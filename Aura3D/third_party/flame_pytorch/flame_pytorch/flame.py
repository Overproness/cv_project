"""FLAME 2020 PyTorch implementation.

Loads ``generic_model.pkl`` (FLAME 2020 from https://flame.is.tue.mpg.de/)
and runs a differentiable forward pass via linear blend skinning (LBS).

FLAME joint structure (5 joints):
    0: global rotation
    1: neck
    2: jaw
    3: left eye
    4: right eye

Pose convention (n_pose=6, the default used in Aura3D):
    pose_params[:, 0:3]  global rotation    (axis-angle)
    pose_params[:, 3:6]  jaw rotation       (axis-angle)
    neck / eye joints   → zero (identity rotation)

FLAME 2020 shapedirs layout:
    shapedirs[:, :, 0:300]      shape blend shapes
    shapedirs[:, :, 300:400]    expression blend shapes
"""
from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def get_config() -> SimpleNamespace:
    """Return a config namespace with sensible Aura3D defaults."""
    return SimpleNamespace(
        flame_model_path=None,
        shape_params=100,
        expression_params=50,
        pose_params=6,
        use_face_contour=False,
    )


# ---------------------------------------------------------------------------
# Rodrigues rotation
# ---------------------------------------------------------------------------

def _batch_rodrigues(thetas: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle → (..., 3, 3) rotation matrix (Rodrigues formula)."""
    shape = thetas.shape[:-1]
    thetas_flat = thetas.reshape(-1, 3)
    angle = torch.norm(thetas_flat, dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    axis = thetas_flat / angle                                              # (N, 3)

    s = torch.sin(angle)       # (N, 1)
    c = torch.cos(angle)       # (N, 1)
    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]

    # Skew-symmetric cross-product matrix K
    zeros = torch.zeros_like(x)
    K = torch.stack([
        torch.cat([ zeros,  -z,     y    ], dim=1),
        torch.cat([ z,       zeros, -x   ], dim=1),
        torch.cat([-y,       x,     zeros], dim=1),
    ], dim=1)  # (N, 3, 3)

    I = torch.eye(3, device=thetas.device, dtype=thetas.dtype).unsqueeze(0)
    # R = I + sin(θ)*K + (1-cos(θ))*K²
    R = I + s.unsqueeze(-1) * K + (1.0 - c.unsqueeze(-1)) * torch.bmm(K, K)
    return R.reshape(*shape, 3, 3)


# ---------------------------------------------------------------------------
# Blend shapes
# ---------------------------------------------------------------------------

def _blend_shapes(params: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    params : (B, n_coeff)
    dirs   : (V*3, n_coeff)  [stored as (V, 3, n_coeff) → reshaped internally]
    returns: (B, V, 3)
    """
    n_coeff = dirs.shape[-1]
    V3 = dirs.shape[0]
    V = V3 // 3
    # dirs: (V*3, n_coeff)  ·  params^T: (n_coeff, B)  →  (V*3, B) → (B, V, 3)
    displacement = (dirs @ params.T).T.reshape(-1, V, 3)
    return displacement


# ---------------------------------------------------------------------------
# FLAME
# ---------------------------------------------------------------------------

class FLAME(nn.Module):
    """Lightweight differentiable FLAME 3D face model.

    Supports the FLAME 2020 pkl format.  All buffers are registered so the
    model can be moved to GPU with ``.to(device)``.
    """

    # FLAME 2020 canonical sizes
    NUM_VERTS: int = 5023
    NUM_FACES: int = 9976
    NUM_JOINTS: int = 5

    # In FLAME 2020, shape blend shapes occupy the first 300 columns of
    # `shapedirs`; expression blend shapes start at column 300.
    SHAPE_DIM_IN_PKL: int = 300

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        model_path = Path(config.flame_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"FLAME model not found at '{model_path}'.\n"
                "Download FLAME 2020 from https://flame.is.tue.mpg.de/ "
                "(free registration required) and extract generic_model.pkl "
                f"to '{model_path.parent}'."
            )

        with open(model_path, "rb") as fh:
            m = pickle.load(fh, encoding="latin1")

        n_shape = int(config.shape_params)
        n_exp   = int(config.expression_params)

        # ---- template vertices -----------------------------------------------
        v_template = torch.as_tensor(np.array(m["v_template"]), dtype=torch.float32)
        self.register_buffer("v_template", v_template)   # (V, 3)

        # ---- shape & expression blend shapes ---------------------------------
        shapedirs_np = np.array(m["shapedirs"])           # (V, 3, ≥300+n_exp)

        shape_dirs = torch.as_tensor(
            shapedirs_np[:, :, :n_shape].reshape(-1, n_shape),
            dtype=torch.float32,
        )  # (V*3, n_shape)
        self.register_buffer("shapedirs", shape_dirs)

        # Expression: stored at offset SHAPE_DIM_IN_PKL in the same array
        expr_offset = self.SHAPE_DIM_IN_PKL
        if shapedirs_np.shape[2] < expr_offset + n_exp:
            # Fallback: older versions store expression separately
            if "exprdirs" in m:
                expr_np = np.array(m["exprdirs"])[:, :, :n_exp]
            else:
                # Zero blend shapes — model will work but won't express
                expr_np = np.zeros((self.NUM_VERTS, 3, n_exp), dtype=np.float32)
        else:
            expr_np = shapedirs_np[:, :, expr_offset: expr_offset + n_exp]

        expr_dirs = torch.as_tensor(
            expr_np.reshape(-1, n_exp), dtype=torch.float32
        )  # (V*3, n_exp)
        self.register_buffer("exprdirs", expr_dirs)

        # ---- pose blend shapes -----------------------------------------------
        posedirs_np = np.array(m["posedirs"])             # (V, 3, n_pose_basis) OR (V*3, n_pose_basis)
        if posedirs_np.ndim == 3:
            V, three, n_pb = posedirs_np.shape
            posedirs_np = posedirs_np.reshape(V * three, n_pb)
        self.register_buffer(
            "posedirs",
            torch.as_tensor(posedirs_np, dtype=torch.float32),
        )  # (V*3, n_pose_basis)

        # ---- joint regressor -------------------------------------------------
        J_reg = m["J_regressor"]
        try:
            J_reg = J_reg.toarray()          # scipy sparse → dense
        except AttributeError:
            J_reg = np.array(J_reg)
        self.register_buffer(
            "J_regressor",
            torch.as_tensor(J_reg, dtype=torch.float32),
        )  # (J, V)

        # ---- skinning weights ------------------------------------------------
        self.register_buffer(
            "weights",
            torch.as_tensor(np.array(m["weights"]), dtype=torch.float32),
        )  # (V, J)

        # ---- kinematic tree --------------------------------------------------
        kintree = m["kintree_table"]
        parents = kintree[0].astype(np.int32).tolist()
        parents[0] = -1          # root has no parent
        self.register_buffer(
            "parents",
            torch.tensor(parents, dtype=torch.long),
        )

        # ---- faces -----------------------------------------------------------
        faces_np = np.array(m["f"], dtype=np.int64)
        self.register_buffer("faces_tensor", torch.as_tensor(faces_np))   # (F, 3)

    # ---------------------------------------------------------------------- #
    # forward                                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        shape_params:      torch.Tensor,   # (B, n_shape)
        expression_params: torch.Tensor,   # (B, n_exp)
        pose_params:       torch.Tensor,   # (B, n_pose)  — at least 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(vertices (B, V, 3), joints (B, J, 3))``."""
        B   = shape_params.shape[0]
        dev = shape_params.device
        dtype = shape_params.dtype

        # 1. Shaped mesh: template + shape blendshapes + expression blendshapes
        V = self.v_template.shape[0]
        v_shaped = (
            self.v_template.unsqueeze(0)
            + _blend_shapes(shape_params,      self.shapedirs)
            + _blend_shapes(expression_params, self.exprdirs)
        )   # (B, V, 3)

        # 2. Joints from the shaped mesh
        J = torch.einsum("jv,bvd->bjd", self.J_regressor, v_shaped)   # (B, J, 3)
        n_j = J.shape[1]

        # 3. Build per-joint rotation matrices
        #    pose_params layout: [global(3), jaw(3), ...]
        #    joints: 0=global, 1=neck, 2=jaw, 3=leye, 4=reye
        pose_aa = torch.zeros(B, n_j, 3, device=dev, dtype=dtype)
        pose_aa[:, 0] = pose_params[:, :3]                    # global
        if pose_params.shape[1] >= 6 and n_j > 2:
            pose_aa[:, 2] = pose_params[:, 3:6]               # jaw

        R = _batch_rodrigues(pose_aa)   # (B, J, 3, 3)

        # 4. Pose blend shapes: driven by (R[1:] - I) for all non-root joints
        I3 = torch.eye(3, device=dev, dtype=dtype)
        pose_feat = (R[:, 1:] - I3).reshape(B, -1)            # (B, (J-1)*9)
        n_pb = self.posedirs.shape[1]
        feat_dim = pose_feat.shape[1]
        if feat_dim > n_pb:
            pose_feat = pose_feat[:, :n_pb]
        elif feat_dim < n_pb:
            pad = torch.zeros(B, n_pb - feat_dim, device=dev, dtype=dtype)
            pose_feat = torch.cat([pose_feat, pad], dim=1)

        v_posed = v_shaped + _blend_shapes(pose_feat, self.posedirs)   # (B, V, 3)

        # 5. LBS
        verts = self._lbs(v_posed, J, R)   # (B, V, 3)

        # 6. Joints in posed space (approx: re-regress on final mesh)
        joints = torch.einsum("jv,bvd->bjd", self.J_regressor, verts)

        return verts, joints

    # ---------------------------------------------------------------------- #
    # LBS                                                                     #
    # ---------------------------------------------------------------------- #

    def _lbs(
        self,
        v: torch.Tensor,   # (B, V, 3) posed vertices (after blend shapes)
        J: torch.Tensor,   # (B, J, 3) rest-pose joint positions
        R: torch.Tensor,   # (B, J, 3, 3) per-joint rotation matrices
    ) -> torch.Tensor:
        """Linear blend skinning → (B, V, 3)."""
        B  = v.shape[0]
        n_j = J.shape[1]
        dev, dtype = v.device, v.dtype

        # Build world transforms via kinematic tree.
        # local[i] = [R[i], J[i]-J[parent[i]]; 0, 1]  (root uses J[0] directly)
        T_world: list[torch.Tensor] = [None] * n_j  # type: ignore[list-item]

        for i in range(n_j):
            p = int(self.parents[i])

            T_l = torch.zeros(B, 4, 4, device=dev, dtype=dtype)
            T_l[:, :3, :3] = R[:, i]
            T_l[:, :3, 3]  = J[:, i] if p < 0 else (J[:, i] - J[:, p])
            T_l[:, 3,  3]  = 1.0

            T_world[i] = T_l if p < 0 else torch.bmm(T_world[p], T_l)

        T_world_t = torch.stack(T_world, dim=1)   # (B, J, 4, 4)

        # G[i] = T_world[i] @ inv_rest[i]  where inv_rest[i] = [I, -J[i]; 0, 1]
        inv_rest_t = torch.eye(4, device=dev, dtype=dtype).unsqueeze(0).unsqueeze(0)
        inv_rest_t = inv_rest_t.expand(B, n_j, -1, -1).contiguous()
        inv_rest_t[:, :, :3, 3] = -J                # (B, J, 4, 4)

        G = torch.einsum("bjkl,bjlm->bjkm", T_world_t, inv_rest_t)   # (B, J, 4, 4)

        # Blend transforms: T_blend[b,v] = Σ_j W[v,j] * G[b,j]
        T_blend = torch.einsum("vj,bjkl->bvkl", self.weights, G)      # (B, V, 4, 4)

        # Apply to homogeneous vertices
        ones = torch.ones(B, v.shape[1], 1, device=dev, dtype=dtype)
        v_h = torch.cat([v, ones], dim=-1)                             # (B, V, 4)
        v_out = torch.einsum("bvkl,bvl->bvk", T_blend, v_h)          # (B, V, 4)

        return v_out[:, :, :3]
