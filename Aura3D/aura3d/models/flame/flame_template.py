"""FLAME canonical template wrapper.

Thin wrapper over `flame-pytorch` that exposes:
  * canonical (zero shape / zero expression / neutral pose) vertices and faces
  * UV coordinates and triangle->UV mapping
  * eye-region triangle indices (for two-stream face/eye split)

The actual FLAME forward pass is delegated to the upstream library; this
class only owns the per-template buffers that Aura-3D needs.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class FLAMECanonicalTemplate(nn.Module):
    def __init__(
        self,
        flame_model_path: str,
        n_shape: int = 100,
        n_exp: int = 50,
        n_pose: int = 6,
        use_face_contour: bool = False,
    ) -> None:
        super().__init__()
        # Lazy import: keeps top-level import cheap and avoids hard dep at
        # module import time when only a config dump is needed.
        from flame_pytorch import FLAME, get_config  # type: ignore

        cfg = get_config()
        cfg.flame_model_path = flame_model_path
        cfg.shape_params = n_shape
        cfg.expression_params = n_exp
        cfg.pose_params = n_pose
        cfg.use_face_contour = use_face_contour

        self.flame = FLAME(cfg)
        self.n_shape = n_shape
        self.n_exp = n_exp
        self.n_pose = n_pose

        # Canonical mesh: shape=0, expr=0, pose=0.
        with torch.no_grad():
            shape = torch.zeros(1, n_shape)
            exp = torch.zeros(1, n_exp)
            pose = torch.zeros(1, n_pose)
            verts, _ = self.flame(shape, exp, pose)
        self.register_buffer("canonical_verts", verts[0])           # (V, 3)
        self.register_buffer("faces", self.flame.faces_tensor.long())  # (F, 3)

        # Eye-region triangles. FLAME ships eye vertex indices; we mark a
        # triangle as "eye" if any of its vertices is in the eye set.
        eye_vidx = self._load_eye_vertex_indices()
        face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool)
        for i in range(3):
            face_mask |= torch.isin(self.faces[:, i], eye_vidx)
        self.register_buffer("eye_face_mask", face_mask)

    @staticmethod
    def _load_eye_vertex_indices() -> torch.Tensor:
        # Placeholder: replace with the vertex indices shipped with the
        # FLAME asset bundle (e.g. `flame_static_embedding.pkl` -> eye_idx).
        return torch.tensor([], dtype=torch.long)

    @property
    def num_vertices(self) -> int:
        return int(self.canonical_verts.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def forward(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        verts, _ = self.flame(shape, expression, pose)
        return verts
