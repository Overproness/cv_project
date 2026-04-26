"""GERR — Gaze-conditioned Eye Rigid Rotation + residual.

Pure-MLP eye rotation prediction is what causes "drifting eyeball"
artefacts (see GazeGaussian §3.3). We instead apply an *explicit* rigid
rotation around each eyeball center driven by the input pitch/yaw, and
let a tiny MLP predict only a small residual to absorb modelling error.

Inputs:
  * gaussians  : BoundGaussians for the whole face (we only modify the
                 eye-region indices)
  * eye_mask   : (F,) bool — which FLAME triangles belong to the eyes
  * eye_centers: (B, 2, 3) world-space centers of left/right eyeballs
                 (derived from posed FLAME landmarks upstream)
  * left_idx / right_idx: (F,) bool partition of eye_mask into L/R
  * gaze       : (B, 2)  pitch, yaw in radians
  * identity_feat: (B, C) for the residual conditioning

We do NOT touch color / opacity / scale here — only position & rotation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gaussians.flame_binding import BoundGaussians, _quat_mul


def _euler_pitchyaw_to_matrix(pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """(B,) each -> (B, 3, 3). Pitch around X, yaw around Y, applied as Ry @ Rx."""
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(pitch)
    one = torch.ones_like(pitch)
    Rx = torch.stack(
        [one, zero, zero, zero, cp, -sp, zero, sp, cp], dim=-1
    ).view(-1, 3, 3)
    Ry = torch.stack(
        [cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=-1
    ).view(-1, 3, 3)
    return Ry @ Rx


def _matrix_to_quaternion_simple(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) -> (B, 4) wxyz; assumes valid rotations near identity."""
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    s = torch.sqrt((t + 1.0).clamp_min(1e-8)) * 2
    w = 0.25 * s
    x = (R[..., 2, 1] - R[..., 1, 2]) / s
    y = (R[..., 0, 2] - R[..., 2, 0]) / s
    z = (R[..., 1, 0] - R[..., 0, 1]) / s
    q = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(q, dim=-1)


class GERREyeBranch(nn.Module):
    def __init__(
        self,
        identity_dim: int,
        residual_hidden: tuple[int, ...] = (128, 128),
        enable_residual: bool = True,
    ) -> None:
        super().__init__()
        self.enable_residual = enable_residual
        if not enable_residual:
            return
        in_dim = identity_dim + 2  # +pitch, yaw
        layers: list[nn.Module] = []
        prev = in_dim
        for h in residual_hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        # 3 (pos) + 3 (axis-angle residual rotation), zero-init.
        self.head = nn.Linear(prev, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _axis_angle_to_quat(aa: torch.Tensor) -> torch.Tensor:
        theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis = aa / theta
        w = torch.cos(theta / 2)
        xyz = axis * torch.sin(theta / 2)
        return F.normalize(torch.cat([w, xyz], dim=-1), dim=-1)

    def forward(
        self,
        g: BoundGaussians,
        left_idx: torch.Tensor,    # (F*K,) bool
        right_idx: torch.Tensor,   # (F*K,) bool
        eye_centers: torch.Tensor, # (B, 2, 3) [left, right]
        gaze: torch.Tensor,        # (B, 2) pitch, yaw
        identity_feat: torch.Tensor,
    ) -> BoundGaussians:
        b = g.xyz.shape[0]
        pitch, yaw = gaze[:, 0], gaze[:, 1]
        R_eye = _euler_pitchyaw_to_matrix(pitch, yaw)            # (B, 3, 3)
        q_eye = _matrix_to_quaternion_simple(R_eye)              # (B, 4)

        xyz = g.xyz.clone()
        rot = g.rotation.clone()

        for side, mask, center_idx in (("L", left_idx, 0), ("R", right_idx, 1)):
            if mask.sum() == 0:
                continue
            c = eye_centers[:, center_idx].unsqueeze(1)          # (B, 1, 3)
            sub = g.xyz[:, mask] - c                              # (B, n, 3)
            sub = torch.einsum("bij,bnj->bni", R_eye, sub) + c
            xyz[:, mask] = sub

            # Apply rigid rotation to each Gaussian's orientation as well.
            q_b = q_eye.unsqueeze(1).expand(-1, mask.sum(), -1)
            rot[:, mask] = F.normalize(_quat_mul(q_b, g.rotation[:, mask]), dim=-1)

        if self.enable_residual:
            cond = torch.cat([identity_feat, gaze], dim=-1)      # (B, C+2)
            res = self.head(self.trunk(cond))                    # (B, 6)
            d_pos = res[:, :3].unsqueeze(1)                      # (B, 1, 3)
            d_aa = res[:, 3:].unsqueeze(1)                       # (B, 1, 3)
            d_q = self._axis_angle_to_quat(d_aa)                 # (B, 1, 4)

            eye_any = left_idx | right_idx
            xyz[:, eye_any] = xyz[:, eye_any] + d_pos
            d_q_e = d_q.expand(-1, int(eye_any.sum()), -1)
            rot[:, eye_any] = F.normalize(_quat_mul(d_q_e, rot[:, eye_any]), dim=-1)

        return BoundGaussians(
            xyz=xyz,
            scale=g.scale,
            rotation=rot,
            color=g.color,
            opacity=g.opacity,
            triangle_idx=g.triangle_idx,
        )
