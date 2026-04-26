"""Face-branch deformation MLP.

The FLAME mesh already gives us the bulk of the expression / pose motion.
This MLP predicts a *residual* per-triangle correction conditioned on the
identity feature and the current (expression, pose) so that:

  * fine-scale dynamics (skin sliding, dimples, lip thickness) are captured
  * eye-region triangles are excluded — those are handled by GERR
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FaceDeformDelta:
    d_position: torch.Tensor   # (B, F, 3)
    d_scale: torch.Tensor      # (B, F, 3)
    d_rotation: torch.Tensor   # (B, F, 4) tangent-space quaternion delta (will be normalised later)


class FaceDeformMLP(nn.Module):
    def __init__(
        self,
        identity_dim: int,
        expr_dim: int,
        pose_dim: int,
        hidden: tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        in_dim = identity_dim + expr_dim + pose_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Heads: small zero-init so the residual starts as identity.
        self.head_pos = nn.Linear(prev, 3)
        self.head_scale = nn.Linear(prev, 3)
        self.head_rot = nn.Linear(prev, 4)
        for m in (self.head_pos, self.head_scale, self.head_rot):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        with torch.no_grad():
            # Identity quaternion (wxyz)
            self.head_rot.bias[0] = 1.0

    def forward(
        self,
        identity_feat: torch.Tensor,   # (B, C)
        expression: torch.Tensor,      # (B, expr_dim)
        pose: torch.Tensor,            # (B, pose_dim)
        num_triangles: int,
    ) -> FaceDeformDelta:
        cond = torch.cat([identity_feat, expression, pose], dim=-1)  # (B, in_dim)
        h = self.trunk(cond)                                         # (B, hidden)
        # Broadcast a single per-frame correction across triangles. A future
        # version can replace this with per-triangle tokens; for Stage-1
        # overfitting a global expression-driven residual is sufficient.
        h = h.unsqueeze(1).expand(-1, num_triangles, -1)
        return FaceDeformDelta(
            d_position=self.head_pos(h),
            d_scale=self.head_scale(h),
            d_rotation=self.head_rot(h),
        )
