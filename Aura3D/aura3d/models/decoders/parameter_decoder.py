"""Parameter decoder.

Consumes the fused multi-view ViT feature and produces per-FLAME-triangle
Gaussian attribute offsets that personalise the canonical template to the
identity in the reference images.

Design choice: predict a UV feature map (CNN decoder), then sample it at
each triangle's UV centroid. This gives smooth, locally-coherent offsets
and trains far more stably than predicting an unstructured offset list
directly with an MLP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GaussianAttrOffsets:
    position_offset: torch.Tensor   # (B, F, 3) local-frame xyz delta
    scale: torch.Tensor             # (B, F, 3) log-scale
    rotation: torch.Tensor          # (B, F, 4) quaternion (xyzw, normalised)
    color: torch.Tensor             # (B, F, 3) RGB or SH-DC
    opacity: torch.Tensor           # (B, F, 1) pre-sigmoid


class UVParameterDecoder(nn.Module):
    """Token feature -> UV map -> per-triangle Gaussian attribute offsets."""

    def __init__(
        self,
        in_dim: int,
        feature_dim: int = 128,
        uv_resolution: int = 256,
        triangle_uv_centroids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.uv_resolution = uv_resolution
        self.feature_dim = feature_dim

        # Triangle UV centroids in [0, 1]^2; registered as a buffer so it
        # follows .to(device). Can be set later via `set_triangle_uvs`.
        self.register_buffer("tri_uv", triangle_uv_centroids, persistent=False)

        # Project token feature to a low-res spatial grid, then upsample.
        self.token_to_grid = nn.Linear(in_dim, feature_dim * 16 * 16)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim, 4, 2, 1), nn.GELU(),  # 32
            nn.ConvTranspose2d(feature_dim, feature_dim, 4, 2, 1), nn.GELU(),  # 64
            nn.ConvTranspose2d(feature_dim, feature_dim, 4, 2, 1), nn.GELU(),  # 128
            nn.ConvTranspose2d(feature_dim, feature_dim, 4, 2, 1), nn.GELU(),  # 256
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
        )

        # Per-attribute heads applied at each triangle.
        self.head_pos = nn.Linear(feature_dim, 3)
        self.head_scale = nn.Linear(feature_dim, 3)
        self.head_rot = nn.Linear(feature_dim, 4)
        self.head_color = nn.Linear(feature_dim, 3)
        self.head_opacity = nn.Linear(feature_dim, 1)

        # Small init so the canonical template is the starting point.
        for m in (self.head_pos, self.head_scale, self.head_color, self.head_opacity):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head_rot.weight)
        with torch.no_grad():
            self.head_rot.bias.zero_()
            self.head_rot.bias[3] = 1.0   # identity quaternion

    def set_triangle_uvs(self, tri_uv: torch.Tensor) -> None:
        # Re-assign the buffer in place; persistent=False mirrors __init__.
        self.tri_uv = tri_uv.clamp(0.0, 1.0).to(
            device=next(self.parameters()).device,
        )

    def forward(self, identity_feat: torch.Tensor) -> GaussianAttrOffsets:
        """identity_feat: (B, C) global identity vector from the encoder."""
        if self.tri_uv is None:
            raise RuntimeError("Call set_triangle_uvs(...) before forward.")

        b = identity_feat.shape[0]
        grid_feat = self.token_to_grid(identity_feat).view(b, self.feature_dim, 16, 16)
        uv_map = self.up(grid_feat)                             # (B, C, 256, 256)

        # Sample at triangle centroids. grid_sample expects coords in [-1, 1].
        sample_grid = (self.tri_uv * 2.0 - 1.0).view(1, -1, 1, 2).expand(b, -1, -1, -1)
        sampled = F.grid_sample(uv_map, sample_grid, align_corners=False)  # (B, C, F, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)           # (B, F, C)

        rot = self.head_rot(sampled)
        rot = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        return GaussianAttrOffsets(
            position_offset=self.head_pos(sampled),
            scale=self.head_scale(sampled),
            rotation=rot,
            color=self.head_color(sampled),
            opacity=self.head_opacity(sampled),
        )
