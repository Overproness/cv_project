"""FLAME -> Gaussian binding (GaussianAvatars-style).

Given a posed FLAME mesh (B, V, 3) and per-triangle attribute offsets from
the parameter decoder, produce a set of world-space 3D Gaussians:

  * For each triangle we attach K Gaussians (default K=1).
  * Each Gaussian's world position = triangle centroid +
        R_local @ (position_offset * triangle_scale)
  * R_local is the orthonormal triangle frame (tangent, normal, bitangent).
  * Gaussian scale = decoder log-scale * triangle_scale (so scale follows
        the underlying mesh as the face deforms / talks).
  * Gaussian rotation = R_local ⊗ decoder_quat (compose triangle frame
        with decoded residual rotation).

This keeps the Gaussians rigidly attached to the moving FLAME surface;
animation = just re-running this binding on the new mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BoundGaussians:
    xyz: torch.Tensor          # (B, F*K, 3) world-space positions
    scale: torch.Tensor        # (B, F*K, 3) log-scale (raw, post-exp = real scale)
    rotation: torch.Tensor     # (B, F*K, 4) unit quaternion (wxyz)
    color: torch.Tensor        # (B, F*K, 3) RGB or SH-DC
    opacity: torch.Tensor      # (B, F*K, 1) pre-sigmoid logits
    triangle_idx: torch.Tensor # (F*K,) which FLAME triangle each Gaussian came from


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, both (..., 4) in wxyz order."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) wxyz, numerically stable.

    Uses the sign-stable Shepperd-style formula: each component is the
    square root of a positive combination of diagonal terms, with sign
    copied from the matching off-diagonal difference. This avoids the
    branchy ``argmax(diag)`` fallback and is fully vectorised.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(1e-8))
    qx = 0.5 * torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-8))
    qy = 0.5 * torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(1e-8))
    qz = 0.5 * torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(1e-8))
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    return F.normalize(torch.stack([qw, qx, qy, qz], dim=-1), dim=-1)


class FLAMEGaussianBinding(nn.Module):
    """Bind decoder-predicted attributes to a posed FLAME mesh."""

    def __init__(self, faces: torch.Tensor, n_gaussians_per_triangle: int = 1) -> None:
        super().__init__()
        self.k = n_gaussians_per_triangle
        self.register_buffer("faces", faces.long())  # (F, 3)
        # (F*K,) lookup so eye-mask etc. can be broadcast to gaussians.
        f = faces.shape[0]
        tri_idx = torch.arange(f).repeat_interleave(self.k)
        self.register_buffer("triangle_idx", tri_idx)

    @staticmethod
    def _triangle_frame(verts: torch.Tensor, faces: torch.Tensor):
        """verts (B, V, 3), faces (F, 3) -> (centroid, R_local, edge_scale).

        R_local is (B, F, 3, 3) with columns [tangent, bitangent, normal].
        edge_scale (B, F, 1) is the average edge length, used to scale local
        offsets so they live in mesh-units rather than absolute meters.
        """
        v0 = verts[:, faces[:, 0]]
        v1 = verts[:, faces[:, 1]]
        v2 = verts[:, faces[:, 2]]
        centroid = (v0 + v1 + v2) / 3.0

        e1 = v1 - v0
        e2 = v2 - v0
        n = torch.cross(e1, e2, dim=-1)
        n = F.normalize(n, dim=-1)
        t = F.normalize(e1, dim=-1)
        b = torch.cross(n, t, dim=-1)
        R = torch.stack([t, b, n], dim=-1)  # (B, F, 3, 3)

        edge_scale = (
            e1.norm(dim=-1) + e2.norm(dim=-1) + (v2 - v1).norm(dim=-1)
        ).unsqueeze(-1) / 3.0
        return centroid, R, edge_scale

    def forward(
        self,
        verts: torch.Tensor,
        offsets,
    ) -> BoundGaussians:
        """verts: (B, V, 3); offsets: GaussianAttrOffsets with shapes (B, F, *)."""
        centroid, R, edge_scale = self._triangle_frame(verts, self.faces)

        # World-space position: centroid + R @ (offset * edge_scale).
        local = offsets.position_offset * edge_scale  # (B, F, 3)
        xyz = centroid + torch.einsum("bfij,bfj->bfi", R, local)

        # Compose triangle frame with decoder residual rotation.
        q_local = _matrix_to_quaternion(R)            # (B, F, 4)
        q_world = _quat_mul(q_local, offsets.rotation)
        q_world = F.normalize(q_world, dim=-1)

        # Scale tracks the mesh edge length so Gaussians shrink/grow with it.
        log_scale = offsets.scale + torch.log(edge_scale.clamp_min(1e-6))

        # If K > 1 we'd repeat-interleave here; default K=1 keeps it trivial.
        if self.k != 1:
            xyz = xyz.repeat_interleave(self.k, dim=1)
            log_scale = log_scale.repeat_interleave(self.k, dim=1)
            q_world = q_world.repeat_interleave(self.k, dim=1)
            color = offsets.color.repeat_interleave(self.k, dim=1)
            opacity = offsets.opacity.repeat_interleave(self.k, dim=1)
        else:
            color = offsets.color
            opacity = offsets.opacity

        return BoundGaussians(
            xyz=xyz,
            scale=log_scale,
            rotation=q_world,
            color=color,
            opacity=opacity,
            triangle_idx=self.triangle_idx,
        )
