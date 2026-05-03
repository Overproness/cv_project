"""Thin wrapper around `diff-gaussian-rasterization`.

Keeps the rest of the codebase agnostic to the rasterizer's calling
convention. We deliberately do NOT modify the upstream CUDA kernel —
all dynamic offsets / deformations happen in PyTorch upstream of this
wrapper, so we only feed the rasterizer with final per-Gaussian tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..gaussians.flame_binding import BoundGaussians


@dataclass
class RenderCamera:
    """Minimum camera parameters the rasterizer needs."""
    world_view_transform: torch.Tensor  # (4, 4) row-major (Gaussian-Splatting convention)
    full_proj_transform: torch.Tensor   # (4, 4)
    camera_center: torch.Tensor         # (3,)
    image_height: int
    image_width: int
    fovx: float
    fovy: float


class GaussianRenderer(nn.Module):
    """diff-gaussian-rasterization wrapper. Inference path = rasterize only."""

    def __init__(self, sh_degree: int = 1, bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.register_buffer("bg_color", torch.tensor(bg_color, dtype=torch.float32))

    def forward(self, g: BoundGaussians, cam: RenderCamera) -> dict:
        # Lazy import: keeps the module importable on machines without the
        # compiled CUDA extension (e.g. CI / dev laptops without a GPU).
        from diff_gaussian_rasterization import (  # type: ignore
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        assert g.xyz.shape[0] == 1, "Renderer expects batch size 1 (per-view)."
        means3D = g.xyz[0]
        scales = torch.exp(g.scale[0])
        rotations = g.rotation[0]                  # already unit-norm
        opacities = torch.sigmoid(g.opacity[0])
        colors = g.color[0]                        # treated as precomputed RGB

        tanfovx = float(torch.tan(torch.tensor(cam.fovx) * 0.5))
        tanfovy = float(torch.tan(torch.tensor(cam.fovy) * 0.5))

        settings = GaussianRasterizationSettings(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=0,
            campos=cam.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)

        screenspace_pts = torch.zeros_like(means3D, requires_grad=True)
        result = rasterizer(
            means3D=means3D,
            means2D=screenspace_pts,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        # Newer diff-gaussian-rasterization returns (rgb, radii, depth);
        # older versions return (rgb, radii). Unpack defensively.
        rgb, radii = result[0], result[1]
        depth = result[2] if len(result) > 2 else None
        return {"rgb": rgb, "radii": radii, "depth": depth,
                "screenspace_pts": screenspace_pts}
