"""Camera utilities — convert (K, w2c) into the row-major matrices that
diff-gaussian-rasterization expects.

Convention: NeRSemble ships OpenCV-style world->cam (Z forward, Y down).
3DGS rasterizer expects column-vectors transformed by row-major matrices,
where the projection maps from camera-space directly to NDC ([-1, 1]^3
with Z in [0, 1]).
"""
from __future__ import annotations

import math

import torch

from ..models.renderer.gs_renderer import RenderCamera


def _opencv_to_opengl_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """Flip Y and Z so the matrix matches OpenGL right-handed convention."""
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=w2c.device, dtype=w2c.dtype))
    return flip @ w2c


def projection_matrix(fovx: float, fovy: float, znear: float = 0.01, zfar: float = 100.0,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    tan_x = math.tan(fovx / 2)
    tan_y = math.tan(fovy / 2)
    P = torch.zeros(4, 4, device=device, dtype=dtype)
    P[0, 0] = 1.0 / tan_x
    P[1, 1] = 1.0 / tan_y
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def make_render_camera(
    K: torch.Tensor,           # (3, 3)
    w2c: torch.Tensor,         # (4, 4) OpenCV
    image_height: int,
    image_width: int,
) -> RenderCamera:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    fovx = 2 * math.atan(image_width / (2 * fx))
    fovy = 2 * math.atan(image_height / (2 * fy))

    w2c_gl = _opencv_to_opengl_w2c(w2c)
    # diff-gs uses row-major: viewmatrix is the transpose of the column-major
    # world->view used in standard CG textbooks.
    view = w2c_gl.T.contiguous()
    proj = projection_matrix(fovx, fovy, device=K.device, dtype=K.dtype)
    full_proj = (proj @ w2c_gl).T.contiguous()

    # Camera center in world space.
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_center = -R.T @ t

    return RenderCamera(
        world_view_transform=view,
        full_proj_transform=full_proj,
        camera_center=cam_center,
        image_height=image_height,
        image_width=image_width,
        fovx=fovx,
        fovy=fovy,
    )
