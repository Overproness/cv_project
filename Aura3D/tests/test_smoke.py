"""Aura-3D smoke tests.

These tests are designed to run on CPU without external assets:
  * No DINOv2 weights download
  * No FLAME .pkl
  * No CUDA rasterizer

Instead we test the building blocks individually and assemble a tiny
mock model end-to-end through `animate(camera=None)`.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from aura3d.data.datasets.nersemble import (
    NeRSemblePhase1Dataset,
    collate_samples,
)
from aura3d.losses.perceptual import (
    FLAMERegLoss,
    GazeLoss,
    IdentityLoss,
    LPIPSLoss,
)
from aura3d.losses.photometric import PhotometricLoss, ssim
from aura3d.models.aura3d_model import Aura3DModel, IdentityCode
from aura3d.models.decoders.parameter_decoder import (
    GaussianAttrOffsets,
    UVParameterDecoder,
)
from aura3d.models.deformation.face_deform import FaceDeformMLP
from aura3d.models.eye.gerr import GERREyeBranch
from aura3d.models.encoders.multiview_vit import EncoderOutput
from aura3d.models.gaussians.flame_binding import (
    BoundGaussians,
    FLAMEGaussianBinding,
    _matrix_to_quaternion,
    _quat_mul,
)
from aura3d.utils.camera import make_render_camera


# ---------------------------------------------------------------- utilities
def _make_unit_tetrahedron(b: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (verts (B, V, 3), faces (F, 3))."""
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0).expand(b, -1, -1).contiguous()
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long
    )
    return verts, faces


def _zero_offsets(b: int, f: int) -> GaussianAttrOffsets:
    return GaussianAttrOffsets(
        position_offset=torch.zeros(b, f, 3),
        scale=torch.zeros(b, f, 3),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(b, f, 4).contiguous(),
        color=torch.zeros(b, f, 3),
        opacity=torch.zeros(b, f, 1),
    )


# ---------------------------------------------------------------- quaternions
def test_quat_identity_from_identity_matrix():
    R = torch.eye(3).expand(5, 3, 3)
    q = _matrix_to_quaternion(R)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(5, 4)
    assert torch.allclose(q.abs(), expected.abs(), atol=1e-4)


def test_quat_mul_identity():
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 4)
    p = torch.randn(3, 4)
    p = p / p.norm(dim=-1, keepdim=True)
    out = _quat_mul(q, p)
    assert torch.allclose(out, p, atol=1e-6)


# ---------------------------------------------------------------- decoder
def test_uv_decoder_zero_init_produces_identity_offsets():
    dec = UVParameterDecoder(in_dim=64, feature_dim=16, uv_resolution=64)
    f = 32
    dec.set_triangle_uvs(torch.rand(f, 2))
    feat = torch.randn(2, 64)
    out = dec(feat)
    assert out.position_offset.shape == (2, f, 3)
    assert torch.allclose(out.position_offset, torch.zeros_like(out.position_offset))
    # Identity quat is (x=0,y=0,z=0,w=1) per decoder convention.
    assert torch.allclose(out.rotation[..., 3], torch.ones(2, f), atol=1e-6)


# ---------------------------------------------------------------- binding
def test_flame_binding_identity_offsets_produces_centroids():
    verts, faces = _make_unit_tetrahedron(b=2)
    binder = FLAMEGaussianBinding(faces, n_gaussians_per_triangle=1)
    offsets = _zero_offsets(b=2, f=faces.shape[0])
    # Decoder convention is (x,y,z,w) but binding consumes wxyz order; the
    # identity quat is the same up to permutation. We pass identity wxyz.
    offsets = GaussianAttrOffsets(
        position_offset=offsets.position_offset,
        scale=offsets.scale,
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, faces.shape[0], 4).contiguous(),
        color=offsets.color,
        opacity=offsets.opacity,
    )
    g = binder(verts, offsets)
    # With zero offsets the Gaussian xyz must equal the triangle centroid.
    expected_centroids = verts[:, faces].mean(dim=2)
    assert torch.allclose(g.xyz, expected_centroids, atol=1e-6)
    assert g.rotation.shape == (2, 4, 4)
    assert torch.allclose(g.rotation.norm(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_flame_binding_K_greater_than_one_repeats():
    verts, faces = _make_unit_tetrahedron()
    binder = FLAMEGaussianBinding(faces, n_gaussians_per_triangle=3)
    offsets = _zero_offsets(b=1, f=faces.shape[0])
    g = binder(verts, offsets)
    assert g.xyz.shape == (1, faces.shape[0] * 3, 3)
    assert g.triangle_idx.shape == (faces.shape[0] * 3,)


# ---------------------------------------------------------------- deformation
def test_face_deform_zero_init_returns_identity():
    mlp = FaceDeformMLP(identity_dim=32, expr_dim=8, pose_dim=6)
    delta = mlp(
        identity_feat=torch.randn(2, 32),
        expression=torch.randn(2, 8),
        pose=torch.randn(2, 6),
        num_triangles=10,
    )
    assert torch.allclose(delta.d_position, torch.zeros_like(delta.d_position))
    assert torch.allclose(delta.d_scale, torch.zeros_like(delta.d_scale))
    # First component of d_rotation is identity quaternion (w=1).
    assert torch.allclose(delta.d_rotation[..., 0], torch.ones(2, 10))


# ---------------------------------------------------------------- eye branch
def test_gerr_zero_gaze_with_no_eye_triangles_is_no_op():
    branch = GERREyeBranch(identity_dim=16, residual_hidden=(8,), enable_residual=True)
    f = 8
    g = BoundGaussians(
        xyz=torch.randn(1, f, 3),
        scale=torch.zeros(1, f, 3),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, f, 4).contiguous(),
        color=torch.zeros(1, f, 3),
        opacity=torch.zeros(1, f, 1),
        triangle_idx=torch.arange(f),
    )
    left = torch.zeros(f, dtype=torch.bool)
    right = torch.zeros(f, dtype=torch.bool)
    out = branch(
        g,
        left_idx=left,
        right_idx=right,
        eye_centers=torch.zeros(1, 2, 3),
        gaze=torch.zeros(1, 2),
        identity_feat=torch.zeros(1, 16),
    )
    assert torch.allclose(out.xyz, g.xyz)


def test_gerr_rotates_eye_gaussians_around_center():
    branch = GERREyeBranch(identity_dim=4, residual_hidden=(4,), enable_residual=False)
    # Single Gaussian on left eye, 1m to the right of the center.
    f = 1
    eye_center = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    g = BoundGaussians(
        xyz=torch.tensor([[[1.0, 0.0, 0.0]]]),
        scale=torch.zeros(1, f, 3),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, f, 4).contiguous(),
        color=torch.zeros(1, f, 3),
        opacity=torch.zeros(1, f, 1),
        triangle_idx=torch.arange(f),
    )
    left = torch.tensor([True])
    right = torch.tensor([False])
    # Yaw 90 deg => point should land near (0,0,-1) under Ry(90).
    gaze = torch.tensor([[0.0, math.pi / 2]])
    out = branch(
        g,
        left_idx=left,
        right_idx=right,
        eye_centers=eye_center,
        gaze=gaze,
        identity_feat=torch.zeros(1, 4),
    )
    expected = torch.tensor([[[0.0, 0.0, -1.0]]])
    assert torch.allclose(out.xyz, expected, atol=1e-5)


# ---------------------------------------------------------------- losses
def test_ssim_identity_is_one():
    img = torch.rand(1, 3, 32, 32)
    s = ssim(img, img)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3)


def test_photometric_loss_drops_with_better_pred():
    target = torch.rand(1, 3, 32, 32)
    far = torch.zeros_like(target)
    near = target.clone() + 0.01 * torch.randn_like(target)
    loss = PhotometricLoss()
    assert loss(near, target)["total"] < loss(far, target)["total"]


def test_perceptual_loss_fallbacks_are_safe():
    # All four perceptual losses must not raise when their backends are
    # missing. They are explicitly designed to degrade gracefully.
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)
    LPIPSLoss()(pred, target)            # falls back to MSE if lpips missing
    IdentityLoss(embedder=None)(pred, target)
    GazeLoss(estimator=None)(pred, torch.zeros(1, 2))
    FLAMERegLoss()(torch.zeros(1, 100), torch.zeros(1, 50))


# ---------------------------------------------------------------- camera
def test_render_camera_shapes_and_center():
    K = torch.eye(3) * 256
    K[2, 2] = 1.0
    K[0, 2] = K[1, 2] = 256
    w2c = torch.eye(4)
    w2c[2, 3] = 1.5
    cam = make_render_camera(K, w2c, image_height=512, image_width=512)
    assert cam.world_view_transform.shape == (4, 4)
    assert cam.full_proj_transform.shape == (4, 4)
    assert cam.camera_center.shape == (3,)
    # Camera center for w2c=Translate(0,0,1.5) is at world origin shifted -1.5.
    assert torch.allclose(cam.camera_center, torch.tensor([0.0, 0.0, -1.5]))
    assert cam.fovx == pytest.approx(2 * math.atan(512 / (2 * 256)))


# ---------------------------------------------------------------- dataset
def test_synthetic_dataset_yields_correct_shapes():
    ds = NeRSemblePhase1Dataset(
        root="this/path/does/not/exist",
        num_ref_views=4,
        image_size=32,
        n_shape=100,
        n_exp=50,
        n_pose=6,
        max_timesteps=2,
        synthetic=True,
    )
    assert len(ds) > 0
    s = ds[0]
    assert s.ref_imgs.shape == (4, 3, 32, 32)
    assert s.target_img.shape == (3, 32, 32)
    assert s.target_K.shape == (3, 3)
    assert s.target_w2c.shape == (4, 4)
    assert s.shape.shape == (100,)
    assert s.expression.shape == (50,)
    assert s.pose.shape == (6,)
    assert s.gaze.shape == (2,)

    batch = collate_samples([ds[0], ds[1]])
    assert batch["ref_imgs"].shape == (2, 4, 3, 32, 32)
    assert batch["target_img"].shape == (2, 3, 32, 32)
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2


# ----------------------------------------------- end-to-end animate (no-rast)
class _StubEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(3, out_dim)

    def forward(self, ref_imgs: torch.Tensor) -> EncoderOutput:
        # Pool over views/spatial: (B, V, 3, H, W) -> (B, 3) -> (B, C)
        cls = self.proj(ref_imgs.mean(dim=(1, 3, 4)))
        return EncoderOutput(
            tokens=cls.unsqueeze(1).unsqueeze(1),
            fused=cls.unsqueeze(1),
            cls=cls,
        )


class _StubFLAME(nn.Module):
    """Tiny stand-in for FLAMECanonicalTemplate (no .pkl required)."""

    def __init__(self) -> None:
        super().__init__()
        verts, faces = _make_unit_tetrahedron(b=1)
        self.register_buffer("canonical_verts", verts[0])
        self.register_buffer("faces", faces)
        eye = torch.zeros(faces.shape[0], dtype=torch.bool)
        eye[0] = True  # mark first triangle as left eye for test
        self.register_buffer("eye_face_mask", eye)
        self.register_buffer("left_eye_face_mask", eye)
        self.register_buffer("right_eye_face_mask",
                             torch.zeros_like(eye))
        self.num_faces_v = faces.shape[0]

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.canonical_verts.shape[0])

    def forward(self, shape, expression, pose):
        b = shape.shape[0]
        return self.canonical_verts.unsqueeze(0).expand(b, -1, -1).contiguous()

    def eye_centers(self, verts):
        b = verts.shape[0]
        return torch.zeros(b, 2, 3, device=verts.device, dtype=verts.dtype)


def _build_mock_aura3d() -> Aura3DModel:
    """Construct an Aura3DModel without DINOv2/FLAME asset deps."""
    cfg = {
        "model": {
            "encoder": {"backbone": "stub", "pretrained": False, "out_dim": 16,
                        "num_ref_views": 2, "cross_view_attn_layers": 1,
                        "freeze_backbone": False},
            "flame": {"flame_model_path": "<unused>", "n_shape": 4,
                      "n_exp": 4, "n_pose": 6, "use_face_contour": False},
            "decoder": {"type": "uv_decoder", "feature_dim": 16,
                        "uv_resolution": 32, "n_gaussians_per_triangle": 1,
                        "predict_attrs": []},
            "eye_branch": {"enabled": True, "eyeball_radius": 0.012,
                           "use_gerr": True, "residual_mlp_hidden": [8, 8]},
            "face_deform_mlp": {"hidden": [16, 16],
                                "expr_cond_dim": 4, "pose_cond_dim": 6},
            "renderer": {"sh_degree": 1, "bg_color": [0.0, 0.0, 0.0],
                         "use_egnr": False, "egnr_train_only": True},
        },
        "training": {}, "data": {},
    }
    # Bypass __init__ to avoid loading DINOv2 + FLAME .pkl. Construct the
    # nn.Module with the same attribute layout.
    model = Aura3DModel.__new__(Aura3DModel)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.encoder = _StubEncoder(out_dim=16)
    model.flame = _StubFLAME()
    model.decoder = UVParameterDecoder(in_dim=16, feature_dim=8, uv_resolution=32)
    model.decoder.set_triangle_uvs(torch.rand(model.flame.num_faces, 2))
    model.binding = FLAMEGaussianBinding(model.flame.faces, n_gaussians_per_triangle=1)
    model.face_deform = FaceDeformMLP(identity_dim=16, expr_dim=4, pose_dim=6,
                                      hidden=(16, 16))
    model.eye_branch = GERREyeBranch(identity_dim=16, residual_hidden=(8, 8),
                                     enable_residual=True)
    model.renderer = nn.Identity()  # never invoked when camera=None
    return model


def test_aura3d_animate_runs_without_camera():
    model = _build_mock_aura3d()
    ref = torch.rand(1, 2, 3, 16, 16)
    identity = model.encode_identity(ref)
    out = model.animate(
        identity=identity,
        shape=torch.zeros(1, 4),
        expression=torch.zeros(1, 4),
        pose=torch.zeros(1, 6),
        gaze=torch.zeros(1, 2),
        camera=None,
    )
    g = out["gaussians"]
    assert g.xyz.shape == (1, model.flame.num_faces, 3)
    assert g.rotation.shape == (1, model.flame.num_faces, 4)
    assert torch.isfinite(g.xyz).all()


def test_aura3d_animate_supports_backprop():
    model = _build_mock_aura3d()
    ref = torch.rand(1, 2, 3, 16, 16)
    identity = model.encode_identity(ref)
    out = model.animate(
        identity=identity,
        shape=torch.zeros(1, 4),
        expression=torch.zeros(1, 4),
        pose=torch.zeros(1, 6),
        gaze=torch.zeros(1, 2),
        camera=None,
    )
    out["gaussians"].xyz.sum().backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad
