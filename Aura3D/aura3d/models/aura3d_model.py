"""Top-level Aura-3D feed-forward avatar model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders.parameter_decoder import GaussianAttrOffsets, UVParameterDecoder
from .deformation.face_deform import FaceDeformMLP
from .encoders.multiview_vit import EncoderOutput, MultiViewViTEncoder
from .eye.gerr import GERREyeBranch
from .flame.flame_template import FLAMECanonicalTemplate
from .gaussians.flame_binding import BoundGaussians, FLAMEGaussianBinding, _quat_mul
from .renderer.gs_renderer import GaussianRenderer, RenderCamera


@dataclass
class IdentityCode:
    """Cached personalisation produced once per user."""
    encoder_out: EncoderOutput
    gaussian_offsets: GaussianAttrOffsets


class Aura3DModel(nn.Module):
    """Feed-forward 3DGS avatar synthesizer.

    Pipeline:
        ref_imgs -> encoder -> identity feat -> decoder -> per-triangle offsets
        FLAME(shape,expr,pose) -> verts -> binding -> world-space Gaussians
        face_deform residual on non-eye triangles
        GERR rigid eye rotation + residual on eye triangles
        rasterizer -> RGB
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg

        enc_cfg = cfg["model"]["encoder"]
        flame_cfg = cfg["model"]["flame"]
        dec_cfg = cfg["model"]["decoder"]
        eye_cfg = cfg["model"]["eye_branch"]
        deform_cfg = cfg["model"]["face_deform_mlp"]
        rend_cfg = cfg["model"]["renderer"]

        self.encoder = MultiViewViTEncoder(
            backbone=enc_cfg["backbone"],
            pretrained=enc_cfg["pretrained"],
            out_dim=enc_cfg["out_dim"],
            num_ref_views=enc_cfg["num_ref_views"],
            cross_view_attn_layers=enc_cfg["cross_view_attn_layers"],
            freeze_backbone=enc_cfg["freeze_backbone"],
        )

        self.flame = FLAMECanonicalTemplate(
            flame_model_path=flame_cfg["flame_model_path"],
            n_shape=flame_cfg["n_shape"],
            n_exp=flame_cfg["n_exp"],
            n_pose=flame_cfg["n_pose"],
            use_face_contour=flame_cfg["use_face_contour"],
        )

        self.decoder = UVParameterDecoder(
            in_dim=enc_cfg["out_dim"],
            feature_dim=dec_cfg["feature_dim"],
            uv_resolution=dec_cfg["uv_resolution"],
        )

        self.binding = FLAMEGaussianBinding(
            faces=self.flame.faces,
            n_gaussians_per_triangle=dec_cfg["n_gaussians_per_triangle"],
        )

        self.face_deform = FaceDeformMLP(
            identity_dim=enc_cfg["out_dim"],
            expr_dim=deform_cfg["expr_cond_dim"],
            pose_dim=deform_cfg["pose_cond_dim"],
            hidden=tuple(deform_cfg["hidden"]),
        )

        self.eye_branch: Optional[nn.Module]
        if eye_cfg["enabled"]:
            self.eye_branch = GERREyeBranch(
                identity_dim=enc_cfg["out_dim"],
                residual_hidden=tuple(eye_cfg["residual_mlp_hidden"]),
                enable_residual=eye_cfg["use_gerr"],
            )
        else:
            self.eye_branch = None

        self.renderer = GaussianRenderer(
            sh_degree=rend_cfg["sh_degree"],
            bg_color=tuple(rend_cfg["bg_color"]),
        )

    # ---- Personalisation (run once per identity) -------------------------
    def encode_identity(self, ref_imgs: torch.Tensor) -> IdentityCode:
        """ref_imgs: (B, V, 3, H, W). Returns cached identity code."""
        enc_out = self.encoder(ref_imgs)
        offsets = self.decoder(enc_out.cls)
        return IdentityCode(encoder_out=enc_out, gaussian_offsets=offsets)

    # ---- Per-frame animation (run every webcam frame) --------------------
    def animate(
        self,
        identity: IdentityCode,
        shape: torch.Tensor,
        expression: torch.Tensor,
        pose: torch.Tensor,
        gaze: Optional[torch.Tensor] = None,
        camera: Optional[RenderCamera] = None,
    ) -> dict:
        """Drive the cached identity and (optionally) rasterize.

        When ``camera`` is ``None`` we return only the posed
        :class:`BoundGaussians` — useful for tests and for trainers that
        share one Gaussian set across many cameras.
        """
        # 1. Pose FLAME mesh.
        verts = self.flame(shape, expression, pose)

        # 2. Bind decoder offsets to the posed mesh.
        gaussians = self.binding(verts, identity.gaussian_offsets)

        # 3. Face-branch residual deformation (skips eye triangles).
        delta = self.face_deform(
            identity_feat=identity.encoder_out.cls,
            expression=expression,
            pose=pose,
            num_triangles=self.flame.num_faces,
        )
        eye_mask_tri = self.flame.eye_face_mask
        face_mask_g = (~eye_mask_tri)[gaussians.triangle_idx]
        d_pos_g = delta.d_position[:, gaussians.triangle_idx]
        d_scale_g = delta.d_scale[:, gaussians.triangle_idx]
        d_rot_g = F.normalize(delta.d_rotation[:, gaussians.triangle_idx], dim=-1)

        xyz = gaussians.xyz.clone()
        scale = gaussians.scale.clone()
        rot = gaussians.rotation.clone()
        if face_mask_g.any():
            xyz[:, face_mask_g] = xyz[:, face_mask_g] + d_pos_g[:, face_mask_g]
            scale[:, face_mask_g] = scale[:, face_mask_g] + d_scale_g[:, face_mask_g]
            rot[:, face_mask_g] = F.normalize(
                _quat_mul(d_rot_g[:, face_mask_g], rot[:, face_mask_g]),
                dim=-1,
            )
        gaussians = BoundGaussians(
            xyz=xyz, scale=scale, rotation=rot,
            color=gaussians.color, opacity=gaussians.opacity,
            triangle_idx=gaussians.triangle_idx,
        )

        # 4. Eye branch: explicit rigid rotation + tiny residual (GERR).
        if self.eye_branch is not None and gaze is not None:
            left_g = self.flame.left_eye_face_mask[gaussians.triangle_idx]
            right_g = self.flame.right_eye_face_mask[gaussians.triangle_idx]
            eye_centers = self.flame.eye_centers(verts)
            gaussians = self.eye_branch(
                gaussians,
                left_idx=left_g,
                right_idx=right_g,
                eye_centers=eye_centers,
                gaze=gaze,
                identity_feat=identity.encoder_out.cls,
            )

        # 5. Rasterize.
        if camera is None:
            return {"gaussians": gaussians}
        out = self.renderer(gaussians, camera)
        out["gaussians"] = gaussians
        return out

    def forward(self, ref_imgs: torch.Tensor) -> IdentityCode:
        return self.encode_identity(ref_imgs)
