"""Top-level Aura-3D feed-forward avatar model.

Pipeline (forward pass):
    ref_imgs ─► MultiViewViTEncoder ─► identity feature
                                       │
                                       ▼
                          UVParameterDecoder ─► per-triangle Gaussian
                                                attribute offsets
                                       │
                  FLAME(shape, expr, pose) gives current mesh
                                       │
                                       ▼
                  bind offsets to triangles -> world-space Gaussians
                                       │
                                       ▼
                  rasterizer ─► RGB image (+ alpha, depth)

Animation at inference time replaces (expr, pose, gaze) every frame while
keeping the encoder/decoder output frozen per identity, giving a true
zero-optimization personalized avatar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .decoders.parameter_decoder import GaussianAttrOffsets, UVParameterDecoder
from .encoders.multiview_vit import EncoderOutput, MultiViewViTEncoder
from .flame.flame_template import FLAMECanonicalTemplate


@dataclass
class IdentityCode:
    """Cached personalisation produced once per user."""
    encoder_out: EncoderOutput
    gaussian_offsets: GaussianAttrOffsets


class Aura3DModel(nn.Module):
    """Feed-forward 3DGS avatar synthesizer."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg

        enc_cfg = cfg["model"]["encoder"]
        flame_cfg = cfg["model"]["flame"]
        dec_cfg = cfg["model"]["decoder"]

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

        # Deformation MLPs (face branch) and eye branch (GERR) will be
        # wired in next; declared here so the assembly is visible.
        self.face_deform: Optional[nn.Module] = None
        self.eye_branch: Optional[nn.Module] = None
        self.renderer: Optional[nn.Module] = None

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
        camera: Optional[dict] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Wire face_deform + eye_branch + renderer in the next iteration."
        )

    def forward(self, ref_imgs: torch.Tensor) -> IdentityCode:
        return self.encode_identity(ref_imgs)
