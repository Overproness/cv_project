"""Multi-view image encoder.

Extracts a per-view patch-token feature grid from a Vision Transformer
(DINOv2 by default) and fuses information across reference views with a
small cross-view transformer block. Output is a fused identity feature
suitable for a UV-space parameter decoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOutput:
    tokens: torch.Tensor       # (B, V, N_patch, C) per-view patch tokens
    fused: torch.Tensor        # (B, N_patch, C)   cross-view fused tokens
    cls: torch.Tensor          # (B, C)            global identity vector


class MultiViewViTEncoder(nn.Module):
    """ViT/DINOv2 backbone shared across reference views + cross-view fusion."""

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        pretrained: bool = True,
        out_dim: int = 768,
        num_ref_views: int = 4,
        cross_view_attn_layers: int = 2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_ref_views = num_ref_views
        self.out_dim = out_dim

        self.backbone = self._build_backbone(backbone, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Learnable per-view positional embedding (added to patch tokens
        # before cross-view attention so the network can distinguish views).
        self.view_embed = nn.Parameter(torch.zeros(1, num_ref_views, 1, out_dim))
        nn.init.trunc_normal_(self.view_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=8,
            dim_feedforward=out_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.cross_view = nn.TransformerEncoder(encoder_layer, num_layers=cross_view_attn_layers)

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> nn.Module:
        if name.startswith("dinov2"):
            # torch.hub provides DINOv2 with patch tokens via forward_features.
            return torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
        raise ValueError(f"Unsupported backbone: {name}")

    def _extract_tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B*V, 3, H, W) -> (B*V, N_patch, C)."""
        # DINOv2 requires H and W to be multiples of patch_size (14).
        h, w = imgs.shape[-2:]
        pad_h = (14 - h % 14) % 14
        pad_w = (14 - w % 14) % 14
        if pad_h > 0 or pad_w > 0:
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h))
        feats = self.backbone.forward_features(imgs)
        # DINOv2 returns dict with 'x_norm_patchtokens' and 'x_norm_clstoken'.
        return feats["x_norm_patchtokens"]

    def forward(self, ref_imgs: torch.Tensor) -> EncoderOutput:
        """ref_imgs: (B, V, 3, H, W) with V <= num_ref_views."""
        b, v, _, h, w = ref_imgs.shape
        assert v <= self.num_ref_views, f"got {v} views, max {self.num_ref_views}"

        tokens = self._extract_tokens(ref_imgs.flatten(0, 1))      # (B*V, N, C)
        n_patch, c = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(b, v, n_patch, c)
        tokens = tokens + self.view_embed[:, :v]

        # Flatten (V, N) into a single sequence for cross-view attention.
        seq = tokens.reshape(b, v * n_patch, c)
        seq = self.cross_view(seq)
        fused = seq.view(b, v, n_patch, c).mean(dim=1)             # (B, N, C)
        cls = fused.mean(dim=1)                                    # (B, C)

        return EncoderOutput(tokens=tokens, fused=fused, cls=cls)
