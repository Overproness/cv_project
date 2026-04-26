"""Training-only refinement wrapper (CodeFormer / SimSwap pseudo-target).

This module is **never** used at inference — its role (per Liu et al.
EGNR) is to clean up 2D supervision targets when ground-truth multi-view
is unavailable. Given an in-the-wild RGB frame, it produces a higher-
quality reconstruction that we then treat as the target image during
loss computation.

We keep the wrapper minimal and lazy-loaded so Phase-1 trainers don't
have to install CodeFormer.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TargetRefiner(nn.Module):
    """Wraps a pretrained CodeFormer instance; identity passthrough fallback."""

    def __init__(self, ckpt_path: Optional[str] = None, fidelity: float = 0.7) -> None:
        super().__init__()
        self.fidelity = fidelity
        self.model: Optional[nn.Module] = None
        if ckpt_path is None:
            return
        try:
            # Expect users to point at a checkpoint compatible with the
            # `basicsr` CodeFormer architecture (sczhou/CodeFormer).
            from basicsr.archs.codeformer_arch import CodeFormer  # type: ignore

            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=("32", "64", "128", "256"),
            )
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = sd.get("params_ema", sd)
            net.load_state_dict(sd, strict=True)
            net.eval()
            for p in net.parameters():
                p.requires_grad = False
            self.model = net
        except Exception as exc:  # pragma: no cover — depends on user env
            print(f"[TargetRefiner] CodeFormer unavailable ({exc}); "
                  f"falling back to identity.")

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) in [0, 1]. Returns refined image in [0, 1]."""
        if self.model is None:
            return img
        x = img * 2 - 1  # CodeFormer expects [-1, 1]
        out, _ = self.model(x, w=self.fidelity, adain=True)
        return (out.clamp(-1, 1) + 1) / 2
