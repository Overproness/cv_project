"""Photometric losses for Aura-3D training.

Phase-1: L1 + SSIM.
Phase-2: adds LPIPS perceptual loss (lazy-imported so Phase-1 machines
without the `lpips` package still work if w_lpips=0).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).unsqueeze(0)  # (1, W)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM for (B, C, H, W) images in [0, 1]. Returns scalar mean SSIM."""
    c = img1.shape[1]
    device, dtype = img1.device, img1.dtype
    g1d = _gaussian_window(window_size, 1.5, device, dtype)
    window = (g1d.T @ g1d).expand(c, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=c)
    mu2 = F.conv2d(img2, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)
    )
    return ssim_map.mean()


class PhotometricLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_ssim: float = 0.2,
        w_lpips: float = 0.0,
        lpips_net: str = "vgg",
    ) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self._lpips_fn: Optional[nn.Module] = None
        if w_lpips > 0.0:
            try:
                import lpips  # type: ignore
                self._lpips_fn = lpips.LPIPS(net=lpips_net)
                # Freeze LPIPS weights — we never want to update VGG
                for p in self._lpips_fn.parameters():
                    p.requires_grad_(False)
            except ImportError:
                raise ImportError(
                    "lpips package not found. Install it with: pip install lpips"
                )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self._lpips_fn is not None:
            self._lpips_fn = self._lpips_fn.to(*args, **kwargs)
        return self

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """pred / target: (3, H, W) or (B, 3, H, W), values in [0, 1]."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        l1 = F.l1_loss(pred, target)
        s = 1.0 - ssim(pred, target)
        total = self.w_l1 * l1 + self.w_ssim * s

        out: dict = {"total": total, "l1": l1.detach(), "ssim": s.detach()}

        if self.w_lpips > 0.0 and self._lpips_fn is not None:
            # Downsample to 256×256 before LPIPS to save ~4× activation memory
            # (LPIPS was designed at 64px; 256px retains plenty of perceptual detail).
            if pred.shape[-1] > 256:
                p_small = F.interpolate(pred, size=(256, 256), mode="bilinear", align_corners=False)
                t_small = F.interpolate(target.detach(), size=(256, 256), mode="bilinear", align_corners=False)
            else:
                p_small, t_small = pred, target.detach()
            # LPIPS expects inputs in [-1, 1]
            p_norm = p_small * 2.0 - 1.0
            t_norm = t_small * 2.0 - 1.0
            lp = self._lpips_fn(p_norm, t_norm).mean()
            total = total + self.w_lpips * lp
            out["total"] = total
            out["lpips"] = lp.detach()

        return out
