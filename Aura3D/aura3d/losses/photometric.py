"""Photometric losses for Phase-1 overfitting.

L1 + SSIM is sufficient to drive a 3DGS overfit to convergence. LPIPS
and gaze / id losses are added in Phase-2; they are intentionally NOT
imported here so Phase-1 has zero heavy dependencies.
"""
from __future__ import annotations

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
    def __init__(self, w_l1: float = 1.0, w_ssim: float = 0.2) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l1 = F.l1_loss(pred, target)
        s = 1.0 - ssim(pred.unsqueeze(0) if pred.dim() == 3 else pred,
                       target.unsqueeze(0) if target.dim() == 3 else target)
        total = self.w_l1 * l1 + self.w_ssim * s
        return {"total": total, "l1": l1.detach(), "ssim": s.detach()}
