"""Phase-2 perceptual & task losses.

LPIPS is provided by the `lpips` package (lazy-loaded). Identity loss
uses a frozen ArcFace embedding (lazy-loaded too) — for now we ship a
clean interface and a lightweight fallback that returns zero so Phase-1
trainers don't break if these packages are not installed.

Gaze loss: angular error between the gaze direction predicted by an
external gaze estimator on the rendered image and the ground-truth
(pitch, yaw) used to drive the avatar. The estimator is treated as a
black-box module that consumes (B, 3, H, W) and returns (B, 2).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    """Wraps the `lpips` package; degrades to L2 if not installed."""

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        try:
            import lpips  # type: ignore

            self.net = lpips.LPIPS(net=net)
            self.available = True
        except Exception:
            self.net = None
            self.available = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return F.mse_loss(pred, target)
        # lpips expects [-1, 1].
        return self.net(pred * 2 - 1, target * 2 - 1).mean()


class IdentityLoss(nn.Module):
    """Cosine distance between ArcFace embeddings of pred and target.

    A user-supplied `embedder` callable (e.g. one of GazeGaussian's
    `face_recognition` model handlers) is preferred. If none is given,
    we return zero so the training loop still works.
    """

    def __init__(self, embedder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.embedder = embedder

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.embedder is None:
            return pred.new_zeros(())
        e_p = F.normalize(self.embedder(pred), dim=-1)
        e_t = F.normalize(self.embedder(target), dim=-1)
        return (1.0 - (e_p * e_t).sum(dim=-1)).mean()


class GazeLoss(nn.Module):
    """Angular error in radians between predicted and target gaze.

    `estimator` consumes a (B, 3, H, W) image in [0, 1] and returns
    (B, 2) (pitch, yaw). When unset, returns zero.
    """

    def __init__(self, estimator: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.estimator = estimator

    @staticmethod
    def _pitchyaw_to_vec(py: torch.Tensor) -> torch.Tensor:
        pitch, yaw = py[:, 0], py[:, 1]
        x = torch.cos(pitch) * torch.sin(yaw)
        y = torch.sin(pitch)
        z = -torch.cos(pitch) * torch.cos(yaw)
        return F.normalize(torch.stack([x, y, z], dim=-1), dim=-1)

    def forward(self, pred_img: torch.Tensor, target_pitchyaw: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            return pred_img.new_zeros(())
        pred_py = self.estimator(pred_img)
        v_p = self._pitchyaw_to_vec(pred_py)
        v_t = self._pitchyaw_to_vec(target_pitchyaw)
        cos = (v_p * v_t).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos).mean()


class FLAMERegLoss(nn.Module):
    """Tikhonov regulariser on FLAME parameters."""

    def forward(self, shape: torch.Tensor, expression: torch.Tensor) -> torch.Tensor:
        return (shape ** 2).mean() + (expression ** 2).mean()
