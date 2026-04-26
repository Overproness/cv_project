"""Phase-1 overfit trainer.

Goal: take 1-3 NeRSemble subjects, drive a single Aura3DModel for tens
of thousands of steps, and verify that the photometric loss closes —
i.e. the encoder + decoder + binding + deformation + GERR + rasterizer
are all wired correctly. This is a SANITY trainer, not a research one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..data.datasets.nersemble import NeRSemblePhase1Dataset, collate_samples
from ..losses.photometric import PhotometricLoss
from ..models.aura3d_model import Aura3DModel
from ..utils.camera import make_render_camera


@dataclass
class TrainerState:
    step: int = 0
    best_loss: float = float("inf")


class OverfitTrainer:
    def __init__(self, cfg: dict, model: Aura3DModel, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.state = TrainerState()

        train_cfg = cfg["training"]
        data_cfg = cfg["data"]
        self.dataset = NeRSemblePhase1Dataset(
            root=data_cfg["root"],
            num_ref_views=train_cfg["num_ref_views"],
            image_size=data_cfg["image_size"],
            n_shape=cfg["model"]["flame"]["n_shape"],
            n_exp=cfg["model"]["flame"]["n_exp"],
            n_pose=cfg["model"]["flame"]["n_pose"],
            synthetic=data_cfg.get("synthetic", False),
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 0),
            collate_fn=collate_samples,
            pin_memory=device == "cuda",
        )

        # Per-component LRs as specified in the config.
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(),  "lr": train_cfg["lr_encoder"]},
                {"params": self.model.decoder.parameters(),  "lr": train_cfg["lr_decoder"]},
                {"params": self.model.face_deform.parameters(), "lr": train_cfg["lr_deform"]},
                {"params": [p for n, p in self.model.named_parameters()
                            if p.requires_grad
                            and not n.startswith(("encoder.", "decoder.", "face_deform.",
                                                  "flame.", "binding."))],
                 "lr": train_cfg["lr_gaussian_attrs"]},
            ]
        )

        losses_cfg = train_cfg["losses"]
        self.photo = PhotometricLoss(w_l1=losses_cfg["l1"], w_ssim=losses_cfg["ssim"])

        self.steps = train_cfg["steps"]
        self.log_every = train_cfg.get("log_every", 50)

    # -------------------------------------------------------------- core step
    def _step(self, batch: dict) -> dict:
        b = batch["ref_imgs"].shape[0]
        assert b == 1, "OverfitTrainer assumes batch_size=1 (one camera per step)."

        ref_imgs = batch["ref_imgs"].to(self.device)
        target = batch["target_img"].to(self.device)
        K = batch["target_K"].to(self.device)[0]
        w2c = batch["target_w2c"].to(self.device)[0]

        identity = self.model.encode_identity(ref_imgs)
        camera = make_render_camera(K, w2c, target.shape[-2], target.shape[-1])
        out = self.model.animate(
            identity=identity,
            shape=batch["shape"].to(self.device),
            expression=batch["expression"].to(self.device),
            pose=batch["pose"].to(self.device),
            gaze=batch["gaze"].to(self.device),
            camera=camera,
        )
        pred = out["rgb"]                         # (3, H, W)
        loss_dict = self.photo(pred, target[0])
        return loss_dict

    def fit(self, ckpt_dir: Optional[str] = None) -> None:
        ckpt_path = Path(ckpt_dir) if ckpt_dir else None
        if ckpt_path:
            ckpt_path.mkdir(parents=True, exist_ok=True)

        self.model.train()
        data_iter = iter(self.loader)
        while self.state.step < self.steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step(batch)
            loss["total"].backward()
            self.optimizer.step()
            self.state.step += 1

            if self.state.step % self.log_every == 0:
                print(
                    f"[step {self.state.step:>6}] "
                    f"total={loss['total'].item():.4f}  "
                    f"l1={loss['l1'].item():.4f}  "
                    f"ssim={loss['ssim'].item():.4f}"
                )
                if loss["total"].item() < self.state.best_loss:
                    self.state.best_loss = loss["total"].item()
                    if ckpt_path:
                        torch.save(
                            {"model": self.model.state_dict(),
                             "step": self.state.step},
                            ckpt_path / "best.pt",
                        )
