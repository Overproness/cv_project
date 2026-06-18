"""Phase-1 overfit trainer.

Goal: take 1-3 NeRSemble subjects, drive a single Aura3DModel for tens
of thousands of steps, and verify that the photometric loss closes —
i.e. the encoder + decoder + binding + deformation + GERR + rasterizer
are all wired correctly. This is a SANITY trainer, not a research one.
"""
from __future__ import annotations

import math
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
        print(
            f"Dataset: root={data_cfg['root']}  synthetic={self.dataset.synthetic}  "
            f"samples={len(self.dataset)}",
            flush=True,
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
        self.photo = PhotometricLoss(
            w_l1=losses_cfg["l1"],
            w_ssim=losses_cfg["ssim"],
            w_lpips=losses_cfg.get("lpips", 0.0),
            lpips_net=losses_cfg.get("lpips_net", "vgg"),
        )
        self.photo = self.photo.to(device)
        self.w_scale_reg = losses_cfg.get("scale_reg", 1e-2)

        self.steps = train_cfg["steps"]
        self.log_every = train_cfg.get("log_every", 50)
        self.ckpt_every = train_cfg.get("ckpt_every", 500)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

        # Cosine decay: LR starts at 1.0× and decays to eta_min_factor× over all steps.
        eta_min = train_cfg.get("lr_eta_min_factor", 0.05)
        cosine_fn = lambda s: eta_min + (1.0 - eta_min) * 0.5 * (
            1.0 + math.cos(math.pi * min(s, self.steps) / self.steps)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cosine_fn)

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

        # Soft penalty on large Gaussian log-scales to prevent explosion.
        # Penalises any log-scale above -3 (real scale > 0.05 m, our renderer cap),
        # leaving normal face-detail values in [-7, -3] completely untouched.
        gaussians = out["gaussians"]
        scale_excess = gaussians.scale.clamp_min(-1.897) - (-1.897)  # 0 when log_scale ≤ log(0.15m)
        scale_reg = self.w_scale_reg * scale_excess.pow(2).mean()
        loss_dict["scale_reg"] = scale_reg
        loss_dict["total"] = loss_dict["total"] + scale_reg

        return loss_dict

    def resume(self, ckpt_path: Path) -> None:
        """Load a checkpoint saved by fit(). Skips silently if none found."""
        best = ckpt_path / "best.pt"
        latest = ckpt_path / "latest.pt"
        # Prefer latest only if it also has scheduler state (i.e. saved by this version).
        # Old latest.pt files (no scheduler key) have stale Adam momentum that causes
        # permanent scale collapse when resumed — always fall back to best.pt in that case.
        src = None
        if latest.exists():
            probe = torch.load(latest, map_location="cpu", weights_only=False)
            if "scheduler" in probe:
                src = latest
        if src is None and best.exists():
            src = best
        if src is None:
            return
        ckpt = torch.load(src, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  [NOTE] Checkpoint missing keys (expected for old ckpts): {missing}")
        if "scheduler" in ckpt:
            # Full checkpoint: restore optimizer + scheduler state exactly.
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Checkpoint from before cosine-decay was added — keep fresh Adam state
            # to avoid stale momentum driving scale collapse, but advance the LR
            # schedule to the correct position so decay continues smoothly.
            self.scheduler.last_epoch = ckpt["step"]
        self.state.step = ckpt["step"]
        self.state.best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from {src.name} at step {self.state.step} "
              f"(best_loss={self.state.best_loss:.4f})", flush=True)

    def fit(self, ckpt_dir: Optional[str] = None) -> None:
        ckpt_path = Path(ckpt_dir) if ckpt_dir else None
        if ckpt_path:
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.resume(ckpt_path)

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

            # Skip NaN/Inf losses to avoid corrupting parameters
            if not torch.isfinite(loss["total"]):
                print(f"WARNING: non-finite loss at step {self.state.step}, skipping.", flush=True)
                self.state.step += 1
                continue

            loss["total"].backward()

            # Clip gradients to prevent scale/opacity collapse from a bad batch
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.state.step += 1

            if self.state.step % self.log_every == 0:
                lpips_str = (
                    f"  lpips={loss['lpips'].item():.4f}" if "lpips" in loss else ""
                )
                print(
                    f"[step {self.state.step:>6}] "
                    f"total={loss['total'].item():.4f}  "
                    f"l1={loss['l1'].item():.4f}  "
                    f"ssim={loss['ssim'].item():.4f}  "
                    f"scale_reg={loss['scale_reg'].item():.5f}"
                    f"{lpips_str}",
                    flush=True,
                )
                if loss["total"].item() < self.state.best_loss:
                    self.state.best_loss = loss["total"].item()
                    if ckpt_path:
                        torch.save(
                            {"model": self.model.state_dict(),
                             "optimizer": self.optimizer.state_dict(),
                             "scheduler": self.scheduler.state_dict(),
                             "step": self.state.step,
                             "best_loss": self.state.best_loss},
                            ckpt_path / "best.pt",
                        )

            # Periodic checkpoint (independent of best-loss tracking)
            if ckpt_path and self.state.step % self.ckpt_every == 0:
                torch.save(
                    {"model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict(),
                     "step": self.state.step,
                     "best_loss": self.state.best_loss},
                    ckpt_path / "latest.pt",
                )
