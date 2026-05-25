"""Stage-1 evaluation script.

Loads best.pt, runs the full Aura-3D forward pass on held-out
(participant, sequence, timestep, camera) samples from NeRSemble, and
reports SSIM / PSNR / LPIPS.  Rendered frames and ground-truth frames are
saved side-by-side as PNGs.

Usage:
    python -m aura3d.scripts.evaluate \
        --config  aura3d/configs/aura3d_default.yaml \
        --ckpt    /mnt/d/GitHub/cv_project/runs/stage1_real/best.pt \
        --out-dir /mnt/d/GitHub/cv_project/runs/stage1_real/eval \
        --n-samples 50
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Optional LPIPS — graceful fallback if not installed.
# ---------------------------------------------------------------------------
try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM between two (3, H, W) float32 tensors in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(pred.unsqueeze(0), 11, 1, 5)
    mu2 = F.avg_pool2d(gt.unsqueeze(0), 11, 1, 5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sig1_sq = F.avg_pool2d(pred.unsqueeze(0) ** 2, 11, 1, 5) - mu1_sq
    sig2_sq = F.avg_pool2d(gt.unsqueeze(0) ** 2, 11, 1, 5) - mu2_sq
    sig12   = F.avg_pool2d(pred.unsqueeze(0) * gt.unsqueeze(0), 11, 1, 5) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return float((num / den).mean().item())


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = float(F.mse_loss(pred, gt).item())
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _save_side_by_side(pred: torch.Tensor, gt: torch.Tensor, path: Path,
                       ssim: float, psnr: float) -> None:
    """Save a side-by-side comparison image: [GT | PRED | DIFF]."""
    try:
        import cv2
    except ImportError:
        return
    def t2bgr(t):
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gt_bgr   = t2bgr(gt)
    pred_bgr = t2bgr(pred)
    diff = np.abs(gt_bgr.astype(np.int16) - pred_bgr.astype(np.int16))
    diff_bgr = np.clip(diff * 4, 0, 255).astype(np.uint8)  # 4× amplified diff
    # Annotate
    h = gt_bgr.shape[0]
    font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, h / 600, (255, 255, 255), 1
    cv2.putText(gt_bgr,   "GT",   (4, 20), font, scale, color, thick)
    cv2.putText(pred_bgr, "PRED", (4, 20), font, scale, color, thick)
    cv2.putText(diff_bgr, f"SSIM={ssim:.3f}  PSNR={psnr:.1f}", (4, 20), font, scale, color, thick)
    panel = np.concatenate([gt_bgr, pred_bgr, diff_bgr], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aura-3D Stage-1 evaluation")
    parser.add_argument("--config",    required=True, help="Path to aura3d_default.yaml")
    parser.add_argument("--ckpt",      required=True, help="Path to best.pt (or latest.pt)")
    parser.add_argument("--out-dir",   default="runs/stage1_real/eval")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of random evaluation samples")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-lpips",  action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ model
    from aura3d.models.aura3d_model import Aura3DModel
    from aura3d.utils.camera import make_render_camera

    print("Building model …")
    model = Aura3DModel(cfg)
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [NOTE] Checkpoint missing keys (expected for old ckpts): {missing}")
    model.eval().to(args.device)
    print(f"Loaded checkpoint  step={ckpt['step']}  best_loss={ckpt.get('best_loss', float('nan')):.4f}")

    # ------------------------------------------------------------------ LPIPS
    lpips_fn = None
    if _LPIPS_AVAILABLE and not args.no_lpips:
        lpips_fn = _lpips_lib.LPIPS(net="alex").to(args.device)
        lpips_fn.eval()
        print("LPIPS (AlexNet) enabled.")
    else:
        print("LPIPS not available — skipping (install with `pip install lpips`).")

    # ----------------------------------------------------------------- dataset
    from aura3d.data.datasets.nersemble import NeRSemblePhase1Dataset, collate_samples
    from torch.utils.data import DataLoader, Subset

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    flame_cfg = cfg["model"]["flame"]

    print("Loading dataset index …")
    dataset = NeRSemblePhase1Dataset(
        root=data_cfg["root"],
        num_ref_views=train_cfg["num_ref_views"],
        image_size=data_cfg["image_size"],
        n_shape=flame_cfg["n_shape"],
        n_exp=flame_cfg["n_exp"],
        n_pose=flame_cfg["n_pose"],
        synthetic=data_cfg.get("synthetic", False),
    )
    print(f"Dataset size: {len(dataset)} samples")

    n = min(args.n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=collate_samples)

    # ----------------------------------------------------------------- eval loop
    ssim_vals, psnr_vals, lpips_vals = [], [], []

    print(f"Evaluating {n} samples …\n{'─'*60}")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ref_imgs = batch["ref_imgs"].to(args.device)    # (1, V, 3, H, W)
            target   = batch["target_img"].to(args.device)  # (1, 3, H, W)
            K        = batch["target_K"].to(args.device)[0]
            w2c      = batch["target_w2c"].to(args.device)[0]
            shape    = batch["shape"].to(args.device)
            expr     = batch["expression"].to(args.device)
            pose     = batch["pose"].to(args.device)
            gaze     = batch["gaze"].to(args.device)

            H, W = target.shape[-2], target.shape[-1]
            camera = make_render_camera(K, w2c, H, W)

            identity = model.encode_identity(ref_imgs)
            out = model.animate(
                identity=identity,
                shape=shape,
                expression=expr,
                pose=pose,
                gaze=gaze,
                camera=camera,
            )
            pred = out["rgb"].clamp(0.0, 1.0)   # (3, H, W)
            gt   = target[0]                    # (3, H, W)

            s = _ssim(pred, gt)
            p = _psnr(pred, gt)
            ssim_vals.append(s)
            psnr_vals.append(p)

            lv = float("nan")
            if lpips_fn is not None:
                lv = float(lpips_fn(
                    pred.unsqueeze(0) * 2 - 1,
                    gt.unsqueeze(0)   * 2 - 1,
                ).item())
                lpips_vals.append(lv)

            # collate_samples turns list[meta_dict] into list[meta_dict];
            # batch["meta"] is a list with one entry per batch item.
            meta_list = batch.get("meta", [{}])
            meta0 = meta_list[0] if isinstance(meta_list, list) else meta_list
            pid = str(meta0.get("pid", "?"))
            seq = str(meta0.get("seq", "?"))
            t   = str(meta0.get("t",   "?"))
            cam = str(meta0.get("target_cam", "?"))

            lpips_str = f"  lpips={lv:.3f}" if not math.isnan(lv) else ""
            print(f"[{i+1:3d}/{n}]  pid={pid}  seq={seq}  t={t}  cam={cam}"
                  f"  ssim={s:.4f}  psnr={p:.2f}dB{lpips_str}")

            img_name = f"{i:04d}_pid{pid}_t{t}_{cam}.png"
            _save_side_by_side(pred, gt, out_dir / "frames" / img_name, s, p)

    # ----------------------------------------------------------------- summary
    mean_ssim  = float(np.mean(ssim_vals))
    mean_psnr  = float(np.mean(psnr_vals))
    mean_lpips = float(np.mean(lpips_vals)) if lpips_vals else float("nan")

    print(f"\n{'═'*60}")
    print(f"  Samples  : {n}")
    print(f"  SSIM  ↑  : {mean_ssim:.4f}  (std {float(np.std(ssim_vals)):.4f})")
    print(f"  PSNR  ↑  : {mean_psnr:.2f} dB  (std {float(np.std(psnr_vals)):.2f})")
    if not math.isnan(mean_lpips):
        print(f"  LPIPS ↓  : {mean_lpips:.4f}  (std {float(np.std(lpips_vals)):.4f})")
    print(f"{'═'*60}\n")

    # Write a results summary text file.
    summary_path = out_dir / "results.txt"
    with open(summary_path, "w") as f:
        f.write(f"checkpoint : {args.ckpt}\n")
        f.write(f"step       : {ckpt['step']}\n")
        f.write(f"best_loss  : {ckpt.get('best_loss', float('nan')):.4f}\n")
        f.write(f"n_samples  : {n}\n")
        f.write(f"ssim       : {mean_ssim:.4f}  std={float(np.std(ssim_vals)):.4f}\n")
        f.write(f"psnr       : {mean_psnr:.2f} dB  std={float(np.std(psnr_vals)):.2f}\n")
        if not math.isnan(mean_lpips):
            f.write(f"lpips      : {mean_lpips:.4f}  std={float(np.std(lpips_vals)):.4f}\n")
        f.write(f"\nper_sample:\n")
        for i, idx in enumerate(indices):
            lv_s = f"  lpips={lpips_vals[i]:.3f}" if i < len(lpips_vals) else ""
            f.write(f"  dataset_idx={idx}  ssim={ssim_vals[i]:.4f}  psnr={psnr_vals[i]:.2f}{lv_s}\n")
    print(f"Results written to {summary_path}")
    print(f"Frame panels  saved to {out_dir / 'frames'}/")


if __name__ == "__main__":
    main()
