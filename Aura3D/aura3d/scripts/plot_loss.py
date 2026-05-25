"""Parse train.log and save a loss-curve figure.

Usage:
    python -m aura3d.scripts.plot_loss \
        --log  /mnt/d/GitHub/cv_project/runs/stage1_real/train.log \
        --out  /mnt/d/GitHub/cv_project/runs/stage1_real/loss_curve.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_log(log_path: str) -> tuple[list[int], list[float], list[float], list[float]]:
    steps, totals, l1s, ssims = [], [], [], []
    pat = re.compile(
        r'\[step\s+(\d+)\]\s+total=([\d.]+)\s+l1=([\d.]+)\s+ssim=([\d.]+)'
    )
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                totals.append(float(m.group(2)))
                l1s.append(float(m.group(3)))
                ssims.append(float(m.group(4)))
    return steps, totals, l1s, ssims


def smooth(values: list[float], window: int = 50) -> list[float]:
    """Simple box-car moving average."""
    arr = np.array(values, dtype=np.float32)
    kernel = np.ones(window) / window
    pad = np.pad(arr, (window // 2, window // 2), mode='edge')
    return list(np.convolve(pad, kernel, mode='valid')[:len(arr)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="loss_curve.png")
    parser.add_argument("--smooth-window", type=int, default=100,
                        help="Moving-average window (in log lines, not steps)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    steps, totals, l1s, ssims = parse_log(args.log)
    if not steps:
        print("No step lines found in log.")
        return

    # Remove duplicate steps (happens when training is resumed — earlier run
    # entries overlap with the new run's step numbers).
    # Keep only the last occurrence of each step so we get a monotone curve.
    seen: dict[int, int] = {}
    for i, s in enumerate(steps):
        seen[s] = i
    idx = sorted(seen.values())
    steps   = [steps[i]  for i in idx]
    totals  = [totals[i] for i in idx]
    l1s     = [l1s[i]    for i in idx]
    ssims   = [ssims[i]  for i in idx]

    steps_k = [s / 1000.0 for s in steps]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Aura-3D Stage-1 Training Loss", fontsize=14, fontweight="bold")

    for ax, raw, label, color in zip(
        axes,
        [totals, l1s, ssims],
        ["Total loss (L1 + SSIM)", "L1 loss", "SSIM loss"],
        ["#2196F3", "#4CAF50", "#FF9800"],
    ):
        sm = smooth(raw, window=args.smooth_window)
        ax.plot(steps_k, raw, alpha=0.25, linewidth=0.6, color=color)
        ax.plot(steps_k, sm, linewidth=1.8, color=color, label=f"{label} (smoothed)")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fk"))

    axes[-1].set_xlabel("Training step (×1000)", fontsize=10)

    # Annotate best total loss
    best_idx = int(np.argmin(totals))
    axes[0].axvline(steps_k[best_idx], color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].annotate(
        f"best={totals[best_idx]:.4f}\n@ {steps[best_idx]//1000}k",
        xy=(steps_k[best_idx], totals[best_idx]),
        xytext=(steps_k[best_idx] + max(steps_k) * 0.03, totals[best_idx] + 0.01),
        fontsize=7.5, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved loss curve → {out_path}")
    print(f"  Total steps logged : {len(steps)}")
    print(f"  Step range         : {steps[0]} – {steps[-1]}")
    print(f"  Best total loss    : {min(totals):.4f} @ step {steps[best_idx]}")


if __name__ == "__main__":
    main()
