"""Phase-1 launcher: overfit Aura-3D on a small NeRSemble subset.

Usage:
    python -m aura3d.scripts.train_stage1_overfit \
        --config aura3d/configs/aura3d_default.yaml \
        --ckpt-dir runs/stage1

Use `--synthetic` for a no-data smoke test. Real training now fails loudly
when `data.root` is missing, so a bad machine-specific path cannot silently
train on random images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aura3d.models.aura3d_model import Aura3DModel
from aura3d.training.trainer import OverfitTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default="runs/stage1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the synthetic data fallback (smoke test).")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training.steps from the config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.synthetic:
        cfg["data"]["synthetic"] = True
    if args.steps is not None:
        cfg["training"]["steps"] = args.steps

    model = Aura3DModel(cfg)
    trainer = OverfitTrainer(cfg, model, device=args.device)
    trainer.fit(ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
