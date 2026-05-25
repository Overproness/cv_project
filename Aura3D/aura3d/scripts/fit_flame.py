"""Run DECA / FLAME tracking on NeRSemble frames and dump per-frame npz.

Output layout (matches what `NeRSemblePhase1Dataset` expects):

    <root>/<participant>/flame_tracking/<sequence>/<timestep:06d>.npz

Keys: shape, expression, pose, gaze (optional).

This script is a thin orchestrator — the heavy lifting is delegated to
DECA (Feng et al.). DECA is loaded lazily so the rest of the project
does not require it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _build_deca(device: str):
    from decalib.deca import DECA  # type: ignore
    from decalib.utils.config import cfg as deca_cfg  # type: ignore

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    return DECA(config=deca_cfg, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="NeRSemble download root.")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--ref-camera", default=None,
                        help="Camera to use for tracking (front-facing recommended).")
    parser.add_argument("--n-shape", type=int, default=100)
    parser.add_argument("--n-exp", type=int, default=50)
    parser.add_argument("--n-pose", type=int, default=6)
    parser.add_argument("--max-timesteps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from nersemble_data.data.nersemble_data import (  # type: ignore
        NeRSembleParticipantDataManager,
    )

    out_dir = Path(args.root) / args.participant / "flame_tracking" / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    pdm = NeRSembleParticipantDataManager(args.root, int(args.participant))
    cams = pdm.list_cameras(args.sequence)
    ref_cam = args.ref_camera or cams[len(cams) // 2]

    deca = _build_deca(args.device)

    n_written = 0
    for t in range(args.max_timesteps):
        try:
            img_np = pdm.load_image(args.sequence, ref_cam, t,
                                     apply_color_correction=False)
        except (IndexError, EOFError, StopIteration):
            # Video ran out of frames
            break
        except Exception as e:
            print(f"  WARNING: could not load frame t={t}: {e}")
            break
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        # load_image with as_uint8=False already returns [0,1] floats
        if img.max() > 1.5:
            img = img / 255.0
        img = img.unsqueeze(0).to(args.device)
        # DECA ResNet encoder expects 224x224 face crops
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            codedict = deca.encode(img)

        np.savez(
            out_dir / f"{t:06d}.npz",
            shape=codedict["shape"].cpu().numpy()[0, : args.n_shape],
            expression=codedict["exp"].cpu().numpy()[0, : args.n_exp],
            pose=codedict["pose"].cpu().numpy()[0, : args.n_pose],
            # DECA does not estimate gaze; leave zeros — Phase-2 plugs in
            # an external gaze estimator (xgaze / mpiifacegaze) later.
            gaze=np.zeros(2, dtype=np.float32),
        )
        n_written += 1
        if t % 10 == 0:
            print(f"  fit t={t:04d}")

    print(f"Wrote {n_written} frames to {out_dir}")


if __name__ == "__main__":
    main()
