"""Render an animated MP4 driving one NeRSemble subject through a timestep sequence.

For each timestep, the FLAME params for that frame are loaded, the identity
is encoded once from one reference camera, then the model is animated for
every timestep and the resulting frames are assembled into a side-by-side
[GT | PRED] MP4.

Usage:
    python -m aura3d.scripts.render_video \
        --config   aura3d/configs/aura3d_default.yaml \
        --ckpt     /mnt/d/GitHub/cv_project/runs/stage1_real/best.pt \
        --out-dir  /mnt/d/GitHub/cv_project/runs/stage1_real/videos \
        --participant 038 \
        --sequence    EXP-1-head \
        --render-cam  222200042 \
        --device      cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def _to_bgr(t: torch.Tensor) -> np.ndarray:
    import cv2
    arr = (t.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aura-3D: render animated video")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--out-dir",     default="runs/stage1_real/videos")
    parser.add_argument("--participant", default="038")
    parser.add_argument("--sequence",    default="EXP-1-head")
    parser.add_argument("--render-cam",  default=None,
                        help="Camera to render. Defaults to the first cam in the sequence.")
    parser.add_argument("--ref-cam",     default=None,
                        help="Camera used as reference view for identity encoding. "
                             "Defaults to second cam in the sequence.")
    parser.add_argument("--n-frames",    type=int, default=50)
    parser.add_argument("--fps",         type=int, default=10)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    import cv2

    # Seed PyTorch RNG before model construction so that tri_uvs (the UV
    # centroid buffer in the decoder) is generated deterministically.  This
    # must be consistent with the fixed-seed path in Aura3DModel.__init__
    # (which now uses its own Generator(seed=0), so this global seed is only
    # a belt-and-suspenders guard).
    torch.manual_seed(42)

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

    # ---------------------------------------------------------------- dataset helpers
    from nersemble_data.data.nersemble_data import (  # type: ignore
        NeRSembleParticipantDataManager,
    )

    data_cfg  = cfg["data"]
    flame_cfg = cfg["model"]["flame"]
    img_size  = data_cfg["image_size"]
    pid       = args.participant

    print(f"Loading participant {pid} / sequence {args.sequence} …")
    pdm   = NeRSembleParticipantDataManager(str(data_cfg["root"]), int(pid))
    cams  = pdm.list_cameras(args.sequence)
    print(f"  Available cameras: {cams}")

    render_cam = args.render_cam if args.render_cam else cams[0]
    ref_cam    = args.ref_cam    if args.ref_cam    else cams[1]
    if render_cam not in cams:
        raise ValueError(f"render-cam {render_cam!r} not in {cams}")
    if ref_cam not in cams:
        raise ValueError(f"ref-cam {ref_cam!r} not in {cams}")
    print(f"  Render cam : {render_cam}")
    print(f"  Reference  : {ref_cam}")

    calib = pdm.load_camera_calibration()

    def load_K_w2c(cam: str) -> tuple[torch.Tensor, torch.Tensor]:
        K_np  = np.array(calib.intrinsics).astype("float32")
        w2c_np = np.array(calib.world_2_cam[cam]).astype("float32")
        K = torch.from_numpy(K_np)
        # Rescale intrinsics to resized image (NeRSemble native w=3208).
        scale = img_size / 3208.0
        K = K.clone()
        K[0, 0] *= scale; K[1, 1] *= scale
        K[0, 2] *= scale; K[1, 2] *= scale
        return K, torch.from_numpy(w2c_np)

    def load_img(cam: str, t: int) -> torch.Tensor:
        import cv2
        img = pdm.load_image(args.sequence, cam, t, as_uint8=True,
                             apply_color_correction=False)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        t_img = torch.from_numpy(img.astype("float32")) / 255.0
        return t_img.permute(2, 0, 1).contiguous()

    def load_flame(t: int) -> dict:
        path = (Path(data_cfg["root"]) / f"{int(pid):03d}" / "flame_tracking"
                / args.sequence / f"{t:06d}.npz")
        n_shape = flame_cfg["n_shape"]
        n_exp   = flame_cfg["n_exp"]
        n_pose  = flame_cfg["n_pose"]
        if not path.exists():
            return {k: torch.zeros(n) for k, n in
                    [("shape", n_shape), ("expression", n_exp),
                     ("pose", n_pose), ("gaze", 2)]}
        d = np.load(path)
        return {
            "shape":      torch.from_numpy(d["shape"]).float()[:n_shape],
            "expression": torch.from_numpy(d["expression"]).float()[:n_exp],
            "pose":       torch.from_numpy(d["pose"]).float()[:n_pose],
            "gaze":       torch.from_numpy(d["gaze"]).float() if "gaze" in d.files
                          else torch.zeros(2),
        }

    # ---------------------------------------------------------------- render camera setup
    K_render, w2c_render = load_K_w2c(render_cam)
    # Move to the same device as the model so rasterizer matrices are on CUDA.
    K_render  = K_render.to(args.device)
    w2c_render = w2c_render.to(args.device)

    # ---------------------------------------------------------------- encode identity (t=0)
    print("Encoding identity from reference view at t=0 …")
    num_ref = cfg["model"]["encoder"]["num_ref_views"]
    # Sample `num_ref` distinct cameras if possible, else tile the single ref cam.
    ref_cams_for_enc = []
    for c in cams:
        if c != render_cam and len(ref_cams_for_enc) < num_ref:
            ref_cams_for_enc.append(c)
    while len(ref_cams_for_enc) < num_ref:
        ref_cams_for_enc.append(ref_cams_for_enc[0])
    ref_imgs = torch.stack([load_img(c, 0) for c in ref_cams_for_enc])  # (V, 3, H, W)
    ref_batch = ref_imgs.unsqueeze(0).to(args.device)   # (1, V, 3, H, W)
    print(f"  Reference cameras : {ref_cams_for_enc}")
    with torch.no_grad():
        identity = model.encode_identity(ref_batch)
    print("  Identity encoded.")

    # ---------------------------------------------------------------- render loop
    n_frames = min(args.n_frames, 50)
    video_path = out_dir / f"pid{pid}_{args.sequence}_{render_cam}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    print(f"Rendering {n_frames} frames → {video_path}")
    for t in range(n_frames):
        gt_img = load_img(render_cam, t)         # (3, H, W)  CPU
        flame  = load_flame(t)

        with torch.no_grad():
            camera = make_render_camera(K_render, w2c_render, img_size, img_size)
            out = model.animate(
                identity=identity,
                shape=flame["shape"].unsqueeze(0).to(args.device),
                expression=flame["expression"].unsqueeze(0).to(args.device),
                pose=flame["pose"].unsqueeze(0).to(args.device),
                gaze=flame["gaze"].unsqueeze(0).to(args.device),
                camera=camera,
            )
        pred = out["rgb"].clamp(0.0, 1.0)   # (3, H, W)

        gt_bgr   = _to_bgr(gt_img)
        pred_bgr = _to_bgr(pred)

        # Add labels
        font, scale, color, thick = (cv2.FONT_HERSHEY_SIMPLEX,
                                     img_size / 600, (255, 255, 255), 1)
        cv2.putText(gt_bgr,   f"GT   t={t}", (4, 20), font, scale, color, thick)
        cv2.putText(pred_bgr, f"PRED t={t}", (4, 20), font, scale, color, thick)

        panel = np.concatenate([gt_bgr, pred_bgr], axis=1)  # side by side

        if writer is None:
            h, w = panel.shape[:2]
            writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))

        writer.write(panel)

        if (t + 1) % 10 == 0 or t == n_frames - 1:
            print(f"  frame {t+1}/{n_frames}")

    if writer:
        writer.release()
    print(f"Video saved → {video_path}")


if __name__ == "__main__":
    main()
