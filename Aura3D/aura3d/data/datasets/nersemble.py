"""NeRSemble Phase-1 dataset.

Phase-1 goal: overfit Aura-3D on 1-3 subjects, each with ~16 cameras and
~50 timesteps. We treat every (subject, timestep, camera) triple as a
training sample: K of the 16 cameras are sampled as REFERENCE views (fed
to the encoder) and one DIFFERENT camera is the target supervision view.

This dataset depends on the `nersemble_data` package for IO. If that
package is not installed (e.g. in CI / a dev laptop without the data),
we fall back to a synthetic dataset with the exact same sample schema so
the training pipeline can be smoke-tested end-to-end.

NOTE: FLAME tracking parameters (shape/expr/pose) are not part of the
raw NeRSemble release. Phase-1 expects them to live alongside the
extracted frames at:

    <root>/<participant>/flame_tracking/<sequence>/<timestep:06d>.npz

with keys: shape (n_shape,), expression (n_exp,), pose (n_pose,),
gaze (2,) optional. A separate preprocessing script (DECA / FLAME-fit)
is responsible for producing those .npz files. For smoke tests we
generate zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NeRSembleSample:
    ref_imgs: torch.Tensor          # (V, 3, H, W) reference views
    target_img: torch.Tensor        # (3, H, W)    supervision view
    target_K: torch.Tensor          # (3, 3) intrinsics for target view
    target_w2c: torch.Tensor        # (4, 4) world->cam (OpenCV)
    shape: torch.Tensor             # (n_shape,)
    expression: torch.Tensor        # (n_exp,)
    pose: torch.Tensor              # (n_pose,)
    gaze: torch.Tensor              # (2,)
    meta: dict


class NeRSemblePhase1Dataset(Dataset):
    def __init__(
        self,
        root: str,
        participants: list[str] | None = None,
        sequences: list[str] | None = None,
        num_ref_views: int = 4,
        image_size: int = 512,
        n_shape: int = 100,
        n_exp: int = 50,
        n_pose: int = 6,
        max_timesteps: int = 50,
        synthetic: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_ref_views = num_ref_views
        self.image_size = image_size
        self.n_shape, self.n_exp, self.n_pose = n_shape, n_exp, n_pose
        self.synthetic = synthetic or not self.root.exists()

        if self.synthetic:
            self._build_synthetic_index(participants, sequences, max_timesteps)
            self._dm_cache: dict = {}
            return

        # Lazy import — only required when actually loading real data.
        from nersemble_data.data.nersemble_data import (  # type: ignore
            NeRSembleDataManager,
        )

        self._dm_cache = {}
        dm = NeRSembleDataManager(str(self.root))
        # list_participants() returns ints; normalise to zero-padded strings so
        # Path arithmetic (self.root / pid / ...) works correctly.
        all_participants = [f"{p:03d}" for p in dm.list_participants()]
        if participants is None:
            participants = all_participants[:3]
        # Accept both int and str inputs from callers.
        participants = [f"{int(p):03d}" for p in participants]
        self.participants = [p for p in participants if p in all_participants]
        if not self.participants:
            raise RuntimeError(f"No requested participants found in {self.root}")

        self.index: list[tuple[str, str, int, str]] = []
        for pid in self.participants:
            pdm = self._participant_manager(pid)
            seqs = sequences or pdm.list_sequences()[:1]
            for seq in seqs:
                if seq not in pdm.list_sequences():
                    continue
                cams = pdm.list_cameras(seq)
                if len(cams) < num_ref_views + 1:
                    continue
                # NeRSemble videos are long; cap at max_timesteps for Phase-1.
                for t in range(min(max_timesteps, self._num_timesteps(pdm, seq, cams[0]))):
                    for target_cam in cams:
                        self.index.append((pid, seq, t, target_cam))

    # ------------------------------------------------------------------ utils
    def _participant_manager(self, pid: str):
        if pid in self._dm_cache:
            return self._dm_cache[pid]
        from nersemble_data.data.nersemble_data import (  # type: ignore
            NeRSembleParticipantDataManager,
        )
        m = NeRSembleParticipantDataManager(str(self.root), int(pid))
        self._dm_cache[pid] = m
        return m

    @staticmethod
    def _num_timesteps(pdm, seq: str, cam: str) -> int:
        # NeRSemble manager doesn't expose a frame count directly; probe.
        # Cap probing to something reasonable.
        for hi in (50, 100, 200, 500):
            try:
                pdm.load_image(seq, cam, hi - 1)
            except Exception:
                return hi - 1
        return 500

    def _build_synthetic_index(
        self,
        participants: list[str] | None,
        sequences: list[str] | None,
        max_timesteps: int,
    ) -> None:
        participants = participants or ["synthA", "synthB"]
        sequences = sequences or ["EXP-1-head"]
        cams = [f"cam{i:03d}" for i in range(16)]
        self.participants = participants
        self.index = [
            (p, s, t, c)
            for p in participants
            for s in sequences
            for t in range(min(max_timesteps, 8))
            for c in cams
        ]
        self._cams_synth = cams

    def __len__(self) -> int:
        return len(self.index)

    # -------------------------------------------------------------- IO helpers
    def _load_image(self, pdm, seq: str, cam: str, t: int) -> torch.Tensor:
        img = pdm.load_image(seq, cam, t, as_uint8=True,
                             apply_color_correction=False)  # HWC uint8
        return self._to_tensor(img)

    def _to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        import cv2  # lazy

        h_target = self.image_size
        if img_np.shape[0] != h_target or img_np.shape[1] != h_target:
            img_np = cv2.resize(img_np, (h_target, h_target), interpolation=cv2.INTER_AREA)
        # img_np is uint8 [0,255] from load_image(as_uint8=True)
        t = torch.from_numpy(img_np.astype('float32')) / 255.0
        return t.permute(2, 0, 1).contiguous()

    def _load_flame(self, pid: str, seq: str, t: int) -> dict:
        if self.synthetic:
            return {
                "shape": torch.zeros(self.n_shape),
                "expression": torch.zeros(self.n_exp),
                "pose": torch.zeros(self.n_pose),
                "gaze": torch.zeros(2),
            }
        path = self.root / pid / "flame_tracking" / seq / f"{t:06d}.npz"
        if not path.exists():
            return {
                "shape": torch.zeros(self.n_shape),
                "expression": torch.zeros(self.n_exp),
                "pose": torch.zeros(self.n_pose),
                "gaze": torch.zeros(2),
            }
        data = np.load(path)
        return {
            "shape": torch.from_numpy(data["shape"]).float()[: self.n_shape],
            "expression": torch.from_numpy(data["expression"]).float()[: self.n_exp],
            "pose": torch.from_numpy(data["pose"]).float()[: self.n_pose],
            "gaze": torch.from_numpy(data["gaze"]).float()
            if "gaze" in data.files else torch.zeros(2),
        }

    def _load_camera(self, pdm, cam: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.synthetic:
            K = torch.eye(3)
            K[0, 0] = K[1, 1] = self.image_size  # focal ~ image size
            K[0, 2] = K[1, 2] = self.image_size / 2
            w2c = torch.eye(4)
            w2c[2, 3] = 1.0  # 1m in front of head
            return K, w2c
        calib = pdm.load_camera_calibration()
        import numpy as np
        K = torch.from_numpy(np.array(calib.intrinsics).astype('float32'))
        w2c = torch.from_numpy(np.array(calib.world_2_cam[cam]).astype('float32'))
        # Rescale K to the resized image.
        # NeRSemble shipping resolution is 3208x2200; assume original w=3208.
        scale = self.image_size / 3208.0
        K = K.clone()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        return K, w2c

    # ---------------------------------------------------------------- sampling
    def __getitem__(self, idx: int) -> NeRSembleSample:
        pid, seq, t, target_cam = self.index[idx]
        rng = np.random.default_rng(seed=idx)

        if self.synthetic:
            cams = self._cams_synth
            ref_cams = list(rng.choice([c for c in cams if c != target_cam],
                                       size=self.num_ref_views, replace=False))
            ref_imgs = torch.stack([
                torch.rand(3, self.image_size, self.image_size) for _ in ref_cams
            ])
            target_img = torch.rand(3, self.image_size, self.image_size)
            K, w2c = self._load_camera(None, target_cam)
        else:
            pdm = self._participant_manager(pid)
            cams = pdm.list_cameras(seq)
            ref_cams = list(rng.choice([c for c in cams if c != target_cam],
                                       size=self.num_ref_views, replace=False))
            ref_imgs = torch.stack([self._load_image(pdm, seq, c, t) for c in ref_cams])
            target_img = self._load_image(pdm, seq, target_cam, t)
            K, w2c = self._load_camera(pdm, target_cam)

        flame = self._load_flame(pid, seq, t)

        return NeRSembleSample(
            ref_imgs=ref_imgs,
            target_img=target_img,
            target_K=K,
            target_w2c=w2c,
            shape=flame["shape"],
            expression=flame["expression"],
            pose=flame["pose"],
            gaze=flame["gaze"],
            meta={"pid": pid, "seq": seq, "t": t, "target_cam": target_cam,
                  "ref_cams": ref_cams},
        )


def collate_samples(batch: list[NeRSembleSample]) -> dict:
    """Custom collate: stack tensors, keep meta as list of dicts."""
    return {
        "ref_imgs": torch.stack([b.ref_imgs for b in batch]),
        "target_img": torch.stack([b.target_img for b in batch]),
        "target_K": torch.stack([b.target_K for b in batch]),
        "target_w2c": torch.stack([b.target_w2c for b in batch]),
        "shape": torch.stack([b.shape for b in batch]),
        "expression": torch.stack([b.expression for b in batch]),
        "pose": torch.stack([b.pose for b in batch]),
        "gaze": torch.stack([b.gaze for b in batch]),
        "meta": [b.meta for b in batch],
    }
