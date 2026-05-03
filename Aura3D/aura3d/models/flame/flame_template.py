"""FLAME canonical template wrapper.

Thin wrapper over ``flame_pytorch`` (local package) that exposes:
  * canonical (zero shape / zero expression / neutral pose) vertices and faces
  * UV coordinates and triangle->UV mapping
  * eye-region triangle indices (for two-stream face/eye split)

When ``flame_model_path`` does not exist on disk the class falls back to a
**synthetic** mode that uses fixed-shape dummy geometry.  This allows the
full Aura3D pipeline to run end-to-end (for architecture smoke-tests and
synthetic-data training) without the FLAME assets, which require a manual
download from https://flame.is.tue.mpg.de/.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn


class FLAMECanonicalTemplate(nn.Module):
    # Canonical FLAME 2020 sizes — used even in synthetic mode so that all
    # downstream buffers (binding, decoder UV grid, etc.) have the right shape.
    NUM_VERTS: int = 5023
    NUM_FACES: int = 9976

    def __init__(
        self,
        flame_model_path: str,
        n_shape: int = 100,
        n_exp: int = 50,
        n_pose: int = 6,
        use_face_contour: bool = False,
    ) -> None:
        super().__init__()
        self.n_shape = n_shape
        self.n_exp = n_exp
        self.n_pose = n_pose

        model_path = Path(flame_model_path)
        self._synthetic = not model_path.exists()

        if self._synthetic:
            warnings.warn(
                f"FLAME model not found at '{model_path}'. "
                "Running in SYNTHETIC mode with dummy geometry. "
                "Download FLAME 2020 from https://flame.is.tue.mpg.de/ "
                "for real training.",
                UserWarning,
                stacklevel=2,
            )
            self._init_synthetic()
        else:
            self._init_real(model_path, n_shape, n_exp, n_pose, use_face_contour)

        # ---- Eye-region triangles ----------------------------------------
        # Populated by _load_eye_vertex_indices() once FLAME assets and
        # flame_static_embedding.pkl are available (Task 3).
        left_v, right_v = self._load_eye_vertex_indices(model_path)
        self.register_buffer("left_eye_vidx",  left_v)
        self.register_buffer("right_eye_vidx", right_v)

        def _tri_mask(vidx: torch.Tensor) -> torch.Tensor:
            m = torch.zeros(self.faces.shape[0], dtype=torch.bool)
            for i in range(3):
                m |= torch.isin(self.faces[:, i], vidx)
            return m

        left_mask  = _tri_mask(left_v)
        right_mask = _tri_mask(right_v)
        self.register_buffer("left_eye_face_mask",  left_mask)
        self.register_buffer("right_eye_face_mask", right_mask)
        self.register_buffer("eye_face_mask", left_mask | right_mask)

    # ------------------------------------------------------------------ init

    def _init_real(
        self,
        model_path: Path,
        n_shape: int,
        n_exp: int,
        n_pose: int,
        use_face_contour: bool,
    ) -> None:
        """Load the real FLAME model from pkl and cache canonical geometry."""
        from flame_pytorch import FLAME, get_config  # type: ignore

        cfg = get_config()
        cfg.flame_model_path = str(model_path)
        cfg.shape_params = n_shape
        cfg.expression_params = n_exp
        cfg.pose_params = n_pose
        cfg.use_face_contour = use_face_contour

        self.flame = FLAME(cfg)

        with torch.no_grad():
            shape = torch.zeros(1, n_shape)
            exp   = torch.zeros(1, n_exp)
            pose  = torch.zeros(1, n_pose)
            verts, _ = self.flame(shape, exp, pose)

        self.register_buffer("canonical_verts", verts[0])                   # (V, 3)
        self.register_buffer("faces",           self.flame.faces_tensor.long())  # (F, 3)

    def _init_synthetic(self) -> None:
        """Register dummy buffers (correct shapes, random values) for dry-runs."""
        V, F = self.NUM_VERTS, self.NUM_FACES
        self.flame = None

        # Vertices: zero — pipeline uses centroid + offset, so zeros are fine
        verts = torch.zeros(V, 3)
        # Faces: valid triangle indices covering all vertices
        rng = torch.Generator()
        rng.manual_seed(42)
        faces = torch.randint(0, V, (F, 3), dtype=torch.long, generator=rng)
        self.register_buffer("canonical_verts", verts)
        self.register_buffer("faces",           faces)

    # ------------------------------------------------------------------ eye

    @staticmethod
    def _load_eye_vertex_indices(
        model_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (left_eye_vidx, right_eye_vidx) as long tensors.

        Loads from ``flame_static_embedding.pkl`` when the file is present
        (Task 3).  Returns empty tensors otherwise so construction stays
        functional for smoke-tests.

        FLAME 68-landmark convention:
            left  eye ring → indices 36-41  (6 points)
            right eye ring → indices 42-47  (6 points)
        """
        empty = torch.tensor([], dtype=torch.long)

        embedding_path = model_path.parent / "flame_static_embedding.pkl"
        if not embedding_path.exists():
            return empty, empty

        import pickle
        with open(embedding_path, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")

        # flame_static_embedding stores per-landmark face index + barycentric
        # coords.  We recover the vertex indices of the landmark faces and use
        # them as an approximation of the "eye ring" vertex set.
        try:
            faces_np = data["lmk_face_idx"]   # (68,) face index per landmark
            b_coords = data["lmk_b_coords"]   # (68, 3) barycentric weights
        except KeyError:
            return empty, empty

        # We need access to the FLAME face array to convert face→vertex indices.
        # Load it from the generic_model.pkl next to the embedding file.
        import pickle, numpy as np
        generic_path = model_path  # same directory
        if not generic_path.exists():
            return empty, empty
        with open(generic_path, "rb") as fh:
            gm = pickle.load(fh, encoding="latin1")
        faces = np.array(gm["f"], dtype=np.int64)   # (F, 3)

        # Left eye: landmarks 36-41, right eye: 42-47
        left_lmk_idx  = list(range(36, 42))
        right_lmk_idx = list(range(42, 48))

        def _vidx_from_lmk(lmk_indices):
            vidx_set = set()
            for li in lmk_indices:
                fi = int(faces_np[li])
                for vi in faces[fi]:
                    vidx_set.add(int(vi))
            return torch.tensor(sorted(vidx_set), dtype=torch.long)

        return _vidx_from_lmk(left_lmk_idx), _vidx_from_lmk(right_lmk_idx)

    # ------------------------------------------------------------------ util

    def eye_centers(self, verts: torch.Tensor) -> torch.Tensor:
        """(B, V, 3) -> (B, 2, 3) world-space [left, right] eye centers."""
        b = verts.shape[0]
        if self.left_eye_vidx.numel() == 0 or self.right_eye_vidx.numel() == 0:
            return torch.zeros(b, 2, 3, device=verts.device, dtype=verts.dtype)
        left  = verts[:, self.left_eye_vidx].mean(dim=1)
        right = verts[:, self.right_eye_vidx].mean(dim=1)
        return torch.stack([left, right], dim=1)

    @property
    def num_vertices(self) -> int:
        return int(self.canonical_verts.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        shape: torch.Tensor,
        expression: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """Return posed vertices (B, V, 3)."""
        if self._synthetic:
            B = shape.shape[0]
            return self.canonical_verts.unsqueeze(0).expand(B, -1, -1).to(
                device=shape.device, dtype=shape.dtype
            )
        verts, _ = self.flame(shape, expression, pose)
        return verts
