# Aura-3D: Feed-Forward 3D Gaussian Splatting Avatar Synthesizer

A feed-forward neural system that reconstructs a photorealistic, fully-animatable 3D head avatar from **1–4 monocular reference photographs** in a single forward pass — with **zero per-user optimization at inference**. Driven by [FLAME](https://flame.is.tue.mpg.de/) parametric mesh and rendered via [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

> **Status (May 29, 2026):** Stage 1 (per-subject overfit, 3 NeRSemble subjects) complete. SSIM 0.7379 · PSNR 22.06 dB · LPIPS 0.5156 at step 184,450. Stage 2 (cross-subject generalization) and Stage 3 (in-the-wild) are future work.

---

## Key Results

| Metric  | Value        | std    |
| ------- | ------------ | ------ |
| SSIM ↑  | **0.7379**   | ±0.044 |
| PSNR ↑  | **22.06 dB** | ±1.38  |
| LPIPS ↓ | **0.5156**   | ±0.057 |

Evaluated on 50 held-out frames from 3 NeRSemble subjects (step 139,400 best checkpoint).

---

## Repository Layout

```
cv_project/
├── Aura3D/                  ← Main codebase (this project)
│   ├── aura3d/
│   │   ├── configs/
│   │   │   └── aura3d_default.yaml   # Full model + training config
│   │   ├── models/
│   │   │   ├── aura3d_model.py       # Top-level Aura3DModel
│   │   │   ├── encoders/             # DINOv2 multi-view ViT encoder
│   │   │   ├── decoders/             # UV-space CNN parameter decoder
│   │   │   ├── flame/                # FLAME canonical template wrapper
│   │   │   ├── gaussians/            # FLAME-to-triangle Gaussian binding
│   │   │   ├── deformation/          # Face expression/pose deform MLP
│   │   │   ├── eye/                  # GERR gaze rotation branch
│   │   │   ├── renderer/             # diff-gaussian-rasterization wrapper
│   │   │   └── refine/               # CodeFormer (training-time only)
│   │   ├── data/                     # NeRSemble / Multiface / FaceScape loaders
│   │   ├── losses/                   # L1, SSIM, LPIPS, gaze, identity, FLAME reg
│   │   ├── training/
│   │   │   └── trainer.py            # OverfitTrainer (Stage 1)
│   │   ├── scripts/
│   │   │   ├── train_stage1_overfit.py
│   │   │   ├── train_stage2_crosssubject.py  # (future)
│   │   │   └── train_stage3_inthewild.py     # (future)
│   │   └── utils/
│   └── third_party/
│       ├── flame_pytorch/            # git submodule
│       └── gaussian-splatting/       # git submodule
├── GazeGaussian/            ← Reference codebase (Wei et al., AAAI 2025)
├── data/
│   └── nersemble/           # 030, 038, 240 subjects + FLAME tracking .npz
├── runs/
│   └── stage1_real/         # Checkpoints: best.pt, latest.pt
└── third_party/
    └── DECA/                # Face tracker (git submodule)
```

---

## Architecture

### Inference Pipeline

```
ref_imgs (1–4)
    │
    ▼
[DINOv2-ViT-B/14 Encoder]        shared backbone + cross-view attention (2 layers)
    │ identity codes (B, 768)
    ▼
[UV-Space CNN Decoder]            Linear → 4× ConvTranspose2d → (B, 128, 256, 256)
    │                             → grid_sample at FLAME triangle UV centroids
    │ per-triangle Gaussian attrs (Δxyz, Δscale, Δrot, ΔRGB, Δopacity)
    ▼
[FLAME Canonical Template]        ~5000 triangles; eye-region mask separates two streams
    │
    ├── [Face Deform MLP]         3-layer skip-MLP, conditioned on expr (50) + pose (6)
    │        │
    ├── [GERR Eye Branch]         rigid gaze rotation around eyeball center + residual MLP
    │        │
    └────────┴──► [diff-gaussian-rasterization]  →  rendered frame
```

### Two-Stream Design

| Stream | Input conditioning               | Method                                         |
| ------ | -------------------------------- | ---------------------------------------------- |
| Face   | FLAME expression (50) + pose (6) | 3-layer skip-MLP (256-dim)                     |
| Eye    | Gaze direction (3)               | Rigid rotation (GERR) + residual MLP (128-dim) |

### Key Components

| Component         | Implementation                                                  |
| ----------------- | --------------------------------------------------------------- |
| Encoder backbone  | DINOv2-ViT-B/14 (768-dim patch tokens)                          |
| Cross-view fusion | 2-layer `TransformerEncoder` (heads=8)                          |
| Decoder           | `Linear(768→128×16×16)` → 4× `ConvTranspose2d` → UV grid sample |
| Rasterizer        | INRIA `diff-gaussian-rasterization` (unmodified CUDA kernel)    |
| Face model        | `flame-pytorch` (FLAME 2020, 100 shape + 50 expr coefficients)  |
| Eye rotation      | Explicit rigid rotation around eyeball center (r=0.012 m)       |

---

## Setup

### Requirements

- Linux (tested on WSL2 / Ubuntu 22.04)
- CUDA 12.1+
- Python 3.11
- Conda

### Environment

```bash
# Create and activate the pytorch environment
conda create -n aura3d python=3.11 -y
conda activate aura3d

# PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install einops timm lpips open3d pytorch3d
pip install PyYAML tqdm tensorboard

# Build CUDA submodules (from Aura3D/)
cd Aura3D
pip install ./third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install ./third_party/gaussian-splatting/submodules/simple-knn
```

### FLAME Model

1. Register and download the **FLAME 2020** model from https://flame.is.tue.mpg.de/
2. Place `generic_model.pkl` at:
   ```
   Aura3D/third_party/flame_pytorch/data/FLAME2020/generic_model.pkl
   ```

### Data

Download NeRSemble subject(s) and run FLAME tracking:

```bash
# FLAME tracking script (requires DECA environment)
bash run_flame_tracking.sh
```

Expected layout after download + tracking:

```
data/nersemble/
  030/EXP-2-eyes/   camera*/images/, flame_tracking/*.npz
  038/EXP-1-head/
  038/EXP-4-lips/
  240/EXP-1-head/
  240/EXP-4-lips/
```

---

## Training

All commands run from `Aura3D/`.

### Stage 1 — Per-Subject Overfit

Trains on 3 subjects (2,400 samples) to validate the full pipeline.

```bash
python -u -m aura3d.scripts.train_stage1_overfit \
    --config aura3d/configs/aura3d_default.yaml \
    --ckpt-dir ../runs/stage1_real \
    --device cuda
```

Checkpoints are saved to `runs/stage1_real/best.pt` and `runs/stage1_real/latest.pt` every 500 steps.

**Smoke test (no data required):**

```bash
python -m aura3d.scripts.train_stage1_overfit \
    --config aura3d/configs/aura3d_default.yaml \
    --synthetic --steps 100
```

### Stage 2 — Cross-Subject Generalization _(future)_

Requires full NeRSemble (220+ subjects) + Multiface dataset.

### Stage 3 — In-the-Wild _(future)_

Requires FaceScape + EG3D-synthesized data. Uses SimSwap + CodeFormer for 2D-supervised training.

---

## Configuration

All hyperparameters live in [`Aura3D/aura3d/configs/aura3d_default.yaml`](Aura3D/aura3d/configs/aura3d_default.yaml).

Key settings:

| Parameter                     | Value           | Notes                                      |
| ----------------------------- | --------------- | ------------------------------------------ |
| `model.encoder.backbone`      | `dinov2_vitb14` | Can swap to `dinov2_vits14`                |
| `model.encoder.num_ref_views` | `4`             | Supports 1–4 reference images              |
| `model.decoder.uv_resolution` | `256`           | UV feature map resolution                  |
| `training.lr_encoder`         | `1e-5`          | Lower LR for pretrained ViT                |
| `training.lr_decoder`         | `5e-4`          |                                            |
| `training.max_grad_norm`      | `0.5`           | Gradient clipping                          |
| `data.image_size`             | `518`           | Must be multiple of 14 (DINOv2 patch size) |
| `training.losses.lpips`       | `0.05`          | Phase 2 perceptual loss weight             |

---

## Evaluation

```bash
# Evaluation artifacts are written to runs/stage1_real/eval/
ls runs/stage1_real/eval/
```

Phase 2 evaluation (with LPIPS loss) outputs go to `runs/stage1_real/eval_phase2/`.

---

## Reference Codebase

[`GazeGaussian/`](GazeGaussian/) contains the reference implementation of **GazeGaussian (Wei et al., AAAI 2025)**, which served as the architectural blueprint for the two-stream face/eye design and GERR gaze rotation. It is included for reference only and is not part of the Aura-3D training pipeline.

---

## Tech Stack

| Component       | Library                                              |
| --------------- | ---------------------------------------------------- |
| Deep learning   | PyTorch 2.5.1                                        |
| ViT backbone    | DINOv2 (torch.hub)                                   |
| 3DGS rasterizer | diff-gaussian-rasterization (INRIA CUDA, unmodified) |
| Face model      | flame-pytorch (FLAME 2020)                           |
| 3D ops          | pytorch3d                                            |
| Face tracker    | DECA                                                 |
| Perceptual loss | LPIPS (alex net)                                     |
| Dataset         | NeRSemble                                            |

---

## References

- **GazeGaussian:** Wei et al., _Gaze-Directed 3D Gaussian Splatting for Real-Time Gaze-Aware Rendering_, AAAI 2025
- **GaussianAvatars:** Qian et al., _GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians_, CVPR 2024
- **3D Gaussian Splatting:** Kerbl et al., _3D Gaussian Splatting for Real-Time Radiance Field Rendering_, SIGGRAPH 2023
- **NeRSemble:** Kirschstein et al., _NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads_, SIGGRAPH 2023
- **FLAME:** Li et al., _Learning a model of facial shape and expression from 4D scans_, SIGGRAPH Asia 2017
- **DINOv2:** Oquab et al., _DINOv2: Learning Robust Visual Features without Supervision_, TMLR 2024
