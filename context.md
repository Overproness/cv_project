# Project Context: Aura-3D

## Overview

This workspace contains two co-dependent codebases:

1. **GazeGaussian** – A cloned, reference research codebase (Wei et al., AAAI 2025). Used as the primary architectural blueprint.
2. **Aura-3D** – A new, original system being built in this workspace, using GazeGaussian as its foundation.

The long-term goal of Aura-3D is to: given 1–4 monocular 2D reference images of a face, instantly produce a fully animatable, photorealistic 3D Gaussian Splatting (3DGS) head avatar driveable in real-time (60+ FPS) from a webcam — with ZERO per-user optimization at inference.

---

## Repository Structure

```
cv_project/
  GazeGaussian/             Reference codebase (Wei et al.)
  Aura3D/                   New system being built
    README.txt              Architecture overview + dataset plan
    aura3d/
      configs/
        aura3d_default.yaml  Full Hydra-style config (model + training)
      models/
        aura3d_model.py      Top-level Aura3DModel class
        encoders/
          multiview_vit.py   DINOv2 multi-view encoder
        decoders/
          parameter_decoder.py  UV-space Gaussian attribute decoder
        flame/
          flame_template.py    FLAME canonical template wrapper
      __init__.py
```

---

## GazeGaussian Reference Codebase — Deep Dive

### Purpose

Per-subject, per-scene optimization-based gaze redirection system using 3DGS. NOT feed-forward. Trains one model _per identity_ on a dataset's training split (ETH-XGaze), then evaluates on a test set. Serves as the animation backbone (deformation MLPs, eye rotation) that Aura-3D will reuse.

### Key Files

| File                                      | Role                                                                                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `models/gaze_gaussian.py`                 | Top-level `GazeGaussianNet`: wires face branch + eye branch                                                                                    |
| `models/gaussian_model.py`                | `GaussianModel`: holds Gaussian attrs (xyz, feature, rotation, scale, opacity) as `nn.Parameter`; implements face deform and eye rotation MLPs |
| `models/neural_renderer.py`               | `NeuralRenderer`: UNet-style pixel-shuffle upsampler; converts rasterized feature maps → RGB images                                            |
| `models/MLP.py`                           | Generic skip-connection `MLP` (Conv1d-based, used for all deformation/color/attribute heads)                                                   |
| `models/mesh_head.py`                     | `MeshHeadModule`: SDF-based neutral mesh reconstruction using DMTet; produces the initialization mesh for Gaussians                            |
| `models/camera_module.py`                 | Camera projection utilities                                                                                                                    |
| `models/discriminator.py`                 | `PatchGAN` discriminator (optional)                                                                                                            |
| `losses/gazenerf_loss.py`                 | Combined loss: L1, SSIM, LPIPS (via `lpips`), angular gaze loss (VGG-based gaze estimator), GAN losses                                         |
| `configs/gazegaussian_options.py`         | `BaseOptions`: all hyperparameters as a plain Python class                                                                                     |
| `trainer/gazegaussian_trainer.py`         | Full training loop, multi-view data loading, checkpointing                                                                                     |
| `trainer/meshhead_trainer.py`             | First-stage trainer: optimizes the SDF neutral mesh                                                                                            |
| `dataloader/*.py`                         | Dataset wrappers for ETH-XGaze, ColumbiaGaze, MPIIFaceGaze, GazeCapture                                                                        |
| `gaze_estimation/`                        | Pre-trained ResNet50/VGG gaze estimator used for the angular loss                                                                              |
| `face_recognition/`                       | FaceX-Zoo-based face recognition for identity similarity evaluation                                                                            |
| `utils/`                                  | Camera math, SH utilities, render utils, trainer utils                                                                                         |
| `submodules/diff-gaussian-rasterization/` | INRIA's CUDA rasterizer (unmodified)                                                                                                           |
| `submodules/simple-knn/`                  | KNN for Gaussian initialization                                                                                                                |

### Two-Stream Gaussian Architecture (GazeGaussian)

```
Neutral Mesh (DMTet SDF)
       │
       ├── Face region ──► canonical face Gaussians {xyz_f, feat_f, q_f, s_f, α_f}
       │                        │
       │            expr+pose codes (θ, β)
       │                        │
       │          FaceDeformMLP (Φ_f): Conv1d skip-MLP
       │              ├── position: exp_def + pose_def (λ-weighted blend)
       │              ├── color:    exp_col + pose_col
       │              └── attrs:    exp_att + pose_att (Δq, Δs, Δα)
       │                        │
       │                Deformed Face Gaussians
       │
       └── Eye region ──► canonical eye Gaussians {xyz_e, feat_e, q_e, s_e_sphere, α_e}
                               │
                    gaze direction φ + identity codes θ
                               │
                   EyeDeformMLP (Φ_e): gaze_rot + exp_def
                       ├── GERR: rigid rotate eye Gaussians around eyeball center
                       │          + learned residual from gaze_rot MLP
                       ├── color: exp_att + gaze_col
                       └── attrs: exp_att + gaze_att
                               │
                      Rotated Eye Gaussians

Both streams → rasterizer → feature maps {M_f, M_e, M_h}
                                │
                  Expression-Guided Neural Renderer (EGNR)
                  (cross-attention: q=expr_codes, k/v=bottleneck feat)
                                │
                       Final RGB images {I_f, I_e, I_h}
```

### MLP Architecture (Conv1d skip-MLPs)

- Input layer, mid-layer skip-connection (concat input at halfway), Conv1d operations.
- Face shape MLP: `[272, 256, 256, 256, 256, 256, 3]`
- Face color MLP: `[272, 256, 256, 32]`
- Eye deform MLP: `[45, 128, 128, 3]`
- Eye rotate MLP: `[45, 128, 128, 4]` (quaternion output)
- Positional encoding: Fourier features (`pos_freq=4`, `gaze_freq=4`)

### Loss Function (GazeGaussianNet)

```
L_total = λ_I · L_I + λ_G · L_G
L_I = L_f_I + L_e_I + L_h_I + L_f_M + L_e_M + L_h_M
      (each = L1 + λ_SSIM·SSIM + λ_VGG·LPIPS, with eye mask)
L_G = angular_error(ψ_g(I_wf), ψ_g(I_gt))   (pre-trained gaze estimator)
λ_I=1.0, λ_G=0.1, λ_SSIM=λ_VGG=0.1
```

### Training Protocol (GazeGaussian)

- Stage 1: Train `MeshHeadModule` (SDF + DMTet neutral mesh)
- Stage 2: Transfer to `GazeGaussianNet` (deformation MLPs + EGNR)
- ETH-XGaze: 80 subjects × 18 views × 10 frames = 14,400 images
- Resolution: 512×512, normalized/cropped
- Optimizer: AdamW, LR=1e-4, step decay

### Inference Speed

74 FPS on unspecified GPU (paper claims), competes with GazeNeRF (46 FPS). NOT real-time 60+ FPS — that is Aura-3D's target.

---

## Aura-3D — New System Being Built

### Core Thesis

**Feed-forward identity prediction**: encode identity from 1–4 reference images → predict personalized 3DGS avatar in one forward pass. No per-user optimization at inference.

### Architectural Decisions Made

#### 1. Multi-View ViT Encoder (`encoders/multiview_vit.py`)

- Backbone: **DINOv2-ViT-B/14** (768-dim patch tokens)
- Shared backbone across all reference views (V ≤ 4)
- Learnable per-view positional embedding added to patch tokens
- Cross-view fusion: 2-layer TransformerEncoder (standard PyTorch; heads=8, d_model=768)
- Output: `EncoderOutput(tokens, fused, cls)` — cls is the (B, 768) global identity vector
- Backbone can be frozen for early training stages

#### 2. FLAME Canonical Template (`flame/flame_template.py`)

- Wraps `flame-pytorch` library
- Exposes: `canonical_verts` (V, 3), `faces` (F, 3), `eye_face_mask` (F,)
- Zero-expression, zero-shape, neutral-pose canonical state stored as buffer
- Eye triangle mask derived from eye vertex indices (separates two-stream)
- `forward(shape, expression, pose)` calls FLAME and returns deformed verts

#### 3. UV-Space Parameter Decoder (`decoders/parameter_decoder.py`)

- Why UV? Per-triangle prediction via UV map sampling gives smooth, spatially-coherent offsets; much more stable than unstructured MLP
- `token_to_grid`: Linear(768 → 128*16*16), reshaped to (B, 128, 16, 16)
- 4× ConvTranspose2d upsampling → (B, 128, 256, 256) UV feature map
- Sample at FLAME triangle UV centroids via `F.grid_sample`
- Per-attribute heads (zero-initialized): position Δxyz, log-scale, quaternion (identity init), RGB color, opacity
- `set_triangle_uvs(tri_uv)` must be called before first forward

#### 4. Top-Level Aura3DModel (`models/aura3d_model.py`)

- `encode_identity(ref_imgs)` → `IdentityCode(encoder_out, gaussian_offsets)` — **run once per user**
- `animate(identity, shape, expression, pose, gaze, camera)` → rendered frame — **run every webcam frame**
- `face_deform`, `eye_branch`, `renderer` declared as `None` — to be implemented next
- Config-driven: all components instantiated from `aura3d_default.yaml`

#### 5. Key Architectural Decisions (Critique of Original Plan)

| Original Plan                        | Corrected Decision                     | Reason                          |
| ------------------------------------ | -------------------------------------- | ------------------------------- |
| Predict raw Gaussian offsets via MLP | Predict via UV-space CNN decoder       | More stable, spatially coherent |
| Single image input                   | 1–4 reference views (multi-view ViT)   | Single view under-constrains 3D |
| EGNR at inference                    | EGNR training-only                     | Breaks 60 FPS target            |
| Fork diff-gaussian-rasterization     | Use unmodified; offsets in PyTorch     | Unnecessary; all upstream       |
| Pure MLP eye rotation                | Explicit rigid rotation + MLP residual | Avoids drifting eye artifacts   |

---

## Tech Stack

| Component                         | Library/Tool                                           |
| --------------------------------- | ------------------------------------------------------ |
| Deep learning                     | PyTorch                                                |
| ViT backbone                      | DINOv2 (torch.hub)                                     |
| 3D Gaussian rasterizer            | `diff-gaussian-rasterization` (INRIA CUDA, unmodified) |
| 3D ops / mesh                     | `pytorch3d`                                            |
| KNN                               | `simple-knn`                                           |
| Face parametric model             | `flame-pytorch`                                        |
| Real-time tracking                | DECA / MediaPipe                                       |
| Face refinement (train-time only) | CodeFormer / GFPGAN                                    |
| 2D supervision (train-time)       | SimSwap (pretrained)                                   |
| SDF / DMTet                       | Already in GazeGaussian utils                          |
| Config                            | Hydra / YAML                                           |

---

## Dataset Plan (Phased)

| Phase   | Dataset                               | Subjects/Frames                          | Size     | Purpose                      |
| ------- | ------------------------------------- | ---------------------------------------- | -------- | ---------------------------- |
| Phase 1 | NeRSemble subset                      | 2–3 subjects, 16 views, 50 frames        | ~5 GB    | Sanity check / overfit       |
| Phase 2 | NeRSemble full + Multiface mini       | ~30 subjects + Multiface 13 identities   | ~25 GB   | Cross-subject generalization |
| Phase 3 | FaceScape multi-view + EG3D synthetic | 359 subjects + ~50k synthetic identities | ~120 GB+ | In-the-wild generalization   |

### Dataset Access

- **NeRSemble**: Request form at `https://forms.gle/rYRoGNh2ed51TDWX9`; pip install `nersemble_data`; 16 cameras, 7.1 MP, 73 FPS
- **Multiface**: Clone `facebookresearch/multiface`; use `download_dataset.py`; mini-dataset = 16.2 GB
- **FaceScape**: Email `nju3dv@nju.edu.cn` with signed license; 847 subjects × 20 expressions, 4K textures
- **EG3D synthetic**: Generate with pretrained EG3D checkpoint; ~50k diverse identities

---

## Training Stages

| Stage | Script                         | Data                       | Goal                             |
| ----- | ------------------------------ | -------------------------- | -------------------------------- |
| 1     | `train_stage1_overfit.py`      | NeRSemble 2–3 subjects     | Verify pipeline end-to-end       |
| 2     | `train_stage2_crosssubject.py` | NeRSemble full + Multiface | Feed-forward identity prediction |
| 3     | `train_stage3_inthewild.py`    | FaceScape + EG3D           | In-the-wild generalization       |

---

## What Still Needs to Be Built (Aura-3D)

| Component                | File                                 | Status |
| ------------------------ | ------------------------------------ | ------ |
| Face deformation MLP     | `models/deformation/face_deform.py`  | TODO   |
| GERR eye branch          | `models/eye/gerr.py`                 | TODO   |
| FLAME→Gaussian binding   | `models/gaussians/flame_binding.py`  | TODO   |
| Rasterizer wrapper       | `models/renderer/rasterizer.py`      | TODO   |
| EGNR (training-only)     | `models/renderer/egnr.py`            | TODO   |
| CodeFormer refine unit   | `models/refine/refine_unit.py`       | TODO   |
| NeRSemble dataloader     | `data/datasets/nersemble.py`         | TODO   |
| DECA preprocessing       | `data/preprocessing/deca_tracker.py` | TODO   |
| Training loop            | `training/trainer.py`                | TODO   |
| Live inference driver    | `inference/driver.py`                | TODO   |
| Losses                   | `losses/`                            | TODO   |
| animate() in Aura3DModel | `models/aura3d_model.py`             | TODO   |

---

## Key Papers Referenced

| Paper                                                     | Key Contribution Used                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| GazeGaussian (Wei et al., AAAI 2025)                      | Two-stream face/eye 3DGS, GERR, EGNR, MLP deformation architecture                          |
| GaussianAvatars (Qian et al., CVPR 2024)                  | FLAME→triangle Gaussian binding, per-triangle local-frame transformation                    |
| Controllable 3D Deepfake (Liu et al., arXiv:2509.11624v1) | 2D-supervised multi-view training, CodeFormer refine unit (train-time supervision cleaning) |
| 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)       | Core rasterization primitive                                                                |
| NeRSemble (Kirschstein et al., SIGGRAPH 2023)             | Primary training dataset                                                                    |
| DINOv2 (Oquab et al., 2024)                               | ViT backbone for identity encoder                                                           |
| FLAME (Li et al., ACM TOG 2017)                           | Parametric face model for canonical template                                                |
| DMTet (Shen et al., NeurIPS 2021)                         | SDF-to-mesh for Gaussian initialization (via GazeGaussian)                                  |

---

## Notes & Decisions Log

- **2026-04-24**: Initial architecture scaffolded. Key insight: UV-space decoder is strictly better than unstructured MLP for per-Gaussian prediction. EGNR confirmed training-only (inference: pure rasterization). Multi-view ViT with cross-view attention chosen over single-view.
- **2026-04-25**: context.md and synopsis created. Next priority: FLAME→Gaussian binding + face deformation MLP + NeRSemble dataloader, so Stage 1 training loop can run end-to-end.
