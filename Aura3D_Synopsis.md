&nbsp;

---

# PROJECT SYNOPSIS

## Aura-3D: A Feed-Forward 3D Gaussian Splatting Avatar Synthesizer with Real-Time Gaze-Aware Animation

| | |
|---|---|
| **Submitted By:** | \[Student Name\] |
| **CMS ID:** | \[Student ID\] |

&nbsp;

*April 25, 2026*

---

## Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Description](#2-project-description)
   - 2.1 [Introduction and Motivation](#21-introduction-and-motivation)
   - 2.2 [Existing Research](#22-existing-research-3-5-papers)
   - 2.3 [Problem Statement](#23-problem-statement)
     - 2.3.1 [Research Questions](#231-research-questions)
   - 2.4 [Research Gap Identification](#24-research-gap-identification)
   - 2.5 [Solution Statement and Research Objectives](#25-solution-statement-and-research-objectives)
     - 2.5.1 [Research Objectives](#251-research-objectives)
3. [Methodology](#3-methodology)
   - 3.1 [Materials: Data Collection and Preprocessing](#31-materials-data-collection-and-preprocessing)
     - 3.1.1 [Data Collection and Ethical Considerations](#311-data-collection-and-ethical-considerations)
     - 3.1.2 [Data Labeling / Procedure of Data Screening](#312-data-labeling--procedure-of-data-screening)
     - 3.1.3 [Data Preprocessing](#313-data-preprocessing)
   - 3.2 [Methods](#32-methods)
     - 3.2.1 [Work Packages (WPs)](#321-work-packages-wps)
   - 3.3 [Model Architecture / Conceptual Framework](#33-model-architecture--conceptual-framework)
   - 3.4 [Justification of Selected Methodology](#34-justification-of-selected-methodology)
   - 3.5 [Novelty of the Work](#35-novelty-of-the-work)
   - 3.6 [Validation and Performance Metrics](#36-validation-and-performance-metrics)
4. [Evaluation](#4-evaluation)
   - 4.1 [Evaluation Metrics](#41-evaluation-metrics)
   - 4.2 [Evaluation Methodology](#42-evaluation-methodology)
   - 4.3 [Preliminary Results / Pilot Testing](#43-preliminary-results--pilot-testing)
   - 4.4 [Comparative Analysis](#44-comparative-analysis)
5. [Implementation Timeline](#5-implementation-timeline)
   - 5.1 [Gantt Chart](#51-gantt-chart)
   - 5.2 [Major Tasks and Deliverables](#52-major-tasks-and-deliverables)
   - 5.3 [Plan for Dissemination](#53-plan-for-dissemination)
- [Appendices](#appendices)
  - [Appendix A: List of Abbreviations](#appendix-a-list-of-abbreviations)
  - [Appendix B: Additional Materials](#appendix-b-additional-materials)
- [References](#references)

---

## 1. Executive Summary

Photorealistic, controllable 3D head avatars are a foundational technology for next-generation human–computer interaction, yet every practical deployment is currently blocked by a shared bottleneck: all high-fidelity methods require hours of per-person neural-network *optimization* over dense multi-view video captured in a controlled studio. This means a new user cannot be onboarded in real time, and a consumer smartphone photograph cannot serve as input. This project proposes **Aura-3D**, a system that eliminates this bottleneck entirely.

Aura-3D trains a feed-forward deep network that, given only **1–4 monocular reference photographs** of a new person, synthesizes a fully personalized, animatable 3D Gaussian Splatting (3DGS) head avatar in a **single forward pass** (< 100 ms), with **zero per-user optimization** at inference. The resulting avatar supports photorealistic gaze redirection and head-pose animation driven at ≥ 60 FPS from a live webcam.

The approach fuses three streams of recent work: (i) the two-stream face/eye Gaussian decoupling and Gaussian Eye Rotation Representation (GERR) of GazeGaussian \[Wei et al., 2025\]; (ii) the FLAME-mesh-to-Gaussian triangle binding of GaussianAvatars \[Qian et al., 2024\]; and (iii) a novel feed-forward multi-view Vision Transformer (DINOv2-ViT-B/14) identity encoder with a UV-space CNN decoder that predicts per-triangle Gaussian attribute offsets.

Training uses publicly available multi-view datasets — NeRSemble (Phase 1–2) and FaceScape + EG3D-synthetic data (Phase 3) — augmented with a 2D-supervised training strategy \[Liu et al., 2025\] that enables in-the-wild generalization without laboratory capture. Ethical compliance is maintained by using only datasets with explicit research-use licenses and by ensuring the system cannot be deployed for non-consensual identity manipulation.

Expected contributions are: (1) the first truly zero-optimization, feed-forward 3DGS avatar system with accurate gaze control; (2) a UV-space Gaussian parameter prediction framework transferable to other 3D generation tasks; (3) fully open-source code and pre-trained checkpoints.

---

## 2. Project Description

### 2.1 Introduction and Motivation

Realistic digital avatars of human faces are central to virtual reality, video conferencing, entertainment, accessibility technology, and medical simulation. The dominant paradigm — per-subject Neural Radiance Fields (NeRF) or 3D Gaussian Splatting (3DGS) — achieves stunning visual quality, but each new identity requires 30 minutes to several hours of GPU compute over dozens of synchronized camera views captured in a studio. This is economically and logistically prohibitive for consumer applications.

3D Gaussian Splatting \[Kerbl et al., 2023\] represents a scene as a set of anisotropic 3D Gaussians parameterized by position, covariance, spherical-harmonic color, and opacity, rendered at real-time framerates via a CUDA tile-based rasterizer. While breakthroughs in 3DGS-based head animation (e.g., GaussianAvatars \[Qian et al., 2024\], GazeGaussian \[Wei et al., 2025\]) have pushed quality to photorealistic levels, each still trains one model per subject.

Meanwhile, feed-forward 3D reconstruction methods (e.g., pixelNeRF, Large Reconstruction Model) have demonstrated that a trained network can generalize across identities, predicting a 3D representation from a single image — but these methods do not yet produce Gaussian splat representations with accurate facial animation control, particularly for the subtle, high-frequency region of the eyes and gaze.

Aura-3D is motivated by the gap between these two lines of work: it unifies the photorealistic quality of per-subject 3DGS head avatars with the zero-shot, single-image generalization of feed-forward reconstruction networks, while also incorporating state-of-the-art gaze control.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AURA-3D CONCEPTUAL OVERVIEW                    │
│                                                                     │
│   1–4 Reference Photos                                              │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────┐    global    ┌───────────────┐  per-triangle       │
│  │ DINOv2 ViT  │─ identity ──►│  UV Decoder   │─ Gaussian offsets ─►│
│  │ (shared,    │  vector      │  (CNN, 256×256│                     │
│  │  V views)   │              │   UV space)   │                     │
│  └─────────────┘              └───────────────┘                     │
│                                       │                             │
│              FLAME canonical template ◄─┘ (bind to triangles)       │
│                       │                                             │
│         ┌─────────────┴──────────────┐                             │
│         ▼                            ▼                              │
│  Face Deform MLP              Eye Branch (GERR)                     │
│  (expr + pose codes)          (rigid gaze rotation                  │
│                                + learned residual)                  │
│         └─────────────┬──────────────┘                             │
│                        ▼                                            │
│               diff-gaussian-rasterizer                              │
│                        │                                            │
│                    RGB Frame  ──► ≥ 60 FPS                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Existing Research (3–5 Papers)

**GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting** \[Wei et al., AAAI 2025\]

GazeGaussian introduced the first 3DGS-based gaze redirection system, outperforming NeRF-based methods (6.62° gaze error, 74 FPS vs. GazeNeRF's 6.94°, 46 FPS). Its key insight is a *two-stream* architecture: face-only Gaussians are driven by expression/pose MLPs, while eye Gaussians use a dedicated Gaussian Eye Rotation Representation (GERR) that applies an explicit rigid rotation to eye Gaussians around the eyeball center, then adds small learned offsets as residuals. An Expression-Guided Neural Renderer (EGNR) injects expression codes via cross-attention into a U-Net decoder to improve generalization across subjects. **Weakness:** still an optimization-based, per-subject system requiring a full training run per identity; cannot generalize to new faces at test time.

**GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians** \[Qian et al., CVPR 2024\]

GaussianAvatars binds each 3D Gaussian to a specific triangle on a FLAME parametric mesh. A local-to-world transformation defined by triangle orientation and barycenter moves each Gaussian rigidly with its host triangle during animation, providing a strong geometric prior. Additional MLPs predict fine-grained appearance offsets per Gaussian. This binding strategy gives anatomically grounded animation without additional optimization at inference time. **Weakness:** the Gaussian parameters themselves are still per-subject learnable parameters, not predicted from a reference image; cannot be personalized instantly.

**A Controllable 3D Deepfake Generation Framework with Gaussian Splatting** \[Liu et al., arXiv:2509.11624, 2025\]

Liu et al. combined FLAME-Gaussian binding with a 2D-supervised training strategy: a pretrained 2D face-swapping model (SimSwap) generates pseudo-multi-view supervision targets from in-the-wild monocular video, enabling training without expensive studio capture. A CodeFormer-based Refine Unit cleans artifacts in the 2D supervision targets before they are used for 3D training, improving optimization stability. The system achieves real-time reenactment at 14.45 FPS on an RTX 2080. **Weakness:** face-swapping introduces identity leakage; still per-subject; 14 FPS is insufficient for smooth real-time interaction.

**3D Gaussian Splatting for Real-Time Radiance Field Rendering** \[Kerbl et al., SIGGRAPH 2023\]

The foundational 3DGS paper establishes the representation and CUDA rasterizer all subsequent methods build on. Gaussians are initialized from a SfM point cloud, then densified/pruned during training via adaptive density control. The tile-based alpha-compositing rasterizer achieves 100+ FPS rendering. **Limitation:** designed for static scenes; dynamic / animatable extensions require additional structure (e.g., FLAME binding).

**NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads** \[Kirschstein et al., SIGGRAPH 2023\]

NeRSemble provides a large-scale dataset (220+ subjects, 16 cameras, 7.1 MP, 73 FPS) with calibrated camera parameters and diverse expressions, alongside a dynamic NeRF baseline using a hash-ensemble deformation field. The dataset is the primary training data for Aura-3D. **Limitation as a method:** computationally heavy NeRF inference; no single-image generalization.

---

### 2.3 Problem Statement

Current state-of-the-art 3D Gaussian Splatting head avatar systems are optimization-bound: each new identity requires a full per-subject training run over densely captured multi-view video, then produces a model that works only for that identity. This makes them practically unusable in any consumer-facing or real-time deployment scenario. Furthermore, gaze direction — the single most important social cue in face-to-face interaction — is either absent from or only implicitly controlled in most existing systems, producing avatars with visually unconvincing, static eyes.

#### 2.3.1 Research Questions

1. Can a feed-forward neural network learn to predict personalized 3DGS Gaussian attribute offsets from 1–4 monocular reference photographs, without any per-subject optimization at inference?
2. What encoder-decoder architecture best preserves high-frequency identity cues (skin texture, eye color, facial geometry) needed for photorealistic personalization?
3. How can explicit gaze control (GERR) be integrated into a feed-forward pipeline so that the eye branch generalizes across identities rather than overfitting to a single person's eye geometry?
4. What is the minimum data volume and diversity needed to achieve cross-identity generalization with no per-subject fine-tuning?

---

### 2.4 Research Gap Identification

| Gap | Current State | Aura-3D Addresses |
|-----|---------------|-------------------|
| Per-subject optimization required | All 3DGS avatars train one model per person | Single forward pass; no per-user optimization |
| Multi-view studio capture needed | GazeGaussian, GaussianAvatars require 16–40 synchronized cameras | 1–4 consumer photos as input |
| No cross-identity generalization | Each model encodes one identity | DINOv2 encoder generalizes across all seen identities |
| Gaze control missing or implicit | GaussianAvatars has no gaze; GazeNeRF is implicit | Explicit GERR rigid rotation + residual MLP, inherited and adapted |
| Inference too slow for real-time | Liu et al.: 14 FPS; GazeNeRF: 46 FPS | Pure rasterization at inference (EGNR off): target ≥ 60 FPS |
| In-the-wild images unsupported | All methods need calibrated multi-view video | 2D-supervised training phase enables monocular in-the-wild input |

---

### 2.5 Solution Statement and Research Objectives

Aura-3D proposes a **feed-forward 3D Gaussian Splatting avatar synthesizer** that separates identity personalization (a one-time encoder-decoder pass over reference images) from animation (a per-frame FLAME-driven Gaussian deformation applied at ≥ 60 FPS). The two-stream face/eye architecture of GazeGaussian is reused for animation, but its per-subject Gaussian parameters are replaced by the output of a universal identity predictor, enabling the system to generalize across all identities in the training distribution.

#### 2.5.1 Research Objectives

1. **Design and validate** a multi-view DINOv2-ViT-B/14 encoder with cross-view transformer fusion that produces a stable 768-dimensional identity embedding from 1–4 reference face images.
2. **Develop** a UV-space CNN decoder that maps the identity embedding to per-FLAME-triangle 3D Gaussian attribute offsets (position Δxyz, log-scale, quaternion rotation, RGB color, opacity), zero-initialized so the canonical FLAME template is the prior.
3. **Implement** FLAME-to-triangle Gaussian binding (following GaussianAvatars) and a two-stream face deformation MLP + GERR eye branch supporting explicit gaze direction control.
4. **Train** the full pipeline in three stages (per-subject overfit → cross-subject → in-the-wild) using NeRSemble, Multiface, FaceScape, and EG3D-synthetic data.
5. **Achieve** real-time avatar animation at ≥ 60 FPS on a mid-range GPU (RTX 3080) with zero per-user optimization at inference.
6. **Evaluate** cross-identity generalization with standard metrics (gaze angular error, SSIM, PSNR, LPIPS, FID, identity cosine similarity) and compare against GazeGaussian, GaussianAvatars, and GazeNeRF.

---

## 3. Methodology

### 3.1 Materials: Data Collection and Preprocessing

#### 3.1.1 Data Collection and Ethical Considerations

All datasets used are established academic benchmarks with explicit non-commercial research licenses. No new data collection involving human subjects is conducted by this project.

| Dataset | Source | License | Scale | Role |
|---------|--------|---------|-------|------|
| NeRSemble | TU Munich | Restricted research use; form access required | 220+ subjects, 16 cams, 73 FPS | Primary training (Phase 1–2) |
| Multiface (mini) | Meta Reality Labs | CC-BY-NC 4.0 | 13 identities, 40 cams | Expression diversity (Phase 2) |
| FaceScape | Nanjing University | Non-commercial research; signed license required | 359 subjects × 20 expressions, ≈60 views | Pose + identity range (Phase 3) |
| EG3D synthetic | Generated; pretrained EG3D checkpoint (NVIDIA) | No real subjects — no privacy risk | ~50,000 synthetic identities | In-the-wild augmentation (Phase 3) |

**Ethical safeguards:** (i) All real-subject data is used solely for training internal model weights; no subject images are published. (ii) The system is designed to require explicit consent-image input and cannot synthesize a face without a user-provided reference, mitigating non-consensual misuse. (iii) FaceScape privacy policy is observed: subjects flagged as non-publishable are excluded from any visualizations.

#### 3.1.2 Data Labeling / Procedure of Data Screening

- **NeRSemble:** Camera calibration (intrinsics/extrinsics) and expression metadata are provided with the dataset. DECA \[Feng et al., 2021\] is run on every frame to produce FLAME shape, expression, and pose coefficients. Gaze direction labels are estimated by a pre-trained ResNet-50 gaze estimator following the ETH-XGaze normalization protocol \[Zhang et al., 2018\].
- **Multiface:** Provided tracked mesh and headpose data are used directly. Camera calibrations are included in the metadata bundle.
- **FaceScape / EG3D:** DECA-based FLAME fitting is applied. Frames with fitting confidence below a threshold (reprojection error > 5 px) are discarded.
- **Quality screening:** Frames flagged by face-parsing as occluded (< 40% visible face area), heavily blurred (Laplacian variance < 50), or with extreme expression artifacts are removed before training.

#### 3.1.3 Data Preprocessing

For all datasets:

1. **Background removal:** BiSeNet-V2 face parsing \[Yu et al., 2021\] produces per-frame foreground masks; backgrounds are replaced with a uniform color.
2. **Gaze normalization:** The image normalization procedure of \[Zhang et al., 2018\] is applied: each image is warped to a canonical camera with fixed focal length, aligning head pitch to zero, producing 512×512 crops.
3. **FLAME fitting:** DECA produces per-frame shape codes β (100-dim), expression codes θ (50-dim), and pose β_pose (6-dim). Per-subject mean shape is cached.
4. **Multi-view pairing:** For each training step, a random subset of V ∈ {1, 2, 4} views of the same subject/frame are selected as reference input; a held-out view is the reconstruction target.
5. **Normalization:** Image pixels are normalized to ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]) before passing to the ViT encoder.
6. **2D supervision (Phase 3):** SimSwap \[Chen et al., 2020\] is run on in-the-wild Multiface/FaceScape frames to generate pseudo-multi-view identity-transferred targets. CodeFormer \[Zhou et al., 2022\] post-processes these targets to remove artifacts before use as training supervision.

---

### 3.2 Methods

Aura-3D's training pipeline proceeds in three sequential stages, each building on the previous:

- **Stage 1 — Per-subject overfit:** Train the full pipeline on 2–3 NeRSemble subjects. Encoder backbone is frozen. Only FLAME binding, face deformation MLP, eye GERR branch, and UV decoder are trained. Goal: verify end-to-end differentiability and loss convergence.
- **Stage 2 — Cross-subject generalization:** Unfreeze the encoder backbone (with low LR). Train on all NeRSemble subjects + Multiface mini. The network must now generalize its identity code across different people.
- **Stage 3 — In-the-wild:** Add FaceScape multi-view + EG3D synthetic data. Add the 2D-supervised training branch (SimSwap + CodeFormer) for monocular inputs. Fine-tune the full model.

#### 3.2.1 Work Packages (WPs)

| WP | Title | Description |
|----|-------|-------------|
| WP1 | Gaussian Binding Infrastructure | Implement FLAME→triangle binding; canonical Gaussian initialization from FLAME mesh; eye-region separation |
| WP2 | Multi-View Identity Encoder | DINOv2-ViT-B/14 shared backbone; per-view positional embeddings; cross-view transformer fusion |
| WP3 | UV-Space Parameter Decoder | Token→grid projection; 4× transposed-conv upsampler; UV centroid sampling; per-attribute heads with zero init |
| WP4 | Face Deformation Branch | Expression + pose conditioned MLP; λ_exp distance-weighting; position/color/attribute outputs |
| WP5 | Eye Branch (GERR) | Explicit rigid gaze rotation around eyeball center; residual MLP for sclera/lid deformation; spherical scale constraint |
| WP6 | Training Loop & Losses | Combined L1 + SSIM + LPIPS + gaze angular + identity regularization loss; Stage 1/2/3 schedulers |
| WP7 | NeRSemble Dataloader | Multi-view video streaming; DECA preprocessing; gaze normalization; multi-view pairing sampler |
| WP8 | Real-Time Inference Driver | DECA/MediaPipe webcam tracker → FLAME params → animate() → rasterizer; targeting ≥ 60 FPS |
| WP9 | Evaluation & Baselines | Reproduce GazeGaussian, GaussianAvatars baselines; full metric suite; ablation study |

---

### 3.3 Model Architecture / Conceptual Framework

The Aura-3D model is composed of five neural modules that separate cleanly into a *personalization path* (run once per user) and an *animation path* (run every frame):

**Module 1 — Multi-View ViT Encoder**

A DINOv2-ViT-B/14 backbone (patch size 14, embedding dim 768) is shared across all V reference views, processing them as a batch of (B·V, 3, H, W) images. A learnable per-view positional embedding of shape (1, V, 1, 768) is added to each view's patch tokens to distinguish viewpoints. Two standard Transformer encoder layers (8 heads, d_model=768, FFN dim 3072) then perform cross-view attention by operating on the flattened (V · N_patch)-length sequence, fusing information across views. Mean pooling over fused patch tokens and view dimension yields a (B, 768) global identity vector.

**Module 2 — UV-Space Parameter Decoder**

The identity vector is projected by a linear layer from 768 dimensions to a 128×16×16 spatial volume, reshaped, then progressively upsampled by four transposed convolution blocks (each doubling spatial resolution with GELU activations) to a (B, 128, 256, 256) UV feature map. Per-FLAME-triangle Gaussian attribute offsets (Δxyz, Δscale, Δrotation, ΔRGB, Δopacity) are extracted by bilinear sampling at each triangle's UV centroid coordinate. All five attribute prediction heads are zero-weight-initialized so that, at initialization, the network outputs the canonical FLAME template unchanged.

**Module 3 — FLAME Canonical Template + Binding**

The FLAME model \[Li et al., 2017\] provides a canonical mesh (zero shape, zero expression, neutral pose). Each mesh triangle hosts one 3D Gaussian at its barycenter. The local-to-world transformation for each Gaussian is defined by the triangle's normal, tangent, and barycenter — following GaussianAvatars \[Qian et al., 2024\]. When FLAME is deformed by expression/pose parameters, Gaussians translate and rotate rigidly with their host triangles, providing anatomically correct macro-scale animation at zero additional cost.

**Module 4 — Face Deformation MLP (Face Branch)**

A Conv1d skip-MLP (architecture: [272, 256, 256, 256, 3] for position; similar for color and attributes) processes per-Gaussian feature vectors concatenated with expression codes θ (positional-encoded, freq=4) and head-pose codes β. A distance-based blending weight λ_exp(x) — piecewise linear in the minimum distance from each Gaussian to the 3D face landmark set — gates expression vs. pose influence per Gaussian (t₁=0.15, t₂=0.25 in normalized FLAME scale), following the GazeGaussian design.

**Module 5 — Eye Branch with GERR**

Eye-region triangles (those with at least one vertex in the FLAME eye-vertex set) form a separate stream. GERR first computes an explicit rigid rotation matrix from target gaze direction (pitch/yaw) to rotate eye Gaussians around the estimated eyeball center. A separate gaze-conditioned MLP then predicts small residual offsets to the Gaussian positions, colors, and attributes to account for eyelid deformation and gaze-label noise. Eye Gaussian scales are constrained to be isotropic (scalar, not 3-vector) to enforce eyeball sphericity.

```
        Reference Images (B, V, 3, H, W)
                │
        ┌───────▼──────────────────────────────────────┐
        │   DINOv2-ViT-B/14  (shared across V views)   │
        │   + per-view embedding + cross-view attn×2   │
        └───────────────────────┬──────────────────────┘
                                │ identity vector (B, 768)
        ┌───────────────────────▼──────────────────────┐
        │   UV Decoder: Linear → 16×16 → 256×256 map   │
        │   sample at FLAME tri UV centroids            │
        │   → (Δpos, Δscale, Δrot, ΔRGB, Δopacity)     │
        └───────────────────────┬──────────────────────┘
                                │ per-triangle canonical offsets
                                ▼
                  FLAME(shape, expr, pose)
                  → deformed mesh → tri transforms
                                │
               bind canonical Gaussians to triangles
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
       Face Branch                         Eye Branch
       FaceDeformMLP                       GERR (gaze rot)
       (expr + pose)                       + residual MLP
               │                                 │
               └────────────────┬────────────────┘
                                ▼
             diff-gaussian-rasterization (CUDA)
                                │
                           RGB frame
                    [Training: + EGNR (training only)]
```

---

### 3.4 Justification of Selected Methodology

**Why 3D Gaussian Splatting over NeRF?** 3DGS uses an explicit, point-cloud-like representation that renders at 60–100+ FPS without any ray marching, making the ≥ 60 FPS inference target achievable on consumer hardware. NeRF-based methods (including GazeNeRF) are limited to 15–50 FPS.

**Why DINOv2 over ResNet or CLIP?** DINOv2 \[Oquab et al., 2024\] produces dense, spatially aware patch tokens superior for geometric reconstruction tasks. Its self-supervised training on 142M images provides a face feature space robust to illumination and viewpoint variation without face-specific fine-tuning, outperforming supervised ResNet features on face reconstruction benchmarks.

**Why UV-space CNN decoder over unstructured MLP?** Predicting per-Gaussian offsets with an unstructured MLP requires the network to reason about arbitrarily ordered 3D points, which is both sample-inefficient and prone to spatial inconsistency. A UV-space map enforces a natural 2D spatial structure on the face surface, providing smooth, locally coherent predictions and dramatically better training stability than MLP alternatives.

**Why FLAME binding over free Gaussians?** Free Gaussians lack animation structure — every frame requires either re-optimization or a complex deformation MLP to handle macro-scale head motion. FLAME provides a strong, anatomically grounded motion prior that decouples macro-scale animation (handled by FLAME kinematics) from fine-grained appearance detail (handled by residual MLPs), enabling generalization to unseen head poses.

**Why keep EGNR training-only?** The Expression-Guided Neural Renderer adds a U-Net pass and cross-attention over the full rasterized feature map. On a mid-range GPU this costs ~8 ms per frame — enough to push the system below 60 FPS when combined with DECA tracking (~5 ms) and rasterization (~4 ms). The EGNR's primary role is to regularize training by providing a stronger appearance signal; at inference the rasterizer alone produces sufficient quality.

---

### 3.5 Novelty of the Work

1. **First feed-forward 3DGS avatar with accurate gaze control.** No existing system combines zero-optimization identity personalization (from reference photos) with explicit, GERR-based gaze redirection.
2. **UV-space Gaussian parameter prediction.** Predicting 3DGS attributes in UV texture space — rather than on unstructured 3D point clouds — is a novel design choice that makes the prediction task spatially coherent and compatible with standard CNN architectures.
3. **Decoupled personalization vs. animation throughput.** The strict separation of the (slow) identity encoding step from the (fast, ≥ 60 FPS) animation step is architecturally novel among 3DGS avatar systems, enabling instant deployment without re-training.
4. **2D-supervised in-the-wild training for 3DGS avatars.** Adapting the SimSwap + CodeFormer supervision pipeline of \[Liu et al., 2025\] to the feed-forward setting, enabling training on monocular in-the-wild images without multi-view capture.

---

### 3.6 Validation and Performance Metrics

See Section 4.1 for the full metric list. The primary validation strategy consists of:

- **Within-dataset evaluation:** Train on ETH-XGaze training split; evaluate on its 15-subject test set (200 labeled images per subject) following the GazeNeRF pairing protocol.
- **Cross-dataset generalization:** Evaluate the same trained model (zero fine-tuning) on ColumbiaGaze, MPIIFaceGaze, and GazeCapture test sets.
- **Ablation study:** Sequentially remove (a) multi-view input (drop to 1 view), (b) UV decoder (replace with MLP), (c) GERR (replace with pure MLP rotation), (d) cross-view attention, measuring impact on all metrics.

---

## 4. Evaluation

### 4.1 Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Gaze accuracy | Angular error (°) | Mean angle between estimated and ground-truth gaze vectors (lower is better) |
| Head pose | Angular error (°) | Mean angle between estimated and ground-truth head pose (lower is better) |
| Image quality | SSIM ↑ | Structural similarity to ground-truth frame |
| Image quality | PSNR ↑ (dB) | Peak signal-to-noise ratio |
| Image quality | LPIPS ↓ | Learned perceptual image patch similarity (AlexNet) |
| Image quality | FID ↓ | Fréchet Inception Distance (distribution-level realism) |
| Identity | Cosine similarity ↑ | FaceX-Zoo \[Wang et al., 2021\] face embedding similarity between rendered and ground-truth frames |
| Speed | FPS ↑ | Frames per second for the animation path on a fixed GPU |
| Personalization cost | Encoding latency (ms) | Time for one forward pass of encoder + decoder (one-time cost per user) |

### 4.2 Evaluation Methodology

- **Hardware:** All experiments run on a single NVIDIA RTX 3090 (24 GB VRAM). FPS is measured on an RTX 3080 to reflect mid-range consumer targets.
- **Baseline reproduction:** GazeGaussian, GaussianAvatars, and GazeNeRF are run with their released checkpoints or retrained on the same splits for fair comparison.
- **Reference view count ablation:** Each cross-dataset evaluation is run with V ∈ {1, 2, 4} reference views to characterize the accuracy–convenience tradeoff.
- **Gaze estimator:** A pre-trained ResNet-50 estimator \[He et al., 2016\] following ETH-XGaze normalization, identical to that used in GazeNeRF, ensures comparability.

### 4.3 Preliminary Results / Pilot Testing

At the time of this synopsis, Stage 1 scaffold code (encoder, decoder, FLAME template, model assembly) has been implemented and tested for forward-pass shape correctness. Key findings:

- DINOv2-ViT-B/14 produces 257 patch tokens of dimension 768 for a 224×224 input (32 content patches + CLS; adjusted to 37×37 = 1,369 patches for 518×518 inputs at patch size 14).
- UV decoder forward pass (B=2, F=9,976 FLAME triangles) completes in ~12 ms on CPU (expected < 2 ms on GPU).
- Zero-initialization of attribute heads confirmed: at initialization, all Gaussian attribute offsets are exactly zero, and the canonical FLAME template is recovered.

Full training has not yet commenced pending NeRSemble dataset access and DECA preprocessing pipeline completion.

### 4.4 Comparative Analysis

Expected comparison against published numbers on the ETH-XGaze within-dataset benchmark:

| Method | Gaze ↓ | Head Pose ↓ | SSIM ↑ | PSNR ↑ | LPIPS ↓ | ID Sim ↑ | FPS ↑ | Optimiz./Subject |
|--------|--------|------------|--------|--------|---------|---------|-------|------------------|
| STED \[Zheng et al., 2020\] | 16.22° | 13.15° | 0.726 | 17.53 | 0.300 | 24.3 | 18 | Yes |
| GazeNeRF \[Ruzzi et al., 2023\] | 6.94° | 3.47° | 0.733 | 15.45 | 0.291 | 45.2 | 46 | Yes |
| GazeGaussian \[Wei et al., 2025\] | 6.62° | 2.13° | 0.823 | 18.73 | 0.216 | 67.7 | 74 | Yes |
| GaussianAvatars \[Qian et al., 2024\] | 30.96° | 13.56° | 0.638 | 12.11 | 0.359 | 27.3 | 91 | Yes |
| **Aura-3D (proposed)** | **≤ 8°** | **≤ 4°** | **> 0.80** | **> 17** | **< 0.25** | **> 55** | **≥ 60** | **No** |

*Aura-3D targets are estimated based on architectural design choices; final numbers pending training completion.*

---

## 5. Implementation Timeline

### 5.1 Gantt Chart

```
Task                                  | Apr | May | Jun | Jul | Aug | Sep |
--------------------------------------|-----|-----|-----|-----|-----|-----|
WP1: FLAME binding + Gaussian init    |████ |     |     |     |     |     |
WP2: Multi-view ViT encoder           |████ |     |     |     |     |     |
WP3: UV-space parameter decoder       |████ |     |     |     |     |     |
WP4: Face deformation MLP             |     |████ |     |     |     |     |
WP5: Eye branch (GERR)                |     |████ |     |     |     |     |
WP6: Training loop + losses           |     |████ |     |     |     |     |
WP7: NeRSemble dataloader + DECA prep |     |████ |████ |     |     |     |
Stage 1 training (overfit 2-3 subj.)  |     |     |████ |     |     |     |
Stage 2 training (cross-subject)      |     |     |     |████ |████ |     |
WP8: Real-time inference driver       |     |     |     |████ |     |     |
Stage 3 training (in-the-wild)        |     |     |     |     |████ |████ |
WP9: Evaluation + baselines           |     |     |     |     |     |████ |
Writing: paper / thesis chapter        |     |     |████ |████ |████ |████ |
```

### 5.2 Major Tasks and Deliverables

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M1 | FLAME binding + GaussianModel init; forward pass verified | End of April 2026 |
| M2 | Face deformation MLP + eye GERR; animation forward pass verified | Mid May 2026 |
| M3 | NeRSemble dataloader + DECA preprocessing pipeline; Stage 1 training converges | End of May 2026 |
| M4 | Stage 1 training: overfit to 2–3 subjects; visual quality check | End of June 2026 |
| M5 | Stage 2 training: cross-subject generalization on full NeRSemble + Multiface | End of July 2026 |
| M6 | Real-time webcam driver running at ≥ 60 FPS | Mid July 2026 |
| M7 | Stage 3 training: FaceScape + EG3D in-the-wild generalization | End of August 2026 |
| M8 | Full evaluation: all metrics, ablations, baseline comparisons complete | End of September 2026 |
| M9 | Code release + pre-trained checkpoints published | October 2026 |

### 5.3 Plan for Dissemination

#### 5.3.1 Target Conference

- **Primary:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) — Submission deadline: November 2026 (for CVPR 2027)
- **Secondary:** European Conference on Computer Vision (ECCV) 2026 — if results are ready by the July submission window

#### 5.3.2 Target Journal

- **Primary:** IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI) — extended journal version with additional ablations and user study
- **Secondary:** ACM Transactions on Graphics (ACM TOG) — given the strong rendering focus of the work

---

## Appendices

### Appendix A: List of Abbreviations

| Abbreviation | Full Form |
|---|---|
| 3DGS | 3D Gaussian Splatting |
| NeRF | Neural Radiance Fields |
| FLAME | Faces Learned with an Articulated Model and Expressions |
| DECA | Detailed Expression Capture and Animation |
| GERR | Gaussian Eye Rotation Representation |
| EGNR | Expression-Guided Neural Renderer |
| ViT | Vision Transformer |
| DINOv2 | Self-DIstillation with NO labels v2 (Meta AI) |
| LPIPS | Learned Perceptual Image Patch Similarity |
| SSIM | Structural Similarity Index Measure |
| PSNR | Peak Signal-to-Noise Ratio |
| FID | Fréchet Inception Distance |
| MLP | Multi-Layer Perceptron |
| SDF | Signed Distance Function |
| DMTet | Deep Marching Tetrahedra |
| SH | Spherical Harmonics |
| CUDA | Compute Unified Device Architecture (NVIDIA) |
| FPS | Frames Per Second |
| UV | Texture Coordinate Space (U and V axes) |
| CNN | Convolutional Neural Network |

### Appendix B: Additional Materials

**B.1 Repository Layout**
```
cv_project/
  GazeGaussian/                    Reference codebase (Wei et al., 2025)
    models/
      gaze_gaussian.py             Two-stream GazeGaussianNet top-level model
      gaussian_model.py            GaussianModel: Gaussian attrs + deform MLPs
      neural_renderer.py           NeuralRenderer: pixel-shuffle UNet
      MLP.py                       Generic Conv1d skip-MLP
      mesh_head.py                 MeshHeadModule: DMTet SDF neutral mesh
    configs/
      gazegaussian_options.py      BaseOptions: all hyperparameters
    losses/
      gazenerf_loss.py             L1 + SSIM + LPIPS + angular gaze + GAN
    trainer/
      gazegaussian_trainer.py      Full training loop
    dataloader/                    ETH-XGaze, Columbia, MPII, GazeCapture
    submodules/
      diff-gaussian-rasterization/ INRIA CUDA rasterizer (unmodified)
      simple-knn/

  Aura3D/
    README.txt                     Architecture overview + dataset plan
    aura3d/
      configs/
        aura3d_default.yaml        Full model + training config
      models/
        aura3d_model.py            Aura3DModel top-level class (encode_identity / animate)
        encoders/
          multiview_vit.py         DINOv2 multi-view encoder + cross-view fusion
        decoders/
          parameter_decoder.py     UV-space CNN decoder → GaussianAttrOffsets
        flame/
          flame_template.py        FLAME wrapper: canonical verts, faces, eye mask
      __init__.py

  context.md                       Full project context reference
  Aura3D_Synopsis.md               This document
```

**B.2 Key Design Decisions Summary**

| Decision | Chosen Approach | Rejected Alternative | Reason |
|----------|----------------|---------------------|--------|
| Identity representation | UV-space CNN decoder | Unstructured MLP | Spatial coherence, training stability |
| Input views | 1–4 reference images | Strictly single image | Single view under-constrains 3D |
| Inference renderer | Pure rasterization | EGNR at inference | ≥ 60 FPS requirement |
| Animation backbone | FLAME triangle binding | Free Gaussians | Anatomical prior, no per-frame optim |
| Eye control | Explicit rigid rotation + MLP residual | Pure MLP rotation | Prevents drifting-eye artifacts |
| Backbone | DINOv2-ViT-B/14 | ResNet-50, CLIP | Dense patch tokens, spatial detail |

---

## References

\[1\] Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM Transactions on Graphics*, 42(4), 139.

\[2\] Wei, X., Chen, P., Li, G., Lu, M., Chen, H., and Tian, F. (2025). GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting. *AAAI 2025*.

\[3\] Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., and Nießner, M. (2024). GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians. *CVPR 2024*, pp. 20299–20309.

\[4\] Liu, W., Liang, S., Nguyen, H. H., and Echizen, I. (2025). A Controllable 3D Deepfake Generation Framework with Gaussian Splatting. *arXiv:2509.11624v1*.

\[5\] Kirschstein, T., Qian, S., Giebenhain, S., Walter, T., and Nießner, M. (2023). NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads. *ACM Transactions on Graphics (SIGGRAPH 2023)*, 42(4), 1–14.

\[6\] Li, T., Bolkart, T., Black, M. J., Li, H., and Romero, J. (2017). Learning a Model of Facial Shape and Expression from 4D Scans. *ACM Transactions on Graphics*, 36(6), 194.

\[7\] Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research (TMLR)*.

\[8\] Feng, Y., Feng, H., Black, M. J., and Bolkart, T. (2021). Learning an Animatable Detailed 3D Face Model from In-the-Wild Images (DECA). *ACM Transactions on Graphics*, 40(4), 1–13.

\[9\] Wuu, C., Zheng, N., Ardisson, S., et al. (2022). Multiface: A Dataset for Neural Face Rendering. *arXiv:2207.11243*.

\[10\] Yang, H., Zhu, H., Wang, Y., et al. (2023). FaceScape: 3D Facial Dataset and Benchmark for Single-View 3D Face Reconstruction. *IEEE TPAMI*.

\[11\] Zhou, S., Chan, K. C. K., Li, C., and Loy, C. C. (2022). Towards Robust Blind Face Restoration with Codebook Lookup Transformer (CodeFormer). *NeurIPS 2022*.

\[12\] Shen, T., Gao, J., Yin, K., Liu, M.-Y., and Fidler, S. (2021). Deep Marching Tetrahedra: A Hybrid Representation for High-Resolution 3D Shape Synthesis. *NeurIPS 2021*, pp. 6087–6101.

\[13\] Ruzzi, A., Shi, X., Wang, X., et al. (2023). GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields. *CVPR 2023*, pp. 9676–9685.

\[14\] Zhang, X., Park, S., Beeler, T., Bradley, D., Tang, S., and Hilliges, O. (2020). ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation. *ECCV 2020*.

\[15\] Chen, R., Chen, X., Ni, B., and Ge, Y. (2020). SimSwap: An Efficient Framework for High Fidelity Face Swapping. *ACM MM 2020*, pp. 2003–2011.

\[16\] Wang, J., Liu, Y., Hu, Y., Shi, H., and Mei, T. (2021). FaceX-Zoo: A PyTorch Toolbox for Face Recognition. *ACM MM 2021*, pp. 3779–3782.

\[17\] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*, pp. 770–778.

\[18\] Zheng, Y., Park, S., Zhang, X., De Mello, S., and Hilliges, O. (2020). Self-Learning Transformations for Improving Gaze and Head Redirection. *NeurIPS 2020*.
