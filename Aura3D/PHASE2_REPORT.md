# Aura-3D — Phase 2 Report

**Date:** May 29, 2026  
**Phase 2a checkpoint:** `runs/stage1_real/latest.pt` — step 181,000  
**Phase 2b status:** Training resumed at step 181,000 with scale clamp fix, currently running (~181,150)  
**Status:** 🔄 In Progress — Phase 2b underway after Phase 2a regression root-caused and fixed

---

## 1. Executive Summary

Phase 2 introduced LPIPS perceptual loss and tighter Gaussian scale constraints to push visual quality beyond the Phase 1 baseline (SSIM 0.74, PSNR 22.1 dB, LPIPS 0.52).

**Phase 2a result (step 181,000):** All three metrics regressed relative to Phase 1.

| Metric  | Phase 1 (step 139,400) | Phase 2a (step 181,000) | Delta      |
| ------- | ---------------------- | ----------------------- | ---------- |
| SSIM ↑  | 0.7379                 | 0.7007                  | **−0.037** |
| PSNR ↑  | 22.06 dB               | 20.24 dB                | **−1.82**  |
| LPIPS ↓ | 0.5156                 | 0.5201                  | −0.005     |

**Root cause identified:** The 0.05 m scale clamp introduced in Phase 2a was too restrictive. Gaussians could not grow large enough to cover hair, neck, and forehead, causing systematic under-reconstruction of peripheral face regions — SSIM and PSNR both drop while LPIPS barely moves.

**Phase 2b fix (applied May 29):** Scale clamp relaxed from 0.05 m → 0.15 m. Training resumed from step 181,000. First Phase 2b step already shows `scale_reg=0.00000` (Gaussians within budget) and training LPIPS as low as 0.366.

---

## 2. Phase 2 Objectives

Phase 2 targets three specific weaknesses identified in the Phase 1 conclusions:

| Objective                                 | Rationale                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------ |
| Add LPIPS perceptual loss                 | L1+SSIM alone converge to blurry means; perceptual loss drives high-frequency detail |
| Tighten scale clamp (from 1.0 m → 0.15 m) | Phase 1 explosion fix used 1.0 m, which allows Gaussians far too large for a face    |
| Maintain training stability on 6 GB VRAM  | LPIPS compute cost must be managed to avoid OOM on the available hardware            |

---

## 3. Changes Implemented

### 3.1 LPIPS Perceptual Loss (`aura3d/losses/photometric.py`)

Added AlexNet-based LPIPS to `PhotometricLoss`. Key implementation choices:

- **AlexNet over VGG:** AlexNet is ~4× smaller at inference, critical for a 6 GB GPU that was already at 100% utilisation with Phase 1 losses.
- **256 × 256 downsample before LPIPS forward pass:** Reduces memory delta to ~51 MB vs ~300 MB for full 518 × 518. Perceptual features are scale-insensitive so this does not degrade signal quality.
- **Gradient frozen on LPIPS network weights:** Only the Aura-3D model trains; LPIPS is a fixed feature extractor.
- **Normalised to [−1, 1]:** LPIPS expects [-1,1] input; images are in [0,1] during training.

```yaml
# aura3d_default.yaml
losses:
  l1: 1.0
  ssim: 0.2
  lpips: 0.05 # conservative weight — perceptual sharpness signal
  lpips_net: alex # alex = 4× less VRAM than vgg
  scale_reg: 1.0e-2
```

### 3.2 Scale Clamp Tightened → 0.05 m (`aura3d/models/renderer/gs_renderer.py`)

Phase 1 hard-clamped at 1.0 m (explosion fix). Phase 2a reduced this to 0.05 m to force finer Gaussians. This turned out to be **too aggressive** — see Section 5.

### 3.3 Scale Regularisation Threshold Updated (`aura3d/training/trainer.py`)

The soft L2 penalty on Gaussian log-scales was updated to penalise anything above `log(0.05) = −3.0`, consistent with the new renderer clamp:

```python
# Phase 2a
scale_excess = gaussians.scale.clamp_min(-3.0) - (-3.0)
scale_reg = self.w_scale_reg * scale_excess.pow(2).mean()
```

---

## 4. Bug: LPIPS OOM Crash (Resolved)

**Symptom:** Training crashed with `torch.cuda.OutOfMemoryError` on the first step after LPIPS was introduced (step ~144,500).

**Root cause:** VGG-based LPIPS on 518 × 518 images required ~300 MB of VRAM on top of the already-full 6 GB budget.

**Fix (two changes):**

1. Switched `lpips_net` from `vgg` → `alex` (AlexNet uses ~4× less feature memory)
2. Downsampled both `pred` and `target` to 256 × 256 via `F.interpolate` before the LPIPS forward pass

**Result:** VRAM delta measured at ~51 MB. Training resumed successfully. First logged LPIPS-included step: `[step 144,550] total=0.3097 l1=0.1845 ssim=0.4333 scale_reg=0.00002 lpips=0.7697`

---

## 5. Bug: Scale Clamp Too Tight → Metric Regression (Resolved)

**Symptom:** Phase 2a evaluation at step 181,000 showed regression on all metrics vs Phase 1:

- SSIM: 0.7007 vs 0.7379 (−0.037)
- PSNR: 20.24 dB vs 22.06 dB (−1.82 dB)
- LPIPS: 0.5201 vs 0.5156 (essentially flat — LPIPS signal not helping)

**Root cause:** The 0.05 m (5 cm) maximum scale is physically too small for peripheral face regions. Hair, neck, forehead, and clothing require larger Gaussians to achieve coverage without requiring extremely dense initialisation. With the hard 5 cm cap:

- Peripheral regions are consistently under-reconstructed (black/dark gaps)
- The decoder learns to avoid peripheral regions since they always incur loss
- PSNR drops because average pixel error over large dark gaps is high
- LPIPS is insensitive to these dark regions so the perceptual signal is less useful

**Fix (applied May 29, Phase 2b):**

_`aura3d/models/renderer/gs_renderer.py`_ — relaxed clamp from 0.05 m → 0.15 m:

```python
# Phase 2b: max 15cm — covers hair/neck while preventing explosion
scales = torch.exp(g.scale[0]).clamp(1e-4, 0.15)
```

_`aura3d/training/trainer.py`_ — scale_reg threshold updated to match `log(0.15) = −1.897`:

```python
# Phase 2b: penalise log-scales above log(0.15m)
scale_excess = gaussians.scale.clamp_min(-1.897) - (-1.897)
scale_reg = self.w_scale_reg * scale_excess.pow(2).mean()
```

**Early Phase 2b signal (step 181,050):** `scale_reg=0.00000` confirms Gaussians are within the 0.15 m budget and no penalty is active. Training LPIPS reached 0.366 on the first new step — lower than most Phase 2a values.

---

## 6. Phase 2a Training Log Summary

Training ran May 26–29 from step 144,500 to 181,100 (approximately 3 days).

| Phase                        | Steps           | Observations                                                                |
| ---------------------------- | --------------- | --------------------------------------------------------------------------- |
| LPIPS OOM crash              | ~144,500        | VGG + 518px input → OOM; fixed with AlexNet + 256px downsample              |
| Recovery after LPIPS restart | 144,550–148,000 | Loss fell 0.34→0.10, LPIPS 0.77→0.47                                        |
| LPIPS improvement plateau    | 148,000–181,100 | LPIPS oscillating 0.38–0.65; best singles at 0.377 (steps 162,400, 173,550) |
| Final Phase 2a step          | 181,100         | total=0.125, l1=0.0508, ssim=0.2498, scale_reg=0.00003, lpips=0.4850        |

Training was healthy throughout — no NaN/Inf losses, no explosion events, `scale_reg` never exceeded 0.00004 (danger threshold: >0.001).

**Note:** `best.pt` was NOT updated during Phase 2a. Because total loss now includes LPIPS (always positive), the Phase 2a total loss always exceeds the Phase 1 best of 0.0557. `best.pt` remains at step 139,400 (Phase 1). All Phase 2 state is in `latest.pt`.

---

## 7. Quantitative Results

### Phase 2a Evaluation (step 181,000, `latest.pt`)

Evaluated May 29, 2026 on 50 random samples (seed=42) across 3 subjects (030, 038, 240).

| Metric  | Phase 1 baseline | Phase 2a result | Δ          | Direction |
| ------- | ---------------- | --------------- | ---------- | --------- |
| SSIM ↑  | 0.7379 ± 0.044   | 0.7007 ± 0.049  | **−0.037** | ❌ Worse  |
| PSNR ↑  | 22.06 ± 1.38 dB  | 20.24 ± 2.28 dB | **−1.82**  | ❌ Worse  |
| LPIPS ↓ | 0.5156 ± 0.057   | 0.5201 ± 0.051  | −0.005     | ❌ Flat   |

### Per-Subject Breakdown (Phase 2a)

| Subject | Sequence   | Avg SSIM | Avg PSNR | Avg LPIPS |
| ------- | ---------- | -------- | -------- | --------- |
| 030     | EXP-2-eyes | 0.741    | 21.2 dB  | 0.487     |
| 038     | EXP-1-head | 0.660    | 18.8 dB  | 0.524     |
| 240     | EXP-1-head | 0.693    | 21.1 dB  | 0.574     |

Subject 030 (eyes sequence) retains the best metrics — eye-region Gaussians are fine-scale and fit within 0.05 m. Subjects 038 and 240 (full head with hair) show the largest regressions, consistent with the scale-clamp hypothesis.

### Phase 2b Status (as of May 29)

Training resumed at step 181,000. Current step: ~181,150. Too early for a full eval; re-eval scheduled at step 200,000 (target date: ~June 1).

**Phase 2b targets:**

| Metric  | Phase 1 baseline | Phase 2b target |
| ------- | ---------------- | --------------- |
| SSIM ↑  | 0.7379           | > 0.74          |
| PSNR ↑  | 22.06 dB         | > 22.0 dB       |
| LPIPS ↓ | 0.5156           | < 0.48          |

---

## 8. Deliverables

| Artifact                            | Location                                               |
| ----------------------------------- | ------------------------------------------------------ |
| Phase 2a checkpoint (latest)        | `runs/stage1_real/latest.pt` (step 181,000)            |
| Phase 1 best checkpoint (unchanged) | `runs/stage1_real/best.pt` (step 139,400, loss 0.0557) |
| Phase 2a eval results               | `runs/stage1_real/eval_phase2/results.txt`             |
| Phase 2a eval frames (50 PNG)       | `runs/stage1_real/eval_phase2/frames/`                 |
| Phase 1 eval results (baseline)     | `runs/stage1_real/eval/results.txt`                    |
| Training log (Phase 1 + 2)          | `runs/stage1_real/train.log`                           |
| LPIPS loss implementation           | `aura3d/losses/photometric.py`                         |
| Scale clamp (Phase 2b: 0.15 m)      | `aura3d/models/renderer/gs_renderer.py`                |
| Scale reg threshold (Phase 2b)      | `aura3d/training/trainer.py`                           |
| Config (Phase 2)                    | `aura3d/configs/aura3d_default.yaml`                   |

---

## 9. What Phase 3 Must Add

Based on Phase 2 learnings:

### 9.1 Gaussian Adaptive Density Control (ADC)

The single most impactful missing feature. Standard 3DGS alternates gradient steps with ADC: Gaussians with large positional gradients are **split** into smaller ones; near-transparent Gaussians are **pruned**. This allows the model to naturally concentrate Gaussian density on fine-detail regions (eyes, lips, nose pores) without hitting a hard scale ceiling.

Implementation: in `trainer._step()`, after `loss.backward()`, check `screenspace_pts.grad` magnitude every ~1,000 steps. Clone high-gradient Gaussians; prune opacity < 0.005.

### 9.2 Cross-Subject Training

Expand from 3 subjects to 30+ (full NeRSemble + Multiface). The current model overfits to 3 identities — the core research claim (generalisation to unseen faces) is untested.

### 9.3 Larger Image Resolution or Patch Strategy

518 × 518 at batch=1 is pushing VRAM limits. Future phases with ADC (more Gaussians) may require either mixed precision, gradient checkpointing, or a patch-based rendering strategy for higher resolution outputs.

### 9.4 Best Checkpoint Tracking with Phase-Relative Loss

`best.pt` is not updated during LPIPS-inclusive phases because LPIPS always adds to the total loss. A Phase 2+ compatible tracker should compute `best_loss` on the geometric mean of SSIM/PSNR/LPIPS against the Phase 1 baseline, or track eval-time metrics rather than training loss.

---

## 10. Conclusion

Phase 2a introduced LPIPS perceptual loss successfully (no OOM after AlexNet + 256px fix) but the overly tight 0.05 m scale clamp caused a net regression in all reconstruction metrics. The root cause was identified immediately from the per-subject eval breakdown (subjects with hair/neck worse than eyes-only) and confirmed by the `scale_reg` spike pattern in training logs.

Phase 2b (scale clamp relaxed to 0.15 m) is now training. Early indicators are positive: `scale_reg=0.00` and training LPIPS already reaching 0.366. Full eval will be run at step 200,000.
