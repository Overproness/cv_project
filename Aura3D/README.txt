Aura-3D: Feed-Forward 3D Gaussian Splatting Avatar Synthesizer

Project layout
==============

aura3d/
  configs/                 Hydra/YAML configs for model + training
  data/
    datasets/              Dataset wrappers (NeRSemble, Multiface, FaceScape)
    preprocessing/         Face tracking (DECA), FLAME fitting, mask extraction
    transforms.py
  models/
    encoders/              ViT / DINOv2 multi-view image encoder
    decoders/              UV-space parameter decoder (predicts per-FLAME-vertex Gaussian attrs)
    flame/                 FLAME wrapper, mesh utilities, UV unwrapping
    gaussians/             Gaussian model, FLAME->Gaussian binding, two-stream face+eye
    deformation/           Expression deformation MLPs (face branch)
    eye/                   GERR: explicit gaze rotation + residual MLP (eye branch)
    renderer/              diff-gaussian-rasterization wrapper, optional EGNR (training only)
    refine/                CodeFormer wrapper (training-time supervision cleanup ONLY)
    aura3d_model.py        Top-level Aura3D_Model assembling all components
  losses/                  L1, LPIPS, SSIM, gaze, identity, FLAME regularizers
  training/
    trainer.py             Lightning-style trainer
    schedulers.py
  inference/
    predictor.py           Single-shot avatar build from N reference images
    driver.py              Live webcam -> FLAME params -> render loop (target 60+ FPS)
  utils/
    camera.py
    sh.py
    logging.py
  scripts/
    train_stage1_overfit.py
    train_stage2_crosssubject.py
    train_stage3_inthewild.py
    fit_flame.py
  third_party/
    flame_pytorch/         git submodule
    DECA/                  git submodule
    diff-gaussian-rasterization/   git submodule (unmodified)
    simple-knn/            git submodule

Two-stage rendering pipeline (inference)
========================================
  ref_imgs (1-4) ──► ViT encoder ──► UV feature map ──► Decoder ──► personalized
                                                                    Gaussian attrs
                                                                    bound to FLAME

  webcam ──► DECA tracker ──► (expr, pose, gaze) ──► FaceDeformMLP + GERR
                                                       ──► transformed Gaussians
                                                       ──► rasterizer ──► frame

Datasets needed (phased)
========================
  Phase 1 — sanity / overfit:
    NeRSemble subset: 2-3 subjects, 16 views, 50 frames  (~5 GB)
  Phase 2 — cross-subject generalization:
    NeRSemble full release (~30 subjects)
    Multiface mini-dataset (16 GB)
  Phase 3 — in-the-wild:
    FaceScape multi-view (~120 GB, request access)
    EG3D-synthesized multi-view samples (generate ~50k identities)
