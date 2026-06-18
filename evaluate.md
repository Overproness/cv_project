Run evaluation from Aura3D/ against either best.pt or latest.pt.

```
export PROJECT=/mnt/d/niche-4/dental/agent1
conda activate aal
cd "$PROJECT/Aura3D"

export PYTHONPATH="$PROJECT/Aura3D:$PYTHONPATH"

RUN_DIR="$PROJECT/runs/stage1_real_fresh_YYYYMMDD_HHMMSS"  # change this
```


Quick eval, useful while training:
```
python -m aura3d.scripts.evaluate \
  --config aura3d/configs/aura3d_default.yaml \
  --ckpt "$RUN_DIR/latest.pt" \
  --out-dir "$RUN_DIR/eval_latest_$(date +%Y%m%d_%H%M%S)" \
  --n-samples 10 \
  --device cuda
```


More reliable eval:
```
python -m aura3d.scripts.evaluate \
  --config aura3d/configs/aura3d_default.yaml \
  --ckpt "$RUN_DIR/best.pt" \
  --out-dir "$RUN_DIR/eval_best_$(date +%Y%m%d_%H%M%S)" \
  --n-samples 50 \
  --device cuda
```


Each eval writes:
```
$RUN_DIR/eval_*/results.txt
$RUN_DIR/eval_*/frames/
```


The frames/ folder is important: it saves [GT | PRED | DIFF] images. Don’t trust metrics alone; look at those panels.
To plot training loss:
```
python -m aura3d.scripts.plot_loss \
  --log "$RUN_DIR/train.log" \
  --out "$RUN_DIR/loss_curve.png"
```

To render a short visual video:
```
python -m aura3d.scripts.render_video \
  --config aura3d/configs/aura3d_default.yaml \
  --ckpt "$RUN_DIR/best.pt" \
  --out-dir "$RUN_DIR/videos_$(date +%Y%m%d_%H%M%S)" \
  --participant 038 \
  --sequence EXP-1-head \
  --n-frames 50 \
  --device cuda
```

When to evaluate:
Run a tiny eval after the first 500-1000 steps just to catch broken rendering. Then run 10 samples every 5k-10k steps early on. After around 25k steps, evaluate every 25k steps or whenever the training loss improves noticeably. Run the full 50 sample eval before stopping training, after major loss drops, and at the final checkpoint.
If you only have one GPU, evaluation while training may OOM. In that case, wait for a checkpoint, pause/stop training after a save, run eval, then restart training from the same RUN_DIR; the trainer will resume automatically from latest.pt.




to tail logs: 
tail -f "$(ls -td "$PROJECT/runs"/stage1_real_fixed_* | head -n 1)/train.log"