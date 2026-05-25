#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flame_tracking.sh — Run DECA/FLAME tracking for all NeRSemble subjects.
#
# Pre-requisites
# --------------
#   1. setup_flame.sh must have been run (generic_model.pkl in place)
#   2. NeRSemble data at /mnt/d/GitHub/cv_project/data/nersemble/
#   3. conda env at /mnt/d/envs/pytorch
# ---------------------------------------------------------------------------
set -e

CONDA_ENV="/mnt/d/envs/pytorch"
AURA3D="/mnt/d/GitHub/cv_project/Aura3D"
NERSEMBLE_ROOT="/mnt/d/GitHub/cv_project/data/nersemble"
MAX_T=50
DEVICE="cuda"

# Participant → sequence → ref-camera
declare -A SEQ
SEQ["030/EXP-2-eyes"]="222200037"
SEQ["038/EXP-1-head"]="222200037"
SEQ["038/EXP-4-lips"]="222200037"
SEQ["240/EXP-1-head"]="222200037"
SEQ["240/EXP-4-lips"]="222200037"

cd "$AURA3D"

for KEY in "${!SEQ[@]}"; do
    IFS='/' read -r PARTICIPANT SEQUENCE <<< "$KEY"
    REF_CAM="${SEQ[$KEY]}"
    OUT="$NERSEMBLE_ROOT/$PARTICIPANT/flame_tracking/$SEQUENCE"

    # Skip if already tracked (all 50 frames present)
    if [[ -f "$OUT/$(printf '%06d' $((MAX_T-1))).npz" ]]; then
        echo "Already tracked: $PARTICIPANT / $SEQUENCE  (skipping)"
        continue
    fi

    echo "=== FLAME tracking: participant=$PARTICIPANT  seq=$SEQUENCE ==="
    conda run --prefix "$CONDA_ENV" --no-capture-output \
        python -m aura3d.scripts.fit_flame \
            --root "$NERSEMBLE_ROOT" \
            --participant "$PARTICIPANT" \
            --sequence "$SEQUENCE" \
            --ref-camera "$REF_CAM" \
            --max-timesteps "$MAX_T" \
            --device "$DEVICE"
done

echo ""
echo "All FLAME tracking complete. Results at $NERSEMBLE_ROOT/*/flame_tracking/"
