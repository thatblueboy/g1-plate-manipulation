#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on a converted move_plate dataset (see convert_dataset.py).
#   DATA=datasets/groot/move_plate_r1 OUT=checkpoints/groot-r1 bash submission/groot/train.sh
# SMOKE=1 runs 20 steps first: it generates meta/stats.json and checks that the
# data loader, modality config and model accept the dataset before a long run.
# Variables: GROOT_EMBODIMENT (new_embodiment | unitree_g1), STEPS (20000),
# BATCH (32), SAVE_EVERY (5000), SAVE_LIMIT (10), SAVE_ONLY_MODEL (unset; set
# to 1 to skip optimizer state: ~12 GB instead of ~24 GB per checkpoint, but
# the run cannot be resumed), NUM_GPUS (1), GROOT_DIR.
set -euo pipefail
cd "$(dirname "$0")/../.."
GROOT_DIR=${GROOT_DIR:-$HOME/Isaac-GR00T}
PY="$GROOT_DIR/.venv/bin/python"
DATA=${DATA:?set DATA to a converted dataset directory}
OUT=${OUT:?set OUT to a new checkpoint directory}
export GROOT_EMBODIMENT=${GROOT_EMBODIMENT:-new_embodiment}
TAG=$(echo "$GROOT_EMBODIMENT" | tr a-z A-Z)
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
SAVE_EVERY=${SAVE_EVERY:-5000}
NUM_GPUS=${NUM_GPUS:-1}

run () {  # steps output_dir
  "$PY" "$GROOT_DIR/gr00t/experiment/launch_finetune.py" \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$PWD/$DATA" \
    --embodiment-tag "$TAG" \
    --modality-config-path "$PWD/submission/groot/modality_config.py" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$2" \
    --max-steps "$1" \
    --save-steps "$SAVE_EVERY" \
    --save-total-limit "${SAVE_LIMIT:-10}" \
    ${SAVE_ONLY_MODEL:+--save-only-model} \
    --global-batch-size "$BATCH" \
    --dataloader-num-workers 8
}

if [ "${SMOKE:-0}" = 1 ]; then
  echo "== $(date +%T) smoke: 20 steps ($GROOT_EMBODIMENT)"
  run 20 "$OUT-smoke"
  ls "$PWD/$DATA/meta"
fi
echo "== $(date +%T) fine-tune $STEPS steps, batch $BATCH, $GROOT_EMBODIMENT -> $OUT"
mkdir -p "$OUT"
cp submission/groot/layout.py submission/groot/modality_config.py "$OUT/"
echo "GROOT_EMBODIMENT=$GROOT_EMBODIMENT DATA=$DATA STEPS=$STEPS BATCH=$BATCH" > "$OUT/run.txt"
run "$STEPS" "$OUT"
echo "== $(date +%T) done"
