#!/usr/bin/env bash
# Evaluate a fine-tuned GR00T checkpoint with the unmodified evaluator.
#   CKPT=checkpoints/groot-r1/checkpoint-20000 bash submission/groot/eval.sh
# Default: 60 episodes on the private validation block (seed 6000000), the same
# seeds as every ACT run, then a paired comparison against the ACT baseline.
# SMOKE=1 runs one 50-step smoke episode instead (never a scored result).
# Variables: EPISODES (60), SEED_START (6000000), GROOT_EXEC_HORIZON (8),
# GROOT_EMBODIMENT (must match training), GROOT_DIR, NAME.
set -euo pipefail
cd "$(dirname "$0")/../.."
GROOT_DIR=${GROOT_DIR:-$HOME/Isaac-GR00T}
CKPT=${CKPT:?set CKPT to a checkpoint directory}
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
export GROOT_CHECKPOINT="$PWD/$CKPT"
export GROOT_EXEC_HORIZON=${GROOT_EXEC_HORIZON:-8}
export GROOT_EMBODIMENT=${GROOT_EMBODIMENT:-new_embodiment}
EPISODES=${EPISODES:-60}
SEED_START=${SEED_START:-6000000}
ID="$(basename "$(dirname "$CKPT")")-$(basename "$CKPT")"
NAME=${NAME:-$(date +%Y%m%d-%H%M)-val-$EPISODES-groot-$ID-h$GROOT_EXEC_HORIZON}
common=(--policy submission.groot.policy:make_policy
        --policy-python "$GROOT_DIR/.venv/bin/python"
        --method "groot-n1.7-$GROOT_EMBODIMENT" --checkpoint "$ID")

if [ "${SMOKE:-0}" = 1 ]; then
  uv run advaitic-eval "${common[@]}" --smoke --episodes 1 --max-steps 50 \
    --out "results/smoke-groot-$(date +%H%M%S)"
  exit
fi
uv run advaitic-eval "${common[@]}" --episodes "$EPISODES" --seed-start "$SEED_START" \
  --out "results/$NAME" 2>&1 | tee "results/$NAME.log"
if [ "$SEED_START" = 6000000 ] && [ "$EPISODES" = 60 ]; then
  uv run python -m submission.tools.compare_runs \
    results/20261110-0920-val-60-last-ensemble "results/$NAME"
fi
