#!/usr/bin/env bash
# Overnight ACT baseline: train, compare checkpoints on a private validation
# seed block, then run ONE scored evaluation on the public seeds.
#   setsid nohup bash submission/act/run_overnight.sh > checkpoints/act-baseline/overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
CK=checkpoints/act-baseline
VAL_SEEDS=6000000   # private validation block, disjoint from demos (18-147) and public eval (7310000+)
TAG=$(date +%Y%m%d-%H%M)
mkdir -p "$CK" results

echo "== $(date) train"
"$PY" -m submission.act.train --data datasets/move_plate --out "$CK" \
  --steps 200000 --batch 32 --max-hours 9 --save-every 25000 || echo "TRAIN FAILED"

evaluate () {  # name ckpt mode seed_start [extra args]
  local name=$1 ckpt=$2 mode=$3 seeds=$4; shift 4
  [ -f "$ckpt" ] || { echo "skip $name: no $ckpt"; return; }
  local id; id=$(sha256sum "$ckpt" | cut -c1-16)
  echo "== $(date) eval $name ($ckpt, mode=$mode, seeds=$seeds)"
  ACT_CHECKPOINT="$PWD/$ckpt" ACT_MODE=$mode uv run advaitic-eval \
    --policy submission.act.policy:make_policy --policy-python "$PY" \
    --method "act-baseline-$mode" --checkpoint "$(basename "$ckpt")-$id" \
    --episodes 30 --seed-start "$seeds" --out "results/$TAG-$name" "$@" 2>&1 | tail -4
}

# Validation (private seeds): checkpoint and execution-mode comparison.
evaluate val-step100k-ensemble "$CK/step_0100000.pt" ensemble $VAL_SEEDS
evaluate val-last-ensemble     "$CK/last.pt"         ensemble $VAL_SEEDS
evaluate val-last-chunk10      "$CK/last.pt"         chunk    $VAL_SEEDS

# One scored run on the public seed block, final checkpoint, default mode.
evaluate public-last-ensemble  "$CK/last.pt"         ensemble 7310000 --video-episodes 3

echo "== $(date) ALL DONE"
for d in results/$TAG-*; do
  python3 -c "import json,sys;r=json.load(open(sys.argv[1]+'/report.json'));print(sys.argv[1], r.get('successes'), '/', r.get('completed_episodes'), r.get('success_rate'), r.get('per_source_slot'))" "$d" 2>/dev/null
done
