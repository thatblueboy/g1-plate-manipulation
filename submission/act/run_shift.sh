#!/usr/bin/env bash
# ACT trained ONLY on shifted-placement demos (datasets/move_plate_shift, from
# submission/tools/augment_sidestep.py: slots 3/4/5), same settings as the
# baseline, then evaluated on the private validation block (same 60 seeds).
#   setsid nohup bash submission/act/run_shift.sh > results/shift.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
NEW=checkpoints/act-shift
TAG=$(date +%Y%m%d-%H%M)
mkdir -p results

evaluate () {  # name ckpt seed_start episodes
  local name=$1 ckpt=$2 start=$3 n=$4
  [ -f "$ckpt" ] || { echo "skip $name: no $ckpt"; return; }
  echo "== $(date +%T) eval $name ($ckpt, seeds $start, $n ep)"
  ACT_CHECKPOINT="$PWD/$ckpt" ACT_MODE=ensemble uv run advaitic-eval \
    --policy submission.act.policy:make_policy --policy-python "$PY" \
    --method "act-$name" --checkpoint "$(basename "$ckpt")-$(sha256sum "$ckpt" | cut -c1-16)" \
    --episodes "$n" --seed-start "$start" --out "results/$TAG-$name" \
    > "results/$TAG-$name.log" 2>&1
  echo "== $(date +%T) done $name"
}

echo "== $(date +%T) train shifted demos only"
"$PY" -m submission.act.train --data datasets/move_plate_shift --out "$NEW" \
  --steps 100000 --batch 32 --save-every 25000 --max-hours 6 || echo "TRAIN FAILED"

evaluate val-60-shift-50k  "$NEW/step_0050000.pt" 6000000 60
evaluate val-60-shift-100k "$NEW/last.pt"         6000000 60

echo "== $(date +%T) summary"
uv run python -m submission.tools.compare_runs results/20261110-0920-val-60-last-ensemble \
  results/*-val-60-base-100k results/$TAG-val-60-shift-50k results/$TAG-val-60-shift-100k
echo "== $(date +%T) ALL DONE"
