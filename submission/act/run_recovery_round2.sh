#!/usr/bin/env bash
# Recovery round 2 (HG-DAgger style): retrain ACT on the demos plus ALL successful
# human recoveries (13 from round 1 on baseline failures + 9 from round 2 on
# failures of the round-1 model), same settings as round 1, then evaluate on
# the private validation block (same 60 seeds).
#   setsid nohup bash submission/act/run_recovery_round2.sh > results/recovery_round2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
DATA=datasets/recovery_r2
NEW=checkpoints/act-recovery-r2
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

echo "== $(date +%T) pack all successful recoveries"
[ -d "$DATA" ] || uv run python -m submission.act.pack_recoveries --successful-only --out "$DATA" || exit 1

echo "== $(date +%T) train demos + round-1/2 recoveries"
"$PY" -m submission.act.train --data datasets/move_plate "$DATA" --out "$NEW" \
  --steps 100000 --batch 32 --save-every 25000 --max-hours 6 || echo "TRAIN FAILED"

evaluate val-60-r2-50k  "$NEW/step_0050000.pt" 6000000 60
evaluate val-60-r2-100k "$NEW/last.pt"         6000000 60

echo "== $(date +%T) summary"
uv run python -m submission.tools.compare_runs results/20261110-2249-val-60-base-100k \
  results/20261110-2249-val-60-r1-100k results/$TAG-val-60-r2-50k results/$TAG-val-60-r2-100k
echo "== $(date +%T) ALL DONE"
