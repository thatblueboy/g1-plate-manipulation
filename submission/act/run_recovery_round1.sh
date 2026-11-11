#!/usr/bin/env bash
# Recovery round 1 (HG-DAgger style): retrain ACT on demos + human recoveries,
# compare against the baseline on the private validation block (same 60 seeds),
# and screen fresh collection seeds with the new policy for the next round.
#   setsid nohup bash submission/act/run_recovery_round1.sh > results/recovery_round1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
BASE=checkpoints/act-baseline
NEW=checkpoints/act-recovery-r1
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

echo "== $(date +%T) pack recoveries"
uv run python -m submission.act.pack_recoveries --successful-only || exit 1

# Baseline at the same training length, evaluated while the new model trains.
evaluate val-60-base-100k "$BASE/step_0100000.pt" 6000000 60 &
BASE_PID=$!

echo "== $(date +%T) train demos + recoveries"
"$PY" -m submission.act.train --data datasets/move_plate datasets/recovery --out "$NEW" \
  --steps 100000 --batch 32 --save-every 25000 --max-hours 6 || echo "TRAIN FAILED"
wait $BASE_PID

evaluate val-60-r1-50k  "$NEW/step_0050000.pt" 6000000 60
evaluate val-60-r1-100k "$NEW/last.pt"         6000000 60

# Failing seeds of the NEW policy for recovery round 2 (collection block only).
for start in 5800000 5850000 5900000; do
  evaluate "screen-r1-$start" "$NEW/last.pt" $start 42 &
done
wait

echo "== $(date +%T) summary"
uv run python -m submission.tools.compare_runs results/20261110-0920-val-60-last-ensemble \
  results/$TAG-val-60-base-100k results/$TAG-val-60-r1-50k results/$TAG-val-60-r1-100k
python3 - "$TAG" <<'PY'
import csv, glob, sys
fails = []
for f in sorted(glob.glob(f"results/{sys.argv[1]}-screen-r1-*/episodes.csv")):
    rows = list(csv.DictReader(open(f)))
    fails += [r["seed"] for r in rows if r["success"] != "True"]
    print(f, sum(r["success"] == "True" for r in rows), "/", len(rows))
print("ROUND-2 FAILING SEEDS:", " ".join(fails))
PY
echo "== $(date +%T) ALL DONE"
