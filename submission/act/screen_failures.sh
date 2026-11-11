#!/usr/bin/env bash
# Find seeds the current policy fails on, in the recovery-collection block
# (5,500,000+), so human recoveries can be recorded on them without touching
# validation (6,000,000+) or public evaluation (7,310,000+) seeds.
#   bash submission/act/screen_failures.sh > results/screen.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
CKPT=checkpoints/act-baseline/last.pt
TAG=$(date +%Y%m%d-%H%M)
# Three disjoint 42-episode blocks (balanced-slot seeding may skip seeds, so starts are far apart).
for start in 5500000 5600000 5700000; do
  ACT_CHECKPOINT="$PWD/$CKPT" uv run advaitic-eval \
    --policy submission.act.policy:make_policy --policy-python "$PY" \
    --method act-baseline-ensemble --checkpoint last.pt --episodes 42 \
    --seed-start $start --out "results/$TAG-screen-$start" > "results/$TAG-screen-$start.log" 2>&1 &
done
wait
python3 - "$TAG" <<'PY'
import csv, glob, sys
fails = []
for f in sorted(glob.glob(f"results/{sys.argv[1]}-screen-*/episodes.csv")):
    rows = list(csv.DictReader(open(f)))
    fails += [r["seed"] for r in rows if r["success"] != "True"]
    print(f, sum(r["success"] == "True" for r in rows), "/", len(rows))
print("FAILING SEEDS:", " ".join(fails))
PY
