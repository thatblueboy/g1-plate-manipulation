#!/usr/bin/env bash
# Chunk-length ablation: same data as recovery round 1 (demos + recoveries),
# chunk K=20 instead of 50, evaluated on the same 60 validation seeds.
# Waits for run_recovery_round1.sh to finish so the GPU is not shared.
#   setsid nohup bash submission/act/run_chunk20.sh > results/chunk20.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
OUT=checkpoints/act-recovery-r1-k20

while pgrep -f run_recovery_round1.sh > /dev/null; do sleep 60; done
TAG=$(date +%Y%m%d-%H%M)

echo "== $(date +%T) train K=20 on demos + recoveries"
"$PY" -m submission.act.train --data datasets/move_plate datasets/recovery --out "$OUT" \
  --chunk 20 --steps 100000 --batch 32 --save-every 25000 --max-hours 6 || echo "TRAIN FAILED"

echo "== $(date +%T) eval"
ACT_CHECKPOINT="$PWD/$OUT/last.pt" ACT_MODE=ensemble uv run advaitic-eval \
  --policy submission.act.policy:make_policy --policy-python "$PY" \
  --method act-r1-k20 --checkpoint "last.pt-$(sha256sum "$OUT/last.pt" | cut -c1-16)" \
  --episodes 60 --seed-start 6000000 --out "results/$TAG-val-60-r1-k20-100k" \
  > "results/$TAG-val-60-r1-k20-100k.log" 2>&1

echo "== $(date +%T) summary (baseline 200k, round-1 K=50 100k, round-1 K=20 100k)"
uv run python -m submission.tools.compare_runs results/20261110-0920-val-60-last-ensemble \
  $(ls -d results/*-val-60-r1-100k | tail -1) "results/$TAG-val-60-r1-k20-100k"
echo "== $(date +%T) ALL DONE"
