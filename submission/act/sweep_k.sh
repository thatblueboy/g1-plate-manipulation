#!/usr/bin/env bash
# Test-time sweep of the temporal-ensembling weight k (w_i = exp(-k*i), i=0 oldest)
# on the private validation block. Same checkpoint, same 60 seeds for every k.
#   bash submission/act/sweep_k.sh > results/sweep_k.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
export ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl
PY="$PWD/.venv-act/bin/python"
CKPT=checkpoints/act-baseline/last.pt
ID=$(sha256sum "$CKPT" | cut -c1-16)
TAG=$(date +%Y%m%d-%H%M)
KS=(${KS:--0.1 -0.05 -0.01 0 0.05 0.01})
PARALLEL=${PARALLEL:-3}

run () {
  local k=$1 out="results/$TAG-val-60-k$1"
  echo "== $(date +%T) start k=$k -> $out"
  ACT_CHECKPOINT="$PWD/$CKPT" ACT_MODE=ensemble ACT_ENSEMBLE_K=$k uv run advaitic-eval \
    --policy submission.act.policy:make_policy --policy-python "$PY" \
    --method "act-baseline-ensemble-k$k" --checkpoint "last.pt-$ID" \
    --episodes 60 --seed-start 6000000 --out "$out" > "$out.log" 2>&1
  echo "== $(date +%T) done  k=$k: $(tail -1 "$out.log")"
}

for k in "${KS[@]}"; do
  run "$k" &
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
done
wait
echo "== ALL DONE $TAG"
