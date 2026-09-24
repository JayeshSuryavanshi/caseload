#!/bin/bash
# Evaluate every trained seed on the same held-out drift episodes, in parallel.
set -u
cd "$(dirname "$0")/.."
CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)
WORKERS=$(( CORES - 2 )); [ "$WORKERS" -lt 1 ] && WORKERS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "evaluating $(ls results/fleet/ppo_s*.pt 2>/dev/null | wc -l) models, $WORKERS at a time"
for m in results/fleet/ppo_s*.pt; do
  [ -e "$m" ] || { echo "no models in results/fleet/"; exit 1; }
  s=$(basename "$m" .pt | sed 's/ppo_s//')
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
  ./.venv/bin/python -u scripts/evaluate.py --model "$m" --seeds 20 \
    --out "results/fleet/eval_s${s}.json" > "results/fleet/eval_s${s}.log" 2>&1 &
done
wait
echo "ALL EVALUATIONS COMPLETE $(date +%H:%M:%S)"
