#!/bin/bash
# Train many seeds in parallel. The bottleneck is the environment, which refits a
# scikit-learn ensemble every round, so this is CPU-bound and scales with cores, not
# with GPUs. One process per seed, each pinned to a single thread, is far faster than
# a few processes each trying to use every core: the refits then fight each other.
#
#   ./scripts/run_fleet.sh 30 200        # 30 seeds, 200 updates each
set -u
SEEDS="${1:-30}"
UPDATES="${2:-200}"
cd "$(dirname "$0")/.."
mkdir -p results/fleet

CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)
# leave two cores for the OS; never run more workers than seeds
WORKERS=$(( CORES - 2 )); [ "$WORKERS" -lt 1 ] && WORKERS=1
[ "$WORKERS" -gt "$SEEDS" ] && WORKERS=$SEEDS
echo "$CORES cores detected, running $WORKERS seeds at a time, $SEEDS total, $UPDATES updates each"

# one thread per worker, or the sklearn refits oversubscribe and everything crawls
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_one() {
  s="$1"
  ./.venv/bin/python -u scripts/train.py \
    --updates "$UPDATES" --episodes-per-update 8 --rounds 18 --budget 0.10 \
    --seed "$s" --out "results/fleet/ppo_s${s}.pt" \
    > "results/fleet/train_s${s}.log" 2>&1
  echo "  seed $s finished $(date +%H:%M:%S)"
}
export -f run_one 2>/dev/null || true

pids=()
for s in $(seq 0 $((SEEDS - 1))); do
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 5; done
  run_one "$s" &
  pids+=($!)
done
wait
echo "ALL $SEEDS SEEDS COMPLETE $(date +%H:%M:%S)"
ls results/fleet/*.pt | wc -l | xargs echo "models written:"
