#!/bin/bash
# Three training seeds, sequentially, 200 updates x 8 episodes = 1,600 episodes each.
# Sequential rather than parallel because the environment refits a gradient boosting
# detector every round and three of those contend badly on six cores.
# Uses $PYTHON if set (e.g. PYTHON=.venv/bin/python), otherwise python on the PATH.
set -u
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  echo "########## seed $s starting $(date +%H:%M:%S) ##########"
  "${PYTHON:-python}" -u scripts/train.py \
    --updates 200 --episodes-per-update 8 --rounds 18 --budget 0.10 \
    --seed "$s" --out "results/ppo_s${s}.pt"
  echo "########## seed $s done $(date +%H:%M:%S) ##########"
done
echo "ALL SEEDS COMPLETE $(date +%H:%M:%S)"
