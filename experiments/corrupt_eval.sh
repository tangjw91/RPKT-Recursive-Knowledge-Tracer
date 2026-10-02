#!/usr/bin/env bash
# Evaluate policies on corrupted prerequisite graphs (held-out Metacademy targets, overclaim 0.2).
set -e
cd "$(dirname "$0")/.."
POL="${POLICIES:-rpkt_v1 eig eig_noverify rl}"
COMMON="--domains metacademy --targets-file models/metacademy_holdout_targets.txt --n-learners 20 --min-ancestors 15 --budget 40 --seeds 0 1 2 --overclaim 0.2 --policies $POL"
for d in 0.0 0.2 0.4; do for a in 0.0 0.2; do
  nice -n 10 python experiments/run_baselines.py $COMMON --edge-drop $d --edge-add $a --tag "${TAG:-corrupt}_d${d}_a${a}"
done; done
echo CORRUPT EVAL DONE
