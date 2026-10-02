#!/usr/bin/env bash
# Remaining cells of both corrupted-graph grids, single-threaded torch (no core oversubscription).
# Usage: corrupt_eval_rest.sh <pid-to-wait-for-before-running-grid-1-tail>
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BASE="--domains metacademy --targets-file models/metacademy_holdout_targets.txt --n-learners 20 --min-ancestors 15 --budget 40 --seeds 0 1 2 --overclaim 0.2"
for cell in "0.2 0.0" "0.2 0.2" "0.4 0.0" "0.4 0.2"; do set -- $cell
  nice -n 10 python experiments/run_baselines.py $BASE --policies bc rl_robust --edge-drop $1 --edge-add $2 --tag "corrupt2_d$1_a$2"
done
echo CORRUPT EVAL 2 DONE
