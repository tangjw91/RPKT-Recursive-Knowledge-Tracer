#!/usr/bin/env bash
# Wait for the running grid-1 cell (pid $1) to finish, then run its last cell single-threaded.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while kill -0 "$1" 2>/dev/null; do sleep 30; done
BASE="--domains metacademy --targets-file models/metacademy_holdout_targets.txt --n-learners 20 --min-ancestors 15 --budget 40 --seeds 0 1 2 --overclaim 0.2"
nice -n 10 python experiments/run_baselines.py $BASE --policies rpkt_v1 eig eig_noverify rl --edge-drop 0.4 --edge-add 0.2 --tag corrupt_d0.4_a0.2
echo CORRUPT EVAL 1 DONE
