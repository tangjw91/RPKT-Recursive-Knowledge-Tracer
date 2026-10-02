#!/usr/bin/env bash
# Paper A sweeps on Metacademy: overclaim rate, probe accuracy x verify cost, misspecified belief.
set -e
cd "$(dirname "$0")/.."
COMMON="--domains metacademy --n-learners 20 --min-ancestors 15 --budget 40 --seeds 0 1 2 --policies rpkt_v1 random eig eig_noverify"
run() { nice -n 10 python experiments/run_baselines.py $COMMON "$@"; }
run --overclaim 0.0 0.1 0.2 0.3 0.4 --tag sweep_overclaim
for q in 0.7 0.95; do for vc in 1 2; do run --overclaim 0.2 --probe-acc $q --verify-cost $vc --tag sweep_q${q}_vc${vc}; done; done
run --overclaim 0.2 --verify-cost 2 --tag sweep_q0.85_vc2
run --overclaim 0.0 0.1 0.2 0.3 0.4 --belief-overclaim 0.1 --tag sweep_misspec_b0.1
echo SWEEPS DONE
