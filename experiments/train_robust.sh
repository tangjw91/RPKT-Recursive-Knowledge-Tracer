#!/usr/bin/env bash
# Warm start from EIG (DAgger) on clean+corrupted graphs, then PPO fine-tune with random termination.
set -e
cd "$(dirname "$0")/.."
nice -n 10 python experiments/train_bc.py --iters 3 --episodes-per-iter 60 --epochs 6 --threads 2 \
  --edge-drop-range 0 0.4 --edge-add-range 0 0.3 --out models/bc_policy.pt
nice -n 10 python experiments/train_rl.py --graphs synth:6,8,3 synth:8,12,4 metacademy --updates 120 \
  --episodes-per-update 24 --n-particles 800 --threads 2 --lam 0.004 --random-termination 0.05 \
  --edge-drop-range 0 0.4 --edge-add-range 0 0.3 --init-from models/bc_policy.pt --out models/rl_robust.pt
echo TRAIN ROBUST DONE
