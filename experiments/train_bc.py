"""Imitation warm start: train the GNN policy to reproduce the EIG heuristic's decisions (DAgger).

Iteration 0 follows EIG's own trajectories; later iterations follow the student's trajectories
while still labelling every state with EIG's choice. Writes a checkpoint loadable by
train_rl.py --init-from and by the 'rl' policy registry entry.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpkt2.env import BoundaryEnv  # noqa: E402
from rpkt2.graph import Universe, candidate_targets, corrupt_universe, load_graph  # noqa: E402
from rpkt2.learner import NoiseModel, SimulatedLearner  # noqa: E402
from rpkt2.policies.eig import EIGPolicy  # noqa: E402
from rpkt2.policies.rl import GNNPolicyNet, RLPolicy  # noqa: E402


def build_pool(graphs, seed, holdout_frac):
    pool = []
    for name in graphs:
        g = load_graph(name, np.random.default_rng(seed))
        targets = candidate_targets(g, min_ancestors=15)
        if name == "metacademy":
            targets = sorted(targets)
            np.random.default_rng(1234).shuffle(targets)
            targets = targets[int(len(targets) * holdout_frac):]
        pool.extend((name, g, t) for t in targets)
    return pool


def sample_episode_setup(pool, rng, args):
    name, g, target = pool[rng.integers(len(pool))]
    u = Universe(g, target)
    p = float(rng.uniform(*args.overclaim_range))
    noise = NoiseModel(p_overclaim=p, r_underclaim=0.05, q_probe=0.85)
    learner = SimulatedLearner.sample(u, float(rng.choice(args.masteries)), noise, rng)
    drop, add = float(rng.uniform(*args.edge_drop_range)), float(rng.uniform(*args.edge_add_range))
    u_pol = corrupt_universe(u, drop, add, rng) if (drop or add) else u
    budget = float(rng.integers(int(args.budget_range[0]), int(args.budget_range[1]) + 1))
    return u_pol, learner, noise, budget


def collect(pool, rng, args, net, beta):
    """One episode; returns list of (x, src, dst, g, mask, expert_action)."""
    u_pol, learner, noise, budget = sample_episode_setup(pool, rng, args)
    expert = EIGPolicy(noise=noise, rng=np.random.default_rng(rng.integers(1 << 31)), n_particles=args.n_particles)
    student = RLPolicy(net=net, noise=noise, rng=np.random.default_rng(rng.integers(1 << 31)),
                       n_particles=args.n_particles, deterministic=False)
    env = BoundaryEnv(u_pol, learner, budget=budget)
    obs = env.reset()
    expert.reset(u_pol)
    student.reset(u_pol)
    data, done = [], False
    while not done:
        a_exp = expert.act(obs)
        n = u_pol.n
        a_idx = 2 * n if a_exp[0] == "stop" else (a_exp[1] if a_exp[0] == "ask" else n + a_exp[1])
        a_stu = student.act(obs)  # also fills student.last with tensors + mask
        t = student.last
        data.append((t.x, t.src, t.dst, t.g, t.mask, a_idx))
        action = a_exp if rng.random() < beta else a_stu
        obs, done = env.step(action)
    return data


def fit(net, opt, data, epochs, batch=64):
    n = len(data)
    for _ in range(epochs):
        perm = np.random.permutation(n)
        tot, correct = 0.0, 0
        for s in range(0, n, batch):
            loss = 0.0
            for j in perm[s:s + batch]:
                x, src, dst, g, mask, a = data[j]
                logits, _ = net(torch.from_numpy(x), torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(g).unsqueeze(0))
                logits = logits.masked_fill(~torch.from_numpy(mask), -1e9)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([a]))
                correct += int(torch.argmax(logits) == a)
            loss = loss / len(perm[s:s + batch])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(perm[s:s + batch])
    return tot / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["synth:6,8,3", "synth:8,12,4", "metacademy"])
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=3, help="DAgger iterations (first one is pure behaviour cloning)")
    ap.add_argument("--episodes-per-iter", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--budget-range", nargs=2, type=float, default=[10, 35])
    ap.add_argument("--overclaim-range", nargs=2, type=float, default=[0.0, 0.4])
    ap.add_argument("--edge-drop-range", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--edge-add-range", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--masteries", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--n-particles", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="models/bc_policy.pt")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pool = build_pool(args.graphs, args.seed, args.holdout_frac)
    net = GNNPolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    data, t0 = [], time.time()
    for it in range(args.iters):
        beta = 1.0 if it == 0 else 0.5 ** it
        for _ in range(args.episodes_per_iter):
            data.extend(collect(pool, rng, args, net, beta))
        loss, acc = fit(net, opt, data, args.epochs)
        print(f"iter {it}  beta {beta:.2f}  labels {len(data)}  loss {loss:.3f}  agreement {acc:.3f}  {time.time() - t0:.0f}s", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "args": {**vars(args), "no_belief": False}}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
