"""Train the GNN probe policy with PPO in the boundary-search environment.

Reward: change in balanced accuracy of the estimate after each action, minus lambda per unit cost.
Usage: python experiments/train_rl.py --graphs synth:6,8,3 synth:8,12,4 metacademy --updates 150
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpkt2.env import BoundaryEnv  # noqa: E402
from rpkt2.graph import Universe, candidate_targets, corrupt_universe, load_graph  # noqa: E402
from rpkt2.learner import NoiseModel, SimulatedLearner  # noqa: E402
from rpkt2.metrics import unknown_set_scores  # noqa: E402
from rpkt2.policies.rl import GNNPolicyNet, RLPolicy, Transition  # noqa: E402


def collect_episode(policy: RLPolicy, env: BoundaryEnv, learner: SimulatedLearner, lam: float,
                    random_termination: float = 0.0, rng: np.random.Generator | None = None):
    obs = env.reset()
    policy.reset(env.u)
    true_unknown = learner.true_unknown
    prev = unknown_set_scores(policy.estimate(obs), true_unknown)["bal_acc"]
    trans = []
    done = False
    while not done:
        action = policy.act(obs)
        cost_before = env.cost_used
        obs, done = env.step(action)
        score = unknown_set_scores(policy.estimate(obs), true_unknown)["bal_acc"]
        t = policy.last
        t.reward = (score - prev) - lam * (env.cost_used - cost_before)
        if not done and random_termination and rng is not None and rng.random() < random_termination:
            done = True
        t.done = done
        prev = score
        trans.append(t)
    return trans, prev, env.cost_used


def ppo_update(net, opt, batch, gamma=0.9, lam_gae=0.95, clip=0.2, epochs=4, ent_coef=0.01, vf_coef=0.5):
    # GAE per episode (batch is a list of episodes, each a list of Transition)
    xs, advs, rets, acts, old_logps = [], [], [], [], []
    for ep in batch:
        values = np.array([t.value for t in ep] + [0.0])
        rewards = np.array([t.reward for t in ep])
        adv = np.zeros(len(ep))
        g = 0.0
        for k in reversed(range(len(ep))):
            delta = rewards[k] + gamma * values[k + 1] - values[k]
            g = delta + gamma * lam_gae * g
            adv[k] = g
        advs.extend(adv)
        rets.extend(adv + values[:-1])
        xs.extend(ep)
        acts.extend(t.action for t in ep)
        old_logps.extend(t.logp for t in ep)
    advs = np.array(advs)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    rets, acts, old_logps = np.array(rets), np.array(acts), np.array(old_logps)
    n = len(xs)
    stats = {}
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, 64):
            idx = perm[start:start + 64]
            loss_pi = loss_v = loss_ent = 0.0
            for j in idx:
                t: Transition = xs[j]
                logits, value = net(torch.from_numpy(t.x), torch.from_numpy(t.src), torch.from_numpy(t.dst),
                                    torch.from_numpy(t.g).unsqueeze(0))
                logits = logits.masked_fill(~torch.from_numpy(t.mask), -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(torch.tensor(acts[j]))
                ratio = torch.exp(logp - old_logps[j])
                a = float(advs[j])
                loss_pi = loss_pi - torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
                loss_v = loss_v + (value - float(rets[j])) ** 2
                loss_ent = loss_ent - dist.entropy()
            loss = (loss_pi + vf_coef * loss_v + ent_coef * loss_ent) / len(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats = {"loss_pi": float(loss_pi.detach()) / len(idx), "loss_v": float(loss_v.detach()) / len(idx)}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["synth:6,8,3", "synth:8,12,4"])
    ap.add_argument("--holdout-frac", type=float, default=0.3, help="fraction of Metacademy targets held out")
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--episodes-per-update", type=int, default=24)
    ap.add_argument("--lam", type=float, default=0.004, help="cost penalty per probe")
    ap.add_argument("--budget-range", nargs=2, type=float, default=[10, 35])
    ap.add_argument("--overclaim-range", nargs=2, type=float, default=[0.0, 0.4])
    ap.add_argument("--masteries", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--edge-drop-range", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--edge-add-range", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--init-from", default=None, help="checkpoint to warm-start from (e.g. models/bc_policy.pt)")
    ap.add_argument("--random-termination", type=float, default=0.0,
                    help="per-step probability the episode ends early, so accuracy at every budget matters")
    ap.add_argument("--n-particles", type=int, default=800)
    ap.add_argument("--no-belief", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/rl_policy.pt")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # training pool of (graph, target)
    pool = []
    for name in args.graphs:
        g = load_graph(name, np.random.default_rng(args.seed))
        targets = candidate_targets(g, min_ancestors=15)
        if name == "metacademy":
            targets = sorted(targets)
            rng_split = np.random.default_rng(1234)
            rng_split.shuffle(targets)
            n_hold = int(len(targets) * args.holdout_frac)
            held = targets[:n_hold]
            targets = targets[n_hold:]
            Path("models").mkdir(exist_ok=True)
            Path("models/metacademy_holdout_targets.txt").write_text("\n".join(held))
        pool.extend((name, g, t) for t in targets)
    print(f"training pool: {len(pool)} (graph, target) pairs", flush=True)

    net = GNNPolicyNet()
    if args.init_from:
        net.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=False)["state_dict"])
        print("warm start from", args.init_from, flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    t0 = time.time()
    for upd in range(args.updates):
        batch, finals, costs = [], [], []
        for _ in range(args.episodes_per_update):
            name, g, target = pool[rng.integers(len(pool))]
            u = Universe(g, target)
            p = float(rng.uniform(*args.overclaim_range))
            noise = NoiseModel(p_overclaim=p, r_underclaim=0.05, q_probe=0.85)
            learner = SimulatedLearner.sample(u, float(rng.choice(args.masteries)), noise, rng)
            budget = float(rng.integers(int(args.budget_range[0]), int(args.budget_range[1]) + 1))
            drop, add = float(rng.uniform(*args.edge_drop_range)), float(rng.uniform(*args.edge_add_range))
            u = corrupt_universe(u, drop, add, rng) if (drop or add) else u
            policy = RLPolicy(net=net, noise=noise, rng=np.random.default_rng(rng.integers(1 << 31)),
                              n_particles=args.n_particles, use_belief=not args.no_belief, deterministic=False)
            env = BoundaryEnv(u, learner, budget=budget)
            trans, final, cost = collect_episode(policy, env, learner, args.lam, args.random_termination, rng)
            batch.append(trans)
            finals.append(final)
            costs.append(cost)
        stats = ppo_update(net, opt, batch)
        if upd % 5 == 0 or upd == args.updates - 1:
            print(f"update {upd:4d}  bal_acc {np.mean(finals):.3f}  cost {np.mean(costs):5.1f}  "
                  f"loss_pi {stats.get('loss_pi', 0):+.3f}  {time.time() - t0:5.0f}s", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "args": vars(args)}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
