"""Run one policy on one (target, learner) episode and record the metric trajectory."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .env import BoundaryEnv
from .learner import SimulatedLearner
from .metrics import cost_to_reach, unknown_set_scores
from .policies.base import Policy


BUDGET_CHECKPOINTS = (5, 10, 15, 20, 30)


def score_at_budget(trajectory: List[Tuple[float, float]], budget: float, initial: float) -> float:
    """Score of the estimate the policy held when its spend last fitted inside `budget`."""
    score = initial
    for cost, s in trajectory:
        if cost > budget:
            break
        score = s
    return score


def run_episode(policy: Policy, env: BoundaryEnv, learner: SimulatedLearner, f1_target: float = 0.9) -> Dict:
    """f1_target is the balanced-accuracy threshold used for cost_to_target."""
    obs = env.reset()
    policy.reset(env.u)
    traj: List[Tuple[float, float]] = []
    true_unknown = learner.true_unknown
    initial = unknown_set_scores(policy.estimate(obs), true_unknown)["bal_acc"]
    done = False
    n_ask = n_verify = 0
    while not done:
        action = policy.act(obs)
        if action[0] == "ask":
            n_ask += 1
        elif action[0] == "verify":
            n_verify += 1
        obs, done = env.step(action)
        traj.append((env.cost_used, unknown_set_scores(policy.estimate(obs), true_unknown)["bal_acc"]))
    est = policy.estimate(obs)
    out = unknown_set_scores(est, true_unknown)
    over = learner.overclaimed
    out.update({
        "cost": env.cost_used,
        "n_ask": n_ask,
        "n_verify": n_verify,
        "cost_to_target": cost_to_reach(traj, f1_target),
        "overclaims_total": int(over.sum()),
        "overclaims_caught": int((over & est).sum()),
        "universe_size": env.u.n,
        **{f"balacc_at_{b}": score_at_budget(traj, b, initial) for b in BUDGET_CHECKPOINTS},
        "true_unknown_size": int(true_unknown.sum()),
    })
    return out
