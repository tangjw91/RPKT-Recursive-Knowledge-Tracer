"""Learned probe policy: a message-passing GNN over the local prerequisite DAG, trained with PPO.

State per node: structural features (depth, in/out degree), evidence (asked?, said-know?,
verified?, passed?), and optionally the particle-posterior marginal. Actions: ask(i), verify(i),
or stop; invalid actions are masked. The particle belief is always kept for the final estimate so
that the learned policy and the EIG heuristic are scored identically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..belief import ParticleBelief
from ..env import Action, Observation
from ..graph import Universe
from .base import Policy

N_NODE_FEATS = 9
N_GLOBAL_FEATS = 3


def node_features(obs: Observation, belief: Optional[ParticleBelief], use_belief: bool) -> np.ndarray:
    u, ev = obs.u, obs.evidence
    n = u.n
    x = np.zeros((n, N_NODE_FEATS), dtype=np.float32)
    x[:, 0] = u.depth / max(u.depth.max(), 1)
    x[:, 1] = np.array([len(p) for p in u.prereqs]) / 6.0
    x[:, 2] = np.array([len(d) for d in u.dependents]) / 6.0
    for i, a in ev.self_report.items():
        x[i, 3] = 1.0
        x[i, 4] = 1.0 if a else -1.0
    for i, a in ev.probe.items():
        x[i, 5] = 1.0
        x[i, 6] = 1.0 if a else -1.0
    if use_belief and belief is not None:
        m = belief.marginals()
        x[:, 7] = m
        x[:, 8] = -(np.clip(m, 1e-6, 1 - 1e-6) * np.log(np.clip(m, 1e-6, 1 - 1e-6))
                    + (1 - np.clip(m, 1e-6, 1 - 1e-6)) * np.log(1 - np.clip(m, 1e-6, 1 - 1e-6)))
    x[u.target_idx, 7] = 0.0
    return x


def edge_index(u: Universe) -> Tuple[np.ndarray, np.ndarray]:
    src, dst = [], []
    for i in range(u.n):
        for p in u.prereqs[i]:
            src.append(p)
            dst.append(i)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


class GNNPolicyNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(N_NODE_FEATS + N_GLOBAL_FEATS, hidden)
        self.down = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))  # prereq -> dependent
        self.up = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))    # dependent -> prereq
        self.self_ = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.ask_head = nn.Linear(hidden, 1)
        self.verify_head = nn.Linear(hidden, 1)
        self.stop_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, g: torch.Tensor):
        n = x.shape[0]
        h = F.relu(self.inp(torch.cat([x, g.expand(n, -1)], dim=1)))
        for lin_d, lin_u, lin_s in zip(self.down, self.up, self.self_):
            agg_down = torch.zeros_like(h).index_add_(0, dst, h[src])
            agg_up = torch.zeros_like(h).index_add_(0, src, h[dst])
            h = F.relu(lin_s(h) + lin_d(agg_down) + lin_u(agg_up)) + h
        pooled = h.mean(dim=0, keepdim=True)
        logits = torch.cat([self.ask_head(h).squeeze(1), self.verify_head(h).squeeze(1),
                            self.stop_head(pooled).squeeze(1)])
        return logits, self.value_head(pooled).squeeze()


@dataclass
class Transition:
    x: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    g: np.ndarray
    mask: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class RLPolicy(Policy):
    """Acts with a trained GNNPolicyNet; estimates with the particle posterior."""
    name = "rl"

    def __init__(self, net: Optional[GNNPolicyNet] = None, n_particles: int = 1500, use_belief: bool = True,
                 verify_cost: float = 1.0, deterministic: bool = True, allow_verify: bool = True, **kw):
        super().__init__(**kw)
        self.net = net or GNNPolicyNet()
        self.n_particles, self.use_belief = n_particles, use_belief
        self.verify_cost, self.deterministic, self.allow_verify = verify_cost, deterministic, allow_verify
        self.last: Optional[Transition] = None

    def reset(self, u: Universe) -> None:
        super().reset(u)
        self.belief = ParticleBelief(u, self.noise, self.rng, n_particles=self.n_particles)
        self._seen_ask, self._seen_verify = set(), set()
        self.src, self.dst = edge_index(u)

    def _absorb(self, obs: Observation) -> None:
        for i, a in obs.evidence.self_report.items():
            if i not in self._seen_ask:
                self._seen_ask.add(i)
                self.belief.update("ask", i, a)
        for i, a in obs.evidence.probe.items():
            if i not in self._seen_verify:
                self._seen_verify.add(i)
                self.belief.update("verify", i, a)

    def action_mask(self, obs: Observation) -> np.ndarray:
        n = self.u.n
        remaining = obs.budget - obs.cost_used
        mask = np.zeros(2 * n + 1, dtype=bool)
        for i in range(n - 1):
            if i not in obs.evidence.self_report and remaining >= 1.0:
                mask[i] = True
            if self.allow_verify and i not in obs.evidence.probe and remaining >= self.verify_cost:
                mask[n + i] = True
        mask[2 * n] = True
        return mask

    def tensors(self, obs: Observation):
        x = node_features(obs, self.belief, self.use_belief)
        g = np.array([(obs.budget - obs.cost_used) / 40.0, obs.cost_used / 40.0, self.u.n / 60.0], dtype=np.float32)
        return x, g

    def act(self, obs: Observation) -> Action:
        self._absorb(obs)
        x, g = self.tensors(obs)
        mask = self.action_mask(obs)
        with torch.no_grad():
            logits, value = self.net(torch.from_numpy(x), torch.from_numpy(self.src), torch.from_numpy(self.dst),
                                     torch.from_numpy(g).unsqueeze(0))
        logits = logits.masked_fill(~torch.from_numpy(mask), -1e9)
        if self.deterministic:
            a = int(torch.argmax(logits))
            logp = 0.0
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a_t = dist.sample()
            a, logp = int(a_t), float(dist.log_prob(a_t))
        self.last = Transition(x, self.src, self.dst, g, mask, a, logp, float(value), 0.0, False)
        n = self.u.n
        if a == 2 * n:
            return ("stop", None)
        return ("ask", a) if a < n else ("verify", a - n)

    def estimate(self, obs: Observation) -> np.ndarray:
        self._absorb(obs)
        est = self.belief.marginals() < 0.5
        est[self.u.target_idx] = True
        return est
