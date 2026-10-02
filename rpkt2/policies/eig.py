"""Expected-information-gain boundary search over a particle posterior (hand-designed heuristic)."""
from __future__ import annotations

import numpy as np

from ..belief import ParticleBelief
from ..env import Action, Observation
from ..graph import Universe
from .base import Policy


class EIGPolicy(Policy):
    name = "eig"

    def __init__(self, n_particles: int = 2000, allow_verify: bool = True, verify_cost: float = 1.0,
                 min_gain: float = 0.02, stop_entropy: float = 0.5, **kw):
        super().__init__(**kw)
        self.n_particles, self.allow_verify = n_particles, allow_verify
        self.verify_cost, self.min_gain, self.stop_entropy = verify_cost, min_gain, stop_entropy

    def reset(self, u: Universe) -> None:
        super().reset(u)
        self.belief = ParticleBelief(u, self.noise, self.rng, n_particles=self.n_particles)
        self._seen_ask, self._seen_verify = set(), set()

    def _absorb(self, obs: Observation) -> None:
        for i, a in obs.evidence.self_report.items():
            if i not in self._seen_ask:
                self._seen_ask.add(i)
                self.belief.update("ask", i, a)
        for i, a in obs.evidence.probe.items():
            if i not in self._seen_verify:
                self._seen_verify.add(i)
                self.belief.update("verify", i, a)

    def act(self, obs: Observation) -> Action:
        self._absorb(obs)
        h_now = self.belief.entropy()
        if h_now < self.stop_entropy:
            return ("stop", None)
        best, best_score = ("stop", None), self.min_gain
        remaining = obs.budget - obs.cost_used
        for i in range(self.u.n - 1):
            if i not in obs.evidence.self_report and remaining >= 1.0:
                gain = (h_now - self.belief.expected_entropy_after("ask", i)) / 1.0
                if gain > best_score:
                    best, best_score = ("ask", i), gain
            if self.allow_verify and i not in obs.evidence.probe and remaining >= self.verify_cost:
                gain = (h_now - self.belief.expected_entropy_after("verify", i)) / self.verify_cost
                if gain > best_score:
                    best, best_score = ("verify", i), gain
        return best

    def estimate(self, obs: Observation) -> np.ndarray:
        self._absorb(obs)
        est = self.belief.marginals() < 0.5
        est[self.u.target_idx] = True
        return est
