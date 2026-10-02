from __future__ import annotations

import numpy as np

from ..env import Action, Observation
from ..graph import Universe, monotone_closure
from ..learner import NoiseModel


class Policy:
    name = "base"

    def __init__(self, noise: NoiseModel | None = None, rng: np.random.Generator | None = None, **_):
        self.noise = noise or NoiseModel()
        self.rng = rng or np.random.default_rng(0)

    def reset(self, u: Universe) -> None:
        self.u = u

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def estimate(self, obs: Observation) -> np.ndarray:
        """Estimated unknown set (bool over universe). Default: monotone closure of the evidence."""
        ev = obs.evidence
        known = [i for i, a in ev.self_report.items() if a]
        unknown = [i for i, a in ev.self_report.items() if not a]
        # verification outcomes override self-report
        known = [i for i in known if ev.probe.get(i, True)] + [i for i, p in ev.probe.items() if p]
        unknown = unknown + [i for i, p in ev.probe.items() if not p]
        return monotone_closure(self.u, known, unknown)
