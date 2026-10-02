"""Simulated learner with a downward-closed true knowledge state and miscalibrated self-report."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import Universe, sample_order_ideal


@dataclass
class NoiseModel:
    p_overclaim: float = 0.2   # P(says "know" | does not know)   (Dunning-Kruger knob)
    r_underclaim: float = 0.05  # P(says "don't know" | knows)
    q_probe: float = 0.85      # P(passes micro-probe | knows) = P(fails | does not know)


class SimulatedLearner:
    """Answers are pre-sampled per concept so that repeated questions are consistent."""

    def __init__(self, u: Universe, known: np.ndarray, noise: NoiseModel, rng: np.random.Generator):
        self.u = u
        self.known = known.copy()
        self.noise = noise
        n = u.n
        say_know = np.where(known, rng.random(n) >= noise.r_underclaim, rng.random(n) < noise.p_overclaim)
        say_know[u.target_idx] = False
        self._self_report = say_know
        self._probe_pass = np.where(known, rng.random(n) < noise.q_probe, rng.random(n) >= noise.q_probe)
        self._probe_pass[u.target_idx] = False

    @classmethod
    def sample(cls, u: Universe, mastery: float, noise: NoiseModel, rng: np.random.Generator) -> "SimulatedLearner":
        return cls(u, sample_order_ideal(u, mastery, rng), noise, rng)

    def self_report(self, i: int) -> bool:
        return bool(self._self_report[i])

    def probe(self, i: int) -> bool:
        return bool(self._probe_pass[i])

    @property
    def true_unknown(self) -> np.ndarray:
        return ~self.known

    @property
    def overclaimed(self) -> np.ndarray:
        """Concepts the learner would claim to know but does not: the unknown unknowns."""
        return self._self_report & ~self.known
