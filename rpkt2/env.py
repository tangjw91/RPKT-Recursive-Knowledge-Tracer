"""Boundary-search environment: a policy spends a probe budget to locate the learner's knowledge boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .graph import Universe
from .learner import SimulatedLearner

Action = Tuple[str, Optional[int]]  # ("ask", i) | ("verify", i) | ("stop", None)


@dataclass
class Evidence:
    self_report: Dict[int, bool] = field(default_factory=dict)
    probe: Dict[int, bool] = field(default_factory=dict)


@dataclass
class Observation:
    u: Universe
    evidence: Evidence
    cost_used: float
    budget: float


class BoundaryEnv:
    def __init__(self, u: Universe, learner: SimulatedLearner, budget: float = 40.0,
                 ask_cost: float = 1.0, verify_cost: float = 1.0):
        self.u, self.learner, self.budget = u, learner, budget
        self.ask_cost, self.verify_cost = ask_cost, verify_cost
        self.reset()

    def reset(self) -> Observation:
        self.evidence = Evidence()
        self.cost_used = 0.0
        self.done = False
        return self.observe()

    def observe(self) -> Observation:
        return Observation(self.u, self.evidence, self.cost_used, self.budget)

    def can_afford(self, action: Action) -> bool:
        kind = action[0]
        c = self.ask_cost if kind == "ask" else self.verify_cost if kind == "verify" else 0.0
        return self.cost_used + c <= self.budget + 1e-9

    def step(self, action: Action) -> Tuple[Observation, bool]:
        kind, i = action
        if self.done:
            return self.observe(), True
        if kind == "stop" or not self.can_afford(action):
            self.done = True
            return self.observe(), True
        if kind == "ask":
            if i in self.evidence.self_report:
                raise ValueError(f"concept {i} already asked")
            self.evidence.self_report[i] = self.learner.self_report(i)
            self.cost_used += self.ask_cost
        elif kind == "verify":
            if i in self.evidence.probe:
                raise ValueError(f"concept {i} already verified")
            self.evidence.probe[i] = self.learner.probe(i)
            self.cost_used += self.verify_cost
        else:
            raise ValueError(kind)
        if self.cost_used >= self.budget - 1e-9:
            self.done = True
        return self.observe(), self.done
