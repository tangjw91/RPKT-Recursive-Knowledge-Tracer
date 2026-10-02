"""Non-learning baselines: RPKT v1 recursion, full enumeration, random probing."""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from ..env import Action, Observation
from ..graph import Universe
from .base import Policy


class RPKTv1Policy(Policy):
    """Exact re-implementation of the published RPKT procedure (Algorithm 1 of the paper).

    Ask every prerequisite of the target; whenever a concept is reported unknown, enqueue its
    prerequisites (bounded by max_depth, at most max_children per concept as the LLM prompt
    returns 2-4). Self-report only, each concept asked once, no verification, no stopping rule.
    """
    name = "rpkt_v1"

    def __init__(self, max_depth: int = 6, max_children: Optional[int] = None, **kw):
        super().__init__(**kw)
        self.max_depth, self.max_children = max_depth, max_children

    def reset(self, u: Universe) -> None:
        super().reset(u)
        self.queue: Deque[tuple[int, int]] = deque()
        self.enqueued = set()
        self._enqueue_children(u.target_idx, 1)

    def _enqueue_children(self, i: int, depth: int) -> None:
        if depth > self.max_depth:
            return
        children = list(self.u.prereqs[i])
        if self.max_children is not None and len(children) > self.max_children:
            children = list(self.rng.choice(children, size=self.max_children, replace=False))
        for c in children:
            if c not in self.enqueued:
                self.enqueued.add(c)
                self.queue.append((c, depth))

    def act(self, obs: Observation) -> Action:
        # process answers to previously asked concepts: expand the unknown ones
        for i, a in list(obs.evidence.self_report.items()):
            if not a and not getattr(self, "_expanded", set()).__contains__(i):
                self._expanded = getattr(self, "_expanded", set()) | {i}
                self._enqueue_children(i, self._depth_of[i] + 1)
        while self.queue:
            i, d = self.queue.popleft()
            if i not in obs.evidence.self_report:
                self._depth_of = getattr(self, "_depth_of", {})
                self._depth_of[i] = d
                return ("ask", i)
        return ("stop", None)


class EnumeratePolicy(Policy):
    """Ask every concept in the universe (accuracy ceiling for self-report, maximal cost)."""
    name = "enumerate"

    def act(self, obs: Observation) -> Action:
        for i in range(self.u.n - 2, -1, -1):  # from the target's immediate prerequisites downward
            if i not in obs.evidence.self_report:
                return ("ask", i)
        return ("stop", None)


class RandomPolicy(Policy):
    name = "random"

    def act(self, obs: Observation) -> Action:
        unasked = [i for i in range(self.u.n - 1) if i not in obs.evidence.self_report]
        if not unasked:
            return ("stop", None)
        return ("ask", int(self.rng.choice(unasked)))
