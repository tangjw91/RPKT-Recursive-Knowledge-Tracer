"""Metrics comparing an estimated unknown set with the learner's true unknown set."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def unknown_set_scores(est_unknown: np.ndarray, true_unknown: np.ndarray) -> Dict[str, float]:
    tp = float((est_unknown & true_unknown).sum())
    fp = float((est_unknown & ~true_unknown).sum())
    fn = float((~est_unknown & true_unknown).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tn = float((~est_unknown & ~true_unknown).sum())
    spec = tn / (tn + fp) if tn + fp else 1.0  # recall on the known set
    return {"precision": precision, "recall": recall, "f1": f1, "boundary_error": fp + fn,
            "bal_acc": 0.5 * (recall + spec)}


def cost_to_reach(trajectory: List[Tuple[float, float]], target: float = 0.9) -> float:
    """First cumulative cost at which the tracked score >= target; inf if never reached."""
    for cost, score in trajectory:
        if score >= target:
            return cost
    return float("inf")
