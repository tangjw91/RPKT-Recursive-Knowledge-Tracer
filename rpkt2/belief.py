"""Particle posterior over downward-closed knowledge states (order ideals) of a prerequisite DAG.

Prior: topological Bernoulli(m) ideals, m drawn from a mixture. Likelihood: per-concept noisy
self-report and micro-probe outcomes. Degeneracy is handled by resampling plus Metropolis
rejuvenation with single-node flips that preserve downward-closure.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .env import Evidence
from .graph import Universe
from .learner import NoiseModel


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class ParticleBelief:
    def __init__(self, u: Universe, noise: NoiseModel, rng: np.random.Generator,
                 n_particles: int = 2000, prior_masteries=(0.3, 0.5, 0.7, 0.9), mcmc_mastery: float = 0.5):
        self.u, self.noise, self.rng = u, noise, rng
        self.M, self.n = n_particles, u.n
        self.m0 = mcmc_mastery
        self.X = np.zeros((self.M, self.n), dtype=bool)
        masteries = rng.choice(prior_masteries, size=self.M)
        for i in range(self.n - 1):
            pre = u.prereqs[i]
            ok = np.ones(self.M, dtype=bool) if pre.size == 0 else self.X[:, pre].all(axis=1)
            self.X[:, i] = ok & (rng.random(self.M) < masteries)
        self.logw = np.zeros(self.M)
        # per-node log-likelihood of accumulated evidence if known / if unknown
        self.ll_known = np.zeros(self.n)
        self.ll_unknown = np.zeros(self.n)
        self._prereq_minus: Dict[Tuple[int, int], np.ndarray] = {}

    # ---------- evidence ----------
    def _lik(self, kind: str, answer: bool) -> Tuple[float, float]:
        eps = 1e-4  # keep likelihoods strictly positive so noise-free settings stay well defined
        p, r, q = (float(np.clip(v, eps, 1 - eps)) for v in
                   (self.noise.p_overclaim, self.noise.r_underclaim, self.noise.q_probe))
        if kind == "ask":   # answer == says "know"
            return (1 - r, p) if answer else (r, 1 - p)
        return (q, 1 - q) if answer else (1 - q, q)  # verify: answer == passed

    def update(self, kind: str, i: int, answer: bool) -> None:
        lk, lu = self._lik(kind, answer)
        self.ll_known[i] += np.log(lk)
        self.ll_unknown[i] += np.log(lu)
        self.logw += np.where(self.X[:, i], np.log(lk), np.log(lu))
        if self.ess() < self.M / 2:
            self.rejuvenate()

    def sync(self, ev: Evidence) -> None:
        """Rebuild from scratch (used when a policy is handed a fresh observation)."""
        for i, a in ev.self_report.items():
            self.update("ask", i, a)
        for i, a in ev.probe.items():
            self.update("verify", i, a)

    # ---------- queries ----------
    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def marginals(self) -> np.ndarray:
        return self.weights() @ self.X

    def entropy(self) -> float:
        return float(_binary_entropy(self.marginals()).sum())

    def ess(self) -> float:
        w = self.weights()
        return 1.0 / float((w ** 2).sum())

    def expected_entropy_after(self, kind: str, i: int) -> float:
        w = self.weights()
        total = 0.0
        for answer in (True, False):
            lk, lu = self._lik(kind, answer)
            lik = np.where(self.X[:, i], lk, lu)
            wa = w * lik
            z = wa.sum()
            if z <= 0:
                continue
            total += z * float(_binary_entropy((wa / z) @ self.X).sum())
        return total

    # ---------- rejuvenation ----------
    def _pre_minus(self, j: int, i: int) -> np.ndarray:
        key = (j, i)
        if key not in self._prereq_minus:
            self._prereq_minus[key] = self.u.prereqs[j][self.u.prereqs[j] != i]
        return self._prereq_minus[key]

    def rejuvenate(self, n_sweeps: int = 2) -> None:
        w = self.weights()
        idx = self.rng.choice(self.M, size=self.M, p=w)
        self.X = self.X[idx].copy()
        self.logw = np.zeros(self.M)
        log_m, log_1m = np.log(self.m0), np.log(1 - self.m0)
        for _ in range(n_sweeps):
            for i in self.rng.permutation(self.n - 1):
                cur = self.X[:, i]
                pre, deps = self.u.prereqs[i], self.u.dependents[i]
                pre_ok = np.ones(self.M, dtype=bool) if pre.size == 0 else self.X[:, pre].all(axis=1)
                dep_free = np.ones(self.M, dtype=bool) if deps.size == 0 else ~self.X[:, deps].any(axis=1)
                to_known = ~cur & pre_ok
                to_unknown = cur & dep_free
                feasible = to_known | to_unknown
                if not feasible.any():
                    continue
                # log prior change for flipping i to known: own factor + dependents that become "eligible"
                d_prior = np.full(self.M, log_m - log_1m)
                for j in deps:
                    pm = self._pre_minus(j, i)
                    others_ok = np.ones(self.M, dtype=bool) if pm.size == 0 else self.X[:, pm].all(axis=1)
                    d_prior += np.where(others_ok, log_1m, 0.0)
                d_lik = self.ll_known[i] - self.ll_unknown[i]
                delta = np.where(to_known, d_prior + d_lik, -(d_prior + d_lik))
                accept = feasible & (np.log(self.rng.random(self.M)) < delta)
                self.X[accept, i] = ~cur[accept]
