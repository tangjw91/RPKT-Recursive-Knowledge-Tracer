import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpkt2.env import BoundaryEnv
from rpkt2.graph import Universe, candidate_targets, is_order_ideal, load_alcpl, sample_order_ideal
from rpkt2.learner import NoiseModel, SimulatedLearner
from rpkt2.policies import EIGPolicy, EnumeratePolicy, RPKTv1Policy
from rpkt2.runner import run_episode


def _setup(domain="data_mining", mastery=0.7, p=0.0, seed=1):
    g = load_alcpl(domain)
    rng = np.random.default_rng(seed)
    target = candidate_targets(g)[0]
    u = Universe(g, target)
    noise = NoiseModel(p_overclaim=p, r_underclaim=0.0, q_probe=1.0)
    return u, SimulatedLearner.sample(u, mastery, noise, rng), noise, rng


def test_sampled_state_is_order_ideal():
    u, learner, _, rng = _setup()
    for m in (0.3, 0.6, 0.9):
        for _ in range(20):
            assert is_order_ideal(u, sample_order_ideal(u, m, rng))
    assert is_order_ideal(u, learner.known)
    assert not learner.known[u.target_idx]


def test_noise_free_v1_and_enumerate_are_exact():
    u, learner, noise, rng = _setup()
    for pol in (RPKTv1Policy(noise=noise, rng=rng), EnumeratePolicy(noise=noise, rng=rng)):
        res = run_episode(pol, BoundaryEnv(u, learner, budget=1000), learner)
        assert res["f1"] == 1.0, (pol.name, res)


def test_v1_cost_never_exceeds_universe():
    u, learner, noise, rng = _setup(mastery=0.3)
    res = run_episode(RPKTv1Policy(noise=noise, rng=rng), BoundaryEnv(u, learner, budget=1000), learner)
    assert res["cost"] <= u.n - 1


def test_eig_runs_and_is_reasonable():
    u, learner, noise, rng = _setup(p=0.2)
    pol = EIGPolicy(noise=noise, rng=rng, n_particles=500)
    res = run_episode(pol, BoundaryEnv(u, learner, budget=30), learner)
    assert 0 <= res["f1"] <= 1 and res["cost"] <= 30
    assert res["n_ask"] + res["n_verify"] == res["cost"]


def test_corrupted_universe_keeps_nodes_and_is_acyclic():
    from rpkt2.graph import corrupt_universe
    u, learner, noise, rng = _setup()
    v = corrupt_universe(u, 0.3, 0.3, rng)
    assert v.n == u.n and v.nodes == u.nodes and v.target_idx == u.target_idx
    assert all((i < j) for i, j in v.edge_list())  # oriented along gold topological order => acyclic
    assert len(v.edge_list()) != len(u.edge_list())
    res = run_episode(EIGPolicy(noise=noise, rng=rng, n_particles=300), BoundaryEnv(v, learner, budget=20), learner)
    assert 0 <= res["bal_acc"] <= 1
