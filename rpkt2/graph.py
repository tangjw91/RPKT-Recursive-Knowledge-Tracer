"""Prerequisite graphs: loading, ancestry, and sampling of downward-closed knowledge states."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

DOMAINS = ["data_mining", "geometry", "physics", "precalculus"]
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "alcpl"


def load_alcpl(domain: str, data_dir: Path = DATA_DIR) -> nx.DiGraph:
    """Edges point prerequisite -> concept. AL-CPL lines are `concept,prerequisite`."""
    g = nx.DiGraph()
    with open(data_dir / f"{domain}.preqs", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            concept, prereq = row[0].strip(), row[1].strip()
            if concept and prereq and concept != prereq:
                g.add_edge(prereq, concept)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(f"{domain}: prerequisite graph has cycles")
    return g


class Universe:
    """The sub-DAG relevant to one target: the target and all its (transitive) prerequisites.

    Nodes are indexed 0..n-1 in a topological order (prerequisites before dependents);
    the target is always the last index.
    """

    def __init__(self, g: nx.DiGraph, target: str):
        anc = nx.ancestors(g, target)
        sub = g.subgraph(anc | {target})
        order = [n for n in nx.topological_sort(sub) if n != target] + [target]
        self.g = g
        self.target = target
        self.nodes: List[str] = order
        self.index: Dict[str, int] = {n: i for i, n in enumerate(order)}
        self.n = len(order)
        self.target_idx = self.n - 1
        self.prereqs: List[np.ndarray] = [
            np.array([self.index[p] for p in sub.predecessors(n)], dtype=int) for n in order
        ]
        self.dependents: List[np.ndarray] = [
            np.array([self.index[d] for d in sub.successors(n)], dtype=int) for n in order
        ]
        self.depth = np.zeros(self.n, dtype=int)  # distance from target (target = 0)
        lengths = nx.single_source_shortest_path_length(sub.reverse(copy=False), target)
        for n, d in lengths.items():
            self.depth[self.index[n]] = d

    def ancestors_idx(self, i: int) -> np.ndarray:
        return np.array([self.index[a] for a in nx.ancestors(self.g, self.nodes[i]) if a in self.index], dtype=int)

    def descendants_idx(self, i: int) -> np.ndarray:
        return np.array([self.index[d] for d in nx.descendants(self.g, self.nodes[i]) if d in self.index], dtype=int)


def sample_order_ideal(u: Universe, mastery: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a downward-closed known set: a node can be known only if all its prerequisites are known.

    Processed in topological order with Bernoulli(mastery); the target is always unknown.
    """
    known = np.zeros(u.n, dtype=bool)
    for i in range(u.n - 1):
        pre = u.prereqs[i]
        if (pre.size == 0 or known[pre].all()) and rng.random() < mastery:
            known[i] = True
    return known


def is_order_ideal(u: Universe, known: np.ndarray) -> bool:
    return all((not known[i]) or known[u.prereqs[i]].all() for i in range(u.n))


def monotone_closure(u: Universe, known_evidence: Sequence[int], unknown_evidence: Sequence[int]) -> np.ndarray:
    """Rule-based estimate used by non-Bayesian policies.

    Known evidence propagates to prerequisites (downward); everything else counts as unknown,
    which mirrors RPKT v1's learning path that lists unassessed nodes as to-learn. Direct
    unknown evidence wins over implied knowledge.
    """
    est_known = np.zeros(u.n, dtype=bool)
    for i in known_evidence:
        est_known[i] = True
        est_known[u.ancestors_idx(i)] = True
    for i in unknown_evidence:
        est_known[i] = False
    est_known[u.target_idx] = False
    return ~est_known  # estimated unknown set


def load_metacademy(data_dir: Path = DATA_DIR.parent / "metacademy") -> nx.DiGraph:
    g = nx.DiGraph()
    with open(data_dir / "edges.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            g.add_edge(row["prerequisite"], row["concept"])
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("metacademy: prerequisite graph has cycles")
    return g


def load_graph(name: str, rng: np.random.Generator | None = None) -> nx.DiGraph:
    """`name` is an AL-CPL domain, "metacademy", or "synth:<layers>,<width>,<prereqs>"."""
    if name in DOMAINS:
        return load_alcpl(name)
    if name == "metacademy":
        return load_metacademy()
    if name.startswith("synth:"):
        L, W, P = (int(x) for x in name[6:].split(","))
        return synthetic_layered_dag(L, W, P, rng or np.random.default_rng(0))
    raise ValueError(name)


def candidate_targets(g: nx.DiGraph, min_ancestors: int = 8) -> List[str]:
    return sorted(n for n in g.nodes if len(nx.ancestors(g, n)) >= min_ancestors)


def synthetic_layered_dag(n_layers: int, width: int, n_prereqs: int, rng: np.random.Generator,
                          target_name: str = "TARGET") -> nx.DiGraph:
    """Synthetic prerequisite DAG for controlled scaling studies.

    Layer 0 holds foundational concepts; each node in layer k>0 draws n_prereqs prerequisites
    from layer k-1 (and occasionally from deeper layers, as real graphs do); a single target on
    top depends on n_prereqs nodes of the last layer. Universe size ~ n_layers * width + 1.
    """
    g = nx.DiGraph()
    layers = [[f"L{k}_{j}" for j in range(width)] for k in range(n_layers)]
    for k in range(1, n_layers):
        for node in layers[k]:
            pool = layers[k - 1] + (layers[k - 2] if k >= 2 else [])
            for p in rng.choice(pool, size=min(n_prereqs, len(pool)), replace=False):
                g.add_edge(p, node)
    for p in rng.choice(layers[-1], size=min(n_prereqs, width), replace=False):
        g.add_edge(p, target_name)
    assert nx.is_directed_acyclic_graph(g)
    return g
