"""Table 1 of Paper A: boundary recovery vs interaction cost for every policy, domain, and noise level.

Usage: python experiments/run_baselines.py --n-learners 20 --domains data_mining geometry
Outputs results/<tag>.csv and results/<tag>.md.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpkt2.env import BoundaryEnv  # noqa: E402
from rpkt2.graph import DOMAINS, Universe, candidate_targets, load_graph  # noqa: E402
from rpkt2.learner import NoiseModel, SimulatedLearner  # noqa: E402
from rpkt2.policies import REGISTRY  # noqa: E402
from rpkt2.runner import run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=DOMAINS,
                    help="AL-CPL domain names, 'metacademy', or 'synth:L,W,P'")
    ap.add_argument("--policies", nargs="+", default=[k for k in REGISTRY if not k.startswith("rl")])
    ap.add_argument("--targets-file", default=None, help="restrict targets to the names listed in this file")
    ap.add_argument("--n-learners", type=int, default=20)
    ap.add_argument("--masteries", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--overclaim", nargs="+", type=float, default=[0.0, 0.2])
    ap.add_argument("--underclaim", type=float, default=0.05)
    ap.add_argument("--probe-acc", type=float, default=0.85)
    ap.add_argument("--verify-cost", type=float, default=1.0)
    ap.add_argument("--belief-overclaim", type=float, default=None,
                    help="overclaim rate assumed by the belief-based policies (default: the true rate)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="overrides --seed with several seeds")
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--n-particles", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ancestors", type=int, default=15)
    ap.add_argument("--tag", default="baselines")
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    seeds = args.seeds or [args.seed]
    for domain in args.domains:
        g = load_graph(domain, np.random.default_rng(args.seed))
        targets = candidate_targets(g, min_ancestors=args.min_ancestors)
        if args.targets_file:
            keep = set(Path(args.targets_file).read_text().split())
            targets = [t for t in targets if t in keep]
        for p in args.overclaim:
            noise = NoiseModel(p_overclaim=p, r_underclaim=args.underclaim, q_probe=args.probe_acc)
            belief_noise = noise if args.belief_overclaim is None else NoiseModel(
                p_overclaim=args.belief_overclaim, r_underclaim=args.underclaim, q_probe=args.probe_acc)
            for seed, k, mastery in itertools.product(seeds, range(args.n_learners), args.masteries):
                rng = np.random.default_rng([seed, k, int(mastery * 100), int(p * 100), sum(map(ord, domain)) % 997])
                target = targets[rng.integers(len(targets))]
                u = Universe(g, target)
                learner = SimulatedLearner.sample(u, mastery, noise, rng)
                for name in args.policies:
                    extra = ({"n_particles": args.n_particles, "verify_cost": args.verify_cost}
                             if name.startswith(("eig", "rl")) else {})
                    policy = REGISTRY[name](noise=belief_noise, rng=np.random.default_rng(rng.integers(1 << 31)), **extra)
                    env = BoundaryEnv(u, learner, budget=args.budget, verify_cost=args.verify_cost)
                    res = run_episode(policy, env, learner)
                    res.update({"domain": domain, "policy": name, "mastery": mastery, "overclaim": p,
                                "learner": k, "target": target, "seed": seed, "probe_acc": args.probe_acc,
                                "verify_cost": args.verify_cost,
                                "belief_overclaim": belief_noise.p_overclaim})
                    rows.append(res)
        print(f"{domain} done, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"{args.tag}.csv", index=False)

    def summarise(d: pd.DataFrame) -> pd.Series:
        reached = np.isfinite(d["cost_to_target"])
        return pd.Series({
            "bal_acc": d["bal_acc"].mean(), "F1": d["f1"].mean(), "precision": d["precision"].mean(),
            "recall": d["recall"].mean(), "cost": d["cost"].mean(), "verify": d["n_verify"].mean(),
            "cost_to_balacc>=0.9": d.loc[reached, "cost_to_target"].median() if reached.any() else np.nan,
            "reached_0.9": reached.mean(), "universe": d["universe_size"].mean(),
            **{f"ba@{b}": d[f"balacc_at_{b}"].mean() for b in (5, 10, 15, 20, 30)},
            "overclaims_caught": d["overclaims_caught"].sum() / max(d["overclaims_total"].sum(), 1),
            "n": len(d),
        })

    table = df.groupby(["overclaim", "policy"]).apply(summarise).round(3)
    md = ["# " + args.tag, "", f"budget={args.budget}, masteries={args.masteries}, underclaim={args.underclaim}, "
          f"probe_acc={args.probe_acc}, domains={args.domains}, learners/domain/mastery={args.n_learners}", "",
          table.to_markdown(), "", "## by domain", "",
          df.groupby(["domain", "overclaim", "policy"]).apply(summarise).round(3).to_markdown()]
    (out / f"{args.tag}.md").write_text("\n".join(md))
    print(table.to_string())


if __name__ == "__main__":
    main()
