"""Paper A figures from sweep CSVs (matplotlib, vector PDF + PNG).

Fig 1: balanced accuracy vs probes spent (one panel per overclaim rate).
Fig 2: final balanced accuracy and overclaims caught vs overclaim rate, 95% CI bands.
Usage: python experiments/figures.py results/sweep_overclaim.csv --out figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Fixed categorical order (validated palette, light surface); never cycled.
COLORS = {"eig": "#2a78d6", "eig_noverify": "#eb6834", "rpkt_v1": "#1baf7a", "random": "#eda100",
          "enumerate": "#e87ba4", "rl": "#4a3aa7", "rl_nobelief": "#e34948", "rl_robust": "#008300", "bc": "#898781"}
LABELS = {"eig": "Boundary search (EIG + verify)", "eig_noverify": "Boundary search (EIG, ask only)",
          "rpkt_v1": "RPKT v1 (recursive expansion)", "random": "Random", "enumerate": "Enumerate",
          "rl": "Learned policy (GNN+PPO)", "rl_nobelief": "Learned policy (no belief features)",
          "rl_robust": "Learned policy (warm start, corrupted-graph training)", "bc": "Imitation of EIG"}
ORDER = ["rpkt_v1", "random", "enumerate", "eig_noverify", "eig", "rl", "rl_nobelief", "bc", "rl_robust"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def mean_curve(trajs: pd.Series, max_cost: int) -> np.ndarray:
    grid = np.arange(0, max_cost + 1)
    out = np.zeros((len(trajs), len(grid)))
    for k, s in enumerate(trajs):
        pts = json.loads(s)
        costs = np.array([c for c, _ in pts])
        scores = np.array([v for _, v in pts])
        idx = np.searchsorted(costs, grid, side="right") - 1
        out[k] = scores[np.clip(idx, 0, len(pts) - 1)]
    return out.mean(axis=0)


def fig_curves(df: pd.DataFrame, out: Path, max_cost: int = 30, rates=(0.0, 0.2, 0.4)) -> None:
    rates = [r for r in rates if r in set(df.overclaim.round(2))]
    fig, axes = plt.subplots(1, len(rates), figsize=(2.6 * len(rates), 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, rates):
        d = df[df.overclaim.round(2) == p]
        for pol in ORDER:
            if pol not in set(d.policy):
                continue
            y = mean_curve(d[d.policy == pol]["traj"], max_cost)
            ax.plot(np.arange(max_cost + 1), y, color=COLORS[pol], lw=1.6, label=LABELS[pol])
        ax.set_title(f"overclaim rate p = {p:.1f}", fontsize=8, color=INK)
        ax.set_xlabel("probes spent")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("balanced accuracy of unknown set")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_curves.pdf")
    fig.savefig(out / "fig_curves.png", dpi=200)


def fig_overclaim(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["caught"] = df.overclaims_caught / df.overclaims_total.clip(lower=1)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
    for ax, metric, ylabel in zip(axes, ["bal_acc", "caught"], ["final balanced accuracy", "overclaims caught (share)"]):
        for pol in ORDER:
            d = df[df.policy == pol]
            if d.empty:
                continue
            if metric == "caught":
                d = d[d.overclaims_total > 0]
            g = d.groupby("overclaim")[metric]
            m, se = g.mean(), g.std(ddof=1) / np.sqrt(g.size())
            ax.plot(m.index, m.values, color=COLORS[pol], lw=1.6, marker="o", ms=3.5, label=LABELS[pol])
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=COLORS[pol], alpha=0.15, lw=0)
        ax.set_xlabel("overclaim rate p")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[0].set_ylim(0.6, 1.0)
    axes[1].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_overclaim.pdf")
    fig.savefig(out / "fig_overclaim.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--max-cost", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    fig_curves(df, out, args.max_cost)
    fig_overclaim(df, out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
