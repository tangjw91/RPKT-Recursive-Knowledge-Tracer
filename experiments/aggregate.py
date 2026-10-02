"""Aggregate sweep CSVs into tables with 95% confidence intervals (mean +- 1.96 * s.e. over episodes).

Usage: python experiments/aggregate.py results/sweep_overclaim.csv [more.csv ...] --by overclaim --out results/sweep_overclaim_ci.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ci(x: pd.Series) -> str:
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return f"{m:.3f} ± {1.96 * se:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--by", nargs="+", default=["overclaim"])
    ap.add_argument("--metrics", nargs="+", default=["bal_acc", "balacc_at_10", "balacc_at_15", "balacc_at_20", "cost", "n_verify"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True)
    df["caught_rate"] = df["overclaims_caught"] / df["overclaims_total"].clip(lower=1)
    keys = args.by + ["policy"]
    table = df.groupby(keys)[args.metrics].agg(ci)
    caught = df[df.overclaims_total > 0].groupby(keys)["caught_rate"].agg(ci).rename("overclaims_caught")
    table = table.join(caught, how="left")
    table["n"] = df.groupby(keys).size()
    md = table.to_markdown()
    print(md)
    if args.out:
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
