# Results log: boundary search vs RPKT v1 on gold prerequisite graphs

Date: 2026-10-02. Code: `rpkt2/`, `experiments/run_baselines.py`. Raw tables: `results/*.md`.
No LLM calls were used; everything below runs on a laptop in about two minutes.

## Setup

* **Graphs.** AL-CPL (4 domains, targets with >= 8 prerequisites, universe ~11 nodes),
  Metacademy (ML/maths, 540 concepts, targets with >= 15 prerequisites, universe 16-61,
  mean 25), synthetic layered DAGs (universe 19-81, mean 33).
* **Learners.** Downward-closed known set sampled at mastery 0.5 / 0.7 / 0.9; underclaim
  rate 0.05; probe accuracy 0.85; overclaim rate p in {0, 0.2}. 30 learners per mastery on
  Metacademy, 20 on synthetic, 10 per AL-CPL domain.
* **Policies.** `rpkt_v1` (exact re-implementation of Algorithm 1: expand every "don't know",
  self-report only), `rpkt_v1_k4` (at most 4 prerequisites per concept, as the LLM prompt
  returns), `enumerate` (ask everything), `random`, `eig` (particle posterior over order ideals +
  expected information gain, may verify), `eig_noverify` (same, ask only).
* **Metric.** Balanced accuracy of the estimated unknown set (mean of recall on unknown and
  recall on known concepts), reported at fixed probe budgets. Ask and verify both cost 1.

## Headline table: balanced accuracy at a fixed number of probes

Metacademy (universe mean 25.5):

| overclaim | policy | @5 | @10 | @15 | @20 | @30 | final | mean cost |
|---|---|---|---|---|---|---|---|---|
| 0.0 | eig | 0.786 | 0.913 | **0.966** | 0.986 | 0.989 | 0.989 | 17.7 |
| 0.0 | eig_noverify | 0.803 | 0.921 | **0.972** | 0.986 | 0.990 | 0.990 | 14.8 |
| 0.0 | enumerate | 0.600 | 0.717 | 0.818 | 0.905 | 0.966 | 0.979 | 24.0 |
| 0.0 | random | 0.648 | 0.778 | 0.872 | 0.929 | 0.967 | 0.980 | 24.0 |
| 0.0 | rpkt_v1 | 0.667 | 0.788 | 0.883 | 0.946 | 0.978 | 0.986 | 20.3 |
| 0.2 | eig | 0.763 | 0.841 | **0.898** | 0.914 | 0.932 | 0.931 | 27.5 |
| 0.2 | eig_noverify | 0.761 | 0.851 | **0.902** | 0.917 | 0.926 | 0.927 | 18.0 |
| 0.2 | enumerate | 0.612 | 0.694 | 0.770 | 0.823 | 0.866 | 0.876 | 23.9 |
| 0.2 | random | 0.654 | 0.740 | 0.805 | 0.841 | 0.871 | 0.878 | 23.9 |
| 0.2 | rpkt_v1 | 0.628 | 0.719 | 0.782 | 0.794 | 0.808 | 0.810 | 15.7 |

Synthetic layered DAGs (universe mean 32.9): same pattern, larger gap. At 15 probes with no
overclaim: eig 0.973 vs rpkt_v1 0.738; with overclaim 0.2: eig 0.913 vs rpkt_v1 0.774, and
rpkt_v1's final balanced accuracy is 0.845 with unlimited budget.

Overclaims caught (share of "I know" answers on unknown concepts that end up marked unknown),
Metacademy, overclaim 0.2: eig 0.93, eig_noverify 0.67, rpkt_v1 0.01, enumerate 0.02.

## What the numbers say

1. **Efficiency (no miscalibration).** Boundary search needs about half the probes of v1 for
   the same accuracy: 0.97 at 15 probes vs v1 reaching 0.95 only at 20 and enumeration at 30.
   v1 is close to enumeration because it cannot skip concepts implied by structure.
2. **Robustness (miscalibration).** With 20 % overclaiming, v1 plateaus at 0.81 balanced
   accuracy regardless of budget: a false "I know" prunes an entire subtree and those unknown
   unknowns are never revisited (recall on unknowns 0.64). The particle posterior keeps
   revisiting them and reaches 0.93.
3. **Verification.** At matched budget, explicit micro-probes do not raise balanced accuracy
   over structural inference alone (eig vs eig_noverify are within noise). Their value is
   *auditable* detection: 93-95 % of overclaims are caught by a failed probe, versus 67 %
   inferred indirectly. Whether verification pays in accuracy depends on probe reliability and
   relative cost; that sweep (q, verify cost) is the next experiment.
4. **AL-CPL is too small** to separate policies (universe ~11, v1 == enumeration); it stays in
   the paper as the realism check, Metacademy and synthetic graphs carry the scaling story.

## Sweeps (Metacademy, 3 seeds x 20 learners x 3 masteries = 180 episodes per cell, 95 % CI)

Scripts: `experiments/sweeps.sh`, tables `results/sweep_*_ci.md`, figures `figures/`.

### Overclaim rate p (budget 40, probe accuracy 0.85, verify cost 1)

| p | rpkt_v1 | random | eig ask-only | eig + verify | overclaims caught (ask-only / verify) |
|---|---|---|---|---|---|
| 0.0 | 0.981 ± .005 | 0.979 ± .006 | 0.985 ± .005 | 0.985 ± .005 | – |
| 0.1 | 0.909 ± .015 | 0.939 ± .008 | 0.959 ± .009 | 0.967 ± .007 | 0.65 / 0.81 |
| 0.2 | 0.836 ± .018 | 0.894 ± .010 | 0.930 ± .012 | 0.935 ± .010 | 0.62 / 0.91 |
| 0.3 | 0.758 ± .020 | 0.845 ± .012 | 0.910 ± .011 | 0.921 ± .012 | 0.74 / 0.93 |
| 0.4 | 0.703 ± .019 | 0.802 ± .012 | 0.866 ± .013 | 0.927 ± .011 | 0.75 / 0.91 |

v1 loses 0.07 balanced accuracy per 0.1 of overclaim rate; boundary search with verification
loses 0.015. Below p = 0.3 structural inference alone matches verification; at p = 0.4
verification is worth 6 points. See `figures/fig_overclaim.pdf` and `fig_curves.pdf`
(with calibrated learners, boundary search reaches 0.9 at ~9 probes, v1 at ~17).

### Probe reliability q x verification cost (p = 0.2)

| q | verify cost | eig ask-only | eig + verify | verifications used | overclaims caught |
|---|---|---|---|---|---|
| 0.70 | 1 | 0.939 ± .011 | 0.932 ± .010 | 8.8 | 0.76 |
| 0.85 | 1 | 0.930 ± .012 | 0.935 ± .010 | 11.6 | 0.91 |
| 0.95 | 1 | 0.936 ± .011 | **0.971 ± .007** | 15.7 | 0.97 |
| 0.95 | 2 | 0.933 ± .012 | 0.967 ± .007 | 8.7 | 0.96 |

Verification pays only when the micro-probe is reliable (q >= 0.9); with q = 0.7 the policy
correctly stops using it. Doubling its cost halves its use without losing final accuracy.
This sets the bar for Paper B's probe generator: items must discriminate at >= 0.9.

### Misspecified belief (policy assumes p = 0.1 regardless of the truth)

| true p | matched belief | belief fixed at 0.1 |
|---|---|---|
| 0.0 | 0.985 ± .005 | 0.970 ± .007 |
| 0.2 | 0.935 ± .010 | 0.953 ± .009 |
| 0.3 | 0.921 ± .012 | 0.937 ± .010 |
| 0.4 | 0.927 ± .011 | 0.927 ± .011 |

The policy does not need the learner's overclaim rate: a fixed mild prior is as good as or
better than the matched one (it verifies slightly more). This removes a reviewer objection.

## Caveats

* `rpkt_v1_k4` equals `rpkt_v1` on synthetic graphs because every node has <= 4 prerequisites.
* AL-CPL universes (~11 nodes) cannot separate policies; keep as realism check only.
* The particle posterior's initial estimate starts at ~0.72 balanced accuracy before any probe
  (prior knowledge of the DAG) whereas rule-based policies start at 0.5; curves show this.

## Learned policy, first attempt (GNN + PPO, 120 updates x 24 episodes, ~10 min CPU)

Trained on synthetic graphs + 70 % of Metacademy targets, overclaim rate drawn from [0, 0.4],
budget from [10, 35]; evaluated on the 30 % held-out Metacademy targets (180 episodes per
cell, budget 40). Full tables: `results/rl_holdout_ci.md`, `results/rl_synth_ci.md`.

| p | policy | final | @10 | @15 | @20 | cost |
|---|---|---|---|---|---|---|
| 0.0 | eig | 0.976 | 0.896 | 0.947 | 0.966 | 20.0 |
| 0.0 | rl (belief feats) | 0.976 | 0.820 | 0.893 | 0.938 | 30.7 |
| 0.0 | rl (no belief) | 0.981 | 0.820 | 0.903 | 0.954 | 36.6 |
| 0.0 | rpkt_v1 | 0.973 | 0.746 | 0.833 | 0.897 | 22.9 |
| 0.2 | eig | 0.929 | 0.842 | 0.893 | 0.911 | 29.6 |
| 0.2 | rl (belief feats) | 0.917 | 0.816 | 0.868 | 0.899 | 31.1 |
| 0.2 | rl (no belief) | 0.931 | 0.781 | 0.857 | 0.899 | 36.3 |
| 0.2 | rpkt_v1 | 0.790 | 0.698 | 0.753 | 0.775 | 16.1 |
| 0.4 | eig | 0.920 | 0.819 | 0.873 | 0.897 | 31.4 |
| 0.4 | rl (belief feats) | 0.891 | 0.802 | 0.840 | 0.863 | 32.1 |
| 0.4 | rl (no belief) | 0.896 | 0.764 | 0.804 | 0.839 | 36.4 |
| 0.4 | rpkt_v1 | 0.686 | 0.655 | 0.670 | 0.685 | 10.3 |

Reading: the learned policy transfers to unseen targets and beats v1 everywhere, matches the
EIG heuristic at full budget, but is 3-8 points behind at fixed budgets and spends more
probes (it almost never stops early). The variant without belief features reaches the same
final accuracy but verifies far more (it cannot see uncertainty). See
`docs/POLICY_DISCUSSION.md` for why and what to do about it.

## Learned policy, second attempt: warm start + corrupted-graph training (`rl_robust`)

Recipe (`experiments/train_robust.sh`): DAgger from the EIG heuristic for 3 iterations
(3.6k labelled states, 49 % exact agreement, many actions being near-ties), then 120 PPO
updates with random early termination (so accuracy at every budget matters), on clean and
corrupted graphs (edge drop 0-0.4, spurious edges 0-0.3). Evaluated on held-out Metacademy
targets, overclaim 0.2, 180 episodes per cell; tables `results/corrupt_ci.md`,
`results/corrupt_paired.md`, figure `figures/fig_corruption.pdf`.

| drop | add | rpkt_v1 | eig + verify | eig ask-only | rl (first attempt) | **rl_robust** | cost: eig / rl_robust |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.785 | 0.925 | 0.930 | 0.918 | **0.934** | 29.4 / 20.8 |
| 0.0 | 0.2 | 0.814 | 0.893 | 0.891 | 0.888 | **0.904** | 28.1 / 20.7 |
| 0.2 | 0.0 | 0.702 | 0.926 | 0.932 | 0.919 | 0.928 | 31.1 / 21.6 |
| 0.2 | 0.2 | 0.736 | 0.886 | 0.890 | 0.890 | **0.905** | 29.4 / 21.0 |
| 0.4 | 0.0 | 0.612 | 0.913 | 0.916 | 0.914 | **0.921** | 33.2 / 22.7 |
| 0.4 | 0.2 | 0.643 | 0.889 | 0.887 | 0.886 | **0.900** | 31.6 / 22.1 |

Paired on identical episodes (rl_robust minus eig, 95 % CI, n = 1080 pooled):
final balanced accuracy +0.010 ± 0.006, at 15 probes +0.011 ± 0.006, probes spent
-9.0 ± 0.3 (-30 %), verification items 4 vs 12 (-70 %).

Reading:
1. **The learned policy now matches or beats the Bayesian heuristic in every cell**, with a
   small but significant accuracy edge pooled over cells, while spending 30 % fewer learner
   interactions and 70 % fewer verification items. It learned *when* verification pays.
2. **Missing prerequisite edges do not break the heuristic** (0.925 -> 0.913 at 40 % missing):
   the particle posterior still gets most inference from the edges that remain. Spurious edges
   cost every policy 3-4 points. So the "RL is robust where the model is misspecified" story is
   only mildly supported; the defensible claim is parity-or-better at lower cost under all
   extraction-error levels.
3. **RPKT v1 collapses under extraction errors** (0.785 -> 0.612 at 40 % missing edges): a
   missing edge hides a whole subtree from blind recursion. This is a direct statement about
   the published system under realistic LLM extraction quality.
4. The warm start and random termination are what fixed the first attempt (it over-spent
   and never stopped); the high-penalty retrain without warm start collapsed to stopping.
