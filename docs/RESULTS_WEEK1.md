# Week 1 results: boundary search vs RPKT v1 on gold prerequisite graphs

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

## Caveats to fix before the paper

* The eig policy is given the true noise parameters (p, r, q). Add a misspecified-noise run.
* Only p in {0, 0.2}; sweep p in {0, 0.1, 0.2, 0.3, 0.4} for the robustness figure.
* Single seed per configuration; add 3 seeds and confidence intervals.
* `rpkt_v1_k4` equals `rpkt_v1` on synthetic graphs because every node has <= 4 prerequisites.

## Next

1. Noise and cost sweeps (p, q, verify cost) and misspecified-noise robustness.
2. RL policy (GNN + PPO) in the same environment; compare with eig at fixed budgets;
   leave-one-graph-family-out transfer.
3. Figures: balanced accuracy vs probes, probes-to-0.9 vs overclaim rate.
