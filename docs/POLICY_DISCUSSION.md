# Can a better policy or RL algorithm beat the information-gain heuristic?

Status 2026-10-02: PPO + GNN matches the particle-EIG heuristic at full budget on held-out
targets, is behind at fixed budgets, and over-spends probes. This note explains why and
ranks the options.

## Why the first RL attempt is behind at fixed budgets

1. **Objective mismatch.** Reward = sum of balanced-accuracy gains minus 0.004 per probe,
   discount 0.99. That maximises accuracy *at the sampled budget*, with almost no pressure to
   gain early, so the agent is not optimising the accuracy-vs-probes curve we report. The
   retrain with discount 0.9 and penalty 0.01 **collapsed to stopping immediately** within
   20 updates (cost 0, accuracy 0.78 = the prior's estimate): the stop action became dominant
   before the policy learned which probes pay. The objective needs random early termination
   (so every budget matters) rather than a larger per-probe penalty, plus a warm start (option 2).
2. **Data.** 2,880 episodes of on-policy data versus a heuristic that uses an explicit
   Bayesian model and needs no training. PPO is sample-hungry; the policy is still improving.
3. **Stopping.** With a 0.004 penalty any expected gain above 0.4 points justifies a probe,
   so "stop" is almost never chosen; cost 31-36 of 40.
4. **The heuristic is strong.** Greedy expected information gain over a correct Bayesian
   model of a monotone target is close to optimal for this problem (it is the noisy version
   of generalised binary search). A learned policy can only beat it where the model is wrong
   or where one-step greed is myopic.

## Options, ranked by expected payoff for the paper

| # | Option | What it would show | Effort | Chance to beat EIG |
|---|---|---|---|---|
| 1 | **Train and test on corrupted graphs** (missing / spurious prerequisite edges, as LLM extraction produces; Paper B measures the real rates) | RL learns when to distrust structure; EIG trusts the DAG and degrades. Ties Paper A to Paper B. | 1-2 days (edge-noise option in env + retrain) | High: this is where the Bayesian model is misspecified |
| 2 | **Imitation then RL** (DAgger from EIG, then PPO fine-tune) | Amortised Bayesian experimental design: same decisions as EIG at a fraction of the compute, then improvements on top | 1 day | High for "matches", medium for "beats" |
| 3 | **Fix the objective** (random early termination so every budget matters; keep a small penalty; warm start) | Fair fixed-budget comparison | tried penalty 0.01 / discount 0.9: collapsed to immediate stop | Medium |
| 4 | **Non-myopic planning**: 2-step lookahead or MCTS over the particle belief | Improves stop / verify timing; no training | 1-2 days, slow inference | Low-medium; gains on long chains only |
| 5 | **Algorithm swap** (DQN with action masking, or REINFORCE with the EIG score as baseline) | Sample efficiency | 1 day | Low: objective and data matter more than the algorithm |
| 6 | **More training** (5-10x episodes, GPU) | Closes the fixed-budget gap | hours | Medium, but only matches |

## Recommendation

Do 2 first (warm start from EIG prevents the stop collapse), then 1 as the main RL experiment of Paper A:
*"a learned probe policy is robust to prerequisite-graph errors that break a model-based
heuristic"*. Use 2 as the initialisation so the learned policy starts at EIG quality.
This keeps the honest fallback (EIG is near-optimal on correct graphs) and gives RL a claim
that the heuristic cannot make, which is exactly the setting the LLM-built graphs of Paper
B create.

Graph corruption to simulate, matched later to Paper B's measured extraction errors:
* drop each true edge with probability d (missed prerequisite, the common LLM error);
* add a spurious edge from a random non-ancestor with probability a (hallucinated
  prerequisite), keeping the graph acyclic;
* the learner's true state lives on the gold graph; the policy only sees the corrupted one.
