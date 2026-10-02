# RPKT‑2 proposal: from a demo to a measurable system

Working title: **"Probe, Don't Ask: Active Knowledge‑Boundary Search with Verified Self‑Assessment for Discovering Unknown Unknowns"**
(alternatives: *ABKT – Active Boundary Knowledge Tracing*; *RPKT‑2*)

Target: 6‑page IEEE conference paper (mid/lower tier), same problem family as
RPKT (FMLDS 2025, arXiv:2508.11892).

---

## 1. Is the problem still worth working on?

Yes. The "unknown unknowns" framing is still open and, if anything, more
crowded in 2026 than in 2025, which is a signal that the community thinks it
matters. Work that appeared after RPKT and targets the same gap:

| Year | Work | What it does | What it does *not* do |
|---|---|---|---|
| 2025 | AutoPRE (LLM agents for prerequisite prediction) | Multi‑role LLM agents predict prerequisite pairs; +13% accuracy | No learner in the loop, no gap discovery |
| 2025 | LLM Protégés (BEA 2025) | LLM tutee with knowledge gaps | Gaps are the *LLM's*, not the learner's |
| 2025 | LOOM (learner memory graph from daily LLM chats) | Builds a long‑term learner graph; surfaces gaps the learner "senses but cannot name" | Needs weeks of chat history; passive, not top‑down from a target topic |
| 2026 | ALIGNAgent | Gap identification + next‑step guidance | Starts from learner work, not from a target question |
| 2026 | Capture‑Calibrate‑Coach | Graph‑based knowledge‑monitoring (calibration) estimation | Estimates calibration, does not search for the boundary |
| 2026 | "Confirming Correct, Missing the Rest" | Shows LLM tutoring agents fail exactly where feedback matters (missing gaps) | Diagnostic paper, no method |
| 2026 | K12‑KGraph / K12‑Bench | Curriculum‑aligned prerequisite KG + prerequisite‑reasoning benchmark | Resource, not a tutoring method |
| 2025–26 | StudentSim, SSKG, "Sycophantic simulators" | LLM student simulators and their faithfulness limits | Evaluation tooling we can reuse (with care) |

Positioning sentence for the new paper: *existing work either builds
prerequisite graphs offline (AutoPRE, K12‑KGraph), or models the learner from
accumulated interaction (LOOM, ALIGNAgent, knowledge tracing); none starts
from a single target question and actively searches for the learner's
knowledge boundary with a provably small number of probes while
correcting for the learner's own miscalibration.*

## 2. What was weak in RPKT v1 (and becomes the v2 contribution)

1. **Self‑report contradicts the motivation.** RPKT motivates itself with
   Dunning‑Kruger, then relies on the learner clicking "I know this". An
   over‑confident learner stops the recursion early and the unknown unknown
   stays unknown. → *v2: verify each "I know" with one cheap LLM‑generated
   micro‑probe (a 1‑item check). Treat self‑report as a prior, not as truth.*
2. **No real stopping rule.** Recursion stops at a depth cap or a hard‑coded
   list of ten "fundamental" strings. → *v2: stop when the boundary is
   resolved, using the prerequisite structure itself.*
3. **Probe explosion.** Every "don't know" spawns 2–4 new concepts; a tree of
   depth 4 can ask the learner 40+ questions. → *v2: exploit monotonicity.
   If you know X you (very likely) know X's prerequisites; if you don't know a
   prerequisite you (very likely) don't know X. The learner's knowledge
   state is a downward‑closed set in the prerequisite DAG, so finding the
   boundary is a search problem on a partial order and can be done with far
   fewer probes than full enumeration (expected‑information‑gain or
   binary‑search‑on‑chains policies).*
4. **Unverified prerequisite extraction.** Nobody checked that GPT‑4o's
   prerequisites are correct, non‑duplicated, or acyclic. Names drift
   ("Derivative" vs "Differentiation"). → *v2: canonicalise concept names by
   embedding similarity, enforce a DAG, and measure extraction quality
   against gold prerequisite datasets.*
5. **No evaluation.** The paper is a single case demonstration.
   → *v2: three tiers of evaluation (below), two of which need no humans.*

## 3. Proposed system (RPKT‑2)

```
target question Q
   │
   ▼
[1] LLM prerequisite expansion (on demand, cached)  ──►  grows a local DAG G
        + name canonicalisation (embeddings)           (duplicates merged,
        + cycle check                                   cycles rejected)
   │
   ▼
[2] Learner‑state model: each concept c has P(known_c); prior from
    education level + depth; constrained by monotonicity on G
   │
   ▼
[3] Probe policy: pick the concept whose answer most reduces uncertainty
    about the boundary (expected information gain / entropy over the cut),
    not "expand every unknown child"
   │
   ▼
[4] Verified assessment: learner answers know / don't‑know (binary, as v1);
    a "know" on a concept with high prior uncertainty triggers one
    LLM‑generated micro‑probe (1 short question). Disagreement = an
    *unknown unknown surfaced*. Update P(known) for c and propagate on G.
   │
   ▼
[5] Stop when boundary entropy < τ or probe budget exhausted
   │
   ▼
[6] Learning path = topological order of the unknown set; personalised
    explanation as in v1
```

Three claimed contributions (fits a 6‑page paper):

* **C1 – Boundary search instead of blind recursion.** Formulate gap
  discovery as finding a downward‑closed set in a prerequisite DAG; an
  active probe policy that needs a fraction of the probes of v1's full
  expansion for the same recovered boundary.
* **C2 – Verified self‑assessment.** Micro‑probes that catch over‑confident
  "I know" answers; quantify how many unknown unknowns this surfaces that
  pure self‑report misses.
* **C3 – Measured dynamic prerequisite discovery.** First quantitative
  check of LLM on‑the‑fly prerequisite extraction against gold prerequisite
  graphs, across several models (GPT‑4o as in v1, a current frontier model,
  and one open model) so the paper also reads as an "AI moved fast, here is
  what changed" study.

## 4. Evaluation plan

### Tier 1 – Simulation on gold prerequisite graphs (no humans, cheap, the core of the results section)

**Data.** AL‑CPL (CC BY‑NC‑SA, downloads from GitHub, verified 2026‑10‑02).
All four domains are acyclic and deep enough for recursive tracing:

| Domain | Concepts | Prereq edges | Longest chain | Mean prereqs / concept |
|---|---|---|---|---|
| Data Mining | 90 | 292 | 6 | 4.4 |
| Geometry | 88 | 524 | 9 | 6.2 |
| Physics | 124 | 487 | 6 | 4.8 |
| Precalculus | 196 | 699 | 7 | 3.7 |

Optional second source: LectureBank 2.0 / LectureBankCD (NLP, CV,
bioinformatics; 322 concepts) if the annotation files are obtainable, or
MOOCCube (17k prerequisite pairs) for scale.

**Synthetic learners.** For each domain sample N learners (e.g. 500):
pick a target concept, sample a downward‑closed known set (consistent with
the DAG) at several mastery levels (20/50/80 % of ancestors known), then
add *overconfidence noise*: with probability p ∈ {0, 0.1, 0.2, 0.3} a
learner self‑reports "know" on a concept they do not know (this is the
Dunning‑Kruger knob). Micro‑probe accuracy is modelled as a noisy oracle
(e.g. 85–90 % correct) and also ablated.

**Policies compared.**
1. RPKT v1: expand every "don't know", self‑report only (the published method).
2. Full enumeration: ask every ancestor (accuracy upper bound, probe cost upper bound).
3. Random probing with the same budget.
4. Depth‑first / BFS without monotone propagation.
5. RPKT‑2 boundary search, with and without micro‑probe verification.

**Metrics.**
* Precision / recall / F1 of the recovered *unknown set* vs ground truth.
* Boundary error (symmetric difference of the recovered cut and the true cut).
* Number of learner probes to reach F1 ≥ 0.9 (the headline efficiency number).
* Unknown unknowns surfaced: concepts self‑reported "known" but actually unknown that the method caught.
* Robustness curves vs overconfidence p.
* LLM calls and dollar cost per session.

This whole tier runs on the gold graph, so it isolates the *algorithm* from
LLM extraction quality and needs **zero API calls**. It can be finished in
days.

### Tier 2 – Quality of LLM prerequisite extraction (API calls, still no humans)

Run the v1 prompt and an improved v2 prompt (canonicalisation + DAG check)
on every concept of AL‑CPL and compare generated prerequisites to the gold
edges: precision, recall, F1 at k = 2…4, duplicate rate, cycle rate, and
cross‑domain edges. Do this for 3 models (GPT‑4o as v1 baseline, one current
frontier model, one open‑weight model). Then re‑run Tier 1 *on the
LLM‑generated graph* instead of the gold graph to show end‑to‑end
performance. Note that gold datasets are incomplete (a correct LLM
prerequisite can be absent from gold), so report a small manually‑judged
sample (e.g. 100 pairs, 2 raters, Cohen's κ) of "LLM says prerequisite,
gold says no".

### Tier 3 – Human evidence (small, makes reviewers comfortable)

Pick one of these depending on IRB and time:

* **(a) Within‑subject user study, N = 20–40 CS undergrads/grads**, 2–3
  target topics (e.g. backpropagation, TCP congestion control, Bayes'
  theorem). Conditions: RPKT‑2 vs RPKT v1 vs plain chat explanation.
  Measures: pre/post test on prerequisite items (learning gain), *calibration*
  (confidence vs correctness, Brier score, before/after), number of unknown
  unknowns surfaced per learner (self‑reported "known", failed probe), probes
  asked, time on task, SUS or NASA‑TLX.
* **(b) Expert rating, 3 instructors**, if no IRB: rate v1 vs v2 prerequisite
  trees and learning paths for correctness, completeness, and ordering on 10
  topics; report inter‑rater agreement.
* **(c) LLM‑simulated students** (StudentSim / SSKG style) as a *secondary*
  check only; cite the 2026 faithfulness critique and do not make it the
  main evidence.

For a mid/lower‑tier venue, **Tier 1 + Tier 2 + Tier 3(b)** is a complete,
defensible paper; adding Tier 3(a) makes it competitive for a better venue.

## 5. Paper skeleton (6 pages, IEEE two‑column)

1. Introduction (0.75 p): unknown unknowns; why self‑report alone fails; three contributions.
2. Related work (0.75 p): prerequisite extraction (AL‑CPL, AutoPRE, K12‑KGraph); knowledge tracing and cold start; LLM tutors and gap diagnosis (LOOM, ALIGNAgent, Capture‑Calibrate‑Coach, "Confirming Correct"); student simulators and their limits.
3. Method (1.5 p): problem formulation (downward‑closed set on a DAG), learner‑state model, probe policy, verified assessment, stopping rule, extraction + canonicalisation. One algorithm box, one architecture figure.
4. Experimental setup (0.75 p): datasets, synthetic learners, baselines, metrics, models.
5. Results (1.5 p): Table: F1 vs probes for all policies × domains. Figure: probe count vs overconfidence p. Table: extraction P/R/F1 by model. Short human/expert results table.
6. Discussion + limitations + conclusion (0.75 p): gold‑graph incompleteness, simulator realism, cost.

## 6. Work plan (rough, 6–8 weeks part‑time)

| Week | Work |
|---|---|
| 1 | Simulation harness: load AL‑CPL, synthetic learners, metrics, v1 baseline reproduced in code |
| 2 | Boundary‑search policy + micro‑probe verification; run Tier 1 grid |
| 3 | Extraction evaluation (Tier 2) across 3 models; canonicalisation |
| 4 | End‑to‑end on LLM graphs; cost accounting; app integration (Streamlit) |
| 5 | Expert rating or user study; figures |
| 6–8 | Writing, ablations reviewers will ask for (noise levels, budget caps, model swaps) |

## 7. Risks and how to pre‑empt reviewer objections

* *"Synthetic learners are unrealistic."* → The monotone + overconfidence
  model is the standard assumption in prerequisite‑based learner modelling;
  we sweep its parameters and add Tier 3.
* *"Gold prerequisite graphs are incomplete."* → Manual adjudication sample
  with κ; report both strict and adjudicated scores.
* *"Just prompt engineering."* → C1 is an algorithmic result that holds on
  the gold graph with no LLM at all; the LLM only supplies the graph.
* *"Why not fine‑tune a KT model?"* → Cold start: a single new question, no
  interaction history; cite the 2025 cold‑start KT studies.

## 8. Sources consulted (titles; arXiv/IEEE were not reachable from this environment, so abstracts were read via search snippets only — verify before citing)

* RPKT, FMLDS 2025, arXiv:2508.11892 (full PDF read).
* AL‑CPL dataset, github.com/harrylclc/AL-CPL-dataset (downloaded, statistics above computed locally).
* AutoPRE: Discovering Concept Prerequisites with LLM Agents (Springer, 2025).
* K12‑KGraph: A Curriculum‑Aligned Knowledge Graph for Benchmarking and Training Educational LLMs, arXiv:2605.09635.
* ALIGNAgent: Adaptive Learner Intelligence for Gap Identification and Next‑step guidance, arXiv:2601.15551.
* LOOM: Personalized Learning Informed by Daily LLM Conversations…, arXiv:2511.21037.
* Capture‑Calibrate‑Coach: A Graph‑Based Framework for Knowledge Monitoring Estimation and Adaptive Feedback, arXiv:2605.25419.
* Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most, arXiv:2605.16207.
* Simulating Students or Sycophantic Problem Solving? On Misconception Faithfulness of LLM Simulators, arXiv:2605.12748.
* StudentSim: Training LLM‑based Student Simulators, arXiv:2609.01591; SSKG, arXiv:2608.21668.
* FoundationalASSIST, arXiv:2602.00070; Cold Start Problem in KT, arXiv:2505.21517; CLST, arXiv:2406.10296.
* LLMs Protégés: Tutoring LLMs with Knowledge Gaps Improves Student Learning Outcome, BEA 2025.
* Prerequisite Relation Learning: A Survey and Outlook, ACM 2025.
