# RPKT‑2: two‑paper plan

Follow‑up to RPKT (FMLDS 2025, arXiv:2508.11892). Decisions so far:
no human study is possible; add an RL component; split into two papers.
Date of plan: 2026‑10‑02.

---

## 0. Why two papers, and in which order

The full idea has two separable halves with separate evidence:

| | Paper A (first) | Paper B (second) |
|---|---|---|
| Question | *Given* a prerequisite graph and a miscalibrated learner, how do you find the knowledge boundary with the fewest probes? | How do you *build* that graph and the probes on the fly with LLMs, and does the whole system beat RPKT v1 end to end? |
| Nature | Formulation + simulator + learned policy | System + LLM evaluation |
| LLM needed | No (gold graphs) | Yes |
| Venue | ICLR 2027 workshop, 4 pages, non‑archival (deadlines typically Feb 2027) | Mid/lower‑tier IEEE conference or EDM, 6 pages, archival (deadlines Mar–Jun 2027) |
| Cost | Zero API cost | API cost for extraction + simulated learners |

Paper A goes first because (1) the simulator it builds is the environment
Paper B needs anyway, (2) its deadline is earliest, (3) it costs nothing to
run, and (4) B cites A for the policy and is free to focus on the LLM side.

Non‑overlap rule (to avoid a self‑plagiarism / salami objection):
* Only A claims the formulation, the benchmark, and the policy result.
* Only B claims graph construction, probe generation, end‑to‑end results,
  model‑generation comparison, and cost. B uses A's policy as a component
  and cites it; B's tables never re‑report A's gold‑graph numbers except as
  one reference row.
* Confirm the workshop is non‑archival before submitting A.

---

## Paper A — "Knowledge‑Boundary Search: Learning to Probe Miscalibrated Learners on Prerequisite Graphs"

### Claims
* **A1 (formulation).** Discovering a learner's unknown unknowns is finding
  a downward‑closed set (order ideal) of a prerequisite DAG using a noisy,
  overconfident respondent, with two probe types of different cost and
  reliability (self‑report, verification item). This connects the education
  problem to active learning of monotone functions / group testing on
  posets, which nobody has done for prerequisite tracing.
* **A2 (benchmark).** A released simulator built on gold prerequisite
  DAGs (AL‑CPL: Data Mining, Geometry, Physics, Precalculus; optionally
  LectureBank/MOOCCube) with a parametric learner model: mastery level,
  overconfidence rate p (false "know"), under‑claim rate r (false "don't
  know"), verification‑item noise q. Gym‑style interface.
* **A3 (policy).** A learned probe policy (GNN encoder + PPO) that decides
  *which* concept to ask, *whether* to spend a verification item, and
  *when to stop*, reaches the same boundary F1 with far fewer learner
  interactions than RPKT v1's recursive expansion and than information‑
  gain heuristics, and transfers to a held‑out domain and to unseen
  overconfidence levels.

### Novelty versus prior work
* Knowledge tracing (DKT, AKT, LLM‑KT): needs interaction history; here
  there is none (one new question). Cold‑start KT papers confirm the gap.
* Computerised adaptive testing: picks items by IRT difficulty; ignores
  prerequisite structure and does not target the boundary.
* Prerequisite extraction (AL‑CPL, AutoPRE, K12‑KGraph): builds graphs;
  no learner in the loop.
* RPKT v1: blind recursion, self‑report only, no stopping rule.
* Capture‑Calibrate‑Coach (2026): estimates calibration; does not search.

### Steps
1. **Simulator (week 1–2).**
   * Load AL‑CPL `.preqs` → DAG per domain (verified acyclic; chains 6–9 deep).
   * Learner generator: pick target t; sample an order ideal K of
     ancestors(t) at mastery m ∈ {0.2, 0.5, 0.8}; responses: self‑report
     says "know" w.p. 1−r if c∈K, w.p. p if c∉K; verification item correct
     w.p. q if c∈K, 1−q if c∉K.
   * Episode: observation = local DAG + per‑node evidence; actions =
     {ask(c), verify(c), stop}; reward = −cost per action + terminal
     F1(unknown set) (or −|Δ boundary|). Budget cap B.
2. **Baselines (week 2).**
   * RPKT v1 re‑implemented exactly (expand all children on "don't know",
     self‑report only, depth cap).
   * Full enumeration (ask every ancestor) — accuracy ceiling.
   * Random, BFS, DFS at equal budget.
   * EIG heuristic: Bayesian belief over nodes with monotone propagation
     on the DAG; pick argmax expected entropy reduction over the cut;
     verify when posterior is near 0.5 and the node is a "know" claim.
   * Chain binary search (for comparison on long chains).
3. **RL agent (week 3–4).** Node features: belief, depth, claimed?,
   verified?, in/out‑degree, distance to target. GraphSAGE/GAT encoder;
   policy head over nodes × {ask, verify} + stop; PPO; curriculum over m,
   p, q. Train on three domains, test on the fourth (leave‑one‑domain‑out).
4. **Experiments (week 4–5).**
   * Table 1: probes to reach F1 ≥ 0.9, and F1 at budgets 10/20/40, per
     policy × domain.
   * Figure 1: probes vs overconfidence p (v1 degrades, A3 flat).
   * Figure 2: unknown unknowns caught (false "know" flipped) vs
     verification budget.
   * Table 2: held‑out‑domain and held‑out‑noise transfer.
   * Ablations: no verify action; no monotone propagation; reward shaping.
5. **Writing (week 6–7).** 4 pages + appendix; release code and simulator.

### If RL does not beat the EIG heuristic
The paper still stands: formulation + benchmark + a strong heuristic, with
the RL result reported honestly as "matches the heuristic, learns the
noise model without being told it". Decide the framing in week 4.

---

## Paper B — "RPKT‑2: On‑Demand Prerequisite Graphs and Verified Self‑Assessment for Discovering Unknown Unknowns with LLMs"

### Claims
* **B1 (construction).** An agentic extraction loop (Proposer → Critic →
  Canonicaliser, with embedding‑based name merging and DAG enforcement)
  produces prerequisite graphs with higher precision, fewer duplicates and
  zero cycles compared with the single‑prompt extraction of RPKT v1,
  measured against AL‑CPL gold across three model generations (GPT‑4o as
  the v1 baseline, one current frontier model, one open‑weight model).
* **B2 (verification).** LLM‑generated one‑item micro‑probes, filtered by a
  Judge agent for validity, catch a measurable share of overclaims that
  self‑report misses, in end‑to‑end runs with profile‑conditioned LLM
  student simulators and, if obtainable, learner states derived from real
  MOOCCube watching logs.
* **B3 (system).** The full system (B1 + B2 + Paper A's policy) recovers
  the learner's boundary with N× fewer interactions than RPKT v1 at a
  stated dollar cost and latency per session, and is deployed in the
  existing Streamlit app.

### Novelty versus prior work
* First system that constructs the prerequisite graph on demand *and*
  actively searches the boundary *and* verifies self‑report, from a single
  target question with no history (LOOM needs weeks of chat; ALIGNAgent
  starts from learner work; AutoPRE/K12‑KGraph build graphs offline).
* First quantitative evaluation of dynamic LLM prerequisite discovery
  against gold graphs across model generations (RPKT v1 never measured it).
* Verified self‑assessment as a counted metric ("unknown unknowns surfaced").

### Steps
1. **Graph builder (week 8–9).** Agentic extraction; embedding
   canonicalisation (cosine threshold tuned on AL‑CPL aliases); cycle
   rejection; caching; three models behind one interface.
2. **Extraction evaluation (week 9–10).** For every AL‑CPL concept: P/R/F1
   at k = 2…4 vs gold edges, duplicate rate, cycle rate. Gold is
   incomplete, so adjudicate a 100‑pair sample of "LLM yes, gold no" with
   two author raters and Cohen's κ; report strict and adjudicated scores.
3. **Micro‑probe generator + Judge (week 10).** Item validity measured by
   discriminability: a mastery‑conditioned simulated student passes, a
   non‑mastery one fails.
4. **End‑to‑end (week 11–12).** Learners: (a) profile‑conditioned LLM
   students (StudentSim/SSKG‑style prompting; cite the 2026 faithfulness
   critique and report agreement with the profile), (b) MOOCCube
   log‑derived states if the user logs are downloadable. Conditions: v1,
   v2 without verification, v2 without agentic extraction, v2 with
   heuristic policy, v2 full. Metrics as in A plus overclaims caught,
   cost, latency.
5. **App integration (week 12).** Replace the recursive expansion in
   `knowledge_tracer.py` / `app.py` with the policy loop; keep the
   binary UI; add the micro‑probe card.
6. **Writing (week 13–14).** 6 pages IEEE two‑column.

---

## Shared timeline (from 2026‑10‑02)

| Weeks | Deliverable |
|---|---|
| 1–2 | Simulator + v1 baseline + EIG heuristic (also serves B) |
| 3–5 | RL agent, Paper A experiments |
| 6–7 | Paper A written; submit when ICLR 2027 workshop CFPs open (check archival policy) |
| 8–10 | Graph builder + extraction evaluation |
| 11–12 | End‑to‑end runs, app integration |
| 13–14 | Paper B written; pick IEEE/EDM venue by deadline |

## Risks
* RL marginal over heuristic → see Paper A fallback.
* Workshop is archival → choose another workshop or fold A into B.
* MOOCCube logs unavailable → B relies on LLM‑simulated learners only;
  say so and lean on the faithfulness‑agreement check.
* Gold graph incompleteness → adjudicated sample with κ.
* AL‑CPL licence is CC BY‑NC‑SA: fine for research, cite it.

## Sources consulted
RPKT PDF (read in full); AL‑CPL repo (downloaded, DAG statistics computed
locally). The 2025–2026 papers below were seen only through search
snippets because arXiv, IEEE Xplore and Semantic Scholar are blocked from
this environment; verify before citing: AutoPRE (2025); K12‑KGraph
(2605.09635); ALIGNAgent (2601.15551); LOOM (2511.21037);
Capture‑Calibrate‑Coach (2605.25419); Confirming Correct, Missing the Rest
(2605.16207); Sycophantic simulators (2605.12748); StudentSim (2609.01591);
SSKG (2608.21668); FoundationalASSIST (2602.00070); cold‑start KT
(2505.21517); CLST (2406.10296); LLM Protégés (BEA 2025); Prerequisite
Relation Learning survey (ACM 2025).
