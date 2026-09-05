# A4 — Scalability and Cost Analysis

**AC required revision #4:** *"Add a scalability/cost analysis"*
**Also answers:** R1 W3 — *"Running pairwise verification across parameter sweeps on a battery of over 200 probe images introduces significant computational latency and resource costs. Scaling this execution-heavy approach to repositories containing thousands of skills could become a major bottleneck."*
**Partly answers:** R2 — *"Two skills may have same output but require vastly different token budgets"* and the maintainability critique

---

## ⚠ Build this first, report it last

A4 is fourth by priority but **its instrumentation must exist before A1 and A3 runs happen.** Timing, token, and cache telemetry cannot be reconstructed after the fact — retrofitting means re-running everything.

**Week 0 deliverable: the telemetry layer.** Everything else in A4 is analysis of data those hooks collect.

---

## Files

| File | Contents |
|---|---|
| [`01_measurement_protocol.md`](01_measurement_protocol.md) | What to instrument, how, and the curator-roster expansion |
| [`02_analysis_and_deliverables.md`](02_analysis_and_deliverables.md) | Complexity model, extrapolation, Pareto analysis, tie-breakers |

---

## The core argument

R1's concern assumes naive `O(n²)` pairwise verification over 177 probes. **The paper already has the answer and never quantifies it** — `sec/appendix.tex` §Candidate Generation describes fingerprint-based pruning:

> *"Evaluating the full comparator stack over all `C(n,2)` pairs is wasteful. We compute a cheap, purely output-derived fingerprint per skill… and propose as candidates only pairs whose fingerprints lie within a generous radius."*

Two facts defuse R1's objection, and both are currently unmeasured:

1. **Signatures are computed once per skill and reused across every pair.** The expensive part is `O(n)`, not `O(n²)`.
2. **Fingerprint pruning reduces the candidate set to `|C| ≪ C(n,2)`.**

**But a reduction ratio without a recall number is meaningless** — pruning everything gives perfect reduction and useless results. The required deliverable is the **reduction/recall tradeoff curve**, not a single reduction figure. That curve is the centerpiece of A4.

---

## Three cost dimensions

| Dimension | Question | Who cares |
|---|---|---|
| **Verification cost** | GPU-hours to verify a library of `n` skills | R1 |
| **Curation cost** | LLM tokens and USD per curation episode | R2, practitioners |
| **Skill-level cost** | Do two behaviorally-equivalent skills cost the same to *run*? | R2 |

The third is R2's point and is usually forgotten. It also yields the merge tie-breaker in [`02_analysis_and_deliverables.md`](02_analysis_and_deliverables.md) §4.

---

## Timeline

| Week | Work |
|---|---|
| **W0** | **Build the telemetry layer.** Wrap signature computation, pair verification, and the agent loop. |
| **W1** | Measure on the existing 100-skill library; candidate reduction/recall sweep |
| **W2** | Collect timing from A1 Corpus A + B runs (free — instrumentation already in place) |
| **W3** | Curator roster expansion; token/USD collection across models |
| **W3–W4** | A3 neural cost collection; extrapolation model; Pareto plots |
| **W4** | Tie-breaker experiment; write §5.x |

---

## Checklist

- [ ] **W0: telemetry layer wrapping all execution paths**
- [ ] Per-skill signature time; per-pair verification time by cascade stage
- [ ] Cache hit rate (the `O(n)` vs `O(n²)` argument)
- [ ] **Candidate reduction AND recall vs. fingerprint radius** — the required curve
- [ ] End-to-end wall clock + GPU-hours for `n = 100`
- [ ] Extrapolation to `n ∈ {10², 10³, 10⁴}`, naive vs. pruned
- [ ] LLM tokens per episode per model → USD/episode, USD/repair
- [ ] Expand curator roster (Claude, Gemini, DeepSeek families)
- [ ] Neural verification cost from A3
- [ ] Battery-size cost lever from A2's probe ablation
- [ ] Pareto plots: F1 vs. USD, F1 vs. wall-clock
- [ ] Non-behavioral tie-breaker experiment
