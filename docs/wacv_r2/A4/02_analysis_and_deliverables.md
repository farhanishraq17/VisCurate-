# A4 · Analysis, Extrapolation, and Deliverables

---

## 1. Complexity model (analytical — goes in the method section)

### 1.1 Naive
Full comparator stack over all pairs:

```
Cost_naive = C(n,2) · |Θ_grid| · |B| · t_forward
           = O(n² · |Θ| · |B|)
```

This is the cost R1 is objecting to, and R1 is right that it does not scale.

### 1.2 With signature caching + fingerprint pruning
```
Cost_actual = n · |Θ_grid| · |B| · t_forward     ← signatures, computed ONCE
            + n · t_fingerprint                    ← cheap screening summary
            + t_candgen(n)                         ← radius query
            + |C| · t_pair                         ← distance computation on CACHED features
```

**Two structural points:**

1. **The `|B| · t_forward` term — the genuinely expensive one — is `O(n)`, not `O(n²)`.**
2. **`t_pair` operates on cached features**, so it is a distance computation, not a forward pass.

> ⚠ **State this precisely — do not summarize the method as "the expensive part is O(n)."** Per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §32, that summary is misleading on three counts:
>
> - **pairwise comparison is still `O(|C|)`**, which may grow quadratically if candidate density scales with `n`
> - **feature storage is `O(n · |B| · |Θ| · d)`** and was omitted entirely — see §1.4
> - `cache_hit` telemetry measures reuse **in one workload**; it does not prove an asymptotic bound. **Prove the bound from the implementation**, and use telemetry only to estimate constants and confirm no recomputation occurs.

### 1.3 ⚠ Eager vs. lazy features — pick one; the original claimed both

The original asserted *both* that features are precomputed for every skill (the caching argument) **and** that stage short-circuiting saves compute. **These are in tension:**

| Implementation | Caching argument | Short-circuit savings |
|---|---|---|
| **Eager** — all learned features precomputed per skill | ✅ holds: forward cost is `O(n)` | ❌ minimal: forward cost already paid; short-circuiting saves only distance computation |
| **Lazy** — learned features computed on demand when a pair reaches that stage | ❌ weakened: forward cost depends on which pairs reach which stage | ✅ real: expensive backends often never invoked |

**Determine which the implementation actually does, document it, and make the complexity equation and telemetry match.** If eager, drop the short-circuit compute claim and keep it only as a distance-computation saving. If lazy, the `O(n)` framing needs qualifying and cost becomes workload-dependent.

Either way, report the **empirical distribution of deciding stage** (`sec/6_methodology.tex:72`, stop-at-first) — it is informative regardless, but what it *implies about cost* depends entirely on this choice.

### 1.4 ⚠ Memory may be the binding constraint, not compute

Absent from the original analysis. Feature-cache size is:

```
mem = n · |B| · |Θ_grid| · d · bytes_per_float
```

At `n = 10⁴`, `|B| = 177`, `|Θ| = 5`, `d = 768` (DINO ViT-B), float32:

```
10⁴ × 177 × 5 × 768 × 4 B  ≈  27 GB     (DINO alone)
```

With LPIPS and CLIP features as well, roughly **80 GB**; at `n = 10⁵`, roughly **800 GB**.

**Report memory alongside compute in the extrapolation table.** At scale the practical bottleneck may be feature storage rather than GPU time — and saying so is a more credible engineering analysis than a compute-only projection. Note the mitigations (lower-precision storage, on-disk memmap, recompute-instead-of-cache above a size threshold) and their cost implications.

---

## 2. Extrapolation

Using measured constants, project to `n ∈ {100, 500, 1000, 5000, 10000}`:

| `n` | Naive GPU-h | Pruned GPU-h | `\|C\|` | Signature GPU-h | Speedup |
|---|---|---|---|---|---|
| 100 | (measured) | (measured) | | | |
| 1,000 | | | | | |
| 10,000 | | | | | |

**Rules for honesty:**
- Mark measured rows vs. extrapolated rows distinctly. Do not present a projection as a measurement.
- `|C|` growth is **not** guaranteed linear — it depends on how fingerprint density scales with library size. Measure `|C|` at several sub-sampled library sizes (`n = 25, 50, 100`, and Corpus B's real `n`) and **fit** the growth rate rather than assuming it.
- State the assumption explicitly: *"assuming candidate density scales as measured on libraries of size ≤ N."*

Corpus B is valuable here precisely because it gives a second, larger, *real* value of `n` to anchor the fit.

---

## 3. Pareto analysis — the most practically useful artifact

Two plots, one point per curator model:

**F1 vs. USD per episode** and **F1 vs. wall-clock per episode.**

Annotate the Pareto frontier. This is the figure a practitioner reading the paper actually needs, and no comparable benchmark reports it.

Expected story, based on the existing results table:
- GPT-5.5 leads on quality (μF1 0.583) **and** has the lowest action cost (75.15) — likely Pareto-optimal
- Qwen 3.5 27B (μF1 0.505) is the open-weight frontier point
- The small models cluster near zero F1 at non-trivial cost — **their cost is nearly pure waste**, which the action-outcome table already hints at (Llama 3.2 3B: 194.81 invalid actions out of 200)

That last point is worth stating plainly: a model that exhausts a 200-action budget to produce one applied edit is not merely weak, it is *expensive* in proportion to its weakness. The cost axis makes the existing analysis sharper.

---

## 4. Non-behavioral tie-breakers (R2's cost and maintainability critique)

### 4.1 The gap
`sec/appendix.tex:52` says `EXACT`/`PERCEPTUAL` license *"merge to a single canonical skill"* — but never specifies **which** skill survives. R2 correctly notes that two output-identical skills may differ enormously in token budget, dependency weight, and maintainability.

### 4.2 The experiment
When a merge is licensed, score both candidates on non-behavioral attributes:

| Attribute | Measurement |
|---|---|
| Runtime | median execution time over the probe battery (from `skill_execute` telemetry) |
| Peak memory | max over battery |
| Lines of code | AST-counted, comments excluded |
| Cyclomatic complexity | standard static analysis |
| Direct dependencies | imports |
| Transitive dependency weight | resolved package count / install size |
| Has tests / docstring | boolean |

> ⚠ **A weighted-sum score is arbitrary** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §33). Runtime, dependency weight, memory, and test coverage are **incomparable objectives** — any set of weights encodes an unstated value judgment that a reviewer can reasonably dispute.
>
> **Use one of these instead:**
>
> | Approach | When |
> |---|---|
> | **Lexicographic rule** | a documented priority order (e.g. correctness-adjacent attributes first, then dependency weight, then runtime), stated in the paper |
> | **Pareto frontier** | report non-dominated survivors and let the user pick; the honest choice when priorities are genuinely application-specific |
>
> **Compare against meaningful named policies**, not merely "arbitrary choice": *fastest*, *fewest dependencies*, *best-tested*, *first-registered*. Report how often each disagrees with the others — if all policies pick the same survivor most of the time, the choice barely matters and saying so is a useful finding. Report **sensitivity** to the priority order.

### 4.3 What to report
| Metric | Meaning |
|---|---|
| **Survivor-change rate** | how often the tie-breaker picks a different skill than an arbitrary choice |
| **Library-level runtime delta** | aggregate execution time after curation, tie-broken vs. arbitrary |
| **Library-level dependency delta** | total transitive dependencies removed |
| **Library-level LOC delta** | total code retained |

Corpus B is the right setting: real ComfyUI nodes differ genuinely in dependency weight and implementation quality, whereas the hand-authored `L_0` was written uniformly by one team and will show a small, uninteresting effect.

### 4.4 Why this is worth the small effort
It answers **two** of R2's four bullets directly, it is a genuine usability improvement rather than a defensive measurement, and it converts an unspecified implementation detail into a stated, evaluated policy — closing a hole a future reviewer would otherwise find.

---

## 5. Deliverables

| ID | Artifact |
|---|---|
| **T24** | Per-component timing: signature, fingerprint, pair-verify by stage, cache hit rate |
| **T25** | End-to-end cost decomposition for `n = 100` and for Corpus B's real `n` |
| **T26** | Extrapolation to `n ∈ {10², 10³, 10⁴}`, naive vs. pruned, measured rows marked |
| **T27** | Curator cost table: tokens, USD/episode, **USD/successful repair**, wall-clock |
| **T28** | Neural verification cost from A3, reported separately |
| **T29** | Tie-breaker results: survivor-change rate + library-level deltas |
| **F10** | **Candidate reduction vs. recall** as fingerprint radius varies — operating point marked |
| **F11** | **Pareto: Mean F1 vs. USD/episode**, frontier annotated |
| **F12** | Cost-lever summary: savings from pruning, battery size, backend choice, `K` |

**Target sentences:**
> *"Signature computation, the dominant cost, is `O(n)` rather than `O(n²)`: signatures are computed once per skill and reused across every pair. Fingerprint pruning further reduces the candidate set to X% of all pairs at Y% candidate recall, giving an end-to-end projection of Z GPU-hours for a 10,000-skill library."*

> *"Curation cost varies by two orders of magnitude across models: the strongest curator repairs a defect for \$A, while the weakest models exhaust their action budget without producing a single correct repair."*

---

## 6. Paper integration

| Section | Edit |
|---|---|
| `sec/6_methodology.tex` or `sec/appendix.tex` §Candidate Generation | Add the complexity model (§1) — the section describes pruning but never analyzes it |
| **New `sec/7_results.tex` subsection "Cost and scalability"** | T24–T28, F10–F12 |
| Table 1 (model evaluation) | Add USD/episode and USD/repair columns, or a companion table |
| `sec/appendix.tex` §Training Details | **Fill the `\textcolor{red}{}` placeholder** — hyperparameters, effort levels, API versions |
| `sec/appendix.tex` §Output-Gated Curation | Document the tie-break policy (§4) |
| `sec/8_discussion.tex` | Cost is currently absent from the limitations discussion; add the measured boundary |

---

## 7. Checklist

- [ ] Complexity model written with measured constants
- [ ] Empirical deciding-stage distribution → expected `t_pair`
- [ ] **Reduction/recall curve** with operating point and its recall
- [ ] Verify the "generous radius / always includes hard negatives" claim empirically
- [ ] `|C|` growth **fitted** across ≥3 library sizes, not assumed
- [ ] Extrapolation table, measured vs. projected clearly distinguished
- [ ] Curator cost table incl. **USD per successful repair**
- [ ] Expanded roster run; verify IDs and rates at implementation time
- [ ] Neural cost reported separately
- [ ] Tie-breaker experiment on Corpus B
- [ ] Pareto plots with frontier annotated
- [ ] Cost-lever summary figure
- [ ] **Fill the appendix Training Details placeholder**
