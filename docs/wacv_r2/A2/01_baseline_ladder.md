# A2 · The Baseline Ladder

**Answers:** AC #2 (first half) · R3 W2 — *"The paper never directly tests its main claim… I expected to see a comparison against a text- or embedding-based matching approach."*

---

## 1. Hypothesis, stated before results

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §14. The original framed this as *"a ladder where the gap narrows but never closes"* — an **anticipated result presented as a design principle.** Corrected to a testable hypothesis with pre-declared refutation conditions.

**H1.** No baseline that reads only text, code, or structure — without executing the skill — attains the verifier's safety level on the merge decision at matched operating points.

**H2.** No non-execution baseline can resolve the six-relation taxonomy; each expresses at most a binary merge-vs-distinct split.

**Pre-declared outcomes:**

| Observation | Conclusion |
|---|---|
| Every rung admits false merges the verifier rejects | H1 supported |
| **Any rung matches or beats the verifier on merge safety** | **H1 refuted** — report it plainly, and rest the contribution on H2 |
| A rung resolves subsumption direction or complementarity | H2 refuted — a substantially weaker paper; report anyway |

**H2 is the more robust claim** and does not depend on beating anyone on a scalar. Text baselines cannot represent `SUBSUMPTION` direction, `SEMANTIC-PRESERVING`, or `COMPLEMENTARY` at all — a structural limitation, not a performance gap. If H1 falls, H2 still carries the contribution.

Declare both before looking at results. `sec/7_results.tex:39` currently asserts the H1-supported outcome (*"added sophistication shrinks the gap without closing the structural blind spot"*) — that sentence must be **re-earned** against the new rungs, not carried over.

---

## 2. The eight rungs

| # | Baseline | Reads | Status | Threshold |
|---|---|---|---|---|
| 1 | **name-match** | name + identifier tokens | ✅ built | Jaccard ≥ 0.5 |
| 2 | **TF–IDF cosine** | name + description + family | ✅ built | cosine ≥ 0.6 |
| 3 | **sentence-embedding** | name + description + family | ❌ **new** | swept |
| 4 | **code-embedding** | source code | ❌ **new** | swept |
| 5 | **AST clone detection** | normalized AST | ❌ **new** | swept |
| 6 | **LLM-on-descriptions** | name + description | ✅ built | n/a (categorical) |
| 7 | **LLM-on-source-code** | full source | ❌ **new** | n/a (categorical) |
| 8 | **output-grounded (VisCurate)** | executed outputs only | ✅ built | calibrated |

### Why rungs 4, 5, 7 are non-optional
`sec/0_abstract.tex` and `sec/1_intro.tex:15` both claim a skill's *"name, description, **or source code**"* is a poor proxy for behavior. **The source-code half of that claim is currently asserted and never tested.** A reviewer with R3's confidence level will notice. Rungs 4, 5, and 7 close it.

Rung 7 in particular is the **strongest possible non-execution baseline**: a frontier LLM reading the actual implementation. If it still fails on the cases execution catches, the argument is essentially closed. This is the rung most worth investing in.

---

## 3. Implementation per rung

### Rung 3 — sentence-embedding
Replaces TF–IDF as the "serious" text baseline. TF–IDF alone is dismissible as a strawman.

- Models: `all-mpnet-base-v2` (default), `gte-large`, or an `E5`-family encoder. Report the best.
- Input: `f"{name}. {description}. Family: {family}"`
- Score: cosine similarity, `L2`-normalized
- **Sweep the threshold**; report the full P–R curve and AUPRC

### Rung 4 — code-embedding
- Models: `UniXcoder`, `CodeT5+`, or `CodeBERT`. Report the best.
- Input: the skill's function source, **with the docstring stripped** — otherwise this rung leaks rung 3's signal and the ladder stops being monotone in modality
- Score: cosine similarity
- Sweep threshold; P–R curve + AUPRC

> **Docstring stripping is essential.** Without it, "code embedding" is partly a description embedding and the experiment does not test what it claims to test.

### Rung 5 — AST clone detection (**two variants, both required**)
The strongest *structural* non-execution method. Catches renamed-variable duplicates that embeddings miss.

> ⚠ **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §17. The original canonicalized all literals to type placeholders. That makes the baseline **blind to defect type (v), parameter-default drift, by construction** — rigging a baseline to fail, which is the mirror image of the strawman problem this ladder exists to avoid.

Run **both** variants and report both:

| Variant | Normalization | Purpose |
|---|---|---|
| **structure-only** | α-rename identifiers, strip docstrings/comments, **canonicalize literals** | pure structural clone detection |
| **semantics-preserving** | α-rename identifiers, strip docstrings/comments, **retain literal values, default arguments, and called API names** | the fair, strong baseline |

The semantics-preserving variant is the one the paper's claim must beat. If only the structure-only variant were reported, a reviewer could correctly object that the baseline was handicapped against exactly the defect type the benchmark injects.

- Parse both functions with Python `ast`
- Score: normalized tree edit distance (Zhang–Shasha), or token-sequence Jaccard over the normalized AST serialization (cheaper, near-equivalent in practice)
- Sweep threshold; P–R curve + AUPRC for each variant

**Predicted failure modes to highlight in the paper** — these make the point precisely:
- *identical code path, different default parameter* → AST says duplicate, execution says `DISTINCT`
- *different implementations, identical behavior* → AST says distinct, execution says `EXACT`

Both are exactly the errors the paper exists to prevent, and neither is fixable with a better parser.

### Rung 7 — LLM-on-source-code
- **Model must be disjoint from every curation subject** — `sec/6_methodology.tex:163` already promises this for rung 6; keep the promise here
- Prompt: both function sources, asked to return exactly one relation word from the taxonomy

> ⚠ **Two corrections, 2026-08-18** — per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §16. Both apply to **rung 6 as already built**, not only to the new rung 7.

**(a) Do not map unparseable replies to `DISTINCT`.** The current protocol (`sec/6_methodology.tex:163`) treats unparseable output *"conservatively as `DISTINCT`."* That folds model failure into a safety number — the baseline looks safer precisely when it fails hardest, and its failure rate becomes invisible.

**Record `ABSTAIN`/`INVALID` as its own outcome**, report its rate, and present a coverage–risk view: safety among *parsed* answers, alongside the fraction unparsed. Report the old convention too if continuity with the existing table is needed, but lead with the honest one.

**(b) "Deterministic decoding" and "three seeds" are mutually inconsistent.** At temperature 0 there is no seed to vary. Pick one and state it:
- *If the API exposes real seed control:* vary the seed, report variance across seeds.
- *Otherwise (the usual case):* **repeat the identical call N times** and report observed nondeterminism as a measured property. Do not describe this as seed variation.

Either way, a single categorical run is not a measurement — repeat and report dispersion.

---

## 4. Fair comparison — one target, all rungs swept

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §4 and §15. Two separate problems were found.

### 4.1 ⚠ The calibration target is not the reported safety metric

The paper calibrates on **precision with `DISTINCT` as the positive class** (`sec/6_methodology.tex:169`) but headlines the **false-merge rate on distinct pairs** (`sec/6_methodology.tex:192`). These are different quantities, and the first is **inflated by class imbalance**: with 926 distinct vs. ~18 non-distinct pairs in `G_0`, predicting `DISTINCT` for everything already yields precision 0.981. A 0.99 constraint is therefore a weak one.

The quantity that actually governs silent merges is the **false discovery rate on the merge decision** — and it is never reported.

**Fix: define `mergeable` as the positive class and report all three, so nothing hides behind anything else:**

| Quantity | Definition | Role |
|---|---|---|
| **FDR on merges** | FP / (TP + FP) among predicted-mergeable | **the safety number that matters** |
| **FPR on distinct** | FP / (all truly distinct) | the paper's current headline; keep for continuity |
| Precision on non-equivalence | TN / (TN + FN) | the current calibration target; keep for continuity |
| **Recall on merges** | TP / (all truly mergeable) | what safety costs — estimable only on `G_ρ` |

**Calibrate against merge-decision FDR**, since that is the property the paper claims.

### 4.2 ⚠ Sweep the existing baselines too

The original swept only the *new* rungs while leaving `name-match` at 0.5 and TF–IDF at 0.6. That is an unfair comparison **in the direction that flatters our result** — the old baselines are pinned at arbitrary thresholds while new ones get optimized operating points.

**Sweep and match every threshold-based rung identically**, including the two already in the draft. Their published numbers (44/926, 1/926) are at fixed thresholds and must be re-derived at matched operating points before they can be compared with the rest.

### 4.3 Reporting

| Method | Op. point | **FDR_merge** | FPR_dist | FM_hard [95% CI] | Abstain/Invalid | AUPRC |
|---|---|---|---|---|---|---|
| name-match | **matched** (re-derived) | | | | | |
| TF–IDF | **matched** (re-derived) | | | | | |
| sentence-emb | matched | | | | | |
| code-emb | matched | | | | | |
| AST clone (structure-only) | matched | | | | | |
| AST clone (semantics-preserving) | matched | | | | | |
| LLM-desc | categorical | | | | | — |
| LLM-source | categorical | | | | | — |
| **output-grounded** | calibrated | | | | | — |

- **Report AUPRC** for swept rungs — threshold-free, forecloses "you picked a bad threshold."
- **Every rate carries raw counts and an exact (Clopper–Pearson) interval.** See [`../README.md`](../README.md) §Statistical rules — the existing `0/6` hard-negative figure has a 95% upper bound of **39%**, so it cannot support the safety claim without a larger slice.

### If a baseline reaches 0 false merges at matched precision
Report it plainly. Then make the argument that actually matters and that no text method can touch:

> Text baselines express **at most a binary merge-vs-distinct split.** They cannot represent `SUBSUMPTION` direction, `SEMANTIC-PRESERVING`, or `COMPLEMENTARY` at all — precisely the distinctions that license `parameterize` rather than `merge`. The tracks differ in kind, not only in degree.

That argument is already in `sec/6_methodology.tex` §Metrics. If a baseline matches on safety, **lead with the taxonomy argument instead** and say so honestly rather than hunting for a threshold where the baseline looks worse.

---

## 5. Evaluation sets

Run the full ladder on **all three**:

| Set | Pairs | Purpose |
|---|---|---|
| `G_0` (clean designed graph) | 944 | current benchmark; comparable to Table 2 |
| `G_ρ` (corrupted graphs) | pooled over 300 instances | **the only set with `EXACT`/`PERCEPTUAL` support** |
| **A1 Corpus A** (real cross-library) | ~250–320 | the ladder on real code with external labels |

The third is the one that answers R3. See [`../A1/01_corpus_A_cross_library.md`](../A1/01_corpus_A_cross_library.md).

---

## 6. Deliverables

| ID | Artifact |
|---|---|
| **T7** | Extended safety/divergence table — all 8 rungs at matched precision, on `G_0` |
| **T8** | Same ladder on `G_ρ` — **with `EXACT`/`PERCEPTUAL` support** |
| **T9** | Same ladder on A1 Corpus A (real pairs) |
| **F3** | P–R curves for rungs 3/4/5 with the verifier's operating point marked |
| **A-6** | Supplementary: prompts for rungs 6 and 7, plus variance across 3 seeds |

**Target sentence:**
> *"Across a ladder of seven non-execution baselines spanning names, descriptions, sentence embeddings, code embeddings, AST clone detection, and frontier LLMs reading full source, none resolves the behavioral taxonomy, and all admit silent merges that output grounding rejects."*

---

## 7. Checklist

- [ ] Implement rung 3 (sentence-embedding), sweep, AUPRC
- [ ] Implement rung 4 (code-embedding) — **strip docstrings**
- [ ] Implement rung 5 (AST clone) with α-renaming + literal canonicalization
- [ ] Implement rung 7 (LLM-on-source), model disjoint from curation subjects
- [ ] Run rungs 6 and 7 **3× with different seeds**; report variance
- [ ] Implement matched-precision operating-point selection
- [ ] Run full ladder on `G_0`, `G_ρ`, and A1 Corpus A
- [ ] Produce T7, T8, T9, F3
- [ ] If any baseline ties on safety → **pivot the narrative to the taxonomy argument, honestly**
