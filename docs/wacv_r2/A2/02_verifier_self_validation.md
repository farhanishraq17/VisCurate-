# A2 · Verifier Self-Validation

**Answers:** AC #2 (second half) · R3 W3 — *"The verifier itself is not validated. Since the ground-truth relationships are known by construction, it should be straightforward to report the verifier's precision and recall."*

R3 is right that it is straightforward. The current draft reports it on the wrong graph.

---

## 0. ★ BLOCKER — resolve the merge-policy contradiction before running anything

> **Added 2026-08-18.** See [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §1. **Nothing else in A2 is meaningful until this is settled**, because it determines the definition of the positive class every safety metric is computed against.

`SEMANTIC-PRESERVING` is described **four different ways** in the manuscript:

| # | Location | Statement |
|---|---|---|
| 1 | `sec/appendix.tex:52` | licenses `parameterize` **or `merge`** |
| 2 | `sec/6_methodology.tex:125` | licenses parameterization or **unification** rather than an outright merge |
| 3 | `sec/7_results.tex:60` | licenses **only parameterization** |
| 4 | `sec/6_methodology.tex:160` | *"**Because only `EXACT` and `PERCEPTUAL` license a merge**, the binary mergeable decision is the axis on which every method is directly comparable"* |

**Statement 4 is the dangerous one.** The entire comparability argument of the equivalence benchmark — the basis for every method comparison in Table 2 — rests on the premise that only `EXACT` and `PERCEPTUAL` license a merge. If statement 1 is what the code does, then:

- the binary `mergeable` positive class in Table 2 is **defined wrongly**
- the headline `0/926` and `0/6` false-merge figures are computed against the **wrong positive class**
- and since `SEMANTIC-PRESERVING` has **precision 0.037** with a high false-positive rate on truly-`DISTINCT` pairs, a semantic-licensed merge policy would produce a false-merge rate nowhere near zero

The reported zero false merges strongly suggest the implementation merges only on `EXACT`/`PERCEPTUAL` and that `sec/appendix.tex:52` is a documentation error. **Verify against the gate implementation — do not assume.**

### Required actions (in order)
1. **Read the gate code.** Determine what relations actually trigger `merge`.
2. Record the finding in the artifact. If the code *does* license semantic merges, **the published safety numbers are invalid** and must be recomputed and the discrepancy disclosed.
3. **Fix all four statements to agree.**
4. **Recommended policy:** only `EXACT` and `PERCEPTUAL` license `merge`; `SEMANTIC-PRESERVING` licenses `parameterize` only, never a merge. This is the only policy under which precision 0.037 is defensible as cost-benign — and the cost-benign argument at `sec/7_results.tex:60` is currently the paper's sole defense of that number.
5. Re-run all safety metrics under the confirmed policy and **state the policy explicitly** in the results section rather than leaving it to the appendix.

---

## 1. The hole, precisely

`sec/7_results.tex:60` states:

> *"`EXACT` and `PERCEPTUAL` have no support in `G_0`, since a clean library contains no true duplicates by construction, so the mergeable positive class is not estimable from this graph."*

And `sec/8_discussion.tex:9` repeats it:

> *"Merge recall is also not estimable on the current clean graph because it contains no designed `EXACT` or `PERCEPTUAL` pairs."*

**So the verifier's precision and recall on exactly the two relations that authorize a destructive merge are unreported.** Table 3 reports `SUBSUMPTION`, `SEMANTIC-PRESERVING`, `COMPLEMENTARY`, and `DISTINCT` — every relation *except* the ones that can silently delete a capability.

This is the single most exploitable gap in the paper for a reviewer with R3's expertise.

---

## 2. The fix is free

The corrupted graphs `G_ρ` **contain the positive class by construction.** From `sec/6_methodology.tex:144`, the defect factories include:

- **(iii) duplicate** — *"inserts an exact or perceptual copy of an existing skill"* → `EXACT` / `PERCEPTUAL` ground truth
- **(iv) subsumption** — *"inserts a fixed-parameter specialization of a more general skill"* → `SUBSUMPTION` ground truth

And `G_ρ = G_0 ∪ Δ(log)` is derived automatically from the corruption log, with the ideal-action key. **The answer key already exists.** The 300-instance grid has already been run for the curation-agent study.

> **This is pure re-analysis of data you already have. No new compute. Do it in week 0.**

If the per-pair verifier verdicts were not persisted during the original agent runs, the re-run is still cheap: verification is deterministic and does not need the LLM agents in the loop at all — run the verifier directly over each `G_ρ`'s pair set.

---

## 3. Protocol

### 3.0 ⚠ First verify the answer key is complete

> **Added 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §18. **Do not skip this because the re-analysis is cheap** — a defective answer key produces confidently wrong numbers.

`G_ρ = G_0 ∪ Δ(log)` is derived from the corruption log, which records the relations each defect **injects** (donor→target). It may not record every *incidental* relation the injection creates. Example: inserting a duplicate of skill `S` also creates relations between that duplicate and everything `S` was already related to — is that transitive closure materialized in `Δ(log)`, or only the direct donor edge?

**If incidental relations are unlabelled and the evaluation treats unlabelled pairs as `DISTINCT` by default, the ground truth is wrong** and the verifier will be penalized for correct verdicts.

**Check:** for a sample of instances, manually enumerate the true relation set and compare against `G_ρ`. If incidental relations are missing, either materialize the transitive closure or restrict evaluation to the pairs the log explicitly labels — and **say which** in the paper.

### 3.1 Steps

1. For each of the 300 corrupted instances, reconstruct `L_ρ` and `G_ρ` by replay (the paper guarantees exact reproducibility from `(L_0, log)`).
2. Run the verifier over all candidate pairs at the **frozen operating point** (Table 4). No re-calibration.
3. Pool predictions and ground truth across instances.
4. Report per-relation precision / recall / F1 **with support counts**, one-vs-rest.
5. Break down by corruption rate `ρ ∈ {0.1 … 1.0}` — does verification degrade as the library gets messier? A reviewer will ask, and it is free to answer.
6. Report the **full 7×7 confusion matrix** over the taxonomy (including `UNCERTAIN`). Confusion structure is more informative than per-class F1 and shows *where* errors land — the cost-asymmetry argument depends on errors landing in benign cells.
7. **Bootstrap by corruption instance / base skill / seed — never by individual pair.** Pairs within an instance share skills and are strongly dependent; pair-level bootstrap would understate variance substantially.
8. **Report exact (Clopper–Pearson) intervals on every rate**, with raw counts.

### Target table

| Relation | Support | P | R | F1 |
|---|---|---|---|---|
| `EXACT` | **(now non-zero)** | | | |
| `PERCEPTUAL` | **(now non-zero)** | | | |
| `SUBSUMPTION` | | | | |
| `SEMANTIC-PRESERVING` | | | | |
| `COMPLEMENTARY` | | | | |
| `DISTINCT` | | | | |
| `UNCERTAIN` (abstention) | | rate only | | |

Plus: **P/R/F1 on the binary mergeable decision** — the axis on which every method in the ladder is comparable, now with a populated positive class.

---

## 4. Fixing the two broken rows in Table 3

### 4.1 `SEMANTIC-PRESERVING` precision = 0.037

**Diagnosis.** `sec/7_results.tex:60` explains it: *"under a permissive semantic band, many truly-`DISTINCT` pairs fall inside it, which also depresses `DISTINCT` recall to 0.605."* So `τ_sm = 0.15` (DINO p90) is too loose for `G_0`'s composition.

Note the existing safeguard is already two-sided — `sec/6_methodology.tex:125` says the CLIP distance is taken as the larger, *"so a match must satisfy both."* Requiring both is therefore **not** an available fix; it is already in place. Look elsewhere.

> ⚠ **No test-driven threshold repair** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §20). Fixes 2–4 below are *motivated by a bad test result*. Selecting them on the test set is leakage and would invalidate the repaired number.
>
> **Required procedure:** develop and select any new pre-filter, margin, or threshold on a **fresh validation split**, then evaluate **once** on untouched test families. Report the validation-selection process explicitly. If there is no budget for a clean split, do fix 1 only — a curve is honest without any repair at all.

**Candidate fixes, in order of preference:**

1. **Report a P–R curve over `τ_sm ∈ [0.02, 0.30]` instead of a single point.** The honest fix, requiring no new method: it reframes 0.037 as a chosen operating point rather than an accident. **Do this regardless of what else you do**, and it carries no leakage risk.
2. **Add a pre-filter before the semantic stage** — require matching spatial extent and channel semantics. Many false positives are likely separable on cheap structural grounds. *(needs a clean split)*
3. **Re-calibrate `τ_sm`** using the same precision-first rule as the other thresholds. Note the current rule targets precision on *non-equivalence*; `SEMANTIC-PRESERVING` may need its own explicitly stated criterion. *(needs a clean split)*
4. **Add a margin guard** — reject a semantic match if the pair's pixel/perceptual distance is far above threshold, so the loosest stage cannot rescue a pair the strict stages rejected decisively. *(needs a clean split)*

**Keep the existing cost-benign argument** (`sec/7_results.tex:60`: the errors are over-labelled as semantic or routed to `UNCERTAIN`, never merged; and `SEMANTIC-PRESERVING` licenses only parameterization). It is a good argument. It is just not a substitute for a curve.

### 4.2 `COMPLEMENTARY` F1 = 0.000

**Diagnosis.** `G_0` contains **one** complementary pair. F1 at n=1 is not a measurement.

### ⚠ First define the unit of analysis

> **Added 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §19.

Complementarity is **not a property of a pair** — it is a property of a **composition**: a triple `(a, b, target)` under a specified operator and order. And most candidate examples are **lossy**: blur∘sharpen, rotate θ ∘ rotate −θ, and resize-up ∘ resize-down only *approximately* recover the identity, so "does it compose?" has no answer until a tolerance is fixed.

**Before reporting any F1, declare:**

| Element | Must be specified |
|---|---|
| Composition operator | `f_a ∘ f_b` — and note the paper does **not** require commutation (`sec/6_methodology.tex:127`), so order matters |
| Order(s) evaluated | both `a∘b` and `b∘a`, reported separately |
| Target behavior | identity? a named third operation (opening/closing)? |
| Tolerance | which relation certifies "composes to target" — `PERCEPTUAL`? `SEMANTIC-PRESERVING`? |
| Ground truth | what makes a triple *truly* complementary, decided before measurement |

Without these, an F1 for `COMPLEMENTARY` is not interpretable regardless of its value.

**Then build the probe set.** Target **20–30 triples**, with the lossy ones marked:

| Pair | Composition target | Lossy? |
|---|---|---|
| dilate / erode | opening, closing | no (exact for the composed op) |
| forward FFT / inverse FFT | identity | precision-sensitive |
| split channels / merge channels | identity | no |
| pad / crop | identity at matched extents | no |
| encode / decode (JPEG) | approximate identity | **yes** |
| blur / sharpen | approximate identity | **yes** |
| resize up / resize down | approximate identity | **yes** |
| rotate θ / rotate −θ | identity | **yes** (interpolation loss) |
| log / exp transform | identity | **yes** (quantization) |
| threshold / invert | complementary masks | no |

Report exact and lossy composition cases **separately**. A method that certifies exact complements and abstains on lossy ones is behaving correctly, and pooling them would hide that.

**Free real pairs from A1:** the published Albumentations→Kornia row `RandomBrightnessContrast → RandomBrightness / RandomContrast` is exactly one transform decomposing into two — a real complementary instance. See [`../A1/01_corpus_A_cross_library.md`](../A1/01_corpus_A_cross_library.md) §2.1.

**Alternative if the detector cannot be made to work:** remove the row and state explicitly that `COMPLEMENTARY` is advisory, gates no edit, and is not quantitatively evaluated. The draft already says it is advisory (`sec/6_methodology.tex:127`), so this is defensible — but **(a) is strictly better** and the pairs are cheap to build.

> Whichever path you take, **do not ship a 0.000 in a results table.** A reviewer scanning tables sees that number before reading the surrounding argument, and it reads as a broken method.

---

## 5. Deliverables

| ID | Artifact |
|---|---|
| **T10** | Per-relation verifier P/R/F1 on `G_ρ` — **with `EXACT`/`PERCEPTUAL` support** |
| **T11** | 7×7 confusion matrix over the taxonomy incl. `UNCERTAIN` |
| **T12** | Verifier quality vs. corruption rate `ρ` |
| **T13** | Revised Table 3 — `SEMANTIC-PRESERVING` as a curve, `COMPLEMENTARY` with real support |
| **F4** | P–R curve over `τ_sm` with the chosen operating point marked |

### Text edits this unlocks
- `sec/7_results.tex:60` — the "not estimable" sentence must be **rewritten**, not deleted; say it is not estimable on the *clean* graph and report it on `G_ρ`
- `sec/8_discussion.tex:9` — same sentence appears here; update both
- `sec/6_methodology.tex:160` — the equivalence benchmark description should mention the `G_ρ` evaluation

---

## 6. Checklist

- [ ] **Week 0:** replay 300 `G_ρ` instances; run verifier at frozen operating point
- [ ] Pool predictions; compute per-relation P/R/F1 **with support counts**
- [ ] Produce the 7×7 confusion matrix
- [ ] Break down by `ρ`; bootstrap CIs across 5 seeds
- [ ] Report binary mergeable P/R/F1 with a populated positive class
- [ ] Build `τ_sm` P–R curve; choose and justify the operating point
- [ ] Test the spatial/channel pre-filter and the margin guard
- [ ] Build 20–30 complementary pairs (incl. real ones from A1)
- [ ] Re-evaluate `COMPLEMENTARY` on the dedicated set
- [ ] **Rewrite the two "not estimable" sentences** in results and discussion
