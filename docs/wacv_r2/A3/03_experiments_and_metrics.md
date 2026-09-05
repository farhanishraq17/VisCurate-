# A3 · Experiments, Metrics, and Deliverables

---

## 1. Experimental sequence

Run in this order. Each step gates the next, and skipping the early ones produces uninterpretable results later.

### Step 1 — Within-model variance characterization ★ do this first
Before comparing *different* models, measure each model against **itself**.

> ⚠ **Use two independent seed batches** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §28). Comparing a sample set *with itself* trivially yields near-zero discrepancy and badly understates variance. Draw batch A (`K` seeds) and batch B (`K` different seeds) independently and compare A against B. Repeat over several batch pairs to characterize the null distribution rather than producing a single number.

Report per operator: MMD to itself, CI width, and whether the self-pair certifies `EXACT`/`PERCEPTUAL`.

**This is both a sanity gate and a result.**
- **As a gate:** any operator whose self-pair does not certify as equivalent indicates a broken harness — or genuine non-determinism large enough that the operator cannot be certified against anything. Resolve before proceeding.
- **As a result:** if a model's self-variance exceeds its distance to a *different* model, that is a genuine and interesting negative finding about neural tool equivalence, and it belongs in the paper. Do not hide it.

This step also establishes the **noise floor** — the smallest discrepancy that is distinguishable from run-to-run variation. `τ` cannot meaningfully sit below it, which is an empirical constraint on calibration.

### Step 2 — Cross-validation against the deterministic path
On Tier 3 seeded classical operators where both paths apply, confirm the distributional verdict matches the seed-matched deterministic verdict.

**Report the agreement rate.** Anything below ~100% is a bug to fix, not a result to report — and finding it here is far cheaper than having a reviewer find it.

### Step 3 — `K`-sweep
Per [`02_method_distributional_equivalence.md`](02_method_distributional_equivalence.md) §6. Fix `K` before full runs, on a small operator subset. Oversampling `K` across the whole roster is the easiest way to burn the GPU budget.

### Step 4 — Threshold calibration per modality
Cluster-disjoint by **model family**, frozen and hashed before test metrics.

### Step 5 — Full evaluation on the designed pair set
30–50 pairs, per-relation P/R/F1 with support, hard negatives reported separately.

### Step 6 — Safety check
False-merge rate on the designed `DISTINCT` pairs, especially the two engineered hard negatives (depth-vs-disparity, instance-vs-semantic masks).

**This is the headline.** The paper's central safety claim is `0/926` false merges on classical operators. The neural equivalent — *does the safety property survive the extension?* — is the number a reviewer will look for.

---

## 2. Metrics

### 2.1 Primary — safety preservation

| Metric | Definition |
|---|---|
| **FM_neural** | false-merge rate on adjudicated `DISTINCT` neural pairs |
| **FM_hard** | false-merge rate on the hard negatives |
| **Abstention rate** | fraction routed to `UNCERTAIN` |

Present alongside the classical numbers, **with exact intervals on both**:

| Track | FM_dist [95% CI] | FM_hard [95% CI] | Abstention |
|---|---|---|---|
| Classical (existing) | 0/926 **[0, 0.32%]** | 0/6 **[0, 39.3%]** | 0.01 |
| **Neural (new)** | | | |

> ⚠ **Do not claim "safety preserved" from a small pair set** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §3, §29). With a handful of hard negatives, a zero-error observation has an upper bound near 40% and supports no such claim. **n ≥ 59** hard negatives are required for a ≤5% upper bound at 95% confidence.
>
> If A3's hard-negative count is small — which it will be at 30–50 total pairs — then **state the bound and do not use the word "preserved."** Write *"no false merges were observed among N hard negatives (95% upper bound X%)"*, which is true and checkable. This is the difference between a scoped feasibility result and an overclaim a reviewer can puncture.

Expect a **higher abstention rate** on the neural track. That is correct behavior, not a weakness — tie it to the decision rule: wide bounds at high variance produce abstention by design.

### 2.2 Per-relation quality
One-vs-rest P/R/F1 with support counts, on the designed pair set. Same format as the revised Table 3 from A2 so the two are directly comparable.

### 2.3 Method-internal
| Metric | Purpose |
|---|---|
| Within-model self-variance per operator | the noise floor (Step 1) |
| Deterministic/distributional agreement rate | correctness check (Step 2) |
| Certificate-flip rate vs. `K` | stability (Step 3) |
| CI width vs. `K` | estimator precision |
| Per-modality calibrated `τ` | operating points |

### 2.4 Cost
Forward passes, GPU-hours, wall-clock per pair, by modality — **hand to A4**. Neural verification is far more expensive than classical, and A4's scalability story needs that number to be honest about the extension's cost.

---

## 3. What to do if results are poor

Likely, and manageable. Pre-decide the framing so it is not improvised under deadline pressure:

| Outcome | Framing |
|---|---|
| High abstention on generative operators | **Correct behavior.** The rule refuses to certify what it cannot establish. Report the abstention rate as the finding and note that generative equivalence needs larger `K` than is practical — a scoped, honest limit. |
| Self-variance exceeds between-model distance for some operator | **A real finding about neural tools.** Report it; it justifies the abstention design and is genuinely informative to the field. |
| Segmentation works, diffusion does not | **Report the boundary explicitly.** "The extension certifies segmentation and depth operators reliably; generative operators remain beyond reliable certification at practical sample sizes." That is a *scoped* claim, which is worth far more than an overreaching one. |
| False merges appear on the neural track | **Do not hide this.** Diagnose: comparator problem, threshold problem, or genuine limit? Report the diagnosis. A reviewer who finds an unreported false merge on a safety-first method will reject. |

**The governing principle:** AC item #3 asks for distributional *support*, demonstrated. A scoped, honest, well-measured extension satisfies it. An overreaching one that a reviewer can puncture does not.

---

## 4. Deliverables

| ID | Artifact |
|---|---|
| **T19** | Within-model self-variance per operator (the noise floor) |
| **T20** | Deterministic vs. distributional agreement on Tier 3 ops |
| **T21** | Per-modality calibrated thresholds (companion to Table 4) |
| **T22** | Per-relation P/R/F1 on the neural pair set, with support |
| **T23** | **Safety comparison: classical vs. neural track** — FM, FM_hard, abstention |
| **F8** | Certificate-flip rate and CI width vs. `K`, knee marked |
| **F9** | Qualitative: a certified `SEMANTIC-PRESERVING` neural pair and a rejected hard negative, side by side with output samples |
| **A-7** | Supplementary: model roster with verified IDs, versions, and `B_neural` |

**Target sentence:**
> *"Extending the verifier with distributional signatures and a confidence-interval decision rule certifies behavioral relations among neural segmentation and depth operators while preserving the safety property — no false merges on engineered hard negatives — at the cost of a higher abstention rate (X% vs. 1% on classical operators)."*

---

## 5. Paper integration

| Section | Edit |
|---|---|
| `sec/6_methodology.tex` | New subsection: distributional signatures + decision rule |
| Algorithm 1 | Extended cascade with CI-based conditions |
| Table 4 | Companion per-modality threshold table (T21) |
| `sec/7_results.tex` | New subsection: neural/stochastic results (T22, T23, F8, F9) |
| `sec/1_intro.tex:13` | **Rewrite** — two-tier scope |
| `sec/0_abstract.tex` | Mention distributional certification |
| `sec/8_discussion.tex:9` | **Rewrite** — narrow to what genuinely remains out of scope |
| `sec/appendix.tex` | Extend §Behavioral Signatures special-case classes |

> **Update all three scope statements together.** A stale "we leave stochastic operators to future work" surviving in the intro while §5 reports neural results reads as a bolted-on revision and undermines the whole extension.

---

## 6. Checklist

- [ ] Step 1: within-model variance; **self-pair sanity gate must pass**
- [ ] Establish the noise floor; constrain `τ` accordingly
- [ ] Step 2: cross-validate distributional vs. seed-matched on Tier 3
- [ ] Step 3: `K`-sweep on a subset; fix `K`; report the knee
- [ ] Step 4: per-modality calibration, cluster-disjoint by model family, **frozen before test**
- [ ] Step 5: full evaluation on the frozen 30–50 pair set
- [ ] Step 6: safety check on hard negatives — **the headline number**
- [ ] Implement affine alignment for depth (**non-negotiable**)
- [ ] Confirm generative comparison uses feature MMD, never pixel distance
- [ ] Instrument for A4 cost
- [ ] Produce T19–T23, F8, F9, A-7
- [ ] **Rewrite all three scope statements together**
