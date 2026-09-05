# A3 · Method — Distributional Behavioral Equivalence

> **Revised 2026-08-18** following [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §2, §25–31. The original pooled-MMD formulation was **mathematically invalid** and is documented in §0 so the error is not reintroduced.

---

## 0. The error this design replaces — do not reintroduce it

The original formulation pooled every probe and every seed into a single sample set per skill:

```
σ_pooled(s, θ) = { φ(f_s(x_i, θ, ω_k)) : i = 1…m, k = 1…K }      ← WRONG
```

and took MMD between two such pooled sets.

**This destroys conditioning on the input.** Counterexample: skill A outputs `blur(x₁), sharpen(x₂)`; skill B outputs `sharpen(x₁), blur(x₂)`. The pooled output distributions are **identical**. MMD ≈ 0. The pair certifies as equivalent, while their per-image behavior is completely different.

**That is a false merge by construction** — in a framework whose central claim is that it never produces false merges.

The corrected design below compares **conditional** distributions `P(output | input, parameters)` per probe, then aggregates conservatively across probes. This is not merely a fix: it is what the deterministic path already does (`sec/6_methodology.tex:77` takes the max over the grid), so the corrected version is *more* consistent with the existing method than the pooled version was.

---

## 1. Conditional distributional signature

For a fixed probe `x_i` and parameter setting `θ`, the skill induces a **conditional output distribution**. Estimate it with `K` independent executions:

```
S(s, x_i, θ) = { φ( f_s(x_i, θ, ω_k) ) : k = 1…K }
```

where `ω_k` are independent draws of execution randomness (seed, sampler noise, non-deterministic kernels).

The full behavioral signature is the **indexed family**, not a pooled bag:

```
σ_dist(s, θ) = ( S(s, x₁, θ), …, S(s, x_m, θ) )
```

The index `i` is preserved throughout. **Never pool across `i`.**

---

## 2. Per-probe discrepancy

For each probe `x_i` and each grid point `θ`, compute a discrepancy between the two conditional sample sets:

```
D_i(θ) = MMD²_u ( S(s_a, x_i, θ),  S(s_b, x_i, θ) )
```

with the unbiased estimator

```
            1                       1                        2
MMD²_u =  ───────  Σ  k(pᵢ,pⱼ) + ───────  Σ  k(qᵢ,qⱼ)  −  ────  Σ  k(pᵢ,qⱼ)
          K(K−1)   i≠j            K(K−1)   i≠j             K²    i,j
```

### 2.1 Kernel bandwidth must be frozen, not per-pair
A median heuristic computed **per pair** makes distances pair-adaptive and therefore **not comparable against a common threshold** — which silently breaks any global `τ`.

**Fit the bandwidth once per modality on the calibration split, then freeze it** alongside the thresholds, with the same provenance stamp the paper already uses (`sec/appendix.tex` §Threshold Calibration). Report the frozen value.

---

## 3. Aggregation across probes — conservative, matching the deterministic path

Aggregate the per-probe discrepancies with the **same worst-case rule the deterministic cascade already uses**:

```
D(s_a, s_b) = max        D_i(θ)
              θ ∈ Θ_grid
              i ∈ 1…m
```

The paper's existing justification carries over verbatim (`sec/6_methodology.tex:81`): worst-case aggregation is what blocks spurious merges between skills that coincide on part of their domain.

**Predeclare the aggregation rule.** If a high quantile (p95) is used instead of a strict max for robustness to a single outlier probe, declare it in advance and report both. Choosing between max and quantile after seeing results is a researcher degree of freedom.

**Multiplicity:** aggregating `m × |Θ|` tests inflates the family-wise error rate. Apply a predeclared correction (Bonferroni or Holm over probes) when combining per-probe decisions, or use the max-statistic's own null distribution obtained by permutation (§4.2), which handles multiplicity natively and is preferred.

---

## 4. Inference — valid, not naive

### 4.1 Why a plain bootstrap CI is not enough
The unbiased MMD² estimator can be **negative**, and under the null `P = Q` its asymptotic distribution is **degenerate** — an infinite weighted sum of chi-squared variables, not Gaussian. A BCa bootstrap interval will therefore have **wrong coverage exactly in the regime that matters** (near equality, which is precisely where equivalence decisions are made).

Placing a BCa interval around an MMD statistic and calling it TOST is not a valid procedure.

### 4.2 Use a permutation null
For each `(x_i, θ)`:
1. Pool the `2K` feature vectors from both skills.
2. Randomly re-split into two groups of `K`, recompute `MMD²_u`. Repeat `P = 1000` times.
3. This yields the null distribution and a valid p-value.
4. For the aggregated statistic, permute **within probe** and take the max across probes per permutation — this gives the null of the max statistic directly and handles multiplicity without a separate correction.

For the equivalence direction, use a **wild/multiplier bootstrap** or an established kernel equivalence-testing procedure to obtain an upper confidence bound on `D`.

### 4.3 Validate the inference before trusting it
**Run a simulation study** before applying this to real operators:
- generate paired samples with known equal and known unequal distributions at realistic `K` and feature dimension
- verify empirical type-I error ≈ α and that the equivalence bound achieves nominal coverage

Report the simulation. Inference on high-dimensional features at small `K` is where this design is most likely to be quietly wrong, and a reviewer with statistics expertise will ask.

### 4.4 Decision rule
Let `[D_lo, D_hi]` be the validated `(1−2α)` bound on the aggregated discrepancy, `α = 0.05`.

| Condition | Verdict |
|---|---|
| `D_hi < τ` | **certify equivalent** — the whole plausible range is inside tolerance |
| `D_lo > τ + δ` | **certify `DISTINCT`** |
| otherwise | **`UNCERTAIN`** → human review |

**Why this fits the paper's design.** Certifying equivalence requires the *upper* bound below `τ` — the conservative direction, matching the existing calibration rule. And at low `K` the bound is wide, so nearly everything lands in `UNCERTAIN`: **the method abstains when it lacks evidence rather than guessing.** That safety property falls out of the statistics for free and is worth stating explicitly in the paper.

---

## 5. `K = 1`: an explicit branch, not an algebraic identity

The original claimed `K = 1` recovers the deterministic method exactly. **It does not.** The unbiased estimator has `K(K−1)` denominators and is undefined at `K = 1`, and MMD in DINO feature space is not the existing `L∞`/LPIPS/SSIM cascade.

**Implement an explicit branch:**

```
if skill is deterministic (verified by repeated-execution check):
    → existing deterministic cascade, unchanged
else:
    → conditional distributional path (§1–4)
```

Determinism is **tested**, not assumed: execute twice at fixed seed and compare. Record the verdict per skill. Do not claim the two paths are the same procedure; claim only that the deterministic path is used wherever it applies, leaving all existing results unchanged.

---

## 6. Modality-specific comparators

| Output | Canonicalization `φ` | Per-probe distance |
|---|---|---|
| **Image** | existing canonicalization | LPIPS / DINO features (existing) |
| **Binary mask** | resize to common extent, binarize at 0.5 | `1 − IoU`; boundary F-measure as second view |
| **Instance masks** | Hungarian matching between instance sets | mean matched `1 − IoU`; penalize unmatched |
| **Boxes + labels** | canonical sort | greedy IoU matching at 0.5; agreement over matched + unmatched |
| **Depth map** | **per-image affine alignment** (least-squares scale + shift) | AbsRel or `δ₁` |
| **Generative image** | DINO/CLIP features | **feature-space MMD only — never pixel `L∞`** |

### Two comparators that matter more than they look

**Depth requires affine alignment.** Monocular depth is defined only up to scale and shift; comparing raw maps measures output conventions, not behavior. **However** — per the critique — alignment can also erase calibration behavior that matters to a user. So report **both**: affine-invariant error as the *behavioral* metric, and native resolution plus calibration convention as **separate interface-compatibility attributes**. A pair may be behaviorally equivalent and interface-incompatible; the curator needs to know both.

**Generative outputs must never use pixel distance.** Two samples from the *same* diffusion model at different seeds have enormous pixel distance. Only feature-space conditional comparison is meaningful. The self-pair sanity gate catches this if it is got wrong.

---

## 7. Calibration for the distributional path

Existing thresholds are calibrated for classical operators and **do not transfer** to MMD in feature space or IoU on masks — different metrics, different scales.

Calibrate a separate operating point **per modality**, using the existing protocol:
- cluster-disjoint by **model family**, not merely by operator
- frozen and hashed before any test metric
- provenance stamp (split hash + date)
- **frozen kernel bandwidth recorded alongside the threshold** (§2.1)

### Honest limitation on the guarantee
The classical track calibrates precision on non-equivalence to ≥ 0.99. **A3 cannot support a claim at that confidence.** With a realistic number of independent model families and designed pairs, the achievable confidence interval is wide.

**State this plainly rather than implying the guarantee transfers.** A3 is scoped as a **feasibility study**: it demonstrates that the framework extends with a principled decision rule and reports measured behavior on a limited operator set. Overclaiming a 0.99 guarantee from ~30 pairs is exactly the kind of thing a confident reviewer will check and reject.

---

## 8. Sample-size analysis

Sweep `K ∈ {1, 2, 4, 8, 16, 32}` and report:

| Metric vs. `K` | What it shows |
|---|---|
| Certificate-flip rate vs. the high-`K` reference | decision stability |
| Bound width | precision of the estimate |
| **Abstention rate** | should be high at low `K` — the rule correctly refuses to guess |
| Wall-clock / GPU-hours | **→ hand to A4** |

**The high-`K` run is a reference, not truth.** Repeat it with independent seed batches and quantify its own uncertainty. For high-dimensional generative features, `K = 8–16` is likely underpowered — **allow "not practically certifiable at feasible `K`" as a reportable outcome.** That is a legitimate scientific result, not a failure.

---

## 9. Where the extension goes in the paper

| Section | Edit |
|---|---|
| `sec/6_methodology.tex` §Equivalence Taxonomy | New subsection: conditional distributional signature + decision rule. **Do not claim `K=1` identity** — describe the explicit branch. |
| Algorithm 1 | Extend the cascade with bound-based conditions |
| Table 4 | Companion per-modality table: thresholds **and frozen bandwidths** |
| `sec/appendix.tex` §Behavioral Signatures | Extend classes (a)/(b)/(c) with the conditional distributional case; add the inference-validation simulation |
| `sec/1_intro.tex:13`, `sec/0_abstract.tex`, `sec/8_discussion.tex:9` | **Rewrite all three scope statements together**, scoped as feasibility |
