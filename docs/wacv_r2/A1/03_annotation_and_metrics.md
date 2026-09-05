# A1 · Annotation Protocol, Metrics, and Deliverables

Covers both corpora. The two need **different** ground-truth strategies, and that distinction must be explicit in the paper — a reviewer who thinks we claimed an answer key for real libraries will reject on that alone.

---

## 1. Ground truth: two different problems

| | Corpus A | Corpus B |
|---|---|---|
| External labels exist? | **Yes** — maintainer equivalence claims | No |
| Role of those labels | **The hypothesis under test**, not ground truth | n/a |
| Annotation scope | **All** disagreements (tens) | **Stratified sample** (300–500) |
| Annotators | In-house acceptable if guideline + raw outputs published | External recommended |

### Corpus A — external judgments as the object of study
The maintainer labels are **not** ground truth. They are category-level hypotheses being tested (see [`01_corpus_A_cross_library.md`](01_corpus_A_cross_library.md) §1.1). Where VisCurate and the label disagree, a human adjudicates.

- 3 annotators, **blind to the verifier verdict**
- shown only: the two functions' outputs on a sample of probes **at the divergent parameter setting**, plus both docstrings
- adjudicate to one of:
  - **label supported** → VisCurate false-splits (a verifier error)
  - **label not supported at aligned parameters** → the finding
  - **genuinely ambiguous** → report separately, do not force

**Adjudicate every disagreement** (there should be only tens), **and additionally sample the agreements.**

> ⚠ **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §12. The original annotated *only* disagreements. That yields the disagreement rate but **cannot estimate verifier accuracy, false-split rate, or relation recall** — those require probability sampling of agreements too, with known inclusion probabilities. Sample agreements at a recorded rate and weight by inverse inclusion probability.

### Corpus B — stratified sampling with inverse-propensity weighting
No external labels, so sample deliberately:

| Stratum | Sampling | Why |
|---|---|---|
| Certified `EXACT` / `PERCEPTUAL` | **all** (cap 150) | these license destructive merges — precision here is what a curator cares about |
| Certified `SUBSUMPTION` | **all** (cap 60) | licenses parameterization |
| `UNCERTAIN` | **all** (cap 60) | validates the abstention band actually catches borderline cases |
| `DISTINCT` | random sample (~100) | estimates false-split rate |
| **Fingerprint-pruned before verification** | **powered design — see below** | **the only way to estimate candidate-generation recall** — A4 needs this too |

Target **300–500 annotated pairs** total.

> The pruned-pairs stratum is easy to forget and impossible to reconstruct later. If candidate generation silently drops true duplicates, every downstream recall number is inflated and unfixable without a re-run.

**Cluster uncertainty by pack and by skill, not by pair.** Pairs sharing a skill are dependent; treating hundreds of thousands of pairs as independent understates variance badly. Bootstrap at the pack/skill level.

### ⚠ The pruned-pair audit needs power, not a convenience sample

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §13.

A random sample of ~80 pruned pairs from a universe of hundreds of thousands cannot demonstrate high candidate recall — it would contain almost no positives, so the estimate carries essentially no information. Use a three-part design:

1. **Injected known positives.** Plant duplicate pairs with known ground truth into the corpus and measure what fraction survives pruning — a direct recall estimate at useful precision.
2. **Exhaustive evaluation within manageable strata.** Pick a few packs or operator families small enough to evaluate *all* pairs unpruned, and compute exact recall there.
3. **Powered random audit** over the remainder, sample size chosen from a stated target precision.

**Report an interval, not a point estimate.**

### Candidate recall must also be measured on *real* positives

Candidate generation *always* includes same-family pairs and the engineered hard negatives (`sec/appendix.tex:43`). So recall measured on `G_0`/`G_ρ` — where the positives largely **are** same-family or engineered — is **partly guaranteed by construction** and overstates real performance.

Report recall separately on **real adjudicated positives** from Corpus A and Corpus B, broken down by source and family. That is the number A4's scalability argument actually needs.

---

## 2. Annotation instrument

### 2.1 What the annotator sees
- Side-by-side outputs on **8–12 probes**, chosen to span the parameter grid and include the max-divergence point
- A difference map (absolute difference, normalized) per probe
- Both function names and docstrings
- **Never** the verifier's verdict, the distances, or the other annotators' labels

### 2.2 Label set
Use the paper's own six relations plus an escape hatch:

`EXACT` · `PERCEPTUAL` · `SUBSUMPTION` (with direction) · `SEMANTIC-PRESERVING` · `COMPLEMENTARY` · `DISTINCT` · `CANNOT-TELL`

`CANNOT-TELL` is essential. Forcing annotators to guess manufactures agreement and corrupts α.

### 2.3 Guideline
Write it before annotation starts, publish it in the appendix. It must define, with worked examples:
- how much visual difference separates `PERCEPTUAL` from `SEMANTIC-PRESERVING`
- how to judge `SUBSUMPTION` direction
- that "would a library maintainer be comfortable deleting one of these?" is the operational test for merge relations
- that spatial-extent differences block merge relations

### 2.4 Calibration round
Run **20 pilot pairs** with all annotators, compute α, discuss disagreements, revise the guideline, **then** start the real batch. Report both the pilot and final α.

---

## 3. Agreement and adjudication

- **Krippendorff's α** on the nominal label set — report it. Below ~0.67 the guideline needs revision before the results are usable.
- Disagreements resolved by discussion to consensus; if no consensus → `CANNOT-TELL`, reported separately.
- Report the **`CANNOT-TELL` rate** as its own number. A high rate is informative: it means real-library equivalence is genuinely ambiguous, which supports the paper's abstention design.

---

## 4. Metrics

### 4.1 Primary — safety (matches the paper's existing headline)
Keep the cost-asymmetric framing already established in `sec/6_methodology.tex` §Metrics.

| Metric | Definition |
|---|---|
| **FM (false-merge rate)** | of pairs certified `EXACT`/`PERCEPTUAL`, fraction human-adjudicated as not mergeable |
| **Abstention rate** | fraction routed to `UNCERTAIN` |
| **False-split rate** | of pairs certified `DISTINCT`, fraction human-adjudicated as mergeable |

FM is the headline. It is directly comparable to the `0/926` and `0/6` in Table 2 — now on real data.

### 4.2 Per-relation quality
Precision / recall / F1 one-vs-rest on the adjudicated set, **with support counts**. Recall on Corpus B is IPS-weighted from the stratified sample; report **confidence intervals**, and be explicit that precision is direct while recall is estimated.

### 4.3 Corpus A only — claim-vs-execution
The matrix from [`01_corpus_A_cross_library.md`](01_corpus_A_cross_library.md) §6, plus:

| Metric | Definition |
|---|---|
| **Over-claim rate** | of maintainer "direct equivalent" pairs, fraction executing as `DISTINCT` |
| **Missed-consolidation rate** | of maintainer `–` pairs, fraction executing as `EXACT`/`PERCEPTUAL` |
| **Adjudicated over-claim rate** | over-claims confirmed by human adjudication (the defensible number) |

### 4.4 Corpus B only — decay characterization
| Metric | Definition |
|---|---|
| **Redundancy rate** | % of nodes certified redundant with ≥1 other |
| **Cross-pack vs. within-pack split** | independent reimplementation vs. within-maintainer sloppiness |
| **Compression potential** | library size reduction if all certified merges applied |
| **Defect-type correspondence** | observed real defects mapped onto the 7 synthetic types + **any uncovered type** |
| **Type-contract violation rate** | nodes declaring `IMAGE` but returning something else |

### 4.5 Baseline comparison
Run the full A2 ladder on both corpora. Compare **at matched precision on non-equivalence**, not at each baseline's arbitrary default threshold — see [`../A2/01_baseline_ladder.md`](../A2/01_baseline_ladder.md) §4.

---

## 5. Complete deliverable list

| ID | Artifact | Corpus | Section |
|---|---|---|---|
| **T1** | Corpus statistics + exclusion funnel with reasons | B | §5.x |
| **T2** | Claim-vs-execution matrix | A | §5.x |
| **T3** | Full baseline ladder on real pairs | A (+B) | §5.x |
| **T4** | Verifier precision per relation on adjudicated pairs + FM rate | A + B | §5.x |
| **T5** | Redundancy census | B | §5.x |
| **T6** | Defect-type correspondence: synthetic taxonomy vs. observed reality | B | §5.x |
| **F1** | **Qualitative: 4–6 real over-claimed pairs**, outputs side by side, divergence annotated. Lead with CLAHE. | A | §5.x |
| **F2** | Redundancy distribution across packs / families | B | §5.x |
| **A-1** | Parameter alignment maps (YAML) | A | appendix |
| **A-2** | Enumerated over-claim list with divergence parameters | A | appendix |
| **A-3** | Frozen pack list with commit SHAs | B | appendix |
| **A-4** | Full exclusion log | B | appendix |
| **A-5** | Annotation guideline + pilot/final α | both | appendix |

---

## 6. Writing the section

**Structure for §5.x "Audit of real, uncurated libraries":**

1. **Motivation** (1 para) — the reviewers' point, conceded plainly and without defensiveness
2. **Corpus A setup** (1 para) — emphasize the external authorship of the equivalence claims; this is the argument, not a detail
3. **Corpus A results** — T2, T3, F1
4. **Corpus B setup** (1 para) — sampling frame + inclusion filter, stated precisely
5. **Corpus B results** — T1, T5, T6, F2
6. **Adjudication** — T4, α, `CANNOT-TELL` rate
7. **What this does and does not establish** (1 para) — bound the claim explicitly

### Framing notes
- **Concede first.** Open by agreeing the synthetic-only evaluation was a real limitation. Reviewers reward this; defensiveness costs credibility.
- **Lead with the external-oracle argument.** It is the strongest thing in the revision and answers R3's specific objection directly.
- **Do not bury the exclusion rate.** Putting it up front reads as confidence.
- **If the numbers are unflattering, say so in the same paragraph as the good ones.** A generalization gap reported by the authors is a finding; one discovered by a reviewer is a rejection.

---

## 7. Checklist

- [ ] Write annotation guideline; publish in appendix
- [ ] **Recruit annotators in W0** — the most common slip point in this plan
- [ ] Run 20-pair pilot; compute α; revise guideline; report both α values
- [ ] Corpus A: adjudicate **every** disagreement
- [ ] Corpus B: build all five strata, **including fingerprint-pruned pairs**
- [ ] Run annotation batch, 3 annotators, blind to verifier
- [ ] Compute Krippendorff's α; adjudicate to consensus
- [ ] Report `CANNOT-TELL` rate separately
- [ ] Compute all metrics with raw counts and CIs
- [ ] Run baseline ladder at **matched precision**
- [ ] Produce T1–T6, F1–F2, A-1–A-5
- [ ] **Check whether `sec/7_results.tex:39` ("no baseline recovers a redundancy the verifier misses") still holds** — softening it may be required
