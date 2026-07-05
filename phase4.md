# Output-Grounded Equivalence Verification for Skill-Library Curation

## Summary

Curating a large library of image-processing operators ("skills") requires deciding, for
every pair of operators, whether they are *redundant* — and if so, in what sense. The
conventional approach compares operators by their **text surface** (names, descriptions,
tags) and merges those that read alike. We argue this is unsafe: text similarity is neither
necessary nor sufficient for behavioral equivalence. Two operators can share almost identical
prose yet produce different pixels, and two operators with unrelated descriptions can compute
the identical function.

We evaluate an **output-grounded equivalence verifier** that decides equivalence by *executing*
both operators on a shared battery of probe images and comparing their outputs with calibrated
perceptual and semantic metrics, organized into a hierarchical six-relation taxonomy with an
explicit abstention class. Against three text-only baselines — token name-matching, TF–IDF
description-cosine, and a strong large-language-model judge that reads descriptions — the
output-grounded verifier is the only method that never licenses a false merge on the
engineered hard-negative slice (0/6), while the text baselines over-merge between 17% and 83%
of those pairs. The verifier keeps every genuinely distinct pair separate (0/926 false merges)
and abstains on ≈1% of decisions via its calibrated uncertainty band.

The reference graph, evaluation protocol, thresholds (calibrated on a cluster-disjoint split),
and all reported numbers are drawn from a single reproducible run over **100 skills / 944
scored pairs / a 177-image probe battery**, seed `1234`.

---

## 1. Problem statement

Let a library contain \(N\) skills, each a deterministic (up to seed) image-to-image operator
with optional parameters. Curation asks a pairwise question for every ordered pair \((A, B)\):
are \(A\) and \(B\) redundant, and what curation action does that redundancy license (merge,
parameterize/unify, or keep separate)? A curation policy that merges too aggressively silently
destroys distinct capabilities; one that merges too timidly leaves the library bloated and
duplicative. The costly error is the **false merge** — collapsing two operators that behave
differently — so the operating point of any equivalence oracle must protect *precision on
non-equivalence* before it maximizes compression.

Text-based deduplication (embedding the description and merging on cosine similarity, or an
LLM asked to compare two descriptions) is the standard, cheap baseline. Its failure mode is
structural: it can only observe what the author *wrote*, not what the operator *does*. This
work replaces the text surface with the operator's **observable behavior** as the ground for
the equivalence decision.

---

## 2. The output-grounded verifier

### 2.1 Execution on a shared probe battery

Both operators are run on a fixed battery of probe images (here, 177 probes spanning
synthetic gradients, textures, edges, and natural-image crops). Comparison is always made on
the **common** set of probes both operators produced, on canonicalized outputs, under a fixed
seed. Because the operators are executed rather than described, the verdict is a *fact about
outputs* rather than a guess about intent.

### 2.2 A six-relation taxonomy with abstention

The verifier classifies each pair into exactly one relation, plus an abstention class:

| Relation | Meaning | Licensed action |
|---|---|---|
| **EXACT** | Identical output on every probe (hash-identical, or worst-case \(L_\infty \le \varepsilon\)) | merge |
| **PERCEPTUAL** | Visually indistinguishable (worst-case LPIPS below threshold, SSIM floor satisfied) | merge / parameterize |
| **SUBSUMPTION** | One operator is a special case of the other (directional; every binding of the specific is reproduced by the general) | parameterize / keep separate |
| **SEMANTIC_PRESERVING** | Same kind of transformation, different algorithm (semantic-feature distance below threshold) | parameterize / keep separate |
| **COMPLEMENTARY** | Orthogonal operations that compose (commute within tolerance) | keep separate |
| **DISTINCT** | Fails every relation above (residual class) | keep separate |
| **UNCERTAIN** | Deciding distance falls inside the calibrated band around a threshold | route to human review |

Only **EXACT ∪ PERCEPTUAL** license an actual *merge*; this binary "mergeable" decision is the
axis on which every method (output-grounded and text) is directly comparable.

### 2.3 Classification is hierarchical and stop-at-first

Relations are checked cheapest-and-strictest first (EXACT → PERCEPTUAL → SUBSUMPTION →
SEMANTIC_PRESERVING → COMPLEMENTARY → DISTINCT) and the first that holds wins. Two design
choices matter for correctness:

- **Aggregation semantics differ by relation.** EXACT and PERCEPTUAL are *universally
  quantified* over the parameter sweep × probes, so they aggregate by **worst case (max)**: a
  single probe on which the outputs diverge disqualifies the merge. SEMANTIC_PRESERVING is a
  distributional claim and aggregates by a **high quantile (p90) plus the mean**, tolerating a
  few outliers.
- **Subsumption is checked before the semantic relation**, so a genuine directional
  special-case relationship (which licenses parameterization) is not masked by a looser
  semantic match. For parameterized operators that share no matched sweep axis, subsumption is
  established by a grid search testing mutual reproduction across parameter grids.

### 2.4 Distance metrics

- **Perceptual.** LPIPS (AlexNet backbone) as the primary perceptual distance, guarded by an
  SSIM floor. The floor closes an LPIPS blind spot: a pair below the LPIPS band but failing the
  SSIM floor is *not* accepted as PERCEPTUAL and falls through to the next relation.
- **Semantic.** Cosine distance in a self-supervised **DINO ViT-B/16** feature space. An
  optional **CLIP ViT-B/32** view is combined **conservatively** — the semantic distance is
  taken as the *maximum* of the DINO and CLIP distances, so adding CLIP can only make the
  verifier *more* cautious about declaring a semantic match, never less.
- **Numerical exactness.** \(\varepsilon\) is a pixel-rounding tolerance (\(1/255\approx
  0.0039\)), not a learned operating point.

### 2.5 Calibrated thresholds and an abstention band

Thresholds are **calibrated, not hand-set**. On a labeled validation split, each threshold is
chosen as the most permissive value (predict "closer" iff distance \(\le t\)) that keeps
**precision on non-equivalence \(\ge 0.99\)** subject to a recall floor of \(0.5\) — i.e.
"compress as much as possible without licensing a false merge." An abstention band
\([\tau(1-\delta),\,\tau(1+\delta)\)] around each perceptual/semantic threshold returns
UNCERTAIN for band-straddling pairs; \(\delta\) is the smallest band on a grid that lifts
decisive-pair precision to the target. The split is **cluster-disjoint** by operator family (a
family lies wholly in calibration or wholly in test; cross-cluster pairs are dropped), so
calibration cannot leak into the test metrics. Every calibrated configuration carries a
provenance stamp (split hash + date) so a reported metric can never silently use uncalibrated
thresholds.

Calibrated operating point used in this run:

| Threshold | Value | Role |
|---|---|---|
| \(\varepsilon\) (exact \(L_\infty\)) | \(1/255 \approx 0.0039\) | numerical-equality tolerance |
| \(\tau_\text{perc}\) (LPIPS) | 0.05 | worst-case perceptual merge bound |
| SSIM floor (\(1-\)SSIM) | 0.10 | LPIPS blind-spot guard |
| \(\tau_\text{sem}\) (DINO, p90) | 0.15 | semantic-preserving bound |
| complementary (LPIPS) | 0.05 | commute tolerance |
| abstention \(\delta\) | 0.05 | half-width of the UNCERTAIN band |

---

## 3. Baselines

All three baselines read only the **text surface** of a skill (name, description, family tag)
and emit a binary mergeable decision plus a similarity score.

1. **name-match** — Jaccard token overlap over the union of name and id tokens; merge iff
   overlap \(\ge 0.5\). A renamed-duplicate detector.
2. **embedding-cosine** — cosine similarity over deterministic **TF–IDF** embeddings
   (smoothed IDF, L2-normalized) of the concatenated text; merge iff cosine \(\ge 0.6\). This
   is the classic "description-embedding dedup" strawman.
3. **llm-on-descriptions** — a large-language-model judge (a fixed, strong instruction model,
   queried through a hosted API with reasoning disabled for one-word determinism) shown both
   descriptions and asked to return exactly one relation word. Unparseable replies are treated
   conservatively as DISTINCT. This judge is deliberately *distinct* from any model used
   elsewhere as a curation subject, so the baseline and any downstream agent experiment are
   never confounded.

---

## 4. Experimental setup

- **Library / pairs.** 100 skills; **944 designed ordered pairs** scored against a designed
  reference relation graph \(G\); seed `1234`.
- **Answer key.** \(G\) is hand-authored and validated (subsumption is a DAG; EXACT is
  transitive; PERCEPTUAL/SEMANTIC/COMPLEMENTARY symmetric but not transitive; each pair carries
  at most one designed relation). It is **fixed before any metric runs** and is never derived
  from the metrics under test. Any pair not listed is DISTINCT by default. A small set of
  **engineered hard negatives** are DISTINCT pairs *flagged* so their slice can be reported
  separately — pairs constructed to *sound* alike but *behave* differently.
- **Probe battery.** 177 probes.
- **Backends.** perceptual `lpips-alex`; semantic `vit_base_patch16_224.dino`; second semantic
  view `clip-ViT-B-32-quickgelu (openai)`; computed on GPU.
- **Calibration split.** Thresholds calibrated on **713** cluster-disjoint pairs drawn from 8
  operator families (blur, color, denoise, edges, frequency, geometric, mask, morphology), with
  **100** held-out test pairs from disjoint families confirming no family leakage. The
  comparative safety/divergence analysis below is reported over the full 944-pair graph.

### 4.1 Metrics

- **Per-relation precision / recall / F1** (one-vs-rest) — how well a track recovers the fine
  taxonomy; only the output-grounded track can resolve all six relations, text judges resolve
  at most merge-vs-distinct.
- **Safety on non-equivalence** — the **false-merge rate** on truly-DISTINCT pairs, and
  separately on the engineered hard-negative slice. This is the cost-asymmetric number the
  curator most cares about.
- **Divergence** — the rate at which a text track and the output track disagree on the
  mergeable decision, broken down by true relation, with the hard-negative slice reported
  separately. *Over-merge* = text says merge, output does not (the silent-merge danger);
  *under-merge* = output finds redundancy the text track misses.

---

## 5. Results

### 5.1 Safety: false merges on non-equivalent pairs

This is the headline safety result. The engineered hard negatives are exactly the pairs
designed to fool a text judge; the output-grounded verifier is the only method that resists all
of them.

| Method | False-merge on DISTINCT | False-merge on hard-negatives | Abstention |
|---|---|---|---|
| **Output-grounded** | **0 / 926 (0.0%)** | **0 / 6 (0.0%)** | 0.01 |
| name-match | 44 / 926 (4.8%) | 5 / 6 (**83.3%**) | 0.00 |
| embedding-cosine (TF–IDF) | 1 / 926 (0.1%) | 0 / 6 (0.0%) | 0.00 |
| llm-on-descriptions | 1 / 926 (0.1%) | 1 / 6 (**16.7%**) | 0.00 |

### 5.2 Divergence from the output-grounded verifier

Disagreement on the mergeable decision, per true relation. Every text disagreement in this run
is an **over-merge** (text merges where the output verifier does not); no baseline found
redundancy the output verifier missed.

**vs name-match** — total divergence **51 / 944 (5.4%)**

| True relation | n | disagree | rate | over-merge |
|---|---:|---:|---:|---:|
| SUBSUMPTION | 7 | 1 | 0.14 | 1 |
| SEMANTIC_PRESERVING | 10 | 5 | 0.50 | 5 |
| COMPLEMENTARY | 1 | 1 | 1.00 | 1 |
| DISTINCT | 926 | 44 | 0.05 | 44 |
| *hard-negative* | 6 | 5 | 0.83 | 5 |
| **ALL** | 944 | 51 | 0.05 | 51 |

**vs embedding-cosine** — total divergence **2 / 944 (0.2%)**

| True relation | n | disagree | rate | over-merge |
|---|---:|---:|---:|---:|
| SEMANTIC_PRESERVING | 10 | 1 | 0.10 | 1 |
| DISTINCT | 926 | 1 | 0.00 | 1 |
| **ALL** | 944 | 2 | 0.00 | 2 |

**vs llm-on-descriptions** — total divergence **1 / 944 (0.1%)**

| True relation | n | disagree | rate | over-merge |
|---|---:|---:|---:|---:|
| DISTINCT | 926 | 1 | 0.00 | 1 |
| *hard-negative* | 6 | 1 | 0.17 | 1 |
| **ALL** | 944 | 1 | 0.00 | 1 |

### 5.3 Output-grounded per-relation quality

| Relation | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUBSUMPTION | 0.667 | 0.857 | 0.750 | 7 |
| SEMANTIC_PRESERVING | 0.037 | 0.400 | 0.067 | 10 |
| COMPLEMENTARY | 0.000 | 0.000 | 0.000 | 1 |
| DISTINCT | 0.988 | 0.605 | 0.750 | 926 |

(EXACT and PERCEPTUAL have zero support in this reference graph — see §6.)

---

## 6. Analysis

**The text/output divergence is real and concentrates exactly where the argument predicts.**
The whole premise is that behavioral equivalence and textual similarity come apart on
adversarially-constructed pairs. On the hard-negative slice — pairs written to *sound*
identical — name-match over-merges 5 of 6 and even the strong LLM judge over-merges 1 of 6,
whereas the output-grounded verifier over-merges none. This is the load-bearing result: a
description-only oracle, however sophisticated, inherits the ambiguity of the description.

**The baselines form a clear quality ladder.** name-match is brittle (5.4% total divergence,
driven by shared tokens: it collapses half the SEMANTIC_PRESERVING pairs and the lone
COMPLEMENTARY pair). TF–IDF cosine is far tighter (2 disagreements total). The LLM judge is the
strongest text method (1 disagreement total) — but its single remaining error is precisely a
hard negative, the case the output grounding exists to catch. Increasing text-judge
sophistication shrinks the divergence but does not remove the structural blind spot.

**The verifier's merge decision is safe; its fine-grained semantic label is noisy.** The
safety-critical *merge* decision is clean (0 false merges on both DISTINCT and hard negatives).
The fine SEMANTIC_PRESERVING label, by contrast, has low precision (0.037): with a permissive
DINO threshold, many truly-DISTINCT pairs fall under the semantic band and are labeled
SEMANTIC_PRESERVING. This inflates the "DISTINCT recall" gap (0.605) — the missing 39.5% of
DISTINCT pairs are not *merged*, they are over-labeled as SEMANTIC or routed to UNCERTAIN.
Because SEMANTIC_PRESERVING licenses only *parameterize*, never *merge*, this is a cost-benign
error: it can over-suggest unification for human review, but it cannot silently destroy a
capability. SUBSUMPTION, the other actionable relation, is recovered well (F1 0.750).

---

## 7. Limitations and honesty notes

- **Merge-positive precision/recall are not estimable from this reference graph.** The current
  \(G\) contains no designed EXACT or PERCEPTUAL pairs (clean-library EXACT is empty by
  construction), so the positive class of the mergeable decision has zero support and its
  precision/recall/F1 are degenerate (0.000) for *every* method, including the output verifier.
  The comparison here is therefore correctly framed around **non-equivalence safety** and
  **divergence**, not around merge-recall. Estimating merge-positive recall requires a reference
  graph seeded with true equivalences.
- **Label certification is pending.** The SEMANTIC_PRESERVING and SUBSUMPTION labels in \(G\)
  are design intent pending human re-certification, and inter-annotator agreement (\(\kappa\))
  is not yet computed — the judgment-laden slice is exported for annotation and \(\kappa\) is
  never fabricated.
- **Small support on some relations.** SUBSUMPTION (n=7), SEMANTIC_PRESERVING (n=10), and
  COMPLEMENTARY (n=1) have small support; per-relation F1 on these should be read as indicative,
  not precise. The DISTINCT and hard-negative conclusions rest on much larger n.
- **Backbone dependence.** Perceptual/semantic verdicts inherit the biases of the LPIPS, DINO,
  and CLIP backbones. The conservative max-combination of DINO and CLIP is a mitigation, not a
  proof of backbone-independence.

---

## 8. Reproducibility

All numbers above come from one deterministic run (seed `1234`): 100 skills, 944 scored pairs,
177-probe battery, backends `lpips-alex` / `vit_base_patch16_224.dino` /
`clip-ViT-B-32-quickgelu-openai`, thresholds calibrated on a 713-pair cluster-disjoint split
(100 held-out test pairs) with the provenance stamp recorded in the emitted configuration. The
run emits the divergence report, per-pair distance table, calibrated-threshold configuration,
a divergence figure, and a human-review template for the label-certification pass.
