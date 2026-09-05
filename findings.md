# VisCurate — WACV-2027 Round-2 findings

**Compiled:** 2026-08-23 · **Code:** `TimeWarp/VisCurate-` · **Artifacts:** `results/wacv_r2/`
**Plans executed:** `visCurate-experiments/RUNBOOK.md` (Stages 0–5) → `NEXT_ACTIONS.md` → `SPRINT_3DAY.md`
**Target:** `overleaf/viscurate/reviewer_concerns.md`

Numbers are reported exactly as produced. Where a result does not exist yet it is marked
**`[PENDING]`** with the artifact path it will land in — never estimated, never inferred. Where a
result is blocked, the blocker is named. `[PENDING]` and *blocked* and *deliberately not done* are
kept as three separate categories throughout.

---

## 0. Scorecard against the metareview's four required revisions

| # | Required revision | Evidence | Status |
|---|---|---|---|
| 1 | Evaluate on ≥1 real, uncurated open-source library | §3 fork pair (`pilgram`/`pilgram2`), §4 Corpus A | ✅ **done** |
| 2a | Add a text/embedding baseline | §2 A2 ladder, all 8 rungs on the same benchmark | ✅ **done** |
| 2b | Verifier self-validation | §1 `G_ρ` merge recall | ✅ **done at n=300** — recall 0.5124, **0 false merges**, 0 errors |
| 3 | Distributional/soft similarity for neural & stochastic tools | §9.1 T7 | ✅ **done** — gates 6/6 self, 2/2 hard neg |
| 4 | Scalability / cost analysis | §5 A4.1–A4.6 | ✅ **done** |

Reviewer-named weaknesses beyond the four:

| Source | Weakness | Track | Status |
|---|---|---|---|
| R3 (minor) | "no ablation showing how performance changes with the number or diversity of probes" | §9.2 T5 | ✅ **done** — diversity, not count, drives safety |
| Metareview + R2 | "two skills matching on a finite probe battery may still diverge out-of-distribution" | §9.3 T6 | ✅ **done** — flip rate 0.0135, upper 0.0625 |
| Metareview | Corpus B — literal "uncurated open-source" (ComfyUI) | T8 | 🚫 blocked (§10) |

**All four required revisions now have numbers, and so do both extra reviewer-named weaknesses.**
Requirement 3 (T7) was the last gap; `SPRINT_3DAY.md` had marked it "first to cut" while the
metareview required it, and the metareview won. What remains is **human**, not computational
(§13).

---

## 1. ★ Verifier self-validation — merge recall (R3's "the verifier itself is not validated")

`G_0` designates **zero** EXACT and **zero** PERCEPTUAL pairs by construction, so the mergeable
positive class is empty there and merge recall has never been measurable. Every prior run printed
`0.000/0.000/0.000` on that row, which reads as *measured failure* rather than *absent data*.
`G_ρ` injects duplicates and is therefore the only set that can produce the number.

### 1.1 Result at n=34 instances (complete, `results/wacv_r2/grho_replay/report.md`)

34 instances (stratified, 4 per ρ, seed 1234) · **1,639 pairs** · **0 errors**

| quantity | value |
|---|---|
| mergeable support (EXACT ∪ PERCEPTUAL) | **436** |
| **merge recall** | **0.4931** — exact 95% CI [0.4452, 0.5411]; instance-bootstrap [0.4553, 0.5368] |
| precision (within labelled set) | **1.0000** |
| F1 | 0.6605 |
| **FDR on merges** | **0.0000** |
| tp / fp / fn / tn | 215 / **0** / 221 / 1203 |

**The verifier recovers 49.3% of genuinely mergeable pairs while making zero false merges.** That
is the safety/compression trade-off stated as a measurement for the first time; the paper could
previously assert the safety half and had no way to quantify its cost.

### 1.2 Per-relation — the cost is almost entirely ONE confusion

| relation | support | predicted | recall [95% CI] | precision | F1 |
|---|---:|---:|---|---|---|
| EXACT | 204 | 199 | **0.9755** [0.944, 0.992] | 1.0000 | 0.9876 |
| **PERCEPTUAL** | **232** | **16** | **0.0560** [0.030, 0.094] | 0.8125 | 0.1048 |
| SUBSUMPTION | 549 | 440 | 0.7978 [0.762, 0.831] | 0.9955 | 0.8857 |
| SEMANTIC_PRESERVING | 397 | 486 | 0.3149 [0.269, 0.363] | 0.2572 | 0.2831 |
| COMPLEMENTARY | 35 | 64 | 0.0000 [0.000, 0.100] | 0.0000 | — |
| DISTINCT | 222 | 434 | 0.3694 [0.306, 0.437] | 0.1889 | 0.2500 |
| UNCERTAIN | 0 | 0 | — | — | — |

**219 of 232 PERCEPTUAL pairs are called SEMANTIC_PRESERVING** — same semantics, different pixels,
one notch too conservative to license a merge. Of the 221 false negatives, **2 come from EXACT and
219 from PERCEPTUAL: 99.1% of all missed consolidation is a single confusion.** Byte-identical
duplicates are found essentially always (0.976); near-duplicates 5.6% of the time. Real libraries
accumulate *near*-duplicates, so the relation the verifier is worst at carries most of the
practical consolidation — and the failure is threshold placement at the calibrated operating point,
not a missing signal.

Two further facts the confusion matrix settles:

- **`UNCERTAIN` is never emitted** in 1,639 pairs. The advertised abstention is dead code at this
  operating point; the paper should say so or justify it.
- **COMPLEMENTARY: 0.000 recall on 35 designed positives while predicted 64 times** —
  independently reproducing the `G_0` finding (§8) on a second dataset. Two-dataset result now.

### 1.3 Safety does not decay with corruption rate

| ρ | pairs | mergeable support | merge recall | false merges |
|---|---:|---:|---|---:|
| 0.1 | 113 | 9 | 0.3333 | **0** |
| 0.2 | 95 | 13 | 0.4615 | **0** |
| 0.3 | 116 | 26 | 0.5769 | **0** |
| 0.4 | 172 | 38 | 0.5789 | **0** |
| 0.5 | 130 | 32 | 0.5312 | **0** |
| 0.6 | 143 | 34 | 0.4706 | **0** |
| 0.7 | 258 | 88 | 0.4318 | **0** |
| 0.8 | 165 | 48 | 0.5208 | **0** |
| 0.9 | 215 | 79 | 0.4684 | **0** |
| 1.0 | 232 | 69 | 0.5217 | **0** |

**Zero false merges at all ten levels** — the safety property does not degrade as the library gets
dirtier, which is the claim a reviewer would most want stress-tested. Recall has no monotone ρ
trend; the ρ=0.1 cell rests on 9 positives and its 0.3333 must not be read as one.

### 1.4 The answer key had to be repaired first

`g_rho.json` records the relations a defect *injects*, not the incidental ones cloning creates.
Audited over all 300 instances: **1,546 unrecorded incidental relations against 12,790 recorded
(~12%), in 280 of 300 instances (93.3%)** — semantic_preserving 816 · subsumption 420 ·
hard-negative 282 · complementary 28. Unlabelled pairs default to `DISTINCT`, so the replay as
originally scoped would have scored the verifier **wrong on ~1,500 pairs it gets right**.

Repaired by soundness, not convenience (`run_grho_replay.py:repair_key`):

| clone type | inheritance | treatment |
|---|---|---|
| EXACT | outputs byte-identical ⇒ relations transfer exactly | **materialised** |
| PERCEPTUAL | within tolerance, relations are threshold-based and can flip | **excluded, never guessed** |

At n=34 this yields 1,639 labelled pairs with **100 PERCEPTUAL-inherited pairs excluded**.
Independently recomputed from scratch: 1,639 / 436 mergeable / 100 excluded / 0 missing-skill.

> An earlier draft projected mergeable support at 473. The measured value is **436**; the
> projection was prose, never produced by a script.

### 1.5 ★ COMPLETE at n=300 — the headline held

**300/300 instances · 13,658 pairs scored · 0 errors** (all 14 CUDA-OOM rows repaired by re-running the affected instances)
(`results/wacv_r2/grho_full/report.md`). Determinism check across re-run shards: **0 verdict
disagreements**.

| quantity | n=34 | **n=300 (final)** |
|---|---|---|
| pairs scored | 1,639 | **13,658** |
| mergeable support | 436 | **3,310** |
| **merge recall** | 0.4931 [0.4452, 0.5411] | **0.5124** [0.4952, 0.5295] |
| instance-clustered bootstrap | [0.4553, 0.5368] | [0.4965, 0.5284] |
| precision (within labelled set) | 1.0000 | **1.0000** |
| F1 | 0.6605 | 0.6776 |
| **FDR on merges** | 0.0000 | **0.0000** |
| tp / **fp** / fn / tn | 215/**0**/221/1203 | 1,696 / **0** / 1,614 / 10,348 |

**Zero false merges across 13,658 pairs and all ten ρ levels.** The interval
narrowed ~3x and the point estimate moved by only 0.019, so the n=34 result was not a small-sample
artifact. The safety half of the claim is now supported at 20x the evidence.

| relation | support | recall [95% CI] | precision |
|---|---:|---|---|
| EXACT | 1,640 | **0.9683** [0.959, 0.976] | 1.0000 |
| **PERCEPTUAL** | 1,667 | **0.0438** [0.034, 0.055] | 0.6952 |
| SUBSUMPTION | 4,672 | 0.7581 | 0.9961 |
| SEMANTIC_PRESERVING | 3,424 | 0.3186 | 0.2784 |
| COMPLEMENTARY | 310 | **0.0000** | 0.0000 |
| DISTINCT | 1,942 | 0.3836 | 0.1880 |

The §1.2 story is confirmed at scale: **1,592 of 1,662 PERCEPTUAL pairs are called
SEMANTIC_PRESERVING**, and COMPLEMENTARY is still **0.0000 recall on 310 designed positives**.

★ **TWO CORRECTIONS to what n=34 suggested** — both are small-sample artifacts that the full run
overturns, and the earlier text must not be quoted:

1. **`UNCERTAIN` is NOT dead code.** It is emitted **2 times in 13,658 pairs**. Vanishingly rare,
   but the n=34 claim "never emitted in 1,639 pairs" was an artifact of sample size. The paper
   should say *effectively never* with the count, not *never*.
2. **EXACT loses 52 pairs, not 2.** The confusion row is 1,588 EXACT · 32 -> PERCEPTUAL (still
   mergeable) · 14 -> SUBSUMPTION · 6 -> SEMANTIC_PRESERVING. The tidy "99.1% of misses are one
   confusion" from n=34 becomes **~98.8%**, which is the number to quote.

✅ **Zero ERROR rows.** 14 CUDA-OOM failures (vLLM co-tenancy, `resize_*` pairs) were found and all 14 were repaired by re-running the 10 affected instances on a quieter card. The scored set is now the complete labelled key.

> The merger deduplicates by `(instance, a, b)` and lets a successful verdict **repair** an ERROR
> row from an earlier shard. Keeping the first occurrence — the original behaviour — would have
> made error-repair a silent no-op. Two *successful* copies that disagree are still reported as a
> determinism bug; none occurred.

## 2. ★ Text/embedding baseline — the paper's central claim, tested

R3: *"the paper never directly tests its main claim… I expected a comparison against a text- or
embedding-based matching approach on the same benchmark."* All eight rungs, same 944-pair `G_0`
benchmark, same frozen thresholds (`results/wacv_r2/a2_ladder_g0/report.md`):

| rung | operating point | FDR on merges | FPR distinct | **false merges on hard negatives** | abstain |
|---|---|---|---|---|---|
| 1 name-match | τ=0.500 | **1.0000** | 0.0540 | **5/6** [0.359, 0.996] | 0 |
| 2 tf-idf cosine | τ=0.600 | **1.0000** | 0.0021 | 0/6 [0.000, 0.459] | 0 |
| 3 sentence embedding | τ=0.600 | **1.0000** | 0.0561 | **5/6** [0.359, 0.996] | 0 |
| 4 code embedding | τ=0.900 | **1.0000** | 0.0180 | 4/6 [0.223, 0.957] | 0 |
| 5a AST structure-only | τ=0.900 | — | 0.0000 | 0/6 | 0 |
| 5b AST semantics-preserving | τ=0.900 | — | 0.0000 | 0/6 | 0 |
| 6 LLM on descriptions | categorical | **1.0000** | 0.0011 | 1/6 [0.004, 0.641] | 0 |
| 7 LLM on source | 120/944 × 3 repeats | — | — | mean 0 | 0 |
| **8 output-grounded (ours)** | categorical | — | **0.0000** | **0/6** [0.000, 0.459] | 14 (1.5%) |

**Name-match and sentence-embedding each produce 5 of 6 possible false merges; the output-grounded
verifier produces 0.** That is the central claim demonstrated rather than illustrated.

**Two honest qualifications.**

1. **The AUPRC column is empty for every rung.** `G_0` has no mergeable positives, so a
   precision-recall curve is undefined there and the ladder reports threshold points only. Fixing
   this is **T9** (§7).
2. **"A strong LLM judge over-merges" is NOT supported.** With a current judge (gpt-5.4) the
   llm-on-descriptions row is 0/926 + 0/6 on the Stage-1 rerun, not 1/926 + 1/6. Only name-match,
   sentence-embedding and code-embedding over-merge. **Lead with the taxonomy claim (H2), not the
   safety-scalar claim (H1).**

---

## 3. ★ Real uncurated library #1 — the fork pair (requirement 1)

`pilgram` 2.0.0 vs `pilgram2` 2.0.10: a genuine fork, both pip-installable side by side, **25
shared filter names**, 14 added by the fork, 0 removed, and **nothing in either library would ever
notice the overlap**. Unlike every other family in Corpus A this is not independent
reimplementation — it is unmanaged divergence from a common ancestor.

| quantity | value |
|---|---|
| implementations / families | 50 / 25 |
| **self-pair sanity gate** | **50/50** pass, exact 95% CI [0.929, 1.000] |
| **EXACT** | **23 / 25** — byte-identical on all 177 probes, `licenses_merge=True` |
| SEMANTIC_PRESERVING | 2 / 25 (`aden`, `perpetua`) — correctly **not** licensed for merge |
| exclusions | 0 |

**92% of a real fork's shared surface is exactly redundant.** Set against Corpus A, where only
10 of 17 published-`direct` rows survived execution as mergeable, the contrast *is* the argument:
**independent reimplementations diverge; forks stay identical and nobody checks.**

★ **`perpetua` is worth quoting in the paper: LPIPS 0.2464 (visibly different) against DINO p90
0.0008 (semantically indistinguishable).** A perceptual-only gate calls it divergence; a
semantic-only gate merges it. It is precisely the case the six-relation taxonomy exists to name,
found in the wild rather than constructed. `aden`: LPIPS 0.0472, DINO p90 0.0250.

*Provenance:* both divergent filters were independently flagged by a single random-probe smoke test
before the 177-probe run, which returned the same two. The alignment file is generated, not
hand-authored — the filters take **no parameters**, so there is no alignment choice an observed
output could contaminate; `corpus_a_forkpair.yaml`'s header states the authoring order explicitly
rather than implying it was written blind like the hand-built corpus.

---

## 4. Real uncurated library #2 — Corpus A cross-library audit

93 implementations across 20 families, 182 within-family pairs, 177-probe battery, thresholds
carried over and **not** recalibrated (`results/wacv_r2/corpusA/report.md`).

- **self-pair gate: 93/93** pass, exact 95% CI [0.961, 1.000]
- **Category-judgment survival rate (published `direct` rows): 10/17 = 0.588** [0.329, 0.816] —
  the fraction of pairs a published table groups as the same operation that execution certifies as
  mergeable at aligned parameters. Read as *a category-level judgment that does not survive
  execution*, never as a maintainer error.
- Restricted to FULL alignments only: **10/14 = 0.714** [0.419, 0.916]

### 4.1 The differential oracle (T2) — a mechanical answer key

Corpus A has **no** ground truth: its `relation` column is the verifier's own output, so scoring a
baseline against it is circular, and the published `direct` rows are n=17 category-level claims.
The oracle replaces the absent annotators (`results/wacv_r2/oracle_corpusA/`).

| quantity | value |
|---|---|
| pairs labelled | **182 / 182** |
| **AMBIGUOUS rate** | **0.0000** — every pair decided at Tier 1 |
| EQUIVALENT | 75 (41.2%) |
| DISTINCT | 107 (58.8%) |
| self-pair gate | **93/93 pass** |
| hard-negative gate | **0/0 — vacuous on this corpus** (see below) |

A zero AMBIGUOUS rate is a strong outcome: the cheap pixel-space tier settled everything, no pair
needed the learned backends, and nothing had to be excluded.

★ **Legitimacy is two claims, not one, and must be labelled as such wherever the numbers appear:**
- against the **text and code** rungs (1–7) the oracle is **independent ground truth** — those
  rungs never execute, so an execution-derived label is a different modality;
- against the **verifier** (rung 8) it is **approximation-vs-exhaustive, NOT independent** — both
  execute the same skills through the same adapters. Standard differential testing, but it is not
  an independent check and must never be presented as one.

⚠ **Caveat to carry:** the hard-negative half of the done-test reported 0/0 on Corpus A because
`G_0`'s engineered hard negatives do not exist among Corpus-A skill ids. **Only the self-pair gate
is meaningful on this corpus.** On `G_0` both gates pass: self **100/100** EQUIVALENT, hard
negatives **6/6** DISTINCT (`results/wacv_r2/oracle_gate/`).

The oracle is never tuned toward the verifier: `eps = 1.5/255` (one 8-bit LSB plus float
round-trip) and `delta = 0.10` come from the representation, not from any observed agreement rate.

---

## 5. ★ Scalability and cost — A4.1 … A4.6 (requirement 4)

`results/wacv_r2/a4_full/report.md`. 262,833 telemetry events across 156 files.

### 5.1 A4.1 — the shipped screening operating point is DOMINATED

Recorded telemetry held exactly **one** radius (0.5) plus a `-1.0` sentinel, so this curve had
never existed. Swept k × radius over 10 `G_ρ` instances (80 mergeable positives, 76,031 pairs):

| setting | merge recall | reduction |
|---|---|---|
| shipped: k=5, radius=0.5 | 0.9625 | 93.49% |
| **k=10, radius=0.05** | **1.0000** | **95.07%** |

**`k` binds, not radius.** Recall saturates at 0.9625 and does not improve from radius 0.05 to 2.0
— a duplicate outside a skill's top-5 nearest neighbours is never a candidate at *any* radius, so
widening the radius only admits non-duplicates. The full grid at radius 0.05:

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| merge recall | 0.7875 | 0.9125 | 0.9625 | **1.0000** | 1.0000 | 1.0000 |
| reduction | 99.27% | 98.04% | 97.05% | **95.07%** | 92.90% | 92.34% |

**Consequence beyond tuning:** in a deployed pipeline the screen sits *in front* of the verifier,
so at the shipped setting it silently discards **3.75% of mergeable pairs before verification ever
sees them** — compounding with the verifier's own 49.3% (§1). At k=10 that term goes to zero.

*(`G_0` supports only the reduction curve and a designed-pair retention curve, reported separately
and never called merge recall: it designates zero mergeable pairs.)*

### 5.2 A4.5 — |C| grows near-linearly, not quadratically

Nested random subsamples of `G_0`, 8 library sizes, seed recorded:

| n | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|---|---|
| \|C\| | 30 | 73 | 112 | 152 | 200 | 240 | 329 | 405 |
| \|C\|/all | 66.7% | 38.4% | 25.7% | 19.5% | 16.3% | 13.6% | 10.4% | **8.2%** |

**Fitted `|C| = 2.379 · n^1.126`**, log-log least squares, **R² = 0.998**. Exponent **1.13** against
the **2.00** of exhaustive pairing — screening converts a quadratic problem into a near-linear one,
measured rather than asserted.

Projected to n=10⁴ (**marked projected — the fit has support only to n=100**): |C| ≈ **75,795**
against 49,995,000 exhaustive pairs, ≈660× reduction.

### 5.3 A4.2 — cost per correct repair (totals, never mean-of-ratios)

| model | episodes | actions | applied | **correct repairs (tp)** | tokens in | tokens out | **tokens / repair** |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt-5.5 | 41 | 4,233 | 2,537 | **2,200** | 18,284,862 | 1,958,803 | **9,202** |
| gpt-5.6-luna | 41 | 6,122 | 2,994 | **2,338** | 22,544,274 | 2,394,685 | **10,667** |

The denominator is **correct repairs (`tp`)**, not actions applied: an agent that applies 200 wrong
edits has not performed 200 repairs. Both rows are restricted to the same 41 telemetry-instrumented
episodes — see §11.2 for the bug this fixed.

**Zero-repair case, defined:** `llama3.2-1b` scores F1 = 0.0000 over 300 episodes, so its cost per
repair is **undefined (division by zero)**. Reporting a large finite number would be wrong; the
correct reading is that any nonzero spend buys zero repairs.

⚠ **Coverage gap, stated not filled:** telemetry was instrumented at Stage 0, *after* the eight
self-hosted vLLM cells of Table 1 had run. Those rows carry **no token or wall telemetry**.
Re-running 8 models × 300 episodes to populate a cost column would be a backfill, which the plan
forbids. Hosted-model cost is measured; self-hosted cost is not.

### 5.4 A4.3 — where the time goes

| deciding stage | n | share |
|---|---:|---|
| residual | 6,020 | 47.2% |
| semantic | 2,672 | 21.0% |
| subsumption | 2,032 | 15.9% |
| complementary | 938 | 7.4% |
| exact | 918 | 7.2% |
| perceptual | 110 | 0.9% |
| abstention_band | 55 | 0.4% |

★ **54.9% of pairs are decided before any learned backend runs.** The cascade earns its keep — the
expensive perceptual and semantic stages are reached by a minority of pairs. Signature cache hit
rate 78.2% (184,824/236,305); pair verification median 401 ms, p95 64.5 s.

### 5.5 A4.4 — the cache is output images, and that is the whole cost

**Measured unit: 214.8 MB per skill** (one `OutputSet` over the 177-probe battery, median of 8
builtin skills) — this is what the LRU actually holds.

| library size | output cache (measured) | DINO-features-only (analytic) |
|---|---|---|
| 100 | 21.5 GB | 0.05 GB |
| 1,000 | 214.8 GB | 0.54 GB |
| 10,000 | **2.1 TB** | **5.4 GB** |

★ **A factor of ~395×.** Caching outputs is what costs; a features-only cache is a concrete design
lever. The measured unit also explains the Stage-4 benchmark's 161 GB peak on 100 skills **without
needing a leak hypothesis**. Eviction "never changes a result" (`compare.py:153`), so bounding the
cache trades wall-clock for memory without altering a verdict.

### 5.6 A4.6 — quality/cost Pareto

Quality axis is Table 1 (§6) with instance-clustered bootstrap CIs; cost axis is §5.3, populated
for hosted models only. Produced by `aggregate_curation.py` — **never per-pair bootstrap**, since
episodes on one corruption instance share base skills and a defect draw.

---

## 6. Table 1 — nine curators, all at n=300

`results/wacv_r2/table1/report.md`

| model | n | mean F1 [95% CI] | precision | recall | intrinsic | action cost |
|---|---:|---|---|---|---|---|
| gpt-5.6-luna *(41, hard tail only)* | 41 | 0.6201 [0.5946, 0.6471] | 0.8001 | 0.5165 | 0.5061 | 148.68 |
| **gpt-5.5** | 300 | **0.5868** [0.5719, 0.6012] | 0.7346 | 0.5305 | 0.4959 | 78.86 |
| qwen3.5-27b | 300 | 0.5050 [0.4903, 0.5206] | 0.6621 | 0.4233 | 0.3577 | 109.59 |
| gemma4-12b | 300 | 0.2900 [0.2754, 0.3059] | 0.7768 | 0.2036 | 0.1588 | 172.23 |
| qwen3.5-4b | 300 | 0.1090 [0.0967, 0.1222] | 0.2899 | 0.0977 | 0.0626 | 185.37 |
| qwen3.5-9b | 300 | 0.1010 [0.0848, 0.1174] | 0.3572 | 0.0739 | 0.0601 | 171.52 |
| llama3.1-8b | 300 | 0.0948 [0.0847, 0.1060] | 0.6150 | 0.0609 | 0.0491 | 200.00 |
| llama3.2-3b | 300 | 0.0237 [0.0194, 0.0284] | 0.4433 | 0.0127 | 0.0119 | 200.00 |
| qwen3.5-2b | 300 | 0.0130 [0.0086, 0.0182] | 0.1338 | 0.0077 | 0.0070 | 128.35 |
| llama3.2-1b | 300 | **0.0000** [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | 152.11 |

**Every cell is now n=300**, retiring the asymmetry the plan called "the first thing a careful
reviewer checks". GPT-5.5's 41 missing episodes were its *strongest*: the archived 259-episode cell
gave 0.5826, the completed 300-episode cell gives **0.5868** (+0.0042), and the 41 filled-in
episodes alone score 0.6136.

★ **Two abstract-level corrections follow:**
1. **recall 0.5376 → 0.5305**, so *"recovers only about half of the required repairs, recall 0.538"*
   becomes **recall 0.531**. "About half" survives; the digits do not.
2. **ρ is NOT monotonically harder, and the direction is model-dependent.** gpt-5.5's F1 *rises*
   with ρ (0.421 at 0.1 → 0.630 at 0.9, +0.210); qwen3.5-27b is flat; gemma4-12b and qwen3.5-4b
   fall ~0.18. More defects = more chances to score for a competent curator.

`gpt-5.6-luna` covers the 41 highest-ρ instances only, scoring **0.6201** [0.5946, 0.6471] where
qwen3.5-27b scores 0.489 and everything ≤9B is below 0.035. **Do not pool luna's 41 with gpt-5.5's
259** — different model, complementary stratum.

---

## 7. Hard negatives — the safety bound

| slice | false merges | exact 95% CI | one-sided 95% upper |
|---|---|---|---|
| original designed 6 | 0/6 | [0.000, 0.459] | 0.393 |
| **expanded 69** | **0/69** | **[0.0000, 0.0521]** | **0.0425** |

Cross-shard determinism check: **0 disagreements** over pairs verified independently by more than
one shard.

⚠ **The ≤5% claim is conditional.** 26 of the 69 receive a non-DISTINCT (but still non-mergeable)
relation. If a human adjudicator sides with the verifier the slice shrinks to 43 and the bound
rises to **6.7% — above 5%**. n=59 is the exact minimum. Disagreements enumerated in
`results/wacv_r2/hardneg69/design_disagreements.json`.

★ **A wording fix the paper needs before a reviewer opens `pairs.csv`.** The verifier calls only
**1 of the 6** designed hard negatives DISTINCT; the other **5** (resize ×3, pad_reflect/replicate,
posterize/quantize) it calls SEMANTIC_PRESERVING — confirmed identically in the archived Jul-2 run
*and* this one. `0/6` is true (SEMANTIC_PRESERVING does not license a merge) but supports only
*"never licenses a false merge"*, **NOT** *"keeps hard negatives apart as distinct"*.
Fix `sec/6_methodology.tex:192`.

### 7.1 `[PENDING]` — the 26 adjudications

- **Package delivered and turnkey:** `results/wacv_r2/adjudication/` — 26 pairs × 3 annotators,
  723 rendered images, structural blinding (verdict lives only in `key/`; pair order and A/B side
  shuffled per annotator with recorded seeds), `CANNOT-TELL` a first-class label, diff maps carry
  their per-probe amplification factor (125 of 241 probes needed none).
- **Blocker:** three co-author annotators. ~1 hour of human time.
- **Scoring ready:** `score_adjudication.py` (κ, `CANNOT-TELL` rate, both bounds). It deliberately
  does not reuse `load_review_labels`, which silently drops any label that is not a `Relation` and
  would swallow `CANNOT-TELL`.
- **Placeholders:** Fleiss κ `[PENDING]` · `CANNOT-TELL` rate `[PENDING]` · resolved slice size
  `[PENDING]` · final one-sided upper bound `[PENDING]`

★ **Why this was NOT completed by an automated annotator, having been considered and rejected.**

1. **Three annotators is the measurement, not a quantity.** `GUIDELINES.md` says it directly —
   *"the agreement statistic … is the whole point of running three of you"*. κ quantifies agreement
   between **independent** judges; three passes from one system are correlated draws and would
   inflate it. There is no valid κ to be had this way.
2. **The agent is already unblinded.** The package blinds structurally (verdicts live only in
   `key/`; pair order and A/B side shuffled per annotator with recorded seeds), but the agent had
   read the reports and therefore knows these 26 are exactly the pairs where the verifier returned
   non-DISTINCT against a DISTINCT design — which *is* the variable under adjudication. The
   guidelines say *"do not consult any other file"*; that condition is already violated.

A single, clearly-labelled **model-annotator** pass (written outside `annotator_1/2/3`) would give
a provisional read on whether the bound lands nearer 4.25% or 6.7%. It would not produce κ and
would carry the contamination caveat above. **Not run without an explicit human decision**, because
the risk is that it gets read as "the adjudication is done" when it is not.

### 7.2 `[PENDING]` — T9, the ladder scored against oracle labels

The oracle (§4.1) now supplies the answer key Corpus A lacked, which makes a properly-scored ladder
possible for the first time — **AUPRC instead of threshold points** (§2 qualification 1).

- **Artifact:** `results/wacv_r2/a2_ladder_corpusA/` (not yet created)
- **Command:** `run_a2_ladder.py --pairs-csv` over Corpus A with oracle `true_relation`
- **Placeholders:** per-rung AUPRC `[PENDING]` · matched operating points `[PENDING]` ·
  rung-8-vs-rungs-1-7 comparison `[PENDING]`
- ⚠ **Label rung 8's row "approximation-vs-exhaustive", not independent** (§4.1).

---

## 8. Two real defects found and fixed in the codebase

### 8.1 ★ The merge gate certified PERCEPTUAL on zero evidence (safety-critical)

`_sweep_perceptual_worst` seeded its running worst-case maxima at `-1.0` and returned them
unguarded. When two skills share **no** successfully executed probe, worst-case LPIPS is reported
as `-1.0`, which is below every perceptual threshold, so the pair certifies **PERCEPTUAL →
`licenses_merge=True`**. *A skill that raises on every probe was licensed for merge with anything,
on zero evidence.*

- **Why it survived 274 tests:** it only manifests with a perceptual backend attached; the
  `--no-ml` path returns DISTINCT correctly, and every confirmed Table-1 curation run used
  `--no-ml`. It matters in practice — corruption defect type 7 ("dead skill") produces exactly such
  a skill, as does any third-party function that cannot run on the battery.
- **Found by** the Corpus A self-pair gate: `alb.gaussian_blur` failed all 177 probes and came back
  PERCEPTUAL against itself in 7 ms.
- **Fix:** the `saw_common` guard `_sweep_pixel_worst` already had, returning `inf` so the pair
  falls through to DISTINCT. Regression tests in `tests/test_taxonomy_empty_common.py` (4 tests;
  they fail without the fix, and one pins that a genuine self-pair still certifies EXACT).
- **Verified harmless to Stage 1** (§11.1).

#### ★ Completeness audit of the fix (2026-08-25)

The question a safety review must answer is not "was a hole patched" but "is the SET of holes
closed". Every path in `classify()` that can return a merge-licensing relation was traced:

| stage | licenses merge? | empty-common-probe guard |
|---|---|---|
| 1 · EXACT | yes | ✅ `all_hash=False` **and** an explicit `pix.value != inf` |
| 2 · PERCEPTUAL | yes | ❌ **the hole** — fixed here |
| 3 · SUBSUMPTION → `mutual` → EXACT/PERCEPTUAL | yes | ✅ `outputs_match`: `if not common: return False` |
| 4 · SEMANTIC | no (never mergeable) | ✅ `quantile({}) → inf` |

**`_sweep_perceptual_worst` was the ONLY unguarded path of the four, so the fix is complete rather
than partial.** A reviewer can check that single claim instead of re-deriving the analysis.

**Mechanism confirmed.** `_sweep_pixel_worst` already returned `inf` correctly, so Stage 1 was
never the failure point — but Stage 2 was still *entered*, and its unguarded `-1.0` seed sits below
every perceptual threshold. That is how a pair with no comparable output certified PERCEPTUAL.

**Tests:** the 4 targeted regressions pass and fail without the fix; the full suite is
**292/292**, counted from the per-test result characters rather than a summary line.

*(Counting matters here: an earlier invocation passed `--timeout=600` without `pytest-timeout`
installed, so pytest aborted on the argument and still **exited 0**. A green exit code was not
evidence the suite had run.)*

**Minor, deliberately not changed.** When `pix.value == inf` the perceptual sweep re-executes the
whole parameter sweep to rediscover a fact already known, wasting work on exactly the pathological
pairs. An early skip is a clean follow-up; it is left out so the safety change can be reviewed in
isolation.

- 🔴 **STILL REQUIRES A HUMAN SIGN-OFF.** The audit above was performed by the same agent that
  wrote the fix, so it is a **self-review**: it substitutes for a reviewer's *analysis*, never for
  a second person's *judgement*. The change is strictly more conservative and provably cannot alter
  any `--no-ml` number, but it can change verdicts in any ML-backed run containing a skill that
  fails to execute, and the org rule on data-deletion-adjacent code requires a person to sign off.

### 8.2 `ENGINEERED_HARD_NEGATIVES` had drifted from the ground-truth YAML

The 6th engineered pair `(resize_nearest_v1, resize_bicubic_v1)` was missing from
`candidates.py`. Restored, with a comment that `configs/ground_truth_g0.yaml` is authoritative.

### 8.3 COMPLEMENTARY is the weakest element of the taxonomy

On `G_0`: **246/944 pairs (26%) classified COMPLEMENTARY against a designed support of 1,
precision 0.000** — and it was the only verdict to move between runs. Commutation is "necessary,
not sufficient" per the repo's own docs. Reproduced on `G_ρ` (§1.2: 0.000 recall on 35 positives,
predicted 64 times). **Two-dataset result; the best-motivated next piece of methods work.**

---

## 9. T5, T6, T7 — all complete

### 9.1 ★ T7 — neural & stochastic support: COMPLETE (required revision 3)

`results/wacv_r2/a3_neural/a3_result.json`. **6 neural skills · 40 probes ·
16 stochastic draws each · 2,000 permutations · alpha=0.05.**
Models: SAM 2.1 hiera large/small, Depth Anything V2 Large/Small, plus two purpose-built negatives.

| gate | result |
|---|---|
| self-pairs NOT distinguishable (the noise floor) | **6/6** ✅ |
| hard negatives distinguishable | **2/2**, 95% CI [0.158, 1.000] ✅ |

| comparison | domain | result |
|---|---|---|
| DAv2-Large vs DAv2-Small | full | DISTINGUISHABLE · min p=0.0004998 · reject 40/40 |
| SAM2-Large vs SAM2-Small | full | DISTINGUISHABLE · min p=0.0004998 · reject 13/38 |
| SAM2-Large vs SAM2-Small | applicable | NOT_DISTINGUISHABLE · min p=0.03548 · reject 0/10 |
| **depth vs disparity** (hard neg) | full | DISTINGUISHABLE · min p=0.0004998 · reject 38/40 |
| **instance vs semantic** (hard neg) | applicable | DISTINGUISHABLE · min p=0.0009995 · reject 3/10 |

★ **The finding: SAM2-Large and SAM2-Small separate on synthetic patterns (13/38) but NOT on
object-bearing images (0/10).** Two capacities of the same architecture are not distinguishable at
this power where segmentation is actually defined. Read strictly against the noise floor — a
non-rejection is absence of evidence, never equivalence.

★ **A limit of the existing battery, discovered here.** The instance-vs-semantic negative cannot
manifest on inputs with no objects: on `deg_black` the mean L2 is exactly 0.000 and MMD^2 is
*negative* on 7 of 8 early probes, because SAM2's masks collapse to one region so "best mask" and
"union of masks" are identical by construction. The battery is 137/177 synthetic patterns against
40 COCO photos, so **it is not a valid domain for segmentation skills**. Both analyses are reported
(`domain='full'` and `domain='applicable'`) and the docstring discloses that the restriction was
chosen *after* seeing the full-battery result.

⚠ **A harness bug this study found in itself, fixed not caveated.** `dav2_large_disparity` was
distinguishable **from itself** (MMD^2=+0.30 against meanL2=0.002) because `1/clip(depth,1e-3)`
explodes near zero depth, so 1-LSB jitter moved the map enormously and min-max normalisation made
the image hostage to a few pixels. Replaced with a 5th-percentile floor; the unstable run is kept
at `a3_neural_v1_unstable_disparity/`. Its gate also mis-tallied **1/1** by keeping only
applicable-domain rows, dropping depth-vs-disparity; corrected to **2/2**.

**Not claimed:** that the deterministic path's precision guarantee transfers. Counts with exact
bounds only, and n=2 hard negatives gives a wide interval.

### 9.2 ★ T5 — probe-battery ablation: COMPLETE (R3's explicit request)

`results/wacv_r2/ablation/`. 21 configurations over the 24 designed
`G_0` pairs (6 engineered hard negatives). **Nested** subsets, so any change is attributable to
battery SIZE alone; all strategies are data-independent, so selection cannot leak.

**Accuracy is FLAT in battery size** — every interval overlaps every other on n=24, so there is no
knee to report and none is invented:

| strategy | \|B\|=8 | 16 | 32 | 64 | 128 | 177 |
|---|---|---|---|---|---|---|
| stratified | 0.58 | 0.50 | 0.54 | 0.50 | 0.50 | 0.46 |
| first | 0.33 | 0.38 | 0.38 | 0.46 | 0.50 | 0.46 |
| random | 0.50 | 0.50 | 0.50 | 0.42 | 0.46 | 0.46 |

★ **Safety is NOT flat, and that is the result — it is about DIVERSITY, not count:**

| strategy | \|B\|=8 | 16 | 32 | 64 | 128 | 177 |
|---|---|---|---|---|---|---|
| stratified | 0 | 0 | 0 | 0 | 0 | 0 |
| **first** | **1** | **1** | **1** | **1** | 0 | 0 |
| random | 0 | 0 | 0 | 0 | 0 | 0 |

**Taking the first N probes produces a false merge on the engineered hard negatives that stratified
and random sampling never do, and it persists to |B|=64.** The battery is ordered by class, so
`first` means "all probes from one class" — this is probe *diversity* protecting merge safety, and
it is a sharper answer than the size curve R3 asked for. **Recommendation: never select probes by
index; stratify across classes.**

**Backend ablation** (|B|=177, stratified): `pixel_only` matches `full`
(0.46 vs 0.50 accuracy) with **0/6 false merges in every
configuration** — consistent with A4.3's finding that 54.9% of pairs are decided before any learned
backend runs.

> The full battery is a **reference, not an oracle**. A smaller battery that disagrees is
> less-evidenced, not wrong; the designed relations are the only correctness signal.

### 9.3 ★ T6 — OOD certificate-flip: COMPLETE (metareview + R2 structural objection)

`results/wacv_r2/ood/ood_result.json`. `B_ood` = **31 probes** across
6 classes absent from the shipped distribution
(aspect, hdr, medical, satellite, saturated, tiny). Thresholds carried over unchanged; **nothing re-calibrated**.

| quantity | value |
|---|---|
| certified merges on `B` (real third-party code) | **74** |
| **certificate flips on `B_ood`** | **1** |
| **flip rate** | **0.0135** — exact 95% CI [0.0003, 0.0730] |
| one-sided 95% upper bound | **0.0625** |

★ **The objection is real but bounded.** Of 74 merges the verifier actually
licensed on real libraries, **1 did not survive out-of-distribution inputs**:

- `alb.gaussian_blur` vs `cv2.GaussianBlur` — **PERCEPTUAL -> SEMANTIC_PRESERVING**

So *"two skills matching on a finite probe battery may still diverge out-of-distribution"* is
correct, and its magnitude on this corpus is **~1.4%, upper bound 6.3%** — a measured number in
place of an unquantified structural worry. The subjects are real certified merges, so a flip means
"we would have deleted a skill that behaves differently on inputs the battery did not contain".

**Refutation direction only.** Surviving `B_ood` is not evidence of equivalence, and no
reverse-flip statistic is computed: a DISTINCT pair looking mergeable on new inputs would be a
threshold artifact, not a discovery.

## 10. 🚫 Blocked — T8 Corpus B (ComfyUI)

The literal reading of requirement 1 ("uncurated open-source skill collection"). **Cannot be run
from this cluster** and that is a policy limit, not a technical one.

- **Why:** importing a pack *is* executing it — `__init__.py` runs on import, before any function
  is called. There is no safe inspection phase. `curation/sandbox.py` has `allow_untrusted=False`
  and the hardened sandbox is deliberately unimplemented.
- **Required containment if run elsewhere:** disposable VM or Docker with `--network=none
  --read-only --user nonroot --cap-drop=ALL`, tmpfs scratch, **no volume mounts**, block
  `169.254.169.254`, no credentials on the box, pinned commit SHAs, destroy the VM after. **Do not
  treat `sys.addaudithook` as containment.**
- **Protocol when it happens:** stratified sampling with a **popular** stratum *and* a **random
  long-tail** stratum, reported **separately** (never "X% of ComfyUI nodes are redundant" from the
  popular stratum — that measures deployments, not the ecosystem); isolated env per pack; **contract
  violations fail the gate — never clamp or reshape**, since that manufactures false merges; and
  **do not compute compression from connected components** (PERCEPTUAL is not transitive) —
  simulate sequential curation with re-verification.
- **The exclusion funnel is itself a finding** and should be reported.
- **Mitigation already in place:** §3 and §4 answer requirement 1 with two real libraries, one of
  them a genuine fork. **Placeholders:** packs sampled `[PENDING]` · exclusion funnel `[PENDING]` ·
  redundancy rate per stratum `[PENDING]`

---

## 11. Methodology notes that changed a number

### 11.1 Pre-flight — one check saved a full re-run

| # | check | result |
|---|---|---|
| P1 | `G_ρ` key covers incidental relations? | **No** → repaired, §1.4 |
| P2 | `G_ρ` text surfaces degenerate like the wrappers? | **No** — 64/64 distinct descriptions |
| P3 | Does the §8.1 fix change Stage 1? | **No** — provably |
| P4 | Telemetry has the A4 fields? | **Yes**, all of them |
| P5 | GPU capacity | ample |

**P3 in detail:** zero pairs with `lpips < 0` and zero PERCEPTUAL verdicts in **both** the archived
paper run and the Stage-1 re-run. The guard can only fire on an empty common probe set, which `G_0`
never produces. **The planned Stage-1 re-run is unnecessary, not skipped.**

**P2 in detail:** `G_ρ`'s *code* surfaces are degenerate (§11.3) but its *text* surfaces are not —
all 64 synthesized skills in a ρ=1.0 instance carry distinct descriptions, and clones inherit the
donor's verbatim. Ladder rungs 1/2/3 are therefore viable on `G_ρ`; 4/5/7 are not.
⚠ **Caveat:** because the harness copies the donor description verbatim, a description-similarity
rung is handed an *easier* task on `G_ρ` than in a real library, where a duplicate is re-described.

### 11.2 A4.2 numerator/denominator mismatch

`phase6_gpt55_full` holds **300** scored episodes but only **41** carry token telemetry. Dividing
all-300 repairs by 41-episodes-of-tokens made gpt-5.5 look **5× cheaper** than luna (2,087 vs
10,667 tokens/repair). Matched to the same 41 episodes the true figures are **9,202 vs 10,667**.
*Numerator and denominator must come from the same episodes.*

### 11.3 ★ Why the A2 ladder can never run on `G_ρ` (deliverable T8-ladder is blocked, not pending)

Synthesized skills are **delegating wrappers**, not copied bodies:

```python
def dup(image: Image, params: Params, seed: int) -> Image:
    return donor_fn(image, params, seed)
```

On one ρ=1.0 instance, **64 synthesized skills collapse to 4 distinct source strings**, and
**519 of 2,016 cross-original synthesized pairs (25.7%) are byte-identical in source while wrapping
different skills.** Any code-similarity rung would therefore (a) score a clone as maximally
*dissimilar* from its own original — wrapper vs real body — and (b) score two clones of *unrelated*
skills as *perfect duplicates*. Additionally, `build_units()` reads only
`build_builtin_registry()`, and **all 436 mergeable positives involve at least one synthesized
skill**, so restricting to builtin pairs re-empties the positive class.

Unblocking means materialising real copies in the harness or resolving wrapper→donor in
`build_units` — and **the choice determines the answer** (donor-resolution makes AST rungs score
EXACT clones 1.0 by construction). That is an author design decision, not a run.

★ **The artifact is itself an argument for the paper.** A delegating wrapper *executes* identically
to its donor, so output-grounded verification is **immune by construction** to the representation
confound that makes every source-similarity baseline unusable here. The verifier scored 0 false
merges on this set; a source-similarity rung would call 25.7% of cross-original clone pairs
identical. That is the cleanest available demonstration of why grounding in outputs rather than
representations is the right primitive.

### 11.4 Blocker resolved for good — the paper's numbers are real

`CODEBASE_RECONCILIATION.md` §7 could not tell whether Table 2/3 were genuine. They are:
`results_before_phase4/phase4_sonnet46_full/manifest.json` has `lpips-alex` + DINO + CLIP,
`device: cuda`, `thresholds_calibrated: true`, 944 pairs, 177-probe battery, 713/100 calibration
split, and a `report.md` reproducing every published figure. `phase4.md` is accurate;
`WORK_PROCESS_OVERVIEW.md` is stale (it describes the older cpu/24-pair run). Calibration
reproduces byte-identically on an independent split; the verifier relation is identical on
**943/944** pairs.

Documentation fixes: `phase4.md:19` "17% to 83%" → **"0% to 83%"**; `sec/7_results.tex:39` must be
re-earned.

---

## 12. Rules held throughout

1. **Never re-calibrate thresholds on evaluation data.** Every run above carries the Stage-1
   operating point unchanged. No test-driven repair.
2. **Freeze and hash before running** — pair lists, alignment maps, pack lists.
3. **Every rate carries raw counts and an exact interval.** `0/69` is not a result;
   `0/69 [0, 4.25%]` is.
4. **Bootstrap by corruption instance / base skill / seed — never by pair.** Pairs within an
   instance share base skills and a defect draw; pair-level resampling badly understates variance.
5. **Never fabricate.** "Not yet run" is an acceptable answer, and **blocked / deliberately-cut /
   pending** are kept as three distinct categories.
6. **Report numbers exactly as produced.** Where they disagree with the paper draft, the draft is
   what changes — see the abstract recall correction (§6) and the hard-negative wording (§7).
7. **Do not modify the corruption harness.** It is frozen and produced Table 1.

---

## 13. Open items — all remaining work is HUMAN or policy-blocked

| # | Item | Blocks | Who |
|---|---|---|---|
| 1 | **26 hard-negative adjudications** (§7.1) | decides the safety bound: **4.25% vs 6.7%** | 3 co-author annotators, ~1 h |
| 2 | **Human review of the merge-gate fix** (§8.1) | safety-critical code path | 1 reviewer, ~1 h |
| 3 | T9 ladder scored on oracle labels (§7.2) | AUPRC for requirement 2a | **code change first** — `build_units()` knows 0 of Corpus A's 93 ids (§7.2) |
| 4 | T8 Corpus B (§10) | strengthens requirement 1 | blocked — needs a disposable VM |
| 5 | Paper edits (§6, §7, §11.4) | abstract recall 0.538 -> **0.531**; hard-negative wording at `sec/6_methodology.tex:192`; `phase4.md:19` "17%" -> "0%" | authors |

**No experiment is waiting on compute.** Items 1 and 2 need people; item 4 needs a machine this
cluster's policy forbids; item 3 needs a small code change before it can run (its "~1 h,
ready to run" estimate was wrong twice — see §7.2); item 5 is short and unblocked.

For item 2 the analysis is now done (§8.1) and only the sign-off remains, which lowers its cost
without removing the requirement.
