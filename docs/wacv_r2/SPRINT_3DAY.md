# VisCurate — 3-Day Sprint Plan

**Date:** 2026-08-19
**Budget:** 3 days wall-clock. **GPU compute is not a constraint** — do not scale runs down.
**Constraint that IS real:** no human annotators, and 3 days of wall-clock.
**Supersedes:** [`ROADMAP_FINAL.md`](ROADMAP_FINAL.md) sequencing (its technical content still holds).

---

## The one thing that decides whether this works

**Serialization is the enemy, not compute.** If tasks run one after another, you finish ~40%. If the independent tracks launch on Day 1 morning across separate GPUs, all four AC items land.

**Everything in Wave 1 below is mutually independent. Launch all of it before doing anything else.** Then work the CPU-side tasks while the GPUs churn.

---

## Pre-flight — 30 minutes, do before launching anything

These are cheap checks that each prevent losing half a day.

| # | Check | Command / method | If it fails |
|---|---|---|---|
| **P1** | `G_ρ` answer key covers **incidental** relations, not just injected ones | Sample 3 instances; manually enumerate true relations for ~20 pairs; compare to `G_ρ` | Restrict scoring to explicitly-labelled pairs and **say so**. Do not let unlabelled default to `DISTINCT` — it penalises correct verdicts. |
| **P2** | `G_ρ` **text surfaces** aren't degenerate like the wrappers were | Count distinct `description` strings among synthesized skills in one ρ=1.0 instance | If they collapse to a template → **text rungs die on `G_ρ` too**; run verifier-only (still satisfies AC #2). |
| **P3** | `_sweep_perceptual_worst` fix vs. Stage 1 | Diff Stage-1 outputs pre/post fix | If it changes ML numbers → **re-run Stage 1 on Day 1** (it's cheap and it's the run the paper rests on). |
| **P4** | Telemetry actually captured the fields A4 needs | `head results/telemetry.jsonl`; check for `cache_hit`, `deciding_stage`, `tokens_in/out` | Report what exists; note gaps. Do **not** re-run to backfill. |
| **P5** | GPU shard capacity | `nvidia-smi` | Determines the shard map below |

**P1 and P2 gate Wave 1 Task 1. Run them first.**

---

## Wave 1 — launch Day 1 morning, all in parallel

| Shard | Task | Bound | Why it's first |
|---|---|---|---|
| **GPU 1–2** | **T1** — `G_ρ` verifier replay, **full 300 instances** | GPU | **Produces merge recall — the missing half of AC #2.** Never measured before. |
| **GPU 3** | **T2** — Oracle build (dense battery + dense grids) | CPU-heavy, GPU tail | Unblocks all labelling. Replaces annotators. |
| **GPU 4** | **T5** — Ablation suite (probe-size sweep, backend, thresholds, seeds) | GPU | Answers R3's minor point; feeds A4's cost levers. |
| **GPU 5** | **T6** — OOD battery + certificate-flip rate | GPU | R2's structural critique → a measured number. |
| **GPU 6** | **T7** — A3 neural: SAM2 ×2, DepthAnythingV2 ×2 + hard negatives | GPU | 96 GB is enormous for these (<2 GB each). No reason not to. |
| **CPU** | **T3** — Corpus A expansion: install + extract new libraries | CPU | Zero risk, pure setup. |
| **CPU** | **T4** — **A4 full analysis** on existing telemetry | CPU | **Completes AC #4 entirely. No new runs needed.** |
| **CPU** | **T8** — Corpus B container setup + pack extraction | CPU | Highest-risk track; start early so the cut decision is informed. |

**Everything above is independent.** Nothing waits on anything else.

---

## Task cards

### T1 — `G_ρ` verifier replay ★ highest value

**Goal:** merge recall (`EXACT` / `PERCEPTUAL` P/R/F1) — a number that has never existed in this project.

**Why it's valid despite the wrapper confound:** verified in `corruption/mutators.py:189–242` — `make_exact_dup_fn` returns the donor's output byte-identically, `make_perceptual_dup_fn` adds a ≤1 LSB dither, `make_fixed_param_fn` bakes params. **Behavioural ground truth is entirely correct.** Only *source-text* similarity is broken.

**Run:** verifier + rungs 1/2/3/6 across all 300 instances, sharded by instance. **Do not subsample** — you have the compute.

**Report:** per-relation P/R/F1 **with support**; binary mergeable with a populated positive class; 7×7 confusion incl. `UNCERTAIN`; breakdown by ρ; **bootstrap by instance/seed, never by pair**; exact Clopper–Pearson intervals.

| Failure mode | Fallback |
|---|---|
| P1 fails (answer key gaps) | Restrict to labelled pairs; report the restriction and its size |
| P2 fails (template descriptions) | **Verifier-only.** Still delivers AC #2's "verifier self-validation" — text rungs on `G_ρ` were always a bonus |
| Slower than expected | Shard harder across GPUs; last resort ρ ∈ {0.2, 0.5, 0.8, 1.0} × 5 seeds, report n |
| Rungs 4/5/7 tempting to include | **Don't.** They're confounded. They go to Corpus A (T3). |

**Done test:** a non-degenerate `EXACT` row with support > 0 in the per-relation table.

---

### T2 — The exhaustive differential oracle ★ replaces annotators

**Goal:** a mechanical `true_relation` column, so no task needs a human.

**Design — tier it so it's fast:**
```
Tier 1 (CPU, cheap, run on everything):
    max L∞ over DENSE battery × DENSE parameter grid
    d ≤ ε_oracle           → EQUIVALENT
    d ≥ Δ_oracle           → DISTINCT
    else                   → send to Tier 2

Tier 2 (GPU, only the undecided):
    LPIPS + DINO adjudication
    still undecided        → AMBIGUOUS  (excluded, rate reported)
```

**Scale it up — you have the compute.** Target ≥1000 probes (extend the existing 177 with adversarial, degenerate, and boundary inputs) and a grid several times finer than the verifier's. The oracle's entire value is being *strictly more thorough* than the thing it scores.

**Legitimacy — state this precisely in the paper:**

| Scoring | Status |
|---|---|
| Text + code baselines (rungs 1–7) | **Independent ground truth** — they never execute, so an execution-derived label is a separate modality |
| The verifier (rung 8) | **Approximation-vs-exhaustive.** Standard in differential testing, but must be *labelled* as such, not passed off as independent |

| Failure mode | Fallback |
|---|---|
| `AMBIGUOUS` rate high | **Widen the battery first** (compute is free). If still high, report the rate — it measures how much real-library equivalence is genuinely undecidable. Score the decisive subset. |
| Too slow at full density | Tier further: full density on candidate pairs, coarse on the rest |
| Oracle disagrees with verifier a lot | **Not a failure — that's the result.** Never tune the oracle toward agreement; that inverts the experiment. |

**Done test:** self-pairs → `EQUIVALENT` 100%; the 6 original engineered hard negatives → `DISTINCT`. Both must pass or the oracle is broken.

---

### T3 — Corpus A expansion + code rungs

**Goal:** grow 182 → 400+ pairs, and give rungs 4/5/7 their correct home.

**Add:**

| Library | Why | Risk |
|---|---|---|
| **`pilgram` + `pilgram2`** | **A fork pair with organic drift** — ~26 shared filter names, pilgram2 adds 14 and changed non-square handling. Same names, forked code, years of divergence. **This is the thesis in miniature.** | None — pure Python + PIL/numpy |
| **ImageMagick via `Wand`** | ~30 years of accumulated ops, never deduplicated | Needs the ImageMagick binary |
| **`imgaug`** | Heavy overlap with albumentations/kornia | None |
| **`opencv-contrib`** | Extra operator families | None |

**Then run rungs 4/5/7 here**, scored against oracle labels. Real cross-library code — genuinely different source, genuinely similar behaviour — is what those rungs were always for.

**Framing to write down:** each library is individually curated, but **nobody curates across them**, and the fork pair is genuine unmanaged divergence. State this claim explicitly rather than letting a reviewer infer you dodged the AC's wording.

| Failure mode | Fallback |
|---|---|
| Wand/ImageMagick install fails | Drop it, note it. The others carry the corpus. |
| Parameter alignment gaps | Report **alignment-failure rate separately from behavioural non-equivalence** — they are different findings |
| Fewer pairs than hoped | Combine with Corpus B-lite; report as one real-library section with strata |

---

### T4 — A4 full analysis ★ completes an entire AC item, zero new runs

Telemetry has recorded since Stage 0. **This is desk work.**

| # | Deliverable |
|---|---|
| A4.1 | **Candidate reduction AND recall** vs. fingerprint radius — the required curve |
| A4.2 | Cost table — **totals, not mean-of-ratios**; define the zero-repair case (Llama 3.2 1B) explicitly |
| A4.3 | Per-stage timing, cache-hit rate, deciding-stage distribution |
| A4.4 | Feature-cache memory + extrapolation (~27 GB DINO-only at n=10⁴) |
| A4.5 | Scaling projection, **fitted** \|C\| growth across ≥3 library sizes, measured vs. projected marked |
| A4.6 | Pareto via `aggregate_curation.py`, instance-clustered bootstrap |

**Note on cost units:** Stage 2 used self-hosted vLLM, not a paid API. **Report GPU-seconds per repair, not USD** — it's more reproducible and doesn't rot when prices change. Report tokens separately.

| Failure mode | Fallback |
|---|---|
| A telemetry field was never wired | Report what exists; note the gap. **Do not re-run to backfill.** |
| Candidate recall needs real positives | Use oracle labels from T2 once available; until then report synthetic recall **with the "guaranteed by construction" caveat stated** |

---

### T5 — Ablation suite

Probe-size sweep `|B| ∈ {8,16,32,64,128,177}` (add larger points only if the battery is genuinely extended), three selection strategies, probe-class drop-one, backend ablation (LPIPS-only / DINO-only / CLIP-only / pixel-only / full), threshold sensitivity **including the joint `(τ_pc, τ_sm)` grid**, seed stability over ≥5 calibration seeds.

**Leakage rule:** probe selection is learned on **calibration families** and tested on **disjoint** families. Selecting on the same pairs you evaluate leaks.

**Hand the battery knee to T4** — it's a measured cost lever.

---

### T6 — OOD divergence

Build `B_ood`: medical, satellite, documents, extreme aspect ratios, high/low dynamic range, and **adversarial boundary probes** (kernel ≈ image size, rotation ≈ 90°, empty masks, saturated inputs). The last class is the highest-yield and cheapest.

**Headline:** certificate-flip rate — of pairs certified `EXACT`/`PERCEPTUAL` on `B`, what fraction diverge on `B_ood`.

**Refutation direction only.** A pair that agrees on OOD data is *not* evidence of equivalence — the original divergence still stands. Do not report a "reverse flip."

If OOD failures motivate extending the default battery: label the first study **discovery**, extend, then evaluate on a **fresh untouched holdout**. Report both.

---

### T7 — A3 neural feasibility

**Do it properly — 96 GB VRAM makes these trivial (<2 GB each).**

Models: SAM 2 large/small (segmentation), Depth Anything V2 large/small (depth). Hard negatives: depth-vs-disparity, instance-vs-semantic masks.

**Two mandatory technical points:**
- **Conditional per-probe comparison, never pooled.** Pooling destroys conditioning: A doing `blur(x₁), sharpen(x₂)` and B doing `sharpen(x₁), blur(x₂)` have *identical* pooled distributions and would certify equivalent while behaving completely differently. That's a false merge by construction.
- **Permutation null, not naive bootstrap.** The unbiased MMD estimator can be negative and is degenerate under equality — exactly where equivalence decisions are made.
- **Depth needs affine alignment** (scale + shift per image) or you measure output conventions, not behaviour.

**Ground truth:** the oracle again — dense multi-seed distributional comparison. No author-asserted relations.

**Must not claim** the 0.99 precision guarantee transfers. Report counts with exact bounds.

| Failure mode | Fallback |
|---|---|
| Model download fails | Use whatever loads; report n |
| Self-variance > between-model distance | **Genuine finding, not a failure.** Report it — it justifies the abstention design. |
| Too few pairs for a claim | Report as feasibility with exact bounds; that was always the framing |

---

### T8 — Corpus B (ComfyUI) — the only literal answer to AC #1

**Containment (non-negotiable):** Docker with `--network=none --read-only --user nonroot --cap-drop=ALL`, tmpfs scratch, **no volume mounts**. Setup ~1 hour. Blocks exfiltration and file access — the realistic threats.

> ⚠ **Importing a pack is already executing it.** `__init__.py` runs on import, before you call any function. There is no safe inspection phase. Also block the cloud metadata endpoint `169.254.169.254`.

**Then:** stratified sampling (popular stratum + **random long-tail stratum**, reported separately), inclusion filter (required inputs = `IMAGE` + primitives only), isolated env per pack, **contract violations fail the gate — never clamp or reshape** (that manufactures false merges).

**Do not compute compression from connected components** — `PERCEPTUAL` isn't transitive. Simulate sequential curation with re-verification.

| Failure mode | Fallback |
|---|---|
| Container issues | Disposable cloud VM |
| Dependency hell → few executable nodes | **The exclusion funnel IS a finding.** Report it. Reduce to top-N packs. |
| Not landing by end of Day 2 | **Hard cut.** Corpus A + fork pair is the fallback real-library result. |

---

## Day-by-day

### Day 1
- **AM:** P1–P5 pre-flight (30 min) → **launch Wave 1 across all shards**
- **PM:** T4 (A4 analysis) end-to-end on CPU while GPUs run · T3 installs · T8 container
- **EOD gate:** T1 producing a non-empty `EXACT` row? T2 passing its done-test?

### Day 2
- **AM:** T2 oracle labels land → **unblocks T9 (Corpus A ladder) and the 26 hard-negative labels**
- **PM:** rungs 4/5/7 on Corpus A · resolve the 26 → firm up the **4.25%** bound · finish rung 7 (120/944 → full; compute is free)
- **EOD gate:** **Corpus B decision — landed or cut.** Do not let it eat Day 3.

### Day 3
- **AM:** consolidate every table; re-run anything the `_sweep` fix invalidated
- **PM:** write-up — abstract **0.538 → 0.531**, results, limitations
- Stretch only if genuinely free: extend A3, extend Corpus B strata

---

## Decision gates

| When | Question | If no |
|---|---|---|
| Day 1 EOD | T1 producing `EXACT` support? | Verifier-only mode; still satisfies AC #2 |
| Day 1 EOD | Oracle passes its done-test? | Fix before Day 2 — everything downstream depends on it |
| Day 2 EOD | Corpus B executing real nodes? | **Cut it.** Corpus A + fork pair + explicit limitation |
| Day 2 EOD | A3 producing certificates? | Report as feasibility with bounds, or cut with a written limitation |
| Day 3 AM | Any table missing? | Report "not run" — never fabricate |

---

## Minimum viable outcome

If everything else slips, these three make a defensible rebuttal:

1. **T1** — merge recall exists → AC #2 substantively complete
2. **T4** — A4 analysis → AC #4 complete
3. **T2 + T3** — oracle-labelled Corpus A incl. the fork pair → AC #1 partially, with an honest limitation on "uncurated"

A3 is then an explicit scoped cut with a written limitation paragraph — which is a legitimate answer to AC #3, not a silent omission.

---

## Expected end state

| Item | Outcome |
|---|---|
| **A1** | Corpus A 400+ pairs, oracle-labelled, T9 scored, fork-pair study. Corpus B if it lands. |
| **A2** | **Merge recall measured.** Rungs 1/2/3/6 on `G_ρ`, all rungs on Corpus A. 4.25% bound resolved. Ablations + OOD done. |
| **A3** | Real feasibility result on 4 neural models, or explicit cut. |
| **A4** | Complete. |

**Four of four AC items addressed** — three substantively, A3 as a scoped feasibility study.

---

## Rules that do not bend, even at speed

1. **Never re-calibrate on evaluation data.** No test-driven repair.
2. **Freeze and hash before running.**
3. **Every rate carries raw counts and an exact interval.** `0/69` is not a result; `0/69 [0, 4.25%]` is.
4. **Never fabricate.** "Not yet run" is acceptable. Keep blocked / deliberately-cut / partial as distinct categories.
5. **Report numbers exactly as produced.** If they disagree with the draft, the draft changes.
6. **Do not modify the corruption harness.** It's frozen and it produced Table 1; changing it invalidates the 300-instance grid.
