# A3 · Model Roster

> **Verify every model identifier against its current model hub page at implementation time.** The IDs below are indicative, not authoritative — hub names get renamed, deprecated, and re-versioned. Do not transcribe them into the paper without checking.

---

## 1. Selection principles

1. **R1 named names.** The review cites *"Grounding DINO, SAM3, etc."* Include those families or explicitly justify the substitution.
2. **Populate the positive class.** Deliberately include near-duplicate pairs, or every relation but `DISTINCT` will have zero support — repeating the exact flaw A2 is fixing in Table 3.
3. **Include hard negatives.** Depth vs. disparity, instance vs. semantic masks — pairs that *sound* interchangeable and are not.
4. **Prefer standard pipelines.** `transformers` / `diffusers` wrappers over research repos with bespoke installs. Setup cost is the main schedule risk.
5. **Span output modalities.** Masks, boxes, depth, and images each need a different comparator (see [`02_method_distributional_equivalence.md`](02_method_distributional_equivalence.md) §4).

---

## 2. Roster

### Tier 1 — Reduced scope (commit to this)

| Category | Operators | Why |
|---|---|---|
| **Promptable segmentation** | SAM 2 (large), SAM 2 (small/tiny), MobileSAM, FastSAM | Named by R1. Mask output → clean IoU comparator. SAM2-large vs. MobileSAM is a genuine near-duplicate pair; large vs. tiny is a genuine `SUBSUMPTION`-flavored pair. |
| **Monocular depth** | Depth Anything V2 (large), Depth Anything V2 (small), MiDaS/DPT, Marigold | Dense continuous output → affine-invariant comparator. DA-V2 vs. MiDaS is a real "same task, different scale convention" hard case. |

**8 operators, 2 modalities, clean comparators, both genuinely neural.** This alone answers AC #3.

### Tier 2 — Full scope

| Category | Operators | Notes |
|---|---|---|
| **Open-vocab detection / grounding** | Grounding DINO, MM-Grounding-DINO, OWLv2 | Named by R1. Box+label output → matched-IoU comparator. Text-prompted, so the prompt is part of `Θ_s`. |
| **Super-resolution / restoration** | Real-ESRGAN (×2, ×4), SwinIR, GFPGAN, CodeFormer | Image→image, **spatial extent changes** — exercises the shape gate and `SUBSUMPTION` across scale factors |
| **Background removal / matting** | BiRefNet, RMBG-2.0, DIS | Near-duplicate cluster; alpha output |
| **Generative image→image** | SD img2img, InstructPix2Pix, ControlNet variants | The hardest case: high-variance, seed-chaotic. **Feature-space MMD only — never pixel `L∞`.** |

### Tier 3 — Seeded-stochastic classical (cheap, include regardless)

Additive Gaussian noise, salt-and-pepper, Perlin/Voronoi generation, dithering, seeded k-means quantization, random-crop-with-seed.

These are cheap, CPU-runnable, and **already partly handled** by the paper's seed-fixing (`sec/appendix.tex` §Behavioral Signatures classes (a) and (c)). They serve as the **validation set for the distributional machinery**: the distributional path must agree with the existing seed-matched path where both apply. That agreement check is a genuine correctness test for the new method and is nearly free.

---

## 3. Designed pair set

As with `G_0`, design the relation graph **before** running anything, and freeze it.

> ⚠ **These are hypotheses, not ground truth** ([`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §30). Asserting that SAM2-small vs. MobileSAM is `PERCEPTUAL`, or that DA-V2 vs. MiDaS is `SEMANTIC-PRESERVING`, is an author expectation. **Scoring P/R/F1 against our own expectations is circular** and a reviewer will say so.
>
> **Required:** freeze the designed relations as *hypotheses*, then establish labels independently by
> - **task-aware expert adjudication** (annotators who know the modality, blind to the verifier), and/or
> - **dataset ground truth** where it exists — e.g. compare both segmentation models against reference masks and derive the relation from task-level agreement rather than asserting it
>
> Report agreement between the frozen hypotheses and the adjudicated labels. Where they differ, the adjudicated label governs, and the disagreement is itself informative about how contestable neural equivalence is.

| Pair | Designed relation | Rationale |
|---|---|---|
| SAM2-large vs. SAM2-large (different seed/run) | `EXACT` or `PERCEPTUAL` | **self-pair sanity gate** — must pass or the harness is broken |
| SAM2-small vs. MobileSAM | `PERCEPTUAL` / `SEMANTIC-PRESERVING` | near-duplicate efficient variants |
| SAM2-large vs. SAM2-tiny | `SEMANTIC-PRESERVING` | same task, quality gap |
| DA-V2-large vs. DA-V2-small | `SEMANTIC-PRESERVING` | same family, capacity gap |
| DA-V2 vs. MiDaS | `SEMANTIC-PRESERVING` | same task, different scale convention |
| **Depth vs. disparity output** | **`DISTINCT`** (hard negative) | reciprocal relationship — *looks* interchangeable, is not |
| **Instance masks vs. semantic masks** | **`DISTINCT`** (hard negative) | same modality, different semantics |
| Real-ESRGAN ×2 vs. ×4 | `DISTINCT` (shape gate) | different spatial extent |
| Real-ESRGAN ×4 vs. SwinIR ×4 | `SEMANTIC-PRESERVING` | competing SR methods |
| BiRefNet vs. RMBG-2.0 | `PERCEPTUAL` / `SEMANTIC-PRESERVING` | near-duplicate matting |
| Grounding DINO vs. OWLv2 (same prompt) | `SEMANTIC-PRESERVING` | competing open-vocab detectors |
| SD img2img (strength 0.1) vs. identity | `SUBSUMPTION`-adjacent | low-strength img2img ≈ near-identity |
| Seeded Gaussian noise vs. itself, matched seed | `EXACT` | **cross-validates distributional vs. seed-matched paths** |

**Target: 30–50 designed ordered pairs**, with the hard negatives flagged for separate reporting exactly as `G_0` does (`sec/6_methodology.tex:141`).

---

## 4. Probes for neural operators

The existing 177-probe battery is built for classical ops and includes degenerate cases (1×1, all-black) that are meaningless or crash-inducing for neural models.

**Build a neural sub-battery, `B_neural`:**
- 40–60 natural images at model-native resolutions
- spanning: single salient object, multiple objects, cluttered scene, low-light, textured/no-object
- **for promptable models**, the prompt is part of `Θ_s` — sweep a small fixed prompt set (e.g. `"the main object"`, `"person"`, `"background"`) and treat prompts as grid points
- **exclude degenerate probes**; record the exclusion and justify it

**State plainly in the paper that `B_neural` is separate from `B`**, and why. Silently swapping batteries between experiments would be a legitimate reviewer complaint.

---

## 5. Compute budget

| Item | Estimate |
|---|---|
| Operators | 8 (Tier 1) to ~20 (Full) |
| Probes | 40–60 |
| Prompt/parameter grid points | 3–5 |
| Seeds `K` | determined by the `K`-sweep; assume 8–16 |
| **Forward passes** | `ops × probes × grid × K` ≈ 8 × 50 × 3 × 16 ≈ **19k** (Tier 1) |

Tractable on a single GPU for Tier 1. Full scope with diffusion operators is the expensive part — **run the `K`-sweep on a small subset first** to fix `K` before committing to full runs, or you will pay for oversampling across every operator.

**Cache aggressively.** Signatures are computed once per (operator, probe, grid point, seed) and reused across all pairs — the same caching argument A4 makes for the classical path. Instrument it so A4 gets the timing.

---

## 6. Checklist

- [ ] Verify all model IDs against current hub pages
- [ ] Stand up Tier 1 (8 operators) first; confirm all load and run
- [ ] Write adapters normalizing each to the skill tuple `⟨k, μ, d, f_s, Θ_s⟩`
- [ ] Build `B_neural` (40–60 probes); document exclusion of degenerate probes
- [ ] Design and **freeze** the 30–50 pair relation graph with hard negatives flagged
- [ ] **Pass the self-pair sanity gate** before interpreting any result
- [ ] Run the `K`-sweep on a subset; fix `K`
- [ ] Include Tier 3 classical stochastic ops as the cross-validation set
- [ ] Instrument for A4 timing
