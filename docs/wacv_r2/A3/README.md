# A3 — Distributional Equivalence for Neural and Stochastic Skills

**AC required revision #3:** *"Add distributional or soft-similarity support for neural and stochastic tools"*
**Also answers:** R1 W1 — *"The framework explicitly limits skills to classical, deterministic image transformations… deliberately excluding stochastic, neural, or generative vision tools (e.g., Grounding DINO, SAM3, etc.). This restriction… limits the benchmark's immediate applicability to modern agent ecosystems."*

**This is the largest genuinely-new methodological contribution in the revision.** It is also the most cuttable — see §Descoping.

---

## Files

| File | Contents |
|---|---|
| [`01_model_roster.md`](01_model_roster.md) | Which operators to include, and why |
| [`02_method_distributional_equivalence.md`](02_method_distributional_equivalence.md) | The math: distributional signatures, MMD, equivalence testing |
| [`03_experiments_and_metrics.md`](03_experiments_and_metrics.md) | Protocol, metrics, deliverables |

---

## The core problem

The paper's equivalence machinery assumes determinism. `sec/6_methodology.tex:10` handles stochasticity only by seed-fixing:

> *"For stochastic skills, a random seed is part of the input, making the transformation deterministic when conditioned on the seed."*

That works for seeded classical noise. It **does not** work for:
- models whose outputs vary with hardware, batch composition, or non-deterministic kernels
- diffusion operators where the seed→output map is chaotic, so seed-matching is meaningless across two *different* models
- comparing two different models that have no shared seed space at all

**The generalization:** replace the point-valued behavioral signature with a **distribution**, and replace the point distance with a **distributional discrepancy plus a statistical decision rule**.

Critically, this preserves the paper's existing design commitment — precision on non-equivalence, abstain on borderline — rather than introducing a new philosophy. The abstention band was already there (`sec/6_methodology.tex:133`); it just becomes statistically grounded instead of a fixed-width margin.

---

## What "success" means here — A3 is a scoped feasibility study

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §29. The original implied the classical safety guarantee would transfer. **It cannot be established at this sample size**, and claiming it would be an overclaim a confident reviewer can puncture.

A3 does **not** need to make VisCurate work well on diffusion models. It needs to show that:

1. the framework **extends** to stochastic and neural operators with a principled decision rule
2. the extension is **evaluated**, with honest numbers about where it works and where it does not
3. the **boundary of practical certifiability** is reported

**What A3 must not claim:** that the 0.99 non-equivalence precision guarantee transfers. A few independent model families and 30–50 pairs cannot support a claim at that confidence — the interval is far too wide. Report observed counts with exact bounds and let the reader see the limit.

> ⚠ The original success criterion #2 read *"preserves the safety property (no false merges)."* **Removed.** Zero false merges among a handful of hard negatives has a 95% upper bound near 40% and establishes nothing. See [`03_experiments_and_metrics.md`](03_experiments_and_metrics.md) §2.1.

A reviewer asked for the capability to exist and be demonstrated, not for it to be solved. Overreaching here is a bigger risk than underreaching — and A3 is the **first item to cut** if the schedule tightens.

---

## Descoping ladder (if time runs short)

| Level | Scope | Still answers AC #3? |
|---|---|---|
| **Full** | All 6 modality categories, ~16 operators | Yes, strongly |
| **Reduced** | Segmentation + depth only (~8 operators) | **Yes** — cleanest comparators, both non-image outputs |
| **Minimal** | Seeded-stochastic classical ops only (noise, dithering, k-means) | Partially — shows the machinery works but dodges "neural" |
| **Paper-only** | Method section + formalism, no experiments | **No** — do not do this |

*Recommendation:* plan for Full, commit to Reduced. Segmentation and depth have the cleanest comparators (IoU and affine-invariant error), both are genuinely neural, and both appear by name in R1's critique.

---

## Timeline

| Week | Work |
|---|---|
| **W2** | Model roster setup; weights downloaded; adapters written; determinism characterization |
| **W2–W3** | Distributional signature implementation; MMD; bootstrap/TOST gate |
| **W3** | `K`-sweep (sample-size analysis); modality comparators |
| **W3–W4** | Full evaluation on the neural pair set; safety check |
| **W4** | Write method extension + results; **rewrite the scope statements** |

---

## Scope-statement rewrites this forces

Three places currently declare the exclusion and must be updated **together** — a reviewer noticing one stale sentence will assume the extension is bolted on:

| Location | Current | After A3 |
|---|---|---|
| `sec/1_intro.tex:13` | *"leaving stochastic and generative operators for future work"* | two-tier scope: deterministic → exact/perceptual certificates; stochastic/neural → distributional certificates |
| `sec/0_abstract.tex` | implies deterministic image-to-image scope | mention distributional certification |
| `sec/8_discussion.tex:9` | *"Our scope is limited to deterministic classical operations"* | narrow to what genuinely remains out of scope |

---

## Risks

| Risk | Mitigation |
|---|---|
| Model zoo setup consumes the schedule | Use `transformers`/`diffusers` pipelines wherever possible; avoid research repos with bespoke install steps. Budget a full week just for setup. |
| Non-determinism swamps the signal | Characterize *within-model* variance first (§`03` step 1). If a model's self-variance exceeds its distance to a different model, report that — it is a real and interesting negative result about neural tool equivalence. |
| GPU budget | The `K`-sweep tells you the minimum viable `K`. Run it early on a small operator subset before committing to full runs. |
| Reviewers expect SAM3/Grounding DINO **specifically** | R1 named both. Include them or explain the substitution. |
| Comparators are contestable | Use standard, citable metrics per modality (IoU, boundary-F, affine-invariant depth error). Do not invent new ones. |
