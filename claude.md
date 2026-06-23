# VisCurate — Implementation Roadmap

*Worked-out plan to start building the project. Derived from CLAUDE.md, VisCurate_Proposal.md, VisCurate_Synthetic_Dataset_Construction.md, viscurate_phase_guide.md, skills_1-100__Examples_Claude.md, AnswersToQuestions.pdf, and the two architecture figures (update_1/2).*
*Drafted 23 June 2026. Codename: `viscurate`. Status: pre-Phase-0, awaiting go-ahead.*

> This roadmap states the dataset, the skill library (the load-bearing piece), and the methods, then lays out a phase-by-phase build. Decisions locked during our consultation are marked **[LOCKED]**. Items still needing a human call are marked **[CONFIRM]**. Where I made a judgment call, the assumption is stated inline.

---

## 0. Decisions locked in consultation (the spine)

| # | Decision | What we chose | Why |
|---|---|---|---|
| D1 | **Skill-library source** | Start from a **100-skill clean base library `L0`**, derived from your hand-authored blueprint. Its `known_equivalences` become the **designed relation graph `G0`**. | Reuses your work, gives a clean base to corrupt from, keeps "designed structure" separate from "injected pollution." |
| D2 | **Experiment shape** | Generate **corrupted libraries `L_ρ`** from `L0` at pollution rates ρ ∈ {10%,20%,…,100%}; show pollution-vs-performance and that output-grounding survives where text judges fail. | Your core experiment. The dial-able noise curve is the headline application result. |
| D3 | **Defect composition** | **Single-defect-per-skill** for the main curve (one of 7 types per corrupted skill, fixed composition), **plus a mixed/realistic run** where defects co-occur. ≥5 seeds per (ρ, composition). | Interpretable ablations + realism + error bars (CVPR expects variance). |
| D4 | **Venue / scale** | Build at **CVPR A\*** rigor (5-layer dataset, error bars, realism audit, construct-validity study), but **validate on a small pilot first** (100 skills, 200–500 probes) before scaling. | Confirms the idea works cheaply before spending compute. Architecture is seed/config-parameterized so scaling needs no rework. |
| D5 | **Compute / dev env** | Develop under **WSL2** (you have it, 32 GB RAM). GPU = **RTX 3050 6 GB** locally for the pilot; scale to 5090/cloud only if needed. | Comparators are inference-only and small; the pilot fits in 6 GB. The 3050 is not the bottleneck — agent LLM latency is. |
| D6 | **Sandbox** | **Lightweight isolation now** (subprocess + wall-clock timeout + resource limits) since all 100 skills are human-authored/trusted; **hardened sandbox (network namespace, chroot/container) deferred to Phase 6** when agent-generated skills appear. | Matches CLAUDE.md §7's explicit allowance to restrict execution to the trusted starter set first. |
| D7 | **Models** | **Multi-model**: enumerate locally-installed Ollama models for the agent and text baselines; comparators = **LPIPS (AlexNet) + DINO ViT-B/16 + CLIP ViT-B/32**; install additional free models as useful. Try several and compare. | You want to see which models behave best; cost-free local-first iteration. |

**Two experiments ride on this machinery — neither can be dropped:**

1. **Equivalence benchmark** (the contribution): per-pair relation prediction → precision/recall/F1 per relation, per judge track (text-only / embedding / LLM-on-text / output-grounded), and the **divergence table** (rate at which text judges and the output judge disagree, by true relation).
2. **Curation application** (the payoff): run the curation agent on `L_ρ`, measure **downstream task success vs ρ**, library compression, and action cost as a Pareto front; show output-gated curation beats text-gated curation.

---

## 1. The skill library (most important)

### 1.1 What we have and what we build

Your `skills_1-100` file is **natural-language pseudocode**, not runnable code, with hand-written metadata: `known_equivalences`, `is_buggy`, `is_dead`. We convert it into the project's executable substrate in three moves:

1. **Implement** each of the 100 as a deterministic Python `fn(image, params, seed) -> image` over PIL/OpenCV/numpy, wrapped in the `Skill` model (CLAUDE.md §2). The pseudocode maps almost 1:1 to OpenCV calls already named in the blueprint.
2. **Clean** `L0`: set `is_buggy=False` everywhere (fix the two planted bugs — skill 30 `convert_hsv_to_rgb` and skill 55 `sharpen_laplacian_kernel`); the planted cosmetic-rename duplicate (skill 92 `gamma_ray_despeckle` ≈ skill 53) and the four `is_dead` skills (20, 74, 83, 87) are **not** baked into `L0` — "dead" is a property of the query layer (§3), and duplicates/bugs are **injected later** at controlled ρ. The result is a defect-free, fully-correct base.
3. **Lift the relations** in `known_equivalences` into the **designed relation graph `G0`** (§2.3). These are legitimate library *design* (e.g. `crop_center ⊑ crop_bounding_box`, `dilate` ⟂ `erode` complementary), not pollution.

The 100 skills already span the seven operation families the method needs: geometric/canvas (1–25), color/exposure/contrast (26–50), signal/blur/edges/morphology (51–75), and masks/reconstruction/synthesis (76–100). That diversity is exactly what produces the relational clusters the benchmark depends on.

### 1.2 The `Skill` model

```
Skill:
  id: str                  # stable, never reused (e.g. "blur_gaussian_core_v1")
  name: str
  description: str         # text baselines ONLY; output-grounded path must never read it
  fn: Callable             # (np.ndarray, params: dict, seed: int) -> np.ndarray, deterministic
  params_schema: dict      # typed, with defaults + valid ranges
  metadata: dict           # family tag, provenance, trusted: bool,
                           # internal-only labels (is_buggy/is_dead) — never shown to the agent
```

Architectural rule enforced in code: the comparator modules **import nothing that touches `description`**. The text/embedding baselines live in a separate package that may read it. This separation is the whole point — it must be a hard boundary, not a convention.

### 1.3 The output canonicalization contract (an engineering gap I'm closing now)

Skills produce heterogeneous outputs — grayscale (1-channel), RGBA, binary masks, edge maps, shape-changing geometric ops. Pixel/LPIPS comparison needs a fixed rule, versioned alongside the probe manifest (`equivalence/canonicalize.py`):

- **dtype/range:** coerce to `float32 ∈ [0,1]` for metric computation; keep a `uint8` copy for hashing.
- **channels:** 1-channel → replicate to 3; RGBA → composite over a fixed mid-gray background for RGB comparison *and* track alpha separately (alpha-IoU / alpha L∞).
- **shape:** identical shape is a **precondition for EXACT/PERCEPTUAL**. Differing shapes can still be SEMANTIC_PRESERVING/SUBSUMPTION/COMPLEMENTARY/DISTINCT; the semantic comparator resizes to a common size before feature extraction (DINO/CLIP are resize-tolerant).
- **binary masks:** compared via exact match + IoU, not LPIPS.

This contract is part of the verifier (which only ever sees outputs), and it is itself a versioned object so labels are reproducible.

### 1.4 Determinism handling

Determinism given `(image, params, seed)` is required so "same output" is decidable. Per-skill classification:

- **Deterministic:** most geometric/color/filter ops — bit-reproducible.
- **Seeded-stochastic:** noise (57, 58, 72), random crop (24), pixel-sort (94), salt mask (87) — accept an explicit `seed`, deterministic at fixed seed; compare at matched seeds.
- **Precision-sensitive:** FFT ops (70, 71, 74) — compare with PERCEPTUAL tolerance, not EXACT; mark `metadata.precision_sensitive=True`.
- **Platform-sensitive:** k-means palette (99) via `cv2.kmeans` — reproducible within a platform but not guaranteed across; mark `metadata.platform_sensitive=True`, compare within-platform only, document it.

No generative/diffusion skills in v1 (CLAUDE.md §2, proposal §2). [CONFIRM] this stays true for the whole CVPR pass.

---

## 2. The dataset (5-layer construction, anchored on the 100)

Adapted from `VisCurate_Synthetic_Dataset_Construction.md`, with `L0` = the cleaned 100. Family of datasets indexed by `(ρ, composition c, seed)` — never a single static file. Everything the dataset asserts as truth is decided **before any equivalence metric runs**; the metrics are what's under test.

```
 Layer A  Audited primitive ops      ← the trusted atoms behind each skill's fn
 Layer B  Clean base library L0 (100) ───►  designed relation graph G0
            freeze reference outputs                  │
 Layer C  Probe battery P (versioned) ───────────────┤
            inject defects (rate ρ, composition c)    │
 Layer D  Corrupted library L_ρ ──►  updated graph G_ρ + corruption log + ideal-action key
 Layer E  Task / usage layer T ──►  query relevance + usage frequency (Dead-Skill, utility)
```

### 2.1 Probe battery `P` (Layer C)

Versioned, deterministic, content-hashed; per-image manifest entry (`probe_id`, `sha256`, domain, channel/format, resolution, source, **license**, notes). Diversity axes that must be covered (a defect is detectable only if the battery exercises it):

- **Domain:** natural photos, documents/screenshots, textures, synthetic gradients, degenerate cases (all-black, all-white, single-color, 1×1, very-high-res).
- **Channel/format:** RGB, **RGBA**, grayscale, 16-bit, palettized — so domain-scoped bugs have somewhere to fire.
- **Signal:** low/high frequency, color profile, aspect ratio.
- **Parameter sweeps:** grids over each parameterized skill's declared range (blur(k) and box(k) agree at small k, diverge at large k).

Pilot target **~200 base images** → expands to thousands of `(skill, image, param)` evaluations after sweeps; scale to **500** for the CVPR run. License-clean sources only (COCO CC BY, OpenImages, self-generated synthetics). **[CONFIRM] face-domain source** — recommend synthetic faces (StyleGAN2 / SD with documented license), avoiding real PII (CLAUDE.md §8).

**Freeze the oracle:** before any corruption, execute every `L0` skill over `P` + sweeps, hash and store all outputs. This reference oracle (a) confirms corruption took effect and (b) scores verifier/agent — it is **never** used to assign relation labels.

### 2.2 The seven corruption types (Layer D)

Each has an injection mechanic, a logged ground-truth label delta, a QA assertion confirming the defect is "active," and a stated verifier-detectability — the asymmetry that justifies the verifier/agent split.

| # | Type | Mechanic | Verifier catches? | Stresses |
|---|---|---|---|---|
| 1 | Implementation Bug | mutate `fn` (off-by-one kernel, wrong axis, missing clip, sign swap) | **Yes** (diverges from oracle) | verifier functional-failure detection |
| 2 | Metadata Mislead | keep `fn`+schema, rewrite name/description/tags | **No** (outputs unchanged) | text baselines |
| 3 | Exact/Perceptual Duplicate | clone under new id (copy `fn` = EXACT, re-implement = PERCEPTUAL) | **Yes** | redundancy detection, `merge` |
| 4 | Subsumption Pair | add a specialization/generalization | **Yes (directional)** | subsumption search, `parameterize` |
| 5 | Parameter-Schema Bug | keep `fn`, corrupt schema (range/default/type) | **Partial** | agent–executor interaction |
| 6 | Domain-Scoped Bug | `fn` correct on RGB, broken on a domain (RGBA/grayscale) | **Only if battery covers it** | probe diversity, conditional verification |
| 7 | Dead Skill | add a correct skill no query needs | **No** (needs usage) | agent utility reasoning |

The three "No/Partial/conditional" rows are the dataset's core argument made concrete: Metadata Mislead and Dead Skill are output-invisible and force the agent's distinct job. QA assertion for those types confirms outputs are **unchanged**, the opposite of Type 1.

**ρ semantics (pinned down):** ρ = fraction of `L0`'s skills corrupted. At ρ=30% with 100 skills, 30 skills are corrupted; each gets exactly one defect (single-defect mode), drawn from a fixed **composition vector `c`** over the 7 types. Run ≥3 compositions (uniform, duplicate-heavy, metadata/text-heavy) and a mixed mode where defects co-occur. `G_ρ` and the ideal-action key are derived deterministically from `G0` + the corruption log — no relabeling at any ρ.

### 2.3 The relation graph `G0`/`G_ρ` (Layer B)

Labeled graph over `(A,B)` pairs, 6-way (+ subsumption direction). `O(N²)` pairs are overwhelmingly DISTINCT, so we keep **dense relational clusters** (blur, edges, quantize, geometric, color, morphology, mask) where the non-DISTINCT classes concentrate, surrounded by a DISTINCT sea with **engineered hard negatives** on cluster boundaries (`blur_gaussian` vs `blur_box`, `posterize` vs `palette_reduce`, `rotate(90)` vs `flip+transpose`). Enforced properties: subsumption is a DAG; EXACT is transitive (closure checked); PERCEPTUAL/SEMANTIC are symmetric but **not** assumed transitive.

### 2.4 Task/usage layer `T` (Layer E)

Natural-language image tasks, each annotated with the ground-truth set of skills that satisfy it; a synthetic (Zipfian) usage log; query→skill relevance for the `retrieve` action. This is what makes "should we merge?" diverge from "can we merge?" (a verified SUBSUMPTION the agent *declines* because the specialized skill is heavily used) and what makes Dead-Skill removal a utility decision the verifier can't see.

### 2.5 Ground-truth bundle + splits

Each `(ρ,c,seed)` instance ships: `G_ρ`, ideal-action key, corruption log, reference oracle, battery manifest, task/usage tables, and a reproducibility manifest (generator version, pool hash, battery hash, seeds, answer-key hashes). **Human verification** for PERCEPTUAL borderline and SEMANTIC_PRESERVING (intrinsically judgment-laden) with reported inter-annotator agreement (κ). **Leakage control:** splits are **cluster-level disjoint** (a blur-cluster skill never appears in both calibration and test); calibration/validation/test frozen before any metric runs.

---

## 3. The methods

### 3.1 Output-grounded verifier (the engine)

Hierarchical pipeline over the probe battery + sweeps, stopping at the first relation that holds. Reads outputs only.

1. **EXACT** — content hash + max per-pixel diff ≤ ε on every probe → **Merge**.
2. **PERCEPTUAL** — EXACT fails, LPIPS ≤ τ_perceptual on every probe (SSIM as a structural cross-check, not a substitute) → **Merge after validation**.
3. **SEMANTIC_PRESERVING** — both fail, DINO/CLIP feature distance ≤ τ_semantic → **Unify/parameterize, conditional** (leans on human labels).
4. **SUBSUMPTION (A ⊑ B)** — directional; **Parameterize** (fold special case into general, never reverse).
5. **COMPLEMENTARY** — orthogonal/composable, fails equivalence+subsumption but not noise → **Keep both**.
6. **DISTINCT** — residual; the relation that blocks silent merges → **Reject merge**.

**Subsumption mechanics (the second engineering gap I'm closing):** A ⊑ B iff for every param setting of A there exists a param setting of B reproducing A's output within EXACT/PERCEPTUAL tolerance across the whole battery. For fixed-param specializations that expose no params (e.g. `rotate_90 ⊑ rotate_canvas_degrees`), A is a single point and we only search B's grid for a match. Coarse grids by default (config knob), early-exit on the first failing probe. Direction is recorded; near-miss specializations (param slightly off → **not** subsumed) are planted as hard cases.

**Threshold calibration:** ε, τ_perceptual, τ_semantic are **calibrated** on the human-labeled validation split (not guessed, not gradient-learned) to hit a target operating point — **high precision on non-equivalence** (a false merge is worse than a missed compression) — and recorded in config with date + split path. The calibration split is cluster-disjoint from test.

### 3.2 Verifier / agent split + gating

The single most important architectural decision (proposal §6): the **verifier** answers "what *is* the relation?" (a fact about outputs; never mutates the library; blind to usage). The **agent** answers "what *should* be done?" (a repository decision over the whole library, usage, history, budget) and chooses among `add, remove, modify, retrieve, merge, split, parameterize, end`.

Gating: `merge`, `split`, `parameterize` (structural edits that change functional coverage) **cannot proceed without a certifying relation from the verifier**. A rejected merge returns a structured reason (`{"rejected": true, "reason": "DISTINCT — max pixel diff 47.3, LPIPS 0.31", "alternatives": ["parameterize","keep_separate"]}`). `add`/`modify` produce `trusted=False` skills (blocked from execution until reviewed); `modify` to an `fn` re-triggers verification of any relation that skill was in; `remove` consults usage stats (Dead-Skill).

### 3.3 Downstream evaluation

A separate, simpler **solver agent** is given a library and a held-out query; it retrieves skill(s), executes them, returns an image. Correctness = perceptual match to reference (LPIPS ≤ τ) **plus** task-specific predicates (e.g. grayscale → all channels equal; resize → exact shape). Query images are **disjoint from probes**; dev/test query splits are stratified by noise type and **disjoint in skills and queries**.

### 3.4 Baselines + the four studies

Baselines: name-match dedup, **description-embedding cosine dedup** (the direct strawman), LLM-judge-on-descriptions, accumulate-only, no-curation, random edits, and an **oracle used only as an upper bound** (never a scoring target / string-similarity reference).

- **Study 1 — Equivalence F1** per judge track on human-verified labels; output-grounded should win decisively on DISTINCT (the same-description/different-output pairs).
- **Study 2 — Curation Pareto** over (downstream success ↑, compression, action cost) with CIs from multiple seeds.
- **Study 3 — Construct validity** — Pearson/Spearman between an intrinsic curation score and downstream success across many libraries (validate the proxy, don't assume it).
- **Study 4 — Vision-matters ablation** (the spine) — identical pipeline, merge/parameterize gated by text-embedding cosine vs by the output verifier; text-gating's erroneous merges degrade downstream performance on blur-sensitive queries.

---

## 3.5 The equivalence & curation engine — detailed design (optimal approach)

This is the technical core. It answers, end to end, **(1)** how a pair `(A, B)` is assigned exactly one of the six relations from executed outputs, and **(2)** how the agent turns that relation into a curation action. Where there is a real design choice, the chosen option is stated with its justification and the rejected alternative — this is the "best and optimal approach" the project commits to.

### 3.5.1 The comparison primitive

A **skill instance** is `σ = (skill, param_binding)`. Executing σ over probe battery `P` yields a canonicalized output set `O_σ = {ô_σ(p) : p ∈ P}` (canonicalization per §1.3). The atomic comparison computes a **per-probe distance** at three levels, then aggregates:

```
compare(σA, σB, P):
    for p in P:
        if shape(ôA(p)) != shape(ôB(p)):  d_pix[p] = ∞          # shape gate
        else:                             d_pix[p] = max_abs_pixel_diff(ôA(p), ôB(p))   # L∞, 0..255
        d_lpips[p] = LPIPS_alex(ôA(p), ôB(p))                   # ~0..1
        d_dino[p]  = 1 - cos(φ(ôA(p)), φ(ôB(p)))                # DINO/CLIP feature cosine distance
    return aggregate(d_pix), aggregate(d_lpips), aggregate(d_dino)
```

**Aggregation rule — a deliberate choice:**
- **EXACT and PERCEPTUAL use the worst probe (`max` over P).** Equivalence is a universally-quantified claim — "indistinguishable on *every* probe." A pair matching on 199 probes and diverging on 1 is **not** mergeable; that one probe is a silent-merge bug in waiting. Worst-case aggregation is what makes the verifier conservative in the safety-critical direction.
- **SEMANTIC_PRESERVING uses a high quantile (p90) plus the mean**, not max. "Same semantic transformation" is a distributional claim that tolerates a few hard probes; a pure max would make it unattainable. We require the p90 distance below `τ_sem` with the mean well below it.
- *Rejected alternative:* mean for all levels — it lets a few catastrophic probes hide behind many easy ones, which is exactly the failure mode we exist to expose.

### 3.5.2 Parameter handling — the part most designs get wrong

A and B rarely share a parameter schema, so "run both and compare" is underspecified. Three regimes, selected per pair:

1. **Fixed-vs-fixed** — both unparameterized or compared at defaults: a single `compare` call.
2. **Matched sweep** — testing whether two *parameterized* skills realize the same operation (e.g. `blur_gaussian_core` vs `blur_box_uniform`): evaluate both over a **matched grid along a shared semantic axis** (here kernel size `k ∈ {3,5,7,9,…}`) and require the relation to hold **at every grid point**. This is the headline case: the two agree at small `k` and diverge at large `k`, so the matched-sweep worst-case correctly returns DISTINCT where a default-only check would wrongly merge them. The shared axis and its mapping live in a config artifact (`param_alignment`), never hard-coded.
3. **Asymmetric search** — subsumption (§3.5.4).

When no sensible shared axis exists, the pair cannot be EXACT/PERCEPTUAL across a sweep and falls through to the semantic/complementary/distinct stages on default bindings. The `param_alignment` map is small and auditable; the *absence* of an alignment is itself informative (the pair is likely DISTINCT or COMPLEMENTARY).

### 3.5.3 The decision pipeline (stop at first match) + an abstention band

```
classify(A, B):
    # Stage 0 — cheap output-based screening already selected this candidate (§3.5.6)
    dpix, dlpips, ddino = compare(A, B over matched grid)          # worst-case for L1/L2
    if dpix  ≤ ε        and not in_band:               return EXACT
    if dlpips ≤ τ_perc  and SSIM ok and not in_band:   return PERCEPTUAL
    if subsumption_search(A, B) certifies direction:   return SUBSUMPTION(dir)
    if ddino_p90 ≤ τ_sem:                              return SEMANTIC_PRESERVING   # → human-verified slice
    if complementarity_test(A, B):                     return COMPLEMENTARY
    return DISTINCT
```

**Order rationale:** cheapest-and-strictest first (a hash is free; LPIPS is one forward pass; subsumption search and composition are expensive). Subsumption is checked *before* SEMANTIC because a true subsumption is a stronger, directional, actionable relation that we don't want masked by a loose semantic match.

**Abstention band (chosen).** Around each threshold define a calibrated margin `[τ(1−δ), τ(1+δ)]`. Pairs landing inside are returned **UNCERTAIN** with the offending distance rather than forced into a class. This (a) protects the precision-on-non-equivalence operating point, (b) gives a clean review queue for borderline PERCEPTUAL/SEMANTIC pairs — exactly the ones the dataset doc routes to human verification, and (c) is honest about calibration noise. `δ` is calibrated, not guessed.

### 3.5.4 Per-relation detectors

**EXACT** — pre-check SHA-256 of canonicalized outputs; all-match → EXACT with no metric. Else worst-case L∞ ≤ ε (ε ≈ 1/255, rounding only). Catches exact duplicates, including re-implementations that are bit-identical.

**PERCEPTUAL** — EXACT fails; worst-case LPIPS(AlexNet) ≤ τ_perc with an SSIM structural floor as cross-check (guards LPIPS blind spots). Catches numerical-precision variants (float32/64, PIL-vs-cv2 resize with matched interpolation, equivalent kernel normalizations, JPEG-q100 round-trip).

**SUBSUMPTION (A ⊑ B), directional.** For each binding `a` in A's grid, search B's grid for `b` with `compare(A@a, B@b)` EXACT-or-PERCEPTUAL on all probes. If every `a` has such a `b` but not conversely, certify `A ⊑ B` and store the binding map; if both directions hold it is EXACT/PERCEPTUAL, not subsumption. Coarse grid + early-exit on the first failing probe; grid resolution is a config knob. Your blueprint already plants these (`crop_center_percentage ⊑ crop_bounding_box`; the fixed-angle rotations ⊑ `rotate_canvas_degrees`; `linear_contrast_stretch ⊑ contrast_stretching_percentile`). Plant **near-miss** specializations (fixed param slightly off) as hard negatives that must return *not* subsumed.

**SEMANTIC_PRESERVING** — EXACT/PERCEPTUAL/SUBSUMPTION all fail; DINO ViT-B/16 feature cosine distance (quantile-aggregated) ≤ τ_sem, optionally with CLIP as a second, more-conservative view (take the larger distance). This is the only relation **not** settled by construction; the metric's job is to *propose* it for human confirmation, never to decide alone. Your blueprint is rich here: Sobel/Scharr/Prewitt/Canny edges, BT.601-vs-BT.709 grayscale, the two inpainting algorithms, gamma-vs-exposure-stops, global-vs-CLAHE equalization.

**COMPLEMENTARY (detector specified here, since the docs leave it abstract).** A pair is complementary if it operates on **disjoint image aspects** and composes meaningfully. Operational test — all three must hold:
1. **Both non-trivial** — each of A, B differs from identity on the battery (rules out a no-op).
2. **Not equivalent/subsuming** — failed every stage above.
3. **Approximate commutation / disjoint effect** — `D(A(B(x)), B(A(x)))` is PERCEPTUAL-small across probes: order does not matter, the hallmark of orthogonal ops (geometry × color, e.g. `rotate_canvas_degrees` × `saturation_gain_multiplier`; or `morphology_dilate` ↔ `morphology_erode` as the blueprint labels).

This requires the verifier to **execute compositions** — a small added capability. Where composition is shape-incompatible or order clearly matters, the test fails and the pair is DISTINCT. *Rejected alternative:* declaring COMPLEMENTARY from metadata/family tags — rejected because it smuggles text back into the output-grounded path, violating the project's load-bearing-modality commitment.

**DISTINCT** — the residual; explicitly includes the engineered hard negatives (`blur_gaussian` vs `blur_box`, `posterize` vs `palette_reduce`, `bicubic` vs `nearest` resize, reflect-vs-replicate border). This is the relation that blocks silent merges.

### 3.5.5 Threshold calibration (operating point)

`ε, τ_perc, τ_sem`, and the abstention `δ` are calibrated on the **human-labeled validation split** to maximize **precision on non-equivalence** (a false merge costs more than a missed compression) subject to a recall floor. Calibration is cluster-disjoint from test and frozen before any test metric runs (§2.5). All values recorded in `configs/` with date, split hash, and model-checkpoint IDs. No threshold is a literal in code.

### 3.5.6 Scaling: output-based candidate generation (a correctness point, not just speed)

Full verification executes skills over the whole battery — too expensive for all `O(N²)` pairs (100 skills → ~5k pairs at the pilot, far more after sweeps). The naive fix, "only verify pairs whose *descriptions* embed closely," **reintroduces the exact text bias the paper attacks** and would miss the different-description/same-output redundancy that is half the contribution. Chosen design:

- Compute, once per skill, a cheap **output fingerprint** on a small fixed *screening* sub-battery (8–16 probes): a low-dim perceptual hash + mean DINO feature.
- ANN/cluster over fingerprints to propose candidate pairs; **always also include** same-family pairs and the engineered hard negatives so the boundary cases are never skipped.
- Run **full output-grounded verification only on candidates**; the rest default to DISTINCT.

Because the fingerprint is *output-based*, two skills with unrelated descriptions but identical behavior still collide and get verified — the redundancy text-based pruning misses. Cache every certified relation keyed by `(id_A, params_hash_A, id_B, params_hash_B, battery_version)`; invalidate on any `modify` to a skill's `fn`.

### 3.5.7 From relations to curation (the agent loop)

The verifier returns **facts**; the agent makes **repository decisions**. The loop:

```
while not done and budget remains:
    observe(library summaries [NO internal labels], usage stats, action history)
    propose action ∈ {add, remove, modify, retrieve, merge, split, parameterize, end}
    if action is structural (merge / split / parameterize):
        rel = verifier.classify(A, B)                  # hard gate
        if rel licenses the action:  apply
        else:                        reject_with_reason(rel, alternatives)
    else:
        apply   # remove consults usage; modify→fn re-verifies affected relations
    log(action, outcome, verifier_result, size_before, size_after)
```

**Relation → action map** (what is *permitted*; the agent still decides what is *desirable* using usage and budget):

| Relation | Permitted structural action | Agent's usage / quality gate |
|---|---|---|
| EXACT | merge to one canonical | keep the faster / better-documented impl |
| PERCEPTUAL | merge after validating the bound holds at param extremes | keep the more robust / efficient impl |
| SEMANTIC_PRESERVING | parameterize/unify **or** keep both | unify only if the variation is a useful knob; never silently collapse |
| SUBSUMPTION (A⊑B) | parameterize (fold A into B) | **decline if A is heavily used** |
| COMPLEMENTARY | keep both | optionally annotate as composable |
| DISTINCT | keep separate | — |

**Two things the docs leave implicit, made explicit here:**
- **Bug and dead-skill detection are not pure pairwise equivalence.** An Implementation Bug surfaces either as DISTINCT between a buggy variant and a correct sibling (when a duplicate exists) or as **downstream task failure** via the query layer — the agent has no clean oracle at curation time. A Dead Skill is invisible to the verifier and removed only on **utility** grounds (zero query relevance plus low usage). This is precisely why the verifier/agent split exists.
- **Rejection feedback is structured and actionable** — e.g. `{"rejected": true, "relation": "DISTINCT", "reason": "max L∞ 47.3 @ probe tex_007, LPIPS 0.31", "alternatives": ["keep_separate","parameterize"]}` — so a rejected `merge(blur_gaussian, blur_box)` becomes a deliberate keep-separate decision, not a retry loop.

### 3.5.8 What the equivalence engine can and cannot catch (honest mapping)

| Corruption type | Caught by the equivalence engine? | By what |
|---|---|---|
| Exact / Perceptual Duplicate | **Yes** | EXACT / PERCEPTUAL → merge |
| Subsumption Pair | **Yes (directional)** | subsumption search → parameterize |
| Implementation Bug | **Only vs a correct sibling**, else downstream | DISTINCT-from-sibling / task failure |
| Domain-Scoped Bug | **Only if the battery covers that domain** | worst-probe divergence on the domain |
| Parameter-Schema Bug | **Partial** | passes at valid params, fails when the schema is trusted in use |
| Metadata Mislead | **No** | text channel only — the agent's job |
| Dead Skill | **No** | utility / usage — the agent's job |

This table *is* the architectural argument: the verifier is a precise instrument with a **defined blind spot**, and the agent covers that blind spot with usage and text reasoning.

### 3.5.9 Metrics for the equivalence stage

Per-relation precision / recall / F1 and a 6×6 confusion matrix, **per judge track** (name-match, embedding-cosine, LLM-on-descriptions, output-grounded, hybrid). The headline **divergence statistic**: fraction of pairs where the text track and the output track disagree, broken down by *true* (injected) relation — with the **hard-negative slice reported separately**, since that is where text judges fail and the contribution lives. Also report the **abstention rate** and **precision-on-DISTINCT** (the safety-critical number: how often the verifier wrongly licenses a merge).

### 3.5.10 Optimal-approach summary (decisions locked in this section)

1. Worst-probe aggregation for EXACT/PERCEPTUAL; p90+mean for SEMANTIC.
2. Matched-sweep evaluation for parameterized pairs via an auditable `param_alignment` map.
3. Hierarchical stop-at-first pipeline (EXACT → PERCEPTUAL → SUBSUMPTION → SEMANTIC → COMPLEMENTARY → DISTINCT) with a calibrated **UNCERTAIN abstention band**.
4. COMPLEMENTARY detected by non-triviality + approximate commutation (the verifier executes compositions) — never by metadata.
5. SUBSUMPTION by directional grid search with near-miss hard negatives.
6. **Output-based** candidate generation (screening fingerprints), never description-based — the load-bearing modality is preserved even inside the speed optimization.
7. Verifier supplies facts; the agent applies the relation→action map with usage gates and structured rejection feedback.
8. An explicit blind-spot table (bugs / metadata / dead) drives the verifier/agent division of labour.

These slot into **Phase 3** (comparators, subsumption, calibration, candidate generation) and **Phase 6** (the agent loop, gating, composition capability for COMPLEMENTARY); they are exercised against injected ground truth in **Phase 4** and **Phase 8**.

---

## 4. Phase-by-phase build (the roadmap)

Order is strict; each phase ends with passing tests + a short written summary of assumptions. Durations assume part-time solo work and include the pilot-first discipline (validate on small scale, then scale config knobs).

| Phase | Deliverable | Exit criteria | Pilot focus |
|---|---|---|---|
| **0 — Scaffold** | `viscurate/` repo, `pyproject.toml`, ruff+mypy(strict)+pytest CI, `config.py` (Pydantic-validated YAML), `rng.py` (explicit seeds, no global state), structlog JSON logging | `pip install -e ".[dev]"`, `pytest`, `mypy src/`, `ruff check` all clean; every subpackage imports | — |
| **1 — Skills + harness** | `Skill` model + registry (JSON-serializable); **lightweight sandboxed executor** (subprocess + timeout + rlimits, `trusted` flag, exec log); implement the 100 as deterministic `fn`s; the canonicalization contract | each skill runs on a 256² probe and returns a valid array; same `(img,params,seed)` → byte-identical; timeout skill returns error not hang; `trusted=False` → BLOCKED; registry round-trips | implement ~25–30 first (geometric+blur+color) to exercise the harness, then the rest |
| **2 — Probe battery** | `build_probes.py`, `data/probe_images/` + `manifest.json` (sha256, license), coverage across all diversity axes; freeze reference oracle | builder reproducible (same seed → same hashes); ≥N per domain + each degenerate case; no `license=unknown` | 200 images; verify oracle freeze + hashing |
| **3 — Comparators + taxonomy** | exact (hash+ε), perceptual (LPIPS, SSIM cross-check), semantic (DINO ViT-B/16, CLIP optional), subsumption search; taxonomy engine; calibration procedure | hand-built EXACT/PERCEPTUAL/DISTINCT synthetic pairs classified correctly; thresholds calibrated on held-out labels and written to config | confirm LPIPS+DINO fit in 6 GB (load one at a time; batch-extract features once) |
| **4 — Equivalence benchmark** ⚠️ | candidate pairs incl. hard cases; auto-labels; text baselines; **human-verify** SEMANTIC/SUBSUMPTION subset; the **divergence table + figure** | divergence pattern matches hypothesis (text judge merges `blur_gaussian`/`blur_box`; output judge says DISTINCT) | **CRITICAL REVIEW CHECKPOINT** — get reviewed before Phase 5. If no divergence, stop and understand why. |
| **5 — Corruption generator** | the 7 injectors with per-type QA assertions; emit `L_ρ`, `G_ρ`, corruption log, ideal-action key over a `(ρ,c,seed)` grid | same seed → byte-identical library; expected per-type counts; buggy skills measurably wrong vs oracle | the graded ρ-series (10→100%) you described |
| **6 — Curation environment** | state representation, action API (`add/remove/modify/retrieve/merge/split/parameterize/end`), verifier gating + rejection-with-reason, agent adapter (Ollama multi-model; Claude API optional), action logging; **hardened sandbox review** | `merge(blur_gaussian, blur_box)` rejected with LPIPS in the reason; exact-duplicate merge approved; `end()` clean; untrusted skill → BLOCKED; actions logged | this is where agent-generated code appears → harden the sandbox here |
| **7 — Query stream + downstream eval** | `Query` objects (instruction, input, reference, predicates), solver agent, correctness scoring, stratified disjoint dev/test | solver fails on some dev queries with the noisy library; succeeds more on a hand-curated one; no probe image appears in any query | confirms Dead-Skill queries have zero retrieval |
| **8 — Metrics, baselines, studies** | the four studies; all baselines; Pareto front; construct-validity correlation; vision-matters ablation; mean ± CI across seeds | every number from a real run with a manifest; Pareto figure shows all methods | start on 3050; scale to 5090/cloud only if a sweep is too slow |
| **9 — Experiment runner + paper artifacts** | per-experiment YAML configs (git SHA, seeds, model versions, battery SHA, thresholds); run manifests; one-command table/figure generators; realism audit report | every paper number traces to a manifest; no placeholders; licenses documented; one-command repro in README | — |

**Where to start:** Phase 0 + the first ~25–30 skills of Phase 1. That alone validates the harness, determinism, and serialization with zero ML dependencies.

---

## 5. Honesty, reproducibility, security (non-negotiable, from CLAUDE.md §6–9)

- **Never fabricate results.** `results/` and every table/figure come from a real run with a committed manifest. "Not yet run" is an acceptable answer; invented numbers are not — not even as placeholders or in docstrings.
- **Verify third-party APIs against current docs before use** (Pillow, OpenCV, `lpips`, `scikit-image`, `timm`/`torch.hub` DINO, `open_clip`, Anthropic SDK). Pin exact versions only after a working install. Don't trust signatures from memory.
- **Sandbox is security-sensitive.** Agent-generated skill execution stays **blocked pending hardened-sandbox review** (network namespace, restricted FS, CPU/mem caps, hard timeout). No `eval`/`exec` of skill code in the main process. Validate params against schema before execution.
- **No secrets / no PII.** `ANTHROPIC_API_KEY` via env var only; license-clean probe images; synthetic faces preferred.
- **Deterministic by default.** Seed numpy/torch/random/skill RNG; record seeds in manifests.

---

## 6. Open items still needing your call [CONFIRM]

These don't block Phase 0 — I'll proceed with the stated default and flag in summaries — but flagging now so they're not surprises:

1. **Face-domain probe source** — recommend synthetic (StyleGAN2/SD, documented license). Default if you don't object: include a small synthetic-face set; otherwise drop faces from v1.
2. **16-bit / palettized domains** — mandatory core or extension? Default: include as a small mandatory slice (cheap, and they're where domain-scoped bugs hide).
3. **Citations** — `SkillClone (arXiv:2603.22447)` and `SkillBrew (arXiv:2605.29440)` are 2026 IDs I can't verify from memory. I'll verify them against arXiv before you lean on the related-work positioning. The proposal already flags refs as "to be finalized."
4. **Annotator pool** for the SEMANTIC slice (size, whether any IRB-style review is needed) — affects κ reliability.
5. **Agent action/compute budget** per curation episode (caps the action-cost axis of the Pareto).
6. **Generative skills** — confirmed out of v1; confirm they stay out for the whole CVPR pass.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **No divergence appears** (text judge ≈ output judge) — kills the premise | low–med | Phase-4 checkpoint *before* building curation; engineered hard negatives guarantee at least the constructed divergence; if real divergence is small, report honestly and analyze why. |
| Hand labels in the blueprint are wrong/inconsistent | med | Treat them as *design intent*; re-certify via the verifier + human spot-check; the answer key for experiments comes from **injected** truth, not the hand labels. |
| Cross-platform non-determinism (FFT, k-means) | med | Mark precision/platform-sensitive skills; compare within-platform; use PERCEPTUAL tolerance where exact is unsafe. |
| Calibration leakage inflates results | med | Cluster-level disjoint splits, frozen before any metric. |
| Subsumption search too slow at scale | med | Coarse grids + early-exit; grid resolution is a config knob; cluster-restrict candidate pairs. |
| 6 GB GPU OOM on a heavy sweep | low | Load one model at a time; batch feature extraction once per probe set; offload to 5090/cloud only for the full CVPR sweep. |
| Sandbox not ready when agent generates code | med | Agent-generated skills are `trusted=False` and **cannot execute** until the Phase-6 hardened sandbox is human-reviewed. |

---

## 8. Immediate next step

On your go-ahead I'll start **Phase 0** (scaffold the repo, CI, config, logging, seeding) and stub the `Skill` model + registry — all dependency-light and runnable under WSL2. I can prototype it in the Linux workspace and hand you the files, or target your WSL2 path directly. Phase 1's first ~25–30 skills follow immediately so we have a runnable harness to validate determinism before any ML dependency lands.
