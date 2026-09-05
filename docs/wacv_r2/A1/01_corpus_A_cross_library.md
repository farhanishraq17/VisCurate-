# A1 · Corpus A — Cross-Library Claimed-Equivalence Audit

**Effort: LOW (one person, one week). Signal: HIGHEST in the entire revision.**
**Proves:** C1 — description-derived equivalence judgments fail on real code.

---

## 1. The core idea

> **Revised 2026-08-18** per [`../CRITIQUE_RESPONSE.md`](../CRITIQUE_RESPONSE.md) §5–7. The original framing measured an "over-claim rate" *against maintainers*, which misattributes to them a claim they never made. The corrected construct is below — read §1.1 before writing any of this up.

Library maintainers publish **category-level equivalence judgments** to help users migrate code: *"these two do the same broad operation."* That judgment is derived from names, documentation, and intent — the exact modality the paper argues is unreliable, and **the exact judgment a text-based curator produces.**

**We execute both sides and measure how often a category-level judgment survives testing at aligned parameters.**

### 1.1 ⚠ What these tables are, and are not

The Kornia mapping page states that a named counterpart means Kornia has a built-in API for **"the same broad operation."** That is a **migration/category mapping, not an assertion of output equivalence at aligned parameters.**

**Therefore:**

| Do | Do not |
|---|---|
| Quote the tables' exact semantics when describing them | Call them "equivalence claims" or "ground truth" |
| Treat a label as a **noisy external hypothesis** | Report an "over-claim rate" as a maintainer error rate |
| Report results **broken down by label type** | Collapse *direct* / *partial* / `–` into one "claimed equivalent" bucket |
| Say: *"a category-level judgment of the kind text-based curation produces"* | Imply maintainers asserted pixel-level equivalence |

**The argument survives the reframe intact.** The value was never that maintainers are wrong — it is that these judgments were **not constructed by us**, so they cannot be dismissed as a strawman built to lose. A category-level "same operation" judgment is precisely what an LLM curator reading two descriptions emits, and testing it by execution is exactly the paper's thesis.

### 1.2 Why this answers R3
R3 wrote that the central claim "is never actually demonstrated experimentally" and that the paper "only gives a couple of illustrative examples." Any baseline *we* build invites the reply that we built it to lose. These judgments come from maintainers of a widely-used library, written for their own users, published before this paper existed.

---

## 2. Sources and pair inventory

### 2.1 Externally authored pairs (the primary asset)

| Source | Pairs | URL |
|---|---|---|
| Albumentations ↔ Torchvision v2 | ~94 | `https://albumentations.ai/docs/albumentations-vs-torchvision/transforms/` |
| Albumentations ↔ Kornia | 75 | `https://albumentations.ai/docs/albumentations-vs-kornia/transforms/` |

Label taxonomy used by both pages:
- **named counterpart** = direct equivalent
- **multiple counterparts** or `(partial)` = partial equivalent
- **`–`** = *"does not support that transform as a built-in augmentation primitive"*

Two rows already visible in the published tables land directly on the paper's taxonomy:

| Published row | Predicted VisCurate relation |
|---|---|
| `ColorJitter` → `ColorJitter / ColorJiggle` | two candidates for one operation — `EXACT`, `PERCEPTUAL`, or a silent hard negative |
| `RandomBrightnessContrast` → `RandomBrightness / RandomContrast` | **one transform = composition of two** — a textbook `COMPLEMENTARY` / `SUBSUMPTION` case |

That second row matters: `COMPLEMENTARY` currently sits at F1 = 0.000 in Table 3 on a support of one synthetic pair. Corpus A supplies real ones.

### 2.2 Self-constructed pairs (declare the provenance)

Target **80–150 additional pairs** across OpenCV ↔ Pillow ↔ scikit-image ↔ Kornia, where no published mapping exists.

**Construct these by text alone** — matching on operation name and documented description, blind to any output. Then:
- freeze the list, hash it, commit it **before running any metric**
- state the provenance in the paper: these are description-derived claims *we* made, which makes them an additional description-based baseline whose failure rate we measure

> **Contamination warning.** If anyone peeks at outputs while building this list, the experiment is compromised and cannot be repaired after the fact. Assign it to one person, have them work from documentation only, and hash the result before the first execution.

### 2.3 Operator families to cover

High-yield families where cross-library divergence is known to live:

| Group | Operations |
|---|---|
| Blur | Gaussian, box/mean, median, bilateral, motion |
| Resize | nearest, bilinear, bicubic, area, Lanczos |
| Geometric | rotate, affine/warp, crop, **pad (border modes!)**, flip, transpose |
| Photometric | brightness, contrast, saturation, hue, gamma, normalize |
| Histogram | global equalization, **CLAHE** |
| Edges | Sobel, Scharr, Laplacian, Canny |
| Morphology | erode, dilate, open, close, gradient, tophat |
| Quantization | threshold, posterize, solarize, invert |
| Enhancement | sharpen, unsharp mask, emboss |
| Codec | JPEG compression |
| Color | RGB↔grayscale, RGB↔HSV, RGB↔LAB |

**Border handling, `align_corners`, and dtype/range conventions are where these libraries genuinely diverge.** Those are the hard negatives — and unlike the paper's current hard negatives, they are real rather than engineered.

### 2.4 Total inventory

| Block | Pairs |
|---|---|
| Externally authored (published tables) | ~169 |
| Self-constructed (name/doc-matched) | 80–150 |
| **Total ordered pairs** | **~250–320** |

Small, but every pair is high-signal and externally grounded.

---

## 3. Critical design decision: audit the *functional* API

> ⚠ **Construct caveat — state this in the paper.** The published tables map **transform classes**, which include sampling behavior, application probability, and parameter distributions. Auditing the **functional cores** tests a different object than the one the tables map. The connection to the external artifact is therefore *candidate generation*, not claim-testing.
>
> **The honest formulation:** the mapping tables supply the **candidate pairs**; the equivalence hypothesis tested at the functional level is **ours**. Corpus A is an author-constructed functional-API audit whose pair list is externally sourced — which retains the anti-strawman property (we did not choose which functions to call equivalent) without misdescribing what is tested.
>
> **Secondary check:** where a public wrapper is deterministic given a fixed seed, additionally test the wrapper itself and report both results. Where the two disagree, the disagreement localizes to sampling behavior rather than to the core operation — itself a reportable finding.

Most mapped transforms are **random** augmentations (`RandomFog`, `RandomBrightnessContrast`). Comparing those directly drags in stochasticity and makes Corpus A depend on A3 finishing.

**Audit the functional cores.** Every library exposes a deterministic functional core beneath the random wrapper:

| Library | Functional entry point |
|---|---|
| Albumentations | `albumentations.augmentations.functional` |
| Kornia | `kornia.enhance.*`, `kornia.filters.*`, `kornia.geometry.*`, `kornia.morphology.*` |
| torchvision v2 | `torchvision.transforms.v2.functional` |
| OpenCV | `cv2.*` |
| Pillow | `PIL.ImageOps`, `PIL.ImageFilter`, `PIL.ImageEnhance` |
| scikit-image | `skimage.filters`, `skimage.transform`, `skimage.exposure`, `skimage.morphology` |

A random transform is just *sampling + functional core*. Auditing the functional layer:
- keeps Corpus A **entirely inside the paper's existing deterministic scope**
- requires zero new machinery
- is a cleaner scientific comparison

**State this explicitly in the paper** — a reviewer will otherwise ask why stochastic transforms were compared deterministically.

---

## 4. Parameter alignment — the real work

To compare `cv2.GaussianBlur(ksize, sigmaX)` against `kornia.filters.gaussian_blur2d(kernel_size, sigma)` against `PIL.ImageFilter.GaussianBlur(radius)`, the parameter schemas must be aligned. This is manual, per-family, and is genuinely part of the contribution.

### 4.1 Protocol

Parameter alignment is the **largest source of researcher degrees of freedom** in Corpus A. If we choose the alignment, we partly choose the result. Guard it accordingly:

1. Hand-author an alignment map per operator family: a bijection (or partial map) between parameter schemas, plus the shared evaluation grid `Θ^grid`.
2. **Author from documentation only**, before executing anything.
3. **Two independent reviewers** author alignments for every ambiguous family; disagreements are recorded, not silently resolved.
4. **Preserve alternative valid alignments** rather than picking one. Where more than one defensible mapping exists, evaluate under each and **report sensitivity across them.** A divergence that appears under only one of several valid alignments is not a behavioral finding — it is an alignment artifact, and reporting it as the former would be a serious error.
5. **Alignment failure and behavioral non-equivalence are separate outcomes** and must never be merged into one bucket. Where no alignment exists — PIL's `radius` and OpenCV's `ksize` are genuinely different parameterizations — record an **alignment failure**. That is a finding about cross-library interoperability, not evidence of behavioral divergence.
6. Pre-register the maps (hash + commit) and publish them as a supplementary artifact.

### 4.2 Suggested schema

```yaml
family: gaussian_blur
canonical_params:
  sigma:   {type: float, grid: [0.5, 1.0, 2.0, 4.0, 8.0]}
  ksize:   {type: int,   grid: [3, 5, 9, 17, 33], constraint: odd}
implementations:
  - id: cv2.GaussianBlur
    call: "cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)"
    maps: {ksize: ksize, sigma: sigma}
    notes: "borderType defaults to BORDER_REFLECT_101"
  - id: kornia.filters.gaussian_blur2d
    call: "gaussian_blur2d(img, (ksize, ksize), (sigma, sigma))"
    maps: {ksize: kernel_size, sigma: sigma}
    notes: "border_type defaults to 'reflect'"
  - id: PIL.ImageFilter.GaussianBlur
    call: "img.filter(ImageFilter.GaussianBlur(radius))"
    maps: {sigma: radius}
    alignment: PARTIAL
    notes: "PIL radius is not sigma; no exact ksize control. Flag as alignment failure."
provenance: {author: <name>, date: <YYYY-MM-DD>, source: docs_only}
```

### 4.3 Why worst-case aggregation matters here
The paper's existing max-over-grid aggregation (`sec/6_methodology.tex:77`) does the heavy lifting: two blurs that agree at small kernels and diverge at large ones get caught. That is exactly the Gaussian-vs-box example already in the text — now demonstrated on **real library code** instead of a hand-built pair.

---

## 5. Execution protocol

1. **Environment.** Single pinned venv, all six libraries at pinned versions. Record versions in the artifact — cross-library behavior changes between releases and this is a reproducibility requirement.
2. **Probe battery.** Reuse the existing 177-probe battery `B` unchanged. Do not build a new one for Corpus A — reusing it means results are directly comparable to the synthetic study.
3. **Canonicalization.** Extend `φ` to normalize: `uint8`↔`float32`, `[0,255]`↔`[0,1]`, `HWC`↔`CHW`, `BGR`↔`RGB`, PIL `Image`↔`ndarray`↔`torch.Tensor`. **Log every conversion applied per pair** — a conversion bug would masquerade as a behavioral difference and is the most likely source of a false over-claim finding.
4. **Verification.** Run the unmodified verifier from `sec/6_methodology.tex` §Equivalence Taxonomy at the **frozen synthetic operating point** (Table 4 thresholds). **Do not re-calibrate on this corpus.**
5. **Baseline ladder.** Run all A2 baselines over the same pair list (see [`../A2/01_baseline_ladder.md`](../A2/01_baseline_ladder.md)) so Corpus A reproduces the full ladder on real code.

### 5.1 Sanity gate before trusting any result
Before interpreting divergences, verify the harness on **self-pairs**: every function compared against itself must certify `EXACT`. Any self-pair that does not indicates a canonicalization or determinism bug. Fix before proceeding. Report this gate in the paper as a harness-validity check.

---

## 6. Headline metric — the claim-vs-execution matrix

For every pair, cross-tabulate the maintainer/self-constructed label against what execution certifies:

| Claimed label | `EXACT` | `PERCEPTUAL` | `SUBSUMPTION` | `SEMANTIC-PRES.` | `UNCERTAIN` | **`DISTINCT`** |
|---|---|---|---|---|---|---|
| Direct equivalent (n=…) | | | | | | **← the money cell** |
| Partial equivalent (n=…) | | | | | | |
| No equivalent `–` (n=…) | | | | | | |

### Three findings emerge from this one table

**1. Category-judgment survival rate.** Of pairs a maintainer grouped as the same operation, the fraction that execute as behaviorally equivalent at aligned parameters. Pairs executing as `DISTINCT` are real silent-merge hazards for any text-based curator. **Enumerate them individually in the appendix** with the deciding distance and the divergent parameter setting. This is the most persuasive content the revision can contain.

> **Wording discipline:** report this as *"a category-level judgment that does not survive execution"* — **never** as *"the maintainers were wrong."* They documented a migration counterpart, not pixel equivalence. See §1.1.

**2. Missed consolidation — requires a separate protocol, not this table.**

> ⚠ **A `–` is not a pair.** A dash means *"no built-in counterpart listed"* — it names no second function, so a missed-consolidation rate **cannot be computed from these rows.** The original plan defined it incorrectly.
>
> To measure it properly, run a **candidate-search protocol**: for each function on the `–` side, fingerprint it and search the *entire* other library for behavioral matches, then adjudicate any hits. That is a legitimate and interesting experiment — real cross-library duplicates nobody documented — but it is a **different experiment** with its own candidate-generation design, and it must be scoped and reported separately. Mark it **optional**; do not attempt it if Corpus B is at risk.

> Note: `sec/7_results.tex:39` claims *"no baseline recovers a redundancy the verifier misses."* If the candidate-search protocol runs and finds real missed consolidations, that sentence needs softening. Flag to whoever edits §5.

**3. Taxonomy coverage on real code.** The partial-equivalent row should populate `SUBSUMPTION` and `SEMANTIC-PRESERVING` with real support — directly addressing the thin-support problem in Table 3.

**All rates get exact (Clopper–Pearson) confidence intervals with raw numerators and denominators.** See [`../README.md`](../README.md) §Statistical rules.

---

## 7. The motivating example for the introduction

[kornia/kornia#3408](https://github.com/kornia/kornia/discussions/3408) is a user report that **Kornia's CLAHE and OpenCV's CLAHE are not equivalent** — same name, same textbook algorithm, same documented purpose — differing in dtype requirements (`uint8`/`uint16` vs `float32`), `clipLimit` semantics, and runtime (~0.5s vs ~2s). The reporter concluded "not equivalent" only after comparing actual outputs; the maintainer's reply pointed at a tutorial and did not resolve whether the outputs are mathematically equivalent.

**This is the paper's entire thesis occurring in the wild, documented by someone who had never heard of VisCurate.**

*Recommendation:* move it into `sec/1_intro.tex` as the motivating example, replacing or supplementing the currently-unverified "20,000 skills / 5,600 distinct operations" statistic flagged in [`PLAN.md`](../../Viscurarte_Rebuttal/PLAN.md) §7. Reproduce the divergence ourselves so the paper reports a measured distance, not just a link to a forum thread.

---

## 8. Deliverables

| ID | Artifact |
|---|---|
| **T2** | Claim-vs-execution matrix (§6) |
| **T3** | Full baseline ladder on real pairs — name / TF–IDF / sentence-emb / code-emb / AST / LLM-desc / LLM-source / output-grounded |
| **T4a** | Verifier precision per relation on human-adjudicated Corpus A pairs |
| **F1** | **Qualitative figure: 4–6 real over-claimed pairs** — the two functions, the probe where they diverge, outputs side by side, deciding distance annotated. **Lead with CLAHE.** |
| **A-1** | Supplementary: parameter alignment maps (YAML) |
| **A-2** | Supplementary: full enumerated over-claim list with divergence parameters |

**Target headline sentence:**
> *"Of N cross-library transform pairs that library maintainers document as direct equivalents, execution certifies only X% as behaviorally equivalent; Y pairs diverge beyond perceptual tolerance at documented parameter settings."*

---

## 9. Checklist

- [ ] Scrape and freeze the ~169 published pairs; record retrieval date and page snapshots
- [ ] Construct 80–150 self-matched pairs **from documentation only**
- [ ] Hash + commit the frozen pair list **before any execution**
- [ ] Author parameter alignment maps for ≥15 operator families
- [ ] Pin library versions; record in artifact
- [ ] Extend `φ` for cross-library dtype/layout/range; log conversions per pair
- [ ] **Pass the self-pair sanity gate** (every function vs. itself → `EXACT`)
- [ ] Execute at frozen synthetic operating point — **no re-calibration**
- [ ] Run full baseline ladder over the same pairs
- [ ] Produce claim-vs-execution matrix
- [ ] Adjudicate all disagreements (small n — do all, do not sample)
- [ ] Reproduce the CLAHE divergence with a measured distance
- [ ] Build F1 qualitative figure
- [ ] Publish alignment maps + over-claim list as supplementary
