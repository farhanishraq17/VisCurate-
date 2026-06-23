# ClaudeCode — VisCurate build log

A running record of what has been implemented and what comes next, written to complement
the `claude.md` roadmap. This document covers **Phase 1 (skills + harness)** in detail and
lays out the concrete plan for **Phase 2 (probe battery + reference oracle)**.

> Status at time of writing (2026-06-23): Phase 0 complete; **Phase 1 complete — all 100
> skills implemented**; gate fully green (`ruff`, `ruff format`, `mypy --strict`, 144 tests).

---

## 1. Phase 1 — what was implemented

Phase 1 turns the roadmap's natural-language skill blueprint into the project's **executable
substrate**: a typed skill model, a serializable registry, the output canonicalization
contract, a sandboxed executor, and a first batch of deterministic skills. Everything here
is **ML-dependency-free** by design (CLAUDE.md §4) — torch/LPIPS/DINO only enter in Phase 3.

### 1.1 The `Skill` model — [src/viscurate/skills/model.py](src/viscurate/skills/model.py)

A skill is `fn(image, params, seed) -> image` plus typed metadata. The model encodes two
load-bearing boundaries **structurally** (by type), so they can't be violated by accident:

- **`Skill.comparator_view()`** returns a `ComparatorView` dataclass that **has no
  `description` attribute at all**. The output-grounded verifier is handed this object, so a
  comparator that reaches for text fails immediately rather than silently leaking the modality
  the paper is trying to beat (CLAUDE.md §1.2).
- **`SkillMetadata.agent_view()`** omits the internal-only ground-truth labels
  (`is_buggy`, `is_dead`) — these are never shown to the curation agent.

Supporting pieces:

- **`ParamSpec`** — one typed parameter (`int`/`float`/`bool`/`str`/`enum`) with
  default, range (`minimum`/`maximum`), and `choices`. `coerce()` validates and converts a
  value; the default is itself validated at construction.
- **`ParamsSchema`** — an ordered tuple of `ParamSpec` with `validate_params()` that fills
  defaults, coerces/range-checks provided values, and **rejects unknown keys**.
- **`SkillSpec`** — the JSON-serializable view of a skill (everything except the callable),
  used for persistence.

### 1.2 The registry — [src/viscurate/skills/registry.py](src/viscurate/skills/registry.py)

An ordered, id-unique collection. It is **JSON-serializable**: `to_json()` emits `SkillSpec`
records (no callables); `from_json(text, resolver)` rebinds each skill's `fn` via an
id→callable **resolver**. For the built-in library that resolver is the library's function
table. Skill ids are stable and never reused, so the mapping is a durable contract. Tested
for round-trip fidelity *and* that a reloaded skill executes byte-identically.

### 1.3 The canonicalization contract — [src/viscurate/skills/canonicalize.py](src/viscurate/skills/canonicalize.py)

Skills emit heterogeneous outputs (grayscale, RGBA, binary masks, edge maps, shape-changing
geometric ops). Comparison needs one fixed, **versioned** rule (`CANON_VERSION = "1.0.0"`):

| Aspect | Rule |
|---|---|
| dtype / range | coerce to `float32 ∈ [0,1]` for metrics; keep a `uint8` copy for hashing (uint8÷255, uint16÷65535, bool→{0,1}, float clipped) |
| channels | 1-channel → replicate to 3; **RGBA → composite over fixed mid-gray** for the RGB view, alpha tracked separately |
| shape | identical shape is a precondition for EXACT/PERCEPTUAL; `max_abs_pixel_diff` returns `∞` on mismatch (the shape gate) |
| binary masks | single-channel `{0,max}` outputs flagged; compared by exact match + **IoU**, not LPIPS |

`content_hash()` (SHA-256 over the uint8 view + shape + mask flag + alpha + version) makes
EXACT duplicates collide and guarantees different shapes never do. The contract is itself a
versioned object so relation labels stay reproducible.

### 1.4 The sandboxed executor — [src/viscurate/skills/executor.py](src/viscurate/skills/executor.py)

Lightweight isolation appropriate for the trusted 100-skill starter set (CLAUDE.md D6):

- Each execution runs in a **fresh subprocess** ([_worker.py](src/viscurate/skills/_worker.py))
  with a **wall-clock timeout**, so a hanging or crashing skill returns a structured
  `ExecutionResult` instead of taking down the harness.
- On POSIX (the WSL2 dev target) it adds `RLIMIT_AS` / `RLIMIT_CPU` via `preexec_fn`. On
  Windows `resource` is unavailable, so it degrades to timeout-only and logs that once.
- **Hard trusted gate:** `trusted=False` skills are *blocked* and never spawn a child —
  agent-generated code waits for the Phase-6 hardened sandbox (CLAUDE.md §5).
- Every execution is logged (structured JSON).

`ExecutionResult` carries `ok / output / error / duration_s / timed_out / blocked /
returncode`. Intentionally-misbehaving skills for testing the failure paths live in
[_testkit.py](src/viscurate/skills/_testkit.py) (sleeper, crasher) — an importable module so
the subprocess can always resolve them.

### 1.5 The skill library (100 / 100 — complete) — [src/viscurate/skills/library/](src/viscurate/skills/library/)

The clean base **L0**: defect-free (no bugs, duplicates, or dead skills — those are
*injected later* at a controlled rate ρ in Phase 5). All 100 deterministic skills, across
the four roadmap super-families (granular `metadata.family` sub-tags in parentheses):

| Super-family | Count | Representative skills |
|---|---|---|
| geometric / canvas (`geometric`) | 25 | flip H/V, rotate 90/180/270/45/arbitrary, transpose, translate, shear H/V, scale_xy, crop center%/bbox, random_crop (seeded), zoom, resize nearest/bilinear/bicubic/fixed, pad reflect/replicate/constant, pad-to-square, tile 2×2 |
| colour / exposure / contrast (`color`) | 25 | grayscale bt601/bt709, invert, brightness add/mul, exposure_stops, contrast, gamma, saturation, desaturate, hue_shift, sepia, temperature, colour-balance, channel isolate/swap, threshold, solarize, posterize, quantize, linear/percentile stretch, levels, equalize global/CLAHE |
| signal / blur / edges / morphology (`blur`,`sharpen`,`frequency`,`edges`,`texture`,`morphology`,`denoise`,`noise`,`stylize`) | 25 | blur gaussian/box/median/bilateral/motion, unsharp + laplacian sharpen, high-pass spatial, emboss, edges sobel/scharr/prewitt/laplacian/canny, DoG, gabor, **low/high-pass FFT (precision-sensitive)**, dilate/erode/open/close, NL-means denoise, gaussian + salt-pepper noise (seeded) |
| masks / reconstruction / synthesis (`reconstruction`,`segmentation`,`quantize`,`stylize`,`synthesis`,`mask`) | 25 | inpaint Telea/NS, flood-fill, connected-components, contours, distance-transform, threshold Otsu/adaptive, **k-means palette (platform-sensitive)**, dither, pixel-sort, mosaic, vignette, duotone, checkerboard, frame, grid, sketch, cartoonize, per-channel equalize, background-blur composite, **mask→RGBA**, halftone, value-noise (seeded), swirl |

Determinism flags set per §1.4: **seeded-stochastic** (4) — `random_crop`, `add_gaussian_noise`,
`add_salt_pepper_noise`, `value_noise_synthesize`; **precision-sensitive** (2) —
`low_pass_fft`, `high_pass_fft`; **platform-sensitive** (1) — `palette_reduce_kmeans`.

**Planted relations** (so the Phase-4 benchmark has structure to find) are baked into this
set already and verified by tests:

- **Subsumption:** `rotate_90/180/270/45 ⊑ rotate_canvas_degrees`,
  `crop_center_percentage ⊑ crop_bounding_box`,
  `linear_contrast_stretch = contrast_stretching_percentile(0,100)` and `⊑ levels_adjust`.
- **Semantic neighbours:** `grayscale_bt601` vs `grayscale_bt709`; the
  Sobel/Scharr/Prewitt/Canny/Laplacian edge family; `gamma_correct` vs `exposure_stops`;
  `equalize_histogram_global` vs `equalize_clahe`; `inpaint_telea` vs `inpaint_ns`;
  `threshold_otsu` vs `threshold_adaptive_mean`.
- **Hard negatives (DISTINCT):** `blur_gaussian` vs `blur_box` (agree at small k, diverge at
  large k), nearest/bilinear/bicubic resize, reflect vs replicate padding, `posterize` vs
  `quantize_uniform`.
- **Complementary:** `morphology_dilate` ↔ `morphology_erode` (and `open`/`close` built on them).
- **Seeded-stochastic determinism probes:** `add_gaussian_noise`, `add_salt_pepper_noise`,
  `random_crop`, `value_noise_synthesize` (deterministic at a fixed seed).
- **RGBA / alpha path:** `mask_to_rgba` emits a 4-channel output that exercises the alpha
  branch of the canonicalization contract.

Conventions: arrays are **RGB**; colour ops coerce to 3-channel and drop alpha, geometric ops
preserve channel count; `crop_bounding_box` parameters are **fractional** so the op is
size-independent and the center-crop subsumption holds exactly.

### 1.6 Supporting scaffold (Phase 0, recap)

[config.py](src/viscurate/config.py) (frozen Pydantic YAML; `calibrated=True` requires
provenance), [rng.py](src/viscurate/rng.py) (BLAKE2b seed derivation, no global state),
[logging.py](src/viscurate/logging.py) (structlog JSON), [cli.py](src/viscurate/cli.py),
and CI across Python 3.11–3.13.

### 1.7 Tests & verification

[tests/](tests/) — 144 tests covering config validation, RNG determinism/independence,
param coercion, the modality/label boundaries, registry round-trip + identical re-execution,
the full canonicalization contract, all 100 skills (determinism + canonicalization,
parametrized), the determinism-flag assignments, the planted relations (incl. the RGBA path),
and the executor (blocked / timeout / crash / success-matches-in-process / byte-identical
sandboxed runs).

```
ruff check          All checks passed!
ruff format --check  all files already formatted
mypy src (strict)    Success: no issues found in 19 source files
pytest               144 passed
```

### 1.8 How to run

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

ruff check . && ruff format --check . && mypy src && pytest   # the gate
python -m viscurate.cli skills                                 # list the library
python -m viscurate.cli config -c configs/default.yaml         # validate a config
```

### 1.9 Phase 1 — complete ✅

All four super-families (100 skills) are implemented, including the masks / reconstruction /
synthesis family (inpainting, palette-reduce/k-means, FFT ops, pixel-sort, dithering,
synthesis) and the precision-sensitive (FFT) and platform-sensitive (`cv2.kmeans`) members
flagged per CLAUDE.md §1.4. Their *handling* in the comparators (PERCEPTUAL tolerance for the
FFT pair, within-platform-only comparison for k-means) is wired in **Phase 3** when the
comparators exist. Phase 1's exit criteria are met; the harness is ready for Phase 2 to scale
on it.

---

## 2. Phase 2 — plan (probe battery + reference oracle)

**Goal (CLAUDE.md Phase 2 / §2.1):** build the versioned probe battery `P` and freeze the
reference oracle, so corruption can later be *proven* to have taken effect and the verifier/
agent can be scored. A defect is only detectable if the battery exercises it — coverage is
the whole point.

### 2.1 Deliverables

1. **`build_probes.py`** — a reproducible generator (same seed → same image bytes → same
   hashes), driven by config and `SeedManager`. Writes to `data/probe_images/`.
2. **`data/probe_images/manifest.json`** — one entry per image:
   `probe_id`, `sha256`, `domain`, `channel/format`, `resolution`, `source`, **license**,
   notes. No `license=unknown` permitted.
3. **The probe diversity axes** (§2.1) covered with per-domain minimum counts:
   - **Domain:** natural photos, documents/screenshots, textures, synthetic gradients, and
     **degenerate cases** (all-black, all-white, single-color, 1×1, very-high-res).
   - **Channel/format:** RGB, **RGBA**, grayscale, 16-bit, palettized — so domain-scoped bugs
     (Type 6) have somewhere to fire.
   - **Signal:** low/high frequency, color profile, aspect ratio.
   - **Parameter sweeps:** grids over each parameterized skill's declared range (e.g. blur(k)
     and box(k) agree at small k, diverge at large k) — the matched-sweep inputs Phase 3 needs.
4. **The frozen reference oracle** — execute **every L0 skill over P + sweeps**, then hash and
   store every output (`data/oracle/`). This oracle (a) later confirms corruption took effect
   and (b) scores the verifier/agent. **It is never used to assign relation labels.**
5. **A probe/oracle reproducibility manifest** — generator version, pool hash, battery hash,
   seeds, canonicalization version (`CANON_VERSION`), so a run is fully traceable (CLAUDE.md §5).

### 2.2 Approach & design choices

- **License-clean sources only** — self-generated synthetics (gradients, textures, shapes,
  noise fields, degenerate cases) need no external license and are fully reproducible from
  seeds; these form the backbone of the **pilot ~200 images**. Natural photos come from
  license-clean sets (COCO CC BY / OpenImages) added with explicit license fields. Synthetic
  faces (documented license) are the recommended face-domain default (avoids PII; the
  `[CONFIRM]` in §6 of the roadmap still applies).
- **Determinism** — every synthetic probe is produced from `SeedManager(root).generator(...)`
  so the battery is byte-reproducible; the manifest records the hash of each image.
- **Canonicalize at oracle-freeze time** — store both the raw output and its `content_hash`
  under `CANON_VERSION`, so later equivalence checks are apples-to-apples.
- **Sweeps as first-class artifacts** — the `param_alignment` notion from §3.5.2 starts here:
  define, per parameterized skill, the shared semantic axis (e.g. kernel size `k`) and the
  grid, stored in config so Phase 3's matched-sweep comparison is auditable, not hard-coded.
- **Storage** — images + oracle outputs are large and regenerable, so they stay **out of git**
  (already in `.gitignore`); only the manifests (small, defining) are tracked.

### 2.3 Exit criteria (from the roadmap)

- Builder reproducible: same seed → identical hashes.
- ≥ N images per domain and **each degenerate case present**.
- No `license=unknown` entries.
- Oracle freeze + hashing verified: re-running a skill over `P` reproduces the stored hash.

### 2.4 New modules (anticipated)

```
src/viscurate/probes/
  build.py         # the generator (synthetics + loaders), seed-driven
  synthetics.py    # gradient/texture/shape/noise/degenerate generators
  manifest.py      # Probe + Manifest pydantic models, sha256, license field (no "unknown")
  oracle.py        # freeze: run every L0 skill over P+sweeps, hash + store, verify
configs/probes.yaml  # per-domain counts, resolutions, sweep grids (param_alignment seed)
```

### 2.5 Risks / watch-items for Phase 2

- **Coverage gaps** silently hide future defects — enforce per-domain minimums in code and
  assert them in tests.
- **16-bit / palettized** handling must round-trip through the canonicalization contract
  (already supports uint16 + single-channel); add probes that exercise those paths.
- **Oracle size** — batch and stream; only the pilot 200 (×sweeps) for now, scale via config.

---

*Next action after Phase 2: Phase 3 — the LPIPS / DINO / CLIP comparators (the `[ml]` extra
lands here), subsumption search, and threshold calibration — leading into the Phase-4
divergence checkpoint, the project's first go/no-go gate.*
