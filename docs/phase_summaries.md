# Phase summaries

Per the roadmap, each phase ends with passing tests + a short written summary of
assumptions and decisions. This file accumulates them.

---

## Phase 0 — Scaffold ✅

**Delivered**

- `pyproject.toml` (src-layout, setuptools). Core deps are **dependency-light**
  (numpy, Pillow, opencv, pydantic, pyyaml, structlog); the output-grounded ML comparators
  (torch, lpips, timm, open-clip, scikit-image) are isolated in the optional `[ml]` extra so
  Phases 0–1 import with zero ML dependencies (CLAUDE.md §4).
- `config.py` — Pydantic-validated YAML, frozen + `extra="forbid"`. Thresholds carry a
  `calibrated` flag and provenance fields; `calibrated=True` *requires* a split hash + date
  (CLAUDE.md §3.5.5). `configs/default.yaml` holds placeholders only.
- `rng.py` — explicit seed derivation via BLAKE2b, **no global RNG state**; `SeedManager`
  hands out per-component `np.random.Generator`s deterministically (CLAUDE.md §1.4).
- `logging.py` — structlog JSON logging.
- `cli.py` — `viscurate config` / `viscurate skills`.
- CI (`.github/workflows/ci.yml`) runs ruff + ruff-format + mypy(strict) + pytest on 3.11–3.13.

**Exit criteria** — `pip install -e ".[dev]"`, `pytest`, `mypy src`, `ruff check` all clean;
every subpackage imports. ✅

**Assumptions / notes**

- Dev box has `cv2` via **opencv-contrib-python** (a superset of opencv-python); the
  declared dep is `opencv-python` for normal installs. Either satisfies `import cv2`.
- The logging config field is `json_format` (not `json`) to avoid shadowing a Pydantic
  BaseModel attribute.

---

## Phase 1 — Skills + harness ✅ (complete: 100/100 skills)

**Delivered**

- **`Skill` model** (`skills/model.py`): `id, name, description, fn, params_schema, metadata`.
  Two boundaries are enforced *by type*, not convention:
  - `Skill.comparator_view()` returns a `ComparatorView` with **no `description` attribute** —
    the output-grounded path cannot read text even by accident (CLAUDE.md §1.2).
  - `SkillMetadata.agent_view()` omits the internal-only `is_buggy` / `is_dead` labels.
  - `ParamSpec` / `ParamsSchema` do typed coercion, range checks, default-fill, and reject
    unknown params.
- **Registry** (`skills/registry.py`): ordered, id-unique, JSON-serializable. Round-trips
  `SkillSpec` records (no callables) and re-binds `fn` via an id→callable resolver; the
  built-in resolver is the library function table.
- **Canonicalization contract** (`skills/canonicalize.py`, `CANON_VERSION = 1.0.0`): float32
  [0,1] metric view + uint8 hash view; 1-ch→3-ch replicate; RGBA composited over fixed
  mid-gray with alpha tracked separately; binary-mask detection + IoU; shape gate (L∞ = ∞ on
  mismatch). Versioned so labels are reproducible (CLAUDE.md §1.3).
- **Sandboxed executor** (`skills/executor.py`): runs each skill in a **fresh subprocess**
  with a wall-clock timeout; POSIX adds `RLIMIT_AS`/`RLIMIT_CPU` via `preexec_fn`. Hard
  **trusted gate** — `trusted=False` skills are blocked and never spawn a child. Every
  execution is logged.
- **100 deterministic skills** across geometric (25), colour (25), signal/blur/edges/
  morphology (25), and masks/reconstruction/synthesis (25), including the planted relations
  the benchmark depends on: `rotate_90/180/270/45 ⊑ rotate_canvas_degrees`,
  `crop_center_percentage ⊑ crop_bounding_box`, `linear_contrast_stretch =
  contrast_stretching_percentile(0,100)`, the BT.601/709, gamma/exposure, global/CLAHE, and
  Telea/NS-inpaint SEMANTIC pairs, `blur_gaussian` vs `blur_box` (hard negative),
  `dilate`/`erode` (complementary), an RGBA-emitting `mask_to_rgba`, four seeded-stochastic
  skills, two precision-sensitive FFT ops, and one platform-sensitive k-means skill.

**Exit criteria** — each skill runs on a 256²-class probe and returns a valid array; same
`(img,params,seed)` → byte-identical; timeout skill returns an error (does not hang);
`trusted=False` → BLOCKED; registry round-trips. ✅ (144 tests pass.)

**Assumptions / decisions**

- Array convention is **RGB** (PIL load order); colour ops coerce to 3-channel RGB and drop
  alpha, geometric ops preserve channel count.
- `crop_bounding_box` parameters are **fractional** ([0,1]) rather than absolute pixels, so
  the op is size-independent and the center-crop subsumption holds exactly.
- The executor passes the whole (pickled) `Skill` to the worker. This is safe **only because
  untrusted skills are blocked before a child is spawned**; agent-generated code waits for the
  Phase-6 hardened sandbox.
- On Windows the executor degrades to timeout-only (no `resource` module) and logs it; the
  rlimit path is exercised on the WSL2 dev target.

**Phase 1 is complete.** All 100 skills are implemented; the FFT (precision-sensitive) and
k-means (platform-sensitive) members are flagged per §1.4, with their tolerant comparison
deferred to the Phase-3 comparators. Next: **Phase 2** (probe battery + frozen oracle).

---

## Phase 2 — Probe battery + reference oracle ✅

**Delivered** — `src/viscurate/probes/`

- **`manifest.py`** — `ProbeEntry` / `ProbeManifest` (Pydantic, frozen) + a `License` model that
  **rejects empty/`unknown` names** and any license that is not redistributable-with-derivatives.
  `assert_coverage` enforces per-domain floors, all channel formats, and the named degenerate
  cases — a defect is only detectable if the battery exercises it.
- **`synthetics.py`** — deterministic, license-free (CC0) generators seeded from
  `SeedManager`: gradients (linear/radial/angular), textures (value-noise/checker/stripes/
  Gabor), shapes, document pages (PIL), colour charts, noise, and the degenerate cases
  (all-black/white, single-colour, 1×1, thin 1×N / N×1, high-res 1024²). Channel formats
  **RGB / RGBA / grayscale / 16-bit / palette** are spread across domains. Probes are stored as
  raw `.npy` so dtype + channels survive exactly.
- **`coco.py`** — license-clean natural photos from COCO test2017. The image licenses are
  per-Flickr-image (not the dataset's CC BY 4.0); we keep only **CC BY 2.0 (id 4)**, *no known
  copyright restrictions* (7), and *US-Gov* (8), and **exclude NoDerivs (6)** because our skills
  produce derivatives. The license table was *verified against the live metadata*. Selection is
  seeded; provenance (license + COCO source URL + sha256) is recorded per image.
- **`build.py` + `configs/probes.yaml`** — the orchestrator: writes probes, computes a
  representation-independent `array_sha256`, assembles a coverage-checked manifest (generator
  version + canon version + seed for reproducibility).
- **`oracle.py`** — freezes every `(skill, probe)` output hash, **self-auditing determinism**
  (a pair whose two runs disagree is recorded `nondeterministic` rather than frozen as an
  unstable reference) and recording legitimate `error`s (e.g. a skill that cannot run on a 1×1
  probe). `verify_oracle` re-runs and confirms zero divergence. cv2 is pinned single-threaded
  for cross-run reproducibility; `inpaint` is guarded on degenerate thin/tiny inputs.

**Exit criteria (all met)** — builder reproducible (same seed → identical hashes); per-domain
minimums + every degenerate case present; **no `license=unknown`** (verified: only CC0 / CC BY
2.0 / no-known-restrictions in the real battery); oracle re-run reproduces stored hashes.

**Real pilot battery + oracle built:** 177 probes (137 synthetic CC0 + 40 COCO CC BY), all 8
domains and 5 channel formats, zero non-redistributable entries. The full reference oracle
froze **clean — 17,700 pairs (100 skills × 177 probes), all `ok`, 0 error, 0 nondeterministic**
— and a spot re-run reproduced every hash (0 mismatches). 13 Phase-2 tests; full gate green
(157 tests, ruff + ruff-format + mypy --strict).

**Assumptions / notes**

- Probes are stored as `.npy` arrays (not PNG/JPEG) so 16-bit, RGBA, and palette-index probes
  reach the skills byte-exactly; the `palette` format is modelled as a small-valued index array
  (the LUT itself is not modelled in v1).
- COCO test2017 metadata lacks per-photographer fields, so attribution records the license +
  COCO source URL; finer attribution traces through COCO.
- Parameter sweeps (the matched-grid inputs for §3.5.2) are deferred to Phase 3, where the
  comparators consume them; the oracle currently freezes skills at their defaults.

---

## Phase 3 — Comparators + taxonomy ✅

The output-grounded equivalence engine: it decides, **from executed outputs alone**, which of
six relations holds between a pair of skills. All of it lives in the new
`src/viscurate/equivalence/` package and is the first code to use the optional `[ml]` extra.

**The modality boundary is enforced by type.** Nothing in the package reads a skill's
`description`. `classify` takes a `ComparatorView` (no `description` attribute) plus an
`OutputProvider` (yields *outputs*, never a `Skill`). `BatteryEvaluator` is the single trusted
bridge — it holds skills (with `fn`) and probe arrays internally and executes them, but exposes
only `outputs()`/`compose_outputs()`/`comparator_view()`/`param_grid()`. A comparator cannot
reach a description through the typed interface.

**Delivered**

- **`backends.py`** — `PerceptualBackend` / `SemanticBackend` **protocols** and the real
  `LpipsBackend` (AlexNet), `DinoBackend` (`vit_base_patch16_224.dino`), `ClipBackend`
  (`ViT-B-32-quickgelu`, `openai`) plus `ssim_distance` (scikit-image) and `cosine_distance`.
  `torch`/`lpips`/`timm`/`open_clip` are **imported lazily inside constructors**, so the
  package imports cleanly *without* the `[ml]` extra — only constructing a real backend needs
  it. Each backend owns **one** model and frees it on `close()` (context manager): the
  PERCEPTUAL model loads for its stage, frees, then the SEMANTIC model — the 6 GB-GPU
  one-model-at-a-time discipline (CLAUDE.md D5). Features are batch-extracted.
- **`compare.py`** — the comparison primitive. Per-probe distances (`pixel`/`lpips`/`dino`)
  and the **aggregation rule**: worst-case (`max`) for EXACT/PERCEPTUAL (equivalence is
  universally quantified — one diverging probe is a silent-merge bug), p90 + mean for SEMANTIC.
  `BatteryEvaluator` caches outputs per `(skill, params, seed)` and can run **compositions**
  `outer(inner(x))` for the COMPLEMENTARY test.
- **`param_alignment.py` + `configs/param_alignment.yaml`** — the auditable shared-axis map
  (never hard-coded). *Symmetric* matched-sweep `axes` (e.g. `blur_gaussian`/`blur_box` over
  `ksize ∈ {3..21}`) drive the worst-case EXACT/PERCEPTUAL sweep; *asymmetric* `subsumption_grids`
  supply explicit search grids for a generalizing skill (rotate angles incl. 90°, centered
  crop boxes, percentile `(0,100)`).
- **`subsumption.py`** — directional grid search with **early-exit on the first failing probe**;
  reports `A⊑B`, `B⊑A`, mutual (→ EXACT/PERCEPTUAL), or none (near-miss specializations
  correctly return none).
- **`complementary.py`** — COMPLEMENTARY by **non-triviality + approximate commutation**
  (`D(A(B(x)), B(A(x)))` small), executed on real compositions — never from metadata/family.
- **`taxonomy.py`** — the stop-at-first pipeline EXACT → PERCEPTUAL → SUBSUMPTION → SEMANTIC →
  COMPLEMENTARY → DISTINCT with a calibrated **UNCERTAIN abstention band** `[τ(1−δ), τ(1+δ)]`
  around the PERCEPTUAL/SEMANTIC thresholds. Returns a structured `RelationResult` (relation,
  direction, deciding distances, worst-case probe, permitted actions) — the actionable
  rejection reason of CLAUDE.md §3.5.7.
- **`candidates.py`** — output-based candidate generation: an output fingerprint (perceptual
  average-hash ‖ mean DINO feature) over a screening sub-battery, nearest-neighbour proposal,
  **plus same-family pairs and the engineered hard negatives always included**. Output-based by
  construction, so different-description/same-output redundancy still collides — the redundancy
  text-based pruning misses.
- **`calibrate.py`** — the calibration *procedure*: `select_threshold` maximizes recall subject
  to a precision-on-non-equivalence floor; `calibrate_thresholds` fits τ_perc/τ_sem/δ on a
  labeled split and **stamps provenance** (split hash + date, `calibrated=True`). It ships **no
  numbers** — there is no human-labeled split until Phase 4, and inventing one is forbidden.

**Key design decision (a bug the tests caught).** EXACT/PERCEPTUAL on a single default binding
is authoritative **only** when the pair shares a real matched-sweep axis *or* both skills are
parameter-free. Two parameterized skills that merely coincide at their defaults (e.g.
`crop_center_percentage(50%)` ≡ `crop_bounding_box`'s default centered box) must **not** be
called EXACT — the grid search decides, yielding the correct `crop_center ⊑ crop_bounding_box`
subsumption.

**Exit criteria (met)** — hand-built EXACT / PERCEPTUAL / DISTINCT synthetic pairs classify
correctly; SUBSUMPTION (rotate_90 ⊑ rotate_canvas, crop_center ⊑ crop_bbox), COMPLEMENTARY,
SEMANTIC, and the UNCERTAIN band each exercised; the matched-sweep worst-case **blocks a false
merge** of `blur_gaussian`/`blur_box`. The calibration procedure is implemented and provenance-
stamped (thresholds remain `calibrated=false` placeholders pending the Phase-4 labeled split).
LPIPS+DINO+CLIP load and produce sane distances (real-backend smoke tests, `-m slow`).

**Environment (verified install).** torch `2.10.0+cpu` was already present; the **matched**
`torchvision==0.25.0+cpu` was installed from the PyTorch CPU index with `--no-deps` (a plain
PyPI `torchvision` risks dragging in a CUDA torch). Model weights (AlexNet, DINO ViT-B/16, CLIP
ViT-B/32) are cached under `~/.cache`. mypy was pinned at 3.11 but scikit-image pulls `tifffile`
(3.12-only `type` syntax) → a `follow_imports = "skip"` override for `skimage`/`tifffile` keeps
mypy clean without weakening typing elsewhere.

**Honest limitations (carried to Phase 4).**

- **Thresholds are uncalibrated placeholders.** Real labels and calibration land in Phase 4;
  every reported metric must use a calibrated config (the provenance rule enforces this).
- **The commutation-based COMPLEMENTARY test is necessary, not sufficient.** Two linear filters
  (e.g. the two blurs) commute yet are *not* "disjoint aspects"; in the full pipeline they are
  caught earlier (SEMANTIC) before COMPLEMENTARY, and the residual label is a calibration
  question. The safety property the tests assert is the robust one: **no false merge**.
- **Center-crop subsumption is exact only on even-sided probes** (center-crop floors, the
  fractional bbox rounds); on odd sizes a 1-px offset can appear. The PERCEPTUAL tolerance and
  the Phase-4 battery/calibration absorb this; it is documented, not hidden.
- **CPU-only here.** The GPU 6 GB budget is met by design (one model at a time); it is not yet
  measured on the target RTX 3050.

**Tests:** 21 new (`tests/test_equivalence.py` — 17 fake-backend, deterministic; +
`tests/test_equivalence_ml.py` — 4 real-backend smoke, marked `slow`). Full gate green: **178
tests, ruff + ruff-format + mypy --strict**.
