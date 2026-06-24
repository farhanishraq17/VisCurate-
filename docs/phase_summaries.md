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

---

## Phase 4 — Equivalence benchmark ⚠️ (machinery complete; the real divergence run is the go/no-go)

Phase 4 turns the Phase-3 engine into a scored benchmark: run the **output-grounded verifier**
and the **text baselines** over candidate pairs, score both against the designed relation graph
``G0``, and produce the **divergence table** — the project's first go/no-go checkpoint. The
machinery, its answer key, and a full deterministic test suite are done; the **real** divergence
numbers (real LPIPS/DINO/CLIP over the full battery) are a single GPU-targeted command and are
**not run here** (no fabricated numbers — CLAUDE.md §5).

**Delivered**

- **The answer key `G0`** — `configs/ground_truth_g0.yaml` + `benchmark/ground_truth.py`. The
  blueprint's planted relations lifted into a validated graph: **subsumption is a DAG**, EXACT is
  **transitively closed**, symmetric relations are stored unordered, every id is checked against
  the live library, and a pair may carry only one relation. 24 designed pairs (7 subsumption, 10
  semantic, 1 complementary, 6 DISTINCT hard negatives); everything else defaults to DISTINCT.
  Fixed before any metric runs and never derived from the metrics under test (CLAUDE.md §2.5).
- **Text baselines in a separate package** — `viscurate/baselines/` is the one place allowed to
  read `description` (the modality boundary is a *package* boundary, CLAUDE.md §1.2):
  **name-match** (token Jaccard), **embedding-cosine** (the direct strawman — dependency-light,
  deterministic TF-IDF behind a swappable `TextEmbedder` protocol), and **LLM-on-descriptions**
  (built behind an `LlmClient` protocol; with no client it records *not run* rather than
  fabricating — there is no offline answer stub).
- **Metrics** (`benchmark/metrics.py`) — per-relation P/R/F1 + 6×6 confusion, the **mergeable**
  decision (EXACT∪PERCEPTUAL) scored one-vs-rest (the axis every track shares), the
  **false-merge / precision-on-DISTINCT** safety numbers, abstention rate, and the headline
  **divergence-by-true-relation** (over-merge vs under-merge), with the hard-negative slice
  reported separately.
- **The runner** (`benchmark/runner.py`) — candidate pairs (output-based generation ∪ the
  planted `G0` structure) → per-track verdicts → full per-pair distance measurements (for
  calibration, computed independently of stop-at-first) → metrics. Device-parameterized for the
  GPU run; the verifier path stays text-blind (handed a `ComparatorView` + `OutputProvider`).
- **Human-verification + κ** (`benchmark/human_review.py`) — extracts the SEMANTIC/SUBSUMPTION ∪
  UNCERTAIN slice to a JSON labeling template, loads completed annotator files, and computes
  **Cohen's / Fleiss' κ**. With no annotations the agreement is reported `status="pending"` — the
  κ value is never invented.
- **Calibration wiring** — `calibrate_from_result` does the **cluster-disjoint** split (a family
  is wholly in calibration or in test; cross-cluster pairs dropped to prevent leakage) and fits
  τ_perc/τ_sem/δ on the calibration cluster, stamping the provenance the config validator
  requires (CLAUDE.md §3.5.5).
- **Report** (`benchmark/report.py`) — `report.md`, `divergence.csv`, `pairs.csv`, a run
  `manifest.json` (git SHA, canon version, battery hash, backends, seed), the `review_template.json`,
  and an optional `divergence.png` (matplotlib, the `viz` extra; the table is emitted regardless).
- **CLI** — `viscurate run-benchmark` (`--device cuda` for the H200, `--clip`, `--calibrate`,
  `--no-ml` for an offline wiring smoke).

**How the real run is launched (on the H200)**

```bash
pip install -e ".[ml,viz]"
viscurate build-probes -c configs/probes.yaml -o data/probe_images   # 177-probe battery incl. COCO
viscurate run-benchmark --device cuda --clip --calibrate --date 2026-06-24 \
    -o results/phase4_benchmark
```

**Exit criteria (status)**

- ✅ candidate pairs incl. hard cases; auto-labels (`G0`); text baselines; divergence table +
  figure; calibration on a cluster-disjoint split — **all built and tested**.
- ⏳ the **divergence pattern itself** (text judge merges `blur_gaussian`/`blur_box`; output judge
  says DISTINCT) must be confirmed on the **real backends** — that is the go/no-go and is **not
  yet run**. The engineered hard negatives + the planted structure guarantee the *constructed*
  divergence exists at an appropriate operating point; the magnitude is an empirical question for
  the GPU run.
- ⏳ κ on the SEMANTIC/SUBSUMPTION slice — pending real annotators (infra + template ready).

**Wiring smoke (offline, this box).** `run-benchmark --no-ml` over a 137-probe synthetic battery
runs end to end and renders every artifact. Even **without** ML the verifier scores SUBSUMPTION
at **P/R/F1 1.000/0.857/0.923** (grid search is EXACT-only, so one rotation needing PERCEPTUAL
tolerance is missed) and `name-match` over-merges **5/6** hard negatives the verifier keeps
separate — the divergence shape is already visible. PERCEPTUAL/SEMANTIC resolution and the
embedding-cosine operating point need the ML run.

**Honest limitations / watch-items carried forward**

- **No real numbers produced here.** The CPU box lacks the `[ml]` backends; the H200 run is the
  source of every reported metric. Nothing in `results/` is committed yet (Phase 9 ties manifests).
- **Embedding-cosine operating point.** A single fixed τ on the full-corpus TF-IDF can be
  conservative (few merges → low divergence). Per-pair similarities are recorded in `pairs.csv`,
  so the fair comparison — the text baseline at *its own* best operating point (or a swept ROC) —
  is a reporting choice, not a re-architecture. Recommended for the real run.
- **`G0` is design intent.** SEMANTIC/SUBSUMPTION labels (esp. the looser edge-family neighbours,
  Canny/Laplacian) are routed to human re-certification; the answer key for the *experiments*
  comes from injected truth in Phase 5, not these hand labels (CLAUDE.md risk register).

**Tests:** 30 new (`tests/test_baselines.py` — 6; `tests/test_benchmark.py` — 24), all
deterministic fake-backend / pure-metric. Source gate green: **ruff + ruff-format + mypy
--strict** clean across 44 source files. (See the Phase-4 environment note below re: the 4
pre-existing macOS-only executor failures.)

**Environment note (macOS dev box).** This phase was built on macOS, where the Phase-1
`SandboxedExecutor` cannot set `RLIMIT_AS` (`setrlimit` raises `ValueError: current limit exceeds
maximum limit` — a Darwin quirk), so its 4 subprocess tests fail locally. This is **pre-existing
and unrelated to Phase 4** (the benchmark uses the in-process `BatteryEvaluator`); it passes on
the Linux/WSL2/H200 target. A graceful-degradation fix is recommended but deferred for
human review (the executor is the security-sensitive sandbox).

---

## Phase 5 — Corruption generator ✅ (machinery + QA complete)

Phase 5 turns the clean base library ``L0`` into a *family* of corrupted libraries ``L_ρ``
indexed by ``(ρ, composition c, seed, mode)`` by injecting the seven defect types (CLAUDE.md
§2.2) at a controlled rate. It lives in the new `src/viscurate/corruption/` package and is
**ML-free** (pure numpy/pydantic) — the graded ρ-series the headline pollution-vs-performance
curve rides on.

> **Go/no-go note.** This is *generation machinery*, not an experiment: building it fabricates
> no results and does not presume the Phase-4 divergence outcome. The experiments that *consume*
> ``L_ρ`` (Phases 7–8) still wait on the Phase-4 go/no-go (the real LPIPS/DINO/CLIP divergence run
> on the H200). Phase 5 was requested explicitly and is safe to land ahead of that gate.

**Architecture — two pure functions around a serializable log.** The corruption **log is the
canonical artifact**; everything else is derived from ``(L0, log)``:

```
plan_corruption(L0, ρ, c, seed, mode) -> CorruptionLog        # deterministic
apply_corruption(L0, log, G0)         -> CorruptedLibrary      # pure replay → L_ρ, G_ρ, key
```

Skills are never pickled — corrupted/added fns are reconstructed from **named factories**
(`mutators.py`) the log references (a mutator name, a baked value, an outer-op id …). So
"same ``(ρ, c, seed, mode)`` → byte-identical ``L_ρ``" reduces to "the planner is deterministic"
(it is) — verified on `registry.to_json()`, the `G_ρ` spec, and the ideal-action key.

**Delivered** — `src/viscurate/corruption/`

- **`types.py`** — `CorruptionType` (the 7), `CorruptionEntry` / `CorruptionLog` (replayable,
  flat-union fields, `bool`-before-`int` so a baked angle `90` is not coerced to `90.0`),
  `IdealAction` / `IdealActionKind` (the **ideal-action key** — `merge/parameterize/modify/remove`),
  `CorruptionManifest`, and the three built-in compositions (`uniform`, `duplicate_heavy`,
  `metadata_heavy`).
- **`mutators.py`** — the named, replayable fn factories: generic output mutators (`roll_h`,
  `swap_channels`, `zero_border`, `invert_values`) for IMPLEMENTATION_BUG / (conditional)
  DOMAIN_SCOPED_BUG, each **self-guaranteeing divergence** (a mutator that no-ops on a given
  output — e.g. `swap_channels` on a grayscale-replicated output — cascades to a guaranteed
  change, keeping "buggy ⇒ measurably wrong" an invariant); the EXACT / PERCEPTUAL (≤1-LSB
  dither) duplicate wrappers; the fixed-parameter SUBSUMPTION specialization; and the
  cross-family DEAD_SKILL composition `outer(inner(x))`.
- **`plan.py`** — deterministic selection of ``K = round(ρ·N)`` distinct **sites**, with
  **eligibility-aware Hamilton apportionment** over the composition (SUBSUMPTION needs a numeric
  param to bake, PARAM_SCHEMA_BUG needs a param with an alternate value; the rest accept any
  skill). Restricted pools are claimed first; an unfillable heavily-weighted restricted type
  **spills its deficit to the flexible types and the realized composition is recorded honestly**
  — never silently forced. Mixed mode adds a co-occurring defect under the rule **"at most one
  output-altering defect per site, plus optional misleading metadata"** (the realistic "broken
  *and* mis-described" case, kept independently verifiable).
- **`apply.py`** — replays the log: folds all in-place defects per site into one mutated skill,
  builds the added skills, and derives ``G_ρ`` by **appending** the log's relation deltas
  (EXACT/PERCEPTUAL for duplicates, directional SUBSUMPTION for specializations) to ``G0`` and
  re-running the existing `GroundTruthGraph` validators (DAG + EXACT transitive closure) — *no
  existing label is ever recomputed*. Corrupted skills **masquerade**: `provenance` stays
  `"builtin"`; the defect lives only in the internal-only `is_buggy` / `is_dead` labels or in
  the relational structure, so the agent must *discover* it (CLAUDE.md §1.2).
- **`qa.py`** — the per-type confirming assertions (CLAUDE.md §2.2): IMPLEMENTATION_BUG /
  DOMAIN_SCOPED_BUG **diverge** from the clean reference (the latter only on its targeted
  domain, unchanged on RGB); METADATA_MISLEAD / PARAM_SCHEMA_BUG leave outputs **unchanged**
  (text/schema changed, fn untouched); duplicates are EXACT / within-tolerance PERCEPTUAL;
  the SUBSUMPTION specialization reproduces the donor at its baked binding; DEAD_SKILL is
  non-trivial, distinct from its donors, and flagged. Compared at **matched seeds** against
  freshly-run clean ``L0`` (self-contained — no oracle file needed). Co-occurrence-aware in
  mixed mode (an output-invariant assertion checks only the part still isolable).
- **`grid.py` + `configs/corruption.yaml`** — the ``(ρ, c, seed, mode)`` driver: ρ ∈
  {10%…100%} × 3 compositions × 5 seeds × {single, mixed}, writing per instance the full
  ground-truth bundle (`corruption_log.json`, `library.json`, `g_rho.json`,
  `ideal_actions.json`, `qa_report.json`, `manifest.json` with `l0`/`g0` hashes, canon version,
  and realized counts).
- **CLI** — `viscurate corrupt -c configs/corruption.yaml -o data/corruption` (QA runs when a
  built battery is present; `--no-qa` for a battery-free wiring run).

**Exit criteria (all met).**

- ✅ **same seed → byte-identical library** — `plan` deterministic, `apply` pure; verified on
  the log, `registry.to_json()`, and the `G_ρ` spec.
- ✅ **expected per-type counts** — uniform ρ=0.3 → `{impl 5, meta 5, dup 4, sub 4, param 4,
  domain 4, dead 4}` (Hamilton, seed-independent); single-type compositions concentrate;
  restricted-pool deficits spill and are recorded.
- ✅ **buggy skills measurably wrong vs the reference** — confirmed by the per-type QA over a
  multi-domain battery: **all 150 (ρ × c × mode × seed) sampled instances QA-clean**, including
  every ρ=1.0 grid point.
- ✅ **G_ρ derived deterministically from G0 + log (no relabeling)** — deltas appended, DAG +
  EXACT-closure validated even at ρ=1.0 `duplicate_heavy`; planted G0 relations (subsumptions,
  hard negatives) survive untouched.

**Tests:** 29 new (`tests/test_corruption.py`), all deterministic/ML-free — determinism &
purity, ρ-semantics & Hamilton counts, the eligibility-cap spill, each of the 7 injectors'
invariants, `G_ρ` derivation/validation, the ideal-action key, mixed-mode co-occurrence, and
the grid driver's artifacts. **Source gate green: ruff + ruff-format + mypy --strict clean across
51 source files; 223 passed, 4 deselected (slow).**

**Honest limitations / decisions worth remembering.**

- **PERCEPTUAL duplicates are a ≤1-LSB dither**, not a full re-implementation — within any
  reasonable τ_perceptual yet not byte-identical. At the *uncalibrated* default ε (1/255) a
  PERCEPTUAL dup may classify as EXACT; both license `merge`, so it is a within-mergeable
  confusion, not a wrong action. Documented, not hidden.
- **Donor-as-site accounting.** For the add-types (duplicate / subsumption / dead) the *donor*
  is the counted corruption site (it stays clean; the new skill carries the defect). This keeps
  "K skills are corruption sites" well-defined; it is the stated interpretation of ρ.
- **Mixed mode** is bounded to ≤1 output-altering defect per site (+ optional metadata) so each
  defect's invariant stays observable; richer co-occurrence is a documented extension point.
- **The 4 macOS executor failures persist** (pre-existing `RLIMIT_AS` Darwin quirk, unrelated —
  Phase 5 uses in-process `Skill.run`, never the sandbox); green on Linux/H200.

---

## Phase 6 — Curation environment ✅ (machinery + sandbox boundary complete)

Phase 6 builds the **agent half** of the verifier/agent split (CLAUDE.md §3.2, §3.5.7): the
curatable library, the eight-action API, the **hard verifier gate** on structural edits, the
relation→action map, usage-aware advisories, agent adapters, action logging, and the **sandbox
trust boundary** that keeps agent-generated code blocked. It lives in the new
`src/viscurate/curation/` package and reuses the Phase-3 verifier unchanged.

> **Go/no-go note.** This is *environment machinery*, not an experiment — it fabricates no
> results and does not presume the Phase-4 divergence outcome. The studies that *consume* it
> (Phases 7–8) still wait on the Phase-4 go/no-go. Phase 6 was requested explicitly and is safe
> to land ahead of that gate.

**The verifier/agent split is enforced structurally.** The environment is the *only* place the
two meet. It hands the verifier a `ComparatorView` + an `OutputProvider` (never a description),
gates `merge` / `parameterize` on the certifying relation, and applies a structural edit only
through `CurationEnvironment.apply`, which calls the gate first.

**Delivered** — `src/viscurate/curation/`

- **`actions.py`** — the eight actions (`add / remove / modify / retrieve / merge / split /
  parameterize / end`) as a frozen, flat-union `Action`, plus `ActionResult` (the canonical,
  JSON-serializable log record) with `ActionStatus` (`applied / rejected / blocked / noop /
  invalid`) and the structured `rejection_feedback()` of §3.5.7. Field convention
  (`primary` = acted-on, `secondary` = survivor) mirrors `IdealAction` so Phase-8 scoring lines
  up directly.
- **`state.py`** — `CurationState` / `SkillSummary` built from `SkillMetadata.agent_view`, so the
  internal `is_buggy` / `is_dead` labels are **absent by construction** — a buggy/dead skill is
  indistinguishable from a clean one in the state (CLAUDE.md §1.2). `UsageStats` is the **Layer-E
  seam** (CLAUDE.md §2.4): counts + query-relevance from any source (empty / hand-specified /
  synthetic); the query-driven usage that makes "should we merge?" diverge from "can we?" lands
  in Phase 7.
- **`sandbox.py`** — the **execution-trust boundary**. `ExecutionPolicy` blocks `trusted=False`
  (agent-generated) skills from being executed or output-verified, with a structured
  review-required reason; `allow_untrusted` MUST stay `False` outside a reviewed sandbox. The
  deferred, human-review-gated controls (network namespace, restricted FS, rlimits + timeout, no
  `eval`/`exec` of skill source) are documented in `HARDENING_PLAN`. **No untrusted execution is
  implemented or enabled** — the boundary is made explicit and testable (CLAUDE.md §5).
- **`gating.py`** — the relation→action map as a pure function of the verifier's
  `RelationResult`: `merge` needs EXACT/PERCEPTUAL; `parameterize` needs SUBSUMPTION in the
  correct direction (`primary ⊑ secondary`; the reverse is rejected with a swap hint) or
  SEMANTIC_PRESERVING. A denied edit returns the relation + deciding distances + permitted
  alternatives.
- **`environment.py`** — `CurationEnvironment.apply` dispatches every action: structural edits
  pass the **trust gate** (both skills must be verifiable) then the **verifier gate**; permitted
  edits fold the duplicate/specialization away and keep the survivor; `remove` surfaces a usage
  warning; `modify` does output-preserving metadata/schema repair (stays trusted; fn edits are
  the untrusted path, deferred); `add` registers a `trusted=False` skill blocked from execution;
  `split` is blocked pending the hardened sandbox. Every action is structlog-logged and appended
  to the serializable history. `run_episode` drives an agent to `end` / budget / `max_steps` and
  returns an `EpisodeResult` with the library-compression Pareto axis.
- **`agent.py`** — the `CurationAgent` protocol with `ScriptedAgent` (deterministic; the
  `no-curation` / `accumulate-only` baseline substrate) and `LlmCurationAgent` behind the
  swappable `LlmClient` text-completion protocol (the same one the Phase-4 judge uses). Two
  clients ship: `OllamaClient` (local, multi-model, dependency-free stdlib HTTP; `list_ollama_models`
  enumerates installed models, CLAUDE.md D7) and `AnthropicClient` (Claude API, optional `[agent]`
  extra, lazy import; **Claude Opus 4.8 + adaptive thinking**, key from env per §5). With no
  client the agent raises rather than fabricating; an unparseable reply ends the episode cleanly.
- **CLI** — `viscurate curate` (`--instance` to curate an L_ρ bundle by replaying its corruption
  log, `--actions` for a scripted policy, `--ollama-model` / `--anthropic` for the LLM, `--no-ml`
  for an offline EXACT/SUBSUMPTION run); writes `action_log.json` + `episode.json`.
- **Config** — `CurationConfig` (`budget`, `usage_fold_threshold`) — the action-cost Pareto axis
  and the §3.5.7 usage gate (closes roadmap open item 5 with a default + knob).
- **Verifier reason enrichment (Phase-3 touch).** `taxonomy.classify` now threads the worst-case
  LPIPS measured in the PERCEPTUAL stage into the fall-through relations' distances (and the
  DISTINCT reason string), so a rejected merge carries the `"… L∞ …, LPIPS …"` evidence of
  CLAUDE.md §3.2/§3.5.7. All Phase-3/4 tests stay green.

**Exit criteria (all met, asserted in tests).**

- ✅ **`merge(blur_gaussian, blur_box)` rejected, with LPIPS in the rejection** — over the
  matched-sweep battery the verifier returns a non-mergeable relation and the `ActionResult`
  carries the worst-case `lpips`; a DISTINCT pair's rejection *reason string* literally quotes
  `LPIPS`.
- ✅ **exact-duplicate merge approved** — two skills sharing an `fn` classify EXACT, the merge
  folds the duplicate away (size 2→1), the canonical survives.
- ✅ **untrusted skill BLOCKED** — an agent-`add`ed skill is `trusted=False`; any structural edit
  touching it is BLOCKED with the review-required reason; the in-process verifier never runs it.
- ✅ **`end()` clean** — `end` is a NOOP that terminates `run_episode`; the budget also bounds
  the episode.
- ✅ **actions logged** — every action appends a JSON-serializable `ActionResult`; the episode
  log round-trips.

**Tests:** 20 new (`tests/test_curation.py`), all deterministic / ML-free (fake perceptual
backend, like Phase 3). Source gate green: **ruff + ruff-format + mypy --strict clean across 58
source files; 243 passed, 4 deselected (slow)**.

**Honest limitations / decisions worth remembering.**

- **The hardened sandbox is deliberately *not* implemented.** Agent-generated code stays blocked
  (CLAUDE.md §5); `allow_untrusted` is a documented, review-gated switch. `split` and fn-level
  `modify` (both needing trusted new code) are correspondingly out of v1.
- **`parameterize` removes the redundant specialization but does not yet materially extend the
  generalizer's schema** — functional coverage is preserved because the survivor subsumes the
  folded skill; richer "absorb the param knob" edits are a documented extension point.
- **Usage is supplied, not yet query-driven.** Phase 6 carries `UsageStats` and applies the
  usage advisory; the Zipfian/query-relevance usage that makes the §3.5.7 "decline if heavily
  used" gate bite is wired in Phase 7's query stream. The verifier *permits* on the relation
  (a fact); using usage to decline a permitted edit is the agent's call.
- **The 4 macOS executor failures persist** (pre-existing `RLIMIT_AS` Darwin quirk, unrelated —
  Phase 6's tests use the in-process `BatteryEvaluator`, never the subprocess sandbox); green on
  Linux/H200.

---

## Phase 7 — Query stream + downstream eval ✅ (machinery complete)

Phase 7 adds the task/usage layer that Phase 6 was waiting for: held-out `Query` objects,
query-derived `UsageStats`, solver policies, downstream scoring, and one-command artifacts. This
is still machinery, not a study result — the full pollution-vs-performance runs remain gated on
the Phase-4 divergence go/no-go.

**Delivered** — `src/viscurate/downstream/`

- **`query.py`** — `QueryStep`, `PredicateSpec`, `Query`, and `QueryManifest`. The manifest records
  split labels, input/reference hashes, the clean L0 reference pipeline, expected skill ids, and
  task predicates. It validates unique ids, dev/test skill-disjointness, and optional query/probe
  hash disjointness.
- **`build.py` + `configs/queries.yaml`** — deterministic held-out query generation. Inputs and
  clean references are written as `.npy` arrays; the manifest is the small artifact. Default pilot
  queries cover color, geometry, masks, blur, and edges across dev/test with skill-disjoint splits.
- **`predicates.py`** — task predicates layered on top of reference matching:
  exact shape, grayscale/channel equality, binary mask, RGBA output, and changed-from-input.
- **`usage.py`** — query relevance now produces the Phase-6 `UsageStats` object. Counts are
  deterministic Zipf-style synthetic usage frequencies over referenced skills; unreferenced skills
  remain zero/absent. This closes the Phase-6 limitation that usage was only supplied by hand.
- **`solver.py`** — `SolverAgent` protocol plus `ExpectedSkillSolver` (upper-bound/reference
  pipeline), `KeywordRetrievalSolver` (dependency-free lexical retriever), and `NoOpSolver`.
  Plan execution blocks `trusted=False` skills, preserving the Phase-6 sandbox boundary.
- **`evaluate.py`** — runs a solver over a query manifest and scores success as
  reference-output match (canonical L∞, optional LPIPS) **and** all task predicates passing.
- **`report.py`** — writes `report.md`, `scores.csv`, `scores.json`, `summary.json`, and a run
  manifest. Reports explicitly state when thresholds are still uncalibrated placeholders.
- **CLI** — `viscurate build-queries`, `viscurate run-downstream`, and `viscurate curate
  --queries-dir` so curation episodes can observe query-derived usage before choosing actions.

**Exit criteria (met, asserted in tests).**

- ✅ Query manifests are deterministic and dev/test skill-disjoint.
- ✅ Query inputs can be checked against the probe manifest and rejected on hash overlap.
- ✅ Reference outputs regenerate from clean L0 pipelines and the expected-skill solver succeeds
  on the clean library.
- ✅ Query-derived `UsageStats` marks expected skills as referenced and leaves unused skills
  unreferenced.
- ✅ A deliberately corrupted skill degrades downstream success, and restoring the clean skill
  restores it.
- ✅ Untrusted skills remain blocked during downstream plan execution.
- ✅ Downstream reports write traceable artifacts; no study numbers are fabricated.

**Tests / gate.** 7 new tests (`tests/test_downstream.py`). Focused checks pass:
`pytest tests/test_downstream.py`, `ruff check` + `ruff format --check` on touched files, and
`mypy src` (strict) across 66 source files.

**Honest limitations / decisions worth remembering.**

- The default query stream is a pilot-sized, deterministic task layer. Scaling it to the full
  Phase-8 studies is a config/data task, not a change to the evaluator.
- `ExpectedSkillSolver` is an upper bound and wiring oracle. The dependency-free
  `KeywordRetrievalSolver` is a baseline; stronger solver/LLM policies belong in Phase 8.
- LPIPS scoring is optional and only used when the `[ml]` backend is constructed. Offline runs use
  the canonical L∞ gate and clearly record that no perceptual backend ran.
- The Phase-4 divergence go/no-go is still the blocker for interpreting any downstream curves.

---

## Phase 8 — Metrics, baselines, and studies ✅ (aggregation machinery complete)

Phase 8 adds the study-level aggregation layer over the artifacts produced by Phases 4, 6, and 7.
It computes the paper-facing quantities — action quality against the ideal-action key, curation
Pareto inputs, construct-validity correlations, and the output-vs-text gating ablation — without
executing experiments or fabricating missing rows.

**Delivered** — `src/viscurate/studies/`

- **`metrics.py`** — pure study metrics:
  - `StudyPoint` seed-level rows for one method on one `(ρ, composition, seed, mode)` instance.
  - `score_actions()` scores applied curation actions against `ideal_actions.json`; `merge` and
    `parameterize` match directionally on `(kind, primary, secondary)`, while `modify` / `remove`
    match on `(kind, primary)`. `KEEP` ideals are not required actions; `retrieve` / `end` are not
    predicted repairs.
  - `intrinsic_curation_score()` uses ideal-action F1 with a penalty for rejected / blocked /
    invalid budget-spending actions. This gives the construct-validity proxy without peeking at
    downstream success.
  - `aggregate_points()` computes mean ± normal-approximation 95% CI across seed rows.
  - `pareto_front()` / `aggregate_pareto_front()` compute the non-dominated curation front
    (success and compression maximize; action cost minimizes).
  - `construct_validity()` computes Pearson and Spearman correlation between intrinsic curation
    score and downstream success.
  - `vision_matters_ablation()` matches output-gated and text-gated rows by
    `(ρ, composition, seed, mode)` and reports success / compression / action-cost deltas.
  - `equivalence_track_summaries()` reuses the Phase-4 benchmark result to produce Study-1 rows
    for each judge track (mergeable P/R/F1, false-merge rate, hard-negative false-merge rate,
    abstention).
  - `load_study_points()` reads JSON or CSV seed-level rows, so raw study runs can be aggregated
    without adding another execution path.
- **`report.py`** — writes the Phase-8 artifact bundle: `report.md`, `points.csv`,
  `aggregates.csv`, `pareto.csv`, `vision_matters_ablation.csv`, `construct_validity.json`,
  `vision_matters_ablation.json`, `manifest.json`, and an optional `pareto.png` when matplotlib is
  installed. The manifest records the input point file, methods, gates, matched ablation count,
  correlation summary, and git SHA from the CLI.
- **CLI** — `viscurate phase8 --points <points.json|points.csv> -o results/phase8_studies`.
  The command aggregates supplied real seed-level rows and refuses an empty input. It is a report
  generator, not a hidden experiment runner.

**Exit criteria (machinery met, asserted in tests).**

- ✅ Action logs are scored against the ideal-action key with direction-sensitive structural
  matching and budget penalties.
- ✅ Curation Pareto aggregation reports success / compression / action-cost means and CIs.
- ✅ Construct-validity correlation computes Pearson and Spearman over seed-level libraries.
- ✅ Vision-matters ablation matches output-gated vs text-gated rows and reports deltas.
- ✅ Phase-8 reports write traceable CSV/JSON/Markdown artifacts and a manifest.
- ✅ The CLI smoke test writes the complete report bundle from a supplied point manifest.

**Tests / gate.** 5 new tests (`tests/test_studies.py`) cover action scoring, aggregation, Pareto
fronts, correlations, ablation matching, report writing, JSON/CSV loaders, CLI smoke, and reuse of
Phase-4 benchmark metrics. Focused checks pass: `pytest tests/test_studies.py
tests/test_benchmark.py::test_runner_produces_the_headline_divergence`, `ruff check` on touched
files, and `mypy src/viscurate/studies src/viscurate/cli.py`.

**Honest limitations / decisions worth remembering.**

- This is **aggregation machinery**, not the completed experiment grid. Real Phase-8 numbers still
  require the Phase-4 divergence go/no-go and actual `(ρ, composition, seed, mode, method)` runs.
- Confidence intervals are normal-approximation 95% CIs over supplied seed rows. If the final paper
  uses bootstrap CIs, this module can add a bootstrap option without changing the row schema.
- The CLI consumes summarized `StudyPoint` rows; Phase 9 remains responsible for the full
  one-command experiment runner tying raw run directories, configs, battery hashes, model versions,
  and manifests together.

---

## Phase 9 — Experiment runner + paper artifacts ✅ (machinery complete)

Phase 9 ties the lower-phase artifact writers into a manifest-backed reproducibility bundle. It
does **not** fabricate missing empirical results: absent Phase-4/Phase-8 artifacts are recorded as
`pending` in the realism audit.

**Delivered** — `src/viscurate/experiments/`

- **`config.py` + `configs/phase9.yaml`** — one declarative experiment surface: runtime config,
  probe/query/corruption configs, ground-truth and param-alignment paths, data/result artifact
  directories, benchmark backend knobs, and the optional real `StudyPoint` input file.
- **`manifest.py`** — writes a Phase-9 run manifest with git SHA, seed, Python/platform,
  VisCurate/canonicalization versions, thresholds, optional model package versions, config-file
  hashes, probe/query/oracle manifest hashes, Phase-4/Phase-8 result manifest hashes, and
  `reproduce.sh` commands.
- **`audit.py`** — the realism audit: probe licenses/domain/format coverage, query split
  disjointness and query/probe hash disjointness, corruption-grid completeness, Phase-4 benchmark
  artifact presence/calibration status, `StudyPoint` presence, and Phase-8 report artifacts.
- **`runner.py`** — `run_phase9` writes `run_manifest.json`, `realism_audit.json`,
  `realism_audit.md`, `reproduce.sh`, `experiment_config.json`, `index.json`, and, when a real
  `points_path` is configured, Phase-8 paper tables/figures under `paper_artifacts/`.
- **CLI / README** — `viscurate phase9 -c configs/phase9.yaml -o results/phase9` is the
  one-command repro entry point.

**Exit criteria (met as machinery).**

- ✅ Every generated paper-artifact bundle traces back to manifests and config hashes.
- ✅ Missing empirical runs are explicitly `pending`; no placeholder numbers are emitted.
- ✅ Probe licenses are audited from the manifest; query/probe disjointness is checked when both
  manifests are present.
- ✅ One-command Phase-9 repro is documented in `README.md`.

**Tests / gate.** 2 new tests (`tests/test_experiments.py`) cover the runner bundle, audit
statuses, paper-artifact generation from real `StudyPoint` rows, and CLI smoke. Focused checks pass:
`pytest tests/test_experiments.py`, `ruff check src/viscurate/experiments src/viscurate/cli.py
tests/test_experiments.py`, and `mypy src/viscurate/experiments src/viscurate/cli.py`.

**Honest limitations / decisions worth remembering.**

- Phase 9 is the orchestration/reporting layer. It makes empirical gaps visible; it does not run the
  full GPU benchmark or curation grid by itself.
- Real paper numbers still require the Phase-4 divergence go/no-go and actual
  `(ρ, composition, seed, mode, method)` study rows.
