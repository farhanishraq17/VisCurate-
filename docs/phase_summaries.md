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
