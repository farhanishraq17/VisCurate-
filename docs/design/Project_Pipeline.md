# VisCurate — Project Pipeline

The complete build plan, end to end, for review before I continue. It is derived from
`claude.md` (the roadmap) and reflects what is already built. Nothing here changes the
roadmap's locked decisions — it makes the execution order, data flow, and per-phase
contracts explicit so you can sanity-check direction before I spend effort.

**Legend:** ✅ done · 🔜 next · ⬜ planned · ⚠️ go/no-go checkpoint

---

## 0. The one-paragraph thesis (what the whole pipeline serves)

A *skill* is a deterministic image→image function. The contribution is to decide whether two
skills are **equivalent by executing them and comparing outputs**, not by comparing their
text descriptions — and then to use those output-grounded relations to **gate library
curation** (merge / split / parameterize). Two results ride on the machinery and neither can
be dropped: (1) an **equivalence benchmark** showing output-grounding wins where text judges
fail, and (2) a **curation application** showing output-gated curation beats text-gated
curation on downstream task success. Everything below exists to produce those two results
honestly, with error bars and a reproducibility manifest.

---

## 1. The data & method flow (the spine)

```
 Layer A  Audited primitive ops        ← trusted atoms behind each skill.fn
 Layer B  Clean base library L0 (100) ─► designed relation graph G0
            freeze reference outputs               │
 Layer C  Probe battery P (versioned) ─────────────┤
            inject defects (rate ρ, composition c) │
 Layer D  Corrupted library L_ρ ─► graph G_ρ + corruption log + ideal-action key
 Layer E  Task / usage layer T ─► query relevance + usage frequency
                         │
        ┌────────────────┴───────────────────┐
   VERIFIER (outputs only)            AGENT (repository decisions)
   "what IS the relation?"            "what SHOULD be done?"
        │  certifying relation gates structural edits  │
        └──────────────► curation actions ◄────────────┘
                         │
            Downstream solver + 4 studies ─► paper artifacts
```

**The hard architectural rule threaded through every phase:** the output-grounded path never
reads a skill's `description`. It is enforced *by type* (the verifier is handed a
`ComparatorView` with no `description` attribute), not by convention.

---

## 2. Phase-by-phase pipeline

### Phase 0 — Scaffold ✅
- **Build:** package layout, Pydantic-validated config, seeded RNG (no global state),
  structlog JSON logging, ruff + mypy(strict) + pytest, CI.
- **Exit:** install/lint/type/test all clean; every subpackage imports. **Met.**

### Phase 1 — Skills + harness  ✅ (100/100) — COMPLETE

Phase 1 turns the roadmap's natural-language skill blueprint into the project's executable
substrate, ML-dependency-free by design (torch/LPIPS/DINO only enter in Phase 3). The
detail below records everything that was built and verified.

**Exit criteria (all met):** each skill runs on a 256²-class probe and returns a valid array;
same `(img, params, seed)` → byte-identical output; the timeout skill returns an error rather
than hanging; `trusted=False` → BLOCKED; the registry round-trips through JSON and a reloaded
skill executes identically. **144 tests green; `ruff`, `ruff format --check`, `mypy --strict`
clean.**

#### (a) Components built

- **`Skill` model** — [src/viscurate/skills/model.py](../../src/viscurate/skills/model.py).
  `id, name, description, fn, params_schema, metadata`. Two load-bearing boundaries are
  enforced *by type*, not convention:
  - `Skill.comparator_view()` → a `ComparatorView` dataclass with **no `description`
    attribute**, so the output-grounded path cannot read text even by accident (§1.2);
  - `SkillMetadata.agent_view()` omits the internal-only `is_buggy` / `is_dead` labels.
  - `ParamSpec` / `ParamsSchema` give typed coercion (`int/float/bool/str/enum`), range +
    `choices` checks, default-fill, and rejection of unknown params; `SkillSpec` is the
    JSON-serializable view (everything but the callable).
- **Registry** — [src/viscurate/skills/registry.py](../../src/viscurate/skills/registry.py).
  Ordered, id-unique, JSON-serializable; persists `SkillSpec` records and re-binds `fn` by id
  via a resolver (the built-in library's function table). Round-trip fidelity and
  identical-re-execution are tested.
- **Canonicalization contract** —
  [src/viscurate/skills/canonicalize.py](../../src/viscurate/skills/canonicalize.py),
  `CANON_VERSION = "1.0.0"`. float32 [0,1] metric view + uint8 hash view; 1-ch → 3-ch
  replicate; RGBA composited over fixed mid-gray with alpha tracked separately;
  binary-mask detection + IoU; shape gate (L∞ = ∞ on mismatch); `content_hash` (SHA-256)
  that collides for EXACT duplicates and never across shapes.
- **Sandboxed executor** —
  [src/viscurate/skills/executor.py](../../src/viscurate/skills/executor.py) +
  [_worker.py](../../src/viscurate/skills/_worker.py) +
  [_testkit.py](../../src/viscurate/skills/_testkit.py). Fresh subprocess per execution, wall-clock
  timeout, POSIX `RLIMIT_AS`/`RLIMIT_CPU` (timeout-only on Windows), a **hard trusted gate**
  (`trusted=False` never spawns a child), and structured `ExecutionResult`
  (`ok / output / error / duration_s / timed_out / blocked / returncode`).
- **The 100-skill library** — [src/viscurate/skills/library/](../../src/viscurate/skills/library/)
  (`geometric.py`, `color.py`, `filtering.py`, `reconstruction.py`, with `_ops.py` helpers
  and `_build.py` builders). The clean base **L0**: defect-free (bugs/duplicates/dead skills
  are injected later in Phase 5).

#### (b) The 100 skills (complete enumeration)

**Geometric / canvas — 25** (`geometric`): `flip_horizontal`, `flip_vertical`, `rotate_90`,
`rotate_180`, `rotate_270`, `transpose`, `rotate_canvas_degrees`, `rotate_45`, `translate`,
`shear_horizontal`, `shear_vertical`, `scale_xy`, `crop_center_percentage`,
`crop_bounding_box`, `random_crop`, `zoom_in_center`, `resize_nearest`, `resize_bilinear`,
`resize_bicubic`, `resize_fixed`, `pad_reflect`, `pad_replicate`, `pad_constant`,
`pad_to_square`, `tile_2x2`.

**Colour / exposure / contrast — 25** (`color`): `grayscale_bt601`, `grayscale_bt709`,
`invert`, `brightness_add`, `brightness_multiply`, `exposure_stops`, `contrast_scale`,
`gamma_correct`, `saturation_gain`, `desaturate_partial`, `hue_shift`, `sepia_tone`,
`color_temperature_shift`, `color_balance_rgb`, `channel_isolate` (enum param),
`channel_swap_rgb_bgr`, `threshold_binary`, `solarize`, `posterize`, `quantize_uniform`,
`linear_contrast_stretch`, `contrast_stretching_percentile`, `levels_adjust`,
`equalize_histogram_global`, `equalize_clahe`.

**Signal / blur / edges / morphology — 25** (`blur`, `sharpen`, `frequency`, `edges`,
`texture`, `morphology`, `denoise`, `noise`, `stylize`): `blur_gaussian`, `blur_box`,
`blur_median`, `blur_bilateral`, `blur_motion`, `sharpen_unsharp`, `sharpen_laplacian_kernel`,
`high_pass_spatial`, `emboss`, `edges_sobel`, `edges_scharr`, `edges_prewitt`,
`edges_laplacian`, `edges_canny`, `difference_of_gaussians`, `gabor_filter`, `low_pass_fft`,
`high_pass_fft`, `morphology_dilate`, `morphology_erode`, `morphology_open`,
`morphology_close`, `denoise_nlmeans`, `add_gaussian_noise`, `add_salt_pepper_noise`.

**Masks / reconstruction / synthesis — 25** (`reconstruction`, `segmentation`, `quantize`,
`stylize`, `synthesis`, `mask`): `inpaint_telea`, `inpaint_ns`, `flood_fill_center`,
`connected_components_colormap`, `contours_draw`, `distance_transform`, `threshold_otsu`,
`threshold_adaptive_mean`, `palette_reduce_kmeans`, `dither_floyd_steinberg`,
`pixel_sort_rows`, `mosaic_pixelate`, `vignette`, `gradient_map_duotone`,
`checkerboard_synthesize`, `border_frame`, `overlay_grid`, `sketch_pencil`, `cartoonize`,
`equalize_per_channel`, `blur_background_composite`, `mask_to_rgba`, `halftone_dots`,
`value_noise_synthesize`, `swirl_distort`. (All ids carry a `_v1` suffix; ids are stable and
never reused.)

#### (c) Determinism classification (§1.4, asserted in tests)

- **Seeded-stochastic (4):** `random_crop`, `add_gaussian_noise`, `add_salt_pepper_noise`,
  `value_noise_synthesize` — deterministic at a fixed seed, so comparable at matched seeds.
- **Precision-sensitive (2):** `low_pass_fft`, `high_pass_fft` — FFT round-trip; compared with
  PERCEPTUAL tolerance in Phase 3, not EXACT.
- **Platform-sensitive (1):** `palette_reduce_kmeans` (`cv2.kmeans`, seeded via `cv2.setRNGSeed`)
  — reproducible within a platform; compared within-platform only.

#### (d) Planted relations baked into L0 (what Phase 4 will find)

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
- **Complementary:** `morphology_dilate` ↔ `morphology_erode` (with `open`/`close` built on them).
- **RGBA / alpha path:** `mask_to_rgba` emits a 4-channel output exercising the alpha branch
  of the canonicalization contract.

#### (e) Tests (144) — [tests/](../../tests/)

config validation (incl. the `calibrated`-requires-provenance rule); RNG
determinism/independence/stability; param coercion + schema validation; the modality
boundary (`comparator_view` has no `description`) and the label boundary (`agent_view` hides
internal labels); registry round-trip + identical re-execution; the full canonicalization
contract (float range, 1-ch replicate, RGBA composite + alpha, mask IoU, shape gate, hash
collisions); **all 100 skills parametrized** for determinism + canonicalization; the
determinism-flag assignments; the planted relations (rotate_45/rotate_90 subsumptions, center
crop = bbox, linear = percentile, gaussian≠box, mask→RGBA, seeded-noise matching); and the
executor (blocked / timeout / crash / success-matches-in-process / byte-identical runs).

#### (f) Engineering notes / decisions worth remembering

- **Editable install in `compat` mode** (`pip install -e . --no-deps --config-settings
  editable_mode=compat`): the default strict editable finder uses a static module map, so
  new modules give `ModuleNotFoundError` until reinstall; compat puts `src/` on the path via a
  `.pth` and avoids that churn.
- **cv2 stubs vs runtime:** opencv's bundled type stubs are stricter than runtime for two calls
  accepting `None` (`normalize`, `kmeans`). `distance_transform` was rewritten to normalize in
  numpy; the `kmeans` `bestLabels=None` overload is narrowly `type: ignore`d. cv2 stays
  type-checked everywhere else.
- **Array convention is RGB;** colour ops coerce to 3-channel and drop alpha, geometric ops
  preserve channel count; `crop_bounding_box` parameters are fractional so the op is
  size-independent and the center-crop subsumption holds exactly.
- **Windows vs WSL2:** `resource` (rlimits) is POSIX-only, so the executor degrades to
  timeout-only on Windows (logged once); the POSIX branch is guarded by `sys.platform` so mypy
  on Windows treats it as unreachable.

#### (g) File map (Phase 0 + 1)

```
src/viscurate/
  config.py rng.py logging.py cli.py            # Phase 0 scaffold
  skills/
    model.py registry.py canonicalize.py        # model + registry + §1.3 contract
    executor.py _worker.py _testkit.py           # sandboxed execution + failure-path skills
    library/{geometric,color,filtering,reconstruction}.py  # the 100 skills (25 each)
    library/{_ops,_build}.py                      # shared helpers + skill builders
configs/default.yaml   tests/   docs/phase_summaries.md   ClaudeCode.md
```

### Phase 2 — Probe battery + reference oracle  ✅
- **Built** (`src/viscurate/probes/`): `synthetics.py` (deterministic CC0 generators across all
  domains + RGB/RGBA/gray/16-bit/palette + degenerate cases), `coco.py` (license-filtered
  natural photos — CC BY 2.0 / PD only, NoDerivs excluded, verified against live COCO metadata),
  `manifest.py` (License model that forbids `unknown`/non-redistributable; coverage assertions),
  `build.py` + `configs/probes.yaml` (orchestrator, `array_sha256`, reproducibility manifest),
  `oracle.py` (freeze with determinism self-audit + `verify_oracle`). CLI: `build-probes`,
  `freeze-oracle`.
- **Key contract:** the oracle confirms corruption took effect and scores verifier/agent —
  it is **never** used to assign relation labels.
- **Exit (met):** builder reproducible (same seed → same hashes); per-domain minimums + every
  degenerate case; no `license=unknown`; oracle re-run reproduces stored hashes. **Real pilot
  battery built (177 probes: 137 CC0 synthetic + 40 COCO CC BY, all domains/formats); reference
  oracle frozen clean — 17,700 pairs, all ok, spot-verify reproduces.**
- **Deferred to Phase 3:** parameter sweeps (the matched-grid inputs the comparators consume).

### Phase 3 — Comparators + taxonomy ✅  *(the `[ml]` extra lands here)*
- **Built** (`src/viscurate/equivalence/`): EXACT (hash + ε), PERCEPTUAL (LPIPS + SSIM
  cross-check), SEMANTIC (DINO ViT-B/16, CLIP optional), directional **subsumption search**,
  the hierarchical stop-at-first taxonomy with the **UNCERTAIN abstention band**, output-based
  candidate generation (fingerprints — *never* description-based), and the COMPLEMENTARY
  detector (non-triviality + approximate commutation, executing compositions). ML backends sit
  behind protocols with **lazy `torch` imports** (package imports without `[ml]`) and load
  **one model at a time**. The text-blind boundary is enforced *by type*: `classify` is handed a
  `ComparatorView` + an `OutputProvider`, never a `Skill`/`description`.
- **Calibration:** the *procedure* is implemented (`calibrate.py`) — τ_perc, τ_sem, δ fit to
  **precision on non-equivalence** subject to a recall floor, written to config with date +
  split hash. No numbers shipped: the labeled split is produced in Phase 4, so thresholds stay
  `calibrated=false` placeholders. No threshold is a literal in code.
- **Exit (met):** hand-built EXACT/PERCEPTUAL/DISTINCT pairs classify correctly; SUBSUMPTION /
  COMPLEMENTARY / SEMANTIC / UNCERTAIN each exercised; matched-sweep worst-case blocks a false
  `blur_gaussian`/`blur_box` merge; LPIPS+DINO+CLIP load and give sane distances (one model at a
  time; GPU 6 GB budget met by design, CPU-verified here). **178 tests green.**
- **Carried to Phase 4:** thresholds are uncalibrated; the commutation COMPLEMENTARY test is
  necessary-not-sufficient (two linear filters commute); center-crop subsumption is exact only
  on even-sided probes (PERCEPTUAL tolerance absorbs the rest). See `docs/phase_summaries.md`.

### Phase 4 — Equivalence benchmark ⚠️  **(first go/no-go checkpoint — machinery ✅, real run ⏳)**
- **Built (✅):** the answer key `G0` (`configs/ground_truth_g0.yaml` + `benchmark/ground_truth.py`,
  DAG/closure-validated); the text baselines in a **separate** `viscurate/baselines/` package
  (name-match, dependency-light TF-IDF embedding-cosine, LLM-on-descriptions behind a client
  protocol); the metrics (per-relation P/R/F1, mergeable decision, false-merge/precision-on-DISTINCT,
  **divergence-by-true-relation** with the hard-negative slice separate); the runner; the
  **human-verify + κ** infra (κ pending real annotators); **cluster-disjoint calibration** wiring;
  the report (table + CSV + manifest + optional figure); and the `viscurate run-benchmark` CLI
  (`--device cuda` for the H200). 30 new tests; source gate green.
- **The test of the premise (⏳ not yet run):** text judge merges `blur_gaussian`/`blur_box`;
  output judge says DISTINCT. The real divergence numbers require the **H200 run** with real
  LPIPS/DINO/CLIP over the full battery — one command (`run-benchmark --device cuda --clip
  --calibrate`). **If no divergence appears, stop and understand why before building curation.**
  No numbers are fabricated here; an offline `--no-ml` wiring smoke confirms the pipeline + report.

### Phase 5 — Corruption generator ✅ (machinery + QA complete; experiments wait on the Phase-4 gate)
- **Built** (`src/viscurate/corruption/`): the 7 defect injectors as **named, replayable fn
  factories** (`mutators.py`), a **deterministic planner** (`plan.py`, eligibility-aware Hamilton
  apportionment with honest deficit-spill), a **pure replay** (`apply.py` → `L_ρ`, `G_ρ`,
  ideal-action key), **per-type QA assertions** (`qa.py` — Type 1/6 diverge from the clean
  reference; Metadata-Mislead & Param-Schema leave outputs **unchanged**; Dead-Skill is
  flagged-and-distinct), and the **`(ρ, c, seed, mode)` grid driver** (`grid.py` +
  `configs/corruption.yaml`: ρ ∈ 10…100% × 3 compositions × 5 seeds × {single, mixed}). CLI:
  `viscurate corrupt`. The corruption **log is canonical**; everything derives from `(L0, log)`,
  so no skill is ever pickled.
- **Key contract:** `G_ρ` is `G0` **plus** the log's relation deltas (EXACT/PERCEPTUAL for
  duplicates, directional SUBSUMPTION for specializations), re-validated by the existing
  DAG/closure checks — *no existing label is recomputed*. Corrupted skills masquerade
  (`provenance="builtin"`); the defect lives only in `is_buggy`/`is_dead` or the relations.
- **Exit (met):** same seed → byte-identical library (log + `registry.to_json()` + `G_ρ` spec);
  expected per-type counts (uniform ρ=0.3 → `{5,5,4,4,4,4,4}` via Hamilton, seed-independent);
  buggy skills measurably wrong (**150 sampled `(ρ,c,mode,seed)` instances QA-clean**, incl. all
  ρ=1.0); `G_ρ` derived deterministically from `G0` + log. **29 tests; source gate green.**
- **Go/no-go:** this is *generation machinery* (fabricates nothing); the experiments that consume
  `L_ρ` (Phases 7–8) still wait on the **Phase-4 divergence run** (H200). Requested explicitly and
  safe to land ahead of the gate.

### Phase 6 — Curation environment ✅ (machinery + sandbox boundary complete)
- **Built** (`src/viscurate/curation/`): the state representation (`CurationState` /
  `SkillSummary` from `agent_view` — **no internal labels**) + `UsageStats` (Layer-E seam); the
  action API (`add/remove/modify/retrieve/merge/split/parameterize/end`) with serializable
  `ActionResult` logs; **verifier gating** (`gating.py`) — the relation→action map with the
  structured §3.5.7 rejection (relation + deciding distances + alternatives); the
  `CurationEnvironment` (trust gate → verifier gate → apply, usage advisories, structlog logging)
  + `run_episode`/`EpisodeResult`; the **sandbox trust boundary** (`sandbox.py` — untrusted code
  blocked, hardening plan documented, **not** enabled); and agent adapters (`ScriptedAgent`,
  `LlmCurationAgent` behind a swappable `LlmClient`, `OllamaClient` multi-model + optional
  `AnthropicClient`/Claude Opus 4.8). CLI: `viscurate curate`. `CurationConfig` adds the episode
  budget (Pareto action-cost axis). The Phase-3 verifier was touched only to thread worst-case
  LPIPS into rejection reasons.
- **Key contract:** the output-grounded path stays text-blind by type (the verifier gets a
  `ComparatorView` + `OutputProvider`); a structural edit can only be applied through
  `apply`, which calls the gate first; agent-generated skills are `trusted=False` and blocked
  from execution/verification pending the human-reviewed hardened sandbox (CLAUDE.md §5).
- **Exit (met):** `merge(blur_gaussian, blur_box)` rejected with LPIPS in the rejection;
  exact-dup merge approved (2→1); untrusted (added) skill BLOCKED; `end()` clean; every action
  logged + JSON round-trips. **20 tests; source gate green (ruff + ruff-format + mypy --strict;
  243 passed, 4 deselected slow).**
- **Carried forward:** the hardened sandbox is deliberately unimplemented (untrusted exec stays
  blocked); `split` and fn-`modify` need trusted new code (out of v1); usage is supplied, not yet
  query-driven (Phase 7 wires the Zipfian/query usage that makes the "decline if heavily used"
  gate bite).

### Phase 7 — Query stream + downstream eval ✅ (machinery complete)
- **Built** (`src/viscurate/downstream/`): `Query` / `QueryManifest`, deterministic query
  generation (`configs/queries.yaml`), task predicates, query-derived `UsageStats`,
  `ExpectedSkillSolver` / `KeywordRetrievalSolver` / `NoOpSolver`, downstream scoring, and
  report artifacts. CLI: `viscurate build-queries`, `viscurate run-downstream`, and
  `viscurate curate --queries-dir`.
- **Exit (met as machinery):** deterministic split-disjoint query manifests; clean L0 references
  regenerate and the expected-skill solver succeeds; deliberately corrupted skills degrade
  downstream success; untrusted skills remain blocked; reports write traceable artifacts. **7 tests;
  focused source gate green.**
- **Carried forward:** this is the task/usage/evaluator substrate. Interpreting pollution curves
  still waits on the Phase-4 go/no-go and Phase-8 study runs.

### Phase 8 — Metrics, baselines, studies ✅ (aggregation machinery complete)
- **Built** (`src/viscurate/studies/`): seed-level `StudyPoint` rows; ideal-action scoring for
  curation logs (direction-sensitive `merge` / `parameterize`, `modify` / `remove` by primary);
  intrinsic curation score; mean ± 95% CI aggregation across seeds; Pareto-front extraction over
  downstream success ↑, compression ↑, action cost ↓; construct-validity Pearson/Spearman
  correlation; output-vs-text gating ablation matched by `(ρ, composition, seed, mode)`; Phase-4
  equivalence-track summaries for Study 1; JSON/CSV row loaders; report writer for
  `report.md`, CSV/JSON tables, manifest, and optional Pareto figure. CLI:
  `viscurate phase8 --points <points.json|points.csv>`.
- **Exit (met as machinery):** action logs score against `ideal_actions.json`; CIs, Pareto front,
  construct-validity correlation, and vision-matters deltas are deterministic; report artifacts are
  manifest-backed and the CLI refuses empty input. **5 tests; focused source gate green.**
- **Carried forward:** this is the aggregation/report layer, not the completed study grid. Real
  Phase-8 numbers still require the Phase-4 divergence go/no-go and actual runs across the
  `(ρ, c, seed, method)` grid.

### Phase 9 — Experiment runner + paper artifacts ✅ (machinery complete)
- **Built** (`src/viscurate/experiments/`): per-experiment YAML config
  (`configs/phase9.yaml`), run manifest writer (git SHA, seed, package/model versions, config
  hashes, battery/query/oracle/result manifest hashes, thresholds), `reproduce.sh` generation,
  a realism audit over probe licenses, query/probe disjointness, corruption-grid completeness,
  Phase-4 benchmark artifacts, and Phase-8 study artifacts, plus paper table/figure generation
  when supplied real `StudyPoint` rows. CLI: `viscurate phase9`.
- **Exit (met as machinery):** every generated paper-artifact bundle traces to manifests; missing
  empirical runs are marked `pending`, not filled; licenses are audited from the probe manifest;
  README has the one-command repro entry. **Focused Phase-9 gate green.**
- **Carried forward:** real paper numbers still require the Phase-4 divergence go/no-go run and
  actual `(ρ, composition, seed, mode, method)` study rows.

---

## 3. Dependency order (what blocks what)

```
0 ─► 1 ─► 2 ─► 3 ─► 4 ⚠️ ─► 5 ─► 6 ─► 7 ─► 8 ─► 9
                    │              ▲
         (Phase 4 is the gate)     └─ Phase 5 machinery may be built ahead of the gate
                                       (it fabricates nothing); the runs that USE L_ρ
                                       (7–8) still wait for divergence to be confirmed.
```
- Phases 0–4 are **inference-light / dependency-light** until Phase 3 pulls in torch+LPIPS+
  DINO; **Phase 5 is ML-free again** (pure numpy/pydantic). Pilot everything small (100 skills,
  ~200 probes) before scaling via config knobs.
- The agent/LLM work (Phase 6) is where latency, not the GPU, dominates.

## 4. Immediate next step

Phases 0–3 are complete (100 skills; a 177-probe license-clean battery + frozen oracle; the
full output-grounded equivalence engine). Phase 4's **machinery** is complete; Phase 5 (the
corruption generator + per-type QA + the `(ρ, c, seed, mode)` grid), Phase 6 (the curation
environment + verifier gating + sandbox boundary + agent adapter), Phase 7 (query stream +
downstream evaluator + query-derived usage), and **Phase 8 (study metrics, aggregation, Pareto /
correlation / ablation reporting)** are now built and tested as machinery.

The blocking item remains the **Phase-4 divergence go/no-go**: the real LPIPS/DINO/CLIP run on
the H200 (`viscurate run-benchmark --device cuda --clip --calibrate`) to confirm the premise —
text judge merges `blur_gaussian`/`blur_box`; output judge says DISTINCT. **Phases 5–8 were
built ahead of this gate because they are machinery that fabricates nothing** (corruption
generation, curation, query construction/evaluation, and study aggregation code), but the *study
runs* that consume them must wait for divergence to be confirmed. If text and output judges do not
diverge, stop and understand why before running the curation studies.

Phase 9 is now built as machinery: `viscurate phase9 -c configs/phase9.yaml -o results/phase9`
writes the manifest-backed reproducibility bundle, realism audit, and paper artifacts when real
seed-level rows are supplied. The blocking item remains the Phase-4 divergence go/no-go before
interpreting or publishing study curves.

## 5. Non-negotiables carried through every phase

- **Never fabricate results.** `results/` and every figure come from a real run with a
  committed manifest. "Not yet run" is acceptable; invented numbers are not.
- **Deterministic by default.** Seeds recorded in manifests; "same output" must be decidable.
- **Output-grounded path is text-blind** (enforced by type).
- **Sandbox is security-sensitive.** Agent-generated skills stay blocked until the hardened
  sandbox is reviewed (Phase 6).
- **Calibration never leaks.** Cluster-disjoint splits, frozen before any test metric.
- **Verify third-party APIs against current docs**; pin versions only after a working install.

---

## 6. Open items still needing your call (from roadmap §6)

These don't block the immediate next step — I'll proceed with the stated default and flag in
summaries — but they affect Phase 2+:
1. **Face-domain probe source** — default: a small synthetic-face set (documented license).
2. **16-bit / palettized** — default: include as a small mandatory slice.
3. **Citations** (`SkillClone`, `SkillBrew`) — verify against arXiv before leaning on them.
4. **Annotator pool** for the SEMANTIC slice (size, review) — affects κ.
5. **Agent action/compute budget** — closed as a config knob (`curation.budget`, default 50);
   still tune per Phase-8 run manifest.
6. **Generative skills** — confirmed out of v1; confirm they stay out for the full CVPR pass.
