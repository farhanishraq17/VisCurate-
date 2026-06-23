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

- **`Skill` model** — [src/viscurate/skills/model.py](src/viscurate/skills/model.py).
  `id, name, description, fn, params_schema, metadata`. Two load-bearing boundaries are
  enforced *by type*, not convention:
  - `Skill.comparator_view()` → a `ComparatorView` dataclass with **no `description`
    attribute**, so the output-grounded path cannot read text even by accident (§1.2);
  - `SkillMetadata.agent_view()` omits the internal-only `is_buggy` / `is_dead` labels.
  - `ParamSpec` / `ParamsSchema` give typed coercion (`int/float/bool/str/enum`), range +
    `choices` checks, default-fill, and rejection of unknown params; `SkillSpec` is the
    JSON-serializable view (everything but the callable).
- **Registry** — [src/viscurate/skills/registry.py](src/viscurate/skills/registry.py).
  Ordered, id-unique, JSON-serializable; persists `SkillSpec` records and re-binds `fn` by id
  via a resolver (the built-in library's function table). Round-trip fidelity and
  identical-re-execution are tested.
- **Canonicalization contract** —
  [src/viscurate/skills/canonicalize.py](src/viscurate/skills/canonicalize.py),
  `CANON_VERSION = "1.0.0"`. float32 [0,1] metric view + uint8 hash view; 1-ch → 3-ch
  replicate; RGBA composited over fixed mid-gray with alpha tracked separately;
  binary-mask detection + IoU; shape gate (L∞ = ∞ on mismatch); `content_hash` (SHA-256)
  that collides for EXACT duplicates and never across shapes.
- **Sandboxed executor** —
  [src/viscurate/skills/executor.py](src/viscurate/skills/executor.py) +
  [_worker.py](src/viscurate/skills/_worker.py) +
  [_testkit.py](src/viscurate/skills/_testkit.py). Fresh subprocess per execution, wall-clock
  timeout, POSIX `RLIMIT_AS`/`RLIMIT_CPU` (timeout-only on Windows), a **hard trusted gate**
  (`trusted=False` never spawns a child), and structured `ExecutionResult`
  (`ok / output / error / duration_s / timed_out / blocked / returncode`).
- **The 100-skill library** — [src/viscurate/skills/library/](src/viscurate/skills/library/)
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

#### (e) Tests (144) — [tests/](tests/)

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

### Phase 3 — Comparators + taxonomy ⬜  *(the `[ml]` extra lands here)*
- **Build:** EXACT (hash + ε), PERCEPTUAL (LPIPS + SSIM cross-check), SEMANTIC
  (DINO ViT-B/16, CLIP optional), directional **subsumption search**, the hierarchical
  stop-at-first taxonomy engine with the **UNCERTAIN abstention band**, output-based
  candidate generation (fingerprints — *never* description-based), and the COMPLEMENTARY
  detector (non-triviality + approximate commutation, executing compositions).
- **Calibration:** ε, τ_perc, τ_sem, δ calibrated on the human-labeled validation split for
  **precision on non-equivalence**, cluster-disjoint from test, written to config with date +
  split hash. No threshold is a literal in code.
- **Exit:** hand-built EXACT/PERCEPTUAL/DISTINCT synthetic pairs classified correctly;
  thresholds calibrated and recorded; LPIPS+DINO fit in 6 GB (load one at a time).

### Phase 4 — Equivalence benchmark ⚠️  **(first go/no-go checkpoint)**
- **Build:** candidate pairs incl. hard cases; auto-labels; the text baselines
  (name-match, embedding-cosine, LLM-on-descriptions); **human-verify** the SEMANTIC /
  SUBSUMPTION subset (report κ); the **divergence table + figure**.
- **The test of the premise:** text judge merges `blur_gaussian`/`blur_box`; output judge says
  DISTINCT. **If no divergence appears, stop and understand why before building curation.**

### Phase 5 — Corruption generator ⬜
- **Build:** the 7 defect injectors, each with a per-type QA assertion (Type 1 diverges from
  oracle; Metadata-Mislead and Dead-Skill leave outputs **unchanged**). Emit `L_ρ`, `G_ρ`,
  corruption log, and ideal-action key over a `(ρ, composition c, seed)` grid; ρ ∈ 10…100%,
  ≥5 seeds, ≥3 compositions + a mixed mode.
- **Exit:** same seed → byte-identical library; expected per-type counts; buggy skills
  measurably wrong vs oracle; `G_ρ` derived deterministically from `G0` + log (no relabeling).

### Phase 6 — Curation environment ⬜
- **Build:** state representation (no internal labels), the action API
  (`add/remove/modify/retrieve/merge/split/parameterize/end`), **verifier gating** with
  structured rejection-and-reason, the relation→action map with usage gates, agent adapter
  (Ollama multi-model; Claude API optional), action logging. **Harden the sandbox here**
  (agent-generated code → `trusted=False`, blocked until reviewed).
- **Exit:** `merge(blur_gaussian, blur_box)` rejected with LPIPS in the reason; exact-dup
  merge approved; untrusted skill BLOCKED; `end()` clean; actions logged.

### Phase 7 — Query stream + downstream eval ⬜
- **Build:** `Query` objects (instruction, input, reference, predicates), the solver agent,
  correctness scoring (perceptual match + task predicates), stratified **disjoint** dev/test
  (no probe image appears in any query; splits disjoint in skills and queries).
- **Exit:** solver fails on some dev queries with the noisy library, succeeds more on a
  curated one; Dead-Skill queries have zero retrieval.

### Phase 8 — Metrics, baselines, studies ⬜
- **Build & run:** all baselines (incl. oracle as upper-bound only) and the four studies —
  (1) Equivalence F1 per judge track, (2) Curation Pareto (success ↑, compression, action
  cost) with CIs, (3) Construct validity (intrinsic score vs downstream success),
  (4) Vision-matters ablation (text-cosine gating vs output gating).
- **Exit:** every number from a real run with a manifest; Pareto figure shows all methods;
  mean ± CI across seeds.

### Phase 9 — Experiment runner + paper artifacts ⬜
- **Build:** per-experiment YAML configs (git SHA, seeds, model versions, battery SHA,
  thresholds), run manifests, one-command table/figure generators, realism-audit report.
- **Exit:** every paper number traces to a manifest; no placeholders; licenses documented;
  one-command repro in README.

---

## 3. Dependency order (what blocks what)

```
0 ─► 1 ─► 2 ─► 3 ─► 4 ⚠️ ─► 5 ─► 6 ─► 7 ─► 8 ─► 9
                    │
         (Phase 4 is the gate: do not build 5–9 until divergence is confirmed)
```
- Phases 0–4 are **inference-light / dependency-light** until Phase 3 pulls in torch+LPIPS+
  DINO. Pilot everything small (100 skills, ~200 probes) before scaling via config knobs.
- The agent/LLM work (Phase 6) is where latency, not the GPU, dominates.

## 4. Immediate next step

Phases 0–2 are complete (100 skills; a 177-probe license-clean battery + frozen oracle; 157
tests green). The next step is **Phase 3** — the output-grounded comparators (EXACT / PERCEPTUAL
LPIPS+SSIM / SEMANTIC DINO+CLIP), directional subsumption search, the stop-at-first taxonomy
with the UNCERTAIN abstention band, and threshold calibration. This is where the optional
`[ml]` extra (torch is already present; lpips / timm / open-clip / scikit-image install here)
lands, leading into the Phase-4 divergence go/no-go checkpoint.

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
5. **Agent action/compute budget** per curation episode — caps the Pareto action-cost axis.
6. **Generative skills** — confirmed out of v1; confirm they stay out for the full CVPR pass.
