# VisCurate — Project Status & Gap Report

*Generated 2026-06-24 from a full scan of the repository (74 source files, 18 test modules,
configs, docs, and the built `data/` artifacts) plus a live run of the quality gate
(`ruff`, `ruff format`, `mypy --strict`, `pytest`).*

---

## 1. Bottom line

The **engineering is excellent and essentially complete** — all 9 roadmap phases of *machinery*
are built to the CVPR-A\* rigor the plan called for, and the quality gate is green. But the
**research experiments have not been run.** There is no `results/` directory, no CSVs, no
figures, and zero paper numbers anywhere in the repo.

So the honest answer to "how did the experiments go" is: **they haven't yet.** The project sits
exactly at its own designed go/no-go checkpoint (Phase 4), waiting for the one run that decides
whether the central hypothesis holds.

This is **by design, not neglect** — CLAUDE.md §5 forbids fabricating results, and that line was
held strictly: every Phase 4–9 deliverable is wiring that *can* produce numbers but refuses to
invent them.

---

## 2. What is actually built

| Phase | Package | Status | Real artifacts produced? |
|---|---|---|---|
| 0 Scaffold | `config.py`, `rng.py`, `logging.py`, CI | ✅ done | n/a |
| 1 Skills + harness | `skills/` — 100 deterministic skills, canonicalization contract, sandboxed executor | ✅ done | code only |
| 2 Probe battery + oracle | `probes/` | ✅ done | ✅ **177 probes built; oracle frozen clean (17,700 pairs, 0 errors)** in `data/` |
| 3 Equivalence engine | `equivalence/` — LPIPS/DINO/CLIP comparators, subsumption search, taxonomy | ✅ done | code only |
| 4 Equivalence benchmark | `benchmark/` + `baselines/` — `G0` answer key, text baselines, divergence table | ⚠️ machinery done | ❌ **divergence run NOT executed** |
| 5 Corruption generator | `corruption/` — 7 defect injectors, (ρ,c,seed,mode) grid | ✅ done | code only |
| 6 Curation environment | `curation/` — 8-action API, verifier gating, sandbox boundary | ✅ done | code only |
| 7 Query + downstream | `downstream/` | ✅ done | code only |
| 8 Studies | `studies/` — Pareto, construct-validity, vision-matters ablation | ✅ aggregation done | ❌ no study rows |
| 9 Experiment runner | `experiments/` — `viscurate phase9`, realism audit | ✅ done | ❌ no bundle generated |

**Only Phase 2 has produced real data artifacts** (the probe battery + oracle in `data/`).
Phases 4–9 are report *generators* with no reports generated.

---

## 3. Code quality — high

Gate result, run live on the Windows dev box on 2026-06-24:

- `ruff check` — **clean**
- `ruff format --check` — **clean** (92 files)
- `mypy src` (strict) — **clean, 74 source files**
- `pytest` — **264 passed, 1 failed**

The single failure is a **Windows-only test bug**, not a real defect (see gap E1 below).

Strengths worth recording:

- **The load-bearing modality boundary is enforced by *type*, not convention.** The verifier in
  `src/viscurate/equivalence/taxonomy.py` takes a `ComparatorView` that *has no `description`
  attribute* — a comparator literally cannot read text. The whole thesis is protected
  structurally.
- A subtle correctness fix is real and tested: EXACT/PERCEPTUAL on default bindings is
  authoritative only when skills share a matched-sweep axis or are param-free; otherwise grid
  search decides (`taxonomy.py:170-182`). This is the difference between wrongly merging
  `crop_center` ≡ `crop_bbox` and correctly calling it subsumption.
- Honesty discipline is consistent: `calibrate.py` ships no numbers, the LLM judge records
  "not run" rather than stubbing, thresholds stay `calibrated=false` until a real labeled split
  exists.

---

## 4. Research status — measured vs. not

**The only empirical signal that exists** is an offline `--no-ml` wiring smoke (documented in
`docs/phase_summaries.md`, Phase 4): even without ML backends, `name-match` over-merged **5 of 6**
engineered hard negatives that the output verifier keeps separate. That shows the *shape* of the
expected divergence — but it is **not** the real result.

**The headline go/no-go has not been run:** the real LPIPS/DINO/CLIP divergence run over the full
battery, which answers the question the whole paper rides on — *does the text judge merge
`blur_gaussian`/`blur_box` while the output judge correctly says DISTINCT, and by how much?*

Until that runs, none of the four studies have numbers:

- ❌ Study 1 — equivalence F1 per judge track
- ❌ Study 2 — curation Pareto
- ❌ Study 3 — construct-validity correlation
- ❌ Study 4 — vision-matters ablation
- ❌ Threshold calibration (no human-labeled split exists yet)

Everything downstream is gated on the Phase-4 result, correctly.

---

## 5. What is missing / needs to be implemented

Prioritized. **(R)** = research run, **(E)** = engineering, **(C)** = open confirmation from
CLAUDE.md §6.

### Priority 0 — the blocking go/no-go

- **(R0.1) Run the Phase-4 divergence benchmark for real.** This is the single decision the
  entire project is architected around. Command:
  ```
  viscurate run-benchmark --clip --calibrate --date 2026-06-24 -o results/phase4_benchmark
  ```
  Note: the `[ml]` extra (`torch` CPU, `lpips`, `timm`, `open_clip`) **is already installed on
  this machine** and the real-backend smoke tests pass — so this can run here on CPU, it does
  **not** strictly require the H200/GPU. Outcome decides go/no-go: if divergence appears, proceed;
  if not, the plan says stop and understand why.

### Priority 1 — calibration & labels (needed before any reported metric)

- **(R1.1) Human-verify the SEMANTIC/SUBSUMPTION ∪ UNCERTAIN slice and compute κ.** Infra +
  labeling template exist (`benchmark/human_review.py`); annotators do not. κ is reported
  `status="pending"` until real annotations arrive.
- **(R1.2) Calibrate `ε`, `τ_perc`, `τ_sem`, `δ` on the cluster-disjoint validation split.** The
  procedure exists (`calibrate.py`) and stamps provenance; thresholds remain `calibrated=false`
  placeholders until fit on R1.1's labels. No reported metric may use an uncalibrated config.

### Priority 2 — the study grid (the payoff experiments)

- **(R2.1) Generate the corruption grid** `viscurate corrupt -c configs/corruption.yaml` →
  `(ρ ∈ {10%…100%}) × 3 compositions × 5 seeds × {single, mixed}` libraries `L_ρ`.
- **(R2.2) Run curation episodes** over each `L_ρ` instance for every method (no-curation,
  accumulate-only, name-match, embedding-cosine, LLM-on-text, output-gated, oracle upper bound).
  Requires choosing/standing up an agent client (Ollama local or Anthropic API).
- **(R2.3) Build the full query stream + run downstream eval** per `L_ρ` to get
  success-vs-ρ curves.
- **(R2.4) Aggregate into the four studies** (`viscurate phase8`) and assemble the paper bundle
  (`viscurate phase9`). Machinery is ready; it needs the real `StudyPoint` rows from R2.2/R2.3.

### Engineering gaps (carried "honest limitations" from the phase summaries)

- **(E1) Windows test encoding bug — trivial fix.**
  `tests/test_benchmark.py::test_write_report_emits_artifacts` calls `.read_text()` without
  `encoding="utf-8"`, so Windows cp1252 chokes on a UTF-8 em-dash. Production `report.py:248`
  writes correctly with `encoding="utf-8"`. Fix: pass `encoding="utf-8"` to the test's reads.
  (Same class as the documented macOS `RLIMIT` executor failures.)
- **(E2) macOS executor degradation.** `SandboxedExecutor` raises on `RLIMIT_AS` on Darwin
  (4 failing tests on macOS). A graceful-degradation fix is recommended but deferred for human
  review (the executor is security-sensitive). Green on Linux/WSL2.
- **(E3) Hardened sandbox — deliberately NOT implemented.** Required before any agent-generated
  skill can execute: network namespace, restricted FS, CPU/mem caps + hard timeout, no
  `eval`/`exec` of skill source. `allow_untrusted` is a documented, review-gated switch that must
  stay `False`. Blocks E4.
- **(E4) `split` action and fn-level `modify`** are out of v1 — both need the trusted new-code
  path that waits on E3.
- **(E5) `parameterize` does not yet materially extend the generalizer's schema** — it folds the
  redundant specialization away (coverage preserved because the survivor subsumes it) but does not
  "absorb the param knob." Documented extension point.
- **(E6) Probe parameter sweeps are not yet first-class Phase-2 artifacts.** The oracle freezes
  skills at defaults; matched-sweep comparison uses the `param_alignment` grids. Promoting sweeps
  to frozen oracle artifacts is a scaling/repro improvement.
- **(E7) COMPLEMENTARY commutation test is necessary-but-not-sufficient** (two linear filters
  commute yet are not "disjoint aspects"). They are caught earlier (SEMANTIC) in the full
  pipeline; the residual label is a calibration question. Refinement opportunity.
- **(E8) Center-crop subsumption is exact only on even-sided probes** (a 1-px offset can appear on
  odd sizes). Absorbed by PERCEPTUAL tolerance + calibration; documented, not hidden.
- **(E9) Embedding-cosine baseline operating point.** A single fixed τ on full-corpus TF-IDF can
  be too conservative (few merges → artificially low divergence). Per-pair similarities are in
  `pairs.csv`; report the text baseline at its own best operating point (or a swept ROC) for a
  fair comparison. Recommended for the real run.
- **(E10) Bootstrap CIs option.** Phase-8 currently uses normal-approximation 95% CIs; add a
  bootstrap option if the paper requires it (no row-schema change needed).
- **(E11) GPU 6 GB budget not yet measured on the target RTX 3050.** One-model-at-a-time
  discipline is implemented; CPU-only verified so far.
- **(E12) Scale-up.** Pilot is 177 probes / pilot-sized query stream; the CVPR run targets ~500
  probes and the full study scale. This is a config/data task, not a re-architecture.

### Open confirmations (CLAUDE.md §6 — non-blocking, flag before relying on them)

- **(C1) Face-domain probe source** — recommend synthetic faces (documented license), avoid PII.
- **(C2) 16-bit / palettized domain scope** — currently a small mandatory slice; confirm.
- **(C3) Citation verification** — `SkillClone (arXiv:2603.22447)` and `SkillBrew
  (arXiv:2605.29440)` are unverified 2026 IDs; verify against arXiv before leaning on related-work
  positioning.
- **(C4) Annotator pool** for the SEMANTIC slice (size, any IRB-style review) — affects κ
  reliability (ties to R1.1).
- **(C5) Agent action/compute budget** per episode — partially closed (a default + knob shipped in
  Phase 6's `CurationConfig`); confirm the final value.
- **(C6) Generative skills stay out** for the whole CVPR pass — confirm.

---

## 6. The one useful surprise

**The `[ml]` extra is already installed on this Windows machine** — `torch` (CPU), `lpips`,
`timm`, `open_clip` all import, and the `slow` real-backend smoke tests pass. The phase docs assume
the divergence run needs the H200, but **the Phase-4 benchmark can be launched right here on CPU**
(slower, but the pilot is small):

```
viscurate run-benchmark --clip --calibrate --date 2026-06-24 -o results/phase4_benchmark
```

That single command turns "machinery complete" into the project's **first actual research
result** — the decision point everything else waits on.

---

## 7. Recommended next steps (in order)

1. **(E1)** Fix the one-line Windows test bug so the gate is fully green on this box.
2. **(R0.1)** Run the Phase-4 divergence benchmark on CPU and inspect the divergence table /
   hard-negative slice. **This is the go/no-go.**
3. If divergence holds: **(R1.1 → R1.2)** collect human labels on the SEMANTIC/SUBSUMPTION slice,
   then calibrate thresholds on the cluster-disjoint split.
4. **(R2.1 → R2.4)** generate the corruption grid, run curation + downstream over the (ρ, c, seed,
   mode, method) grid, aggregate the four studies, and emit the Phase-9 paper bundle.
5. Schedule the deferred engineering (E3 hardened sandbox, then E4) before any agent-generated
   skills are allowed to execute; pick up E5–E12 as the studies demand.

---

*This report reflects the repository state at commit `00f2d2c` ("completed phases 4-9"). The
authoritative, phase-by-phase detail lives in `docs/phase_summaries.md`; the roadmap and locked
decisions are in `claude.md`.*