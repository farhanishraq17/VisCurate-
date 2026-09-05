# VisCurate GPU Runbook

**For:** whoever is running this on the 96 GB-VRAM machines
**Repo:** `VisCurate_Experiments` (currently at `D:\Research_Ishmam\Viscurate\Serious Deadlines\VisCurate_Experiments\`)
**Written:** 2026-08-18

---

## Read this first — the two things that matter most

**1. Most of the pipeline already exists and has never been run on real backends.** The repo has a tested end-to-end CLI (264 passing tests, `mypy --strict` clean). Stage 1 below is not new code — it is running what is already there, on a GPU, for the first time. Its own status report calls this the project's go/no-go.

**2. Do Stage 0 before any long run.** The telemetry hooks cannot be added retroactively. A 12-hour run without them produces no cost data and has to be repeated.

**Do not skip stages.** Stage 1 is a genuine go/no-go: if the divergence does not appear, stop and report back rather than proceeding to Stage 2. The project is architected around that checkpoint.

---

## Hardware notes

96 GB VRAM is far more than this pipeline needs. The comparators (LPIPS-AlexNet, DINO ViT-B/16, CLIP ViT-B/32) are small, inference-only, and the code deliberately loads **one model at a time** (`equivalence/backends.py`, designed for a 6 GB budget).

**Implications:**
- Do **not** try to raise batch sizes to "use" the VRAM — the bottleneck is skill execution (CPU/OpenCV) and, in Stage 3, LLM API latency.
- **Multiple GPUs help most by running independent instances in parallel** — Stage 2's corruption grid is 300 embarrassingly-parallel instances. One process per GPU, `CUDA_VISIBLE_DEVICES` per shard.
- Expect GPU util to look *low*. That is correct, not a misconfiguration.

---

## Stage 0 — Setup and verification (~30 min)

### 0.1 Environment

```bash
cd /path/to/VisCurate_Experiments
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,viz]"
```

> **Torch/torchvision pinning matters.** The repo notes that a plain PyPI `torchvision` can drag in a mismatched CUDA torch. Install the matched pair from the PyTorch CUDA index explicitly if the default resolution misbehaves.

### 0.2 Verify the gate is green

```bash
ruff check . && ruff format --check . && mypy src && pytest -q
```

**Expected:** all clean. Two known, harmless failures may appear:
- a **Windows-only** UTF-8 test bug in `tests/test_benchmark.py` (`.read_text()` missing `encoding="utf-8"`)
- **macOS-only** `RLIMIT_AS` executor failures (4 tests)

Both are documented and environment-specific. **On Linux both should pass** — if they fail on Linux, stop and report.

### 0.3 Confirm the ML backends actually load on GPU

```bash
pytest -q -m slow          # real-backend smoke tests (LPIPS + DINO + CLIP)
```

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), "| n_gpu:", torch.cuda.device_count())
print("device0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY
```

### 0.4 Install the telemetry layer ← **do not skip**

Copy from this planning folder into the repo:

```bash
mkdir -p src/viscurate/instrument
cp <plans>/code/viscurate_instrument_telemetry.py src/viscurate/instrument/telemetry.py
printf 'from viscurate.instrument.telemetry import recorder, timer, gpu_timer\n' \
  > src/viscurate/instrument/__init__.py
```

Then wire the call sites (small, mechanical — see [`code/viscurate_instrument_telemetry.py`](code/viscurate_instrument_telemetry.py) docstring for the event schema):

| Wire into | Emit |
|---|---|
| `equivalence/compare.py` — `BatteryEvaluator` output cache | `signature_compute(..., cache_hit=...)` |
| `equivalence/taxonomy.py` — `classify` return | `pair_verify(..., deciding_stage=..., short_circuited=...)` |
| `equivalence/candidates.py` — `candidate_pairs` | `candidate_gen(...)` |
| `curation/agent.py` — `AnthropicClient` / `OllamaClient` | `llm_call(tokens_in=..., tokens_out=...)` |
| `curation/environment.py` — `run_episode` | `agent_episode(...)` |

```bash
export VISCURATE_TELEMETRY="$PWD/results/telemetry.jsonl"
```

**Sanity check:** run any short command and confirm `results/telemetry.jsonl` gains a `run_manifest` line plus real events.

---

## Stage 1 — The go/no-go divergence run ★ HIGHEST PRIORITY

**This is the single most valuable thing on the list.** It produces the paper's Table 2 and Table 3 for the first time, and resolves an open question about where the draft's current numbers came from.

### 1.1 Build the probe battery

```bash
viscurate build-probes -c configs/probes.yaml -o data/probe_images
```

**Verify:** `data/probe_images/manifest.json` exists, has **177** entries, and **no entry has `license: unknown`**.

### 1.2 Freeze the reference oracle

```bash
viscurate freeze-oracle -c configs/probes.yaml --probes-dir data/probe_images -o data/oracle
```

Re-running a skill over the battery must reproduce the stored hash. This oracle proves corruption took effect later; it is **never** used to assign relation labels.

### 1.3 The divergence benchmark ← the go/no-go

```bash
viscurate run-benchmark \
    --device cuda \
    --clip \
    --calibrate \
    --date "$(date +%F)" \
    --probes-dir data/probe_images \
    --ground-truth configs/ground_truth_g0.yaml \
    --param-alignment configs/param_alignment.yaml \
    -o results/phase4_benchmark
```

Runtime: minutes to a couple of hours. Watch for OOM (shouldn't happen at 96 GB) and for skills erroring on degenerate probes.

### 1.4 ★ The go/no-go check

Open `results/phase4_benchmark/report.md` and `divergence.csv`, and answer:

| Question | Where | Pass condition |
|---|---|---|
| Does the output verifier call `blur_gaussian` vs `blur_box` **DISTINCT**? | `pairs.csv`, hard-negative slice | **Yes** — this is the headline case |
| Do text judges **over-merge** hard negatives the verifier keeps apart? | `divergence.csv` | Yes, some over-merge |
| False merges on truly-DISTINCT pairs? | `report.md` safety table | Ideally 0 |
| Did calibration produce real thresholds? | run `manifest.json` | `calibrated: true` with provenance |

**If divergence appears → proceed to Stage 2.**
**If it does not → STOP and report back.** The project plan is explicit: understand why before building further. Do not proceed on the assumption it will show up later.

### 1.5 Send back

`report.md`, `divergence.csv`, `pairs.csv`, `manifest.json`, `divergence.png`, `results/telemetry.jsonl`.

> **Note on `G_0` size.** `configs/ground_truth_g0.yaml` defines **24 designed pairs** (7 subsumption, 10 semantic, 1 complementary, 6 hard negatives), everything else defaulting to DISTINCT. The paper draft currently claims 944 designed pairs. **Report the number this run actually produces** — do not adjust anything to match the paper. Resolving that discrepancy is the point. See [`CODEBASE_RECONCILIATION.md`](CODEBASE_RECONCILIATION.md) §1.

> ⚠ **The 6-pair hard-negative slice is statistically very weak.** Observing 0 false merges out of 6 has an exact 95% upper bound of **39%**. It cannot support a safety claim. **n ≥ 59** hard negatives are needed for a ≤5% bound. Report `0/6` with its interval; expanding the slice is a separate task (Stage 5).

---

## Stage 2 — Corruption grid + curation episodes

**Only after Stage 1 passes.** This produces the paper's Table 1.

### 2.1 Generate the grid

```bash
viscurate corrupt \
    -c configs/corruption.yaml \
    --ground-truth configs/ground_truth_g0.yaml \
    --probes-dir data/probe_images \
    -o data/corruption
```

Produces `(ρ ∈ {0.1…1.0}) × 3 compositions × 5 seeds × {single, mixed}` = **300 instances**, each with `L_ρ`, `G_ρ`, corruption log, and ideal-action key. Same seed → byte-identical library.

**Verify:** 300 instance dirs; per-type defect counts match the composition vectors; QA assertions pass (Type 2 metadata-mislead and Type 7 dead-skill must leave outputs **unchanged**; Type 1 must change them).

### 2.2 Curation episodes

One episode per (instance × model). **This is the expensive stage — it is LLM-latency-bound, not GPU-bound.**

```bash
export ANTHROPIC_API_KEY=...   # never commit this

viscurate curate \
    --instance data/corruption/<instance_name> \
    --probes-dir data/probe_images \
    --device cuda --clip \
    --anthropic --model claude-sonnet-5 \
    -o results/curation/<model>/<instance_name>
```

> ⚠ **The CLI default `--model claude-opus-4-8` is stale.** Pass a current id explicitly: `claude-opus-5`, `claude-sonnet-5`, or `claude-haiku-4-5`. Local models via `--ollama-model <name>`.

**Cost control — do this before the full sweep:**
1. Run **10 instances** with one model.
2. Read actual token counts from `results/telemetry.jsonl`.
3. Extrapolate to `300 × n_models`. **Confirm the total before committing.**
4. If too expensive, subsample the grid — and record that the grid was subsampled.

**Parallelism:** shard instances across GPUs.

```bash
# one shard per GPU
CUDA_VISIBLE_DEVICES=0 ./run_shard.sh 0 &
CUDA_VISIBLE_DEVICES=1 ./run_shard.sh 1 &
```

### 2.3 Aggregate

```bash
viscurate phase8 --points results/curation/points.json -o results/phase8_studies
viscurate phase9 -c configs/phase9.yaml -o results/phase9
```

`phase8` gives the Pareto front, construct-validity correlation, and the vision-matters ablation. `phase9` writes the manifest bundle and realism audit.

---

## Stage 3 — A2 new baselines (CPU-friendly; can run alongside Stage 2)

Closes the paper's unevaluated claim that **source code** is a poor proxy — currently only name and description are tested.

### 3.1 Install

```bash
cp <plans>/code/viscurate_baselines_embedders.py   src/viscurate/baselines/embedders.py
cp <plans>/code/viscurate_baselines_code_judges.py src/viscurate/baselines/code_judges.py
pip install sentence-transformers transformers
```

### 3.2 The four new rungs

| Rung | Class | Notes |
|---|---|---|
| 3 sentence-embedding | `SentenceTransformerEmbedder` → existing `EmbeddingCosineJudge` | no new judge needed — the embedder protocol is already swappable |
| 4 code-embedding | `CodeEmbeddingJudge` | **docstrings stripped** — otherwise it leaks rung 3's signal |
| 5 AST clone | `AstCloneJudge` | **run BOTH variants**, see below |
| 7 LLM-on-source | `LlmSourceJudge` | model must be disjoint from every curation subject |

> ⚠ **Rung 5 must report both variants.** `structure-only` canonicalizes literals to type placeholders, which makes it blind to defect type (v) — parameter-default drift — *by construction*. Reporting only that variant rigs the baseline to fail. `semantics-preserving` (literals + defaults + called APIs retained) is the one the paper's claim must actually beat.

### 3.3 Fairness rules — these change the result, not just presentation

1. **Sweep every threshold-based rung and report AUPRC**, including the two already shipped. `name-match` is currently pinned at τ=0.5 and TF-IDF at τ=0.6 — comparing those fixed points against optimized new rungs is unfair *in our favour*, which a reviewer will catch. The repo's own Phase-4 notes already flag this as a known issue.
2. **Compare at matched operating points** — for each rung, the most permissive threshold holding precision-on-non-equivalence ≥ 0.99 subject to recall ≥ 0.5.
3. **Report false-discovery rate on the merge decision**, not only precision-on-non-equivalence. With 926 distinct vs ~18 non-distinct pairs, the latter is inflated by class imbalance — predicting DISTINCT for everything already scores 0.981.
4. **Unparseable LLM replies are `UNCERTAIN`, not `DISTINCT`.** The shipped `LlmJudge` maps them to DISTINCT, which hides model failure inside a safety number. Report the abstention rate separately.

---

## Stage 4 — A1 Corpus A: cross-library audit (CPU-only, no sandbox needed)

The fast, high-signal half of the real-library requirement. **No GPU required**, no untrusted code.

```bash
pip install opencv-python pillow scikit-image kornia albumentations torchvision
```

**Steps:**
1. Scrape the Albumentations↔Torchvision (~94 pairs) and Albumentations↔Kornia (75 pairs) published mapping tables. Record retrieval date and page snapshots.
2. Hand-author cross-library **parameter alignment maps** in the repo's existing `configs/param_alignment.yaml` format — the mechanism already exists (`equivalence/param_alignment.py`), only the maps are new.
3. **Hash and commit the frozen pair list + alignment maps BEFORE any execution.** Authored from documentation only.
4. Audit the **functional cores** (`cv2.*`, `kornia.filters.*`, `torchvision.transforms.v2.functional`, `PIL.ImageOps`), not the random transform wrappers — keeps it deterministic and inside the paper's current scope.
5. **Pass the self-pair sanity gate** — every function vs. itself must certify EXACT — before interpreting any divergence.
6. Run through the existing `taxonomy.classify` at the **frozen Stage-1 operating point**. **Do not re-calibrate on this corpus.**

Full protocol: [`A1/01_corpus_A_cross_library.md`](A1/01_corpus_A_cross_library.md).

> **Wording discipline:** these mapping tables are *category-level* judgments ("same broad operation"), not claims of pixel equivalence. Report "a category-level judgment that does not survive execution" — **never** "the maintainers were wrong."

---

## Stage 5 — Blocked / longer-lead items

| Item | Blocker |
|---|---|
| **A1 Corpus B** (ComfyUI real nodes) | Needs the **hardened sandbox**, which is deliberately unimplemented (`curation/sandbox.py`; `allow_untrusted` must stay `False`). Every community node is untrusted third-party code. |
| **A2 trojan defect** | Same sandbox boundary, higher stakes — deliberately executing malicious side effects needs real OS-level containment, not `sys.addaudithook` (that is observability, not containment). |
| **Expand hard negatives to n ≥ 59** | Design work in `configs/ground_truth_g0.yaml` — no blocker, just needs doing. **High value per hour**: it is what makes the headline safety claim statistically meaningful. |
| **A3 neural/stochastic** | Largest new build (conditional per-probe distributional inference). Also crosses a scope line the project explicitly deferred (`claude.md`: *"No generative/diffusion skills in v1"*, marked `[CONFIRM]` and never closed). Confirm before spending GPU time. |

---

## Reporting back

After each stage, send:

```
results/<stage>/report.md
results/<stage>/manifest.json
results/<stage>/*.csv
results/telemetry.jsonl
```

Plus a short note: what ran, what failed, anything surprising.

**Report numbers exactly as produced.** If they disagree with the paper draft, that is the finding — the draft's current tables do not trace to this codebase, and establishing what is actually true is the whole point of Stage 1.

---

## Rules that hold across every stage

1. **Never re-calibrate thresholds on evaluation data.** Fit on the frozen cluster-disjoint calibration split; carry it over unchanged. The repo enforces provenance stamping; do not work around it.
2. **Freeze and hash before running** — pair lists, pack lists, alignment maps.
3. **Every rate gets raw counts and an exact (Clopper–Pearson) interval.** `0/6` is not a result; `0/6 [0, 39%]` is.
4. **Never fabricate.** "Not yet run" is an acceptable answer; the repo already holds this line strictly (`CLAUDE.md §5`) and its report generators refuse to invent missing rows.
5. **Report negative results.** If VisCurate underperforms, that is the finding. A generalization gap found by us is a contribution; one found by a reviewer is a rejection.
