---
name: run-curation-ml-sweep
description: Run the VisCurate Phase-6 curation agent over the full corruption sweep with the REAL visual stack (LPIPS + DINO + CLIP) gating, served by a local vLLM, parallelized across workers. Use when the user asks to run/benchmark/sweep a (local vLLM) model as the curation agent with real perceptual/semantic gating. Turnkey: only $MODEL changes between runs.
---

# Run Phase-6 Curation Sweep (real LPIPS + DINO + CLIP)

## TL;DR — the whole run, one knob

```bash
cd /scratch/general/nfs1/u1592009/TimeWarp/VisCurate-
MODEL=Qwen/Qwen3.5-4B bash phase6_run.sh          # default
MODEL=meta-llama/Llama-3.1-8B-Instruct bash phase6_run.sh
```

`phase6_run.sh` is the single entry point. It starts (or reuses) one vLLM server and fans the
sweep out across parallel curate workers with the real ML gate. **Everything except `MODEL` is
baked in and vetted** — do not re-derive it. Output → `results/phase6_<model_alias>_full/`.
It is **resumable**: re-run the same command and completed instances are skipped.

## The run config (identical every time — do not change unless asked)

| Setting | Value | Why |
|---|---|---|
| Metrics | LPIPS + DINO + CLIP (`--clip`, real ML) | full perceptual+semantic gate; `NO_CLIP=1` drops CLIP |
| Thresholds | `$CONFIG` calibrated yaml (Phase 4) | merge/parameterize gated on *measured* distances |
| `--device` | `cuda` | vision models on GPU |
| `--max-steps` | 200 | agent step cap |
| Cache | `--max-cache-gb 22` (byte-bounded) | predictable RAM; 12 workers fit ~545 GB |
| Workers | `NWORKERS=12` | measured best: 14 inst/h; more workers win (see Tuning) |
| Server | TP=1 on GPU 0 | TP>1 crashes on NCCL init |
| Vision GPUs | `1 2 3` | workers' LPIPS/DINO/CLIP spread off the server GPU |
| Env | `viscurate` conda env (server **and** clients) | has vLLM 0.24 + torch/lpips/timm/open_clip |

## Environment (critical)

- **One env for everything: `viscurate`** (`/scratch/general/nfs1/u1592009/miniconda3/envs/viscurate`).
  It holds vLLM **0.24** *and* the `[ml]` stack. `phase6_run.sh` uses its **absolute** python/vllm
  paths, so it does **not** depend on the shell's active env or `PATH`.
- **Qwen3.5 (`qwen3_5` / `Qwen3_5ForConditionalGeneration`, a GDN/linear-attention arch) needs
  vLLM ≥ 0.23.** Older envs (`tw_r1_q3` = 0.8.5.post1) cannot serve it and reject
  `--gdn-prefill-backend`. If vLLM is missing/old in `viscurate`, install it:
  `viscurate/bin/pip install vllm` (pulls vLLM 0.24 + torch 2.11; the client ML libs still import).
- The model must be present in the HF cache (`/scratch/general/nfs1/u1592009/.cache/huggingface/hub`);
  `TRANSFORMERS_CACHE`/`HF_HOME` is already set in the allocation shell.

## Serving — TP=1, single GPU

`phase6_run.sh` starts the server itself (or reuses a healthy one on `$PORT`):

```
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --port 8001 --host 0.0.0.0 --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len 40960 --enforce-eager
```

- **Keep TP=1.** With ≥2 visible GPUs the old `start_vlm.sh` auto-selected `--tensor-parallel-size 2`,
  which **crashes** at engine init with a NCCL `ncclNetPluginInit`/`ncclNetInit` failure (nccl
  2.28.9/cu13). 4–14B models fit one A800 (40 GB). Only set `SERVER_TP>1` for a model too big for
  one GPU (and then expect to debug NCCL).
- The `deep_gemm ... ImportError: libnvrtc.so.13` line during startup is a **benign warning**
  (optional kernel); the server still reaches `Application startup complete`.

## Parallelism & memory tuning (the real bottleneck)

The vLLM server is **idle** even under many clients (`Running: 0 reqs`) — the work is **client-side**:
each instance runs ~178 agent steps, and every merge/parameterize check renders skills over the 177
probes + LPIPS/DINO/CLIP. So we parallelize *instances*, not the server.

- **More workers win — the ceiling is GPU vision-contention + CPU rendering, NOT cache-thrash.**
  Measured on the 4-GPU/700 GB/60-CPU node: **12 workers × 22 GB = 14 inst/h** vs **8 × 45 GB =
  6 inst/h** (bigger cache, fewer workers, tested *slower* and used *more* RAM). So pack in as many
  workers as RAM allows; don't trade workers for a bigger cache.
- **RAM, not CPUs, is the hard limit.** Each worker ≈ base (~13 GB: torch + 3 vision models) +
  cache (`MAX_CACHE_GB`) + ~12 GB transient render buffers. At 22 GB cache that's ~45 GB/worker →
  **12 workers ≈ 545 GB peak** on a 700 GB node (safe). Watch for transient spikes: 14 workers
  briefly hit 653 GB. Leave ≥120 GB headroom. Rough sizing: `NWORKERS ≈ (RAM_GB − 100) / 47`.
- **Use a BYTE cache bound (`--max-cache-gb`), never a count bound.** OutputSet sizes vary ~10×
  (`metadata_heavy`/`uniform` instances ≈ 3 GB each vs ≈ 0.4 GB for others), so a count bound gives
  wildly unpredictable RAM (caused repeated near-OOM at 24/18/12 workers). Bytes are predictable;
  eviction never changes a result (pure recompute), so the bound is safe.
- To speed it up further you'd cut **GPU vision contention**: batch the LPIPS/DINO/CLIP inference
  (currently `batch=1` in `equivalence/backends.py`) and/or spread workers over more GPUs. Not done
  yet — a code change to the ML backend.

## Monitor & verify

```bash
# progress / rate (score.json per instance)
find results/phase6_<alias>_full -name score.json | wc -l          # of 300
# total worker RSS vs the cgroup mem limit (watch it stay well under):
#   the sampler used during setup sums /proc/<pid>/VmRSS over all "viscurate.cli curate" procs
tail results/phase6_logs/phase6_parallel.log                       # worker progress / FAILs
cat  results/phase6_<alias>_full/summary.json                      # mean P/R/F1 when done
```

Typical rate on the 4-GPU/700 GB/60-CPU A800 node: ~15 min/instance/worker; 8 workers → the full
300 in roughly 8–10 h. Small models score low (F1 ~0.2); larger ones higher (Llama-3.1-8B ~0.6).

## Gotchas (cost real time before)

1. **Fresh `srun`/`salloc` shells start in the PARENT dir** (`…/TimeWarp`), not the repo, and
   reset `PATH`/active env. `phase6_run.sh` `cd`s itself and uses absolute paths, so just invoke it
   by absolute path or after `cd …/VisCurate-`.
2. **Don't wrap the run in an outer `tee`** — a backgrounded server holding the pipe can hang it.
   Server output is already redirected to its own log; redirect the sweep with `>` if you want a log.
3. **`results/` may be renamed** (e.g. `results_before_phase4/`). Point `CONFIG` at wherever the
   calibrated thresholds actually live; output goes to a fresh `results/`.
4. Each instance is a fresh process → RAM resets between instances; only the worst *single* instance
   must fit. Verify a stable peak before walking away.

## Files
- `phase6_run.sh` — turnkey orchestrator (serve + parallel sweep). Change only `$MODEL`.
- `phase6_parallel.sh` — the worker pool (NWORKERS curate clients sharing the server). `NWORKERS=1`
  gives the old sequential behavior.
- `src/viscurate/equivalence/compare.py` — `BatteryEvaluator(max_cache_bytes=…)` byte-bounded cache.
- `src/viscurate/cli.py` — `curate --max-cache-gb` / `--max-cache-entries`.
