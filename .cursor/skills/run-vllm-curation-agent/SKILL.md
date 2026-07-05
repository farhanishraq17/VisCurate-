---
name: run-vllm-curation-agent
description: Run a local vLLM-served model as the VisCurate Phase 6 curation agent. Use when the user asks to benchmark, run, sweep, or evaluate a vLLM/local model as a curation agent over clean or corrupted VisCurate skill libraries.
---

# Run vLLM Curation Agent

## Purpose

Use this skill for VisCurate workflows where a local vLLM model should act as the **Phase 6 curation agent**. Do not confuse this with the Phase 4 `llm-on-descriptions` judge. The local vLLM agent proposes curation actions; VisCurate verifies/gates those actions with output-grounded checks.

## Before Running

1. Confirm the repo root is `/scratch/general/nfs1/u1592009/TimeWarp/VisCurate-`.
2. Check tmux panes/windows before launching long jobs:
   ```bash
   tmux list-windows -a -F '#{session_name}:#{window_index} #{window_name} #{window_active}'
   tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_path} #{pane_current_command}'
   ```
3. Do not print `.env` or secrets. Only check whether required keys are present.
4. Prefer running long curation sweeps inside tmux, not the agent shell.
5. Use `--no-ml` for smoke or broad sweeps unless the user explicitly asks for LPIPS/DINO/CLIP gates.

## Single-Instance Local vLLM Curation

Use `run.sh` for one corrupted instance. It starts `start_vlm.sh`, runs `viscurate curate`, saves artifacts, and stops vLLM unless `KEEP_SERVER=1`.

Default command:
```bash
cd /scratch/general/nfs1/u1592009/TimeWarp/VisCurate-
./run.sh \
  --instance data/corruption/rho010_uniform_seed1234_single \
  --out results/phase6_vllm_qwen3_4b_rho010 \
  --max-steps 200
```

Useful overrides:
```bash
MODEL=Qwen/Qwen3-4B PORT=8001 KEEP_SERVER=1 ./run.sh ...
```

Outputs:
- `action_log.json`
- `episode.json`

## Full Corruption Sweep

Use `run_vllm_curation_sweep.sh` to run the local vLLM model over all generated corruption instances.

Recommended command:
```bash
cd /scratch/general/nfs1/u1592009/TimeWarp/VisCurate-
FORCE=1 MAX_STEPS=200 MODEL=Qwen/Qwen3-4B bash run_vllm_curation_sweep.sh
```

Default output root:
```text
results/phase6_vllm_qwen3_4b_full/
```

Per-instance outputs:
- `<instance>/action_log.json`
- `<instance>/episode.json`
- `<instance>/score.json`

Final aggregate outputs:
- `summary.json`
- `scores.jsonl`

`score.json` is computed against the instance’s `ideal_actions.json` and includes precision, recall, F1, TP/FP/FN, action cost, and intrinsic score.

## Monitoring Progress

Count completed scores:
```bash
python - <<'PY'
from pathlib import Path
root = Path("results/phase6_vllm_qwen3_4b_full")
print(len(list(root.glob("*/score.json"))))
PY
```

Estimate remaining time:
```bash
/scratch/general/nfs1/u1592009/miniconda3/envs/tw_r1_q3/bin/python - <<'PY'
from pathlib import Path
import time
root = Path("results/phase6_vllm_qwen3_4b_full")
inst_root = Path("data/corruption")
total = sum(1 for p in inst_root.iterdir() if (p / "corruption_log.json").exists())
scores = sorted(root.glob("*/score.json"))
completed = len(scores)
remaining = total - completed
print(f"completed={completed} total={total} remaining={remaining}")
if len(scores) > 1:
    first = min(scores, key=lambda p: p.stat().st_mtime)
    last = max(scores, key=lambda p: p.stat().st_mtime)
    elapsed = max(1e-9, last.stat().st_mtime - first.stat().st_mtime)
    rate = (completed - 1) / elapsed
    print(f"rate_per_min={rate * 60:.2f}")
    print(f"eta_min={remaining / rate / 60:.1f}")
print(time.strftime("%H:%M:%S"))
PY
```

## Interpreting Results

For curation-agent benchmarking, use action metrics unless downstream evaluation has also been run:

- `precision`: fraction of applied predicted repair actions that match the ideal key.
- `recall`: fraction of ideal repair actions found.
- `f1`: action-level repair F1.
- `intrinsic_score`: action F1 penalized for rejected, blocked, or invalid actions.
- `action_cost`: number of non-`end` actions.
- `compression`: `size_before - size_after` from `episode.json`.

Do not call this downstream task success. Downstream success requires a separate `viscurate run-downstream` pass.

## Common Gotchas

- `MAX_STEPS` is per corrupted library instance. The effective cap is `min(MAX_STEPS, curation.budget)`; the default config budget may cap runs before `MAX_STEPS`.
- If `start_vlm.sh` blocks after saying the server is ready, ensure wrapper scripts call it directly rather than piping it through an outer `tee`.
- Local vLLM uses the OpenAI-compatible `/v1` API through `--openai-model` and `--openai-base-url http://localhost:<port>/v1`.
- Hosted GPT/OpenAI curation uses `run_openai_curation_sweep.sh`, not this local vLLM workflow.
