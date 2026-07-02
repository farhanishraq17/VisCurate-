#!/usr/bin/env bash
# Phase 4 — the equivalence benchmark done for real.
#
# This is the project go/no-go. Unlike the Phase-6 curation sweeps (which ran with NO_ML=1),
# this runs the REAL visual stack — LPIPS (perceptual) + DINO (semantic) + CLIP (2nd semantic
# view) — and CALIBRATES the thresholds. It compares the output-grounded verifier against three
# text baselines (name-match, embedding-cosine, and an LLM-on-descriptions judge) and emits the
# divergence report. The calibrated thresholds it produces are what a proper Phase 6 run needs.
#
# The LLM-on-descriptions baseline here is Claude (default: Sonnet 4.6) via the Claude API — a
# fixed, strong text judge. Do NOT point this at the local vLLM curation models: those are the
# SUBJECTS of the Phase-6 agent experiment, and reusing them as the baseline confounds the two
# tracks (--llm-anthropic keeps them separate).
#
# Prerequisites:
#   * The [ml] + agent stack (torch / lpips / timm / open_clip / anthropic) — the `viscurate` env.
#     NOTE: that env has CPU-only torch, so this runs on CPU. To use a GPU you need a CUDA-torch
#     env that also has the full visual stack (none of the current envs do — see PYTHON_BIN).
#   * ANTHROPIC_API_KEY in .env at the repo root (run-benchmark loads .env automatically).
#   * A probe battery + G0 ground truth (data/probe_images*, configs/ground_truth_g0.yaml).
#
# Env overrides (all optional):
#   DEVICE=cpu|cuda    torch device (default cpu; cuda only works if PYTHON_BIN points at a CUDA-torch env)
#   PROBES_DIR=...      probe battery dir (default data/probe_images = 43-probe smoke; full = data/probe_images_full)
#   JUDGE_MODEL=...     Claude model id for the LLM judge (default claude-sonnet-4-6)
#   OUT=...             output dir (default results/phase4_sonnet46)
#   DATE=YYYY-MM-DD     calibration date stamp (default: today)
#   NO_CLIP=1           drop the optional CLIP semantic view (keeps LPIPS + DINO)
#   JUDGE_THINKING=1    let the Claude judge use adaptive thinking (default off: fast, deterministic,
#                       one-word answers with no truncation risk — bump --llm-max-tokens if you enable it)
#   PYTHON_BIN=...      python interpreter (default: the viscurate conda env)
#
# Usage:
#   ./phase4_run.sh                                     # 43-probe smoke battery, CPU, Sonnet 4.6 judge
#   PROBES_DIR=data/probe_images_full ./phase4_run.sh   # full 177-probe battery (the real go/no-go), CPU
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
PROBES_DIR="${PROBES_DIR:-data/probe_images}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-6}"
OUT="${OUT:-results/phase4_sonnet46}"
DATE="${DATE:-$(date +%F)}"
MAX_CACHE="${MAX_CACHE:-256}"   # LRU bound on the output cache (0 = unbounded)
PYTHON_BIN="${PYTHON_BIN:-/scratch/general/nfs1/u1592009/miniconda3/envs/viscurate/bin/python}"

# Run from the repo root so configs/, data/, and .env resolve regardless of CWD.
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "${PROBES_DIR}/manifest.json" ]]; then
    echo "ERROR: missing probe battery at ${PROBES_DIR}/manifest.json" >&2
    echo "       build one first: viscurate build-probes ..." >&2
    exit 1
fi
if [[ ! -f .env ]]; then
    echo "WARNING: no .env at repo root — the Claude judge needs ANTHROPIC_API_KEY." >&2
    echo "         run-benchmark loads .env automatically; add ANTHROPIC_API_KEY=... to it." >&2
fi

# Optional CLIP second semantic view (recommended: on).
CLIP_FLAG=(--clip)
[[ "${NO_CLIP:-0}" -eq 1 ]] && CLIP_FLAG=()

# Judge thinking: off by default so the one-word relation answer can't be truncated by thinking
# tokens (an empty reply is parsed conservatively as DISTINCT). Enable with JUDGE_THINKING=1.
THINK_FLAG=(--llm-no-thinking)
[[ "${JUDGE_THINKING:-0}" -eq 1 ]] && THINK_FLAG=()

echo "Phase 4 benchmark:"
echo "  device=${DEVICE}  probes=${PROBES_DIR}  judge=${JUDGE_MODEL}  out=${OUT}  date=${DATE}"
echo "  clip=$([[ ${#CLIP_FLAG[@]} -gt 0 ]] && echo on || echo off)  judge_thinking=$([[ ${#THINK_FLAG[@]} -gt 0 ]] && echo off || echo on)"

set -x
"$PYTHON_BIN" -m viscurate.cli run-benchmark \
    --probes-dir "$PROBES_DIR" \
    --device "$DEVICE" \
    "${CLIP_FLAG[@]}" \
    --calibrate \
    --date "$DATE" \
    --llm-anthropic \
    --llm-anthropic-model "$JUDGE_MODEL" \
    "${THINK_FLAG[@]}" \
    -o "$OUT"
