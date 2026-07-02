#!/bin/bash
# run.sh — smoke test: serve a local vLLM model with start_vlm.sh, then run it as the
# Phase-6 curation agent over a corrupted skill-library instance.
#
# Pipeline:
#   1. build a small synthetic probe battery (offline) if one is not already present
#   2. start the vLLM server via ./start_vlm.sh and wait for /health
#   3. run `viscurate curate` with --openai-model pointed at the served model
#   4. stop the server (unless KEEP_SERVER=1)
#
# PREREQUISITE: a GPU must already be allocated in THIS shell — start_vlm.sh queries nvidia-smi.
# Allocate one in your tmux pane first, e.g.:  salloc --gres=gpu:1 --time=02:00:00 ...
#
# Usage:
#   ./run.sh                                  # Qwen/Qwen3-4B as curation agent on :8001
#   ./run.sh --model Qwen/Qwen3-4B --port 8002
#   ./run.sh --instance data/corruption/rho010_uniform_seed1234_single
#   ./run.sh --no-build                       # reuse an existing probe battery
#   KEEP_SERVER=1 ./run.sh                    # leave the vLLM server running afterwards
set -euo pipefail

# --- defaults -------------------------------------------------------------------------------
MODEL="Qwen/Qwen3-4B"
PORT=8001
PROBES_CONFIG="configs/probes_smoke.yaml"
PROBES_DIR="data/probe_images"
INSTANCE_DIR="data/corruption/rho010_uniform_seed1234_single"
OUT_DIR="results/phase6_vllm_curation"
BUILD_PROBES=1
SERVER_LOG="run_server.log"
MAX_STEPS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --probes-config) PROBES_CONFIG="$2"; shift 2;;
        --probes-dir) PROBES_DIR="$2"; shift 2;;
        --instance) INSTANCE_DIR="$2"; shift 2;;
        --out) OUT_DIR="$2"; shift 2;;
        --max-steps) MAX_STEPS="$2"; shift 2;;
        --no-build) BUILD_PROBES=0; shift;;
        -h|--help) grep '^#' "$0" | sed 's/^#\( \|$\)//'; exit 0;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 1;;
    esac
done

# Run from the repo root so configs/, data/, and start_vlm.sh resolve regardless of CWD.
cd "$(dirname "$0")"
BASE_URL="http://localhost:${PORT}/v1"

# Prefer the installed console script; fall back to running from source (no pip install needed).
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
if command -v viscurate &>/dev/null; then
    VISCURATE=(viscurate)
elif command -v python &>/dev/null; then
    VISCURATE=(python -m viscurate.cli)
else
    VISCURATE=(python3 -m viscurate.cli)
fi

# --- preflight: GPU --------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi &>/dev/null; then
    echo "ERROR: no GPU visible (nvidia-smi failed). Allocate a GPU in this tmux pane first," >&2
    echo "       e.g.  salloc --gres=gpu:1 --time=02:00:00 ...   then re-run ./run.sh" >&2
    exit 1
fi

# --- step 1: probe battery -------------------------------------------------------------------
if [[ -f "${PROBES_DIR}/manifest.json" ]]; then
    echo "[1/3] Reusing probe battery at ${PROBES_DIR}"
elif [[ "$BUILD_PROBES" -eq 1 ]]; then
    echo "[1/3] Building synthetic probe battery -> ${PROBES_DIR}"
    "${VISCURATE[@]}" build-probes -c "$PROBES_CONFIG" -o "$PROBES_DIR"
else
    echo "ERROR: no probe battery at ${PROBES_DIR}/manifest.json and --no-build was set." >&2
    exit 1
fi

if [[ -n "$INSTANCE_DIR" && ! -f "${INSTANCE_DIR}/corruption_log.json" ]]; then
    echo "ERROR: no corruption_log.json under ${INSTANCE_DIR}." >&2
    echo "       Run 'viscurate corrupt -c configs/corruption.yaml -o data/corruption' first," >&2
    echo "       or pass --instance '' to curate the clean base library." >&2
    exit 1
fi

# --- step 2: serve the model -----------------------------------------------------------------
SERVER_STARTED=0
cleanup() {
    if [[ "$SERVER_STARTED" -eq 1 && "${KEEP_SERVER:-0}" -ne 1 ]]; then
        echo "Stopping vLLM server (port ${PORT})..."
        pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[2/3] Starting vLLM server: ${MODEL} on :${PORT} (log: ${SERVER_LOG})"
# start_vlm.sh already tees the server process to vllm_server.log. Call it directly so its
# background vLLM process does not keep an outer tee open and block the curation step.
bash start_vlm.sh --model "$MODEL" --port "$PORT"
SERVER_STARTED=1

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "ERROR: server reported ready but http://localhost:${PORT}/health is not responding." >&2
    exit 1
fi

# --- step 3: curation ------------------------------------------------------------------------
echo "[3/3] Running curation agent — ${MODEL} @ ${BASE_URL}"
CURATE_ARGS=(
    curate
    --probes-dir "$PROBES_DIR"
    --out "$OUT_DIR"
    --no-ml
    --openai-model "$MODEL"
    --openai-base-url "$BASE_URL"
    --openai-no-thinking
    --max-steps "$MAX_STEPS"
)
if [[ -n "$INSTANCE_DIR" ]]; then
    CURATE_ARGS+=(--instance "$INSTANCE_DIR")
fi
"${VISCURATE[@]}" "${CURATE_ARGS[@]}"

echo
echo "Done. Curation artifacts -> ${OUT_DIR}"
echo "(Smoke run uses --no-ml. For perceptual/semantic verifier gates, remove --no-ml and add"
echo " --device cuda in step 3.)"
