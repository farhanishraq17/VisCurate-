#!/usr/bin/env bash
# Phase 6 — run the curation agent with the REAL visual stack (LPIPS + DINO + CLIP) gating,
# over the whole corruption sweep, served by a local vLLM. TURNKEY: to run a different model
# you change ONLY $MODEL; every other decision below is baked in and vetted.
#
#   MODEL=Qwen/Qwen3.5-4B                bash phase6_run.sh   # default
#   MODEL=meta-llama/Llama-3.1-8B-Instruct bash phase6_run.sh
#
# What this does (see .cursor/skills/run-curation-ml-sweep/SKILL.md for the full rationale):
#   1. Uses the `viscurate` conda env for BOTH the vLLM server and the curate clients — it has
#      vLLM 0.24 (needed for Qwen3.5 / qwen3_5 GDN arch) AND the [ml] stack (torch/lpips/
#      timm/open_clip). Absolute paths => no dependence on the shell's PATH/active env.
#   2. Starts ONE single-GPU (TP=1) vLLM server on SERVER_GPU, or reuses a healthy one.
#      TP=1 is deliberate: TP=2 crashes on NCCL init (ncclNetPluginInit) with the CUDA-13 stack,
#      and 4-14B models fit one A800. Set SERVER_TP>1 only for a model too big for one GPU.
#   3. Fans the sweep out over NWORKERS curate clients (phase6_parallel.sh), all sharing that
#      one server, each with a BYTE-BOUNDED verifier cache (--max-cache-gb) so per-worker RAM is
#      predictable (a count bound is not: OutputSet sizes vary ~10x across skills). Vision runs
#      spread over VISION_GPUS. Resumable: completed instances are skipped.
#
# Real ML is always on here: LPIPS (perceptual) + DINO (semantic) + CLIP (2nd semantic view),
# gated by the CALIBRATED thresholds from Phase 4.
#
# Env overrides (all optional; MODEL is the one you normally change):
#   MODEL         vLLM-served model id (default Qwen/Qwen3.5-4B)
#   CONFIG        calibrated thresholds yaml (default results_before_phase4/phase4_sonnet46_full/…)
#   NWORKERS      parallel curate workers (default 12) — RAM-bound (~45 GB each at 22 GB cache).
#                 More workers beat fewer: the sweep is bound by GPU vision-contention + CPU
#                 rendering, not cache-thrash, so add workers until RAM (not cores) runs out.
#   MAX_CACHE_GB  per-worker verifier cache byte budget, GB (default 22) — sized so 12 workers
#                 fit in ~545 GB. Bigger cache + fewer workers tested SLOWER; don't do it.
#   SERVER_GPU    GPU index for the vLLM server (default 0)
#   VISION_GPUS   GPUs for workers' LPIPS/DINO/CLIP, space-separated (default "1 2 3")
#   SERVER_TP     tensor-parallel size for the server (default 1; keep 1 unless model > 1 GPU)
#   NO_CLIP=1     drop the CLIP view (keeps LPIPS + DINO)
#   PORT DEVICE MAX_STEPS PROBES_DIR INSTANCES_DIR OUT_ROOT PYTHON_BIN KEEP_SERVER GPU_MEM_UTIL MAX_MODEL_LEN
set -uo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

VISCURATE_ENV="/scratch/general/nfs1/u1592009/miniconda3/envs/viscurate"
PYTHON_BIN="${PYTHON_BIN:-${VISCURATE_ENV}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VISCURATE_ENV}/bin/vllm}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8001}"
SERVER_GPU="${SERVER_GPU:-0}"
SERVER_TP="${SERVER_TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
CONFIG="${CONFIG:-results_before_phase4/phase4_sonnet46_full/calibrated_thresholds.yaml}"
NWORKERS="${NWORKERS:-12}"             # vetted on 700 GB/60 CPU/4 GPU: 14 inst/h, ~545 GB peak.
MAX_CACHE_GB="${MAX_CACHE_GB:-22}"     # more-workers-smaller-cache beat fewer-bigger (vision-contention bound, not thrash).
VISION_GPUS="${VISION_GPUS:-1 2 3}"
KEEP_SERVER="${KEEP_SERVER:-0}"        # 0 = stop the server we started when the sweep finishes
MODEL_ALIAS="$(echo "$MODEL" | tr '/A-Z.' '_a-z_')"

mkdir -p results/phase6_logs
SERVER_LOG="results/phase6_logs/vllm_${MODEL_ALIAS}.log"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: missing calibrated thresholds at ${CONFIG}" >&2
    echo "       point CONFIG at your Phase-4 calibrated_thresholds.yaml." >&2
    exit 1
fi

SERVER_STARTED=0
cleanup() {
    if [[ "$SERVER_STARTED" -eq 1 && "${KEEP_SERVER}" -ne 1 ]]; then
        echo "Stopping vLLM server (port ${PORT})…"
        pkill -f "vllm serve ${MODEL} .*--port ${PORT}" 2>/dev/null || \
            pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- 1. serve (reuse a healthy server, else start TP=SERVER_TP on SERVER_GPU) ------------------
if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "Reusing existing vLLM server at http://localhost:${PORT}/v1"
else
    echo "Starting vLLM server: ${MODEL} (TP=${SERVER_TP}) on GPU ${SERVER_GPU}, port ${PORT}"
    tp_flag=(); [[ "$SERVER_TP" -gt 1 ]] && tp_flag=(--tensor-parallel-size "$SERVER_TP")
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" nohup "$VLLM_BIN" serve "$MODEL" \
        --port "$PORT" --host 0.0.0.0 --trust-remote-code \
        --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager "${tp_flag[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_STARTED=1
    echo "  waiting for server (log: ${SERVER_LOG})…"
    for _ in $(seq 1 150); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
        sleep 5
    done
    if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "ERROR: vLLM server did not become healthy. Last log lines:" >&2
        tail -25 "$SERVER_LOG" >&2
        exit 1
    fi
    echo "  server ready."
fi

# --- 2. run the parallel curation sweep against that server -----------------------------------
MODEL="$MODEL" PORT="$PORT" CONFIG="$CONFIG" DEVICE="${DEVICE:-cuda}" \
INSTANCES_DIR="${INSTANCES_DIR:-data/corruption}" PROBES_DIR="${PROBES_DIR:-data/probe_images_full}" \
OUT_ROOT="${OUT_ROOT:-results/phase6_${MODEL_ALIAS}_full}" MAX_STEPS="${MAX_STEPS:-200}" \
NWORKERS="$NWORKERS" MAX_CACHE_GB="$MAX_CACHE_GB" VISION_GPUS="$VISION_GPUS" \
NO_CLIP="${NO_CLIP:-0}" PYTHON_BIN="$PYTHON_BIN" \
    bash phase6_parallel.sh
