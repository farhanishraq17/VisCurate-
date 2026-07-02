#!/usr/bin/env bash
# Sequentially benchmark several local vLLM-served models as VisCurate curation agents.
#
# Each model runs the full corruption sweep through run_vllm_curation_sweep.sh and writes
# per-model artifacts under results/phase6_vllm_<alias>_full/.
set -uo pipefail

MAX_STEPS="${MAX_STEPS:-200}"
FORCE="${FORCE:-1}"
NO_ML="${NO_ML:-1}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
QUEUE_ROOT="${QUEUE_ROOT:-results/vllm_model_queue}"

cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
mkdir -p "$QUEUE_ROOT"

MODEL_IDS=(
  "Qwen/Qwen3.5-27B"
  "google/gemma-4-12B-it"
  "meta-llama/Llama-3.1-8B-Instruct"
)

ALIASES=(
  "qwen35_27b"
  "gemma4_12b"
  "llama31_8b_instruct"
)

EXTRA_ARGS=(
  "--gdn-prefill-backend triton"
  ""
  ""
)

echo "vLLM curation model queue"
echo "models=${#MODEL_IDS[@]} max_steps=${MAX_STEPS} force=${FORCE} no_ml=${NO_ML}"
echo "queue_root=${QUEUE_ROOT}"

for idx in "${!MODEL_IDS[@]}"; do
  model="${MODEL_IDS[$idx]}"
  alias="${ALIASES[$idx]}"
  extra_args="${EXTRA_ARGS[$idx]}"
  out_root="results/phase6_vllm_${alias}_full"
  log_path="${QUEUE_ROOT}/${alias}.log"
  status_path="${QUEUE_ROOT}/${alias}.status"

  {
    echo "model=${model}"
    echo "alias=${alias}"
    echo "out_root=${out_root}"
    echo "status=running"
    echo "started_at=$(date -Is)"
  } > "$status_path"

  echo
  echo "================================================================================"
  echo "[$((idx + 1))/${#MODEL_IDS[@]}] ${alias}: ${model}"
  echo "output=${out_root}"
  echo "log=${log_path}"
  echo "================================================================================"

  MODEL="$model" \
  OUT_ROOT="$out_root" \
  MAX_STEPS="$MAX_STEPS" \
  FORCE="$FORCE" \
  NO_ML="$NO_ML" \
  PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  VLLM_EXTRA_ARGS="$extra_args" \
  bash run_vllm_curation_sweep.sh 2>&1 | tee "$log_path"
  exit_code="${PIPESTATUS[0]}"

  {
    echo "model=${model}"
    echo "alias=${alias}"
    echo "out_root=${out_root}"
    echo "exit_code=${exit_code}"
    echo "finished_at=$(date -Is)"
    if [[ "$exit_code" -eq 0 ]]; then
      echo "status=completed"
    else
      echo "status=failed"
    fi
  } > "$status_path"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "ERROR: ${alias} failed with exit code ${exit_code}; continuing to next model."
  fi
done

echo
echo "Queue finished. Status files:"
ls -1 "${QUEUE_ROOT}"/*.status 2>/dev/null || true
