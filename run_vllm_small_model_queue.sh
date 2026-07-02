#!/usr/bin/env bash
# Sequentially benchmark only the new small local vLLM models as VisCurate curation agents.
#
# This is intentionally separate from run_vllm_model_queue.sh so already benchmarked
# models are not queued again.
set -uo pipefail

MAX_STEPS="${MAX_STEPS:-200}"
FORCE="${FORCE:-0}"
NO_ML="${NO_ML:-1}"
PORT="${PORT:-8010}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
QUEUE_ROOT="${QUEUE_ROOT:-results/vllm_model_queue_small}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/general/nfs1/u1592009/miniconda3/envs/tw_r1_q3/bin/python}"
ENV_BIN="$(dirname "$PYTHON_BIN")"
ENV_ROOT="$(dirname "$ENV_BIN")"

cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export PATH="${ENV_BIN}:${PATH}"
export VIRTUAL_ENV="${ENV_ROOT}"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
mkdir -p "$QUEUE_ROOT"

MODEL_IDS=(
  "Qwen/Qwen3.5-2B"
  "meta-llama/Llama-3.2-1B-Instruct"
)

ALIASES=(
  "qwen35_2b"
  "llama32_1b_instruct"
)

EXTRA_ARGS=(
  "--gdn-prefill-backend triton"
  ""
)

TOTAL_INSTANCES="$(
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
root = Path("data/corruption")
print(sum(1 for p in root.iterdir() if (p / "corruption_log.json").exists()))
PY
)"

score_count() {
  local root="$1"
  ROOT="$root" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["ROOT"])
print(len(list(root.glob("*/score.json"))) if root.exists() else 0)
PY
}

clear_stale_server() {
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "Clearing existing vLLM server on port ${PORT} before switching models..."
    pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
    sleep 5
  fi
}

echo "vLLM small-model curation queue"
echo "models=${#MODEL_IDS[@]} max_steps=${MAX_STEPS} force=${FORCE} no_ml=${NO_ML}"
echo "port=${PORT} queue_root=${QUEUE_ROOT} total_instances=${TOTAL_INSTANCES}"

for idx in "${!MODEL_IDS[@]}"; do
  model="${MODEL_IDS[$idx]}"
  alias="${ALIASES[$idx]}"
  extra_args="${EXTRA_ARGS[$idx]}"
  out_root="results/phase6_vllm_${alias}_full"
  log_path="${QUEUE_ROOT}/${alias}.log"
  status_path="${QUEUE_ROOT}/${alias}.status"
  completed="$(score_count "$out_root")"

  if [[ "$FORCE" -ne 1 && "$completed" -ge "$TOTAL_INSTANCES" ]]; then
    {
      echo "model=${model}"
      echo "alias=${alias}"
      echo "out_root=${out_root}"
      echo "score_count=${completed}"
      echo "status=skipped_existing"
      echo "finished_at=$(date -Is)"
    } > "$status_path"
    echo
    echo "skip ${alias}: ${completed}/${TOTAL_INSTANCES} scores already exist"
    continue
  fi

  {
    echo "model=${model}"
    echo "alias=${alias}"
    echo "out_root=${out_root}"
    echo "log_path=${log_path}"
    echo "score_count_at_start=${completed}"
    echo "status=running"
    echo "started_at=$(date -Is)"
  } > "$status_path"

  echo
  echo "================================================================================"
  echo "[$((idx + 1))/${#MODEL_IDS[@]}] ${alias}: ${model}"
  echo "output=${out_root}"
  echo "log=${log_path}"
  echo "starting_scores=${completed}/${TOTAL_INSTANCES}"
  echo "================================================================================"

  clear_stale_server

  MODEL="$model" \
  OUT_ROOT="$out_root" \
  MAX_STEPS="$MAX_STEPS" \
  FORCE="$FORCE" \
  NO_ML="$NO_ML" \
  PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  PYTHON_BIN="$PYTHON_BIN" \
  VLLM_EXTRA_ARGS="$extra_args" \
  bash run_vllm_curation_sweep.sh 2>&1 | tee "$log_path"
  exit_code="${PIPESTATUS[0]}"
  completed="$(score_count "$out_root")"

  {
    echo "model=${model}"
    echo "alias=${alias}"
    echo "out_root=${out_root}"
    echo "log_path=${log_path}"
    echo "score_count=${completed}"
    echo "exit_code=${exit_code}"
    echo "finished_at=$(date -Is)"
    if [[ "$exit_code" -eq 0 ]]; then
      echo "status=completed"
    else
      echo "status=failed"
    fi
  } > "$status_path"

  clear_stale_server

  if [[ "$exit_code" -ne 0 ]]; then
    echo "ERROR: ${alias} failed with exit code ${exit_code}; continuing to next model."
  fi
done

echo
echo "Small-model queue finished. Status files:"
ls -1 "${QUEUE_ROOT}"/*.status 2>/dev/null || true
