#!/usr/bin/env bash
# Smoke-test vLLM models without running the full benchmark.
#
# For each model:
#   1. start vLLM
#   2. run one tiny chat-completion request
#   3. run one VisCurate curation action on one corruption instance
#   4. stop vLLM
set -uo pipefail

PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
NO_ML="${NO_ML:-1}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/general/nfs1/u1592009/miniconda3/envs/tw_r1_q3/bin/python}"
INSTANCES_DIR="${INSTANCES_DIR:-data/corruption}"
PROBES_DIR="${PROBES_DIR:-data/probe_images}"
OUT_ROOT="${OUT_ROOT:-results/vllm_smoke_tests}"
LOG_ROOT="${LOG_ROOT:-results/vllm_smoke_tests/logs}"

cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_IDS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3.5-27B"
)

ALIASES=(
  "llama31_8b_instruct"
  "qwen35_27b"
)

EXTRA_ARGS=(
  ""
  "--gdn-prefill-backend triton"
)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

SMOKE_INSTANCE="$(
  INSTANCES_DIR="$INSTANCES_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["INSTANCES_DIR"])
for path in sorted(p for p in root.iterdir() if (p / "corruption_log.json").exists()):
    print(path)
    break
PY
)"

if [[ -z "$SMOKE_INSTANCE" ]]; then
  echo "ERROR: no corruption instances found under ${INSTANCES_DIR}" >&2
  exit 1
fi

stop_server() {
  pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
  sleep 5
}

echo "vLLM smoke tests"
echo "instance=${SMOKE_INSTANCE}"
echo "max_model_len=${MAX_MODEL_LEN} smoke_steps=${SMOKE_STEPS}"

for idx in "${!MODEL_IDS[@]}"; do
  model="${MODEL_IDS[$idx]}"
  alias="${ALIASES[$idx]}"
  extra_args="${EXTRA_ARGS[$idx]}"
  out_dir="${OUT_ROOT}/${alias}"
  status_path="${OUT_ROOT}/${alias}.status"
  server_log="${LOG_ROOT}/${alias}_server.log"
  chat_log="${LOG_ROOT}/${alias}_chat.json"

  echo
  echo "================================================================================"
  echo "[$((idx + 1))/${#MODEL_IDS[@]}] ${alias}: ${model}"
  echo "================================================================================"

  if [[ -n "${SMOKE_ONLY_ALIAS:-}" && "$alias" != "$SMOKE_ONLY_ALIAS" ]]; then
    echo "skip ${alias} (SMOKE_ONLY_ALIAS=${SMOKE_ONLY_ALIAS})"
    continue
  fi

  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  {
    echo "model=${model}"
    echo "alias=${alias}"
    echo "status=starting"
    echo "started_at=$(date -Is)"
  } > "$status_path"

  stop_server

  if ! MODEL="$model" MAX_MODEL_LEN="$MAX_MODEL_LEN" VLLM_EXTRA_ARGS="$extra_args" bash start_vlm.sh --model "$model" --port "$PORT" >"$server_log" 2>&1; then
    {
      echo "model=${model}"
      echo "alias=${alias}"
      echo "status=server_failed"
      echo "finished_at=$(date -Is)"
    } > "$status_path"
    echo "ERROR: server failed for ${alias}; see ${server_log}"
    stop_server
    continue
  fi

  if ! MODEL="$model" PORT="$PORT" CHAT_LOG="$chat_log" "$PYTHON_BIN" - <<'PY'; then
import json
import os
import urllib.request

model = os.environ["MODEL"]
port = os.environ["PORT"]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: VisCurate smoke OK"}],
    "max_tokens": 16,
}
req = urllib.request.Request(
    f"http://localhost:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    body = resp.read().decode("utf-8")
open(os.environ["CHAT_LOG"], "w", encoding="utf-8").write(body)
data = json.loads(body)
print(data["choices"][0]["message"]["content"])
PY
    {
      echo "model=${model}"
      echo "alias=${alias}"
      echo "status=chat_failed"
      echo "finished_at=$(date -Is)"
    } > "$status_path"
    echo "ERROR: chat completion failed for ${alias}; see ${server_log}"
    stop_server
    continue
  fi

  args=(
    curate
    --instance "$SMOKE_INSTANCE"
    --probes-dir "$PROBES_DIR"
    --out "$out_dir"
    --openai-model "$model"
    --openai-base-url "http://localhost:${PORT}/v1"
    --openai-no-thinking
    --openai-max-tokens 512
    --max-steps "$SMOKE_STEPS"
  )
  if [[ "$NO_ML" -eq 1 ]]; then
    args+=(--no-ml)
  else
    args+=(--device cuda)
  fi

  if "$PYTHON_BIN" -m viscurate.cli "${args[@]}"; then
    {
      echo "model=${model}"
      echo "alias=${alias}"
      echo "status=passed"
      echo "out_dir=${out_dir}"
      echo "server_log=${server_log}"
      echo "chat_log=${chat_log}"
      echo "finished_at=$(date -Is)"
    } > "$status_path"
    echo "PASS: ${alias}"
  else
    {
      echo "model=${model}"
      echo "alias=${alias}"
      echo "status=curate_failed"
      echo "out_dir=${out_dir}"
      echo "server_log=${server_log}"
      echo "chat_log=${chat_log}"
      echo "finished_at=$(date -Is)"
    } > "$status_path"
    echo "ERROR: curation smoke failed for ${alias}"
  fi

  stop_server
done

echo
echo "Smoke statuses:"
ls -1 "${OUT_ROOT}"/*.status 2>/dev/null || true
