#!/usr/bin/env bash
# Run a local vLLM model as the Phase-6 curation agent over every generated corruption instance.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
PORT="${PORT:-8001}"
INSTANCES_DIR="${INSTANCES_DIR:-data/corruption}"
PROBES_DIR="${PROBES_DIR:-data/probe_images}"
OUT_ROOT="${OUT_ROOT:-results/phase6_vllm_qwen3_4b_full}"
MAX_STEPS="${MAX_STEPS:-200}"
FORCE="${FORCE:-0}"
NO_ML="${NO_ML:-1}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/general/nfs1/u1592009/miniconda3/envs/tw_r1_q3/bin/python}"

cd "$(dirname "$0")"
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

VISCURATE=("$PYTHON_BIN" -m viscurate.cli)

if [[ ! -f "${PROBES_DIR}/manifest.json" ]]; then
    echo "ERROR: missing probe battery at ${PROBES_DIR}/manifest.json" >&2
    exit 1
fi

if [[ ! -d "${INSTANCES_DIR}" ]]; then
    echo "ERROR: missing corruption instance directory: ${INSTANCES_DIR}" >&2
    exit 1
fi

SERVER_STARTED=0
cleanup() {
    if [[ "$SERVER_STARTED" -eq 1 && "${KEEP_SERVER:-0}" -ne 1 ]]; then
        echo "Stopping vLLM server (port ${PORT})..."
        pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "Reusing existing vLLM server at http://localhost:${PORT}/v1"
else
    echo "Starting vLLM server: ${MODEL} on :${PORT}"
    bash start_vlm.sh --model "$MODEL" --port "$PORT"
    SERVER_STARTED=1
fi

mkdir -p "$OUT_ROOT"

mapfile -t INSTANCES < <(
    INSTANCES_DIR="$INSTANCES_DIR" \
    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["INSTANCES_DIR"])
for path in sorted(p for p in root.iterdir() if (p / "corruption_log.json").exists()):
    print(path)
PY
)

total="${#INSTANCES[@]}"
echo "Curation sweep: ${total} instances -> ${OUT_ROOT}"
echo "model=${MODEL} max_steps=${MAX_STEPS} no_ml=${NO_ML}"

for idx in "${!INSTANCES[@]}"; do
    instance="${INSTANCES[$idx]}"
    name="$(basename "$instance")"
    out="${OUT_ROOT}/${name}"
    if [[ "$FORCE" -ne 1 && -f "${out}/action_log.json" && -f "${out}/episode.json" ]]; then
        echo "[$((idx + 1))/${total}] skip ${name} (already complete)"
    else
        echo "[$((idx + 1))/${total}] curate ${name}"
        args=(
            curate
            --instance "$instance"
            --probes-dir "$PROBES_DIR"
            --out "$out"
            --openai-model "$MODEL"
            --openai-base-url "http://localhost:${PORT}/v1"
            --openai-no-thinking
            --openai-max-tokens 2048
            --max-steps "$MAX_STEPS"
        )
        if [[ "$NO_ML" -eq 1 ]]; then
            args+=(--no-ml)
        else
            args+=(--device cuda)
        fi
        "${VISCURATE[@]}" "${args[@]}"
    fi

    INSTANCE="$instance" OUT_DIR="$out" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from viscurate.corruption.types import IdealAction
from viscurate.curation.actions import ActionResult
from viscurate.studies.metrics import action_cost, intrinsic_curation_score, score_actions

instance = Path(os.environ["INSTANCE"])
out = Path(os.environ["OUT_DIR"])
log = [ActionResult.model_validate(x) for x in json.loads((out / "action_log.json").read_text())]
ideal = [IdealAction.model_validate(x) for x in json.loads((instance / "ideal_actions.json").read_text())]
score = score_actions(log, ideal)
payload = {
    "instance": instance.name,
    "precision": score.precision,
    "recall": score.recall,
    "f1": score.f1,
    "tp": score.tp,
    "fp": score.fp,
    "fn": score.fn,
    "n_ideal": score.n_ideal,
    "n_predicted": score.n_predicted,
    "action_cost": action_cost(log),
    "intrinsic_score": intrinsic_curation_score(log, score),
}
(out / "score.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("  score f1={f1:.3f} precision={precision:.3f} recall={recall:.3f}".format(**payload))
PY
done

OUT_ROOT="$OUT_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for path in sorted(root.glob("*/score.json")):
    rows.append(json.loads(path.read_text()))
summary = {
    "n": len(rows),
    "mean_precision": sum(r["precision"] for r in rows) / len(rows) if rows else 0.0,
    "mean_recall": sum(r["recall"] for r in rows) / len(rows) if rows else 0.0,
    "mean_f1": sum(r["f1"] for r in rows) / len(rows) if rows else 0.0,
    "mean_intrinsic_score": sum(r["intrinsic_score"] for r in rows) / len(rows) if rows else 0.0,
    "mean_action_cost": sum(r["action_cost"] for r in rows) / len(rows) if rows else 0.0,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(root / "scores.jsonl").write_text(
    "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
    encoding="utf-8",
)
print("Summary:", json.dumps(summary, indent=2))
PY
