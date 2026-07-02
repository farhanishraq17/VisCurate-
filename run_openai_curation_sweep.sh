#!/usr/bin/env bash
# Run a hosted OpenAI-compatible model as the Phase-6 curation agent over every corruption instance.
set -euo pipefail

MODEL="${MODEL:-gpt-5.5}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
INSTANCES_DIR="${INSTANCES_DIR:-data/corruption}"
PROBES_DIR="${PROBES_DIR:-data/probe_images}"
OUT_ROOT="${OUT_ROOT:-results/phase6_openai_gpt55_full}"
MAX_STEPS="${MAX_STEPS:-200}"
FORCE="${FORCE:-0}"
NO_ML="${NO_ML:-1}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/general/nfs1/u1592009/miniconda3/envs/tw_r1_q3/bin/python}"

cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Add it to .env as OPENAI_API_KEY=..." >&2
    exit 1
fi

if command -v viscurate &>/dev/null; then
    VISCURATE=(viscurate)
else
    VISCURATE=("$PYTHON_BIN" -m viscurate.cli)
fi

if [[ ! -f "${PROBES_DIR}/manifest.json" ]]; then
    echo "ERROR: missing probe battery at ${PROBES_DIR}/manifest.json" >&2
    exit 1
fi

if [[ ! -d "${INSTANCES_DIR}" ]]; then
    echo "ERROR: missing corruption instance directory: ${INSTANCES_DIR}" >&2
    exit 1
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
echo "OpenAI curation sweep: ${total} instances -> ${OUT_ROOT}"
echo "model=${MODEL} base_url=${BASE_URL} max_steps=${MAX_STEPS} no_ml=${NO_ML}"

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
            --openai-base-url "$BASE_URL"
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
rows = [json.loads(path.read_text()) for path in sorted(root.glob("*/score.json"))]
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
