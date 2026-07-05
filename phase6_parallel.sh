#!/usr/bin/env bash
# Phase 6 curation, PARALLELIZED: NWORKERS concurrent curate clients sharing ONE vLLM server.
#
# The vLLM server is client-bound-idle in the sequential runner (each instance is ~200
# CPU/GPU-heavy verification steps while the server waits), so we fan out instances across
# several curate processes that all hit the same server. Resumable: skips instances that
# already have action_log.json + episode.json. Each worker pins vision to GPU1 and uses a
# reduced --max-cache-entries so NWORKERS processes fit in host RAM.
#
# Prereq: a healthy vLLM server already running on PORT (this script does NOT start one).
#
# Env: NWORKERS(4) MAX_CACHE(64) MODEL PORT DEVICE CONFIG INSTANCES_DIR PROBES_DIR OUT_ROOT
#      MAX_STEPS(200) VISION_GPU(1) PYTHON_BIN NO_CLIP
set -uo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8001}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-results_before_phase4/phase4_sonnet46_full/calibrated_thresholds.yaml}"
INSTANCES_DIR="${INSTANCES_DIR:-data/corruption}"
PROBES_DIR="${PROBES_DIR:-data/probe_images_full}"
MODEL_ALIAS="$(echo "$MODEL" | tr '/A-Z.' '_a-z_')"
OUT_ROOT="${OUT_ROOT:-results/phase6_${MODEL_ALIAS}_full}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_CACHE="${MAX_CACHE:-256}"
MAX_CACHE_GB="${MAX_CACHE_GB:-0}"        # byte budget per worker (GB); 0 = off. Primary bound for parallel.
NWORKERS="${NWORKERS:-4}"
VISION_GPUS="${VISION_GPUS:-1}"          # space-separated GPU ids; workers round-robin across them
read -ra VGPUS <<< "$VISION_GPUS"
PYTHON_BIN="${PYTHON_BIN:-/scratch/general/nfs1/u1592009/miniconda3/envs/viscurate/bin/python}"
CLIP_FLAG=(--clip); [[ "${NO_CLIP:-0}" -eq 1 ]] && CLIP_FLAG=()
VISCURATE=("$PYTHON_BIN" -m viscurate.cli)

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "ERROR: no healthy vLLM server on :${PORT}. Start one first." >&2; exit 1
fi
mkdir -p "$OUT_ROOT"

mapfile -t ALL < <(INSTANCES_DIR="$INSTANCES_DIR" OUT_ROOT="$OUT_ROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["INSTANCES_DIR"]); out = Path(os.environ["OUT_ROOT"])
for p in sorted(x for x in root.iterdir() if (x / "corruption_log.json").exists()):
    o = out / p.name
    if not ((o / "action_log.json").exists() and (o / "episode.json").exists()):
        print(p)
PY
)
total=${#ALL[@]}
echo "parallel curation: ${total} incomplete instances | workers=${NWORKERS} cache=${MAX_CACHE} vision_gpus='${VISION_GPUS}' -> ${OUT_ROOT}"
[[ "$total" -eq 0 ]] && { echo "nothing to do (all complete)"; exit 0; }

curate_one() {
    local instance="$1" gpu="$2" name out
    name="$(basename "$instance")"; out="${OUT_ROOT}/${name}"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "${VISCURATE[@]}" curate \
        --config "$CONFIG" --instance "$instance" --probes-dir "$PROBES_DIR" \
        --out "$out" --device "$DEVICE" "${CLIP_FLAG[@]}" \
        --openai-model "$MODEL" --openai-base-url "http://localhost:${PORT}/v1" \
        --openai-no-thinking --openai-max-tokens 2048 \
        --max-steps "$MAX_STEPS" --max-cache-entries "$MAX_CACHE" --max-cache-gb "$MAX_CACHE_GB" \
        >"${out}.curate.log" 2>&1 || { echo "[$(date +%T)] FAIL curate ${name}"; return; }
    INSTANCE="$instance" OUT_DIR="$out" "$PYTHON_BIN" - >>"${out}.curate.log" 2>&1 <<'PY'
import json, os
from pathlib import Path
from viscurate.corruption.types import IdealAction
from viscurate.curation.actions import ActionResult
from viscurate.studies.metrics import action_cost, intrinsic_curation_score, score_actions
instance = Path(os.environ["INSTANCE"]); out = Path(os.environ["OUT_DIR"])
log = [ActionResult.model_validate(x) for x in json.loads((out / "action_log.json").read_text())]
ideal = [IdealAction.model_validate(x) for x in json.loads((instance / "ideal_actions.json").read_text())]
score = score_actions(log, ideal)
payload = {"instance": instance.name, "precision": score.precision, "recall": score.recall,
    "f1": score.f1, "tp": score.tp, "fp": score.fp, "fn": score.fn, "n_ideal": score.n_ideal,
    "n_predicted": score.n_predicted, "action_cost": action_cost(log),
    "intrinsic_score": intrinsic_curation_score(log, score)}
(out / "score.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
    echo "[$(date +%T)] done ${name}"
}

worker() {
    local w="$1" i gpu
    gpu="${VGPUS[$((w % ${#VGPUS[@]}))]}"
    for ((i = w; i < total; i += NWORKERS)); do
        echo "[$(date +%T)] worker${w}(gpu${gpu}) -> [$((i + 1))/${total}] $(basename "${ALL[$i]}")"
        curate_one "${ALL[$i]}" "$gpu"
    done
    echo "[$(date +%T)] worker${w} finished its shard"
}

for ((w = 0; w < NWORKERS; w++)); do worker "$w" & done
wait
echo "[$(date +%T)] all workers done"

OUT_ROOT="$OUT_ROOT" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["OUT_ROOT"]); rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/score.json"))]
summary = {"n": len(rows),
    "mean_precision": sum(r["precision"] for r in rows)/len(rows) if rows else 0.0,
    "mean_recall": sum(r["recall"] for r in rows)/len(rows) if rows else 0.0,
    "mean_f1": sum(r["f1"] for r in rows)/len(rows) if rows else 0.0,
    "mean_intrinsic_score": sum(r["intrinsic_score"] for r in rows)/len(rows) if rows else 0.0,
    "mean_action_cost": sum(r["action_cost"] for r in rows)/len(rows) if rows else 0.0}
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(root / "scores.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows), encoding="utf-8")
print("Summary:", json.dumps(summary, indent=2))
PY
