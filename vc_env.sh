#!/bin/bash
# VisCurate WACV-R2 run environment. conda activate is broken post-purge; use the env bin directly.
# Resolve the repo root from this file's own location so the script survives the tree being moved.
export VC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VC_PY=/scratch/general/nfs1/u1592009/miniconda3/envs/viscurate/bin/python
export PATH=/scratch/general/nfs1/u1592009/miniconda3/envs/viscurate/bin:$PATH
export VISCURATE_TELEMETRY="$VC_ROOT/results/wacv_r2/telemetry.jsonl"
export HF_HOME=/scratch/general/nfs1/u1592009/hf_cache
export TOKENIZERS_PARALLELISM=false
# Credentials. VisCurate-/.env's OPENAI_API_KEY and ANTHROPIC_API_KEY are both expired (401 as of
# 2026-08-19), so TimeWarp/.env's working OPENAI_API_KEY is sourced second and wins. No Anthropic
# credential exists anywhere on this machine: unset it so the Claude path fails loudly at startup
# rather than 401-ing once per pair, mid-run.
set -a
[ -f "$VC_ROOT/.env" ] && . "$VC_ROOT/.env"
[ -f /scratch/general/nfs1/u1592009/TimeWarp/.env ] && . /scratch/general/nfs1/u1592009/TimeWarp/.env
set +a
unset ANTHROPIC_API_KEY
cd "$VC_ROOT"
