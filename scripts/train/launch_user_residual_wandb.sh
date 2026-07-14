#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <gpu-id> <configs/active/run.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3

[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
[[ "$(realpath "$config")" == "$root"/configs/active/*.yaml ]] || {
  echo "formal runs must use configs/active/*.yaml: $config" >&2
  exit 2
}
grep -Eq '^report_to:[[:space:]]*wandb[[:space:]]*$' "$config" || {
  echo "config does not report to W&B: $config" >&2
  exit 2
}
grep -Eq '^num_train_epochs:[[:space:]]*1([.]0)?[[:space:]]*$' "$config" || {
  echo "user residual formal run must be exactly one epoch" >&2
  exit 2
}
grep -Eq '^save_strategy:[[:space:]]*("no"|no)[[:space:]]*$' "$config" || {
  echo "user residual formal run must use save_strategy: no" >&2
  exit 2
}
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"

if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online formal run: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi
export WANDB_MODE=online

"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" "$root/scripts/train/train_user_residual_retkl.py" "$config"
