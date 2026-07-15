#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <gpu-id> <configs/active/run.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
default_bin=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin
python_bin=${LLAMAFACTORY_PYTHON:-$default_bin/python3}
if [[ -n "${LLAMAFACTORY_CLI:-}" ]]; then
  trainer_bin=$LLAMAFACTORY_CLI
elif [[ -n "${LLAMAFACTORY_PYTHON:-}" && -x "$(dirname "$python_bin")/llamafactory-cli" ]]; then
  trainer_bin=$(dirname "$python_bin")/llamafactory-cli
else
  trainer_bin=$default_bin/llamafactory-cli
fi

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ -x "$trainer_bin" ]] || { echo "llamafactory-cli is not executable: $trainer_bin" >&2; exit 2; }

[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
[[ "$(realpath "$config")" == "$root"/configs/active/*.yaml ]] || {
  echo "formal runs must use configs/active/*.yaml: $config" >&2
  exit 2
}
if grep -Eq '^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$config"; then
  echo "refusing a disabled training config: $config" >&2
  exit 2
fi
grep -Eq '^report_to:[[:space:]]*wandb[[:space:]]*$' "$config" || {
  echo "config does not report to W&B: $config" >&2
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

exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$trainer_bin" train "$config"
