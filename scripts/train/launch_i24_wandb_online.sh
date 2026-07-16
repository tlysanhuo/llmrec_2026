#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <single-gpu-id-or-uuid> <configs/active/i23_action_ansretkl_v1.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
trainer=$root/scripts/train/train_i23_action_retkl.py
expected_config=$root/configs/active/i23_action_ansretkl_v1.yaml
parent=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
data=$root/assets/derived/processed/data_user_residual_retention_v1.jsonl
output=$root/checkpoints/i23_action_ansretkl_v1

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ "$gpu_id" != *,* ]] || { echo "I-24 formal training must expose exactly one GPU" >&2; exit 2; }
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$expected_config" ]] || {
  echo "I-24 launcher only accepts $expected_config" >&2
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

expected_parent=0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8
expected_parent_config=b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7
expected_data=bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0
expected_trainer=bfeea1dceb3f247c6853359abe0d71f2757d65e5882cfeade0e1fcc2ce5270a7
expected_config_hash=0903f2358fe28d849cacede466c92887329960958735f3deed9ec4b2474938de

[[ "$(sha256sum "$parent/adapter_model.safetensors" | awk '{print $1}')" == "$expected_parent" ]] || {
  echo "I-23 parent adapter checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$parent/adapter_config.json" | awk '{print $1}')" == "$expected_parent_config" ]] || {
  echo "I-23 parent config checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$data" | awk '{print $1}')" == "$expected_data" ]] || {
  echo "I-24 training data checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$trainer" | awk '{print $1}')" == "$expected_trainer" ]] || {
  echo "I-24 trainer checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$expected_config_hash" ]] || {
  echo "I-24 config checksum drifted" >&2; exit 2;
}
[[ ! -e "$output" ]] || {
  echo "refusing to overwrite I-24 output directory: $output" >&2; exit 2;
}

"$python_bin" "$trainer" --self-test
"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  I24_REFERENCE_ADAPTER="$parent" \
  I24_ACTION_KL=5.0 \
  I24_RETENTION_KL=50.0 \
  I24_TRACE_KL_TOKENS=96 \
  I24_BODY_KL_TOKENS=48 \
  I24_TAIL_KL_TOKENS=16 \
  I24_MAX_KL_TOKENS=160 \
  I24_TERMINAL_MULTIPLIER=2.0 \
  I24_LOGIT_CHUNK=8 \
  "$python_bin" "$trainer" "$config"
