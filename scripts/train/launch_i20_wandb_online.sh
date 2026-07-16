#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <gpu-id-or-uuid> <configs/active/run.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$root"/configs/active/*.yaml ]] || {
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

expected_parent=71bc3c2c86beb1c1aaafd41f98915ba94a7f964b6e8450079a883aebc32ffd5b
expected_data=0c08b8f505fe55acbc0ec5a25a8efe90b25fccf1cec5188ea7cb7cf0461162fa
expected_trainer=ed11aed3c71b529886d9632d5c8edc4bf5b51901113f61589ff11d40ff6bb642
[[ "$(sha256sum "$root/submissions/e3_userres_r80_retkl_v3_s875_platform/adapter_model.safetensors" | awk '{print $1}')" == "$expected_parent" ]] || {
  echo "I-13 parent adapter checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/assets/derived/processed/data_i20_prod_ad_positive_retkl_v1.jsonl" | awk '{print $1}')" == "$expected_data" ]] || {
  echo "I-20 training data checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/scripts/train/train_i20_positive_retkl.py" | awk '{print $1}')" == "$expected_trainer" ]] || {
  echo "I-20 trainer checksum drifted" >&2; exit 2;
}
[[ ! -e "$root/checkpoints/i13_s875_posrec_pa_ansretkl_v1" ]] || {
  echo "refusing to overwrite I-20 output directory" >&2; exit 2;
}

"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  I20_REFERENCE_ADAPTER="$root/submissions/e3_userres_r80_retkl_v3_s875_platform" \
  I20_POSITIVE_KL=0.20 \
  I20_RETENTION_KL=4.0 \
  I20_RETENTION_MAX_TOKENS=128 \
  I20_LOGIT_CHUNK=8 \
  "$python_bin" "$root/scripts/train/train_i20_positive_retkl.py" "$config"
