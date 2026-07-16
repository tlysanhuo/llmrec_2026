#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <gpu-id-or-uuid> <configs/active/run.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-$root/LLaMA-Factory/.venv/bin/python3}

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$root"/configs/active/*.yaml ]] || {
  echo "formal runs require configs/active/*.yaml: $config" >&2; exit 2;
}
grep -Eq '^report_to:[[:space:]]*wandb[[:space:]]*$' "$config" || {
  echo "config does not report to W&B" >&2; exit 2;
}
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
[[ "${WANDB_MODE:-online}" == online ]] || { echo "W&B must be online" >&2; exit 2; }
export WANDB_MODE=online

expected_parent=71bc3c2c86beb1c1aaafd41f98915ba94a7f964b6e8450079a883aebc32ffd5b
expected_data=bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0
expected_trainer=e8890cca4d965885b9468b118009e6b8e578c22f0281a78fbb8c855177bd26cf
[[ "$(sha256sum "$root/submissions/e3_userres_r80_retkl_v3_s875_platform/adapter_model.safetensors" | awk '{print $1}')" == "$expected_parent" ]] || {
  echo "I-13 parent checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/assets/derived/processed/data_user_residual_retention_v1.jsonl" | awk '{print $1}')" == "$expected_data" ]] || {
  echo "I-21 data checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/scripts/train/train_i21_topic_retkl.py" | awk '{print $1}')" == "$expected_trainer" ]] || {
  echo "I-21 trainer checksum drifted" >&2; exit 2;
}
[[ ! -e "$root/checkpoints/i13_s875_topic_ansretkl_v1" ]] || {
  echo "refusing to overwrite I-21 output" >&2; exit 2;
}

"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  I21_REFERENCE_ADAPTER="$root/submissions/e3_userres_r80_retkl_v3_s875_platform" \
  I21_TOPIC_KL=2.0 \
  I21_RETENTION_KL=50.0 \
  I21_RETENTION_MAX_TOKENS=128 \
  "$python_bin" "$root/scripts/train/train_i21_topic_retkl.py" "$config"
