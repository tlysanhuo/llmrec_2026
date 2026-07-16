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
expected_data=81c7da5f9b6510b7663c5036d39f6269deea2ffe647c78c2d753ad68d8ce8350
expected_holdout=8aa4306f139afc0a00cacd91508de90aa9fa2cbd9942af9cdb665d895721402a
expected_builder=a67978a5c4846ecb03650fe638f05ad979f1df581f09599d172277d275e2a28b
expected_trainer=1cb9458670ddd550727b3cb558575d4a5e38262df7c42eb7544c2e2029caa044
expected_config=ee443f8272f5258bfaa3a8f0115735946a3b80d971fd767882e48a72396a898e
[[ "$(sha256sum "$root/submissions/e3_userres_r80_retkl_v3_s875_platform/adapter_model.safetensors" | awk '{print $1}')" == "$expected_parent" ]] || {
  echo "I-13 parent checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/assets/derived/processed/data_i22_world_retkl_v1.jsonl" | awk '{print $1}')" == "$expected_data" ]] || {
  echo "I-22 data checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl" | awk '{print $1}')" == "$expected_holdout" ]] || {
  echo "I-22 holdout checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/scripts/data/build_i22_world_retkl_v1.py" | awk '{print $1}')" == "$expected_builder" ]] || {
  echo "I-22 builder checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$root/scripts/train/train_i22_world_retkl.py" | awk '{print $1}')" == "$expected_trainer" ]] || {
  echo "I-22 trainer checksum drifted" >&2; exit 2;
}
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$expected_config" ]] || {
  echo "I-22 config checksum drifted" >&2; exit 2;
}
[[ ! -e "$root/checkpoints/i13_s875_world_ansretkl_v1" ]] || {
  echo "refusing to overwrite I-22 output" >&2; exit 2;
}

"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  I22_REFERENCE_ADAPTER="$root/submissions/e3_userres_r80_retkl_v3_s875_platform" \
  I22_WORLD_KL=2.0 \
  I22_RETENTION_KL=50.0 \
  I22_RETENTION_MAX_TOKENS=128 \
  "$python_bin" "$root/scripts/train/train_i22_world_retkl.py" "$config"
