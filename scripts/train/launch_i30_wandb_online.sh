#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=llmrec-2026 $0 <single-gpu-index> <i30-config>" >&2
  exit 2
fi

gpu_id=$1
config=$(realpath "$2")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
expected_config=$root/configs/active/i30_r96_material_teacher_retkl_r8_v1.yaml
trainer=$root/scripts/train/train_i30_r96_material_teacher_retkl.py
builder=$root/scripts/data/build_i30_r96_material_teacher_v1.py
data=$root/assets/derived/processed/data_i30_r96_material_teacher_retkl_v1.jsonl
holdout=$root/assets/evaluation/holdout/data_i30_r96_material_teacher_gate_v1.jsonl
ledger=$root/logs/data/i30_r96_material_teacher_selection_v1.jsonl
audit=$root/logs/data/i30_r96_material_teacher_retkl_v1_audit.json
dataset_registry=$root/configs/datasets/i30_r96_material_teacher_retkl_v1/dataset_info.json
gate=$root/configs/evaluation/i30_r96_material_teacher_checkpoint_gate.json
parent=$root/submissions/i19_world_external_r96_s875_platform
teacher=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
output=$root/checkpoints/i30_r96_material_teacher_retkl_r8_v1

[[ "$config" == "$expected_config" ]] || { echo "I-30 accepts only $expected_config" >&2; exit 2; }
[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || { echo "I-30 requires exactly one GPU" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "I-30 output already exists; never resume or overwrite: $output" >&2; exit 2; }
grep -Fq 'PREREGISTERED_AND_HASH_FROZEN_READY_FOR_FORMAL_TRAINING' "$gate" || {
  echo "I-30 gate is not launch-authorized" >&2
  exit 2
}
if grep -Eq 'PENDING_|^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$gate" "$config"; then
  echo "I-30 still contains a pending or disabled marker" >&2
  exit 2
fi
grep -Fq 'data_i30_r96_material_teacher_retkl_v1.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-30 data is not in the authoritative ASSETS registry" >&2
  exit 2
}
grep -Fq 'i30_r96_material_teacher_retkl_r8_v1' "$root/docs/EXPERIMENT_INDEX.md" || {
  echo "I-30 is not preregistered in EXPERIMENT_INDEX" >&2
  exit 2
}

check_sha256() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" ]] || { echo "missing $label: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label checksum drifted: $actual/$expected" >&2
    exit 2
  }
}

check_sha256 "$base_config" 5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4 "O6 config"
check_sha256 "$parent/adapter_model.safetensors" 4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e "r96 parent"
check_sha256 "$parent/adapter_config.json" 78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f "r96 parent config"
check_sha256 "$teacher/adapter_model.safetensors" 0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8 "I-23 teacher"
check_sha256 "$teacher/adapter_config.json" b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7 "I-23 teacher config"
check_sha256 "$data" 0df9a192976eb61eb8dd333fd59edb994d1fcad482710e1282f36dd792bfc4a4 "I-30 data"
check_sha256 "$holdout" dd744ee2d2f584b9bcae938cde1f5976801a9eece39aa1972b284641603f97a0 "I-30 holdout"
check_sha256 "$ledger" b303a501dddb1f7ae3afef192298f349dd211cbda229eb1618a94634c82b5b3d "I-30 selection ledger"
check_sha256 "$audit" 6b46775be12f96ff8e3258266129633cf372dfb7cda28080d8581279a5d49226 "I-30 data audit"
check_sha256 "$dataset_registry" fd3b55c5d4d5c05422b9cfa01514c6b6bcbbdcb325d4e88ccf40357218024872 "I-30 dataset registry"
check_sha256 "$builder" a57c17e28900d4d5f14acdc2130e61f6f8e0c0e1282102bd765f1ceb7ca6ab8a "I-30 builder"
check_sha256 "$trainer" 60fa31d5aa18d7bdb1d039f590a74c7edd4547def0b8f10734fdb54591db9f39 "I-30 trainer"
check_sha256 "$config" e351827cca5f9cf2b4632a960006dbcc27af02280e167004b51445ca16d99db4 "I-30 config"
check_sha256 "$gate" 895a9a6292f0ec4fdbc06865a8809be9c1845af04726530a18d2da606ae5eadc "I-30 gate"

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
[[ "$WANDB_ENTITY" == "thaongocnguyendo0-" ]] || { echo "unexpected W&B entity" >&2; exit 2; }
[[ "$WANDB_PROJECT" == "llmrec-2026" ]] || { echo "unexpected W&B project" >&2; exit 2; }
[[ "${WANDB_MODE:-online}" == online ]] || { echo "I-30 refuses non-online W&B" >&2; exit 2; }
export WANDB_MODE=online

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config"
