#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=llmrec-2026 $0 <single-gpu-index> <i33-config>" >&2
  exit 2
fi

gpu_id=$1
config=$(realpath "$2")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
expected_config=$root/configs/active/i33_r96_material_desc2sid_retkl_r8_v1.yaml
trainer=$root/scripts/train/train_i33_r96_material_desc2sid_retkl.py
builder=$root/scripts/data/build_i33_r96_material_desc2sid_v1.py
data=$root/assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl
control=$root/assets/derived/processed/data_i33_r96_material_bidirectional_e_clean_control_v1.jsonl
holdout=$root/assets/evaluation/holdout/data_i33_r96_material_desc2sid_gate_v1.jsonl
world_holdout=$root/assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl
ledger=$root/logs/data/i33_r96_material_desc2sid_selection_v1.jsonl
control_ledger=$root/logs/data/i33_r96_material_bidirectional_control_selection_v1.jsonl
replacement_ledger=$root/logs/data/i33_r96_e_clean_retention_replacements_v1.jsonl
audit=$root/logs/data/i33_r96_material_desc2sid_retkl_v1_audit.json
dataset_registry=$root/configs/datasets/i33_r96_material_desc2sid_retkl_v1/dataset_info.json
gate=$root/configs/evaluation/i33_r96_material_desc2sid_checkpoint_gate.json
evaluator=$root/scripts/eval/audit_i33_material_desc2sid_gate.py
combiner=$root/scripts/train/combine_lora_adapters.py
parent=$root/submissions/i19_world_external_r96_s875_platform
teacher=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
output=$root/checkpoints/i33_r96_material_desc2sid_retkl_r8_v1

[[ "$config" == "$expected_config" ]] || { echo "I-33 accepts only $expected_config" >&2; exit 2; }
[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || { echo "I-33 requires exactly one GPU" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "I-33 output already exists; never resume or overwrite: $output" >&2; exit 2; }
grep -Fq 'PREREGISTERED_AND_HASH_FROZEN_READY_FOR_FORMAL_TRAINING' "$gate" || {
  echo "I-33 gate is not launch-authorized" >&2
  exit 2
}
if grep -Eq 'PENDING_|^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$gate" "$config"; then
  echo "I-33 still contains a pending or disabled marker" >&2
  exit 2
fi
grep -Fq 'data_i33_r96_material_desc2sid_retkl_v1.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-33 data is not in the authoritative ASSETS registry" >&2
  exit 2
}
grep -Fq 'i33_r96_material_desc2sid_retkl_r8_v1' "$root/docs/EXPERIMENT_INDEX.md" || {
  echo "I-33 is not preregistered in EXPERIMENT_INDEX" >&2
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
check_sha256 "$data" 7d6a1e4a44238a79dcb0d31384f147c02baea95cd870224e2a6815444f8470fd "I-33 treatment data"
check_sha256 "$control" 812a6f711ff239737a9dde50acdff0588f6ac4885de037664dea58575c707d40 "I-33 control data"
check_sha256 "$holdout" 76acc6a39b248e4501a11e99bc889871f5208c811070485791b554538e658f99 "I-33 holdout"
check_sha256 "$world_holdout" f75106758792163dd33d1d52639ba507a6d9e69094d8213d5f3b0969ee272f62 "I-33 world holdout"
check_sha256 "$ledger" 22e05c2181947995500dc300b68cad8c39ece9041a56da5bf6a9829793d588af "I-33 treatment selection ledger"
check_sha256 "$control_ledger" 54fa513c2396e079ee8dc542b6117b81ce86047734cd6d04aa784003cbb36024 "I-33 control selection ledger"
check_sha256 "$replacement_ledger" 8c4774300a515e960abc8b49950030834e88af4c870828a6871d136ffcb18ca8 "I-33 retention replacement ledger"
check_sha256 "$audit" 5e731caad45ece5c1192c4d0b1193babee9ff119608ad6f64152ec778ef62bc9 "I-33 data audit"
check_sha256 "$dataset_registry" 4fb0ef52686c8fc8f9caa49da75f5c463591c4440c0c4c01fecbc1ce4642e8bd "I-33 dataset registry"
check_sha256 "$builder" 9c27a666f4a95e834742e491e46fd8cf649bec98c654be0fbe302a13e1c5ed1b "I-33 builder"
check_sha256 "$trainer" a437e3513b9577acd794b143d49162dcf9c957c473a41a2320396d5aa69ed383 "I-33 trainer"
check_sha256 "$config" c47c7f5d95bcb8fb0c671edca03601fbf4849ee0e97d8aba6d79803dfd9f53bb "I-33 config"
check_sha256 "$gate" b6279bda6167d311cd0d139261f02d15f94aa8453cb24565d4ecef6e4b183b26 "I-33 gate"
check_sha256 "$evaluator" fb4ff78e2f5615d18851c077712ee2a5504832f5fdf4f252822db910f72de44d "I-33 evaluator"
check_sha256 "$combiner" aad9370860c4af498d773629a3e6daea47853a13868e9f56cd0cd37c12d841c1 "LoRA combiner"

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
[[ "$WANDB_ENTITY" == "thaongocnguyendo0-" ]] || { echo "unexpected W&B entity" >&2; exit 2; }
[[ "$WANDB_PROJECT" == "llmrec-2026" ]] || { echo "unexpected W&B project" >&2; exit 2; }
[[ "${WANDB_MODE:-online}" == online ]] || { echo "I-33 refuses non-online W&B" >&2; exit 2; }
export WANDB_MODE=online

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config"
