#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> WANDB_MODE=online $0 <single-gpu-id> <configs/active/i36_i35_user_expand_retkl_r16_v1.yaml> [--dry-run]" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

gpu_id=$1
config_arg=$2
mode=${3:-}
if [[ -n "$mode" && "$mode" != "--dry-run" ]]; then
  usage
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_python=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3
python_bin=${LLAMAFACTORY_PYTHON:-$runtime_python}
expected_config=$root/configs/active/i36_i35_user_expand_retkl_r16_v1.yaml
trainer=$root/scripts/train/train_i36_i35_user_expand_retkl.py
dataset_registry=$root/configs/datasets/i36_i35_user_expand_retkl_v1/dataset_info.json
audit=$root/logs/data/i36_i35_user_expand_retkl_v1_audit.json
assets_registry=$root/docs/reference/ASSETS.md
experiment_index=$root/docs/EXPERIMENT_INDEX.md
todo=$root/docs/TODO.md
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
parent=$root/submissions/i35_r96_video_boundary_retkl_r112_step548_platform
data=$root/assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl
output=$root/checkpoints/i36_i35_user_expand_retkl_r16_v1

expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00
expected_parent_config=4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996
expected_trainer=e760cba91fe02553e1545d1fff8f3da303bfa4304974a0106f3c00a1db9ff9e3
expected_config_hash=2a2194ecef159786368c37c334166922dbfedcf3f366bec7353c073c79f43db3
expected_dataset_registry=dee54b0c94a12bf04edc6c99b45fe20cce9950033d0aa4f48e7d132b19f4ffce
expected_data=2720746a2e8aa7804d519698ce9f2b127e9be2db1d4488e642e800a5337b692d
expected_audit=eb426018525f9e3e1d682e1c89e5ca3dc8963b0a57c104911e1197324e464240
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ "$gpu_id" =~ ^[0-9]+$ ]] || {
  echo "I-36 requires exactly one numeric GPU id" >&2
  exit 2
}
[[ -x "$python_bin" ]] || { echo "missing executable Python: $python_bin" >&2; exit 2; }
[[ -f "$config_arg" ]] || { echo "missing config: $config_arg" >&2; exit 2; }
config=$(realpath "$config_arg")
[[ "$config" == "$expected_config" ]] || {
  echo "I-36 launcher accepts only $expected_config" >&2
  exit 2
}

gpu_memory_used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$gpu_memory_used" =~ ^[0-9]+$ ]] || {
  echo "I-36 could not read GPU $gpu_id memory usage" >&2
  exit 2
}
if (( gpu_memory_used > 1024 )); then
  echo "I-36 requires an empty GPU; GPU $gpu_id already uses ${gpu_memory_used} MiB" >&2
  exit 2
fi

check_sha256() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" ]] || { echo "missing $label: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label checksum drifted: $actual/$expected" >&2
    exit 2
  }
}

check_sha256 "$base_config" "$expected_base_config" "O6 base config"
check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "I-35 step548 parent adapter"
check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "I-35 step548 parent config"
check_sha256 "$trainer" "$expected_trainer" "I-36 trainer"
check_sha256 "$config" "$expected_config_hash" "I-36 config"
check_sha256 "$dataset_registry" "$expected_dataset_registry" "I-36 dataset registry"
check_sha256 "$data" "$expected_data" "I-36 formal data"
check_sha256 "$audit" "$expected_audit" "I-36 formal audit"

"$python_bin" - "$root" "$config" "$dataset_registry" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

root, config_path, registry_path = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("I-36 YAML must contain a mapping")
expected = {
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(root / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"),
    "create_new_adapter": True,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "dataset": "data_i36_i35_user_expand_retkl_v1",
    "dataset_dir": str(root / "configs/datasets/i36_i35_user_expand_retkl_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i36_i35_user_expand_retkl_r16_v1"),
    "save_strategy": "steps",
    "save_steps": 2063,
    "save_total_limit": 2,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i36_i35_user_expand_retkl_r16_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-6,
    "max_steps": 4125,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "bf16": True,
    "seed": 19260836,
}
for key, value in expected.items():
    observed = config.get(key)
    if isinstance(value, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SystemExit(f"I-36 config drift for {key}: {observed!r}/{value!r}")
    elif observed != value:
        raise SystemExit(f"I-36 config drift for {key}: {observed!r}/{value!r}")
if config.get("resume_from_checkpoint"):
    raise SystemExit("I-36 cannot resume from a checkpoint")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i36_i35_user_expand_retkl_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected I-36 registry keys: {sorted(registry)}")
entry = registry[key]
expected_columns = {
    "prompt": "input",
    "response": "output",
    "history": "history",
    "system": "instruction",
}
if entry.get("formatting") != "alpaca" or entry.get("columns") != expected_columns:
    raise SystemExit("I-36 registry must map instruction to system and input to user")
expected_data = root / "assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl"
if Path(entry.get("file_name", "")).resolve() != expected_data.resolve():
    raise SystemExit("I-36 registry points to the wrong formal data path")
PY

if [[ "$mode" == "--dry-run" ]]; then
  echo "I-36 dry-run PASS: static model/config/data/audit contracts are locked"
  echo "entrypoint=$trainer"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || {
  echo "I-36 requires W&B entity $expected_wandb_entity" >&2
  exit 2
}
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || {
  echo "I-36 requires W&B project $expected_wandb_project" >&2
  exit 2
}
[[ "${WANDB_MODE:-online}" == online ]] || {
  echo "I-36 refuses non-online W&B (WANDB_MODE=${WANDB_MODE:-unset})" >&2
  exit 2
}
export WANDB_MODE=online

for required in "$assets_registry" "$experiment_index" "$todo"; do
  [[ -f "$required" ]] || { echo "missing formal I-36 registry: $required" >&2; exit 2; }
done
grep -Fq 'data_i36_i35_user_expand_retkl_v1.jsonl' "$assets_registry" || {
  echo "I-36 training data is not registered in ASSETS.md" >&2
  exit 2
}
grep -Fq 'i36_i35_user_expand_retkl_r16_v1' "$experiment_index" || {
  echo "I-36 is not registered in EXPERIMENT_INDEX.md" >&2
  exit 2
}
grep -Fq 'I-36单GPU/W&B正式训练' "$todo" || {
  echo "I-36 formal training is not registered in TODO.md" >&2
  exit 2
}
[[ ! -e "$output" ]] || {
  echo "I-36 refuses to overwrite its reserved output: $output" >&2
  exit 2
}

export I36_PARENT_ADAPTER="$parent"
export I36_TRAINING_DATA="$data"
export I36_AUDIT="$audit"
export I36_OUTPUT_DIR="$output"

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config"
