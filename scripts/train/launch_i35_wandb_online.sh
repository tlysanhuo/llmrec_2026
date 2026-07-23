#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> WANDB_MODE=online $0 <single-gpu-id> <configs/active/i35_r96_video_boundary_retkl_r16_v1.yaml> [--dry-run]" >&2
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
python_bin=${LLAMAFACTORY_PYTHON:-$root/LLaMA-Factory/.venv/bin/python3}
expected_config=$root/configs/active/i35_r96_video_boundary_retkl_r16_v1.yaml
trainer=$root/scripts/train/train_i35_video_boundary_retkl.py
dataset_registry=$root/configs/datasets/i35_video_boundary_retkl_v1/dataset_info.json
audit=$root/logs/data/i35_video_boundary_retkl_v1_audit.json
assets_registry=$root/docs/reference/ASSETS.md
experiment_index=$root/docs/EXPERIMENT_INDEX.md
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
parent=$root/submissions/i19_world_external_r96_s875_platform
data=$root/assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl
sidecar=$root/assets/derived/processed/data_i35_video_boundary_retkl_v1_sidecar.jsonl
output=$root/checkpoints/i35_r96_video_boundary_retkl_r16_v1

expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e
expected_parent_config=78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f
expected_trainer=927ebe805945507d04a1a3d4c149c94497768eb84cdce17788f04ed768de84e4
expected_config_hash=0a1491af2913b29294b17f6dc80a3b6cca0cb1c2cc6b881c86c6a14418341ebd
expected_dataset_registry=87278e058573c2742accf1f253b076d270256d39e0a2fd9ba5f1b105eba893ff
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ "$gpu_id" =~ ^[0-9]+$ ]] || {
  echo "I-35 requires exactly one numeric GPU id" >&2
  exit 2
}
[[ -x "$python_bin" ]] || { echo "missing executable Python: $python_bin" >&2; exit 2; }
[[ -f "$config_arg" ]] || { echo "missing config: $config_arg" >&2; exit 2; }
config=$(realpath "$config_arg")
[[ "$config" == "$expected_config" ]] || {
  echo "I-35 launcher accepts only $expected_config" >&2
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

check_sha256 "$base_config" "$expected_base_config" "O6 base config"
check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "r96 parent adapter"
check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "r96 parent config"
check_sha256 "$trainer" "$expected_trainer" "I-35 trainer"
check_sha256 "$config" "$expected_config_hash" "I-35 config"
check_sha256 "$dataset_registry" "$expected_dataset_registry" "I-35 dataset registry"

"$python_bin" - "$root" "$config" "$dataset_registry" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

root, config_path, registry_path = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("I-35 YAML must contain a mapping")
expected = {
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(root / "submissions/i19_world_external_r96_s875_platform"),
    "create_new_adapter": True,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "dataset": "data_i35_video_boundary_retkl_v1",
    "dataset_dir": str(root / "configs/datasets/i35_video_boundary_retkl_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i35_r96_video_boundary_retkl_r16_v1"),
    "save_strategy": "steps",
    "save_steps": 137,
    "save_total_limit": 5,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i35_r96_video_boundary_retkl_r16_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1.0e-5,
    "max_steps": 685,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "bf16": True,
    "seed": 19260835,
}
for key, value in expected.items():
    observed = config.get(key)
    if isinstance(value, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise SystemExit(f"I-35 config drift for {key}: {observed!r}/{value!r}")
    elif observed != value:
        raise SystemExit(f"I-35 config drift for {key}: {observed!r}/{value!r}")
if config.get("resume_from_checkpoint"):
    raise SystemExit("I-35 cannot resume from a checkpoint")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i35_video_boundary_retkl_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected I-35 registry keys: {sorted(registry)}")
entry = registry[key]
expected_columns = {
    "prompt": "input",
    "response": "output",
    "history": "history",
    "system": "instruction",
}
if entry.get("formatting") != "alpaca" or entry.get("columns") != expected_columns:
    raise SystemExit("I-35 registry must map instruction to system and input to user")
expected_data = root / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"
if Path(entry.get("file_name", "")).resolve() != expected_data.resolve():
    raise SystemExit("I-35 registry points to the wrong formal data path")
PY

if [[ "$mode" == "--dry-run" ]]; then
  echo "I-35 dry-run PASS: static model/config/registry contracts are locked"
  echo "expected_audit=$audit"
  echo "entrypoint=$trainer"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || {
  echo "I-35 requires W&B entity $expected_wandb_entity" >&2
  exit 2
}
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || {
  echo "I-35 requires W&B project $expected_wandb_project" >&2
  exit 2
}
[[ "${WANDB_MODE:-online}" == online ]] || {
  echo "I-35 refuses non-online W&B (WANDB_MODE=${WANDB_MODE:-unset})" >&2
  exit 2
}
export WANDB_MODE=online

for required in "$audit" "$data" "$sidecar" "$assets_registry" "$experiment_index"; do
  [[ -f "$required" ]] || { echo "missing formal I-35 input: $required" >&2; exit 2; }
done
grep -Fq 'data_i35_video_boundary_retkl_v1.jsonl' "$assets_registry" || {
  echo "I-35 training data is not registered in ASSETS.md" >&2
  exit 2
}
grep -Fq 'data_i35_video_boundary_retkl_v1_sidecar.jsonl' "$assets_registry" || {
  echo "I-35 sidecar is not registered in ASSETS.md" >&2
  exit 2
}
grep -Fq 'i35_r96_video_boundary_retkl_r16_v1' "$experiment_index" || {
  echo "I-35 is not registered in EXPERIMENT_INDEX.md" >&2
  exit 2
}
[[ ! -e "$output" ]] || {
  echo "I-35 refuses to overwrite its reserved output: $output" >&2
  exit 2
}

export I35_PARENT_ADAPTER="$parent"
export I35_TRAINING_DATA="$data"
export I35_SIDECAR="$sidecar"
export I35_AUDIT="$audit"
export I35_OUTPUT_DIR="$output"

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config"
