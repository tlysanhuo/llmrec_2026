#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> WANDB_MODE=online $0 <single-gpu-id> <configs/active/i37_strict_future_rec_r8_v1.yaml> [--dry-run]" >&2
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
expected_config=$root/configs/active/i37_strict_future_rec_r8_v1.yaml
trainer=$root/scripts/train/train_i37_strict_future_rec.py
dataset_registry=$root/configs/datasets/i37_strict_future_rec_v1/dataset_info.json
audit=$root/logs/data/i37_strict_future_rec_v1_audit.json
data=$root/assets/derived/processed/data_i37_strict_future_rec_v1.jsonl
parent=$root/submissions/i35_r96_video_boundary_retkl_r112_step548_platform
output=$root/checkpoints/i37_strict_future_rec_r8_v1
train_log=$root/logs/train/i37_strict_future_rec_r8_v1.log

expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00
expected_parent_config=4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996
expected_trainer=568fb843c33d3c719d73915f58697c72a60e62b7c077c44f8b20a78d1f48ef61
expected_config_hash=371a1df3be3694b6fc4d79b4b1056393ed010b2e40690d5332dce46e5d17fdd1
expected_dataset_registry=0d7b8a0b038f4c51a7acbc2307a1afea70b0e000f1b28772a84c183b7d9e2bd5
expected_data=2f663a7e4f477126d765a9c8e8aaa676caf1a014b0be206978f3c93f19e948b4
expected_audit=c30f0940319459616853923b7e5b19e92c1c4f0324f58ec21fd6aba0333c6ad0
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "I-37 requires one numeric GPU id" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "missing executable Python: $python_bin" >&2; exit 2; }
[[ -f "$config_arg" ]] || { echo "missing config: $config_arg" >&2; exit 2; }
config=$(realpath "$config_arg")
[[ "$config" == "$expected_config" ]] || { echo "I-37 launcher accepts only $expected_config" >&2; exit 2; }

gpu_state=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu_memory_used=$(cut -d, -f1 <<<"$gpu_state" | tr -d '[:space:]')
gpu_utilization=$(cut -d, -f2 <<<"$gpu_state" | tr -d '[:space:]')
[[ "$gpu_memory_used" =~ ^[0-9]+$ && "$gpu_utilization" =~ ^[0-9]+$ ]] || {
  echo "I-37 could not read GPU $gpu_id state" >&2
  exit 2
}
if [[ "$mode" != "--dry-run" ]] && (( gpu_memory_used > 24576 || gpu_utilization > 70 )); then
  echo "I-37 GPU $gpu_id is too busy: ${gpu_memory_used} MiB, ${gpu_utilization}%" >&2
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

check_sha256 "$root/models/OneReason-0.8B-pretrain-competition/config.json" "$expected_base_config" "O6 base config"
check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "I-35 step548 parent adapter"
check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "I-35 step548 parent config"
check_sha256 "$trainer" "$expected_trainer" "I-37 trainer"
check_sha256 "$config" "$expected_config_hash" "I-37 config"
check_sha256 "$dataset_registry" "$expected_dataset_registry" "I-37 dataset registry"
check_sha256 "$data" "$expected_data" "I-37 formal data"
check_sha256 "$audit" "$expected_audit" "I-37 formal audit"

"$python_bin" - "$root" "$config" "$dataset_registry" "$audit" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

root, config_path, registry_path, audit_path = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected = {
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(root / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"),
    "create_new_adapter": True,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "dataset": "data_i37_strict_future_rec_v1",
    "dataset_dir": str(root / "configs/datasets/i37_strict_future_rec_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i37_strict_future_rec_r8_v1"),
    "save_strategy": "steps",
    "save_steps": 256,
    "save_total_limit": 2,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i37_strict_future_rec_r8_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-6,
    "max_steps": 512,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "bf16": True,
    "seed": 19260837,
}
for key, value in expected.items():
    observed = config.get(key)
    if isinstance(value, float):
        if not isinstance(observed, (int, float)) or not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"I-37 config drift for {key}: {observed!r}/{value!r}")
    elif observed != value:
        raise SystemExit(f"I-37 config drift for {key}: {observed!r}/{value!r}")
if config.get("resume_from_checkpoint"):
    raise SystemExit("I-37 cannot resume from a checkpoint")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i37_strict_future_rec_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected I-37 registry keys: {sorted(registry)}")
entry = registry[key]
expected_columns = {"prompt": "input", "response": "output", "history": "history", "system": "instruction"}
if entry.get("formatting") != "alpaca" or entry.get("columns") != expected_columns:
    raise SystemExit("I-37 dataset registry drifted")
if Path(entry.get("file_name", "")).resolve() != (root / "assets/derived/processed/data_i37_strict_future_rec_v1.jsonl").resolve():
    raise SystemExit("I-37 registry points to the wrong data")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
contract = audit.get("contract", {})
if contract.get("total_rows") != 2048 or contract.get("route_counts") != {"future_ce": 1024, "retention_kl": 1024}:
    raise SystemExit("I-37 audit contract drifted")
PY

if [[ "$mode" == "--dry-run" ]]; then
  echo "I-37 dry-run PASS: single GPU $gpu_id; static model/config/data/audit contracts are locked; current GPU state ${gpu_memory_used} MiB/${gpu_utilization}%"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || { echo "I-37 requires W&B entity $expected_wandb_entity" >&2; exit 2; }
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || { echo "I-37 requires W&B project $expected_wandb_project" >&2; exit 2; }
[[ "${WANDB_MODE:-online}" == online ]] || { echo "I-37 refuses non-online W&B" >&2; exit 2; }
export WANDB_MODE=online

grep -Fq 'data_i37_strict_future_rec_v1.jsonl' "$root/docs/reference/ASSETS.md" || { echo "I-37 data is not registered" >&2; exit 2; }
grep -Fq 'i37_strict_future_rec_r8_v1' "$root/docs/EXPERIMENT_INDEX.md" || { echo "I-37 experiment is not registered" >&2; exit 2; }
grep -Fq 'I-37单GPU/W&B正式训练' "$root/docs/TODO.md" || { echo "I-37 training is not registered in TODO" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "I-37 refuses to overwrite $output" >&2; exit 2; }

export I37_TRAINING_DATA="$data"
export I37_AUDIT="$audit"
export I37_OUTPUT_DIR="$output"

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

mkdir -p "$(dirname "$train_log")"
env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config" 2>&1 | tee "$train_log"
