#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> WANDB_MODE=online $0 <single-gpu-id> <configs/active/i38_i23_material_i35_teacher_retkl_r16_v1.yaml> [--dry-run]" >&2
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
expected_config=$root/configs/active/i38_i23_material_i35_teacher_retkl_r16_v1.yaml
trainer=$root/scripts/train/train_i38_i23_material_i35_teacher_retkl.py
dataset_registry=$root/configs/datasets/i38_i23_material_i35_teacher_retkl_v1/dataset_info.json
data=$root/assets/derived/processed/data_i38_i23_material_i35_teacher_retkl_v1.jsonl
audit=$root/logs/data/i38_i23_material_i35_teacher_retkl_v1_audit.json
gate=$root/assets/evaluation/holdout/data_i38_i23_material_i35_teacher_gate_v1.jsonl
gate_audit=$root/logs/data/i38_i23_material_i35_teacher_gate_v1_audit.json
gate_config=$root/configs/evaluation/i38_i23_material_i35_teacher_checkpoint_gate.json
evaluator=$root/scripts/eval/audit_i38_i23_material_i35_teacher_gate.py
start=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
teacher=$root/submissions/i35_r96_video_boundary_retkl_r112_step548_platform
output=$root/checkpoints/i38_i23_material_i35_teacher_retkl_r16_v1
train_log=$root/logs/train/i38_i23_material_i35_teacher_retkl_r16_v1.log

expected_base=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_start=0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8
expected_start_config=b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7
expected_teacher=52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00
expected_teacher_config=4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996
expected_trainer=1111bd6dc7dfc54edbb599da5e987de51cf50f75acb5eecaa3f4a4987b0b41c7
expected_config_hash=45a35e6fe21d4afdd80555bfa56ef3be2247188ac76d442e6beb7b36b0c71893
expected_registry=de6260b8083bec7bae47a0ec35d1901256af7ac7c71a67fb98df3e0f1899a7c0
expected_data=5d8ca1a6fa9190841187543559ead1d497d48a50b082382c9fa8501add928d58
expected_audit=f2f0cb5df91d370763c8dbf36854a05b3240284cb6e7cabbdbddfa9e0903a99c
expected_gate=311b298f939a953aed7a8a11a694e257518e273a1e506537d76953720eaed41f
expected_gate_audit=b281ef99821853b373fe7b947b1d2421330c9d8c703f3dc389347b3286f1692b
expected_gate_config=7b3e5e5e6c7c9cc74eb23140c71b2d53ada6e50a07f4c1a6e619f5a0b8ee5a5f
expected_evaluator=73d23815a087be8de7ca41d3b29d6bf712ac4d3d590ed639fc729fe6497535c8
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "I-38 requires exactly one numeric GPU id" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "missing executable Python: $python_bin" >&2; exit 2; }
[[ -f "$config_arg" ]] || { echo "missing config: $config_arg" >&2; exit 2; }
config=$(realpath "$config_arg")
[[ "$config" == "$expected_config" ]] || { echo "I-38 launcher accepts only $expected_config" >&2; exit 2; }

gpu_state=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu_memory_used=$(cut -d, -f1 <<<"$gpu_state" | tr -d '[:space:]')
gpu_utilization=$(cut -d, -f2 <<<"$gpu_state" | tr -d '[:space:]')
[[ "$gpu_memory_used" =~ ^[0-9]+$ && "$gpu_utilization" =~ ^[0-9]+$ ]] || {
  echo "I-38 could not read GPU $gpu_id state" >&2
  exit 2
}
if [[ "$mode" != "--dry-run" ]] && (( gpu_memory_used > 8192 || gpu_utilization > 50 )); then
  echo "I-38 GPU $gpu_id is too busy: ${gpu_memory_used} MiB, ${gpu_utilization}%" >&2
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

check_sha256 "$root/models/OneReason-0.8B-pretrain-competition/config.json" "$expected_base" "O6 base config"
check_sha256 "$start/adapter_model.safetensors" "$expected_start" "I-23 start adapter"
check_sha256 "$start/adapter_config.json" "$expected_start_config" "I-23 start config"
check_sha256 "$teacher/adapter_model.safetensors" "$expected_teacher" "I-35 teacher adapter"
check_sha256 "$teacher/adapter_config.json" "$expected_teacher_config" "I-35 teacher config"
check_sha256 "$trainer" "$expected_trainer" "I-38 trainer"
check_sha256 "$config" "$expected_config_hash" "I-38 config"
check_sha256 "$dataset_registry" "$expected_registry" "I-38 dataset registry"
check_sha256 "$data" "$expected_data" "I-38 formal data"
check_sha256 "$audit" "$expected_audit" "I-38 formal data audit"
check_sha256 "$gate" "$expected_gate" "I-38 frozen gate"
check_sha256 "$gate_audit" "$expected_gate_audit" "I-38 gate audit"
check_sha256 "$gate_config" "$expected_gate_config" "I-38 checkpoint gate"
check_sha256 "$evaluator" "$expected_evaluator" "I-38 frozen evaluator"

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
    "adapter_name_or_path": str(root / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"),
    "create_new_adapter": True,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "dataset": "data_i38_i23_material_i35_teacher_retkl_v1",
    "dataset_dir": str(root / "configs/datasets/i38_i23_material_i35_teacher_retkl_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i38_i23_material_i35_teacher_retkl_r16_v1"),
    "save_strategy": "no",
    "save_only_model": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i38_i23_material_i35_teacher_retkl_r16_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-6,
    "max_steps": 685,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "bf16": True,
    "seed": 19260838,
}
for key, value in expected.items():
    observed = config.get(key)
    if isinstance(value, float):
        if not isinstance(observed, (int, float)) or not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"I-38 config drift for {key}: {observed!r}/{value!r}")
    elif observed != value:
        raise SystemExit(f"I-38 config drift for {key}: {observed!r}/{value!r}")
if config.get("resume_from_checkpoint"):
    raise SystemExit("I-38 cannot resume from a checkpoint")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i38_i23_material_i35_teacher_retkl_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected I-38 registry keys: {sorted(registry)}")
entry = registry[key]
expected_columns = {"prompt": "input", "response": "output", "history": "history", "system": "instruction"}
if entry.get("formatting") != "alpaca" or entry.get("columns") != expected_columns:
    raise SystemExit("I-38 dataset registry drifted")
expected_path = root / "assets/derived/processed/data_i38_i23_material_i35_teacher_retkl_v1.jsonl"
if Path(entry.get("file_name", "")).resolve() != expected_path.resolve():
    raise SystemExit("I-38 registry points to the wrong data")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
output = audit.get("output", {})
if output.get("rows") != 2740 or output.get("route_counts") != {
    "material_anchor_i23": 1370,
    "retention_teacher_i35": 1370,
}:
    raise SystemExit("I-38 audit route contract drifted")
mix = audit.get("mix", {})
if any(mix.get(key) != 0 for key in ("T_rows", "E_rows", "model_generated_rows")):
    raise SystemExit("I-38 audit contains a forbidden training source")
PY

if [[ "$mode" == "--dry-run" ]]; then
  echo "I-38 dry-run PASS: one GPU $gpu_id; static model/data/config/gate contracts locked; GPU ${gpu_memory_used} MiB/${gpu_utilization}%"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || { echo "I-38 requires W&B entity $expected_wandb_entity" >&2; exit 2; }
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || { echo "I-38 requires W&B project $expected_wandb_project" >&2; exit 2; }
[[ "${WANDB_MODE:-online}" == online ]] || { echo "I-38 refuses non-online W&B" >&2; exit 2; }
export WANDB_MODE=online

grep -Fq 'data_i38_i23_material_i35_teacher_retkl_v1.jsonl' "$root/docs/reference/ASSETS.md" || { echo "I-38 data is not registered" >&2; exit 2; }
grep -Fq 'i38_i23_material_i35_teacher_retkl_r16_v1' "$root/docs/EXPERIMENT_INDEX.md" || { echo "I-38 experiment is not registered" >&2; exit 2; }
grep -Fq 'I-38M单GPU/W&B正式训练' "$root/docs/TODO.md" || { echo "I-38 training is not registered in TODO" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "I-38 refuses to overwrite $output" >&2; exit 2; }

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$python_bin" "$evaluator" --self-test
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

mkdir -p "$(dirname "$train_log")"
env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config" 2>&1 | tee "$train_log"
