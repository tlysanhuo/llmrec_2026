#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> WANDB_MODE=online $0 <single-gpu-id> <configs/active/i34_r96_material_beam_margin_retkl_r16_v1.yaml> [--dry-run]" >&2
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
expected_config=$root/configs/active/i34_r96_material_beam_margin_retkl_r16_v1.yaml
trainer=$root/scripts/train/train_i34_material_beam_margin_retkl.py
dataset_registry=$root/configs/datasets/i34_r96_material_beam_margin_retkl_v1/dataset_info.json
gate=$root/configs/evaluation/i34_r96_material_beam_margin_checkpoint_gate.json
assets_registry=$root/docs/reference/ASSETS.md
experiment_index=$root/docs/EXPERIMENT_INDEX.md
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
parent=$root/submissions/i19_world_external_r96_s875_platform
teacher=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
data=$root/assets/derived/processed/data_i34_material_beam_margin_retkl_v1.jsonl
sidecar=$root/assets/derived/processed/data_i34_material_beam_margin_retkl_v1_sidecar.jsonl
output=$root/checkpoints/i34_r96_material_beam_margin_retkl_r16_v1

expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e
expected_parent_config=78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f
expected_teacher=0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8
expected_teacher_config=b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7
expected_trainer=ad5fb60d877a75538999cc319d992a79d673c216e4c353d0ec1774ddd69f7dde
expected_config_hash=1a29c2f474838630fe594d3698c7aadd112d05374bf2cfad6307871ac277d757
expected_dataset_registry=f7a35670e5124fd20fb9e37a2062191b76f2ad37768a9ce91bc942b61713c084
expected_gate_hash=f3090a7354bd6da25c34474f15046ac0cf49dbedf28682af8719478b65142b49
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || {
  echo "I-34 requires exactly one visible GPU" >&2
  exit 2
}
[[ -x "$python_bin" ]] || { echo "missing executable Python: $python_bin" >&2; exit 2; }
[[ -f "$config_arg" ]] || { echo "missing config: $config_arg" >&2; exit 2; }
config=$(realpath "$config_arg")
[[ "$config" == "$expected_config" ]] || {
  echo "I-34 launcher accepts only $expected_config" >&2
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

for required in \
  "$trainer" "$dataset_registry" "$gate" "$assets_registry" "$experiment_index" \
  "$base_config" "$parent/adapter_model.safetensors" "$parent/adapter_config.json" \
  "$teacher/adapter_model.safetensors" "$teacher/adapter_config.json"; do
  [[ -f "$required" ]] || { echo "missing frozen I-34 input: $required" >&2; exit 2; }
done

check_sha256 "$base_config" "$expected_base_config" "O6 base config"
check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "r96 parent adapter"
check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "r96 parent config"
check_sha256 "$teacher/adapter_model.safetensors" "$expected_teacher" "I-23 construction teacher"
check_sha256 "$teacher/adapter_config.json" "$expected_teacher_config" "I-23 construction teacher config"
check_sha256 "$trainer" "$expected_trainer" "I-34 trainer"
check_sha256 "$config" "$expected_config_hash" "I-34 config"
check_sha256 "$dataset_registry" "$expected_dataset_registry" "I-34 dataset registry"
check_sha256 "$gate" "$expected_gate_hash" "I-34 checkpoint gate"

"$python_bin" - "$root" "$config" "$dataset_registry" "$trainer" "$gate" <<'PY'
import json
import math
import sys
from pathlib import Path

root, config_path, registry_path, trainer_path, gate_path = map(Path, sys.argv[1:])
try:
    import yaml
except Exception as error:
    raise SystemExit(f"I-34 static validation requires PyYAML: {error}")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit("I-34 YAML must contain a mapping")
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
    "dataset": "data_i34_material_beam_margin_retkl_v1",
    "dataset_dir": str(root / "configs/datasets/i34_r96_material_beam_margin_retkl_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i34_r96_material_beam_margin_retkl_r16_v1"),
    "save_strategy": "steps",
    "save_steps": 64,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i34_r96_material_beam_margin_retkl_r16_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_steps": 128,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "bf16": True,
    "seed": 19260834,
}
for key, value in expected.items():
    observed = config.get(key)
    if isinstance(value, float):
        if not isinstance(observed, (int, float)) or not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"I-34 config drift for {key}: {observed!r}/{value!r}")
    elif observed != value:
        raise SystemExit(f"I-34 config drift for {key}: {observed!r}/{value!r}")
if "resume_from_checkpoint" in config and config["resume_from_checkpoint"]:
    raise SystemExit("I-34 must start a fresh residual and cannot resume")
if float(config.get("learning_rate", -1.0)) != 1.0e-5:
    raise SystemExit("I-34 learning_rate must be 1e-5")
if int(config.get("save_total_limit", 0)) < 2:
    raise SystemExit("I-34 must retain both adapter-only checkpoints 64 and 128")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i34_material_beam_margin_retkl_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected I-34 registry keys: {sorted(registry)}")
entry = registry[key]
expected_columns = {"prompt": "instruction", "query": "input", "response": "output", "history": "history"}
if entry.get("formatting") != "alpaca" or entry.get("columns") != expected_columns:
    raise SystemExit("I-34 dataset registry schema drifted")
expected_data = root / "assets/derived/processed/data_i34_material_beam_margin_retkl_v1.jsonl"
if Path(entry.get("file_name", "")).resolve() != expected_data.resolve():
    raise SystemExit("I-34 registry does not point to the reserved formal data path")

trainer_source = trainer_path.read_text(encoding="utf-8")
for marker in (
    "CustomSeq2SeqTrainer.compute_loss",
    "from llamafactory.train.tuner import run_exp",
    "run_exp()",
    "generation_max_length",
    "save_only_model",
):
    if marker not in trainer_source:
        raise SystemExit(f"I-34 trainer integration marker missing: {marker}")
cli_source = (root / "LLaMA-Factory/src/llamafactory/launcher.py").read_text(encoding="utf-8")
if "from .train.tuner import run_exp" not in cli_source:
    raise SystemExit("LLaMA-Factory CLI no longer dispatches train to run_exp")

gate = json.loads(gate_path.read_text(encoding="utf-8"))
recipe = gate.get("frozen_training_recipe_after_admission", {})
for key, value in {
    "single_gpu": True,
    "residual_rank": 16,
    "residual_alpha": 16,
    "residual_dropout": 0.05,
    "target_modules": "all linear",
    "material_rows": 128,
    "retention_rows": 384,
    "batch": 1,
    "gradient_accumulation": 4,
    "optimizer_steps": 128,
    "learning_rate": 0.00001,
    "scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "cutoff": 16384,
    "seed": 19260834,
}.items():
    observed = recipe.get(key)
    if isinstance(value, float):
        if not isinstance(observed, (int, float)) or not math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"I-34 gate recipe drift for {key}: {observed!r}/{value!r}")
    elif observed != value:
        raise SystemExit(f"I-34 gate recipe drift for {key}: {observed!r}/{value!r}")
if gate.get("checkpoint_gate", {}).get("candidate_steps_in_order") != [64, 128]:
    raise SystemExit("I-34 checkpoint candidates must remain [64, 128]")
PY

if [[ "$mode" == "--dry-run" ]]; then
  gate_status=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$gate")
  echo "I-34 dry-run: no training or data preflight started"
  echo "gate_status=$gate_status"
  echo "expected_training_data=$data"
  echo "expected_sidecar=$sidecar"
  echo "entrypoint=$trainer (patches CustomSeq2SeqTrainer, then calls run_exp)"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || {
  echo "I-34 requires W&B entity $expected_wandb_entity" >&2
  exit 2
}
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || {
  echo "I-34 requires W&B project $expected_wandb_project" >&2
  exit 2
}
[[ "${WANDB_MODE:-online}" == online ]] || {
  echo "I-34 refuses non-online W&B (WANDB_MODE=${WANDB_MODE:-unset})" >&2
  exit 2
}
export WANDB_MODE=online

gate_status=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$gate")
[[ "$gate_status" == "PREREGISTERED_AND_HASH_FROZEN_READY_FOR_FORMAL_TRAINING" ]] || {
  echo "I-34 is not launch-authorized: gate status=$gate_status" >&2
  exit 2
}
if grep -Eq 'PENDING_|^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$gate" "$config"; then
  echo "I-34 still contains a pending or disabled marker" >&2
  exit 2
fi
grep -Fq 'data_i34_material_beam_margin_retkl_v1.jsonl' "$assets_registry" || {
  echo "I-34 training data is not registered in ASSETS.md" >&2
  exit 2
}
grep -Fq 'data_i34_material_beam_margin_retkl_v1_sidecar.jsonl' "$assets_registry" || {
  echo "I-34 sidecar is not registered in ASSETS.md" >&2
  exit 2
}
grep -Fq 'i34_r96_material_beam_margin_r16_v1' "$experiment_index" || {
  echo "I-34 is not recorded in EXPERIMENT_INDEX.md" >&2
  exit 2
}

[[ "${I34_TRAINING_DATA:-$data}" == "$data" ]] || {
  echo "I-34 launcher does not permit a training-data path override" >&2
  exit 2
}
[[ "${I34_SIDECAR:-$sidecar}" == "$sidecar" ]] || {
  echo "I-34 launcher does not permit a sidecar path override" >&2
  exit 2
}
: "${I34_TRAINING_DATA_SHA256:?I34_TRAINING_DATA_SHA256 is required after data admission}"
: "${I34_SIDECAR_SHA256:?I34_SIDECAR_SHA256 is required after sidecar admission}"
[[ "$I34_TRAINING_DATA_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid I34_TRAINING_DATA_SHA256" >&2; exit 2; }
[[ "$I34_SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid I34_SIDECAR_SHA256" >&2; exit 2; }
check_sha256 "$data" "$I34_TRAINING_DATA_SHA256" "I-34 training data"
check_sha256 "$sidecar" "$I34_SIDECAR_SHA256" "I-34 sidecar"

[[ ! -e "$output" ]] || {
  echo "I-34 refuses to overwrite its reserved output: $output" >&2
  exit 2
}

export I34_PARENT_ADAPTER="$parent"
export I34_TRAINING_DATA="$data"
export I34_SIDECAR="$sidecar"
export I34_OUTPUT_DIR="$output"

# The wrapper is intentional: it installs the I-34 loss/adapter guards before
# invoking run_exp(), the exact function reached by `llamafactory-cli train`.
# Calling llamafactory-cli directly would train ordinary SFT and bypass I-34.
"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" "$trainer" "$config"
