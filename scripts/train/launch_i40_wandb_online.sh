#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=thaongocnguyendo0- WANDB_PROJECT=llmrec-2026 WANDB_MODE=online $0 <single-gpu-id> [--dry-run]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

gpu_id=$1
mode=${2:-}
if [[ -n "$mode" && "$mode" != "--dry-run" ]]; then
  usage
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3
config=$root/configs/active/i40_i35_direct_user_continue_r112_v1.yaml
trainer=$root/scripts/train/train_i40_i35_direct_user_continue.py
dataset_registry=$root/configs/datasets/i40_i35_direct_user_continue_v1/dataset_info.json
data=$root/assets/derived/processed/data_i40_i35_direct_user_continue_v1.jsonl
sidecar=$root/assets/derived/processed/data_i40_i35_direct_user_continue_v1_sidecar.jsonl
audit=$root/logs/data/i40_i35_direct_user_continue_v1_audit.json
parent=$root/submissions/i35_r96_video_boundary_retkl_r112_step548_platform
output=$root/checkpoints/i40_i35_direct_user_continue_r112_v1
train_log=$root/logs/train/i40_i35_direct_user_continue_r112_v1.log

expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00
expected_parent_config=4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996
expected_trainer=cd5eb4095a692015bfb26831b10c02da1c0c2836a56581096ecf90e02f915473
expected_config=992825270a4f16ed9a4ec26f2e0d974603a9c2c1f9ae9c4e79fcb50da2cdfbc3
expected_registry=6bd0f06fb18ea4d2a864e6515c1d0c85ce6f5f4bb2c0f0f22f047a61ba5d06e0
expected_data=483a4bb2f98d41497600d078032634d4f36fe2970a53d98b4a7fccc488910c18
expected_sidecar=e9bc129cd834bff161247985cc5430cf46872006cbae4a86fd37c3666b60acb2
expected_audit=c5c2323b2c9aa1dddd4e49936bad09a5cb342a1f32d592f3574cc442b6985b7c
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ "$gpu_id" =~ ^[0-9]+$ ]] || {
  echo "I-40 requires exactly one numeric GPU id" >&2
  exit 2
}
[[ -x "$python_bin" ]] || {
  echo "missing executable Python: $python_bin" >&2
  exit 2
}

if ! gpu_state=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits); then
  echo "I-40 could not read GPU $gpu_id state" >&2
  exit 2
fi
gpu_memory_used=$(cut -d, -f1 <<<"$gpu_state" | tr -d '[:space:]')
gpu_utilization=$(cut -d, -f2 <<<"$gpu_state" | tr -d '[:space:]')
[[ "$gpu_memory_used" =~ ^[0-9]+$ && "$gpu_utilization" =~ ^[0-9]+$ ]] || {
  echo "I-40 could not parse GPU $gpu_id state" >&2
  exit 2
}
if [[ "$mode" != "--dry-run" ]] && (( gpu_memory_used > 1024 || gpu_utilization > 10 )); then
  echo "I-40 requires an empty GPU; GPU $gpu_id uses ${gpu_memory_used} MiB at ${gpu_utilization}%" >&2
  exit 2
fi

check_sha256() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "$path" ]] || {
    echo "missing $label: $path" >&2
    exit 2
  }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label checksum drifted: $actual/$expected" >&2
    exit 2
  }
}

verify_frozen_assets() {
  check_sha256 "$root/models/OneReason-0.8B-pretrain-competition/config.json" "$expected_base_config" "O6 base config"
  check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "I-35 step548 policy/reference adapter"
  check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "I-35 step548 adapter config"
  check_sha256 "$trainer" "$expected_trainer" "I-40 trainer"
  check_sha256 "$config" "$expected_config" "I-40 config"
  check_sha256 "$dataset_registry" "$expected_registry" "I-40 dataset registry"
  check_sha256 "$data" "$expected_data" "I-40 formal data"
  check_sha256 "$sidecar" "$expected_sidecar" "I-40 full routing sidecar"
  check_sha256 "$audit" "$expected_audit" "I-40 formal audit"
}

verify_frozen_assets
[[ ! -e "$output" ]] || {
  echo "I-40 refuses to overwrite $output" >&2
  exit 2
}
[[ ! -e "$train_log" ]] || {
  echo "I-40 refuses to overwrite $train_log" >&2
  exit 2
}

"$python_bin" - "$root" "$config" "$dataset_registry" "$data" "$sidecar" "$audit" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

root, config_path, registry_path, data_path, sidecar_path, audit_path = map(
    Path, sys.argv[1:]
)
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected = {
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(
        root / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"
    ),
    "create_new_adapter": False,
    "trust_remote_code": True,
    "flash_attn": "fa2",
    "enable_liger_kernel": False,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "data_i40_i35_direct_user_continue_v1",
    "dataset_dir": str(root / "configs/datasets/i40_i35_direct_user_continue_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "overwrite_cache": True,
    "preprocessing_num_workers": 16,
    "dataloader_num_workers": 8,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i40_i35_direct_user_continue_r112_v1"),
    "logging_steps": 25,
    "save_strategy": "steps",
    "save_steps": 515,
    "save_total_limit": 4,
    "save_only_model": True,
    "plot_loss": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i40_i35_direct_user_continue_r112_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-7,
    "max_steps": 2060,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "max_grad_norm": 0.5,
    "bf16": True,
    "seed": 19260840,
}
for key, expected_value in expected.items():
    observed = config.get(key)
    if isinstance(expected_value, float):
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isclose(
                float(observed), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise SystemExit(
                f"I-40 config drift for {key}: {observed!r}/{expected_value!r}"
            )
    elif observed != expected_value:
        raise SystemExit(
            f"I-40 config drift for {key}: {observed!r}/{expected_value!r}"
        )
for forbidden in (
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "additional_target",
    "resume_from_checkpoint",
):
    if config.get(forbidden) not in (None, False, ""):
        raise SystemExit(f"I-40 forbids a fresh/resume override: {forbidden}")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i40_i35_direct_user_continue_v1"
expected_entry = {
    "file_name": str(data_path),
    "formatting": "alpaca",
    "columns": {
        "prompt": "input",
        "response": "output",
        "history": "history",
        "system": "instruction",
    },
}
if registry != {key: expected_entry}:
    raise SystemExit("I-40 dataset registry drifted")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
if (
    audit.get("schema_version") != "i40-i35-direct-user-continue-r112-v1"
    or audit.get("status") != "FORMAL_DATA_FROZEN_TRAINING_AUTHORIZED"
    or audit.get("formal_training_generated") is not True
    or audit.get("seed") != 19260840
):
    raise SystemExit("I-40 audit does not authorize this run")
if audit.get("intersections") != {
    "user_retention_exact_prompt": 0,
    "user_retention_mode_prompt": 0,
    "forbidden_E_rows": 0,
    "third_party_rows": 0,
}:
    raise SystemExit("I-40 audit intersection contract drifted")

mix = audit.get("mix") or {}
routes = mix.get("routes") or {}
if (
    mix.get("total_rows") != 8240
    or mix.get("optimizer_steps_batch1_acc4") != 2060
    or (routes.get("user_ce") or {}).get("rows") != 5500
    or (routes.get("retention_kl") or {}).get("rows") != 2740
    or (routes.get("retention_kl") or {}).get("old_i35_objective_reused")
    is not False
):
    raise SystemExit("I-40 audit mix/loss contract drifted")
if (routes.get("user_ce") or {}).get("by_task") != {
    "action": 4000,
    "topic": 1500,
}:
    raise SystemExit("I-40 user task mix drifted")
if (routes.get("retention_kl") or {}).get("by_task") != {
    "material_desc2sid": 1370,
    "action": 207,
    "topic": 206,
    "rec_video": 206,
    "rec_prod": 207,
    "rec_ad": 206,
    "rec_living": 207,
    "world": 131,
}:
    raise SystemExit("I-40 retention task mix drifted")

expected_outputs = {
    "training_data": (
        data_path,
        8240,
        "483a4bb2f98d41497600d078032634d4f36fe2970a53d98b4a7fccc488910c18",
    ),
    "sidecar": (
        sidecar_path,
        8240,
        "e9bc129cd834bff161247985cc5430cf46872006cbae4a86fd37c3666b60acb2",
    ),
}
outputs = audit.get("outputs") or {}
for name, (expected_path, rows, digest) in expected_outputs.items():
    entry = outputs.get(name) or {}
    observed_path = Path(str(entry.get("path") or ""))
    if not observed_path.is_absolute():
        observed_path = root / observed_path
    if (
        observed_path.resolve() != expected_path.resolve()
        or entry.get("rows") != rows
        or entry.get("sha256") != digest
    ):
        raise SystemExit(f"I-40 audit output drifted: {name}")

selection = audit.get("selection") or {}
if (
    selection.get("unique_normalized_rows") != 8215
    or selection.get("duplicate_row_exposures") != 25
    or selection.get("maximum_normalized_row_exposure") != 2
):
    raise SystemExit("I-40 inherited duplicate exposure contract drifted")
tokenization = audit.get("tokenization") or {}
if (
    tokenization.get("rows") != 8240
    or tokenization.get("unique_routing_token_hashes") != 8215
    or tokenization.get("maximum_qwen3_nothink_tokens") != 8864
    or tokenization.get("cutoff") != 16384
):
    raise SystemExit("I-40 tokenization contract drifted")
PY

if [[ "$mode" == "--dry-run" ]]; then
  echo "I-40 dry-run PASS: direct existing r112 continuation; one GPU $gpu_id; frozen model/config/data/sidecar/audit locked; output absent; GPU ${gpu_memory_used} MiB/${gpu_utilization}%"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || {
  echo "I-40 requires W&B entity $expected_wandb_entity" >&2
  exit 2
}
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || {
  echo "I-40 requires W&B project $expected_wandb_project" >&2
  exit 2
}
[[ "${WANDB_MODE:-}" == online ]] || {
  echo "I-40 requires explicit WANDB_MODE=online" >&2
  exit 2
}
case "${WANDB_DISABLED:-false}" in
  1|true|TRUE|yes|YES)
    echo "I-40 refuses WANDB_DISABLED" >&2
    exit 2
    ;;
esac
[[ -z "${WANDB_RUN_ID:-}" ]] || {
  echo "I-40 refuses a preselected W&B run id" >&2
  exit 2
}
[[ "${WANDB_RESUME:-never}" == never ]] || {
  echo "I-40 refuses W&B resume" >&2
  exit 2
}
export WANDB_MODE=online
export WANDB_DISABLED=false
export WANDB_RESUME=never

grep -Fq 'data_i40_i35_direct_user_continue_v1.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-40 formal data is not registered" >&2
  exit 2
}
grep -Fq 'data_i40_i35_direct_user_continue_v1_sidecar.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-40 sidecar is not registered" >&2
  exit 2
}
grep -Fq 'i40_i35_direct_user_continue_r112_v1' "$root/docs/EXPERIMENT_INDEX.md" || {
  echo "I-40 experiment is not registered" >&2
  exit 2
}
grep -Fq 'I-40唯一单卡训练' "$root/docs/TODO.md" || {
  echo "I-40 training is not registered in TODO" >&2
  exit 2
}

export I40_PARENT_ADAPTER="$parent"
export I40_TRAINING_DATA="$data"
export I40_SIDECAR="$sidecar"
export I40_AUDIT="$audit"
export I40_OUTPUT_DIR="$output"

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$python_bin" - <<'PY'
import os

import wandb

if os.environ.get("WANDB_MODE") != "online":
    raise SystemExit("I-40 W&B mode is not online")
viewer = wandb.Api(timeout=15).viewer
if not viewer:
    raise SystemExit("I-40 W&B authentication failed")
print(f"W&B authenticated as {viewer}", flush=True)
PY

verify_frozen_assets
[[ ! -e "$output" ]] || {
  echo "I-40 output appeared during preflight: $output" >&2
  exit 2
}
[[ ! -e "$train_log" ]] || {
  echo "I-40 log appeared during preflight: $train_log" >&2
  exit 2
}

mkdir -p "$(dirname "$train_log")"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  I40_PARENT_ADAPTER="$parent" \
  I40_TRAINING_DATA="$data" \
  I40_SIDECAR="$sidecar" \
  I40_AUDIT="$audit" \
  I40_OUTPUT_DIR="$output" \
  "$python_bin" "$trainer" "$config" \
  > >(tee "$train_log") 2>&1
