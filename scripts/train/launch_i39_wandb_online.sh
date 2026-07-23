#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> WANDB_MODE=online $0 <single-gpu-id> [--dry-run]" >&2
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
config=$root/configs/active/i39_i35_userab_firstdiv_retkl_r8_v1.yaml
trainer=$root/scripts/train/train_i39_ab_firstdiv_retkl.py
dataset_registry=$root/configs/datasets/i39_i35_userab_firstdiv_retkl_v1/dataset_info.json
data=$root/assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1.jsonl
sidecar=$root/assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1_sidecar.jsonl
gate=$root/assets/evaluation/holdout/data_i39_userab_firstdiv_gate_v1.jsonl
gate_config=$root/configs/evaluation/i39_i35_ab_firstdiv_material_checkpoint_gate_v1.json
audit=$root/logs/data/i39_i35_userab_firstdiv_retkl_v1_audit.json
parent=$root/submissions/i35_r96_video_boundary_retkl_r112_step548_platform
output=$root/checkpoints/i39_i35_userab_firstdiv_retkl_r8_v1
train_log=$root/logs/train/i39_i35_userab_firstdiv_retkl_r8_v1.log

expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00
expected_parent_config=4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996
expected_trainer=a979a84391d8994bdffa5605de34b4234b633bd73f50bdeb9719c145ceddb343
expected_config=ebe0227dcb6b395e68a1f7424fa66fcad8dbb12cab0936e0d60e80a3e407605d
expected_registry=6cdb35254640e8836c18e2f946ee48d211b04ed43730d3791345bf02bb143221
expected_data=0a5cb2e55fff2c21deb1452216e08eae104bb5ce7d7e68a599ac52908261a3e2
expected_sidecar=d9d74eb573523eb70d0593076542c49d047fdcd4eb616a88d777476e5532bd14
expected_gate=293fc361295db56196acc035bd639d63e426168b93ea36f2d21c3890c2a34d40
expected_gate_config=89737747e68161a68607d06fdd8767cbcae9a31373252bdc7e6a764df32ed504
expected_audit=c52921cacaa42aac569b5be72eb8ad31112193b48afed892141380419e75a718
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ "$gpu_id" =~ ^[0-9]+$ ]] || {
  echo "I-39 requires exactly one numeric GPU id" >&2
  exit 2
}
[[ -x "$python_bin" ]] || {
  echo "missing executable Python: $python_bin" >&2
  exit 2
}

if ! gpu_state=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits); then
  echo "I-39 could not read GPU $gpu_id state" >&2
  exit 2
fi
gpu_memory_used=$(cut -d, -f1 <<<"$gpu_state" | tr -d '[:space:]')
gpu_utilization=$(cut -d, -f2 <<<"$gpu_state" | tr -d '[:space:]')
[[ "$gpu_memory_used" =~ ^[0-9]+$ && "$gpu_utilization" =~ ^[0-9]+$ ]] || {
  echo "I-39 could not parse GPU $gpu_id state" >&2
  exit 2
}
if [[ "$mode" != "--dry-run" ]] && (( gpu_memory_used > 8192 || gpu_utilization > 50 )); then
  echo "I-39 GPU $gpu_id is too busy: ${gpu_memory_used} MiB, ${gpu_utilization}%" >&2
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
  check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "I-35 step548 parent adapter"
  check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "I-35 step548 parent config"
  check_sha256 "$trainer" "$expected_trainer" "I-39 trainer"
  check_sha256 "$config" "$expected_config" "I-39 config"
  check_sha256 "$dataset_registry" "$expected_registry" "I-39 dataset registry"
  check_sha256 "$data" "$expected_data" "I-39 formal data"
  check_sha256 "$sidecar" "$expected_sidecar" "I-39 full routing sidecar"
  check_sha256 "$gate" "$expected_gate" "I-39 frozen AB gate"
  check_sha256 "$gate_config" "$expected_gate_config" "I-39 frozen gate config"
  check_sha256 "$audit" "$expected_audit" "I-39 formal audit"
}

verify_frozen_assets
[[ ! -e "$output" ]] || {
  echo "I-39 refuses to overwrite $output" >&2
  exit 2
}
[[ ! -e "$train_log" ]] || {
  echo "I-39 refuses to overwrite $train_log" >&2
  exit 2
}

"$python_bin" - "$root" "$config" "$dataset_registry" "$data" "$sidecar" "$gate" "$audit" <<'PY'
import json
import math
import sys
from pathlib import Path

import yaml

root, config_path, registry_path, data_path, sidecar_path, gate_path, audit_path = map(
    Path, sys.argv[1:]
)
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected = {
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(
        root / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"
    ),
    "create_new_adapter": True,
    "trust_remote_code": True,
    "flash_attn": "fa2",
    "enable_liger_kernel": False,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_rank": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "dataset": "data_i39_i35_userab_firstdiv_retkl_v1",
    "dataset_dir": str(root / "configs/datasets/i39_i35_userab_firstdiv_retkl_v1"),
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "generation_max_length": 16384,
    "packing": False,
    "overwrite_cache": True,
    "preprocessing_num_workers": 16,
    "dataloader_num_workers": 8,
    "val_size": 0,
    "eval_strategy": "no",
    "output_dir": str(root / "checkpoints/i39_i35_userab_firstdiv_retkl_r8_v1"),
    "logging_steps": 8,
    "save_strategy": "no",
    "save_only_model": True,
    "plot_loss": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "run_name": "i39_i35_userab_firstdiv_retkl_r8_v1",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-6,
    "max_steps": 640,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.001,
    "bf16": True,
    "seed": 19260839,
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
                f"I-39 config drift for {key}: {observed!r}/{expected_value!r}"
            )
    elif observed != expected_value:
        raise SystemExit(
            f"I-39 config drift for {key}: {observed!r}/{expected_value!r}"
        )
if config.get("resume_from_checkpoint"):
    raise SystemExit("I-39 cannot resume from a checkpoint")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i39_i35_userab_firstdiv_retkl_v1"
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
    raise SystemExit("I-39 dataset registry drifted")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("schema_version") != "i39-i35-userab-firstdiv-retkl-v1":
    raise SystemExit("I-39 audit schema drifted")
if audit.get("formal_training_generated") is not True or audit.get("seed") != 19260839:
    raise SystemExit("I-39 audit does not authorize this formal run")
if audit.get("asset_class") != "D(O3; D-I36 user overlap; D-I12 retention; M-I35 Beam64 filter)":
    raise SystemExit("I-39 audit provenance drifted")

mix = audit.get("mix") or {}
expected_routes = {
    "material_firstdiv": {
        "rows": 512,
        "ratio": 0.2,
        "by_objective": {
            "a_firstdiv": 128,
            "b_firstdiv": 128,
            "c_firstdiv": 192,
            "full_anchor": 64,
        },
    },
    "user_micro_ce": {
        "rows": 128,
        "ratio": 0.05,
        "by_task": {"action": 96, "topic": 32},
    },
    "retention_kl": {
        "rows": 1920,
        "ratio": 0.75,
        "by_task": {
            "action": 256,
            "material_desc2sid": 128,
            "material_sid2desc": 128,
            "rec_ad": 240,
            "rec_living": 240,
            "rec_prod": 240,
            "rec_video": 240,
            "topic": 256,
            "world": 192,
        },
    },
}
if (
    mix.get("total_rows") != 2560
    or mix.get("optimizer_steps_batch1_acc4") != 640
    or mix.get("routes") != expected_routes
):
    raise SystemExit("I-39 audit mix contract drifted")

expected_outputs = {
    "training_data": (data_path, 2560, "0a5cb2e55fff2c21deb1452216e08eae104bb5ce7d7e68a599ac52908261a3e2"),
    "sidecar": (sidecar_path, 2560, "d9d74eb573523eb70d0593076542c49d047fdcd4eb616a88d777476e5532bd14"),
    "gate": (gate_path, 313, "293fc361295db56196acc035bd639d63e426168b93ea36f2d21c3890c2a34d40"),
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
        raise SystemExit(f"I-39 audit output drifted: {name}")

sidecar_contract = audit.get("sidecar_contract") or {}
if (
    sidecar_contract.get("rows") != 2560
    or sidecar_contract.get("routes")
    != {"material_firstdiv": 512, "retention_kl": 1920, "user_micro_ce": 128}
    or sidecar_contract.get("objectives")
    != {
        "a_firstdiv": 128,
        "b_firstdiv": 128,
        "c_firstdiv": 192,
        "full_anchor": 64,
    }
    or sidecar_contract.get("parent_adapter_sha256")
    != "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
    or sidecar_contract.get("model_candidates_are_negatives_only") is not True
    or sidecar_contract.get("single_parent_compatibility_teacher_score_removed")
    is not True
):
    raise SystemExit("I-39 sidecar contract drifted")

leakage = audit.get("leakage") or {}
zero_leakage = (
    "exact_cross_route_prompt_overlap",
    "gate_all_route_ab_overlap",
    "gate_all_route_full_sid_overlap",
    "gate_formal_exact_prompt_overlap",
    "gate_formal_mode_prompt_overlap",
    "gate_material_ab_overlap",
    "mode_cross_route_prompt_overlap",
)
if any(leakage.get(name) != 0 for name in zero_leakage):
    raise SystemExit("I-39 leakage contract drifted")
if leakage.get("formal_material_unique_ab") != 480 or leakage.get("gate_unique_ab") != 256:
    raise SystemExit("I-39 AB coverage contract drifted")

tokenizer = audit.get("tokenizer") or {}
if (
    tokenizer.get("template") != "qwen3_nothink"
    or tokenizer.get("cutoff") != 16384
    or tokenizer.get("sidecar_rows") != 2560
    or not isinstance(tokenizer.get("max_total_tokens"), int)
    or tokenizer["max_total_tokens"] > 16384
):
    raise SystemExit("I-39 tokenizer contract drifted")
PY

if [[ "$mode" == "--dry-run" ]]; then
  echo "I-39 dry-run PASS: exactly one GPU $gpu_id; frozen model/config/data/sidecar/gate/audit contracts locked; output absent; GPU ${gpu_memory_used} MiB/${gpu_utilization}%"
  echo "command=CUDA_VISIBLE_DEVICES=$gpu_id $python_bin $trainer $config"
  exit 0
fi

[[ "${WANDB_ENTITY:-}" == "$expected_wandb_entity" ]] || {
  echo "I-39 requires W&B entity $expected_wandb_entity" >&2
  exit 2
}
[[ "${WANDB_PROJECT:-}" == "$expected_wandb_project" ]] || {
  echo "I-39 requires W&B project $expected_wandb_project" >&2
  exit 2
}
[[ "${WANDB_MODE:-}" == online ]] || {
  echo "I-39 requires explicit WANDB_MODE=online" >&2
  exit 2
}
case "${WANDB_DISABLED:-false}" in
  1|true|TRUE|yes|YES)
    echo "I-39 refuses WANDB_DISABLED" >&2
    exit 2
    ;;
esac
[[ -z "${WANDB_RUN_ID:-}" ]] || {
  echo "I-39 refuses a preselected W&B run id" >&2
  exit 2
}
[[ "${WANDB_RESUME:-never}" == never ]] || {
  echo "I-39 refuses W&B resume" >&2
  exit 2
}
export WANDB_MODE=online
export WANDB_DISABLED=false
export WANDB_RESUME=never

grep -Fq 'data_i39_i35_userab_firstdiv_retkl_v1.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-39 formal data is not registered" >&2
  exit 2
}
grep -Fq 'data_i39_i35_userab_firstdiv_retkl_v1_sidecar.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-39 sidecar is not registered" >&2
  exit 2
}
grep -Fq 'data_i39_userab_firstdiv_gate_v1.jsonl' "$root/docs/reference/ASSETS.md" || {
  echo "I-39 frozen gate is not registered" >&2
  exit 2
}
grep -Fq 'i39_i35_userab_firstdiv_retkl_r8_v1' "$root/docs/EXPERIMENT_INDEX.md" || {
  echo "I-39 experiment is not registered" >&2
  exit 2
}
grep -Fq 'I-39' "$root/docs/TODO.md" || {
  echo "I-39 training is not registered in TODO" >&2
  exit 2
}

export I39_PARENT_ADAPTER="$parent"
export I39_TRAINING_DATA="$data"
export I39_SIDECAR="$sidecar"
export I39_AUDIT="$audit"
export I39_OUTPUT_DIR="$output"

"$python_bin" "$trainer" --self-test
"$python_bin" "$trainer" --data-preflight
"$python_bin" - <<'PY'
import os

import wandb

if os.environ.get("WANDB_MODE") != "online":
    raise SystemExit("I-39 W&B mode is not online")
viewer = wandb.Api(timeout=15).viewer
if not viewer:
    raise SystemExit("I-39 W&B authentication failed")
print(f"W&B authenticated as {viewer}", flush=True)
PY

verify_frozen_assets
[[ ! -e "$output" ]] || {
  echo "I-39 output appeared during preflight: $output" >&2
  exit 2
}
[[ ! -e "$train_log" ]] || {
  echo "I-39 log appeared during preflight: $train_log" >&2
  exit 2
}

mkdir -p "$(dirname "$train_log")"
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  I39_PARENT_ADAPTER="$parent" \
  I39_TRAINING_DATA="$data" \
  I39_SIDECAR="$sidecar" \
  I39_AUDIT="$audit" \
  I39_OUTPUT_DIR="$output" \
  "$python_bin" "$trainer" "$config" \
  > >(tee "$train_log") 2>&1
