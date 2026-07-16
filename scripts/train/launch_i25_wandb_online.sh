#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <single-gpu-id-or-uuid> <configs/active/i23_actionres_r16_ansretkl_ep1.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
trainer=$root/scripts/train/train_i23_actionres_retkl.py
expected_config=$root/configs/active/i23_actionres_r16_ansretkl_ep1.yaml
gate=$root/configs/evaluation/i23_actionres_r16_checkpoint_gate.json
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
parent=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
data=$root/assets/derived/processed/data_user_residual_retention_v1.jsonl
dataset_registry=$root/configs/datasets/user_residual_retention_v1/dataset_info.json
builder=$root/scripts/data/build_user_residual_retention_v1.py
audit=$root/logs/data/user_residual_retention_v1_audit.json
output=$root/checkpoints/i23_actionres_r16_ansretkl_ep1

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || {
  echo "I-25 formal training must expose exactly one GPU" >&2
  exit 2
}
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$expected_config" ]] || {
  echo "I-25 launcher only accepts $expected_config" >&2
  exit 2
}
if grep -Eq '^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$config"; then
  echo "refusing a disabled training config: $config" >&2
  exit 2
fi

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online formal run: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi
export WANDB_MODE=online

expected_base=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_parent=0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8
expected_parent_config=b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7
expected_data=bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0
expected_registry=06b04bc3e29d5be783acd0867b831b87152ddbacb924cb80b18715e2c5c51608
expected_builder=454cf7d3f4cd1380886536406d0bd1730670041279431857d7b58ff2b66d6e43
expected_audit=0437b70ba1b323707560aaae5fdaf1167b732375328fd84673fb99ba3904e054
expected_trainer=0071a0885cff480222c5d905f68ccf22c3f90a54d7967df55b08b6d951907c02
expected_config_hash=da46f0b153a06244a4b8015c64055dbcab0e44788138db434bdff5f5605c5dbd
expected_gate=53b5b375630ba2255dada2fd04d8fc4cd1b694e3afde10079ef492135dcef212

check_hash() {
  local path=$1 expected=$2 label=$3
  [[ -f "$path" ]] || { echo "missing $label: $path" >&2; exit 2; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    echo "$label checksum drifted: $path" >&2
    exit 2
  }
}

check_hash "$base_config" "$expected_base" "O6 base config"
check_hash "$parent/adapter_model.safetensors" "$expected_parent" "I-23 parent adapter"
check_hash "$parent/adapter_config.json" "$expected_parent_config" "I-23 parent adapter config"
check_hash "$data" "$expected_data" "I-25 registered training data"
check_hash "$dataset_registry" "$expected_registry" "I-25 dataset registry"
check_hash "$builder" "$expected_builder" "I-25 upstream builder"
check_hash "$audit" "$expected_audit" "I-25 upstream audit"
check_hash "$trainer" "$expected_trainer" "I-25 trainer"
check_hash "$config" "$expected_config_hash" "I-25 config"
check_hash "$gate" "$expected_gate" "I-25 preregistered gate"
[[ "$(wc -l <"$data")" -eq 6106 ]] || {
  echo "I-25 training data row count is not 6106" >&2
  exit 2
}
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "refusing to overwrite I-25 output directory: $output" >&2
  exit 2
}

"$python_bin" - "$config" "$gate" "$root" <<'PY'
import sys
import hashlib
from pathlib import Path

import yaml

path = Path(sys.argv[1])
gate_path = Path(sys.argv[2])
root = Path(sys.argv[3])
config = yaml.safe_load(path.read_text(encoding="utf-8"))
expected = {
    "create_new_adapter": True,
    "lora_rank": 16,
    "lora_alpha": 16,
    "dataset": "data_user_residual_retention_v1",
    "packing": False,
    "report_to": "wandb",
    "save_strategy": "steps",
    "save_steps": 250,
    "save_total_limit": 6,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-5,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 46,
    "seed": 19260821,
}
drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if drift:
    raise SystemExit(f"I-25 locked config fields drifted: {drift}")
if "max_steps" in config:
    raise SystemExit("I-25 must cover one full epoch; max_steps is forbidden")
if "warmup_ratio" in config:
    raise SystemExit("I-25 locks explicit warmup_steps=46; warmup_ratio is forbidden")

import json

gate = json.loads(gate_path.read_text(encoding="utf-8"))
if gate.get("status") != "PREREGISTERED_BEFORE_I25_FORMAL_LAUNCH":
    raise SystemExit(f"I-25 gate is not active: {gate.get('status')!r}")

def pointer(document, value):
    node = document
    for component in value.strip("/").split("/"):
        node = node[int(component)] if isinstance(node, list) else node[component]
    return node

for required in gate["prelaunch_lock"]["required_json_pointers_before_launch"]:
    try:
        value = pointer(gate, required)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise SystemExit(f"I-25 gate missing required pointer {required}: {error}")
    if value in (None, "", "PENDING"):
        raise SystemExit(f"I-25 gate pointer is unresolved: {required}={value!r}")

artifacts = gate["formal_training_artifacts"]
artifact_expected = {
    "config": ("configs/active/i23_actionres_r16_ansretkl_ep1.yaml", "da46f0b153a06244a4b8015c64055dbcab0e44788138db434bdff5f5605c5dbd"),
    "trainer": ("scripts/train/train_i23_actionres_retkl.py", "0071a0885cff480222c5d905f68ccf22c3f90a54d7967df55b08b6d951907c02"),
}
for name, (relative, expected_hash) in artifact_expected.items():
    record = artifacts[name]
    if (record.get("path"), record.get("sha256")) != (relative, expected_hash):
        raise SystemExit(f"I-25 gate {name} lock drifted: {record}")
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected_hash:
        raise SystemExit(f"I-25 locked {name} hash mismatch: {actual}")
launcher_paths = {
    "online_launcher": "scripts/train/launch_i25_wandb_online.sh",
    "detached_launcher": "scripts/train/launch_i25_wandb_detached.sh",
}
for name, expected_path in launcher_paths.items():
    if artifacts[name] != {"path": expected_path}:
        raise SystemExit(f"I-25 gate launcher path drifted: {name}={artifacts[name]}")

invariants = gate["training_invariants"]
invariant_expected = {
    "single_gpu": True,
    "wandb_enabled": True,
    "wandb_mode": "online",
    "epochs": 1,
    "optimizer_steps": 1527,
    "effective_batch_size": 4,
    "residual_rank": 16,
    "residual_alpha": 16,
    "parent_is_frozen": True,
    "reference_is_frozen": True,
    "learning_rate": 5.0e-5,
    "lr_schedule": "cosine",
    "warmup_steps": 46,
    "action_ce_weight": 1.0,
    "action_kl_weight": 0.05,
    "retention_kl_weight": 2.0,
    "seed": 19260821,
}
invariant_drift = {
    key: (invariants.get(key), value)
    for key, value in invariant_expected.items()
    if invariants.get(key) != value
}
if invariant_drift:
    raise SystemExit(f"I-25 gate training invariants drifted: {invariant_drift}")
expected_kl = (
    "parent-to-policy forward KL on every contiguous supervised assistant-response "
    "target position, from the first response token through the final formatter-"
    "supervised token; no KL position sampling; action gold CE remains restricted "
    "to the post-</think> answer body"
)
if invariants.get("kl_position_protocol") != expected_kl:
    raise SystemExit("I-25 gate KL-position protocol drifted")

axis = gate["checkpoint_axis"]
steps = [250, 500, 750, 1000, 1250, 1527]
expected_root = "checkpoints/i23_actionres_r16_ansretkl_ep1"
expected_paths = {str(step): f"{expected_root}/checkpoint-{step}" for step in steps}
if axis.get("output_root") != expected_root:
    raise SystemExit(f"I-25 gate output root drifted: {axis.get('output_root')}")
if axis.get("candidates_in_ascending_order") != steps:
    raise SystemExit("I-25 gate checkpoint candidate order drifted")
if axis.get("expected_paths") != expected_paths:
    raise SystemExit("I-25 gate expected checkpoint paths drifted")
print("[i25] locked YAML fields PASS")
print("[i25] active preregistered gate fields PASS")
PY

printf '[i25] preregistered_gate_sha256=%s\n' "$expected_gate"

"$python_bin" "$trainer" --self-test
"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

set +e
env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  I25_ACTION_KL=0.05 \
  I25_RETENTION_KL=2.0 \
  I25_TERMINAL_MULTIPLIER=2.0 \
  I25_LOGIT_CHUNK=16 \
  "$python_bin" "$trainer" "$config"
train_rc=$?
set -e
if [[ "$train_rc" -ne 0 ]]; then
  echo "I-25 trainer exited with status $train_rc; final checkpoint will not be synthesized" >&2
  exit "$train_rc"
fi

"$python_bin" - "$output/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing completed trainer state: {path}")
state = json.loads(path.read_text(encoding="utf-8"))
if state.get("global_step") != 1527:
    raise SystemExit(f"I-25 did not complete step 1527: {state.get('global_step')}")
print("[i25] completed optimizer step 1527 PASS")
PY

for step in 250 500 750 1000 1250 1500; do
  checkpoint=$output/checkpoint-$step
  [[ -f "$checkpoint/adapter_model.safetensors" && -f "$checkpoint/adapter_config.json" ]] || {
    echo "missing I-25 scheduled adapter checkpoint: $checkpoint" >&2
    exit 1
  }
done
for path in "$output/adapter_model.safetensors" "$output/adapter_config.json"; do
  [[ -f "$path" ]] || { echo "missing completed I-25 root artifact: $path" >&2; exit 1; }
done
forbidden_state=$(find "$output" -type f \( \
  -name 'optimizer.pt' -o -name 'scheduler.pt' -o -name 'scaler.pt' -o \
  -name 'rng_state*.pth' \) -print -quit)
[[ -z "$forbidden_state" ]] || {
  echo "I-25 checkpoint retained forbidden training state: $forbidden_state" >&2
  exit 1
}

final_checkpoint=$output/checkpoint-1527
temporary_checkpoint=$output/.checkpoint-1527.tmp.$$
[[ ! -e "$final_checkpoint" && ! -L "$final_checkpoint" ]] || {
  echo "refusing to overwrite explicit I-25 final checkpoint: $final_checkpoint" >&2
  exit 1
}
[[ ! -e "$temporary_checkpoint" && ! -L "$temporary_checkpoint" ]] || {
  echo "refusing to reuse I-25 final-checkpoint staging path: $temporary_checkpoint" >&2
  exit 1
}
cleanup_final_stage() {
  if [[ -d "$temporary_checkpoint" ]]; then
    rm -f -- "$temporary_checkpoint/adapter_model.safetensors" "$temporary_checkpoint/adapter_config.json"
    rmdir -- "$temporary_checkpoint" 2>/dev/null || true
  fi
}
trap cleanup_final_stage EXIT
mkdir -m 0755 -- "$temporary_checkpoint"
cp -- "$output/adapter_model.safetensors" "$temporary_checkpoint/adapter_model.safetensors"
cp -- "$output/adapter_config.json" "$temporary_checkpoint/adapter_config.json"
[[ "$(find "$temporary_checkpoint" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 2 ]] || {
  echo "I-25 final checkpoint staging directory must contain exactly two files" >&2
  exit 1
}
cmp -s "$output/adapter_model.safetensors" "$temporary_checkpoint/adapter_model.safetensors"
cmp -s "$output/adapter_config.json" "$temporary_checkpoint/adapter_config.json"
mv -T -- "$temporary_checkpoint" "$final_checkpoint"
trap - EXIT

cmp -s "$output/adapter_model.safetensors" "$final_checkpoint/adapter_model.safetensors"
cmp -s "$output/adapter_config.json" "$final_checkpoint/adapter_config.json"
[[ "$(find "$final_checkpoint" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 2 ]] || {
  echo "I-25 checkpoint-1527 must contain only adapter/config" >&2
  exit 1
}
printf '[i25] checkpoint-1527 adapter_sha256=%s config_sha256=%s root_byte_identical=PASS\n' \
  "$(sha256sum "$final_checkpoint/adapter_model.safetensors" | awk '{print $1}')" \
  "$(sha256sum "$final_checkpoint/adapter_config.json" | awk '{print $1}')"
