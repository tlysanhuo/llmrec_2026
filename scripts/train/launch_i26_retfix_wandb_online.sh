#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <single-gpu-id-or-uuid> <configs/active/i23_actionres_r16_ansretkl_ep1_retfix.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
trainer=$root/scripts/train/train_i23_actionres_retkl.py
expected_config=$root/configs/active/i23_actionres_r16_ansretkl_ep1_retfix.yaml
gate=$root/configs/evaluation/i23_actionres_r16_ansretkl_ep1_retfix_checkpoint_gate.json
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
parent=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
data=$root/assets/derived/processed/data_user_residual_retention_v1.jsonl
dataset_registry=$root/configs/datasets/user_residual_retention_v1/dataset_info.json
builder=$root/scripts/data/build_user_residual_retention_v1.py
audit=$root/logs/data/user_residual_retention_v1_audit.json
output=$root/checkpoints/i23_actionres_r16_ansretkl_ep1_retfix

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || {
  echo "I-26 formal training must expose exactly one GPU" >&2
  exit 2
}
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$expected_config" ]] || {
  echo "I-26 launcher only accepts $expected_config" >&2
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
expected_config_hash=c9a0c156e299384bdf63571bb03fb0821fc1fcfdde10eb82b965e9d4823e3194
expected_gate=2e3d3730b4cff13b0c0e0c99f7b80fb6c4bc4d3424f3f59bfb95a3be645f8ed1

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
check_hash "$data" "$expected_data" "I-26 registered training data"
check_hash "$dataset_registry" "$expected_registry" "I-26 dataset registry"
check_hash "$builder" "$expected_builder" "I-26 upstream builder"
check_hash "$audit" "$expected_audit" "I-26 upstream audit"
check_hash "$trainer" "$expected_trainer" "I-26 trainer"
check_hash "$config" "$expected_config_hash" "I-26 config"
check_hash "$gate" "$expected_gate" "I-26 preregistered gate"
[[ "$(wc -l <"$data")" -eq 6106 ]] || {
  echo "I-26 training data row count is not 6106" >&2
  exit 2
}
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "refusing to overwrite I-26 output directory: $output" >&2
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
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(root / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"),
    "create_new_adapter": True,
    "lora_rank": 16,
    "lora_alpha": 16,
    "dataset": "data_user_residual_retention_v1",
    "packing": False,
    "report_to": "wandb",
    "save_strategy": "steps",
    "save_steps": 250,
    "save_total_limit": 7,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "output_dir": str(root / "checkpoints/i23_actionres_r16_ansretkl_ep1_retfix"),
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5.0e-5,
    "num_train_epochs": 1,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 46,
    "seed": 19260821,
    "run_name": "i23_actionres_r16_ansretkl_ep1_retfix",
}
drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if drift:
    raise SystemExit(f"I-26 locked config fields drifted: {drift}")
if "max_steps" in config:
    raise SystemExit("I-26 must cover one full epoch; max_steps is forbidden")
if "warmup_ratio" in config:
    raise SystemExit("I-26 locks explicit warmup_steps=46; warmup_ratio is forbidden")
if "resume_from_checkpoint" in config:
    raise SystemExit("I-26 is a clean O6+I23 start; resume_from_checkpoint is forbidden")

import json

gate = json.loads(gate_path.read_text(encoding="utf-8"))
if gate.get("status") != "PREREGISTERED_BEFORE_I26_FORMAL_LAUNCH":
    raise SystemExit(f"I-26 gate is not active: {gate.get('status')!r}")

def pointer(document, value):
    node = document
    for component in value.strip("/").split("/"):
        node = node[int(component)] if isinstance(node, list) else node[component]
    return node

for required in gate["prelaunch_lock"]["required_json_pointers_before_launch"]:
    try:
        value = pointer(gate, required)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise SystemExit(f"I-26 gate missing required pointer {required}: {error}")
    if value in (None, "", "PENDING"):
        raise SystemExit(f"I-26 gate pointer is unresolved: {required}={value!r}")

artifacts = gate["formal_training_artifacts"]
artifact_expected = {
    "config": ("configs/active/i23_actionres_r16_ansretkl_ep1_retfix.yaml", "c9a0c156e299384bdf63571bb03fb0821fc1fcfdde10eb82b965e9d4823e3194"),
    "trainer": ("scripts/train/train_i23_actionres_retkl.py", "0071a0885cff480222c5d905f68ccf22c3f90a54d7967df55b08b6d951907c02"),
}
for name, (relative, expected_hash) in artifact_expected.items():
    record = artifacts[name]
    if (record.get("path"), record.get("sha256")) != (relative, expected_hash):
        raise SystemExit(f"I-26 gate {name} lock drifted: {record}")
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected_hash:
        raise SystemExit(f"I-26 locked {name} hash mismatch: {actual}")
launcher_paths = {
    "online_launcher": "scripts/train/launch_i26_retfix_wandb_online.sh",
    "detached_launcher": "scripts/train/launch_i26_retfix_wandb_detached.sh",
}
for name, expected_path in launcher_paths.items():
    if artifacts[name] != {"path": expected_path}:
        raise SystemExit(f"I-26 gate launcher path drifted: {name}={artifacts[name]}")

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
    "save_total_limit": 7,
    "seed": 19260821,
}
invariant_drift = {
    key: (invariants.get(key), value)
    for key, value in invariant_expected.items()
    if invariants.get(key) != value
}
if invariant_drift:
    raise SystemExit(f"I-26 gate training invariants drifted: {invariant_drift}")
expected_kl = (
    "parent-to-policy forward KL on every contiguous supervised assistant-response "
    "target position, from the first response token through the final formatter-"
    "supervised token; no KL position sampling; action gold CE remains restricted "
    "to the post-</think> answer body"
)
if invariants.get("kl_position_protocol") != expected_kl:
    raise SystemExit("I-26 gate KL-position protocol drifted")

axis = gate["checkpoint_axis"]
steps = [250, 500, 750, 1000, 1250, 1527]
retained_steps = [250, 500, 750, 1000, 1250, 1500, 1527]
expected_root = "checkpoints/i23_actionres_r16_ansretkl_ep1_retfix"
expected_paths = {str(step): f"{expected_root}/checkpoint-{step}" for step in steps}
expected_retained_paths = {
    str(step): f"{expected_root}/checkpoint-{step}" for step in retained_steps
}
if axis.get("output_root") != expected_root:
    raise SystemExit(f"I-26 gate output root drifted: {axis.get('output_root')}")
if axis.get("candidates_in_ascending_order") != steps:
    raise SystemExit("I-26 gate checkpoint candidate order drifted")
if axis.get("expected_paths") != expected_paths:
    raise SystemExit("I-26 gate expected checkpoint paths drifted")
if axis.get("retained_checkpoint_steps") != retained_steps:
    raise SystemExit("I-26 gate retained checkpoint order drifted")
if axis.get("expected_retained_paths") != expected_retained_paths:
    raise SystemExit("I-26 gate retained checkpoint paths drifted")
print("[i26] locked YAML fields PASS")
print("[i26] active preregistered gate fields PASS")
PY

printf '[i26] preregistered_gate_sha256=%s\n' "$expected_gate"

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
  echo "I-26 trainer exited with status $train_rc; refusing post-training acceptance" >&2
  exit "$train_rc"
fi

"$python_bin" - "$output" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
state_path = output / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(f"missing completed trainer state: {state_path}")
state = json.loads(state_path.read_text(encoding="utf-8"))
if state.get("global_step") != 1527:
    raise SystemExit(f"I-26 did not complete step 1527: {state.get('global_step')}")

expected_steps = [250, 500, 750, 1000, 1250, 1500, 1527]
observed = {}
for path in output.iterdir():
    if path.is_dir() and path.name.startswith("checkpoint-"):
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            raise SystemExit(f"unexpected checkpoint directory: {path}")
        observed[int(suffix)] = path
if sorted(observed) != expected_steps:
    raise SystemExit(
        f"I-26 retained checkpoint set mismatch: {sorted(observed)} != {expected_steps}"
    )

for step, checkpoint in observed.items():
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (checkpoint / name).is_file():
            raise SystemExit(f"checkpoint-{step} missing {name}")

terminal_state_path = observed[1527] / "trainer_state.json"
if not terminal_state_path.is_file():
    raise SystemExit(
        "checkpoint-1527 lacks Trainer state; refusing a synthesized terminal checkpoint"
    )
terminal_state = json.loads(terminal_state_path.read_text(encoding="utf-8"))
if terminal_state.get("global_step") != 1527:
    raise SystemExit(
        f"checkpoint-1527 Trainer state drifted: {terminal_state.get('global_step')}"
    )

forbidden_names = {"optimizer.pt", "scheduler.pt", "scaler.pt"}
for path in output.rglob("*"):
    if not path.is_file():
        continue
    if path.name in forbidden_names or (
        path.name.startswith("rng_state") and path.suffix == ".pth"
    ):
        raise SystemExit(f"I-26 retained forbidden training state: {path}")

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

terminal = observed[1527]
hashes = {}
for name in ("adapter_model.safetensors", "adapter_config.json"):
    root_path = output / name
    terminal_path = terminal / name
    if not root_path.is_file():
        raise SystemExit(f"missing completed I-26 root artifact: {root_path}")
    root_hash = digest(root_path)
    terminal_hash = digest(terminal_path)
    if root_hash != terminal_hash:
        raise SystemExit(
            f"I-26 root/checkpoint-1527 {name} mismatch: "
            f"{root_hash} != {terminal_hash}"
        )
    hashes[name] = root_hash

print("[i26] completed optimizer step 1527 PASS")
print(f"[i26] retained direct Trainer checkpoints PASS: {expected_steps}")
print("[i26] no optimizer/scheduler/scaler/RNG state PASS")
print(
    "[i26] checkpoint-1527 "
    f"adapter_sha256={hashes['adapter_model.safetensors']} "
    f"config_sha256={hashes['adapter_config.json']} "
    "root_byte_identical=PASS"
)
PY
