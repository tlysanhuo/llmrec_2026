#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <locked-gpu-uuid> <configs/active/i25_step250_deterministic_replay.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
llamafactory_root=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory
python_bin=${LLAMAFACTORY_PYTHON:-$llamafactory_root/.venv/bin/python3}
expected_gpu=GPU-d3c522d6-ed0f-2579-01cd-2d97da749980

formal_config=$root/configs/active/i23_actionres_r16_ansretkl_ep1.yaml
formal_trainer=$root/scripts/train/train_i23_actionres_retkl.py
recovery_config=$root/configs/active/i25_step250_deterministic_replay.yaml
recovery_trainer=$root/scripts/train/train_i25_step250_deterministic_replay.py
plan=$root/configs/evaluation/i25_step250_deterministic_recovery_plan.json
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
parent=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
data=$root/assets/derived/processed/data_user_residual_retention_v1.jsonl
dataset_registry=$root/configs/datasets/user_residual_retention_v1/dataset_info.json
builder=$root/scripts/data/build_user_residual_retention_v1.py
upstream_audit=$root/logs/data/user_residual_retention_v1_audit.json
formal_output=$root/checkpoints/i23_actionres_r16_ansretkl_ep1
formal_log=$root/logs/train/i23_actionres_r16_ansretkl_ep1.log
formal_status=$root/logs/train/i23_actionres_r16_ansretkl_ep1.exit_code
recovery_output=$root/checkpoints/i25_step250_deterministic_replay
receipt=$root/logs/train/i25_step250_deterministic_replay_receipt.json

expected_plan=94da5c04650ff71f9117502ae323bf1367aa67fc3837fc6ab613b74decaba1ec
expected_formal_config=da46f0b153a06244a4b8015c64055dbcab0e44788138db434bdff5f5605c5dbd
expected_formal_trainer=0071a0885cff480222c5d905f68ccf22c3f90a54d7967df55b08b6d951907c02
expected_recovery_config=da3345db8ee86a12f71a61374b8f36e12a5c2df510bbb4416beee4c567092346
expected_recovery_trainer=e6d3a0589eb2610b8db75912ff4dbe659fa364270b1b835157fcffa316b62cae
expected_step250=4af7296737209fb00df4908a09e21382ac6a9c987663b658e2f58e123fc928e9
expected_adapter_config=6c127a34b497e7a8672763b238a50801dd9a3e6cbb662c3c981f9a9b4a4e976a

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ "$gpu_id" == "$expected_gpu" ]] || {
  echo "I-25 recovery is locked to the original physical GPU: $expected_gpu" >&2
  exit 2
}
[[ -f "$config" ]] || { echo "missing recovery config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$recovery_config" ]] || {
  echo "I-25 recovery received the wrong config: $config" >&2
  exit 2
}
if grep -Eq '^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$config"; then
  echo "refusing a disabled recovery config: $config" >&2
  exit 2
fi

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online recovery: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi
export WANDB_MODE=online

check_hash() {
  local path=$1 expected=$2 label=$3
  [[ -f "$path" ]] || { echo "missing $label: $path" >&2; exit 2; }
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label checksum drifted: $actual != $expected ($path)" >&2
    exit 2
  }
}

check_hash "$plan" "$expected_plan" "I-25 recovery plan"
check_hash "$formal_config" "$expected_formal_config" "formal I-25 config"
check_hash "$formal_trainer" "$expected_formal_trainer" "formal I-25 trainer"
check_hash "$recovery_config" "$expected_recovery_config" "I-25 recovery config"
check_hash "$recovery_trainer" "$expected_recovery_trainer" "I-25 recovery trainer"
check_hash "$base_config" 5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4 "O6 base config"
check_hash "$parent/adapter_model.safetensors" 0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8 "I-23 parent adapter"
check_hash "$parent/adapter_config.json" b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7 "I-23 parent config"
check_hash "$data" bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0 "I-25 training data"
check_hash "$dataset_registry" 06b04bc3e29d5be783acd0867b831b87152ddbacb924cb80b18715e2c5c51608 "I-25 dataset registry"
check_hash "$builder" 454cf7d3f4cd1380886536406d0bd1730670041279431857d7b58ff2b66d6e43 "I-25 upstream builder"
check_hash "$upstream_audit" 0437b70ba1b323707560aaae5fdaf1167b732375328fd84673fb99ba3904e054 "I-25 upstream audit"
check_hash "$formal_log" 3cbefe68cfba632f901e6f2543ed463afd288569bf4e5d5e07e959f5123c693f "formal I-25 log"
[[ "$(wc -l <"$data")" -eq 6106 ]] || {
  echo "I-25 registered data is not 6106 rows" >&2
  exit 2
}

runtime_files=(
  "$llamafactory_root/src/llamafactory/hparams/parser.py"
  "$llamafactory_root/src/llamafactory/train/tuner.py"
  "$llamafactory_root/src/llamafactory/train/sft/workflow.py"
  "$llamafactory_root/src/llamafactory/train/sft/trainer.py"
  "$llamafactory_root/.venv/lib/python3.11/site-packages/transformers/trainer.py"
  "$llamafactory_root/.venv/lib/python3.11/site-packages/transformers/optimization.py"
  "$llamafactory_root/.venv/lib/python3.11/site-packages/transformers/trainer_callback.py"
)
runtime_hashes=(
  0b378ce63cba37ebb9cc32a8dc3f882943db4421a9928c528ee8596e0c0f2ae1
  623d402abd50173e9496a5298778181b33f24bca809de1d461e2838ed59c2e38
  aaa1f92b66c4985fe1c1d47a6c28831a1cd0b107d77e3819c2ea174e37a1384b
  4016aa09576891d53f2f3ee8f0cd21cd0f5c80c8ee1f45ab964a00750253d7c9
  14efc56b1e3f0fc30f797c5fd15ca59a02bfb0745dcc4697713ed5778cacc476
  985f8bef608fa57dd2c01e93869424ad193ffe82f03390f22cfd3d3ca99dad27
  a9703b60f3d585627054ed976eca3dbea69fddcfd4eb01b1cc19bff67bccf00a
)
for index in "${!runtime_files[@]}"; do
  check_hash "${runtime_files[$index]}" "${runtime_hashes[$index]}" "locked training-runtime source"
done

[[ ! -e "$recovery_output" && ! -L "$recovery_output" ]] || {
  echo "refusing to overwrite I-25 recovery output: $recovery_output" >&2
  exit 2
}
[[ ! -e "$receipt" && ! -L "$receipt" ]] || {
  echo "refusing to overwrite I-25 recovery receipt: $receipt" >&2
  exit 2
}
[[ ! -e "$formal_output/checkpoint-250" && ! -L "$formal_output/checkpoint-250" ]] || {
  echo "formal checkpoint-250 already exists; recovery is single-use and non-overwriting" >&2
  exit 2
}

"$python_bin" - "$root" "$plan" "$formal_config" "$recovery_config" "$formal_status" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import accelerate
import datasets
import peft
import safetensors
import torch
import transformers
import yaml

root = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
formal_config_path = Path(sys.argv[3])
recovery_config_path = Path(sys.argv[4])
formal_status_path = Path(sys.argv[5])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
if plan.get("status") != "PREREGISTERED_BEFORE_ONE_RECOVERY_REPLAY":
    raise SystemExit(f"recovery plan is not active: {plan.get('status')!r}")

versions = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets.__version__,
    "safetensors": safetensors.__version__,
}
expected_versions = plan["runtime_fingerprint"]["packages"]
if versions != expected_versions:
    raise SystemExit(f"training runtime version drift: {versions} != {expected_versions}")

formal_config = yaml.safe_load(formal_config_path.read_text(encoding="utf-8"))
recovery_config = yaml.safe_load(recovery_config_path.read_text(encoding="utf-8"))
delta = {
    key: (formal_config.get(key), recovery_config.get(key))
    for key in sorted(set(formal_config) | set(recovery_config))
    if formal_config.get(key) != recovery_config.get(key)
}
if set(delta) != {"output_dir", "run_name"}:
    raise SystemExit(f"recovery config has non-identity training drift: {delta}")
if "max_steps" in recovery_config or "warmup_ratio" in recovery_config:
    raise SystemExit("recovery config must retain epoch-derived 1527 steps and explicit warmup")
expected = {
    "num_train_epochs": 1,
    "learning_rate": 5.0e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 46,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "save_steps": 250,
    "save_total_limit": 6,
    "save_only_model": True,
    "seed": 19260821,
    "create_new_adapter": True,
    "lora_rank": 16,
    "lora_alpha": 16,
}
drift = {key: (recovery_config.get(key), value) for key, value in expected.items() if recovery_config.get(key) != value}
if drift:
    raise SystemExit(f"locked recovery fields drifted: {drift}")

formal = root / plan["incident"]["formal_output_root"]
state_path = formal / "trainer_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
if (state.get("global_step"), state.get("max_steps")) != (1527, 1527):
    raise SystemExit(f"formal I-25 state drifted: {state.get('global_step')}/{state.get('max_steps')}")
if formal_status_path.read_text(encoding="utf-8").strip() != "1":
    raise SystemExit("formal incident exit code must remain exactly 1")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

incident = plan["incident"]
if sha(state_path) != incident["formal_trainer_state_sha256"]:
    raise SystemExit("formal trainer_state changed after incident registration")
if sha(formal / "adapter_model.safetensors") != incident["root_adapter_sha256"]:
    raise SystemExit("formal root adapter drifted")
if sha(formal / "adapter_config.json") != incident["root_config_sha256"]:
    raise SystemExit("formal root config drifted")
for step, expected_hash in incident["surviving_adapter_sha256"].items():
    checkpoint = formal / f"checkpoint-{step}"
    if sha(checkpoint / "adapter_model.safetensors") != expected_hash:
        raise SystemExit(f"formal checkpoint-{step} adapter drifted")
    if sha(checkpoint / "adapter_config.json") != incident["root_config_sha256"]:
        raise SystemExit(f"formal checkpoint-{step} config drifted")
for name in ("adapter_model.safetensors", "adapter_config.json"):
    if (formal / name).read_bytes() != (formal / "checkpoint-1527" / name).read_bytes():
        raise SystemExit(f"formal checkpoint-1527 is not root-identical: {name}")
print("[i25-recovery] formal incident and config-identity preflight PASS")
print(f"[i25-recovery] runtime versions PASS: {versions}")
PY

gpu_fingerprint=$(nvidia-smi --id="$gpu_id" --query-gpu=uuid,name,driver_version --format=csv,noheader)
[[ "$gpu_fingerprint" == "$expected_gpu, NVIDIA H100 80GB HBM3, 535.230.02" ]] || {
  echo "I-25 recovery GPU fingerprint drifted: $gpu_fingerprint" >&2
  exit 2
}
IFS=',' read -r used_mib free_mib utilization <<<"$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
[[ "$used_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || {
  echo "could not parse I-25 recovery GPU baseline" >&2
  exit 2
}
(( used_mib >= 25000 && used_mib <= 30000 && free_mib >= 50000 && utilization == 0 )) || {
  echo "I-25 recovery GPU is not at the registered platform baseline: used=$used_mib free=$free_mib util=$utilization" >&2
  exit 2
}
printf '[i25-recovery] gpu baseline PASS: %s used=%sMiB free=%sMiB util=%s%%\n' "$gpu_id" "$used_mib" "$free_mib" "$utilization"
printf '[i25-recovery] recovery_plan_sha256=%s\n' "$expected_plan"

"$python_bin" "$recovery_trainer" --self-test
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
  "$python_bin" "$recovery_trainer" "$config"
train_rc=$?
set -e
if [[ "$train_rc" -ne 0 ]]; then
  echo "I-25 step-250 replay exited with status $train_rc; nothing was installed" >&2
  exit "$train_rc"
fi

source_checkpoint=$recovery_output/checkpoint-250
for path in \
  "$source_checkpoint/adapter_model.safetensors" \
  "$source_checkpoint/adapter_config.json" \
  "$recovery_output/adapter_model.safetensors" \
  "$recovery_output/adapter_config.json" \
  "$recovery_output/trainer_state.json"; do
  [[ -f "$path" ]] || { echo "missing recovery output: $path" >&2; exit 1; }
done
forbidden_state=$(find "$recovery_output" -type f \( \
  -name 'optimizer.pt' -o -name 'scheduler.pt' -o -name 'scaler.pt' -o \
  -name 'rng_state*.pth' \) -print -quit)
[[ -z "$forbidden_state" ]] || {
  echo "I-25 recovery retained forbidden state: $forbidden_state" >&2
  exit 1
}
[[ "$(find "$recovery_output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | wc -l)" -eq 1 ]] || {
  echo "I-25 recovery produced an unexpected checkpoint set" >&2
  exit 1
}
cmp -s "$source_checkpoint/adapter_model.safetensors" "$recovery_output/adapter_model.safetensors"
cmp -s "$source_checkpoint/adapter_config.json" "$recovery_output/adapter_config.json"

"$python_bin" - "$recovery_output/trainer_state.json" <<'PY'
import json
import sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
if (state.get("global_step"), state.get("max_steps")) != (250, 1527):
    raise SystemExit(f"recovery trainer state is not 250/1527: {state.get('global_step')}/{state.get('max_steps')}")
print("[i25-recovery] trainer_state global_step/max_steps=250/1527 PASS")
PY

actual_adapter=$(sha256sum "$source_checkpoint/adapter_model.safetensors" | awk '{print $1}')
actual_config=$(sha256sum "$source_checkpoint/adapter_config.json" | awk '{print $1}')
actual_bytes=$(stat -c '%s' "$source_checkpoint/adapter_model.safetensors")
if [[ "$actual_adapter" != "$expected_step250" || "$actual_config" != "$expected_adapter_config" || "$actual_bytes" -ne 40422168 ]]; then
  echo "I-25 deterministic replay REJECTED; no formal artifact was installed" >&2
  echo "observed adapter=$actual_adapter bytes=$actual_bytes config=$actual_config" >&2
  echo "expected adapter=$expected_step250 bytes=40422168 config=$expected_adapter_config" >&2
  exit 1
fi
printf '[i25-recovery] exact pre-rotation identity PASS: adapter=%s config=%s\n' "$actual_adapter" "$actual_config"

"$python_bin" - "$source_checkpoint" "$formal_output/checkpoint-250" "$receipt" "$expected_plan" "$formal_log" <<'PY'
import ctypes
import ctypes.util
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
receipt = Path(sys.argv[3])
plan_sha = sys.argv[4]
formal_log = Path(sys.argv[5])
expected = {
    "adapter_model.safetensors": "4af7296737209fb00df4908a09e21382ac6a9c987663b658e2f58e123fc928e9",
    "adapter_config.json": "6c127a34b497e7a8672763b238a50801dd9a3e6cbb662c3c981f9a9b4a4e976a",
}

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if destination.exists() or destination.is_symlink():
    raise SystemExit(f"refusing to overwrite recovery destination: {destination}")
if receipt.exists() or receipt.is_symlink():
    raise SystemExit(f"refusing to overwrite recovery receipt: {receipt}")
temporary = destination.parent / f".checkpoint-250.recovery.{os.getpid()}"
if temporary.exists() or temporary.is_symlink():
    raise SystemExit(f"refusing to reuse recovery staging path: {temporary}")

temporary.mkdir(mode=0o755)
installed = False
try:
    for name, expected_hash in expected.items():
        source_file = source / name
        if sha(source_file) != expected_hash:
            raise RuntimeError(f"source hash changed before install: {source_file}")
        target_file = temporary / name
        shutil.copyfile(source_file, target_file)
        os.chmod(target_file, 0o644)
        with target_file.open("rb") as handle:
            os.fsync(handle.fileno())
        if sha(target_file) != expected_hash:
            raise RuntimeError(f"staged hash mismatch: {target_file}")
    directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(temporary),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    installed = True
    parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
finally:
    if not installed and temporary.exists():
        shutil.rmtree(temporary)

if sorted(path.name for path in destination.iterdir()) != sorted(expected):
    raise SystemExit("installed checkpoint-250 does not contain exactly two locked files")
for name, expected_hash in expected.items():
    if sha(destination / name) != expected_hash:
        raise SystemExit(f"installed hash mismatch after atomic rename: {name}")

record = {
    "status": "EXACT_STEP250_RECOVERED_AND_INSTALLED",
    "installed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "recovery_plan_sha256": plan_sha,
    "formal_incident_log": str(formal_log),
    "formal_incident_log_sha256": sha(formal_log),
    "formal_incident_exit_code": 1,
    "formal_incident_exit_code_was_modified": False,
    "replay_global_step": 250,
    "replay_planned_max_steps": 1527,
    "replay_microbatches": 1000,
    "replay_routes": {"action": 278, "retention": 722},
    "source_checkpoint": str(source),
    "install_destination": str(destination),
    "install_contents": expected,
    "identity": "byte-exact match to the adapter/config hashes observed before formal checkpoint rotation",
}
temporary_receipt = receipt.with_name(f".{receipt.name}.tmp.{os.getpid()}")
if temporary_receipt.exists() or temporary_receipt.is_symlink():
    raise SystemExit(f"refusing to reuse receipt staging path: {temporary_receipt}")
payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
with temporary_receipt.open("x", encoding="utf-8") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
os.rename(temporary_receipt, receipt)
receipt_parent_fd = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(receipt_parent_fd)
finally:
    os.close(receipt_parent_fd)
print(f"[i25-recovery] atomic two-file install PASS: {destination}")
print(f"[i25-recovery] receipt: {receipt}")
PY

check_hash "$formal_output/checkpoint-250/adapter_model.safetensors" "$expected_step250" "installed formal checkpoint-250 adapter"
check_hash "$formal_output/checkpoint-250/adapter_config.json" "$expected_adapter_config" "installed formal checkpoint-250 config"
[[ "$(find "$formal_output/checkpoint-250" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 2 ]] || {
  echo "installed formal checkpoint-250 must contain exactly two files" >&2
  exit 1
}
echo "[i25-recovery] recovery complete; original formal exit code remains 1 by design"
