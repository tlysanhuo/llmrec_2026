#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <single-gpu-id-or-uuid> <configs/active/i23_actionres_r16_ansretkl_ep1_retfix.yaml> <logs/train/run.log>" >&2
  exit 2
fi

gpu_id=$1
config=$2
log_path=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
gate=$root/configs/evaluation/i23_actionres_r16_ansretkl_ep1_retfix_checkpoint_gate.json
expected_gate=2e3d3730b4cff13b0c0e0c99f7b80fb6c4bc4d3424f3f59bfb95a3be645f8ed1

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online formal run: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi

[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || {
  echo "I-26 detached training must expose exactly one GPU" >&2
  exit 2
}
[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$root/configs/active/i23_actionres_r16_ansretkl_ep1_retfix.yaml" ]] || {
  echo "I-26 detached launcher received the wrong config: $config" >&2
  exit 2
}
[[ -f "$gate" ]] || { echo "missing I-26 preregistered gate: $gate" >&2; exit 2; }
[[ "$(sha256sum "$gate" | awk '{print $1}')" == "$expected_gate" ]] || {
  echo "I-26 preregistered gate checksum drifted" >&2
  exit 2
}
"$python_bin" - "$gate" "$root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

gate_path = Path(sys.argv[1])
root = Path(sys.argv[2])
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
expected_paths = {
    "config": "configs/active/i23_actionres_r16_ansretkl_ep1_retfix.yaml",
    "trainer": "scripts/train/train_i23_actionres_retkl.py",
    "online_launcher": "scripts/train/launch_i26_retfix_wandb_online.sh",
    "detached_launcher": "scripts/train/launch_i26_retfix_wandb_detached.sh",
}
for name, relative in expected_paths.items():
    if artifacts[name].get("path") != relative:
        raise SystemExit(f"I-26 gate {name} path drifted: {artifacts[name]}")
for name in ("config", "trainer"):
    record = artifacts[name]
    actual = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
    if actual != record.get("sha256"):
        raise SystemExit(f"I-26 gate {name} hash mismatch: {actual}")

invariants = gate["training_invariants"]
expected_invariants = {
    "single_gpu": True,
    "wandb_enabled": True,
    "wandb_mode": "online",
    "epochs": 1,
    "optimizer_steps": 1527,
    "effective_batch_size": 4,
    "residual_rank": 16,
    "residual_alpha": 16,
    "learning_rate": 5.0e-5,
    "lr_schedule": "cosine",
    "warmup_steps": 46,
    "action_ce_weight": 1.0,
    "action_kl_weight": 0.05,
    "retention_kl_weight": 2.0,
    "seed": 19260821,
}
drift = {key: (invariants.get(key), value) for key, value in expected_invariants.items() if invariants.get(key) != value}
if drift:
    raise SystemExit(f"I-26 gate invariant drift: {drift}")
axis = gate["checkpoint_axis"]
if axis.get("output_root") != "checkpoints/i23_actionres_r16_ansretkl_ep1_retfix":
    raise SystemExit("I-26 gate output root drifted")
if axis.get("candidates_in_ascending_order") != [250, 500, 750, 1000, 1250, 1527]:
    raise SystemExit("I-26 gate candidate order drifted")
print("[i25] detached preregistered gate precheck PASS")
PY
[[ ! -e "$root/checkpoints/i23_actionres_r16_ansretkl_ep1_retfix" && ! -L "$root/checkpoints/i23_actionres_r16_ansretkl_ep1_retfix" ]] || {
  echo "refusing to overwrite I-26 output directory" >&2
  exit 2
}

logs_root=$(realpath "$root/logs/train")
log_path=$(realpath -m "$log_path")
[[ "$log_path" == "$logs_root"/*.log ]] || {
  echo "formal run log must be logs/train/*.log: $log_path" >&2
  exit 2
}
pid_path=${log_path%.log}.pid
status_path=${log_path%.log}.exit_code
for path in "$log_path" "$pid_path" "$status_path"; do
  [[ ! -e "$path" && ! -L "$path" ]] || {
    echo "refusing to overwrite runtime record: $path" >&2
    exit 2
  }
done

nohup setsid --wait bash -c '
  launcher=$1
  gpu=$2
  config=$3
  status=$4
  set +e
  "$launcher" "$gpu" "$config"
  rc=$?
  printf "%s\n" "$rc" >"$status"
  exit "$rc"
' bash "$root/scripts/train/launch_i26_retfix_wandb_online.sh" "$gpu_id" "$config" "$status_path" \
  </dev/null >"$log_path" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_path"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "detached launcher exited during startup; inspect $log_path" >&2
  exit 1
fi
echo "detached_pid=$pid"
echo "log=$log_path"
echo "status=$status_path"
