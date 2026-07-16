#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <locked-gpu-uuid> <configs/active/i25_step250_deterministic_replay.yaml> <logs/train/i25_step250_deterministic_replay.log>" >&2
  exit 2
fi

gpu_id=$1
config=$2
log_path=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}
expected_gpu=GPU-d3c522d6-ed0f-2579-01cd-2d97da749980
expected_config=$root/configs/active/i25_step250_deterministic_replay.yaml
plan=$root/configs/evaluation/i25_step250_deterministic_recovery_plan.json
online=$root/scripts/train/launch_i25_step250_recovery_online.sh
recovery_output=$root/checkpoints/i25_step250_deterministic_replay
formal_destination=$root/checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-250
receipt=$root/logs/train/i25_step250_deterministic_replay_receipt.json

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online recovery: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi
[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ "$gpu_id" == "$expected_gpu" ]] || {
  echo "I-25 recovery is locked to $expected_gpu" >&2
  exit 2
}
[[ -f "$config" ]] || { echo "missing recovery config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$expected_config" ]] || {
  echo "I-25 detached recovery received the wrong config: $config" >&2
  exit 2
}
[[ "$(sha256sum "$config" | awk '{print $1}')" == da3345db8ee86a12f71a61374b8f36e12a5c2df510bbb4416beee4c567092346 ]] || {
  echo "I-25 recovery config checksum drifted" >&2
  exit 2
}
[[ "$(sha256sum "$plan" | awk '{print $1}')" == 94da5c04650ff71f9117502ae323bf1367aa67fc3837fc6ab613b74decaba1ec ]] || {
  echo "I-25 recovery plan checksum drifted" >&2
  exit 2
}
[[ "$(sha256sum "$online" | awk '{print $1}')" == 0e5a1e04117500ae4617ef24050dfb1bc2c30c35d521dcc8f96c587dd45a48cc ]] || {
  echo "I-25 online recovery launcher checksum drifted" >&2
  exit 2
}

"$python_bin" - "$plan" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
if plan.get("status") != "PREREGISTERED_BEFORE_ONE_RECOVERY_REPLAY":
    raise SystemExit(f"I-25 recovery plan is not active: {plan.get('status')!r}")
rule = plan["scientific_equivalence"]["scheduler_rule"]
if "state.max_steps=1527" not in rule or "max_steps remains absent" not in rule:
    raise SystemExit("I-25 recovery scheduler lock drifted")
acceptance = plan["acceptance_and_install"]
if acceptance.get("required_global_step") != 250 or acceptance.get("required_state_max_steps") != 1527:
    raise SystemExit("I-25 recovery step lock drifted")
print("[i25-recovery] detached preregistration precheck PASS")
PY

[[ ! -e "$recovery_output" && ! -L "$recovery_output" ]] || {
  echo "refusing to overwrite I-25 recovery output" >&2
  exit 2
}
[[ ! -e "$formal_destination" && ! -L "$formal_destination" ]] || {
  echo "formal checkpoint-250 already exists; recovery is single-use" >&2
  exit 2
}
[[ ! -e "$receipt" && ! -L "$receipt" ]] || {
  echo "refusing to overwrite I-25 recovery receipt" >&2
  exit 2
}

logs_root=$(realpath "$root/logs/train")
log_path=$(realpath -m "$log_path")
[[ "$log_path" == "$logs_root/i25_step250_deterministic_replay.log" ]] || {
  echo "recovery log path must be exactly logs/train/i25_step250_deterministic_replay.log" >&2
  exit 2
}
pid_path=${log_path%.log}.pid
status_path=${log_path%.log}.exit_code
for path in "$log_path" "$pid_path" "$status_path"; do
  [[ ! -e "$path" && ! -L "$path" ]] || {
    echo "refusing to overwrite recovery runtime record: $path" >&2
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
' bash "$online" "$gpu_id" "$config" "$status_path" \
  </dev/null >"$log_path" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_path"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "detached I-25 recovery exited during startup; inspect $log_path" >&2
  exit 1
fi
echo "detached_pid=$pid"
echo "log=$log_path"
echo "status=$status_path"
echo "receipt=$receipt"
