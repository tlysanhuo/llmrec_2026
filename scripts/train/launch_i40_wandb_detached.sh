#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: WANDB_ENTITY=thaongocnguyendo0- WANDB_PROJECT=llmrec-2026 WANDB_MODE=online $0 <single-gpu-id>" >&2
  exit 2
fi

gpu_id=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
online_launcher=$root/scripts/train/launch_i40_wandb_online.sh
formal_log=$root/logs/train/i40_i35_direct_user_continue_r112_v1.log
output=$root/checkpoints/i40_i35_direct_user_continue_r112_v1
detached_log=$root/logs/train/i40_i35_direct_user_continue_r112_v1_detached_launcher.log
pid_path=$root/logs/train/i40_i35_direct_user_continue_r112_v1_detached_launcher.pid
status_path=$root/logs/train/i40_i35_direct_user_continue_r112_v1_detached_launcher.exit_code
expected_online_launcher=48a403eb7080a98b74e7371b8e1eebd855f3a5f56cc248dc9aa9e0c590288f1f

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
[[ "${WANDB_MODE:-}" == online ]] || {
  echo "I-40 detached launcher requires explicit WANDB_MODE=online" >&2
  exit 2
}
[[ "${WANDB_RESUME:-never}" == never ]] || {
  echo "I-40 detached launcher refuses W&B resume" >&2
  exit 2
}
[[ "$gpu_id" =~ ^[0-9]+$ ]] || {
  echo "I-40 detached launcher requires one numeric GPU id" >&2
  exit 2
}
[[ -x "$online_launcher" ]] || {
  echo "I-40 online launcher is missing or not executable" >&2
  exit 2
}
[[ "$(sha256sum "$online_launcher" | awk '{print $1}')" == "$expected_online_launcher" ]] || {
  echo "I-40 online launcher checksum drifted" >&2
  exit 2
}
for path in "$formal_log" "$output" "$detached_log" "$pid_path" "$status_path"; do
  [[ ! -e "$path" && ! -L "$path" ]] || {
    echo "I-40 detached launcher refuses to overwrite $path" >&2
    exit 2
  }
done

nohup setsid --wait bash -c '
  launcher=$1
  gpu=$2
  status=$3
  set +e
  "$launcher" "$gpu"
  rc=$?
  printf "%s\n" "$rc" >"$status"
  exit "$rc"
' bash "$online_launcher" "$gpu_id" "$status_path" \
  </dev/null >"$detached_log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_path"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "I-40 detached launcher exited during startup; inspect $detached_log" >&2
  exit 1
fi
echo "detached_pid=$pid"
echo "formal_log=$formal_log"
echo "launcher_log=$detached_log"
echo "status=$status_path"
