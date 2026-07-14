#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <gpu-id-or-uuid> <configs/active/run.yaml> <logs/train/run.log>" >&2
  exit 2
fi

gpu_id=$1
config=$2
log_path=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online formal run: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi

[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$root"/configs/active/*.yaml ]] || {
  echo "formal runs must use configs/active/*.yaml: $config" >&2
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
  [[ ! -e "$path" ]] || { echo "refusing to overwrite runtime record: $path" >&2; exit 2; }
done
mkdir -p "$(dirname "$log_path")"

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
' bash "$root/scripts/train/launch_wandb_online.sh" "$gpu_id" "$config" "$status_path" \
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
