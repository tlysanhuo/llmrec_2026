#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

default_python=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3
if [[ -n "${LLAMAFACTORY_PYTHON:-}" ]]; then
  python_bin=$LLAMAFACTORY_PYTHON
elif [[ -x "$default_python" ]]; then
  python_bin=$default_python
else
  python_bin=python3
fi

parent_config=configs/active/i13_repro_parent_r64_ep3.yaml
residual_config=configs/active/i13_repro_residual_r16_retkl_ep1.yaml
parent_dir=checkpoints/i13_repro_parent_r64_ep3
residual_dir=checkpoints/i13_repro_residual_r16_retkl_ep1
combined_dir=checkpoints/i13_repro_combined_r80_s875
combine_audit=checkpoints/i13_repro_combined_r80_s875.audit.json

usage() {
  cat >&2 <<'EOF'
usage:
  scripts/reproduce/i13_highscore.sh verify-data
  scripts/reproduce/i13_highscore.sh restore-data
  scripts/reproduce/i13_highscore.sh self-test
  scripts/reproduce/i13_highscore.sh train-parent <gpu-id>
  scripts/reproduce/i13_highscore.sh train-residual <gpu-id>
  scripts/reproduce/i13_highscore.sh combine
  scripts/reproduce/i13_highscore.sh all <gpu-id>

Training commands require WANDB_ENTITY and WANDB_PROJECT. Set
LLAMAFACTORY_PYTHON/LLAMAFACTORY_CLI when not using the registered cluster venv.
EOF
  exit 2
}

verify_data() {
  python3 scripts/data/restore_i13_highscore_data.py --verify-only
}

restore_data() {
  python3 scripts/data/restore_i13_highscore_data.py
}

self_test() {
  "$python_bin" scripts/train/train_user_residual_retkl.py --self-test
  "$python_bin" scripts/train/combine_lora_adapters.py --self-test
}

train_parent() {
  local gpu_id=${1:?gpu-id is required}
  LLAMAFACTORY_PYTHON="$python_bin" \
    scripts/train/launch_wandb_online.sh "$gpu_id" "$parent_config"
}

train_residual() {
  local gpu_id=${1:?gpu-id is required}
  [[ -f "$parent_dir/adapter_model.safetensors" ]] || {
    echo "missing stage-1 parent adapter: $parent_dir/adapter_model.safetensors" >&2
    exit 2
  }
  USERRES_USER_KL=0.05 \
  USERRES_RETENTION_KL=2.0 \
  USERRES_TERMINAL_MULTIPLIER=2.0 \
  USERRES_LOGIT_CHUNK=16 \
  LLAMAFACTORY_PYTHON="$python_bin" \
    scripts/train/launch_user_residual_wandb.sh "$gpu_id" "$residual_config"
}

combine() {
  [[ -f "$parent_dir/adapter_model.safetensors" ]] || {
    echo "missing stage-1 parent adapter: $parent_dir/adapter_model.safetensors" >&2
    exit 2
  }
  [[ -f "$residual_dir/adapter_model.safetensors" ]] || {
    echo "missing stage-2 residual adapter: $residual_dir/adapter_model.safetensors" >&2
    exit 2
  }
  if [[ -e "$combined_dir" && "${I13_REPRO_FORCE:-0}" != 1 ]]; then
    echo "refusing to replace $combined_dir; set I13_REPRO_FORCE=1 if intentional" >&2
    exit 2
  fi
  "$python_bin" scripts/train/combine_lora_adapters.py \
    "$parent_dir" \
    "$residual_dir" \
    "$combined_dir" \
    --residual-scale 0.875 \
    --audit "$combine_audit"
}

command=${1:-}
case "$command" in
  verify-data)
    [[ $# -eq 1 ]] || usage
    verify_data
    ;;
  restore-data)
    [[ $# -eq 1 ]] || usage
    restore_data
    ;;
  self-test)
    [[ $# -eq 1 ]] || usage
    self_test
    ;;
  train-parent)
    [[ $# -eq 2 ]] || usage
    train_parent "$2"
    ;;
  train-residual)
    [[ $# -eq 2 ]] || usage
    train_residual "$2"
    ;;
  combine)
    [[ $# -eq 1 ]] || usage
    combine
    ;;
  all)
    [[ $# -eq 2 ]] || usage
    restore_data
    self_test
    train_parent "$2"
    train_residual "$2"
    combine
    ;;
  *)
    usage
    ;;
esac
