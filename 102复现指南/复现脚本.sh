#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法: bash 复现脚本.sh <llmrec_2026-main路径> <阶段>
阶段:
  check       检查代码、模型、I-13 parent 和两份源数据
  build-data 构建 3,146 行 world/retention 混合数据
  self-test   运行训练损失和 LoRA 拼接自测
  train       单卡训练 rank-16 I-19 residual
  combine     以 residual-scale=0.875 拼成 rank-96 adapter
  audit       可选：运行多 scale 的本地 CE/KL Pareto 审计
  all         依次运行 check、build-data、self-test、train、combine

可覆盖环境变量:
  GPU=0
  BASE_MODEL=models/OneReason-0.8B-pretrain-competition
  I13_PARENT=checkpoints/i13_repro_combined_r80_s875
  WORLD_DATA=assets/derived/releases/i19_frinkleko_world_1578/data_i19_frinkleko_world_1578_clean.jsonl
  RETAIN_DATA=assets/derived/releases/e3_userres_r80_retkl_v3_s875/data_seed_teacher_v1.jsonl
EOF
}

if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
REPO_DIR="$(cd "$1" && pwd)"
STAGE="$2"
cd "${REPO_DIR}"

GPU="${GPU:-0}"
BASE_MODEL="${BASE_MODEL:-models/OneReason-0.8B-pretrain-competition}"
I13_PARENT="${I13_PARENT:-checkpoints/i13_repro_combined_r80_s875}"
WORLD_DATA="${WORLD_DATA:-assets/derived/releases/i19_frinkleko_world_1578/data_i19_frinkleko_world_1578_clean.jsonl}"
RETAIN_DATA="${RETAIN_DATA:-assets/derived/releases/e3_userres_r80_retkl_v3_s875/data_seed_teacher_v1.jsonl}"
MIXED_DATA="assets/derived/releases/i19_userres_retention_v1/data_world_residual_retention_v1.jsonl"
RESIDUAL="checkpoints/i19_world_userres_retkl_r16_ep1_i13retain_v1"
COMBINED="checkpoints/i19_world_userres_retkl_r16_ep1_i13retain_v1_combined_r96"
CONFIG="configs/active/i19_world_userres_retkl_r16_ep1_i13retain_v1.yaml"

line_count() { python - "$1" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open("rb") as source:
    print(sum(1 for line in source if line.strip()))
PY
}

check() {
  local missing=0
  for path in \
    "${BASE_MODEL}" "${I13_PARENT}/adapter_config.json" "${I13_PARENT}/adapter_model.safetensors" \
    "${WORLD_DATA}" "${RETAIN_DATA}" "${CONFIG}" \
    scripts/data/build_world_residual_retention_v1.py scripts/data/build_seed_scoremax_v1.py \
    scripts/train/train_world_residual_retkl.py scripts/train/combine_lora_adapters.py; do
    if [[ ! -e "${path}" ]]; then echo "缺少: ${path}" >&2; missing=1; fi
  done
  [[ ${missing} -eq 0 ]] || exit 1
  [[ "$(line_count "${WORLD_DATA}")" == 1573 ]] || { echo "WORLD_DATA 必须是 1,573 行 clean 版" >&2; exit 1; }
  [[ "$(line_count "${RETAIN_DATA}")" == 32644 ]] || { echo "RETAIN_DATA 必须是 32,644 行 I-13 parent 数据" >&2; exit 1; }
  echo "检查通过: base、I-13 parent、两份源数据与代码均就绪。"
}

build_data() {
  python scripts/data/build_world_residual_retention_v1.py \
    --world-source "${WORLD_DATA}" \
    --retain-source "${RETAIN_DATA}" \
    --out "${MIXED_DATA}" \
    --audit assets/derived/releases/i19_userres_retention_v1/manifest.json \
    --seed 19260821
  [[ "$(line_count "${MIXED_DATA}")" == 3146 ]] || { echo "混合数据行数不是 3,146" >&2; exit 1; }
}

self_test() {
  WORLDRES_TEACHER_BASE="${BASE_MODEL}" python scripts/train/train_world_residual_retkl.py --self-test
  python scripts/train/combine_lora_adapters.py --self-test
}

train() {
  if [[ "${BASE_MODEL}" != "models/OneReason-0.8B-pretrain-competition" || "${I13_PARENT}" != "checkpoints/i13_repro_combined_r80_s875" ]]; then
    echo "train 阶段要求默认相对路径；请建立软链接，或同步修改 ${CONFIG}。" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  WORLDRES_TEACHER_BASE="${BASE_MODEL}" \
  WORLDRES_WORLD_KL=0.05 \
  WORLDRES_RETENTION_KL=2.0 \
  WORLDRES_TERMINAL_MULTIPLIER=2.0 \
  WORLDRES_LOGIT_CHUNK=16 \
    python scripts/train/train_world_residual_retkl.py "${CONFIG}"
}

combine() {
  python scripts/train/combine_lora_adapters.py \
    "${I13_PARENT}" "${RESIDUAL}" "${COMBINED}" \
    --residual-scale 0.875 \
    --audit "${COMBINED}/.audit.json"
}

audit() {
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/eval/audit_world_residual_delta.py \
    --base "${BASE_MODEL}" \
    --parent "${I13_PARENT}" \
    --residual "${RESIDUAL}" \
    --train-mixture "${MIXED_DATA}" \
    --retention-source "${RETAIN_DATA}" \
    --scales 0,0.25,0.5,0.75,0.8,0.875,0.9,1.0 \
    --gpu "${GPU}" \
    --output logs/audits/i19_world_residual_pareto.json
}

case "${STAGE}" in
  check) check ;;
  build-data) build_data ;;
  self-test) self_test ;;
  train) train ;;
  combine) combine ;;
  audit) audit ;;
  all) check; build_data; self_test; train; combine ;;
  *) usage >&2; exit 2 ;;
esac
