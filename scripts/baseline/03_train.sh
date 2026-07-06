#!/usr/bin/env bash
# 03_train.sh — 全量 SFT 训练（本地 2×H100，GPU 3,6）
# 复现官方 demo，配置见 configs/baseline/baseline_sft_v1.yaml
set -euo pipefail

REPRO=${REPRO:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026}
LF=$REPRO/LLaMA-Factory
VENV=$LF/.venv
PROJ=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026
CONFIG=$PROJ/configs/baseline/baseline_sft_v1.yaml
OUT=$PROJ/checkpoints/baseline_sft_v1
LOG=$OUT/train.log

source "$VENV/bin/activate"

# 只用空闲的 GPU（默认 3,6；可用 GPUS 环境变量覆盖）
export CUDA_VISIBLE_DEVICES=${GPUS:-3,6}
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1   # 多卡 DDP

mkdir -p "$OUT"
echo "[run] llamafactory-cli train $CONFIG  (GPUS=$CUDA_VISIBLE_DEVICES)"
echo "[run] log -> $LOG"
llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOG"
