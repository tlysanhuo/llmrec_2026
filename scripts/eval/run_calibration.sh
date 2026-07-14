#!/usr/bin/env bash
# run_calibration.sh — 离线台校准批跑(docs/offline_eval.md §3 锚集,双卡轮转,断点续跑)
# 用法: bash scripts/eval/run_calibration.sh <gpu_id> <queue: A|B>
set -u
V=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
PROJ=$V/llmrec_2026
PY=$V/miniconda3/envs/verl_v071/bin/python
cd "$PROJ"
GPU=${1:?gpu id}
QUEUE=${2:?A or B}

# 信息量优先排序:物料阶梯极值/方差对先跑,半途中断也已有可校准子集
A_LIST=(
  checkpoints/riders_fk_lora_ep1_merged
  checkpoints/exp_seed_ep3
  submissions/recipe2_w5_ep1_platform
  checkpoints/pstack_v2_ep3
  checkpoints/run_a_r2
  checkpoints/rebal_mat_ep3
  checkpoints/fk_lora_embed_ep1_merged
  checkpoints/global_v1_lora_ep1_merged
)
B_LIST=(
  checkpoints/recipe1_bs32_lr1e4_ep3
  checkpoints/seed_ep5
  models/OneReason-0.8B-pretrain-competition
  checkpoints/tokengeo_v1_ep3
  checkpoints/rebal_world_ep3
  checkpoints/baseline_sft_v1
  checkpoints/run_c_material
)
if [ "$QUEUE" = "A" ]; then LIST=("${A_LIST[@]}"); else LIST=("${B_LIST[@]}"); fi

for M in "${LIST[@]}"; do
  TAG=$(basename "$M")
  if ls logs/offline_eval/"${TAG}"_v4_*.json >/dev/null 2>&1; then
    echo "[skip] $TAG 已有 v4 读数"
    continue
  fi
  echo "[run ] $TAG on GPU$GPU $(date +%H:%M:%S)"
  $PY scripts/eval/offline_eval.py --model "$M" --gpu "$GPU" --tag "$TAG" \
    > "logs/offline_eval/${TAG}_run.log" 2>&1
  echo "[done] $TAG rc=$? $(date +%H:%M:%S)"
done
echo "queue $QUEUE finished"
