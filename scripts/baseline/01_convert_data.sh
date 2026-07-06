#!/usr/bin/env bash
# 01_convert_data.sh — 官方 convert_jsonl.py：12 个赛道 jsonl → Alpaca data_final.jsonl
set -euo pipefail

REPRO=${REPRO:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026}
LF=$REPRO/LLaMA-Factory
PROJ=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026
# 官方下发数据解压目录（每行 [{system,prompt,response}]）
EXTRACTED=${EXTRACTED:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/extracted}
OUT=$REPRO/data/data_final.jsonl

source "$LF/.venv/bin/activate"
mkdir -p "$REPRO/data"

if [ -s "$OUT" ]; then
  echo "[skip] $OUT exists ($(wc -l < "$OUT") lines); delete to regenerate"; exit 0
fi

python "$PROJ/docs/demo_baseline/convert_jsonl.py" \
  --input  "$EXTRACTED" \
  --output "$OUT" \
  --shuffle --shuffle-seed 2026

echo "[ok] wrote $(wc -l < "$OUT") records to $OUT"
