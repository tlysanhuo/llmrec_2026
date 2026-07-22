#!/usr/bin/env bash
# 用 eval/data/competition_smoke.jsonl（当前版本，含 material 修正与新增样本）
# 依次对 A~E 五个锚点跑冒烟评测，再执行聚合校准脚本。
# 用法：bash eval/run_calibration_smoke.sh
set -euo pipefail

cd "$(dirname "$0")"

MODEL=/home/hadoop-ba-rc/one_reason/OneReason-0.8B
GPU=0
# material 维度样本量大（546 条数据集中 533 条为 material），用多进程并行 beam
# search 绕开单进程 GIL 瓶颈，充分利用多核 CPU；4 个 worker 共享同一张 GPU。
ENGINE_MODE=multiprocess
MATERIAL_WORKERS=4

SCRIPT_START=$(date +%s)

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')][run_calibration_smoke] $1"
}

# 当前运行的 run_smoke_eval.py 子进程 pid（供 trap 转发终止信号用）。
CURRENT_PY_PID=""

cleanup_on_signal() {
  log "收到终止信号，尝试清理子进程..."
  if [[ -n "$CURRENT_PY_PID" ]] && kill -0 "$CURRENT_PY_PID" 2>/dev/null; then
    # 转发 SIGTERM 给 run_smoke_eval.py，触发它自己注册的清理逻辑（终止
    # material_worker.py 及其 fork 出的 vLLM EngineCore 子进程，释放显存）。
    kill -TERM "$CURRENT_PY_PID" 2>/dev/null || true
    wait "$CURRENT_PY_PID" 2>/dev/null || true
  fi
  exit 1
}
trap cleanup_on_signal SIGINT SIGTERM

run_anchor() {
  local step="$1"; local name="$2"; local tag="$3"; local output="$4"; shift 4
  local t0
  t0=$(date +%s)
  log "=== [${step}/6] Anchor ${tag}: ${name} 开始 ==="
  # 放后台运行并记录 pid：这样 trap 才能在脚本收到 SIGINT/SIGTERM 时立刻转发
  # 信号给 run_smoke_eval.py（若用前台阻塞调用，bash 要等它跑完才处理信号）。
  python3 run_smoke_eval.py \
    --model "$MODEL" \
    --tag "$tag" \
    --output "$output" \
    --gpu "$GPU" \
    --engine-mode "$ENGINE_MODE" \
    --material-workers "$MATERIAL_WORKERS" \
    "$@" &
  CURRENT_PY_PID=$!
  wait "$CURRENT_PY_PID"
  CURRENT_PY_PID=""
  local t1
  t1=$(date +%s)
  log "=== [${step}/6] Anchor ${tag}: ${name} 完成，耗时 $((t1 - t0))s（累计 $((t1 - SCRIPT_START))s）==="
}

log "开始运行 A~E 五锚点冒烟评测，engine_mode=${ENGINE_MODE} material_workers=${MATERIAL_WORKERS}"

run_anchor 1 "base (no LoRA)" A_base output/calibration_smoke_A.json

run_anchor 2 "Frinkleko baseline" B_frinkleko output/calibration_smoke_B.json \
  --lora /home/hadoop-ba-rc/one_reason/demo/output/onereason_0.8b_lora_frinkleko

run_anchor 3 "I-19 world gold combined r96" C_i19_world_gold output/calibration_smoke_C.json \
  --lora /home/hadoop-ba-rc/one_reason/llmrec_2026-main/checkpoints/i19_world_gold_combined_r96

run_anchor 4 "I-20 scale=0.25 combined r96" D_i20_scale025 output/calibration_smoke_D.json \
  --lora /home/hadoop-ba-rc/one_reason/llmrec_2026-main/checkpoints/i20_i19a1_scale025_combined_r96

run_anchor 5 "I-13 repro combined r80 s875" E_i13_repro output/calibration_smoke_E.json \
  --lora /home/hadoop-ba-rc/one_reason/llmrec_2026-main/checkpoints/i13_repro_combined_r80_s875

log "=== [6/6] Aggregate calibration report 开始 ==="
python3 calibrate_smoke.py
log "=== [6/6] Aggregate calibration report 完成 ==="

log "全部完成，总耗时 $(( $(date +%s) - SCRIPT_START ))s。See output/calibration_smoke_report.json"
