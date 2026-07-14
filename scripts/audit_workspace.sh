#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME=/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

for name in official derived third_party evaluation archive; do
  test -d "$RUNTIME/data/$name" || fail "missing runtime data category: $name"
done

for name in seed_sft hf_raw sft_aligned general_pretrain general_sft base_model; do
  test -e "$ROOT/assets/official/$name" || fail "missing official asset entry: $name"
done

check_parquet_set() {
  local path=$1 expected_count=$2 expected_bytes=$3 label=$4
  local count bytes
  count=$(find -L "$path" -maxdepth 1 -type f -name '*.parquet' | wc -l)
  bytes=$(find -L "$path" -maxdepth 1 -type f -name '*.parquet' -printf '%s\n' | awk '{s+=$1} END {printf "%.0f",s}')
  test "$count" -eq "$expected_count" || fail "$label parquet count is $count, expected $expected_count"
  test "$bytes" = "$expected_bytes" || fail "$label parquet bytes are $bytes, expected $expected_bytes"
}

check_parquet_set "$ROOT/assets/official/general_pretrain" 310 27139522149 "General-Pretrain"
check_parquet_set "$ROOT/assets/official/general_sft" 301 24685081929 "General-SFT"

root_data_count=$(find "$ROOT" -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.parquet' -o -name '*.tar.gz' -o -name '*.safetensors' \) | wc -l)
test "$root_data_count" -eq 0 || fail "dataset/model files are scattered in repository root"

mapfile -t checkpoint_dirs < <(find -L "$ROOT/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
artifact_index="$ROOT/docs/EXPERIMENT_INDEX.md"
for name in "${checkpoint_dirs[@]}"; do
  grep -Fq "checkpoints/$name/" "$artifact_index" || fail "retained checkpoint is not registered: $name"
done

mapfile -t intermediate_dirs < <(find -L "$ROOT/checkpoints" -type d -name 'checkpoint-*' -print | sort)
for path in "${intermediate_dirs[@]}"; do
  relative=${path#"$ROOT"/}
  grep -Fq "$relative/" "$artifact_index" || fail "intermediate checkpoint is not registered: $relative"
done

registry="$RUNTIME/LLaMA-Factory/data/dataset_info.json"
test -f "$registry" || fail "missing LLaMA-Factory dataset registry"
while IFS=$'\t' read -r key path; do
  test -f "$path" || fail "registered dataset is missing: $key -> $path"
done < <(jq -r 'to_entries[] | select((.value.file_name // "") | contains("llmrec_2026")) | [.key,.value.file_name] | @tsv' "$registry")

shopt -s nullglob
for cfg in "$ROOT"/configs/active/*.yaml; do
  grep -Eq '^num_train_epochs:[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$' "$cfg" || fail "active config has invalid epoch count: $cfg"
  if grep -Eq '^save_strategy:[[:space:]]*epoch[[:space:]]*$' "$cfg"; then
    grep -Eq '^save_total_limit:[[:space:]]*[1-9][0-9]*[[:space:]]*$' "$cfg" || fail "epoch-saving config has no positive save limit: $cfg"
    grep -Eq '^save_only_model:[[:space:]]*true[[:space:]]*$' "$cfg" || fail "epoch-saving config would retain optimizer state: $cfg"
  else
    grep -Eq '^save_strategy:[[:space:]]*("no"|no)[[:space:]]*$' "$cfg" || fail "unsupported save strategy in active config: $cfg"
  fi
  ! grep -q 'PENDING_DISTILL_COMPLETION' "$cfg" || fail "active config asset ledger is not finalized: $cfg"
  grep -Eq '^report_to:[[:space:]]*wandb[[:space:]]*$' "$cfg" || fail "active config does not report to W&B: $cfg"
done

if test -f "$ROOT/configs/secrets/deepseek_api.env"; then
  mode=$(stat -c '%a' "$ROOT/configs/secrets/deepseek_api.env")
  test "$mode" = 600 || fail "DeepSeek credential mode is $mode, expected 600"
fi

echo "[PASS] workspace structure, checkpoint retention, dataset registry, and active config policy"
