#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <single-gpu-index-or-full-uuid> <configs/active/i28_i23_rec_multigold_proposal_retkl_v1.yaml>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${LLAMAFACTORY_PYTHON:-/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/python3}

expected_config=$root/configs/active/i28_i23_rec_multigold_proposal_retkl_v1.yaml
trainer=$root/scripts/train/train_i28_multigold_proposal_retkl.py
set_path_evaluator=$root/scripts/eval/audit_i28_multigold_set_path.py
parent=$root/submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
data=$root/assets/derived/processed/data_i28_video_multigold_proposal_retkl_v1.jsonl
dataset_registry=$root/configs/datasets/i28_video_multigold_proposal_retkl_v1/dataset_info.json
builder=$root/scripts/data/build_i28_video_multigold_proposal_v1.py
audit=$root/logs/data/i28_video_multigold_proposal_v1_audit.json
holdout=$root/assets/evaluation/holdout/data_i28_video_multigold_proposal_v1_gate.jsonl
gate=$root/configs/evaluation/i28_i23_rec_multigold_proposal_checkpoint_gate.json
assets_registry=$root/docs/reference/ASSETS.md
experiment_index=$root/docs/EXPERIMENT_INDEX.md
output=$root/checkpoints/i28_i23_rec_multigold_proposal_retkl_v1

# This launcher deliberately remains impossible to execute until the final data,
# trainer, config, registry, and audit hashes are frozen and copied below.  Do
# not replace this value before ASSETS.md and EXPERIMENT_INDEX.md authorize the
# formal run.
release_state=READY_FOR_PREREGISTERED_I28_FORMAL_RUN
[[ "$release_state" == READY_FOR_PREREGISTERED_I28_FORMAL_RUN ]] || {
  echo "I-28 launcher is fail-closed: $release_state" >&2
  exit 2
}

expected_parent=0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8
expected_parent_config=b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7
expected_base_config=5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4
expected_data=f74a22b96bf7651a97cdb5b578f346eea0109045329efafb80092f91756fbb6a
expected_dataset_registry=1e8ba55adf40ec281a7ee1bf427c4f6e9bfaf083d4790b32b205a8f3d6a72e62
expected_builder=98b4129343db55c334c3768ddacbd908414eadb0d3be0e7ddc4e71ae20e4bf11
expected_audit=b0036b97fa210bd8738e75af348a6f0bbced83f5159c5e658624b5ee3c643be2
expected_holdout=48dd7f4224e7ca9e98805d966ca901814fdb76b85471afcf1ec7d98a0c22c7e5
expected_trainer=72fa991433698cd7f705a700d7d72c467356e5530e1f625547e84a6ecaef7253
expected_set_path_evaluator=647f002dac412b9ba3c08f5bfb7ad18d73e7de51eb5ab11aa4d8887a0850f2d3
expected_config_hash=f6086e468b91a9df93804d7ab6421549bb8767a07522a1340aac9939772f077a
expected_gate_hash=f57aad89db68f779e4eacf14617986ecb9956cd2a0b41145c17d1d5bf120f006
expected_wandb_entity=thaongocnguyendo0-
expected_wandb_project=llmrec-2026

[[ -x "$python_bin" ]] || { echo "python is not executable: $python_bin" >&2; exit 2; }
[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || {
  echo "I-28 formal training must expose exactly one GPU" >&2
  exit 2
}
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$expected_config" ]] || {
  echo "I-28 launcher only accepts $expected_config" >&2
  exit 2
}
if grep -Eq '^### (ABORTED|DO_NOT_RUN|HISTORICAL_ONLY)' "$config"; then
  echo "refusing a disabled or not-yet-authorized training config: $config" >&2
  exit 2
fi
grep -Eq '^report_to:[[:space:]]*wandb[[:space:]]*$' "$config" || {
  echo "config does not report to W&B: $config" >&2
  exit 2
}
: "${WANDB_ENTITY:?WANDB_ENTITY is required}"
: "${WANDB_PROJECT:?WANDB_PROJECT is required}"
[[ "$WANDB_ENTITY" == "$expected_wandb_entity" ]] || {
  echo "I-28 requires the write-verified W&B entity $expected_wandb_entity" >&2
  exit 2
}
[[ "$WANDB_PROJECT" == "$expected_wandb_project" ]] || {
  echo "I-28 requires the frozen W&B project $expected_wandb_project" >&2
  exit 2
}
if [[ "${WANDB_MODE:-online}" != online ]]; then
  echo "refusing non-online formal run: WANDB_MODE=${WANDB_MODE:-unset}" >&2
  exit 2
fi
export WANDB_MODE=online

for required in \
  "$trainer" "$set_path_evaluator" "$parent/adapter_model.safetensors" "$parent/adapter_config.json" \
  "$base_config" "$data" "$dataset_registry" "$builder" "$audit" "$holdout" "$gate"; do
  [[ -f "$required" ]] || { echo "missing frozen I-28 input: $required" >&2; exit 2; }
done
grep -Fq 'data_i28_video_multigold_proposal_retkl_v1.jsonl' "$assets_registry" || {
  echo "I-28 formal D asset is not registered in ASSETS.md" >&2
  exit 2
}
grep -Fq 'i28_i23_rec_multigold_proposal_retkl_v1' "$experiment_index" || {
  echo "I-28 formal run is not preregistered in EXPERIMENT_INDEX.md" >&2
  exit 2
}

check_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "$label checksum drifted: $actual/$expected" >&2
    exit 2
  }
}

check_sha256 "$parent/adapter_model.safetensors" "$expected_parent" "I-23 parent adapter"
check_sha256 "$parent/adapter_config.json" "$expected_parent_config" "I-23 parent config"
check_sha256 "$base_config" "$expected_base_config" "O6 base config"
check_sha256 "$data" "$expected_data" "I-28 training data"
check_sha256 "$dataset_registry" "$expected_dataset_registry" "I-28 dataset registry"
check_sha256 "$builder" "$expected_builder" "I-28 builder"
check_sha256 "$audit" "$expected_audit" "I-28 build audit"
check_sha256 "$holdout" "$expected_holdout" "I-28 rollout gate holdout"
check_sha256 "$trainer" "$expected_trainer" "I-28 trainer"
check_sha256 "$set_path_evaluator" "$expected_set_path_evaluator" "I-28 set-path evaluator"
check_sha256 "$config" "$expected_config_hash" "I-28 config"
check_sha256 "$gate" "$expected_gate_hash" "I-28 preregistered gate"

"$python_bin" - "$root" "$gate" "$dataset_registry" "$data" "$audit" "$holdout" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root, gate_path, registry_path, data_path, audit_path, holdout_path = map(
    Path, sys.argv[1:]
)

gate_text = gate_path.read_text(encoding="utf-8")
if "PENDING_" in gate_text:
    raise SystemExit("I-28 gate still contains a PENDING field")
gate = json.loads(gate_text)
if gate.get("status") != "PREREGISTERED_BEFORE_I28_FORMAL_LAUNCH":
    raise SystemExit(f"I-28 gate is not launch-authorized: {gate.get('status')!r}")
if gate.get("candidate_steps_in_order") != [64, 128]:
    raise SystemExit("I-28 checkpoint axis drifted")
recipe = gate.get("frozen_training_recipe", {})
expected_recipe = {
    "single_gpu": True,
    "create_new_adapter": False,
    "per_device_batch": 1,
    "gradient_accumulation": 4,
    "effective_batch": 4,
    "optimizer_steps": 128,
    "complete_dataset_exposures": 1,
    "learning_rate": 1e-7,
    "warmup_steps": 16,
}
for key, expected in expected_recipe.items():
    if recipe.get(key) != expected:
        raise SystemExit(f"I-28 gate recipe drifted for {key}: {recipe.get(key)!r}/{expected!r}")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_i28_video_multigold_proposal_retkl_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected I-28 dataset registry keys: {sorted(registry)}")
entry = registry[key]
if entry.get("formatting") != "alpaca":
    raise SystemExit("I-28 dataset registry must use alpaca formatting")
expected_columns = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "history": "history",
}
if entry.get("columns") != expected_columns:
    raise SystemExit(f"I-28 dataset columns drifted: {entry.get('columns')!r}")
registered_data = (registry_path.parent / entry["file_name"]).resolve()
if registered_data != data_path.resolve():
    raise SystemExit(f"I-28 dataset registry resolves to {registered_data}, not {data_path.resolve()}")

rows = []
with data_path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            raise SystemExit(f"blank I-28 training line: {line_number}")
        rows.append(json.loads(line))
if len(rows) != 512:
    raise SystemExit(f"I-28 data row count drifted: {len(rows)}/512")
route_counts = Counter(row.get("route") for row in rows)
if route_counts != {"proposal_ce": 128, "retention_kl": 384}:
    raise SystemExit(f"I-28 data route counts drifted: {route_counts}")
proposal_group_counts = Counter(
    row.get("group_id") for row in rows if row.get("route") == "proposal_ce"
)
if len(proposal_group_counts) != 64 or set(proposal_group_counts.values()) != {2}:
    raise SystemExit("I-28 proposal groups are not exactly 64 groups x 2 positives")

with holdout_path.open(encoding="utf-8") as source:
    holdout_rows = [json.loads(line) for line in source if line.strip()]
if len(holdout_rows) != 128 or Counter(row.get("route") for row in holdout_rows) != {
    "gate_only": 128
}:
    raise SystemExit("I-28 E gate row/route count drifted")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
training = audit.get("training_rows", {})
if training.get("rows") != 512 or training.get("route_counts") != {
    "proposal_ce": 128,
    "retention_kl": 384,
}:
    raise SystemExit("I-28 audit row/route contract drifted")
proposal = training.get("proposal", {})
if (
    proposal.get("groups") != 64
    or proposal.get("rows") != 128
    or proposal.get("rows_per_group") != 2
    or proposal.get("selected_golds_also_in_immutable_prompt") != 0
):
    raise SystemExit("I-28 proposal audit contract drifted")
retention = training.get("retention", {})
expected_retention = {
    "material_desc2sid": 96,
    "material_sid2desc": 96,
    "action": 32,
    "topic": 32,
    "rec_prod": 32,
    "rec_ad": 32,
    "rec_living": 32,
    "world": 32,
}
if (
    retention.get("rows") != 384
    or retention.get("selected_task_counts") != expected_retention
    or retention.get("selected_strict_proposal_signature_rows") != 0
    or retention.get("teacher_core_field_changes") != 0
):
    raise SystemExit("I-28 retention audit contract drifted")
token_audit = training.get("qwen3_response_token_audit", {})
if (
    token_audit.get("status") != "PASS"
    or token_audit.get("rows_checked") != 512
    or token_audit.get("route_counts") != {"proposal_ce": 128, "retention_kl": 384}
    or token_audit.get("empty_think_exact_itemic_proposal_signature_by_route")
    != {"proposal_ce": 128, "retention_kl": 0}
    or token_audit.get("qwen3_nothink_formatted_cutoff", {}).get("overflow_rows") != 0
):
    raise SystemExit("I-28 O6-tokenizer audit contract drifted")
forbidden = audit.get("forbidden_sources", {})
if forbidden != {
    "T_rows": 0,
    "E_rows_in_training": 0,
    "model_or_teacher_rollout_rows": 0,
    "O3_target_metadata_rows": 0,
}:
    raise SystemExit(f"I-28 forbidden-source audit drifted: {forbidden}")
split = audit.get("split", {})
for overlap_key in (
    "train_gate_normalized_prompt_overlap",
    "train_i27_normalized_prompt_overlap",
    "gate_i27_normalized_prompt_overlap",
    "complete_train_gate_normalized_prompt_overlap",
    "complete_train_i27_normalized_prompt_overlap",
):
    if split.get(overlap_key) != 0:
        raise SystemExit(f"I-28 prompt overlap gate failed: {overlap_key}")
gate_audit = audit.get("gate", {})
if (
    gate_audit.get("rows") != 128
    or gate_audit.get("groups") != 128
    or gate_audit.get("primary_non_history_gold_count") != 539
    or gate_audit.get("training_allowed") is not False
):
    raise SystemExit("I-28 E gate audit contract drifted")
if audit.get("formal_training_started") is not False:
    raise SystemExit("I-28 immutable build audit no longer records pre-training state")

if output := gate.get("formal_training_data", {}).get("path"):
    if (root / output).resolve() != data_path.resolve():
        raise SystemExit("I-28 gate training-data path drifted")
PY

[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "refusing to overwrite I-28 output directory: $output" >&2
  exit 2
}

gpu_row=$(nvidia-smi \
  --query-gpu=index,uuid,memory.free,utilization.gpu \
  --format=csv,noheader,nounits | \
  awk -F ', ' -v target="$gpu_id" '$1 == target || $2 == target {print; exit}')
[[ -n "$gpu_row" ]] || {
  echo "GPU target is not an exact index or full UUID on this host: $gpu_id" >&2
  exit 2
}
IFS=', ' read -r observed_index observed_uuid free_mib util_pct <<<"$gpu_row"
if (( free_mib < 50000 )); then
  echo "I-28 requires at least 50000 MiB free at launch; GPU $observed_index has $free_mib MiB" >&2
  exit 2
fi
if (( util_pct > 10 )); then
  echo "I-28 refuses a GPU above 10% utilization at launch; GPU $observed_index is $util_pct%" >&2
  exit 2
fi
echo "I-28 GPU preflight PASS: index=$observed_index uuid=$observed_uuid free_mib=$free_mib util_pct=$util_pct"

"$python_bin" "$trainer" --self-test
"$root/scripts/audit_workspace.sh"
"$python_bin" -c 'import wandb; print(f"W&B authenticated as {wandb.Api(timeout=15).viewer}")'

exec env \
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  I28_REFERENCE_ADAPTER="$parent" \
  I28_PROPOSAL_KL=0.20 \
  I28_RETENTION_KL=4.0 \
  I28_RETENTION_MAX_TOKENS=128 \
  I28_LOGIT_CHUNK=8 \
  I28_EXPECTED_MICROBATCHES=512 \
  I28_EXPECTED_PROPOSALS=128 \
  I28_EXPECTED_RETENTIONS=384 \
  I28_EXPECTED_PROPOSAL_DOMAIN=video \
  "$python_bin" "$trainer" "$config"
