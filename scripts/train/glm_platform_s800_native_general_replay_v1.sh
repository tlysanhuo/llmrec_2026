#!/usr/bin/env bash
# GLM Platform Training Task container entry; this is not an SSH submit command.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: WANDB_ENTITY=<entity> WANDB_PROJECT=<project> $0 <single-gpu-index-or-uuid> <config>" >&2
  exit 2
fi

gpu_id=$1
config=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
personal_root=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
runtime_root=$personal_root/ai_runtime
python_bin=${LLAMAFACTORY_PYTHON:-$runtime_root/llmrec_2026/LLaMA-Factory/.venv/bin/python3}

expected_config=$root/configs/active/s800_native_general_replay_r8_v1.yaml
trainer=$root/scripts/train/train_s800_native_general_replay.py
builder=$root/scripts/data/build_s800_native_general_replay_v1.py
data=$root/assets/derived/processed/data_s800_native_general_replay_v1.jsonl
route_manifest=$root/assets/derived/official_general/s800_native_general_replay_v1_routes.json
audit=$root/logs/data/s800_native_general_replay_v1_audit.json
registry=$root/configs/datasets/s800_native_general_replay_v1/dataset_info.json
checkpoint_gate=$root/configs/evaluation/s800_native_general_replay_checkpoint_gate_v1.json
retention_gate=$root/assets/evaluation/holdout/s800_native_general_replay_retention_gate_v1.jsonl
retention_gate_builder=$root/scripts/data/build_s800_native_general_replay_gate_v1.py
retention_gate_audit=$root/logs/data/s800_native_general_replay_retention_gate_v1_audit.json
general_gate=$root/assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl
general_baseline=$root/logs/probe/official_general_world_mc_v1_s800_baseline.json
parent=$root/submissions/e3_userres_r80_retkl_v3_s800_platform
base_config=$root/models/OneReason-0.8B-pretrain-competition/config.json
output=$root/checkpoints/s800_native_general_replay_r8_v1

[[ -d "$personal_root" ]] || { echo "missing personal volume: $personal_root" >&2; exit 2; }
mountpoint -q "$personal_root" || { echo "personal volume is not mounted: $personal_root" >&2; exit 2; }
test -w "$personal_root" || { echo "personal volume is not writable: $personal_root" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "training Python is not executable: $python_bin" >&2; exit 2; }
[[ -n "$gpu_id" && "$gpu_id" != *,* ]] || {
  echo "formal replay must expose exactly one GPU" >&2
  exit 2
}
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
config=$(realpath "$config")
[[ "$config" == "$expected_config" ]] || {
  echo "entry only accepts $expected_config" >&2
  exit 2
}
[[ ! -e "$output" && ! -L "$output" ]] || {
  echo "refusing to overwrite formal output: $output" >&2
  exit 2
}

: "${WANDB_API_KEY:?WANDB_API_KEY must be configured in the Training Task secret environment}"
: "${WANDB_ENTITY:?WANDB_ENTITY must be configured in the Training Task environment}"
: "${WANDB_PROJECT:?WANDB_PROJECT must be configured in the Training Task environment}"
[[ "${WANDB_MODE:-online}" == online ]] || {
  echo "refusing non-online W&B mode: ${WANDB_MODE:-unset}" >&2
  exit 2
}
[[ "$WANDB_ENTITY" == "thaongocnguyendo0-" ]] || {
  echo "unexpected W&B entity: $WANDB_ENTITY" >&2
  exit 2
}
[[ "$WANDB_PROJECT" == "llmrec-2026" ]] || {
  echo "unexpected W&B project: $WANDB_PROJECT" >&2
  exit 2
}

export WANDB_MODE=online
export TMPDIR=$runtime_root/tmp
export HF_HOME=$runtime_root/hf
export TRANSFORMERS_CACHE=$runtime_root/hf
export WANDB_DIR=$runtime_root/wandb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$HF_HOME" "$WANDB_DIR"
for directory in "$TMPDIR" "$HF_HOME" "$WANDB_DIR"; do
  resolved=$(realpath "$directory")
  [[ "$resolved" == "$runtime_root"/* ]] || {
    echo "runtime directory escapes personal volume: $directory -> $resolved" >&2
    exit 2
  }
done

declare -A expected_hashes=(
  ["$base_config"]="5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
  ["$parent/adapter_model.safetensors"]="bb86eb8af0efd3560b7b7c8440f3830627e9255f4fcc2265b9274a27668f63c6"
  ["$parent/adapter_config.json"]="e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0"
  ["$data"]="87097135eb7ddb866b78ae6427c24b8cc2712f898c892d7da92d58ff7e9fddd2"
  ["$route_manifest"]="630ec7d0c60363eb64e2a655057200cd70c49bcd4797c671cc7a14750f4004af"
  ["$audit"]="96b6c63696d5908cc25dd1008a7fb9a64e1a31650d942a6f48af3e28e4b25e04"
  ["$builder"]="ca9e1ee57c2b7d7ed0838d0c3b6677b85c6cac2f07dcd7cb079c36c8b4f37130"
  ["$trainer"]="cee8c25828016f9a5d7df08507589a1fa996d5a845c2b498c347ed2373078071"
  ["$registry"]="571ea5521d6ccbdabb034a72ff254c2c212051a1824ff455a9052dc4cedf27c8"
  ["$config"]="c95c6116e321a2fcbcbb1a9ccd28a2e794d3f04c359ae12760c6baca74b0de2f"
  ["$checkpoint_gate"]="970d169ddfb42e1064ca30f5cad4e1f85bf9e99f669726daadf44f2ee100c416"
  ["$retention_gate"]="3206e91ac465ca4f1410e3f8a9219a60c11cbb1beb3d4eb2fa9fa69c4b89c30f"
  ["$retention_gate_builder"]="62482f3638b4ba09cd6e93fa4003a8ae2fc14169bdb2289a44d8b62188c1ce9e"
  ["$retention_gate_audit"]="5a4095f06dd0f8c85b87bc1766a590b6d8513d5177be455ccdb82bd83fd4d6be"
  ["$general_gate"]="fb67b76d8d071799ba372185bd89cb556afef9065a1b188fb9dd86a9131e13df"
  ["$general_baseline"]="fc5f477528c1a446c2ef7f16394ea071c7487a3eaae7c239e6a865588abdceb4"
)
for path in "${!expected_hashes[@]}"; do
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "${expected_hashes[$path]}" ]] || {
    echo "frozen input hash drifted: $path $actual/${expected_hashes[$path]}" >&2
    exit 2
  }
done

[[ $(wc -l <"$data") -eq 513 ]] || { echo "training JSONL is not 513 rows" >&2; exit 2; }

"$python_bin" - "$config" "$registry" "$data" "$route_manifest" "$audit" "$root" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

config_path, registry_path, data_path, manifest_path, audit_path, root = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected = {
    "model_name_or_path": str(root / "models/OneReason-0.8B-pretrain-competition"),
    "adapter_name_or_path": str(root / "submissions/e3_userres_r80_retkl_v3_s800_platform"),
    "create_new_adapter": True,
    "lora_rank": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "lora_target": "all",
    "dataset": "data_s800_native_general_replay_v1",
    "template": "qwen3_nothink",
    "cutoff_len": 16384,
    "packing": False,
    "output_dir": str(root / "checkpoints/s800_native_general_replay_r8_v1"),
    "save_strategy": "no",
    "save_total_limit": 4,
    "save_only_model": True,
    "overwrite_output_dir": False,
    "report_to": "wandb",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "max_steps": 129,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 8,
    "weight_decay": 0.0,
    "seed": 19260831,
}
drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if drift:
    raise SystemExit(f"locked training config drifted: {drift}")
if "warmup_ratio" in config or "resume_from_checkpoint" in config:
    raise SystemExit("warmup ratio and resume are forbidden")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
key = "data_s800_native_general_replay_v1"
if set(registry) != {key}:
    raise SystemExit(f"unexpected dataset registry: {sorted(registry)}")
entry = registry[key]
expected_columns = {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "history": "history",
}
if entry.get("formatting") != "alpaca" or entry.get("columns") != expected_columns:
    raise SystemExit("dataset registry formatting drifted")
if Path(entry["file_name"]).resolve() != data_path.resolve():
    raise SystemExit("dataset registry points to the wrong training JSONL")

rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
if len(rows) != 513 or Counter(row.get("route") for row in rows) != {
    "general_ce": 129,
    "retention_kl": 384,
}:
    raise SystemExit("training route signature drifted")
retention = Counter(row.get("task") for row in rows if row.get("route") == "retention_kl")
expected_retention = {
    "material_desc2sid": 48,
    "material_sid2desc": 48,
    "action": 48,
    "topic": 48,
    "rec_video": 48,
    "rec_prod": 48,
    "rec_ad": 48,
    "rec_living": 48,
}
if retention != expected_retention:
    raise SystemExit(f"retention quota drifted: {retention}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("training_data_sha256")
    != "87097135eb7ddb866b78ae6427c24b8cc2712f898c892d7da92d58ff7e9fddd2"
    or len(manifest.get("general_ce_target_sha256", [])) != 129
    or len(manifest.get("retention_kl_target_sha256", [])) != 384
    or manifest.get("cross_route_target_sha256_collisions") != 0
):
    raise SystemExit("route manifest contract drifted")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
training = audit.get("training_rows", {})
if training.get("rows") != 513 or training.get("route_counts") != {
    "general_ce": 129,
    "retention_kl": 384,
}:
    raise SystemExit("build audit row contract drifted")
token_audit = training.get("qwen3_token_audit", {})
if token_audit.get("status") != "PASS" or token_audit.get("formatted_tokens", {}).get("overflow_rows") != 0:
    raise SystemExit("qwen3 token/cutoff audit did not pass")
if audit.get("formal_training_started") is not False:
    raise SystemExit("build audit no longer records pre-training state")
PY

# Verify the Training Task's own secret/network context. The development SSH
# shell is intentionally irrelevant and is not allowed to force offline mode.
"$python_bin" - <<'PY'
import os
import wandb

if not os.environ.get("WANDB_API_KEY"):
    raise SystemExit("WANDB_API_KEY is missing")
if not wandb.login(key=os.environ["WANDB_API_KEY"], verify=True):
    raise SystemExit("W&B online key verification failed")
print("[s800-general] W&B online connectivity PASS", flush=True)
PY

export CUDA_VISIBLE_DEVICES=$gpu_id
exec "$python_bin" "$trainer" "$config"
