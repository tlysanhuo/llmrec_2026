#!/usr/bin/env python3
"""Admit I-34 beam ledgers and build the frozen margin/KL training mix.

The development ledger is used only for the preregistered admission decision.
Formal material rows always come from the hash-locked O1-derived train pool;
development rows and beam-generated non-gold candidates are never copied into
the training JSONL.  The only model-derived training metadata is the bounded
hard-negative sidecar consumed by the fail-closed I-34 trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SEED = 19260834
SCHEMA_VERSION = "i34-material-beam-margin-v1"
AUDIT_SCHEMA_VERSION = "i34-material-beam-formal-audit-v1"
POOL_SCHEMA_VERSION = "i34-material-beam-pool-v1"
RUNNER_AUDIT_SCHEMA_VERSION = "i34-material-beam-gap-audit-v1"
TASK = "material_desc2sid"

TRAIN_POOL = ROOT / "logs/data/i34_material_beam_candidate_pool_v1.jsonl"
DEV_POOL = ROOT / "assets/evaluation/holdout/data_i34_material_beam_dev_v1.jsonl"
RETENTION_POOL = ROOT / "logs/data/i34_material_beam_retention_pool_v1.jsonl"
POOL_AUDIT = ROOT / "logs/data/i34_material_beam_pool_v1_audit.json"
TRAIN_LEDGER = ROOT / "logs/data/i34_material_beam_train_ledger_v1.jsonl"
DEV_LEDGER = ROOT / "logs/probe/i34_material_beam_dev_ledger_v1.jsonl"
RUNNER_AUDIT = ROOT / "logs/probe/i34_material_beam_gap_audit_v1.json"

FORMAL_DATA = (
    ROOT / "assets/derived/processed/data_i34_material_beam_margin_retkl_v1.jsonl"
)
FORMAL_SIDECAR = (
    ROOT
    / "assets/derived/processed/data_i34_material_beam_margin_retkl_v1_sidecar.jsonl"
)
FORMAL_AUDIT = ROOT / "logs/data/i34_material_beam_margin_retkl_v1_audit.json"

EXPECTED_FILES: dict[str, tuple[Path, int | None, str]] = {
    "train_pool": (
        TRAIN_POOL,
        1024,
        "cb5500a3485aa5b093e70c3d3c53ac73d4485839f1945bc8d58b5eb3d5c19022",
    ),
    "dev_pool": (
        DEV_POOL,
        256,
        "fec7f5cb5dd642e83addd4d23ec1f7f0c6d3e285960a417e0520d27b6938401c",
    ),
    "retention_pool": (
        RETENTION_POOL,
        384,
        "a1edf44988127be969cd881a2a93dc0a496d91d4f9711cdc85b7041116fb1493",
    ),
    "pool_audit": (
        POOL_AUDIT,
        None,
        "fdbc32a80a4e358a879b1d89cbdcfa11e6eea8a4c47ef99b3a12c194631e48e5",
    ),
}

RUNNER = ROOT / "scripts/eval/generate_i34_material_beam_gap_v1.py"
RUNNER_SHA256 = "0af4943f8a61f33695132bccf97dec311589521a270e1019cff80202301320cc"
POOL_BUILDER_SHA256 = "616398bf058385d950d1281942a1171a4a4e763d935c395b0a0f80dae7b4b663"
MATERIAL_SOURCE_ROWS = 32644
MATERIAL_SOURCE_SHA256 = "13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f"
RETENTION_SOURCE_ROWS = 2048
RETENTION_SOURCE_SHA256 = "7d6a1e4a44238a79dcb0d31384f147c02baea95cd870224e2a6815444f8470fd"
BASE_ARTIFACT_SHA256 = "431cc7546a1813ed21a184974a1ac739139b7bdc4643d04e521d066f6ad20652"
PARENT_ARTIFACT_SHA256 = "3c6b694627803f5121ce2020cb4a32242c8a6f1671ec0e4f811f31579e937ba6"
TEACHER_ARTIFACT_SHA256 = "7c193b8db334fe23a2cc74774b8adbee15ce6ba0a260b3afd3fefbbe3cbbb4f1"
PARENT_ADAPTER_SHA256 = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
PARENT_CONFIG_SHA256 = "78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f"
TEACHER_ADAPTER_SHA256 = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
TEACHER_CONFIG_SHA256 = "b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7"

DOMAIN_ORDER = ("ad", "living", "prod", "video")
DOMAIN_TOKEN_IDS = {"video": 176245, "prod": 176247, "living": 176249, "ad": 176251}
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244
EOS_ID = 151645
BEAM_WIDTH = 64
MAX_HARD_NEGATIVES = 12
MAX_NEGATIVES_PER_DIVERGENCE = 4
FORMAL_MATERIAL_ROWS = 128
FORMAL_RETENTION_ROWS = 384
MIN_TRAIN_GAP = 128
MIN_DEV_GAP = 32
MIN_TRAIN_DOMAIN_GAP = 16
MIN_DEV_DOMAIN_GAP = 4

RETENTION_QUOTAS = {
    "action": 55,
    "topic": 55,
    "rec_video": 55,
    "rec_prod": 55,
    "rec_ad": 55,
    "rec_living": 55,
    "world": 54,
}

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
GOLD_RE = re.compile(
    r"^<\|(video|ad|prod|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)


class ContractError(RuntimeError):
    """Raised when a frozen I-34 input or admission invariant drifts."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_hex64(value: Any, field: str) -> str:
    require(isinstance(value, str) and HEX64_RE.fullmatch(value) is not None, f"invalid {field}")
    return value


def finite_number(value: Any, field: str) -> float:
    require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def read_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON input: {path}")
    if expected_sha256 is not None:
        require(file_sha256(path) == expected_sha256, f"SHA256 drift: {path}")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"invalid JSON input {path}: {error}") from error
    require(isinstance(value, dict), f"JSON input is not an object: {path}")
    return value


def read_jsonl(
    path: Path,
    *,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL input: {path}")
    if expected_sha256 is not None:
        require(file_sha256(path) == expected_sha256, f"SHA256 drift: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
            try:
                value = strict_json_loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ContractError(f"invalid JSON at {path}:{line_number}: {error}") from error
            require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    if expected_rows is not None:
        require(len(rows) == expected_rows, f"row-count drift for {path}: {len(rows)}/{expected_rows}")
    return rows


def core_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required = ("instruction", "input", "output", "history")
    for field in required[:3]:
        require(isinstance(row.get(field), str), f"row {field} must be a string")
    require(isinstance(row.get("history"), list), "row history must be a list")
    return {field: row[field] for field in required}


def source_prompt_sha256(row: Mapping[str, Any]) -> str:
    value = core_row(row)
    return canonical_sha256([value["instruction"], value["input"], value["history"]])


def mode_prompt_sha256(row: Mapping[str, Any]) -> str:
    value = core_row(row)
    text = value["input"].rstrip()
    for suffix in ("/no_think", "/think"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            break
    return canonical_sha256([value["instruction"], text, value["history"]])


def trainer_prompt(row: Mapping[str, Any]) -> str:
    value = core_row(row)
    query = "\n".join(part for part in (value["instruction"], value["input"]) if part)
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


def trainer_prompt_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(trainer_prompt(row).encode("utf-8")).hexdigest()


def answer_body(row: Mapping[str, Any]) -> str:
    output = core_row(row)["output"]
    require(output.count("</think>") == 1 and output.startswith("<think>"), "invalid think framing")
    think, body = output.split("</think>", 1)
    require(think[len("<think>") :].strip() == "", "material think block must be empty")
    return body.strip()


def gold_abc(row: Mapping[str, Any]) -> tuple[str, tuple[str, str, str]]:
    match = GOLD_RE.fullmatch(answer_body(row))
    require(match is not None, "material row must contain exactly one full gold SID")
    assert match is not None
    return match.group(1), (match.group(2), match.group(3), match.group(4))


def expected_gold_tokens(domain: str, abc: Sequence[str], observed: Any) -> list[int]:
    require(
        isinstance(observed, list)
        and len(observed) == 5
        and all(not isinstance(item, bool) and isinstance(item, int) for item in observed),
        "gold_tokens must be five integers",
    )
    tokens = [int(item) for item in observed]
    require(tokens[0] == DOMAIN_TOKEN_IDS[domain], "gold domain token mismatch")
    require(A_LO <= tokens[1] <= A_HI, "gold s_a token outside range")
    require(B_LO <= tokens[2] <= B_HI, "gold s_b token outside range")
    require(C_LO <= tokens[3] <= C_HI, "gold s_c token outside range")
    require(tokens[4] == EOS_ID, "gold EOS token mismatch")
    # OneReason's item tokens are contiguous and the runner also records the
    # textual ABC values.  Equality here catches tokenizer/model drift.
    require(tokens[1] == A_LO + int(abc[0]), "gold s_a text/token mismatch")
    require(tokens[2] == B_LO + int(abc[1]), "gold s_b text/token mismatch")
    require(tokens[3] == C_LO + int(abc[2]), "gold s_c text/token mismatch")
    return tokens


def stable_key(namespace: str, *parts: Any) -> str:
    return canonical_sha256([SEED, namespace, *parts])


def validate_pool_audit(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version") == POOL_SCHEMA_VERSION, "pool audit schema drift")
    builder = value.get("builder")
    require(isinstance(builder, dict), "pool audit builder missing")
    require(builder.get("sha256") == POOL_BUILDER_SHA256, "pool builder hash drift")
    source = value.get("source")
    require(isinstance(source, dict), "pool audit material source missing")
    require(source.get("upstream_asset_id") == "data_seed_teacher_v1", "pool audit material source ID drift")
    require(source.get("rows") == MATERIAL_SOURCE_ROWS, "pool audit material source row drift")
    require(source.get("sha256") == MATERIAL_SOURCE_SHA256, "pool audit material source hash drift")
    outputs = value.get("outputs")
    require(isinstance(outputs, dict), "pool audit outputs missing")
    for audit_key, expected_key in (
        ("candidate", "train_pool"),
        ("development", "dev_pool"),
        ("retention", "retention_pool"),
    ):
        entry = outputs.get(audit_key)
        _, rows, digest = EXPECTED_FILES[expected_key]
        require(isinstance(entry, dict), f"pool audit {audit_key} missing")
        require(entry.get("rows") == rows and entry.get("sha256") == digest, f"pool audit {audit_key} drift")
    cross = value.get("final_cross_intersections")
    require(isinstance(cross, dict) and cross and all(item == 0 for item in cross.values()), "pool audit has forbidden intersections")
    scope = value.get("scope")
    require(isinstance(scope, dict), "pool audit scope missing")
    require(scope.get("model_loaded") is False, "pool audit unexpectedly loaded a model")
    require(scope.get("training_started") is False, "pool audit unexpectedly started training")
    require(scope.get("training_projection_created") is False, "pool audit already created a projection")
    retention = value.get("retention")
    require(isinstance(retention, dict), "pool audit retention source missing")
    require(
        retention.get("source")
        == "assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl",
        "pool audit retention source ID drift",
    )


def validate_material_pool(
    rows: Sequence[dict[str, Any]], role: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    prompts: set[str] = set()
    modes: set[str] = set()
    tokens: set[tuple[str, str, str, str]] = set()
    expected_pool_role = "candidate" if role == "train" else "development"
    for index, row in enumerate(rows, start=1):
        location = f"{role} pool row {index}"
        require(row.get("schema_version") == POOL_SCHEMA_VERSION, f"{location} schema drift")
        require(row.get("pool_role") == expected_pool_role, f"{location} role drift")
        require(row.get("task") == TASK, f"{location} task drift")
        value = core_row(row)
        require(value["history"] == [], f"{location} history must remain empty")
        require(value["input"].rstrip().endswith("/no_think"), f"{location} mode drift")
        domain, abc = gold_abc(value)
        row_hash = canonical_sha256(value)
        source_prompt = source_prompt_sha256(value)
        mode_prompt = mode_prompt_sha256(value)
        require(row.get("row_sha256") == row_hash, f"{location} row hash mismatch")
        require(row.get("prompt_sha256") == source_prompt, f"{location} prompt hash mismatch")
        require(row.get("mode_prompt_sha256") == mode_prompt, f"{location} mode hash mismatch")
        require(row.get("gold_domain") == domain, f"{location} domain mismatch")
        require(
            [row.get("gold_s_a"), row.get("gold_s_b"), row.get("gold_s_c")]
            == [int(value) for value in abc],
            f"{location} gold ABC mismatch",
        )
        require(row.get("prefix_group") == f"{domain}:{abc[0]}:{abc[1]}", f"{location} prefix group mismatch")
        require(row_hash not in result, f"duplicate material row hash: {row_hash}")
        require(source_prompt not in prompts and mode_prompt not in modes, f"duplicate material prompt: {location}")
        full_sid = (domain, *abc)
        require(full_sid not in tokens, f"duplicate material full SID: {location}")
        prompts.add(source_prompt)
        modes.add(mode_prompt)
        tokens.add(full_sid)
        result[row_hash] = row
    return result


def validate_retention_pool(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: Counter[str] = Counter()
    row_hashes: set[str] = set()
    prompts: set[str] = set()
    modes: set[str] = set()
    for index, row in enumerate(rows, start=1):
        location = f"retention pool row {index}"
        require(row.get("schema_version") == POOL_SCHEMA_VERSION, f"{location} schema drift")
        require(row.get("pool_role") == "retention", f"{location} role drift")
        require(row.get("route") == "retention_reference", f"{location} route drift")
        task = row.get("task")
        require(task in RETENTION_QUOTAS, f"{location} task drift")
        value = core_row(row)
        require(value["history"] == [], f"{location} history must remain empty")
        row_hash = canonical_sha256(value)
        prompt = source_prompt_sha256(value)
        mode = mode_prompt_sha256(value)
        require(row.get("source_row_sha256") == row_hash, f"{location} row hash mismatch")
        require(row.get("prompt_sha256") == prompt, f"{location} prompt hash mismatch")
        require(row.get("mode_prompt_sha256") == mode, f"{location} mode hash mismatch")
        require(row_hash not in row_hashes, f"duplicate retention row: {location}")
        require(prompt not in prompts and mode not in modes, f"duplicate retention prompt: {location}")
        row_hashes.add(row_hash)
        prompts.add(prompt)
        modes.add(mode)
        tasks[str(task)] += 1
    require(dict(tasks) == RETENTION_QUOTAS, f"retention quotas drifted: {dict(tasks)}")
    return list(rows)


def validate_runner_audit(
    audit: Mapping[str, Any], train_ledger_path: Path, dev_ledger_path: Path
) -> tuple[str, str]:
    require(audit.get("schema_version") == RUNNER_AUDIT_SCHEMA_VERSION, "runner audit schema drift")
    require(audit.get("status") == "complete", "runner audit is not complete")
    require(audit.get("formal_training_generated") is False, "runner generated formal training unexpectedly")
    require(
        audit.get("selection_definition") == "teacher_full_gold_hit_and_parent_full_gold_miss",
        "runner gap definition drift",
    )
    require(audit.get("script_sha256") == RUNNER_SHA256, "runner audit script hash drift")
    require(file_sha256(RUNNER) == RUNNER_SHA256, "current runner script hash drift")

    inputs = audit.get("input_pools")
    require(isinstance(inputs, dict), "runner input pool audit missing")
    for key, expected_key, route in (
        ("train", "train_pool", "beam_train_pool"),
        ("dev", "dev_pool", "beam_gate_pool"),
    ):
        entry = inputs.get(key)
        _, rows, digest = EXPECTED_FILES[expected_key]
        require(isinstance(entry, dict), f"runner {key} input audit missing")
        require(entry.get("sha256") == digest, f"runner {key} input hash drift")
        require(entry.get("total_rows") == rows and entry.get("selected_rows") == rows, f"runner {key} row count drift")
        require(entry.get("route") == route, f"runner {key} route drift")

    artifacts = audit.get("artifacts")
    require(isinstance(artifacts, dict), "runner artifact audit missing")
    base = artifacts.get("base")
    require(isinstance(base, dict) and base.get("artifact_sha256") == BASE_ARTIFACT_SHA256, "runner base artifact drift")
    adapters = artifacts.get("adapters")
    require(isinstance(adapters, list) and len(adapters) == 2, "runner must contain exactly parent and teacher adapters")
    parent, teacher = adapters
    for entry, rank, model_hash, config_hash, artifact_hash, label in (
        (parent, 96, PARENT_ADAPTER_SHA256, PARENT_CONFIG_SHA256, PARENT_ARTIFACT_SHA256, "parent"),
        (teacher, 64, TEACHER_ADAPTER_SHA256, TEACHER_CONFIG_SHA256, TEACHER_ARTIFACT_SHA256, "teacher"),
    ):
        require(isinstance(entry, dict), f"runner {label} artifact missing")
        require(entry.get("rank") == rank, f"runner {label} rank drift")
        require(entry.get("adapter_model_sha256") == model_hash, f"runner {label} model hash drift")
        require(entry.get("adapter_config_sha256") == config_hash, f"runner {label} config hash drift")
        artifact = entry.get("artifact")
        require(isinstance(artifact, dict) and artifact.get("artifact_sha256") == artifact_hash, f"runner {label} artifact hash drift")
    parent_name = parent.get("name")
    teacher_name = teacher.get("name")
    require(isinstance(parent_name, str) and parent_name, "runner parent name missing")
    require(isinstance(teacher_name, str) and teacher_name and teacher_name != parent_name, "runner teacher name invalid")

    adapter_roles = audit.get("adapters")
    require(isinstance(adapter_roles, dict), "runner adapter roles missing")
    require(adapter_roles.get("parent") == parent_name, "runner parent role drift")
    require(adapter_roles.get("teacher") == teacher_name, "runner teacher role drift")
    require(adapter_roles.get("optional_candidates") == [], "runner optional candidate adapters are forbidden")
    require(adapter_roles.get("distinct_lora_ids_and_names") is True, "runner LoRA identity contract drift")

    runtime = audit.get("runtime")
    require(isinstance(runtime, dict), "runner runtime audit missing")
    expected_runtime = {
        "vllm_version": "0.12.0",
        "dtype": "bfloat16",
        "seed": 42,
        "single_engine": True,
        "native_no_think_required": True,
        "empty_think_required": True,
        "beam_width": BEAM_WIDTH,
        "max_tokens": 3,
        "beam_constraints": {},
        "rank_base": 0,
        "hard_negative_max_total": MAX_HARD_NEGATIVES,
        "hard_negative_max_per_first_divergence": MAX_NEGATIVES_PER_DIVERGENCE,
    }
    for field, expected in expected_runtime.items():
        require(runtime.get(field) == expected, f"runner runtime {field} drift")

    outputs = audit.get("outputs")
    require(isinstance(outputs, dict), "runner output audit missing")
    for key, path, rows in (
        ("train_ledger", train_ledger_path, 1024),
        ("dev_ledger", dev_ledger_path, 256),
    ):
        entry = outputs.get(key)
        require(isinstance(entry, dict), f"runner {key} audit missing")
        require(entry.get("rows") == rows, f"runner {key} row count drift")
        require(Path(str(entry.get("path"))).resolve() == path.resolve(), f"runner {key} path drift")
        require(entry.get("sha256") == file_sha256(path), f"runner {key} hash drift")
    require(outputs.get("formal_train") is None, "runner audit unexpectedly names formal training")
    return parent_name, teacher_name


def first_divergence(gold: Sequence[int], negative: Sequence[int]) -> int:
    require(len(gold) == len(negative) == 3, "ABC triples must have length three")
    for index, (expected, observed) in enumerate(zip(gold, negative)):
        if expected != observed:
            return index
    raise ContractError("hard negative duplicates the gold triple")


def validate_negative(value: Any, gold: Sequence[int], location: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{location} must be an object")
    tokens = value.get("tokens")
    require(
        isinstance(tokens, list)
        and len(tokens) == 3
        and all(not isinstance(item, bool) and isinstance(item, int) for item in tokens),
        f"{location}.tokens must be three integers",
    )
    normalized_tokens = [int(item) for item in tokens]
    require(A_LO <= normalized_tokens[0] <= A_HI, f"{location} s_a outside range")
    require(B_LO <= normalized_tokens[1] <= B_HI, f"{location} s_b outside range")
    require(C_LO <= normalized_tokens[2] <= C_HI, f"{location} s_c outside range")
    abc = value.get("abc")
    require(
        isinstance(abc, list)
        and len(abc) == 3
        and all(isinstance(item, str) and item.isdigit() for item in abc),
        f"{location}.abc must be three numeric strings",
    )
    require(
        normalized_tokens
        == [A_LO + int(abc[0]), B_LO + int(abc[1]), C_LO + int(abc[2])],
        f"{location} text/token mismatch",
    )
    divergence = first_divergence(gold, normalized_tokens)
    require(value.get("first_divergence") == divergence, f"{location} divergence mismatch")
    rank = value.get("parent_beam_rank")
    require(not isinstance(rank, bool) and isinstance(rank, int) and 0 <= rank < BEAM_WIDTH, f"{location} rank invalid")
    parent_score = finite_number(value.get("parent_score"), f"{location}.parent_score")
    result = {
        "tokens": normalized_tokens,
        "first_divergence": divergence,
        "parent_beam_rank": int(rank),
        "parent_score": parent_score,
    }
    if value.get("teacher_score") is not None:
        result["teacher_score"] = finite_number(value.get("teacher_score"), f"{location}.teacher_score")
    return result


def validate_beam_block(
    block: Any,
    *,
    name: str,
    domain: str,
    gold: Sequence[str],
    location: str,
) -> tuple[bool, int, int]:
    require(isinstance(block, dict), f"{location} missing")
    if "name" in block:
        require(block.get("name") == name, f"{location} name drift")
    require(block.get("beam_count") == BEAM_WIDTH, f"{location} beam count drift")
    require(block.get("invalid_count") == 0, f"{location} has invalid candidates")
    require(block.get("invalid_ranks") == [], f"{location} invalid ranks drift")
    candidates = block.get("valid_candidates")
    require(isinstance(candidates, list) and len(candidates) == BEAM_WIDTH, f"{location} valid candidate count drift")
    hit = False
    ranks: set[int] = set()
    for index, candidate in enumerate(candidates):
        item_location = f"{location}.valid_candidates[{index}]"
        require(isinstance(candidate, dict), f"{item_location} must be an object")
        abc = candidate.get("abc")
        require(isinstance(abc, list) and len(abc) == 3 and all(isinstance(item, str) and item.isdigit() for item in abc), f"{item_location}.abc invalid")
        require(all(0 <= int(item) <= 8191 for item in abc), f"{item_location}.abc outside range")
        rank = candidate.get("rank")
        require(not isinstance(rank, bool) and isinstance(rank, int) and 0 <= rank < BEAM_WIDTH, f"{item_location}.rank invalid")
        require(rank not in ranks, f"{location} duplicate beam rank")
        ranks.add(rank)
        finite_number(candidate.get("cum_logprob"), f"{item_location}.cum_logprob")
        text = candidate.get("text")
        require(isinstance(text, str) and GOLD_RE.fullmatch(text) is not None, f"{item_location}.text invalid")
        text_match = GOLD_RE.fullmatch(text)
        assert text_match is not None
        require(text_match.group(1) == domain, f"{item_location} domain mismatch")
        require(list(text_match.groups()[1:]) == abc, f"{item_location} text/ABC mismatch")
        require(candidate.get("token_count") == 3, f"{item_location}.token_count must be three")
        hit = hit or tuple(abc) == tuple(gold)
    require(ranks == set(range(BEAM_WIDTH)), f"{location} ranks are not 0..63")
    require(block.get("full_gold_hit") is hit, f"{location} full-gold flag mismatch")
    return hit, 0, len(candidates)


def validate_ledger(
    rows: Sequence[dict[str, Any]],
    pool_by_hash: Mapping[str, dict[str, Any]],
    *,
    route: str,
    parent_name: str,
    teacher_name: str,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(len(rows) == len(pool_by_hash), f"{label} ledger/pool row count mismatch")
    seen_rows: set[str] = set()
    seen_prompt_tokens: set[str] = set()
    normalized: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "rows": len(rows),
        "gap_selected": 0,
        "trainer_ready": 0,
        "parent_full_gold_hits": 0,
        "teacher_full_gold_hits": 0,
        "parent_invalid_candidates": 0,
        "teacher_invalid_candidates": 0,
        "parent_valid_candidates": 0,
        "teacher_valid_candidates": 0,
        "hard_negatives": 0,
        "hard_negative_pool": 0,
        "hard_negative_dropped": 0,
    }
    for index, row in enumerate(rows, start=1):
        location = f"{label} ledger row {index}"
        require(row.get("schema_version") == SCHEMA_VERSION, f"{location} schema drift")
        require(row.get("task") == TASK and row.get("route") == route, f"{location} task/route drift")
        row_hash = require_hex64(row.get("row_sha256"), f"{location}.row_sha256")
        require(row_hash in pool_by_hash, f"{location} is absent from the locked pool")
        require(row_hash not in seen_rows, f"duplicate {label} ledger row")
        seen_rows.add(row_hash)
        pool_row = pool_by_hash[row_hash]
        domain, abc = gold_abc(pool_row)
        require(row.get("domain") == domain, f"{location} domain drift")
        require(row.get("gold_abc") == list(abc), f"{location} gold ABC drift")
        require(row.get("source_prompt_sha256") == pool_row.get("prompt_sha256"), f"{location} source prompt drift")
        require(row.get("source_mode_prompt_sha256") == pool_row.get("mode_prompt_sha256"), f"{location} source mode drift")
        require(row.get("prompt_sha256") == trainer_prompt_sha256(pool_row), f"{location} trainer prompt drift")
        prompt_token_hash = require_hex64(row.get("prompt_token_sha256"), f"{location}.prompt_token_sha256")
        require(prompt_token_hash not in seen_prompt_tokens, f"duplicate {label} prompt token hash")
        seen_prompt_tokens.add(prompt_token_hash)
        require(not isinstance(row.get("prompt_token_count"), bool) and isinstance(row.get("prompt_token_count"), int) and row["prompt_token_count"] > 0, f"{location} prompt token count invalid")
        require_hex64(row.get("renderer_prompt_sha256"), f"{location}.renderer_prompt_sha256")
        require(row.get("parent_adapter_sha256") == PARENT_ADAPTER_SHA256, f"{location} parent hash drift")
        require(row.get("teacher_adapter_sha256") == TEACHER_ADAPTER_SHA256, f"{location} teacher hash drift")
        gold_tokens = expected_gold_tokens(domain, abc, row.get("gold_tokens"))
        expected_positive = [gold_tokens[1:4]]
        require(row.get("positive_tokens") == expected_positive, f"{location} positive set is not gold-only")

        parent_hit, parent_invalid, parent_valid = validate_beam_block(
            row.get("parent"), name=parent_name, domain=domain, gold=abc, location=f"{location}.parent"
        )
        teacher_hit, teacher_invalid, teacher_valid = validate_beam_block(
            row.get("teacher"), name=teacher_name, domain=domain, gold=abc, location=f"{location}.teacher"
        )
        candidate_results = row.get("candidate_results")
        require(isinstance(candidate_results, dict) and set(candidate_results) == {parent_name, teacher_name}, f"{location} candidate result names drift")
        validate_beam_block(candidate_results[parent_name], name=parent_name, domain=domain, gold=abc, location=f"{location}.candidate_parent")
        validate_beam_block(candidate_results[teacher_name], name=teacher_name, domain=domain, gold=abc, location=f"{location}.candidate_teacher")
        parent_compact = dict(row["parent"])
        teacher_compact = dict(row["teacher"])
        parent_compact.pop("name", None)
        teacher_compact.pop("name", None)
        require(candidate_results[parent_name] == parent_compact, f"{location} duplicate parent block disagrees")
        require(candidate_results[teacher_name] == teacher_compact, f"{location} duplicate teacher block disagrees")

        raw_negatives = row.get("hard_negatives")
        require(isinstance(raw_negatives, list), f"{location} hard negatives missing")
        require(len(raw_negatives) <= MAX_HARD_NEGATIVES, f"{location} has too many hard negatives")
        negatives: list[dict[str, Any]] = []
        negative_tokens: set[tuple[int, int, int]] = set()
        divergence_counts: Counter[int] = Counter()
        for neg_index, negative in enumerate(raw_negatives):
            clean = validate_negative(negative, gold_tokens[1:4], f"{location}.hard_negatives[{neg_index}]")
            token_tuple = tuple(clean["tokens"])
            require(token_tuple not in negative_tokens, f"{location} has duplicate hard negatives")
            negative_tokens.add(token_tuple)
            divergence_counts[clean["first_divergence"]] += 1
            require(divergence_counts[clean["first_divergence"]] <= MAX_NEGATIVES_PER_DIVERGENCE, f"{location} has too many negatives at one divergence")
            negatives.append(clean)

        gap = row.get("gap_selection")
        require(isinstance(gap, dict), f"{location} gap selection missing")
        selected = teacher_hit and not parent_hit
        trainer_ready = selected and bool(negatives)
        require(gap.get("definition") == "teacher_full_gold_hit_and_parent_full_gold_miss", f"{location} gap definition drift")
        require(gap.get("selected") is selected, f"{location} selected flag drift")
        require(gap.get("trainer_ready") is trainer_ready, f"{location} trainer-ready flag drift")
        require(gap.get("hard_negative_count") == len(negatives), f"{location} hard-negative count drift")
        pool_count = gap.get("hard_negative_pool_count")
        dropped_count = gap.get("hard_negative_dropped_count")
        require(not isinstance(pool_count, bool) and isinstance(pool_count, int) and pool_count >= len(negatives), f"{location} negative pool count invalid")
        require(dropped_count == pool_count - len(negatives), f"{location} negative dropped count invalid")
        require(row.get("formal_training_generated") is False, f"{location} formal flag drift")

        counts["gap_selected"] += int(selected)
        counts["trainer_ready"] += int(trainer_ready)
        counts["parent_full_gold_hits"] += int(parent_hit)
        counts["teacher_full_gold_hits"] += int(teacher_hit)
        counts["parent_invalid_candidates"] += parent_invalid
        counts["teacher_invalid_candidates"] += teacher_invalid
        counts["parent_valid_candidates"] += parent_valid
        counts["teacher_valid_candidates"] += teacher_valid
        counts["hard_negatives"] += len(negatives)
        counts["hard_negative_pool"] += pool_count
        counts["hard_negative_dropped"] += dropped_count
        normalized.append(
            {
                **row,
                "_pool_row": pool_row,
                "_gold_tokens": gold_tokens,
                "_hard_negatives": negatives,
                "_selected": selected,
                "_trainer_ready": trainer_ready,
            }
        )
    require(seen_rows == set(pool_by_hash), f"{label} ledger does not cover the locked pool")
    return normalized, counts


def admission_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["_selected"]]
    ready = [row for row in rows if row["_trainer_ready"]]
    return {
        "gap_selected": len(selected),
        "trainer_ready": len(ready),
        "gap_selected_by_domain": dict(sorted(Counter(str(row["domain"]) for row in selected).items())),
        "trainer_ready_by_domain": dict(sorted(Counter(str(row["domain"]) for row in ready).items())),
    }


def apply_admission_gate(
    train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    train = admission_counts(train_rows)
    dev = admission_counts(dev_rows)
    failures: list[str] = []
    if train["gap_selected"] < MIN_TRAIN_GAP:
        failures.append(f"train gap {train['gap_selected']} < {MIN_TRAIN_GAP}")
    if train["trainer_ready"] < MIN_TRAIN_GAP:
        failures.append(f"train trainer-ready {train['trainer_ready']} < {MIN_TRAIN_GAP}")
    if dev["gap_selected"] < MIN_DEV_GAP:
        failures.append(f"dev gap {dev['gap_selected']} < {MIN_DEV_GAP}")
    for domain in DOMAIN_ORDER:
        if train["gap_selected_by_domain"].get(domain, 0) < MIN_TRAIN_DOMAIN_GAP:
            failures.append(f"train {domain} gap < {MIN_TRAIN_DOMAIN_GAP}")
        if train["trainer_ready_by_domain"].get(domain, 0) < MIN_TRAIN_DOMAIN_GAP:
            failures.append(f"train {domain} trainer-ready < {MIN_TRAIN_DOMAIN_GAP}")
        if dev["gap_selected_by_domain"].get(domain, 0) < MIN_DEV_DOMAIN_GAP:
            failures.append(f"dev {domain} gap < {MIN_DEV_DOMAIN_GAP}")
    report = {
        "status": "pass" if not failures else "fail",
        "thresholds": {
            "train_gap_min": MIN_TRAIN_GAP,
            "train_trainer_ready_min": MIN_TRAIN_GAP,
            "dev_gap_min": MIN_DEV_GAP,
            "train_each_domain_gap_and_trainer_ready_min": MIN_TRAIN_DOMAIN_GAP,
            "dev_each_domain_gap_min": MIN_DEV_DOMAIN_GAP,
            "invalid_candidate_count_max": 0,
            "schema_or_runtime_error_count_max": 0,
        },
        "train": train,
        "dev": dev,
        "invalid_candidate_count": 0,
        "schema_or_runtime_error_count": 0,
        "failures": failures,
    }
    require(not failures, "I34 admission failed: " + "; ".join(failures))
    return report


def select_material(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAIN_ORDER:
        eligible = [row for row in rows if row["_trainer_ready"] and row.get("domain") == domain]
        eligible.sort(key=lambda row: stable_key("formal-material-selection", domain, row["row_sha256"]))
        require(len(eligible) >= MIN_TRAIN_DOMAIN_GAP, f"not enough trainer-ready {domain} rows")
        by_domain[domain] = eligible

    selected: list[dict[str, Any]] = []
    offsets: dict[str, int] = {}
    for domain in DOMAIN_ORDER:
        selected.extend(by_domain[domain][:MIN_TRAIN_DOMAIN_GAP])
        offsets[domain] = MIN_TRAIN_DOMAIN_GAP
    while len(selected) < FORMAL_MATERIAL_ROWS:
        progressed = False
        for domain in DOMAIN_ORDER:
            offset = offsets[domain]
            if offset < len(by_domain[domain]):
                selected.append(by_domain[domain][offset])
                offsets[domain] = offset + 1
                progressed = True
                if len(selected) == FORMAL_MATERIAL_ROWS:
                    break
        require(progressed, "trainer-ready rows exhausted before selecting 128")
    require(len({row["row_sha256"] for row in selected}) == FORMAL_MATERIAL_ROWS, "formal material selection duplicated rows")
    return selected


def formal_training_row(row: Mapping[str, Any], route: str, task: str) -> dict[str, Any]:
    return {**core_row(row), "route": route, "task": task}


def sidecar_row(row: Mapping[str, Any]) -> dict[str, Any]:
    gold = list(row["_gold_tokens"])
    negatives = list(row["_hard_negatives"])
    require(0 < len(negatives) <= MAX_HARD_NEGATIVES, "selected row has no bounded hard negatives")
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "prompt_token_sha256": row["prompt_token_sha256"],
        "prompt_sha256": row["prompt_sha256"],
        "row_sha256": row["row_sha256"],
        "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        "teacher_adapter_sha256": TEACHER_ADAPTER_SHA256,
        "domain": row["domain"],
        "gold_tokens": gold,
        "positive_tokens": [gold[1:4]],
        "hard_negatives": negatives,
    }


def encoded_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def encoded_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_new_files(payloads: Sequence[tuple[Path, bytes]]) -> None:
    paths = [path.resolve() for path, _ in payloads]
    require(len(paths) == len(set(paths)), "output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    require(not existing, "refusing to overwrite existing output: " + ", ".join(existing))
    temporary: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        for path, payload in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            temp_path = Path(temp_name)
            temporary.append((temp_path, path))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for temp_path, path in temporary:
            os.link(temp_path, path)
            created.append(path)
        for temp_path, _ in temporary:
            temp_path.unlink()
    except Exception:
        for temp_path, _ in temporary:
            temp_path.unlink(missing_ok=True)
        # A collision after the precheck is treated as a failed transaction;
        # files linked by this call are removed, but pre-existing files never are.
        for path in created:
            path.unlink(missing_ok=True)
        raise


def intersection_report(
    selected: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    material_pool = [row["_pool_row"] for row in selected]
    dev_pool = [row["_pool_row"] for row in dev_rows]

    def hashes(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], set[str]]:
        return (
            {canonical_sha256(core_row(row)) for row in rows},
            {source_prompt_sha256(row) for row in rows},
            {mode_prompt_sha256(row) for row in rows},
        )

    material_core, material_prompt, material_mode = hashes(material_pool)
    dev_core, dev_prompt, dev_mode = hashes(dev_pool)
    retention_core, retention_prompt, retention_mode = hashes(retention_rows)
    material_groups = {row["_pool_row"]["prefix_group"] for row in selected}
    dev_groups = {row["_pool_row"]["prefix_group"] for row in dev_rows}
    material_tokens = {row["prompt_token_sha256"] for row in selected}
    dev_tokens = {row["prompt_token_sha256"] for row in dev_rows}
    return {
        "formal_material_vs_dev_core_row": len(material_core & dev_core),
        "formal_material_vs_dev_prompt": len(material_prompt & dev_prompt),
        "formal_material_vs_dev_mode_prompt": len(material_mode & dev_mode),
        "formal_material_vs_dev_prefix_group": len(material_groups & dev_groups),
        "formal_material_vs_dev_prompt_token": len(material_tokens & dev_tokens),
        "formal_material_vs_retention_core_row": len(material_core & retention_core),
        "formal_material_vs_retention_prompt": len(material_prompt & retention_prompt),
        "formal_material_vs_retention_mode_prompt": len(material_mode & retention_mode),
        "dev_vs_retention_core_row": len(dev_core & retention_core),
        "dev_vs_retention_prompt": len(dev_prompt & retention_prompt),
        "dev_vs_retention_mode_prompt": len(dev_mode & retention_mode),
    }


def load_and_admit(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "train_pool": Path(args.train_pool),
        "dev_pool": Path(args.dev_pool),
        "retention_pool": Path(args.retention_pool),
        "pool_audit": Path(args.pool_audit),
        "train_ledger": Path(args.train_ledger),
        "dev_ledger": Path(args.dev_ledger),
        "runner_audit": Path(args.runner_audit),
    }
    # v1 is tied to one frozen path and content set.  Custom paths are allowed
    # only as aliases to those exact bytes; their hashes remain non-negotiable.
    train_pool_rows = read_jsonl(paths["train_pool"], expected_rows=1024, expected_sha256=EXPECTED_FILES["train_pool"][2])
    dev_pool_rows = read_jsonl(paths["dev_pool"], expected_rows=256, expected_sha256=EXPECTED_FILES["dev_pool"][2])
    retention_pool_rows = read_jsonl(paths["retention_pool"], expected_rows=384, expected_sha256=EXPECTED_FILES["retention_pool"][2])
    pool_audit = read_json(paths["pool_audit"], EXPECTED_FILES["pool_audit"][2])
    validate_pool_audit(pool_audit)
    train_pool = validate_material_pool(train_pool_rows, "train")
    dev_pool = validate_material_pool(dev_pool_rows, "dev")
    retention = validate_retention_pool(retention_pool_rows)

    require(set(train_pool).isdisjoint(dev_pool), "train/dev pool row overlap")
    train_groups = {row["prefix_group"] for row in train_pool.values()}
    dev_groups = {row["prefix_group"] for row in dev_pool.values()}
    require(train_groups.isdisjoint(dev_groups), "train/dev prefix group overlap")

    runner_audit = read_json(paths["runner_audit"])
    parent_name, teacher_name = validate_runner_audit(
        runner_audit, paths["train_ledger"], paths["dev_ledger"]
    )
    outputs = runner_audit["outputs"]
    train_ledger_rows = read_jsonl(
        paths["train_ledger"],
        expected_rows=1024,
        expected_sha256=outputs["train_ledger"]["sha256"],
    )
    dev_ledger_rows = read_jsonl(
        paths["dev_ledger"],
        expected_rows=256,
        expected_sha256=outputs["dev_ledger"]["sha256"],
    )
    train_ledger, train_counts = validate_ledger(
        train_ledger_rows,
        train_pool,
        route="beam_train_pool",
        parent_name=parent_name,
        teacher_name=teacher_name,
        label="train",
    )
    dev_ledger, dev_counts = validate_ledger(
        dev_ledger_rows,
        dev_pool,
        route="beam_gate_pool",
        parent_name=parent_name,
        teacher_name=teacher_name,
        label="dev",
    )
    audit_counts = runner_audit.get("counts")
    require(isinstance(audit_counts, dict), "runner count audit missing")
    require(audit_counts.get("train") == train_counts, "runner train counts disagree with ledger")
    require(audit_counts.get("dev") == dev_counts, "runner dev counts disagree with ledger")
    admission = apply_admission_gate(train_ledger, dev_ledger)
    return {
        "paths": paths,
        "runner_audit": runner_audit,
        "train": train_ledger,
        "dev": dev_ledger,
        "retention": retention,
        "runner_counts": {"train": train_counts, "dev": dev_counts},
        "admission": admission,
    }


def admission_report(state: Mapping[str, Any]) -> dict[str, Any]:
    paths = state["paths"]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "admission_only",
        "status": "admitted",
        "formal_training_generated": False,
        "seed": SEED,
        "locks": {
            "train_pool_sha256": file_sha256(paths["train_pool"]),
            "dev_pool_sha256": file_sha256(paths["dev_pool"]),
            "retention_pool_sha256": file_sha256(paths["retention_pool"]),
            "pool_audit_sha256": file_sha256(paths["pool_audit"]),
            "train_ledger_sha256": file_sha256(paths["train_ledger"]),
            "dev_ledger_sha256": file_sha256(paths["dev_ledger"]),
            "runner_audit_sha256": file_sha256(paths["runner_audit"]),
            "runner_sha256": RUNNER_SHA256,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "teacher_adapter_sha256": TEACHER_ADAPTER_SHA256,
        },
        "runner_counts": state["runner_counts"],
        "admission": state["admission"],
    }


def build_formal(state: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selected = select_material(state["train"])
    retention = list(state["retention"])
    cross = intersection_report(selected, state["dev"], retention)
    require(all(value == 0 for value in cross.values()), f"formal cross-intersection is nonzero: {cross}")

    material_rows = [formal_training_row(row["_pool_row"], "material_margin", TASK) for row in selected]
    retention_formal = [formal_training_row(row, "retention", str(row["task"])) for row in retention]
    tagged: list[tuple[str, str, dict[str, Any]]] = []
    for selected_row, formal_row in zip(selected, material_rows):
        tagged.append(("material_margin", selected_row["row_sha256"], formal_row))
    for pool_row, formal_row in zip(retention, retention_formal):
        tagged.append(("retention", str(pool_row["source_row_sha256"]), formal_row))
    tagged.sort(key=lambda item: stable_key("formal-fixed-seed-shuffle", item[0], item[1]))
    training_rows = [item[2] for item in tagged]
    sidecar_rows = [sidecar_row(row) for row in selected]
    sidecar_rows.sort(key=lambda row: row["prompt_token_sha256"])

    require(len(training_rows) == 512, "formal training row count drift")
    require(Counter(row["route"] for row in training_rows) == {"material_margin": 128, "retention": 384}, "formal route mix drift")
    require(len(sidecar_rows) == 128, "formal sidecar count drift")
    sidecar_keys = {row["prompt_token_sha256"] for row in sidecar_rows}
    require(sidecar_keys == {row["prompt_token_sha256"] for row in selected}, "sidecar key set drift")

    data_payload = encoded_jsonl(training_rows)
    sidecar_payload = encoded_jsonl(sidecar_rows)
    output_data = Path(args.output)
    output_sidecar = Path(args.sidecar_output)
    output_audit = Path(args.audit_output)
    paths = state["paths"]
    selected_by_domain = dict(sorted(Counter(str(row["domain"]) for row in selected).items()))
    retention_by_task = dict(sorted(Counter(str(row["task"]) for row in retention).items()))
    selection_ledger = [
        {
            "selection_order": index,
            "domain": row["domain"],
            "row_sha256": row["row_sha256"],
            "source_prompt_sha256": row["source_prompt_sha256"],
            "source_mode_prompt_sha256": row["source_mode_prompt_sha256"],
            "prompt_token_sha256": row["prompt_token_sha256"],
            "gold_abc": row["gold_abc"],
            "stable_selection_key": stable_key("formal-material-selection", row["domain"], row["row_sha256"]),
            "hard_negative_count": len(row["_hard_negatives"]),
        }
        for index, row in enumerate(selected, start=1)
    ]
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "formal_built",
        "seed": SEED,
        "builder": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "classification": "D(O1; M-I19/I23 beam construction filter) plus frozen D retention",
        "upstreams": {
            "material_gold_source": {
                "asset_id": "data_seed_teacher_v1",
                "class": "D(O1)",
                "path": "assets/derived/processed/data_seed_teacher_v1.jsonl",
                "rows": MATERIAL_SOURCE_ROWS,
                "sha256": MATERIAL_SOURCE_SHA256,
                "formal_rows_are_unchanged_material_desc2sid_gold": True,
            },
            "material_selection_pool": {
                "asset_id": "i34_material_beam_candidate_pool_v1",
                "class": "D(O1) candidate; M filters only select rows",
                "selection_pool_path": str(paths["train_pool"].resolve().relative_to(ROOT)),
                "selection_pool_rows": 1024,
                "selection_pool_sha256": file_sha256(paths["train_pool"]),
            },
            "development_gate": {
                "class": "E(D(O1))",
                "path": str(paths["dev_pool"].resolve().relative_to(ROOT)),
                "rows": 256,
                "sha256": file_sha256(paths["dev_pool"]),
                "copied_to_training_rows": 0,
            },
            "retention": {
                "asset_id": "i34_material_beam_retention_pool_v1",
                "class": "D(O1,O2.* including General), frozen I33 E-clean retention",
                "path": str(paths["retention_pool"].resolve().relative_to(ROOT)),
                "rows": 384,
                "sha256": file_sha256(paths["retention_pool"]),
                "parent_asset_id": "data_i33_r96_material_desc2sid_retkl_v1",
                "parent_path": "assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl",
                "parent_rows": RETENTION_SOURCE_ROWS,
                "parent_sha256": RETENTION_SOURCE_SHA256,
            },
            "beam_ledgers": {
                "train_path": str(paths["train_ledger"].resolve().relative_to(ROOT)),
                "train_rows": 1024,
                "train_sha256": file_sha256(paths["train_ledger"]),
                "dev_path": str(paths["dev_ledger"].resolve().relative_to(ROOT)),
                "dev_rows": 256,
                "dev_sha256": file_sha256(paths["dev_ledger"]),
                "runner_audit_path": str(paths["runner_audit"].resolve().relative_to(ROOT)),
                "runner_audit_sha256": file_sha256(paths["runner_audit"]),
                "runner_sha256": RUNNER_SHA256,
                "pool_builder_sha256": POOL_BUILDER_SHA256,
            },
            "model_filters": {
                "parent_i19_adapter_sha256": PARENT_ADAPTER_SHA256,
                "parent_i19_config_sha256": PARENT_CONFIG_SHA256,
                "teacher_i23_adapter_sha256": TEACHER_ADAPTER_SHA256,
                "teacher_i23_config_sha256": TEACHER_CONFIG_SHA256,
                "non_gold_teacher_beams_used_as_positive": 0,
            },
        },
        "admission": state["admission"],
        "runner_counts": state["runner_counts"],
        "selection": {
            "definition": "train trainer_ready rows only",
            "score_sorting": False,
            "stable_hash_seed": SEED,
            "algorithm": "hash-sort within domain; take 16/domain; then round-robin ad,living,prod,video until 128",
            "domain_order": list(DOMAIN_ORDER),
            "rows": len(selected),
            "by_domain": selected_by_domain,
            "ledger": selection_ledger,
        },
        "mix": {
            "total_rows": 512,
            "material_margin": {"rows": 128, "ratio": 0.25, "by_domain": selected_by_domain},
            "retention": {"rows": 384, "ratio": 0.75, "by_task": retention_by_task},
            "fixed_seed_hash_shuffle": True,
            "dataset_exposures": 1,
        },
        "sidecar_contract": {
            "rows": len(sidecar_rows),
            "positive_definition": "the one O1 full-gold SID only",
            "positive_rows_per_prompt": 1,
            "hard_negative_source": "complete wrong r96 beam triples only",
            "hard_negative_max_total": MAX_HARD_NEGATIVES,
            "hard_negative_max_per_first_divergence": MAX_NEGATIVES_PER_DIVERGENCE,
            "all_emitted_scores_finite": True,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "teacher_adapter_sha256": TEACHER_ADAPTER_SHA256,
        },
        "intersections": cross,
        "forbidden_sources": {
            "third_party_rows": 0,
            "development_or_other_E_training_rows": 0,
            "model_generated_positive_rows": 0,
            "teacher_non_gold_positive_rows": 0,
        },
        "outputs": {
            "training_data": {
                "path": str(output_data.resolve().relative_to(ROOT)),
                "rows": len(training_rows),
                "bytes": len(data_payload),
                "sha256": bytes_sha256(data_payload),
            },
            "sidecar": {
                "path": str(output_sidecar.resolve().relative_to(ROOT)),
                "rows": len(sidecar_rows),
                "bytes": len(sidecar_payload),
                "sha256": bytes_sha256(sidecar_payload),
            },
        },
    }
    audit_payload = encoded_json(audit)
    write_new_files(
        (
            (output_data, data_payload),
            (output_sidecar, sidecar_payload),
            (output_audit, audit_payload),
        )
    )
    return audit


def self_test() -> None:
    gold = [A_LO + 1, B_LO + 2, C_LO + 3]
    clean = validate_negative(
        {
            "tokens": [A_LO + 4, B_LO + 2, C_LO + 3],
            "abc": ["4", "2", "3"],
            "first_divergence": 0,
            "parent_beam_rank": 1,
            "parent_score": -1.25,
            "teacher_score": -2.5,
        },
        gold,
        "self-test",
    )
    require(clean["first_divergence"] == 0, "negative self-test failed")
    try:
        finite_number(float("nan"), "self-test")
    except ContractError:
        pass
    else:
        raise AssertionError("non-finite score self-test failed")

    synthetic: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        for index in range(40):
            synthetic.append(
                {
                    "domain": domain,
                    "row_sha256": hashlib.sha256(f"{domain}:{index}".encode()).hexdigest(),
                    "_trainer_ready": True,
                }
            )
    selected = select_material(synthetic)
    require(Counter(row["domain"] for row in selected) == Counter({domain: 32 for domain in DOMAIN_ORDER}), "round-robin selection self-test failed")
    require(
        [row["row_sha256"] for row in selected]
        == [row["row_sha256"] for row in select_material(list(reversed(synthetic)))],
        "selection depends on input order",
    )

    payload = encoded_jsonl([{"value": 1}])
    with tempfile.TemporaryDirectory(prefix="i34_formal_builder_test_") as directory:
        path = Path(directory) / "new.jsonl"
        write_new_files(((path, payload),))
        require(path.read_bytes() == payload, "new-file writer self-test failed")
        try:
            write_new_files(((path, payload),))
        except ContractError:
            pass
        else:
            raise AssertionError("writer overwrite self-test failed")
    print("[i34-formal-builder] self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--admission-report", type=Path)
    parser.add_argument("--train-pool", type=Path, default=TRAIN_POOL)
    parser.add_argument("--dev-pool", type=Path, default=DEV_POOL)
    parser.add_argument("--retention-pool", type=Path, default=RETENTION_POOL)
    parser.add_argument("--pool-audit", type=Path, default=POOL_AUDIT)
    parser.add_argument("--train-ledger", type=Path, default=TRAIN_LEDGER)
    parser.add_argument("--dev-ledger", type=Path, default=DEV_LEDGER)
    parser.add_argument("--runner-audit", type=Path, default=RUNNER_AUDIT)
    parser.add_argument("--output", type=Path, default=FORMAL_DATA)
    parser.add_argument("--sidecar-output", type=Path, default=FORMAL_SIDECAR)
    parser.add_argument("--audit-output", type=Path, default=FORMAL_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        require(not args.admission_only and args.admission_report is None, "--self-test cannot be combined with admission options")
        self_test()
        return
    if args.admission_report is not None:
        require(args.admission_only, "--admission-report requires --admission-only")
    state = load_and_admit(args)
    if args.admission_only:
        report = admission_report(state)
        if args.admission_report is not None:
            write_new_files(((args.admission_report, encoded_json(report)),))
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return
    audit = build_formal(state, args)
    print(json.dumps(audit["outputs"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
