#!/usr/bin/env python3
"""Score the fixed I29 I-23/s800 x legacy/canonical renderer grid.

The GPU generator is intentionally label-blind.  This CPU-only companion is the
first point where the four rollout cells are joined with the frozen first-16
multi-gold ledger.  It validates the complete 16 x 4 x 8 shape and reports
longest-contiguous-prefix (LCP) matches against every non-history gold in each
group.

``--preflight`` only reads the frozen gold ledger and the already-existing
s800 x legacy rollout.  It must reproduce the preregistered historical counts
before any new cell is trusted.  Normal scoring never overwrites an existing
report and publishes through an atomic same-directory hard link.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = (
    PROJECT_ROOT
    / "assets/derived/processed/o1_rec_multigold_v1_gold_ledger.jsonl"
)
DEFAULT_S800_LEGACY = (
    PROJECT_ROOT / "assets/derived/processed/o1_rec_multigold_v1_rollouts.jsonl"
)
DEFAULT_S800_CANONICAL = (
    PROJECT_ROOT / "logs/probe/i29_renderer_s800_canonical_n16.jsonl"
)
DEFAULT_I23_LEGACY = PROJECT_ROOT / "logs/probe/i29_renderer_i23_legacy_n16.jsonl"
DEFAULT_I23_CANONICAL = (
    PROJECT_ROOT / "logs/probe/i29_renderer_i23_canonical_n16.jsonl"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "logs/probe/i29_i23_s800_renderer_calibration_n16.json"
)

GROUP_COUNT = 16
REASONING_SAMPLES = 4
ITEM_CANDIDATES = 8
GLOBAL_SEED = 19260829

GOLD_SCHEMA = "o1-rec-gold-ledger-v1"
ROLLOUT_SCHEMA = "o1-rec-rollouts-v1"
REPORT_SCHEMA = "i29-renderer-calibration-report-v1"

GOLD_FULL_SHA256 = "ec3f39054e4ba1d3e4a476ff8deea3b057cd52eb4a28d07d1237962ab7081cf5"
GOLD_FIRST16_RAW_SHA256 = (
    "d300cf7db5c3ff335133de89a49352ecefbee5d35b2eeef4281bb2f60986d04c"
)
S800_LEGACY_FULL_SHA256 = (
    "c3dfe9bc5a2dbfb3161a9aa0d241692b1ad0616e7f954e096e9dd5caf4198fac"
)
S800_LEGACY_FIRST16_RAW_SHA256 = (
    "53df39ccaaa0946e32bd6e16951feed25b7a6bb73a0e7eb5af94f4fd7e04884f"
)
S800_LEGACY_CONFIG_FILE_SHA256 = (
    "c72fce0cf5b2c7b54d7bf632a33bd97d6b2501a298a3c11b6d5df1bfb47b7513"
)
BASE_ARTIFACT_SHA256 = (
    "431cc7546a1813ed21a184974a1ac739139b7bdc4643d04e521d066f6ad20652"
)
S800_ARTIFACT_SHA256 = (
    "ed5366a6c38a3e4da3c90970d243bd1b0f86fe7aad3ea08074fd7f32c2633c51"
)
I23_ARTIFACT_SHA256 = (
    "7c193b8db334fe23a2cc74774b8adbee15ce6ba0a260b3afd3fefbbe3cbbb4f1"
)
LEGACY_GENERATOR_SHA256 = (
    "668a7e09b1460bb57e80baa6bcbfc28af50e2d99352784294805b1a4c5fa8c0d"
)
MANIFEST_FULL_SHA256 = (
    "c75e6a326dd02da07b671787a0bbc76cc391c0ec1254a7eaed0fa1cc250d0300"
)
MANIFEST_FIRST16_RAW_SHA256 = (
    "b3d7300f57847ce1a5e83e8ca438e167df00d80f720034278d3ce0280c2f5a57"
)
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/eval/generate_i29_renderer_pair.py"
LEGACY_GENERATOR_SCRIPT = PROJECT_ROOT / "scripts/rft/generate_rec_rft_rollouts.py"
NEW_CONFIG_SCHEMA = "i29-renderer-calibration-cell-config-v1"
NEW_CONFIG_PROTOCOL = "i29-i23-s800-renderer-calibration-n16-v1"
NEW_METADATA_SCHEMA = "i29-renderer-calibration-cell-metadata-v1"

EXPECTED_DECODE = {
    "reasoning": {
        "samples": 4,
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 50,
        "stop": ["</think>"],
        "seed": GLOBAL_SEED,
        "per_prompt_seed": "(seed + rollout_seed) mod 2**31",
    },
    "item_beam": {
        "beam_width": 8,
        "max_tokens": 3,
        "temperature": 0.0,
        "ignore_eos": False,
        "length_penalty": 1.0,
    },
    "runtime": {
        "batch_prompts": 16,
        "beam_batch_prompts": 16,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 4096,
        "max_logprobs": 16,
        "vllm_version": "0.12.0",
    },
}

RENDERER_LOCKS = {
    "canonical_user": {
        "name": "canonical_user",
        "messages": ["user"],
        "query": "instruction + newline + input",
    },
    "legacy_system": {
        "name": "legacy_system",
        "messages": ["system", "user"],
        "query": "system=instruction; user=input",
    },
}

# Preregistered cell IDs deliberately describe both factors.  The generator's
# shorter names are preserved in its configs/metadata; this explicit alias is
# the only permitted translation between the two naming schemes.
NEW_CELL_ALIASES = {
    "s800_canonical_user": {
        "generator_cell": "s800_canonical",
        "adapter_name": "s800",
        "adapter_sha256": S800_ARTIFACT_SHA256,
        "adapter_rank": 80,
        "renderer_name": "canonical_user",
        "arg_name": "s800_canonical",
    },
    "i23_legacy_system": {
        "generator_cell": "i23_legacy",
        "adapter_name": "i23",
        "adapter_sha256": I23_ARTIFACT_SHA256,
        "adapter_rank": 64,
        "renderer_name": "legacy_system",
        "arg_name": "i23_legacy",
    },
    "i23_canonical_user": {
        "generator_cell": "i23_canonical",
        "adapter_name": "i23",
        "adapter_sha256": I23_ARTIFACT_SHA256,
        "adapter_rank": 64,
        "renderer_name": "canonical_user",
        "arg_name": "i23_canonical",
    },
}

NEW_CONFIG_KEYS = {
    "schema_version",
    "protocol",
    "script",
    "frozen_generator",
    "cell",
    "base",
    "adapter",
    "manifest",
    "reused_fourth_cell",
    "decode",
    "output",
}
NEW_METADATA_KEYS = {
    "schema_version",
    "status",
    "cell",
    "config_sha256",
    "rollout_sha256",
    "rows",
    "traces",
    "candidates",
}

EXPECTED_LEGACY = {
    "groups": 16,
    "reasoning_traces": 64,
    "candidates": 512,
    "valid_candidates": 512,
    "reasoning_stop_closed": 62,
    "candidate_lcp_histogram": {"0": 414, "1": 87, "2": 11, "3": 0},
    "group_max_lcp_histogram": {"0": 8, "1": 6, "2": 2, "3": 0},
    "group_any_lcp_at_least": {"1": 8, "2": 2, "3": 0},
    "candidate_prefix_mass": 109,
}

HEX64 = re.compile(r"[0-9a-f]{64}")
ITEM_RE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|>"
    r"(<s_a_\d+>)(<s_b_\d+>)(<s_c_\d+>)"
)
DOMAINS = {"video", "prod", "ad", "living"}

GOLD_KEYS = {
    "schema_version",
    "group_id",
    "prompt_sha256",
    "domain",
    "prompt_group_id",
    "source_prompt_group_size",
    "source_group_size",
    "gold_count",
    "golds",
    "original_thought_sha256s",
    "original_thought_stripped_sha256s",
}
GOLD_ENTRY_KEYS = {
    "itemic",
    "itemic_sha256",
    "answer",
    "output_prefix",
    "output_suffix",
    "output_shell_sha256",
    "source_row_indices",
    "source_row_sha256s",
    "target_in_prompt",
}
ROLLOUT_KEYS = {
    "schema_version",
    "group_id",
    "prompt_sha256",
    "domain",
    "generator",
    "traces",
}
GENERATOR_KEYS = {"config_sha256", "base_sha256", "adapter_sha256", "seed"}
TRACE_KEYS = {"trace_id", "reasoning_index", "thought", "reasoning", "candidates"}
REASONING_KEYS = {
    "text",
    "raw_text",
    "finish_reason",
    "stop_reason",
    "token_count",
    "seed",
}
CANDIDATE_KEYS = {
    "text",
    "item",
    "valid",
    "finish_reason",
    "stop_reason",
    "token_count",
    "cumulative_logprob",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-score the fixed four-cell I29 renderer calibration."
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--s800-legacy", type=Path, default=DEFAULT_S800_LEGACY)
    parser.add_argument("--s800-canonical", type=Path, default=DEFAULT_S800_CANONICAL)
    parser.add_argument("--i23-legacy", type=Path, default=DEFAULT_I23_LEGACY)
    parser.add_argument("--i23-canonical", type=Path, default=DEFAULT_I23_CANONICAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate frozen inputs and reproduce the old s800 x legacy baseline.",
    )
    mode.add_argument(
        "--self-test", action="store_true", help="Run deterministic in-memory tests."
    )
    return parser.parse_args(argv)


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str, location: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON at {location}: {exc}") from exc


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text_sha256(payload)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_n_raw_sha256(path: Path, count: int) -> str:
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as source:
        for line in source:
            if seen == count:
                break
            digest.update(line)
            seen += 1
    if seen != count:
        raise ValueError(f"{path} has only {seen} lines; expected at least {count}")
    return digest.hexdigest()


def require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} keys differ; missing={missing}, extra={extra}")
    return value


def require_hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def require_non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def validate_nullable_reason(value: Any, context: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (str, int))
    ):
        raise ValueError(f"{context} must be null, string, or integer")


def parse_item(value: Any, expected_domain: str, context: str) -> tuple[str, str, str] | None:
    if not isinstance(value, str):
        return None
    match = ITEM_RE.fullmatch(value)
    if match is None:
        return None
    domain, a_token, b_token, c_token = match.groups()
    if domain != expected_domain:
        raise ValueError(
            f"{context} item domain {domain!r} does not match {expected_domain!r}"
        )
    return a_token, b_token, c_token


def longest_prefix(
    candidate: tuple[str, str, str], gold: tuple[str, str, str]
) -> int:
    depth = 0
    for candidate_token, gold_token in zip(candidate, gold):
        if candidate_token != gold_token:
            break
        depth += 1
    return depth


def validate_gold_row(value: Any, location: str) -> dict[str, Any]:
    row = require_exact_keys(value, GOLD_KEYS, f"gold row at {location}")
    if row["schema_version"] != GOLD_SCHEMA:
        raise ValueError(f"wrong gold schema at {location}")
    require_hex64(row["group_id"], f"group_id at {location}")
    require_hex64(row["prompt_sha256"], f"prompt_sha256 at {location}")
    require_hex64(row["prompt_group_id"], f"prompt_group_id at {location}")
    domain = row["domain"]
    if domain not in DOMAINS:
        raise ValueError(f"invalid domain at {location}: {domain!r}")
    for key in ("source_prompt_group_size", "source_group_size", "gold_count"):
        require_non_negative_int(row[key], f"{key} at {location}")
    golds = row["golds"]
    if not isinstance(golds, list) or not golds:
        raise ValueError(f"golds must be a non-empty list at {location}")
    if row["gold_count"] != len(golds):
        raise ValueError(f"gold_count mismatch at {location}")
    seen_items: set[str] = set()
    for index, gold_value in enumerate(golds):
        context = f"gold {index} at {location}"
        gold = require_exact_keys(gold_value, GOLD_ENTRY_KEYS, context)
        parsed = parse_item(gold["itemic"], domain, f"itemic in {context}")
        if parsed is None:
            raise ValueError(f"invalid itemic in {context}")
        if gold["itemic"] in seen_items:
            raise ValueError(f"duplicate itemic in {context}")
        seen_items.add(gold["itemic"])
        if text_sha256(gold["itemic"]) != gold["itemic_sha256"]:
            raise ValueError(f"itemic_sha256 mismatch in {context}")
        if not all(
            isinstance(gold[key], str)
            for key in ("answer", "output_prefix", "output_suffix")
        ):
            raise ValueError(f"output shell fields must be strings in {context}")
        expected_shell = gold["output_prefix"] + "{thought}" + gold["output_suffix"]
        if text_sha256(expected_shell) != gold["output_shell_sha256"]:
            raise ValueError(f"output_shell_sha256 mismatch in {context}")
        if gold["target_in_prompt"] is not False:
            raise ValueError(f"target_in_prompt must be false in {context}")
        indices = gold["source_row_indices"]
        hashes = gold["source_row_sha256s"]
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"source_row_indices missing in {context}")
        if not isinstance(hashes, list) or len(hashes) != len(indices):
            raise ValueError(f"source provenance length mismatch in {context}")
        for source_index in indices:
            require_non_negative_int(source_index, f"source row index in {context}")
        for source_hash in hashes:
            require_hex64(source_hash, f"source row hash in {context}")
    for key in ("original_thought_sha256s", "original_thought_stripped_sha256s"):
        values = row[key]
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list at {location}")
        for digest in values:
            require_hex64(digest, f"{key} digest at {location}")
    return row


def load_fixed_gold(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"gold ledger not found: {resolved}")
    full_hash = file_sha256(resolved)
    if full_hash != GOLD_FULL_SHA256:
        raise ValueError(
            f"gold ledger SHA-256 mismatch: expected {GOLD_FULL_SHA256}, got {full_hash}"
        )
    raw_hash = first_n_raw_sha256(resolved, GROUP_COUNT)
    if raw_hash != GOLD_FIRST16_RAW_SHA256:
        raise ValueError(
            "first-16 gold raw SHA-256 mismatch: "
            f"expected {GOLD_FIRST16_RAW_SHA256}, got {raw_hash}"
        )
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as source:
        for line_number in range(1, GROUP_COUNT + 1):
            line = source.readline()
            if not line:
                raise ValueError(f"gold ledger ended before line {line_number}")
            if not line.strip():
                raise ValueError(f"blank gold line at {resolved}:{line_number}")
            rows.append(
                validate_gold_row(
                    strict_json_loads(line, f"{resolved}:{line_number}"),
                    f"{resolved}:{line_number}",
                )
            )
    group_ids = [row["group_id"] for row in rows]
    if len(set(group_ids)) != GROUP_COUNT:
        raise ValueError("first-16 gold group_id values are not unique")
    return rows, {
        "path": str(resolved),
        "sha256": full_hash,
        "first16_raw_sha256": raw_hash,
        "selected_rows": GROUP_COUNT,
        "selected_gold_items": sum(row["gold_count"] for row in rows),
    }


def validate_generator(value: Any, context: str) -> dict[str, Any]:
    generator = require_exact_keys(value, GENERATOR_KEYS, context)
    for key in ("config_sha256", "base_sha256", "adapter_sha256"):
        require_hex64(generator[key], f"{key} in {context}")
    seed = generator["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError(f"seed in {context} must be a 31-bit non-negative integer")
    if seed != GLOBAL_SEED:
        raise ValueError(f"global seed drift in {context}: {seed} != {GLOBAL_SEED}")
    return generator


def validate_rollout_row(
    value: Any, gold: dict[str, Any], location: str
) -> tuple[dict[str, Any], int]:
    row = require_exact_keys(value, ROLLOUT_KEYS, f"rollout row at {location}")
    if row["schema_version"] != ROLLOUT_SCHEMA:
        raise ValueError(f"wrong rollout schema at {location}")
    for key in ("group_id", "prompt_sha256", "domain"):
        if row[key] != gold[key]:
            raise ValueError(f"{key} does not match fixed gold population at {location}")
    validate_generator(row["generator"], f"generator at {location}")
    traces = row["traces"]
    if not isinstance(traces, list) or len(traces) != REASONING_SAMPLES:
        raise ValueError(f"expected {REASONING_SAMPLES} traces at {location}")
    trace_ids: set[str] = set()
    reasoning_indices: list[int] = []
    row_seed: int | None = None
    for trace_offset, trace_value in enumerate(traces):
        trace_context = f"trace {trace_offset} at {location}"
        trace = require_exact_keys(trace_value, TRACE_KEYS, trace_context)
        trace_id = require_hex64(trace["trace_id"], f"trace_id in {trace_context}")
        if trace_id in trace_ids:
            raise ValueError(f"duplicate trace_id in {trace_context}")
        trace_ids.add(trace_id)
        reasoning_index = require_non_negative_int(
            trace["reasoning_index"], f"reasoning_index in {trace_context}"
        )
        reasoning_indices.append(reasoning_index)
        if not isinstance(trace["thought"], str):
            raise ValueError(f"thought must be a string in {trace_context}")
        reasoning = require_exact_keys(
            trace["reasoning"], REASONING_KEYS, f"reasoning in {trace_context}"
        )
        if reasoning["text"] != trace["thought"] or not isinstance(
            reasoning["raw_text"], str
        ):
            raise ValueError(f"reasoning text mismatch in {trace_context}")
        require_non_negative_int(
            reasoning["token_count"], f"reasoning token_count in {trace_context}"
        )
        validate_nullable_reason(
            reasoning["finish_reason"], f"reasoning finish_reason in {trace_context}"
        )
        validate_nullable_reason(
            reasoning["stop_reason"], f"reasoning stop_reason in {trace_context}"
        )
        seed = reasoning["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError(f"invalid reasoning seed in {trace_context}")
        if row_seed is None:
            row_seed = seed
        elif seed != row_seed:
            raise ValueError(f"reasoning traces use different seeds at {location}")
        candidates = trace["candidates"]
        if not isinstance(candidates, list) or len(candidates) != ITEM_CANDIDATES:
            raise ValueError(f"expected {ITEM_CANDIDATES} candidates in {trace_context}")
        for candidate_offset, candidate_value in enumerate(candidates):
            candidate_context = (
                f"candidate {candidate_offset} in {trace_context}"
            )
            candidate = require_exact_keys(
                candidate_value, CANDIDATE_KEYS, candidate_context
            )
            if not isinstance(candidate["text"], str):
                raise ValueError(f"candidate text must be a string in {candidate_context}")
            parsed = parse_item(
                candidate["text"], row["domain"], f"candidate text in {candidate_context}"
            )
            expected_valid = parsed is not None
            if candidate["valid"] is not expected_valid:
                raise ValueError(f"incorrect valid flag in {candidate_context}")
            if candidate["item"] != (candidate["text"] if expected_valid else None):
                raise ValueError(f"incorrect parsed item in {candidate_context}")
            require_non_negative_int(
                candidate["token_count"], f"token_count in {candidate_context}"
            )
            score = candidate["cumulative_logprob"]
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"invalid cumulative_logprob in {candidate_context}")
            validate_nullable_reason(
                candidate["finish_reason"], f"finish_reason in {candidate_context}"
            )
            validate_nullable_reason(
                candidate["stop_reason"], f"stop_reason in {candidate_context}"
            )
    if sorted(reasoning_indices) != list(range(REASONING_SAMPLES)):
        raise ValueError(f"reasoning_index values are not 0..3 at {location}")
    assert row_seed is not None
    return row, row_seed


def validate_runtime_artifact_block(
    value: Any,
    *,
    expected_sha256: str,
    expected_rank: int | None,
    context: str,
) -> dict[str, Any]:
    expected_keys = {"path", "artifact_sha256", "files"}
    if expected_rank is not None:
        expected_keys |= {"rank", "vllm_max_lora_rank"}
    block = require_exact_keys(value, expected_keys, context)
    if not isinstance(block["path"], str) or not block["path"]:
        raise ValueError(f"runtime artifact path missing in {context}")
    if block["artifact_sha256"] != expected_sha256:
        raise ValueError(
            f"runtime artifact SHA-256 mismatch in {context}: "
            f"expected {expected_sha256}, got {block['artifact_sha256']}"
        )
    files = block["files"]
    if not isinstance(files, list):
        raise ValueError(f"runtime artifact files must be a list in {context}")
    previous_path: str | None = None
    for index, file_value in enumerate(files):
        file_context = f"runtime file {index} in {context}"
        runtime_file = require_exact_keys(
            file_value, {"path", "size", "sha256"}, file_context
        )
        if not isinstance(runtime_file["path"], str) or not runtime_file["path"]:
            raise ValueError(f"empty runtime file path in {file_context}")
        if previous_path is not None and runtime_file["path"] <= previous_path:
            raise ValueError(f"runtime files are not strictly sorted in {context}")
        previous_path = runtime_file["path"]
        require_non_negative_int(runtime_file["size"], f"size in {file_context}")
        require_hex64(runtime_file["sha256"], f"sha256 in {file_context}")
    if expected_rank is not None:
        if block["rank"] != expected_rank:
            raise ValueError(
                f"adapter rank mismatch in {context}: "
                f"expected {expected_rank}, got {block['rank']}"
            )
        if block["vllm_max_lora_rank"] != 128:
            raise ValueError(f"vLLM max LoRA rank mismatch in {context}")
    return block


def validate_new_config_lock(
    config: Any,
    rollout_path: Path,
    prereg_cell_name: str,
    alias: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    locked = require_exact_keys(config, NEW_CONFIG_KEYS, context)
    if locked["schema_version"] != NEW_CONFIG_SCHEMA:
        raise ValueError(f"new cell config schema mismatch in {context}")
    if locked["protocol"] != NEW_CONFIG_PROTOCOL:
        raise ValueError(f"new cell config protocol mismatch in {context}")

    script = require_exact_keys(locked["script"], {"path", "sha256"}, context)
    if (
        not isinstance(script["path"], str)
        or Path(script["path"]).resolve() != GENERATOR_SCRIPT.resolve()
    ):
        raise ValueError(f"I29 generator path mismatch in {context}")
    generator_script_hash = file_sha256(GENERATOR_SCRIPT.resolve())
    if script["sha256"] != generator_script_hash:
        raise ValueError(f"I29 generator script SHA-256 mismatch in {context}")

    frozen = require_exact_keys(
        locked["frozen_generator"], {"path", "sha256", "reuse"}, context
    )
    if (
        not isinstance(frozen["path"], str)
        or Path(frozen["path"]).resolve() != LEGACY_GENERATOR_SCRIPT.resolve()
        or frozen["sha256"] != LEGACY_GENERATOR_SHA256
        or frozen["reuse"] != "generate_batch with scoped renderer override"
    ):
        raise ValueError(f"frozen I27 generator lock mismatch in {context}")

    cell = require_exact_keys(
        locked["cell"], {"name", "adapter_name", "renderer"}, context
    )
    expected_renderer = RENDERER_LOCKS[alias["renderer_name"]]
    renderer = require_exact_keys(
        cell["renderer"], {"name", "messages", "query"}, context
    )
    expected_cell = (alias["generator_cell"], alias["adapter_name"])
    actual_cell = (cell["name"], cell["adapter_name"])
    if actual_cell != expected_cell:
        raise ValueError(
            f"cell alias mismatch in {context}: prereg={prereg_cell_name}, "
            f"expected generator identity={expected_cell}, got={actual_cell}"
        )
    if renderer != expected_renderer:
        raise ValueError(
            f"complete renderer lock mismatch in {context}: "
            f"expected {expected_renderer}, got {renderer}"
        )

    validate_runtime_artifact_block(
        locked["base"],
        expected_sha256=BASE_ARTIFACT_SHA256,
        expected_rank=None,
        context=f"base in {context}",
    )
    validate_runtime_artifact_block(
        locked["adapter"],
        expected_sha256=alias["adapter_sha256"],
        expected_rank=alias["adapter_rank"],
        context=f"adapter in {context}",
    )

    manifest = require_exact_keys(
        locked["manifest"],
        {"path", "sha256", "selection", "selected_rows", "first16_sha256", "contains_labels"},
        context,
    )
    expected_manifest_path = (
        PROJECT_ROOT
        / "assets/derived/processed/o1_rec_multigold_v1_prompt_manifest.jsonl"
    ).resolve()
    if (
        not isinstance(manifest["path"], str)
        or Path(manifest["path"]).resolve() != expected_manifest_path
        or manifest["sha256"] != MANIFEST_FULL_SHA256
        or manifest["selection"] != "physical first 16 JSONL rows"
        or manifest["selected_rows"] != GROUP_COUNT
        or manifest["first16_sha256"] != MANIFEST_FIRST16_RAW_SHA256
        or manifest["contains_labels"] is not False
    ):
        raise ValueError(f"prompt-manifest lock mismatch in {context}")

    reused = require_exact_keys(
        locked["reused_fourth_cell"],
        {"path", "sha256", "first16_sha256", "cell"},
        context,
    )
    expected_old_path = DEFAULT_S800_LEGACY.resolve()
    if (
        not isinstance(reused["path"], str)
        or Path(reused["path"]).resolve() != expected_old_path
        or reused["sha256"] != S800_LEGACY_FULL_SHA256
        or reused["first16_sha256"] != S800_LEGACY_FIRST16_RAW_SHA256
        or reused["cell"] != "s800_legacy"
    ):
        raise ValueError(f"reused fourth-cell lock mismatch in {context}")
    if locked["decode"] != EXPECTED_DECODE:
        raise ValueError(f"complete decode lock mismatch in {context}")
    if (
        not isinstance(locked["output"], str)
        or Path(locked["output"]).resolve() != rollout_path.resolve()
    ):
        raise ValueError(f"output path mismatch in {context}")
    return {
        "preregistered_cell": prereg_cell_name,
        "generator_cell": alias["generator_cell"],
        "adapter_name": alias["adapter_name"],
        "renderer_name": alias["renderer_name"],
        "alias_explicitly_validated": True,
    }


def load_companion_config(
    rollout_path: Path,
    generator: dict[str, Any],
    *,
    expected_adapter_sha256: str,
    prereg_cell_name: str | None,
    cell_alias: dict[str, Any] | None,
    expected_file_sha256: str | None,
) -> dict[str, Any]:
    config_path = rollout_path.with_suffix(rollout_path.suffix + ".config.json")
    if not config_path.is_file():
        raise FileNotFoundError(f"rollout companion config not found: {config_path}")
    config_file_hash = file_sha256(config_path)
    if expected_file_sha256 is not None and config_file_hash != expected_file_sha256:
        raise ValueError(
            f"config file SHA-256 mismatch for {config_path}: "
            f"expected {expected_file_sha256}, got {config_file_hash}"
        )
    config = strict_json_loads(
        config_path.read_text(encoding="utf-8"), str(config_path)
    )
    if not isinstance(config, dict):
        raise ValueError(f"rollout companion config must be an object: {config_path}")
    config_identity = canonical_sha256(config)
    if config_identity != generator["config_sha256"]:
        raise ValueError(
            f"canonical config identity mismatch for {config_path}: "
            f"row={generator['config_sha256']}, config={config_identity}"
        )
    if generator["base_sha256"] != BASE_ARTIFACT_SHA256:
        raise ValueError(
            f"base runtime artifact drift in {rollout_path}: "
            f"{generator['base_sha256']} != {BASE_ARTIFACT_SHA256}"
        )
    if generator["adapter_sha256"] != expected_adapter_sha256:
        raise ValueError(
            f"adapter runtime artifact drift in {rollout_path}: "
            f"{generator['adapter_sha256']} != {expected_adapter_sha256}"
        )
    if config.get("base", {}).get("artifact_sha256") != BASE_ARTIFACT_SHA256:
        raise ValueError(f"base artifact mismatch inside {config_path}")
    if config.get("adapter", {}).get("artifact_sha256") != expected_adapter_sha256:
        raise ValueError(f"adapter artifact mismatch inside {config_path}")

    identity_report: dict[str, Any]
    if prereg_cell_name is None:
        if cell_alias is not None:
            raise AssertionError("old cell cannot have a new-cell alias")
        if config.get("schema_version") != "o1-rec-rollout-config-v1":
            raise ValueError(f"old rollout config schema mismatch in {config_path}")
        identity_report = {
            "cell_name": "reused_s800_legacy",
            "adapter_name": "s800",
            "renderer_name": "legacy_system",
            "identity_source": "frozen old config SHA plus generator/decode provenance",
        }
    else:
        if cell_alias is None:
            raise AssertionError("new cell requires an explicit alias")
        identity_report = validate_new_config_lock(
            config,
            rollout_path,
            prereg_cell_name,
            cell_alias,
            str(config_path),
        )
    return {
        "path": str(config_path),
        "sha256": config_file_hash,
        "canonical_sha256": config_identity,
        "cell_identity": identity_report,
    }


def new_cell_specifications(
    args: argparse.Namespace,
) -> list[tuple[str, Path, dict[str, Any]]]:
    return [
        (prereg_name, Path(getattr(args, alias["arg_name"])), alias)
        for prereg_name, alias in NEW_CELL_ALIASES.items()
    ]


def blind_precheck_new_cells(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Verify all completion artifacts before any code is allowed to open gold."""
    specifications = new_cell_specifications(args)
    paths: dict[str, dict[str, Path]] = {}
    missing: list[Path] = []
    for prereg_name, rollout_value, _ in specifications:
        rollout = rollout_value.resolve()
        config = rollout.with_suffix(rollout.suffix + ".config.json")
        metadata = rollout.with_suffix(rollout.suffix + ".meta.json")
        cell_paths = {"rollout": rollout, "config": config, "metadata": metadata}
        paths[prereg_name] = cell_paths
        missing.extend(path for path in cell_paths.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "label-blind completion precheck failed before gold access; missing: "
            + ", ".join(str(path) for path in missing)
        )

    checked: dict[str, dict[str, Any]] = {}
    for prereg_name, _, alias in specifications:
        cell_paths = paths[prereg_name]
        rollout_path = cell_paths["rollout"]
        config_path = cell_paths["config"]
        metadata_path = cell_paths["metadata"]
        config = strict_json_loads(
            config_path.read_text(encoding="utf-8"), str(config_path)
        )
        identity = validate_new_config_lock(
            config, rollout_path, prereg_name, alias, str(config_path)
        )
        config_identity = canonical_sha256(config)
        config_file_hash = file_sha256(config_path)
        rollout_hash = file_sha256(rollout_path)

        metadata = require_exact_keys(
            strict_json_loads(
                metadata_path.read_text(encoding="utf-8"), str(metadata_path)
            ),
            NEW_METADATA_KEYS,
            f"completion metadata at {metadata_path}",
        )
        if metadata["schema_version"] != NEW_METADATA_SCHEMA:
            raise ValueError(f"completion metadata schema mismatch at {metadata_path}")
        if metadata["status"] != "complete":
            raise ValueError(f"cell is not complete according to {metadata_path}")
        if metadata["cell"] != alias["generator_cell"]:
            raise ValueError(
                f"completion metadata cell alias mismatch at {metadata_path}"
            )
        require_hex64(
            metadata["config_sha256"], f"config_sha256 at {metadata_path}"
        )
        require_hex64(
            metadata["rollout_sha256"], f"rollout_sha256 at {metadata_path}"
        )
        if metadata["config_sha256"] != config_identity:
            raise ValueError(
                f"completion metadata/config canonical SHA mismatch at {metadata_path}"
            )
        if metadata["rollout_sha256"] != rollout_hash:
            raise ValueError(
                f"completion metadata/rollout SHA mismatch at {metadata_path}"
            )
        expected_counts = {
            "rows": GROUP_COUNT,
            "traces": GROUP_COUNT * REASONING_SAMPLES,
            "candidates": GROUP_COUNT * REASONING_SAMPLES * ITEM_CANDIDATES,
        }
        for key, expected in expected_counts.items():
            actual = require_non_negative_int(
                metadata[key], f"{key} at {metadata_path}"
            )
            if actual != expected:
                raise ValueError(
                    f"completion metadata {key} mismatch at {metadata_path}: "
                    f"expected {expected}, got {actual}"
                )
        checked[prereg_name] = {
            "rollout_path": str(rollout_path),
            "rollout_sha256": rollout_hash,
            "config_path": str(config_path),
            "config_file_sha256": config_file_hash,
            "config_canonical_sha256": config_identity,
            "metadata_path": str(metadata_path),
            "metadata_sha256": file_sha256(metadata_path),
            "completion_marker": {
                "schema_version": metadata["schema_version"],
                "status": metadata["status"],
                "cell": metadata["cell"],
                **expected_counts,
            },
            "identity": identity,
        }
    if len({value["rollout_path"] for value in checked.values()}) != len(checked):
        raise ValueError("new-cell aliases resolve to duplicate rollout paths")
    if len(
        {value["config_canonical_sha256"] for value in checked.values()}
    ) != len(checked):
        raise ValueError("new-cell configs do not have unique canonical identities")
    return checked


def load_rollout_cell(
    path: Path,
    gold_rows: list[dict[str, Any]],
    *,
    allow_extra_rows: bool,
    expected_adapter_sha256: str,
    prereg_cell_name: str | None,
    cell_alias: dict[str, Any] | None,
    expected_full_sha256: str | None = None,
    expected_first16_raw_sha256: str | None = None,
    expected_config_file_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"rollout cell not found: {resolved}")
    full_hash = file_sha256(resolved)
    if expected_full_sha256 is not None and full_hash != expected_full_sha256:
        raise ValueError(
            f"rollout SHA-256 mismatch for {resolved}: "
            f"expected {expected_full_sha256}, got {full_hash}"
        )
    raw_hash = first_n_raw_sha256(resolved, GROUP_COUNT)
    if (
        expected_first16_raw_sha256 is not None
        and raw_hash != expected_first16_raw_sha256
    ):
        raise ValueError(
            f"first-16 rollout raw SHA-256 mismatch for {resolved}: "
            f"expected {expected_first16_raw_sha256}, got {raw_hash}"
        )
    all_values: list[tuple[Any, str]] = []
    with resolved.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            location = f"{resolved}:{line_number}"
            if not line.strip():
                raise ValueError(f"blank rollout line at {location}")
            all_values.append((strict_json_loads(line, location), location))
    if not allow_extra_rows and len(all_values) != GROUP_COUNT:
        raise ValueError(
            f"{resolved} must contain exactly {GROUP_COUNT} rows; got {len(all_values)}"
        )
    if len(all_values) < GROUP_COUNT:
        raise ValueError(f"{resolved} has only {len(all_values)} rollout rows")
    selected_values = all_values[:GROUP_COUNT]
    rows: list[dict[str, Any]] = []
    reasoning_seeds: dict[str, int] = {}
    for index, ((value, location), gold) in enumerate(zip(selected_values, gold_rows)):
        if not isinstance(value, dict) or value.get("group_id") != gold["group_id"]:
            raise ValueError(
                f"rollout row order/group drift at selected index {index} in {resolved}"
            )
        row, reasoning_seed = validate_rollout_row(value, gold, location)
        rows.append(row)
        reasoning_seeds[row["group_id"]] = reasoning_seed
    generators = {json.dumps(row["generator"], sort_keys=True) for row in rows}
    if len(generators) != 1:
        raise ValueError(f"generator identity changes within rollout cell {resolved}")
    generator = rows[0]["generator"]
    companion_config = load_companion_config(
        resolved,
        generator,
        expected_adapter_sha256=expected_adapter_sha256,
        prereg_cell_name=prereg_cell_name,
        cell_alias=cell_alias,
        expected_file_sha256=expected_config_file_sha256,
    )
    return rows, {
        "path": str(resolved),
        "sha256": full_hash,
        "first16_raw_sha256": raw_hash,
        "source_rows": len(all_values),
        "selected_rows": GROUP_COUNT,
        "generator": generator,
        "companion_config": companion_config,
    }, reasoning_seeds


def score_cell(rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_lcp = Counter({depth: 0 for depth in range(4)})
    group_max_lcp = Counter({depth: 0 for depth in range(4)})
    group_any = Counter({depth: 0 for depth in range(1, 4)})
    gold_unique_prefixes = Counter({depth: 0 for depth in range(1, 4)})
    generated_unique_prefixes = Counter({depth: 0 for depth in range(1, 4)})
    matched_unique_prefixes = Counter({depth: 0 for depth in range(1, 4)})
    valid_candidates = 0
    reasoning_stop_closed = 0
    per_group: list[dict[str, Any]] = []

    for row, gold in zip(rows, gold_rows):
        gold_items = [
            parse_item(entry["itemic"], gold["domain"], "validated gold")
            for entry in gold["golds"]
        ]
        if any(item is None for item in gold_items):
            raise AssertionError("validated gold unexpectedly failed item parsing")
        typed_gold_items = [item for item in gold_items if item is not None]
        group_candidate_lcp = Counter({depth: 0 for depth in range(4)})
        generated_items: list[tuple[str, str, str]] = []
        group_valid = 0
        group_closed = 0
        for trace in row["traces"]:
            if trace["reasoning"]["stop_reason"] == "</think>":
                reasoning_stop_closed += 1
                group_closed += 1
            for candidate in trace["candidates"]:
                if candidate["valid"]:
                    parsed = parse_item(candidate["item"], row["domain"], "validated candidate")
                    assert parsed is not None
                    generated_items.append(parsed)
                    valid_candidates += 1
                    group_valid += 1
                    depth = max(
                        longest_prefix(parsed, gold_item)
                        for gold_item in typed_gold_items
                    )
                else:
                    depth = 0
                candidate_lcp[depth] += 1
                group_candidate_lcp[depth] += 1
        maximum = max(
            (depth for depth, count in group_candidate_lcp.items() if count),
            default=0,
        )
        group_max_lcp[maximum] += 1
        for depth in range(1, 4):
            if maximum >= depth:
                group_any[depth] += 1
        group_gold_unique: dict[str, int] = {}
        group_generated_unique: dict[str, int] = {}
        group_matched_unique: dict[str, int] = {}
        for depth in range(1, 4):
            gold_prefix_set = {item[:depth] for item in typed_gold_items}
            generated_prefix_set = {item[:depth] for item in generated_items}
            matched_prefix_set = gold_prefix_set & generated_prefix_set
            gold_unique_prefixes[depth] += len(gold_prefix_set)
            generated_unique_prefixes[depth] += len(generated_prefix_set)
            matched_unique_prefixes[depth] += len(matched_prefix_set)
            group_gold_unique[str(depth)] = len(gold_prefix_set)
            group_generated_unique[str(depth)] = len(generated_prefix_set)
            group_matched_unique[str(depth)] = len(matched_prefix_set)
        per_group.append(
            {
                "group_id": row["group_id"],
                "domain": row["domain"],
                "gold_count": gold["gold_count"],
                "reasoning_stop_closed": group_closed,
                "valid_candidates": group_valid,
                "candidate_lcp_histogram": {
                    str(depth): group_candidate_lcp[depth] for depth in range(4)
                },
                "max_lcp": maximum,
                "unique_prefixes": {
                    "gold": group_gold_unique,
                    "generated": group_generated_unique,
                    "matched_gold": group_matched_unique,
                },
            }
        )

    reasoning_traces = len(rows) * REASONING_SAMPLES
    candidates = reasoning_traces * ITEM_CANDIDATES
    prefix_mass = sum(depth * candidate_lcp[depth] for depth in range(4))
    coverage = {
        str(depth): (
            matched_unique_prefixes[depth] / gold_unique_prefixes[depth]
            if gold_unique_prefixes[depth]
            else None
        )
        for depth in range(1, 4)
    }
    return {
        "counts": {
            "groups": len(rows),
            "reasoning_traces": reasoning_traces,
            "candidates": candidates,
            "valid_candidates": valid_candidates,
            "invalid_candidates": candidates - valid_candidates,
        },
        "reasoning": {
            "stop_closed": reasoning_stop_closed,
            "not_stop_closed": reasoning_traces - reasoning_stop_closed,
            "stop_closed_rate": reasoning_stop_closed / reasoning_traces,
        },
        "candidate_lcp_histogram": {
            str(depth): candidate_lcp[depth] for depth in range(4)
        },
        "candidate_prefix_mass": prefix_mass,
        "candidate_mean_lcp": prefix_mass / candidates,
        "group_max_lcp_histogram": {
            str(depth): group_max_lcp[depth] for depth in range(4)
        },
        "group_any_lcp_at_least": {
            str(depth): group_any[depth] for depth in range(1, 4)
        },
        "exact": {
            "candidates": candidate_lcp[3],
            "groups": group_any[3],
        },
        "unique_prefix_coverage": {
            "gold": {
                str(depth): gold_unique_prefixes[depth] for depth in range(1, 4)
            },
            "generated": {
                str(depth): generated_unique_prefixes[depth]
                for depth in range(1, 4)
            },
            "matched_gold": {
                str(depth): matched_unique_prefixes[depth]
                for depth in range(1, 4)
            },
            "matched_gold_rate": coverage,
            "deduplication_scope": "within_group_then_sum_across_groups",
        },
        "per_group": per_group,
    }


def assert_expected_legacy(stats: dict[str, Any]) -> None:
    actual = {
        "groups": stats["counts"]["groups"],
        "reasoning_traces": stats["counts"]["reasoning_traces"],
        "candidates": stats["counts"]["candidates"],
        "valid_candidates": stats["counts"]["valid_candidates"],
        "reasoning_stop_closed": stats["reasoning"]["stop_closed"],
        "candidate_lcp_histogram": stats["candidate_lcp_histogram"],
        "group_max_lcp_histogram": stats["group_max_lcp_histogram"],
        "group_any_lcp_at_least": stats["group_any_lcp_at_least"],
        "candidate_prefix_mass": stats["candidate_prefix_mass"],
    }
    if actual != EXPECTED_LEGACY:
        raise AssertionError(
            "frozen s800 x legacy baseline did not reproduce: "
            + json.dumps({"expected": EXPECTED_LEGACY, "actual": actual}, sort_keys=True)
        )


def scalar_view(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_prefix_mass": stats["candidate_prefix_mass"],
        "candidate_mean_lcp": stats["candidate_mean_lcp"],
        "candidate_lcp_at_least_1": sum(
            stats["candidate_lcp_histogram"][str(depth)] for depth in range(1, 4)
        ),
        "candidate_lcp_at_least_2": sum(
            stats["candidate_lcp_histogram"][str(depth)] for depth in range(2, 4)
        ),
        "exact_candidates": stats["exact"]["candidates"],
        "exact_groups": stats["exact"]["groups"],
        "group_any_a": stats["group_any_lcp_at_least"]["1"],
        "group_any_ab": stats["group_any_lcp_at_least"]["2"],
        "group_any_abc": stats["group_any_lcp_at_least"]["3"],
        "matched_unique_a": stats["unique_prefix_coverage"]["matched_gold"]["1"],
        "matched_unique_ab": stats["unique_prefix_coverage"]["matched_gold"]["2"],
        "matched_unique_abc": stats["unique_prefix_coverage"]["matched_gold"]["3"],
        "valid_candidates": stats["counts"]["valid_candidates"],
        "reasoning_stop_closed": stats["reasoning"]["stop_closed"],
    }


def compare_cells(
    minuend_name: str,
    minuend: dict[str, Any],
    subtrahend_name: str,
    subtrahend: dict[str, Any],
) -> dict[str, Any]:
    left = scalar_view(minuend)
    right = scalar_view(subtrahend)
    delta = {key: left[key] - right[key] for key in left}
    return {
        "definition": f"{minuend_name} minus {subtrahend_name}",
        "minuend": left,
        "subtrahend": right,
        "delta": delta,
    }


def direction_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    delta = comparison["delta"]
    mass_delta = delta["candidate_prefix_mass"]
    if mass_delta > 0:
        primary = "s800_higher_candidate_prefix_mass"
    elif mass_delta < 0:
        primary = "i23_higher_candidate_prefix_mass"
    else:
        primary = "candidate_prefix_mass_tie"
    return {
        "primary_metric": "candidate_prefix_mass",
        "primary_direction": primary,
        "candidate_prefix_mass_delta_s800_minus_i23": mass_delta,
        "secondary_group_any_ab_delta_s800_minus_i23": delta["group_any_ab"],
        "secondary_exact_group_delta_s800_minus_i23": delta["exact_groups"],
        "boundary": (
            "Directional 16-group smoke only. Exact=0 is not a rejection rule; "
            "do not infer leaderboard gain or authorize training from this report."
        ),
    }


def assert_cross_cell_integrity(
    metadata: dict[str, dict[str, Any]],
    reasoning_seeds: dict[str, dict[str, int]],
) -> None:
    seed_maps = list(reasoning_seeds.values())
    reference = seed_maps[0]
    for cell_name, seed_map in reasoning_seeds.items():
        if seed_map != reference:
            raise ValueError(f"per-group reasoning seed drift in cell {cell_name}")
    base_hashes = {
        cell_metadata["generator"]["base_sha256"]
        for cell_metadata in metadata.values()
    }
    if len(base_hashes) != 1:
        raise ValueError(f"base artifact differs across cells: {sorted(base_hashes)}")
    if (
        metadata["s800_legacy_system"]["generator"]["adapter_sha256"]
        != metadata["s800_canonical_user"]["generator"]["adapter_sha256"]
    ):
        raise ValueError("s800 adapter differs between renderer cells")
    if (
        metadata["i23_legacy_system"]["generator"]["adapter_sha256"]
        != metadata["i23_canonical_user"]["generator"]["adapter_sha256"]
    ):
        raise ValueError("I-23 adapter differs between renderer cells")
    if (
        metadata["s800_canonical_user"]["generator"]["adapter_sha256"]
        == metadata["i23_canonical_user"]["generator"]["adapter_sha256"]
    ):
        raise ValueError("I-23 and s800 unexpectedly have the same adapter identity")


def atomic_write_json_no_replace(path: Path, value: Any) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {destination}")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        # Same-filesystem hard linking is atomic and fails if destination exists.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def make_synthetic_gold(index: int) -> dict[str, Any]:
    domain = ("video", "prod", "ad", "living")[index % 4]
    group_id = text_sha256(f"synthetic-group-{index}")
    prompt_sha = text_sha256(f"synthetic-prompt-{index}")
    item = f"<|{domain}_begin|><s_a_{index}><s_b_{index + 1}><s_c_{index + 2}>"
    suffix = f"</think>\n{item}"
    return {
        "schema_version": GOLD_SCHEMA,
        "group_id": group_id,
        "prompt_sha256": prompt_sha,
        "domain": domain,
        "prompt_group_id": text_sha256(f"synthetic-prompt-group-{index}"),
        "source_prompt_group_size": 1,
        "source_group_size": 1,
        "gold_count": 1,
        "golds": [
            {
                "itemic": item,
                "itemic_sha256": text_sha256(item),
                "answer": item,
                "output_prefix": "<think>",
                "output_suffix": suffix,
                "output_shell_sha256": text_sha256("<think>{thought}" + suffix),
                "source_row_indices": [index],
                "source_row_sha256s": [text_sha256(f"synthetic-source-{index}")],
                "target_in_prompt": False,
            }
        ],
        "original_thought_sha256s": [text_sha256(f"thought-{index}")],
        "original_thought_stripped_sha256s": [text_sha256(f"thought-{index}")],
    }


def make_synthetic_rollout(gold: dict[str, Any], index: int) -> dict[str, Any]:
    exact_item = gold["golds"][0]["itemic"]
    parsed = parse_item(exact_item, gold["domain"], "synthetic item")
    assert parsed is not None
    a_token, b_token, _ = parsed
    miss_item = (
        f"<|{gold['domain']}_begin|>{a_token}{b_token}<s_c_{9000 + index}>"
    )
    traces = []
    for reasoning_index in range(REASONING_SAMPLES):
        candidates = []
        for candidate_index in range(ITEM_CANDIDATES):
            item = exact_item if candidate_index == 0 else miss_item
            candidates.append(
                {
                    "text": item,
                    "item": item,
                    "valid": True,
                    "finish_reason": None,
                    "stop_reason": None,
                    "token_count": 3,
                    "cumulative_logprob": -float(candidate_index),
                }
            )
        thought = f"thought {index} {reasoning_index}"
        traces.append(
            {
                "trace_id": text_sha256(
                    f"synthetic-trace-{index}-{reasoning_index}"
                ),
                "reasoning_index": reasoning_index,
                "thought": thought,
                "reasoning": {
                    "text": thought,
                    "raw_text": f"<think>{thought}",
                    "finish_reason": "stop",
                    "stop_reason": "</think>",
                    "token_count": 3,
                    "seed": 1000 + index,
                },
                "candidates": candidates,
            }
        )
    return {
        "schema_version": ROLLOUT_SCHEMA,
        "group_id": gold["group_id"],
        "prompt_sha256": gold["prompt_sha256"],
        "domain": gold["domain"],
        "generator": {
            "config_sha256": "a" * 64,
            "base_sha256": "b" * 64,
            "adapter_sha256": "c" * 64,
            "seed": GLOBAL_SEED,
        },
        "traces": traces,
    }


def make_synthetic_locked_config(
    rollout_path: Path, prereg_cell_name: str, alias: dict[str, Any]
) -> dict[str, Any]:
    del prereg_cell_name  # Kept explicit so fixtures use the same alias interface.
    runtime_file = {
        "path": "synthetic.bin",
        "size": 1,
        "sha256": "d" * 64,
    }
    return {
        "schema_version": NEW_CONFIG_SCHEMA,
        "protocol": NEW_CONFIG_PROTOCOL,
        "script": {
            "path": str(GENERATOR_SCRIPT.resolve()),
            "sha256": file_sha256(GENERATOR_SCRIPT.resolve()),
        },
        "frozen_generator": {
            "path": str(LEGACY_GENERATOR_SCRIPT.resolve()),
            "sha256": LEGACY_GENERATOR_SHA256,
            "reuse": "generate_batch with scoped renderer override",
        },
        "cell": {
            "name": alias["generator_cell"],
            "adapter_name": alias["adapter_name"],
            "renderer": json.loads(
                json.dumps(RENDERER_LOCKS[alias["renderer_name"]])
            ),
        },
        "base": {
            "path": str(PROJECT_ROOT / "models/OneReason-0.8B-pretrain-competition"),
            "artifact_sha256": BASE_ARTIFACT_SHA256,
            "files": [runtime_file],
        },
        "adapter": {
            "path": f"synthetic/{alias['adapter_name']}",
            "artifact_sha256": alias["adapter_sha256"],
            "files": [runtime_file],
            "rank": alias["adapter_rank"],
            "vllm_max_lora_rank": 128,
        },
        "manifest": {
            "path": str(
                (
                    PROJECT_ROOT
                    / "assets/derived/processed/o1_rec_multigold_v1_prompt_manifest.jsonl"
                ).resolve()
            ),
            "sha256": MANIFEST_FULL_SHA256,
            "selection": "physical first 16 JSONL rows",
            "selected_rows": GROUP_COUNT,
            "first16_sha256": MANIFEST_FIRST16_RAW_SHA256,
            "contains_labels": False,
        },
        "reused_fourth_cell": {
            "path": str(DEFAULT_S800_LEGACY.resolve()),
            "sha256": S800_LEGACY_FULL_SHA256,
            "first16_sha256": S800_LEGACY_FIRST16_RAW_SHA256,
            "cell": "s800_legacy",
        },
        "decode": json.loads(json.dumps(EXPECTED_DECODE)),
        "output": str(rollout_path.resolve()),
    }


def make_blind_test_args(
    root: Path,
    *,
    omit_metadata_for: str | None = None,
    corrupt_decode_for: str | None = None,
    corrupt_renderer_for: str | None = None,
) -> argparse.Namespace:
    arguments: dict[str, Any] = {
        "gold": root / "GOLD_MUST_NOT_BE_OPENED.jsonl",
        "s800_legacy": root / "legacy-must-not-be-opened.jsonl",
        "out": root / "report-must-not-be-written.json",
    }
    rollout_payload = b"{}\n"
    for prereg_name, alias in NEW_CELL_ALIASES.items():
        rollout_path = (root / f"{alias['arg_name']}.jsonl").resolve()
        arguments[alias["arg_name"]] = rollout_path
        rollout_path.write_bytes(rollout_payload)
        config = make_synthetic_locked_config(rollout_path, prereg_name, alias)
        if prereg_name == corrupt_decode_for:
            config["decode"]["reasoning"]["temperature"] = 0.7
        if prereg_name == corrupt_renderer_for:
            config["cell"]["renderer"]["messages"] = ["system"]
        config_path = rollout_path.with_suffix(rollout_path.suffix + ".config.json")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        metadata = {
            "schema_version": NEW_METADATA_SCHEMA,
            "status": "complete",
            "cell": alias["generator_cell"],
            "config_sha256": canonical_sha256(config),
            "rollout_sha256": hashlib.sha256(rollout_payload).hexdigest(),
            "rows": GROUP_COUNT,
            "traces": GROUP_COUNT * REASONING_SAMPLES,
            "candidates": GROUP_COUNT * REASONING_SAMPLES * ITEM_CANDIDATES,
        }
        if prereg_name != omit_metadata_for:
            metadata_path = rollout_path.with_suffix(rollout_path.suffix + ".meta.json")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return argparse.Namespace(**arguments)


def assert_blind_rejection_before_gold(
    args: argparse.Namespace, expected_message_fragment: str
) -> None:
    try:
        run_score(args)
    except (FileNotFoundError, ValueError) as exc:
        message = str(exc)
        if "gold ledger" in message or "GOLD_MUST_NOT_BE_OPENED" in message:
            raise AssertionError("gold was reached before blind precheck rejected") from exc
        if expected_message_fragment not in message:
            raise AssertionError(
                f"unexpected blind rejection: expected {expected_message_fragment!r}, "
                f"got {message!r}"
            ) from exc
    else:
        raise AssertionError("invalid blind artifacts were accepted")


def run_self_test() -> None:
    assert parse_item(
        "<|video_begin|><s_a_1><s_b_2><s_c_3>", "video", "self-test"
    ) == ("<s_a_1>", "<s_b_2>", "<s_c_3>")
    assert parse_item(
        "<|video_begin|><s_a_1><s_b_2><s_c_3> ", "video", "self-test"
    ) is None
    assert longest_prefix(("a", "b", "x"), ("a", "b", "c")) == 2
    gold_rows = [validate_gold_row(make_synthetic_gold(i), f"self-test:{i}") for i in range(16)]
    rollout_rows = []
    for index, gold in enumerate(gold_rows):
        row, seed = validate_rollout_row(
            make_synthetic_rollout(gold, index), gold, f"self-test:{index}"
        )
        assert seed == 1000 + index
        rollout_rows.append(row)
    stats = score_cell(rollout_rows, gold_rows)
    assert stats["counts"] == {
        "groups": 16,
        "reasoning_traces": 64,
        "candidates": 512,
        "valid_candidates": 512,
        "invalid_candidates": 0,
    }
    assert stats["candidate_lcp_histogram"] == {
        "0": 0,
        "1": 0,
        "2": 448,
        "3": 64,
    }
    assert stats["candidate_prefix_mass"] == 1088
    with tempfile.TemporaryDirectory(prefix="i29-score-self-test-") as directory:
        output = Path(directory) / "report.json"
        atomic_write_json_no_replace(output, {"ok": True})
        assert json.loads(output.read_text()) == {"ok": True}
        try:
            atomic_write_json_no_replace(output, {"ok": False})
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic writer overwrote an existing path")
        blind_root = Path(directory) / "blind-valid"
        blind_root.mkdir()
        valid_args = make_blind_test_args(blind_root)
        checked = blind_precheck_new_cells(valid_args)
        assert set(checked) == set(NEW_CELL_ALIASES)

        missing_root = Path(directory) / "blind-missing-meta"
        missing_root.mkdir()
        missing_args = make_blind_test_args(
            missing_root, omit_metadata_for="s800_canonical_user"
        )
        assert_blind_rejection_before_gold(missing_args, "missing")

        decode_root = Path(directory) / "blind-wrong-decode"
        decode_root.mkdir()
        decode_args = make_blind_test_args(
            decode_root, corrupt_decode_for="i23_legacy_system"
        )
        assert_blind_rejection_before_gold(decode_args, "decode lock mismatch")

        renderer_root = Path(directory) / "blind-wrong-renderer"
        renderer_root.mkdir()
        renderer_args = make_blind_test_args(
            renderer_root, corrupt_renderer_for="i23_canonical_user"
        )
        assert_blind_rejection_before_gold(renderer_args, "renderer lock mismatch")
    print("score_i29_renderer_pair.py self-test: PASS")


def load_legacy_preflight(
    gold_path: Path, legacy_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    gold_rows, gold_metadata = load_fixed_gold(gold_path)
    legacy_rows, legacy_metadata, _ = load_rollout_cell(
        legacy_path,
        gold_rows,
        allow_extra_rows=True,
        expected_adapter_sha256=S800_ARTIFACT_SHA256,
        prereg_cell_name=None,
        cell_alias=None,
        expected_full_sha256=S800_LEGACY_FULL_SHA256,
        expected_first16_raw_sha256=S800_LEGACY_FIRST16_RAW_SHA256,
        expected_config_file_sha256=S800_LEGACY_CONFIG_FILE_SHA256,
    )
    legacy_stats = score_cell(legacy_rows, gold_rows)
    assert_expected_legacy(legacy_stats)
    return gold_rows, gold_metadata, legacy_metadata, legacy_stats


def run_preflight(gold_path: Path, legacy_path: Path) -> None:
    _, gold_metadata, legacy_metadata, legacy_stats = load_legacy_preflight(
        gold_path, legacy_path
    )
    payload = {
        "status": "PASS",
        "protocol": "i29-fixed-first16-renderer-calibration",
        "gold": gold_metadata,
        "s800_legacy_system": legacy_metadata,
        "reproduced": {
            "counts": legacy_stats["counts"],
            "reasoning": legacy_stats["reasoning"],
            "candidate_lcp_histogram": legacy_stats["candidate_lcp_histogram"],
            "candidate_prefix_mass": legacy_stats["candidate_prefix_mass"],
            "group_max_lcp_histogram": legacy_stats["group_max_lcp_histogram"],
            "group_any_lcp_at_least": legacy_stats["group_any_lcp_at_least"],
            "exact": legacy_stats["exact"],
            "unique_prefix_coverage": legacy_stats["unique_prefix_coverage"],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_score(args: argparse.Namespace) -> None:
    # This must remain the first operation: the three completion-marked,
    # label-blind cells are fully authenticated before gold can be opened.
    blind_completion = blind_precheck_new_cells(args)
    gold_rows, gold_metadata, legacy_metadata, legacy_stats = load_legacy_preflight(
        args.gold, args.s800_legacy
    )
    cells: dict[str, list[dict[str, Any]]] = {"s800_legacy_system": []}
    metadata: dict[str, dict[str, Any]] = {
        "s800_legacy_system": legacy_metadata
    }
    seed_maps: dict[str, dict[str, int]] = {}
    legacy_rows, _, legacy_seeds = load_rollout_cell(
        args.s800_legacy,
        gold_rows,
        allow_extra_rows=True,
        expected_adapter_sha256=S800_ARTIFACT_SHA256,
        prereg_cell_name=None,
        cell_alias=None,
        expected_full_sha256=S800_LEGACY_FULL_SHA256,
        expected_first16_raw_sha256=S800_LEGACY_FIRST16_RAW_SHA256,
        expected_config_file_sha256=S800_LEGACY_CONFIG_FILE_SHA256,
    )
    cells["s800_legacy_system"] = legacy_rows
    seed_maps["s800_legacy_system"] = legacy_seeds
    for cell_name, path, alias in new_cell_specifications(args):
        completed = blind_completion[cell_name]
        rows, cell_metadata, reasoning_seeds = load_rollout_cell(
            path,
            gold_rows,
            allow_extra_rows=False,
            expected_adapter_sha256=alias["adapter_sha256"],
            prereg_cell_name=cell_name,
            cell_alias=alias,
            expected_full_sha256=completed["rollout_sha256"],
            expected_config_file_sha256=completed["config_file_sha256"],
        )
        cells[cell_name] = rows
        metadata[cell_name] = cell_metadata
        seed_maps[cell_name] = reasoning_seeds
    assert_cross_cell_integrity(metadata, seed_maps)
    stats = {
        "s800_legacy_system": legacy_stats,
        "s800_canonical_user": score_cell(cells["s800_canonical_user"], gold_rows),
        "i23_legacy_system": score_cell(cells["i23_legacy_system"], gold_rows),
        "i23_canonical_user": score_cell(cells["i23_canonical_user"], gold_rows),
    }
    canonical_model_comparison = compare_cells(
        "s800_canonical_user",
        stats["s800_canonical_user"],
        "i23_canonical_user",
        stats["i23_canonical_user"],
    )
    comparisons = {
        "canonical_s800_minus_i23": canonical_model_comparison,
        "legacy_s800_minus_i23": compare_cells(
            "s800_legacy_system",
            stats["s800_legacy_system"],
            "i23_legacy_system",
            stats["i23_legacy_system"],
        ),
        "s800_canonical_minus_legacy": compare_cells(
            "s800_canonical_user",
            stats["s800_canonical_user"],
            "s800_legacy_system",
            stats["s800_legacy_system"],
        ),
        "i23_canonical_minus_legacy": compare_cells(
            "i23_canonical_user",
            stats["i23_canonical_user"],
            "i23_legacy_system",
            stats["i23_legacy_system"],
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "population": "frozen first 16 manifest/gold groups in source order",
            "cells": "I-23/s800 x legacy-system/canonical-user renderer",
            "shape_per_cell": {
                "groups": GROUP_COUNT,
                "reasoning_samples_per_group": REASONING_SAMPLES,
                "item_candidates_per_reasoning": ITEM_CANDIDATES,
            },
            "candidate_metric": (
                "For each candidate, max LCP depth over every gold (a,b,c) in its "
                "group; invalid candidate depth is 0."
            ),
            "reasoning_stop_closed": "reasoning.stop_reason equals </think>",
            "gold_join_boundary": "CPU scorer only; generator receives no gold path",
        },
        "inputs": {
            "gold": gold_metadata,
            "label_blind_completion_precheck": blind_completion,
            "rollout_cells": metadata,
        },
        "integrity": {
            "status": "PASS",
            "global_seed": GLOBAL_SEED,
            "per_group_reasoning_seeds_identical_across_cells": True,
            "base_artifact_identical_across_cells": True,
            "same_adapter_within_each_renderer_pair": True,
            "new_cells_completion_marked_before_gold_access": True,
            "new_cell_aliases": NEW_CELL_ALIASES,
            "legacy_baseline_reproduced": True,
        },
        "cells": stats,
        "comparisons": comparisons,
        "canonical_model_direction": direction_summary(canonical_model_comparison),
    }
    atomic_write_json_no_replace(args.out, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "report": str(args.out.resolve()),
                "report_sha256": file_sha256(args.out.resolve()),
                "canonical_model_direction": report["canonical_model_direction"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
    elif args.preflight:
        run_preflight(args.gold, args.s800_legacy)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
