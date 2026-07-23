#!/usr/bin/env python3
"""Run the I34 material beam-gap probe.

This runner is deliberately a *ledger generator*, not a training-data
builder.  It evaluates the parent I19 adapter and the I23 teacher on the
same base model and the exact material renderer used by ``offline_eval.py``:
the fixed domain prefix followed by an unconstrained 64-wide, three-token
beam.  Optional ``--candidate NAME=PATH`` adapters use the same engine and
decode protocol, which makes later checkpoint comparisons reproducible.

The input pools are Alpaca rows with ``route`` set to ``beam_train_pool`` or
``beam_gate_pool``.  Gold labels are parsed strictly from ``output``.  Every
row is retained in the train/dev beam ledgers; the gap predicate is recorded
as metadata only:

    teacher full-gold hit AND parent full-gold miss

No formal training JSONL is emitted by this program.  The ledgers contain
complete valid ``(a,b,c)`` candidates and cumulative log probabilities,
invalid counts, token hashes, and enough metadata for an audited downstream
selector/trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE = PROJECT_ROOT / "models/OneReason-0.8B-pretrain-competition"
DEFAULT_PARENT = (
    PROJECT_ROOT / "submissions/i19_world_external_r96_s875_platform"
)
DEFAULT_TEACHER = (
    PROJECT_ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
)
DEFAULT_TRAIN_INPUT = PROJECT_ROOT / "logs/data/i34_material_beam_candidate_pool_v1.jsonl"
DEFAULT_DEV_INPUT = PROJECT_ROOT / "assets/evaluation/holdout/data_i34_material_beam_dev_v1.jsonl"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "logs/data/i34_material_beam_train_ledger_v1.jsonl"
DEFAULT_DEV_OUT = PROJECT_ROOT / "logs/probe/i34_material_beam_dev_ledger_v1.jsonl"
DEFAULT_AUDIT_OUT = PROJECT_ROOT / "logs/probe/i34_material_beam_gap_audit_v1.json"

# These are the currently published pool locks.  The data builder may publish
# a replacement freeze; callers should pass --hash-lock (or explicit expected
# hashes) for that freeze rather than silently accepting a changed file.
DEFAULT_TRAIN_INPUT_SHA256 = (
    "cb5500a3485aa5b093e70c3d3c53ac73d4485839f1945bc8d58b5eb3d5c19022"
)
DEFAULT_DEV_INPUT_SHA256 = (
    "fec7f5cb5dd642e83addd4d23ec1f7f0c6d3e285960a417e0520d27b6938401c"
)
DEFAULT_BASE_ARTIFACT_SHA256 = (
    "431cc7546a1813ed21a184974a1ac739139b7bdc4643d04e521d066f6ad20652"
)
DEFAULT_PARENT_ARTIFACT_SHA256 = (
    "3c6b694627803f5121ce2020cb4a32242c8a6f1671ec0e4f811f31579e937ba6"
)
DEFAULT_TEACHER_ARTIFACT_SHA256 = (
    "7c193b8db334fe23a2cc74774b8adbee15ce6ba0a260b3afd3fefbbe3cbbb4f1"
)

PARENT_ADAPTER_MODEL_SHA256 = (
    "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
)
TEACHER_ADAPTER_MODEL_SHA256 = (
    "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
)

SCHEMA_VERSION = "i34-material-beam-margin-v1"
AUDIT_SCHEMA_VERSION = "i34-material-beam-gap-audit-v1"
TASK = "material_desc2sid"
TRAIN_ROUTE = "beam_train_pool"
DEV_ROUTE = "beam_gate_pool"
EXPECTED_VLLM_VERSION = "0.12.0"
BEAM_WIDTH = 64
MAX_BEAM_TOKENS = 3
MAX_LORA_RANK = 128
MAX_HARD_NEGATIVES = 12
MAX_NEGATIVES_PER_DIVERGENCE = 4
DEFAULT_BATCH_SIZE = 16
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_MODEL_LEN = 40960
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SEED = 42
SMOKE_LIMIT = 2

DOMAIN_TOKENS = {
    "video": "<|video_begin|>",
    "ad": "<|ad_begin|>",
    "prod": "<|prod_begin|>",
    "living": "<|living_begin|>",
}
DOMAIN_ALIASES = {"live": "living", **{key: key for key in DOMAIN_TOKENS}}
GOLD_RE = re.compile(
    r"^<\|(video|ad|prod|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)
ITEM_RE = re.compile(
    r"^<\|(video|ad|prod|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_RUNTIME_SUFFIXES = {
    ".bin",
    ".json",
    ".jinja",
    ".model",
    ".safetensors",
    ".tiktoken",
    ".txt",
}


class InputError(ValueError):
    """Raised for a malformed or protocol-incompatible pool row."""


@dataclass(frozen=True)
class PreparedRow:
    """Validated input row plus the two prompt representations used here."""

    raw: dict[str, Any]
    row_sha256: str
    prompt_sha256: str
    canonical_prompt: str
    renderer_prompt: str
    renderer_prompt_sha256: str
    domain: str
    gold_abc: tuple[str, str, str]
    route: str
    source_prompt_sha256: str | None
    source_mode_prompt_sha256: str | None


@dataclass(frozen=True)
class Pool:
    path: Path
    route: str
    file_sha256: str
    total_rows: int
    rows: tuple[PreparedRow, ...]


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    path: Path
    request_id: int
    artifact: dict[str, Any]
    rank: int
    adapter_model_sha256: str
    adapter_config_sha256: str


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise InputError(f"non-finite JSON number is forbidden: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
    )


def normalize_domain(value: str) -> str:
    normalized = DOMAIN_ALIASES.get(value)
    if normalized is None:
        raise InputError(f"unsupported material domain: {value!r}")
    return normalized


def parse_gold_output(
    output: str, *, require_empty_think: bool = True
) -> tuple[str, tuple[str, str, str]]:
    """Parse one empty-think response and exactly one domain/SID triple."""
    if not isinstance(output, str):
        raise InputError("output must be a string")
    pieces = output.split("</think>")
    if len(pieces) < 2:
        raise InputError("output has no </think> delimiter")
    if require_empty_think:
        think_part = pieces[-2].strip()
        if think_part != "<think>":
            raise InputError(
                "I34 material output must use an empty <think> block; "
                "use --allow-nonempty-think only for an explicitly old freeze"
            )
    body = pieces[-1].strip()
    match = GOLD_RE.fullmatch(body)
    if match is None:
        raise InputError(
            "material output must end in exactly "
            "<|domain_begin|><s_a_n><s_b_n><s_c_n>"
        )
    domain, a, b, c = match.groups()
    return normalize_domain(domain), (a, b, c)


def canonical_user_prompt(row: Mapping[str, Any]) -> str:
    """Canonical trainer prompt (instruction/input only, no domain prefix)."""
    instruction = row["instruction"]
    user_input = row["input"]
    parts = [part for part in (instruction, user_input) if part]
    query = "\n".join(parts)
    return (
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def offline_eval_prompt(
    row: Mapping[str, Any], domain: str, *, require_native_no_think: bool = True
) -> str:
    """Exact offline material renderer + domain prefix.

    The I34 freeze is required to carry the native ``/no_think`` suffix.  An
    explicit compatibility mode can retain the historical offline_eval
    suffix append, but the default refuses a ``/think``/ambiguous row so a
    pool cannot silently change the model's context.
    """
    user_input = row["input"]
    if require_native_no_think:
        if not user_input.rstrip().endswith("/no_think"):
            raise InputError(
                "I34 material beam rows must end in native /no_think; "
                "use --allow-mode-rewrite only for an explicitly old freeze"
            )
    elif not user_input.rstrip().endswith("/no_think"):
        user_input += "/no_think"
    prompt = ""
    if row.get("instruction"):
        prompt += f"<|im_start|>system\n{row['instruction']}<|im_end|>\n"
    prompt += (
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )
    return prompt + DOMAIN_TOKENS[domain]


def validate_input_row(
    value: Any,
    route: str,
    location: str,
    *,
    require_native_no_think: bool = True,
    require_empty_think: bool = True,
) -> PreparedRow:
    if not isinstance(value, dict):
        raise InputError(f"{location} must be a JSON object")
    required = {"instruction", "input", "output", "history", "task"}
    missing = sorted(required - set(value))
    if missing:
        raise InputError(f"{location} is missing required keys: {missing}")
    for key in ("instruction", "input", "output", "task"):
        if not isinstance(value[key], str):
            raise InputError(f"{location}.{key} must be a string")
    if not isinstance(value["history"], list):
        raise InputError(f"{location}.history must be a list")
    declared_route = value.get("route")
    pool_role = value.get("pool_role")
    inferred_route = {
        "candidate": TRAIN_ROUTE,
        "development": DEV_ROUTE,
    }.get(pool_role)
    if declared_route is None:
        declared_route = inferred_route
    if declared_route != route:
        raise InputError(
            f"{location}.route/pool_role does not identify {route!r}"
        )
    if value["task"] != TASK:
        raise InputError(f"{location}.task must be {TASK!r}")
    if "schema_version" in value and value["schema_version"] != "i34-material-beam-pool-v1":
        raise InputError(
            f"{location}.schema_version is not i34-material-beam-pool-v1"
        )
    if not value["input"].strip():
        raise InputError(f"{location}.input must not be empty")
    domain, gold_abc = parse_gold_output(
        value["output"], require_empty_think=require_empty_think
    )
    canonical_prompt = canonical_user_prompt(value)
    renderer_prompt = offline_eval_prompt(
        value, domain, require_native_no_think=require_native_no_think
    )
    core_row = {
        "instruction": value["instruction"],
        "input": value["input"],
        "output": value["output"],
        "history": value["history"],
    }
    row_sha = canonical_sha256(core_row)
    source_row_sha = value.get("row_sha256")
    if source_row_sha is not None:
        if not isinstance(source_row_sha, str) or source_row_sha != row_sha:
            raise InputError(
                f"{location}.row_sha256 does not match normalized core row"
            )
    source_prompt_sha = value.get("prompt_sha256")
    expected_source_prompt_sha = canonical_sha256(
        [value["instruction"], value["input"], value["history"]]
    )
    if source_prompt_sha is not None:
        if not isinstance(source_prompt_sha, str) or source_prompt_sha != expected_source_prompt_sha:
            raise InputError(f"{location}.prompt_sha256 does not match source prompt digest")
    source_mode_sha = value.get("mode_prompt_sha256")
    if source_mode_sha is not None and not isinstance(source_mode_sha, str):
        raise InputError(f"{location}.mode_prompt_sha256 must be a string")
    expected_sid = f"{DOMAIN_TOKENS[domain]}<s_a_{gold_abc[0]}><s_b_{gold_abc[1]}><s_c_{gold_abc[2]}>"
    if "gold_sid" in value and value["gold_sid"] != expected_sid:
        raise InputError(f"{location}.gold_sid disagrees with output")
    if "gold_domain" in value and normalize_domain(str(value["gold_domain"])) != domain:
        raise InputError(f"{location}.gold_domain disagrees with output")
    for key, expected in (
        ("gold_s_a", int(gold_abc[0])),
        ("gold_s_b", int(gold_abc[1])),
        ("gold_s_c", int(gold_abc[2])),
    ):
        if key in value and value[key] != expected:
            raise InputError(f"{location}.{key} disagrees with output")
    raw_with_route = dict(value)
    raw_with_route.setdefault("route", route)
    return PreparedRow(
        raw=raw_with_route,
        row_sha256=row_sha,
        prompt_sha256=hashlib.sha256(
            canonical_prompt.encode("utf-8")
        ).hexdigest(),
        canonical_prompt=canonical_prompt,
        renderer_prompt=renderer_prompt,
        renderer_prompt_sha256=hashlib.sha256(
            renderer_prompt.encode("utf-8")
        ).hexdigest(),
        domain=domain,
        gold_abc=gold_abc,
        route=route,
        source_prompt_sha256=source_prompt_sha,
        source_mode_prompt_sha256=source_mode_sha,
    )


def parse_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "all", "none"}:
        return None
    if text == "smoke":
        return SMOKE_LIMIT
    try:
        parsed = int(text)
    except ValueError as exc:
        raise InputError("--limit must be an integer, 'smoke', or 'all'") from exc
    if parsed < 1:
        raise InputError("--limit integer must be >= 1")
    return parsed


def load_pool(
    path: Path,
    route: str,
    limit: int | None,
    *,
    require_native_no_think: bool = True,
    require_empty_think: bool = True,
) -> Pool:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_digest = file_sha256(path)
    rows: list[PreparedRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise InputError(f"blank JSONL row at {path}:{line_number}")
            try:
                value = strict_json_loads(line)
                prepared = validate_input_row(
                    value,
                    route,
                    f"{path}:{line_number}",
                    require_native_no_think=require_native_no_think,
                    require_empty_think=require_empty_think,
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise InputError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if prepared.row_sha256 in seen:
                raise InputError(
                    f"duplicate canonical row hash at {path}:{line_number}: "
                    f"{prepared.row_sha256}"
                )
            seen.add(prepared.row_sha256)
            rows.append(prepared)
    selected = rows if limit is None else rows[:limit]
    if not selected:
        raise InputError(f"pool is empty: {path}")
    return Pool(
        path=path,
        route=route,
        file_sha256=raw_digest,
        total_rows=len(rows),
        rows=tuple(selected),
    )


def token_hash(ids: Sequence[int]) -> str:
    """Hash exactly the trainer's little-endian uint32 token byte stream."""
    normalized: list[int] = []
    for index, value in enumerate(ids):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"token id {index} is not an integer: {value!r}")
        if not 0 <= value < 2**32:
            raise ValueError(f"token id {index} is outside uint32: {value}")
        normalized.append(value)
    packed = struct.pack(f"<{len(normalized)}I", *normalized)
    return hashlib.sha256(packed).hexdigest()


def tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if ids is None and isinstance(encoded, (list, tuple)):
        ids = encoded
    if ids is None:
        raise ValueError("tokenizer did not return input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(
        ids[0], (list, tuple)
    ):
        ids = ids[0]
    if not isinstance(ids, (list, tuple)):
        raise ValueError("tokenizer input_ids must be a flat sequence")
    result: list[int] = []
    for value in ids:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"tokenizer returned a non-integer token id: {value!r}")
        result.append(value)
    return result


def convert_token_ids(tokenizer: Any, tokens: Sequence[str]) -> list[int]:
    converted = tokenizer.convert_tokens_to_ids(list(tokens))
    if isinstance(converted, int) and not isinstance(converted, bool):
        converted = [converted]
    if not isinstance(converted, (list, tuple)) or len(converted) != len(tokens):
        raise ValueError(f"tokenizer could not convert tokens: {tokens!r}")
    result: list[int] = []
    for token, value in zip(tokens, converted):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid token id for {token!r}: {value!r}")
        result.append(value)
    return result


def row_token_metadata(row: PreparedRow, tokenizer: Any) -> dict[str, Any]:
    prompt_ids = tokenizer_encode(tokenizer, row.canonical_prompt)
    domain_id, a_id, b_id, c_id = convert_token_ids(
        tokenizer,
        [
            DOMAIN_TOKENS[row.domain],
            f"<s_a_{row.gold_abc[0]}>",
            f"<s_b_{row.gold_abc[1]}>",
            f"<s_c_{row.gold_abc[2]}>",
        ],
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if hasattr(eos_id, "item"):
        eos_id = eos_id.item()
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("tokenizer has no valid eos_token_id")
    return {
        "prompt_token_sha256": token_hash(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "gold_tokens": [domain_id, a_id, b_id, c_id, eos_id],
        "positive_tokens": [[a_id, b_id, c_id]],
        "gold_token_parts": {
            "domain": domain_id,
            "a": a_id,
            "b": b_id,
            "c": c_id,
            "eos": eos_id,
        },
    }


def artifact_fingerprint(path: Path, *, adapter: bool) -> dict[str, Any]:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    if adapter:
        candidates = [root / "adapter_config.json", root / "adapter_model.safetensors"]
        missing = [str(item) for item in candidates if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"invalid adapter; missing {missing}")
    else:
        if not (root / "config.json").is_file():
            raise FileNotFoundError(f"base model has no config.json: {root}")
        candidates = sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in MODEL_RUNTIME_SUFFIXES
        )
        if not any(item.suffix.lower() in {".safetensors", ".bin"} for item in candidates):
            raise FileNotFoundError(f"base model has no local weight file: {root}")
    files: list[dict[str, Any]] = []
    for item in candidates:
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    return {
        "path": str(root),
        "artifact_sha256": canonical_sha256(files),
        "files": files,
    }


def adapter_rank(path: Path) -> int:
    config_path = path / "adapter_config.json"
    value = strict_json_loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"adapter config is not an object: {config_path}")
    ranks: list[int] = []
    rank = value.get("r")
    if isinstance(rank, int) and not isinstance(rank, bool):
        ranks.append(rank)
    pattern = value.get("rank_pattern", {})
    if isinstance(pattern, dict):
        ranks.extend(
            item
            for item in pattern.values()
            if isinstance(item, int) and not isinstance(item, bool)
        )
    if not ranks or min(ranks) < 1 or max(ranks) > MAX_LORA_RANK:
        raise ValueError(
            f"adapter rank must be 1..{MAX_LORA_RANK}: {config_path}"
        )
    return max(ranks)


def parse_candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not NAME_RE.fullmatch(name):
        raise ValueError(
            "candidate NAME must contain only letters, digits, '_', '-', or '.'"
        )
    if not path:
        raise ValueError("candidate PATH must not be empty")
    return name, Path(path).expanduser()


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_hash_lock_value(value: Any, aliases: set[str]) -> str | None:
    wanted = {_normalized_key(item) for item in aliases}
    if isinstance(value, dict):
        for key, child in value.items():
            if _normalized_key(str(key)) in wanted and isinstance(child, str):
                if HEX64_RE.fullmatch(child):
                    return child
            found = find_hash_lock_value(child, aliases)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_hash_lock_value(child, aliases)
            if found is not None:
                return found
    return None


def load_hash_lock(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--hash-lock must contain a JSON object")
    return value


def expected_hash(
    cli_value: str | None,
    lock: Mapping[str, Any],
    aliases: set[str],
    label: str,
) -> str | None:
    value = cli_value or find_hash_lock_value(lock, aliases)
    if value is None:
        return None
    if not HEX64_RE.fullmatch(value):
        raise ValueError(f"{label} expected hash is not a lowercase SHA256")
    return value


def check_expected(actual: str, expected: str | None, label: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def finite_score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sequence_text_and_tokens(
    sequence: Any, prompt: str, tokenizer: Any
) -> tuple[str, list[int], int]:
    logprobs = getattr(sequence, "logprobs", None)
    generated_count = len(logprobs) if logprobs is not None else 0
    tokens = getattr(sequence, "tokens", None)
    generated_tokens: list[int] = []
    if tokens is not None and generated_count:
        generated_tokens = list(tokens)[-generated_count:]
    generated_text = ""
    if generated_tokens:
        generated_text = tokenizer.decode(generated_tokens)
    sequence_text = getattr(sequence, "text", None)
    if isinstance(sequence_text, str):
        if sequence_text.startswith(prompt):
            generated_text = sequence_text[len(prompt) :]
        elif not generated_text:
            generated_text = sequence_text
    return generated_text, generated_tokens, generated_count


def parse_candidate_text(text: str) -> tuple[str, str, str] | None:
    match = ITEM_RE.fullmatch(text)
    return tuple(match.groups()[1:]) if match is not None else None


def parse_beam_sequences(
    beam_output: Any, prompt: str, domain: str, tokenizer: Any
) -> tuple[list[dict[str, Any]], list[int]]:
    sequences = getattr(beam_output, "sequences", None)
    if not isinstance(sequences, (list, tuple)):
        raise RuntimeError("vLLM beam output has no sequence list")
    if len(sequences) != BEAM_WIDTH:
        raise RuntimeError(
            f"vLLM returned {len(sequences)} beams; expected {BEAM_WIDTH}"
        )
    prefix = DOMAIN_TOKENS[domain]
    valid: list[dict[str, Any]] = []
    invalid_ranks: list[int] = []
    for rank, sequence in enumerate(sequences):
        generated_text, generated_tokens, generated_count = _sequence_text_and_tokens(
            sequence, prompt, tokenizer
        )
        full_text = generated_text if generated_text.startswith(prefix) else prefix + generated_text
        abc = parse_candidate_text(full_text)
        score = finite_score(getattr(sequence, "cum_logprob", None))
        if abc is None:
            invalid_ranks.append(rank)
            continue
        valid.append(
            {
                "abc": list(abc),
                "cum_logprob": score,
                "rank": rank,
                "text": full_text,
                "token_count": generated_count,
                "finish_reason": getattr(sequence, "finish_reason", None),
                "stop_reason": getattr(sequence, "stop_reason", None),
                "generated_token_ids": generated_tokens,
            }
        )
    return valid, invalid_ranks


def first_divergence(gold: Sequence[str], candidate: Sequence[str]) -> int:
    for index, (expected, actual) in enumerate(zip(gold, candidate)):
        if expected != actual:
            return index
    raise ValueError("candidate is identical to gold; cannot be a hard negative")


def _score_map(candidates: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], float | None]:
    result: dict[tuple[str, str, str], float | None] = {}
    for candidate in candidates:
        key = tuple(str(item) for item in candidate["abc"])
        score = candidate.get("cum_logprob")
        if key not in result:
            result[key] = score
        elif score is not None and (
            result[key] is None or score > result[key]
        ):
            result[key] = score
    return result


def hard_negative_rows(
    gold: Sequence[str],
    parent_candidates: Sequence[Mapping[str, Any]],
    teacher_candidates: Sequence[Mapping[str, Any]],
    token_ids_by_abc: Mapping[tuple[str, str, str], list[int]],
    *,
    tokenizer: Any | None = None,
    max_total: int | None = MAX_HARD_NEGATIVES,
    max_per_divergence: int | None = MAX_NEGATIVES_PER_DIVERGENCE,
) -> list[dict[str, Any]]:
    teacher_scores = _score_map(teacher_candidates)
    seen: set[tuple[str, str, str]] = set()
    divergence_counts = [0, 0, 0]
    negatives: list[dict[str, Any]] = []
    for candidate in parent_candidates:
        key = tuple(str(item) for item in candidate["abc"])
        if key in seen or key == tuple(gold):
            continue
        seen.add(key)
        token_ids = token_ids_by_abc.get(key)
        if token_ids is None and tokenizer is not None:
            token_ids = convert_token_ids(
                tokenizer,
                [
                    f"<s_a_{key[0]}>",
                    f"<s_b_{key[1]}>",
                    f"<s_c_{key[2]}>",
                ],
            )
        if token_ids is None:
            continue
        parent_score = finite_score(candidate.get("cum_logprob"))
        # The trainer uses parent_score as a ranking signal; null/non-finite
        # scores are retained in the complete beam ledger but are not emitted
        # as hard negatives.
        if parent_score is None:
            continue
        divergence = first_divergence(gold, key)
        if max_per_divergence is not None and divergence_counts[divergence] >= max_per_divergence:
            continue
        if max_total is not None and len(negatives) >= max_total:
            break
        teacher_score = finite_score(teacher_scores.get(key))
        negative = {
            "tokens": list(token_ids),
            "abc": list(key),
            "first_divergence": divergence,
            "parent_beam_rank": int(candidate["rank"]),
            "parent_score": parent_score,
        }
        if teacher_score is not None:
            negative["teacher_score"] = teacher_score
        negatives.append(
            negative
        )
        divergence_counts[divergence] += 1
    return negatives


def _compact_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for candidate in candidates:
        compact.append(
            {
                "abc": list(candidate["abc"]),
                "cum_logprob": candidate.get("cum_logprob"),
                "rank": int(candidate["rank"]),
                "text": candidate.get("text", ""),
                "token_count": int(candidate.get("token_count", 0)),
                "finish_reason": candidate.get("finish_reason"),
                "stop_reason": candidate.get("stop_reason"),
            }
        )
    return compact


def build_ledger_row(
    row: PreparedRow,
    token_meta: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    parent_name: str,
    teacher_name: str,
    token_ids_by_abc: Mapping[tuple[str, str, str], list[int]],
    *,
    tokenizer: Any | None = None,
    parent_adapter_sha256: str | None = None,
    teacher_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    parent = results[parent_name]
    teacher = results[teacher_name]
    gold_key = tuple(row.gold_abc)
    parent_hit = any(tuple(item["abc"]) == gold_key for item in parent["candidates"])
    teacher_hit = any(tuple(item["abc"]) == gold_key for item in teacher["candidates"])
    all_negatives = hard_negative_rows(
        row.gold_abc,
        parent["candidates"],
        teacher["candidates"],
        token_ids_by_abc,
        tokenizer=tokenizer,
        max_total=None,
        max_per_divergence=None,
    )
    negatives = hard_negative_rows(
        row.gold_abc,
        parent["candidates"],
        teacher["candidates"],
        token_ids_by_abc,
        tokenizer=tokenizer,
    )
    candidate_results: dict[str, Any] = {}
    for name, result in results.items():
        candidate_results[name] = {
            "full_gold_hit": any(
                tuple(item["abc"]) == gold_key for item in result["candidates"]
            ),
            "valid_candidates": _compact_candidates(result["candidates"]),
            "invalid_count": len(result["invalid_ranks"]),
            "invalid_ranks": list(result["invalid_ranks"]),
            "beam_count": int(result["beam_count"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "route": row.raw["route"],
        "row_sha256": row.row_sha256,
        "prompt_sha256": row.prompt_sha256,
        "source_prompt_sha256": row.source_prompt_sha256,
        "source_mode_prompt_sha256": row.source_mode_prompt_sha256,
        "prompt_token_sha256": token_meta["prompt_token_sha256"],
        "prompt_token_count": token_meta["prompt_token_count"],
        "renderer_prompt_sha256": row.renderer_prompt_sha256,
        "domain": row.domain,
        "gold_abc": list(row.gold_abc),
        "parent_adapter_sha256": parent_adapter_sha256,
        "teacher_adapter_sha256": teacher_adapter_sha256,
        "gold_tokens": list(token_meta["gold_tokens"]),
        "positive_tokens": [list(item) for item in token_meta["positive_tokens"]],
        "parent": {
            "name": parent_name,
            "full_gold_hit": parent_hit,
            "valid_candidates": _compact_candidates(parent["candidates"]),
            "invalid_count": len(parent["invalid_ranks"]),
            "invalid_ranks": list(parent["invalid_ranks"]),
            "beam_count": int(parent["beam_count"]),
        },
        "teacher": {
            "name": teacher_name,
            "full_gold_hit": teacher_hit,
            "valid_candidates": _compact_candidates(teacher["candidates"]),
            "invalid_count": len(teacher["invalid_ranks"]),
            "invalid_ranks": list(teacher["invalid_ranks"]),
            "beam_count": int(teacher["beam_count"]),
        },
        "candidate_results": candidate_results,
        "gap_selection": {
            "definition": "teacher_full_gold_hit_and_parent_full_gold_miss",
            "selected": teacher_hit and not parent_hit,
            "hard_negative_count": len(negatives),
            "hard_negative_pool_count": len(all_negatives),
            "hard_negative_dropped_count": len(all_negatives) - len(negatives),
            "trainer_ready": bool(teacher_hit and not parent_hit and negatives),
        },
        "hard_negatives": negatives,
        "formal_training_generated": False,
    }


def run_adapter_beam(
    llm: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    request: Any,
    domains: Sequence[str],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run the frozen offline beam call for one adapter, in stable order."""
    from vllm.sampling_params import BeamSearchParams

    params = BeamSearchParams(beam_width=BEAM_WIDTH, max_tokens=MAX_BEAM_TOKENS)
    outputs: list[dict[str, Any]] = []
    for offset in range(0, len(prompts), batch_size):
        prompt_chunk = prompts[offset : offset + batch_size]
        domain_chunk = domains[offset : offset + batch_size]
        beam_outputs = llm.beam_search(
            [{"prompt": prompt} for prompt in prompt_chunk],
            params,
            lora_request=request,
            use_tqdm=False,
        )
        if len(beam_outputs) != len(prompt_chunk):
            raise RuntimeError("vLLM returned a different number of beam responses")
        for prompt, domain, beam_output in zip(
            prompt_chunk, domain_chunk, beam_outputs
        ):
            candidates, invalid_ranks = parse_beam_sequences(
                beam_output, prompt, domain, tokenizer
            )
            outputs.append(
                {
                    "candidates": candidates,
                    "invalid_ranks": invalid_ranks,
                    "beam_count": BEAM_WIDTH,
                }
            )
    return outputs


def atomic_write_bytes(path: Path, payload: bytes, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing output (use --overwrite): {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def atomic_write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], overwrite: bool = False
) -> str:
    payload = jsonl_payload(rows)
    atomic_write_bytes(path, payload, overwrite=overwrite)
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any], overwrite: bool = False) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, payload, overwrite=overwrite)
    return hashlib.sha256(payload).hexdigest()


def make_adapter_specs(
    parent_path: Path,
    teacher_path: Path,
    candidate_values: Sequence[str],
) -> tuple[AdapterSpec, ...]:
    parsed: list[tuple[str, Path]] = [
        ("parent_i19_r96", parent_path),
        ("teacher_i23_r64", teacher_path),
    ]
    parsed.extend(parse_candidate_spec(value) for value in candidate_values)
    names: set[str] = set()
    specs: list[AdapterSpec] = []
    for request_id, (name, path) in enumerate(parsed, 1):
        if name in names:
            raise ValueError(f"duplicate adapter name: {name}")
        names.add(name)
        resolved = path.expanduser().resolve()
        rank = adapter_rank(resolved)
        artifact = artifact_fingerprint(resolved, adapter=True)
        model_sha = next(
            item["sha256"]
            for item in artifact["files"]
            if item["path"] == "adapter_model.safetensors"
        )
        config_sha = next(
            item["sha256"]
            for item in artifact["files"]
            if item["path"] == "adapter_config.json"
        )
        specs.append(
            AdapterSpec(
                name=name,
                path=resolved,
                request_id=request_id,
                artifact=artifact,
                rank=rank,
                adapter_model_sha256=model_sha,
                adapter_config_sha256=config_sha,
            )
        )
    return tuple(specs)


def build_token_id_lookup(tokenizer: Any, rows: Sequence[PreparedRow]) -> dict[tuple[str, str, str], list[int]]:
    lookup: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        key = tuple(row.gold_abc)
        if key in lookup:
            continue
        lookup[key] = convert_token_ids(
            tokenizer,
            [
                f"<s_a_{key[0]}>",
                f"<s_b_{key[1]}>",
                f"<s_c_{key[2]}>",
            ],
        )
    return lookup


def summarize_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
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
    for row in rows:
        parent = row["parent"]
        teacher = row["teacher"]
        result["gap_selected"] += int(row["gap_selection"]["selected"])
        result["trainer_ready"] += int(row["gap_selection"]["trainer_ready"])
        result["parent_full_gold_hits"] += int(parent["full_gold_hit"])
        result["teacher_full_gold_hits"] += int(teacher["full_gold_hit"])
        result["parent_invalid_candidates"] += int(parent["invalid_count"])
        result["teacher_invalid_candidates"] += int(teacher["invalid_count"])
        result["parent_valid_candidates"] += len(parent["valid_candidates"])
        result["teacher_valid_candidates"] += len(teacher["valid_candidates"])
        result["hard_negatives"] += len(row["hard_negatives"])
        result["hard_negative_pool"] += int(row["gap_selection"]["hard_negative_pool_count"])
        result["hard_negative_dropped"] += int(row["gap_selection"]["hard_negative_dropped_count"])
    return result


def expected_values(args: argparse.Namespace, lock: Mapping[str, Any]) -> dict[str, str | None]:
    # The published default freeze is locked even when the caller does not
    # provide a separate lock file.  Custom pools remain opt-in and must pass
    # their own expected hashes if they need a hard lock.
    train_cli = args.expected_train_sha256
    dev_cli = args.expected_dev_sha256
    base_cli = args.expected_base_artifact_sha256
    parent_cli = args.expected_parent_artifact_sha256
    teacher_cli = args.expected_teacher_artifact_sha256
    parent_model_cli = args.expected_parent_model_sha256
    teacher_model_cli = args.expected_teacher_model_sha256
    if train_cli is None and Path(args.train_input).resolve() == DEFAULT_TRAIN_INPUT.resolve():
        train_cli = DEFAULT_TRAIN_INPUT_SHA256
    if dev_cli is None and Path(args.dev_input).resolve() == DEFAULT_DEV_INPUT.resolve():
        dev_cli = DEFAULT_DEV_INPUT_SHA256
    if base_cli is None and Path(args.base).resolve() == DEFAULT_BASE.resolve():
        base_cli = DEFAULT_BASE_ARTIFACT_SHA256
    if parent_cli is None and Path(args.parent_adapter).resolve() == DEFAULT_PARENT.resolve():
        parent_cli = DEFAULT_PARENT_ARTIFACT_SHA256
    if teacher_cli is None and Path(args.teacher_adapter).resolve() == DEFAULT_TEACHER.resolve():
        teacher_cli = DEFAULT_TEACHER_ARTIFACT_SHA256
    if parent_model_cli is None and Path(args.parent_adapter).resolve() == DEFAULT_PARENT.resolve():
        parent_model_cli = PARENT_ADAPTER_MODEL_SHA256
    if teacher_model_cli is None and Path(args.teacher_adapter).resolve() == DEFAULT_TEACHER.resolve():
        teacher_model_cli = TEACHER_ADAPTER_MODEL_SHA256
    return {
        "train_input": expected_hash(
            train_cli,
            lock,
            {"train_input_sha256", "train_sha256", "i34_train_sha256"},
            "train input",
        ),
        "dev_input": expected_hash(
            dev_cli,
            lock,
            {"dev_input_sha256", "gate_input_sha256", "dev_sha256", "i34_dev_sha256"},
            "dev input",
        ),
        "base_artifact": expected_hash(
            base_cli,
            lock,
            {"base_artifact_sha256", "base_sha256", "model_artifact_sha256"},
            "base artifact",
        ),
        "parent_artifact": expected_hash(
            parent_cli,
            lock,
            {"parent_artifact_sha256", "parent_sha256", "i19_artifact_sha256"},
            "parent artifact",
        ),
        "teacher_artifact": expected_hash(
            teacher_cli,
            lock,
            {"teacher_artifact_sha256", "teacher_sha256", "i23_artifact_sha256"},
            "teacher artifact",
        ),
        "parent_model": expected_hash(
            parent_model_cli,
            lock,
            {"parent_adapter_model_sha256", "parent_model_sha256", "i19_model_sha256"},
            "parent adapter model",
        ),
        "teacher_model": expected_hash(
            teacher_model_cli,
            lock,
            {"teacher_adapter_model_sha256", "teacher_model_sha256", "i23_model_sha256"},
            "teacher adapter model",
        ),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    limit = parse_limit(args.limit)
    require_native = not args.allow_mode_rewrite
    require_empty = not args.allow_nonempty_think
    train = load_pool(
        Path(args.train_input),
        TRAIN_ROUTE,
        limit,
        require_native_no_think=require_native,
        require_empty_think=require_empty,
    )
    dev = load_pool(
        Path(args.dev_input),
        DEV_ROUTE,
        limit,
        require_native_no_think=require_native,
        require_empty_think=require_empty,
    )
    base_artifact = artifact_fingerprint(Path(args.base), adapter=False)
    specs = make_adapter_specs(Path(args.parent_adapter), Path(args.teacher_adapter), args.candidate)
    lock = load_hash_lock(Path(args.hash_lock) if args.hash_lock else None)
    expected = expected_values(args, lock)
    check_expected(train.file_sha256, expected["train_input"], "train input")
    check_expected(dev.file_sha256, expected["dev_input"], "dev input")
    check_expected(base_artifact["artifact_sha256"], expected["base_artifact"], "base artifact")
    check_expected(specs[0].artifact["artifact_sha256"], expected["parent_artifact"], "parent artifact")
    check_expected(specs[1].artifact["artifact_sha256"], expected["teacher_artifact"], "teacher artifact")
    check_expected(specs[0].adapter_model_sha256, expected["parent_model"], "parent adapter model")
    check_expected(specs[1].adapter_model_sha256, expected["teacher_model"], "teacher adapter model")
    return {
        "train": {"path": str(train.path), "rows": len(train.rows), "total_rows": train.total_rows, "sha256": train.file_sha256},
        "dev": {"path": str(dev.path), "rows": len(dev.rows), "total_rows": dev.total_rows, "sha256": dev.file_sha256},
        "base": base_artifact,
        "adapters": [
            {
                "name": spec.name,
                "path": str(spec.path),
                "rank": spec.rank,
                "artifact_sha256": spec.artifact["artifact_sha256"],
                "adapter_model_sha256": spec.adapter_model_sha256,
                "adapter_config_sha256": spec.adapter_config_sha256,
            }
            for spec in specs
        ],
        "limit": limit,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if "," in args.gpu or not args.gpu.strip():
        raise ValueError("--gpu must name exactly one CUDA device")
    limit = parse_limit(args.limit)
    require_native = not args.allow_mode_rewrite
    require_empty = not args.allow_nonempty_think
    train = load_pool(
        Path(args.train_input),
        TRAIN_ROUTE,
        limit,
        require_native_no_think=require_native,
        require_empty_think=require_empty,
    )
    dev = load_pool(
        Path(args.dev_input),
        DEV_ROUTE,
        limit,
        require_native_no_think=require_native,
        require_empty_think=require_empty,
    )
    base = Path(args.base).expanduser().resolve()
    base_artifact = artifact_fingerprint(base, adapter=False)
    specs = make_adapter_specs(Path(args.parent_adapter), Path(args.teacher_adapter), args.candidate)
    lock = load_hash_lock(Path(args.hash_lock) if args.hash_lock else None)
    expected = expected_values(args, lock)
    check_expected(train.file_sha256, expected["train_input"], "train input")
    check_expected(dev.file_sha256, expected["dev_input"], "dev input")
    check_expected(base_artifact["artifact_sha256"], expected["base_artifact"], "base artifact")
    check_expected(specs[0].artifact["artifact_sha256"], expected["parent_artifact"], "parent artifact")
    check_expected(specs[1].artifact["artifact_sha256"], expected["teacher_artifact"], "teacher artifact")
    check_expected(specs[0].adapter_model_sha256, expected["parent_model"], "parent adapter model")
    check_expected(specs[1].adapter_model_sha256, expected["teacher_model"], "teacher adapter model")

    output_paths = [Path(args.train_output), Path(args.dev_output), Path(args.audit_output)]
    if len({str(path.resolve()) for path in output_paths}) != len(output_paths):
        raise ValueError("train, dev, and audit outputs must be distinct")
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing outputs (use --overwrite): "
                + ", ".join(existing)
            )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import vllm
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    version = str(getattr(vllm, "__version__", "unknown"))
    if version != args.expected_vllm_version:
        raise RuntimeError(
            f"vLLM version mismatch: expected {args.expected_vllm_version}, got {version}"
        )
    llm = LLM(
        model=str(base),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=args.seed,
        enable_prefix_caching=True,
        trust_remote_code=True,
        max_logprobs=130,
        enable_lora=True,
        max_lora_rank=MAX_LORA_RANK,
    )
    tokenizer = llm.get_tokenizer()
    requests = {
        spec.name: LoRARequest(spec.name, spec.request_id, lora_path=str(spec.path))
        for spec in specs
    }
    all_rows = list(train.rows) + list(dev.rows)
    token_meta_by_hash = {
        row.row_sha256: row_token_metadata(row, tokenizer) for row in all_rows
    }
    token_ids_by_abc = build_token_id_lookup(tokenizer, all_rows)

    def generate_pool(pool: Pool) -> list[dict[str, Any]]:
        prompts = [row.renderer_prompt for row in pool.rows]
        domains = [row.domain for row in pool.rows]
        generated: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            generated[spec.name] = run_adapter_beam(
                llm,
                tokenizer,
                prompts,
                requests[spec.name],
                domains,
                args.batch_size,
            )
        # Candidate beams contain triples not present as gold labels.  Resolve
        # every legal triple once so hard-negative sidecar entries always have
        # numeric token IDs rather than silently dropping unseen candidates.
        for result_rows in generated.values():
            for result in result_rows:
                for candidate in result["candidates"]:
                    key = tuple(str(item) for item in candidate["abc"])
                    if key not in token_ids_by_abc:
                        token_ids_by_abc[key] = convert_token_ids(
                            tokenizer,
                            [
                                f"<s_a_{key[0]}>",
                                f"<s_b_{key[1]}>",
                                f"<s_c_{key[2]}>",
                            ],
                        )
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(pool.rows):
            per_row = {name: generated[name][index] for name in generated}
            rows.append(
                build_ledger_row(
                    row,
                    token_meta_by_hash[row.row_sha256],
                    per_row,
                    specs[0].name,
                    specs[1].name,
                    token_ids_by_abc,
                    tokenizer=tokenizer,
                    parent_adapter_sha256=specs[0].adapter_model_sha256,
                    teacher_adapter_sha256=specs[1].adapter_model_sha256,
                )
            )
        return rows

    train_ledger = generate_pool(train)
    dev_ledger = generate_pool(dev)
    train_output_sha = atomic_write_jsonl(
        Path(args.train_output), train_ledger, overwrite=args.overwrite
    )
    dev_output_sha = atomic_write_jsonl(
        Path(args.dev_output), dev_ledger, overwrite=args.overwrite
    )
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "complete",
        "formal_training_generated": False,
        "selection_definition": "teacher_full_gold_hit_and_parent_full_gold_miss",
        "input_pools": {
            "train": {
                "path": str(train.path),
                "route": train.route,
                "total_rows": train.total_rows,
                "selected_rows": len(train.rows),
                "sha256": train.file_sha256,
            },
            "dev": {
                "path": str(dev.path),
                "route": dev.route,
                "total_rows": dev.total_rows,
                "selected_rows": len(dev.rows),
                "sha256": dev.file_sha256,
            },
        },
        "artifacts": {
            "base": base_artifact,
            "adapters": [
                {
                    "name": spec.name,
                    "request_id": spec.request_id,
                    "path": str(spec.path),
                    "rank": spec.rank,
                    "adapter_model_sha256": spec.adapter_model_sha256,
                    "adapter_config_sha256": spec.adapter_config_sha256,
                    "artifact": spec.artifact,
                }
                for spec in specs
            ],
        },
        "runtime": {
            "vllm_version": version,
            "gpu": args.gpu,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "single_engine": True,
            "renderer": "offline_eval.prompt_of(mode='nothink') + fixed domain prefix",
            "native_no_think_required": not args.allow_mode_rewrite,
            "empty_think_required": not args.allow_nonempty_think,
            "beam_width": BEAM_WIDTH,
            "max_tokens": MAX_BEAM_TOKENS,
            "beam_constraints": {},
            "batch_size": args.batch_size,
            "rank_base": 0,
            "hard_negative_max_total": MAX_HARD_NEGATIVES,
            "hard_negative_max_per_first_divergence": MAX_NEGATIVES_PER_DIVERGENCE,
        },
        "adapters": {
            "parent": specs[0].name,
            "teacher": specs[1].name,
            "optional_candidates": [spec.name for spec in specs[2:]],
            "distinct_lora_ids_and_names": True,
        },
        "counts": {
            "train": summarize_counts(train_ledger),
            "dev": summarize_counts(dev_ledger),
        },
        "outputs": {
            "train_ledger": {
                "path": str(Path(args.train_output).resolve()),
                "rows": len(train_ledger),
                "sha256": train_output_sha,
            },
            "dev_ledger": {
                "path": str(Path(args.dev_output).resolve()),
                "rows": len(dev_ledger),
                "sha256": dev_output_sha,
            },
            "formal_train": None,
        },
        "hash_lock": {
            "source": str(Path(args.hash_lock).resolve()) if args.hash_lock else None,
            "expected": expected,
        },
        "script_sha256": file_sha256(Path(__file__).resolve()),
    }
    audit_sha = atomic_write_json(
        Path(args.audit_output), audit, overwrite=args.overwrite
    )
    audit["audit_sha256"] = audit_sha
    return audit


class _SelfTestTokenizer:
    """Tiny tokenizer sufficient to exercise hash and beam logic without vLLM."""

    eos_token_id = 999

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [index + 1 for index, _ in enumerate(text)]}

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        result: list[int] = []
        for token in tokens:
            if token not in self._ids:
                self._ids[token] = 1000 + len(self._ids)
            result.append(self._ids[token])
        return result

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join({1: "<s_a_1>", 2: "<s_b_2>", 3: "<s_c_3>"}.get(item, "?") for item in tokens)


def _self_test() -> None:
    import tempfile

    row = {
        "instruction": "生成商品token",
        "input": "商品描述/no_think",
        "output": "<think>\n\n</think>\n<|prod_begin|><s_a_1><s_b_2><s_c_3>",
        "history": [],
        "route": TRAIN_ROUTE,
        "task": TASK,
    }
    prepared = validate_input_row(row, TRAIN_ROUTE, "self-test")
    expected_canonical = (
        "<|im_start|>user\n生成商品token\n商品描述/no_think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert prepared.canonical_prompt == expected_canonical
    assert prepared.renderer_prompt.endswith("<|prod_begin|>")
    assert prepared.gold_abc == ("1", "2", "3")
    tokenizer = _SelfTestTokenizer()
    metadata = row_token_metadata(prepared, tokenizer)
    expected_hash = hashlib.sha256(
        struct.pack(f"<{len(tokenizer_encode(tokenizer, expected_canonical))}I", *tokenizer_encode(tokenizer, expected_canonical))
    ).hexdigest()
    assert metadata["prompt_token_sha256"] == expected_hash
    assert metadata["gold_tokens"][-1] == tokenizer.eos_token_id

    valid_sequence = SimpleNamespace(
        tokens=[1, 2, 3],
        logprobs=[{}, {}, {}],
        cum_logprob=-0.25,
        text="",
        finish_reason="stop",
        stop_reason=None,
    )
    invalid_sequence = SimpleNamespace(
        tokens=[42],
        logprobs=[{}],
        cum_logprob=-1.0,
        text="x",
        finish_reason="length",
        stop_reason=None,
    )
    beam = SimpleNamespace(sequences=[valid_sequence] + [invalid_sequence] * 63)
    candidates, invalid = parse_beam_sequences(
        beam, prepared.renderer_prompt, prepared.domain, tokenizer
    )
    assert len(candidates) == 1 and candidates[0]["abc"] == ["1", "2", "3"]
    assert len(invalid) == 63 and invalid[0] == 1
    parent_candidate = {
        "candidates": [
            {"abc": ["1", "2", "4"], "cum_logprob": -0.1, "rank": 0, "text": "", "token_count": 3},
        ],
        "invalid_ranks": [],
        "beam_count": 64,
    }
    teacher_candidate = {
        "candidates": [
            {"abc": ["1", "2", "3"], "cum_logprob": -0.2, "rank": 0, "text": "", "token_count": 3},
        ],
        "invalid_ranks": [],
        "beam_count": 64,
    }
    lookup = {("1", "2", "4"): [11, 12, 14]}
    ledger = build_ledger_row(
        prepared,
        metadata,
        {"parent_i19_r96": parent_candidate, "teacher_i23_r64": teacher_candidate},
        "parent_i19_r96",
        "teacher_i23_r64",
        lookup,
    )
    assert ledger["gap_selection"]["selected"] is True
    assert ledger["hard_negatives"][0]["first_divergence"] == 2
    assert ledger["formal_training_generated"] is False

    with tempfile.TemporaryDirectory(prefix="i34-self-test-") as directory:
        directory_path = Path(directory)
        train_path = directory_path / "train.jsonl"
        train_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
        pool = load_pool(train_path, TRAIN_ROUTE, None)
        assert pool.total_rows == 1 and pool.rows[0].row_sha256 == prepared.row_sha256
        output_path = directory_path / "ledger.jsonl"
        output_hash = atomic_write_jsonl(output_path, [ledger])
        assert output_hash == file_sha256(output_path)
        try:
            atomic_write_jsonl(output_path, [ledger])
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic writer allowed an implicit overwrite")
    assert parse_limit("smoke") == SMOKE_LIMIT
    assert parse_limit("all") is None
    print("i34 material beam-gap self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU-only protocol tests")
    parser.add_argument("--preflight", action="store_true", help="validate inputs/artifacts without vLLM")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--parent-adapter", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--teacher-adapter", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH", help="optional repeatable adapter candidate")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV_OUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--hash-lock", type=Path, default=None)
    parser.add_argument("--expected-train-sha256", "--expected-train-input-sha256", dest="expected_train_sha256", default=None)
    parser.add_argument("--expected-dev-sha256", "--expected-dev-input-sha256", dest="expected_dev_sha256", default=None)
    parser.add_argument("--expected-base-artifact-sha256", "--expected-base-sha256", dest="expected_base_artifact_sha256", default=None)
    parser.add_argument("--expected-parent-artifact-sha256", "--expected-parent-sha256", dest="expected_parent_artifact_sha256", default=None)
    parser.add_argument("--expected-teacher-artifact-sha256", "--expected-teacher-sha256", dest="expected_teacher_artifact_sha256", default=None)
    parser.add_argument("--expected-parent-model-sha256", dest="expected_parent_model_sha256", default=None)
    parser.add_argument("--expected-teacher-model-sha256", dest="expected_teacher_model_sha256", default=None)
    parser.add_argument("--limit", default=None, help="all, smoke (2 per pool), or a positive row count")
    parser.add_argument(
        "--allow-mode-rewrite",
        action="store_true",
        help="compatibility mode: append /no_think to legacy rows missing it",
    )
    parser.add_argument(
        "--allow-nonempty-think",
        action="store_true",
        help="compatibility mode: accept legacy non-empty reasoning in gold output",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default=DEFAULT_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-vllm-version", default=EXPECTED_VLLM_VERSION)
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace existing outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if args.preflight:
            parser.error("--self-test and --preflight are mutually exclusive")
        _self_test()
        return 0
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_model_len < 1:
        parser.error("--max-model-len must be >= 1")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("--gpu-memory-utilization must be in (0,1)")
    try:
        if args.preflight:
            print(json.dumps(preflight(args), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            audit = run(args)
            print(json.dumps(audit["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    except (FileNotFoundError, InputError, ValueError, RuntimeError, OSError) as exc:
        print(f"i34 runner failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
