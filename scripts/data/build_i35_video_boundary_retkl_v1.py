#!/usr/bin/env python3
"""Build the I-35 video-boundary residual training set.

The I-35 objective is deliberately narrower than ordinary material SFT.  The
verified r96 submission is the only parent used here.  A row is a ``boundary``
example when the parent's gold SID is in Beam128 ranks 64..127 (zero based);
all other rows are ``preserve`` examples.  Boundary rows carry a bounded
sidecar of wrong parent beams, while preserve rows carry no extra positive or
negative labels.  The formal JSONL itself contains only ordinary
``instruction``/``input``/``output``/``history`` rows so it remains compatible
with the existing single-GPU trainer.  A small, explicitly bounded set of
malformed Beam sequences is tolerated in the ledger; those ranks are excluded
from gold and hard-negative selection.

This script is intentionally fail-closed.  It does not generate anything
until both Beam128 ledgers and the completed runner audit exist, refuses to
overwrite any output, and records every upstream hash in the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "i35-video-boundary-retkl-v1"
POOL_SCHEMA_VERSION = "i35-video-material-beam-pool-v1"
TASK = "material_desc2sid"
MATERIAL_ROUTE = "material_boundary"
RETENTION_ROUTE = "retention_kl"
SEED = 19260835

POOL_TRAIN = ROOT / "logs/data/i35_video_material_beam128_pool_v1.jsonl"
POOL_DEV = ROOT / "logs/data/i35_video_material_beam128_pool_v1_dev.jsonl"
POOL_AUDIT = ROOT / "logs/data/i35_video_material_beam128_pool_v1_audit.json"
BEAM_TRAIN = ROOT / "logs/data/i35_video_material_beam128_train_ledger_v1.jsonl"
BEAM_DEV = ROOT / "logs/data/i35_video_material_beam128_dev_ledger_v1.jsonl"
BEAM_AUDIT = ROOT / "logs/probe/i35_video_material_beam128_audit_v1.json"
RETENTION_SOURCE = ROOT / "assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl"

OUTPUT = ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"
SIDECAR = ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1_sidecar.jsonl"
AUDIT = ROOT / "logs/data/i35_video_boundary_retkl_v1_audit.json"

EXPECTED_POOL_ROWS = 1370
EXPECTED_TRAIN_POOL_ROWS = 1369
EXPECTED_DEV_POOL_ROWS = 1
EXPECTED_RETENTION_SOURCE_ROWS = 2048
EXPECTED_RETENTION_ROWS = 1536
FORMAL_MATERIAL_ROWS = 1370
FORMAL_RETENTION_ROWS = 1370

OFFICIAL_SYSTEM = "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。"
OFFICIAL_USER_PREFIX = "请解析以下视频内容并输出对应的视频token：\n\n"
PARENT_ADAPTER_SHA256 = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
PARENT_CONFIG_SHA256 = "78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f"
BASE_CONFIG_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"

BEAM_WIDTH = 128
# vLLM can occasionally return a malformed sequence in an otherwise complete
# beam. Keep the ledger usable in that case, but cap the tolerated fraction
# and require an exact valid/invalid rank partition below.
MAX_INVALID_CANDIDATES = 16
BOUNDARY_LO = 64
BOUNDARY_HI = 127
NEGATIVE_RANK_LO = 56
NEGATIVE_RANK_HI = 63
MAX_HARD_NEGATIVES = 12
MAX_NEGATIVES_PER_DIVERGENCE = 4

DOMAIN_TOKEN_ID = 176245
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244
EOS_ID = 151645

RETENTION_TASK_ORDER = ("action", "topic", "rec_video", "rec_prod", "rec_ad", "rec_living", "world")
RETENTION_SOURCE_COUNTS = {
    "action": 235,
    "topic": 234,
    "rec_video": 234,
    "rec_prod": 234,
    "rec_ad": 234,
    "rec_living": 234,
    "world": 131,
}

SID_RE = re.compile(r"^<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised when an input or output invariant is not satisfied."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    """Return a logical registry path without dereferencing volume links."""
    value = path if path.is_absolute() else ROOT / path
    try:
        return str(value.relative_to(ROOT))
    except ValueError as exc:
        raise ContractError(f"path is outside the logical repository: {value}") from exc


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_hex(value: Any, name: str) -> str:
    require(isinstance(value, str) and HEX64_RE.fullmatch(value) is not None, f"{name} is not SHA256")
    return value


def finite(value: Any, name: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{name} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{name} is not finite")
    return result


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing input: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"invalid JSON input {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
                value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                require(isinstance(value, dict), f"non-object JSONL row at {path}:{line_number}")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"invalid JSONL input {path}: {exc}") from exc
    return rows


def load_jsonl_flexible(path: Path) -> list[dict[str, Any]]:
    """Load registered E JSONL, whose historical rows may be one-item lists."""
    require(path.is_file(), f"missing input: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
                value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                values = value if isinstance(value, list) else [value]
                require(values and all(isinstance(item, dict) for item in values), f"invalid E row at {path}:{line_number}")
                rows.extend(values)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"invalid flexible JSONL input {path}: {exc}") from exc
    return rows


def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    instruction = row.get("instruction")
    if instruction is None:
        instruction = row.get("system", "")
    user_input = row.get("input")
    if user_input is None:
        user_input = row.get("user")
    if user_input is None:
        user_input = row.get("prompt", "")
    output = row.get("output")
    if output is None:
        output = row.get("response", "")
    history = row.get("history") or []
    return {
        "instruction": str(instruction or ""),
        "input": str(user_input or ""),
        "output": str(output or ""),
        "history": history,
    }


def core_hash(row: Mapping[str, Any]) -> str:
    value = normalized(row)
    require(isinstance(value["history"], list), "history must be a list")
    # The pool builders lock row_sha256 over this four-field object (not a
    # positional list).  Keep the exact representation so ledger joins cannot
    # silently accept a semantically equivalent but differently hashed row.
    return digest({
        "instruction": value["instruction"],
        "input": value["input"],
        "output": value["output"],
        "history": value["history"],
    })


def prompt_hash(row: Mapping[str, Any]) -> str:
    value = normalized(row)
    return digest([value["instruction"], value["input"], value["history"]])


def mode_prompt_hash(row: Mapping[str, Any]) -> str:
    value = normalized(row)
    text = value["input"].rstrip()
    for suffix in ("/no_think", "/think"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            break
    return digest([value["instruction"], text, value["history"]])


def beam_prompt(row: Mapping[str, Any]) -> str:
    """Legacy Beam runner prompt (kept only for ledger provenance checks)."""
    value = normalized(row)
    user = "\n".join(part for part in (value["instruction"], value["input"]) if part)
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def formal_prompt(row: Mapping[str, Any]) -> str:
    """The actual system/user ChatML template consumed by the I-35 trainer."""
    value = normalized(row)
    system = f"<|im_start|>system\n{value['instruction']}<|im_end|>\n" if value["instruction"] else ""
    return system + f"<|im_start|>user\n{value['input']}<|im_end|>\n<|im_start|>assistant\n"


def beam_prompt_hash(row: Mapping[str, Any]) -> str:
    return sha256_bytes(beam_prompt(row).encode("utf-8"))


def formal_prompt_hash(row: Mapping[str, Any]) -> str:
    return sha256_bytes(formal_prompt(row).encode("utf-8"))


def token_hash(token_ids: Sequence[int]) -> str:
    values = [int(value) for value in token_ids]
    require(all(0 <= value <= 0xFFFFFFFF for value in values), "token ID outside uint32")
    return sha256_bytes(struct.pack(f"<{len(values)}I", *values))


def load_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            ROOT / "models/OneReason-0.8B-pretrain-competition",
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as exc:  # pragma: no cover - environment-specific import errors
        raise ContractError(f"cannot load O6 tokenizer for formal prompt hashes: {exc}") from exc


def renderer_prompt(row: Mapping[str, Any]) -> str:
    value = normalized(row)
    return (
        f"<|im_start|>system\n{value['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{value['input']}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n<|video_begin|>"
    )


def renderer_prompt_hash(row: Mapping[str, Any]) -> str:
    return sha256_bytes(renderer_prompt(row).encode("utf-8"))


def parse_video_sid(output: str) -> tuple[str, str, str]:
    require(isinstance(output, str), "material output must be a string")
    parts = output.split("</think>")
    require(len(parts) == 2, "material output must have one </think>")
    require(parts[0].startswith("<think>"), "material output must start with <think>")
    require(parts[0][len("<think>") :].strip() == "", "material think block is not empty")
    match = SID_RE.fullmatch(parts[1].strip())
    require(match is not None, "material output is not a video SID")
    assert match is not None
    return tuple(match.groups())  # type: ignore[return-value]


def gold_tokens(abc: Sequence[str]) -> list[int]:
    require(len(abc) == 3 and all(str(value).isdigit() for value in abc), "invalid gold ABC")
    values = [int(value) for value in abc]
    require(0 <= values[0] <= 8191 and 0 <= values[1] <= 8191 and 0 <= values[2] <= 8191, "ABC outside codebook")
    return [DOMAIN_TOKEN_ID, A_LO + values[0], B_LO + values[1], C_LO + values[2], EOS_ID]


def first_divergence(gold: Sequence[str], candidate: Sequence[str]) -> int:
    for index, (expected, observed) in enumerate(zip(gold, candidate)):
        if expected != observed:
            return index
    raise ContractError("hard negative duplicates gold")


def load_pool_audit(path: Path, train_path: Path, dev_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = load_json(path)
    require(isinstance(audit, dict), "pool audit must be an object")
    require(audit.get("schema_version") == POOL_SCHEMA_VERSION, "I35 pool schema drift")
    require(audit.get("eligible_rows") == EXPECTED_POOL_ROWS, "I35 pool eligible count drift")
    source = audit.get("source")
    require(isinstance(source, dict), "I35 pool source audit missing")
    require(source.get("rows") == 1621, "I35 O1 source count drift")
    require(source.get("asset_id") == "O1.懂物料part4", "I35 source asset drift")
    outputs = audit.get("outputs")
    require(isinstance(outputs, dict), "I35 pool outputs missing")
    expected_paths = {
        "train_pool": train_path,
        "dev_pool": dev_path,
    }
    rows: list[dict[str, Any]] = []
    for key, expected_path in expected_paths.items():
        entry = outputs.get(key)
        require(isinstance(entry, dict), f"pool audit {key} missing")
        observed_path = Path(str(entry.get("path")))
        if not observed_path.is_absolute():
            observed_path = ROOT / observed_path
        require(observed_path.resolve() == expected_path.resolve(), f"pool audit {key} path drift")
        require(expected_path.is_file(), f"missing pool file: {expected_path}")
        actual_hash = sha256_file(expected_path)
        require(entry.get("sha256") == actual_hash, f"pool {key} hash disagrees with audit")
        expected_count = EXPECTED_TRAIN_POOL_ROWS if key == "train_pool" else EXPECTED_DEV_POOL_ROWS
        require(entry.get("rows") == expected_count, f"pool {key} count drift")
        loaded = load_jsonl(expected_path)
        require(len(loaded) == expected_count, f"pool {key} physical count drift")
        rows.extend(loaded)
    require(len(rows) == EXPECTED_POOL_ROWS, "combined I35 pool count drift")
    return audit, rows


def load_e_manifest(pool_audit: Mapping[str, Any]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    manifest = pool_audit.get("e_manifest")
    require(isinstance(manifest, list) and manifest, "I35 E manifest missing")
    exact: set[str] = set()
    modes: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for entry in manifest:
        require(isinstance(entry, dict), "invalid E manifest entry")
        path = ROOT / str(entry.get("path"))
        require(path.resolve() not in seen_paths, "duplicate E manifest path")
        seen_paths.add(path.resolve())
        require(path.is_file(), f"registered E asset is missing: {path}")
        require(sha256_file(path) == entry.get("sha256"), f"E asset hash drift: {path}")
        rows = load_jsonl_flexible(path)
        require(len(rows) == entry.get("rows"), f"E asset row count drift: {path}")
        for row in rows:
            value = normalized(row)
            require(isinstance(value["history"], list), f"E history is not a list: {path}")
            exact.add(prompt_hash(value))
            modes.add(mode_prompt_hash(value))
            all_rows.append(value)
    return exact, modes, all_rows


def validate_pool_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    prompt_seen: set[str] = set()
    mode_seen: set[str] = set()
    sid_seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(rows, 1):
        value = normalized(raw)
        # The frozen I-35 pool predates a per-row schema field; its audit
        # carries the schema/version lock.  Accept an absent field, but reject
        # any explicit value from another pool generation.
        require(raw.get("schema_version") in {None, POOL_SCHEMA_VERSION}, f"pool row {index} schema drift")
        require(raw.get("task") == TASK, f"pool row {index} task drift")
        require(raw.get("route") in {"beam_train_pool", "beam_gate_pool"}, f"pool row {index} route drift")
        require(value["instruction"] == OFFICIAL_SYSTEM, f"pool row {index} system drift")
        require(value["input"].startswith(OFFICIAL_USER_PREFIX) and value["input"].endswith("/no_think"), f"pool row {index} user template drift")
        require(value["history"] == [], f"pool row {index} history drift")
        abc = parse_video_sid(value["output"])
        require(raw.get("gold_domain") == "video", f"pool row {index} gold domain drift")
        require(
            [raw.get("gold_s_a"), raw.get("gold_s_b"), raw.get("gold_s_c")] == [int(item) for item in abc],
            f"pool row {index} gold ABC drift",
        )
        require(raw.get("gold_sid") == value["output"].split("</think>", 1)[1].strip(), f"pool row {index} gold SID drift")
        row_hash = core_hash(value)
        require(raw.get("row_sha256") == row_hash, f"pool row {index} core hash drift")
        require(raw.get("prompt_sha256") == prompt_hash(value), f"pool row {index} prompt hash drift")
        require(raw.get("mode_prompt_sha256") == mode_prompt_hash(value), f"pool row {index} mode hash drift")
        require(row_hash not in by_hash, f"duplicate pool row {index}")
        require(prompt_hash(value) not in prompt_seen and mode_prompt_hash(value) not in mode_seen, f"duplicate pool prompt {index}")
        require(tuple(abc) not in sid_seen, f"duplicate pool SID {index}")
        prompt_seen.add(prompt_hash(value))
        mode_seen.add(mode_prompt_hash(value))
        sid_seen.add(tuple(abc))
        by_hash[row_hash] = dict(raw)
    require(len(by_hash) == EXPECTED_POOL_ROWS, "pool unique row count drift")
    return by_hash


def validate_beam_audit(path: Path, train_path: Path, dev_path: Path) -> dict[str, Any]:
    audit = load_json(path)
    require(isinstance(audit, dict), "Beam audit must be an object")
    require(audit.get("status") in {"complete", "finished"}, "Beam audit is not complete")
    require(audit.get("formal_training_generated") in {False, None}, "Beam runner generated formal data")
    runtime = audit.get("runtime")
    if isinstance(runtime, dict):
        require(runtime.get("beam_width") == BEAM_WIDTH, "Beam width is not 128")
    found_parent = False
    artifacts = audit.get("artifacts")
    if isinstance(artifacts, dict):
        adapters = artifacts.get("adapters")
        if isinstance(adapters, list):
            for item in adapters:
                if isinstance(item, dict) and item.get("adapter_model_sha256") == PARENT_ADAPTER_SHA256:
                    found_parent = True
                    if item.get("adapter_config_sha256") is not None:
                        require(item.get("adapter_config_sha256") == PARENT_CONFIG_SHA256, "parent config hash drift")
    adapters = audit.get("adapters")
    if isinstance(adapters, list):
        for item in adapters:
            if isinstance(item, dict) and item.get("adapter_model_sha256") == PARENT_ADAPTER_SHA256:
                found_parent = True
    if isinstance(adapters, dict) and adapters.get("parent"):
        found_parent = found_parent or adapters.get("parent_adapter_sha256") == PARENT_ADAPTER_SHA256
    require(found_parent, "Beam audit does not identify the verified r96 parent")
    outputs = audit.get("outputs")
    if isinstance(outputs, dict):
        for key, path_value, expected_rows in (("train_ledger", train_path, EXPECTED_TRAIN_POOL_ROWS), ("dev_ledger", dev_path, EXPECTED_DEV_POOL_ROWS)):
            entry = outputs.get(key)
            if isinstance(entry, dict):
                observed_path = Path(str(entry.get("path")))
                if not observed_path.is_absolute():
                    observed_path = ROOT / observed_path
                require(observed_path.resolve() == path_value.resolve(), f"Beam audit {key} path drift")
                require(entry.get("rows") == expected_rows, f"Beam audit {key} count drift")
                require(entry.get("sha256") == sha256_file(path_value), f"Beam audit {key} hash drift")
    return audit


def candidate_abc(candidate: Mapping[str, Any], location: str) -> tuple[str, str, str]:
    abc = candidate.get("abc")
    require(isinstance(abc, list) and len(abc) == 3 and all(isinstance(value, (str, int)) and str(value).isdigit() for value in abc), f"{location}.abc invalid")
    values = tuple(str(value) for value in abc)
    require(all(0 <= int(value) <= 8191 for value in values), f"{location}.abc outside codebook")
    return values  # type: ignore[return-value]


def validate_parent_candidates(
    block: Mapping[str, Any], gold: tuple[str, str, str], location: str
) -> tuple[list[dict[str, Any]], int | None, list[int]]:
    require(block.get("beam_count") == BEAM_WIDTH, f"{location}.beam_count is not 128")

    invalid_count = block.get("invalid_count")
    require(
        isinstance(invalid_count, int) and not isinstance(invalid_count, bool),
        f"{location}.invalid_count is invalid",
    )
    require(
        0 <= invalid_count <= MAX_INVALID_CANDIDATES,
        f"{location} has too many invalid candidates: {invalid_count}/{MAX_INVALID_CANDIDATES}",
    )
    invalid_ranks_raw = block.get("invalid_ranks")
    require(isinstance(invalid_ranks_raw, list), f"{location}.invalid_ranks is invalid")
    require(
        len(invalid_ranks_raw) == invalid_count,
        f"{location}.invalid_count disagrees with invalid_ranks",
    )
    invalid_ranks: list[int] = []
    invalid_rank_set: set[int] = set()
    for index, rank in enumerate(invalid_ranks_raw):
        require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and 0 <= rank < BEAM_WIDTH,
            f"{location}.invalid_ranks[{index}] is invalid",
        )
        require(rank not in invalid_rank_set, f"{location} has duplicate invalid rank {rank}")
        invalid_rank_set.add(rank)
        invalid_ranks.append(rank)

    candidates = block.get("valid_candidates")
    require(
        isinstance(candidates, list) and len(candidates) == BEAM_WIDTH - invalid_count,
        f"{location} candidate count drift",
    )
    seen_ranks: set[int] = set()
    seen_abc: set[tuple[str, str, str]] = set()
    clean: list[dict[str, Any]] = []
    gold_ranks: list[int] = []
    for index, candidate in enumerate(candidates):
        require(isinstance(candidate, dict), f"{location}[{index}] is not an object")
        abc = candidate_abc(candidate, f"{location}[{index}]")
        rank = candidate.get("rank")
        require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and 0 <= rank < BEAM_WIDTH,
            f"{location}[{index}].rank invalid",
        )
        require(rank not in seen_ranks and abc not in seen_abc, f"duplicate parent candidate at {location}[{index}]")
        seen_ranks.add(rank)
        seen_abc.add(abc)
        score = candidate.get("cum_logprob")
        if score is not None:
            finite(score, f"{location}[{index}].cum_logprob")
        if abc == gold:
            gold_ranks.append(rank)
        clean.append(dict(candidate))

    require(seen_ranks.isdisjoint(invalid_rank_set), f"{location} valid/invalid ranks overlap")
    require(
        seen_ranks | invalid_rank_set == set(range(BEAM_WIDTH)),
        f"{location} valid/invalid ranks do not cover 0..127",
    )
    require(len(gold_ranks) <= 1, f"{location} contains duplicate gold")
    hit = bool(gold_ranks)
    require(block.get("full_gold_hit") is hit, f"{location}.full_gold_hit disagrees")
    return clean, (gold_ranks[0] if hit else None), sorted(invalid_ranks)


def shared_prefix_length(gold: Sequence[str], candidate: Sequence[str]) -> int:
    count = 0
    for expected, observed in zip(gold, candidate):
        if expected != observed:
            break
        count += 1
    return count


def build_negatives(gold: tuple[str, str, str], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[tuple[dict[str, Any], tuple[str, str, str], int, int]] = []
    for candidate in candidates:
        abc = candidate_abc(candidate, "parent candidate")
        if abc == gold:
            continue
        rank = int(candidate["rank"])
        score = candidate.get("cum_logprob")
        if score is None or not math.isfinite(float(score)):
            continue
        parsed.append((dict(candidate), abc, rank, shared_prefix_length(gold, abc)))
    rank_window = [item for item in parsed if NEGATIVE_RANK_LO <= item[2] <= NEGATIVE_RANK_HI]
    shared = [item for item in parsed if item[2] >= BOUNDARY_LO and item[3] > 0 and item not in rank_window]
    # Once the explicit 56..63 priority and shared-prefix candidates are
    # exhausted, only remaining near-cutoff candidates (64..127) are useful;
    # pulling ranks far below the cutoff would teach the wrong boundary.
    fallback = [
        item
        for item in parsed
        if item[2] >= BOUNDARY_LO and item not in rank_window and item not in shared
    ]
    rank_window.sort(key=lambda item: item[2])
    shared.sort(key=lambda item: (-item[3], abs(item[2] - BOUNDARY_LO), item[2]))
    fallback.sort(key=lambda item: (abs(item[2] - BOUNDARY_LO), item[2]))
    selected: list[dict[str, Any]] = []
    divergence_counts: Counter[int] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for source, group in (("parent_rank_56_63", rank_window), ("shared_gold_prefix_near_cutoff", shared), ("parent_near_cutoff_fallback", fallback)):
        for candidate, abc, rank, _prefix in group:
            if len(selected) >= MAX_HARD_NEGATIVES:
                break
            if abc in seen:
                continue
            divergence = first_divergence(gold, abc)
            if divergence_counts[divergence] >= MAX_NEGATIVES_PER_DIVERGENCE:
                continue
            seen.add(abc)
            divergence_counts[divergence] += 1
            values = [int(abc[0]), int(abc[1]), int(abc[2])]
            negative = {
                "tokens": [A_LO + values[0], B_LO + values[1], C_LO + values[2]],
                "abc": list(abc),
                "first_divergence": divergence,
                "parent_beam_rank": rank,
                "parent_score": float(candidate["cum_logprob"]),
                "selection_source": source,
            }
            selected.append(negative)
        if len(selected) >= MAX_HARD_NEGATIVES:
            break
    return selected


def validate_ledger_rows(
    path: Path,
    expected_route: str,
    pool_by_hash: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(path)
    expected_count = EXPECTED_TRAIN_POOL_ROWS if expected_route == "beam_train_pool" else EXPECTED_DEV_POOL_ROWS
    require(len(rows) == expected_count, f"{path} row count drift: {len(rows)}/{expected_count}")
    observed: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        location = f"{path.name}:{index}"
        row_hash = require_hex(row.get("row_sha256"), f"{location}.row_sha256")
        require(row_hash in pool_by_hash, f"{location} is not in I35 pool")
        require(row_hash not in observed, f"duplicate ledger row {location}")
        require(row.get("route") == expected_route, f"{location}.route drift")
        pool = pool_by_hash[row_hash]
        require(row.get("source_prompt_sha256") in {None, pool.get("prompt_sha256")}, f"{location}.source_prompt_sha256 drift")
        require(row.get("prompt_sha256") == beam_prompt_hash(pool), f"{location}.prompt_sha256 does not match Beam template")
        require_hex(row.get("prompt_token_sha256"), f"{location}.prompt_token_sha256")
        require(isinstance(row.get("prompt_token_count"), int) and row["prompt_token_count"] > 0, f"{location}.prompt_token_count invalid")
        require(row.get("renderer_prompt_sha256") in {None, renderer_prompt_hash(pool)}, f"{location}.renderer_prompt_sha256 drift")
        require(row.get("parent_adapter_sha256") == PARENT_ADAPTER_SHA256, f"{location}.parent_adapter_sha256 drift")
        gold = tuple(str(value) for value in (pool.get("gold_s_a"), pool.get("gold_s_b"), pool.get("gold_s_c")))
        require(row.get("gold_abc") == list(gold), f"{location}.gold_abc drift")
        expected_tokens = gold_tokens(gold)
        require(row.get("gold_tokens") == expected_tokens, f"{location}.gold_tokens drift")
        require(row.get("positive_tokens") == [expected_tokens[1:4]], f"{location}.positive_tokens drift")
        parent = row.get("parent")
        require(isinstance(parent, dict), f"{location}.parent missing")
        candidates, gold_rank, invalid_ranks = validate_parent_candidates(
            parent, gold, f"{location}.parent.valid_candidates"
        )
        observed[row_hash] = {
            "ledger": row,
            "pool": pool,
            "parent_candidates": candidates,
            "parent_gold_rank": gold_rank,
            "parent_invalid_ranks": invalid_ranks,
        }
        objective = "boundary" if gold_rank is not None and BOUNDARY_LO <= gold_rank <= BOUNDARY_HI else "preserve"
        counts[objective] += 1
        counts["invalid_candidates"] += len(invalid_ranks)
        if invalid_ranks:
            counts["rows_with_invalid"] += 1
        if objective == "boundary":
            negatives = build_negatives(gold, candidates)
            require(negatives, f"{location} boundary row has no finite parent negative")
            counts["hard_negatives"] += len(negatives)
            counts[f"divergence_{sum(1 for _ in negatives)}"] += 0
    require(set(observed) == set(pool_by_hash), f"{path.name} does not cover its pool exactly")
    return observed, {
        "rows": len(rows),
        "boundary": counts["boundary"],
        "preserve": counts["preserve"],
        "hard_negatives": counts["hard_negatives"],
        "invalid_candidates": counts["invalid_candidates"],
        "rows_with_invalid": counts["rows_with_invalid"],
    }


def retention_quotas(source_counts: Mapping[str, int]) -> dict[str, int]:
    require(set(source_counts) == set(RETENTION_TASK_ORDER), "retention task set drift")
    require(source_counts["world"] == RETENTION_SOURCE_COUNTS["world"], "world source count drift")
    target_nonworld = FORMAL_RETENTION_ROWS - source_counts["world"]
    nonworld_total = sum(source_counts[task] for task in RETENTION_TASK_ORDER if task != "world")
    raw = {task: target_nonworld * source_counts[task] / nonworld_total for task in RETENTION_TASK_ORDER if task != "world"}
    result = {task: int(math.floor(value)) for task, value in raw.items()}
    result["world"] = source_counts["world"]
    remaining = target_nonworld - sum(result[task] for task in result if task != "world")
    ranked = sorted(
        raw,
        key=lambda task: (-(raw[task] - math.floor(raw[task])), digest([SEED, "retention-quota", task])),
    )
    for task in ranked[:remaining]:
        result[task] += 1
    require(sum(result.values()) == FORMAL_RETENTION_ROWS, "retention quota sum drift")
    for task, count in result.items():
        require(0 < count <= source_counts[task], f"retention quota invalid for {task}")
    return {task: result[task] for task in RETENTION_TASK_ORDER}


def select_retention(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [row for row in rows if row.get("route") == RETENTION_ROUTE]
    require(len(source) == EXPECTED_RETENTION_ROWS, "I33 retention route count drift")
    by_task: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(source):
        task = row.get("task")
        require(task in RETENTION_TASK_ORDER, f"unknown I33 retention task: {task}")
        value = normalized(row)
        require(isinstance(value["history"], list), "retention history is not a list")
        by_task[str(task)].append((index, dict(row)))
    source_counts = {task: len(by_task[task]) for task in RETENTION_TASK_ORDER}
    require(source_counts == RETENTION_SOURCE_COUNTS, f"I33 retention task counts drift: {source_counts}")
    quotas = retention_quotas(source_counts)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for task in RETENTION_TASK_ORDER:
        ranked = sorted(
            by_task[task],
            key=lambda item: (digest([SEED, "retention-row", task, core_hash(item[1]), item[0]]), item[0]),
        )
        chosen = ranked if task == "world" else ranked[: quotas[task]]
        require(len(chosen) == quotas[task], f"retention selection count drift for {task}")
        for _index, original in chosen:
            value = normalized(original)
            transformed = {
                "instruction": "",
                "input": value["instruction"] + "\n" + value["input"],
                "output": value["output"],
                "history": value["history"],
                "route": RETENTION_ROUTE,
                "task": task,
            }
            # I33 intentionally contains a bounded second exposure for some
            # world rows.  Keep those rows (the world quota is all 131) and
            # use the frozen source line as the identity, rather than
            # incorrectly deduplicating them by core hash.
            key = digest([task, core_hash(original), _index])
            require(key not in selected_keys, "duplicate selected retention source")
            selected_keys.add(key)
            selected.append(transformed)
    require(len(selected) == FORMAL_RETENTION_ROWS, "formal retention count drift")
    selected_counts = dict(Counter(row["task"] for row in selected))
    require(selected_counts == quotas, "selected retention task counts drift")
    return selected, {
        "source_rows": len(source),
        "source_counts": source_counts,
        "quotas": quotas,
        "selected_rows": len(selected),
        "selected_counts": selected_counts,
        "world_rows_all_retained": True,
        "selection": "stable SHA256 sort within task; proportional largest-remainder quotas; world is fully retained",
    }


def make_material_row(pool: Mapping[str, Any]) -> dict[str, Any]:
    value = normalized(pool)
    abc = parse_video_sid(value["output"])
    description = value["input"]
    require(description.startswith(OFFICIAL_USER_PREFIX) and description.endswith("/no_think"), "pool user template drift")
    output = f"<think>\n\n</think>\n<|video_begin|><s_a_{abc[0]}><s_b_{abc[1]}><s_c_{abc[2]}>"
    row = {
        "instruction": OFFICIAL_SYSTEM,
        "input": description,
        "output": output,
        "history": [],
        "route": MATERIAL_ROUTE,
        "task": TASK,
    }
    require(row["input"] == OFFICIAL_USER_PREFIX + row["input"][len(OFFICIAL_USER_PREFIX) :], "material user prefix drift")
    require(parse_video_sid(row["output"]) == abc, "material gold changed during formalization")
    return row


def make_sidecar(
    pool: Mapping[str, Any],
    ledger: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    gold_rank: int | None,
    objective: str,
    formal_token_sha256: str,
    formal_token_count: int,
) -> dict[str, Any]:
    abc = tuple(str(value) for value in (pool["gold_s_a"], pool["gold_s_b"], pool["gold_s_c"]))
    tokens = gold_tokens(abc)
    negatives = build_negatives(abc, candidates) if objective == "boundary" else []
    if objective == "boundary":
        require(gold_rank is not None and BOUNDARY_LO <= gold_rank <= BOUNDARY_HI, "boundary rank drift")
        require(negatives, "boundary sidecar has no negatives")
    require(objective in {"boundary", "preserve"}, "unknown sidecar objective")
    row = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "objective": objective,
        "row_sha256": core_hash(pool),
        "source_prompt_sha256": pool.get("prompt_sha256"),
        "source_mode_prompt_sha256": pool.get("mode_prompt_sha256"),
        # prompt_sha256 is the actual system/user template hash.  The Beam
        # runner's legacy hash is retained separately for provenance.
        "prompt_sha256": formal_prompt_hash(pool),
        "beam_prompt_sha256": beam_prompt_hash(pool),
        "prompt_template_sha256": formal_prompt_hash(make_material_row(pool)),
        "prompt_token_sha256": formal_token_sha256,
        "prompt_token_count": formal_token_count,
        "beam_prompt_token_sha256": ledger["prompt_token_sha256"],
        "renderer_prompt_sha256": renderer_prompt_hash(pool),
        "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "domain": "video",
        "parent_gold_rank_1based": None if gold_rank is None else gold_rank + 1,
        "parent_beam_rank": gold_rank,
        "gold_abc": [tokens[1], tokens[2], tokens[3]],
        "gold_tokens": tokens,
        "positive_tokens": [tokens[1:4]],
        "hard_negatives": negatives,
    }
    return row


def intersections(material: Sequence[Mapping[str, Any]], retention: Sequence[Mapping[str, Any]], e_exact: set[str], e_modes: set[str]) -> dict[str, int]:
    material_exact = {prompt_hash(row) for row in material}
    retention_exact = {prompt_hash(row) for row in retention}
    material_modes = {mode_prompt_hash(row) for row in material}
    retention_modes = {mode_prompt_hash(row) for row in retention}
    return {
        "material_vs_retention_prompt": len(material_exact & retention_exact),
        "material_vs_retention_mode_prompt": len(material_modes & retention_modes),
        "material_vs_E_prompt": len(material_exact & e_exact),
        "material_vs_E_mode_prompt": len(material_modes & e_modes),
        "retention_vs_E_prompt": len(retention_exact & e_exact),
        "retention_vs_E_mode_prompt": len(retention_modes & e_modes),
    }


def encoded_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical(row) + "\n").encode("utf-8") for row in rows)


def encoded_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


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
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            temporary_path = Path(temporary_name)
            temporary.append((temporary_path, path))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary_path, path in temporary:
            os.link(temporary_path, path)
            created.append(path)
        for temporary_path, _path in temporary:
            temporary_path.unlink(missing_ok=True)
    except BaseException:
        for temporary_path, _path in temporary:
            temporary_path.unlink(missing_ok=True)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = (Path(args.output), Path(args.sidecar_output), Path(args.audit_output))
    require(len({path.resolve() for path in output_paths}) == 3, "outputs must be distinct")
    require(all(not path.exists() for path in output_paths), "formal builder refuses to overwrite outputs")

    pool_audit, pool_rows = load_pool_audit(Path(args.pool_audit), Path(args.train_pool), Path(args.dev_pool))
    pool_by_hash = validate_pool_rows(pool_rows)
    e_exact, e_modes, e_rows = load_e_manifest(pool_audit)
    require(len(e_rows) > 0, "E manifest is empty")
    validate_beam_audit(Path(args.beam_audit), Path(args.train_ledger), Path(args.dev_ledger))
    train_ledger, train_counts = validate_ledger_rows(Path(args.train_ledger), "beam_train_pool", {key: value for key, value in pool_by_hash.items() if value.get("route") == "beam_train_pool"})
    dev_ledger, dev_counts = validate_ledger_rows(Path(args.dev_ledger), "beam_gate_pool", {key: value for key, value in pool_by_hash.items() if value.get("route") == "beam_gate_pool"})
    all_ledger = {**train_ledger, **dev_ledger}
    require(set(all_ledger) == set(pool_by_hash), "Beam ledgers do not cover all 1370 pool rows")

    # The Beam ledger's prompt hash was produced by its historical
    # user-only helper.  Re-tokenize the *formal* system/user ChatML template
    # with O6 here; this is the hash the trainer will use at runtime.
    tokenizer = load_tokenizer()
    material_rows: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    material_objectives: Counter[str] = Counter()
    for row_hash in sorted(pool_by_hash, key=lambda value: digest([SEED, "material-order", value])):
        info = all_ledger[row_hash]
        pool = info["pool"]
        gold_rank = info["parent_gold_rank"]
        objective = "boundary" if gold_rank is not None and BOUNDARY_LO <= gold_rank <= BOUNDARY_HI else "preserve"
        material = make_material_row(pool)
        encoded_prompt = tokenizer.encode(formal_prompt(material), add_special_tokens=False)
        require(encoded_prompt, "formal prompt tokenization returned no tokens")
        actual_token_hash = token_hash(encoded_prompt)
        # Keep a deterministic text-level assertion alongside the tokenizer
        # hash so a renderer change cannot be hidden by a coincident hash.
        require(formal_prompt_hash(material) == formal_prompt_hash(pool), "formal prompt text changed")
        material_rows.append(material)
        sidecar_rows.append(
            make_sidecar(
                pool,
                info["ledger"],
                info["parent_candidates"],
                gold_rank,
                objective,
                actual_token_hash,
                len(encoded_prompt),
            )
        )
        material_objectives[objective] += 1
    require(len(material_rows) == FORMAL_MATERIAL_ROWS, "formal material count drift")
    require(len(sidecar_rows) == FORMAL_MATERIAL_ROWS, "sidecar count drift")
    require(material_objectives["boundary"] > 0 and material_objectives["preserve"] > 0, "I35 requires both boundary and preserve material rows")

    retention_source = Path(args.retention_source)
    require(sha256_file(retention_source) == args.expected_retention_sha256, "I33 upstream hash drift")
    retention_source_rows = load_jsonl(retention_source)
    require(len(retention_source_rows) == EXPECTED_RETENTION_SOURCE_ROWS, "I33 upstream row count drift")
    retention_rows, retention_audit = select_retention(retention_source_rows)
    cross = intersections(material_rows, retention_rows, e_exact, e_modes)
    require(all(value == 0 for value in cross.values()), f"prompt intersection is nonzero: {cross}")

    material_hashes = {core_hash(row) for row in material_rows}
    require(len(material_hashes) == FORMAL_MATERIAL_ROWS, "duplicate formal material rows")
    retention_nonworld_hashes = {
        core_hash(row) for row in retention_rows if row.get("task") != "world"
    }
    retention_nonworld_rows = sum(1 for row in retention_rows if row.get("task") != "world")
    require(len(retention_nonworld_hashes) == retention_nonworld_rows, "duplicate non-world retention rows")
    sidecar_keys = {row["row_sha256"] for row in sidecar_rows}
    require(sidecar_keys == {core_hash(row) for row in material_rows}, "sidecar/data key mismatch")
    require(len({row["prompt_token_sha256"] for row in sidecar_rows}) == FORMAL_MATERIAL_ROWS, "duplicate material token hashes")

    tagged = [("material_boundary", core_hash(row), row) for row in material_rows]
    tagged.extend(("retention_kl", core_hash(row), row) for row in retention_rows)
    tagged.sort(key=lambda item: digest([SEED, "formal-shuffle", item[0], item[1]]))
    training_rows = [item[2] for item in tagged]
    data_payload = encoded_jsonl(training_rows)
    sidecar_rows.sort(key=lambda row: row["prompt_token_sha256"])
    sidecar_payload = encoded_jsonl(sidecar_rows)

    task_counts = dict(sorted(Counter(row["task"] for row in retention_rows).items()))
    runtime_steps = math.ceil((FORMAL_MATERIAL_ROWS + FORMAL_RETENTION_ROWS) / 4)
    builder_path = Path(__file__).resolve()
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "formal_built",
        "asset_class": "D(O1; M-I35 Beam128 boundary selection) + D(I33 retention subset)",
        "builder": {"path": str(builder_path.relative_to(ROOT)), "sha256": sha256_file(builder_path)},
        "field_map": {"system": "instruction", "user": "input", "template_note": "trainer consumes instruction/input; semantic system/user values are recorded explicitly"},
        "upstreams": {
            "material_pool": {
                "asset_id": "i35_video_material_beam128_pool_v1",
                "class": "D(O1)",
                "train_path": repo_relative(Path(args.train_pool)),
                "train_rows": EXPECTED_TRAIN_POOL_ROWS,
                "train_sha256": sha256_file(Path(args.train_pool)),
                "dev_path": repo_relative(Path(args.dev_pool)),
                "dev_rows": EXPECTED_DEV_POOL_ROWS,
                "dev_sha256": sha256_file(Path(args.dev_pool)),
                "pool_audit_path": repo_relative(Path(args.pool_audit)),
                "pool_audit_sha256": sha256_file(Path(args.pool_audit)),
                "formal_rows": FORMAL_MATERIAL_ROWS,
            },
            "beam128": {
                "train_path": repo_relative(Path(args.train_ledger)),
                "train_rows": EXPECTED_TRAIN_POOL_ROWS,
                "train_sha256": sha256_file(Path(args.train_ledger)),
                "dev_path": repo_relative(Path(args.dev_ledger)),
                "dev_rows": EXPECTED_DEV_POOL_ROWS,
                "dev_sha256": sha256_file(Path(args.dev_ledger)),
                "audit_path": repo_relative(Path(args.beam_audit)),
                "audit_sha256": sha256_file(Path(args.beam_audit)),
                "beam_width": BEAM_WIDTH,
                "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
                "parent_config_sha256": PARENT_CONFIG_SHA256,
                "teacher_or_other_adapters_used_as_positive": 0,
            },
            "retention": {
                "asset_id": "data_i33_r96_material_desc2sid_retkl_v1",
                "class": "D(O1,O2.*; I33 frozen retention)",
                "path": repo_relative(retention_source),
                "rows": len(retention_source_rows),
                "sha256": sha256_file(retention_source),
                "route": RETENTION_ROUTE,
                "source_route_rows": EXPECTED_RETENTION_ROWS,
                "formal_rows": FORMAL_RETENTION_ROWS,
            },
            "e_manifest": {
                "rows": len(e_rows),
                "assets": pool_audit.get("e_manifest"),
                "copied_to_training": 0,
            },
        },
        "parent": {"adapter_sha256": PARENT_ADAPTER_SHA256, "config_sha256": PARENT_CONFIG_SHA256, "base_config_sha256": BASE_CONFIG_SHA256, "only_parent_used": True},
        "material_selection": {
            "rows": FORMAL_MATERIAL_ROWS,
            "by_objective": dict(sorted(material_objectives.items())),
            "boundary_rank_zero_based": [BOUNDARY_LO, BOUNDARY_HI],
            "preserve_definition": "parent gold rank <64 or absent from Beam128",
            "negative_priority": ["parent ranks 56..63", "shared gold prefix near cutoff", "near-cutoff fallback"],
            "max_hard_negatives": MAX_HARD_NEGATIVES,
            "max_negatives_per_first_divergence": MAX_NEGATIVES_PER_DIVERGENCE,
            "invalid_candidate_policy": {
                "max_per_beam": MAX_INVALID_CANDIDATES,
                "valid_invalid_ranks_must_partition_beam": True,
                "invalid_candidates_are_not_hard_negatives": True,
            },
            "invalid_candidates_total": train_counts["invalid_candidates"]
            + dev_counts["invalid_candidates"],
            "rows_with_invalid_total": train_counts["rows_with_invalid"]
            + dev_counts["rows_with_invalid"],
            "train_ledger_counts": train_counts,
            "dev_ledger_counts": dev_counts,
        },
        "mix": {
            "total_rows": len(training_rows),
            "material": {"rows": FORMAL_MATERIAL_ROWS, "ratio": 0.5, "by_objective": dict(sorted(material_objectives.items()))},
            "retention": {"rows": FORMAL_RETENTION_ROWS, "ratio": 0.5, "by_task": task_counts, **retention_audit},
            "fixed_seed_hash_shuffle": True,
        },
        "seed": SEED,
        "runtime": {"batch_size": 1, "gradient_accumulation": 4, "expected_optimizer_steps": runtime_steps, "single_gpu": True, "wandb_required": True, "tokenizer": "O6 local AutoTokenizer", "formal_prompt_template": "explicit system ChatML + user ChatML"},
        "sidecar_contract": {
            "rows": len(sidecar_rows),
            "objectives": dict(sorted(material_objectives.items())),
            "retention_rows_in_sidecar": 0,
            "positive_definition": "one official O1 video SID",
            "hard_negative_source": "verified r96 parent Beam128 only",
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "token_hash_template_checked": True,
        },
        "intersections": cross,
        "forbidden_sources": {"third_party_rows": 0, "E_rows": 0, "registered_E_rows_copied": 0, "dev_pool_rows_copied": EXPECTED_DEV_POOL_ROWS, "teacher_positive_rows": 0},
        "outputs": {
            "training_data": {"path": repo_relative(Path(args.output)), "rows": len(training_rows), "bytes": len(data_payload), "sha256": sha256_bytes(data_payload)},
            "sidecar": {"path": repo_relative(Path(args.sidecar_output)), "rows": len(sidecar_rows), "bytes": len(sidecar_payload), "sha256": sha256_bytes(sidecar_payload)},
        },
    }
    audit_payload = encoded_json(audit)
    write_new_files(((Path(args.output), data_payload), (Path(args.sidecar_output), sidecar_payload), (Path(args.audit_output), audit_payload)))
    return audit


def self_test() -> None:
    assert retention_quotas(RETENTION_SOURCE_COUNTS)["world"] == 131
    assert sum(retention_quotas(RETENTION_SOURCE_COUNTS).values()) == 1370
    gold = ("1", "2", "3")
    candidates = []
    for rank in range(128):
        abc = ("9", str(rank), "8")
        if rank == 64:
            abc = gold
        candidates.append({"abc": list(abc), "rank": rank, "cum_logprob": -float(rank)})
    negatives = build_negatives(gold, candidates)
    assert negatives and all(item["parent_beam_rank"] != 64 for item in negatives)
    assert max(Counter(item["first_divergence"] for item in negatives).values()) <= 4
    valid_candidates, hit_rank, invalid_ranks = validate_parent_candidates(
        {
            "beam_count": BEAM_WIDTH,
            "invalid_count": 1,
            "invalid_ranks": [127],
            "valid_candidates": candidates[:-1],
            "full_gold_hit": True,
        },
        gold,
        "self_test.parent",
    )
    assert len(valid_candidates) == 127 and hit_rank == 64 and invalid_ranks == [127]
    pool = {"instruction": OFFICIAL_SYSTEM, "input": OFFICIAL_USER_PREFIX + "描述/no_think", "output": "<think>\n\n</think>\n<|video_begin|><s_a_1><s_b_2><s_c_3>", "history": []}
    material = make_material_row(pool)
    assert material["instruction"] == OFFICIAL_SYSTEM
    assert material["input"].startswith(OFFICIAL_USER_PREFIX)
    assert parse_video_sid(material["output"]) == ("1", "2", "3")
    assert beam_prompt_hash(pool) == beam_prompt_hash(material)
    assert formal_prompt_hash(pool) == formal_prompt_hash(material)
    print("i35 video boundary formal builder self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train-pool", type=Path, default=POOL_TRAIN)
    parser.add_argument("--dev-pool", type=Path, default=POOL_DEV)
    parser.add_argument("--pool-audit", type=Path, default=POOL_AUDIT)
    parser.add_argument("--train-ledger", type=Path, default=BEAM_TRAIN)
    parser.add_argument("--dev-ledger", type=Path, default=BEAM_DEV)
    parser.add_argument("--beam-audit", "--runner-audit", dest="beam_audit", type=Path, default=BEAM_AUDIT)
    parser.add_argument("--retention-source", type=Path, default=RETENTION_SOURCE)
    parser.add_argument("--expected-retention-sha256", default="7d6a1e4a44238a79dcb0d31384f147c02baea95cd870224e2a6815444f8470fd")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sidecar-output", type=Path, default=SIDECAR)
    parser.add_argument("--audit-output", type=Path, default=AUDIT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
