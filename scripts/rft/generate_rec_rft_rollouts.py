#!/usr/bin/env python3
"""Generate prompt-only, two-stage recommendation RFT rollouts.

The input is a strict prompt manifest.  It deliberately contains no gold item,
target, output, or source-row reference.  For every selected prompt this script
samples N reasoning traces, then continues each trace with the prompt's domain
token and performs a K-wide, three-token beam search for an itemic candidate.

This is a single-GPU platform job.  All decoding choices are explicit command
line arguments, while the input/model/adapter identities and every relevant
argument are hash-locked in ``<out>.config.json``.  Completed groups are written
and fsynced one batch at a time and can be resumed only with an identical lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_VOLUME = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51")
MANIFEST_SCHEMA = "o1-rec-prompt-manifest-v1"
ROLLOUT_SCHEMA = "o1-rec-rollouts-v1"
CONFIG_SCHEMA = "o1-rec-rollout-config-v1"
PROTOCOL = "sample-reasoning-then-3token-item-beam-v1"

MANIFEST_KEYS = {
    "schema_version",
    "group_id",
    "instruction",
    "input",
    "history",
    "domain",
    "prompt_sha256",
    "rollout_seed",
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
TRACE_KEYS = {
    "trace_id",
    "reasoning_index",
    "thought",
    "reasoning",
    "candidates",
}
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

DOMAIN_TOKENS = {
    "video": "<|video_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "living": "<|living_begin|>",
}
DOMAIN_ALIASES = {"live": "living", **{name: name for name in DOMAIN_TOKENS}}
DOMAIN_ORDER = tuple(DOMAIN_TOKENS)
HEX64 = re.compile(r"[0-9a-f]{64}")
SID_SUFFIX = re.compile(r"<s_a_\d+><s_b_\d+><s_c_\d+>")
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)

# Only files that can affect local Hugging Face/vLLM inference are included.
# This intentionally excludes README files, images, optimizer state, and logs.
MODEL_RUNTIME_SUFFIXES = {
    ".bin",
    ".json",
    ".jinja",
    ".model",
    ".safetensors",
    ".tiktoken",
    ".txt",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two-stage recommendation RFT rollouts from a prompt-only manifest."
    )
    parser.add_argument("--base", required=True, help="Base Hugging Face model directory.")
    parser.add_argument("--adapter", required=True, help="PEFT LoRA adapter directory.")
    parser.add_argument("--manifest", required=True, help="Prompt-only JSONL manifest.")
    parser.add_argument("--out", required=True, help="Output rollout JSONL.")
    parser.add_argument(
        "--domains",
        required=True,
        help="Comma-separated subset of video,prod,ad,living (live aliases living).",
    )
    parser.add_argument("--gpu", required=True, help="One CUDA device index or UUID.")
    parser.add_argument("--reasoning-samples", required=True, type=int)
    parser.add_argument("--item-candidates", required=True, type=int)
    parser.add_argument("--max-reasoning-tokens", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", required=True, type=float)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-prompts", required=True, type=int)
    parser.add_argument(
        "--beam-batch-prompts",
        required=True,
        type=int,
        help="Maximum number of sampled reasoning continuations per beam_search call.",
    )
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def parse_domains(value: str) -> tuple[str, ...]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    if not raw:
        raise ValueError("--domains must select at least one domain")
    unknown = [part for part in raw if part not in DOMAIN_ALIASES]
    if unknown:
        raise ValueError(f"unknown domains: {unknown}")
    normalized = [DOMAIN_ALIASES[part] for part in raw]
    if len(set(normalized)) != len(normalized):
        raise ValueError("--domains contains duplicate domains or aliases")
    selected = set(normalized)
    return tuple(domain for domain in DOMAIN_ORDER if domain in selected)


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    positive = {
        "--reasoning-samples": args.reasoning_samples,
        "--item-candidates": args.item_candidates,
        "--max-reasoning-tokens": args.max_reasoning_tokens,
        "--batch-prompts": args.batch_prompts,
        "--beam-batch-prompts": args.beam_batch_prompts,
        "--max-model-len": args.max_model_len,
        "--num-shards": args.num_shards,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive values required for: {invalid}")
    if args.max_reasoning_tokens + 3 >= args.max_model_len:
        raise ValueError(
            "--max-model-len must leave room beyond reasoning and three item tokens"
        )
    if not 0.0 < args.temperature:
        raise ValueError("--temperature must be positive for sampled reasoning")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("--top-k must be -1 or positive")
    if not 0 <= args.seed < 2**31:
        raise ValueError("--seed must be in [0, 2**31)")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    if not args.gpu.strip() or "," in args.gpu:
        raise ValueError("--gpu must name exactly one visible CUDA device")
    return parse_domains(args.domains)


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
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def prompt_identity(record: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "history": record["history"],
            "input": record["input"],
            "instruction": record["instruction"],
        }
    )


def validate_manifest_record(record: Any, location: str) -> dict[str, Any]:
    row = require_exact_keys(record, MANIFEST_KEYS, f"manifest row at {location}")
    if row["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError(f"wrong manifest schema at {location}")
    if not isinstance(row["group_id"], str) or not HEX64.fullmatch(row["group_id"]):
        raise ValueError(f"invalid group_id at {location}")
    if not isinstance(row["instruction"], str):
        raise ValueError(f"instruction must be a string at {location}")
    if not isinstance(row["input"], str) or not row["input"].endswith("/think"):
        raise ValueError(f"input must end exactly with /think at {location}")
    if row["history"] != []:
        raise ValueError(f"history must be the empty list at {location}")
    if row["domain"] not in DOMAIN_TOKENS:
        raise ValueError(f"invalid domain at {location}: {row['domain']!r}")
    if not isinstance(row["prompt_sha256"], str) or not HEX64.fullmatch(
        row["prompt_sha256"]
    ):
        raise ValueError(f"invalid prompt_sha256 at {location}")
    actual_prompt_hash = prompt_identity(row)
    if row["prompt_sha256"] != actual_prompt_hash:
        raise ValueError(
            f"prompt_sha256 mismatch at {location}: "
            f"expected {actual_prompt_hash}, got {row['prompt_sha256']}"
        )
    rollout_seed = row["rollout_seed"]
    if (
        isinstance(rollout_seed, bool)
        or not isinstance(rollout_seed, int)
        or not 0 <= rollout_seed < 2**31
    ):
        raise ValueError(f"rollout_seed must be a 31-bit non-negative int at {location}")
    return row


def load_manifest(
    path: Path,
    domains: Sequence[str],
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    selected_domains = set(domains)
    selected: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            location = f"{path}:{line_index + 1}"
            if not line.strip():
                raise ValueError(f"blank JSONL line at {location}")
            row = validate_manifest_record(strict_json_loads(line), location)
            group_id = row["group_id"]
            prompt_hash = row["prompt_sha256"]
            if group_id in group_ids:
                raise ValueError(f"duplicate group_id in manifest: {group_id}")
            if prompt_hash in prompt_hashes:
                raise ValueError(f"duplicate prompt_sha256 in manifest: {prompt_hash}")
            group_ids.add(group_id)
            prompt_hashes.add(prompt_hash)
            if (
                row["domain"] in selected_domains
                and line_index % num_shards == shard_index
            ):
                selected.append(row)
    return selected


def build_reasoning_prompt(record: dict[str, Any]) -> str:
    sections: list[str] = []
    if record["instruction"]:
        sections.append(
            f"<|im_start|>system\n{record['instruction']}<|im_end|>\n"
        )
    sections.append(
        f"<|im_start|>user\n{record['input']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return "".join(sections)


def thought_body(raw_text: str) -> str:
    """Remove only exact outer think tags; preserve the sampled body byte-for-byte."""
    body = raw_text
    if body.startswith("<think>"):
        body = body[len("<think>") :]
    if body.endswith("</think>"):
        body = body[: -len("</think>")]
    return body


def build_item_prompt(
    reasoning_prompt: str,
    raw_reasoning: str,
    domain: str,
) -> str:
    closing = "\n" if raw_reasoning.endswith("</think>") else "</think>\n"
    return reasoning_prompt + raw_reasoning + closing + DOMAIN_TOKENS[domain]


def derive_seed(global_seed: int, rollout_seed: int) -> int:
    return (global_seed + rollout_seed) % (2**31)


def safe_reason(value: Any) -> str | int | None:
    if value is None or isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    return str(value)


def parsed_item(candidate_text: str, domain: str) -> str | None:
    prefix = DOMAIN_TOKENS[domain]
    suffix = candidate_text[len(prefix) :] if candidate_text.startswith(prefix) else ""
    if candidate_text.startswith(prefix) and SID_SUFFIX.fullmatch(suffix):
        return candidate_text
    return None


def runtime_artifact_fingerprint(path: Path, *, adapter: bool) -> dict[str, Any]:
    root = path.resolve()
    if adapter:
        candidates = [root / "adapter_config.json", root / "adapter_model.safetensors"]
        missing = [str(item) for item in candidates if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"invalid PEFT adapter; missing {missing}")
    else:
        if not (root / "config.json").is_file():
            raise FileNotFoundError(f"base model has no config.json: {root}")
        candidates = sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in MODEL_RUNTIME_SUFFIXES
        )
        weights = [
            item
            for item in candidates
            if item.suffix.lower() in {".safetensors", ".bin"}
        ]
        if not weights:
            raise FileNotFoundError(f"base model has no local weight file: {root}")

    files: list[dict[str, Any]] = []
    for item in candidates:
        relative = item.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    return {
        "path": str(root),
        "artifact_sha256": canonical_sha256(files),
        "files": files,
    }


def adapter_rank(adapter_dir: Path) -> int:
    config = strict_json_loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    ranks: list[int] = []
    rank = config.get("r") if isinstance(config, dict) else None
    if isinstance(rank, int) and not isinstance(rank, bool):
        ranks.append(rank)
    rank_pattern = config.get("rank_pattern", {}) if isinstance(config, dict) else {}
    if isinstance(rank_pattern, dict):
        ranks.extend(
            value
            for value in rank_pattern.values()
            if isinstance(value, int) and not isinstance(value, bool)
        )
    if not ranks or min(ranks) < 1:
        raise ValueError("adapter_config.json has no valid positive LoRA rank")
    return max(ranks)


def vllm_max_lora_rank(rank: int) -> int:
    try:
        return next(value for value in VLLM_LORA_RANKS if value >= rank)
    except StopIteration as exc:
        raise ValueError(f"adapter rank {rank} exceeds vLLM's supported maximum") from exc


def verify_volume_path(path: Path) -> None:
    volume = Path(os.environ.get("PERSONAL_VOLUME_ROOT", str(DEFAULT_VOLUME))).resolve()
    subprocess.run(["mountpoint", "-q", str(volume)], check=True)
    if not os.access(volume, os.W_OK):
        raise PermissionError(f"personal volume is not writable: {volume}")
    resolved = path.resolve()
    if resolved != volume and volume not in resolved.parents:
        raise ValueError(f"path must be on the personal volume: {resolved}")


def verify_input(path: Path, *, directory: bool) -> None:
    verify_volume_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if directory and not path.is_dir():
        raise NotADirectoryError(path)
    if not directory and not path.is_file():
        raise ValueError(f"expected a file: {path}")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def create_empty_output(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to append rollout batch")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_nullable_reason(value: Any, label: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (str, int))
    ):
        raise ValueError(f"{label} must be null, string, or integer")


def validate_rollout_row(
    value: Any,
    manifest_record: dict[str, Any],
    expected_generator: dict[str, Any],
    reasoning_samples: int,
    item_candidates: int,
    location: str,
) -> dict[str, Any]:
    row = require_exact_keys(value, ROLLOUT_KEYS, f"rollout row at {location}")
    if row["schema_version"] != ROLLOUT_SCHEMA:
        raise ValueError(f"wrong rollout schema at {location}")
    for key in ("group_id", "prompt_sha256", "domain"):
        if row[key] != manifest_record[key]:
            raise ValueError(f"{key} does not match manifest at {location}")
    generator = require_exact_keys(
        row["generator"], GENERATOR_KEYS, f"generator at {location}"
    )
    if generator != expected_generator:
        raise ValueError(f"generator identity mismatch at {location}")
    traces = row["traces"]
    if not isinstance(traces, list) or len(traces) != reasoning_samples:
        raise ValueError(f"wrong trace count at {location}")
    trace_ids: set[str] = set()
    indices: list[int] = []
    domain = manifest_record["domain"]
    for trace_offset, trace_value in enumerate(traces):
        trace_label = f"trace {trace_offset} at {location}"
        trace = require_exact_keys(trace_value, TRACE_KEYS, trace_label)
        if not isinstance(trace["trace_id"], str) or not HEX64.fullmatch(
            trace["trace_id"]
        ):
            raise ValueError(f"invalid trace_id in {trace_label}")
        if trace["trace_id"] in trace_ids:
            raise ValueError(f"duplicate trace_id in {trace_label}")
        trace_ids.add(trace["trace_id"])
        if isinstance(trace["reasoning_index"], bool) or not isinstance(
            trace["reasoning_index"], int
        ):
            raise ValueError(f"invalid reasoning_index in {trace_label}")
        indices.append(trace["reasoning_index"])
        if not isinstance(trace["thought"], str):
            raise ValueError(f"thought must be a string in {trace_label}")

        reasoning = require_exact_keys(
            trace["reasoning"], REASONING_KEYS, f"reasoning in {trace_label}"
        )
        if reasoning["text"] != trace["thought"] or not isinstance(
            reasoning["raw_text"], str
        ):
            raise ValueError(f"reasoning text mismatch in {trace_label}")
        if isinstance(reasoning["token_count"], bool) or not isinstance(
            reasoning["token_count"], int
        ) or reasoning["token_count"] < 0:
            raise ValueError(f"invalid reasoning token_count in {trace_label}")
        if isinstance(reasoning["seed"], bool) or not isinstance(
            reasoning["seed"], int
        ) or not 0 <= reasoning["seed"] < 2**31:
            raise ValueError(f"invalid reasoning seed in {trace_label}")
        validate_nullable_reason(
            reasoning["finish_reason"], f"reasoning finish_reason in {trace_label}"
        )
        validate_nullable_reason(
            reasoning["stop_reason"], f"reasoning stop_reason in {trace_label}"
        )

        candidates = trace["candidates"]
        if not isinstance(candidates, list) or len(candidates) != item_candidates:
            raise ValueError(f"wrong candidate count in {trace_label}")
        for candidate_offset, candidate_value in enumerate(candidates):
            candidate_label = f"candidate {candidate_offset} in {trace_label}"
            candidate = require_exact_keys(
                candidate_value, CANDIDATE_KEYS, candidate_label
            )
            if not isinstance(candidate["text"], str):
                raise ValueError(f"candidate text must be a string in {candidate_label}")
            parsed = parsed_item(candidate["text"], domain)
            expected_valid = parsed is not None
            if candidate["valid"] is not expected_valid:
                raise ValueError(f"incorrect valid flag in {candidate_label}")
            if candidate["item"] != parsed:
                raise ValueError(f"incorrect parsed item in {candidate_label}")
            if isinstance(candidate["token_count"], bool) or not isinstance(
                candidate["token_count"], int
            ) or candidate["token_count"] < 0:
                raise ValueError(f"invalid token_count in {candidate_label}")
            score = candidate["cumulative_logprob"]
            if score is not None and (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError(f"invalid cumulative_logprob in {candidate_label}")
            validate_nullable_reason(
                candidate["finish_reason"], f"finish_reason in {candidate_label}"
            )
            validate_nullable_reason(
                candidate["stop_reason"], f"stop_reason in {candidate_label}"
            )
    if sorted(indices) != list(range(reasoning_samples)):
        raise ValueError(f"reasoning_index values are not contiguous at {location}")
    return row


def load_existing_rows(
    path: Path,
    manifest_by_id: dict[str, dict[str, Any]],
    expected_generator: dict[str, Any],
    reasoning_samples: int,
    item_candidates: int,
) -> tuple[set[str], Counter[str]]:
    completed: set[str] = set()
    domain_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            location = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"blank JSONL line at {location}")
            value = strict_json_loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("group_id"), str):
                raise ValueError(f"missing group_id at {location}")
            group_id = value["group_id"]
            if group_id in completed:
                raise ValueError(f"duplicate existing group_id: {group_id}")
            manifest_record = manifest_by_id.get(group_id)
            if manifest_record is None:
                raise ValueError(f"existing group is outside selected shard: {group_id}")
            row = validate_rollout_row(
                value,
                manifest_record,
                expected_generator,
                reasoning_samples,
                item_candidates,
                location,
            )
            completed.add(group_id)
            domain_counts[row["domain"]] += 1
    return completed, domain_counts


def rollout_config(
    args: argparse.Namespace,
    domains: Sequence[str],
    base: dict[str, Any],
    adapter: dict[str, Any],
    adapter_rank_value: int,
    maximum_lora_rank: int,
    manifest_path: Path,
    output_path: Path,
    vllm_version: str,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "schema_version": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "script": {
            "path": str(script_path),
            "sha256": file_sha256(script_path),
        },
        "base": base,
        "adapter": {
            **adapter,
            "rank": adapter_rank_value,
            "vllm_max_lora_rank": maximum_lora_rank,
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "schema_version": MANIFEST_SCHEMA,
            "strict_keys": sorted(MANIFEST_KEYS),
            "contains_gold": False,
        },
        "output": str(output_path.resolve()),
        "selection": {
            "domains": list(domains),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "shard_method": "zero_based_manifest_line_index_modulo",
        },
        "reasoning_sampling": {
            "samples": args.reasoning_samples,
            "max_tokens": args.max_reasoning_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stop": ["</think>"],
            "global_seed": args.seed,
            "per_prompt_seed": "(global_seed + manifest.rollout_seed) mod 2**31",
        },
        "item_beam": {
            "beam_width": args.item_candidates,
            "max_tokens": 3,
            "temperature": 0.0,
            "ignore_eos": False,
            "length_penalty": 1.0,
        },
        "runtime": {
            "gpu": args.gpu,
            "single_gpu": True,
            "dtype": args.dtype,
            "batch_prompts": args.batch_prompts,
            "beam_batch_prompts": args.beam_batch_prompts,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_logprobs": 2 * args.item_candidates,
            "vllm_version": vllm_version,
        },
    }


def generator_identity(
    config: dict[str, Any],
    base: dict[str, Any],
    adapter: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "config_sha256": canonical_sha256(config),
        "base_sha256": base["artifact_sha256"],
        "adapter_sha256": adapter["artifact_sha256"],
        "seed": seed,
    }


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def generate_batch(
    model: Any,
    SamplingParams: Any,
    BeamSearchParams: Any,
    lora_request: Any,
    records: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    generator: dict[str, Any],
) -> list[dict[str, Any]]:
    reasoning_prompts = [build_reasoning_prompt(record) for record in records]
    effective_seeds = [
        derive_seed(args.seed, record["rollout_seed"]) for record in records
    ]
    sampling_params = [
        SamplingParams(
            n=args.reasoning_samples,
            max_tokens=args.max_reasoning_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=seed,
            stop=["</think>"],
        )
        for seed in effective_seeds
    ]
    reasoning_responses = model.generate(
        reasoning_prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    if len(reasoning_responses) != len(records):
        raise RuntimeError("vLLM returned a different number of reasoning responses")

    trace_states: list[dict[str, Any]] = []
    item_prompts: list[str] = []
    for record, reasoning_prompt, effective_seed, response in zip(
        records, reasoning_prompts, effective_seeds, reasoning_responses
    ):
        if len(response.outputs) != args.reasoning_samples:
            raise RuntimeError(
                f"vLLM returned {len(response.outputs)} reasoning samples for "
                f"{record['group_id']}; expected {args.reasoning_samples}"
            )
        for reasoning_index, output in enumerate(response.outputs):
            raw_text = output.text
            if not isinstance(raw_text, str):
                raise RuntimeError("vLLM returned a non-string reasoning output")
            thought = thought_body(raw_text)
            trace_id = hashlib.sha256(
                (
                    f"{record['group_id']}\0{generator['config_sha256']}\0"
                    f"{reasoning_index}"
                ).encode("utf-8")
            ).hexdigest()
            trace_states.append(
                {
                    "record": record,
                    "trace_id": trace_id,
                    "reasoning_index": reasoning_index,
                    "thought": thought,
                    "reasoning": {
                        "text": thought,
                        "raw_text": raw_text,
                        "finish_reason": safe_reason(output.finish_reason),
                        "stop_reason": safe_reason(output.stop_reason),
                        "token_count": len(output.token_ids),
                        "seed": effective_seed,
                    },
                    "candidates": None,
                }
            )
            item_prompts.append(
                build_item_prompt(reasoning_prompt, raw_text, record["domain"])
            )

    beam_params = BeamSearchParams(
        beam_width=args.item_candidates,
        max_tokens=3,
        ignore_eos=False,
        temperature=0.0,
        length_penalty=1.0,
    )
    beam_outputs: list[Any] = []
    for offset in range(0, len(item_prompts), args.beam_batch_prompts):
        prompt_chunk = item_prompts[offset : offset + args.beam_batch_prompts]
        chunk_outputs = model.beam_search(
            [{"prompt": prompt} for prompt in prompt_chunk],
            beam_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        if len(chunk_outputs) != len(prompt_chunk):
            raise RuntimeError("vLLM returned a different number of beam responses")
        beam_outputs.extend(chunk_outputs)
    if len(beam_outputs) != len(trace_states):
        raise RuntimeError("reasoning/beam response alignment failed")

    tokenizer = model.get_tokenizer()
    for state, beam_output in zip(trace_states, beam_outputs):
        sequences = beam_output.sequences
        if len(sequences) != args.item_candidates:
            raise RuntimeError(
                f"vLLM returned {len(sequences)} item beams for {state['trace_id']}; "
                f"expected {args.item_candidates}"
            )
        domain = state["record"]["domain"]
        prefix = DOMAIN_TOKENS[domain]
        candidates: list[dict[str, Any]] = []
        for sequence in sequences:
            # BeamSearchSequence.tokens contains the prompt as well as generated
            # tokens.  logprobs has exactly one entry per generated token, so
            # slicing tokens is robust to decode-time prompt normalization.
            generated_token_count = len(sequence.logprobs)
            generated_tokens = (
                sequence.tokens[-generated_token_count:]
                if generated_token_count
                else []
            )
            generated_text = tokenizer.decode(generated_tokens)
            full_text = prefix + generated_text
            item = parsed_item(full_text, domain)
            candidates.append(
                {
                    "text": full_text,
                    "item": item,
                    "valid": item is not None,
                    "finish_reason": safe_reason(sequence.finish_reason),
                    "stop_reason": safe_reason(sequence.stop_reason),
                    "token_count": generated_token_count,
                    "cumulative_logprob": finite_float(sequence.cum_logprob),
                }
            )
        state["candidates"] = candidates

    states_by_group: dict[str, list[dict[str, Any]]] = {
        record["group_id"]: [] for record in records
    }
    for state in trace_states:
        record = state.pop("record")
        states_by_group[record["group_id"]].append(state)

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "schema_version": ROLLOUT_SCHEMA,
            "group_id": record["group_id"],
            "prompt_sha256": record["prompt_sha256"],
            "domain": record["domain"],
            "generator": generator,
            "traces": states_by_group[record["group_id"]],
        }
        validate_rollout_row(
            row,
            record,
            generator,
            args.reasoning_samples,
            args.item_candidates,
            "newly generated in-memory row",
        )
        rows.append(row)
    return rows


def write_metadata(
    metadata_path: Path,
    output_path: Path,
    config: dict[str, Any],
    generator: dict[str, Any],
    selected_records: Sequence[dict[str, Any]],
    completed: set[str],
    domain_counts: Counter[str],
    args: argparse.Namespace,
    *,
    complete: bool,
) -> None:
    expected_domain_counts = Counter(record["domain"] for record in selected_records)
    metadata = {
        "schema_version": "o1-rec-rollout-metadata-v1",
        "status": "complete" if complete else "in_progress",
        "config_sha256": generator["config_sha256"],
        "output": str(output_path.resolve()),
        "rows_in_file": len(completed),
        "expected_rows": len(selected_records),
        "traces_in_file": len(completed) * args.reasoning_samples,
        "candidates_in_file": (
            len(completed) * args.reasoning_samples * args.item_candidates
        ),
        "domain_rows_in_file": dict(sorted(domain_counts.items())),
        "expected_domain_rows": dict(sorted(expected_domain_counts.items())),
        "selection": config["selection"],
        "sha256": file_sha256(output_path) if complete else None,
    }
    atomic_write_json(metadata_path, metadata)


def run(args: argparse.Namespace) -> None:
    domains = validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    manifest_path = Path(args.manifest)
    output_path = Path(args.out)
    verify_input(base_path, directory=True)
    verify_input(adapter_path, directory=True)
    verify_input(manifest_path, directory=False)
    verify_volume_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"cannot resume missing output: {output_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"cannot resume without rollout config: {config_path}")
    else:
        collisions = [
            path for path in (output_path, config_path, metadata_path) if path.exists()
        ]
        if collisions:
            raise FileExistsError(
                f"rollout path or sidecar exists; pass --resume only for a matching run: "
                f"{collisions}"
            )

    base_fingerprint = runtime_artifact_fingerprint(base_path, adapter=False)
    adapter_fingerprint = runtime_artifact_fingerprint(adapter_path, adapter=True)
    rank = adapter_rank(adapter_path.resolve())
    maximum_lora_rank = vllm_max_lora_rank(rank)

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

    config = rollout_config(
        args,
        domains,
        base_fingerprint,
        adapter_fingerprint,
        rank,
        maximum_lora_rank,
        manifest_path,
        output_path,
        str(vllm_version),
    )
    generator = generator_identity(
        config, base_fingerprint, adapter_fingerprint, args.seed
    )
    if args.resume:
        saved_config = strict_json_loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("resume arguments or artifact hashes do not match the saved lock")
        if canonical_sha256(saved_config) != generator["config_sha256"]:
            raise ValueError("saved rollout config hash is internally inconsistent")
    else:
        atomic_write_json(config_path, config)
        create_empty_output(output_path)

    selected_records = load_manifest(
        manifest_path, domains, args.num_shards, args.shard_index
    )
    manifest_by_id = {record["group_id"]: record for record in selected_records}
    completed, domain_counts = load_existing_rows(
        output_path,
        manifest_by_id,
        generator,
        args.reasoning_samples,
        args.item_candidates,
    )
    pending = [
        record for record in selected_records if record["group_id"] not in completed
    ]
    if not pending:
        write_metadata(
            metadata_path,
            output_path,
            config,
            generator,
            selected_records,
            completed,
            domain_counts,
            args,
            complete=True,
        )
        print(f"no pending prompts; {len(completed)} rows are complete")
        return

    lora_request = LoRARequest(
        adapter_path.resolve().name,
        1,
        lora_path=str(adapter_path.resolve()),
    )
    model = LLM(
        model=str(base_path.resolve()),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=args.seed,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_logprobs=2 * args.item_candidates,
        enable_lora=True,
        max_lora_rank=maximum_lora_rank,
    )

    generated = 0
    for offset in range(0, len(pending), args.batch_prompts):
        batch = pending[offset : offset + args.batch_prompts]
        rows = generate_batch(
            model,
            SamplingParams,
            BeamSearchParams,
            lora_request,
            batch,
            args,
            generator,
        )
        append_rows(output_path, rows)
        for row in rows:
            completed.add(row["group_id"])
            domain_counts[row["domain"]] += 1
        generated += len(rows)
        write_metadata(
            metadata_path,
            output_path,
            config,
            generator,
            selected_records,
            completed,
            domain_counts,
            args,
            complete=False,
        )
        print(
            f"generated {generated}/{len(pending)} pending prompts; "
            f"total {len(completed)}/{len(selected_records)}",
            flush=True,
        )

    write_metadata(
        metadata_path,
        output_path,
        config,
        generator,
        selected_records,
        completed,
        domain_counts,
        args,
        complete=True,
    )


def self_test() -> None:
    instruction = "You recommend one item."
    input_text = "history only /think"
    prompt_hash = canonical_sha256(
        {"history": [], "input": input_text, "instruction": instruction}
    )
    manifest_record = {
        "schema_version": MANIFEST_SCHEMA,
        "group_id": "a" * 64,
        "instruction": instruction,
        "input": input_text,
        "history": [],
        "domain": "video",
        "prompt_sha256": prompt_hash,
        "rollout_seed": 17,
    }
    validate_manifest_record(manifest_record, "self-test")
    assert parse_domains("video,live") == ("video", "living")
    assert derive_seed(2**31 - 1, 2) == 1
    assert thought_body("<think>\nactual\n") == "\nactual\n"
    assert thought_body("<think>actual</think>") == "actual"
    prompt = build_reasoning_prompt(manifest_record)
    assert "You recommend one item." in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    item = "<|video_begin|><s_a_1><s_b_2><s_c_3>"
    assert parsed_item(item, "video") == item
    assert parsed_item(item + " ", "video") is None

    generator = {
        "config_sha256": "b" * 64,
        "base_sha256": "c" * 64,
        "adapter_sha256": "d" * 64,
        "seed": 11,
    }
    rollout_row = {
        "schema_version": ROLLOUT_SCHEMA,
        "group_id": manifest_record["group_id"],
        "prompt_sha256": prompt_hash,
        "domain": "video",
        "generator": generator,
        "traces": [
            {
                "trace_id": "e" * 64,
                "reasoning_index": 0,
                "thought": "actual",
                "reasoning": {
                    "text": "actual",
                    "raw_text": "<think>actual",
                    "finish_reason": "stop",
                    "stop_reason": "</think>",
                    "token_count": 3,
                    "seed": 28,
                },
                "candidates": [
                    {
                        "text": item,
                        "item": item,
                        "valid": True,
                        "finish_reason": "length",
                        "stop_reason": None,
                        "token_count": 3,
                        "cumulative_logprob": -0.25,
                    }
                ],
            }
        ],
    }
    validate_rollout_row(
        rollout_row, manifest_record, generator, 1, 1, "self-test"
    )
    forbidden = dict(manifest_record)
    forbidden["golds"] = [item]
    try:
        validate_manifest_record(forbidden, "self-test-forbidden")
    except ValueError:
        pass
    else:
        raise AssertionError("manifest label field was not rejected")

    with tempfile.TemporaryDirectory(prefix="rec-rft-rollout-selftest-") as directory:
        root = Path(directory)
        output = root / "rollouts.jsonl"
        metadata = root / "metadata.json"
        create_empty_output(output)
        append_rows(output, [rollout_row])
        completed, counts = load_existing_rows(
            output, {manifest_record["group_id"]: manifest_record}, generator, 1, 1
        )
        assert completed == {manifest_record["group_id"]}
        assert counts == Counter({"video": 1})
        atomic_write_json(metadata, {"ok": True})
        assert strict_json_loads(metadata.read_text(encoding="utf-8")) == {"ok": True}
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> None:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv == ["--self-test"]:
        self_test()
        return
    if "--self-test" in actual_argv:
        raise SystemExit("--self-test must be used alone")
    run(parse_args(actual_argv))


if __name__ == "__main__":
    main()
