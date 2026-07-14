#!/usr/bin/env python3
"""Read-only EDA for registered official assets O4 and O5.

The script deliberately separates exact, cheap statistics (Parquet footers and
the ``source`` column) from content statistics estimated with a deterministic,
source-stratified sample. It never writes to the official asset directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPO_ROOT / "assets" / "official"
DEFAULT_TOKENIZER = OFFICIAL_ROOT / "base_model"

ASSETS = {
    "O4": {
        "name": "OpenOneRec-General-Pretrain",
        "path": OFFICIAL_ROOT / "general_pretrain",
        "revision": "ed57951e14595112eb18d47b850776e9407b8ff9",
    },
    "O5": {
        "name": "OpenOneRec-General-SFT",
        "path": OFFICIAL_ROOT / "general_sft",
        "revision": "4b8e43913aeb8e6c66b9253df4ab64ecc77dfd6c",
    },
}

SOURCE_FAMILIES = {
    "OpenMathReasoning": "math_reasoning",
    "NuminaMath-QwQ-CoT-5M": "math_reasoning",
    "DeepMath103K": "math_reasoning",
    "OpenCodeReasoning_new": "code_reasoning",
    "OpenCoderReasoning": "code_reasoning",
    "KodCode_V1_SFT_R1": "code_reasoning",
    "Reasoning_KodCode_V1_SFT_R1": "code_reasoning",
    "Bespoke-Stratos-17k": "code_reasoning",
    "medical-o1-reasoning-SFT": "medical_reasoning",
    "medical-o1-reasoning-SFT-think": "medical_reasoning",
    "Chinese-Reasoning-Distil-Data": "chinese_general_reasoning",
    "Chinese-Reasoning-Distil-Data-think": "chinese_general_reasoning",
    "reasoning_v1_20m": "general_reasoning",
    "R1-Distill-SFT": "general_reasoning",
    "Infinity_Instruct": "general_instruction",
    "Reasoning_Multi_subject_RLVR": "multi_subject_reasoning",
}

THINK_OPEN_RE = re.compile(r"<think>|<\|begin_of_thought\|>|<analysis>", re.I)
THINK_CLOSE_RE = re.compile(r"</think>|<\|end_of_thought\|>|</analysis>", re.I)
ANSWER_TAG_RE = re.compile(r"<answer>|</answer>|<\|begin_of_solution\|>", re.I)
REASONING_CUE_RE = re.compile(
    r"\b(?:let me|we need|first|second|step[- ]by[- ]step|to solve|let us|"
    r"therefore|thus)\b|(?:首先|先来|需要|分析|思考|解题|因此|所以)",
    re.I,
)
INLINE_CHOICE_RE = re.compile(r"(?<![A-Za-z])\(([A-H])\)", re.I)
CHOICE_MARKER_RE = re.compile(
    r"(?:^|[\n\r]|\s)(?:\(?([A-H])\)|([A-H])[.、:：])(?=\s|[^A-Za-z])",
    re.I,
)
MC_ANSWER_RE = re.compile(
    r"(?:答案|正确选项|answer|option|choice|<answer>)\s*(?:是|为|:|：)?\s*"
    r"[\(\[]?([A-H])[\)\]]?",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=sorted(ASSETS),
        default=sorted(ASSETS),
        help="Registered asset IDs to inspect (default: O4 O5).",
    )
    parser.add_argument(
        "--sample-files",
        type=int,
        default=32,
        help="Hash-selected Parquet files read for content sampling per asset.",
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=400,
        help="Reservoir size retained for each exact source label.",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Local tokenizer used for exact sampled token lengths.",
    )
    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Skip tokenizer loading and token-length statistics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Parquet streaming batch size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON here; stdout is used when omitted.",
    )
    args = parser.parse_args()
    if args.sample_files <= 0:
        parser.error("--sample-files must be positive")
    if args.samples_per_source <= 0:
        parser.error("--samples-per-source must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def official_files(asset_path: Path) -> list[Path]:
    if not asset_path.exists():
        raise FileNotFoundError(f"registered asset path is missing: {asset_path}")
    files = sorted(asset_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no top-level Parquet files at: {asset_path}")
    return files


def canonical_schema(schema: Any) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((field.name, str(field.type)) for field in schema))


def parquet_footer_summary(files: Sequence[Path]) -> dict[str, Any]:
    total_rows = 0
    total_row_groups = 0
    total_bytes = 0
    ordered_schemas: Counter[tuple[tuple[str, str], ...]] = Counter()
    logical_schemas: Counter[tuple[tuple[str, str], ...]] = Counter()
    null_counts: Counter[str] = Counter()
    null_unknown_rows: Counter[str] = Counter()

    for path in files:
        total_bytes += path.stat().st_size
        parquet = pq.ParquetFile(path)
        total_rows += parquet.metadata.num_rows
        total_row_groups += parquet.metadata.num_row_groups
        ordered = tuple((field.name, str(field.type)) for field in parquet.schema_arrow)
        ordered_schemas[ordered] += 1
        logical_schemas[canonical_schema(parquet.schema_arrow)] += 1

        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                name = column.path_in_schema
                statistics = column.statistics
                if statistics is not None and statistics.has_null_count:
                    null_counts[name] += statistics.null_count
                else:
                    null_unknown_rows[name] += row_group.num_rows

    def render_schema_counts(
        counts: Counter[tuple[tuple[str, str], ...]],
    ) -> list[dict[str, Any]]:
        return [
            {"file_count": count, "fields": list(schema)}
            for schema, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_rows": total_rows,
        "row_group_count": total_row_groups,
        "ordered_schema_variants": render_schema_counts(ordered_schemas),
        "logical_schema_variants": render_schema_counts(logical_schemas),
        "footer_null_counts": dict(sorted(null_counts.items())),
        "footer_null_count_unknown_rows": dict(sorted(null_unknown_rows.items())),
    }


def exact_source_counts(files: Sequence[Path], batch_size: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in files:
        parquet = pq.ParquetFile(path)
        if "source" not in parquet.schema_arrow.names:
            counts["<MISSING_COLUMN>"] += parquet.metadata.num_rows
            continue
        for batch in parquet.iter_batches(columns=["source"], batch_size=batch_size):
            counts.update("<NULL>" if value is None else str(value) for value in batch.column(0).to_pylist())
    return counts


def hash_selected_files(
    files: Sequence[Path], asset_id: str, seed: int, count: int
) -> list[Path]:
    def key(path: Path) -> bytes:
        payload = f"{seed}\0{asset_id}\0{path.name}".encode()
        return hashlib.sha256(payload).digest()

    return sorted(sorted(files, key=key)[: min(count, len(files))])


def source_rng(seed: int, asset_id: str, source: str) -> random.Random:
    payload = f"{seed}\0{asset_id}\0{source}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def reservoir_sample(
    files: Sequence[Path],
    asset_id: str,
    seed: int,
    samples_per_source: int,
    batch_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], int]:
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    rngs: dict[str, random.Random] = {}
    rows_scanned = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        columns = [name for name in ("source", "messages", "text") if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            for row in batch.to_pylist():
                rows_scanned += 1
                source = "<NULL>" if row.get("source") is None else str(row["source"])
                seen[source] += 1
                reservoir = reservoirs[source]
                compact_row = {
                    "source": source,
                    "messages": row.get("messages"),
                    "text": row.get("text"),
                    "file": path.name,
                }
                if len(reservoir) < samples_per_source:
                    reservoir.append(compact_row)
                    continue
                rng = rngs.setdefault(source, source_rng(seed, asset_id, source))
                replacement_index = rng.randrange(seen[source])
                if replacement_index < samples_per_source:
                    reservoir[replacement_index] = compact_row

    return reservoirs, seen, rows_scanned


def content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := content_text(item)))
    if isinstance(value, dict):
        if "text" in value:
            return content_text(value["text"])
        if value.get("type") == "text" and "content" in value:
            return content_text(value["content"])
        return "\n".join(
            part
            for key, item in value.items()
            if key not in {"type", "role"} and (part := content_text(item))
        )
    return str(value)


def parse_message_list(raw: Any) -> tuple[list[dict[str, Any]], str | None]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError) as exc:
        return [], type(exc).__name__
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    if not isinstance(value, list):
        return [], "not_a_list"
    if not all(isinstance(message, dict) for message in value):
        return [], "non_object_message"
    return value, None


def language_bucket(text: str) -> str:
    counts = Counter()
    for char in text:
        codepoint = ord(char)
        if 0x3400 <= codepoint <= 0x9FFF or 0x20000 <= codepoint <= 0x3134F:
            counts["han"] += 1
        elif 0x3040 <= codepoint <= 0x30FF:
            counts["kana"] += 1
        elif 0xAC00 <= codepoint <= 0xD7AF:
            counts["hangul"] += 1
        elif char.isascii() and char.isalpha():
            counts["latin"] += 1
        elif 0x0400 <= codepoint <= 0x052F:
            counts["cyrillic"] += 1
        elif 0x0600 <= codepoint <= 0x06FF:
            counts["arabic"] += 1
        elif char.isalpha() and not unicodedata.category(char).startswith("M"):
            counts["other"] += 1

    letters = sum(counts.values())
    if letters == 0:
        return "empty_or_symbolic"
    if counts["kana"] / letters >= 0.03:
        return "ja"
    if counts["hangul"] / letters >= 0.10:
        return "ko"
    han_latin = counts["han"] + counts["latin"]
    if han_latin:
        han_ratio = counts["han"] / han_latin
        if counts["han"] >= 8 and han_ratio >= 0.60:
            return "zh"
        if counts["han"] >= 8 and 0.10 < han_ratio < 0.60:
            return "zh_en_mixed"
        if counts["latin"] >= 12 and han_ratio <= 0.10:
            return "en"
    dominant = max(counts, key=counts.get)
    return {"cyrillic": "cyrillic", "arabic": "arabic"}.get(dominant, "other")


def is_multiple_choice(question: str) -> bool:
    markers = {match.upper() for match in INLINE_CHOICE_RE.findall(question)}
    for match in CHOICE_MARKER_RE.finditer(question):
        marker = match.group(1) or match.group(2)
        if marker:
            markers.add(marker.upper())
    return len(markers) >= 3


def mc_answer_in_tail(answer: str) -> bool:
    tail = answer[-600:]
    if MC_ANSWER_RE.search(tail):
        return True
    stripped = re.sub(r"</?(?:answer|think)>", " ", tail, flags=re.I).strip()
    return bool(re.fullmatch(r"[\(\[]?[A-H][\)\].。]?", stripped, re.I))


def analyze_sample_row(row: dict[str, Any]) -> dict[str, Any]:
    messages, parse_error = parse_message_list(row.get("messages"))
    role_texts: dict[str, list[str]] = defaultdict(list)
    role_counts: Counter[str] = Counter()
    for message in messages:
        role = str(message.get("role", "<MISSING>"))
        role_counts[role] += 1
        role_texts[role].append(content_text(message.get("content")))

    system_text = "\n".join(role_texts.get("system", []))
    user_text = "\n".join(role_texts.get("user", []))
    assistant_text = "\n".join(role_texts.get("assistant", []))
    prompt_text = "\n".join(part for part in (system_text, user_text) if part)
    total_text = "\n".join(part for part in (prompt_text, assistant_text) if part)
    think_open = bool(THINK_OPEN_RE.search(assistant_text))
    think_close = bool(THINK_CLOSE_RE.search(assistant_text))
    explicit_cot = think_open or think_close
    likely_cot = explicit_cot or (
        len(assistant_text) >= 400 and bool(REASONING_CUE_RE.search(assistant_text[:4000]))
    )
    mc = is_multiple_choice(user_text)

    return {
        "source": row["source"],
        "file": row["file"],
        "parse_ok": parse_error is None,
        "parse_error": parse_error,
        "message_count": len(messages),
        "system_count": role_counts["system"],
        "user_count": role_counts["user"],
        "assistant_count": role_counts["assistant"],
        "prompt": prompt_text,
        "response": assistant_text,
        "total": total_text,
        "prompt_chars": len(prompt_text),
        "response_chars": len(assistant_text),
        "total_chars": len(total_text),
        "prompt_language": language_bucket(user_text),
        "response_language": language_bucket(assistant_text),
        "explicit_cot": explicit_cot,
        "paired_think_tags": think_open and think_close,
        "unclosed_think_tags": think_open and not think_close,
        "response_starts_think": bool(
            re.match(r"\s*(?:<think>|<\|begin_of_thought\|>|<analysis>)", assistant_text, re.I)
        ),
        "likely_cot": likely_cot,
        "answer_tag": bool(ANSWER_TAG_RE.search(assistant_text)),
        "multiple_choice": mc,
        "chinese_multiple_choice": mc
        and language_bucket(user_text) in {"zh", "zh_en_mixed"},
        "mc_answer_in_tail": mc and mc_answer_in_tail(assistant_text),
    }


def attach_token_lengths(records: list[dict[str, Any]], tokenizer_path: Path) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    for field in ("prompt", "response", "total"):
        output_name = f"{field}_tokens"
        for start in range(0, len(records), 64):
            chunk = records[start : start + 64]
            encoded = tokenizer(
                [record[field] for record in chunk],
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_length=True,
            )
            for record, length in zip(chunk, encoded["length"], strict=True):
                record[output_name] = int(length)
    return f"{type(tokenizer).__name__}:{len(tokenizer)}"


def weighted_rate(
    records: Sequence[dict[str, Any]],
    weights: Sequence[float],
    predicate: Callable[[dict[str, Any]], bool],
    eligible: Callable[[dict[str, Any]], bool] | None = None,
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record, weight in zip(records, weights, strict=True):
        if eligible is not None and not eligible(record):
            continue
        denominator += weight
        if predicate(record):
            numerator += weight
    return None if denominator == 0 else numerator / denominator


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(zip(values, weights, strict=True), key=lambda pair: pair[0])
    total_weight = sum(weight for _, weight in ordered)
    threshold = q * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def distribution(
    records: Sequence[dict[str, Any]], weights: Sequence[float], field: str
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0.0
    for record, weight in zip(records, weights, strict=True):
        key = str(record[field])
        counts[key] += weight
        total += weight
    if total == 0:
        return {}
    return {key: round(value / total, 6) for key, value in counts.most_common()}


def numeric_summary(
    records: Sequence[dict[str, Any]], weights: Sequence[float], field: str
) -> dict[str, Any] | None:
    pairs = [
        (float(record[field]), weight)
        for record, weight in zip(records, weights, strict=True)
        if field in record
    ]
    if not pairs:
        return None
    values = [value for value, _ in pairs]
    field_weights = [weight for _, weight in pairs]
    result: dict[str, Any] = {
        "mean": rounded(sum(v * w for v, w in pairs) / sum(field_weights), 2),
        "p50": weighted_quantile(values, field_weights, 0.50),
        "p90": weighted_quantile(values, field_weights, 0.90),
        "p95": weighted_quantile(values, field_weights, 0.95),
        "p99": weighted_quantile(values, field_weights, 0.99),
        "max_in_sample": max(values),
    }
    if field.endswith("_tokens"):
        for cutoff in (4096, 8192, 16384, 32768):
            result[f"gt_{cutoff}_rate"] = rounded(
                sum(weight for value, weight in pairs if value > cutoff) / sum(field_weights)
            )
    return result


def summarize_records(
    records: list[dict[str, Any]], source_counts: Counter[str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    sampled_source_counts = Counter(record["source"] for record in records)
    weights = [
        source_counts[record["source"]] / sampled_source_counts[record["source"]]
        for record in records
    ]
    valid = lambda record: bool(record["parse_ok"])
    mc = lambda record: valid(record) and bool(record["multiple_choice"])

    overall = {
        "sample_rows": len(records),
        "source_weighting": "exact source rows / sampled rows for that source",
        "parse_success_rate": rounded(weighted_rate(records, weights, valid)),
        "has_system_rate": rounded(
            weighted_rate(records, weights, lambda r: r["system_count"] > 0, valid)
        ),
        "multi_turn_user_rate": rounded(
            weighted_rate(records, weights, lambda r: r["user_count"] > 1, valid)
        ),
        "missing_assistant_rate": rounded(
            weighted_rate(records, weights, lambda r: r["assistant_count"] == 0, valid)
        ),
        "prompt_language_distribution": distribution(
            [r for r in records if valid(r)],
            [w for r, w in zip(records, weights, strict=True) if valid(r)],
            "prompt_language",
        ),
        "response_language_distribution": distribution(
            [r for r in records if valid(r)],
            [w for r, w in zip(records, weights, strict=True) if valid(r)],
            "response_language",
        ),
        "explicit_cot_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["explicit_cot"]), valid)
        ),
        "likely_cot_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["likely_cot"]), valid)
        ),
        "paired_think_tag_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["paired_think_tags"]), valid)
        ),
        "unclosed_think_tag_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["unclosed_think_tags"]), valid)
        ),
        "response_starts_think_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["response_starts_think"]), valid)
        ),
        "answer_tag_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["answer_tag"]), valid)
        ),
        "multiple_choice_rate": rounded(weighted_rate(records, weights, mc, valid)),
        "chinese_multiple_choice_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["chinese_multiple_choice"]), valid)
        ),
        "mc_answer_in_tail_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["mc_answer_in_tail"]), mc)
        ),
        "lengths": {
            field: summary
            for field in (
                "prompt_chars",
                "response_chars",
                "total_chars",
                "prompt_tokens",
                "response_tokens",
                "total_tokens",
            )
            if (summary := numeric_summary(records, weights, field)) is not None
        },
    }

    source_profiles: dict[str, dict[str, Any]] = {}
    for source in sorted(sampled_source_counts):
        source_records = [record for record in records if record["source"] == source]
        source_weights = [1.0] * len(source_records)
        profile, _ = summarize_unweighted_source(source_records, source_weights)
        profile.update(
            {
                "exact_rows": source_counts[source],
                "exact_row_share": rounded(source_counts[source] / sum(source_counts.values())),
                "source_family": SOURCE_FAMILIES.get(source, "other"),
            }
        )
        source_profiles[source] = profile
    return overall, source_profiles


def summarize_unweighted_source(
    records: list[dict[str, Any]], weights: list[float]
) -> tuple[dict[str, Any], None]:
    valid = lambda record: bool(record["parse_ok"])
    profile = {
        "sample_rows": len(records),
        "parse_success_rate": rounded(weighted_rate(records, weights, valid)),
        "prompt_language_distribution": distribution(
            [r for r in records if valid(r)],
            [w for r, w in zip(records, weights, strict=True) if valid(r)],
            "prompt_language",
        ),
        "explicit_cot_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["explicit_cot"]), valid)
        ),
        "likely_cot_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["likely_cot"]), valid)
        ),
        "multiple_choice_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["multiple_choice"]), valid)
        ),
        "chinese_multiple_choice_rate": rounded(
            weighted_rate(records, weights, lambda r: bool(r["chinese_multiple_choice"]), valid)
        ),
        "total_tokens": numeric_summary(records, weights, "total_tokens"),
    }
    return profile, None


def exact_family_counts(source_counts: Counter[str]) -> dict[str, dict[str, Any]]:
    family_counts: Counter[str] = Counter()
    for source, count in source_counts.items():
        family_counts[SOURCE_FAMILIES.get(source, "other")] += count
    total = sum(family_counts.values())
    return {
        family: {"rows": rows, "share": rounded(rows / total)}
        for family, rows in family_counts.most_common()
    }


def analyze_asset(
    asset_id: str,
    sample_files: int,
    samples_per_source: int,
    seed: int,
    batch_size: int,
    tokenizer_path: Path | None,
) -> dict[str, Any]:
    spec = ASSETS[asset_id]
    files = official_files(spec["path"])
    footer = parquet_footer_summary(files)
    source_counts = exact_source_counts(files, batch_size)
    selected = hash_selected_files(files, asset_id, seed, sample_files)
    reservoirs, sampled_seen, rows_scanned = reservoir_sample(
        selected,
        asset_id,
        seed,
        samples_per_source,
        batch_size,
    )
    records = [
        analyze_sample_row(row)
        for source in sorted(reservoirs)
        for row in reservoirs[source]
    ]
    tokenizer_identity = None
    if tokenizer_path is not None:
        tokenizer_identity = attach_token_lengths(records, tokenizer_path)
    overall, source_profiles = summarize_records(records, source_counts)
    selection_payload = "\n".join(path.name for path in selected).encode()

    return {
        "asset_id": asset_id,
        "name": spec["name"],
        "official_revision": spec["revision"],
        "registered_path": str(spec["path"].relative_to(REPO_ROOT)),
        "parquet": footer,
        "exact_source_counts": dict(source_counts.most_common()),
        "exact_source_shares": {
            source: rounded(count / sum(source_counts.values()))
            for source, count in source_counts.most_common()
        },
        "exact_source_family_distribution": exact_family_counts(source_counts),
        "content_sample": {
            "selection_method": "lowest SHA256(seed, asset_id, filename)",
            "selected_file_count": len(selected),
            "selected_files_sha256": hashlib.sha256(selection_payload).hexdigest(),
            "selected_files": [path.name for path in selected],
            "rows_scanned": rows_scanned,
            "rows_seen_by_source": dict(sampled_seen.most_common()),
            "reservoir_target_per_source": samples_per_source,
            "tokenizer": tokenizer_identity,
            "overall_source_weighted": overall,
            "source_profiles": source_profiles,
        },
    }


def main() -> int:
    args = parse_args()
    tokenizer_path: Path | None = None if args.skip_tokenization else args.tokenizer
    if tokenizer_path is not None and not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer path is missing: {tokenizer_path}")

    result = {
        "eda_version": 1,
        "method": {
            "exact": "all Parquet footers plus a full scan of only the source column",
            "content": (
                "deterministic hash-selected files; uniform reservoir per source; "
                "overall content estimates weighted by exact source row counts"
            ),
            "language": "Unicode-script heuristic on user text; not a language model",
            "multiple_choice": "at least three distinct A-H option markers in user text",
            "likely_cot": "explicit thought tags or a >=400-char answer with reasoning cues",
            "seed": args.seed,
            "sample_files_per_asset": args.sample_files,
            "reservoir_target_per_source": args.samples_per_source,
        },
        "assets": {},
    }
    for asset_id in args.assets:
        print(f"[EDA] analyzing {asset_id}...", file=sys.stderr, flush=True)
        result["assets"][asset_id] = analyze_asset(
            asset_id,
            args.sample_files,
            args.samples_per_source,
            args.seed,
            args.batch_size,
            tokenizer_path,
        )

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
        return 0

    output = args.output.resolve()
    official_root = OFFICIAL_ROOT.resolve()
    if is_relative_to(output, official_root):
        raise ValueError(f"refusing to write inside official assets: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"[EDA] wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
