#!/usr/bin/env python3
"""Verify the frozen O1-O6 official-data EDA snapshot without mutating assets.

This is intentionally a verifier, not another asset registry. Paths and asset
identity remain authoritative only in ``docs/reference/ASSETS.md``. The default
run checks cheap, exact invariants. ``--tokenize-o1`` additionally recomputes
O1 chat-template lengths and target-token shares with the registered O6
tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "assets" / "official"
EVALUATION = ROOT / "assets" / "evaluation"

O1 = OFFICIAL / "seed_sft"
O2 = OFFICIAL / "hf_raw"
O3 = OFFICIAL / "sft_aligned" / "baseline_caption_tag_lists.parquet"
O4 = OFFICIAL / "general_pretrain"
O5 = OFFICIAL / "general_sft"
O6 = OFFICIAL / "base_model"

MODE_RE = re.compile(r"/(?:no_)?think\s*$")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ITEM_RE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)

EXPECTED_ROWS = {
    "OneReason_UserProfile": 500_000,
    "OneReason_Pid2Caption": 21_061_327,
    "OneReason_Pid2Sid": 35_914_095,
    "OneReason_Pid2Tag": 5_417_279,
    "OneReason_General": 152_005,
    "O3": 19_204,
    "O4": 2_655_181,
    "O5": 2_555_706,
}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", MODE_RE.sub("", value.rstrip())).casefold()


def answer_body(response: str) -> str:
    return response.split("</think>", 1)[-1].lstrip("\n")


def task_of(row: dict[str, str]) -> str:
    body = answer_body(row["response"]).strip()
    if body.startswith("["):
        return "user_action"
    if body.startswith("{") and "logic_chain" in body:
        return "user_topic"
    if "该用户最近" in body:
        matches = ITEM_RE.findall(body)
        if not matches:
            return "rec_unknown"
        return f"rec_{matches[-1][0]}"
    prompt_has_item = bool(ITEM_RE.search(row["prompt"]))
    answer_has_item = bool(ITEM_RE.search(body))
    if answer_has_item and not prompt_has_item:
        return "material_desc2sid"
    if prompt_has_item and not answer_has_item:
        return "material_sid2desc"
    return "unknown"


def iter_o1() -> Iterable[tuple[str, int, dict[str, str]]]:
    for path in sorted(O1.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                value = json.loads(line)
                if not isinstance(value, list) or len(value) != 1:
                    raise AssertionError(f"{path}:{line_number}: expected length-1 list")
                row = value[0]
                if set(row) != {"system", "prompt", "response"}:
                    raise AssertionError(f"{path}:{line_number}: unexpected keys")
                yield path.name, line_number, row


def quantile_higher(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("empty quantile input")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered) + 0.999999999) - 1))
    return ordered[index]


def summarize_lengths(values: list[int]) -> dict[str, int]:
    return {
        "n": len(values),
        "sum": sum(values),
        "p50": quantile_higher(values, 0.50),
        "p90": quantile_higher(values, 0.90),
        "p95": quantile_higher(values, 0.95),
        "p99": quantile_higher(values, 0.99),
        "max": max(values),
        "gt_4096": sum(value > 4096 for value in values),
        "gt_8192": sum(value > 8192 for value in values),
        "gt_16384": sum(value > 16384 for value in values),
    }


def encoded_length(value: Any) -> int:
    """Return sequence length for either a token list or BatchEncoding.

    The registered tokenizer currently makes ``apply_chat_template`` return a
    BatchEncoding even without ``return_dict=True``. ``len(BatchEncoding)`` is
    the number of fields, not the number of tokens.
    """
    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise AssertionError("expected one unbatched token sequence")
        value = value[0]
    return len(value)


def inspect_o1(tokenize: bool) -> tuple[dict[str, Any], list[str]]:
    counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    format_errors: Counter[str] = Counter()
    full_hashes: Counter[str] = Counter()
    rec_groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "think": set(), "answers": set()}
    )
    rec_message_hashes: list[str] = []
    rows_for_tokens: list[tuple[str, dict[str, str]]] = []

    for _, _, row in iter_o1():
        task = task_of(row)
        counts[task] += 1
        match = THINK_RE.search(row["response"])
        mode = (
            "no_think"
            if row["prompt"].rstrip().endswith("/no_think")
            else "think"
            if row["prompt"].rstrip().endswith("/think")
            else "missing"
        )
        state = "missing" if match is None else "nonempty" if match.group(1).strip() else "empty"
        modes[f"{mode}|{state}"] += 1
        if mode == "think" and state != "nonempty":
            format_errors["think_mode_mismatch"] += 1
        if mode == "no_think" and state != "empty":
            format_errors["nothink_mode_mismatch"] += 1
        for domain, a, b, c in ITEM_RE.findall(row["prompt"] + row["response"]):
            if any(not 0 <= int(value) <= 8191 for value in (a, b, c)):
                format_errors[f"item_range_{domain}"] += 1

        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        full_hashes[digest_text(canonical)] += 1

        if task.startswith("rec_"):
            core = row["system"] + "\0" + MODE_RE.sub("", row["prompt"].rstrip())
            group = rec_groups[digest_text(core)]
            group["rows"] += 1
            group["think"].add(digest_text(match.group(1).strip() if match else ""))
            group["answers"].add(digest_text(answer_body(row["response"])))
            messages = [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": MODE_RE.sub("", row["prompt"].rstrip())},
                {"role": "assistant", "content": row["response"]},
            ]
            rec_message_hashes.append(
                digest_text(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            )
        if tokenize:
            rows_for_tokens.append((task, row))

    duplicate_groups = [group for group in rec_groups.values() if group["rows"] > 1]
    result: dict[str, Any] = {
        "rows": sum(counts.values()),
        "task_counts": dict(counts),
        "mode_think_counts": dict(modes),
        "format_errors": dict(format_errors),
        "exact_full": {
            "unique": len(full_hashes),
            "extra_rows": sum(value - 1 for value in full_hashes.values()),
            "duplicate_groups": sum(value > 1 for value in full_hashes.values()),
            "max_group": max(full_hashes.values()),
        },
        "recommendation_prompt_groups": {
            "groups": len(rec_groups),
            "duplicate_groups": len(duplicate_groups),
            "rows_in_duplicate_groups": sum(group["rows"] for group in duplicate_groups),
            "max_group": max(group["rows"] for group in rec_groups.values()),
            "duplicate_groups_same_think": sum(len(group["think"]) == 1 for group in duplicate_groups),
            "duplicate_groups_all_distinct_answers": sum(
                len(group["answers"]) == group["rows"] for group in duplicate_groups
            ),
        },
    }

    if tokenize:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            O6, trust_remote_code=True, local_files_only=True
        )
        full_lengths: dict[str, list[int]] = defaultdict(list)
        target_lengths: dict[str, list[int]] = defaultdict(list)
        for task, row in rows_for_tokens:
            prompt_messages = []
            if row["system"]:
                prompt_messages.append({"role": "system", "content": row["system"]})
            prompt_messages.append({"role": "user", "content": row["prompt"]})
            full_messages = prompt_messages + [
                {"role": "assistant", "content": row["response"]}
            ]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            )
            full_length = encoded_length(full_ids)
            prompt_length = encoded_length(prompt_ids)
            full_lengths[task].append(full_length)
            target_lengths[task].append(max(0, full_length - prompt_length))
        total_targets = sum(sum(values) for values in target_lengths.values())
        if total_targets <= 0:
            raise AssertionError("O1 assistant target-token count must be positive")
        result["o6_chat_template_lengths"] = {
            "all_full": summarize_lengths(
                [value for values in full_lengths.values() for value in values]
            ),
            "by_task_full": {
                task: summarize_lengths(values) for task, values in sorted(full_lengths.items())
            },
            "by_task_target": {
                task: {
                    **summarize_lengths(values),
                    "share": round(sum(values) / total_targets, 8),
                }
                for task, values in sorted(target_lengths.items())
            },
        }

    return result, rec_message_hashes


def parquet_dataset_summary(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files: {path}")
    rows = 0
    bytes_ = 0
    schemas: Counter[str] = Counter()
    column_counts: Counter[int] = Counter()
    for file in files:
        parquet = pq.ParquetFile(file)
        rows += parquet.metadata.num_rows
        bytes_ += file.stat().st_size
        schema = tuple((field.name, str(field.type)) for field in parquet.schema_arrow)
        schemas[repr(schema)] += 1
        column_counts[len(schema)] += 1
    return {
        "files": len(files),
        "rows": rows,
        "bytes": bytes_,
        "schema_variants": len(schemas),
        "column_count_file_counts": {
            str(count): file_count
            for count, file_count in sorted(column_counts.items())
        },
        "schema_file_counts": dict(schemas),
    }


def inspect_o2() -> dict[str, Any]:
    result = {}
    for name in sorted(EXPECTED_ROWS):
        if not name.startswith("OneReason_"):
            continue
        result[name] = parquet_dataset_summary(O2 / name)
    return result


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(content_text(item) for item in value)
    if isinstance(value, dict):
        return content_text(value.get("text", value.get("content", "")))
    return "" if value is None else str(value)


def inspect_o3(o1_rec_hashes: list[str]) -> tuple[dict[str, Any], set[str]]:
    parquet = pq.ParquetFile(O3)
    rows = 0
    positions = 0
    caption_hits = 0
    tag_hits = 0
    length_mismatches = 0
    record_ids: list[int] = []
    message_matches = 0
    user_hashes: set[str] = set()
    index = 0
    for batch in parquet.iter_batches(
        columns=["record_id", "messages", "sid_token_list", "caption_list", "tag_list"],
        batch_size=128,
    ):
        for row in batch.to_pylist():
            rows += 1
            record_ids.append(row["record_id"])
            sid = row["sid_token_list"]
            caption = row["caption_list"]
            tag = row["tag_list"]
            positions += len(sid)
            caption_hits += sum(value is not None for value in caption)
            tag_hits += sum(value is not None for value in tag)
            if not (len(sid) == len(caption) == len(tag)):
                length_mismatches += 1
            messages = json.loads(row["messages"])
            flattened_messages = [
                {
                    "role": message.get("role"),
                    "content": content_text(message.get("content")),
                }
                for message in messages
            ]
            canonical = digest_text(
                json.dumps(
                    flattened_messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if index < len(o1_rec_hashes) and canonical == o1_rec_hashes[index]:
                message_matches += 1
            for message in messages:
                if message.get("role") == "user":
                    user_hashes.add(digest_text(normalize(content_text(message.get("content")))))
            index += 1
    return (
        {
            "rows": rows,
            "record_id_contiguous": record_ids == list(range(rows)),
            "sid_positions": positions,
            "caption_hits": caption_hits,
            "caption_rate": round(caption_hits / positions, 8),
            "tag_hits": tag_hits,
            "tag_rate": round(tag_hits / positions, 8),
            "list_length_mismatch_rows": length_mismatches,
            "o1_messages_equal_after_mode_strip": message_matches,
        },
        user_hashes,
    )


def inspect_o6() -> dict[str, Any]:
    config = json.loads((O6 / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (O6 / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer_json = json.loads(
        (O6 / "tokenizer.json").read_text(encoding="utf-8")
    )
    added = json.loads((O6 / "added_tokens.json").read_text(encoding="utf-8"))
    tokenizer_ids = list(tokenizer_json["model"]["vocab"].values()) + [
        token["id"] for token in tokenizer_json["added_tokens"]
    ]
    groups: dict[str, list[int]] = {key: [] for key in "abc"}
    for token, token_id in added.items():
        match = re.fullmatch(r"<s_([abc])_(\d+)>", token)
        if match:
            groups[match.group(1)].append(token_id)
    return {
        "architecture": config["architectures"][0],
        "config_vocab_size": config["vocab_size"],
        "tokenizer_base_vocab_size": len(tokenizer_json["model"]["vocab"]),
        "tokenizer_length_from_ids": max(tokenizer_ids) + 1,
        "added_tokens": len(added),
        "itemic_groups": {
            key: {
                "count": len(values),
                "id_min": min(values),
                "id_max": max(values),
                "contiguous": sorted(values) == list(range(min(values), max(values) + 1)),
            }
            for key, values in groups.items()
        },
        "model_max_position_embeddings": config["max_position_embeddings"],
        "tokenizer_model_max_length": tokenizer_config["model_max_length"],
        "tie_word_embeddings": config["tie_word_embeddings"],
        "generic_sid_begin_registered": "<|sid_begin|>" in added,
        "domain_token_ids": {
            token: added.get(token)
            for token in (
                "<|video_begin|>",
                "<|prod_begin|>",
                "<|living_begin|>",
                "<|ad_begin|>",
            )
        },
    }


def inspect_visible_world(o1_prompt_hashes: set[str], o3_user_hashes: set[str]) -> dict[str, Any]:
    path = EVALUATION / "visible" / "懂世界.jsonl"
    rows = []
    for line in path.open(encoding="utf-8"):
        value = json.loads(line)
        row = value[0] if isinstance(value, list) else value
        rows.append(row)
    valid = rows[:5]
    hashes = [digest_text(normalize(row.get("prompt", row.get("input", "")))) for row in valid]
    return {
        "registered_rows": len(rows),
        "valid_current_rows": len(valid),
        "exact_normalized_in_o1": sum(value in o1_prompt_hashes for value in hashes),
        "exact_normalized_in_o3": sum(value in o3_user_hashes for value in hashes),
        "restriction": "E is diagnostic-only and must never be used for training selection",
    }


def verify_expected(result: dict[str, Any]) -> None:
    assert result["O1"]["rows"] == 32_480
    assert result["O1"]["task_counts"] == {
        "rec_video": 14_868,
        "rec_living": 1_271,
        "rec_ad": 1_576,
        "rec_prod": 1_489,
        "material_desc2sid": 5_597,
        "material_sid2desc": 4_787,
        "user_action": 1_588,
        "user_topic": 1_304,
    }
    assert not result["O1"]["format_errors"]
    assert result["O1"]["exact_full"] == {
        "unique": 32_335,
        "extra_rows": 145,
        "duplicate_groups": 113,
        "max_group": 4,
    }
    assert result["O1"]["recommendation_prompt_groups"] == {
        "groups": 6_460,
        "duplicate_groups": 3_542,
        "rows_in_duplicate_groups": 16_286,
        "max_group": 23,
        "duplicate_groups_same_think": 3_542,
        "duplicate_groups_all_distinct_answers": 3_440,
    }
    assert result["O3"]["rows"] == EXPECTED_ROWS["O3"]
    assert result["O3"]["sid_positions"] == 3_539_794
    assert result["O3"]["caption_hits"] == 3_478_100
    assert result["O3"]["tag_hits"] == 1_356_390
    assert result["O3"]["list_length_mismatch_rows"] == 0
    assert result["O3"]["o1_messages_equal_after_mode_strip"] == 19_204
    for asset_name, summary in result["O2"].items():
        assert summary["rows"] == EXPECTED_ROWS[asset_name]
    assert result["O2"]["OneReason_UserProfile"]["column_count_file_counts"] == {
        "63": 10
    }
    assert result["O4"]["rows"] == EXPECTED_ROWS["O4"]
    assert result["O5"]["rows"] == EXPECTED_ROWS["O5"]
    assert result["O6"]["config_vocab_size"] == 176_253
    assert result["O6"]["tokenizer_base_vocab_size"] == 151_643
    assert result["O6"]["tokenizer_length_from_ids"] == 176_253
    assert all(group["count"] == 8_192 for group in result["O6"]["itemic_groups"].values())
    assert result["E_visible_world_overlap"]["exact_normalized_in_o1"] == 0
    assert result["E_visible_world_overlap"]["exact_normalized_in_o3"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenize-o1",
        action="store_true",
        help="Recompute expensive O1 O6-chat-template length and target-token statistics.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout.")
    args = parser.parse_args()

    o1_prompt_hashes = {
        digest_text(normalize(row["prompt"])) for _, _, row in iter_o1()
    }
    o1, o1_rec_hashes = inspect_o1(args.tokenize_o1)
    o3, o3_user_hashes = inspect_o3(o1_rec_hashes)
    result = {
        "snapshot": "official-data-eda-final-20260712",
        "scope": "registered O1-O6 only; read-only",
        "O1": o1,
        "O2": inspect_o2(),
        "O3": o3,
        "O4": parquet_dataset_summary(O4),
        "O5": parquet_dataset_summary(O5),
        "O6": inspect_o6(),
        "E_visible_world_overlap": inspect_visible_world(
            o1_prompt_hashes, o3_user_hashes
        ),
    }
    verify_expected(result)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output = args.output.resolve()
        if output.is_relative_to(OFFICIAL.resolve()):
            raise ValueError(f"refusing to write inside official assets: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
