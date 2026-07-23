#!/usr/bin/env python3
"""Build a prompt-disjoint eight-task retention gate for General replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_s800_native_general_replay_v1 as replay
from build_seed_scoremax_v1 import task_of


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
TRAIN = ROOT / "assets/derived/processed/data_s800_native_general_replay_v1.jsonl"
TOKENIZER = ROOT / "assets/official/base_model"
OUT = (
    ROOT
    / "assets/evaluation/holdout/s800_native_general_replay_retention_gate_v1.jsonl"
)
AUDIT = ROOT / "logs/data/s800_native_general_replay_retention_gate_v1_audit.json"

SOURCE_SHA256 = replay.RETENTION_SHA256
TRAIN_SHA256 = "87097135eb7ddb866b78ae6427c24b8cc2712f898c892d7da92d58ff7e9fddd2"
SEED = 19_260_831
SCHEMA = "s800-native-general-replay-retention-gate-v1"
QUOTAS = {task: 32 for task in replay.RETENTION_QUOTAS}
EXPECTED_ROWS = sum(QUOTAS.values())


def build(source: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_prompt_keys = {
        (str(row.get("task")), replay.normalized_prompt_key(row))
        for row in train_rows
        if row.get("route") == replay.RETENTION_ROUTE
    }
    if len(train_prompt_keys) != replay.EXPECTED_RETENTION_ROWS:
        raise AssertionError("formal retention prompts are not 384 unique groups")

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: defaultdict(list) for task in QUOTAS
    }
    excluded_train: Counter[str] = Counter()
    for row in source:
        try:
            task = task_of(row)
        except ValueError:
            continue
        if task not in groups or not replay.valid_response_structure(row["output"]):
            continue
        prompt_key = replay.normalized_prompt_key(row)
        if (task, prompt_key) in train_prompt_keys:
            excluded_train[task] += 1
            continue
        groups[task][prompt_key].append(row)

    rows: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    for task, quota in QUOTAS.items():
        available[task] = len(groups[task])
        ordered = sorted(
            groups[task],
            key=lambda prompt_key: replay.stable_hash(
                SEED, "s800-general-retention-gate-v1", task, prompt_key
            ),
        )
        if len(ordered) < quota:
            raise AssertionError(f"only {len(ordered)} gate groups for {task}; need {quota}")
        for prompt_key in ordered[:quota]:
            row = min(
                groups[task][prompt_key],
                key=lambda value: replay.stable_hash(
                    SEED, "s800-general-retention-gate-row-v1", task, replay.canonical_json(value)
                ),
            )
            rows.append(
                {
                    "schema_version": SCHEMA,
                    "route": "gate_only",
                    "task": task,
                    "record_id": replay.stable_hash(SCHEMA, task, prompt_key),
                    "upstream_ids": ["data_user_residual_retention_v1"],
                    **row,
                }
            )

    rows.sort(key=lambda row: replay.stable_hash(SEED, "gate-order", row["record_id"]))
    counts = Counter(row["task"] for row in rows)
    if len(rows) != EXPECTED_ROWS or counts != QUOTAS:
        raise AssertionError(f"gate quota drifted: {len(rows)}/{dict(counts)}")
    prompt_keys = {(row["task"], replay.normalized_prompt_key(row)) for row in rows}
    if len(prompt_keys) != EXPECTED_ROWS or prompt_keys & train_prompt_keys:
        raise AssertionError("retention train/gate prompt split is not disjoint")
    return rows, {
        "rows": len(rows),
        "task_counts": dict(sorted(counts.items())),
        "available_prompt_groups_after_train_exclusion": dict(sorted(available.items())),
        "formal_train_prompt_groups_excluded": dict(sorted(excluded_train.items())),
        "train_gate_prompt_overlap": 0,
        "teacher_core_field_changes": 0,
        "allowed_role": "checkpoint preservation gate only; forbidden from CE and KL training",
    }


def token_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.eos_token != "<|im_end|>":
        tokenizer.eos_token = "<|im_end|>"
    lengths = [
        len(tokenizer.encode(replay.qwen3_formatted_text(row, tokenizer), add_special_tokens=False))
        for row in rows
    ]
    overflow = sum(length > replay.CUTOFF_LEN for length in lengths)
    if overflow:
        raise AssertionError(f"{overflow} retention gate rows exceed cutoff")
    return {
        "status": "PASS",
        "template": "qwen3_nothink",
        "cutoff_len": replay.CUTOFF_LEN,
        "rows_checked": len(rows),
        "overflow_rows": overflow,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not replay.PERSONAL_ROOT.is_mount():
        raise RuntimeError(f"personal volume is not mounted: {replay.PERSONAL_ROOT}")
    if not args.overwrite and (args.out.exists() or args.audit.exists()):
        raise FileExistsError("refusing to overwrite retention gate assets")
    if replay.sha256_file(SOURCE) != SOURCE_SHA256:
        raise AssertionError("retention source hash drifted")
    if replay.sha256_file(TRAIN) != TRAIN_SHA256:
        raise AssertionError("formal replay data hash drifted")

    source = replay.load_jsonl(SOURCE)
    with TRAIN.open(encoding="utf-8") as handle:
        train_rows = [json.loads(line) for line in handle if line.strip()]
    rows, split_audit = build(source, train_rows)
    tokens = token_audit(rows)
    replay.atomic_jsonl(args.out, rows)
    audit = {
        "asset_class": "D-holdout(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": SCHEMA,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": replay.sha256_file(Path(__file__)),
        "seed": SEED,
        "upstream": {
            "parent_distribution": {
                "path": str(SOURCE.resolve()),
                "rows": len(source),
                "sha256": SOURCE_SHA256,
            },
            "formal_replay_train": {
                "path": str(TRAIN.resolve()),
                "rows": len(train_rows),
                "sha256": TRAIN_SHA256,
            },
        },
        "split": split_audit,
        "token_audit": tokens,
        "forbidden_role": "never training data and never online-score estimate",
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(rows),
            "bytes": args.out.stat().st_size,
            "sha256": replay.sha256_file(args.out),
        },
    }
    replay.atomic_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
