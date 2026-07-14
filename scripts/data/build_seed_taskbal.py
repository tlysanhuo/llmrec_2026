#!/usr/bin/env python3
"""Build the official-seed-only dataset for task-balanced LoRA SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT.parent / "ai_runtime" / "llmrec_2026"
DEFAULT_SRC = RUNTIME / "data" / "processed" / "data_final.jsonl"
DEFAULT_OUT = RUNTIME / "data" / "processed" / "data_seed_taskbal.jsonl"
DEFAULT_AUDIT = RUNTIME / "logs" / "data" / "seed_taskbal_audit.json"
MODEL = ROOT / "models" / "OneReason-0.8B-pretrain-competition"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")

ACTION_WEIGHT = 3.0
ACTION_TERMINAL_MULTIPLIER = 2.0
TOPIC_WEIGHT = 0.5
DESC2SID_ANSWER_WEIGHT = 4.0


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if isinstance(row, list):
                row = row[0]
            rows.append(
                {
                    "instruction": row.get("instruction", row.get("system", "")) or "",
                    "input": row.get("input", row.get("prompt", "")) or "",
                    "output": row.get("output", row.get("response", "")) or "",
                    "history": row.get("history") or [],
                }
            )
    return rows


def answer_body(row: dict) -> str:
    return row["output"].split("</think>", 1)[-1].lstrip("\n")


def task_of(row: dict) -> str:
    body = answer_body(row).strip()
    if body.startswith("["):
        return "action"
    if body.startswith("{") and "logic_chain" in body:
        return "topic"
    if "该用户最近" in body:
        match = DOMAIN_RE.search(body)
        return f"rec_{match.group(1) if match else 'unknown'}"
    input_has_sid = "<s_a_" in row["input"]
    output_has_sid = "<s_a_" in body
    if output_has_sid and not input_has_sid:
        return "material_desc2sid"
    if input_has_sid and not output_has_sid:
        return "material_sid2desc"
    raise ValueError(f"cannot classify row: {body[:120]!r}")


def core_prompt(row: dict) -> tuple[str, str]:
    return row["instruction"], MODE_SUFFIX_RE.sub("", row["input"].rstrip())


def to_nothink(row: dict) -> dict:
    converted = dict(row)
    converted["input"] = MODE_SUFFIX_RE.sub("/no_think", row["input"].rstrip())
    converted["output"] = "<think>\n\n</think>\n" + answer_body(row)
    return converted


def response_weight_stats(rows: list[dict]) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    close_think_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    domain_ids = {
        tokenizer.convert_tokens_to_ids(f"<|{domain}_begin|>")
        for domain in ("video", "prod", "ad", "living")
    }

    raw_tokens = 0
    weighted_tokens = 0.0
    per_task = defaultdict(lambda: {"tokens": 0, "weighted_tokens": 0.0})
    for row in rows:
        task = task_of(row)
        ids = tokenizer.encode(row["output"], add_special_tokens=False) + [eos_id]
        weights = [1.0] * len(ids)
        if task == "action":
            weights = [ACTION_WEIGHT] * len(ids)
            # The tokenizer merges the final quote and bracket into one token (e.g. `"]`).
            # Weight the final content token and EOS instead of matching a standalone `]`.
            weights[-2] *= ACTION_TERMINAL_MULTIPLIER
            weights[-1] *= ACTION_TERMINAL_MULTIPLIER
        elif task == "topic":
            weights = [TOPIC_WEIGHT] * len(ids)
        elif task == "material_desc2sid":
            try:
                body_start = ids.index(close_think_id) + 1
            except ValueError as error:
                raise AssertionError("desc2sid row is missing </think>") from error
            while body_start < len(ids) and ids[body_start] not in domain_ids:
                body_start += 1
            if body_start == len(ids):
                raise AssertionError("desc2sid row is missing domain token")
            for index in range(body_start, len(ids) - 1):
                weights[index] = DESC2SID_ANSWER_WEIGHT

        raw_tokens += len(ids)
        weighted_tokens += sum(weights)
        per_task[task]["tokens"] += len(ids)
        per_task[task]["weighted_tokens"] += sum(weights)

    return {
        "raw_target_tokens": raw_tokens,
        "weighted_target_tokens": round(weighted_tokens, 3),
        "loss_mean_weight": weighted_tokens / raw_tokens,
        "per_task": dict(sorted(per_task.items())),
    }


def file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260817)
    args = parser.parse_args()

    rows = load_jsonl(args.src)
    if len(rows) != 32480:
        raise AssertionError(f"expected 32480 official rows, got {len(rows)}")

    rec_groups = defaultdict(list)
    for index, row in enumerate(rows):
        if task_of(row).startswith("rec_"):
            rec_groups[core_prompt(row)].append(index)

    converted = 0
    for indexes in rec_groups.values():
        filled_thinks = {
            match.group(1).strip()
            for index in indexes
            if (match := THINK_RE.search(rows[index]["output"])) and match.group(1).strip()
        }
        if len(indexes) > 1 and len(filled_thinks) == 1:
            for index in indexes[1:]:
                rows[index] = to_nothink(rows[index])
                converted += 1

    task_counts = Counter(task_of(row) for row in rows)
    empty_think_counts = Counter(
        task_of(row)
        for row in rows
        if (match := THINK_RE.search(row["output"])) and not match.group(1).strip()
    )
    if task_counts["action"] != 1588 or task_counts["topic"] != 1304:
        raise AssertionError(f"unexpected user task counts: {task_counts}")
    if task_counts["material_desc2sid"] != 5597 or task_counts["material_sid2desc"] != 4787:
        raise AssertionError(f"unexpected material counts: {task_counts}")
    if sum(value for key, value in task_counts.items() if key.startswith("rec_")) != 19204:
        raise AssertionError(f"unexpected recommendation counts: {task_counts}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "source": str(args.src.resolve()),
        "output": str(args.out.resolve()),
        "rows": len(rows),
        "seed": args.seed,
        "rec_prompt_groups": len(rec_groups),
        "rec_duplicate_cot_converted": converted,
        "task_counts": dict(sorted(task_counts.items())),
        "empty_think_counts": dict(sorted(empty_think_counts.items())),
        "weights": {
            "action": ACTION_WEIGHT,
            "action_terminal_multiplier": ACTION_TERMINAL_MULTIPLIER,
            "topic": TOPIC_WEIGHT,
            "material_desc2sid_answer": DESC2SID_ANSWER_WEIGHT,
            "default": 1.0,
        },
        "token_weight_audit": response_weight_stats(rows),
        "md5": file_hash(args.out),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
