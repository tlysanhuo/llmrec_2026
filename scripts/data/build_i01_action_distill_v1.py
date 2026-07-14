#!/usr/bin/env python3
"""Build the performance-first O1 I-01 + O2 action-distill SFT mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = ROOT / "assets/derived/processed/data_final.jsonl"
DEFAULT_ACTION = ROOT / "assets/derived/processed/action_distill_v5.jsonl"
DEFAULT_ACTION_AUDIT = ROOT / "logs/data/action_distill_v5.audit.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_i01_action_distill_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/i01_action_distill_v1_audit.json"
MODEL = ROOT / "models/OneReason-0.8B-pretrain-competition"
EVAL_SOURCES = (
    ROOT / "assets/derived/processed/r2_gold_v4.jsonl",
    ROOT / "assets/derived/processed/r2_gold_g1.jsonl",
    ROOT / "assets/derived/processed/r2_gold_g2.jsonl",
    ROOT / "assets/derived/processed/r2_gold_local.jsonl",
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
ITEM_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, list):
                if len(row) != 1:
                    raise ValueError(f"unexpected list row in {path}")
                row = row[0]
            rows.append(row)
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def answer_body(row: dict[str, Any]) -> str:
    if "</think>" not in row["output"]:
        raise ValueError("response is missing </think>")
    return row["output"].split("</think>", 1)[1].lstrip("\n")


def task_of(row: dict[str, Any]) -> str:
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


def core_prompt(row: dict[str, Any]) -> tuple[str, str]:
    return row["instruction"], MODE_SUFFIX_RE.sub("", row["input"].rstrip())


def to_nothink(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["input"] = MODE_SUFFIX_RE.sub("/no_think", row["input"].rstrip())
    converted["output"] = "<think>\n\n</think>\n" + answer_body(row)
    return converted


def apply_i01(rows: list[dict[str, Any]]) -> dict[str, int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if task_of(row).startswith("rec_"):
            groups[core_prompt(row)].append(index)

    converted = 0
    singleton_groups = 0
    for indexes in groups.values():
        if len(indexes) == 1:
            singleton_groups += 1
            continue
        think_bodies = []
        for index in indexes:
            match = THINK_RE.search(rows[index]["output"])
            if not match or not match.group(1).strip():
                raise AssertionError("O1 recommendation row unexpectedly lacks original CoT")
            think_bodies.append(match.group(1).strip())
        if len(set(think_bodies)) != 1:
            raise AssertionError("recommendation prompt group has non-identical duplicate CoTs")
        for index in indexes[1:]:
            rows[index] = to_nothink(rows[index])
            converted += 1

    return {
        "rec_prompt_groups": len(groups),
        "rec_singleton_groups": singleton_groups,
        "rec_duplicate_cot_converted": converted,
    }


def validate_action_row(row: dict[str, Any]) -> dict[str, int]:
    if not row["input"].rstrip().endswith("/no_think"):
        raise AssertionError("distilled action prompt is not /no_think")
    think_match = THINK_RE.search(row["output"])
    if not think_match or think_match.group(1).strip():
        raise AssertionError("distilled action response must have an empty think block")
    try:
        selected = json.loads(answer_body(row))
    except json.JSONDecodeError as error:
        raise AssertionError("distilled action response is not valid JSON") from error
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise AssertionError("distilled action response is not a string list")
    if not 5 <= len(selected) <= 18 or len(selected) != len(set(selected)):
        raise AssertionError("distilled action selection count/uniqueness is invalid")
    history_text = row["input"].split("\n\n角色任务", 1)[0]
    history_items = set(ITEM_RE.findall(history_text))
    if any(item not in history_items for item in selected):
        raise AssertionError("distilled action response contains a non-history item")
    theme_match = re.search(r"主题[:：]([^\n]+)", row["input"])
    if not theme_match or not 8 <= len(theme_match.group(1).strip()) <= 40:
        raise AssertionError("distilled action theme is absent or malformed")
    return {"selected": len(selected), "history": len(history_items)}


def accepted_source_indices(audit_path: Path) -> list[int]:
    accepted = []
    for row in read_jsonl(audit_path):
        if row.get("status") == "accepted":
            accepted.append(int(row["src_idx"]))
    return accepted


def eval_source_indices() -> set[int]:
    indices = set()
    for path in EVAL_SOURCES:
        for row in read_jsonl(path):
            indices.add(int(row["_src_idx"]))
    return indices


def token_mix(rows_with_source: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    by_task = Counter()
    by_source = Counter()
    total = 0
    for row, source in rows_with_source:
        count = len(tokenizer.encode(row["output"], add_special_tokens=False)) + 1
        total += count
        by_task[task_of(row)] += count
        by_source[source] += count
    return {
        "total_target_tokens_including_eos": total,
        "by_task": dict(sorted(by_task.items())),
        "by_task_ratio": {
            key: round(value / total, 8) for key, value in sorted(by_task.items())
        },
        "by_source": dict(sorted(by_source.items())),
        "by_source_ratio": {
            key: round(value / total, 8) for key, value in sorted(by_source.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-src", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--action-src", type=Path, default=DEFAULT_ACTION)
    parser.add_argument("--action-audit", type=Path, default=DEFAULT_ACTION_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--n-action", type=int, default=164)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19260817)
    args = parser.parse_args()

    seed_rows = [normalize_row(row) for row in read_jsonl(args.seed_src)]
    if len(seed_rows) != 32480:
        raise AssertionError(f"expected 32,480 O1 rows, got {len(seed_rows)}")
    i01 = apply_i01(seed_rows)
    if i01["rec_prompt_groups"] != 6460 or i01["rec_duplicate_cot_converted"] != 12744:
        raise AssertionError(f"I-01 signature drifted: {i01}")

    all_action_rows = [normalize_row(row) for row in read_jsonl(args.action_src)]
    if len(all_action_rows) < args.n_action:
        raise AssertionError(
            f"need {args.n_action} accepted action rows, only {len(all_action_rows)} available"
        )
    action_rows = all_action_rows[: args.n_action]
    action_stats = [validate_action_row(row) for row in action_rows]

    accepted_indices = accepted_source_indices(args.action_audit)
    if len(accepted_indices) < args.n_action:
        raise AssertionError("action audit has fewer accepted rows than the distilled asset")
    selected_indices = accepted_indices[: args.n_action]
    leaked = sorted(set(selected_indices) & eval_source_indices())
    if leaked:
        raise AssertionError(f"distilled action rows overlap E sources: {leaked[:20]}")
    if len(selected_indices) != len(set(selected_indices)):
        raise AssertionError("distilled action source indices are not unique")

    if args.action_repeat < 1:
        raise AssertionError("action-repeat must be at least one")
    rows_with_source = [(row, "O1_i01") for row in seed_rows]
    for _ in range(args.action_repeat):
        rows_with_source.extend((row, "O2_action_distill_v5") for row in action_rows)
    task_counts = Counter(task_of(row) for row, _ in rows_with_source)
    source_counts = Counter(source for _, source in rows_with_source)
    empty_think_counts = Counter(
        task_of(row)
        for row, _ in rows_with_source
        if (match := THINK_RE.search(row["output"])) and not match.group(1).strip()
    )
    target_mix = token_mix(rows_with_source)

    rng = random.Random(args.seed)
    rng.shuffle(rows_with_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row, _ in rows_with_source:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.out)

    total_rows = len(rows_with_source)
    audit = {
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "upstreams": {
            "O1_D_format": {
                "path": str(args.seed_src.resolve()),
                "rows": len(seed_rows),
                "sha256": sha256(args.seed_src),
            },
            "O2_action_distill_v5": {
                "path": str(args.action_src.resolve()),
                "available_rows": len(all_action_rows),
                "unique_rows": len(action_rows),
                "training_repeats_per_unique_row": args.action_repeat,
                "effective_training_rows": len(action_rows) * args.action_repeat,
                "sha256": sha256(args.action_src),
                "audit_log": str(args.action_audit.resolve()),
                "excluded_eval_source_indices": len(eval_source_indices()),
                "selected_eval_overlap": 0,
            },
        },
        "i01": i01,
        "rows": total_rows,
        "row_mix": {
            key: {"rows": value, "ratio": round(value / total_rows, 8)}
            for key, value in sorted(source_counts.items())
        },
        "task_counts": dict(sorted(task_counts.items())),
        "empty_think_counts": dict(sorted(empty_think_counts.items())),
        "action_qc": {
            "unique_rows": len(action_stats),
            "training_repeats_per_unique_row": args.action_repeat,
            "effective_training_rows": len(action_stats) * args.action_repeat,
            "selected_min": min(item["selected"] for item in action_stats),
            "selected_max": max(item["selected"] for item in action_stats),
            "selected_mean": round(
                sum(item["selected"] for item in action_stats) / len(action_stats), 4
            ),
            "history_unique_item_min": min(item["history"] for item in action_stats),
            "history_unique_item_max": max(item["history"] for item in action_stats),
        },
        "target_token_mix": target_mix,
        "output": str(args.out.resolve()),
        "output_sha256": sha256(args.out),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
