#!/usr/bin/env python3
"""Build the user-residual SFT and parent-retention mixture.

User rows keep their complete histories and original targets. Non-user rows are
used only by the custom trainer's parent-KL branch; their labels define the
teacher-forced answer positions and are not optimized with gold CE.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from build_seed_scoremax_v1 import (
    action_targets,
    answer_body,
    load_jsonl,
    parse_action_events,
    sha256,
    stable_hash,
    task_of,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
DEFAULT_TEACHER = ROOT / "assets/derived/processed/action_distill_v5.jsonl"
DEFAULT_WORLD = ROOT / "assets/derived/official_general/sft_world_knowledge.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/user_residual_retention_v1_audit.json"

RETENTION_COUNTS = {
    "material_desc2sid": 281,
    "material_sid2desc": 281,
    "rec_video": 565,
    "rec_prod": 565,
    "rec_ad": 565,
    "rec_living": 565,
}


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def load_world(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            normalized = normalize(row)
            if not normalized["input"] or not normalized["output"]:
                raise ValueError(f"empty world prompt/response at {path}:{line_number}")
            rows.append(normalized)
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["instruction"], row["input"], row["output"]


def summarize(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
    }


def topic_step_count(row: dict[str, Any]) -> int:
    payload = json.loads(answer_body(row))
    chain = payload.get("logic_chain") if isinstance(payload, dict) else None
    events = chain.get("events") if isinstance(chain, dict) else None
    if not isinstance(events, list):
        raise AssertionError("topic target lacks logic_chain.events")
    return len(events)


def select_retention(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {task: [] for task in RETENTION_COUNTS}
    for row in rows:
        task = task_of(row)
        if task in buckets:
            buckets[task].append(row)

    selected = []
    audit = {}
    for task, wanted in RETENTION_COUNTS.items():
        candidates = buckets[task]
        candidates.sort(
            key=lambda row: stable_hash(seed, "retention", task, *row_key(row))
        )
        if len(candidates) < wanted:
            raise AssertionError(f"only {len(candidates)} {task} rows, need {wanted}")
        chosen = candidates[:wanted]
        selected.extend(chosen)
        audit[task] = {
            "available": len(candidates),
            "selected": len(chosen),
            "selection": "stable_hash_without_replacement",
        }
    return selected, audit


def validate_user_rows(
    rows: list[dict[str, Any]], teacher_keys: set[tuple[str, str, str]]
) -> dict[str, Any]:
    task_counts = Counter()
    action_history_events = []
    action_target_counts = []
    action_chronological = 0
    topic_steps = []
    teacher_matches = 0

    for row in rows:
        task = task_of(row)
        if task not in {"action", "topic"}:
            raise AssertionError(f"non-user task in user branch: {task}")
        if not row["input"].rstrip().endswith("/no_think"):
            raise AssertionError(f"user row is not routed through /no_think: {task}")
        task_counts[task] += 1
        teacher_matches += row_key(row) in teacher_keys

        if task == "action":
            _, events, _ = parse_action_events(row["input"])
            targets = action_targets(row)
            action_history_events.append(len(events))
            action_target_counts.append(len(targets))
            cursor = 0
            ordered = True
            for target in targets:
                match = next(
                    (event.index for event in events[cursor:] if event.token == target),
                    None,
                )
                if match is None:
                    ordered = False
                    break
                cursor = match + 1
            action_chronological += ordered
        else:
            steps = topic_step_count(row)
            if not 1 <= steps <= 5:
                raise AssertionError(f"topic target has {steps} steps, expected 1..5")
            topic_steps.append(steps)

    if task_counts != {"action": 1752, "topic": 1301}:
        raise AssertionError(f"unexpected user task signature: {task_counts}")
    if teacher_matches != 164:
        raise AssertionError(f"expected 164 unique teacher rows once, got {teacher_matches}")

    return {
        "task_counts": dict(sorted(task_counts.items())),
        "teacher_unique_once": teacher_matches,
        "action": {
            "history_events": summarize(action_history_events),
            "selected_targets": summarize(action_target_counts),
            "chronological_rows": action_chronological,
            "non_chronological_or_repeated_alignment_rows": len(action_history_events)
            - action_chronological,
        },
        "topic_logic_chain_steps": summarize(topic_steps),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260821)
    args = parser.parse_args()

    source_rows = [normalize(row) for row in load_jsonl(args.source)]
    if len(source_rows) != 32_644:
        raise AssertionError(f"expected 32,644 source rows, got {len(source_rows)}")
    source_counts = Counter(task_of(row) for row in source_rows)

    teacher_rows = [normalize(row) for row in load_jsonl(args.teacher)]
    teacher_keys = {row_key(row) for row in teacher_rows}
    if len(teacher_rows) != 164 or len(teacher_keys) != 164:
        raise AssertionError("teacher source must contain 164 unique rows")

    excluded_topic_rows = [
        row for row in source_rows if task_of(row) == "topic" and topic_step_count(row) > 5
    ]
    if len(excluded_topic_rows) != 3:
        raise AssertionError(
            f"expected three topic rows above five steps, got {len(excluded_topic_rows)}"
        )
    user_rows = [
        row
        for row in source_rows
        if task_of(row) == "action"
        or (task_of(row) == "topic" and topic_step_count(row) <= 5)
    ]
    user_audit = validate_user_rows(user_rows, teacher_keys)

    retention_rows, retention_audit = select_retention(source_rows, args.seed)
    world_rows = load_world(args.world)
    if len(world_rows) != 231:
        raise AssertionError(f"expected 231 world rows, got {len(world_rows)}")
    retention_rows.extend(world_rows)
    if len(retention_rows) != len(user_rows):
        raise AssertionError(
            f"user/retention rows are not 1:1: {len(user_rows)}/{len(retention_rows)}"
        )

    final_rows = user_rows + retention_rows
    random.Random(args.seed).shuffle(final_rows)
    write_jsonl(args.out, final_rows)

    audit = {
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "upstream": {
            "I10_training_mix": {
                "path": str(args.source.resolve()),
                "rows": len(source_rows),
                "sha256": sha256(args.source),
                "task_counts": dict(sorted(source_counts.items())),
            },
            "O2_teacher_identity_check": {
                "path": str(args.teacher.resolve()),
                "rows": len(teacher_rows),
                "sha256": sha256(args.teacher),
            },
            "world_retention_D_O2_General": {
                "path": str(args.world.resolve()),
                "rows": len(world_rows),
                "sha256": sha256(args.world),
            },
        },
        "training_semantics": {
            "user_rows": "gold CE plus weak parent KL",
            "retention_rows": "parent KL only; response defines teacher-forced mask",
            "parent": "I-10 E3 adapter; no platform E data used",
        },
        "rows": len(final_rows),
        "row_mix": {
            "user_supervision": {
                "rows": len(user_rows),
                "ratio": round(len(user_rows) / len(final_rows), 8),
                "excluded_topic_rows_above_5_steps": len(excluded_topic_rows),
                **user_audit,
            },
            "parent_retention": {
                "rows": len(retention_rows),
                "ratio": round(len(retention_rows) / len(final_rows), 8),
                "source_selection": retention_audit,
                "world_rows": len(world_rows),
                "world_rows_without_close_think": sum(
                    "</think>" not in row["output"] for row in world_rows
                ),
            },
        },
        "forbidden_rows": {"O2_rule": 0, "T": 0, "E": 0},
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
