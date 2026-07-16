#!/usr/bin/env python3
"""Build the I-20 product/ad positive-only and parent-retention mixture.

The supervised branch contains only O1 recommendation positives.  Every
product/ad target is represented once with its original thinking trajectory
and once through the official empty-think `/no_think` route.  The retention
branch is consumed as parent-KL-only data by the I-20 custom trainer; its
response text supplies teacher-forced positions but is never treated as gold
CE.  No preference negatives or platform-evaluation rows are used.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_seed_scoremax_v1 import (
    answer_body,
    core_prompt,
    load_jsonl,
    sha256,
    stable_hash,
    task_of,
    to_nothink,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_O1 = ROOT / "assets/derived/processed/data_final.jsonl"
DEFAULT_RETENTION = ROOT / "assets/derived/processed/data_seed_clean_v1.jsonl"
DEFAULT_WORLD = ROOT / "assets/derived/official_general/sft_world_knowledge.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_i20_prod_ad_positive_retkl_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/i20_prod_ad_positive_retkl_v1_audit.json"

POSITIVE_TASKS = {"rec_prod", "rec_ad"}
RETENTION_COUNTS = {
    "rec_video": 2_099,
    "rec_living": 800,
    "material_desc2sid": 1_000,
    "material_sid2desc": 1_000,
    "action": 500,
    "topic": 500,
}
EXPECTED_O1_COUNTS = {
    "rec_prod": 1_489,
    "rec_ad": 1_576,
}


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def load_world(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            row = normalize(raw)
            if not row["input"] or not row["output"]:
                raise ValueError(f"empty world prompt/response at {path}:{line_number}")
            rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["instruction"], row["input"], row["output"]


def build_positive_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [row for row in rows if task_of(row) in POSITIVE_TASKS]
    counts = Counter(task_of(row) for row in source)
    if dict(counts) != EXPECTED_O1_COUNTS:
        raise AssertionError(f"O1 prod/ad signature drifted: {counts}")

    paired: list[dict[str, Any]] = []
    paired_counts: Counter[str] = Counter()
    for row in source:
        task = task_of(row)
        original = dict(row)
        no_think = to_nothink(row)
        if answer_body(original) != answer_body(no_think):
            raise AssertionError("think/no-think pairing changed the final answer")
        if not no_think["input"].rstrip().endswith("/no_think"):
            raise AssertionError("no-think positive is not routed through /no_think")
        paired.extend((original, no_think))
        paired_counts[f"{task}_think"] += 1
        paired_counts[f"{task}_nothink"] += 1

    return paired, {
        "source_rows": len(source),
        "paired_rows": len(paired),
        "source_task_counts": dict(sorted(counts.items())),
        "paired_mode_counts": dict(sorted(paired_counts.items())),
        "answer_changes": 0,
        "preference_negative_rows": 0,
    }


def select_grouped_recommendation(
    rows: list[dict[str, Any]], task: str, wanted: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if task_of(row) == task:
            groups[core_prompt(row)].append(row)
    if len(groups) < wanted:
        raise AssertionError(f"only {len(groups)} prompt groups for {task}, need {wanted}")

    ordered_groups = sorted(
        groups.items(), key=lambda item: stable_hash(seed, "retention-group", task, *item[0])
    )
    selected: list[dict[str, Any]] = []
    for key, candidates in ordered_groups[:wanted]:
        representative = min(
            candidates,
            key=lambda row: stable_hash(seed, "retention-target", task, key, *row_key(row)),
        )
        selected.append(to_nothink(representative))
    return selected


def select_flat(
    rows: list[dict[str, Any]], task: str, wanted: int, seed: int
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if task_of(row) == task]
    candidates.sort(key=lambda row: stable_hash(seed, "retention", task, *row_key(row)))
    if len(candidates) < wanted:
        raise AssertionError(f"only {len(candidates)} rows for {task}, need {wanted}")
    return [to_nothink(row) for row in candidates[:wanted]]


def build_retention_rows(
    rows: list[dict[str, Any]], world_rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for task, wanted in RETENTION_COUNTS.items():
        if task.startswith("rec_"):
            chosen = select_grouped_recommendation(rows, task, wanted, seed)
            selection = "stable_prompt_group_without_replacement; one target; /no_think view"
        else:
            chosen = select_flat(rows, task, wanted, seed)
            selection = "stable_row_without_replacement; /no_think view"
        selected.extend(chosen)
        audit[task] = {"selected": len(chosen), "selection": selection}

    if len(world_rows) != 231:
        raise AssertionError(f"expected 231 world rows, got {len(world_rows)}")
    selected.extend(world_rows)
    audit["world"] = {
        "selected": len(world_rows),
        "selection": "all registered D(O2.General) world-retention rows",
    }
    return selected, audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o1", type=Path, default=DEFAULT_O1)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260824)
    args = parser.parse_args()

    o1_rows = [normalize(row) for row in load_jsonl(args.o1)]
    retention_source = [normalize(row) for row in load_jsonl(args.retention)]
    world_rows = load_world(args.world)
    if len(o1_rows) != 32_480 or len(retention_source) != 32_480:
        raise AssertionError(
            f"expected two 32,480-row O1-derived sources, got "
            f"{len(o1_rows)}/{len(retention_source)}"
        )

    positive_rows, positive_audit = build_positive_rows(o1_rows)
    retention_rows, retention_audit = build_retention_rows(
        retention_source, world_rows, args.seed
    )
    if len(positive_rows) != 6_130 or len(retention_rows) != 6_130:
        raise AssertionError(
            f"positive/retention mixture is not 1:1: "
            f"{len(positive_rows)}/{len(retention_rows)}"
        )

    final_rows = positive_rows + retention_rows
    random.Random(args.seed).shuffle(final_rows)
    write_jsonl(args.out, final_rows)

    audit = {
        "asset_class": "D(O1,O2.General)",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "upstream": {
            "O1_format_conversion": {
                "asset_id": "D(O1):data_final",
                "path": str(args.o1.resolve()),
                "rows": len(o1_rows),
                "sha256": sha256(args.o1),
            },
            "O1_clean_views": {
                "asset_id": "D(O1):data_seed_clean_v1",
                "path": str(args.retention.resolve()),
                "rows": len(retention_source),
                "sha256": sha256(args.retention),
            },
            "world_retention": {
                "asset_id": "D(O2.General):sft_world_knowledge",
                "path": str(args.world.resolve()),
                "rows": len(world_rows),
                "sha256": sha256(args.world),
            },
        },
        "training_semantics": {
            "positive_rows": "final prod/ad domain+SID token CE plus weak frozen-I13 KL",
            "retention_rows": "frozen-I13 KL only; no gold CE",
            "parent": "I-13 e3_userres_r80_retkl_v3_s875",
        },
        "rows": len(final_rows),
        "row_mix": {
            "prod_ad_positive": {
                "rows": len(positive_rows),
                "ratio": 0.5,
                **positive_audit,
            },
            "parent_retention": {
                "rows": len(retention_rows),
                "ratio": 0.5,
                "task_counts": retention_audit,
            },
        },
        "forbidden_training_rows": {
            "preference_negative": 0,
            "O2_userprofile_pseudo_label": 0,
            "teacher_or_model_rollout": 0,
            "T": 0,
            "E": 0,
        },
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
