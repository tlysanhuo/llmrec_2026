#!/usr/bin/env python3
"""Compare rule-built action labels with accepted teacher labels from the same pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ITEM_RE = __import__("re").compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def targets(row: dict[str, Any]) -> list[str]:
    return json.loads(str(row["output"]).split("</think>", 1)[1].strip())


def metrics(rule: list[str], teacher: list[str]) -> dict[str, float | int | bool]:
    rule_set, teacher_set = set(rule), set(teacher)
    overlap = len(rule_set & teacher_set)
    precision = overlap / len(rule_set) if rule_set else float(not teacher_set)
    recall = overlap / len(teacher_set) if teacher_set else float(not rule_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "rule_targets": len(rule_set),
        "teacher_targets": len(teacher_set),
        "overlap": overlap,
        "exact_set": rule_set == teacher_set,
    }


def summarize(rows: list[dict[str, float | int | bool]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return round(statistics.fmean(float(row[key]) for row in rows), 6)

    return {
        "rows": len(rows),
        "mean_f1": mean("f1"),
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "zero_overlap": sum(int(row["overlap"]) == 0 for row in rows),
        "exact_set": sum(bool(row["exact_set"]) for row in rows),
        "teacher_items": sum(int(row["teacher_targets"]) for row in rows),
        "teacher_items_missing_from_rule": sum(
            int(row["teacher_targets"]) - int(row["overlap"]) for row in rows
        ),
        "rule_items": sum(int(row["rule_targets"]) for row in rows),
        "rule_items_rejected_by_teacher": sum(
            int(row["rule_targets"]) - int(row["overlap"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "assets/derived/processed/r2_base_v3.jsonl")
    parser.add_argument("--teacher", type=Path, default=ROOT / "assets/derived/processed/action_distill_v5.jsonl")
    parser.add_argument("--teacher-audit", type=Path, default=ROOT / "logs/data/action_distill_v5.audit.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/data/action_rule_vs_teacher_audit.json")
    args = parser.parse_args()

    source_rows = load_jsonl(args.source)
    teacher_rows = load_jsonl(args.teacher)
    accepted_indices = [
        int(row["src_idx"])
        for row in load_jsonl(args.teacher_audit)
        if row.get("status") == "accepted"
    ]
    if len(teacher_rows) != len(accepted_indices):
        raise AssertionError("teacher rows and accepted source indices are misaligned")

    all_rows = []
    filtered_rows = []
    for source_index, teacher_row in zip(accepted_indices, teacher_rows):
        source_row = source_rows[source_index]
        rule_targets = targets(source_row)
        teacher_targets = targets(teacher_row)
        result = metrics(rule_targets, teacher_targets)
        all_rows.append(result)

        history = ITEM_RE.findall(str(source_row["input"]).split("\n\n角色任务", 1)[0])
        unique_rule_targets = len(set(rule_targets))
        density = unique_rule_targets / len(history)
        if (
            80 <= len(history) <= 260
            and 4 <= unique_rule_targets <= 15
            and 0.02 <= density <= 0.12
        ):
            filtered_rows.append(result)

    report = {
        "protocol": "action-rule-label-gate-v1",
        "inputs": {
            "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source)},
            "teacher": {"path": str(args.teacher.resolve()), "sha256": sha256(args.teacher)},
            "teacher_audit": {
                "path": str(args.teacher_audit.resolve()),
                "sha256": sha256(args.teacher_audit),
            },
        },
        "all_accepted_teacher_rows": summarize(all_rows),
        "rows_matching_i09_rule_filter": summarize(filtered_rows),
    }
    filtered = report["rows_matching_i09_rule_filter"]
    report["gate"] = {
        "thresholds": {
            "mean_f1_min": 0.60,
            "mean_recall_min": 0.60,
            "zero_overlap_rate_max": 0.05,
        },
        "pass": (
            filtered["mean_f1"] >= 0.60
            and filtered["mean_recall"] >= 0.60
            and filtered["zero_overlap"] / max(filtered["rows"], 1) <= 0.05
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
