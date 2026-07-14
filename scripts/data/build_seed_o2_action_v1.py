#!/usr/bin/env python3
"""Build a low-dose O1 + O2 action-select SFT mixture.

All O1 rows and targets are preserved. O2 contributes unique, full-history
action examples only: accepted teacher rows once each and a filtered subset of
the registered rule-built source. Rule targets are reordered chronologically;
teacher-only Caption/Tag annotations never enter the output dataset.
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
    align_target_events,
    compress_recommendation_cot,
    load_jsonl,
    parse_action_events,
    sha256,
    stable_hash,
    target_token_mix,
    task_of,
    to_nothink,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_O1 = ROOT / "assets/derived/processed/data_final.jsonl"
DEFAULT_R2 = ROOT / "assets/derived/processed/r2_base_v3.jsonl"
DEFAULT_TEACHER = ROOT / "assets/derived/processed/action_distill_v5.jsonl"
DEFAULT_TEACHER_AUDIT = ROOT / "logs/data/action_distill_v5.audit.jsonl"
DEFAULT_EXCLUDES = (
    ROOT / "assets/derived/processed/r2_gold_v4.jsonl",
    ROOT / "assets/derived/processed/r2_gold_g1.jsonl",
    ROOT / "assets/derived/processed/r2_gold_g2.jsonl",
    ROOT / "assets/derived/processed/r2_gold_local.jsonl",
)
DEFAULT_OUT = ROOT / "assets/derived/processed/data_seed_o2_action_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/seed_o2_action_v1_audit.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction") or ""),
        "input": str(row.get("input") or ""),
        "output": str(row.get("output") or ""),
        "history": row.get("history") or [],
    }


def history_text(prompt: str) -> str:
    boundary = "\n\n角色任务"
    if boundary not in prompt:
        raise ValueError("action prompt lacks role-task boundary")
    return prompt.split(boundary, 1)[0]


def load_excluded_indices(paths: tuple[Path, ...]) -> tuple[set[int], dict[str, Any]]:
    excluded: set[int] = set()
    sources = []
    for path in paths:
        rows = read_jsonl(path)
        indices = {int(row["_src_idx"]) for row in rows if "_src_idx" in row}
        if len(indices) != len(rows):
            raise AssertionError(f"missing or duplicate _src_idx in {path}")
        excluded.update(indices)
        sources.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "unique_indices": len(indices),
                "sha256": sha256(path),
            }
        )
    return excluded, {"sources": sources, "union_indices": len(excluded)}


def load_accepted_indices(path: Path) -> tuple[list[int], dict[str, Any]]:
    accepted = [
        int(row["src_idx"])
        for row in read_jsonl(path)
        if row.get("status") == "accepted"
    ]
    if len(accepted) != len(set(accepted)):
        raise AssertionError("teacher audit contains duplicate accepted src_idx values")
    return accepted, {
        "path": str(path.resolve()),
        "accepted_indices": len(accepted),
        "sha256": sha256(path),
    }


def summarize(values: list[int]) -> dict[str, float | int]:
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


def validate_teacher_rows(
    rows: list[dict[str, Any]],
    accepted_indices: list[int],
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(accepted_indices):
        raise AssertionError(
            f"teacher rows/audit accepted mismatch: {len(rows)}/{len(accepted_indices)}"
        )

    selected_lengths = []
    history_lengths = []
    normalized = []
    for row, source_index in zip(rows, accepted_indices):
        clean = training_row(row)
        if task_of(clean) != "action":
            raise AssertionError("teacher row is not action-select")
        if not clean["input"].rstrip().endswith("/no_think"):
            raise AssertionError("teacher row is not routed through /no_think")
        if history_text(clean["input"]) != history_text(str(source_rows[source_index]["input"])):
            raise AssertionError(f"teacher/source history mismatch at src_idx={source_index}")
        _, events, _ = parse_action_events(clean["input"])
        targets = action_targets(clean)
        if align_target_events(events, targets) is None:
            raise AssertionError(f"teacher target order is not chronological at src_idx={source_index}")
        selected_lengths.append(len(targets))
        history_lengths.append(len(events))
        normalized.append(clean)

    return normalized, {
        "rows": len(normalized),
        "history_events": summarize(history_lengths),
        "selected_targets": summarize(selected_lengths),
    }


def chronological_rule_row(
    row: dict[str, Any],
) -> tuple[dict[str, Any], int, int, bool] | None:
    clean = training_row(row)
    if not clean["input"].rstrip().endswith("/no_think"):
        return None
    try:
        _, events, _ = parse_action_events(clean["input"])
        raw_targets = action_targets(clean)
    except (ValueError, json.JSONDecodeError):
        return None

    target_set = set(raw_targets)
    ordered_targets = []
    seen = set()
    for event in events:
        if event.token in target_set and event.token not in seen:
            seen.add(event.token)
            ordered_targets.append(event.token)
    if seen != target_set:
        return None

    clean["output"] = (
        "<think>\n\n</think>\n"
        + json.dumps(ordered_targets, ensure_ascii=False, separators=(",", ":"))
    )
    return clean, len(events), len(ordered_targets), ordered_targets != raw_targets


def select_rule_rows(
    source_rows: list[dict[str, Any]],
    excluded: set[int],
    accepted: set[int],
    wanted: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[int]]:
    rejection_counts = Counter()
    candidates = []
    for source_index, source_row in enumerate(source_rows):
        if source_index in excluded:
            rejection_counts["registered_eval_index"] += 1
            continue
        if source_index in accepted:
            rejection_counts["teacher_accepted_index"] += 1
            continue
        parsed = chronological_rule_row(source_row)
        if parsed is None:
            rejection_counts["empty_or_invalid"] += 1
            continue
        row, n_history, n_selected, reordered = parsed
        if not 80 <= n_history <= 260:
            rejection_counts["history_outside_80_260"] += 1
            continue
        if not 4 <= n_selected <= 15:
            rejection_counts["target_count_outside_4_15"] += 1
            continue
        density = n_selected / n_history
        if not 0.02 <= density <= 0.12:
            rejection_counts["density_outside_0.02_0.12"] += 1
            continue

        rank = (
            abs(n_selected - 8),
            abs(density - 0.0694),
            abs(n_history - 198),
            stable_hash(seed, source_index, row["input"], row["output"]),
        )
        candidates.append((rank, source_index, row, n_history, n_selected, reordered))

    candidates.sort(key=lambda item: item[0])
    if len(candidates) < wanted:
        raise AssertionError(f"only {len(candidates)} eligible O2 rule rows, need {wanted}")
    chosen = candidates[:wanted]
    rows = [item[2] for item in chosen]
    indices = [item[1] for item in chosen]
    history_lengths = [item[3] for item in chosen]
    selected_lengths = [item[4] for item in chosen]
    densities = [round(item[4] / item[3] * 10_000) for item in chosen]
    return rows, {
        "requested_rows": wanted,
        "eligible_before_rank_cut": len(candidates),
        "selected_rows": len(rows),
        "selection_rank": (
            "minimize |targets-8|, |density-0.0694|, |history-198|, then stable hash"
        ),
        "history_events": summarize(history_lengths),
        "selected_targets": summarize(selected_lengths),
        "positive_density_basis_points": summarize(densities),
        "chronologically_reordered": sum(bool(item[5]) for item in chosen),
        "rejections": dict(sorted(rejection_counts.items())),
    }, indices


def assert_unique_prompts(rows: list[dict[str, Any]], label: str) -> None:
    prompts = [row["input"] for row in rows]
    if len(prompts) != len(set(prompts)):
        raise AssertionError(f"{label} contains duplicate prompts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o1", type=Path, default=DEFAULT_O1)
    parser.add_argument("--r2", type=Path, default=DEFAULT_R2)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-audit", type=Path, default=DEFAULT_TEACHER_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--rule-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=19260817)
    args = parser.parse_args()
    if args.rule_rows < 0:
        parser.error("--rule-rows must be non-negative")

    o1_rows = load_jsonl(args.o1)
    if len(o1_rows) != 32_480:
        raise AssertionError(f"expected 32,480 O1 rows, got {len(o1_rows)}")
    o1_source_counts = Counter(task_of(row) for row in o1_rows)
    rec_stats = compress_recommendation_cot(o1_rows, args.seed)
    topic_converted = 0
    for index, row in enumerate(o1_rows):
        if task_of(row) == "topic" and not row["input"].rstrip().endswith("/no_think"):
            o1_rows[index] = to_nothink(row)
            topic_converted += 1

    r2_rows = read_jsonl(args.r2)
    if len(r2_rows) != 4_200:
        raise AssertionError(f"expected 4,200 r2_base_v3 rows, got {len(r2_rows)}")
    excluded, exclusion_audit = load_excluded_indices(DEFAULT_EXCLUDES)
    accepted_indices, teacher_audit = load_accepted_indices(args.teacher_audit)
    if len(excluded) != 354:
        raise AssertionError(f"expected 354 excluded eval indices, got {len(excluded)}")
    if len(accepted_indices) != 164:
        raise AssertionError(f"expected 164 accepted teacher indices, got {len(accepted_indices)}")
    if excluded.intersection(accepted_indices):
        raise AssertionError("teacher accepted indices overlap registered eval indices")

    teacher_rows, teacher_stats = validate_teacher_rows(
        read_jsonl(args.teacher), accepted_indices, r2_rows
    )
    rule_rows, rule_stats, selected_rule_indices = select_rule_rows(
        r2_rows, excluded, set(accepted_indices), args.rule_rows, args.seed
    )
    if excluded.intersection(selected_rule_indices):
        raise AssertionError("selected rule rows overlap registered eval indices")
    if set(accepted_indices).intersection(selected_rule_indices):
        raise AssertionError("selected rule rows overlap teacher rows")
    assert_unique_prompts(teacher_rows, "teacher rows")
    assert_unique_prompts(rule_rows, "rule rows")
    if set(row["input"] for row in teacher_rows).intersection(row["input"] for row in rule_rows):
        raise AssertionError("teacher and rule prompts overlap")

    o2_rows = teacher_rows + rule_rows
    final_rows = o1_rows + o2_rows
    rng = random.Random(args.seed)
    rng.shuffle(final_rows)
    final_counts = Counter(task_of(row) for row in final_rows)

    if rec_stats["prompt_groups"] != 6_460 or rec_stats["duplicate_cot_converted"] != 12_744:
        raise AssertionError(f"recommendation grouping signature drifted: {rec_stats}")
    if o1_source_counts["action"] != 1_588 or o1_source_counts["topic"] != 1_304:
        raise AssertionError(f"O1 user-task signature drifted: {o1_source_counts}")
    if topic_converted != 602:
        raise AssertionError(f"topic think signature drifted: {topic_converted}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in final_rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.out)

    output_sha256 = sha256(args.out)
    audit = {
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "upstream": {
            "O1": {
                "path": str(args.o1.resolve()),
                "rows": len(o1_rows),
                "sha256": sha256(args.o1),
            },
            "O2_rule_source": {
                "path": str(args.r2.resolve()),
                "rows": len(r2_rows),
                "sha256": sha256(args.r2),
            },
            "O2_teacher": {
                "path": str(args.teacher.resolve()),
                "rows": len(teacher_rows),
                "sha256": sha256(args.teacher),
                "audit": teacher_audit,
            },
            "E_exclusions_not_training_data": exclusion_audit,
        },
        "transformations": {
            "O1_recommendation": rec_stats,
            "O1_topic_to_nothink": topic_converted,
            "O2_teacher_validation": teacher_stats,
            "O2_rule_filter_and_reorder": rule_stats,
            "O2_annotations_removed": True,
        },
        "rows": len(final_rows),
        "row_mix": {
            "O1_preserved": {
                "rows": len(o1_rows),
                "ratio": round(len(o1_rows) / len(final_rows), 8),
            },
            "O2_teacher_unique_once": {
                "rows": len(teacher_rows),
                "ratio": round(len(teacher_rows) / len(final_rows), 8),
            },
            "O2_rule_unique": {
                "rows": len(rule_rows),
                "ratio": round(len(rule_rows) / len(final_rows), 8),
            },
        },
        "source_task_counts": dict(sorted(o1_source_counts.items())),
        "final_task_counts": dict(sorted(final_counts.items())),
        "target_token_mix": target_token_mix(final_rows),
        "leakage_gate": {
            "registered_eval_source_indices": len(excluded),
            "selected_overlap": len(
                excluded.intersection(accepted_indices + selected_rule_indices)
            ),
        },
        "output": str(args.out.resolve()),
        "output_sha256": output_sha256,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
