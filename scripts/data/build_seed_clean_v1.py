#!/usr/bin/env python3
"""Build the O1-only clean training set used by the single-adapter r80 run."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from build_seed_scoremax_v1 import (
    compress_recommendation_cot,
    load_jsonl,
    sha256,
    target_token_mix,
    task_of,
    to_nothink,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_O1 = ROOT / "assets/derived/processed/data_final.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_seed_clean_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/seed_clean_v1_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o1", type=Path, default=DEFAULT_O1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260817)
    args = parser.parse_args()

    rows = load_jsonl(args.o1)
    if len(rows) != 32_480:
        raise AssertionError(f"expected 32,480 O1 rows, got {len(rows)}")

    source_counts = Counter(task_of(row) for row in rows)
    source_outputs = [row["output"] for row in rows]
    rec_stats = compress_recommendation_cot(rows, args.seed)

    topic_converted = 0
    for index, row in enumerate(rows):
        if task_of(row) == "topic" and not row["input"].rstrip().endswith("/no_think"):
            rows[index] = to_nothink(row)
            topic_converted += 1

    if rec_stats["prompt_groups"] != 6_460:
        raise AssertionError(f"recommendation prompt groups drifted: {rec_stats}")
    if rec_stats["duplicate_cot_converted"] != 12_744:
        raise AssertionError(f"recommendation conversion drifted: {rec_stats}")
    if topic_converted != 602:
        raise AssertionError(f"topic conversion drifted: {topic_converted}")
    if source_counts["action"] != 1_588 or source_counts["topic"] != 1_304:
        raise AssertionError(f"O1 user-task signature drifted: {source_counts}")

    final_counts = Counter(task_of(row) for row in rows)
    if final_counts != source_counts:
        raise AssertionError("task counts changed during format-only transformations")

    # Both transformations must preserve the supervised answer after </think>.
    for index, (before, row) in enumerate(zip(source_outputs, rows)):
        if before.split("</think>", 1)[1] != row["output"].split("</think>", 1)[1]:
            raise AssertionError(f"target changed at row {index}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.out)

    audit = {
        "asset_class": "D(O1)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "upstream": {
            "asset_id": "O1",
            "path": str(args.o1.resolve()),
            "rows": 32_480,
            "sha256": sha256(args.o1),
        },
        "transformations": {
            "recommendation": rec_stats,
            "topic_to_nothink": topic_converted,
            "targets_preserved": 32_480,
        },
        "rows": 32_480,
        "row_mix": {"O1": {"rows": 32_480, "ratio": 1.0}},
        "task_counts": dict(sorted(final_counts.items())),
        "target_token_mix": target_token_mix(rows),
        "forbidden_training_rows": {"O2": 0, "T": 0, "E": 0},
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
