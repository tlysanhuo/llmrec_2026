#!/usr/bin/env python3
"""Build the I-19 world-residual SFT and parent-retention mixture.

This is the world-task counterpart of `build_user_residual_retention_v1.py`
(I-13's stage-two data builder). I-13's own report
(`docs/I13_SOTA技术报告.md`, section 1.2 / 7 and section 3.3) documents that
I-19 A1 (`train_i19_world_retkl.py`) never applied a retention KL constraint
on the seven non-world tasks' real input distribution -- it only computed
`CE(gold) + 0.5*KL(parent)` on the world rows themselves. That omission is the
same failure mode I-13 already avoided for the user-residual case, and it is
the documented root cause of I-19 A1's severe collateral damage (material
-0.0613, video -0.0288, total -0.0757 relative to I-13).

This builder mirrors I-13's exact recipe for the world task instead:

- World supervision branch: the full, already-frozen, already
  contamination-audited 1,573-row Frinkleko world bucket
  (`data_i19_frinkleko_world_1578_clean.jsonl` -- the `_clean` sibling with
  the 5 rows that leaked into `eval/data/competition_smoke.jsonl` already
  removed; see that release's `contamination_audit_competition_smoke.json`).
  Rows are copied byte-for-byte; their labels receive real gradient (gold CE).
- Retention branch: exactly 1,573 rows stratified across the eight non-world
  task buckets already present in `data_seed_teacher_v1.jsonl` (the same
  32,644-row I-13 parent training mixture): action, topic,
  material_desc2sid, material_sid2desc, rec_video, rec_prod, rec_ad,
  rec_living. These rows' labels are never optimized with gold CE; the
  custom trainer (`train_world_residual_retkl.py`) uses them only to locate
  the teacher-forced response span for a forward-KL-to-parent loss.

The two branches are exactly 1:1 in row count, matching I-13's stage-two
design principle (`RETENTION_COUNTS` sums to the user-branch row count in
`build_user_residual_retention_v1.py`). Selection within each retention
bucket is a deterministic, seeded, without-replacement stable-hash sort
(`stable_hash`), reusing the same primitive I-13 already uses, so the choice
is reproducible and auditable rather than relying on Python's default RNG
ordering.

Every row (both branches) is stamped with a disjoint, exact instruction
prefix (borrowing the sentinel-prefix routing design already used by
`build_i20_world_mopd_routing.py`) so the trainer never has to guess a row's
route by sniffing JSON/list content -- world rows are multiple-choice text
answers, not JSON, so I-13's original "first output token" heuristic does
not apply cleanly here. The trainer (`train_world_residual_retkl.py`)
verifies the sentinel by decoding a small window of leading prompt tokens
back to text (not by matching encoded token ids, which is unsound across a
BPE tokenizer's context-dependent merges) and stripping the qwen3 chat
template's fixed per-turn header first.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from build_seed_scoremax_v1 import load_jsonl, sha256, stable_hash, task_of


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETAIN_SOURCE = (
    ROOT / "assets/derived/releases/e3_userres_r80_retkl_v3_s875/data_seed_teacher_v1.jsonl"
)
DEFAULT_WORLD_SOURCE = (
    ROOT / "assets/derived/releases/i19_frinkleko_world_1578/data_i19_frinkleko_world_1578_clean.jsonl"
)
DEFAULT_OUT = ROOT / "assets/derived/releases/i19_userres_retention_v1/data_world_residual_retention_v1.jsonl"
DEFAULT_AUDIT = ROOT / "assets/derived/releases/i19_userres_retention_v1/manifest.json"

# Must stay byte-identical to WORLD_PREFIX / RETAIN_PREFIX in
# scripts/train/train_world_residual_retkl.py.
WORLD_PREFIX = "[I19-ROUTE:WORLD] "
RETAIN_PREFIX = "[I19-ROUTE:RETAIN] "

EXPECTED_RETAIN_SOURCE_ROWS = 32_644
EXPECTED_WORLD_ROWS = 1_573

# Eight non-world task buckets available in data_seed_teacher_v1.jsonl. Unlike
# I-13's own RETENTION_COUNTS (which excludes action/topic because those were
# I-13's *supervision* target), here the supervision target is world, so all
# eight real business buckets are protected retention domains.
RETENTION_TASKS = (
    "action",
    "topic",
    "material_desc2sid",
    "material_sid2desc",
    "rec_video",
    "rec_prod",
    "rec_ad",
    "rec_living",
)


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


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


def largest_remainder_allocation(total: int, task_count: int) -> list[int]:
    """Split `total` rows across `task_count` buckets as evenly as possible.

    Uses the largest-remainder method so the eight per-task counts differ by
    at most one row and sum exactly to `total` (1,573), instead of silently
    truncating with integer floor division.
    """
    base = total // task_count
    remainder = total - base * task_count
    allocation = [base] * task_count
    for index in range(remainder):
        allocation[index] += 1
    return allocation


def select_retention(
    rows: list[dict[str, Any]], seed: int, total: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {task: [] for task in RETENTION_TASKS}
    for row in rows:
        task = task_of(row)
        if task in buckets:
            buckets[task].append(row)

    counts = largest_remainder_allocation(total, len(RETENTION_TASKS))
    wanted_by_task = dict(zip(RETENTION_TASKS, counts))

    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for task, wanted in wanted_by_task.items():
        candidates = buckets[task]
        candidates.sort(
            key=lambda row: stable_hash(seed, "world_retention", task, *row_key(row))
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


def validate_world_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_lengths = []
    no_think_count = 0
    answer_letter_count = 0
    for row in rows:
        if not row["input"].rstrip().endswith("/no_think"):
            raise AssertionError("world row is not routed through /no_think")
        no_think_count += 1
        if "</think>" not in row["output"]:
            raise AssertionError("world row is missing </think>")
        body = row["output"].split("</think>", 1)[1].lstrip("\n")
        if "正确答案是" not in body and "答案是" not in body:
            raise AssertionError(f"world row does not look like a graded MC answer: {body[:80]!r}")
        answer_letter_count += 1
        output_lengths.append(len(row["output"]))
    return {
        "rows": len(rows),
        "no_think_routed": no_think_count,
        "graded_answer_format": answer_letter_count,
        "output_char_length": summarize(output_lengths),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def stamp(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    stamped = []
    for row in rows:
        new_row = dict(row)
        new_row["instruction"] = prefix + row.get("instruction", "")
        stamped.append(new_row)
    return stamped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retain-source", type=Path, default=DEFAULT_RETAIN_SOURCE)
    parser.add_argument("--world-source", type=Path, default=DEFAULT_WORLD_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260821)
    args = parser.parse_args()

    retain_source_rows = [normalize(row) for row in load_jsonl(args.retain_source)]
    if len(retain_source_rows) != EXPECTED_RETAIN_SOURCE_ROWS:
        raise AssertionError(
            f"expected {EXPECTED_RETAIN_SOURCE_ROWS} retain-source rows, got {len(retain_source_rows)}"
        )
    retain_source_counts = Counter(task_of(row) for row in retain_source_rows)

    world_rows = [normalize(row) for row in load_jsonl(args.world_source)]
    if len(world_rows) != EXPECTED_WORLD_ROWS:
        raise AssertionError(f"expected {EXPECTED_WORLD_ROWS} world rows, got {len(world_rows)}")
    world_audit = validate_world_rows(world_rows)

    retention_rows, retention_audit = select_retention(
        retain_source_rows, args.seed, EXPECTED_WORLD_ROWS
    )
    if len(retention_rows) != len(world_rows):
        raise AssertionError(
            f"world/retention rows are not 1:1: {len(world_rows)}/{len(retention_rows)}"
        )

    world_keys = {row_key(row) for row in world_rows}
    retention_keys = {row_key(row) for row in retention_rows}
    overlap = world_keys & retention_keys
    if overlap:
        raise AssertionError(f"world and retention branches overlap on {len(overlap)} rows")

    stamped_world = stamp(world_rows, WORLD_PREFIX)
    stamped_retention = stamp(retention_rows, RETAIN_PREFIX)
    if WORLD_PREFIX in RETAIN_PREFIX or RETAIN_PREFIX in WORLD_PREFIX:
        raise AssertionError("route sentinels must not be substrings of each other")
    for row in stamped_world:
        if not row["instruction"].startswith(WORLD_PREFIX) or row["instruction"].startswith(RETAIN_PREFIX):
            raise AssertionError("world row failed exact sentinel stamping")
    for row in stamped_retention:
        if not row["instruction"].startswith(RETAIN_PREFIX) or row["instruction"].startswith(WORLD_PREFIX):
            raise AssertionError("retention row failed exact sentinel stamping")

    final_rows = stamped_world + stamped_retention
    random.Random(args.seed).shuffle(final_rows)
    write_jsonl(args.out, final_rows)

    audit = {
        "schema_version": 1,
        "experiment_id": "I-19 (re-trained with I-13-style retention constraint)",
        "asset_class": "T(Frinkleko-world)+D(O1,O2.teacher-unique)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "route_sentinels": {"world": WORLD_PREFIX, "retain": RETAIN_PREFIX},
        "upstream": {
            "world_source": {
                "path": str(args.world_source.resolve()),
                "note": (
                    "1,573-row _clean sibling of data_i19_frinkleko_world_1578.jsonl; "
                    "the 5 rows that leaked into eval/data/competition_smoke.jsonl "
                    "(evaluation_log_visible bucket) have already been removed upstream "
                    "-- see i19_frinkleko_world_1578/contamination_audit_competition_smoke.json"
                ),
                "rows": len(world_rows),
                "sha256": sha256(args.world_source),
            },
            "retain_source": {
                "path": str(args.retain_source.resolve()),
                "note": "same 32,644-row data_seed_teacher_v1.jsonl used to train the I-13 parent",
                "rows": len(retain_source_rows),
                "sha256": sha256(args.retain_source),
                "task_counts": dict(sorted(retain_source_counts.items())),
            },
        },
        "training_semantics": {
            "world_rows": "gold CE plus weak parent KL (mirrors I-13's user-branch USER_PARENT_KL=0.05)",
            "retention_rows": "parent KL only; response defines teacher-forced mask, CE=0 (mirrors I-13's RETENTION_KL_WEIGHT=2.0)",
            "parent": "checkpoints/i13_repro_combined_r80_s875 (current fixed-protocol SOTA, rank-80 combined adapter); no platform E data used",
        },
        "rows": len(final_rows),
        "row_mix": {
            "world_supervision": {
                "rows": len(world_rows),
                "ratio": round(len(world_rows) / len(final_rows), 8),
                **world_audit,
            },
            "parent_retention": {
                "rows": len(retention_rows),
                "ratio": round(len(retention_rows) / len(final_rows), 8),
                "task_allocation": retention_audit,
            },
        },
        "forbidden_rows": {"O2_rule": 0, "E": 0},
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
