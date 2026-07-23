#!/usr/bin/env python3
"""Build the locally reproducible I19-style world/retention mixture.

This is a new local experiment over the retained s800 parent.  It mirrors the
reported I19 data geometry while keeping the five documented Frinkleko leak
rows out of the world branch and recording every upstream identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_seed_scoremax_v1 import load_jsonl, stable_hash, task_of


ROOT = Path(__file__).resolve().parents[2]
WORLD_SOURCE = ROOT / "data/processed/frinkleko_alpaca_32705.jsonl"
RETENTION_SOURCE = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
OUTPUT = ROOT / "assets/derived/processed/data_i19_local_world_residual_retention_v1.jsonl"
AUDIT = ROOT / "logs/data/i19_local_world_residual_retention_v1_audit.json"
SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
LEAK_SOURCE_LINES = (171, 12193, 13741, 15389, 19510)
WORLD_ROUTE = "[I19-ROUTE:WORLD] "
RETAIN_ROUTE = "[I19-ROUTE:RETAIN] "

RETENTION_COUNTS = {
    "action": 197,
    "topic": 197,
    "material_desc2sid": 197,
    "material_sid2desc": 197,
    "rec_video": 197,
    "rec_prod": 196,
    "rec_ad": 196,
    "rec_living": 196,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["instruction"], row["input"], row["output"]


def load_world() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = 0
    removed: list[int] = []
    with WORLD_SOURCE.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            source_rows += 1
            if line_number in LEAK_SOURCE_LINES:
                removed.append(line_number)
                continue
            row = json.loads(raw)
            if row.get("instruction") != SYSTEM or "<s_a_" in json.dumps(row, ensure_ascii=False):
                continue
            normalized = normalize(row)
            if not normalized["input"] or not normalized["output"]:
                raise AssertionError(f"empty world row at source line {line_number}")
            if "/no_think" not in normalized["input"]:
                raise AssertionError(f"world row is not /no_think at line {line_number}")
            if "正确答案是 (在此处填写选项字母)" not in normalized["input"]:
                raise AssertionError(f"world placeholder drift at line {line_number}")
            rows.append(normalized)
    if source_rows != 32705:
        raise AssertionError(f"expected 32705 Frinkleko rows, got {source_rows}")
    if removed != list(LEAK_SOURCE_LINES):
        raise AssertionError(f"leak removal mismatch: {removed}")
    if len(rows) != 1573:
        raise AssertionError(f"expected 1573 clean world rows, got {len(rows)}")
    return rows, {"source_rows": source_rows, "removed_source_lines": removed}


def load_retention(seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = [normalize(row) for row in load_jsonl(RETENTION_SOURCE)]
    if len(source) != 32644:
        raise AssertionError(f"expected 32644 retention source rows, got {len(source)}")
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in RETENTION_COUNTS}
    for row in source:
        task = task_of(row)
        if task in buckets:
            buckets[task].append(row)

    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for task, wanted in RETENTION_COUNTS.items():
        candidates = sorted(
            buckets[task],
            key=lambda row: stable_hash(seed, "i19-local-retention", task, *row_key(row)),
        )
        if len(candidates) < wanted:
            raise AssertionError(f"only {len(candidates)} {task} rows, need {wanted}")
        chosen = candidates[:wanted]
        selected.extend(chosen)
        audit[task] = {"available": len(candidates), "selected": len(chosen)}
    if len(selected) != 1573:
        raise AssertionError(f"expected 1573 retention rows, got {len(selected)}")
    return selected, audit


def with_route(row: dict[str, Any], route: str) -> dict[str, Any]:
    routed = dict(row)
    routed["instruction"] = route + row["instruction"]
    return routed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=19260821)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()

    world, world_audit = load_world()
    retention, retention_audit = load_retention(args.seed)
    world_keys = {row_key(row) for row in world}
    retention_keys = {row_key(row) for row in retention}
    if world_keys & retention_keys:
        raise AssertionError("world and retention branches overlap")

    final = [with_route(row, WORLD_ROUTE) for row in world]
    final.extend(with_route(row, RETAIN_ROUTE) for row in retention)
    random.Random(args.seed).shuffle(final)
    write_jsonl(args.out, final)

    audit = {
        "schema": "i19-local-world-residual-retention-v1",
        "asset_class": "D/MIXED(T-authorized,O1,O2.teacher-unique)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "upstream": {
            "frinkleko_world": {
                "path": str(WORLD_SOURCE.resolve()),
                "sha256": sha256(WORLD_SOURCE),
                "source_class": "T (third-party; explicitly user-authorized)",
                **world_audit,
                "clean_rows": len(world),
            },
            "retention_source": {
                "path": str(RETENTION_SOURCE.resolve()),
                "sha256": sha256(RETENTION_SOURCE),
                "source_class": "D(O1,O2.teacher-unique)",
                "rows": 32644,
            },
        },
        "rows": len(final),
        "row_mix": {
            "world_ce": {"rows": len(world), "ratio": 0.5},
            "retention_kl": {
                "rows": len(retention),
                "ratio": 0.5,
                "task_counts": RETENTION_COUNTS,
                "selection": retention_audit,
            },
        },
        "route_sentinels": {
            "world": WORLD_ROUTE,
            "retention": RETAIN_ROUTE,
            "cross_branch_triple_key_overlap": 0,
        },
        "training_semantics": {
            "world": "full response CE + 0.05 parent KL",
            "retention": "parent KL only; response defines teacher-forced mask",
            "parent": "local e3_userres_r80_retkl_v3_s800 merged parent",
        },
        "output": str(args.out.resolve()),
        "output_sha256": sha256(args.out),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
