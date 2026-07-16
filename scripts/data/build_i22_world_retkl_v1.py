#!/usr/bin/env python3
"""Build I-22 world answer-token supervision plus frozen-I13 retention.

The 231 registered D(O2.General) world rows are matched exactly inside the
registered I-12 mixture.  Four rows without an unambiguous final option are
excluded.  A deterministic 46-row prompt-disjoint holdout is reserved; the
remaining 181 rows are converted to the platform's empty-think answer form
and repeated seven times.  Every non-world I-12 row is retained once and is
consumed as frozen-parent KL only by the I-22 trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
DEFAULT_WORLD = ROOT / "assets/derived/official_general/sft_world_knowledge.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_i22_world_retkl_v1.jsonl"
DEFAULT_HOLDOUT = ROOT / "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/i22_world_retkl_v1_audit.json"

FINAL_ANSWER = re.compile(
    r"(?:正确答案是|最终答案是|Correct answer is)\s*[\(（]?\s*"
    r"([A-J](?:[\s、,，和及/]*[A-J])*)",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(normalize(raw))
    return rows


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def row_key(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_key(seed: int, row: dict[str, Any]) -> str:
    payload = f"{seed}\0{row['instruction']}\0{row['input']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_label(row: dict[str, Any]) -> str | None:
    body = row["output"].split("</think>")[-1].strip()
    match = FINAL_ANSWER.search(body)
    if match is None:
        return None
    letters = re.findall(r"[A-J]", match.group(1).upper())
    label = "".join(dict.fromkeys(letters))
    return label or None


def canonical_world(row: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "instruction": row["instruction"],
        "input": row["input"],
        "output": f"<think>\n\n</think>\n正确答案是 ({label})",
        "history": row["history"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260827)
    parser.add_argument("--holdout-rows", type=int, default=46)
    parser.add_argument("--world-repeat", type=int, default=7)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    world_rows = load_jsonl(args.world)
    if len(base_rows) != 6_106 or len(world_rows) != 231:
        raise AssertionError(f"upstream row signature drifted: {len(base_rows)}/{len(world_rows)}")

    world_keys = {row_key(row) for row in world_rows}
    if len(world_keys) != 231:
        raise AssertionError("registered world source is not row-unique")
    embedded = [row for row in base_rows if row_key(row) in world_keys]
    retention = [row for row in base_rows if row_key(row) not in world_keys]
    if len(embedded) != 231 or len(retention) != 5_875:
        raise AssertionError(f"world/base exact join failed: {len(embedded)}/{len(retention)}")

    valid: list[tuple[dict[str, Any], str]] = []
    excluded: list[dict[str, Any]] = []
    for row in world_rows:
        label = extract_label(row)
        if label is None:
            excluded.append(row)
        else:
            valid.append((row, label))
    if len(valid) != 227 or len(excluded) != 4:
        raise AssertionError(f"world answer parser signature drifted: {len(valid)}/{len(excluded)}")

    valid.sort(key=lambda item: stable_key(args.seed, item[0]))
    holdout_raw = valid[: args.holdout_rows]
    train_raw = valid[args.holdout_rows :]
    if len(holdout_raw) != 46 or len(train_raw) != 181:
        raise AssertionError("unexpected deterministic world split")
    train_unique = [canonical_world(row, label) for row, label in train_raw]
    holdout = [canonical_world(row, label) for row, label in holdout_raw]
    if {row["input"] for row in train_unique} & {row["input"] for row in holdout}:
        raise AssertionError("world train/holdout prompt overlap")

    world_positive = [dict(row) for _ in range(args.world_repeat) for row in train_unique]
    final_rows = retention + world_positive
    random.Random(args.seed).shuffle(final_rows)
    write_jsonl(args.out, final_rows)
    write_jsonl(args.holdout, holdout)

    audit = {
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General)",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "upstream": {
            "i12_registered_mix": {
                "asset_id": "data_user_residual_retention_v1",
                "path": str(args.base.resolve()),
                "rows": len(base_rows),
                "sha256": sha256(args.base),
            },
            "registered_world": {
                "asset_id": "D(O2.General):sft_world_knowledge",
                "path": str(args.world.resolve()),
                "rows": len(world_rows),
                "sha256": sha256(args.world),
            },
        },
        "world_filter": {
            "parseable": len(valid),
            "excluded_ambiguous": len(excluded),
            "train_unique": len(train_unique),
            "holdout_unique": len(holdout),
            "train_holdout_prompt_overlap": 0,
            "canonical_response": "<think>\\n\\n</think>\\n正确答案是 (<ascending letters>)",
            "train_label_counts": dict(sorted(Counter(label for _, label in train_raw).items())),
            "holdout_label_counts": dict(sorted(Counter(label for _, label in holdout_raw).items())),
        },
        "training_mix": {
            "world_answer_ce_rows": len(world_positive),
            "world_unique_rows": len(train_unique),
            "world_repeat": args.world_repeat,
            "frozen_i13_kl_only_rows": len(retention),
            "total_rows": len(final_rows),
            "world_answer_ce_ratio": len(world_positive) / len(final_rows),
            "frozen_i13_kl_only_ratio": len(retention) / len(final_rows),
            "third_party_rows": 0,
            "evaluation_rows": 0,
            "model_rollout_rows": 0,
        },
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(final_rows),
            "sha256": sha256(args.out),
        },
        "holdout": {
            "path": str(args.holdout.resolve()),
            "rows": len(holdout),
            "sha256": sha256(args.holdout),
            "allowed_role": "checkpoint selection only; excluded from CE and KL training",
        },
    }
    write_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
