#!/usr/bin/env python3
"""Build the I-13 domain-balanced recommendation preference subset.

The upstream pairs are already O1-derived, same-prompt, same-domain hard
negatives.  This builder changes no chosen/rejected label; it only performs a
deterministic domain rebalance for a low-dose I-13 DPO experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT / "assets/derived/processed/data_o1_reward_preference_v1_train.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "assets/derived/processed/data_i13_rec_balanced_preference_v1_train.jsonl"
)
DEFAULT_AUDIT = ROOT / "logs/data/i13_rec_balanced_preference_v1_audit.json"
DEFAULT_COUNTS = {
    "rec_ad": 768,
    "rec_prod": 768,
    "rec_living": 768,
    "rec_video": 384,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, row: dict[str, Any]) -> str:
    meta = row["meta"]
    key = "|".join(
        (
            str(seed),
            str(meta["task"]),
            str(meta["source_row"]),
            str(meta["prompt_group"]),
            str(meta["chosen_target"]),
            str(meta["rejected_target"]),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            required = {"instruction", "input", "chosen", "rejected", "meta"}
            if set(row) != required:
                raise ValueError(f"line {line_number}: unexpected columns {sorted(row)}")
            meta = row["meta"]
            if meta.get("split") != "train":
                raise ValueError(f"line {line_number}: non-train pair")
            if meta.get("source") != "D(O1):data_seed_clean_v1":
                raise ValueError(f"line {line_number}: unexpected lineage")
            if row["chosen"] == row["rejected"]:
                raise ValueError(f"line {line_number}: identical pair")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260823)
    args = parser.parse_args()

    for path in (args.output, args.audit):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    rows = load_jsonl(args.source)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["meta"]["task"])].append(row)

    selected: list[dict[str, Any]] = []
    for task, count in DEFAULT_COUNTS.items():
        candidates = sorted(buckets[task], key=lambda row: stable_hash(args.seed, row))
        if len(candidates) < count:
            raise ValueError(f"{task}: requested {count}, found {len(candidates)}")
        for row in candidates[:count]:
            copied = dict(row)
            copied_meta = dict(row["meta"])
            copied_meta["subset"] = "data_i13_rec_balanced_preference_v1_train"
            copied_meta["subset_seed"] = args.seed
            copied["meta"] = copied_meta
            selected.append(copied)

    selected.sort(key=lambda row: stable_hash(args.seed + 1, row))
    pair_ids = {
        (row["meta"]["source_row"], row["meta"]["prompt_group"])
        for row in selected
    }
    if len(pair_ids) != len(selected):
        raise AssertionError("duplicate source-row/prompt-group pair")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    task_counts = Counter(row["meta"]["task"] for row in selected)
    tier_counts = Counter(
        (row["meta"]["task"], row["meta"]["negative_tier"])
        for row in selected
    )
    total = len(selected)
    audit = {
        "status": "COMPLETE_DERIVED_TRAINING_ASSET",
        "asset_id": "data_i13_rec_balanced_preference_v1_train",
        "class": "D(O1)",
        "purpose": "low-dose domain-balanced hard-negative DPO on I-13",
        "upstream": {
            "asset_id": "data_o1_reward_preference_v1_train",
            "path": str(args.source.resolve()),
            "rows": len(rows),
            "sha256": sha256(args.source),
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "seed": args.seed,
            "operation": "deterministic subset only; chosen/rejected labels unchanged",
        },
        "output": {
            "path": str(args.output.resolve()),
            "rows": total,
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
        "mix": {
            "task_counts": dict(sorted(task_counts.items())),
            "task_ratios": {
                task: round(count / total, 8)
                for task, count in sorted(task_counts.items())
            },
            "tier_counts_by_task": {
                f"{task}/{tier}": count
                for (task, tier), count in sorted(tier_counts.items())
            },
            "O1_derived_pairs": total,
            "O2_T_E_teacher_model_rollout_pairs": 0,
        },
        "safety": {
            "unique_source_row_prompt_group_pairs": len(pair_ids),
            "labels_rewritten": 0,
            "evaluation_rows_used": 0,
            "third_party_rows_used": 0,
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
