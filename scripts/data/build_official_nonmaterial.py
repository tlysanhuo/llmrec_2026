#!/usr/bin/env python3
"""Build the unsampled official non-material SFT pool."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def classify(row: dict) -> str:
    body = row["output"].split("</think>")[-1].strip()
    if body.startswith("["):
        return "action"
    if body.startswith("{") and "logic_chain" in body:
        return "topic"
    if "该用户最近" in body:
        return "recommendation"
    return "material"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/processed/data_final.jsonl")
    parser.add_argument("--out", default="data/processed/official_nonmaterial_v1.jsonl")
    parser.add_argument("--audit", default="logs/data/official_nonmaterial_v1_audit.json")
    args = parser.parse_args()

    source_path = Path(args.src)
    output_path = Path(args.out)
    audit_path = Path(args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    output_hash = hashlib.md5()
    output_rows = 0
    with source_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for source_index, line in enumerate(source):
            row = json.loads(line)
            task = classify(row)
            counts[task] += 1
            if task == "material":
                continue
            clean = {
                "instruction": row.get("instruction", ""),
                "input": row["input"],
                "output": row["output"],
                "history": row.get("history") or [],
            }
            encoded = (json.dumps(clean, ensure_ascii=False) + "\n").encode("utf-8")
            output.write(encoded.decode("utf-8"))
            output_hash.update(encoded)
            output_rows += 1

    expected = {
        "recommendation": 19204,
        "material": 10384,
        "action": 1588,
        "topic": 1304,
    }
    if dict(counts) != expected:
        raise AssertionError(f"unexpected official task counts: {dict(counts)}")
    if output_rows != 22096:
        raise AssertionError(f"unexpected non-material row count: {output_rows}")

    audit = {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "source_counts": dict(counts),
        "output_rows": output_rows,
        "excluded_material_rows": counts["material"],
        "synthetic_rows": 0,
        "md5": output_hash.hexdigest(),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
