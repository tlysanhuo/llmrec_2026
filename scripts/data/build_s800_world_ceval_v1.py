#!/usr/bin/env python3
"""Build the explicitly authorized 1,578-row CEval world residual set.

The source is the registered third-party Frinkleko release.  Only the exact
world-format bucket is projected; no evaluation rows or SID-tagged
recommendation rows are included.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/frinkleko_alpaca_32705.jsonl"
OUTPUT = ROOT / "assets/derived/processed/data_s800_world_ceval_v1.jsonl"
AUDIT = ROOT / "logs/data/s800_world_ceval_v1_audit.json"
SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("instruction") != SYSTEM:
                continue
            if "<s_a_" in json.dumps(row, ensure_ascii=False):
                continue
            rows.append(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "output": row["output"],
                    "history": row.get("history", []),
                }
            )
    if len(rows) != 1578:
        raise AssertionError(f"expected 1578 CEval rows, got {len(rows)}")
    if any("/no_think" not in row["input"] for row in rows):
        raise AssertionError("world rows must use the platform /no_think format")
    if any("正确答案是 (在此处填写选项字母)" not in row["input"] for row in rows):
        raise AssertionError("world rows must use the canonical answer placeholder")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "schema": "s800-world-ceval-v1",
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "source_class": "T (third-party; explicitly user-authorized for this run)",
        "rows": len(rows),
        "output": str(OUTPUT.resolve()),
        "output_sha256": sha256(OUTPUT),
        "mix_ratio": "100% authorized Frinkleko CEval world rows; no O/D/E rows",
        "filter": "instruction exact match and no <s_a_*> SID marker",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
