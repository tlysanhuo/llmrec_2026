#!/usr/bin/env python3
"""Remove the five documented evaluation-leak rows from data_riders_fk.

The retained rows are copied byte-for-byte and remain in their original order.
This builder intentionally makes no other data transformation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/derived/processed/data_riders_fk.jsonl"
FK_SOURCE = ROOT / "assets/derived/processed/frinkleko_alpaca_32705.jsonl"
OUTPUT = ROOT / "assets/derived/processed/data_riders_fk_clean.jsonl"
AUDIT = ROOT / "logs/data/riders_fk_clean_audit.json"

# One-based source line numbers documented in docs/experiment_log.md.
LEAK_SOURCE_LINES = (171, 12193, 13741, 15389, 19510)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_row(raw: str) -> str:
    row = json.loads(raw)
    row.setdefault("history", [])
    kept = {
        key: row[key]
        for key in ("instruction", "input", "output", "history")
        if key in row
    }
    return json.dumps(kept, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    if not SOURCE.is_file() or not FK_SOURCE.is_file():
        raise FileNotFoundError(f"missing registered input: {SOURCE} or {FK_SOURCE}")

    leak_rows: dict[str, int] = {}
    with FK_SOURCE.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if line_no in LEAK_SOURCE_LINES:
                key = normalized_row(raw)
                if key in leak_rows:
                    raise AssertionError(f"duplicate leak signature at source line {line_no}")
                leak_rows[key] = line_no

    if set(leak_rows.values()) != set(LEAK_SOURCE_LINES):
        raise AssertionError(f"failed to resolve all leak rows: {sorted(leak_rows.values())}")

    removed: list[dict[str, object]] = []
    source_rows = 0
    output_rows = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding="utf-8") as src, OUTPUT.open("w", encoding="utf-8") as dst:
        for shuffled_line_no, raw in enumerate(src, start=1):
            source_rows += 1
            key = normalized_row(raw)
            source_line_no = leak_rows.get(key)
            if source_line_no is not None:
                removed.append(
                    {
                        "source_line_one_based": source_line_no,
                        "shuffled_line_one_based": shuffled_line_no,
                        "row_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    }
                )
                continue
            dst.write(raw)
            output_rows += 1

    if source_rows != 37267:
        raise AssertionError(f"unexpected source rows: {source_rows}")
    if len(removed) != 5 or output_rows != 37262:
        raise AssertionError(f"unexpected removal result: removed={len(removed)} output={output_rows}")
    if {int(item["source_line_one_based"]) for item in removed} != set(LEAK_SOURCE_LINES):
        raise AssertionError(f"removed rows do not match registry: {removed}")

    audit = {
        "asset_class": "D/MIXED(O1,O2.General,T; E removed)",
        "source": str(SOURCE),
        "source_rows": source_rows,
        "source_sha256": sha256(SOURCE),
        "leak_identity_source": str(FK_SOURCE),
        "leak_identity_source_sha256": sha256(FK_SOURCE),
        "builder": "scripts/data/build_riders_fk_clean.py",
        "output": str(OUTPUT),
        "output_rows": output_rows,
        "output_sha256": sha256(OUTPUT),
        "removed": sorted(removed, key=lambda item: int(item["source_line_one_based"])),
        "mix_rows": {
            "frinkleko_clean_T": 32700,
            "world_zh_D_O2_General": 2824,
            "p3_D_mixed": 1500,
            "world_mc_clean_D": 238,
        },
        "invariants": {
            "retained_rows_byte_identical": True,
            "retained_order_identical": True,
            "only_documented_E_rows_removed": True,
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
