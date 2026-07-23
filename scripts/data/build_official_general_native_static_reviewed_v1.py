#!/usr/bin/env python3
"""Project the task-fit-reviewed subset of native static O2 General SFT.

Review scope is deliberately narrow: reviewers judge whether a prompt is a
static world-knowledge task aligned with the intended General replay role.
They do not certify the official source response as independent factual gold.
The frozen line-number decisions are safe only for the exact upstream hashes
asserted below.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PERSONAL_ROOT = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51")
SOURCE = ROOT / "assets/derived/processed/data_official_general_native_static_v1.jsonl"
SOURCE_LINEAGE = ROOT / "assets/derived/official_general/official_general_native_static_v1_lineage.jsonl"
OUT = ROOT / "assets/derived/processed/data_official_general_native_static_reviewed_v1.jsonl"
LINEAGE_OUT = ROOT / "assets/derived/official_general/official_general_native_static_reviewed_v1_lineage.jsonl"
AUDIT_OUT = ROOT / "logs/data/official_general_native_static_reviewed_v1_audit.json"

SOURCE_SHA256 = "bb96bf7584b3162a94c7e90022bacd37ec1c0076c9a4bc1a8fc365767fa92aea"
SOURCE_LINEAGE_SHA256 = "dc2473c3ab421fee24d9e1e160356a529249c41c2abf3955ba013616c31d5722"
RULESET_VERSION = "official-general-native-static-reviewed-20260718-v1"

# One-based line numbers in the exact 270-row mechanical candidate pool.
# Review criterion: retain static explanatory/factual knowledge; reject
# fictional hypotheticals, personal advice, corporate strategy/current policy,
# careers, health/medical/benefit claims, tool operation, and writing tasks.
APPROVED_LINE_NUMBERS = frozenset({
    # history / culture
    12, 14, 23, 28, 29, 34, 35,
    # geography / environment
    39, 40, 42, 43, 45, 48, 50, 52, 54, 56, 57, 62, 63, 64, 66, 67, 68,
    69, 72, 74, 76, 77, 78, 79, 80, 82, 83, 84, 87, 89, 90, 96, 97,
    # natural science
    99, 103, 107, 108, 110, 111, 112, 114, 118, 119, 120, 121, 124, 125, 127,
    128, 130, 131, 139, 140, 143, 144, 145, 147, 149, 150, 151, 152, 155, 157,
    159, 160, 161, 162,
    # computing / technology
    164, 165, 167, 170, 173, 174,
    # everyday static knowledge
    177, 178, 179, 180, 184, 189, 190, 192, 194, 195, 196, 197, 199, 207, 209,
    211, 212, 215, 216, 220, 221, 227, 229, 230, 231, 232, 234, 235, 236, 237,
    238, 243, 244, 245, 246, 247, 248, 250, 251, 255, 256, 257, 258, 259, 260,
    264, 265, 266, 270,
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
    temp.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    if not PERSONAL_ROOT.is_mount() or not os.access(PERSONAL_ROOT, os.W_OK):
        raise RuntimeError(f"personal volume unavailable or read-only: {PERSONAL_ROOT}")
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise AssertionError("mechanical candidate hash drifted; review decisions are invalid")
    if sha256_file(SOURCE_LINEAGE) != SOURCE_LINEAGE_SHA256:
        raise AssertionError("candidate lineage hash drifted; review decisions are invalid")
    source_rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    source_lineage = [json.loads(line) for line in SOURCE_LINEAGE.open(encoding="utf-8") if line.strip()]
    if len(source_rows) != 270 or len(source_lineage) != 270:
        raise AssertionError("candidate row signature drifted")
    if max(APPROVED_LINE_NUMBERS) > len(source_rows) or min(APPROVED_LINE_NUMBERS) < 1:
        raise AssertionError("review decision line is out of range")

    rows: list[dict] = []
    lineage: list[dict] = []
    rejected: Counter[str] = Counter()
    for line_number, (row, meta) in enumerate(zip(source_rows, source_lineage), start=1):
        if line_number not in APPROVED_LINE_NUMBERS:
            rejected["task_fit_reject"] += 1
            continue
        if meta.get("source_supervision") != "official_native_sft_not_independent_factual_gold":
            raise AssertionError(f"source supervision role drift at line {line_number}")
        if meta.get("quality", {}).get("eval_near_checked") is not True:
            raise AssertionError(f"E near-check missing at line {line_number}")
        if meta.get("mode") != "think" or not row["input"].endswith("/think"):
            raise AssertionError(f"official native think route drift at line {line_number}")
        rows.append(row)
        reviewed_meta = dict(meta)
        reviewed_meta["task_fit_review"] = {
            "status": "pass",
            "scope": "static-world-knowledge task fit and protocol only",
            "factual_gold_certified": False,
            "source_line_number": line_number,
        }
        lineage.append(reviewed_meta)

    if len(rows) != len(APPROVED_LINE_NUMBERS):
        raise AssertionError("approved projection row count drift")
    if len({meta["record_id"] for meta in lineage}) != len(lineage):
        raise AssertionError("duplicate record_id after review projection")
    if len({row["input"] for row in rows}) != len(rows):
        raise AssertionError("duplicate trainer input after review projection")

    atomic_jsonl(OUT, rows)
    atomic_jsonl(LINEAGE_OUT, lineage)
    builder_sha = sha256_file(Path(__file__))
    audit = {
        "asset_class": "D-reviewed(O2.General)-native-static-SFT",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ruleset_version": RULESET_VERSION,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "review": {
            "scope": "static-world-knowledge task fit and official protocol",
            "reviewed_rows": len(source_rows),
            "passed_rows": len(rows),
            "rejected_rows": sum(rejected.values()),
            "rejection_counts": dict(sorted(rejected.items())),
            "independently_verified_factual_gold": False,
            "source_response_role": "official_native_sft_supervision",
        },
        "selected": {
            "domains": dict(sorted(Counter(meta["domain"] for meta in lineage).items())),
            "modes": dict(sorted(Counter(meta["mode"] for meta in lineage).items())),
            "token_stats": {
                "min": min(meta["quality"]["total_tokens"] for meta in lineage),
                "max": max(meta["quality"]["total_tokens"] for meta in lineage),
                "mean": sum(meta["quality"]["total_tokens"] for meta in lineage) / len(lineage),
            },
        },
        "upstream": {
            "training_candidate": {"path": str(SOURCE.resolve()), "sha256": SOURCE_SHA256, "rows": 270},
            "lineage_candidate": {"path": str(SOURCE_LINEAGE.resolve()), "sha256": SOURCE_LINEAGE_SHA256, "rows": 270},
        },
        "release_status": {
            "training_format_created": True,
            "formal_training_mix_approved": False,
        },
        "outputs": {},
    }
    for name, path, count in (
        ("training", OUT, len(rows)),
        ("lineage", LINEAGE_OUT, len(lineage)),
    ):
        audit["outputs"][name] = {
            "path": str(path.resolve()),
            "rows": count,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_json(AUDIT_OUT, audit)
    print(json.dumps({"review": audit["review"], "selected": audit["selected"]}, ensure_ascii=False, sort_keys=True))
    print(f"[OK] training: {OUT}")
    print(f"[OK] lineage: {LINEAGE_OUT}")
    print(f"[OK] audit: {AUDIT_OUT}")


if __name__ == "__main__":
    main()
