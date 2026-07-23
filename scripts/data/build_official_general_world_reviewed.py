#!/usr/bin/env python3
"""Freeze the independently reviewed safe MC subset from official General scans.

This script deliberately creates a reviewed *candidate* asset, not trainer-format
data.  The release gate remains closed unless the reviewed pool is large enough
to support both a useful train split and a permanently held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_official_general_world_clean import (
    atomic_json,
    atomic_jsonl,
    parse_mc_prompt,
    semantic_normalize,
    sha256_file,
    stable_json,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "assets/derived/official_general"
DEFAULT_STRICT = SOURCE_DIR / "world_mc_strict_candidates.jsonl"
DEFAULT_NEAR = SOURCE_DIR / "world_clean_near_rejections.jsonl"
DEFAULT_QA = SOURCE_DIR / "general_zh_short_candidates.jsonl"
DEFAULT_OUT = SOURCE_DIR / "world_mc_human_reviewed_safe.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/official_general_world_reviewed_audit.json"

EXPECTED_STRICT_SHA256 = "b25aefe29cebb236ed339d1675c122349151ded4462bd9ad19469a12a08b3b75"
EXPECTED_NEAR_SHA256 = "f3c26856eaceef7381166a67d1cc74eeb51e2e3ff0faec18c841e9745c17302f"
EXPECTED_QA_SHA256 = "7b3e59c259ae3150faaa46a03f995d71acc18e4a6ec642cb1147659f105075b3"
EXPECTED_STRICT_ROWS = 5
EXPECTED_FORMAT_ONLY_ROWS = 118
EXPECTED_QA_ROWS = 26
MIN_RELEASE_TOTAL = 48
MIN_HOLDOUT_ROWS = 16

# Two strict candidates plus 27 answer-format-only recoveries passed independent
# manual factuality, uniqueness, stability, and risk review on 2026-07-18.
APPROVED_GOLD: dict[str, str] = {
    "0071de1176388ba8299d6547f029e281c87ee51cad7b9693a5a37d3a0c04eff7": "C",
    "0389103d6c320a5bf9f3c9ef4ab228f2ee55993c75cc0e376356195442266479": "C",
    "0aa0dd2dcd4894eeacd2311ed05e3800f6c851f3ec1305424106aaacc15235cb": "A",
    "1b31bdff62136dbbf42851b4cca6235ae05d609fdd85e15044104f738e6e814c": "C",
    "1f9dbba0a0cb8b5e15e85d7c2b3e5a83a82eb73cc287ccd36d138ade45570f20": "B",
    "341ef952dc17c7ca9ec3dea520428763ff83def935a90a949dcf08adb8ca8b75": "D",
    "3f4a84c7f2608df14ab761c64d7e17ff3fc19d67fad8f1dd01ab366dcca9f405": "C",
    "59f63fecbc764cbb630d57e14921949d03bb7df967d3bde1ea4b9c86f5f8e2d2": "B",
    "5bce0e86d1354a20b39de928d271970bbfa71c70fbdc5bda5f7b7ef09f4e8224": "C",
    "625d8c304bfc8ff381842ff1cc6712520099be9a1191c664021782e283c0449e": "B",
    "6df696ee7f3ae090c8b90446e08a5d5afa236a99b9921b35ff70bb9a869e7898": "A",
    "72c8ee0e777b4fcb0d5e1045b92625f308150fb8129baf1206c31dc9658d8e7f": "C",
    "9f327cd1a9a39706ef559c7c76f6b442c326bc20040cfb7f9e3cb8758f5315a7": "C",
    "a7c1285e670a1ba2bd505eeaca298ad92f62dddf34528e75f353288988d97cce": "B",
    "b3ffab4836c42251a5b12379a99fc248d33c7858c22e751fef3363ca5afa8154": "D",
    "b5427e542f68e3287b797bc02b52b081c1ed8abc92e055494f173a1af5f6118a": "D",
    "bcf7cb936beb188990be3e7d1c3a8ea491e71969f8653e7816a4d4cc46a82029": "D",
    "c16d05c062b06a005bb97a36370adf3abeeef4553580776bfbff90a13dc64cf1": "B",
    "c718c99cc09443fb692dac2829bad87edc0ee842244c3889e02a3261e3b2b65a": "B",
    "c9c24758de36878536e316034db2fc327e6eeea763057bcba36c9b997bd3ee23": "B",
    "d4e27d1a9b8a22b59db0025a62c36e67d2264489b96e1848df4c66942f7a32ae": "C",
    "d6629ee6786b9f8f961e38851feb31aa546ca259ee15df8c35a88a84487f102c": "C",
    "dcb108e514bb3fcb00c5a6971c36a756324b241041722aeb4ea8c168df7e0b02": "C",
    "e477b0accd9ca1ee49dbc1d1759dfb43ba048ed61938fb1af7626e54781e063e": "A",
    "ea144be5b108391f941cca474a25bea8c32392dfb7e06e247c92ea6f228f2fa9": "A",
    "f51feb1e0aca56b6277ee5dc10e0e7662ebb3650fc390f4ecc745a49d2f52af4": "A",
    "f61b57a9f6e3e63e152a494c5a405264eace077c5f1f7d918eccabf495a108ad": "A",
    "f77df32ae9245d26ac3f9063307d57e6d214e059462b70fd40e4af33a77a0731": "B",
    "fd2652c7c0622fd989750706d12ceec5573d0c7e7e8dbd16176429b7c1523c0a": "C",
}

STRICT_APPROVED = {
    "6df696ee7f3ae090c8b90446e08a5d5afa236a99b9921b35ff70bb9a869e7898",
    "bcf7cb936beb188990be3e7d1c3a8ea491e71969f8653e7816a4d4cc46a82029",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def fingerprint(builder_sha: str, strict_sha: str, near_sha: str, qa_sha: str) -> str:
    payload = {
        "approved_gold": APPROVED_GOLD,
        "builder_sha256": builder_sha,
        "near_sha256": near_sha,
        "qa_sha256": qa_sha,
        "strict_sha256": strict_sha,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--near", type=Path, default=DEFAULT_NEAR)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder_path = Path(__file__).resolve()
    builder_sha = sha256_file(builder_path)
    strict_sha = sha256_file(args.strict)
    near_sha = sha256_file(args.near)
    qa_sha = sha256_file(args.qa)
    if strict_sha != EXPECTED_STRICT_SHA256:
        raise RuntimeError(f"strict input hash mismatch: {strict_sha}")
    if near_sha != EXPECTED_NEAR_SHA256:
        raise RuntimeError(f"near-rejection input hash mismatch: {near_sha}")
    if qa_sha != EXPECTED_QA_SHA256:
        raise RuntimeError(f"QA input hash mismatch: {qa_sha}")

    strict_rows = read_jsonl(args.strict)
    near_rows = read_jsonl(args.near)
    qa_rows = read_jsonl(args.qa)
    if len(strict_rows) != EXPECTED_STRICT_ROWS:
        raise RuntimeError(f"expected {EXPECTED_STRICT_ROWS} strict rows, got {len(strict_rows)}")
    if len(qa_rows) != EXPECTED_QA_ROWS:
        raise RuntimeError(f"expected {EXPECTED_QA_ROWS} QA rows, got {len(qa_rows)}")
    format_only = [
        row
        for row in near_rows
        if row.get("reason_codes") in (["answer_unparsed"], ["answer_not_final"])
    ]
    if len(format_only) != EXPECTED_FORMAT_ONLY_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_FORMAT_ONLY_ROWS} format-only rows, got {len(format_only)}"
        )

    strict_by_id = {str(row["record_id"]): row for row in strict_rows}
    near_by_id = {str(row["record_id"]): row for row in format_only}
    available = set(strict_by_id) | set(near_by_id)
    missing = sorted(set(APPROVED_GOLD) - available)
    if missing:
        raise RuntimeError(f"approved IDs missing from frozen inputs: {missing}")
    if set(strict_by_id) & set(near_by_id):
        raise RuntimeError("strict and recovery record IDs overlap")
    if STRICT_APPROVED != (set(APPROVED_GOLD) & set(strict_by_id)):
        raise RuntimeError("strict approval partition drifted")

    build_fingerprint = fingerprint(builder_sha, strict_sha, near_sha, qa_sha)
    output: list[dict[str, Any]] = []
    seen_semantic: set[str] = set()
    source_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    for record_id, gold in sorted(APPROVED_GOLD.items()):
        if record_id in strict_by_id:
            source = strict_by_id[record_id]
            clean = source.get("clean")
            if not isinstance(clean, dict):
                raise ValueError(f"strict row {record_id} has no clean payload")
            question = str(clean["question"])
            options = clean["options"]
            if clean.get("answer_letter") != gold:
                raise RuntimeError(f"strict gold mismatch for {record_id}")
            review_bucket = "strict_candidate"
        else:
            source = near_by_id[record_id]
            parsed, reasons = parse_mc_prompt(str(source["prompt"]))
            if parsed is None or reasons:
                raise RuntimeError(f"recovery prompt failed frozen parser for {record_id}: {reasons}")
            question = parsed.question
            options = parsed.options
            review_bucket = "answer_format_recovery"
        if not isinstance(options, dict) or set(options) != set("ABCD"):
            raise ValueError(f"invalid options for {record_id}")
        if gold not in options:
            raise ValueError(f"invalid approved gold for {record_id}: {gold}")
        semantic_payload = semantic_normalize(question) + "\0" + "\0".join(
            semantic_normalize(str(options[label])) for label in "ABCD"
        )
        semantic_hash = hashlib.sha256(semantic_payload.encode("utf-8")).hexdigest()
        if semantic_hash in seen_semantic:
            raise RuntimeError(f"semantic duplicate in reviewed output: {record_id}")
        seen_semantic.add(semantic_hash)
        lineage = source.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError(f"missing lineage for {record_id}")
        asset_id = str(lineage.get("asset_id", "unknown"))
        source_counts[asset_id] += 1
        answer_counts[gold] += 1
        output.append(
            {
                "record_id": record_id,
                "task_type": "world_mc",
                "clean": {
                    "question": question,
                    "options": {label: str(options[label]) for label in "ABCD"},
                    "answer_letter": gold,
                    "answer_text": str(options[gold]),
                },
                "review": {
                    "status": "pass",
                    "factual_correct": True,
                    "unambiguous": True,
                    "low_risk_and_stable": True,
                    "review_bucket": review_bucket,
                    "reviewers": ["independent-manual-review-20260718"],
                },
                "lineage": lineage,
                "builder": {
                    "builder_sha256": builder_sha,
                    "build_fingerprint": build_fingerprint,
                    "ruleset_version": "official-general-world-reviewed-20260718-v1",
                },
            }
        )

    if len(output) != 29:
        raise RuntimeError(f"expected 29 approved rows, got {len(output)}")
    atomic_jsonl(args.out, output)
    output_sha = sha256_file(args.out)
    eligible_for_semantic_split = len(output) >= MIN_RELEASE_TOTAL
    audit = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "asset_class": "D-reviewed-candidate(O2.General,O5); NOT TRAINING DATA",
        "ruleset_version": "official-general-world-reviewed-20260718-v1",
        "builder": str(builder_path),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "inputs": {
            "strict_candidates": {
                "path": str(args.strict.resolve()),
                "rows": len(strict_rows),
                "sha256": strict_sha,
            },
            "near_rejections": {
                "path": str(args.near.resolve()),
                "rows": len(near_rows),
                "format_only_reviewed_rows": len(format_only),
                "sha256": near_sha,
            },
            "qa_candidates": {
                "path": str(args.qa.resolve()),
                "rows": len(qa_rows),
                "sha256": qa_sha,
            },
        },
        "manual_review": {
            "mc_rows_reviewed": len(strict_rows) + len(format_only),
            "mc_pass": len(output),
            "mc_fail": len(strict_rows) + len(format_only) - len(output),
            "strict_pass": len(STRICT_APPROVED),
            "format_only_recovery_pass": len(output) - len(STRICT_APPROVED),
            "qa_rows_reviewed": 26,
            "qa_factually_plausible": 9,
            "qa_training_eligible": 0,
            "qa_exclusion": "open-ended long-answer format mismatches platform Chinese /no_think A-D MC",
        },
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(output),
            "bytes": args.out.stat().st_size,
            "sha256": output_sha,
            "answer_distribution": dict(sorted(answer_counts.items())),
            "upstream_distribution": dict(sorted(source_counts.items())),
        },
        "release_gate": {
            "minimum_total_before_split": MIN_RELEASE_TOTAL,
            "minimum_permanent_holdout": MIN_HOLDOUT_ROWS,
            "approved_total": len(output),
            "shortfall": max(0, MIN_RELEASE_TOTAL - len(output)),
            "eligible_for_semantic_split": eligible_for_semantic_split,
            "training_projection_created": False,
            "decision": "CLOSED_INSUFFICIENT_REVIEWED_MC"
            if not eligible_for_semantic_split
            else "ELIGIBLE_FOR_SEMANTIC_SPLIT_ONLY",
        },
    }
    atomic_json(args.audit, audit)
    print(stable_json(audit))


if __name__ == "__main__":
    main()
