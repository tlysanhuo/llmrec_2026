#!/usr/bin/env python3
"""Publish the prompt-reviewed, physically answer-blind O2 translation packet."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_o5_en_mc_translation_packet as common  # noqa: E402


base = common.base
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "assets/derived/official_general/o2_en_mc_answer_claim_pilot.jsonl"
OUTPUT = ROOT / "assets/derived/official_general/o2_en_mc_translation_packet.jsonl"
LEDGER = ROOT / "logs/data/o2_en_mc_prompt_review_ledger.jsonl"
AUDIT = ROOT / "logs/data/o2_en_mc_prompt_review_audit.json"

EXPECTED_INPUT_SHA256 = "6a78709e887feb0ae034c2392e9d3117abe28ff566365f65c5623dd87a1ab1cf"
EXPECTED_INPUT_ROWS = 38
EXPECTED_PASS_ROWS = 27
RULESET_VERSION = "o2-en-mc-answer-blind-prompt-review-20260718-v1"

REJECT_REASONS = {
    "1b9b0b9e068c59c53bb7862052e8b63bb5a7d454d1418580250742d9d5c66f05": [
        "orthographic_views_orientation_and_dimensions_underspecified"
    ],
    "2b90d9a532f90450543d1ff28687d9c3f10dced75e693f7813748c7b346c750a": [
        "double_escaped_math_delimiters_require_unapproved_repair"
    ],
    "60390babc3e442e444b31071e63383ef1f9b621dc39112731061a2a907218336": [
        "empty_set_notation_and_parameter_domain_ambiguous"
    ],
    "731867514600812bb470037b0109d3690b107521cf373a1e4d0e6d2d5c10f63d": [
        "option_missing_comparison_operator"
    ],
    "aaa8a87c438a1686ca6f5871316846bfd71a68cbab012f2a6ebd88a8345feac3": [
        "set_definition_contains_free_variable"
    ],
    "b5fba95710d90520d4652dfd1ebfc35df501548fcce7d9a028239e4cd964dc01": [
        "broken_heading_or_possible_truncation"
    ],
    "b7295ca762d31fdb754b18aec33b7bd2d00d73eb74102a55230db043ed598797": [
        "acid_definition_framework_unspecified"
    ],
    "bc20ab309d996a1d3d0ee3b92af3a0481a1f8f8f6e8775a6e6c52ece694677d3": [
        "missing_figure_context"
    ],
    "c8d1211b4f02d8f05960a20529a15f199ad47be9b24e04bfa357a576701ab869": [
        "double_escaped_math_delimiters_require_unapproved_repair"
    ],
    "d6505ff17a587512e68ae01b71c09e44bd2979ebab8c07b74e209b48cd1ce312": [
        "natural_language_set_membership_not_unique"
    ],
    "e229b18b22282af0c1c6f62877b91355c898909e6c64340263fdc9f14a9854a9": [
        "plane_counting_object_and_general_position_underspecified"
    ],
}


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.input.resolve() != INPUT.resolve():
        raise RuntimeError(f"input must be canonical O2 candidate ledger: {INPUT}")
    base.ensure_safe_paths((args.output, args.ledger, args.audit))
    input_sha = base.sha256_file(args.input)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise AssertionError(f"O2 candidate ledger drifted: {input_sha} != {EXPECTED_INPUT_SHA256}")
    rows = common._read_jsonl(args.input)
    if len(rows) != EXPECTED_INPUT_ROWS:
        raise AssertionError(f"O2 candidate row count drifted: {len(rows)} != {EXPECTED_INPUT_ROWS}")
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate O2 candidate record_id")
    if set(REJECT_REASONS) - set(ids):
        raise AssertionError("one or more frozen O2 reject ids are absent upstream")

    builder_sha = base.sha256_file(Path(__file__))
    fingerprint = base.hash_text(
        base.stable_json(
            {
                "builder_sha256": builder_sha,
                "input_sha256": input_sha,
                "ruleset_version": RULESET_VERSION,
                "reject_reasons": REJECT_REASONS,
            }
        )
    )
    packet: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["record_id"]):
        record_id = row["record_id"]
        reasons = REJECT_REASONS.get(record_id, [])
        decision = "reject" if reasons else "pass_to_answer_blind_translation"
        ledger.append(
            {
                "record_id": record_id,
                "review": {
                    "reviewer_protocol": "answer-blind-prompt-screen-o2-20260718",
                    "answer_fields_visible": False,
                    "decision": decision,
                    "reason_codes": reasons,
                    "scope": "self_containment_unambiguity_translation_safety_only",
                    "factual_gold_approved": False,
                },
                "builder": {
                    "ruleset_version": RULESET_VERSION,
                    "builder_sha256": builder_sha,
                    "build_fingerprint": fingerprint,
                },
            }
        )
        if reasons:
            continue
        quality = row["quality"]
        original = row["original"]
        packet.append(
            {
                "record_id": record_id,
                "task_type": "world_mc_answer_blind_translation",
                "lineage": row["lineage"],
                "source": {
                    "language": "en",
                    "question": original["question"],
                    "options": original["options"],
                    "prompt_sha256": base.hash_text(
                        original["question"]
                        + "\0"
                        + "\0".join(original["options"][label] for label in "ABCD")
                    ),
                },
                "mechanical_checks": {
                    "topic": quality["topic"],
                    "strict_english": True,
                    "strict_uppercase_abcd": True,
                    "original_eval_overlap_modes": quality["original_eval_overlap_modes"],
                    "original_parent_overlap_modes": quality["original_parent_overlap_modes"],
                    "original_reviewed_overlap_modes": quality["original_reviewed_overlap_modes"],
                    "original_o5_candidate_overlap_modes": quality[
                        "original_o5_candidate_overlap_modes"
                    ],
                },
                "prompt_review": {
                    "status": "pass",
                    "answer_fields_visible": False,
                    "scope": "self_containment_unambiguity_translation_safety_only",
                },
                "translation": {
                    "status": "pending",
                    "required": "faithful_zh_translation_preserving_all_math_and_option_mapping",
                },
                "builder": {
                    "ruleset_version": RULESET_VERSION,
                    "builder_sha256": builder_sha,
                    "build_fingerprint": fingerprint,
                },
            }
        )

    if len(packet) != EXPECTED_PASS_ROWS:
        raise AssertionError(f"O2 prompt-review pass drifted: {len(packet)} != {EXPECTED_PASS_ROWS}")
    forbidden_hits = sorted(common.FORBIDDEN_PACKET_KEYS & common._all_keys(packet))
    if forbidden_hits:
        raise AssertionError(f"O2 answer-blind packet contains forbidden keys: {forbidden_hits}")
    base.atomic_jsonl(args.output, packet)
    base.atomic_jsonl(args.ledger, ledger)
    audit = {
        "asset_class": "D-answer-blind-translation-packet(O2.General); NOT TRAINING DATA",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": fingerprint,
        "ruleset_version": RULESET_VERSION,
        "upstream": {"path": str(args.input.resolve()), "rows": len(rows), "sha256": input_sha},
        "prompt_review": {
            "protocol": "answer-blind-prompt-screen-o2-20260718",
            "answer_fields_visible": False,
            "pass_to_translation": len(packet),
            "rejected": len(REJECT_REASONS),
            "reason_counts": dict(
                sorted(Counter(reason for reasons in REJECT_REASONS.values() for reason in reasons).items())
            ),
            "factual_gold_approved": 0,
        },
        "answer_isolation": {
            "forbidden_packet_keys": sorted(common.FORBIDDEN_PACKET_KEYS),
            "forbidden_key_hits": forbidden_hits,
            "source_answer_claim_copied": False,
            "assistant_response_copied": False,
        },
        "outputs": {
            "translation_packet": {
                "path": str(args.output.resolve()),
                "rows": len(packet),
                "bytes": args.output.stat().st_size,
                "sha256": base.sha256_file(args.output),
            },
            "prompt_review_ledger": {
                "path": str(args.ledger.resolve()),
                "rows": len(ledger),
                "bytes": args.ledger.stat().st_size,
                "sha256": base.sha256_file(args.ledger),
            },
        },
        "release_gate": {
            "translation_completed": False,
            "blind_reviews_completed": False,
            "adjudication_completed": False,
            "gold_labels_created": False,
            "training_projection_created": False,
        },
    }
    base.atomic_json(args.audit, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> None:
    audit = build(parse_args())
    print(
        f"[done] O2 answer-blind translation rows={audit['outputs']['translation_packet']['rows']} "
        f"sha256={audit['outputs']['translation_packet']['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
