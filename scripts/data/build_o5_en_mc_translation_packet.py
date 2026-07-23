#!/usr/bin/env python3
"""Publish the prompt-reviewed, physically answer-blind O5 translation packet.

The upstream file is a sealed answer-claim ledger.  This builder reads it only
to select the frozen prompt-review decisions and then projects *only* the
English question/options plus lineage into a separate translation packet.  No
answer letter, answer text, assistant evidence, or gold-like field is copied.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_official_general_world_clean as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "assets/derived/official_general/o5_en_mc_answer_claim_pilot.jsonl"
OUTPUT = ROOT / "assets/derived/official_general/o5_en_mc_translation_packet.jsonl"
LEDGER = ROOT / "logs/data/o5_en_mc_prompt_review_ledger.jsonl"
AUDIT = ROOT / "logs/data/o5_en_mc_prompt_review_audit.json"

EXPECTED_INPUT_SHA256 = "21c43445b2cb23ca6e46cd5c812b0ae883ab174a763fce3209baef5d76533db3"
EXPECTED_INPUT_ROWS = 51
EXPECTED_PASS_ROWS = 41
RULESET_VERSION = "o5-en-mc-answer-blind-prompt-review-20260718-v1"

# These decisions were made from a projection containing only record_id,
# source, original question/options and topic.  The reviewer had no access to
# source_answer_claim.  A pass means "safe to translate and blind-solve", not
# factual correctness and not gold approval.
REJECT_REASONS = {
    "2328632d5acbf5fe685bd436dad40b4cf8e692bbe1ef80e98a82bcbc6232d896": [
        "computed_unique_value_not_in_options"
    ],
    "6b3668dfc5f1074b6564865f021ab3b98fa74667b99626c9879d909b878584ab": [
        "polar_origin_membership_ambiguity"
    ],
    "6b58f37254fb46385f4fc29d07114cf59c53b0be34ba364dc35a70b42aa16ac0": [
        "broken_scalar_vector_notation"
    ],
    "75489518daa428744a1e74118b502fe336bf518654f16b9c6b9f08a77e617859": [
        "multiple_correct_options"
    ],
    "8eb179d313b14f8cf20fc96ff03c3fe805ea887981522fe33ea69f7663fdd74b": [
        "multiple_correct_options"
    ],
    "b650255b0419d23c9be2442139b8148ee5a2e54f2820e930c1ed1db246d151d8": [
        "underdetermined_geometry"
    ],
    "cc1fd733d9c275ea1ffa481a334afe6fb62f382e1bd8cf7b36ea128fe98b6b64": [
        "odd_degree_polynomial_global_maximum_impossible"
    ],
    "e154b9e079db6aacdd7c1ec1319b0a4ff357d51c057e2d985d9a3e42e4bc0514": [
        "laurent_coefficient_condition_inconsistent"
    ],
    "e8a8010e294b99e9d50d812180ba70a438f80c5fa963d6b793575415b2e656a5": [
        "missing_arrow_pattern_context"
    ],
    "f6be42df8925aa997c6db975726e8700fad92cb9222708378a904e5a73b98a0b": [
        "ambiguous_focal_length_terminology_changes_under_translation"
    ],
}

FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "source_answer_claim",
        "answer",
        "answer_letter",
        "answer_text",
        "correct_answer",
        "evidence",
        "gold",
        "gold_status",
        "metadata_label",
        "response",
        "assistant",
    }
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.input.resolve() != INPUT.resolve():
        raise RuntimeError(f"input must be frozen canonical candidate ledger: {INPUT}")
    base.ensure_safe_paths((args.output, args.ledger, args.audit))
    input_sha = base.sha256_file(args.input)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise AssertionError(
            f"candidate ledger drifted: {input_sha} != {EXPECTED_INPUT_SHA256}"
        )
    rows = _read_jsonl(args.input)
    if len(rows) != EXPECTED_INPUT_ROWS:
        raise AssertionError(f"candidate row count drifted: {len(rows)} != {EXPECTED_INPUT_ROWS}")
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate candidate record_id")
    missing_rejects = set(REJECT_REASONS) - set(ids)
    if missing_rejects:
        raise AssertionError(f"review reject ids missing upstream: {sorted(missing_rejects)}")

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
                    "reviewer_protocol": "answer-blind-prompt-screen-split-20260718",
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
        packet.append(
            {
                "record_id": record_id,
                "task_type": "world_mc_answer_blind_translation",
                "lineage": row["lineage"],
                "source": {
                    "language": "en",
                    "question": row["original"]["question"],
                    "options": row["original"]["options"],
                    "prompt_sha256": base.hash_text(
                        row["original"]["question"]
                        + "\0"
                        + "\0".join(row["original"]["options"][label] for label in "ABCD")
                    ),
                },
                "mechanical_checks": {
                    "topic": quality["topic"],
                    "strict_english": True,
                    "strict_uppercase_abcd": True,
                    "original_eval_overlap_modes": quality["original_eval_overlap_modes"],
                    "original_parent_overlap_modes": quality["original_parent_overlap_modes"],
                    "original_reviewed_overlap_modes": quality["original_reviewed_overlap_modes"],
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
        raise AssertionError(f"prompt-review pass count drifted: {len(packet)} != {EXPECTED_PASS_ROWS}")
    forbidden_hits = sorted(FORBIDDEN_PACKET_KEYS & _all_keys(packet))
    if forbidden_hits:
        raise AssertionError(f"answer-blind packet contains forbidden keys: {forbidden_hits}")
    if any(row["record_id"] in REJECT_REASONS for row in packet):
        raise AssertionError("rejected record leaked into translation packet")

    base.atomic_jsonl(args.output, packet)
    base.atomic_jsonl(args.ledger, ledger)
    audit = {
        "asset_class": "D-answer-blind-translation-packet(O5); NOT TRAINING DATA",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": fingerprint,
        "ruleset_version": RULESET_VERSION,
        "upstream": {
            "path": str(args.input.resolve()),
            "rows": len(rows),
            "sha256": input_sha,
        },
        "prompt_review": {
            "protocol": "answer-blind-prompt-screen-split-20260718",
            "answer_fields_visible": False,
            "pass_to_translation": len(packet),
            "rejected": len(REJECT_REASONS),
            "reason_counts": dict(
                sorted(Counter(reason for reasons in REJECT_REASONS.values() for reason in reasons).items())
            ),
            "factual_gold_approved": 0,
        },
        "answer_isolation": {
            "forbidden_packet_keys": sorted(FORBIDDEN_PACKET_KEYS),
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
        "[done] answer-blind translation packet "
        f"rows={audit['outputs']['translation_packet']['rows']} "
        f"sha256={audit['outputs']['translation_packet']['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
