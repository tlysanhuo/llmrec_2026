#!/usr/bin/env python3
"""Adjudicate O2 Chinese MC translations after two independent blind solves.

Both solvers see only the Chinese answer-free packet.  The quarantined source
assistant answer is revealed only after both review files are complete.  A row
is accepted only when both reviews pass, agree with each other, and their
consensus agrees with the source claim.  The output remains a reviewed
candidate, not trainer-format data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_o5_zh_mc_dual_answer_blind_reviewed as shared
from build_official_general_world_clean import atomic_json, atomic_jsonl, sha256_file, stable_json


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "assets/derived/official_general"
LOGS = ROOT / "logs/data"

CANDIDATE = DERIVED / "o2_en_mc_answer_claim_pilot.jsonl"
PACKET = DERIVED / "o2_en_mc_zh_blind_review_packet.jsonl"
SOLUTION_A = LOGS / "o2_zh_blind_solution_a.jsonl"
SOLUTION_B = LOGS / "o2_zh_blind_solution_b.jsonl"
OUT = DERIVED / "o2_zh_mc_dual_answer_blind_reviewed_safe.jsonl"
LEDGER = LOGS / "o2_zh_mc_dual_answer_blind_adjudication_ledger.jsonl"
AUDIT = LOGS / "o2_zh_mc_dual_answer_blind_adjudication_audit.json"

EXPECTED_HASHES = {
    "candidate": "6a78709e887feb0ae034c2392e9d3117abe28ff566365f65c5623dd87a1ab1cf",
    "packet": "c045d1789fce9b4ffd1027262d2629e2b301e089bdf7a97dca8b617dbfe80707",
    "solution_a": "167b1cf0dbd307decf538c1e315abf6f2f02211eb344bb019208df765a3e21f7",
    "solution_b": "90e93d755f457a17d952c7ab0cd8266c6158f622de9c92da7a7cdedc7ffddd80",
}
EXPECTED_ROWS = 27
RULESET_VERSION = "o2-zh-mc-dual-answer-blind-adjudication-20260718-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return shared.read_jsonl(path)


def require_input(name: str, supplied: Path, canonical: Path) -> str:
    if supplied.resolve() != canonical.resolve():
        raise RuntimeError(f"{name} must use canonical input: {canonical}")
    digest = sha256_file(supplied)
    if digest != EXPECTED_HASHES[name]:
        raise RuntimeError(f"{name} hash mismatch: {digest} != {EXPECTED_HASHES[name]}")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--solution-a", type=Path, default=SOLUTION_A)
    parser.add_argument("--solution-b", type=Path, default=SOLUTION_B)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = {
        "candidate": CANDIDATE,
        "packet": PACKET,
        "solution_a": SOLUTION_A,
        "solution_b": SOLUTION_B,
    }
    supplied = {
        "candidate": args.candidate,
        "packet": args.packet,
        "solution_a": args.solution_a,
        "solution_b": args.solution_b,
    }
    input_hashes = {
        name: require_input(name, supplied[name], canonical[name]) for name in canonical
    }
    candidates = shared.index_unique(read_jsonl(args.candidate), "candidate")
    packet_rows = read_jsonl(args.packet)
    packet = shared.index_unique(packet_rows, "packet")
    solution_a = shared.index_unique(read_jsonl(args.solution_a), "solution_a")
    solution_b = shared.index_unique(read_jsonl(args.solution_b), "solution_b")
    if len(packet) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} packet rows, got {len(packet)}")
    if set(solution_a) != set(packet) or set(solution_b) != set(packet):
        raise RuntimeError("blind-review coverage does not exactly match packet")
    if not set(packet).issubset(candidates):
        raise RuntimeError("packet IDs are absent from candidate source")
    for row in solution_a.values():
        shared.validate_review(row, "solution_a", require_prompt_blind=True)
    for row in solution_b.values():
        shared.validate_review(row, "solution_b", require_prompt_blind=True)

    builder = Path(__file__).resolve()
    builder_sha = sha256_file(builder)
    build_fingerprint = hashlib.sha256(
        stable_json(
            {
                "builder_sha256": builder_sha,
                "input_hashes": input_hashes,
                "min_confidence": shared.MIN_CONFIDENCE,
                "ruleset_version": RULESET_VERSION,
            }
        ).encode("utf-8")
    ).hexdigest()

    accepted: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    semantic_seen: set[str] = set()
    boundary_rows = 0

    for record_id in sorted(packet):
        packet_row = packet[record_id]
        candidate = candidates[record_id]
        review_a = solution_a[record_id]
        review_b = solution_b[record_id]
        translated = packet_row.get("translated")
        if not isinstance(translated, dict):
            raise ValueError(f"missing translated payload: {record_id}")
        question = str(translated.get("question", "")).strip()
        options_raw = translated.get("options")
        if not question or not isinstance(options_raw, dict) or set(options_raw) != set("ABCD"):
            raise ValueError(f"invalid translated prompt: {record_id}")
        options = {label: str(options_raw[label]).strip() for label in "ABCD"}
        checks = packet_row.get("mechanical_checks")
        if not isinstance(checks, dict):
            raise ValueError(f"missing mechanical checks: {record_id}")
        required_true = {
            "strict_chinese_question",
            "strict_uppercase_abcd",
            "digit_multiset_match",
            "latex_command_multiset_match",
        }
        required_false = {
            "eval_overlap",
            "parent_overlap",
            "reviewed_overlap",
            "char3_near_overlap",
            "o5_reviewed_overlap",
        }
        if any(checks.get(key) is not True for key in required_true):
            raise RuntimeError(f"translation invariant failed: {record_id}")
        if any(checks.get(key) is not False for key in required_false):
            raise RuntimeError(f"translation leakage gate failed: {record_id}")
        if "strict_chinese_rule" in checks:
            if checks["strict_chinese_rule"] != "o2_formula_boundary_exact_han5_zero_latin":
                raise RuntimeError(f"unknown language boundary rule: {record_id}")
            boundary_rows += 1

        claim = candidate.get("source_answer_claim")
        if not isinstance(claim, dict) or claim.get("status") != "source_assistant_claim_only_not_gold":
            raise ValueError(f"invalid quarantined source claim: {record_id}")
        claim_letter = claim.get("letter")
        if claim_letter not in "ABCD":
            raise ValueError(f"invalid source claim letter: {record_id}")

        reasons: list[str] = []
        for label, row in (("a", review_a), ("b", review_b)):
            if row["verdict"] != "pass":
                reasons.append(f"blind_review_{label}_reject")
            review = row["review"]
            for flag in ("self_contained", "single_correct", "stable", "low_risk", "visual_independent"):
                if review.get(flag) is not True:
                    reasons.append(f"blind_review_{label}_{flag}_false")
            solution = row["solution"]
            if solution["confidence"] < shared.MIN_CONFIDENCE:
                reasons.append(f"blind_review_{label}_confidence_below_threshold")
            if solution["alternate_correct_letters"]:
                reasons.append(f"blind_review_{label}_alternate_correct_letters")
        letter_a = review_a["solution"]["answer_letter"]
        letter_b = review_b["solution"]["answer_letter"]
        if letter_a != letter_b:
            reasons.append("blind_solution_disagreement")
        if letter_a != claim_letter or letter_b != claim_letter:
            reasons.append("source_claim_disagrees_with_blind_consensus")
        reasons = sorted(set(reasons))
        decision = "accept" if not reasons else "reject"
        for reason in reasons:
            reason_counts[reason] += 1
        ledger.append(
            {
                "record_id": record_id,
                "decision": decision,
                "reason_codes": reasons,
                "blind_review_a": review_a,
                "blind_review_b": review_b,
                "source_answer_claim": {
                    "letter": claim_letter,
                    "status": claim["status"],
                    "role": "agreement_check_not_standalone_gold",
                },
            }
        )
        if reasons:
            continue

        assert letter_a == letter_b == claim_letter
        semantic = shared.semantic_key(question, options)
        if semantic in semantic_seen:
            raise RuntimeError(f"semantic duplicate in adjudicated output: {record_id}")
        semantic_seen.add(semantic)
        lineage = candidate.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError(f"missing lineage: {record_id}")
        reviewers = [
            str(review_a["review"]["reviewer_id"]),
            str(review_b["review"]["reviewer_id"]),
        ]
        if len(set(reviewers)) != 2:
            raise RuntimeError(f"reviewer IDs are not independent: {record_id}")
        answer_counts[claim_letter] += 1
        topic_counts[str(candidate.get("quality", {}).get("topic", "unknown"))] += 1
        source_counts[str(lineage.get("asset_id", "unknown"))] += 1
        accepted.append(
            {
                "record_id": record_id,
                "task_type": "world_mc",
                "clean": {
                    "question": question,
                    "options": options,
                    "answer_letter": claim_letter,
                    "answer_text": options[claim_letter],
                },
                "review": {
                    "status": "pass",
                    "consensus_verified": True,
                    "unambiguous": True,
                    "low_risk_and_stable": True,
                    "translation_mechanical_and_answer_consistency_pass": True,
                    "review_bucket": "o2_english_mc_translated_dual_answer_blind",
                    "review_protocol": "two_independent_answer_blind_solves_plus_source_claim_agreement",
                    "source_answer_claim_role": "agreement_check_not_standalone_gold",
                    "source_answer_claim_blind_review_count": 2,
                    "source_prompt_blind_review_count": 2,
                    "minimum_blind_confidence": min(
                        float(review_a["solution"]["confidence"]),
                        float(review_b["solution"]["confidence"]),
                    ),
                    "reviewers": reviewers,
                },
                "lineage": lineage,
                "builder": {
                    "builder_sha256": builder_sha,
                    "build_fingerprint": build_fingerprint,
                    "ruleset_version": RULESET_VERSION,
                },
            }
        )

    atomic_jsonl(args.out, accepted)
    atomic_jsonl(args.ledger, ledger)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "asset_class": "D-reviewed-candidate(O2.General); NOT TRAINING DATA",
        "ruleset_version": RULESET_VERSION,
        "builder": str(builder),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "inputs": {
            name: {
                "path": str(supplied[name].resolve()),
                "rows": len(read_jsonl(supplied[name])),
                "sha256": input_hashes[name],
            }
            for name in canonical
        },
        "protocol": {
            "source_answer_claim_was_gold_before_adjudication": False,
            "required_independent_answer_blind_reviews": 2,
            "independent_source_prompt_blind_reviews": 2,
            "minimum_confidence": shared.MIN_CONFIDENCE,
            "acceptance_rule": "both reviews pass and agree; consensus equals quarantined source claim",
            "formula_language_boundary_rows": boundary_rows,
        },
        "adjudication": {
            "reviewed_rows": len(packet),
            "accepted_rows": len(accepted),
            "rejected_rows": len(packet) - len(accepted),
            "reason_counts": dict(sorted(reason_counts.items())),
            "answer_distribution": dict(sorted(answer_counts.items())),
            "topic_distribution": dict(sorted(topic_counts.items())),
            "upstream_distribution": dict(sorted(source_counts.items())),
        },
        "outputs": {
            "reviewed_candidate": {
                "path": str(args.out.resolve()),
                "rows": len(accepted),
                "bytes": args.out.stat().st_size,
                "sha256": sha256_file(args.out),
            },
            "adjudication_ledger": {
                "path": str(args.ledger.resolve()),
                "rows": len(ledger),
                "bytes": args.ledger.stat().st_size,
                "sha256": sha256_file(args.ledger),
            },
        },
        "release_gate": {
            "translation_completed": True,
            "blind_reviews_completed": True,
            "adjudication_completed": True,
            "reviewed_labels_created": True,
            "training_projection_created": False,
            "decision": "REVIEWED_CANDIDATE_ONLY_PENDING_COMBINED_SEMANTIC_SPLIT",
        },
    }
    atomic_json(args.audit, audit)
    print(stable_json(audit))


if __name__ == "__main__":
    main()
