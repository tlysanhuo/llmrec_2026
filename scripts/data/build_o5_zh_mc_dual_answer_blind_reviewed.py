#!/usr/bin/env python3
"""Adjudicate O5 Chinese MC translations after two answer-blind solves.

The source assistant answer is deliberately treated as a claim, not as gold.
A row is accepted only when two independent Chinese-prompt solvers both pass
the question, independently choose the same option with confidence >= 0.85,
and that consensus also agrees with the quarantined source answer claim.

The resulting file is a reviewed candidate asset.  It is not trainer-format
data and this builder never creates a training projection or holdout split.
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
    semantic_normalize,
    sha256_file,
    stable_json,
)


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "assets/derived/official_general"
LOGS = ROOT / "logs/data"

DEFAULT_CANDIDATE = DERIVED / "o5_en_mc_answer_claim_pilot.jsonl"
DEFAULT_PACKET = DERIVED / "o5_en_mc_zh_blind_review_packet.jsonl"
DEFAULT_CROSS_A = LOGS / "o5_zh_blind_solution_cross_a.jsonl"
DEFAULT_CROSS_B = LOGS / "o5_zh_blind_solution_cross_b.jsonl"
DEFAULT_INDEPENDENT = LOGS / "o5_zh_blind_solution_independent_c.jsonl"
DEFAULT_OUT = DERIVED / "o5_zh_mc_dual_answer_blind_reviewed_safe.jsonl"
DEFAULT_LEDGER = LOGS / "o5_zh_mc_dual_answer_blind_adjudication_ledger.jsonl"
DEFAULT_AUDIT = LOGS / "o5_zh_mc_dual_answer_blind_adjudication_audit.json"

EXPECTED_HASHES = {
    "candidate": "21c43445b2cb23ca6e46cd5c812b0ae883ab174a763fce3209baef5d76533db3",
    "packet": "8229e7c1ab51dcd1813ec0e7b85417a94d1c2911fab6af7588fa16048616dea0",
    "cross_a": "81871001d8edc4e740a1518aea8739f0cddcc1f00cd5da34f6385bce5d46131a",
    "cross_b": "b88482693b476073bec7cd22bbf62451b5ea6f9f80c4ac6d2e6e41aa8a86f943",
    "independent": "c7370647c8c878088b9adf08cf235cbe4366b10ed67cd4e8c4f853f5ef6f765a",
}
EXPECTED_PACKET_ROWS = 41
MIN_CONFIDENCE = 0.85
RULESET_VERSION = "o5-zh-mc-dual-answer-blind-adjudication-20260718-v1"

TOP_LEVEL_REVIEW_KEYS = {
    "record_id",
    "review",
    "solution",
    "verdict",
    "reason_codes",
}
REVIEW_REQUIRED_KEYS = {
    "reviewer_id",
    "blind_to_source_prompt",
    "blind_to_source_answer_claim",
    "self_contained",
    "single_correct",
    "stable",
    "low_risk",
    "visual_independent",
}
SOLUTION_KEYS = {
    "answer_letter",
    "confidence",
    "rationale",
    "alternate_correct_letters",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def require_canonical(name: str, supplied: Path, canonical: Path) -> str:
    if supplied.resolve() != canonical.resolve():
        raise RuntimeError(f"{name} must use canonical input: {canonical}")
    digest = sha256_file(supplied)
    if digest != EXPECTED_HASHES[name]:
        raise RuntimeError(f"{name} hash mismatch: {digest} != {EXPECTED_HASHES[name]}")
    return digest


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in result:
            raise ValueError(f"{label}: missing or duplicate record_id: {record_id!r}")
        result[record_id] = row
    return result


def validate_review(row: dict[str, Any], label: str, require_prompt_blind: bool) -> None:
    if set(row) != TOP_LEVEL_REVIEW_KEYS:
        raise ValueError(f"{label}: unexpected top-level schema for {row.get('record_id')}")
    review = row.get("review")
    solution = row.get("solution")
    if not isinstance(review, dict) or not REVIEW_REQUIRED_KEYS.issubset(review):
        raise ValueError(f"{label}: invalid review schema for {row['record_id']}")
    if not isinstance(solution, dict) or set(solution) != SOLUTION_KEYS:
        raise ValueError(f"{label}: invalid solution schema for {row['record_id']}")
    if review["blind_to_source_answer_claim"] is not True:
        raise ValueError(f"{label}: reviewer was not answer-claim blind")
    if require_prompt_blind and review["blind_to_source_prompt"] is not True:
        raise ValueError(f"{label}: independent reviewer was not source-prompt blind")
    if row["verdict"] not in {"pass", "reject"}:
        raise ValueError(f"{label}: invalid verdict for {row['record_id']}")
    letter = solution["answer_letter"]
    confidence = solution["confidence"]
    if letter is not None and letter not in "ABCD":
        raise ValueError(f"{label}: invalid answer letter for {row['record_id']}")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"{label}: invalid confidence for {row['record_id']}")
    alternates = solution["alternate_correct_letters"]
    if not isinstance(alternates, list) or any(x not in "ABCD" for x in alternates):
        raise ValueError(f"{label}: invalid alternates for {row['record_id']}")
    if letter in alternates:
        raise ValueError(f"{label}: chosen answer repeated in alternates")
    if row["verdict"] == "pass" and (letter is None or confidence < MIN_CONFIDENCE):
        raise ValueError(f"{label}: pass violates confidence/answer gate")


def semantic_key(question: str, options: dict[str, str]) -> str:
    payload = semantic_normalize(question) + "\0" + "\0".join(
        semantic_normalize(options[label]) for label in "ABCD"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--cross-a", type=Path, default=DEFAULT_CROSS_A)
    parser.add_argument("--cross-b", type=Path, default=DEFAULT_CROSS_B)
    parser.add_argument("--independent", type=Path, default=DEFAULT_INDEPENDENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = {
        "candidate": DEFAULT_CANDIDATE,
        "packet": DEFAULT_PACKET,
        "cross_a": DEFAULT_CROSS_A,
        "cross_b": DEFAULT_CROSS_B,
        "independent": DEFAULT_INDEPENDENT,
    }
    supplied = {
        "candidate": args.candidate,
        "packet": args.packet,
        "cross_a": args.cross_a,
        "cross_b": args.cross_b,
        "independent": args.independent,
    }
    input_hashes = {
        name: require_canonical(name, supplied[name], canonical[name])
        for name in canonical
    }

    candidate_rows = read_jsonl(args.candidate)
    packet_rows = read_jsonl(args.packet)
    cross_a_rows = read_jsonl(args.cross_a)
    cross_b_rows = read_jsonl(args.cross_b)
    independent_rows = read_jsonl(args.independent)
    if len(packet_rows) != EXPECTED_PACKET_ROWS:
        raise RuntimeError(f"expected {EXPECTED_PACKET_ROWS} packet rows, got {len(packet_rows)}")

    candidates = index_unique(candidate_rows, "candidate")
    packet = index_unique(packet_rows, "packet")
    cross_a = index_unique(cross_a_rows, "cross_a")
    cross_b = index_unique(cross_b_rows, "cross_b")
    independent = index_unique(independent_rows, "independent")
    if set(cross_a) & set(cross_b):
        raise RuntimeError("cross-review partitions overlap")
    cross = {**cross_a, **cross_b}
    packet_ids = set(packet)
    if set(cross) != packet_ids or set(independent) != packet_ids:
        raise RuntimeError("blind-review coverage does not exactly match packet")
    if not packet_ids.issubset(candidates):
        raise RuntimeError("blind packet contains IDs absent from source candidate")

    for row in cross.values():
        validate_review(row, "cross", require_prompt_blind=False)
    for row in independent.values():
        validate_review(row, "independent", require_prompt_blind=True)

    builder_path = Path(__file__).resolve()
    builder_sha = sha256_file(builder_path)
    fingerprint_payload = {
        "builder_sha256": builder_sha,
        "input_hashes": input_hashes,
        "min_confidence": MIN_CONFIDENCE,
        "ruleset_version": RULESET_VERSION,
    }
    build_fingerprint = hashlib.sha256(
        stable_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()

    accepted: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    semantic_seen: set[str] = set()

    for record_id in sorted(packet):
        packet_row = packet[record_id]
        candidate = candidates[record_id]
        cross_row = cross[record_id]
        independent_row = independent[record_id]
        translated = packet_row.get("translated")
        if not isinstance(translated, dict):
            raise ValueError(f"missing translated payload: {record_id}")
        question = str(translated.get("question", "")).strip()
        options_raw = translated.get("options")
        if not question or not isinstance(options_raw, dict) or set(options_raw) != set("ABCD"):
            raise ValueError(f"invalid translated prompt: {record_id}")
        options = {label: str(options_raw[label]).strip() for label in "ABCD"}
        if any(not value for value in options.values()):
            raise ValueError(f"empty translated option: {record_id}")
        checks = packet_row.get("mechanical_checks")
        if not isinstance(checks, dict):
            raise ValueError(f"missing packet checks: {record_id}")
        required_true = {
            "digit_multiset_match",
            "latex_command_multiset_match",
            "strict_chinese_question",
            "strict_uppercase_abcd",
        }
        required_false = {
            "char3_near_overlap",
            "eval_overlap",
            "parent_overlap",
            "reviewed_overlap",
        }
        if any(checks.get(key) is not True for key in required_true):
            raise RuntimeError(f"translated invariant failed: {record_id}")
        if any(checks.get(key) is not False for key in required_false):
            raise RuntimeError(f"translated leakage gate failed: {record_id}")

        claim = candidate.get("source_answer_claim")
        if not isinstance(claim, dict) or claim.get("status") != "source_assistant_claim_only_not_gold":
            raise ValueError(f"invalid quarantined source claim: {record_id}")
        claim_letter = claim.get("letter")
        if claim_letter not in "ABCD":
            raise ValueError(f"invalid source claim letter: {record_id}")

        cross_solution = cross_row["solution"]
        independent_solution = independent_row["solution"]
        reasons: list[str] = []
        for label, row in (("cross", cross_row), ("independent", independent_row)):
            review = row["review"]
            if row["verdict"] != "pass":
                reasons.append(f"{label}_review_reject")
            for flag in ("self_contained", "single_correct", "stable", "low_risk", "visual_independent"):
                if review.get(flag) is not True:
                    reasons.append(f"{label}_{flag}_false")
            if row["solution"]["confidence"] < MIN_CONFIDENCE:
                reasons.append(f"{label}_confidence_below_threshold")
            if row["solution"]["alternate_correct_letters"]:
                reasons.append(f"{label}_alternate_correct_letters")
        cross_letter = cross_solution["answer_letter"]
        independent_letter = independent_solution["answer_letter"]
        if cross_letter != independent_letter:
            reasons.append("blind_solution_disagreement")
        if cross_letter != claim_letter or independent_letter != claim_letter:
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
                "cross_review": cross_row,
                "independent_review": independent_row,
                "source_answer_claim": {
                    "letter": claim_letter,
                    "status": claim["status"],
                    "role": "agreement_check_not_standalone_gold",
                },
            }
        )
        if reasons:
            continue

        assert cross_letter == independent_letter == claim_letter
        key = semantic_key(question, options)
        if key in semantic_seen:
            raise RuntimeError(f"semantic duplicate in adjudicated output: {record_id}")
        semantic_seen.add(key)
        lineage = candidate.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError(f"missing lineage: {record_id}")
        source_counts[str(lineage.get("asset_id", "unknown"))] += 1
        topic_counts[str(candidate.get("quality", {}).get("topic", "unknown"))] += 1
        answer_counts[claim_letter] += 1
        reviewer_ids = [
            str(cross_row["review"]["reviewer_id"]),
            str(independent_row["review"]["reviewer_id"]),
        ]
        if len(set(reviewer_ids)) != 2:
            raise RuntimeError(f"reviewers are not independent IDs: {record_id}")
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
                    "review_bucket": "o5_english_mc_translated_dual_answer_blind",
                    "review_protocol": "two_independent_answer_blind_solves_plus_source_claim_agreement",
                    "source_answer_claim_role": "agreement_check_not_standalone_gold",
                    "source_answer_claim_blind_review_count": 2,
                    "source_prompt_blind_review_count": 1,
                    "minimum_blind_confidence": min(
                        float(cross_solution["confidence"]),
                        float(independent_solution["confidence"]),
                    ),
                    "reviewers": reviewer_ids,
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
    output_sha = sha256_file(args.out)
    ledger_sha = sha256_file(args.ledger)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "asset_class": "D-reviewed-candidate(O5); NOT TRAINING DATA",
        "ruleset_version": RULESET_VERSION,
        "builder": str(builder_path),
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
            "independent_source_prompt_blind_reviews": 1,
            "minimum_confidence": MIN_CONFIDENCE,
            "acceptance_rule": "both reviews pass and agree; consensus equals quarantined source claim",
            "cross_partition_rows": {
                "a": len(cross_a),
                "b": len(cross_b),
            },
        },
        "adjudication": {
            "reviewed_rows": len(packet_rows),
            "accepted_rows": len(accepted),
            "rejected_rows": len(packet_rows) - len(accepted),
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
                "sha256": output_sha,
            },
            "adjudication_ledger": {
                "path": str(args.ledger.resolve()),
                "rows": len(ledger),
                "bytes": args.ledger.stat().st_size,
                "sha256": ledger_sha,
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
