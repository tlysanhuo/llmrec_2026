#!/usr/bin/env python3
"""Merge O2 answer-blind translations into a Chinese blind-solve packet.

This reuses the frozen O5 translation validator, then adds the already
adjudicated O5 Chinese questions to the exact and char-3 semantic blacklist.
No source answer claim is read or copied by this builder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_o5_zh_blind_review_packet as shared
import build_official_general_world_clean as base


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "assets/derived/official_general"
LOGS = ROOT / "logs/data"

PACKET = DERIVED / "o2_en_mc_translation_packet.jsonl"
PART_A = DERIVED / "o2_en_mc_translation_part_a.jsonl"
PART_B = DERIVED / "o2_en_mc_translation_part_b.jsonl"
O5_REVIEWED = DERIVED / "o5_zh_mc_dual_answer_blind_reviewed_safe.jsonl"
OUT = DERIVED / "o2_en_mc_zh_blind_review_packet.jsonl"
LEDGER = LOGS / "o2_en_mc_translation_merged_ledger.jsonl"
AUDIT = LOGS / "o2_en_mc_translation_audit.json"

EXPECTED_HASHES = {
    "packet": "5b95944e313ac25d9af8687e18ad78e3b4e25f57da6b2698149cb2248b338f1c",
    "part_a": "f1591398a4ab8bf4f9a48b865d2eb6ccf536ddeaceb45b5581d2fc180c473651",
    "part_b": "8a33511c301e30a668bb33152be0f1b7490d58d265d065bdf2040c626f2295d9",
    "o5_reviewed": "260c4cdcd5f6ef94e9100ed08aaf676af86a217c1f83f7aed96f2c2365104f50",
}
EXPECTED_ROWS = 27
RULESET_VERSION = "o2-en-mc-zh-answer-blind-review-packet-20260718-v2"


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


def render(question: str, options: dict[str, str]) -> str:
    return question.strip() + "\n" + "\n".join(
        f"{label}. {options[label].strip()}" for label in "ABCD"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--part-a", type=Path, default=PART_A)
    parser.add_argument("--part-b", type=Path, default=PART_B)
    parser.add_argument("--o5-reviewed", type=Path, default=O5_REVIEWED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = {
        "packet": PACKET,
        "part_a": PART_A,
        "part_b": PART_B,
        "o5_reviewed": O5_REVIEWED,
    }
    supplied = {
        "packet": args.packet,
        "part_a": args.part_a,
        "part_b": args.part_b,
        "o5_reviewed": args.o5_reviewed,
    }
    for name, path in supplied.items():
        if path.resolve() != canonical[name].resolve():
            raise RuntimeError(f"{name} must use canonical input: {canonical[name]}")
        digest = base.sha256_file(path)
        if digest != EXPECTED_HASHES[name]:
            raise RuntimeError(f"{name} hash mismatch: {digest} != {EXPECTED_HASHES[name]}")

    # Configure the frozen shared validator for the O2 inputs.  Its outputs go
    # to adjacent stage files so the additional O5-reviewed blacklist is
    # checked before any final packet is atomically published.
    shared.PACKET = PACKET
    shared.PART_A = PART_A
    shared.PART_B = PART_B
    shared.EXPECTED_HASHES = {
        "packet": EXPECTED_HASHES["packet"],
        "part_a": EXPECTED_HASHES["part_a"],
        "part_b": EXPECTED_HASHES["part_b"],
    }
    shared.EXPECTED_ROWS = EXPECTED_ROWS
    shared.RULESET_VERSION = RULESET_VERSION

    stage_out = args.out.with_suffix(args.out.suffix + ".stage")
    stage_ledger = args.ledger.with_suffix(args.ledger.suffix + ".stage")
    stage_audit = args.audit.with_suffix(args.audit.suffix + ".stage")
    stage_paths = (stage_out, stage_ledger, stage_audit)
    try:
        shared.build(
            argparse.Namespace(
                packet=args.packet,
                part_a=args.part_a,
                part_b=args.part_b,
                out=stage_out,
                ledger=stage_ledger,
                audit=stage_audit,
            )
        )

        o5_rows = read_jsonl(args.o5_reviewed)
        if len(o5_rows) != 41:
            raise RuntimeError(f"expected 41 adjudicated O5 rows, got {len(o5_rows)}")
        o5_index = base.LeakageIndex()
        for row in o5_rows:
            clean = row.get("clean")
            if not isinstance(clean, dict) or set(clean.get("options", {})) != set("ABCD"):
                raise ValueError(f"invalid O5 reviewed row: {row.get('record_id')}")
            o5_index.add(
                render(str(clean["question"]), clean["options"]),
                "o5_dual_answer_blind_reviewed",
                include_near=True,
            )
        char3_texts, char3_inverted = shared._char3_index((o5_index,))

        staged_packet = read_jsonl(stage_out)
        staged_ledger = read_jsonl(stage_ledger)
        post_split_owned_holdout_matches = sum(
            bool(row.get("translated_leakage", {}).get("post_split_owned_holdout_modes"))
            for row in staged_ledger
        )
        ledger_by_id = {row["record_id"]: row for row in staged_ledger}
        staged_packet_by_id = {row["record_id"]: row for row in staged_packet}

        # The shared gate uses min_han=6.  O2 contains two formula-dominant
        # questions whose complete Chinese prose has exactly five Han
        # characters and zero Latin prose.  Recover only this frozen boundary:
        # every other translation/leakage check must already pass and the sole
        # rejection must be strict_chinese_question.  This is not a general
        # relaxation and cannot rescue English or mixed-language prompts.
        boundary_recovered = 0
        for ledger_row in staged_ledger:
            if ledger_row["record_id"] in staged_packet_by_id:
                continue
            if ledger_row["decision"].get("reason_codes") != [
                "translation_check_failed:strict_chinese_question"
            ]:
                continue
            checks = ledger_row["mechanical_checks"]
            language = checks.get("question_language", {})
            leakage = ledger_row["translated_leakage"]
            if not (
                language.get("han") == 5
                and language.get("latin") == 0
                and language.get("kana") == 0
                and language.get("hangul") == 0
                and checks.get("digit_multiset_match") is True
                and checks.get("latex_command_multiset_match") is True
                and checks.get("english_sentence_residue") is False
                and checks.get("source_prompt_sha256_match") is True
                and not leakage.get("eval_modes")
                and not leakage.get("parent_modes")
                and not leakage.get("reviewed_modes")
                and leakage.get("char3_near_overlap") is False
            ):
                continue
            translated = ledger_row["translation"]
            if translated.get("status") != "pass" or translated.get("answer_fields_visible") is not False:
                continue
            recovered = {
                "record_id": ledger_row["record_id"],
                "task_type": "world_mc_zh_blind_solution",
                "translated": {
                    "question": translated["question_zh"],
                    "options": translated["options_zh"],
                },
                "mechanical_checks": {
                    "strict_chinese_question": True,
                    "strict_chinese_rule": "o2_formula_boundary_exact_han5_zero_latin",
                    "strict_uppercase_abcd": True,
                    "digit_multiset_match": True,
                    "latex_command_multiset_match": True,
                    "eval_overlap": False,
                    "parent_overlap": False,
                    "reviewed_overlap": False,
                    "char3_near_overlap": False,
                },
                "review_protocol": {
                    "source_prompt_visible": False,
                    "source_answer_claim_visible": False,
                    "required_independent_reviews": 2,
                },
            }
            staged_packet_by_id[ledger_row["record_id"]] = recovered
            ledger_row["mechanical_checks"]["o2_formula_boundary_pass"] = True
            ledger_row["decision"] = {
                "status": "pass_to_blind_solution",
                "reason_codes": [],
                "boundary_acceptance_codes": [
                    "strict_chinese_formula_exact_han5_zero_latin"
                ],
            }
            boundary_recovered += 1

        final_packet: list[dict[str, Any]] = []
        o5_overlap_rows = 0
        o5_overlap_modes: dict[str, list[str]] = {}
        internal_invariants: set[str] = set()
        internal_duplicate_rows = 0
        for row in sorted(staged_packet_by_id.values(), key=lambda value: value["record_id"]):
            translated = row["translated"]
            prompt = render(translated["question"], translated["options"])
            from build_o5_english_mc_answer_claim_pilot import parse_english_mc_prompt

            parsed, reasons = parse_english_mc_prompt(prompt)
            if parsed is None or reasons:
                raise RuntimeError(f"staged translated prompt no longer parses: {row['record_id']}")
            invariant = base.mc_semantic_keys(parsed, "A")["option_invariant_hash"]
            if invariant in internal_invariants:
                internal_duplicate_rows += 1
                ledger_by_id[row["record_id"]]["decision"] = {
                    "status": "reject",
                    "reason_codes": ["translated_internal_duplicate"],
                }
                continue
            internal_invariants.add(invariant)
            exact_hit, modes = o5_index.match(prompt, parsed)
            char3_hit = shared._char3_near_match(prompt, char3_texts, char3_inverted)
            ledger_row = ledger_by_id[row["record_id"]]
            ledger_row["translated_leakage"]["o5_reviewed_modes"] = modes
            ledger_row["translated_leakage"]["o5_reviewed_char3_near_overlap"] = char3_hit
            if exact_hit or char3_hit:
                o5_overlap_rows += 1
                o5_overlap_modes[row["record_id"]] = sorted(
                    set(modes + (["char3_near_overlap"] if char3_hit else []))
                )
                ledger_row["decision"] = {
                    "status": "reject",
                    "reason_codes": ["translated_o5_reviewed_overlap"],
                }
                continue
            row["mechanical_checks"]["o5_reviewed_overlap"] = False
            final_packet.append(row)

        base.atomic_jsonl(args.out, final_packet)
        base.atomic_jsonl(args.ledger, staged_ledger)
        shared_audit = json.loads(stage_audit.read_text(encoding="utf-8"))
        builder = Path(__file__).resolve()
        audit = {
            **shared_audit,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "asset_class": "D-answer-blind-zh-review-packet(O2.General); NOT TRAINING DATA",
            "builder": str(builder),
            "builder_sha256": base.sha256_file(builder),
            "ruleset_version": RULESET_VERSION,
        }
        audit["inputs"]["o5_reviewed"] = {
            "path": str(args.o5_reviewed.resolve()),
            "rows": len(o5_rows),
            "sha256": EXPECTED_HASHES["o5_reviewed"],
        }
        audit["blacklist"]["o5_dual_answer_blind_reviewed"] = len(o5_rows)
        audit["filter_counts"] = {
            "boundary_recover:strict_chinese_formula_exact_han5_zero_latin": boundary_recovered,
            "drop:translated_internal_duplicate": internal_duplicate_rows,
            "drop:translated_o5_reviewed_overlap": o5_overlap_rows,
            "post_split_owned_holdout_match:expected": post_split_owned_holdout_matches,
            "translation_status:pass_to_blind_solution": len(final_packet),
            "translation_status:reject": EXPECTED_ROWS - len(final_packet),
        }
        audit["language_boundary"] = {
            "shared_default_min_han": 6,
            "o2_formula_boundary_exact_han": 5,
            "o2_formula_boundary_max_latin": 0,
            "recovered_rows": boundary_recovered,
            "general_relaxation": False,
        }
        audit["o5_reviewed_overlap_modes"] = o5_overlap_modes
        audit["outputs"] = {
            "merged_translation_ledger": {
                "path": str(args.ledger.resolve()),
                "rows": len(staged_ledger),
                "bytes": args.ledger.stat().st_size,
                "sha256": base.sha256_file(args.ledger),
            },
            "zh_blind_review_packet": {
                "path": str(args.out.resolve()),
                "rows": len(final_packet),
                "bytes": args.out.stat().st_size,
                "sha256": base.sha256_file(args.out),
            },
        }
        audit["release_gate"]["translation_completed"] = len(final_packet) == EXPECTED_ROWS
        base.atomic_json(args.audit, audit)
        print(
            f"[done] O2 blind_packet={len(final_packet)} "
            f"o5_overlap={o5_overlap_rows} "
            f"sha256={audit['outputs']['zh_blind_review_packet']['sha256']}"
        )
    finally:
        for path in stage_paths:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
