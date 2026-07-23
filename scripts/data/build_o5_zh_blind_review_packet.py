#!/usr/bin/env python3
"""Merge O5 answer-blind translations and publish a Chinese blind-solve packet.

The output contains only translated Chinese prompts and mechanical checks.  It
does not contain the English source prompt, the upstream assistant response, a
source answer claim, or any gold-like field.  Both exact semantic leakage and a
Chinese char-3 near-duplicate gate run before publication.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_official_general_world_clean as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "assets/derived/official_general/o5_en_mc_translation_packet.jsonl"
PART_A = ROOT / "assets/derived/official_general/o5_en_mc_translation_part_a.jsonl"
PART_B = ROOT / "assets/derived/official_general/o5_en_mc_translation_part_b.jsonl"
OUT = ROOT / "assets/derived/official_general/o5_en_mc_zh_blind_review_packet.jsonl"
LEDGER = ROOT / "logs/data/o5_en_mc_translation_merged_ledger.jsonl"
AUDIT = ROOT / "logs/data/o5_en_mc_translation_audit.json"
OWNED_HOLDOUT = ROOT / "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl"

EXPECTED_HASHES = {
    "packet": "646e8ea49b659b02f2d3c28d9f1b5690d365ffecab79b625f99c1487df913907",
    "part_a": "8b4639df485d2162af8ba79e0e90f3c7f935b40e68b3e9e1abff0cbc6aa73117",
    "part_b": "237695b76b1b3344a52550bd189b140a4d1fc5b26ca06e78820bdca742877ae1",
}
EXPECTED_ROWS = 41
RULESET_VERSION = "o5-en-mc-zh-answer-blind-review-packet-20260718-v2"

# These translated records were exposed by the first completed baseline and
# are therefore permanently E.  The central E loader discovers OWNED_HOLDOUT
# after the first split build; excluding that single owned file from the
# historical index keeps this upstream translation stage idempotent.  We load
# the owned file separately below and fail closed if any other translated
# candidate overlaps it.
FROZEN_POST_SPLIT_TRANSLATED_HOLDOUT_IDS = frozenset(
    {
        "0db195d8394f97a34ffdd52e40f77e9915c939e513777d537ad84e1e3e56116e",
        "1bd8d393ca981d169b4cf1ff623dd677dab576659f5c0bd1e4e95a75c9c1c184",
        "24bff0f77ab267b87f7fc6e962f1d7b083e9572a01977cfc1614503d56234880",
        "29369382f410a7e83ad9c68855864d24d5e835306efd1bed5e79f6e6fd9a32aa",
        "44f890788c4c0b49d6a86b64227aaab9c8cc492f0d2f9b25d436b119dcb48b12",
        "4c5793aefe395ead28824a93243d1acd1384692b88cf6df4d587f2e7872fdded",
        "4cd93d9254b0acc79c6067be39abda4cf3e15113b47536430abf1f19fd797573",
        "6a4650fd9e4f87a4784fb5d26db85713e2760736b7053b4020bdc302c3a54ffa",
        "711277afa70f751f021e7b93b381c7d598e07f3b76ba5d0a47ebc76cc41a5c61",
        "79d8908a27df55a5074b6cd21a3fee9c69cfc98747b61dbaf1dfbbc8d8ac2a03",
        "7fa3683d9222252b4ae6672799efe25c2dc120fb32f21295ff64ce2538dab67e",
        "94f8463bdcf53980144209681b127d556e3ae29c91c2c716193faf97ea5a4ce0",
        "b25a0a2e49e21477ee6a21c28c137c292bfc1788666bef0d2c0f0ab54c12be4d",
        "ca69f3824cae85cb399910f5f5b9055f49691d3a2aa2008c27be418618d3590b",
        "db8e539cfc43d6eb3b8a6e261cf8a1398a669bc2cb883bbe0bf6539ccd257a0d",
        "de56d364e0554c5802482b1c1c62a519f558333ffea97ebd1ca1d8f555f3ef9e",
        "e9576f82ad02237f5fd8972bfa128ad24e1966ae3b017de2e3fd0db5f7edd579",
        "e9f99af759a341f65bac9d75ba55cd4426330bb531d81c70b1b52b901e2b1c20",
    }
)

FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "answer",
        "answer_letter",
        "answer_text",
        "assistant",
        "correct_answer",
        "evidence",
        "gold",
        "gold_status",
        "metadata_label",
        "original",
        "response",
        "source",
        "source_answer_claim",
        "source_prompt",
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
                raise ValueError(f"non-object row: {path}:{line_number}")
            rows.append(row)
    return rows


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def _render(question: str, options: dict[str, str]) -> str:
    return question.strip() + "\n" + "\n".join(
        f"{label}. {options[label].strip()}" for label in "ABCD"
    )


def _digit_tokens(value: str) -> Counter[str]:
    return Counter(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", value))


def _latex_commands(value: str) -> Counter[str]:
    return Counter(re.findall(r"\\[A-Za-z]+", value))


def _strip_math_spans(value: str) -> str:
    return re.sub(
        r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]",
        " ",
        value,
        flags=re.S,
    )


def _translation_checks(source: dict[str, Any], translated: dict[str, Any]) -> dict[str, Any]:
    source_text = source["question"] + "\n" + "\n".join(source["options"].values())
    translated_text = translated["question_zh"] + "\n" + "\n".join(
        translated["options_zh"].values()
    )
    source_digits = _digit_tokens(source_text)
    translated_digits = _digit_tokens(translated_text)
    source_commands = _latex_commands(source_text)
    translated_commands = _latex_commands(translated_text)
    question_prose = _strip_math_spans(translated["question_zh"])
    question_zh, question_stats = base.is_strict_zh(question_prose, min_han=6)
    outside_math = _strip_math_spans(translated_text)
    english_sentence_residue = bool(
        re.search(r"\b[A-Za-z]+(?:\s+[A-Za-z]+){3,}\b", outside_math)
    )
    return {
        "source_prompt_sha256_match": True,
        "digit_multiset_match": source_digits == translated_digits,
        "latex_command_multiset_match": source_commands == translated_commands,
        "strict_chinese_question": question_zh,
        "question_language": question_stats,
        "english_sentence_residue": english_sentence_residue,
    }


def _load_parent_near_index() -> base.LeakageIndex:
    index = base.LeakageIndex()
    paths = (
        ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl",
        ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl",
    )
    expected = {"data_seed_teacher_v1.jsonl": 32644, "data_user_residual_retention_v1.jsonl": 6106}
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompts = list(base.row_prompts(row))
                if not prompts:
                    raise ValueError(f"parent row has no prompt: {path}:{line_number}")
                for prompt in prompts:
                    counts[path.name] += 1
                    # Exact matching is already exhaustive in load_train_index.
                    # The heavier near index is needed only for MC-like parent
                    # prompts because every current candidate is strict A-D.
                    # This avoids materialising 5-grams for tens of thousands
                    # of long recommendation histories without weakening the
                    # relevant translated-MC near-duplicate gate.
                    core = base.strip_world_wrapper(prompt)
                    if len(base.broad_option_labels(core)) >= 3:
                        index.add(prompt, path.name, include_near=True)
    if dict(counts) != expected:
        raise AssertionError(f"parent near-index signature drifted: {dict(counts)} != {expected}")
    return index


def _char3_index(indexes: Iterable[base.LeakageIndex]) -> tuple[list[str], dict[str, set[int]]]:
    texts: list[str] = []
    seen: set[str] = set()
    for index in indexes:
        for value in index.semantic_texts:
            if value and value not in seen:
                seen.add(value)
                texts.append(value)
    inverted: dict[str, set[int]] = defaultdict(set)
    for text_index, value in enumerate(texts):
        for gram in base.char_ngrams(value, 3):
            inverted[gram].add(text_index)
    return texts, inverted


def _char3_near_match(prompt: str, texts: list[str], inverted: dict[str, set[int]]) -> bool:
    value = base.semantic_normalize(base.strip_world_wrapper(prompt))
    if len(value) < 12:
        return False
    grams = base.char_ngrams(value, 3)
    possible: Counter[int] = Counter()
    for gram in grams:
        possible.update(inverted.get(gram, ()))
    for text_index, intersection in possible.most_common(96):
        other = texts[text_index]
        length_ratio = min(len(value), len(other)) / max(len(value), len(other))
        if length_ratio < 0.65:
            continue
        other_grams = base.char_ngrams(other, 3)
        union = len(grams) + len(other_grams) - intersection
        if union and intersection / union >= 0.60:
            return True
        if SequenceMatcher(None, value, other, autojunk=False).ratio() >= 0.82:
            return True
    return False


def build(args: argparse.Namespace) -> dict[str, Any]:
    canonical = {"packet": PACKET, "part_a": PART_A, "part_b": PART_B}
    supplied = {"packet": args.packet, "part_a": args.part_a, "part_b": args.part_b}
    for name in canonical:
        if supplied[name].resolve() != canonical[name].resolve():
            raise RuntimeError(f"{name} must use canonical input: {canonical[name]}")
        actual_hash = base.sha256_file(supplied[name])
        if actual_hash != EXPECTED_HASHES[name]:
            raise AssertionError(f"{name} drifted: {actual_hash} != {EXPECTED_HASHES[name]}")
    base.ensure_safe_paths((args.out, args.ledger, args.audit))

    packet = _read_jsonl(args.packet)
    translations = _read_jsonl(args.part_a) + _read_jsonl(args.part_b)
    if len(packet) != EXPECTED_ROWS or len(translations) != EXPECTED_ROWS:
        raise AssertionError(
            f"translation row count drifted: packet={len(packet)} translations={len(translations)}"
        )
    packet_by_id = {row["record_id"]: row for row in packet}
    translation_by_id = {row["record_id"]: row for row in translations}
    if len(packet_by_id) != EXPECTED_ROWS or len(translation_by_id) != EXPECTED_ROWS:
        raise AssertionError("duplicate record_id in packet or translation")
    if set(packet_by_id) != set(translation_by_id):
        raise AssertionError("translation record ids do not equal packet record ids")

    print("[blacklist] loading E, current parent, reviewed-29, and parent near indexes", flush=True)
    # OWNED_HOLDOUT is a downstream output of this pipeline, not historical E
    # available when candidate selection ran.  Keep it separate so reruns do
    # not reject their own already-frozen evaluation rows.
    eval_index = base.load_eval_index(exclude_paths=(OWNED_HOLDOUT,))
    owned_holdout_index = base.LeakageIndex()
    owned_holdout_rows = 0
    owned_translated_ids: set[str] = set()
    if OWNED_HOLDOUT.exists():
        owned_rows = _read_jsonl(OWNED_HOLDOUT)
        owned_holdout_rows = len(owned_rows)
        for owned_row in owned_rows:
            owned_id = str(owned_row.get("record_id", ""))
            if owned_id in FROZEN_POST_SPLIT_TRANSLATED_HOLDOUT_IDS:
                owned_translated_ids.add(owned_id)
            prompts = list(base.row_prompts(owned_row))
            if not prompts:
                raise ValueError(f"owned holdout row has no prompt: {owned_id}")
            for prompt in prompts:
                owned_holdout_index.add(prompt, OWNED_HOLDOUT.name)
        if owned_translated_ids != FROZEN_POST_SPLIT_TRANSLATED_HOLDOUT_IDS:
            missing = sorted(FROZEN_POST_SPLIT_TRANSLATED_HOLDOUT_IDS - owned_translated_ids)
            extra = sorted(owned_translated_ids - FROZEN_POST_SPLIT_TRANSLATED_HOLDOUT_IDS)
            raise AssertionError(
                f"owned translated holdout ids drifted: missing={missing} extra={extra}"
            )
    train_index, train_counts = base.load_train_index()
    from build_o5_english_mc_answer_claim_pilot import (
        _load_reviewed_index,
        parse_english_mc_prompt,
    )

    reviewed_index, reviewed_count = _load_reviewed_index()
    parent_near_index = _load_parent_near_index()
    char3_texts, char3_inverted = _char3_index(
        (eval_index, reviewed_index, parent_near_index)
    )
    print(
        f"[blacklist] eval={sum(eval_index.source_counts.values())} "
        f"parent={sum(train_counts.values())} reviewed={reviewed_count} "
        f"char3_texts={len(char3_texts)}",
        flush=True,
    )

    ledger: list[dict[str, Any]] = []
    blind_packet: list[dict[str, Any]] = []
    translated_invariants: set[str] = set()
    stats: Counter[str] = Counter()
    for record_id in sorted(packet_by_id):
        packet_row = packet_by_id[record_id]
        translation_row = translation_by_id[record_id]
        translated = translation_row["translation"]
        reasons: list[str] = []
        if translation_row.get("source_prompt_sha256") != packet_row["source"]["prompt_sha256"]:
            reasons.append("source_prompt_sha256_mismatch")
        if translated.get("answer_fields_visible") is not False:
            reasons.append("translator_answer_isolation_not_confirmed")
        if translated.get("status") != "pass":
            reasons.append("translator_rejected")
        if list(translated.get("options_zh", {})) != list("ABCD"):
            reasons.append("translated_options_not_exact_abcd")
        checks = _translation_checks(packet_row["source"], translated)
        for key in (
            "digit_multiset_match",
            "latex_command_multiset_match",
            "strict_chinese_question",
        ):
            if not checks[key]:
                reasons.append(f"translation_check_failed:{key}")
        if checks["english_sentence_residue"]:
            reasons.append("translation_check_failed:english_sentence_residue")

        zh_prompt = _render(translated["question_zh"], translated["options_zh"])
        # This parser validates the exact uppercase multiline A-D structure
        # without treating a stem that describes two subproblems as a
        # multiple-response question.  Language is gated separately above.
        parsed, parse_reasons = parse_english_mc_prompt(zh_prompt)
        if parsed is None:
            reasons.extend(f"translated_mc_parse:{reason}" for reason in parse_reasons)
        else:
            semantic = base.mc_semantic_keys(parsed, "A")
            invariant = semantic["option_invariant_hash"]
            if invariant in translated_invariants:
                reasons.append("translated_internal_duplicate")
            eval_hit, eval_modes = eval_index.match(zh_prompt, parsed)
            owned_holdout_hit, owned_holdout_modes = owned_holdout_index.match(zh_prompt, parsed)
            train_hit, train_modes = train_index.match(zh_prompt, parsed)
            reviewed_hit, reviewed_modes = reviewed_index.match(zh_prompt, parsed)
            parent_near_hit, parent_near_modes = parent_near_index.match(zh_prompt, parsed)
            if eval_hit:
                reasons.append("translated_eval_overlap")
            if owned_holdout_hit:
                if record_id not in FROZEN_POST_SPLIT_TRANSLATED_HOLDOUT_IDS:
                    reasons.append("translated_unexpected_owned_holdout_overlap")
                else:
                    stats["post_split_owned_holdout_match:expected"] += 1
            if train_hit or parent_near_hit:
                reasons.append("translated_parent_overlap")
            if reviewed_hit:
                reasons.append("translated_reviewed_overlap")
            char3_hit = _char3_near_match(zh_prompt, char3_texts, char3_inverted)
            if char3_hit:
                reasons.append("translated_char3_near_overlap")
            leakage = {
                "eval_modes": eval_modes,
                "post_split_owned_holdout_modes": owned_holdout_modes,
                "parent_modes": sorted(set(train_modes + parent_near_modes)),
                "reviewed_modes": reviewed_modes,
                "char3_near_overlap": char3_hit,
            }
            if not reasons:
                translated_invariants.add(invariant)
        if parsed is None:
            leakage = {
                "eval_modes": [],
                "post_split_owned_holdout_modes": [],
                "parent_modes": [],
                "reviewed_modes": [],
                "char3_near_overlap": False,
            }

        status = "pass_to_blind_solution" if not reasons else "reject"
        stats[f"translation_status:{status}"] += 1
        for reason in sorted(set(reasons)):
            stats[f"drop:{reason}"] += 1
        ledger.append(
            {
                "record_id": record_id,
                "source_prompt_sha256": packet_row["source"]["prompt_sha256"],
                "source": packet_row["source"],
                "translation": translated,
                "mechanical_checks": checks,
                "translated_leakage": leakage,
                "decision": {"status": status, "reason_codes": sorted(set(reasons))},
            }
        )
        if reasons:
            continue
        blind_packet.append(
            {
                "record_id": record_id,
                "task_type": "world_mc_zh_blind_solution",
                "translated": {
                    "question": translated["question_zh"],
                    "options": translated["options_zh"],
                },
                "mechanical_checks": {
                    "strict_chinese_question": True,
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
        )

    forbidden_hits = sorted(FORBIDDEN_BLIND_KEYS & _keys(blind_packet))
    if forbidden_hits:
        raise AssertionError(f"blind packet contains forbidden keys: {forbidden_hits}")
    base.atomic_jsonl(args.ledger, ledger)
    base.atomic_jsonl(args.out, blind_packet)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "asset_class": "D-answer-blind-zh-review-packet(O5); NOT TRAINING DATA",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": base.sha256_file(Path(__file__)),
        "ruleset_version": RULESET_VERSION,
        "inputs": {
            name: {
                "path": str(supplied[name].resolve()),
                "rows": len(_read_jsonl(supplied[name])),
                "sha256": EXPECTED_HASHES[name],
            }
            for name in ("packet", "part_a", "part_b")
        },
        "blacklist": {
            "historical_eval_excludes_owned_downstream_holdout": True,
            "eval_prompt_instances": sum(eval_index.source_counts.values()),
            "eval_source_prompt_instances": dict(sorted(eval_index.source_counts.items())),
            "owned_downstream_holdout": {
                "path": str(OWNED_HOLDOUT.resolve()),
                "exists": OWNED_HOLDOUT.exists(),
                "rows": owned_holdout_rows,
                "frozen_translated_ids": len(owned_translated_ids),
                "sha256": base.sha256_file(OWNED_HOLDOUT) if OWNED_HOLDOUT.exists() else None,
            },
            "current_parent_prompt_instances": dict(sorted(train_counts.items())),
            "reviewed_world_candidates": reviewed_count,
            "char3_index_texts": len(char3_texts),
        },
        "filter_counts": dict(sorted(stats.items())),
        "answer_isolation": {
            "forbidden_blind_keys": sorted(FORBIDDEN_BLIND_KEYS),
            "forbidden_key_hits": forbidden_hits,
            "english_source_present_in_blind_packet": False,
            "source_answer_claim_visible_to_blind_reviewer": False,
        },
        "outputs": {
            "merged_translation_ledger": {
                "path": str(args.ledger.resolve()),
                "rows": len(ledger),
                "bytes": args.ledger.stat().st_size,
                "sha256": base.sha256_file(args.ledger),
            },
            "zh_blind_review_packet": {
                "path": str(args.out.resolve()),
                "rows": len(blind_packet),
                "bytes": args.out.stat().st_size,
                "sha256": base.sha256_file(args.out),
            },
        },
        "release_gate": {
            "translation_completed": len(blind_packet) == EXPECTED_ROWS,
            "blind_reviews_completed": False,
            "adjudication_completed": False,
            "gold_labels_created": False,
            "training_projection_created": False,
        },
    }
    base.atomic_json(args.audit, audit)
    print(
        f"[done] blind_packet={len(blind_packet)} rejected={EXPECTED_ROWS-len(blind_packet)} "
        f"sha256={audit['outputs']['zh_blind_review_packet']['sha256']}",
        flush=True,
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--part-a", type=Path, default=PART_A)
    parser.add_argument("--part-b", type=Path, default=PART_B)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
