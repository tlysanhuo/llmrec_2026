#!/usr/bin/env python3
"""Build a quarantined O2.General English A-D answer-claim pilot.

This program scans the complete registered OneReason_General snapshot, but it
does not create training data, translated data, gold labels, short-QA rows, or
synthetic distractors.  A surviving assistant answer is retained only as a
``source_answer_claim`` for later answer-blind translation and human review.

The strict English parser, answer-claim parser, safety patterns, leakage index,
semantic dedupe, and atomic writers are reused from the tested O5 pilot.  O2's
asset/source allowlist remains local to this file; the O5 source policy is not
changed or widened.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_o5_english_mc_answer_claim_pilot as o5  # noqa: E402


base = o5.base
ROOT = o5.ROOT
O2_DIR = ROOT / "assets/official/hf_raw/OneReason_General"
O5_CANDIDATES = ROOT / "assets/derived/official_general/o5_en_mc_answer_claim_pilot.jsonl"
OUT = ROOT / "assets/derived/official_general/o2_en_mc_answer_claim_pilot.jsonl"
AUDIT = ROOT / "logs/data/o2_en_mc_answer_claim_pilot_audit.json"

O2_REVISION = "registry-snapshot-20260717"
RULESET_VERSION = "o2-en-mc-answer-claim-pilot-20260718-v1"
EXPECTED_FILES = 158
EXPECTED_ROWS = 152_005
ALLOWED_SOURCE = "stepfun_general"

DEFAULT_TOTAL_CAP = 100
DEFAULT_MATH_CAP = 40
DEFAULT_ANSWER_CAP = 25


def source_policy_reasons(prompt: str, lineage: Mapping[str, Any]) -> list[str]:
    """Apply O2's fixed source allowlist and O5's tested safety policy."""

    reasons: list[str] = []
    if lineage.get("asset_id") != "O2.General":
        reasons.append("source_asset_not_o2_general")
    if str(lineage.get("source", "")) != ALLOWED_SOURCE:
        reasons.append("source_not_stepfun_general")

    # Keep benchmark/source inspection fail-closed even though O2 metadata does
    # not currently expose an upstream dataset name.  The regexes and rejection
    # meanings are exactly the policy components exercised by the O5 tests.
    context = prompt + "\n" + base.stable_json(dict(lineage))
    checks = (
        (o5.BENCHMARK, context, "source_benchmark_family"),
        (o5.READING_CONTEXT, prompt, "reading_comprehension_context"),
        (o5.MEDICAL, prompt, "high_risk_medical"),
        (o5.LEGAL_POLITICAL, prompt, "high_risk_legal_or_political"),
        (o5.FINANCE, prompt, "risk_finance"),
        (o5.TIME_SENSITIVE, prompt, "risk_time_sensitive"),
        (o5.SUBJECTIVE, prompt, "risk_subjective"),
        (o5.PROMPT_INJECTION, prompt, "risk_prompt_injection"),
        (o5.IDENTITY, prompt, "risk_identity"),
        (o5.TOOL_MEDIA, prompt, "risk_tool_or_media"),
    )
    for pattern, value, reason in checks:
        if pattern.search(value):
            reasons.append(reason)
    return sorted(set(reasons))


def _selection_hash(record_id: str) -> str:
    payload = f"{O2_REVISION}\0{RULESET_VERSION}\0{record_id}"
    return base.hash_text(payload)


def _select_with_caps(
    rows: Sequence[dict[str, Any]], total_cap: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically select without backfill beyond any registered cap."""

    ranked = sorted(rows, key=lambda row: (row["quality"]["selection_hash"], row["record_id"]))
    selected: list[dict[str, Any]] = []
    topic_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    cap_rejections: Counter[str] = Counter()
    for row in ranked:
        topic = row["quality"]["topic"]
        answer = row["source_answer_claim"]["letter"]
        if len(selected) >= total_cap:
            cap_rejections["total_cap"] += 1
            continue
        if topic == "math_logic" and topic_counts[topic] >= DEFAULT_MATH_CAP:
            cap_rejections["topic_cap:math_logic"] += 1
            continue
        if answer_counts[answer] >= DEFAULT_ANSWER_CAP:
            cap_rejections[f"answer_cap:{answer}"] += 1
            continue
        selected.append(row)
        topic_counts[topic] += 1
        answer_counts[answer] += 1

    selected.sort(key=lambda row: row["record_id"])
    return selected, {
        "eligible_rows": len(rows),
        "selected_rows": len(selected),
        "requested_max_rows": total_cap,
        "shortfall_to_max": max(total_cap - len(selected), 0),
        "topic_counts": dict(sorted(topic_counts.items())),
        "answer_counts": dict(sorted(answer_counts.items())),
        "cap_rejections": dict(sorted(cap_rejections.items())),
        "dirty_backfill_rows": 0,
    }


def _load_o5_candidate_index(path: Path) -> tuple[base.LeakageIndex, int, str]:
    """Load only O5 candidate prompts as a required text blacklist."""

    if not path.exists():
        raise FileNotFoundError(f"required O5 candidate blacklist is missing: {path}")
    index = base.LeakageIndex()
    count = 0
    record_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                record_id = str(row["record_id"])
                original = row["original"]
                question = original["question"]
                options = original["options"]
            except Exception as exc:
                raise ValueError(f"malformed O5 candidate: {path}:{line_number}") from exc
            if record_id in record_ids:
                raise ValueError(f"duplicate O5 candidate record_id: {record_id}")
            if original.get("language") != "en" or list(options) != list("ABCD"):
                raise ValueError(f"invalid O5 candidate prompt: {path}:{line_number}")
            if not isinstance(question, str) or any(
                not isinstance(options[label], str) or not options[label].strip()
                for label in "ABCD"
            ):
                raise ValueError(f"empty O5 candidate prompt: {path}:{line_number}")
            prompt = question.strip() + "\n" + "\n".join(
                f"{label}. {options[label]}" for label in "ABCD"
            )
            index.add(prompt, path.name, include_near=True)
            record_ids.add(record_id)
            count += 1
    if count == 0:
        raise AssertionError("O5 candidate blacklist is empty")
    return index, count, base.sha256_file(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.o2_dir.resolve() != O2_DIR.resolve():
        raise RuntimeError(f"O2.General must use registered canonical asset: {O2_DIR}")
    if args.o5_candidates.resolve() != O5_CANDIDATES.resolve():
        raise RuntimeError(f"O5 candidates must use registered canonical asset: {O5_CANDIDATES}")
    base.ensure_safe_paths((args.out, args.audit))

    files = sorted(args.o2_dir.glob("*.parquet"))
    if len(files) != EXPECTED_FILES:
        raise AssertionError(f"O2.General shard signature drifted: {len(files)} != {EXPECTED_FILES}")

    import pyarrow.parquet as pq

    columns = [
        "source",
        "messages",
        "metadata",
        "uuid",
        "label",
        "images",
        "videos",
        "image",
        "video",
    ]
    total_rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(columns) - set(parquet.schema.names)
        if missing:
            raise AssertionError(f"O2.General missing columns {sorted(missing)} in {path.name}")
        total_rows += parquet.metadata.num_rows
    if total_rows != EXPECTED_ROWS:
        raise AssertionError(f"O2.General row signature drifted: {total_rows} != {EXPECTED_ROWS}")

    print("[blacklist] loading E, current parent, reviewed-29, and O5 candidate prompts", flush=True)
    eval_index = base.load_eval_index()
    train_index, train_counts = base.load_train_index()
    reviewed_index, reviewed_count = o5._load_reviewed_index()
    o5_index, o5_candidate_count, o5_candidate_sha = _load_o5_candidate_index(
        args.o5_candidates
    )
    print(
        f"[blacklist] eval={sum(eval_index.source_counts.values())} "
        f"parent={sum(train_counts.values())} reviewed={reviewed_count} "
        f"o5_candidates={o5_candidate_count}",
        flush=True,
    )

    stats: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for file_number, path in enumerate(files, 1):
        file_survivors = 0
        file_rows = pq.ParquetFile(path).metadata.num_rows
        for row_group, row_index, row in base.parquet_rows(path, columns):
            stats["rows_scanned"] += 1
            source = str(row.get("source") or "")
            stats[f"source_seen:{source}"] += 1
            if source != ALLOWED_SOURCE:
                stats["drop:source_not_stepfun_general"] += 1
                continue
            parsed_messages = base.parse_messages(row.get("messages"))
            if parsed_messages is None:
                stats["drop:messages_not_single_plain_round"] += 1
                continue
            _roles, user, assistant = parsed_messages
            if any(
                o5._media_present(row.get(field))
                for field in ("images", "videos", "image", "video")
            ):
                stats["drop:media_columns_present"] += 1
                continue

            record_id, lineage = base.source_locator(
                "O2.General",
                O2_REVISION,
                args.o2_dir,
                path,
                row_group,
                row_index,
                source,
                row.get("uuid"),
                row.get("messages"),
            )
            policy_lineage = dict(lineage)
            policy_lineage["metadata"] = row.get("metadata")
            policy_reasons = source_policy_reasons(user, policy_lineage)
            if policy_reasons:
                for reason in policy_reasons:
                    stats[f"drop:{reason}"] += 1
                continue

            parsed, parse_reasons = o5.parse_english_mc_prompt(user)
            if parsed is None:
                for reason in parse_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            language_reasons, language = o5._language_reasons(parsed)
            if language_reasons:
                for reason in language_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            claim, evidence, answer_reasons = o5.parse_english_mc_answer_claim(assistant)
            if claim is None:
                for reason in answer_reasons:
                    stats[f"drop:{reason}"] += 1
                continue

            metadata_value: Any = row.get("metadata")
            if row.get("label") is not None:
                metadata_value = {"metadata": metadata_value, "label": str(row.get("label"))}
            metadata_label, metadata_reasons = o5.extract_metadata_label(metadata_value)
            if metadata_reasons:
                for reason in metadata_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            if metadata_label is not None and metadata_label != claim:
                stats["drop:metadata_claim_mismatch"] += 1
                continue

            eval_hit, eval_modes = eval_index.match(user, parsed)
            train_hit, train_modes = train_index.match(user, parsed)
            reviewed_hit, reviewed_modes = reviewed_index.match(user, parsed)
            o5_hit, o5_modes = o5_index.match(user, parsed)
            if eval_hit or train_hit or reviewed_hit or o5_hit:
                if eval_hit:
                    stats["drop:eval_overlap"] += 1
                if train_hit:
                    stats["drop:parent_overlap"] += 1
                if reviewed_hit:
                    stats["drop:reviewed_overlap"] += 1
                if o5_hit:
                    stats["drop:o5_candidate_overlap"] += 1
                continue

            semantic = base.mc_semantic_keys(parsed, claim)
            topic = o5.topic_bucket(parsed, source)
            candidate = {
                "record_id": record_id,
                "task_type": "world_mc_translation_candidate",
                "lineage": lineage,
                "original": {
                    "language": "en",
                    "question": parsed.question,
                    "options": parsed.options,
                },
                "source_answer_claim": {
                    "status": "source_assistant_claim_only_not_gold",
                    "letter": claim,
                    "answer_text": parsed.options[claim],
                    "evidence": evidence,
                    "independent_metadata_label": metadata_label is not None,
                    "metadata_label": metadata_label,
                },
                "quality": {
                    "language": language,
                    "topic": topic,
                    "semantic": semantic,
                    "selection_hash": _selection_hash(record_id),
                    "original_eval_overlap_modes": eval_modes,
                    "original_parent_overlap_modes": train_modes,
                    "original_reviewed_overlap_modes": reviewed_modes,
                    "original_o5_candidate_overlap_modes": o5_modes,
                },
                "translation": {"status": "pending_answer_blind_translation"},
                "review": {
                    "status": "pending_two_answer_blind_reviews",
                    "gold_status": "unresolved",
                },
                "builder": {
                    "ruleset_version": RULESET_VERSION,
                    "builder_sha256": None,
                    "build_fingerprint": None,
                },
            }
            candidates.append(candidate)
            stats["pre_dedupe_eligible"] += 1
            stats[f"pre_dedupe_topic:{topic}"] += 1
            file_survivors += 1

        print(
            f"[scan {file_number:03d}/{len(files)}] {path.name} "
            f"rows={file_rows} eligible={file_survivors}",
            flush=True,
        )

    if stats["rows_scanned"] != EXPECTED_ROWS:
        raise AssertionError(f"streamed row signature drifted: {stats['rows_scanned']} != {EXPECTED_ROWS}")
    source_counts = {
        key.removeprefix("source_seen:"): value
        for key, value in stats.items()
        if key.startswith("source_seen:")
    }
    if source_counts != {ALLOWED_SOURCE: EXPECTED_ROWS}:
        raise AssertionError(f"O2.General source signature drifted: {source_counts}")

    deduped = o5._dedupe(candidates, stats)
    selected, selection_audit = _select_with_caps(deduped, args.max_candidates)

    builder_sha = base.sha256_file(Path(__file__))
    build_fingerprint = base.hash_text(
        base.stable_json(
            {
                "builder_sha256": builder_sha,
                "ruleset_version": RULESET_VERSION,
                "o2_revision": O2_REVISION,
                "o2_files": [path.name for path in files],
                "o5_candidate_sha256": o5_candidate_sha,
                "max_candidates": args.max_candidates,
                "math_cap": DEFAULT_MATH_CAP,
                "answer_cap": DEFAULT_ANSWER_CAP,
            }
        )
    )
    for row in selected:
        row["builder"]["builder_sha256"] = builder_sha
        row["builder"]["build_fingerprint"] = build_fingerprint

    ids: set[str] = set()
    invariants: set[str] = set()
    final_topics: Counter[str] = Counter()
    final_answers: Counter[str] = Counter()
    for row in selected:
        record_id = row["record_id"]
        invariant = row["quality"]["semantic"]["option_invariant_hash"]
        assert record_id not in ids
        assert invariant not in invariants
        assert row["lineage"]["asset_id"] == "O2.General"
        assert row["lineage"]["source"] == ALLOWED_SOURCE
        assert list(row["original"]["options"]) == list("ABCD")
        assert row["source_answer_claim"]["letter"] in "ABCD"
        assert row["source_answer_claim"]["status"].endswith("not_gold")
        assert row["translation"]["status"].startswith("pending")
        assert row["review"]["gold_status"] == "unresolved"
        ids.add(record_id)
        invariants.add(invariant)
        final_topics[row["quality"]["topic"]] += 1
        final_answers[row["source_answer_claim"]["letter"]] += 1
    assert len(selected) <= DEFAULT_TOTAL_CAP
    assert final_topics["math_logic"] <= DEFAULT_MATH_CAP
    assert all(final_answers[label] <= DEFAULT_ANSWER_CAP for label in "ABCD")

    base.atomic_jsonl(args.out, selected)
    output_sha = base.sha256_file(args.out)
    audit = {
        "asset_class": "D-quarantine-candidate(O2.General); NOT TRAINING DATA",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "ruleset_version": RULESET_VERSION,
        "upstream": {
            "asset_id": "O2.General",
            "revision": O2_REVISION,
            "path": str(args.o2_dir.resolve()),
            "parquet_files": len(files),
            "rows": total_rows,
            "required_source": ALLOWED_SOURCE,
        },
        "policy": {
            "scan_scope": "all registered O2.General parquet rows",
            "answer_semantics": "assistant output is source_answer_claim, never source gold",
            "allowed_source": ALLOWED_SOURCE,
            "native_input": "strict English uppercase multiline exactly-A-D MC only",
            "short_qa_created": False,
            "distractors_generated": False,
            "math_cap": DEFAULT_MATH_CAP,
            "answer_cap": DEFAULT_ANSWER_CAP,
            "max_candidates": args.max_candidates,
            "dirty_backfill": False,
        },
        "blacklist": {
            "eval_prompt_instances": sum(eval_index.source_counts.values()),
            "current_parent_prompt_instances": dict(sorted(train_counts.items())),
            "reviewed_world_candidates": reviewed_count,
            "o5_candidate_path": str(args.o5_candidates.resolve()),
            "o5_candidate_rows": o5_candidate_count,
            "o5_candidate_sha256": o5_candidate_sha,
            "selection_uses_prompt_text_only": True,
        },
        "filter_counts": dict(sorted(stats.items())),
        "pre_selection": {
            "deduped_eligible_rows": len(deduped),
            "source_counts": o5._counter_nested(deduped, "source"),
            "topic_counts": o5._counter_nested(deduped, "topic"),
            "answer_counts": o5._counter_nested(deduped, "answer"),
        },
        "selection": selection_audit,
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(selected),
            "bytes": args.out.stat().st_size,
            "sha256": output_sha,
        },
        "release_gate": {
            "candidate_only": True,
            "translation_projection_created": False,
            "training_projection_created": False,
            "gold_labels_created": False,
            "short_qa_created": False,
            "synthetic_distractors_created": False,
            "required_next_steps": [
                "create a physically answer-blind translation packet",
                "translate without source_answer_claim access",
                "run translated Chinese E/parent/reviewed leakage checks",
                "collect two independent blind solutions",
                "only then reveal and adjudicate source_answer_claim",
            ],
        },
    }
    base.atomic_json(args.audit, audit)
    print(f"[done] selected={len(selected)} output={args.out} sha256={output_sha}", flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o2-dir", type=Path, default=O2_DIR)
    parser.add_argument("--o5-candidates", type=Path, default=O5_CANDIDATES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_TOTAL_CAP)
    args = parser.parse_args()
    if not 0 <= args.max_candidates <= DEFAULT_TOTAL_CAP:
        parser.error(f"--max-candidates must be in [0,{DEFAULT_TOTAL_CAP}]")
    resolved = {
        "o5_candidates": args.o5_candidates.resolve(strict=False),
        "out": args.out.resolve(strict=False),
        "audit": args.audit.resolve(strict=False),
    }
    if resolved["out"] == resolved["audit"]:
        parser.error("--out and --audit must differ")
    if resolved["o5_candidates"] in {resolved["out"], resolved["audit"]}:
        parser.error("outputs must not overwrite the O5 candidate blacklist")
    return args


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
