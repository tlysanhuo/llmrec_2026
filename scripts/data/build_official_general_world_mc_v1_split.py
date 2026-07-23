#!/usr/bin/env python3
"""Create a permanent holdout and unique-only train projection from reviewed MC.

Inputs are three frozen, fully reviewed official-General cohorts.  The split is
deterministic, cohort- and answer-stratified, and fails closed on exact or near
semantic duplicates.  Holdout rows are physically separated under the E asset
tree and expose a top-level input prompt so future leakage indexes discover
them automatically.  This builder does not create a training mix, config, or
task and does not repeat any training question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import build_official_general_world_clean as base
from build_official_general_world_clean import (
    atomic_json,
    atomic_jsonl,
    char_ngrams,
    semantic_normalize,
    sha256_file,
    stable_json,
)


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "assets/derived/official_general"
PROCESSED = ROOT / "assets/derived/processed"
EVAL = ROOT / "assets/evaluation/holdout"
LOGS = ROOT / "logs/data"

LEGACY = DERIVED / "world_mc_human_reviewed_safe.jsonl"
O5 = DERIVED / "o5_zh_mc_dual_answer_blind_reviewed_safe.jsonl"
O2 = DERIVED / "o2_zh_mc_dual_answer_blind_reviewed_safe.jsonl"
O5_CANDIDATES = DERIVED / "o5_en_mc_answer_claim_pilot.jsonl"
O2_CANDIDATES = DERIVED / "o2_en_mc_answer_claim_pilot.jsonl"
TRAIN_REVIEWED = DERIVED / "official_general_world_mc_v1_train_reviewed.jsonl"
TRAIN_PROJECTION = PROCESSED / "data_official_general_world_mc_v1.jsonl"
HOLDOUT = EVAL / "official_general_world_mc_v1_holdout.jsonl"
AUDIT = LOGS / "official_general_world_mc_v1_split_audit.json"

EXPECTED_INPUTS = {
    "legacy_zh_reviewed": {
        "path": LEGACY,
        "rows": 29,
        "sha256": "75a4249824272ab3afe8255a18ad0a8e13e6d9ccf1e10e69233185082dc1fd74",
    },
    "o5_translated_dual_reviewed": {
        "path": O5,
        "rows": 41,
        "sha256": "260c4cdcd5f6ef94e9100ed08aaf676af86a217c1f83f7aed96f2c2365104f50",
    },
    "o2_translated_dual_reviewed": {
        "path": O2,
        "rows": 27,
        "sha256": "f2cd3538afb80c4237e3a760e831b5308b8890c1c864950d820b70ce4f27df98",
    },
}
TOPIC_INPUTS = {
    "o5_translated_dual_reviewed": {
        "path": O5_CANDIDATES,
        "rows": 51,
        "sha256": "21c43445b2cb23ca6e46cd5c812b0ae883ab174a763fce3209baef5d76533db3",
    },
    "o2_translated_dual_reviewed": {
        "path": O2_CANDIDATES,
        "rows": 38,
        "sha256": "6a78709e887feb0ae034c2392e9d3117abe28ff566365f65c5623dd87a1ab1cf",
    },
}
EXPECTED_REVIEWED_TOTAL = 97
EXPECTED_SPLIT_POOL_TOTAL = 93
EXPECTED_CLEAN_D_OR_SCORING_TOTAL = 91
STRATIFIED_HOLDOUT_TARGET_ROWS = 24
FINAL_HOLDOUT_ROWS = 25
FINAL_SCORING_HOLDOUT_ROWS = 23
FINAL_TRAIN_ROWS = 68
MIN_TRAIN_ROWS = 32
MIN_HOLDOUT_ROWS = 16
SEED = 20260718
RULESET_VERSION = "official-general-world-mc-v1-semantic-split-20260718-v3"

# The first 24-row baseline was completed before the eligible-cohort quota bug
# was found.  Every exposed row remains permanent E even if the corrected
# stratification would select a different row.  After all E and parent
# quarantines, correct stratification adds exactly one further O5 row,
# producing a 25-row final holdout.
FROZEN_BASELINE_HOLDOUT_IDS = frozenset(
    {
        "0aa0dd2dcd4894eeacd2311ed05e3800f6c851f3ec1305424106aaacc15235cb",
        "0db195d8394f97a34ffdd52e40f77e9915c939e513777d537ad84e1e3e56116e",
        "1bd8d393ca981d169b4cf1ff623dd677dab576659f5c0bd1e4e95a75c9c1c184",
        "24bff0f77ab267b87f7fc6e962f1d7b083e9572a01977cfc1614503d56234880",
        "29369382f410a7e83ad9c68855864d24d5e835306efd1bed5e79f6e6fd9a32aa",
        "3f4a84c7f2608df14ab761c64d7e17ff3fc19d67fad8f1dd01ab366dcca9f405",
        "4c5793aefe395ead28824a93243d1acd1384692b88cf6df4d587f2e7872fdded",
        "4cd93d9254b0acc79c6067be39abda4cf3e15113b47536430abf1f19fd797573",
        "625d8c304bfc8ff381842ff1cc6712520099be9a1191c664021782e283c0449e",
        "6a4650fd9e4f87a4784fb5d26db85713e2760736b7053b4020bdc302c3a54ffa",
        "711277afa70f751f021e7b93b381c7d598e07f3b76ba5d0a47ebc76cc41a5c61",
        "79d8908a27df55a5074b6cd21a3fee9c69cfc98747b61dbaf1dfbbc8d8ac2a03",
        "7fa3683d9222252b4ae6672799efe25c2dc120fb32f21295ff64ce2538dab67e",
        "94f8463bdcf53980144209681b127d556e3ae29c91c2c716193faf97ea5a4ce0",
        "9f327cd1a9a39706ef559c7c76f6b442c326bc20040cfb7f9e3cb8758f5315a7",
        "b25a0a2e49e21477ee6a21c28c137c292bfc1788666bef0d2c0f0ab54c12be4d",
        "bcf7cb936beb188990be3e7d1c3a8ea491e71969f8653e7816a4d4cc46a82029",
        "c718c99cc09443fb692dac2829bad87edc0ee842244c3889e02a3261e3b2b65a",
        "ca69f3824cae85cb399910f5f5b9055f49691d3a2aa2008c27be418618d3590b",
        "d4e27d1a9b8a22b59db0025a62c36e67d2264489b96e1848df4c66942f7a32ae",
        "db8e539cfc43d6eb3b8a6e261cf8a1398a669bc2cb883bbe0bf6539ccd257a0d",
        "de56d364e0554c5802482b1c1c62a519f558333ffea97ebd1ca1d8f555f3ef9e",
        "e9576f82ad02237f5fd8972bfa128ad24e1966ae3b017de2e3fd0db5f7edd579",
        "e9f99af759a341f65bac9d75ba55cd4426330bb531d81c70b1b52b901e2b1c20",
    }
)
EXPECTED_CORRECTION_ADDITIONS = frozenset(
    {"44f890788c4c0b49d6a86b64227aaab9c8cc492f0d2f9b25d436b119dcb48b12"}
)

# The central E blacklist reverse-check found two rows from the earlier
# reviewed-29 cohort in CMMLU test.  They are quarantined completely: moving
# them into the newly created holdout would not make them eligible D data.
EXCLUDED_EXISTING_E = {
    "0071de1176388ba8299d6547f029e281c87ee51cad7b9693a5a37d3a0c04eff7": {
        "source": "cmmlu:test/elementary_chinese.csv",
        "modes": ["core_exact", "option_invariant_exact", "ordered_exact", "stem_exact"],
    },
    "1b31bdff62136dbbf42851b4cca6235ae05d609fdd85e15044104f738e6e814c": {
        "source": "cmmlu:test/elementary_chinese.csv",
        "modes": ["stem_exact"],
    },
}

# A final reverse check also found two legacy reviewed rows already present in
# the current parent retention asset.  They carry no new information and are
# quarantined from both D and the new E asset; source locations are frozen so
# the audit can re-confirm the overlap on every build.
EXCLUDED_EXISTING_PARENT = {
    "1f9dbba0a0cb8b5e15e85d7c2b3e5a83a82eb73cc287ccd36d138ade45570f20": {
        "source": "data_user_residual_retention_v1.jsonl",
        "line_number": 1554,
        "parent_prompt_sha256": "4589e68765e98cee3469775ed9638301e7eda61e39e45f0004fd20661fc96415",
    },
    "ea144be5b108391f941cca474a25bea8c32392dfb7e06e247c92ea6f228f2fa9": {
        "source": "data_user_residual_retention_v1.jsonl",
        "line_number": 3077,
        "parent_prompt_sha256": "db407cd71524b56842b0ea20cff73b27d460642aae13fb2ef2414c897ec1fdf6",
    },
}

# These two rows were already exposed in the completed first baseline, so they
# remain permanent E.  A later full holdout-vs-parent reverse check found exact
# copies in the current parent retention asset.  They can never return to D and
# are not eligible for checkpoint selection or evaluation aggregation.
PARENT_CONTAMINATED_PERMANENT_E = {
    "d4e27d1a9b8a22b59db0025a62c36e67d2264489b96e1848df4c66942f7a32ae": {
        "source": "data_user_residual_retention_v1.jsonl",
        "line_number": 3022,
        "parent_prompt_sha256": "323d0e9524936852f627942a9e5515bb6866de23057a8419cacc8f1bdf2bac03",
        "contamination_reason": "current_parent_exact",
    },
    "9f327cd1a9a39706ef559c7c76f6b442c326bc20040cfb7f9e3cb8758f5315a7": {
        "source": "data_user_residual_retention_v1.jsonl",
        "line_number": 5362,
        "parent_prompt_sha256": "22f239803b271e7429786c2c79e119fccc734c44d85b80334a43049e8a18a4c8",
        "contamination_reason": "current_parent_exact",
    },
}

SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
PROMPT_HEAD = "请回答以下问题：\n\n"
PROMPT_TAIL = '\n\n请按以下格式作答："正确答案是 (在此处填写选项字母)"/no_think'
RESPONSE_PREFIX = "<think>\n\n</think>\n\n\n正确答案是 ("


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


def file_manifest(paths: list[Path]) -> dict[str, Any]:
    """Content-address every file used by a reverse leakage gate."""
    entries: list[dict[str, Any]] = []
    for path in sorted({value.resolve() for value in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(f"required gate input is missing: {path}")
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            relative = str(path)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(stable_json(entries).encode("utf-8")).hexdigest()
    return {"files": len(entries), "manifest_sha256": digest, "entries": entries}


def historical_eval_manifest_paths() -> list[Path]:
    """Mirror the central E loader while excluding only our owned output."""
    paths = [
        ROOT / "assets/evaluation/visible/懂世界.jsonl",
        ROOT / "assets/evaluation/offline_eval/dev_world.jsonl",
        ROOT / "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl",
        ROOT / "scripts/eval/precheck.py",
        ROOT / "assets/evaluation/offline_eval/_cmmlu.zip",
    ]
    paths.extend(
        path
        for path in sorted((ROOT / "assets/evaluation/holdout").glob("*.jsonl"))
        if path.resolve() != HOLDOUT.resolve()
    )
    paths.extend(sorted((ROOT / "assets/evaluation/offline_eval/_ceval_val").glob("*.parquet")))
    paths.extend(sorted((ROOT / "logs/eval").glob("*.log")))
    return paths


def parent_manifest_paths() -> list[Path]:
    return [
        ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl",
        ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl",
    ]


def render_stem(clean: dict[str, Any]) -> str:
    return str(clean["question"]).strip() + "\n" + "\n".join(
        f"{label}. {str(clean['options'][label]).strip()}" for label in "ABCD"
    )


def trainer_row(row: dict[str, Any]) -> dict[str, Any]:
    clean = row["clean"]
    return {
        "instruction": SYSTEM,
        "input": PROMPT_HEAD + render_stem(clean) + PROMPT_TAIL,
        "output": RESPONSE_PREFIX + clean["answer_letter"] + ")",
        "history": [],
    }


def stable_split_hash(cohort: str, record_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{cohort}\0{record_id}".encode("utf-8")).hexdigest()


def allocate(total: int, counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder proportional integer allocation with stable ties."""
    population = sum(counts.values())
    if population <= 0 or total < 0 or total > population:
        raise ValueError(f"invalid allocation total/population: {total}/{population}")
    raw = {key: total * value / population for key, value in counts.items()}
    result = {key: min(counts[key], math.floor(raw[key])) for key in counts}
    remaining = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - math.floor(raw[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if result[key] >= counts[key]:
                continue
            result[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("allocation could not satisfy target")
    return result


def semantic_text(row: dict[str, Any]) -> str:
    return semantic_normalize(render_stem(row["clean"]))


def structured_parsed(row: dict[str, Any]) -> base.ParsedMC:
    """Use the reviewed structure for leakage keys, not heuristic acceptance.

    Some valid single-answer questions describe two sub-scenarios inside the
    stem.  The raw-candidate parser conservatively flags that prose as possible
    multi-select, but the reviewed asset already supplies one A-D answer and
    four validated options.  Constructing ParsedMC directly preserves every
    exact/stem/option-invariant leakage mode without weakening the gate.
    """
    clean = row["clean"]
    return base.ParsedMC(
        question=str(clean["question"]).strip(),
        options={label: str(clean["options"][label]).strip() for label in "ABCD"},
        postscript="",
    )


def near_duplicate(a: str, b: str) -> tuple[bool, dict[str, float]]:
    if not a or not b:
        return False, {"char3_jaccard": 0.0, "sequence_ratio": 0.0, "length_ratio": 0.0}
    length_ratio = min(len(a), len(b)) / max(len(a), len(b))
    grams_a = char_ngrams(a, 3)
    grams_b = char_ngrams(b, 3)
    union = len(grams_a | grams_b)
    jaccard = len(grams_a & grams_b) / max(union, 1)
    sequence = SequenceMatcher(None, a, b, autojunk=False).ratio()
    containment = length_ratio >= 0.70 and (a in b or b in a)
    hit = jaccard >= 0.60 or sequence >= 0.82 or containment
    return hit, {
        "char3_jaccard": jaccard,
        "sequence_ratio": sequence,
        "length_ratio": length_ratio,
    }


def validate_reviewed(row: dict[str, Any], cohort: str) -> None:
    if row.get("task_type") != "world_mc":
        raise ValueError(f"{cohort}: invalid task_type for {row.get('record_id')}")
    clean = row.get("clean")
    review = row.get("review")
    lineage = row.get("lineage")
    if not isinstance(clean, dict) or not isinstance(review, dict) or not isinstance(lineage, dict):
        raise ValueError(f"{cohort}: incomplete reviewed row {row.get('record_id')}")
    if not str(clean.get("question", "")).strip():
        raise ValueError(f"{cohort}: empty question")
    options = clean.get("options")
    if not isinstance(options, dict) or set(options) != set("ABCD"):
        raise ValueError(f"{cohort}: invalid options")
    if any(not str(options[label]).strip() for label in "ABCD"):
        raise ValueError(f"{cohort}: empty option")
    answer = clean.get("answer_letter")
    if answer not in "ABCD" or clean.get("answer_text") != options[answer]:
        raise ValueError(f"{cohort}: invalid reviewed answer")
    if review.get("status") != "pass":
        raise ValueError(f"{cohort}: non-pass reviewed row")
    for key in ("unambiguous", "low_risk_and_stable"):
        if review.get(key) is not True:
            raise ValueError(f"{cohort}: review flag {key} is not true")
    if cohort == "legacy_zh_reviewed":
        if review.get("factual_correct") is not True:
            raise ValueError(f"{cohort}: legacy factual review is not true")
    else:
        for key in (
            "consensus_verified",
            "translation_mechanical_and_answer_consistency_pass",
        ):
            if review.get(key) is not True:
                raise ValueError(f"{cohort}: review flag {key} is not true")
        if review.get("source_answer_claim_role") != "agreement_check_not_standalone_gold":
            raise ValueError(f"{cohort}: source claim role is not safely scoped")
        reviewers = review.get("reviewers")
        if not isinstance(reviewers, list) or len(reviewers) != 2 or len(set(reviewers)) != 2:
            raise ValueError(f"{cohort}: expected two distinct answer-blind reviewers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=LEGACY)
    parser.add_argument("--o5", type=Path, default=O5)
    parser.add_argument("--o2", type=Path, default=O2)
    parser.add_argument("--train-reviewed", type=Path, default=TRAIN_REVIEWED)
    parser.add_argument("--train-projection", type=Path, default=TRAIN_PROJECTION)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.ensure_safe_paths(
        (args.train_reviewed, args.train_projection, args.holdout, args.audit)
    )
    supplied = {
        "legacy_zh_reviewed": args.legacy,
        "o5_translated_dual_reviewed": args.o5,
        "o2_translated_dual_reviewed": args.o2,
    }
    input_paths = {path.resolve() for path in supplied.values()}
    output_paths = {
        path.resolve()
        for path in (args.train_reviewed, args.train_projection, args.holdout, args.audit)
    }
    if input_paths & output_paths:
        raise RuntimeError("an output path would overwrite a reviewed input")

    topic_by_cohort: dict[str, dict[str, str]] = {}
    for cohort, spec in TOPIC_INPUTS.items():
        path = spec["path"]
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"{cohort} topic input hash mismatch: {digest} != {spec['sha256']}")
        topic_rows = read_jsonl(path)
        if len(topic_rows) != spec["rows"]:
            raise RuntimeError(
                f"{cohort} topic input row mismatch: {len(topic_rows)} != {spec['rows']}"
            )
        mapping: dict[str, str] = {}
        for row in topic_rows:
            record_id = str(row.get("record_id", ""))
            quality = row.get("quality")
            topic = quality.get("topic") if isinstance(quality, dict) else None
            if not record_id or topic not in {"math_logic", "other_general"}:
                raise ValueError(f"{cohort}: invalid topic metadata for {record_id}")
            if record_id in mapping:
                raise ValueError(f"{cohort}: duplicate topic record_id {record_id}")
            mapping[record_id] = str(topic)
        topic_by_cohort[cohort] = mapping

    cohorts: dict[str, list[dict[str, Any]]] = {}
    for cohort, spec in EXPECTED_INPUTS.items():
        path = supplied[cohort]
        if path.resolve() != spec["path"].resolve():
            raise RuntimeError(f"{cohort} must use canonical input: {spec['path']}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(f"{cohort} hash mismatch: {digest} != {spec['sha256']}")
        rows = read_jsonl(path)
        if len(rows) != spec["rows"]:
            raise RuntimeError(f"{cohort} row mismatch: {len(rows)} != {spec['rows']}")
        for row in rows:
            validate_reviewed(row, cohort)
            row["_split_cohort"] = cohort
            row["_split_hash"] = stable_split_hash(cohort, str(row["record_id"]))
            if cohort == "legacy_zh_reviewed":
                row["_split_topic"] = "legacy_unclassified"
            else:
                try:
                    row["_split_topic"] = topic_by_cohort[cohort][str(row["record_id"])]
                except KeyError as exc:
                    raise RuntimeError(
                        f"{cohort}: reviewed row absent from frozen topic input: {row['record_id']}"
                    ) from exc
        cohorts[cohort] = rows

    reviewed = [row for rows in cohorts.values() for row in rows]
    if len(reviewed) != EXPECTED_REVIEWED_TOTAL:
        raise RuntimeError(
            f"combined reviewed row mismatch: {len(reviewed)} != {EXPECTED_REVIEWED_TOTAL}"
        )
    record_ids = [str(row["record_id"]) for row in reviewed]
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError("record_id collision across reviewed cohorts")
    if not set(EXCLUDED_EXISTING_E).issubset(record_ids):
        raise RuntimeError("frozen existing-E exclusion IDs are absent from reviewed inputs")
    if not set(EXCLUDED_EXISTING_PARENT).issubset(record_ids):
        raise RuntimeError("frozen existing-parent exclusion IDs are absent from reviewed inputs")
    quarantine_ids = set(EXCLUDED_EXISTING_E) | set(EXCLUDED_EXISTING_PARENT)
    if set(EXCLUDED_EXISTING_E) & set(EXCLUDED_EXISTING_PARENT):
        raise RuntimeError("E and parent quarantine IDs unexpectedly overlap")
    combined = [row for row in reviewed if str(row["record_id"]) not in quarantine_ids]
    if len(combined) != EXPECTED_SPLIT_POOL_TOTAL:
        raise RuntimeError(
            f"split-pool row mismatch: {len(combined)} != {EXPECTED_SPLIT_POOL_TOTAL}"
        )

    semantic = {str(row["record_id"]): semantic_text(row) for row in combined}
    if len(set(semantic.values())) != len(semantic):
        raise RuntimeError("exact semantic duplicate across reviewed cohorts")
    near_pairs: list[dict[str, Any]] = []
    max_jaccard = 0.0
    max_sequence = 0.0
    for index, row in enumerate(combined):
        rid = str(row["record_id"])
        for other in combined[:index]:
            oid = str(other["record_id"])
            hit, metrics = near_duplicate(semantic[rid], semantic[oid])
            max_jaccard = max(max_jaccard, metrics["char3_jaccard"])
            max_sequence = max(max_sequence, metrics["sequence_ratio"])
            if hit:
                near_pairs.append({"record_id_a": rid, "record_id_b": oid, **metrics})
    if near_pairs:
        raise RuntimeError(f"near semantic duplicates require cluster adjudication: {near_pairs[:5]}")

    eligible_by_cohort = {
        cohort: [row for row in combined if row["_split_cohort"] == cohort]
        for cohort in cohorts
    }
    cohort_counts = {cohort: len(rows) for cohort, rows in eligible_by_cohort.items()}
    cohort_holdout_quota = allocate(STRATIFIED_HOLDOUT_TARGET_ROWS, cohort_counts)
    corrected_holdout_ids: set[str] = set()
    answer_quota: dict[str, dict[str, int]] = {}
    for cohort, rows in eligible_by_cohort.items():
        by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_answer[row["clean"]["answer_letter"]].append(row)
        counts = {label: len(by_answer.get(label, [])) for label in "ABCD"}
        quotas = allocate(cohort_holdout_quota[cohort], counts)
        answer_quota[cohort] = quotas
        for label in "ABCD":
            candidates = sorted(
                by_answer.get(label, []), key=lambda row: (row["_split_hash"], row["record_id"])
            )
            corrected_holdout_ids.update(
                str(row["record_id"]) for row in candidates[: quotas[label]]
            )

    if len(corrected_holdout_ids) != STRATIFIED_HOLDOUT_TARGET_ROWS:
        raise RuntimeError(
            "corrected stratified holdout selection mismatch: "
            f"{len(corrected_holdout_ids)} != {STRATIFIED_HOLDOUT_TARGET_ROWS}"
        )
    eligible_ids = {str(row["record_id"]) for row in combined}
    if not FROZEN_BASELINE_HOLDOUT_IDS.issubset(eligible_ids):
        raise RuntimeError(
            "a previously exposed baseline holdout row is absent from eligible reviewed inputs"
        )
    correction_additions = corrected_holdout_ids - FROZEN_BASELINE_HOLDOUT_IDS
    if correction_additions != EXPECTED_CORRECTION_ADDITIONS:
        raise RuntimeError(
            f"corrected holdout additions drifted: {sorted(correction_additions)} "
            f"!= {sorted(EXPECTED_CORRECTION_ADDITIONS)}"
        )
    holdout_ids = set(FROZEN_BASELINE_HOLDOUT_IDS | corrected_holdout_ids)
    if len(holdout_ids) != FINAL_HOLDOUT_ROWS:
        raise RuntimeError(
            f"permanent holdout union mismatch: {len(holdout_ids)} != {FINAL_HOLDOUT_ROWS}"
        )
    train_rows = [row for row in combined if str(row["record_id"]) not in holdout_ids]
    holdout_rows = [row for row in combined if str(row["record_id"]) in holdout_ids]
    if len(train_rows) != FINAL_TRAIN_ROWS or len(holdout_rows) != FINAL_HOLDOUT_ROWS:
        raise RuntimeError(
            f"final row signature drifted: train={len(train_rows)} holdout={len(holdout_rows)}"
        )
    if len(train_rows) < MIN_TRAIN_ROWS or len(holdout_rows) < MIN_HOLDOUT_ROWS:
        raise RuntimeError("minimum train/holdout release gate failed")
    train_rows.sort(key=lambda row: (row["_split_hash"], row["record_id"]))
    holdout_rows.sort(key=lambda row: (row["_split_hash"], row["record_id"]))
    scoring_holdout_rows = [
        row
        for row in holdout_rows
        if str(row["record_id"]) not in PARENT_CONTAMINATED_PERMANENT_E
    ]
    if len(scoring_holdout_rows) != FINAL_SCORING_HOLDOUT_ROWS:
        raise RuntimeError(
            "scoring holdout row mismatch: "
            f"{len(scoring_holdout_rows)} != {FINAL_SCORING_HOLDOUT_ROWS}"
        )
    if len(train_rows) + len(scoring_holdout_rows) != EXPECTED_CLEAN_D_OR_SCORING_TOTAL:
        raise RuntimeError("clean D/scoring row total drifted")

    # Final reverse gate: rebuild the central historical-E and parent indexes
    # after the definitive 70/25 split.  Our owned holdout is excluded from
    # historical E and is checked separately through the frozen-ID union.
    print("[reverse-gate] hashing historical E and parent manifests", flush=True)
    eval_manifest = file_manifest(historical_eval_manifest_paths())
    parent_manifest = file_manifest(parent_manifest_paths())
    eval_index = base.load_eval_index(exclude_paths=(HOLDOUT,))
    parent_index, parent_counts = base.load_train_index()
    from build_o5_zh_blind_review_packet import _load_parent_near_index

    parent_near_index = _load_parent_near_index()
    reviewed_by_id = {str(row["record_id"]): row for row in reviewed}
    quarantined_eval_confirmation: dict[str, dict[str, Any]] = {}
    for record_id in sorted(EXCLUDED_EXISTING_E):
        row = reviewed_by_id[record_id]
        prompt = render_stem(row["clean"])
        parsed = structured_parsed(row)
        hit, modes = eval_index.match(prompt, parsed)
        if not hit:
            raise RuntimeError(f"frozen existing-E quarantine no longer matches central E: {record_id}")
        quarantined_eval_confirmation[record_id] = {
            **EXCLUDED_EXISTING_E[record_id],
            "central_eval_match_modes_reconfirmed": modes,
        }
    quarantined_parent_confirmation: dict[str, dict[str, Any]] = {}
    for record_id in sorted(EXCLUDED_EXISTING_PARENT):
        row = reviewed_by_id[record_id]
        prompt = render_stem(row["clean"])
        parsed = structured_parsed(row)
        exact_hit, exact_modes = parent_index.match(prompt, parsed)
        near_hit, near_modes = parent_near_index.match(prompt, parsed)
        if not (exact_hit or near_hit):
            raise RuntimeError(
                f"frozen existing-parent quarantine no longer matches parent: {record_id}"
            )
        quarantined_parent_confirmation[record_id] = {
            **EXCLUDED_EXISTING_PARENT[record_id],
            "parent_match_modes_reconfirmed": sorted(set(exact_modes + near_modes)),
        }
    train_eval_hits: list[dict[str, Any]] = []
    train_parent_hits: list[dict[str, Any]] = []
    for row in train_rows:
        prompt = render_stem(row["clean"])
        parsed = structured_parsed(row)
        eval_hit, eval_modes = eval_index.match(prompt, parsed)
        parent_hit, parent_modes = parent_index.match(prompt, parsed)
        parent_near_hit, parent_near_modes = parent_near_index.match(prompt, parsed)
        if eval_hit:
            train_eval_hits.append(
                {"record_id": row["record_id"], "match_modes": eval_modes}
            )
        if parent_hit or parent_near_hit:
            train_parent_hits.append(
                {
                    "record_id": row["record_id"],
                    "match_modes": sorted(set(parent_modes + parent_near_modes)),
                }
            )
    holdout_eval_hits: list[dict[str, Any]] = []
    holdout_parent_hits: list[dict[str, Any]] = []
    for row in holdout_rows:
        prompt = render_stem(row["clean"])
        parsed = structured_parsed(row)
        eval_hit, eval_modes = eval_index.match(prompt, parsed)
        parent_hit, parent_modes = parent_index.match(prompt, parsed)
        parent_near_hit, parent_near_modes = parent_near_index.match(prompt, parsed)
        if eval_hit:
            holdout_eval_hits.append(
                {"record_id": row["record_id"], "match_modes": eval_modes}
            )
        if parent_hit or parent_near_hit:
            holdout_parent_hits.append(
                {
                    "record_id": row["record_id"],
                    "match_modes": sorted(set(parent_modes + parent_near_modes)),
                }
            )
    observed_contaminated_holdout_ids = {
        str(row["record_id"]) for row in holdout_parent_hits
    }
    expected_contaminated_holdout_ids = set(PARENT_CONTAMINATED_PERMANENT_E)
    if train_eval_hits or train_parent_hits or holdout_eval_hits:
        raise RuntimeError(
            "final reverse leakage gate failed: "
            f"train_eval={train_eval_hits[:5]} train_parent={train_parent_hits[:5]} "
            f"holdout_eval={holdout_eval_hits[:5]}"
        )
    if observed_contaminated_holdout_ids != expected_contaminated_holdout_ids:
        raise RuntimeError(
            "holdout parent contamination signature drifted: "
            f"observed={sorted(observed_contaminated_holdout_ids)} "
            f"expected={sorted(expected_contaminated_holdout_ids)}"
        )

    builder = Path(__file__).resolve()
    builder_sha = sha256_file(builder)
    build_fingerprint = hashlib.sha256(
        stable_json(
            {
                "builder_sha256": builder_sha,
                "stratified_holdout_target_rows": STRATIFIED_HOLDOUT_TARGET_ROWS,
                "final_holdout_rows": FINAL_HOLDOUT_ROWS,
                "final_scoring_holdout_rows": FINAL_SCORING_HOLDOUT_ROWS,
                "frozen_baseline_holdout_ids": sorted(FROZEN_BASELINE_HOLDOUT_IDS),
                "parent_contaminated_permanent_e": PARENT_CONTAMINATED_PERMANENT_E,
                "input_hashes": {key: value["sha256"] for key, value in EXPECTED_INPUTS.items()},
                "topic_input_hashes": {
                    key: value["sha256"] for key, value in TOPIC_INPUTS.items()
                },
                "historical_eval_manifest_sha256": eval_manifest["manifest_sha256"],
                "parent_manifest_sha256": parent_manifest["manifest_sha256"],
                "ruleset_version": RULESET_VERSION,
                "seed": SEED,
            }
        ).encode("utf-8")
    ).hexdigest()

    def reviewed_output(row: dict[str, Any], role: str) -> dict[str, Any]:
        result = {key: value for key, value in row.items() if not key.startswith("_split_")}
        result["split"] = {
            "role": role,
            "seed": SEED,
            "split_hash": row["_split_hash"],
            "cohort": row["_split_cohort"],
            "topic": row["_split_topic"],
            "semantic_cluster_size": 1,
        }
        if role == "permanent_holdout":
            record_id = str(row["record_id"])
            scoring_eligible = record_id not in PARENT_CONTAMINATED_PERMANENT_E
            result["split"]["evaluation_eligible"] = scoring_eligible
            result["split"]["checkpoint_selection_eligible"] = scoring_eligible
            if not scoring_eligible:
                result["split"]["contamination"] = PARENT_CONTAMINATED_PERMANENT_E[
                    record_id
                ]
        result["split_builder"] = {
            "builder_sha256": builder_sha,
            "build_fingerprint": build_fingerprint,
            "ruleset_version": RULESET_VERSION,
        }
        return result

    train_reviewed = [reviewed_output(row, "train") for row in train_rows]
    train_projection = [trainer_row(row) for row in train_rows]
    holdout_output: list[dict[str, Any]] = []
    for row in holdout_rows:
        reviewed_row = reviewed_output(row, "permanent_holdout")
        # Top-level input is intentional: the central E blacklist discovers
        # every holdout prompt on future scans without special-case code.
        holdout_output.append({**trainer_row(row), **reviewed_row})

    if len({stable_json(row) for row in train_projection}) != len(train_projection):
        raise RuntimeError("duplicate trainer rows")
    train_inputs = {row["input"] for row in train_projection}
    holdout_inputs = {row["input"] for row in holdout_output}
    if train_inputs & holdout_inputs:
        raise RuntimeError("train/holdout prompt overlap")
    if any(row["output"] != RESPONSE_PREFIX + row["clean"]["answer_letter"] + ")" for row in holdout_output):
        raise RuntimeError("holdout canonical response mismatch")
    for row in train_projection:
        if set(row) != {"instruction", "input", "output", "history"}:
            raise RuntimeError("trainer projection contains metadata fields")

    atomic_jsonl(args.train_reviewed, train_reviewed)
    atomic_jsonl(args.train_projection, train_projection)
    atomic_jsonl(args.holdout, holdout_output)

    def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "answer": dict(sorted(Counter(row["clean"]["answer_letter"] for row in rows).items())),
            "cohort": dict(sorted(Counter(row["_split_cohort"] for row in rows).items())),
            "topic": dict(sorted(Counter(row["_split_topic"] for row in rows).items())),
            "upstream_asset": dict(
                sorted(Counter(str(row["lineage"].get("asset_id", "unknown")) for row in rows).items())
            ),
        }

    audit = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "asset_class": "D-reviewed-train + E-permanent-holdout derived from official General",
        "ruleset_version": RULESET_VERSION,
        "builder": str(builder),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "inputs": {
            cohort: {
                "path": str(supplied[cohort].resolve()),
                "rows": spec["rows"],
                "sha256": spec["sha256"],
            }
            for cohort, spec in EXPECTED_INPUTS.items()
        },
        "topic_metadata_inputs": {
            cohort: {
                "path": str(spec["path"].resolve()),
                "rows": spec["rows"],
                "sha256": spec["sha256"],
            }
            for cohort, spec in TOPIC_INPUTS.items()
        },
        "semantic_gate": {
            "exact_duplicate_pairs": 0,
            "near_duplicate_pairs": len(near_pairs),
            "char3_jaccard_threshold": 0.60,
            "sequence_ratio_threshold": 0.82,
            "containment_min_length_ratio": 0.70,
            "maximum_observed_char3_jaccard": max_jaccard,
            "maximum_observed_sequence_ratio": max_sequence,
            "all_semantic_clusters_singleton": True,
            "train_holdout_exact_prompt_overlap": 0,
            "train_holdout_near_prompt_overlap": 0,
        },
        "split": {
            "seed": SEED,
            "method": (
                "eligible-cohort-and-answer-stratified stable hash, unioned with every "
                "row exposed by the completed first baseline; semantic clusters kept whole"
            ),
            "stratified_target_rows": STRATIFIED_HOLDOUT_TARGET_ROWS,
            "permanent_holdout_rows_after_exposure_union": FINAL_HOLDOUT_ROWS,
            "cohort_holdout_quota": cohort_holdout_quota,
            "cohort_answer_holdout_quota": answer_quota,
            "frozen_first_baseline_holdout_rows": len(FROZEN_BASELINE_HOLDOUT_IDS),
            "corrected_stratified_holdout_rows": len(corrected_holdout_ids),
            "correction_addition_ids": sorted(correction_additions),
            "frozen_only_ids_retained_as_e": sorted(
                FROZEN_BASELINE_HOLDOUT_IDS - corrected_holdout_ids
            ),
            "permanence_policy": (
                "a row exposed to any completed diagnostic remains E and can never return to D"
            ),
            "train": distribution(train_rows),
            "holdout": distribution(holdout_rows),
            "scoring_holdout": distribution(scoring_holdout_rows),
        },
        "final_reverse_leakage_gate": {
            "historical_eval": {
                "owned_holdout_excluded_from_historical_index": str(HOLDOUT.resolve()),
                "source_prompt_instances": dict(sorted(eval_index.source_counts.items())),
                "manifest": eval_manifest,
                "checked_final_train_rows": len(train_rows),
                "checked_permanent_holdout_rows": len(holdout_rows),
                "train_hits": train_eval_hits,
                "holdout_hits_excluding_owned_holdout_file": holdout_eval_hits,
            },
            "current_parent": {
                "source_prompt_instances": dict(sorted(parent_counts.items())),
                "manifest": parent_manifest,
                "checked_final_train_rows": len(train_rows),
                "checked_permanent_holdout_rows": len(holdout_rows),
                "train_exact_or_mc_near_hits": train_parent_hits,
                "holdout_exact_or_mc_near_hits": holdout_parent_hits,
            },
            "translated_parent_near_gate_inherited_from_upstream": True,
            "decision": "PASS_ZERO_FINAL_TRAIN_HITS_AND_FROZEN_TWO_RETIRED_E_HITS",
        },
        "existing_e_quarantine": {
            "reviewed_input_rows": len(reviewed),
            "excluded_rows": len(EXCLUDED_EXISTING_E),
            "eligible_rows_after_existing_e_exclusion": (
                len(reviewed) - len(EXCLUDED_EXISTING_E)
            ),
            "rows": quarantined_eval_confirmation,
            "policy": "excluded from both train and new holdout; existing E cannot be relabeled as D",
        },
        "existing_parent_quarantine": {
            "excluded_rows": len(EXCLUDED_EXISTING_PARENT),
            "split_pool_rows_after_hard_quarantines": len(combined),
            "rows": quarantined_parent_confirmation,
            "policy": (
                "excluded from both train and new holdout; a current-parent duplicate "
                "cannot be represented as new D information"
            ),
        },
        "parent_contaminated_permanent_e": {
            "rows": len(PARENT_CONTAMINATED_PERMANENT_E),
            "record_ids": sorted(PARENT_CONTAMINATED_PERMANENT_E),
            "details": PARENT_CONTAMINATED_PERMANENT_E,
            "policy": (
                "remain permanent E after completed exposure; never D; excluded from "
                "evaluation and checkpoint-selection aggregation"
            ),
        },
        "outputs": {
            "train_reviewed": {
                "path": str(args.train_reviewed.resolve()),
                "rows": len(train_reviewed),
                "bytes": args.train_reviewed.stat().st_size,
                "sha256": sha256_file(args.train_reviewed),
            },
            "train_projection": {
                "path": str(args.train_projection.resolve()),
                "rows": len(train_projection),
                "bytes": args.train_projection.stat().st_size,
                "sha256": sha256_file(args.train_projection),
                "world_unique_rows": len(train_projection),
                "world_repeat": 1,
            },
            "permanent_holdout": {
                "path": str(args.holdout.resolve()),
                "rows": len(holdout_output),
                "bytes": args.holdout.stat().st_size,
                "sha256": sha256_file(args.holdout),
                "allowed_role": "checkpoint selection and mechanism audit only; never gradient data",
                "scoring_eligible_rows": len(scoring_holdout_rows),
                "retired_parent_contaminated_rows": len(
                    PARENT_CONTAMINATED_PERMANENT_E
                ),
            },
        },
        "release_gate": {
            "minimum_train_rows": MIN_TRAIN_ROWS,
            "minimum_holdout_rows": MIN_HOLDOUT_ROWS,
            "reviewed_total": len(reviewed),
            "split_pool_total_after_hard_quarantines": len(combined),
            "clean_d_or_scoring_total": len(train_rows) + len(scoring_holdout_rows),
            "training_projection_created": True,
            "training_rows_repeated": False,
            "permanent_holdout_created": True,
            "formal_training_config_created": False,
            "formal_training_authorized": False,
            "decision": "DATA_GATE_PASS_AWAITING_BASELINE_DIAGNOSTIC_AND_USER_APPROVAL_OF_TRAINING_PARAMETERS",
        },
    }
    atomic_json(args.audit, audit)
    print(stable_json(audit))


if __name__ == "__main__":
    main()
