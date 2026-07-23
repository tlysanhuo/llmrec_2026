from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/data"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "official_general_world_mc_v1_split",
    SCRIPT_DIR / "build_official_general_world_mc_v1_split.py",
)
assert SPEC is not None and SPEC.loader is not None
split = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = split
SPEC.loader.exec_module(split)


def read_jsonl(relative: str) -> list[dict]:
    path = ROOT / relative
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def recursive_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(recursive_keys(child))
    return keys


def test_answer_blind_packets_are_physically_isolated() -> None:
    forbidden = {
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
    for relative, expected_rows in (
        ("assets/derived/official_general/o5_en_mc_zh_blind_review_packet.jsonl", 41),
        ("assets/derived/official_general/o2_en_mc_zh_blind_review_packet.jsonl", 27),
    ):
        rows = read_jsonl(relative)
        assert len(rows) == expected_rows
        assert not (forbidden & recursive_keys(rows))
        assert all(row["review_protocol"]["source_answer_claim_visible"] is False for row in rows)


def test_translation_audits_remain_bound_to_pre_correction_prompt_set() -> None:
    cases = (
        ("logs/data/o5_en_mc_translation_audit.json", 11),
        ("logs/data/o2_en_mc_translation_audit.json", 7),
    )
    for relative, expected_owned_matches in cases:
        audit = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert audit["blacklist"]["historical_eval_excludes_owned_downstream_holdout"] is True
        owned = audit["blacklist"]["owned_downstream_holdout"]
        assert owned["frozen_translated_ids"] == 18
        assert owned["rows"] == 25
        # These audits were frozen when the same 25 prompt IDs were first
        # exposed.  The current holdout changes only split eligibility metadata
        # for two rows, so the review-time full-file hash remains historical.
        assert owned["sha256"] == (
            "d363ff22f93b6eb4e10eebba818be4c8fb1dfb05ff5e7d0ac8d66cf7c1dcf2db"
        )
        assert (
            audit["filter_counts"]["post_split_owned_holdout_match:expected"]
            == expected_owned_matches
        )
        packet = audit["outputs"]["zh_blind_review_packet"]
        assert packet["sha256"] == hashlib.sha256(Path(packet["path"]).read_bytes()).hexdigest()


def test_adjudicated_assets_have_two_blind_reviews_and_no_rejections() -> None:
    cases = (
        (
            "assets/derived/official_general/o5_zh_mc_dual_answer_blind_reviewed_safe.jsonl",
            "logs/data/o5_zh_mc_dual_answer_blind_adjudication_ledger.jsonl",
            41,
            1,
        ),
        (
            "assets/derived/official_general/o2_zh_mc_dual_answer_blind_reviewed_safe.jsonl",
            "logs/data/o2_zh_mc_dual_answer_blind_adjudication_ledger.jsonl",
            27,
            2,
        ),
    )
    for asset_path, ledger_path, expected_rows, prompt_blind_count in cases:
        rows = read_jsonl(asset_path)
        ledger = read_jsonl(ledger_path)
        assert len(rows) == len(ledger) == expected_rows
        assert all(row["decision"] == "accept" and not row["reason_codes"] for row in ledger)
        for row in rows:
            review = row["review"]
            assert review["consensus_verified"] is True
            assert review["translation_mechanical_and_answer_consistency_pass"] is True
            assert "factual_correct" not in review
            assert "translation_fidelity_pass" not in review
            assert review["source_answer_claim_blind_review_count"] == 2
            assert review["source_prompt_blind_review_count"] == prompt_blind_count
            assert review["source_answer_claim_role"] == "agreement_check_not_standalone_gold"
            assert len(set(review["reviewers"])) == 2


def test_split_allocation_is_deterministic_and_stratified() -> None:
    assert split.allocate(24, {"legacy": 25, "o5": 41, "o2": 27}) == {
        "legacy": 6,
        "o2": 7,
        "o5": 11,
    }
    assert split.allocate(6, {"A": 5, "B": 7, "C": 9, "D": 4}) == {
        "A": 1,
        "B": 2,
        "C": 2,
        "D": 1,
    }


def test_near_duplicate_gate_detects_rewrites_but_not_unrelated_text() -> None:
    same, metrics = split.near_duplicate(
        split.semantic_normalize("太阳系最大的行星是哪一个？A.地球 B.木星 C.火星 D.金星"),
        split.semantic_normalize("太阳系中最大的行星是哪一个？A.地球 B.木星 C.火星 D.金星"),
    )
    assert same
    assert metrics["sequence_ratio"] >= 0.82 or metrics["char3_jaccard"] >= 0.60
    different, _ = split.near_duplicate(
        split.semantic_normalize("太阳系最大的行星是哪一个？"),
        split.semantic_normalize("一个二次函数的判别式如何计算？"),
    )
    assert not different


def test_reverse_gate_uses_reviewed_structure_for_two_scenario_single_choice() -> None:
    rows = read_jsonl(
        "assets/derived/official_general/o5_zh_mc_dual_answer_blind_reviewed_safe.jsonl"
    )
    row = next(
        value
        for value in rows
        if value["record_id"]
        == "fb54aa86b2c826cfaaa7d173cc10939f64ee1f397cbfe8a9b8b982e80036a004"
    )
    raw_parsed, reasons = split.base.parse_mc_prompt(split.render_stem(row["clean"]))
    assert raw_parsed is None and reasons == ["mc_multiselect"]
    reviewed_parsed = split.structured_parsed(row)
    assert reviewed_parsed.question == row["clean"]["question"]
    assert reviewed_parsed.options == row["clean"]["options"]


def test_final_train_holdout_and_quarantine_invariants() -> None:
    train_reviewed = read_jsonl(
        "assets/derived/official_general/official_general_world_mc_v1_train_reviewed.jsonl"
    )
    train_projection = read_jsonl(
        "assets/derived/processed/data_official_general_world_mc_v1.jsonl"
    )
    holdout = read_jsonl(
        "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl"
    )
    assert len(train_reviewed) == len(train_projection) == 68
    assert len(holdout) == 25
    train_ids = {row["record_id"] for row in train_reviewed}
    holdout_ids = {row["record_id"] for row in holdout}
    assert not (train_ids & holdout_ids)
    assert not (set(split.EXCLUDED_EXISTING_E) & (train_ids | holdout_ids))
    assert not (set(split.EXCLUDED_EXISTING_PARENT) & (train_ids | holdout_ids))
    assert split.FROZEN_BASELINE_HOLDOUT_IDS.issubset(holdout_ids)
    assert split.EXPECTED_CORRECTION_ADDITIONS.issubset(holdout_ids)
    assert all(row["split"]["role"] == "train" for row in train_reviewed)
    assert all(row["split"]["role"] == "permanent_holdout" for row in holdout)
    scoring_rows = [row for row in holdout if row["split"]["evaluation_eligible"]]
    retired_rows = [row for row in holdout if not row["split"]["evaluation_eligible"]]
    assert len(scoring_rows) == 23
    assert {row["record_id"] for row in retired_rows} == set(
        split.PARENT_CONTAMINATED_PERMANENT_E
    )
    assert all(
        row["split"]["checkpoint_selection_eligible"] is False
        and row["split"]["contamination"]["contamination_reason"]
        == "current_parent_exact"
        for row in retired_rows
    )
    assert all(set(row) == {"instruction", "input", "output", "history"} for row in train_projection)
    assert len({json.dumps(row, ensure_ascii=False, sort_keys=True) for row in train_projection}) == 68
    assert not ({row["input"] for row in train_projection} & {row["input"] for row in holdout})


def test_split_audit_hashes_and_release_boundary_match_files() -> None:
    audit_path = "logs/data/official_general_world_mc_v1_split_audit.json"
    audit = json.loads((ROOT / audit_path).read_text(encoding="utf-8"))
    outputs = audit["outputs"]
    mapping = {
        "train_reviewed": "assets/derived/official_general/official_general_world_mc_v1_train_reviewed.jsonl",
        "train_projection": "assets/derived/processed/data_official_general_world_mc_v1.jsonl",
        "permanent_holdout": "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl",
    }
    for name, relative in mapping.items():
        assert outputs[name]["sha256"] == sha256(relative)
        assert outputs[name]["bytes"] == (ROOT / relative).stat().st_size
    assert audit["existing_e_quarantine"]["reviewed_input_rows"] == 97
    assert audit["existing_e_quarantine"]["excluded_rows"] == 2
    assert audit["existing_parent_quarantine"]["excluded_rows"] == 2
    assert audit["parent_contaminated_permanent_e"]["rows"] == 2
    assert audit["release_gate"]["split_pool_total_after_hard_quarantines"] == 93
    assert audit["release_gate"]["clean_d_or_scoring_total"] == 91
    assert audit["release_gate"]["formal_training_authorized"] is False
    assert audit["release_gate"]["training_rows_repeated"] is False
    assert audit["semantic_gate"]["near_duplicate_pairs"] == 0
    assert audit["final_reverse_leakage_gate"]["decision"] == (
        "PASS_ZERO_FINAL_TRAIN_HITS_AND_FROZEN_TWO_RETIRED_E_HITS"
    )
    assert not audit["final_reverse_leakage_gate"]["historical_eval"]["train_hits"]
    assert not audit["final_reverse_leakage_gate"]["historical_eval"][
        "holdout_hits_excluding_owned_holdout_file"
    ]
    assert not audit["final_reverse_leakage_gate"]["current_parent"][
        "train_exact_or_mc_near_hits"
    ]
    assert {
        row["record_id"]
        for row in audit["final_reverse_leakage_gate"]["current_parent"]
        ["holdout_exact_or_mc_near_hits"]
    } == set(split.PARENT_CONTAMINATED_PERMANENT_E)
    assert audit["split"]["train"]["topic"] == {
        "legacy_unclassified": 18,
        "math_logic": 49,
        "other_general": 1,
    }
    assert audit["split"]["holdout"]["topic"] == {
        "legacy_unclassified": 7,
        "math_logic": 17,
        "other_general": 1,
    }
    assert audit["split"]["scoring_holdout"]["rows"] == 23


def test_s800_baseline_is_bound_to_frozen_holdout() -> None:
    report = json.loads(
        (ROOT / "logs/probe/official_general_world_mc_v1_s800_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "COMPLETE_BASELINE_NOT_ONLINE_SCORE_ESTIMATE"
    assert report["inputs"]["holdout"] == sha256(
        "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl"
    )
    assert report["inputs"]["train"] == sha256(
        "assets/derived/processed/data_official_general_world_mc_v1.jsonl"
    )
    assert report["inputs"]["permanent_holdout_rows"] == 25
    assert report["inputs"]["scoring_holdout_rows"] == 23
    assert set(report["inputs"]["retired_record_ids"]) == set(
        split.PARENT_CONTAMINATED_PERMANENT_E
    )
    assert report["overall"]["rows"] == 23
    assert report["overall"]["abcd_accuracy"] == 0.2173913
    assert report["by_topic"]["math_logic"]["rows"] == 17
    assert report["by_topic"]["math_logic"]["abcd_accuracy"] == 0.11764706
    assert report["method"]["generation"] is False
