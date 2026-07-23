#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/data/build_o5_english_mc_answer_claim_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("o5_english_mc_answer_claim_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


QUESTION = "Which planet is known as the Red Planet?"
OPTIONS = {
    "A": "Mercury",
    "B": "Venus",
    "C": "Earth",
    "D": "Mars",
}


def mc_prompt(style: str, question: str = QUESTION) -> str:
    renderers = {
        "dot": lambda label, text: f"{label}. {text}",
        "colon": lambda label, text: f"{label}: {text}",
        "close_paren": lambda label, text: f"{label}) {text}",
        "paren": lambda label, text: f"({label}) {text}",
        "bracket": lambda label, text: f"[{label}] {text}",
    }
    render = renderers[style]
    return question + "\n" + "\n".join(render(label, OPTIONS[label]) for label in "ABCD")


def shard_rank(path: Path) -> tuple[str, str]:
    # This is deliberately the two-byte literal backslash-zero separator, not
    # a NUL byte.  The date, asset id, separator, and ordering are all frozen.
    separator = bytes([92, 48])
    payload = (
        b"20260711"
        + separator
        + b"O5"
        + separator
        + path.name.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest(), path.name


def sample_rank(seed: int, stratum: str, record_id: str) -> tuple[str, str]:
    payload = f"{seed}\0{stratum}\0{record_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), record_id


class O5EnglishMcPromptTest(unittest.TestCase):
    def assert_reason_contains(self, reasons: list[str], fragment: str) -> None:
        self.assertTrue(
            any(fragment in reason for reason in reasons),
            f"expected reason containing {fragment!r}, got {reasons!r}",
        )

    def test_five_multiline_uppercase_abcd_styles_are_accepted(self) -> None:
        for style in ("dot", "colon", "close_paren", "paren", "bracket"):
            with self.subTest(style=style):
                parsed, reasons = pilot.parse_english_mc_prompt(mc_prompt(style))
                self.assertEqual(reasons, [])
                self.assertIsNotNone(parsed)
                assert parsed is not None
                self.assertEqual(parsed.question, QUESTION)
                self.assertEqual(parsed.options, OPTIONS)

    def test_inline_options_are_rejected_even_when_abcd_are_present(self) -> None:
        prompt = QUESTION + " A. Mercury B. Venus C. Earth D. Mars"
        parsed, reasons = pilot.parse_english_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assert_reason_contains(reasons, "inline")

    def test_lowercase_labels_are_rejected(self) -> None:
        prompt = QUESTION + "\na. Mercury\nb. Venus\nc. Earth\nd. Mars"
        parsed, reasons = pilot.parse_english_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assert_reason_contains(reasons, "lowercase")

    def test_extra_e_option_is_rejected(self) -> None:
        prompt = mc_prompt("dot") + "\nE. Jupiter"
        parsed, reasons = pilot.parse_english_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assert_reason_contains(reasons, "extra_option")

    def test_duplicate_option_text_is_rejected_after_normalization(self) -> None:
        prompt = (
            QUESTION
            + "\nA. Mercury\nB. Venus\nC.  mercury!  \nD. Mars"
        )
        parsed, reasons = pilot.parse_english_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assert_reason_contains(reasons, "duplicate_option")

    def test_multiselect_wording_is_rejected(self) -> None:
        stems = (
            "Select all that apply: which planets are rocky?",
            "Choose two correct answers about the planets.",
            "Which of the following statements are correct?",
            "There may be more than one correct answer. Which option applies?",
        )
        for stem in stems:
            with self.subTest(stem=stem):
                parsed, reasons = pilot.parse_english_mc_prompt(mc_prompt("dot", stem))
                self.assertIsNone(parsed)
                self.assert_reason_contains(reasons, "multiselect")

    def test_all_none_and_cross_referencing_options_are_rejected(self) -> None:
        forbidden_options = (
            "All of the above",
            "None of the above",
            "Both A and B",
            "Neither A nor B",
        )
        for forbidden in forbidden_options:
            with self.subTest(forbidden=forbidden):
                prompt = (
                    QUESTION
                    + f"\nA. Mercury\nB. Venus\nC. Earth\nD. {forbidden}"
                )
                parsed, reasons = pilot.parse_english_mc_prompt(prompt)
                self.assertIsNone(parsed)
                self.assert_reason_contains(reasons, "all_none")

    def test_negated_or_exception_stems_are_rejected(self) -> None:
        stems = (
            "Which of the following is NOT a planet?",
            "All of the following are planets EXCEPT which one?",
            "Which statement about Mars is incorrect?",
            "Which outcome is least likely?",
        )
        for stem in stems:
            with self.subTest(stem=stem):
                parsed, reasons = pilot.parse_english_mc_prompt(mc_prompt("dot", stem))
                self.assertIsNone(parsed)
                self.assert_reason_contains(reasons, "negat")

    def test_strict_english_accepts_english_formula_text(self) -> None:
        passed, stats = pilot.is_strict_en(
            "For 2x + 3 = 7, which value of x is correct?\n"
            "A. One\nB. Two\nC. Three\nD. Four"
        )
        self.assertTrue(passed, stats)

    def test_mixed_scripts_are_rejected_by_strict_english_gate(self) -> None:
        mixed = (
            "Which planet is called 红色星球?\nA. Mercury\nB. Venus\nC. Earth\nD. Mars",
            "Which answer is 맞습니까?\nA. One\nB. Two\nC. Three\nD. Four",
            "Which answer is 正しい?\nA. One\nB. Two\nC. Three\nD. Four",
        )
        for text in mixed:
            with self.subTest(text=text):
                passed, _stats = pilot.is_strict_en(text)
                self.assertFalse(passed)


class O5EnglishAnswerClaimTest(unittest.TestCase):
    def assert_answer(self, response: str, expected: str) -> None:
        answer, evidence, reasons = pilot.parse_english_mc_answer_claim(response)
        self.assertEqual(reasons, [])
        self.assertEqual(answer, expected)
        self.assertTrue(evidence)

    def assert_rejected(self, response: str, reason_fragment: str) -> None:
        answer, _evidence, reasons = pilot.parse_english_mc_answer_claim(response)
        self.assertIsNone(answer)
        self.assertTrue(
            any(reason_fragment in reason for reason in reasons),
            f"expected reason containing {reason_fragment!r}, got {reasons!r}",
        )

    def test_supported_final_answer_claim_forms(self) -> None:
        examples = (
            ("After comparing the choices.\nFinal Answer: A", "A"),
            ("After comparing the choices.\nThe correct answer is B.", "B"),
            ("After comparing the choices.\nC is correct.", "C"),
            ("After comparing the choices.\n[D]", "D"),
            (r"After comparing the choices." + "\n" + r"\boxed{A}", "A"),
        )
        for response, expected in examples:
            with self.subTest(response=response):
                self.assert_answer(response, expected)

    def test_real_r1_boxed_answer_forms(self) -> None:
        examples = (
            (r"The result follows. Final answer: $\boxed{A}$", "A"),
            ("The result follows.\n" + r"\[\boxed{C}\]", "C"),
            (r"The result follows. \boxed{\text{B}}", "B"),
            (r"The result follows. \boxed{\text{B: } \frac{1}{12}}", "B"),
        )
        for response, expected in examples:
            with self.subTest(response=response):
                self.assert_answer(response, expected)

    def test_boxed_value_without_answer_letter_is_rejected(self) -> None:
        answer, _evidence, reasons = pilot.parse_english_mc_answer_claim(
            r"The final value is \boxed{\frac{1}{2}}"
        )
        self.assertIsNone(answer)
        self.assertTrue(
            any("unparsed" in reason or "no_letter" in reason for reason in reasons),
            reasons,
        )

    def test_conflicting_strong_claims_are_rejected(self) -> None:
        self.assert_rejected(
            "The correct answer is A.\nAfter reconsidering it, Final Answer: B",
            "conflict",
        )

    def test_uncertain_claims_are_rejected(self) -> None:
        for response in (
            "Final Answer: probably B",
            "I think the correct answer is B.",
            "The answer may be B.",
            "Final Answer: B or C",
        ):
            with self.subTest(response=response):
                self.assert_rejected(response, "uncertain")

    def test_claim_must_be_the_last_substantive_line(self) -> None:
        self.assert_rejected(
            "Final Answer: B\nThis is followed by additional substantive explanation.",
            "not_final",
        )

    def test_bracket_or_boxed_letter_inside_prose_is_not_a_final_claim(self) -> None:
        for response in (
            "Option [B] appears in the intermediate calculation only.",
            r"The expression \boxed{C} is considered before the final decision.",
        ):
            with self.subTest(response=response):
                self.assert_rejected(response, "not_final")


class O5EnglishMetadataTest(unittest.TestCase):
    def test_consistent_metadata_answer_fields_are_accepted(self) -> None:
        metadata = {
            "answer": "B",
            "correct_answer": "B",
            "label": "B",
            "gold": "B",
        }
        label, reasons = pilot.extract_metadata_label(metadata)
        self.assertEqual(reasons, [])
        self.assertEqual(label, "B")

    def test_conflicting_metadata_answer_fields_are_rejected(self) -> None:
        label, reasons = pilot.extract_metadata_label(
            {"answer": "B", "correct_answer": "C", "label": "B"}
        )
        self.assertIsNone(label)
        self.assertTrue(any("conflict" in reason for reason in reasons), reasons)

    def test_claim_and_metadata_are_independently_extractable_for_exact_match(self) -> None:
        claim, _evidence, claim_reasons = pilot.parse_english_mc_answer_claim(
            "Final Answer: D"
        )
        metadata, metadata_reasons = pilot.extract_metadata_label(
            {"answer": "D", "label": "D"}
        )
        self.assertEqual(claim_reasons, [])
        self.assertEqual(metadata_reasons, [])
        self.assertEqual(claim, metadata)


class O5EnglishSourcePolicyTest(unittest.TestCase):
    SAFE_LINEAGE = {
        "asset_id": "O5",
        "asset_revision": "frozen-revision",
        "source": "R1-Distill-SFT",
        "shard": "filtered_1_000000.parquet",
        "row_group": 0,
        "row_index": 7,
    }

    def assert_policy_reason(
        self, prompt: str, lineage: dict[str, object], fragment: str
    ) -> None:
        reasons = pilot.source_policy_reasons(prompt, lineage)
        self.assertTrue(
            any(fragment in reason for reason in reasons),
            f"expected reason containing {fragment!r}, got {reasons!r}",
        )

    def test_registered_low_risk_source_is_allowed(self) -> None:
        self.assertEqual(
            pilot.source_policy_reasons(mc_prompt("dot"), dict(self.SAFE_LINEAGE)),
            [],
        )

    def test_medical_legal_and_dynamic_risk_prompts_are_rejected(self) -> None:
        prompts = (
            (
                "A patient has severe chest pain. Which drug dose should be administered?",
                "high_risk",
            ),
            (
                "Under current securities law, which filing is legally required?",
                "high_risk",
            ),
            ("Who is the current president of this country?", "time_sensitive"),
        )
        for prompt, reason in prompts:
            with self.subTest(prompt=prompt):
                self.assert_policy_reason(prompt, dict(self.SAFE_LINEAGE), reason)

    def test_benchmark_derived_sources_are_rejected(self) -> None:
        for source in ("MMLU", "CMMLU", "CEval", "ARC-Challenge"):
            lineage = {**self.SAFE_LINEAGE, "source": source, "benchmark": source}
            with self.subTest(source=source):
                self.assert_policy_reason(mc_prompt("dot"), lineage, "benchmark")

    def test_reading_comprehension_rows_are_rejected(self) -> None:
        prompts = (
            "Read the passage below and answer the question. According to the passage, "
            "why did the narrator leave the town?\nA. Work\nB. Study\nC. Family\nD. Weather",
            "St. Louis Rams release a player. What is this text about?\n"
            "A. World\nB. Sports\nC. Business\nD. Science and technology",
            "Select the topic that this is about: a company built a network.\n"
            "A. World\nB. Sports\nC. Business\nD. Science and technology",
            "Select the topic that this about: a company built a network.\n"
            "A. World\nB. Sports\nC. Business\nD. Science and technology",
            "A published report describes a basketball feud. Which topic is this article about?\n"
            "A. World\nB. Sports\nC. Business\nD. Science and technology",
            "Given the fact that a plant stem stores water, what is the answer to the "
            "question or completion about stems and flowers?\n"
            "A. dogs and cats\nB. cows and cud\nC. bees and pollen\nD. silos and grains",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assert_policy_reason(prompt, dict(self.SAFE_LINEAGE), "reading_comprehension")

    def test_asset_and_source_allowlist_are_fail_closed(self) -> None:
        wrong_asset = {**self.SAFE_LINEAGE, "asset_id": "O2.General"}
        unknown_source = {**self.SAFE_LINEAGE, "source": "unregistered-corpus"}
        self.assert_policy_reason(mc_prompt("dot"), wrong_asset, "asset")
        self.assert_policy_reason(mc_prompt("dot"), unknown_source, "source")


class O5EnglishDeterminismTest(unittest.TestCase):
    def test_fixed_shard_selection_uses_frozen_key_and_file_order(self) -> None:
        paths = [
            Path("/registered/general_sft")
            / f"filtered_{index // 4}_{index % 4:06d}.parquet"
            for index in range(301)
        ]
        expected_names = (
            "filtered_0_000002.parquet",
            "filtered_10_000003.parquet",
            "filtered_11_000000.parquet",
            "filtered_11_000003.parquet",
            "filtered_12_000001.parquet",
            "filtered_21_000000.parquet",
            "filtered_22_000001.parquet",
            "filtered_26_000001.parquet",
            "filtered_2_000003.parquet",
            "filtered_34_000003.parquet",
            "filtered_36_000003.parquet",
            "filtered_38_000003.parquet",
            "filtered_3_000001.parquet",
            "filtered_40_000000.parquet",
            "filtered_40_000003.parquet",
            "filtered_41_000001.parquet",
            "filtered_44_000001.parquet",
            "filtered_46_000000.parquet",
            "filtered_4_000000.parquet",
            "filtered_50_000002.parquet",
            "filtered_52_000000.parquet",
            "filtered_53_000001.parquet",
            "filtered_54_000003.parquet",
            "filtered_58_000003.parquet",
            "filtered_61_000003.parquet",
            "filtered_69_000000.parquet",
            "filtered_6_000002.parquet",
            "filtered_70_000000.parquet",
            "filtered_72_000002.parquet",
            "filtered_73_000000.parquet",
            "filtered_74_000000.parquet",
            "filtered_7_000001.parquet",
        )
        expected_ranked = sorted(paths, key=shard_rank)[:32]
        self.assertEqual(
            tuple(path.name for path in sorted(expected_ranked, key=lambda path: path.name)),
            expected_names,
        )

        selected = pilot.select_fixed_shards(paths)
        reversed_selected = pilot.select_fixed_shards(list(reversed(paths)))

        self.assertEqual(tuple(path.name for path in selected), expected_names)
        self.assertEqual(tuple(path.name for path in reversed_selected), expected_names)
        self.assertEqual(len(selected), 32)
        self.assertEqual(len(set(selected)), 32)

    def test_stable_stratified_quota_is_input_order_independent(self) -> None:
        seed = 7301
        rows = [
            {"record_id": f"r1-{index}", "source_bucket": "r1"}
            for index in range(8)
        ] + [
            {"record_id": f"inf-{index}", "source_bucket": "infinity"}
            for index in range(6)
        ]
        quotas = {"r1": 3, "infinity": 2}

        selected, audit = pilot.stable_stratified_sample(
            rows,
            quotas=quotas,
            seed=seed,
            stratum_key="source_bucket",
            id_key="record_id",
        )
        selected_again, audit_again = pilot.stable_stratified_sample(
            list(reversed(rows)),
            quotas=quotas,
            seed=seed,
            stratum_key="source_bucket",
            id_key="record_id",
        )

        expected_ids: set[str] = set()
        for stratum, quota in quotas.items():
            candidates = [row for row in rows if row["source_bucket"] == stratum]
            ranked = sorted(
                candidates,
                key=lambda row: sample_rank(seed, stratum, row["record_id"]),
            )
            expected_ids.update(row["record_id"] for row in ranked[:quota])

        self.assertEqual(selected, selected_again)
        self.assertEqual(audit, audit_again)
        self.assertEqual({row["record_id"] for row in selected}, expected_ids)
        self.assertEqual(Counter(row["source_bucket"] for row in selected), quotas)
        self.assertEqual(audit["shortfall_rows"], 0)
        self.assertEqual(audit["shortfall_by_stratum"], {})
        self.assertEqual(audit["dirty_backfill_rows"], 0)

    def test_shortfall_is_reported_without_cross_stratum_dirty_backfill(self) -> None:
        rows = [
            {"record_id": "scarce-0", "source_bucket": "scarce"},
            {"record_id": "rich-0", "source_bucket": "rich"},
            {"record_id": "rich-1", "source_bucket": "rich"},
            {"record_id": "rich-2", "source_bucket": "rich"},
            {"record_id": "rich-3", "source_bucket": "rich"},
        ]
        quotas = {"scarce": 2, "rich": 2}

        selected, audit = pilot.stable_stratified_sample(
            rows,
            quotas=quotas,
            seed=41,
            stratum_key="source_bucket",
            id_key="record_id",
        )
        counts = Counter(row["source_bucket"] for row in selected)

        self.assertEqual(counts, {"scarce": 1, "rich": 2})
        self.assertEqual(len(selected), 3)
        self.assertEqual(audit["requested_rows"], 4)
        self.assertEqual(audit["selected_rows"], 3)
        self.assertEqual(audit["shortfall_rows"], 1)
        self.assertEqual(audit["shortfall_by_stratum"], {"scarce": 1})
        self.assertEqual(audit["dirty_backfill_rows"], 0)


if __name__ == "__main__":
    unittest.main()
