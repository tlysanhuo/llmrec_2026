#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/data/build_o5_infinity_nonmath_mc_prompt_packet.py"
SPEC = importlib.util.spec_from_file_location("o5_infinity_nonmath_prompt_packet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

EXPECTED_PACKET_ROWS = 9
EXPECTED_PACKET_SHA256 = "a85719eecf8be553de5563d62567d93450810b13c3a2bac750c9b48883d2e0ad"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FormatNearParserTest(unittest.TestCase):
    def test_inline_abcd_is_recovered_without_response(self) -> None:
        parsed, reasons = builder.parse_format_near_mc(
            "中国最高峰是 A. 珠穆朗玛峰 B. 马卡鲁峰 C. 富士山 D. 喜马拉雅山"
        )
        self.assertEqual(reasons, [])
        assert parsed is not None
        self.assertEqual(parsed.question, "中国最高峰是")
        self.assertEqual(list(parsed.options), list("ABCD"))
        self.assertEqual(parsed.options["A"], "珠穆朗玛峰")

    def test_supported_native_marker_forms_are_recovered(self) -> None:
        prompts = (
            "题干（A）甲（B）乙（C）丙（D）丁",
            "题干 A:甲 B:乙 C:丙 D:丁",
            "题干 a)甲 b)乙 c)丙 d)丁",
            "题干 A．甲 B．乙 C．丙 D．丁",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                parsed, reasons = builder.parse_format_near_mc(prompt)
                self.assertEqual(reasons, [])
                self.assertIsNotNone(parsed)

    def test_three_choices_extra_e_and_repeated_runs_are_rejected(self) -> None:
        prompts = (
            "题干 A.甲 B.乙 C.丙",
            "题干 A.甲 B.乙 C.丙 D.丁 E.戊",
            "题干 A.甲 B.乙 C.丙 D.丁 A.一 B.二 C.三 D.四",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                parsed, reasons = builder.parse_format_near_mc(prompt)
                self.assertIsNone(parsed)
                self.assertTrue(reasons)

    def test_duplicate_choice_text_is_rejected(self) -> None:
        parsed, reasons = builder.parse_format_near_mc("题干 A.甲 B.乙 C.甲 D.丁")
        self.assertIsNone(parsed)
        self.assertIn("duplicate_or_empty_choice", reasons)


class PromptOnlyPolicyTest(unittest.TestCase):
    def assert_rejected(self, prompt: str, reason: str) -> None:
        parsed, parse_reasons = builder.parse_format_near_mc(prompt)
        self.assertEqual(parse_reasons, [])
        policy_reasons, _language = builder.prompt_policy_reasons(prompt, parsed)
        self.assertIn(reason, policy_reasons)

    def test_safe_stable_nonmath_prompt_passes(self) -> None:
        prompt = "中国最高峰是 A.珠穆朗玛峰 B.马卡鲁峰 C.富士山 D.喜马拉雅山"
        parsed, parse_reasons = builder.parse_format_near_mc(prompt)
        self.assertEqual(parse_reasons, [])
        policy_reasons, _language = builder.prompt_policy_reasons(prompt, parsed)
        self.assertEqual(policy_reasons, [])

    def test_required_risk_families_are_rejected(self) -> None:
        cases = (
            ("冠心病患者应选择哪项 A.甲 B.乙 C.丙 D.丁", "medical_or_other_high_risk"),
            ("根据某法律应选择哪项 A.甲 B.乙 C.丙 D.丁", "legal"),
            ("现任总统属于哪项 A.甲 B.乙 C.丙 D.丁", "political"),
            ("股票投资属于哪项 A.甲 B.乙 C.丙 D.丁", "financial"),
            ("目前人口情况属于哪项 A.甲 B.乙 C.丙 D.丁", "time_sensitive"),
            ("请根据文章选择哪项 A.甲 B.乙 C.丙 D.丁", "reading_comprehension"),
            ("哪些说法正确 A.甲 B.乙 C.丙 D.丁", "multiselect_or_combination"),
            ("方程的解是哪项 A.甲 B.乙 C.丙 D.丁", "math_or_computation"),
            ("请编写代码选择哪项 A.甲 B.乙 C.丙 D.丁", "code_or_transformation"),
            ("选择正确的单词填入句子 A.甲 B.乙 C.丙 D.丁", "language_form_task"),
            ("你认为哪个最配 A.甲 B.乙 C.丙 D.丁", "subjective_or_normative"),
            ("哪个认识正确 A.我们要甲 B.乙 C.丙 D.丁", "subjective_or_normative"),
            ("哪个正确 A.甲 B.乙 C.丙 D.以上都不正确", "all_none_or_cross_reference"),
        )
        for prompt, reason in cases:
            with self.subTest(reason=reason):
                self.assert_rejected(prompt, reason)

    def test_numeric_extremum_is_math_even_without_math_keyword(self) -> None:
        self.assert_rejected(
            "从以下选项确定哪一个最小 A.0.001 B.0.01 C.0.1 D.1.0",
            "math_or_computation",
        )

    def test_forbidden_key_fragments_are_recursive(self) -> None:
        bad = {"prompt": {"question": "题干", "answer_excerpt": "秘密"}}
        self.assertEqual(
            builder.forbidden_packet_key_paths(bad),
            ["prompt.answer_excerpt"],
        )


class BuiltAssetTest(unittest.TestCase):
    def test_packet_and_audit_are_prompt_only_and_consistent(self) -> None:
        self.assertTrue(builder.OUTPUT.exists(), "run the packet builder before this asset test")
        self.assertTrue(builder.AUDIT.exists(), "run the packet builder before this asset test")
        with builder.OUTPUT.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        audit = json.loads(builder.AUDIT.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), EXPECTED_PACKET_ROWS)
        self.assertEqual(sha256_file(builder.OUTPUT), EXPECTED_PACKET_SHA256)
        self.assertEqual([row["record_id"] for row in rows], sorted(row["record_id"] for row in rows))
        self.assertEqual(len({row["record_id"] for row in rows}), len(rows))
        self.assertEqual(builder.forbidden_packet_key_paths(rows), [])
        self.assertEqual(audit["input"]["sha256"], builder.EXPECTED_INPUT_SHA256)
        self.assertEqual(audit["input"]["rows"], builder.EXPECTED_INPUT_ROWS)
        self.assertEqual(audit["input"]["target_rows"], builder.EXPECTED_TARGET_ROWS)
        self.assertEqual(audit["filtering"]["accepted_rows"], len(rows))
        self.assertEqual(audit["output"]["sha256"], sha256_file(builder.OUTPUT))
        self.assertEqual(audit["output"]["training_eligible_rows"], 0)
        self.assertEqual(audit["prompt_only_isolation"]["forbidden_key_path_hits"], [])
        self.assertFalse(audit["prompt_only_isolation"]["source_assistant_text_copied"])

        for row in rows:
            self.assertEqual(row["task_type"], "world_mc_prompt_only_format_recovery_candidate")
            self.assertEqual(row["lineage"]["source"], builder.SOURCE_NAME)
            self.assertEqual(list(row["prompt"]["options"]), list("ABCD"))
            self.assertFalse(row["candidate_state"]["training_eligible"])
            self.assertFalse(row["candidate_state"]["factuality_verified"])
            self.assertEqual(row["candidate_state"]["status"], "pending_human_prompt_and_factual_review")


if __name__ == "__main__":
    unittest.main()
