#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/build_official_general_world_clean.py"
SPEC = importlib.util.spec_from_file_location("world_clean", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
world_clean = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = world_clean
SPEC.loader.exec_module(world_clean)


class OfficialGeneralWorldCleanTest(unittest.TestCase):
    def test_strict_mc_accepts_exact_abcd(self) -> None:
        prompt = "中国面积最大的淡水湖是哪个？\nA.鄱阳湖\nB.洞庭湖\nC.太湖\nD.洪泽湖"
        parsed, reasons = world_clean.parse_mc_prompt(prompt)
        self.assertEqual(reasons, [])
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(list(parsed.options), list("ABCD"))

    def test_multiselect_is_rejected_before_boilerplate_removal(self) -> None:
        prompt = "以下是一道多项选择题：\n哪些正确？\nA.甲\nB.乙\nC.丙\nD.丁"
        parsed, reasons = world_clean.parse_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assertIn("mc_multiselect", reasons)

    def test_additional_multiselect_phrases_are_rejected(self) -> None:
        for phrase in (
            "请选择两个正确选项", "有多个正确答案", "一个或多个正确项", "不止一个正确项",
            "有两个正确答案", "有2个正确答案", "有两个选项正确", "有2个选项正确",
            "有两项是正确的", "正确的有两项", "正确答案有两个", "正确选项有两项",
            "两个答案是正确的", "请选两项", "选两个", "选二项", "至少两个选项",
            "至少有两个答案", "可选择多个选项", "答案不唯一", "选出所有正确选项",
            "正确的有哪几项", "以下哪些说法正确",
            "选三项", "选3项", "有三个正确答案", "有3个选项正确", "一个以上",
            "超过一个", "不只一个正确答案", "有多项正确", "多个选项是正确的",
            "勾选所有适用项", "choose all", "select two", "more than one answer",
            "至少一项",
        ):
            with self.subTest(phrase=phrase):
                parsed, reasons = world_clean.parse_mc_prompt(
                    f"{phrase}：\nA.甲\nB.乙\nC.丙\nD.丁"
                )
                self.assertIsNone(parsed)
                self.assertIn("mc_multiselect", reasons)

    def test_nested_boilerplate_is_removed_to_fixed_point(self) -> None:
        prompt = "题目：以下是一道单项选择题：\n中国首都是哪里？\nA.北京\nB.上海\nC.广州\nD.深圳"
        parsed, reasons = world_clean.parse_mc_prompt(prompt)
        self.assertEqual(reasons, [])
        assert parsed is not None
        self.assertEqual(parsed.question, "中国首都是哪里?")

    def test_meta_residue_is_rejected(self) -> None:
        prompt = ",请给出答案以及解析过程。中国首都是哪里？\nA.北京\nB.上海\nC.广州\nD.深圳"
        parsed, reasons = world_clean.parse_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assertIn("mc_meta_residue", reasons)

    def test_same_line_option_meta_is_rejected(self) -> None:
        for tail in (
            "丁。请给出答案和解析。",
            "丁，请选择正确答案。",
            "丁 请一步步思考并给出理由。",
            "丁。解决这个问题。",
        ):
            with self.subTest(tail=tail):
                prompt = f"中国首都是哪里？\nA.甲\nB.乙\nC.丙\nD.{tail}"
                parsed, reasons = world_clean.parse_mc_prompt(prompt)
                self.assertIsNone(parsed)
                self.assertIn("mc_option_meta", reasons)

    def test_same_line_option_multiselect_tail_is_rejected(self) -> None:
        for tail in (
            "丁（选择两项）", "丁。选择两项。", "丁（可多选）", "丁；答案不唯一",
            "丁 choose all that apply",
        ):
            with self.subTest(tail=tail):
                parsed, reasons = world_clean.parse_mc_prompt(
                    f"中国首都是哪里？\nA.甲\nB.乙\nC.丙\nD.{tail}"
                )
                self.assertIsNone(parsed)
                self.assertIn("mc_option_meta", reasons)

    def test_prompt_side_answer_or_explanation_leak_is_rejected(self) -> None:
        question_leaks = (
            "题干 答案：A", "题干（正确答案A）", "题干 参考答案为 B", "题干 Answer: C",
            "题干 解析：内容", "题干 正确选项是 B", "题干 正确选项为C", "题干 本题选D",
            "题干 Key: A", "题干 Ans=B", "题干 答案=B", "题干 正确的是 A",
            "题干 故选 A", "题干 A 为正确答案", "题干 correct answer: A",
        )
        for question in question_leaks:
            with self.subTest(question=question):
                parsed, reasons = world_clean.parse_mc_prompt(
                    f"{question}\nA.甲\nB.乙\nC.丙\nD.丁"
                )
                self.assertIsNone(parsed)
                self.assertIn("mc_prompt_gold_leak", reasons)

        option_leaks = ("丁。正确答案是 A", "丁；答案：A", "丁 参考答案为 B", "丁。Answer: C", "丁。解析：内容")
        for tail in option_leaks:
            with self.subTest(tail=tail):
                parsed, reasons = world_clean.parse_mc_prompt(
                    f"中国首都是哪里？\nA.甲\nB.乙\nC.丙\nD.{tail}"
                )
                self.assertIsNone(parsed)
                self.assertIn("mc_option_gold_leak", reasons)

    def test_option_gold_markers_are_rejected(self) -> None:
        for option_a, option_b in (("甲（正确）", "乙"), ("甲 ✓", "乙"), ("*甲", "乙"), ("甲[答案]", "乙"), ("甲", "乙（错误）")):
            with self.subTest(option_a=option_a, option_b=option_b):
                parsed, reasons = world_clean.parse_mc_prompt(
                    f"中国首都是哪里？\nA.{option_a}\nB.{option_b}\nC.丙\nD.丁"
                )
                self.assertIsNone(parsed)
                self.assertIn("mc_option_gold_marker", reasons)

    def test_mathematical_radical_is_not_a_gold_marker(self) -> None:
        parsed, reasons = world_clean.parse_mc_prompt(
            "方程的参数取值是哪一个？\nA.b=±√2\nB.b=2\nC.b=3\nD.b=4"
        )
        self.assertEqual(reasons, [])
        self.assertIsNotNone(parsed)

        for expression in ("2 × 3 = 6", "x × y", "向量 a × b", "√ 2", "± √ 2"):
            with self.subTest(expression=expression):
                parsed, reasons = world_clean.parse_mc_prompt(
                    f"下列数学表达式是哪一个？\nA.{expression}\nB.普通表达式\nC.另一表达式\nD.最后表达式"
                )
                self.assertEqual(reasons, [])
                self.assertIsNotNone(parsed)

    def test_broad_choice_forms_cannot_route_as_qa(self) -> None:
        for prompt in (
            "选择：\na. 甲\nb. 乙\nc. 丙\nd. 丁",
            "选择：\n(A) 甲\n(B) 乙\n(C) 丙\n(D) 丁",
            "选择： A. 甲 B. 乙 C. 丙 D. 丁",
        ):
            with self.subTest(prompt=prompt):
                self.assertGreaterEqual(len(world_clean.broad_option_labels(prompt)), 3)

    def test_plural_which_mc_is_rejected_in_strict_pool(self) -> None:
        parsed, reasons = world_clean.parse_mc_prompt(
            "这项发明包括哪些重要技术成果？\nA.甲\nB.乙\nC.丙\nD.丁"
        )
        self.assertIsNone(parsed)
        self.assertIn("mc_multiselect", reasons)

    def test_extra_option_is_rejected(self) -> None:
        prompt = "哪个正确？\nA.甲\nB.乙\nC.丙\nD.丁\nE.戊"
        parsed, reasons = world_clean.parse_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assertIn("mc_extra_option", reasons)

    def test_duplicate_option_is_rejected(self) -> None:
        prompt = "哪个正确？\nA.甲\nB.乙\nC.甲\nD.丁"
        parsed, reasons = world_clean.parse_mc_prompt(prompt)
        self.assertIsNone(parsed)
        self.assertIn("mc_duplicate_option", reasons)

    def test_english_prose_does_not_become_answer_letters(self) -> None:
        assertions = world_clean.answer_assertions("The answer is correct because it follows the rule.")
        self.assertEqual(assertions, [])

    def test_single_chinese_answer_is_parsed(self) -> None:
        assistant = "<think>先分析各项。</think>\n因此，正确答案是 (C)。"
        answer, evidence, think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertEqual(reasons, [])
        self.assertEqual(answer, "C")
        self.assertIn("正确答案", evidence or "")
        self.assertEqual(think_status, "closed_think")

    def test_english_stem_with_chinese_instruction_and_options_is_rejected(self) -> None:
        prompt = (
            "请选择正确答案：What is the capital city of this imaginary country?\n"
            "A.这是第一个非常详细的中文选项\nB.这是第二个非常详细的中文选项\n"
            "C.这是第三个非常详细的中文选项\nD.这是第四个非常详细的中文选项"
        )
        row, reasons = world_clean.evaluate_mc(
            record_id="x",
            lineage={},
            user=prompt,
            assistant="正确答案是 (A)",
            eval_index=world_clean.LeakageIndex(),
            train_index=world_clean.LeakageIndex(),
        )
        self.assertIsNone(row)
        self.assertIn("non_zh", reasons)

    def test_high_risk_and_subjective_cues_are_rejected(self) -> None:
        self.assertIn("high_risk", world_clean.risk_reasons("根据证券法判断上市监管要求", "答案"))
        self.assertIn("high_risk", world_clean.risk_reasons("两岸关系中的台湾问题", "答案"))
        self.assertIn("subjective", world_clean.risk_reasons("哪项措施最能提升用户满意度？", "答案"))

    def test_dynamic_facts_and_answer_time_anchors_are_rejected(self) -> None:
        for user, assistant in (
            ("目前世界上人口最多的国家是哪个？", "答案"),
            ("美国总统是谁？", "答案"),
            ("世界上人口最多的国家是哪个？", "答案"),
            ("这个国家是哪个？", "截至2023年的最新数据表明是甲国"),
        ):
            with self.subTest(user=user):
                self.assertIn("time_sensitive", world_clean.risk_reasons(user, assistant))

    def test_missing_context_answer_is_rejected_from_qa(self) -> None:
        row, reasons = world_clean.evaluate_qa(
            record_id="x",
            lineage={},
            user="中国历史上这一制度的定义是什么？",
            assistant="未提供相关文本，无法确定答案，请提供内容。",
            eval_index=world_clean.LeakageIndex(),
            train_index=world_clean.LeakageIndex(),
        )
        self.assertIsNone(row)
        self.assertIn("missing_context_answer", reasons)

    def test_uncertain_qa_answers_are_rejected(self) -> None:
        for assistant in (
            "我不知道这个问题的准确答案",
            "答案并不清楚",
            "可能是蔡伦，但我无法核实",
            "大概是东汉蔡伦",
            "没有足够信息判断是谁",
            "抱歉，我不确定具体是谁",
        ):
            with self.subTest(assistant=assistant):
                row, reasons = world_clean.evaluate_qa(
                    record_id="x",
                    lineage={},
                    user="中国古代造纸术是谁改进的？",
                    assistant=assistant,
                    eval_index=world_clean.LeakageIndex(),
                    train_index=world_clean.LeakageIndex(),
                )
                self.assertIsNone(row)
                self.assertIn("missing_context_answer", reasons)

    def test_conflicting_answer_is_rejected(self) -> None:
        assistant = "<think>正确答案是 A。</think>\n正确答案是 (B)。"
        answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertIsNone(answer)
        self.assertIn("answer_conflict", reasons)

    def test_multiletter_answer_is_rejected(self) -> None:
        assistant = "正确答案是 (AC)"
        answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertIsNone(answer)
        self.assertIn("answer_not_single", reasons)

    def test_nonfinal_uncertain_assertion_is_rejected(self) -> None:
        assistant = "答案是 A 还是 B，需要进一步判断。"
        answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertIsNone(answer)
        self.assertIn("answer_not_final", reasons)

    def test_negated_then_corrected_assertion_is_rejected(self) -> None:
        assistant = "答案是 A，但这个判断不正确，实际应为 B。"
        answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertIsNone(answer)
        self.assertTrue({"answer_conflict", "answer_not_final"}.intersection(reasons))

    def test_negative_final_phrases_never_become_gold(self) -> None:
        for assistant in (
            "错误答案是 A。",
            "不正确的答案是 B。",
            "排除的答案为 C。",
            "不应选 D。",
            "最终不应选 B。",
            "不能选 A。",
            "不要选 A。",
            "不选 A。",
            "切勿选 A。",
            "不该选 A。",
            "不应当选 A。",
            "无需选 A。",
        ):
            with self.subTest(assistant=assistant):
                answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
                self.assertIsNone(answer)
                self.assertTrue({"answer_not_final", "answer_negated"}.intersection(reasons))

    def test_uncertain_final_phrases_never_become_gold(self) -> None:
        for assistant in (
            "猜测答案是 A。",
            "不确定答案是 A。",
            "假设答案是 A。",
            "可能选 A。",
            "猜选 A。",
            "暂且选 A。",
            "大概选 A。",
        ):
            with self.subTest(assistant=assistant):
                answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
                self.assertIsNone(answer)
                self.assertTrue({"answer_not_final", "answer_uncertain"}.intersection(reasons))

    def test_negative_or_uncertain_context_rejects_even_positive_last_line(self) -> None:
        for assistant, reason in (
            ("起初可能选 A。\n正确答案是 A。", "answer_uncertain"),
            ("B 不能选。\n正确答案是 A。", "answer_negated"),
        ):
            with self.subTest(assistant=assistant):
                answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
                self.assertIsNone(answer)
                self.assertIn(reason, reasons)

    def test_positive_final_line_after_reasoning_is_accepted(self) -> None:
        assistant = "先比较四个选项。\n综上所述，正确答案是 (D)。"
        answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertEqual(reasons, [])
        self.assertEqual(answer, "D")

        for positive, expected in (("正确答案是 A。", "A"), ("因此，应选 B。", "B"), ("C", "C")):
            with self.subTest(positive=positive):
                answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(positive)
                self.assertEqual(reasons, [])
                self.assertEqual(answer, expected)

    def test_special_thought_and_solution_tags(self) -> None:
        assistant = (
            "<|begin_of_thought|>先排除其他选项。<|end_of_thought|>"
            "<|begin_of_solution|>B"
        )
        answer, _evidence, think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertEqual(reasons, [])
        self.assertEqual(answer, "B")
        self.assertEqual(think_status, "closed_special_think")

    def test_only_special_thought_answer_has_no_final(self) -> None:
        assistant = "<|begin_of_thought|>正确答案是 A。<|end_of_thought|>"
        answer, _evidence, _think_status, reasons = world_clean.parse_mc_answer(assistant)
        self.assertIsNone(answer)
        self.assertIn("no_final", reasons)

    def test_unclosed_think_is_rejected(self) -> None:
        final, status = world_clean.split_reasoning("<think>没有闭合\n正确答案是 A")
        self.assertIsNone(final)
        self.assertEqual(status, "malformed_think")

    def test_option_shuffle_semantic_key_is_stable(self) -> None:
        left = world_clean.ParsedMC("问题", {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}, "")
        right = world_clean.ParsedMC("问题", {"A": "乙", "B": "甲", "C": "丙", "D": "丁"}, "")
        left_keys = world_clean.mc_semantic_keys(left, "A")
        right_keys = world_clean.mc_semantic_keys(right, "B")
        self.assertEqual(left_keys["semantic_key"], right_keys["semantic_key"])
        self.assertNotEqual(left_keys["ordered_qa_hash"], right_keys["ordered_qa_hash"])

    def test_same_question_conflicting_answers_are_both_rejected(self) -> None:
        parsed = world_clean.ParsedMC("问题", {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}, "")
        common = {
            "task_type": "world_mc",
            "lineage": {},
            "raw": {"user": "问题", "assistant": ""},
            "review": {"status": "pending"},
        }
        rows = []
        for record_id, answer in (("1", "A"), ("2", "B")):
            keys = world_clean.mc_semantic_keys(parsed, answer)
            rows.append({
                **common,
                "record_id": record_id,
                "clean": {"answer": answer},
                "quality": {
                    "option_invariant_hash": keys["option_invariant_hash"],
                    "answer_text_norm": keys["answer_text_norm"],
                },
            })
        kept, rejected = world_clean.dedupe_candidates(rows, "world_mc", world_clean.Counter())
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(row["reason_codes"] == ["duplicate_conflict"] for row in rejected))

    def test_parent_wrapper_matches_raw_mc(self) -> None:
        raw = "中国面积最大的淡水湖是哪个？\nA.鄱阳湖\nB.洞庭湖\nC.太湖\nD.洪泽湖"
        index = world_clean.LeakageIndex()
        index.add(world_clean.WORLD_HEAD + raw + world_clean.WORLD_TAIL, "parent", include_near=False)
        parsed, reasons = world_clean.parse_mc_prompt(raw)
        self.assertEqual(reasons, [])
        assert parsed is not None
        hit, modes = index.match(raw, parsed)
        self.assertTrue(hit)
        self.assertTrue({"core_exact", "stem_exact", "ordered_exact"}.intersection(modes))

    def test_eval_mc_stem_cannot_leak_through_qa_route(self) -> None:
        question = "中国古代造纸术是谁改进的？"
        mc = question + "\nA.东汉蔡伦\nB.东汉张衡\nC.南北朝祖冲之\nD.北宋毕昇"
        index = world_clean.LeakageIndex()
        index.add(mc, "eval", include_near=False)
        hit, modes = index.match(question)
        self.assertTrue(hit)
        self.assertIn("stem_text_exact", modes)
        row, reasons = world_clean.evaluate_qa(
            record_id="x",
            lineage={},
            user=question,
            assistant="东汉时期的蔡伦",
            eval_index=index,
            train_index=world_clean.LeakageIndex(),
        )
        self.assertIsNone(row)
        self.assertIn("eval_overlap", reasons)

    def test_noncanonical_eval_mc_forms_still_blacklist_qa_stem(self) -> None:
        question = "中国古代造纸术是谁改进的？"
        variants = (
            question + "\nA.蔡伦\nB.张衡\nC.祖冲之\nD.毕昇\nE.沈括",
            question + "\na.蔡伦\nb.张衡\nc.祖冲之\nd.毕昇",
            question + "\n(A) 蔡伦\n(B) 张衡\n(C) 祖冲之\n(D) 毕昇",
            question + "\nA.蔡伦\nB.张衡\nC.祖冲之",
        )
        for mc in variants:
            with self.subTest(mc=mc):
                index = world_clean.LeakageIndex()
                index.add(mc, "eval", include_near=False)
                hit, modes = index.match(question)
                self.assertTrue(hit)
                self.assertIn("stem_text_exact", modes)

    def test_broad_stem_restarts_after_fake_a_in_question(self) -> None:
        question = "A、B两地各有一批货物，最终运输量是多少？"
        mc = question + "\nA.三千六百吨\nB.四千吨\nC.五千吨\nD.六千吨"
        self.assertEqual(world_clean.extract_broad_mc_stem(mc), world_clean.normalize_raw(question))
        index = world_clean.LeakageIndex()
        index.add(mc, "eval", include_near=False)
        hit, modes = index.match(question)
        self.assertTrue(hit)
        self.assertIn("stem_text_exact", modes)

    def test_broad_stem_uses_last_abcd_run_for_combination_question(self) -> None:
        question = (
            "阅读以下四个陈述并判断组合：\n"
            "A.第一个陈述\nB.第二个陈述\nC.第三个陈述\nD.第四个陈述"
        )
        mc = question + "\nA.只有一二\nB.只有三四\nC.一二三\nD.全部"
        self.assertEqual(world_clean.extract_broad_mc_stem(mc), world_clean.normalize_raw(question))
        stems = world_clean.extract_broad_mc_stems(mc)
        self.assertEqual(len(stems), 2)
        self.assertEqual(stems[-1], world_clean.normalize_raw(question))
        index = world_clean.LeakageIndex()
        index.add(mc, "eval", include_near=False)
        hit, modes = index.match(question)
        self.assertTrue(hit)
        self.assertIn("stem_text_exact", modes)

    def test_eval_qa_cannot_leak_through_mc_route(self) -> None:
        question = "中国古代造纸术是谁改进的？"
        mc = question + "\nA.东汉蔡伦\nB.东汉张衡\nC.南北朝祖冲之\nD.北宋毕昇"
        index = world_clean.LeakageIndex()
        index.add(question, "eval", include_near=False)
        parsed, parse_reasons = world_clean.parse_mc_prompt(mc)
        self.assertEqual(parse_reasons, [])
        assert parsed is not None
        hit, modes = index.match(mc, parsed)
        self.assertTrue(hit)
        self.assertIn("indexed_prompt_as_stem_exact", modes)
        row, reasons = world_clean.evaluate_mc(
            record_id="x",
            lineage={},
            user=mc,
            assistant="正确答案是 (A)。",
            eval_index=index,
            train_index=world_clean.LeakageIndex(),
        )
        self.assertIsNone(row)
        self.assertIn("eval_overlap", reasons)

    def test_source_locator_respects_supplied_root(self) -> None:
        root = Path("/registered/mirror")
        _record_id, lineage = world_clean.source_locator(
            "O", "r", root, root / "part.parquet", 0, 3, "s", "u", "[]"
        )
        self.assertEqual(lineage["shard"], "part.parquet")

    def test_output_paths_must_be_distinct(self) -> None:
        path = world_clean.DEFAULT_MC_OUT
        with self.assertRaisesRegex(RuntimeError, "pairwise distinct"):
            world_clean.ensure_safe_paths((path, path))

    def test_official_assets_are_never_output_targets(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "official assets"):
            world_clean.ensure_safe_paths((world_clean.O2_DIR / "do_not_write.jsonl",))


if __name__ == "__main__":
    unittest.main()
