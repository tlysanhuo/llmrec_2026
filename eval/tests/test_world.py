#!/usr/bin/env python3
"""tests/test_world.py — 懂世界测试（A类：文档给出完整算例，原样复现）。

文档（docs/评测部分解析.md 第476-506行）给出完整算例：
    题目：三角形数对题（A点(1,5) B点(1,1) C点(3,1)，选项 A锐角 B直角 C钝角 D等腰）
    gold = B
    答案抽取逻辑：按优先级逐条正则扫描全文，第一个匹配到含合法字母的即提取、
    去重、升序返回。
    判错场景：少选、错选、多选、解析失败，都算错误（文档第498行原文）。

本文件先用文档原题文本精确复现 gold=B 的抽取结果（A类核心验收），再用文档
给出的两条正则 pattern 逐条构造匹配样例，并补充"少选、错选、多选、解析失败"
四种官方明确判错场景各至少1例。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.world import accuracy, extract_answer, is_correct

# 文档原文题目（第486-489行）与 gold（第492行）
DOC_QUESTION = (
    "如果A点用数对表示为（1，5），B点用数对表示为（1，1），C点用数对表示为（3，1），"
    "那么三角形ABC一定是（）三角形。\n"
    "A.锐角  B.直角  C.钝角  D.等腰\n"
    '请按以下格式作答："正确答案是(在此处填写选项字母)"'
)
DOC_GOLD = "B"


def test_extract_answer_matches_document_example_exactly():
    """A类核心验收：文档原题模型回答"正确答案是(B)"应精确抽取出 B。"""
    model_response = "正确答案是(B)"
    result = extract_answer(model_response)
    assert result["answer"] == DOC_GOLD
    assert result["pattern_index"] == 0  # 命中文档第503行第1条正则
    assert result["parse_error"] is False


def test_full_pipeline_with_document_question_and_gold():
    """完整跑通：文档题目 + 模型给出正确回答 -> is_correct 应为 True。"""
    model_response = "根据数对计算，AB边和BC边垂直，所以正确答案是(B)"
    assert is_correct(model_response, DOC_GOLD) is True


def test_pattern_1_matches_various_phrasings():
    """文档第503行第1条正则：(?:正确)?答案(?:应该)?(?:是|为|应为|应当是)... 逐条构造匹配样例。"""
    cases = [
        "正确答案是(B)",
        "正确答案是B",
        "答案是B",
        "答案应该是(B)",
        "答案为(B)",
        "答案应为B",
        "答案应当是（B）",  # 中文括号
        "正确答案是：B",
    ]
    for text in cases:
        result = extract_answer(text)
        assert result["answer"] == "B", f"pattern1 should match: {text!r}"
        assert result["pattern_index"] == 0


def test_pattern_2_matches_best_answer_phrasing():
    """文档第504行第2条正则：最佳答案(?:是|为)....

    注：由于第1条正则"(?:正确)?答案..."中"正确"是可选前缀，"最佳答案是(B)"内的
    "答案是(B)"子串本身就能被第1条正则匹配到，按"逐条尝试、第一个匹配即返回"
    的优先级规则（文档第500行），这类文本会先命中 pattern_index=0，这是符合
    优先级链设计本身的行为，不是bug。为了独立验证第2条正则的捕获组本身能正确
    工作，这里直接用正则对象匹配，而不经过完整优先级链。
    """
    from metrics.world import ANSWER_PATTERNS

    pattern_2 = ANSWER_PATTERNS[1]
    cases = ["最佳答案是(B)", "最佳答案为B", "最佳答案是：B"]
    for text in cases:
        match = pattern_2.search(text)
        assert match is not None, f"pattern2 should match: {text!r}"
        assert match.group(1).upper() == "B"

    # 而在完整优先级链中，由于pattern0更宽泛，会先于pattern1命中，这里显式断言该行为
    result = extract_answer("最佳答案是(B)")
    assert result["answer"] == "B"
    assert result["pattern_index"] == 0


def test_pattern_priority_first_match_wins():
    """当文本同时含第1条和第2条pattern可匹配内容时，优先级链应命中排在前面的第1条。"""
    text = "经过分析，最佳答案是(C)，因此正确答案是(B)"
    result = extract_answer(text)
    # 正则 search 从左到右扫描，"最佳答案是(C)"先出现，但优先级链先尝试pattern0再pattern1，
    # pattern0 是"(?:正确)?答案..."，它本身也能匹配"最佳答案是(C)"中的"答案是(C)"部分
    assert result["pattern_index"] == 0


def test_case_underselect_counts_as_error():
    """少选场景：模型只字面提及选项但未给出明确答案格式，解析失败按错误处理。"""
    text = "这道题需要分析三角形的边长关系，可能是直角或钝角。"
    result = extract_answer(text)
    assert result["answer"] is None
    assert result["parse_error"] is True
    assert is_correct(text, DOC_GOLD) is False


def test_case_wrong_choice_counts_as_error():
    """错选场景：明确给出唯一答案但选错了字母。"""
    text = "正确答案是(A)"
    result = extract_answer(text)
    assert result["answer"] == "A"
    assert is_correct(text, DOC_GOLD) is False  # gold=B，预测A，错选


def test_case_multi_select_counts_as_error():
    """多选场景：正则捕获到多个字母（如"AB"），文档规定多选算错误。"""
    text = "正确答案是(AB)"
    result = extract_answer(text)
    assert result["answer"] is None  # 多选时不返回单一答案
    assert result["matched_letters"] == "AB"
    assert result["parse_error"] is True
    assert is_correct(text, DOC_GOLD) is False


def test_case_parse_failure_counts_as_error():
    """解析失败场景：全文没有任何pattern能匹配到合法字母。"""
    text = "我不确定这道题的答案，需要更多信息才能判断。"
    result = extract_answer(text)
    assert result["answer"] is None
    assert result["pattern_index"] is None
    assert result["parse_error"] is True
    assert is_correct(text, DOC_GOLD) is False


def test_accuracy_aggregation_over_document_case_and_error_cases():
    """整体Accuracy：1条正确（文档原题）+ 4条错误（少选/错选/多选/解析失败）= 1/5。"""
    predictions = [
        "正确答案是(B)",  # correct
        "这道题需要分析。",  # underselect / parse fail
        "正确答案是(A)",  # wrong choice
        "正确答案是(AB)",  # multi-select
        "我不确定答案。",  # parse failure
    ]
    golds = [DOC_GOLD] * 5
    result = accuracy(predictions, golds)
    assert result["n"] == 5
    assert result["n_correct"] == 1
    assert abs(result["accuracy"] - 0.2) < 1e-9
    assert result["correct_flags"] == [1, 0, 0, 0, 0]


def test_empty_predictions_and_golds():
    result = accuracy([], [])
    assert result["accuracy"] == 0.0
    assert result["n"] == 0


# --------------------------- 多选题（docs/competition.md 第126-137、520-537行）-----
# 官方多选完整样例（第520-537行）：镧系元素题，gold="ABC"
MULTI_GOLD = "ABC"


def test_extract_answer_matches_official_multi_select_example():
    """A类：competition.md 第520-537行多选完整样例，模型给出"正确答案是ABC"应精确
    抽取出 matched_letters="ABC"（多选场景下 answer 字段按原语义仍为 None，
    但 matched_letters 完整保留，用于 is_correct 的多选判分）。
    """
    model_response = "正确答案是ABC"
    result = extract_answer(model_response)
    assert result["matched_letters"] == MULTI_GOLD
    assert result["answer"] is None  # 沿用原语义：非单字母时 answer 为 None


def test_multi_select_fully_correct_counts_as_correct():
    """多选题：模型预测的字母集合与 gold 完全一致（顺序、大小写、去重后）应判对。

    对应 competition.md 第135行"全部选项完全匹配才得分"。
    """
    assert is_correct("正确答案是(ABC)", MULTI_GOLD) is True
    assert is_correct("正确答案是CBA", MULTI_GOLD) is True  # 顺序不敏感
    assert is_correct("正确答案是abc", MULTI_GOLD) is True  # 大小写不敏感


def test_multi_select_partial_underselect_counts_as_error():
    """多选题-漏选：预测只给出部分正确字母（如"AB"，漏选C），应判错。"""
    assert is_correct("正确答案是(AB)", MULTI_GOLD) is False


def test_multi_select_overselect_or_wrong_letter_counts_as_error():
    """多选题-多选/错选：预测包含不在 gold 中的字母，应判错。"""
    assert is_correct("正确答案是(ABCD)", MULTI_GOLD) is False  # 多选了D
    assert is_correct("正确答案是(ABD)", MULTI_GOLD) is False  # 错选D代替C


def test_multi_select_accuracy_aggregation():
    """多选题整体Accuracy：1条完全正确 + 1条漏选 + 1条多选 + 1条解析失败 = 1/4。"""
    predictions = [
        "正确答案是ABC",  # correct
        "正确答案是AB",  # underselect
        "正确答案是ABCD",  # overselect
        "无法判断",  # parse failure
    ]
    golds = [MULTI_GOLD] * 4
    result = accuracy(predictions, golds)
    assert result["n"] == 4
    assert result["n_correct"] == 1
    assert abs(result["accuracy"] - 0.25) < 1e-9


def test_single_select_still_correct_after_multi_select_support():
    """回归验证：支持多选后，单选题（gold为单字母）判分逻辑不受影响。"""
    assert is_correct("正确答案是(B)", "B") is True
    assert is_correct("正确答案是(A)", "B") is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
