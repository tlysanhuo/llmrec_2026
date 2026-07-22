#!/usr/bin/env python3
"""tests/test_text_match.py — common/text_match.py 基础单元测试。

覆盖 tok_f1 / rouge_l / dice_f1 / optimal_matching 的边界情况：
空集合、完全相同、完全不同、部分重叠。这些是任务1（项目骨架与共享组件）
验收标准要求的基础覆盖，属于自行构造的最小算例（非文档给出的数值）。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text_match import dice_f1, lcs_len, optimal_matching, rouge_l, tok_f1


def approx(a, b, tol=1e-6):
    return math.isclose(a, b, abs_tol=tol)


# --------------------------- tok_f1 ---------------------------------------
def test_tok_f1_empty():
    assert tok_f1([], []) == 0.0
    assert tok_f1([], list("abc")) == 0.0
    assert tok_f1(list("abc"), []) == 0.0


def test_tok_f1_identical():
    assert approx(tok_f1(list("abc"), list("abc")), 1.0)


def test_tok_f1_disjoint():
    assert tok_f1(list("abc"), list("xyz")) == 0.0


def test_tok_f1_partial_overlap():
    # a={a,a,b} b={a,b,b} 交集(多重集) = {a:1, b:1} = 2
    # p = 2/3, r = 2/3, f1 = 2*(2/3)*(2/3)/(4/3) = 2/3
    a, b = list("aab"), list("abb")
    assert approx(tok_f1(a, b), 2 / 3)


# --------------------------- rouge_l ---------------------------------------
def test_rouge_l_empty():
    assert rouge_l([], []) == 0.0
    assert rouge_l([], list("abc")) == 0.0


def test_rouge_l_identical():
    assert approx(rouge_l(list("hello"), list("hello")), 1.0)


def test_rouge_l_disjoint():
    assert rouge_l(list("abc"), list("xyz")) == 0.0


def test_rouge_l_partial():
    # lcs("abcde", "ace") = 3 ("ace")
    a, b = list("abcde"), list("ace")
    l = lcs_len(a, b)
    assert l == 3
    p, r = l / len(b), l / len(a)
    expected = 2 * p * r / (p + r)
    assert approx(rouge_l(a, b), expected)


# --------------------------- dice_f1 ----------------------------------------
def test_dice_f1_empty_both():
    assert dice_f1([], []) == 0.0


def test_dice_f1_identical_single_token():
    assert approx(dice_f1(["<s_a_1>"], ["<s_a_1>"]), 1.0)


def test_dice_f1_disjoint():
    assert dice_f1(["<s_a_1>"], ["<s_a_2>"]) == 0.0


def test_dice_f1_multi_to_one_ratio():
    # |A|=1, |B|=2, 交集=1 → 2*1/(1+2) = 2/3（文档①vs①案例同款结构）
    assert approx(dice_f1(["x"], ["x", "y"]), 2 / 3)


# --------------------------- optimal_matching -------------------------------
def test_optimal_matching_empty():
    assert optimal_matching([], []) == []
    assert optimal_matching([["a"]], []) == []
    assert optimal_matching([], [["a"]]) == []


def test_optimal_matching_identical_sequences():
    gold = [["a"], ["b"], ["c"]]
    pred = [["a"], ["b"], ["c"]]
    pairs = optimal_matching(gold, pred)
    assert len(pairs) == 3
    for gi, pi, score in pairs:
        assert gi == pi
        assert approx(score, 1.0)


def test_optimal_matching_no_overlap():
    gold = [["a"], ["b"]]
    pred = [["x"], ["y"]]
    assert optimal_matching(gold, pred) == []


def test_optimal_matching_cross_matching():
    """验证"跨越匹配"能力：标准链某节点应匹配到生成链中顺序靠后的对应节点，
    即使中间隔着若干条与之无关的生成链节点（对应文档 ③ vs ⑤ 的跨越匹配特性）。
    """
    gold = [["alpha"], ["beta"]]
    pred = [["gamma"], ["delta"], ["alpha"], ["epsilon"], ["beta"]]
    pairs = optimal_matching(gold, pred)
    pair_map = {gi: (pi, score) for gi, pi, score in pairs}
    assert pair_map[0] == (2, 1.0)  # gold[0]="alpha" matches pred[2]
    assert pair_map[1] == (4, 1.0)  # gold[1]="beta" matches pred[4]


def test_optimal_matching_rejects_fully_reversed_order():
    """保序约束回归测试：gold=[A,B] 与 pred=[B,A]（内容相同但顺序完全颠倒）时，
    不应像无约束的全局二分图匹配（匈牙利算法）那样给出两条交叉连线、总分2.0，
    而应受"有序"约束限制，只能选出一条连线，总分不超过1.0。

    对应 docs/评测部分解析.md 中"最优有序匹配（Optimal Ordered Matching）"及
    "保序约束下不能往回匹配已用过的节点"的表述（第224行漏生成判定说明）。
    """
    gold = [["A"], ["B"]]
    pred = [["B"], ["A"]]
    pairs = optimal_matching(gold, pred)
    total_score = sum(score for _, _, score in pairs)
    assert total_score <= 1.0 + 1e-9
    assert len(pairs) == 1  # 只能保留一条连线，不能同时匹配两条交叉的连线


def test_optimal_matching_preserves_relative_order_of_matched_pairs():
    """已匹配的多对 (gold_idx, pred_idx) 之间，pred_idx 必须随 gold_idx 单调不减，
    这是"有序匹配"的核心不变量（允许跳过/跨越，但不允许往回匹配）。
    """
    gold = [["a"], ["b"], ["c"], ["d"]]
    pred = [["d"], ["c"], ["b"], ["a"]]  # 完全逆序
    pairs = optimal_matching(gold, pred)
    pred_indices = [pi for _, pi, _ in pairs]
    assert pred_indices == sorted(pred_indices)
    # 完全逆序时，保序约束下最多只能保留 1 对匹配
    assert len(pairs) == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
