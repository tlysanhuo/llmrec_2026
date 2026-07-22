#!/usr/bin/env python3
"""tests/test_user_f1.py — 懂用户-F1 测试（B类：文档未给完整gold和F1数值，按F1定义自行构造）。

文档（docs/评测部分解析.md 第65-102行）给出的 Answer 用省略号收尾：
    Answer:
    ["<|video_begin|><s_a_6697><s_b_5857><s_c_5563>",
     "<|prod_begin|><s_a_7622><s_b_6454><s_c_1546>", ...]
即只有 2 条明确条目，不是完整 gold，也没有给出 F1 数值演算。本文件用文档样例中
明确列出的这 2 条 itemic token 作为最小 gold 集合基础，自行构造三种场景：
  1. 预测=gold子集 → P=1, R<1
  2. 预测=gold+多余项 → P<1, R=1
  3. 预测与gold部分重叠 → P<1, R<1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.user_f1 import f1_score

# 文档原文明确给出的两条 gold 条目（第90-91行）
DOC_GOLD_ITEM_1 = "<|video_begin|><s_a_6697><s_b_5857><s_c_5563>"
DOC_GOLD_ITEM_2 = "<|prod_begin|><s_a_7622><s_b_6454><s_c_1546>"
# 自行补充第三条 gold（因文档用省略号截断，构造场景需要 >=3 条 gold 才能体现
# "部分重叠"，此条为自造，非文档原文）
EXTRA_GOLD_ITEM = "<|ad_begin|><s_a_4815><s_b_6234><s_c_2693>"  # 取自文档历史条目原文（第74行）

GOLD = [DOC_GOLD_ITEM_1, DOC_GOLD_ITEM_2, EXTRA_GOLD_ITEM]


def test_prediction_equals_gold_gives_f1_one():
    result = f1_score(list(GOLD), list(GOLD))
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_prediction_is_subset_of_gold():
    """场景1：预测=gold子集 → P=1（预测项全对）, R<1（漏了1条）。
    pred = {item1, item2}，gold = {item1, item2, extra} → P=1, R=2/3, F1=2*1*(2/3)/(1+2/3)=0.8
    """
    pred = [DOC_GOLD_ITEM_1, DOC_GOLD_ITEM_2]
    result = f1_score(pred, GOLD)
    assert result["precision"] == 1.0
    assert abs(result["recall"] - 2 / 3) < 1e-9
    assert abs(result["f1"] - 0.8) < 1e-9


def test_prediction_is_gold_plus_extra():
    """场景2：预测=gold+多余项 → P<1（多了1条噪声）, R=1（gold全部覆盖）。
    pred = gold + {noise}，gold不变 → P=3/4, R=1, F1=2*(3/4)*1/((3/4)+1)=6/7
    """
    noise_item = "<|living_begin|><s_a_9999><s_b_9999><s_c_9999>"  # 自造噪声条目
    pred = list(GOLD) + [noise_item]
    result = f1_score(pred, GOLD)
    assert abs(result["precision"] - 0.75) < 1e-9
    assert result["recall"] == 1.0
    assert abs(result["f1"] - 6 / 7) < 1e-9


def test_prediction_partially_overlaps_gold():
    """场景3：预测与gold部分重叠 → P<1, R<1。
    pred = {item1, noise}，gold = {item1, item2, extra}
    overlap=1, P=1/2, R=1/3, F1=2*(1/2)*(1/3)/((1/2)+(1/3))=2/5
    """
    noise_item = "<|ad_begin|><s_a_1><s_b_1><s_c_1>"
    pred = [DOC_GOLD_ITEM_1, noise_item]
    result = f1_score(pred, GOLD)
    assert abs(result["precision"] - 0.5) < 1e-9
    assert abs(result["recall"] - 1 / 3) < 1e-9
    assert abs(result["f1"] - 0.4) < 1e-9


def test_empty_prediction_gives_zero_f1():
    result = f1_score([], GOLD)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_empty_gold_gives_zero_f1_without_error():
    result = f1_score([DOC_GOLD_ITEM_1], [])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_both_empty_gives_zero_f1_without_error():
    result = f1_score([], [])
    assert result["f1"] == 0.0


def test_duplicate_entries_are_deduplicated():
    """预测中重复条目应去重后再计算（集合语义）。"""
    pred = [DOC_GOLD_ITEM_1, DOC_GOLD_ITEM_1, DOC_GOLD_ITEM_2]
    result = f1_score(pred, GOLD)
    assert result["n_pred"] == 2  # 去重后只剩2条


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
