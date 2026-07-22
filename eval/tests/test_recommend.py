#!/usr/bin/env python3
"""tests/test_recommend.py — 懂推荐测试（B类：候选数据取自文档原文，gold为自行假设）。

文档（docs/评测部分解析.md 第406-468行）"演示案例"给出的候选 item_id 列表本身也是
用省略号截断的示例，并非完整32个，明确列出的条目为：

    Thinking mode（6个明确条目）：123, 176487, 2487, 89764, 52, 679843
    Non-thinking mode（6个明确条目）：1456, 5787, 367, 2664, 980, 3165

文档没有给出最终 gold item_id 和 Pass@64 数值，本文件用文档给出的这些明确条目
构造去重合并逻辑测试，并自行假设三种 gold 位置场景：
  1. gold 落在 thinking 路候选中 → 命中
  2. gold 落在 non-thinking 路候选中 → 命中
  3. gold 两路都没有 → 不命中
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.recommend import merge_candidate_pools, pass_at_64_recommend

# 文档原文明确列出的候选 item_id（第431-437行、第461-467行），省略号截断的部分不纳入
THINKING_IDS = [123, 176487, 2487, 89764, 52, 679843]
NON_THINKING_IDS = [1456, 5787, 367, 2664, 980, 3165]


def test_merge_deduplicates_and_preserves_order():
    """两路合并去重（文档第403行），无重叠时合并后长度=两路长度之和。"""
    merged = merge_candidate_pools(THINKING_IDS, NON_THINKING_IDS)
    assert len(merged) == len(THINKING_IDS) + len(NON_THINKING_IDS)
    assert merged[: len(THINKING_IDS)] == THINKING_IDS
    assert merged[len(THINKING_IDS):] == NON_THINKING_IDS


def test_merge_deduplicates_overlapping_ids():
    """两路存在重叠 item_id 时，合并结果应去重（只保留一次）。"""
    overlap_thinking = THINKING_IDS + [1456]  # 与 non-thinking 路第一个id重复
    merged = merge_candidate_pools(overlap_thinking, NON_THINKING_IDS)
    assert len(merged) == len(THINKING_IDS) + len(NON_THINKING_IDS)  # 重复项被去重
    assert merged.count(1456) == 1


def test_gold_hit_in_thinking_path():
    """场景1：gold 落在 thinking 路候选中 → Pass@64 = 1（gold=文档thinking路给出的123）。"""
    result = pass_at_64_recommend(THINKING_IDS, NON_THINKING_IDS, gold_item_ids=[123])
    assert result["pass@64"] == 1
    assert 123 in result["merged_pool"]


def test_gold_hit_in_non_thinking_path():
    """场景2：gold 落在 non-thinking 路候选中 → Pass@64 = 1（gold=文档non-thinking路给出的3165）。"""
    result = pass_at_64_recommend(THINKING_IDS, NON_THINKING_IDS, gold_item_ids=[3165])
    assert result["pass@64"] == 1


def test_gold_miss_in_both_paths():
    """场景3：gold 两路都没有 → Pass@64 = 0（自行假设一个不在两路候选中的 item_id）。"""
    result = pass_at_64_recommend(THINKING_IDS, NON_THINKING_IDS, gold_item_ids=[999999999])
    assert result["pass@64"] == 0


def test_multiple_gold_ids_any_hit_counts():
    """文档platform_guide.md注记：answer可含多个gold item ids，任一命中即通过。
    gold集合包含1个命中项+1个未命中项，仍应判定为命中。
    """
    result = pass_at_64_recommend(THINKING_IDS, NON_THINKING_IDS, gold_item_ids=[999999999, 2487])
    assert result["pass@64"] == 1


def test_empty_candidates_never_hit():
    result = pass_at_64_recommend([], [], gold_item_ids=[123])
    assert result["pass@64"] == 0
    assert result["merged_pool"] == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
