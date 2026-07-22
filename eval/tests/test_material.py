#!/usr/bin/env python3
"""tests/test_material.py — 懂物料 Pass@64 测试（B类：文档未给数值算例，按规则自行构造）。

文档（docs/评测部分解析.md 第29-62行）只给出 1 条输入输出样例：
    Answer (Itemic Pattern Version): <|video_begin|><s_a_3915><s_b_8150><s_c_535>
    Answer (Item ID Version): 1234
以及三步规则描述（beam64生成候选 -> pattern映射item_id -> 64候选任一命中即通过），
但没有给出"64候选命中判定"的数值演算。本文件用文档样例的 pattern/item_id 构造
映射表条目，自行构造以下三个断言场景：
  1. 64候选中含1个命中 → Pass@64 = 1
  2. 64候选全部未命中 → Pass@64 = 0
  3. 同一pattern映射多个item_id时，取最新Pid
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.material import (
    INVALID_ITEM_ID,
    PatternMapping,
    evaluate_material,
    pass_at_64,
    resolve_candidates,
)

# 文档原文样例（第46-50行）
DOC_PATTERN = "<s_a_3915><s_b_8150><s_c_535>"
DOC_ITEM_ID = 1234


def _fake_beam64(hit_pattern: str | None, filler_pattern: str = "<s_a_0><s_b_0><s_c_0>") -> list[str]:
    """构造64个候选 pattern：若 hit_pattern 非空，放在第0位，其余63个用 filler 占位。"""
    if hit_pattern is None:
        return [filler_pattern] * 64
    return [hit_pattern] + [filler_pattern] * 63


def test_resolve_single_pattern_matches_document_example():
    """映射表只含文档样例这一条时，pattern 应精确映射到文档给出的 item_id=1234。"""
    mapping = PatternMapping.from_pairs([(DOC_PATTERN, DOC_ITEM_ID)])
    assert mapping.resolve(DOC_PATTERN) == DOC_ITEM_ID


def test_resolve_unknown_pattern_returns_invalid():
    """映射不到的 pattern 应返回 0（invalid），对齐文档第56行"如映射不到则为0"。"""
    mapping = PatternMapping.from_pairs([(DOC_PATTERN, DOC_ITEM_ID)])
    assert mapping.resolve("<s_a_9999><s_b_9999><s_c_9999>") == INVALID_ITEM_ID


def test_multi_to_one_pattern_takes_latest_pid():
    """一个 pattern 对应多个 item_ids 时取最新 Pid（文档第61行）。
    自行构造：pattern 先后对应 item_id=100（旧）和 1234（新，文档样例值），
    应取 1234。
    """
    mapping = PatternMapping.from_pairs([(DOC_PATTERN, 100), (DOC_PATTERN, DOC_ITEM_ID)])
    assert mapping.resolve(DOC_PATTERN) == DOC_ITEM_ID


def test_pass_at_64_hits_when_one_of_64_candidates_matches_gold():
    """场景1：64候选中含1个命中 → Pass@64 = 1。"""
    mapping = PatternMapping.from_pairs([(DOC_PATTERN, DOC_ITEM_ID)])
    beam = _fake_beam64(hit_pattern=DOC_PATTERN)
    assert len(beam) == 64
    result = evaluate_material(beam, mapping, gold_item_id=DOC_ITEM_ID)
    assert result["pass@64"] == 1
    assert DOC_ITEM_ID in result["candidates"]


def test_pass_at_64_misses_when_none_of_64_candidates_match_gold():
    """场景2：64候选全部未命中 → Pass@64 = 0。"""
    mapping = PatternMapping.from_pairs([(DOC_PATTERN, DOC_ITEM_ID)])
    beam = _fake_beam64(hit_pattern=None)  # 全部是 filler，不含 gold pattern
    result = evaluate_material(beam, mapping, gold_item_id=DOC_ITEM_ID)
    assert result["pass@64"] == 0


def test_pass_at_64_all_invalid_candidates_never_pass():
    """全部候选都映射失败（invalid=0）时，即使凑巧 gold 也是非法值，也不应记为命中。"""
    mapping = PatternMapping.from_pairs([])  # 空映射表，全部 pattern 都映射不到
    beam = _fake_beam64(hit_pattern=DOC_PATTERN)
    result = evaluate_material(beam, mapping, gold_item_id=DOC_ITEM_ID)
    assert result["pass@64"] == 0
    assert result["n_invalid"] == 64


def test_resolve_candidates_length_matches_beam_width():
    mapping = PatternMapping.from_pairs([(DOC_PATTERN, DOC_ITEM_ID)])
    beam = _fake_beam64(hit_pattern=DOC_PATTERN)
    candidates = resolve_candidates(beam, mapping)
    assert len(candidates) == 64


def test_pass_at_64_direct_function():
    assert pass_at_64([DOC_ITEM_ID, 0, 0], DOC_ITEM_ID) == 1
    assert pass_at_64([0, 0, 0], DOC_ITEM_ID) == 0
    assert pass_at_64([1, 2, 3], DOC_ITEM_ID) == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
