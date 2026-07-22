#!/usr/bin/env python3
"""tests/test_sid_utils.py — common/sid_utils.py 单元测试（自行构造，B类）。

覆盖 SemanticID token 解析（parse_sid_tokens）与反查表映射（sid_tokens_to_item_ids）
的核心逻辑，不依赖真实 Pid2Sid parquet 数据（用手工构造的小索引代替），保证测试
在任何环境下都能快速稳定运行。真实数据接入的端到端验证见 tests/test_loaders.py。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.sid_utils import DOMAIN_TO_PREFIX, domain_from_prefix, parse_sid_tokens, sid_tokens_to_item_ids


def test_parse_sid_tokens_single_token():
    text = "商品是<|prod_begin|><s_a_1><s_b_2><s_c_3>，请查收"
    tokens = parse_sid_tokens(text)
    assert tokens == [("prod", 1, 2, 3)]


def test_parse_sid_tokens_multiple_tokens_preserve_order():
    text = "<|video_begin|><s_a_1><s_b_2><s_c_3>，然后<|living_begin|><s_a_9><s_b_8><s_c_7>"
    tokens = parse_sid_tokens(text)
    assert tokens == [("video", 1, 2, 3), ("living", 9, 8, 7)]


def test_parse_sid_tokens_empty_text_returns_empty_list():
    assert parse_sid_tokens("") == []
    assert parse_sid_tokens(None) == []


def test_parse_sid_tokens_ignores_non_sid_text():
    assert parse_sid_tokens("这里没有任何SemanticID") == []


def test_parse_sid_tokens_rejects_unknown_prefix():
    """未知 domain 前缀（不在 video/prod/living/ad 中）不应被解析出来。"""
    text = "<|unknown_begin|><s_a_1><s_b_2><s_c_3>"
    assert parse_sid_tokens(text) == []


def test_sid_tokens_to_item_ids_hits_and_misses():
    index = {("prod", 1, 2, 3): [100, 101], ("video", 9, 8, 7): [200]}
    tokens = [("prod", 1, 2, 3), ("video", 9, 8, 7), ("ad", 0, 0, 0)]  # 最后一个查不到
    result = sid_tokens_to_item_ids(tokens, index)
    assert result == {100, 101, 200}


def test_sid_tokens_to_item_ids_all_miss_returns_empty_set():
    index = {("prod", 1, 2, 3): [100]}
    tokens = [("video", 9, 9, 9)]
    assert sid_tokens_to_item_ids(tokens, index) == set()


def test_domain_from_prefix_roundtrip():
    for domain, prefix in DOMAIN_TO_PREFIX.items():
        assert domain_from_prefix(prefix) == domain


def test_domain_from_prefix_unknown_returns_unknown():
    assert domain_from_prefix("not_a_prefix") == "unknown"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
