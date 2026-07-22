#!/usr/bin/env python3
"""tests/test_text_utils.py — common/text_utils.py 单元测试（自行构造，B类）。

覆盖 strip_think / extract_json_array / extract_json_object 的核心行为与边界情况。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text_utils import extract_json_array, extract_json_object, strip_think


# --------------------------- strip_think ------------------------------------
def test_strip_think_removes_think_block():
    text = "<think>这是思考过程</think>\n这是正文"
    assert strip_think(text) == "\n这是正文"


def test_strip_think_no_think_tag_returns_original():
    text = "没有think标签的纯文本"
    assert strip_think(text) == text


def test_strip_think_unclosed_think_returns_empty():
    text = "<think>思考到一半就截断了，没有闭合标签"
    assert strip_think(text) == ""


def test_strip_think_empty_input():
    assert strip_think("") == ""
    assert strip_think(None) == ""


# --------------------------- extract_json_array ------------------------------
def test_extract_json_array_after_think_block():
    text = '<think>思考过程中提到了[1,2,3]这种数组</think>\n["a", "b"]'
    result = extract_json_array(text)
    assert result == ["a", "b"]


def test_extract_json_array_no_think_block():
    text = '这是回答：["x", "y", "z"]'
    result = extract_json_array(text)
    assert result == ["x", "y", "z"]


def test_extract_json_array_nested_brackets():
    text = '结果是 [{"a": [1, 2]}, {"b": 3}]'
    result = extract_json_array(text)
    assert result == [{"a": [1, 2]}, {"b": 3}]


def test_extract_json_array_no_array_returns_none():
    assert extract_json_array("这里没有任何数组") is None
    assert extract_json_array("") is None


def test_extract_json_array_picks_last_valid_array_in_segment():
    """当正文中出现多个候选 [...] 片段时，优先取能成功解析的那个（从后往前找）。"""
    text = "无关内容 [not valid json] 真正的结果 [1, 2, 3]"
    result = extract_json_array(text)
    assert result == [1, 2, 3]


# --------------------------- extract_json_object -----------------------------
def test_extract_json_object_after_think_block():
    text = '<think>无关思考</think>\n{"logic_chain": {"name": "test", "events": []}}'
    result = extract_json_object(text)
    assert result == {"logic_chain": {"name": "test", "events": []}}


def test_extract_json_object_no_think_block():
    text = '{"key": "value"}'
    assert extract_json_object(text) == {"key": "value"}


def test_extract_json_object_no_object_returns_none():
    assert extract_json_object("没有任何JSON对象") is None
    assert extract_json_object("") is None


def test_extract_json_object_skips_invalid_then_finds_valid():
    text = '前缀噪声 {invalid json here} 然后是 {"valid": true}'
    result = extract_json_object(text)
    assert result == {"valid": True}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
