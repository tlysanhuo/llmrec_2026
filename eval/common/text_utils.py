#!/usr/bin/env python3
"""common/text_utils.py — 通用文本处理：去除 <think> 段、从含前缀/后缀噪声的文本中提取 JSON。

供 `data/loaders.py` 解析官方 sampled 数据（`{"system","prompt","response"}`，
`response` 形如 `<think>...</think>\n正文`）时复用，确保懂物料/懂推荐等维度在从
`response` 中抽取 SemanticID 时先剥离 `<think>` 段，避免把思考过程中提及的历史
SID 误当作 GT（详见 eval/SPEC.md 附录"忠实度对比评估"中记录的踩坑点）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def strip_think(text: str) -> str:
    """去除 <think>...</think> 段（含标签本身），返回剩余正文。

    - 若不存在 `<think>`，原样返回。
    - 若存在 `<think>` 但没有 `</think>`（截断），返回空字符串（保守处理，避免把
      思考过程当正文），调用方应在拿到空结果时自行决定是否退化为使用原文本。
    """
    if not text or "<think>" not in text:
        return text or ""
    if "</think>" not in text:
        return ""
    return _THINK_PATTERN.sub("", text, count=1)


def _find_balanced_span(text: str, start_idx: int, open_ch: str, close_ch: str) -> Optional[str]:
    """从 text[start_idx] == open_ch 起，用括号计数法找到与之匹配的 close_ch，返回子串（含首尾括号）。"""
    depth = 0
    in_str = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _iter_top_level_spans(text: str, open_ch: str, close_ch: str) -> list[str]:
    """从左到右扫描 text，找出所有"顶层"（不嵌套在另一个同类型括号内部）的
    平衡片段，按出现顺序返回。用于避免"内层子结构恰好也能被独立解析"时
    抢先于外层完整结构被选中（如数组套数组、对象套对象的场景）。
    """
    spans = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == open_ch:
            span = _find_balanced_span(text, i, open_ch, close_ch)
            if span is not None:
                spans.append(span)
                i += len(span)
                continue
        i += 1
    return spans


def extract_json_array(text: str) -> Optional[list[Any]]:
    """从文本中找到最后一个可以被 json.loads 成功解析为 list 的顶层 [...] 片段。

    策略：优先在 `</think>` 之后的正文中查找；若找不到，再退化到全文查找。
    只在"顶层"（未嵌套在其它同类括号内部）片段中查找，避免像
    `[{"a": [1, 2]}]` 这种嵌套结构被内层子数组抢先匹配；若同一 segment 内
    出现多个顶层候选（如先有说明性噪声片段、后有真正结果），取最后一个能
    成功解析的。
    """

    def _search(segment: str) -> Optional[list[Any]]:
        spans = _iter_top_level_spans(segment, "[", "]")
        for span in reversed(spans):
            try:
                parsed = json.loads(span)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return parsed
        return None

    if not text:
        return None
    if "</think>" in text:
        after = text.split("</think>", 1)[1]
        result = _search(after)
        if result is not None:
            return result
    return _search(text)


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """从文本中定位 JSON 对象并解析为 dict，优先 `</think>` 之后，再退化全文。

    只在"顶层"（未嵌套在其它同类花括号内部）片段中查找（与 `extract_json_array`
    同一策略，避免嵌套对象被内层子结构抢先匹配），按出现顺序取第一个能成功
    解析为 dict 的顶层片段。
    """

    def _search(segment: str) -> Optional[dict[str, Any]]:
        spans = _iter_top_level_spans(segment, "{", "}")
        for span in spans:
            try:
                parsed = json.loads(span)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    if not text:
        return None
    if "</think>" in text:
        after = text.split("</think>", 1)[1]
        result = _search(after)
        if result is not None:
            return result
    return _search(text)
