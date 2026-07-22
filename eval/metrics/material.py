#!/usr/bin/env python3
"""metrics/material.py — 懂物料：深度理解语义 ID（Pass@64）。

对齐 docs/评测部分解析.md 第 29-62 行定义：

1. 模型 beam search 生成 64 个 itemic patterns（形如 `<s_a_0><s_b_1><s_c_2>`）。
2. 将每个 itemic pattern 映射到 item id（映射不到则为 0，即 invalid）。
3. 将预测得到的 64 个 item id 与 ground truth item_id 对比，64 个候选任一命中即通过，
   全部未命中则为 0。

映射方法（文档第61行）：维护一个 itemic pattern -> item ids 的映射表；当一个 itemic
pattern 对应多个 item ids 时，取最新 Pid（本模块假设映射表条目按 Pid 升序或显式提供
"最新"标记，取列表中给定顺序的最后一个作为"最新"）。
"""
from __future__ import annotations

from dataclasses import dataclass


INVALID_ITEM_ID = 0


@dataclass
class PatternMapping:
    """itemic pattern -> item_ids 映射表。

    内部结构：{pattern: [item_id, ...]}，列表按"从旧到新"的 Pid 顺序存放；
    多对一取最新 Pid 时取列表最后一个元素。
    """

    table: dict[str, list[int]]

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, int]]) -> "PatternMapping":
        """从 (pattern, item_id, ...) 列表按插入顺序构建映射表，同一 pattern
        多次出现视为多个 Pid，按传入顺序保留，最后一个视为最新。
        """
        table: dict[str, list[int]] = {}
        for pattern, item_id in pairs:
            table.setdefault(pattern, []).append(item_id)
        return cls(table=table)

    def resolve(self, pattern: str) -> int:
        """把 itemic pattern 映射到 item_id；映射不到则返回 INVALID_ITEM_ID(0)；
        一个 pattern 对应多个 item_ids 时取最新 Pid（列表最后一个）。
        """
        ids = self.table.get(pattern)
        if not ids:
            return INVALID_ITEM_ID
        return ids[-1]


def resolve_candidates(patterns: list[str], mapping: PatternMapping) -> list[int]:
    """把 beam64 生成的 itemic pattern 列表逐个映射为 item_id 候选列表。"""
    return [mapping.resolve(p) for p in patterns]


def pass_at_64(candidates: list[int], gold_item_id: int) -> int:
    """64 个候选 item_id 任一命中 gold 即通过（返回1），全部未命中返回0。

    INVALID_ITEM_ID(0) 不应被当作"命中"，即使 gold_item_id 意外为0也不计分
    （0 在文档定义中专门代表"映射失败"，不是合法 item id）。
    """
    if gold_item_id == INVALID_ITEM_ID:
        return 0
    return int(gold_item_id in set(candidates) - {INVALID_ITEM_ID})


def evaluate_material(
    beam_patterns: list[str], mapping: PatternMapping, gold_item_id: int
) -> dict:
    """整合完整三步流程：beam patterns -> item_id 候选 -> Pass@64 判定。"""
    candidates = resolve_candidates(beam_patterns, mapping)
    passed = pass_at_64(candidates, gold_item_id)
    return {
        "candidates": candidates,
        "n_invalid": candidates.count(INVALID_ITEM_ID),
        "n_distinct_valid": len(set(candidates) - {INVALID_ITEM_ID}),
        "pass@64": passed,
    }
