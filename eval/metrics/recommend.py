#!/usr/bin/env python3
"""metrics/recommend.py — 懂推荐（Pass@64，两路合并）。

对齐 docs/评测部分解析.md 第 352-473 行定义：

1. Thinking mode：先 rollout 一条 CoT，再生成 32 个 item id 候选。
2. Non-thinking mode：直接（空 think）生成 32 个 item id 候选。
3. 两路结果去重合并，形成 64 item ids 候选池。
4. 只要候选池中任意候选 item id 命中任意 ground truth item id 即通过，否则为 0。

注：文档"answer 可含多个 gold item ids"（见 docs/platform_guide.md 第170行摘要），
因此 gold 用集合表示，候选池与 gold 集合有交集即算命中。
"""
from __future__ import annotations


def merge_candidate_pools(thinking_ids: list[int], non_thinking_ids: list[int]) -> list[int]:
    """两路结果去重合并，形成候选池（文档第403行）。返回去重后的列表，保留首次出现顺序。"""
    seen = set()
    merged = []
    for item_id in thinking_ids + non_thinking_ids:
        if item_id not in seen:
            seen.add(item_id)
            merged.append(item_id)
    return merged


def pass_at_64_recommend(
    thinking_ids: list[int], non_thinking_ids: list[int], gold_item_ids: list[int]
) -> dict:
    """整合两路候选合并 + 任一命中判定（文档第404行）。"""
    merged = merge_candidate_pools(thinking_ids, non_thinking_ids)
    gold_set = set(gold_item_ids)
    hit = bool(set(merged) & gold_set)
    return {
        "merged_pool": merged,
        "pool_size": len(merged),
        "pass@64": int(hit),
    }
