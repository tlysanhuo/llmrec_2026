#!/usr/bin/env python3
"""metrics/user_f1.py — 懂用户：抽取兴趣相关行为（F1-Score）。

对齐 docs/评测部分解析.md 第 65-102 行定义：

1. 模型生成 1 个回答（预测的交互列表，JSON 数组）。
2. 将预测结果与 ground truth 对比，计算 F1-Score。

标准 F1 定义：
    P = |pred ∩ gold| / |pred|
    R = |pred ∩ gold| / |gold|
    F1 = 2PR / (P+R)

预测/gold 均为 itemic pattern（或文本项）字符串的集合，去重后按精确字符串相等比较
（文档示例中条目本身就是完整的 itemic token 字符串，如
`<|video_begin|><s_a_6697><s_b_5857><s_c_5563>`）。
"""
from __future__ import annotations


def f1_score(pred: list[str], gold: list[str]) -> dict:
    """计算预测交互列表与 ground truth 之间的 F1-Score。

    空预测、空gold 均按 0 处理，不抛异常（P/R 分母为0时约定为0，F1随之为0）。
    """
    pred_set = set(pred)
    gold_set = set(gold)
    overlap = len(pred_set & gold_set)

    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
        "n_pred": len(pred_set),
        "n_gold": len(gold_set),
    }
