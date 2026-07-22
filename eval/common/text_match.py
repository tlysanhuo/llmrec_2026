#!/usr/bin/env python3
"""common/text_match.py — 通用文本/序列相似度与最优匹配工具。

供 metrics/user_chain.py（懂用户-逻辑链）等维度复用。所有函数均为纯函数，
不依赖网络/GPU，便于单元测试精确复现 docs/评测部分解析.md 中给出的数值算例。
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence


def lcs_len(a: Sequence, b: Sequence) -> int:
    """最长公共子序列长度（用于 ROUGE-L）。"""
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            m[i + 1][j + 1] = m[i][j] + 1 if a[i] == b[j] else max(m[i][j + 1], m[i + 1][j])
    return m[-1][-1]


def tok_f1(a: Sequence, b: Sequence) -> float:
    """两个 token/字符序列之间的 token 重合 F1（Counter 多重集交集）。"""
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p = overlap / max(sum(cb.values()), 1)
    r = overlap / max(sum(ca.values()), 1)
    return 2 * p * r / (p + r)


def rouge_l(a: Sequence, b: Sequence) -> float:
    """基于最长公共子序列的 ROUGE-L F1。"""
    length = lcs_len(a, b)
    if not length:
        return 0.0
    p = length / max(len(b), 1)
    r = length / max(len(a), 1)
    return 2 * p * r / (p + r)


def dice_f1(a: Sequence, b: Sequence) -> float:
    """成对相似度：Dice 系数形式的类 F1。

    对应文档公式：
        m_F1(A*_k, Â_j) = 2|A*_k ∩ Â_j| / (|A*_k| + |Â_j|)

    这里把 a、b 看作 item token / 关键词的（多重）集合，用 Counter 交集
    统计公共元素个数，支持重复元素（如同一 token 出现两次）。
    """
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    denom = len(a) + len(b)
    if denom == 0:
        return 0.0
    return 2 * inter / denom


def logic_similarity(l1: str, l2: str, *, use_nli: bool = False) -> float:
    """Logic 字段混合相似度：s_logic = 0.5*TokenF1 + 0.5*ROUGE-L。

    官方文档公式（docs/评测部分解析.md 第 321 行）明确只含 Token-F1 与
    ROUGE-L 两项，不含语义模型项；因此默认实现（use_nli=False）严格按
    官方公式计算。

    `use_nli` 是预留的可插拔扩展点：team 从平台日志逆向猜测真实评测器
    EvolutionTopicGenEvaluator 可能还叠加了 NLI CrossEncoder
    (nli-deberta-v3-base) 做语义分量，但这不是官方文档正式定义的公式，
    当前未安装该模型。待用户安装好 NLI 模型后，可在此处接入并把
    use_nli 默认值改为可配置，混合公式与叠加权重需要另行确认，不在本
    次改动范围内。
    """
    if use_nli:
        raise NotImplementedError(
            "NLI CrossEncoder 尚未接入：docs/评测部分解析.md 官方公式不含 NLI 项，"
            "此为预留扩展点，待模型安装后另行实现。"
        )
    a, b = list(l1), list(l2)
    return 0.5 * tok_f1(a, b) + 0.5 * rouge_l(a, b)


def optimal_matching(
    gold_items: Sequence[Sequence],
    pred_items: Sequence[Sequence],
    similarity_fn=dice_f1,
) -> list[tuple[int, int, float]]:
    """在两个"按时间顺序排列"的序列之间求最优**保序**匹配，返回匹配对及其得分。

    【2026-07-18 修订】原实现用 `scipy.optimize.linear_sum_assignment`（匈牙利算法）
    求无约束的全局二分图最大权匹配，允许任意顺序的交叉/颠倒配对。但文档原文明确
    使用"最优**有序**匹配（Optimal Ordered Matching）"这一术语，并在漏生成说明中
    特别强调"标准链④由于……**保序约束下不能往回匹配已用过的节点**，因此判定为
    漏生成"（docs/评测部分解析.md 第224行）——这是对"有序"约束的直接文本证据：
    一旦某个 gold 节点匹配到某个 pred 节点，后续 gold 节点只能匹配"更靠后"的 pred
    节点，不能回头匹配排在已用节点之前的 pred 节点。

    构造反例验证了两种算法的差异：若 gold=[A,B]、pred=[B,A]（内容与顺序完全对调），
    无约束的匈牙利匹配会给出两条交叉连线（A↔pred[1], B↔pred[0]，总分2.0），错误地
    对"完全逆序重排"给出满分；而保序匹配只能选出一条连线（如 A↔pred[1]，总分1.0），
    符合"有序"约束下应有的行为。

    实现改为 DP 保序匹配（等价于加权最长公共子序列对齐）：
        dp[i][j] = max(dp[i-1][j], dp[i][j-1], dp[i-1][j-1] + sim[i-1][j-1])
    其中 i 遍历 gold_items（1..n），j 遍历 pred_items（1..m），不设相似度阈值过滤
    （sim_fn 本身在不相关时接近或等于0，直接累加到 DP 中自然不会主动选中低相似度
    配对）。在文档给出的完整数值算例（标准链①②③④ vs 生成链①②③④⑤，"③跨越匹配⑤"
    只是跳过节点、并未发生真正顺序颠倒）上，本实现与原匈牙利实现给出完全相同的
    匹配结果与 Action Alignment 数值（≈0.593），两种算法在"只跳过不倒序"场景下
    等价，替换不影响已有测试。

    Returns:
        [(gold_idx, pred_idx, score), ...]，按 gold_idx 升序排列，只包含
        score > 0 的匹配对。
    """
    n, m = len(gold_items), len(pred_items)
    if n == 0 or m == 0:
        return []

    sim = [[similarity_fn(gold_items[i], pred_items[j]) for j in range(m)] for i in range(n)]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + sim[i - 1][j - 1]
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], diag)

    # 回溯得到匹配对（0-based 原始索引）
    pairs: list[tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 and j > 0:
        diag = dp[i - 1][j - 1] + sim[i - 1][j - 1]
        if dp[i][j] == diag and diag > dp[i - 1][j] and diag > dp[i][j - 1]:
            if sim[i - 1][j - 1] > 0:
                pairs.append((i - 1, j - 1, sim[i - 1][j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs
