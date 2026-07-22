#!/usr/bin/env python3
"""metrics/user_chain.py — 懂用户：给定兴趣抽取逻辑链条（Action/Logic Alignment）。

严格对齐 docs/评测部分解析.md 第 105-348 行定义的官方公式：

1. 标准链 E* 与生成链 Ê 按 action 做「最优有序匹配」，得到匹配对集合 M_a。
   相似度用逐 event 的 Dice 系数 m_F1(A*_k, Â_j) = 2|A*_k∩Â_j| / (|A*_k|+|Â_j|)，
   把 action 内容看作 item token / 关键词的字符集合。
2. Action Alignment：
       P_a = Σ m_F1 / |Ê|，R_a = Σ m_F1 / |E*|，Action Alignment = 2*P_a*R_a/(P_a+R_a)
3. Logic Alignment：复用 M_a，对每个匹配对的 logic 文本算
       s_logic = 0.5*TokenF1 + 0.5*ROUGE-L
   再走同样的 P_l/R_l/F1 流程。
4. 综合分 = (Action Alignment + Logic Alignment) / 2。

「最优有序匹配」允许跨越匹配（标准链某节点匹配到生成链中顺序靠后、中间跳过若干
未匹配节点的对应节点），本质是在相似度矩阵上求全局最优的二分图匹配（见
common/text_match.optimal_matching），而不是简单的顺序 LCS 逐位对齐。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from common.text_match import dice_f1, logic_similarity, optimal_matching

_ACTION_TYPE_RE = re.compile(r"^\[([^\]]+)\]\s*")
_ITEM_TOKEN_RE = re.compile(r"<\|?\w+_begin\|?><s_a_\d+><s_b_\d+><s_c_\d+>")


@dataclass
class Event:
    """一条 logic_chain 事件。action 是「[行为类型] 具体内容」字符串。"""

    action: str
    logic: str


@dataclass
class ChainScoreResult:
    action_alignment: float
    logic_alignment: float
    overall_score: float
    matched_pairs: list[tuple[int, int, float]]
    precision_action: float
    recall_action: float
    precision_logic: float
    recall_logic: float
    n_gold: int
    n_pred: int


def _action_type(action: str) -> str | None:
    """提取 action 前缀的行为类型标签，如 '[视频-长播]'。"""
    m = _ACTION_TYPE_RE.match(action.strip())
    return m.group(1) if m else None


def _action_tokens(action: str) -> list[str]:
    """把 action 内容切分为可比较的 token/字符集合。

    文档演算的关键约束（第237行 + 262-267行示例）是：
      - 不同 action_type（如 '直播-关注' vs '商品-购买' vs '搜索'）之间即使
        文本表层有重叠字符（如标点、常见汉字），也应判定为交集为空、
        m_F1=0，不应产生虚假的非零相似度。
      - 同 action_type 且内容为 itemic token（如 '<video_begin>...'）时，
        按 itemic token 集合算 Dice（支持'；'合并多个token的情形，如
        文档①vs①案例：1个token vs 2个token合并，Dice=2/3）。
      - 同 action_type 且内容为纯文本（如搜索类）时，按文本字符集合算
        Dice（文档③vs⑤案例：文本完全相同，Dice=1）。

    实现：返回一个带 action_type 前缀标记的 token 列表，使得不同
    action_type 之间天然没有公共元素（交集为0），同 action_type 内部再按
    itemic token 优先、否则按字符切分来构造可比较的集合。
    """
    atype = _action_type(action) or ""
    body = action[len(f"[{atype}]"):].strip() if atype else action.strip()

    item_tokens = _ITEM_TOKEN_RE.findall(body)
    if item_tokens:
        # itemic token 场景：token 本身已足够唯一，无需再加 action_type 前缀
        # 也能保证跨 action_type 不会碰巧共享同一个 itemic token 字符串。
        return item_tokens

    # 纯文本场景（如搜索类）：按字符切分，并加 action_type 前缀保证不同
    # action_type 之间不会因共享汉字/标点产生虚假交集。
    return [f"{atype}::{ch}" for ch in body]


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def score_chain(gold_events: list[Event], pred_events: list[Event]) -> ChainScoreResult:
    """计算一对标准链/生成链的 Action Alignment、Logic Alignment 与综合分。"""
    n_gold, n_pred = len(gold_events), len(pred_events)

    gold_sets = [_action_tokens(e.action) for e in gold_events]
    pred_sets = [_action_tokens(e.action) for e in pred_events]

    pairs = optimal_matching(gold_sets, pred_sets, similarity_fn=dice_f1)

    # ---- Action Alignment ----
    action_sum = sum(score for _, _, score in pairs)
    p_a = action_sum / n_pred if n_pred else 0.0
    r_a = action_sum / n_gold if n_gold else 0.0
    action_alignment = _f1(p_a, r_a)

    # ---- Logic Alignment（复用同一批匹配对） ----
    logic_sum = 0.0
    for gi, pi, _ in pairs:
        logic_sum += logic_similarity(gold_events[gi].logic, pred_events[pi].logic)
    p_l = logic_sum / n_pred if n_pred else 0.0
    r_l = logic_sum / n_gold if n_gold else 0.0
    logic_alignment = _f1(p_l, r_l)

    overall = (action_alignment + logic_alignment) / 2

    return ChainScoreResult(
        action_alignment=action_alignment,
        logic_alignment=logic_alignment,
        overall_score=overall,
        matched_pairs=pairs,
        precision_action=p_a,
        recall_action=r_a,
        precision_logic=p_l,
        recall_logic=r_l,
        n_gold=n_gold,
        n_pred=n_pred,
    )


def parse_events(payload: dict) -> list[Event]:
    """从官方 JSON schema `{"logic_chain": {"events": [...]}}` 解析出 Event 列表。"""
    events = payload.get("logic_chain", {}).get("events", [])
    return [Event(action=e.get("action", ""), logic=e.get("logic", "")) for e in events]
