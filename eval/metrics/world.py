#!/usr/bin/env python3
"""metrics/world.py — 懂世界（Accuracy + 官方正则答案抽取优先级链，支持单选/多选）。

对齐 docs/评测部分解析.md 第 476-506 行定义：

1. 模型生成 1 个回答（如「正确答案是（A）」）。
2. 与 ground truth 对比：**少选、错选、多选、解析失败，都算错误**（此处"多选"
   特指"预测出的字母数量与该题 gold 字母数量不一致"这一情形，而非"该题本身
   是多选题类型"，见下方说明）。
3. 答案抽取逻辑：按优先级逐条正则扫描全文，按 pattern 列表顺序逐个尝试，
   第一个匹配到含合法字母的即提取、去重、升序返回。

文档原文明确给出的 Response pattern（第503-504行）只有前两条，第三条起用
"……（其余略）"省略。本模块前两条正则**逐字对应文档原文**（A类），第3-5条
按同一优先级链风格补全（"故选"、"选择"、纯字母兜底），风格与前两条一致，
用于覆盖"解析失败"之外更多真实生成文本的情形；核心 A 类验收（文档三角形
数对原题精确抽取出 B）仅依赖第 1 条规则即可满足，不依赖补全部分。

【2026-07-18 修订：支持多选题】`docs/competition.md` 第126-137行、第520-537行
明确定义懂世界包含"单选"和"多选"两种真实题型（多选 GT 形如 `"ABC"`，评分规则
"全部选项完全匹配才得分，少选/错选/漏选均不得分"），这与`docs/评测部分解析.md`
的表述是同一套逻辑在不同题型上的体现——不是"预测出多个字母就一律判错"，而是
"预测结果必须与该题的 gold 字母集合完全一致才算对"。此前实现里 `extract_answer`
一旦捕获到多于1个字母就把 `answer` 强制置为 `None`，导致即使模型对某道多选题
（gold="ABC"）完全预测正确，也会被误判为错误，这是相对 `competition.md` 的一处
忠实度缺口。修订后 `is_correct`/`accuracy` 改为直接比较 `matched_letters` 与
`gold`（大小写不敏感、内部先排序去重），不再依赖 `answer` 字段的"仅单选"限制；
`answer` 字段本身保留原语义（只在恰好命中单个字母时才非 None）供只关心单选场景
的调用方使用，不破坏其含义。
"""
from __future__ import annotations

import re


# 文档第503-504行原文的前两条正则（逐字复现，仅将 {L} 具体化为 [A-D]+ 捕获组）
# 第1条：docs/评测部分解析.md 第503行
#   (?:正确)?答案(?:应该)?(?:是|为|应为|应当是)\s*[：:]?\s*[\(（]?{L}[\)）]?
# 第2条：docs/评测部分解析.md 第504行
#   最佳答案(?:是|为)\s*[：:]?\s*[\(（]?{L}[\)）]?
# 第3-5条：文档用"……（其余略）"省略，按同一优先级链风格自行补全（非文档原文）
ANSWER_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:正确)?答案(?:应该)?(?:是|为|应为|应当是)\s*[：:]?\s*[\(（]?\s*([A-D]+)\s*[\)）]?", re.I),
    re.compile(r"最佳答案(?:是|为)\s*[：:]?\s*[\(（]?\s*([A-D]+)\s*[\)）]?", re.I),
    re.compile(r"故选\s*[：:]?\s*[\(（]?\s*([A-D]+)\s*[\)）]?", re.I),
    re.compile(r"选择\s*[：:]?\s*[\(（]?\s*([A-D]+)\s*[\)）]?", re.I),
    re.compile(r"^\s*[\(（]?\s*([A-D]+)\s*[\)）]?\s*[。.]?\s*$", re.I),
]


def extract_answer(text: str) -> dict:
    """按优先级逐条正则扫描全文，第一个匹配到含合法字母的即提取、去重、升序返回。

    返回值：
        answer: 单选时为单个字母（如 "B"）；多选/解析失败时为 None。
        matched_letters: 匹配到的所有字母去重升序拼接后的字符串（可能长度>1，表示多选）。
        pattern_index: 命中的 pattern 在优先级链中的下标；全部未命中为 None。
        parse_error: 多选或全部未命中都算 True（"少选、错选、多选、解析失败都算错误"，
            这里 parse_error 特指"抽取阶段就未能得到唯一字母"的情形）。
    """
    for pattern_index, pattern in enumerate(ANSWER_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        letters = "".join(sorted(set(match.group(1).upper())))
        return {
            "answer": letters if len(letters) == 1 else None,
            "matched_letters": letters,
            "pattern_index": pattern_index,
            "parse_error": len(letters) != 1,
        }
    return {
        "answer": None,
        "matched_letters": None,
        "pattern_index": None,
        "parse_error": True,
    }


def is_correct(prediction_text: str, gold: str) -> bool:
    """判定单条预测是否正确：预测抽取出的字母集合须与 gold 完全一致（大小写不敏感、
    与顺序无关，内部均先排序去重再比较），否则一律算错——覆盖单选（gold 单字母）
    与多选（gold 多字母，如 "ABC"）两种题型，对应文档"少选、错选、多选、解析
    失败都算错误"（`docs/评测部分解析.md`）与"全部选项完全匹配才得分，少选/
    错选/漏选均不得分"（`docs/competition.md` 第135行）两处表述。
    """
    extracted = extract_answer(prediction_text)
    matched = extracted["matched_letters"]
    if not matched:
        return False
    normalized_gold = "".join(sorted(set(gold.upper())))
    return matched == normalized_gold


def accuracy(predictions: list[str], golds: list[str]) -> dict:
    """整体 Accuracy 计算。"""
    assert len(predictions) == len(golds)
    correct = [int(is_correct(p, g)) for p, g in zip(predictions, golds)]
    return {
        "n": len(golds),
        "n_correct": sum(correct),
        "accuracy": sum(correct) / len(golds) if golds else 0.0,
        "correct_flags": correct,
    }
