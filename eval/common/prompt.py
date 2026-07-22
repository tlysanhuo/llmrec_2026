#!/usr/bin/env python3
"""common/prompt.py — 统一 prompt 构造与官方推理参数表。

对齐 docs/评测部分解析.md 「推理参数」一节（第 509-517 行）：

| 任务   | 模式               | Temperature | Top-k | Top-p |
|--------|--------------------|-------------|-------|-------|
| 懂物料 | \\no_think          | 0.7         | 20    | 0.8   |
| 懂用户 | \\no_think          | 0.6         | 20    | 0.95  |
| 懂推荐 | \\no_think + \\think | 0.6         | 50    | 0.95  |
| 懂世界 | \\no_think          | 0.7         | 20    | 0.8   |

口径对齐修订：真实线上评测日志（测评中间输出.md）证实，懂物料
（challenge_itemic_pattern_grounding）与懂推荐（challenge_recommendation_*
四子任务）在生成阶段并非让模型自由生成完整 itemic pattern（含 domain 前缀），
而是评测框架把对应 domain 的 domain-begin token 作为 prompt_token 硬编码
拼进 prompt 末尾（assistant 段的生成起点），模型只需 beam search 生成 3 个
后续 token（s_a/s_b/s_c）。懂物料线上 100% 为 video 域，固定注入 video 前缀；
懂推荐按每条样本的目标域（video/prod/ad/living）注入对应前缀。
DOMAIN_BEGIN_TOKENS 与 build_domain_prompt() 提供这一约束解码前缀的统一构造。
"""
from __future__ import annotations

from dataclasses import dataclass

EMPTY_THINK = "<think>\n\n</think>\n"

# domain 简写前缀（对齐 common/sid_utils.py::DOMAIN_TO_PREFIX 的取值域）
# -> 完整的 domain-begin token 文本。真实线上评测日志证实，懂物料/懂推荐生成阶段
# 由评测框架把对应 domain 的该 token 硬编码拼进 prompt 末尾（约束解码前缀），模型
# 只需 beam search 生成后续 3 个 token（<s_a_x><s_b_y><s_c_z>），详见模块顶部说明。
DOMAIN_BEGIN_TOKENS = {
    "video": "<|video_begin|>",
    "prod": "<|prod_begin|>",
    "ad": "<|ad_begin|>",
    "living": "<|living_begin|>",
}


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_k: int
    top_p: float
    max_tokens: int


# 官方推理参数表（docs/评测部分解析.md 第 511-516 行逐字对齐）
SAMPLING = {
    "material": SamplingConfig(temperature=0.7, top_k=20, top_p=0.8, max_tokens=128),
    "user": SamplingConfig(temperature=0.6, top_k=20, top_p=0.95, max_tokens=4096),
    # 懂推荐：no_think 与 think 两路共享同一组温度参数，仅 assistant 前缀不同
    "recommend": SamplingConfig(temperature=0.6, top_k=50, top_p=0.95, max_tokens=4096),
    "world": SamplingConfig(temperature=0.7, top_k=20, top_p=0.8, max_tokens=128),
}


def build_prompt(system: str, user: str, *, mode: str) -> str:
    """构造 qwen3 chat 格式 prompt。

    mode:
        "no_think" — 用户尾部追加 /no_think（若未带），assistant 起始追加空 think 前缀直出。
        "think"    — 用户尾部追加 /think（若未带），assistant 起始为空，交给模型自行采样 thinking。
    """
    u = user.rstrip()
    if mode == "no_think":
        if not u.endswith("/no_think"):
            u = u + "/no_think"
    elif mode == "think":
        if not u.endswith("/think"):
            u = u + "/think"
    else:
        raise ValueError(f"unknown mode: {mode}")

    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n")
    if mode == "no_think":
        parts.append(EMPTY_THINK)
    return "".join(parts)


def build_domain_prompt(system, user, *, domain_prefix, mode, thinking_text=None):
    """在 build_prompt 基础上，于 assistant 生成起点强制注入 domain-begin token。

    对齐线上评测「Single-stage generation with prompt_token (<|xxx_begin|>)」的
    约束解码协议：assistant 段不再留给模型自由生成 domain 前缀，而是由评测框架
    直接把对应 domain 的 begin-token 拼进 prompt 末尾，模型只需继续生成 3 个
    SemanticID token（s_a/s_b/s_c）。

    Args:
        domain_prefix: DOMAIN_BEGIN_TOKENS 的 key，取值 "video"/"prod"/"ad"/"living"。
        mode:
            "no_think" -- 懂物料 / 懂推荐 no_think 路：空 think 段 + domain 前缀。
            "think"    -- 懂推荐 think 路 Stage 2：需同时传入 thinking_text
                         （Stage 1 采样得到的完整思考文本），拼接为
                         <think>...</think> + domain 前缀。
        thinking_text: 仅 mode="think" 时使用，Stage 1 生成的思考文本
            （若不含起止 <think>/</think> 标签会自动补齐）。

    Returns:
        完整 prompt 字符串，assistant 段末尾以 domain-begin token 结束，
        引擎只需在其后继续生成 3 个 token。
    """
    if domain_prefix not in DOMAIN_BEGIN_TOKENS:
        raise ValueError("unknown domain_prefix: %r" % (domain_prefix,))
    begin_token = DOMAIN_BEGIN_TOKENS[domain_prefix]

    if mode == "no_think":
        base = build_prompt(system, user, mode="no_think")
        return base + begin_token
    elif mode == "think":
        if thinking_text is None:
            raise ValueError("mode=think 时必须提供 thinking_text（Stage 1 采样结果）")
        base = build_prompt(system, user, mode="think")
        t = thinking_text.strip()
        if not t.startswith("<think>"):
            t = "<think>\n" + t
        if "</think>" not in t:
            t = t + "\n</think>\n"
        return base + t + begin_token
    else:
        raise ValueError("unknown mode: %s" % (mode,))


# ---------------------------------------------------------------------------
# 懂世界：官方 system/user 模板（docs/评测部分解析.md 第 480-493 行逐字对齐）
# ---------------------------------------------------------------------------
WORLD_SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
WORLD_USER_TEMPLATE = (
    "请回答以下问题：\n{question}\n"
    "A.{A}\nB.{B}\nC.{C}\nD.{D}\n"
    '请按以下格式作答："正确答案是(在此处填写选项字母)"'
)


def build_world_prompt(question: str, A: str, B: str, C: str, D: str) -> tuple[str, str]:
    """返回 (system, user) 二元组，user 尚未拼接 /no_think（由 build_prompt 处理）。"""
    user = WORLD_USER_TEMPLATE.format(question=question, A=A, B=B, C=C, D=D)
    return WORLD_SYSTEM, user
