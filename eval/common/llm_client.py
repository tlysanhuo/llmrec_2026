#!/usr/bin/env python3
"""common/llm_client.py — 多渠道并行大模型调用客户端。

完整迁移自 docs/llm_api_guide.md 描述的方案，替代旧脚本（label_action_v4.py 等）
使用的单渠道云雾（yunwu）API 依赖。本模块只提供"调用层"通用能力：渠道池、限速、
并发编排、重试；不包含任何具体业务的 prompt/数据构造逻辑。

用法：
    from common.llm_client import call_llm, call_llm_batch

    text = call_llm([{"role": "user", "content": "你好"}])
    results = call_llm_batch([msgs1, msgs2, ...])  # 并发编排，返回与输入等长的结果列表
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. 概述：网关地址与鉴权
# ---------------------------------------------------------------------------
API_URL = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
# 注：APP_ID 是文档 docs/llm_api_guide.md 中直接给出的约定值（非需要从密钥管理系统
# 加载的敏感凭据）；如未来切换为需要保密加载的凭据，请改为从环境变量/密钥文件读取。
APP_ID = "22053007895171985484"

# ---------------------------------------------------------------------------
# 2. 可用渠道池（7 个渠道，均经过验证：单请求可通 + 3并发 QPM=20 压测零失败）
# ---------------------------------------------------------------------------
CHANNEL_MODELS = [
    "deepseek-v4-flash-tencent",
    "deepseek-v4-flash-meituan",
    "deepseek-v4-flash-baidu",
    "deepseek-v4-pro-tencent",
    "deepseek-v3.2-tencent",
    "deepseek-v3.2-doubao",
    "deepseek-v3.2-huawei",
]

# 已验证并剔除的模型（不要加回来）：
#   deepseek-v4-flash-huawei — 3并发/QPM=20压测下48s内15次失败仅1次成功，几乎全429
#   deepseek-v4-flash（无厂商后缀）— 402 Payment Required（该app-id未开通）
#   deepseek-v4-pro（无厂商后缀）— 402 Payment Required（同上）

QPM_PER_CHANNEL = 20
N_WORKERS_PER_CHANNEL = 3


# ---------------------------------------------------------------------------
# 3.1 限速器：令牌桶
# ---------------------------------------------------------------------------
class TokenBucket:
    """令牌桶限速器：保证发出速率不超过 QPM，同时允许多个请求并发在飞。

    与 Semaphore(1) 的区别：
      Semaphore(1) = 同一时刻只有1个请求（严格串行，浪费 API 响应等待时间）
      TokenBucket  = 发出速率 ≤ QPM，但 N 个请求可以同时在网络中等待响应
    """

    def __init__(self, qpm: int):
        self._min_interval = 60.0 / qpm
        self._lock = threading.Lock()
        self._last_sent = 0.0

    def acquire(self) -> None:
        """阻塞直到可以发出下一个请求。"""
        while True:
            with self._lock:
                now = time.time()
                wait = self._min_interval - (now - self._last_sent)
                if wait <= 0:
                    self._last_sent = now
                    return
            time.sleep(max(0.05, wait))


# ---------------------------------------------------------------------------
# 3.2 渠道封装
# ---------------------------------------------------------------------------
class Channel:
    __slots__ = ("model", "limiter")

    def __init__(self, model: str, qpm: int):
        self.model = model
        self.limiter = TokenBucket(qpm)


CHANNELS = [Channel(m, QPM_PER_CHANNEL) for m in CHANNEL_MODELS]


# ---------------------------------------------------------------------------
# 3.3 无状态渠道分配
# ---------------------------------------------------------------------------
def pick_channel(*keys: Any) -> Channel:
    """无状态地把一个任务/批次分配到某个渠道。

    keys 建议传入能唯一标识该任务的信息（如 shard 序号 + batch 起始位置），
    保证同一批次总是分配到同一个渠道，便于排查问题。
    """
    idx = hash(keys) % len(CHANNELS)
    return CHANNELS[idx]


# ---------------------------------------------------------------------------
# 3.4 单次 HTTP 调用（含重试）
# ---------------------------------------------------------------------------
def _load_app_id() -> str:
    """按内部约定加载 app-id。当前直接返回文档给出的约定值。"""
    return APP_ID


def call_api(
    messages: list[dict[str, str]],
    channel: Channel,
    retries: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0,
) -> str:
    """单次（含重试）调用指定渠道。重试也必须经过限速器，不能只在首次请求时限速。"""
    payload = {
        "model": channel.model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {_load_app_id()}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        channel.limiter.acquire()
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("empty content")
            return content
        except Exception as e:  # noqa: BLE001 - 需要捕获所有异常以便重试
            last_err = e
            log.warning("[%s] HTTP retry %d/%d: %s", channel.model, attempt + 1, retries, e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"[{channel.model}] API failed after all HTTP retries: {last_err}")


# ---------------------------------------------------------------------------
# 3.5 单次调用便捷入口（不关心具体走哪个渠道时使用）
# ---------------------------------------------------------------------------
def call_llm(
    messages: list[dict[str, str]],
    *,
    task_key: Any = None,
    retries: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0,
) -> str:
    """单条调用：按 task_key（默认用 messages 内容哈希）无状态选择渠道。"""
    channel = pick_channel(task_key if task_key is not None else str(messages))
    return call_api(messages, channel, retries=retries, max_tokens=max_tokens, temperature=temperature)


# ---------------------------------------------------------------------------
# 并发编排：批量调用
# ---------------------------------------------------------------------------
def call_llm_batch(
    tasks: list[list[dict[str, str]]],
    *,
    retries: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0,
    workers: int | None = None,
) -> list[str | None]:
    """并发编排批量调用。总并发数 = 每渠道并发数 × 渠道数。

    单个任务失败（重试耗尽后）不会中断整批，对应位置返回 None，并记录日志，
    由上层调用方决定降级策略（如重跑该子集）。
    """
    if workers is None:
        workers = N_WORKERS_PER_CHANNEL * len(CHANNELS)
    results: list[str | None] = [None] * len(tasks)

    def _process(task_id: int, messages: list[dict[str, str]]) -> str:
        channel = pick_channel(task_id)
        return call_api(messages, channel, retries=retries, max_tokens=max_tokens, temperature=temperature)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, i, msgs): i for i, msgs in enumerate(tasks)}
        for fut in as_completed(futures):
            task_id = futures[fut]
            try:
                results[task_id] = fut.result()
            except Exception as e:  # noqa: BLE001
                log.error("Task %d failed after all retries: %s", task_id, e)
                results[task_id] = None
    return results


def estimate_minutes(n_requests: int, batch_size: int = 1) -> float:
    """预计分钟数 = 预计样本总数 / (总QPM × batch_size)。"""
    total_qpm = QPM_PER_CHANNEL * len(CHANNELS)
    return n_requests / (total_qpm * batch_size)
