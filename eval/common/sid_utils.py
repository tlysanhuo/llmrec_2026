#!/usr/bin/env python3
"""common/sid_utils.py — SemanticID <-> Item ID 映射工具。

背景（对齐 eval/SPEC.md 附录"忠实度对比评估"发现）：`data/loaders.py` 此前误判
`demo/baseline-data/baseline_data/sampled/` 下的官方数据与 `data/OneReason_Pid2Sid/`
反查表"不可达"，转而用哈希代理值伪造 item_id。实测确认这批文件在当前环境完全可读，
本模块提供正确的 SemanticID 解析与 Pid2Sid 反查表构建/缓存，供 `data/loaders.py`
和 `metrics/material.py`、`metrics/recommend.py` 使用真实 item_id 空间。

- `SID_PATTERN` 从文本中解析 `<|xxx_begin|><s_a_i><s_b_j><s_c_k>` 形式的 SemanticID token。
- Pid2Sid 反查表：`(prefix, a, b, c) -> list[pid]`，从 `data/OneReason_Pid2Sid/part-*.parquet`
  构建，首次构建后 pickle 缓存到磁盘，之后直接加载缓存（约 2177 万条记录，构建耗时约
  2 分钟，已在当前环境实测验证）。这批 parquet 数据来自 HuggingFace 上公开的 17GB 原始
  物料表（从全量物料池中独立采样得到），与 `demo/baseline-data/.../sampled/` 下的 SFT
  样本（另外单独采样构造）是两批彼此独立的采样结果，因此两者之间的 SID 覆盖不完全重合
  是预期内的正常现象（详见 `sid_tokens_to_item_ids` 函数说明与 `eval/SPEC.md` 附录）。
"""
from __future__ import annotations

import glob
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Tuple

SID_PATTERN = re.compile(r"<\|(video|prod|living|ad)_begin\|>((?:<s_[abc]_\d+>){3})")
_SID_PART_PATTERN = re.compile(r"<s_[abc]_(\d+)>")

# domain（parquet 中的取值） -> token 前缀（SID_PATTERN 捕获组1 的取值）
DOMAIN_TO_PREFIX = {
    "video/video": "video",
    "video/ad": "ad",
    "goods": "prod",
    "live": "living",
}

PidSidKey = Tuple[str, int, int, int]


def parse_sid_tokens(text: str | None) -> List[PidSidKey]:
    """从文本中解析出全部 SemanticID token，返回 (prefix, a, b, c) 列表（保留出现顺序，允许重复）。"""
    if not text:
        return []
    results: List[PidSidKey] = []
    for prefix, triple in SID_PATTERN.findall(text):
        parts = _SID_PART_PATTERN.findall(triple)
        if len(parts) != 3:
            continue
        a, b, c = (int(p) for p in parts)
        results.append((prefix, a, b, c))
    return results


def build_pid2sid_index(pid2sid_dir: str) -> Dict[PidSidKey, List[int]]:
    """遍历 pid2sid_dir 下全部 part-*.parquet，构建 (prefix,a,b,c) -> [pid,...] 反查表。"""
    import pandas as pd

    files = sorted(glob.glob(os.path.join(pid2sid_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到任何 parquet 分片: {pid2sid_dir}/part-*.parquet")

    index: Dict[PidSidKey, List[int]] = defaultdict(list)
    for fp in files:
        df = pd.read_parquet(fp, columns=["pid", "domain", "sid_three"])
        prefixes = df["domain"].map(DOMAIN_TO_PREFIX)
        pids = df["pid"].tolist()
        sid_threes = df["sid_three"].tolist()
        prefix_list = prefixes.tolist()
        for pid, prefix, sid_three in zip(pids, prefix_list, sid_threes):
            if prefix is None:
                continue
            a, b, c = (int(x) for x in sid_three)
            key = (prefix, a, b, c)
            index[key].append(int(pid))
    return dict(index)


def load_pid2sid_index(
    cache_path: str, pid2sid_dir: str, force_rebuild: bool = False
) -> Dict[PidSidKey, List[int]]:
    """加载（或构建并缓存）Pid2Sid 反查表。"""
    if not force_rebuild and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    index = build_pid2sid_index(pid2sid_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def sid_tokens_to_item_ids(
    tokens: List[PidSidKey], index: Dict[PidSidKey, List[int]]
) -> set:
    """把一组 (prefix,a,b,c) token 映射为 item_id（pid）集合；映射不到的 token 直接跳过。

    注：实测该反查表相对 sampled 数据存在约 28.6% 的映射未命中（详见 SPEC.md 附录）。
    这是符合预期的正常现象，非缺陷：本反查表数据来自 HuggingFace 上公开的 17GB 原始
    物料表，是从全量物料池中一次独立采样得到的快照；而 sampled 数据（懂物料/懂推荐）
    是从同一物料池另外单独采样构造的，两次采样互不保证子集关系，因此部分 sampled
    样本的 SID 天然查不到对应 item_id，不影响以 HuggingFace 原始数据构造训练集本身。
    映射不到时不抛异常，调用方需自行处理"全部候选映射失败"的样本（视为该样本永远
    无法命中，不应计入分母之外的特殊处理）。
    """
    item_ids = set()
    for key in tokens:
        pids = index.get(key)
        if pids:
            item_ids.update(pids)
    return item_ids


def domain_from_prefix(prefix: str) -> str:
    """token 前缀反推 domain 名（用于统计/分域展示），与 DOMAIN_TO_PREFIX 互为逆映射。"""
    for domain, p in DOMAIN_TO_PREFIX.items():
        if p == prefix:
            return domain
    return "unknown"
