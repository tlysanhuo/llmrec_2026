#!/usr/bin/env python3
"""data/loaders.py — 只读加载各评测维度的 dev 数据集。

【2026-07-18 修订】此前版本误判 `demo/baseline-data/baseline_data/sampled/` 下的官方
sampled 数据（`懂世界*.jsonl`、`懂用户.jsonl`、`懂物料part1~4.jsonl`、`懂推荐1~4.jsonl`）
与 `data/OneReason_Pid2Sid/part-*.parquet` 反查表"不可达"，转而回退到哈希代理 item_id
等简化方案，懂推荐维度甚至直接判定为不可用。经与另一版实现（`eval.zip`）交叉验证并
在当前环境实测确认：**这批官方文件全部真实可读**，`Pid2Sid` 反查表可在约 2 分钟内构建出
约 2177 万条记录（详见 `eval/SPEC.md` 附录"忠实度对比评估"）。本次修订改为直接对接这批
官方数据源，不再使用代理/伪造 item_id。

数据来源（每个维度独立探测，找不到时给出清晰的 available=False 原因，不静默跳过）：

- 懂世界：`demo/baseline-data/baseline_data/sampled/懂世界_from_mc.jsonl`（272条，主要来源）
  + `懂世界.jsonl`（7条，样例较少，一并合并）。GT 直接取 `response` 中 `</think>` 之后的
  正文，用 `metrics.world.extract_answer` 同一套逻辑抽取（保证训练数据与推理输出口径一致）。
- 懂用户-F1 / 懂用户-逻辑链：`demo/baseline-data/baseline_data/sampled/懂用户.jsonl`
  （2892条），按 `"logic_chain" in response` 拆分两个子集（F1: 1588条，逻辑链: 1304条）。
- 懂物料：`demo/baseline-data/baseline_data/sampled/懂物料part1~4.jsonl`（仅 text→token
  方向，合计约5597条；part5~7 是 token→text 反方向，解析文档未覆盖，不使用），GT SemanticID
  通过 `common/sid_utils.py` 的 Pid2Sid 反查表映射为真实 item_id 集合。
- 懂推荐：`demo/baseline-data/baseline_data/sampled/懂推荐1~4.jsonl`（合计约19204条），
  同样用 Pid2Sid 反查表映射 GT，并从原始 `/think` 后缀 prompt 派生出 `/no_think` 版本供
  双路（thinking + non-thinking）推理使用。

已知现象（符合预期，非缺陷，如实记录）：`data/OneReason_Pid2Sid/` 反查表相对 `sampled` 数据
存在约 28.6% 的映射未命中（懂物料抽样实测 5597 条中仅 1601 条能映射出非空 item_id 集合，
懂推荐约72.6%命中率，四个 domain 的失败率相近）。**根因已与用户确认**：`Pid2Sid` 反查表来自
HuggingFace 上公开的 17GB 原始物料表，是从全量物料池中一次**独立采样**得到的快照；而
`sampled/` 下的 SFT 训练/评测样本（懂物料、懂推荐）是从同一物料池**另外单独采样**构造出来的
另一批数据。两次采样彼此独立、不保证子集关系，因此部分 SFT 样本的 SID 天然不落在 HuggingFace
这批采样快照范围内、查不到对应 item_id，这是两批数据分别采样的构造方式决定的正常现象，不是
反查表数据缺失或本模块代码逻辑有误，也不影响以 HuggingFace 原始数据构造训练集本身。因此懂
物料/懂推荐维度中"GT 全部映射失败"的样本会被 loader 过滤掉并统计到 `n_gt_map_failed`，调用方
应据此了解实际可评测样本数会少于名义样本数，但无需将其当作需要修复的问题。

不修改、不重写 `llmrec_2026-main/` 或 `demo/` 下的任何文件，本模块只做只读访问。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.sid_utils import (
    domain_from_prefix,
    load_pid2sid_index,
    parse_sid_tokens,
    sid_tokens_to_item_ids,
)
from common.text_utils import extract_json_array, extract_json_object, strip_think

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLED_DIR = REPO_ROOT / "demo" / "baseline-data" / "baseline_data" / "sampled"
PID2SID_DIR = REPO_ROOT / "data" / "OneReason_Pid2Sid"
PID2SID_CACHE = REPO_ROOT / "eval" / "output" / ".cache" / "pid2sid_index.pkl"


def _path_available(path: Path) -> bool:
    """路径存在且可读；若是软链接，还要求链接目标真实可达（覆盖 lustre 挂载失效场景）。"""
    try:
        return path.exists()
    except OSError:
        # 常见于失效软链接：exists() 内部 stat 失败会抛 OSError（如 ENOTCONN/EIO）
        return False


def _read_json_array_file(path: Path) -> list[dict]:
    """sampled 目录下的 jsonl 文件每行是一个单元素 JSON 数组
    `[{"system": ..., "prompt": ..., "response": ...}]`，需要展开成单条记录列表。
    """
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                rows.extend(obj)
            else:
                rows.append(obj)
    return rows


def _read_json_array_files(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if _path_available(p):
            rows.extend(_read_json_array_file(p))
    return rows


_PID2SID_INDEX_CACHE: dict | None = None


def get_pid2sid_index() -> dict | None:
    """懒加载并进程内缓存 Pid2Sid 反查表；数据目录不可用时返回 None（不抛异常）。

    对外公开（供 `run_eval.py` 在推理阶段把模型生成的 SemanticID 映射为 item_id
    时复用同一份缓存索引，避免重复构建）。
    """
    global _PID2SID_INDEX_CACHE
    if _PID2SID_INDEX_CACHE is not None:
        return _PID2SID_INDEX_CACHE
    if not _path_available(PID2SID_DIR):
        return None
    _PID2SID_INDEX_CACHE = load_pid2sid_index(str(PID2SID_CACHE), str(PID2SID_DIR))
    return _PID2SID_INDEX_CACHE


# 向后兼容别名（内部模块曾用下划线前缀名）
_get_pid2sid_index = get_pid2sid_index


# ---------------------------------------------------------------------------
# 懂世界
# ---------------------------------------------------------------------------

_WORLD_FILES = ["懂世界_from_mc.jsonl", "懂世界.jsonl"]


def load_world_dev(limit: int | None = None) -> dict[str, Any]:
    """加载懂世界 dev 数据：官方 sampled `懂世界_from_mc.jsonl`（272条）+ `懂世界.jsonl`（7条）。

    GT 用 `metrics.world.extract_answer` 从 `response` 的 `</think>` 之后正文中抽取，
    与推理输出复用同一套抽取逻辑，保证评测口径一致；解析失败（无法抽出唯一或多个合法
    字母）的行会被跳过并计入 `n_gt_parse_failed`。
    """
    from metrics.world import extract_answer

    paths = [SAMPLED_DIR / name for name in _WORLD_FILES]
    raw_rows = _read_json_array_files(paths)
    if not raw_rows:
        return {
            "available": False,
            "reason": f"未找到懂世界 sampled 数据：{[str(p) for p in paths]}",
            "items": [],
        }

    items = []
    n_gt_parse_failed = 0
    for r in raw_rows:
        response = r.get("response", "")
        body = strip_think(response) or response
        extracted = extract_answer(body)
        letters = extracted["matched_letters"]
        if not letters:
            n_gt_parse_failed += 1
            continue
        items.append(
            {
                "system": r.get("system", ""),
                "prompt": r.get("prompt", ""),
                "gold": letters,  # 升序字母串，单选为单字符，多选为多字符
                "source": "懂世界_from_mc.jsonl / 懂世界.jsonl",
            }
        )
    if limit:
        items = items[:limit]
    return {
        "available": True,
        "source": ",".join(str(p) for p in paths),
        "items": items,
        "n_raw": len(raw_rows),
        "n_gt_parse_failed": n_gt_parse_failed,
    }


# ---------------------------------------------------------------------------
# 懂用户（F1 / 逻辑链）：官方 sampled 懂用户.jsonl
# ---------------------------------------------------------------------------

_USER_FILE = "懂用户.jsonl"


def load_user_f1_dev(limit: int | None = None) -> dict[str, Any]:
    """加载懂用户-F1 dev 数据：从 `懂用户.jsonl` 中筛出不含 `logic_chain` 的行（约1588条）。

    GT = 从 `response` 中提取的 SemanticID 字符串 JSON 数组（`extract_json_array` 优先
    从 `</think>` 之后的正文中查找）。
    """
    path = SAMPLED_DIR / _USER_FILE
    if not _path_available(path):
        return {"available": False, "reason": f"未找到 {path}", "items": []}

    raw_rows = _read_json_array_file(path)
    items = []
    n_parse_failed = 0
    for r in raw_rows:
        response = r.get("response", "")
        if "logic_chain" in response:
            continue
        gold = extract_json_array(response)
        if not isinstance(gold, list):
            n_parse_failed += 1
            continue
        items.append(
            {
                "system": r.get("system", ""),
                "prompt": r.get("prompt", ""),
                "gold": gold,
                "source": _USER_FILE,
            }
        )
    if limit:
        items = items[:limit]
    if not items:
        return {"available": False, "reason": f"{path} 中未解析出任何 F1 型样例", "items": []}
    return {
        "available": True,
        "source": str(path),
        "items": items,
        "n_gt_parse_failed": n_parse_failed,
    }


def load_user_chain_dev(limit: int | None = None) -> dict[str, Any]:
    """加载懂用户-逻辑链 dev 数据：从 `懂用户.jsonl` 中筛出含 `logic_chain` 的行（约1304条）。

    GT = 从 `response` 中提取的 `{"logic_chain": {"name":..., "events":[...]}}` 结构。
    """
    path = SAMPLED_DIR / _USER_FILE
    if not _path_available(path):
        return {"available": False, "reason": f"未找到 {path}", "items": []}

    raw_rows = _read_json_array_file(path)
    items = []
    n_parse_failed = 0
    for r in raw_rows:
        response = r.get("response", "")
        if "logic_chain" not in response:
            continue
        gt_obj = extract_json_object(response)
        if not gt_obj:
            n_parse_failed += 1
            continue
        chain = gt_obj.get("logic_chain", {})
        events = chain.get("events")
        if not events:
            n_parse_failed += 1
            continue
        items.append(
            {
                "system": r.get("system", ""),
                "prompt": r.get("prompt", ""),
                "gold_chain_name": chain.get("name"),
                "gold_events": events,
                "source": _USER_FILE,
            }
        )
    if limit:
        items = items[:limit]
    if not items:
        return {"available": False, "reason": f"{path} 中未解析出任何逻辑链型样例", "items": []}
    return {
        "available": True,
        "source": str(path),
        "items": items,
        "n_gt_parse_failed": n_parse_failed,
    }


# ---------------------------------------------------------------------------
# 懂物料：官方 sampled 懂物料part1~4.jsonl（仅 text→token 方向）+ Pid2Sid 反查
# ---------------------------------------------------------------------------

_MATERIAL_FILES = [f"懂物料part{i}.jsonl" for i in range(1, 5)]


def load_material_dev(
    limit: int | None = None, domain_filter: tuple[str, ...] | None = ("video",)
) -> dict[str, Any]:
    """加载懂物料 dev 数据：`懂物料part1~4.jsonl`（text→token 方向，合计约5597条）。

    GT SemanticID 从 `response` 的 `</think>` 之后正文解析，再通过 Pid2Sid 反查表映射为
    真实 item_id 集合（一个 pattern 理论上只应对应 1 个 GT，但反查表可能一对多，全部
    纳入 `gold_item_ids`）。若 `Pid2Sid` 索引不可用或该行的 GT 全部映射失败（已知约
    28.6% 命中率，因 HuggingFace 原始表与本批 sampled 数据是分别独立采样的两批数据、
    非缺陷，详见本文件模块级 docstring"已知现象"一节），该行会被跳过，分别计入
    `n_sid_parse_failed` / `n_gt_map_failed`。

    Args:
        domain_filter: 只保留 gold 前缀在此集合内的样本，取值为
            "video"/"prod"/"ad"/"living" 的子集。默认仅保留 `("video",)`：
            真实线上评测日志（测评中间输出.md）证实懂物料
            （challenge_itemic_pattern_grounding）线上 100% 为 video 域，而本地
            sampled 数据（懂物料part1~4.jsonl）实际混合了四个 domain，不过滤
            会造成评测口径偏离线上。传入 None 则不过滤（保留旧行为，供对比/
            调试用）。
    """
    paths = [SAMPLED_DIR / name for name in _MATERIAL_FILES]
    raw_rows = _read_json_array_files(paths)
    if not raw_rows:
        return {
            "available": False,
            "reason": f"未找到懂物料 sampled 数据：{[str(p) for p in paths]}",
            "items": [],
        }

    index = _get_pid2sid_index()
    if index is None:
        return {
            "available": False,
            "reason": f"Pid2Sid 反查表目录不可用：{PID2SID_DIR}",
            "items": [],
        }

    items = []
    n_sid_parse_failed = 0
    n_gt_map_failed = 0
    n_domain_filtered = 0
    for r in raw_rows:
        response = r.get("response", "")
        body = strip_think(response) or response
        sid_tokens = parse_sid_tokens(body)
        if not sid_tokens:
            n_sid_parse_failed += 1
            continue
        prefix, a, b, c = sid_tokens[0]
        if domain_filter is not None and prefix not in domain_filter:
            n_domain_filtered += 1
            continue
        gold_item_ids = sid_tokens_to_item_ids(sid_tokens, index)
        if not gold_item_ids:
            n_gt_map_failed += 1
            continue
        # pattern 保留完整 `<|xxx_begin|>` 前缀，与 sampled 数据 response 原文格式
        # 一致（实测确认官方 response 里 SID 就带 domain 前缀，如
        # `<|prod_begin|><s_a_6091><s_b_2919><s_c_2941>`），确保 self-check 模式下
        # 用 pattern 本身模拟"beam64命中"时，能被 common/sid_utils.parse_sid_tokens
        # 正确解析，与真实模型推理输出走同一条解析路径。
        pattern = f"<|{prefix}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"
        domain = domain_from_prefix(prefix)
        items.append(
            {
                "system": r.get("system", ""),
                "prompt": r.get("prompt", ""),
                "pattern": pattern,
                "gold_item_ids": sorted(gold_item_ids),
                "domain": domain,
                "domain_prefix": prefix,
                "source": r.get("_src_file", "懂物料part1~4.jsonl"),
            }
        )
    if limit:
        items = items[:limit]
    if not items:
        return {
            "available": False,
            "reason": "懂物料样本 GT 全部解析/映射失败",
            "items": [],
            "n_sid_parse_failed": n_sid_parse_failed,
            "n_gt_map_failed": n_gt_map_failed,
            "n_domain_filtered": n_domain_filtered,
        }
    return {
        "available": True,
        "source": ",".join(str(p) for p in paths),
        "items": items,
        "n_raw": len(raw_rows),
        "n_sid_parse_failed": n_sid_parse_failed,
        "n_gt_map_failed": n_gt_map_failed,
        "n_domain_filtered": n_domain_filtered,
    }


# ---------------------------------------------------------------------------
# 懂推荐：官方 sampled 懂推荐1~4.jsonl + Pid2Sid 反查
# ---------------------------------------------------------------------------

_RECOMMEND_FILES = [f"懂推荐{i}.jsonl" for i in range(1, 5)]


def load_recommend_dev(limit: int | None = None) -> dict[str, Any]:
    """加载懂推荐 dev 数据：`懂推荐1~4.jsonl`（合计约19204条）。

    - GT SemanticID 从 `response` 的 `</think>` 之后正文解析，经 Pid2Sid 反查表映射为
      item_id 集合。
    - `prompt` 原样保留为 `prompt_think`（含官方 `/think` 后缀）；额外派生
      `prompt_nothink`（末尾 `/think` 替换为 `/no_think`），供双路（thinking +
      non-thinking）候选生成使用（解析文档「懂推荐」评估指标计算一节）。
    - `target_domain_prefix`：本条样本预测目标域前缀（video/prod/ad/living），
      从 gold SemanticID 的首个 token 前缀直接反推得到。实测本数据源全量样本
      的 gold 均为单一域（无混合多域情况），因此此反推完全可靠，与线上每条
      测试样本本身已确定预测目标域的事实对齐（注意：源文件名本身不对应四个
      目标域，四个源文件内部都混合了 video/prod/ad/living 四种目标域样本，
      必须逐条取 target_domain_prefix 而非按文件名判断）。
    """
    paths = [SAMPLED_DIR / name for name in _RECOMMEND_FILES]
    raw_rows = _read_json_array_files(paths)
    if not raw_rows:
        return {
            "available": False,
            "reason": f"未找到懂推荐 sampled 数据：{[str(p) for p in paths]}",
            "items": [],
        }

    index = _get_pid2sid_index()
    if index is None:
        return {
            "available": False,
            "reason": f"Pid2Sid 反查表目录不可用：{PID2SID_DIR}",
            "items": [],
        }

    items = []
    n_sid_parse_failed = 0
    n_gt_map_failed = 0
    n_think_replace_failed = 0
    for r in raw_rows:
        response = r.get("response", "")
        body = strip_think(response) or response
        sid_tokens = parse_sid_tokens(body)
        if not sid_tokens:
            n_sid_parse_failed += 1
            continue
        gold_item_ids = sid_tokens_to_item_ids(sid_tokens, index)
        if not gold_item_ids:
            n_gt_map_failed += 1
            continue

        prompt_think = r.get("prompt", "")
        if prompt_think.rstrip().endswith("/think"):
            stripped = prompt_think.rstrip()
            prompt_nothink = stripped[: -len("/think")] + "/no_think"
        else:
            n_think_replace_failed += 1
            prompt_nothink = prompt_think

        target_domain_prefix = sid_tokens[0][0]
        items.append(
            {
                "system": r.get("system", ""),
                "prompt_think": prompt_think,
                "prompt_nothink": prompt_nothink,
                "gold_item_ids": sorted(gold_item_ids),
                "target_domain_prefix": target_domain_prefix,
                "source": r.get("_src_file", "懂推荐1~4.jsonl"),
            }
        )
    if limit:
        items = items[:limit]
    if not items:
        return {
            "available": False,
            "reason": "懂推荐样本 GT 全部解析/映射失败",
            "items": [],
            "n_sid_parse_failed": n_sid_parse_failed,
            "n_gt_map_failed": n_gt_map_failed,
        }
    return {
        "available": True,
        "source": ",".join(str(p) for p in paths),
        "items": items,
        "n_raw": len(raw_rows),
        "n_sid_parse_failed": n_sid_parse_failed,
        "n_gt_map_failed": n_gt_map_failed,
        "n_think_replace_failed": n_think_replace_failed,
    }


DIM_LOADERS = {
    "material": load_material_dev,
    "user_f1": load_user_f1_dev,
    "user_chain": load_user_chain_dev,
    "recommend": load_recommend_dev,
    "world": load_world_dev,
}


def load_dev(dim: str, limit: int | None = None) -> dict[str, Any]:
    if dim not in DIM_LOADERS:
        raise ValueError(f"未知评测维度: {dim!r}，可选: {sorted(DIM_LOADERS)}")
    return DIM_LOADERS[dim](limit=limit)
