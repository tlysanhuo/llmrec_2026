#!/usr/bin/env python3
"""在 competition_smoke.jsonl 上运行跨维度基础模型冒烟评测。

推理执行有三种模式（--engine-mode），从旧到新依次为：

- legacy：逐条样本串行调用引擎（最初实现），每次只喂 1 条 prompt 给 vLLM。
  最稳妥但最慢，GPU/CPU 利用率最低。作为其余两种模式失效时的回退路径保留。
- batch（默认）：按维度分组，同一维度的全部样本一次性打包成一个 batch 调用
  引擎，让 vLLM 的 continuous batching 生效。比 legacy 快，实现和 legacy 等价
  （相同 prompt/采样参数/评分逻辑），只是调用粒度从逐条变成整维度。
- multiprocess：在 batch 基础上，当 material 样本数超过阈值时，改用多个独立
  子进程（material_worker.py）各自建立 vLLM 实例、共享同一张 GPU 并行跑
  material 分片，绕开 vLLM 官方 beam_search 在 Python 层单进程单线程（GIL）
  做逐 step 候选构造/排序的瓶颈。其余维度仍复用 batch 模式的单进程实现。
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.engine import VLLMEngine
from common.prompt import SAMPLING, build_domain_prompt, build_prompt
from common.sid_utils import parse_sid_tokens
from common.text_utils import extract_json_array, extract_json_object
from metrics.user_chain import Event, score_chain
from metrics.user_f1 import f1_score
from metrics.world import extract_answer, is_correct


def _domain_prefix_of(gold_pattern: str) -> str:
    """从 gold pattern（完整带 `<|xxx_begin|>` 前缀的 SemanticID 文本）中取出目标
    domain 前缀（video/prod/ad/living）。competition_smoke.jsonl 中每条
    material/recommend 样本的 gold 均为单一域的完整 pattern。
    """
    tokens = parse_sid_tokens(gold_pattern)
    if not tokens:
        raise ValueError(f"无法从 gold 中解析出 domain 前缀: {gold_pattern!r}")
    return tokens[0][0]


# multiprocess 模式下，material 样本数超过该阈值才切分多进程；阈值以下多进程启动
# 开销（每个子进程都要重新加载一次模型）划不来，退化为单进程 batch 调用。
MATERIAL_MULTIPROCESS_THRESHOLD = 30
# multiprocess 模式默认并行的 worker 进程数（共享同一张 GPU，各自限制显存占比）。
DEFAULT_MATERIAL_WORKERS = 4
# 主进程轮询各 worker 存活状态 / 打印心跳日志的间隔（秒）。
WORKER_POLL_INTERVAL = 20


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][run_smoke_eval] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 子进程组清理：material_worker.py 内部会用 vLLM 的多进程后端再 fork 出
# EngineCore 子进程。之前用 subprocess.Popen(cmd) 默认把子进程放在主进程的同一
# 进程组，一旦主进程被 kill -9（没有机会执行清理代码），worker 及其 fork 出的
# EngineCore 会变成孤儿进程继续跑、继续占着 GPU 显存不释放。
#
# 修复：
# 1) 每个 worker 用 start_new_session=True 启动，成为独立的进程组组长；
# 2) 把 (pid, pgid) 登记到全局表，注册 atexit / SIGTERM / SIGINT 处理器，
#    无论主进程正常结束、异常退出还是被信号终止，都尝试用 os.killpg 把整个
#    worker 进程组（含其 fork 出的 EngineCore）一起杀掉。
# ---------------------------------------------------------------------------
_ACTIVE_WORKER_PGIDS: set[int] = set()


def _register_worker_pgid(pgid: int) -> None:
    _ACTIVE_WORKER_PGIDS.add(pgid)


def _unregister_worker_pgid(pgid: int) -> None:
    _ACTIVE_WORKER_PGIDS.discard(pgid)


def _kill_all_worker_pgids() -> None:
    pgids = list(_ACTIVE_WORKER_PGIDS)
    if not pgids:
        return
    # 先礼后兵：SIGTERM 给 vLLM 一点时间做自身清理（释放显存/关闭子进程），
    # 短暂等待后再 SIGKILL 兜底，避免遗留占用显存的孤儿 EngineCore 进程。
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    time.sleep(3)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        _ACTIVE_WORKER_PGIDS.discard(pgid)


def _signal_handler(signum, frame):
    log(f"收到信号 {signum}，清理所有子进程组后退出...")
    _kill_all_worker_pgids()
    sys.exit(128 + signum)


atexit.register(_kill_all_worker_pgids)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            records = value if isinstance(value, list) else [value]
            for record in records:
                record["_line_no"] = line_no
                rows.append(record)
    return rows


def normalize_user_gold(text: str) -> list[str]:
    parsed = extract_json_array(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_chain_gold(text: str) -> list[Event]:
    parsed = extract_json_object(text)
    if parsed:
        return [
            Event(action=event.get("action", ""), logic=event.get("logic", ""))
            for event in parsed.get("logic_chain", {}).get("events", [])
        ]

    # competition.md 的示例答案是“JSON 字符串中的 JSON”，但内部说明文本含未转义
    # 双引号，严格 json.loads 会失败；只对 event 的 action/logic 字段做定界提取。
    event_pattern = re.compile(
        r'\{"date":\s*"[^"]*",\s*"action":\s*"(.*?)",\s*"logic":\s*"(.*?)"\}',
        re.DOTALL,
    )
    return [Event(action=action, logic=logic) for action, logic in event_pattern.findall(text)]


# ---------------------------------------------------------------------------
# legacy：最初实现，逐条样本串行调用引擎。保留作为 batch/multiprocess 失效时的
# 回退路径，行为（prompt 构造、采样参数、评分逻辑）与 batch 模式完全一致。
# ---------------------------------------------------------------------------
def sample_one(engine: VLLMEngine, row: dict, dim: str):
    config_key = "user" if dim.startswith("user_") else dim
    config = SAMPLING[config_key]
    prompt = build_prompt(row.get("system", ""), row["prompt"], mode="no_think")
    return engine.sample(
        [prompt],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )[0].text


def evaluate_row(engine: VLLMEngine, row: dict) -> dict:
    dim = row["dimension"]
    gold = row["gold"]
    detail = {
        "sample_id": row["sample_id"],
        "dimension": dim,
        "task": row["task"],
        "source": row["source"],
    }

    if dim == "material":
        domain_prefix = _domain_prefix_of(gold)
        begin_token = f"<|{domain_prefix}_begin|>"
        prompt = build_domain_prompt(row.get("system", ""), row["prompt"], domain_prefix=domain_prefix, mode="no_think")
        suffixes = engine.beam_decode([prompt], beam_width=64, max_tokens=3)[0]
        predictions = [begin_token + s for s in suffixes]
        gold_tokens = set(parse_sid_tokens(gold))
        candidate_tokens = {token for text in predictions for token in parse_sid_tokens(text)}
        passed = bool(gold_tokens & candidate_tokens)
        detail.update(
            score=int(passed),
            metric="semantic_id_pass@64",
            gold=gold,
            prediction=predictions,
            n_candidates=len(predictions),
        )
        return detail

    if dim == "recommend":
        domain_prefix = _domain_prefix_of(gold)
        begin_token = f"<|{domain_prefix}_begin|>"
        config = SAMPLING["recommend"]

        # no-think 路：直接 beam search 3 token
        prompt_nothink = build_domain_prompt(
            row.get("system", ""), row["prompt"], domain_prefix=domain_prefix, mode="no_think"
        )
        nothink_suffixes = engine.beam_decode([prompt_nothink], beam_width=32, max_tokens=3)[0]
        non_thinking_texts = [begin_token + s for s in nothink_suffixes]

        # think 路：Stage 1 采样 thinking，Stage 2 在其后 beam search 3 token
        prompt_think_stage1 = build_prompt(row.get("system", ""), row["prompt"], mode="think")
        gen_thinking = engine.sample(
            [prompt_think_stage1],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            n=1,
        )
        thinking_text = gen_thinking[0].text
        prompt_think_stage2 = build_domain_prompt(
            row.get("system", ""),
            row["prompt"],
            domain_prefix=domain_prefix,
            mode="think",
            thinking_text=thinking_text,
        )
        think_suffixes = engine.beam_decode([prompt_think_stage2], beam_width=32, max_tokens=3)[0]
        thinking_texts = [begin_token + s for s in think_suffixes]

        outputs = thinking_texts + non_thinking_texts
        gold_tokens = set(parse_sid_tokens(gold))
        candidate_tokens = {token for text in outputs for token in parse_sid_tokens(text)}
        passed = bool(gold_tokens & candidate_tokens)
        detail.update(
            score=int(passed),
            metric="semantic_id_pass@64",
            gold=gold,
            prediction=outputs,
            n_candidates=len(outputs),
            n_unique_semantic_ids=len(candidate_tokens),
        )
        return detail

    output = sample_one(engine, row, dim)
    detail.update(gold=gold, prediction=output)
    if dim == "world":
        parsed = extract_answer(output)["matched_letters"]
        detail.update(score=int(is_correct(output, gold)), metric="accuracy", parsed_prediction=parsed)
    elif dim == "user_f1":
        pred = extract_json_array(output) or []
        result = f1_score([str(item) for item in pred], normalize_user_gold(gold))
        detail.update(score=result["f1"], metric="f1", parsed_prediction=pred, metric_detail=result)
    elif dim == "user_chain":
        payload = extract_json_object(output) or {}
        chain = payload.get("logic_chain", payload)
        pred_events = [
            Event(action=event.get("action", ""), logic=event.get("logic", ""))
            for event in chain.get("events", [])
        ]
        result = score_chain(normalize_chain_gold(gold), pred_events)
        detail.update(
            score=result.overall_score,
            metric="chain_overall",
            parsed_prediction=payload,
            metric_detail={
                "action_alignment": result.action_alignment,
                "logic_alignment": result.logic_alignment,
                "n_gold": result.n_gold,
                "n_pred": result.n_pred,
            },
        )
    else:
        raise ValueError(f"unsupported dimension: {dim}")
    return detail


def evaluate_all_legacy(engine: VLLMEngine, rows: list[dict]) -> list[dict]:
    """legacy 模式入口：逐条串行调用 evaluate_row，与最初实现完全一致。"""
    return [evaluate_row(engine, row) for row in rows]


# ---------------------------------------------------------------------------
# batch：按维度分组，整维度一次性打包调用引擎。
# ---------------------------------------------------------------------------
def base_detail(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"],
        "dimension": row["dimension"],
        "task": row["task"],
        "source": row["source"],
    }


def score_material_predictions(row: dict, predictions: list[str]) -> dict:
    gold = row["gold"]
    gold_tokens = set(parse_sid_tokens(gold))
    candidate_tokens = {token for text in predictions for token in parse_sid_tokens(text)}
    passed = bool(gold_tokens & candidate_tokens)
    detail = base_detail(row)
    detail.update(
        score=int(passed),
        metric="semantic_id_pass@64",
        gold=gold,
        prediction=predictions,
        n_candidates=len(predictions),
    )
    return detail


def eval_material_batch(engine: VLLMEngine, rows: list[dict]) -> list[dict]:
    """懂物料：对齐线上「Single-stage generation with prompt_token (<|xxx_begin|>)」
    协议——每条样本目标域的 begin token 强制拼进 prompt 末尾，一次性打包给
    beam_decode 只生成 3 个后续 token（s_a/s_b/s_c），而非自由生成完整 pattern。
    """
    log(f"[material] 开始，n={len(rows)}")
    t0 = time.monotonic()
    domain_prefixes = [_domain_prefix_of(row["gold"]) for row in rows]
    prompts = [
        build_domain_prompt(row.get("system", ""), row["prompt"], domain_prefix=dp, mode="no_think")
        for row, dp in zip(rows, domain_prefixes)
    ]
    all_suffixes = engine.beam_decode(prompts, beam_width=64, max_tokens=3)
    details = []
    for row, dp, suffixes in zip(rows, domain_prefixes, all_suffixes):
        begin_token = f"<|{dp}_begin|>"
        predictions = [begin_token + s for s in suffixes]
        details.append(score_material_predictions(row, predictions))
    log(f"[material] 完成，n={len(rows)}，耗时 {time.monotonic() - t0:.1f}s")
    return details


def eval_recommend_batch(engine: VLLMEngine, rows: list[dict]) -> list[dict]:
    """懂推荐：对齐线上四个 `challenge_recommendation_*` 子任务的双路约束解码协议。

    - no-think 路：每条样本目标域的 begin token 拼进 prompt 末尾，一次性打包做
      beam search 生成 3 个后续 token（beam_width=32）。
    - think 路：先把全部 prompt 打包做一次 sample（n=1，max_tokens=4096）得到
      Stage 1 thinking 文本，再把 thinking 文本 + 目标域 begin token 拼回 prompt
      末尾，二次打包做 beam search 生成 3 个后续 token（beam_width=32，Stage 2）。
    """
    log(f"[recommend] 开始，n={len(rows)}")
    t0 = time.monotonic()
    config = SAMPLING["recommend"]
    domain_prefixes = [_domain_prefix_of(row["gold"]) for row in rows]

    # --- no-think 路：直接 beam search 3 token ---
    prompts_nothink = [
        build_domain_prompt(row.get("system", ""), row["prompt"], domain_prefix=dp, mode="no_think")
        for row, dp in zip(rows, domain_prefixes)
    ]
    nothink_suffixes_flat = engine.beam_decode(prompts_nothink, beam_width=32, max_tokens=3)

    # --- think 路：Stage 1 批量采样 thinking ---
    prompts_think_stage1 = [
        build_prompt(row.get("system", ""), row["prompt"], mode="think") for row in rows
    ]
    gen_thinking_flat = engine.sample(
        prompts_think_stage1,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        n=1,
    )
    thinking_texts_stage1 = [g.text for g in gen_thinking_flat]

    # --- think 路：Stage 2 批量 beam search 3 token ---
    prompts_think_stage2 = [
        build_domain_prompt(
            row.get("system", ""),
            row["prompt"],
            domain_prefix=dp,
            mode="think",
            thinking_text=think_text,
        )
        for row, dp, think_text in zip(rows, domain_prefixes, thinking_texts_stage1)
    ]
    think_suffixes_flat = engine.beam_decode(prompts_think_stage2, beam_width=32, max_tokens=3)

    details = []
    for row, dp, nothink_suffixes, think_suffixes in zip(
        rows, domain_prefixes, nothink_suffixes_flat, think_suffixes_flat
    ):
        begin_token = f"<|{dp}_begin|>"
        outputs = [begin_token + s for s in think_suffixes] + [begin_token + s for s in nothink_suffixes]
        gold = row["gold"]
        gold_tokens = set(parse_sid_tokens(gold))
        candidate_tokens = {token for text in outputs for token in parse_sid_tokens(text)}
        passed = bool(gold_tokens & candidate_tokens)
        detail = base_detail(row)
        detail.update(
            score=int(passed),
            metric="semantic_id_pass@64",
            gold=gold,
            prediction=outputs,
            n_candidates=len(outputs),
            n_unique_semantic_ids=len(candidate_tokens),
        )
        details.append(detail)
    log(f"[recommend] 完成，n={len(rows)}，耗时 {time.monotonic() - t0:.1f}s")
    return details


def eval_sampled_batch(engine: VLLMEngine, rows: list[dict], dim: str) -> list[dict]:
    """懂世界 / 懂用户-F1 / 懂用户-逻辑链：同一维度所有 prompt 一次性批量采样。"""
    log(f"[{dim}] 开始，n={len(rows)}")
    t0 = time.monotonic()
    config_key = "user" if dim.startswith("user_") else dim
    config = SAMPLING[config_key]
    prompts = [build_prompt(row.get("system", ""), row["prompt"], mode="no_think") for row in rows]
    results = engine.sample(
        prompts,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    details = []
    for row, result in zip(rows, results):
        output = result.text
        gold = row["gold"]
        detail = base_detail(row)
        detail.update(gold=gold, prediction=output)
        if dim == "world":
            parsed = extract_answer(output)["matched_letters"]
            detail.update(score=int(is_correct(output, gold)), metric="accuracy", parsed_prediction=parsed)
        elif dim == "user_f1":
            pred = extract_json_array(output) or []
            result_f1 = f1_score([str(item) for item in pred], normalize_user_gold(gold))
            detail.update(score=result_f1["f1"], metric="f1", parsed_prediction=pred, metric_detail=result_f1)
        elif dim == "user_chain":
            payload = extract_json_object(output) or {}
            chain = payload.get("logic_chain", payload)
            pred_events = [
                Event(action=event.get("action", ""), logic=event.get("logic", ""))
                for event in chain.get("events", [])
            ]
            result_chain = score_chain(normalize_chain_gold(gold), pred_events)
            detail.update(
                score=result_chain.overall_score,
                metric="chain_overall",
                parsed_prediction=payload,
                metric_detail={
                    "action_alignment": result_chain.action_alignment,
                    "logic_alignment": result_chain.logic_alignment,
                    "n_gold": result_chain.n_gold,
                    "n_pred": result_chain.n_pred,
                },
            )
        else:
            raise ValueError(f"unsupported dimension: {dim}")
        details.append(detail)
    log(f"[{dim}] 完成，n={len(rows)}，耗时 {time.monotonic() - t0:.1f}s")
    return details


def evaluate_all_batch(engine: VLLMEngine, rows: list[dict]) -> list[dict]:
    """batch 模式入口：按维度分组，每个维度整体打包批量推理。"""
    by_dim: dict[str, list[dict]] = defaultdict(list)
    order: dict[str, int] = {}
    for idx, row in enumerate(rows):
        by_dim[row["dimension"]].append(row)
        order.setdefault(row["sample_id"], idx)

    details: list[dict] = []
    if by_dim.get("material"):
        details.extend(eval_material_batch(engine, by_dim["material"]))
    if by_dim.get("recommend"):
        details.extend(eval_recommend_batch(engine, by_dim["recommend"]))
    for dim in ("world", "user_f1", "user_chain"):
        if by_dim.get(dim):
            details.extend(eval_sampled_batch(engine, by_dim[dim], dim))

    # 恢复输入数据的原始顺序，方便对照数据文件排查。
    details.sort(key=lambda item: order[item["sample_id"]])
    return details


# ---------------------------------------------------------------------------
# multiprocess：batch 基础上，material 维度改用多个子进程并行跑分片。
# ---------------------------------------------------------------------------
def eval_material_multiprocess(
    model: str,
    lora: str | None,
    gpu: str,
    rows: list[dict],
    n_workers: int,
    beam_width: int = 64,
    max_tokens: int = 3,
) -> list[dict]:
    """把 material rows 切成 n_workers 片，每片起一个独立子进程（material_worker.py）
    各自建立 vLLM 实例、共享同一张 GPU（各自限制显存占比）并行跑 beam_decode。

    绕开的瓶颈：vLLM 官方 `LLM.beam_search()` 在 Python 层是单进程单线程地做逐
    step 候选构造/排序（GIL 限制），单进程内批量传入更多 prompt 并不能提升
    GPU/CPU 利用率；多进程才能真正利用多核 CPU 并行做这部分调度工作。
    """
    worker_script = Path(__file__).resolve().parent / "material_worker.py"
    n_workers = max(1, min(n_workers, len(rows)))
    shard_size = math.ceil(len(rows) / n_workers)
    shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]

    # 显存在 n_workers 个进程间平均分配，每个进程各自独立加载一份模型权重
    # （模型本身很小，约 1.5GB；主要开销在 KV cache，按并行度平分即可）。留一点
    # 余量（0.8 而不是 0.85 满打满算），避免多个进程 profile 显存时的瞬时峰值叠加
    # 导致个别进程初始化失败（vLLM 的 CUDA OOM 有时只报 "Engine core
    # initialization failed"，看不到底层原因）。
    gpu_mem_per_worker = 0.8 / len(shards)
    # vLLM 的 LLM() 初始化会有一段短暂的显存 profiling / NCCL 握手过程，多个进程
    # 完全同时起步容易互相挤占，错峰几秒钟启动能显著降低偶发初始化失败概率。
    stagger_seconds = 8

    with tempfile.TemporaryDirectory(prefix="material_mp_") as tmpdir:
        tmp = Path(tmpdir)
        procs = []
        out_paths = []
        log_paths = []
        for i, shard in enumerate(shards):
            in_path = tmp / f"shard_{i}.json"
            out_path = tmp / f"result_{i}.json"
            log_path = tmp / f"worker_{i}.log"
            in_path.write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
            out_paths.append(out_path)
            log_paths.append(log_path)

            cmd = [
                sys.executable,
                str(worker_script),
                "--model", model,
                "--input", str(in_path),
                "--output", str(out_path),
                "--gpu", gpu,
                "--gpu-memory-utilization", str(gpu_mem_per_worker),
                "--beam-width", str(beam_width),
                "--max-tokens", str(max_tokens),
                "--worker-id", str(i),
            ]
            if lora:
                cmd.extend(["--lora", lora])
            log_handle = log_path.open("w", encoding="utf-8")
            # start_new_session=True 让每个 worker 成为独立的进程组组长，这样才能用
            # os.killpg 把它和它 fork 出的 vLLM EngineCore 子进程一起终止；若仍用
            # 默认进程组，主进程被 kill -9 时这些子进程会变成孤儿继续占用显存。
            proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True
            )
            procs.append(proc)
            _register_worker_pgid(proc.pid)
            log(f"[material/mp] worker {i} 已启动（pid={proc.pid}），分片样本数={len(shard)}")
            if i < len(shards) - 1:
                time.sleep(stagger_seconds)

        try:
            log(f"[material/mp] 全部 {len(shards)} 个 worker 已启动，开始等待完成（每 {WORKER_POLL_INTERVAL}s 打印一次心跳）...")
            remaining = set(range(len(procs)))
            while remaining:
                time.sleep(WORKER_POLL_INTERVAL)
                for i in list(remaining):
                    if procs[i].poll() is not None:
                        remaining.discard(i)
                heartbeat = []
                for i in range(len(procs)):
                    status = "done" if i not in remaining else "running"
                    last_line = ""
                    if log_paths[i].exists():
                        lines = log_paths[i].read_text(encoding="utf-8", errors="replace").splitlines()
                        last_line = lines[-1] if lines else ""
                    heartbeat.append(f"worker{i}[{status}]: {last_line}")
                log("[material/mp] 心跳 - " + " | ".join(heartbeat))

            failures = []
            for i, proc in enumerate(procs):
                ret = proc.wait()
                if ret != 0:
                    failures.append((i, ret))
            if failures:
                detail_lines = []
                for i, ret in failures:
                    tail = log_paths[i].read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                    detail_lines.append(f"--- worker {i} (exit {ret}) tail of {log_paths[i]} ---")
                    detail_lines.extend(tail)
                raise RuntimeError(
                    f"material_worker 子进程失败: {failures}\n" + "\n".join(detail_lines)
                )

            details: list[dict] = []
            for out_path in out_paths:
                details.extend(json.loads(out_path.read_text(encoding="utf-8")))
        finally:
            # 无论正常完成、部分 worker 失败还是中途抛异常，都要确保还存活的 worker
            # 被终止并从全局注册表中移除，不留孤儿进程占用显存。
            for proc in procs:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass
                _unregister_worker_pgid(proc.pid)
    log(f"[material/mp] 全部 worker 完成，n={len(details)}")
    return details


def evaluate_all_multiprocess(
    model: str,
    lora: str | None,
    gpu: str,
    rows: list[dict],
    n_workers: int,
) -> list[dict]:
    """multiprocess 模式入口：material 用多进程 worker 并行，其余维度沿用单进程
    batch 调用（在主进程内新建一个 VLLMEngine，用于处理 material 之外的维度）。
    """
    by_dim: dict[str, list[dict]] = defaultdict(list)
    order: dict[str, int] = {}
    for idx, row in enumerate(rows):
        by_dim[row["dimension"]].append(row)
        order.setdefault(row["sample_id"], idx)

    details: list[dict] = []

    material_rows = by_dim.get("material") or []
    if material_rows:
        if len(material_rows) > MATERIAL_MULTIPROCESS_THRESHOLD:
            details.extend(
                eval_material_multiprocess(model, lora, gpu, material_rows, n_workers)
            )
        else:
            # 样本量太小，多进程启动开销不划算，退化为单进程 batch 调用。
            log(
                f"[material] 样本数 {len(material_rows)} 未超过阈值 "
                f"{MATERIAL_MULTIPROCESS_THRESHOLD}，退化为单进程 batch 调用"
            )
            engine = VLLMEngine(model=model, gpu=gpu, lora_path=lora)
            details.extend(eval_material_batch(engine, material_rows))
            _run_rest_dims(engine, by_dim, details)
            details.sort(key=lambda item: order[item["sample_id"]])
            return details

    # material 已经在独立子进程里跑完并释放显存，这里再为其余维度单独建立引擎。
    if any(by_dim.get(dim) for dim in ("recommend", "world", "user_f1", "user_chain")):
        engine = VLLMEngine(model=model, gpu=gpu, lora_path=lora)
        _run_rest_dims(engine, by_dim, details)

    details.sort(key=lambda item: order[item["sample_id"]])
    return details


def _run_rest_dims(engine: VLLMEngine, by_dim: dict[str, list[dict]], details: list[dict]) -> None:
    if by_dim.get("recommend"):
        details.extend(eval_recommend_batch(engine, by_dim["recommend"]))
    for dim in ("world", "user_f1", "user_chain"):
        if by_dim.get(dim):
            details.extend(eval_sampled_batch(engine, by_dim[dim], dim))


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", default=None, help="可选的 LoRA adapter 目录")
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--data", default=str(root / "data" / "competition_smoke.jsonl"))
    parser.add_argument("--output", default=str(root / "output" / "base_competition_smoke.json"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--engine-mode",
        choices=("legacy", "batch", "multiprocess"),
        default="batch",
        help=(
            "legacy=逐条串行调用（最初实现，最慢，回退用）；"
            "batch=按维度整体打包批量推理（默认）；"
            "multiprocess=在 batch 基础上，material 维度用多进程并行（样本量大时更快）。"
        ),
    )
    parser.add_argument(
        "--material-workers",
        type=int,
        default=DEFAULT_MATERIAL_WORKERS,
        help="multiprocess 模式下并行处理 material 的子进程数",
    )
    args = parser.parse_args()

    started = time.monotonic()
    rows = load_rows(Path(args.data))
    dim_counts = defaultdict(int)
    for row in rows:
        dim_counts[row["dimension"]] += 1
    log(
        f"tag={args.tag} engine_mode={args.engine_mode} model={args.model} "
        f"lora={args.lora or '(none)'} n_total={len(rows)} by_dim={dict(dim_counts)}"
    )

    if args.engine_mode == "multiprocess":
        details = evaluate_all_multiprocess(args.model, args.lora, args.gpu, rows, args.material_workers)
    else:
        log("初始化 vLLM engine...")
        engine = VLLMEngine(model=args.model, gpu=args.gpu, lora_path=args.lora)
        log(f"engine 初始化完成，耗时 {time.monotonic() - started:.1f}s")
        if args.engine_mode == "legacy":
            details = evaluate_all_legacy(engine, rows)
        else:
            details = evaluate_all_batch(engine, rows)

    log(f"全部维度推理完成，n={len(details)}，总耗时 {time.monotonic() - started:.1f}s，开始汇总...")
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in details:
        grouped[item["dimension"]].append(float(item["score"]))
    summary = {
        dim: {"n": len(scores), "mean": sum(scores) / len(scores)}
        for dim, scores in grouped.items()
    }
    log(f"summary={summary}")
    lora_sha256 = None
    if args.lora:
        adapter = Path(args.lora) / "adapter_model.safetensors"
        digest = hashlib.sha256()
        with adapter.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        lora_sha256 = digest.hexdigest()
    result = {
        "protocol": "competition-smoke-v1",
        "timestamp": datetime.now().isoformat(),
        "tag": args.tag,
        "engine_mode": args.engine_mode,
        "model": str(Path(args.model).resolve()),
        "lora": str(Path(args.lora).resolve()) if args.lora else None,
        "lora_sha256": lora_sha256,
        "seed": 42,
        "data": str(Path(args.data).resolve()),
        "n": len(details),
        "elapsed_seconds": time.monotonic() - started,
        "summary": summary,
        "details": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "n": len(details), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
