#!/usr/bin/env python3
"""material_worker.py — 懂物料 beam_decode 的单分片 worker 子进程。

背景：vLLM 官方 `LLM.beam_search()` 在 Python 层是单进程单线程地做逐 step 候选构造/
排序（受 GIL 限制），每步只生成 1 个 token，计算量小、CPU 端调度开销占比高，导致
批量传入更多 prompt 也无法把 GPU/CPU 利用率打满。本 worker 作为独立子进程运行，
配合 run_smoke_eval.py 用多进程把 material 数据切分成 N 片，各自建立独立的 vLLM
实例（共享同一张 GPU，各自限制 gpu_memory_utilization），绕开单进程 GIL 瓶颈。

用法（由 run_smoke_eval.py 内部调用，也可单独调试）：
    python3 material_worker.py --model <path> [--lora <path>] \
        --input <shard.jsonl> --output <result.json> \
        --gpu-memory-utilization 0.2 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.engine import VLLMEngine
from common.prompt import build_domain_prompt
from common.sid_utils import parse_sid_tokens


def _domain_prefix_of(gold_pattern: str) -> str:
    """从 gold pattern（完整带 `<|xxx_begin|>` 前缀的 SemanticID 文本）中取出目标
    domain 前缀（video/prod/ad/living）。
    """
    tokens = parse_sid_tokens(gold_pattern)
    if not tokens:
        raise ValueError(f"无法从 gold 中解析出 domain 前缀: {gold_pattern!r}")
    return tokens[0][0]


def log(worker_id: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][worker {worker_id}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--input", required=True, help="分片后的 rows，JSON 数组")
    parser.add_argument("--output", required=True, help="结果输出路径，JSON 数组")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3,
        help=(
            "beam search 生成的后续 token 数，默认 3（对齐线上「Single-stage "
            "generation with prompt_token」协议：domain-begin token 已硬编码拼进 "
            "prompt，模型只需继续生成 s_a/s_b/s_c 三个 token）。"
        ),
    )
    parser.add_argument("--worker-id", default="?", help="仅用于日志标识，不影响逻辑")
    args = parser.parse_args()
    wid = args.worker_id

    t0 = time.monotonic()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    log(wid, f"分片样本数={len(rows)}，开始初始化 vLLM engine（gpu_mem_util={args.gpu_memory_utilization:.3f}）...")
    if not rows:
        Path(args.output).write_text("[]", encoding="utf-8")
        return 0

    engine = VLLMEngine(
        model=args.model,
        gpu=args.gpu,
        lora_path=args.lora,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    log(wid, f"engine 初始化完成，耗时 {time.monotonic() - t0:.1f}s，开始 beam_decode（beam_width={args.beam_width}）...")
    t1 = time.monotonic()
    domain_prefixes = [_domain_prefix_of(row["gold"]) for row in rows]
    prompts = [
        build_domain_prompt(row.get("system", ""), row["prompt"], domain_prefix=dp, mode="no_think")
        for row, dp in zip(rows, domain_prefixes)
    ]
    # beam_decode 返回值已去除 prompt 前缀（裸 s_a/s_b/s_c，不含 domain-begin token，
    # 因为该 token 已被硬编码进 prompt），评分前需拼回前缀才能被 parse_sid_tokens 正确解析。
    all_suffixes = engine.beam_decode(prompts, beam_width=args.beam_width, max_tokens=args.max_tokens)
    all_predictions = [
        [f"<|{dp}_begin|>" + s for s in suffixes] for dp, suffixes in zip(domain_prefixes, all_suffixes)
    ]
    log(wid, f"beam_decode 完成，耗时 {time.monotonic() - t1:.1f}s，开始评分并写出结果...")

    details = []
    for row, predictions in zip(rows, all_predictions):
        gold = row["gold"]
        gold_tokens = set(parse_sid_tokens(gold))
        candidate_tokens = {token for text in predictions for token in parse_sid_tokens(text)}
        passed = bool(gold_tokens & candidate_tokens)
        details.append(
            {
                "sample_id": row["sample_id"],
                "dimension": row["dimension"],
                "task": row["task"],
                "source": row["source"],
                "score": int(passed),
                "metric": "semantic_id_pass@64",
                "gold": gold,
                "prediction": predictions,
                "n_candidates": len(predictions),
            }
        )

    Path(args.output).write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    log(wid, f"全部完成，总耗时 {time.monotonic() - t0:.1f}s，n={len(details)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
