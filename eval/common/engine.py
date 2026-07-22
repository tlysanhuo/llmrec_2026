#!/usr/bin/env python3
"""common/engine.py — vLLM 推理封装：beam search / 采样解码。

独立于 llmrec_2026-main/scripts/eval/ 下的旧脚本自成一套实现，
不 import 旧脚本代码。vLLM 是重依赖，仅在真正调用时才 import，
以便本模块在没有安装 vLLM 的环境下也能被测试代码（不涉及推理部分）安全导入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]


class VLLMEngine:
    """对 vLLM LLM 的最小封装：beam_decode 用于懂物料/懂推荐的候选生成，
    sample 用于懂用户/懂推荐-thinking路/懂世界的采样生成。
    """

    def __init__(
        self,
        model: str,
        *,
        gpu: str = "0",
        max_model_len: int = 40960,
        gpu_memory_utilization: float = 0.85,
        seed: int = 42,
        lora_path: str | None = None,
        max_logprobs: int = 130,
    ):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            model=model,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            seed=seed,
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_logprobs=max_logprobs,
        )
        self._lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest
            import json
            from pathlib import Path

            adapter_config = json.loads((Path(lora_path) / "adapter_config.json").read_text())
            adapter_rank = int(adapter_config["r"])
            supported_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
            max_lora_rank = next((r for r in supported_ranks if r >= adapter_rank), None)
            if max_lora_rank is None:
                raise ValueError(f"LoRA rank {adapter_rank} exceeds vLLM's supported maximum")
            kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank)
            self._lora_request = LoRARequest("adapter", 1, str(lora_path))

        self.llm = LLM(**kwargs)

    def beam_decode(
        self, prompts: list[str], beam_width: int, max_tokens: int = 3, chunk: int = 30
    ) -> list[list[str]]:
        """beam search，返回每条 prompt 的 beam_width 个候选生成文本（已去除 prompt 前缀）。

        chunk 默认调小到 30：beam search 是 Python 层逐 step 单线程调度（GIL
        限制），单个 chunk 太大会导致长时间没有任何中间输出、外部无法判断进度是
        否卡住；调小 chunk 让每完成一小批就打印一次进度，代价是 chunk 间有极小的
        调度开销（可忽略，相比 beam search 本身的耗时）。
        """
        import time

        from vllm.sampling_params import BeamSearchParams

        params = BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens)
        results: list[list[str]] = []
        n_chunks = (len(prompts) + chunk - 1) // chunk or 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}][beam_decode] 开始，共 {len(prompts)} 条，分 {n_chunks} 个 chunk（每 chunk {chunk} 条）", flush=True)
        for ci, i in enumerate(range(0, len(prompts), chunk), 1):
            batch = prompts[i : i + chunk]
            t0 = time.monotonic()
            outs = self.llm.beam_search(
                [{"prompt": p} for p in batch], params, lora_request=self._lora_request
            )
            for p, o in zip(batch, outs):
                cands = []
                for seq in o.sequences:
                    gen = seq.text[len(p):] if seq.text.startswith(p) else seq.text
                    cands.append(gen)
                results.append(cands)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{ts}][beam_decode] chunk {ci}/{n_chunks} 完成"
                f"（本 chunk {len(batch)} 条，耗时 {time.monotonic() - t0:.1f}s，"
                f"累计已完成 {len(results)}/{len(prompts)}）",
                flush=True,
            )
        return results

    def sample(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        n: int = 1,
        stop: list[str] | None = None,
        seed: int = 42,
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        sp = SamplingParams(
            n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, seed=seed, stop=stop
        )
        outs = self.llm.generate(prompts, sp, lora_request=self._lora_request)
        results = []
        for o in outs:
            for generated in o.outputs:
                results.append(
                    GenerationResult(text=generated.text, token_ids=list(generated.token_ids))
                )
        return results
