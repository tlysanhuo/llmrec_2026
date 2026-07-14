#!/usr/bin/env python3
"""Audit raw chosen/rejected margins for an adapter on the locked I-16 holdout.

This is a deterministic mechanism and drift audit.  It does not estimate an
online competition score and must never feed holdout rows back into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
DEFAULT_DATA = ROOT / "assets/evaluation/holdout/data_o1_reward_preference_v1_holdout.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 8),
        "median": round(percentile(values, 0.5), 8),
        "p10": round(percentile(values, 0.1), 8),
        "p90": round(percentile(values, 0.9), 8),
        "chosen_positive_rate": round(sum(value > 0 for value in values) / len(values), 8),
    }


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    return min(max(cutoff_len - new_target_len, 0), source_len), new_target_len


def load_sample(path: Path, per_task: int) -> tuple[list[dict[str, Any]], str]:
    grouped: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key = f"{row['meta']['source_row']}|{row['meta']['prompt_group']}"
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            grouped[row["meta"]["task"]].append((digest, key, row))

    selected = []
    manifest = []
    for task, values in sorted(grouped.items()):
        if len(values) < per_task:
            raise ValueError(f"task {task} has only {len(values)} rows, requested {per_task}")
        for digest, key, row in sorted(values)[:per_task]:
            selected.append(row)
            manifest.append(f"{task}|{digest}|{key}|{row['meta']['negative_tier']}")
    manifest_hash = hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest()
    return selected, manifest_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--per-task", type=int, default=64)
    parser.add_argument("--cutoff-len", type=int, default=16_384)
    args = parser.parse_args()

    adapter_file = args.adapter / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, manifest_hash = load_sample(args.data, args.per_task)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()

    def encode(row: dict[str, Any], response_key: str) -> tuple[list[int], list[int]]:
        content = "\n".join(value for value in (row["instruction"], row["input"]) if value)
        prompt = (
            "<|im_start|>user\n"
            + content
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(row[response_key] + "<|im_end|>\n", add_special_tokens=False)
        return prompt_ids, response_ids

    results = []
    started = time.time()
    with torch.inference_mode():
        for row in rows:
            prompt_ids, chosen_ids = encode(row, "chosen")
            _, rejected_ids = encode(row, "rejected")
            source_len, target_len = infer_seqlen(
                len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), args.cutoff_len
            )
            sequences = [
                prompt_ids[:source_len] + chosen_ids[:target_len],
                prompt_ids[:source_len] + rejected_ids[:target_len],
            ]
            lengths = [len(sequence) for sequence in sequences]
            maximum = max(lengths)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_ids = torch.tensor(
                [sequence + [pad_id] * (maximum - len(sequence)) for sequence in sequences],
                device="cuda",
            )
            attention_mask = torch.tensor(
                [[1] * len(sequence) + [0] * (maximum - len(sequence)) for sequence in sequences],
                device="cuda",
            )
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            logps = []
            for sequence_index, sequence_len in enumerate(lengths):
                token_logps = -functional.cross_entropy(
                    logits[sequence_index, source_len - 1 : sequence_len - 1].float(),
                    input_ids[sequence_index, source_len:sequence_len],
                    reduction="none",
                )
                logps.append((token_logps.sum().item(), token_logps.mean().item(), len(token_logps)))
            results.append(
                {
                    "task": row["meta"]["task"],
                    "negative_tier": row["meta"]["negative_tier"],
                    "raw_margin": logps[0][0] - logps[1][0],
                    "normalized_margin": logps[0][1] - logps[1][1],
                    "chosen_sum_logp": logps[0][0],
                    "rejected_sum_logp": logps[1][0],
                    "chosen_mean_logp": logps[0][1],
                    "rejected_mean_logp": logps[1][1],
                    "chosen_tokens": logps[0][2],
                    "rejected_tokens": logps[1][2],
                }
            )
            del logits, input_ids, attention_mask

    def grouped_summary(field: str, key: str) -> dict[str, Any]:
        groups: dict[str, list[float]] = defaultdict(list)
        for result in results:
            groups[result[key]].append(result[field])
        return {name: summarize(values) for name, values in sorted(groups.items())}

    report = {
        "status": "COMPLETE_NOT_A_SCORE_ESTIMATE",
        "model": {
            "base": str(args.base.resolve()),
            "adapter": str(args.adapter.resolve()),
            "adapter_sha256": sha256(adapter_file),
        },
        "source": {
            "path": str(args.data.resolve()),
            "class": "E(D(O1))",
            "sha256": sha256(args.data),
        },
        "method": {
            "template": "qwen3_nothink",
            "cutoff_len": args.cutoff_len,
            "per_task": args.per_task,
            "rows": len(rows),
            "sample_manifest_sha256": manifest_hash,
            "task_counts": dict(sorted(Counter(row["meta"]["task"] for row in rows).items())),
            "negative_tier_counts": dict(
                sorted(Counter(row["meta"]["negative_tier"] for row in rows).items())
            ),
            "raw_margin": "sum logp(chosen) - sum logp(rejected)",
            "normalized_margin": "mean token logp(chosen) - mean token logp(rejected)",
            "stochastic_sampling": False,
        },
        "raw_margin": {
            "global": summarize([result["raw_margin"] for result in results]),
            "by_task": grouped_summary("raw_margin", "task"),
            "by_negative_tier": grouped_summary("raw_margin", "negative_tier"),
        },
        "normalized_margin": {
            "global": summarize([result["normalized_margin"] for result in results]),
            "by_task": grouped_summary("normalized_margin", "task"),
            "by_negative_tier": grouped_summary("normalized_margin", "negative_tier"),
        },
        "absolute_logp": {
            "chosen_sum_by_task": grouped_summary("chosen_sum_logp", "task"),
            "rejected_sum_by_task": grouped_summary("rejected_sum_logp", "task"),
            "chosen_mean_by_task": grouped_summary("chosen_mean_logp", "task"),
            "rejected_mean_by_task": grouped_summary("rejected_mean_logp", "task"),
        },
        "resources": {
            "gpu_count": 1,
            "elapsed_seconds": round(time.time() - started, 4),
            "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        },
        "interpretation": "Use recommendation margins to confirm the trained ordering mechanism and action margins only as a drift diagnostic. Do not map these values to an online score.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
