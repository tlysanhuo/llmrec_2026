#!/usr/bin/env python3
"""Paired held-out world answer-path audit for I-22 checkpoints.

The 46-row D(O2.General) selection set is prompt-disjoint from I-22 training
and never participates in backpropagation.  This is a checkpoint-selection
diagnostic, not an online-score estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
DEFAULT_PARENT = ROOT / "submissions/e3_userres_r80_retkl_v3_s875_platform"
DEFAULT_HOLDOUT = ROOT / "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl"
DEFAULT_TRAIN = ROOT / "assets/derived/processed/data_i22_world_retkl_v1.jsonl"
PREFIX = "<think>\n\n</think>\n正确答案是 ("


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=ADAPTER_DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=ADAPTER_DIR")
    return name, Path(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 46:
        raise ValueError(f"expected 46 held-out rows, got {len(rows)}")
    return rows


def prompt_and_target(row: dict[str, Any]) -> tuple[str, str]:
    output = str(row["output"])
    if not output.startswith(PREFIX) or not output.endswith(")"):
        raise ValueError("holdout response is not canonical")
    target = output[len(PREFIX) : -1]
    if not target or any(letter not in "ABCDEFGHIJ" for letter in target):
        raise ValueError(f"invalid held-out label: {target!r}")
    prompt = ""
    if row.get("instruction"):
        prompt += f"<|im_start|>system\n{row['instruction']}<|im_end|>\n"
    prompt += (
        f"<|im_start|>user\n{row['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{PREFIX}"
    )
    return prompt, target


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def absolute_summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "gold_token_logp_mean": round(mean([row["token_logp"] for row in rows]), 8),
        "gold_sum_logp_mean": round(mean([row["sum_logp"] for row in rows]), 8),
        "all_token_top1_rate": round(mean([row["all_token_top1"] for row in rows]), 8),
        "token_top1_mean": round(mean([row["token_top1"] for row in rows]), 8),
    }


def delta_summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    deltas = [row["sum_logp_delta"] for row in rows]
    return {
        "n": len(rows),
        "gold_sum_logp_delta_mean": round(mean(deltas), 8),
        "gold_sum_logp_delta_median": round(statistics.median(deltas), 8),
        "gold_sum_logp_improved_rate": round(mean([value > 0 for value in deltas]), 8),
        "parent_to_candidate_kl_mean": round(mean([row["kl"] for row in rows]), 10),
        "all_token_top1_delta": round(mean([row["all_token_top1_delta"] for row in rows]), 8),
        "token_top1_delta_mean": round(mean([row["token_top1_delta"] for row in rows]), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    adapters = [("parent", args.parent), *args.candidate]
    if len({name for name, _ in adapters}) != len(adapters):
        raise ValueError("adapter names must be unique")
    for name, path in adapters:
        if not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"{name}: {path}")

    rows = load_rows(args.holdout)
    examples = [prompt_and_target(row) for row in rows]
    prompt_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        prompt_groups[(row["instruction"], row["input"])].append(row["output"])
    duplicate_groups = [outputs for outputs in prompt_groups.values() if len(outputs) > 1]
    if len(prompt_groups) != 44 or len(duplicate_groups) != 2:
        raise ValueError("held-out prompt-group signature drifted")
    if any(len(set(outputs)) != 1 for outputs in duplicate_groups):
        raise ValueError("a repeated held-out prompt has conflicting labels")
    train_prompts = {
        json.loads(line)["input"] for line in args.train.open(encoding="utf-8") if line.strip()
    }
    if any(row["input"] in train_prompts for row in rows):
        raise ValueError("held-out prompt leaked into formal I-22 training data")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(
        model, args.parent, adapter_name="parent", is_trainable=False
    ).eval()
    for name, path in args.candidate:
        model.load_adapter(path, adapter_name=name, is_trainable=False, low_cpu_mem_usage=True)

    by_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    started = time.time()
    with torch.inference_mode():
        for index, (prompt, target) in enumerate(examples, start=1):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            ids = prompt_ids + target_ids
            input_ids = torch.tensor([ids], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            positions = torch.arange(len(prompt_ids) - 1, len(ids) - 1, device="cuda")
            targets = torch.tensor([target_ids], device="cuda")

            logits_by_model = {}
            for name, _ in adapters:
                model.set_adapter(name)
                logits_by_model[name] = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=positions,
                ).logits.float()

            parent_logits = logits_by_model["parent"]
            parent_log_probs = F.log_softmax(parent_logits, dim=-1)
            parent_probs = parent_log_probs.exp()
            parent_gold = parent_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            parent_top1 = parent_logits.argmax(-1)
            parent_all = float(parent_top1.eq(targets).all())
            parent_token = float(parent_top1.eq(targets).float().mean())
            for name, _ in adapters:
                logits = logits_by_model[name]
                log_probs = F.log_softmax(logits, dim=-1)
                gold = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                top1 = logits.argmax(-1)
                all_top1 = float(top1.eq(targets).all())
                token_top1 = float(top1.eq(targets).float().mean())
                by_model[name].append(
                    {
                        "sum_logp": float(gold.sum()),
                        "token_logp": float(gold.mean()),
                        "all_token_top1": all_top1,
                        "token_top1": token_top1,
                        "sum_logp_delta": float(gold.sum() - parent_gold.sum()),
                        "kl": float(F.kl_div(log_probs, parent_probs, reduction="none").sum(-1).mean()),
                        "all_token_top1_delta": all_top1 - parent_all,
                        "token_top1_delta": token_top1 - parent_token,
                    }
                )
            if index % 16 == 0:
                print(f"[i22-world-audit] {index}/{len(examples)}", flush=True)

    models = {}
    for name, path in adapters:
        models[name] = {
            "adapter": str(path.resolve()),
            "adapter_sha256": sha256(path / "adapter_model.safetensors"),
            "absolute": absolute_summary(by_model[name]),
            "delta_vs_parent": delta_summary(by_model[name]),
        }

    report = {
        "status": "COMPLETE_NOT_A_SCORE_ESTIMATE",
        "method": {
            "route": "canonical empty-think world answer letters only",
            "rows": len(rows),
            "prompt_groups": len(prompt_groups),
            "duplicate_same_label_prompt_groups": len(duplicate_groups),
            "stochastic_sampling": False,
            "selection_warning": "Prompt-disjoint D holdout for checkpoint selection; do not map to online score.",
        },
        "source": {
            "class": "D(O2.General), excluded from I-22 formal training",
            "holdout_sha256": sha256(args.holdout),
            "formal_train_sha256": sha256(args.train),
            "prompt_overlap": 0,
        },
        "models": models,
        "resources": {
            "gpu_count": 1,
            "elapsed_seconds": round(time.time() - started, 3),
            "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
