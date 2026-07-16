#!/usr/bin/env python3
"""Paired topic/action gold-path audit for I-21 checkpoints.

This is a deterministic E-class mechanism diagnostic, not an online score
estimate.  It compares canonical topic JSON and action-list gold paths from the
registered offline dev set while sharing one base model across all adapters.
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
DEFAULT_DEV = ROOT / "assets/evaluation/offline_eval"


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


def stable_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            packed = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            rows.append((hashlib.sha256(packed.encode()).hexdigest(), row))
    if len(rows) < count:
        raise ValueError(f"{path} has {len(rows)} rows, need {count}")
    return [row for _, row in sorted(rows)[:count]]


def prompt_of(row: dict[str, Any]) -> str:
    user = str(row["user"])
    if not user.rstrip().endswith("/no_think"):
        user += "/no_think"
    prompt = ""
    if row.get("system"):
        prompt += f"<|im_start|>system\n{row['system']}<|im_end|>\n"
    return (
        prompt
        + f"<|im_start|>user\n{user}<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n"
    )


def target_of(task: str, row: dict[str, Any]) -> str:
    if task == "topic":
        value = {"logic_chain": {"name": "", "events": row["gold"]}}
    elif task == "action":
        value = row["gold"]
    else:
        raise ValueError(task)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "gold_token_logp_mean": round(mean([row["token_logp"] for row in rows]), 8),
        "gold_sum_logp_mean": round(mean([row["sum_logp"] for row in rows]), 8),
        "top1_agreement_mean": round(mean([row["top1_agreement"] for row in rows]), 8),
    }


def delta_summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    deltas = [row["sum_logp_delta"] for row in rows]
    return {
        "n": len(rows),
        "gold_sum_logp_delta_mean": round(mean(deltas), 8),
        "gold_sum_logp_delta_median": round(statistics.median(deltas), 8),
        "gold_sum_logp_improved_rate": round(mean([value > 0 for value in deltas]), 8),
        "parent_to_candidate_kl_mean": round(mean([row["kl"] for row in rows]), 10),
        "top1_agreement_delta_mean": round(mean([row["top1_delta"] for row in rows]), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--per-task", type=int, default=32)
    parser.add_argument("--cutoff", type=int, default=16384)
    args = parser.parse_args()

    adapters = [("parent", args.parent), *args.candidate]
    if len({name for name, _ in adapters}) != len(adapters):
        raise ValueError("adapter names must be unique")
    for name, path in adapters:
        if not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"{name}: {path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sources = {
        "topic": args.dev / "dev_topic.jsonl",
        "action": args.dev / "dev_action.jsonl",
    }
    rows: list[dict[str, Any]] = []
    for task, path in sources.items():
        for row in stable_rows(path, args.per_task):
            rows.append({"task": task, "prompt": prompt_of(row), "target": target_of(task, row)})

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

    by_model: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    skipped = 0
    started = time.time()
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
            ids = prompt_ids + target_ids
            if len(ids) > args.cutoff:
                skipped += 1
                continue
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
            for name, _ in adapters:
                logits = logits_by_model[name]
                log_probs = F.log_softmax(logits, dim=-1)
                gold = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                top1 = logits.argmax(-1)
                kl = F.kl_div(log_probs, parent_probs, reduction="none").sum(-1).mean()
                agreement = top1.eq(targets).float().mean()
                parent_agreement = parent_top1.eq(targets).float().mean()
                by_model[name].append(
                    {
                        "task": row["task"],
                        "sum_logp": float(gold.sum()),
                        "token_logp": float(gold.mean()),
                        "top1_agreement": float(agreement),
                        "sum_logp_delta": float(gold.sum() - parent_gold.sum()),
                        "kl": float(kl),
                        "top1_delta": float(agreement - parent_agreement),
                    }
                )
            if index % 16 == 0:
                print(f"[i21-audit] {index}/{len(rows)}", flush=True)

    models = {}
    for name, path in adapters:
        model_rows = by_model[name]
        absolute = {}
        delta = {}
        for task in ("topic", "action"):
            selected = [row for row in model_rows if row["task"] == task]
            absolute[task] = summary(selected)
            delta[task] = delta_summary(selected)
        models[name] = {
            "adapter": str(path.resolve()),
            "adapter_sha256": sha256(path / "adapter_model.safetensors"),
            "absolute": absolute,
            "delta_vs_parent": delta,
        }

    report = {
        "status": "COMPLETE_NOT_A_SCORE_ESTIMATE",
        "method": {
            "route": "canonical /no_think topic JSON and action-list teacher-forced paths",
            "per_task_requested": args.per_task,
            "skipped_over_cutoff": skipped,
            "selection_warning": "E-class mechanism/retention diagnostic only",
        },
        "source": {
            "class": "E",
            "file_sha256": {task: sha256(path) for task, path in sources.items()},
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
