#!/usr/bin/env python3
"""Paired mechanism and held-out retention audit for an E3 user residual.

This is not a score predictor. It measures whether the residual lowers
teacher-forced user loss while remaining close to the merged E3 parent. A
registered source dataset can supply deterministic non-user rows excluded from
the residual's training mix, and multiple residual scales can be tested while
loading the model only once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CLOSE_THINK = "</think>"
IM_END = "<|im_end|>"
DEFAULT_RETENTION_TASKS = ("rec_video", "rec_prod", "rec_ad", "rec_living")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(output: str) -> str:
    body = output.split(CLOSE_THINK, 1)[-1].lstrip()
    if body.startswith("["):
        return "action"
    if body.startswith("{"):
        return "topic"
    return "retention"


def row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("instruction", "")),
        str(row.get("input", "")),
        str(row.get("output", "")),
    )


def parse_scales(raw: str) -> list[float]:
    scales = []
    for part in raw.split(","):
        scale = float(part.strip())
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("scales must be finite and non-negative")
        if scale not in scales:
            scales.append(scale)
    if not scales:
        raise ValueError("at least one scale is required")
    return scales


def scale_key(scale: float) -> str:
    return f"{scale:g}"


def render(row: dict[str, object]) -> tuple[str, str]:
    instruction = str(row.get("instruction", ""))
    query = str(row.get("input", ""))
    user = "\n".join(part for part in (instruction, query) if part)
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{row['output']}{IM_END}\n"
    return prompt, response


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--retention-source", type=Path)
    parser.add_argument("--retention-exclude", type=Path)
    parser.add_argument(
        "--retention-tasks", default=",".join(DEFAULT_RETENTION_TASKS)
    )
    parser.add_argument("--retention-per-task", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--action-n", type=int, default=32)
    parser.add_argument("--topic-n", type=int, default=32)
    parser.add_argument("--retention-n", type=int, default=64)
    parser.add_argument("--cutoff", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=19260821)
    parser.add_argument("--scales", default="1.0")
    args = parser.parse_args()
    scales = parse_scales(args.scales)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Some evaluation environments ship a site-package named ``scripts``;
    # load the repository helpers by path so that it cannot shadow this repo.
    def load_repo_module(relative: str, name: str):
        path = REPO_ROOT / relative
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load repository helper: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    scoremax = load_repo_module("scripts/data/build_seed_scoremax_v1.py", "llmrec_scoremax")
    residual = load_repo_module("scripts/train/train_user_residual_retkl.py", "llmrec_residual")
    load_jsonl, stable_hash, task_of = scoremax.load_jsonl, scoremax.stable_hash, scoremax.task_of
    forward_kl = residual.forward_kl
    task_and_weights = residual.task_and_weights
    weighted_ce = residual.weighted_ce

    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    with args.dataset.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows[classify(str(row["output"]))].append(row)

    rng = random.Random(args.seed)
    requested = {
        "action": args.action_n,
        "topic": args.topic_n,
        "retention": args.retention_n,
    }
    selected: list[tuple[str, dict[str, object]]] = []
    for task, count in requested.items():
        rng.shuffle(rows[task])
        if len(rows[task]) < count:
            raise RuntimeError(f"requested {count} {task} rows, found {len(rows[task])}")
        selected.extend((task, row) for row in rows[task][:count])

    heldout_audit: dict[str, Any] = {}
    if args.retention_source:
        retention_tasks = tuple(
            task.strip() for task in args.retention_tasks.split(",") if task.strip()
        )
        if not retention_tasks:
            raise ValueError("retention tasks cannot be empty")
        excluded = set()
        if args.retention_exclude:
            excluded = {row_key(row) for row in load_jsonl(args.retention_exclude)}
        source_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in load_jsonl(args.retention_source):
            task = task_of(row)
            if task in retention_tasks and row_key(row) not in excluded:
                source_buckets[task].append(row)
        for task in retention_tasks:
            candidates = source_buckets[task]
            candidates.sort(
                key=lambda row: stable_hash(
                    args.seed, "scale-heldout", task, *row_key(row)
                )
            )
            if len(candidates) < args.retention_per_task:
                raise RuntimeError(
                    f"only {len(candidates)} held-out {task} rows, "
                    f"need {args.retention_per_task}"
                )
            chosen = candidates[: args.retention_per_task]
            selected.extend((task, row) for row in chosen)
            heldout_audit[task] = {
                "available_after_exclusion": len(candidates),
                "selected": len(chosen),
            }

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda()
    parent_wrapper = PeftModel.from_pretrained(base, args.parent, is_trainable=False)
    merged_parent = parent_wrapper.merge_and_unload(safe_merge=True)
    model = PeftModel.from_pretrained(merged_parent, args.residual, is_trainable=False)
    model.eval()

    lora_layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    if not lora_layers:
        raise RuntimeError("residual model contains no LoRA layers")

    def set_residual_scale(scale: float) -> None:
        for layer in lora_layers:
            for adapter in layer.active_adapters:
                layer.set_scale(adapter, scale)

    by_scale: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped_too_long = 0
    with torch.no_grad():
        for index, (expected_task, row) in enumerate(selected, start=1):
            prompt, response = render(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(response, add_special_tokens=False)
            input_ids = prompt_ids + target_ids
            if len(input_ids) > args.cutoff:
                skipped_too_long += 1
                continue

            targets = torch.tensor(target_ids, dtype=torch.long, device="cuda")
            routed_task, weights = task_and_weights(targets)
            expected_route = (
                expected_task if expected_task in {"action", "topic"} else "retention"
            )
            if routed_task != expected_route:
                raise RuntimeError(f"task routing mismatch: {expected_task}/{routed_task}")
            positions = torch.arange(
                len(prompt_ids) - 1,
                len(input_ids) - 1,
                dtype=torch.long,
                device="cuda",
            )
            batch = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long, device="cuda"),
                "attention_mask": torch.ones(
                    (1, len(input_ids)), dtype=torch.long, device="cuda"
                ),
                "logits_to_keep": positions,
            }
            disable_adapter = getattr(model, "disable_adapter", None)
            if disable_adapter is None:
                raise RuntimeError("expected a PEFT residual with disable_adapter()")
            context = disable_adapter()
            with context:
                parent_logits = model(**batch).logits.detach()
            parent_ce = None
            if expected_task in {"action", "topic"}:
                parent_ce = float(weighted_ce(parent_logits, targets, weights))
            for scale in scales:
                set_residual_scale(scale)
                candidate_logits = model(**batch).logits.detach()
                if candidate_logits.size(1) != len(target_ids):
                    raise RuntimeError("partial logit length does not match target length")
                metrics = by_scale[scale_key(scale)][expected_task]
                metrics["kl"].append(float(forward_kl(candidate_logits, parent_logits)))
                metrics["top1_agreement"].append(
                    float(
                        candidate_logits.argmax(dim=-1)
                        .eq(parent_logits.argmax(dim=-1))
                        .float()
                        .mean()
                    )
                )
                if parent_ce is not None:
                    candidate_ce = float(weighted_ce(candidate_logits, targets, weights))
                    metrics["parent_ce"].append(parent_ce)
                    metrics["candidate_ce"].append(candidate_ce)
                    metrics["ce_delta"].append(candidate_ce - parent_ce)
                del candidate_logits

            set_residual_scale(1.0)
            del batch, parent_logits, targets
            if index % 16 == 0 or index == len(selected):
                print(f"[paired-audit] {index}/{len(selected)}", flush=True)

    result: dict[str, object] = {
        "scope": (
            "deterministic user mechanism plus O1-held-out parent-retention audit; "
            "not an online score predictor"
        ),
        "seed": args.seed,
        "scales": scales,
        "requested": requested,
        "heldout_retention": heldout_audit,
        "skipped_too_long": skipped_too_long,
        "artifacts": {
            "base": str(args.base.resolve()),
            "parent": str(args.parent.resolve()),
            "parent_adapter_sha256": sha256(args.parent / "adapter_model.safetensors"),
            "residual": str(args.residual.resolve()),
            "residual_adapter_sha256": sha256(args.residual / "adapter_model.safetensors"),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256(args.dataset),
        },
        "metrics_by_scale": {},
    }
    if args.retention_source:
        artifacts = result["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts["retention_source"] = str(args.retention_source.resolve())
        artifacts["retention_source_sha256"] = sha256(args.retention_source)
        if args.retention_exclude:
            artifacts["retention_exclude"] = str(args.retention_exclude.resolve())
            artifacts["retention_exclude_sha256"] = sha256(args.retention_exclude)
    metrics_by_scale = result["metrics_by_scale"]
    assert isinstance(metrics_by_scale, dict)
    for scale, task_values in by_scale.items():
        metrics_by_scale[scale] = {
            task: {name: summarize(values) for name, values in task_metrics.items()}
            for task, task_metrics in task_values.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
