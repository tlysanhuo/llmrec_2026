#!/usr/bin/env python3
"""Gate I-24 material retention against the frozen I-23 parent.

The holdout is selected deterministically from the registered D(O1) material
subset after exact normalized rows present in the I-24 training mix are
removed.  Metrics are teacher-forced full-response parent-to-candidate KL and
parent/candidate top-1 agreement.  They are safety diagnostics, not online
score estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
DEFAULT_SOURCE = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
DEFAULT_EXCLUDE = (
    ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
)
MATERIAL_TASKS = ("material_desc2sid", "material_sid2desc")
LOCKED_SOURCE_SHA256 = (
    "13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f"
)
LOCKED_EXCLUDE_SHA256 = (
    "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0"
)
LOCKED_PARENT_ADAPTER_SHA256 = (
    "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
)
EXPECTED_EXCLUDED_PER_TASK = 281
IM_END = "<|im_end|>"
KL_CHUNK = 8


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Canonical normalized row identity, including the history field."""

    history = json.dumps(
        row.get("history") or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return row["instruction"], row["input"], row["output"], history


def row_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(
        exact_row_key(row), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_digest(seed: int, task: str, row: dict[str, Any]) -> str:
    payload = json.dumps(
        [seed, "i24-material-heldout", task, *exact_row_key(row)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("history"):
        raise ValueError("selected material row unexpectedly has non-empty history")
    user = "\n".join(
        value
        for value in (str(row.get("instruction", "")), str(row.get("input", "")))
        if value
    )
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{row['output']}{IM_END}\n"
    return prompt, response


def load_holdout(
    source_path: Path, exclude_path: Path, seed: int, per_task: int
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    helper_path = ROOT / "scripts/data/build_seed_scoremax_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_scoremax_i24", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load repository helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    load_jsonl, task_of = helper.load_jsonl, helper.task_of

    excluded_keys = {exact_row_key(row) for row in load_jsonl(exclude_path)}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(source_path):
        task = task_of(row)
        if task in MATERIAL_TASKS:
            buckets[task].append(row)

    selected: list[tuple[str, str, dict[str, Any]]] = []
    audit: dict[str, Any] = {}
    combined_manifest = []
    for task in MATERIAL_TASKS:
        source_rows = buckets[task]
        excluded_count = sum(exact_row_key(row) in excluded_keys for row in source_rows)
        if excluded_count != EXPECTED_EXCLUDED_PER_TASK:
            raise RuntimeError(
                f"{task}: expected {EXPECTED_EXCLUDED_PER_TASK} exact training-row "
                f"matches, found {excluded_count}"
            )
        candidates = [
            row for row in source_rows if exact_row_key(row) not in excluded_keys
        ]
        if len(candidates) < per_task:
            raise RuntimeError(
                f"{task}: only {len(candidates)} rows remain after exact exclusion; "
                f"need {per_task}"
            )
        ranked = sorted(
            ((selection_digest(seed, task, row), row_digest(row), row) for row in candidates),
            key=lambda item: (item[0], item[1]),
        )
        chosen = ranked[:per_task]
        task_manifest = [f"{task}|{selection_hash}|{digest}" for selection_hash, digest, _ in chosen]
        combined_manifest.extend(task_manifest)
        selected.extend((task, digest, row) for _, digest, row in chosen)
        audit[task] = {
            "source_rows": len(source_rows),
            "exact_training_rows_excluded": excluded_count,
            "available_after_exclusion": len(candidates),
            "unique_after_exclusion": len({exact_row_key(row) for row in candidates}),
            "selected": len(chosen),
            "selected_manifest_sha256": hashlib.sha256(
                "\n".join(task_manifest).encode("utf-8")
            ).hexdigest(),
        }

    audit["combined_selected_manifest_sha256"] = hashlib.sha256(
        "\n".join(combined_manifest).encode("utf-8")
    ).hexdigest()
    return selected, audit


def chunked_kl(policy_logits: Any, reference_logits: Any) -> float:
    import torch
    import torch.nn.functional as functional

    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(
            f"candidate/parent logit mismatch: {policy_logits.shape}/{reference_logits.shape}"
        )
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), KL_CHUNK):
        end = min(start + KL_CHUNK, policy_logits.size(1))
        candidate = policy_logits[:, start:end].float()
        parent = reference_logits[:, start:end].float()
        total += functional.kl_div(
            functional.log_softmax(candidate, dim=-1),
            functional.softmax(parent, dim=-1),
            reduction="sum",
        )
    value = float(total / policy_logits.size(1))
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite KL: {value}")
    return value


def gold_sum_logp(logits: Any, targets: Any) -> float:
    import torch
    import torch.nn.functional as functional

    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, logits.size(1), KL_CHUNK):
        end = min(start + KL_CHUNK, logits.size(1))
        log_probs = functional.log_softmax(logits[:, start:end].float(), dim=-1)
        total += log_probs.gather(
            -1, targets[start:end].view(1, -1, 1)
        ).sum()
    value = float(total)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite gold log-probability: {value}")
    return value


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def summarize(rows: list[dict[str, float]]) -> dict[str, float | int]:
    token_count = sum(int(row["target_tokens"]) for row in rows)
    kl_values = [row["kl"] for row in rows]
    agreement_values = [row["top1_agreement"] for row in rows]
    gold_deltas = [row["gold_sum_logp_delta"] for row in rows]
    return {
        "n": len(rows),
        "target_tokens": token_count,
        "parent_to_candidate_kl_mean": round(statistics.fmean(kl_values), 10),
        "parent_to_candidate_kl_p95": round(percentile(kl_values, 0.95), 10),
        "parent_to_candidate_kl_max": round(max(kl_values), 10),
        "top1_agreement_mean": round(statistics.fmean(agreement_values), 8),
        "top1_agreement_token_weighted": round(
            sum(row["top1_matches"] for row in rows) / token_count, 8
        ),
        "parent_gold_top1_mean": round(
            statistics.fmean(row["parent_gold_top1"] for row in rows), 8
        ),
        "candidate_gold_top1_mean": round(
            statistics.fmean(row["candidate_gold_top1"] for row in rows), 8
        ),
        "gold_sum_logp_delta_mean": round(statistics.fmean(gold_deltas), 8),
        "gold_sum_logp_improved_rate": round(
            statistics.fmean(delta > 0 for delta in gold_deltas), 8
        ),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one I-24 adapter with I-23 on exact-row-excluded material."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--seed", type=int, default=19260827)
    parser.add_argument("--per-task", type=int, default=96)
    parser.add_argument("--cutoff", type=int, default=16384)
    args = parser.parse_args()

    if args.per_task <= 0 or args.cutoff <= 0:
        raise ValueError("per-task and cutoff must be positive")
    required_files = {
        "base config": args.base / "config.json",
        "parent adapter": args.parent / "adapter_model.safetensors",
        "candidate adapter": args.candidate / "adapter_model.safetensors",
        "source": args.source,
        "exclude": args.exclude,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    source_sha = sha256(args.source)
    exclude_sha = sha256(args.exclude)
    parent_sha = sha256(args.parent / "adapter_model.safetensors")
    locked = {
        "source": (source_sha, LOCKED_SOURCE_SHA256),
        "exclude": (exclude_sha, LOCKED_EXCLUDE_SHA256),
        "parent": (parent_sha, LOCKED_PARENT_ADAPTER_SHA256),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in locked.items()
        if observed != expected
    }
    if mismatches:
        raise RuntimeError(f"locked I-24 gate artifact mismatch: {mismatches}")

    selected, holdout_audit = load_holdout(
        args.source, args.exclude, args.seed, args.per_task
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=True, trust_remote_code=True, use_fast=True
    )
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
    model.load_adapter(
        args.candidate,
        adapter_name="candidate",
        is_trainable=False,
        low_cpu_mem_usage=True,
    )

    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    started = time.time()
    with torch.inference_mode():
        for index, (task, digest, row) in enumerate(selected, start=1):
            prompt, response = render(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(response, add_special_tokens=False)
            input_ids_list = prompt_ids + target_ids
            if not target_ids or len(input_ids_list) > args.cutoff:
                raise RuntimeError(
                    f"{task}/{digest}: invalid length prompt={len(prompt_ids)} "
                    f"target={len(target_ids)} cutoff={args.cutoff}"
                )

            input_ids = torch.tensor([input_ids_list], device="cuda", dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            targets = torch.tensor(target_ids, device="cuda", dtype=torch.long)
            prediction_positions = torch.arange(
                len(prompt_ids) - 1,
                len(input_ids_list) - 1,
                device="cuda",
                dtype=torch.long,
            )
            logits: dict[str, Any] = {}
            for name in ("parent", "candidate"):
                model.set_adapter(name)
                logits[name] = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=prediction_positions,
                ).logits.detach()
                if logits[name].size(1) != len(target_ids):
                    raise RuntimeError(
                        f"{name} partial-logit length mismatch: "
                        f"{logits[name].size(1)}/{len(target_ids)}"
                    )

            parent_top1 = logits["parent"].argmax(-1).squeeze(0)
            candidate_top1 = logits["candidate"].argmax(-1).squeeze(0)
            matches = int(candidate_top1.eq(parent_top1).sum())
            parent_gold = int(parent_top1.eq(targets).sum())
            candidate_gold = int(candidate_top1.eq(targets).sum())
            parent_logp = gold_sum_logp(logits["parent"], targets)
            candidate_logp = gold_sum_logp(logits["candidate"], targets)
            token_count = len(target_ids)
            by_task[task].append(
                {
                    "target_tokens": float(token_count),
                    "kl": chunked_kl(logits["candidate"], logits["parent"]),
                    "top1_matches": float(matches),
                    "top1_agreement": matches / token_count,
                    "parent_gold_top1": parent_gold / token_count,
                    "candidate_gold_top1": candidate_gold / token_count,
                    "gold_sum_logp_delta": candidate_logp - parent_logp,
                }
            )
            del input_ids, attention_mask, targets, logits
            if index % 16 == 0 or index == len(selected):
                print(f"[i24-material] {index}/{len(selected)}", flush=True)

    metrics = {task: summarize(by_task[task]) for task in MATERIAL_TASKS}
    gate_pass = all(
        statistics.fmean(row["kl"] for row in by_task[task]) <= 0.005
        and statistics.fmean(row["top1_agreement"] for row in by_task[task]) >= 0.99
        for task in MATERIAL_TASKS
    )
    report = {
        "status": "COMPLETE_NOT_A_SCORE_ESTIMATE",
        "scope": (
            "I-24 material safety gate on deterministic exact-row-excluded "
            "D(O1) material rows; never training data and not an online-score estimate"
        ),
        "method": {
            "seed": args.seed,
            "per_task": args.per_task,
            "tasks": list(MATERIAL_TASKS),
            "route": "teacher-forced full assistant response including terminal tokens",
            "selection": "ascending SHA256(seed, namespace, task, exact normalized row)",
            "exact_row_fields": ["instruction", "input", "output", "history"],
            "stochastic_sampling": False,
            "gate_thresholds": {
                "each_direction_parent_to_candidate_kl_mean_max": 0.005,
                "each_direction_top1_agreement_mean_min": 0.99,
            },
        },
        "holdout": holdout_audit,
        "artifacts": {
            "base": str(args.base.resolve()),
            "base_config_sha256": sha256(args.base / "config.json"),
            "parent": str(args.parent.resolve()),
            "parent_adapter_sha256": parent_sha,
            "candidate": str(args.candidate.resolve()),
            "candidate_adapter_sha256": sha256(
                args.candidate / "adapter_model.safetensors"
            ),
            "source": str(args.source.resolve()),
            "source_class": "D(O1) material subset of registered data_seed_teacher_v1",
            "source_sha256": source_sha,
            "exact_row_exclude": str(args.exclude.resolve()),
            "exact_row_exclude_sha256": exclude_sha,
        },
        "metrics_by_material_direction": metrics,
        "gate_pass": gate_pass,
        "resources": {
            "gpu_count": 1,
            "elapsed_seconds": round(time.time() - started, 3),
            "peak_gpu_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 2**30, 4
            ),
        },
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
