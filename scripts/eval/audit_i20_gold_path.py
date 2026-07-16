#!/usr/bin/env python3
"""Compare I-20 checkpoints on deterministic direct-route gold SID paths.

This is a mechanism and retention diagnostic, not an online-score estimate.
It uses the registered offline recommendation dev rows and scores only the
three SID tokens generated after the evaluator supplies the domain marker.
One frozen base is shared by the parent and every candidate adapter.
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
DOMAINS = {
    "video": ("video", "<|video_begin|>"),
    "prod": ("prod", "<|prod_begin|>"),
    "ad": ("ad", "<|ad_begin|>"),
    "live": ("living", "<|living_begin|>"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=ADAPTER_DIR")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=ADAPTER_DIR")
    return name, Path(raw_path)


def prompt_of(row: dict[str, Any], domain_token: str) -> str:
    user = row["user"]
    if not user.rstrip().endswith("/no_think"):
        user += "/no_think"
    prompt = ""
    if row.get("system"):
        prompt += f"<|im_start|>system\n{row['system']}<|im_end|>\n"
    prompt += (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
        + domain_token
    )
    return prompt


def load_rows(dev: Path, per_domain: int) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    selected: list[dict[str, Any]] = []
    manifests: list[str] = []
    source_hashes: dict[str, str] = {}
    for label, (file_domain, domain_token) in DOMAINS.items():
        path = dev / f"dev_rec_{label}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[label] = sha256(path)
        candidates = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                key = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                candidates.append((hashlib.sha256(key.encode()).hexdigest(), row))
        if len(candidates) < per_domain:
            raise ValueError(f"{path} has {len(candidates)} rows, need {per_domain}")
        for digest, row in sorted(candidates)[:per_domain]:
            abc = row["gold"]["abc"]
            selected.append(
                {
                    "domain": label,
                    "prompt": prompt_of(row, domain_token),
                    "target_text": "".join(f"<s_{part}_{value}>" for part, value in zip("abc", abc)),
                }
            )
            manifests.append(f"{label}|{file_domain}|{digest}|{'/'.join(abc)}")
    manifest_hash = hashlib.sha256("\n".join(manifests).encode()).hexdigest()
    return selected, manifest_hash, source_hashes


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: list[dict[str, float]]) -> dict[str, float | int]:
    sums = [row["sum_logp"] for row in rows]
    token_means = [row["token_mean_logp"] for row in rows]
    return {
        "n": len(rows),
        "gold_sum_logp_mean": round(mean(sums), 8),
        "gold_sum_logp_median": round(statistics.median(sums), 8),
        "gold_token_logp_mean": round(mean(token_means), 8),
        "all_token_rank_le_32_rate": round(mean([row["all_rank_le_32"] for row in rows]), 8),
        "all_token_rank_le_64_rate": round(mean([row["all_rank_le_64"] for row in rows]), 8),
        "token_rank_mean": round(mean([row["rank_mean"] for row in rows]), 8),
    }


def summarize_delta(rows: list[dict[str, float]]) -> dict[str, float | int]:
    values = [row["sum_logp_delta"] for row in rows]
    return {
        "n": len(rows),
        "gold_sum_logp_delta_mean": round(mean(values), 8),
        "gold_sum_logp_delta_median": round(statistics.median(values), 8),
        "gold_sum_logp_improved_rate": round(mean([value > 0 for value in values]), 8),
        "parent_to_candidate_kl_mean": round(mean([row["kl"] for row in rows]), 10),
        "all_rank_le_32_delta": round(mean([row["rank32_delta"] for row in rows]), 8),
        "all_rank_le_64_delta": round(mean([row["rank64_delta"] for row in rows]), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--per-domain", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    adapters = [("parent", args.parent), *args.candidate]
    names = [name for name, _ in adapters]
    if len(names) != len(set(names)):
        raise ValueError(f"adapter names must be unique: {names}")
    for name, path in adapters:
        if not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"{name}: {path / 'adapter_model.safetensors'}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, manifest_hash, source_hashes = load_rows(args.dev, args.per_domain)
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

    encoded = []
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        target_ids = tokenizer.encode(row["target_text"], add_special_tokens=False)
        if len(target_ids) != 3:
            raise ValueError(f"target is not exactly three SID tokens: {row['target_text']} -> {target_ids}")
        encoded.append({**row, "ids": prompt_ids + target_ids, "targets": target_ids})

    by_model: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    started = time.time()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        for offset in range(0, len(encoded), args.batch_size):
            batch = encoded[offset : offset + args.batch_size]
            maximum = max(len(row["ids"]) for row in batch)
            input_ids = torch.tensor(
                [[pad_id] * (maximum - len(row["ids"])) + row["ids"] for row in batch],
                device="cuda",
            )
            attention_mask = input_ids.ne(pad_id).long()
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)
            prediction_positions = torch.arange(maximum - 4, maximum - 1, device="cuda")
            targets = input_ids[:, -3:]

            logits_by_model = {}
            for name, _ in adapters:
                model.set_adapter(name)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=prediction_positions,
                ).logits.float()
                if logits.shape[:2] != targets.shape:
                    raise RuntimeError(f"partial-logit shape mismatch for {name}: {logits.shape}")
                logits_by_model[name] = logits

            parent_logits = logits_by_model["parent"]
            parent_log_probs = functional.log_softmax(parent_logits, dim=-1)
            parent_probs = parent_log_probs.exp()
            parent_gold = parent_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            parent_ranks = 1 + (parent_logits > parent_logits.gather(-1, targets.unsqueeze(-1))).sum(-1)

            for name, _ in adapters:
                logits = logits_by_model[name]
                log_probs = functional.log_softmax(logits, dim=-1)
                gold = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                ranks = 1 + (logits > logits.gather(-1, targets.unsqueeze(-1))).sum(-1)
                kl = functional.kl_div(log_probs, parent_probs, reduction="none").sum(-1).mean(-1)
                for index, row in enumerate(batch):
                    all32 = float(bool((ranks[index] <= 32).all()))
                    all64 = float(bool((ranks[index] <= 64).all()))
                    parent32 = float(bool((parent_ranks[index] <= 32).all()))
                    parent64 = float(bool((parent_ranks[index] <= 64).all()))
                    by_model[name].append(
                        {
                            "domain": row["domain"],
                            "sum_logp": float(gold[index].sum()),
                            "token_mean_logp": float(gold[index].mean()),
                            "rank_mean": float(ranks[index].float().mean()),
                            "all_rank_le_32": all32,
                            "all_rank_le_64": all64,
                            "sum_logp_delta": float(gold[index].sum() - parent_gold[index].sum()),
                            "kl": float(kl[index]),
                            "rank32_delta": all32 - parent32,
                            "rank64_delta": all64 - parent64,
                        }
                    )
            del logits_by_model, input_ids, attention_mask, position_ids, targets

    report_models = {}
    for name, path in adapters:
        model_rows = by_model[name]
        domains = {
            domain: [row for row in model_rows if row["domain"] == domain]
            for domain in DOMAINS
        }
        target_rows = domains["prod"] + domains["ad"]
        retention_rows = domains["video"] + domains["live"]
        report_models[name] = {
            "adapter": str(path.resolve()),
            "adapter_sha256": sha256(path / "adapter_model.safetensors"),
            "absolute": {
                "by_domain": {domain: summarize(values) for domain, values in domains.items()},
                "target_prod_ad": summarize(target_rows),
                "retention_video_live": summarize(retention_rows),
            },
            "delta_vs_parent": {
                "by_domain": {domain: summarize_delta(values) for domain, values in domains.items()},
                "target_prod_ad": summarize_delta(target_rows),
                "retention_video_live": summarize_delta(retention_rows),
            },
        }

    report = {
        "status": "COMPLETE_NOT_A_SCORE_ESTIMATE",
        "method": {
            "route": "direct /no_think; evaluator-supplied domain marker; three gold SID tokens",
            "per_domain": args.per_domain,
            "batch_size": args.batch_size,
            "sample_manifest_sha256": manifest_hash,
            "stochastic_sampling": False,
            "selection_warning": "Mechanism/retention diagnostic only; do not map to online score.",
        },
        "source": {
            "class": "E",
            "directory": str(args.dev.resolve()),
            "file_sha256": source_hashes,
        },
        "models": report_models,
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
