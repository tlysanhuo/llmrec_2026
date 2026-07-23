#!/usr/bin/env python3
"""Teacher-forced gate for the single I-38M full candidate.

This gate is a mechanism check, not an online score estimator. It requires the
candidate to stay close to I-23 on both material directions and to move toward
I-35 step548 on every non-material task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
I23 = ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
I35 = ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"
HOLDOUT = ROOT / "assets/evaluation/holdout/data_i38_i23_material_i35_teacher_gate_v1.jsonl"

BASE_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
I23_SHA256 = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
I35_SHA256 = "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
HOLDOUT_SHA256 = "311b298f939a953aed7a8a11a694e257518e273a1e506537d76953720eaed41f"

MATERIAL_TASKS = ("material_desc2sid", "material_sid2desc")
RETENTION_TASKS = (
    "action",
    "topic",
    "rec_video",
    "rec_prod",
    "rec_ad",
    "rec_living",
    "world",
)
EXPECTED_TASKS = {
    "material_desc2sid": 128,
    "material_sid2desc": 64,
    "action": 32,
    "topic": 32,
    "rec_video": 32,
    "rec_prod": 32,
    "rec_ad": 32,
    "rec_living": 32,
    "world": 16,
}
CLOSE_THINK_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}
POSITION_CAP = 128
LOGIT_CHUNK = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scored_bounds(tokens: list[int]) -> tuple[int, int]:
    try:
        start = tokens.index(CLOSE_THINK_ID) + 1
    except ValueError:
        start = 0
    while start < len(tokens) and tokens[start] in WHITESPACE_IDS:
        start += 1
    end = len(tokens)
    while end > start and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    if start >= end or tokens[end - 1] != EOS_ID:
        raise RuntimeError("I-38 gate response is empty or missing final EOS")
    return start, end


def sampled_positions(start: int, end: int, cap: int):
    import torch

    count = end - start
    if count <= cap:
        return torch.arange(start, end, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("I-38 gate position cap produced duplicates")
    return relative + start


def forward_kl(candidate, reference) -> float:
    import torch
    import torch.nn.functional as functional

    if candidate.shape != reference.shape:
        raise RuntimeError("I-38 gate KL shape mismatch")
    total = torch.zeros((), device=candidate.device, dtype=torch.float32)
    for start in range(0, candidate.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, candidate.size(1))
        total += functional.kl_div(
            functional.log_softmax(candidate[:, start:end].float(), dim=-1),
            functional.softmax(reference[:, start:end].float(), dim=-1),
            reduction="sum",
        )
    value = float(total / candidate.size(1))
    if not math.isfinite(value):
        raise RuntimeError("non-finite I-38 gate KL")
    return value


def mean_gold_logp(logits, targets) -> float:
    import torch
    import torch.nn.functional as functional

    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, logits.size(1))
        total += functional.log_softmax(logits[:, start:end].float(), dim=-1).gather(
            -1, targets[start:end].view(1, -1, 1)
        ).sum()
    value = float(total / targets.numel())
    if not math.isfinite(value):
        raise RuntimeError("non-finite I-38 gate gold logp")
    return value


def load_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in HOLDOUT.open(encoding="utf-8") if line.strip()]
    counts = Counter(str(row.get("task")) for row in rows)
    if len(rows) != 400 or dict(counts) != EXPECTED_TASKS:
        raise RuntimeError(f"I-38 gate task contract drifted: {len(rows)}/{dict(counts)}")
    return rows


def summarize(rows: list[dict[str, float]], reference_name: str) -> dict[str, Any]:
    tokens = sum(int(row["tokens"]) for row in rows)
    return {
        "rows": len(rows),
        "tokens": tokens,
        f"candidate_to_{reference_name}_kl_mean": statistics.fmean(
            row[f"candidate_to_{reference_name}_kl"] for row in rows
        ),
        f"candidate_{reference_name}_top1_agreement": sum(
            row[f"candidate_{reference_name}_top1_matches"] for row in rows
        ) / tokens,
        f"candidate_gold_logp_delta_vs_{reference_name}": statistics.fmean(
            row[f"candidate_gold_delta_vs_{reference_name}"] for row in rows
        ),
    }


def run_self_test() -> None:
    import torch

    tokens = [151667, 271, CLOSE_THINK_ID, 198, 176251, 151669, 159861, 168053, EOS_ID, 198]
    assert scored_bounds(tokens) == (4, 9)
    torch.manual_seed(38)
    candidate = torch.randn(1, 11, 37)
    reference = torch.randn_like(candidate)
    assert forward_kl(reference, reference) < 1e-6
    assert forward_kl(candidate, reference) > 0
    print("[i38-gate] self-test PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.candidate is None or args.out is None:
        parser.error("--candidate and --out are required")

    locked = {
        BASE / "config.json": BASE_SHA256,
        I23 / "adapter_model.safetensors": I23_SHA256,
        I35 / "adapter_model.safetensors": I35_SHA256,
        HOLDOUT: HOLDOUT_SHA256,
    }
    for path, expected in locked.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"I-38 gate locked artifact missing or drifted: {path}")
    candidate = args.candidate.resolve()
    candidate_weights = candidate / "adapter_model.safetensors"
    candidate_config = candidate / "adapter_config.json"
    if not candidate_weights.is_file() or not candidate_config.is_file():
        raise FileNotFoundError(f"I-38 candidate is not a two-file adapter: {candidate}")
    config = json.loads(candidate_config.read_text(encoding="utf-8"))
    if (int(config.get("r", -1)), int(config.get("lora_alpha", -1))) != (80, 80):
        raise RuntimeError("I-38 gate accepts only the exact I-23 r64 + residual r16 = r80 candidate")
    rows = load_rows()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(
        model, I23, adapter_name="i23", is_trainable=False, low_cpu_mem_usage=True
    ).eval()
    model.load_adapter(I35, adapter_name="i35", is_trainable=False, low_cpu_mem_usage=True)
    model.load_adapter(candidate, adapter_name="candidate", is_trainable=False, low_cpu_mem_usage=True)

    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    started = time.time()
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt_ids, response_ids = template.encode_oneturn(
                tokenizer,
                [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ],
                row["instruction"],
                None,
            )
            body_start, body_end = scored_bounds(response_ids)
            relative = sampled_positions(body_start, body_end, POSITION_CAP)
            input_ids = torch.tensor([prompt_ids + response_ids], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            positions = relative.cuda() + len(prompt_ids) - 1
            targets = torch.tensor(response_ids, device="cuda", dtype=torch.long)[relative.cuda()]
            logits = {}
            for name in ("i23", "i35", "candidate"):
                model.set_adapter(name)
                logits[name] = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=positions,
                ).logits.detach()
            values = {"tokens": float(targets.numel())}
            for reference in ("i23", "i35"):
                values[f"candidate_to_{reference}_kl"] = forward_kl(
                    logits["candidate"], logits[reference]
                )
                values[f"candidate_{reference}_top1_matches"] = float(
                    logits["candidate"].argmax(-1).eq(logits[reference].argmax(-1)).sum()
                )
                values[f"candidate_gold_delta_vs_{reference}"] = (
                    mean_gold_logp(logits["candidate"], targets)
                    - mean_gold_logp(logits[reference], targets)
                )
            values["i23_to_i35_kl"] = forward_kl(logits["i23"], logits["i35"])
            values["i23_i35_top1_matches"] = float(
                logits["i23"].argmax(-1).eq(logits["i35"].argmax(-1)).sum()
            )
            by_task[str(row["task"])].append(values)
            if index % 32 == 0 or index == len(rows):
                print(f"[i38-gate] {index}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)

    material: dict[str, dict[str, Any]] = {}
    for task in MATERIAL_TASKS:
        report = summarize(by_task[task], "i23")
        report["pass"] = (
            report["candidate_to_i23_kl_mean"] <= 0.005
            and report["candidate_i23_top1_agreement"] >= 0.99
            and report["candidate_gold_logp_delta_vs_i23"] >= -0.01
        )
        material[task] = report

    retention: dict[str, dict[str, Any]] = {}
    aggregate_rows: list[dict[str, float]] = []
    for task in RETENTION_TASKS:
        rows_for_task = by_task[task]
        aggregate_rows.extend(rows_for_task)
        report = summarize(rows_for_task, "i35")
        baseline_kl = statistics.fmean(row["i23_to_i35_kl"] for row in rows_for_task)
        baseline_matches = sum(row["i23_i35_top1_matches"] for row in rows_for_task)
        tokens = report["tokens"]
        report.update(
            {
                "i23_to_i35_kl_mean": baseline_kl,
                "i23_i35_top1_agreement": baseline_matches / tokens,
                "teacher_kl_ratio_vs_i23": (
                    report["candidate_to_i35_kl_mean"] / baseline_kl if baseline_kl > 0 else 0.0
                ),
            }
        )
        report["pass"] = (
            report["candidate_to_i35_kl_mean"] < baseline_kl
            and report["candidate_i35_top1_agreement"] >= report["i23_i35_top1_agreement"]
        )
        retention[task] = report
    aggregate = summarize(aggregate_rows, "i35")
    baseline_aggregate = statistics.fmean(row["i23_to_i35_kl"] for row in aggregate_rows)
    aggregate["i23_to_i35_kl_mean"] = baseline_aggregate
    aggregate["teacher_kl_ratio_vs_i23"] = (
        aggregate["candidate_to_i35_kl_mean"] / baseline_aggregate
        if baseline_aggregate > 0 else 0.0
    )
    aggregate["pass"] = aggregate["teacher_kl_ratio_vs_i23"] <= 0.90

    material_pass = all(report["pass"] for report in material.values())
    retention_pass = aggregate["pass"] and all(report["pass"] for report in retention.values())
    report = {
        "status": "COMPLETE_NOT_AN_ONLINE_SCORE_ESTIMATE",
        "candidate": {
            "path": str(candidate),
            "adapter_sha256": sha256(candidate_weights),
            "adapter_config_sha256": sha256(candidate_config),
            "rank_alpha": [80, 80],
        },
        "anchors": {
            "material_i23_adapter_sha256": I23_SHA256,
            "retention_i35_adapter_sha256": I35_SHA256,
        },
        "holdout": {"path": str(HOLDOUT), "rows": len(rows), "sha256": HOLDOUT_SHA256},
        "material": material,
        "retention": retention,
        "retention_aggregate": aggregate,
        "teacher_forced_pass": material_pass and retention_pass,
        "next_required_gate": "itemic breakage 0/60, then package audit; no checkpoint or scale search",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
