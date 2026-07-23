#!/usr/bin/env python3
"""Evaluate I-32 task-restoration candidates on the frozen acceptance set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT = ROOT / "submissions/i19_world_external_r96_s875_platform"
TEACHER = ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
HOLDOUT = ROOT / "assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl"
PARENT_SHA = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
TEACHER_SHA = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
HOLDOUT_SHA = "f75106758792163dd33d1d52639ba507a6d9e69094d8213d5f3b0969ee272f62"
BASE_SHA = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
MATERIAL_TASKS = ("material_desc2sid", "material_sid2desc")
RETENTION_TASKS = ("action", "topic", "rec_video", "rec_prod", "rec_ad", "rec_living", "world")
EXPECTED_COUNTS = {
    **{task: 128 for task in MATERIAL_TASKS},
    **{task: 64 for task in RETENTION_TASKS if task != "world"},
    "world": 16,
}
IM_END = "<|im_end|>"
CLOSE_THINK_ID = 151668
WHITESPACE_IDS = {198, 220, 262, 271}
KL_CHUNK = 8
RETENTION_CAP = 128
ANSWER_PATTERNS = (
    re.compile(r"正确答案是\s*[（(]?\s*([A-I]+)", re.I),
    re.compile(r"最终答案是\s*[（(]?\s*([A-I]+)", re.I),
    re.compile(r"(?:correct answer is|answer is)\s*[（(]?\s*([A-I]+)", re.I),
    re.compile(r"选\s*([A-I])(?:[。.!！]|$)", re.I),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    return name, Path(path)


def render(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("history"):
        raise RuntimeError("I-32 gate rows unexpectedly contain history")
    user = "\n".join(value for value in (row["instruction"], row["input"]) if value)
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        f"{row['output']}{IM_END}\n",
    )


def response_body(ids: list[int], material: bool) -> tuple[int, int]:
    if material:
        try:
            start = ids.index(CLOSE_THINK_ID) + 1
        except ValueError as error:
            raise RuntimeError("material gate response lacks </think>") from error
        while start < len(ids) and ids[start] in WHITESPACE_IDS:
            start += 1
    else:
        if CLOSE_THINK_ID in ids:
            start = ids.index(CLOSE_THINK_ID) + 1
            while start < len(ids) and ids[start] in WHITESPACE_IDS:
                start += 1
        else:
            start = 0
    end = len(ids)
    while end > start and ids[end - 1] in WHITESPACE_IDS:
        end -= 1
    if start >= end:
        raise RuntimeError("empty I-32 gate response body")
    return start, end


def sampled_relative_positions(start: int, end: int, cap: int):
    import torch

    count = end - start
    if count <= cap:
        return torch.arange(start, end, dtype=torch.long)
    values = torch.linspace(0, count - 1, steps=cap).round().long().unique()
    if values.numel() != cap:
        raise RuntimeError("I-32 gate cap produced duplicate positions")
    return values + start


def forward_kl(policy, reference) -> float:
    import torch
    import torch.nn.functional as functional

    total = torch.zeros((), device=policy.device, dtype=torch.float32)
    for start in range(0, policy.size(1), KL_CHUNK):
        end = min(start + KL_CHUNK, policy.size(1))
        total += functional.kl_div(
            functional.log_softmax(policy[:, start:end].float(), dim=-1),
            functional.softmax(reference[:, start:end].float(), dim=-1),
            reduction="sum",
        )
    value = float(total / policy.size(1))
    if not math.isfinite(value):
        raise RuntimeError("non-finite I-32 gate KL")
    return value


def mean_gold_logp(logits, targets) -> float:
    import torch
    import torch.nn.functional as functional

    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, logits.size(1), KL_CHUNK):
        end = min(start + KL_CHUNK, logits.size(1))
        total += functional.log_softmax(logits[:, start:end].float(), dim=-1).gather(
            -1, targets[start:end].view(1, -1, 1)
        ).sum()
    value = float(total / targets.numel())
    if not math.isfinite(value):
        raise RuntimeError("non-finite I-32 gate gold logp")
    return value


def extract_answer(output: str) -> tuple[str, int, int]:
    matches = []
    for pattern in ANSWER_PATTERNS:
        matches.extend(pattern.finditer(output))
    if not matches:
        raise RuntimeError(f"unparsed I-32 world answer: {output[-160:]!r}")
    match = max(matches, key=lambda item: item.start(1))
    answer = match.group(1).upper()
    return answer, match.start(1), match.end(1)


def summarize_material(rows: list[dict[str, float]]) -> dict[str, Any]:
    deltas = [row["gold_delta"] for row in rows]
    teacher_deltas = [row["teacher_kl_delta"] for row in rows]
    return {
        "n": len(rows),
        "gold_mean_logp_delta": statistics.fmean(deltas),
        "gold_improved_rate": statistics.fmean(value > 0 for value in deltas),
        "teacher_kl_delta_mean": statistics.fmean(teacher_deltas),
        "pass": statistics.fmean(deltas) >= 0 and statistics.fmean(value > 0 for value in deltas) >= 0.55 and statistics.fmean(teacher_deltas) <= 0,
    }


def summarize_retention(rows: list[dict[str, float]]) -> dict[str, Any]:
    token_count = sum(int(row["tokens"]) for row in rows)
    return {
        "n": len(rows),
        "tokens": token_count,
        "top1_agreement": sum(row["top1_matches"] for row in rows) / token_count,
        "parent_to_candidate_kl_mean": statistics.fmean(row["parent_kl"] for row in rows),
        "gold_mean_logp_delta": statistics.fmean(row["gold_delta"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--expected-rank", type=int, default=168)
    args = parser.parse_args()
    if args.expected_rank < 1:
        raise ValueError("expected rank must be positive")
    names = [name for name, _ in args.candidate]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate name")
    required = {
        BASE / "config.json": BASE_SHA,
        PARENT / "adapter_model.safetensors": PARENT_SHA,
        TEACHER / "adapter_model.safetensors": TEACHER_SHA,
        HOLDOUT: HOLDOUT_SHA,
    }
    for path, expected in required.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"locked I-32 gate artifact missing or drifted: {path}")
    for name, path in args.candidate:
        if not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"{name}: {path}")
        config = json.loads((path / "adapter_config.json").read_text())
        if (int(config["r"]), int(config["lora_alpha"])) != (
            args.expected_rank,
            args.expected_rank,
        ):
            raise RuntimeError(
                f"{name} is not exact r{args.expected_rank}/alpha{args.expected_rank}"
            )

    rows = [json.loads(line) for line in HOLDOUT.open() if line.strip()]
    if len(rows) != sum(EXPECTED_COUNTS.values()) or Counter(row["task"] for row in rows) != EXPECTED_COUNTS:
        raise RuntimeError("I-32 holdout row/task contract drifted")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    model = PeftModel.from_pretrained(model, PARENT, adapter_name="parent", is_trainable=False).eval()
    model.load_adapter(TEACHER, adapter_name="teacher", is_trainable=False, low_cpu_mem_usage=True)
    adapters = [("parent", PARENT), ("teacher", TEACHER)]
    for name, path in args.candidate:
        model.load_adapter(path, adapter_name=name, is_trainable=False, low_cpu_mem_usage=True)
        adapters.append((name, path))

    by_candidate: dict[str, dict[str, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in names
    }
    world_exact = {name: {"correct": 0, "rows": 0} for name, _ in adapters if name != "teacher"}
    started = time.time()
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prompt, response = render(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            body_start, body_end = response_body(response_ids, row["task"] in MATERIAL_TASKS)
            relative = sampled_relative_positions(
                body_start,
                body_end,
                body_end - body_start if row["task"] in MATERIAL_TASKS else RETENTION_CAP,
            )
            positions = relative.cuda() + len(prompt_ids) - 1
            targets = torch.tensor(response_ids, device="cuda", dtype=torch.long)[relative.cuda()]
            input_ids = torch.tensor([prompt_ids + response_ids], device="cuda", dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            logits = {}
            for name, _ in adapters:
                model.set_adapter(name)
                logits[name] = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=positions,
                ).logits.detach()
            parent_gold = mean_gold_logp(logits["parent"], targets)
            parent_top1 = logits["parent"].argmax(-1).squeeze(0)
            parent_teacher_kl = (
                forward_kl(logits["parent"], logits["teacher"])
                if row["task"] in MATERIAL_TASKS
                else 0.0
            )
            for name in names:
                candidate_gold = mean_gold_logp(logits[name], targets)
                values = {
                    "tokens": float(targets.numel()),
                    "gold_delta": candidate_gold - parent_gold,
                    "top1_matches": float(logits[name].argmax(-1).squeeze(0).eq(parent_top1).sum()),
                    "parent_kl": forward_kl(logits[name], logits["parent"]),
                    "teacher_kl_delta": (
                        forward_kl(logits[name], logits["teacher"]) - parent_teacher_kl
                        if row["task"] in MATERIAL_TASKS
                        else 0.0
                    ),
                }
                by_candidate[name][row["task"]].append(values)

            if row["task"] == "world":
                answer, answer_start, answer_end = extract_answer(row["output"])
                if row["output"][answer_start:answer_end].upper() != answer:
                    raise RuntimeError("I-32 world answer extraction span drifted")
                exact_text = prompt + row["output"][:answer_end]
                encoded_exact = tokenizer(
                    exact_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                exact_ids = encoded_exact["input_ids"]
                offsets = encoded_exact["offset_mapping"]
                global_start = len(prompt) + answer_start
                global_end = len(prompt) + answer_end
                target_indexes = [
                    index
                    for index, (start, end) in enumerate(offsets)
                    if end > global_start and start < global_end
                ]
                if not target_indexes or target_indexes != list(
                    range(target_indexes[0], target_indexes[-1] + 1)
                ):
                    raise RuntimeError("I-32 world answer token span is empty or disjoint")
                if target_indexes[0] == 0:
                    raise RuntimeError("I-32 world answer starts at token zero")
                exact_input = torch.tensor([exact_ids], device="cuda")
                exact_mask = torch.ones_like(exact_input)
                exact_positions = torch.tensor(
                    [index - 1 for index in target_indexes], device="cuda"
                )
                exact_targets = torch.tensor(
                    [exact_ids[index] for index in target_indexes], device="cuda"
                )
                for name, _ in adapters:
                    if name == "teacher":
                        continue
                    model.set_adapter(name)
                    exact_logits = model(
                        input_ids=exact_input,
                        attention_mask=exact_mask,
                        use_cache=False,
                        logits_to_keep=exact_positions,
                    ).logits
                    world_exact[name]["rows"] += 1
                    world_exact[name]["correct"] += int(
                        bool(exact_logits.argmax(-1).squeeze(0).eq(exact_targets).all())
                    )
            if index % 32 == 0 or index == len(rows):
                print(f"[i32-gate] {index}/{len(rows)} elapsed={time.time()-started:.1f}s", flush=True)
            del input_ids, attention_mask, logits, positions, targets

    reports = {}
    parent_world = world_exact["parent"]["correct"]
    for name, path in args.candidate:
        material = {task: summarize_material(by_candidate[name][task]) for task in MATERIAL_TASKS}
        retention = {task: summarize_retention(by_candidate[name][task]) for task in RETENTION_TASKS}
        retention_pass = all(
            values["top1_agreement"] >= 0.99
            and values["parent_to_candidate_kl_mean"] <= 0.005
            and values["gold_mean_logp_delta"] >= -0.01
            for values in retention.values()
        )
        world_pass = (
            world_exact[name]["rows"] == EXPECTED_COUNTS["world"]
            and world_exact[name]["correct"] >= parent_world
        )
        reports[name] = {
            "path": str(path.resolve()),
            "adapter_sha256": sha256(path / "adapter_model.safetensors"),
            "material": material,
            "retention": retention,
            "world_exact": {
                "rows": world_exact[name]["rows"],
                "parent_correct": parent_world,
                "candidate_correct": world_exact[name]["correct"],
                "pass": world_pass,
            },
            "teacher_forced_pass": all(values["pass"] for values in material.values()) and retention_pass and world_pass,
        }
    eligible = [name for name in names if reports[name]["teacher_forced_pass"]]
    report = {
        "status": "COMPLETE_NOT_AN_ONLINE_SCORE_ESTIMATE",
        "holdout": {"path": str(HOLDOUT.resolve()), "sha256": sha256(HOLDOUT), "rows": len(rows)},
        "parent": {"path": str(PARENT.resolve()), "adapter_sha256": PARENT_SHA, "world_exact_correct": parent_world},
        "teacher": {"path": str(TEACHER.resolve()), "adapter_sha256": TEACHER_SHA},
        "candidate_order": names,
        "expected_rank_and_alpha": args.expected_rank,
        "models": reports,
        "earliest_teacher_forced_pass": eligible[0] if eligible else None,
        "next_required_gate": "itemic breakage 0/60 for each teacher-forced-pass candidate in order",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
