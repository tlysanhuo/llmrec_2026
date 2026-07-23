#!/usr/bin/env python3
"""Build I-30 material-teacher rows and a disjoint material gate.

The construction filter compares the frozen I-23 and I19-world r96 adapters
on deterministic O1 material candidates.  It selects rows where I-23 assigns
higher mean log-probability to the answer body.  This is a construction
filter, not an online-score estimator.
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
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
SOURCE = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
RETENTION_SOURCE = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
PARENT = ROOT / "submissions/i19_world_external_r96_s875_platform"
TEACHER = ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
OUTPUT = ROOT / "assets/derived/processed/data_i30_r96_material_teacher_retkl_v1.jsonl"
HOLDOUT = ROOT / "assets/evaluation/holdout/data_i30_r96_material_teacher_gate_v1.jsonl"
LEDGER = ROOT / "logs/data/i30_r96_material_teacher_selection_v1.jsonl"
AUDIT = ROOT / "logs/data/i30_r96_material_teacher_retkl_v1_audit.json"

SOURCE_SHA256 = "13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f"
RETENTION_SHA256 = "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0"
PARENT_SHA256 = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
TEACHER_SHA256 = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
BASE_CONFIG_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"

MATERIAL_TASKS = ("material_desc2sid", "material_sid2desc")
GATE_PER_TASK = 128
CANDIDATE_POOL_PER_TASK = 1024
TRAIN_PER_TASK = 256
RETENTION_QUOTAS = {
    "action": 235,
    "topic": 234,
    "rec_video": 234,
    "rec_prod": 234,
    "rec_ad": 234,
    "rec_living": 234,
    "world": 131,
}
RETENTION_GATE_PER_TASK = 64
SEED = 19260831
IM_END = "<|im_end|>"
CLOSE_THINK_ID = 151668
WHITESPACE_IDS = {198, 220, 262, 271}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_helpers():
    helper_path = ROOT / "scripts/data/build_seed_scoremax_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_i30_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper.load_jsonl, helper.task_of


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", "") or ""),
        "input": str(row.get("input", "") or ""),
        "output": str(row.get("output", "") or ""),
        "history": row.get("history") or [],
    }


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(normalized_row(row)).encode()).hexdigest()


def prompt_digest(row: dict[str, Any]) -> str:
    payload = [row.get("instruction", ""), row.get("input", ""), row.get("history") or []]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def stable_digest(seed: int, namespace: str, task: str, row: dict[str, Any]) -> str:
    payload = [seed, namespace, task, row_digest(row)]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def classify(row: dict[str, Any], task_of) -> str:
    try:
        return task_of(row)
    except ValueError:
        # The registered I-12 retention source contains exactly 231 O2.General
        # world rows spanning several legacy answer renderers, including 41
        # without ChatML think tags.  All O1 task rows classify above.
        if "<s_a_" not in canonical(normalized_row(row)):
            return "world"
        raise


def unique_prompt_rows(
    rows: list[dict[str, Any]], *, reject_conflicting_targets: bool = True
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[prompt_digest(row)].append(row)
    selected = []
    for digest, values in grouped.items():
        targets = {str(row["output"]) for row in values}
        if reject_conflicting_targets and len(targets) != 1:
            raise RuntimeError(f"prompt {digest} has {len(targets)} conflicting targets")
        selected.append(min(values, key=row_digest))
    return selected


def split_material(rows: list[dict[str, Any]], task_of, seed: int):
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = classify(row, task_of)
        if task in MATERIAL_TASKS:
            buckets[task].append(normalized_row(row))

    pools: dict[str, list[dict[str, Any]]] = {}
    gates: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for task in MATERIAL_TASKS:
        unique = unique_prompt_rows(buckets[task])
        ranked = sorted(unique, key=lambda row: stable_digest(seed, "i30-material-split", task, row))
        needed = GATE_PER_TASK + CANDIDATE_POOL_PER_TASK
        if len(ranked) < needed:
            raise RuntimeError(f"{task}: only {len(ranked)} unique prompts, need {needed}")
        gate = ranked[:GATE_PER_TASK]
        pool = ranked[GATE_PER_TASK:needed]
        gates.extend({**row, "route": "gate_only", "task": task} for row in gate)
        pools[task] = pool
        audit[task] = {
            "source_rows": len(buckets[task]),
            "unique_prompt_rows": len(unique),
            "gate_rows": len(gate),
            "candidate_pool_rows": len(pool),
            "gate_prompt_manifest_sha256": hashlib.sha256(
                "\n".join(prompt_digest(row) for row in gate).encode()
            ).hexdigest(),
            "candidate_prompt_manifest_sha256": hashlib.sha256(
                "\n".join(prompt_digest(row) for row in pool).encode()
            ).hexdigest(),
        }
    if {prompt_digest(row) for row in gates} & {
        prompt_digest(row) for task in MATERIAL_TASKS for row in pools[task]
    }:
        raise RuntimeError("material gate and construction pools overlap by prompt")
    return pools, gates, audit


def render(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("history"):
        raise RuntimeError("I-30 material construction expects empty history")
    user = "\n".join(value for value in (row["instruction"], row["input"]) if value)
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return prompt, f"{row['output']}{IM_END}\n"


def body_slice(target_ids: list[int]) -> tuple[int, int]:
    try:
        start = target_ids.index(CLOSE_THINK_ID) + 1
    except ValueError as error:
        raise RuntimeError("material response is missing </think>") from error
    while start < len(target_ids) and target_ids[start] in WHITESPACE_IDS:
        start += 1
    end = len(target_ids)
    while end > start and target_ids[end - 1] in WHITESPACE_IDS:
        end -= 1
    if start >= end:
        raise RuntimeError("material response has an empty answer body")
    return start, end


def score_pools(pools, gpu: str, cutoff: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(model, PARENT, adapter_name="parent", is_trainable=False).eval()
    model.load_adapter(TEACHER, adapter_name="teacher", is_trainable=False, low_cpu_mem_usage=True)

    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = sum(len(values) for values in pools.values())
    completed = 0
    started = time.time()
    with torch.inference_mode():
        for task in MATERIAL_TASKS:
            for row in pools[task]:
                prompt, response = render(row)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                start, end = body_slice(response_ids)
                all_ids = prompt_ids + response_ids
                if len(all_ids) > cutoff:
                    raise RuntimeError(f"{task}/{row_digest(row)} exceeds cutoff: {len(all_ids)}>{cutoff}")
                positions = torch.arange(
                    len(prompt_ids) + start - 1,
                    len(prompt_ids) + end - 1,
                    device="cuda",
                    dtype=torch.long,
                )
                targets = torch.tensor(response_ids[start:end], device="cuda", dtype=torch.long)
                input_ids = torch.tensor([all_ids], device="cuda", dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                scores = {}
                top1 = {}
                for name in ("parent", "teacher"):
                    model.set_adapter(name)
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=positions,
                    ).logits.float()
                    if logits.size(1) != targets.numel():
                        raise RuntimeError(f"{name} partial logits mismatch")
                    gold = functional.log_softmax(logits, dim=-1).gather(
                        -1, targets.view(1, -1, 1)
                    ).squeeze()
                    scores[name] = float(gold.mean())
                    top1[name] = float(logits.argmax(-1).squeeze(0).eq(targets).float().mean())
                    del logits, gold
                delta = scores["teacher"] - scores["parent"]
                if not all(math.isfinite(value) for value in (*scores.values(), delta)):
                    raise RuntimeError("non-finite I-30 construction score")
                scored[task].append(
                    {
                        "row": row,
                        "task": task,
                        "row_sha256": row_digest(row),
                        "prompt_sha256": prompt_digest(row),
                        "answer_tokens": int(targets.numel()),
                        "parent_mean_logp": scores["parent"],
                        "teacher_mean_logp": scores["teacher"],
                        "teacher_minus_parent_mean_logp": delta,
                        "parent_gold_top1_rate": top1["parent"],
                        "teacher_gold_top1_rate": top1["teacher"],
                    }
                )
                completed += 1
                if completed % 32 == 0 or completed == total:
                    print(f"[i30-build] scored {completed}/{total} elapsed={time.time()-started:.1f}s", flush=True)
                del input_ids, attention_mask, positions, targets
    return scored, tokenizer


def select_training(scored):
    selected = []
    selection_audit = {}
    for task in MATERIAL_TASKS:
        positive = [row for row in scored[task] if row["teacher_minus_parent_mean_logp"] > 0]
        ranked = sorted(
            positive,
            key=lambda row: (-row["teacher_minus_parent_mean_logp"], row["row_sha256"]),
        )
        if len(ranked) < TRAIN_PER_TASK:
            raise RuntimeError(f"{task}: only {len(ranked)} positive teacher-advantage rows; need {TRAIN_PER_TASK}")
        chosen = ranked[:TRAIN_PER_TASK]
        selected.extend({**row["row"], "route": "material_teacher", "task": task} for row in chosen)
        deltas = [row["teacher_minus_parent_mean_logp"] for row in chosen]
        selection_audit[task] = {
            "positive_advantage_rows": len(positive),
            "selected": len(chosen),
            "selected_delta_mean": sum(deltas) / len(deltas),
            "selected_delta_min": min(deltas),
            "selected_delta_max": max(deltas),
            "selected_manifest_sha256": hashlib.sha256(
                "\n".join(row["row_sha256"] for row in chosen).encode()
            ).hexdigest(),
        }
    return selected, selection_audit


def select_retention(rows, task_of, seed: int):
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = classify(row, task_of)
        if task in RETENTION_QUOTAS:
            buckets[task].append(normalized_row(row))
    ranked_by_task = {}
    for task, wanted in RETENTION_QUOTAS.items():
        unique = unique_prompt_rows(buckets[task], reject_conflicting_targets=False)
        ranked_by_task[task] = sorted(
            unique, key=lambda row: stable_digest(seed, "i30-retention", task, row)
        )
        if len(ranked_by_task[task]) < wanted + RETENTION_GATE_PER_TASK:
            raise RuntimeError(
                f"{task}: only {len(ranked_by_task[task])} unique retention prompts, "
                f"need {wanted + RETENTION_GATE_PER_TASK}"
            )

    selected = []
    chosen_by_task = {}
    train_prompts: set[str] = set()
    for task, wanted in RETENTION_QUOTAS.items():
        chosen = []
        for row in ranked_by_task[task]:
            digest = prompt_digest(row)
            if digest in train_prompts:
                continue
            chosen.append(row)
            train_prompts.add(digest)
            if len(chosen) == wanted:
                break
        if len(chosen) != wanted:
            raise RuntimeError(f"{task}: could not allocate {wanted} globally unique train prompts")
        chosen_by_task[task] = chosen
        selected.extend({**row, "route": "retention_kl", "task": task} for row in chosen)

    gate = []
    gate_prompts: set[str] = set()
    audit = {}
    for task, wanted in RETENTION_QUOTAS.items():
        heldout = []
        for row in ranked_by_task[task]:
            digest = prompt_digest(row)
            if digest in train_prompts or digest in gate_prompts:
                continue
            heldout.append(row)
            gate_prompts.add(digest)
            if len(heldout) == RETENTION_GATE_PER_TASK:
                break
        if len(heldout) != RETENTION_GATE_PER_TASK:
            raise RuntimeError(
                f"{task}: could not allocate {RETENTION_GATE_PER_TASK} globally disjoint gate prompts"
            )
        gate.extend({**row, "route": "gate_only", "task": task} for row in heldout)
        audit[task] = {
            "available_unique_prompts": len(ranked_by_task[task]),
            "selected": len(chosen_by_task[task]),
            "gate": len(heldout),
        }
    if len(selected) != sum(RETENTION_QUOTAS.values()):
        raise RuntimeError("retention total drifted")
    return selected, gate, audit


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cutoff", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        sample = [151667, 271, CLOSE_THINK_ID, 198, 176251, 151669, 159861, 168053, 151645, 198]
        assert body_slice(sample) == (4, 9)
        assert sum(RETENTION_QUOTAS.values()) == 1536
        print("[i30-build] self-test passed")
        return

    required = {
        SOURCE: SOURCE_SHA256,
        RETENTION_SOURCE: RETENTION_SHA256,
        PARENT / "adapter_model.safetensors": PARENT_SHA256,
        TEACHER / "adapter_model.safetensors": TEACHER_SHA256,
        BASE / "config.json": BASE_CONFIG_SHA256,
    }
    for path, expected in required.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"locked artifact drift: {path} {observed}/{expected}")

    load_jsonl, task_of = load_helpers()
    source_rows = load_jsonl(SOURCE)
    retention_rows = load_jsonl(RETENTION_SOURCE)
    pools, material_gate_rows, split_audit = split_material(source_rows, task_of, args.seed)
    scored, tokenizer = score_pools(pools, args.gpu, args.cutoff)
    material_rows, selection_audit = select_training(scored)
    retention_selected, retention_gate_rows, retention_audit = select_retention(
        retention_rows, task_of, args.seed
    )
    gate_rows = material_gate_rows + retention_gate_rows
    training_rows = material_rows + retention_selected
    random.Random(args.seed).shuffle(training_rows)

    train_prompts = {prompt_digest(row) for row in training_rows}
    gate_prompts = {prompt_digest(row) for row in gate_rows}
    if train_prompts & gate_prompts:
        raise RuntimeError("selected material training and gate prompts overlap")
    if Counter(row["route"] for row in training_rows) != {
        "material_teacher": 512,
        "retention_kl": 1536,
    }:
        raise RuntimeError("final I-30 route signature drifted")

    ledger_rows = []
    for task in MATERIAL_TASKS:
        for row in sorted(scored[task], key=lambda item: item["row_sha256"]):
            ledger_rows.append({key: value for key, value in row.items() if key != "row"})
    atomic_jsonl(OUTPUT, training_rows)
    atomic_jsonl(HOLDOUT, gate_rows)
    atomic_jsonl(LEDGER, ledger_rows)

    audit = {
        "schema": "i30-r96-material-teacher-retkl-v1",
        "asset_class": "D(O1,O2.*; M-I23/I19-world construction filter)",
        "seed": args.seed,
        "builder": str(Path(__file__).resolve()),
        "upstream": {
            "material_source": {"path": str(SOURCE.resolve()), "rows": len(source_rows), "sha256": sha256(SOURCE)},
            "retention_source": {"path": str(RETENTION_SOURCE.resolve()), "rows": len(retention_rows), "sha256": sha256(RETENTION_SOURCE)},
            "base_config_sha256": sha256(BASE / "config.json"),
            "parent_adapter": {"path": str(PARENT.resolve()), "sha256": sha256(PARENT / "adapter_model.safetensors")},
            "teacher_adapter": {"path": str(TEACHER.resolve()), "sha256": sha256(TEACHER / "adapter_model.safetensors")},
        },
        "construction_filter": {
            "metric": "I23 minus r96 teacher-forced answer-body gold token mean-logp",
            "candidate_pool_per_material_direction": CANDIDATE_POOL_PER_TASK,
            "selection_rule": "top 256 strictly-positive rows per direction; ties by row SHA256",
            "warning": "construction filter only; not an online-score estimate",
            "split": split_audit,
            "selection": selection_audit,
            "ledger": str(LEDGER.resolve()),
            "ledger_rows": len(ledger_rows),
            "ledger_sha256": sha256(LEDGER),
        },
        "training": {
            "path": str(OUTPUT.resolve()),
            "rows": len(training_rows),
            "sha256": sha256(OUTPUT),
            "mix": {
                "material_teacher_rows": len(material_rows),
                "material_by_direction": {task: TRAIN_PER_TASK for task in MATERIAL_TASKS},
                "retention_kl_rows": len(retention_selected),
                "retention_by_task": RETENTION_QUOTAS,
                "material_to_retention": "1:3",
                "O1_or_O2_derived_rows": len(training_rows),
                "T_rows": 0,
                "E_rows": 0,
            },
        },
        "gate": {
            "path": str(HOLDOUT.resolve()),
            "rows": len(gate_rows),
            "material_per_direction": GATE_PER_TASK,
            "retention_per_task": RETENTION_GATE_PER_TASK,
            "task_counts": dict(Counter(row["task"] for row in gate_rows)),
            "sha256": sha256(HOLDOUT),
            "prompt_overlap_with_all_training": 0,
            "training_use": False,
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "cutoff": args.cutoff,
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(AUDIT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
