#!/usr/bin/env python3
"""Filter the I-18 CoT repairs by answer utility and build I-23.

The existing I-18 generations are construction inputs, not regenerated here.
``score`` freezes I-10 E3 and measures only the known final-answer tokens for
every unique gold in each of the 538 affected prompt groups.  ``build`` then
patches the I-10 training parent only for groups that pass the preregistered
domain-aware rule.  No evaluation prompt or third-party row is used.

This is a construction filter, not an offline competition-score estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
DEFAULT_ADAPTER = ROOT / "checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1995"
DEFAULT_PARENT = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
DEFAULT_O2_TEACHER = ROOT / "assets/derived/processed/action_distill_v5.jsonl"
DEFAULT_REQUESTS = ROOT / "logs/data/cotfix_v2_requests.jsonl"
DEFAULT_GENERATIONS = ROOT / "logs/data/cotfix_v2_generations.jsonl"
DEFAULT_LEDGER = ROOT / "logs/data/cotfix_v3_answer_utility.jsonl"
DEFAULT_SCORE_SUMMARY = ROOT / "logs/data/cotfix_v3_answer_utility_summary.json"
DEFAULT_OUTPUT = ROOT / "assets/derived/processed/data_seed_teacher_cotfix_v3.jsonl"
DEFAULT_BUILD_AUDIT = ROOT / "logs/data/seed_teacher_cotfix_v3_audit.json"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
ITEM_RE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|>"
    r"<s_a_\d+><s_b_\d+><s_c_\d+>"
)

# Frozen before the valid full 538-group audit.  An initial batched dry run was
# discarded after batch-composition sensitivity was observed; formal scoring
# is batch-one, and this guard excludes sub-material bf16 boundary changes.
POSITIVE_EPS = 5.0e-3
NONVIDEO_MIN_WIN_RATE = 0.50
VIDEO_MIN_WIN_RATE = 0.75
MIN_SELECTED_TOTAL = 80
MIN_SELECTED_NONVIDEO = 48
MIN_SELECTED_DOMAINS = 3
MIN_SELECTED_PER_COUNTED_DOMAIN = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def latest_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in read_jsonl(path)}


def prompt_core(row: dict[str, Any]) -> str:
    return MODE_SUFFIX_RE.sub("", str(row.get("input") or "").rstrip())


def split_output(output: str) -> tuple[str, str, str]:
    """Return thought text, exact answer-leading context, and answer body."""
    match = THINK_RE.search(output)
    if match is None:
        raise ValueError("assistant output lacks a think block")
    suffix = output[match.end() :]
    answer = suffix.lstrip()
    separator = suffix[: len(suffix) - len(answer)]
    context_shell = output[: match.start(1)] + "{thought}" + output[match.end(1) : match.end()]
    return match.group(1).strip(), context_shell + separator, answer


def answer_domain(answer: str) -> str:
    matches = ITEM_RE.findall(answer)
    if not matches:
        raise ValueError(f"recommendation answer has no item token: {answer[:120]!r}")
    return matches[-1]


def qwen_prompt(row: dict[str, Any], input_text: str) -> str:
    """Render exactly the registered qwen3_nothink user-side template."""
    rendered: list[str] = []
    for old_prompt, old_response in row.get("history") or []:
        rendered.append(
            "<|im_start|>user\n"
            + str(old_prompt)
            + "<|im_end|>\n<|im_start|>assistant\n"
            + str(old_response)
            + "<|im_end|>\n"
        )
    query = "\n".join(
        value for value in (str(row.get("instruction") or ""), input_text) if value
    )
    rendered.append(
        "<|im_start|>user\n"
        + query
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    return "".join(rendered)


def natural_nothink_input(input_text: str) -> str:
    stripped = input_text.rstrip()
    if not stripped.endswith("/think"):
        raise ValueError(f"retained CoT input is not /think: {input_text[-80:]!r}")
    return MODE_SUFFIX_RE.sub("/no_think", stripped)


def collect_groups(
    parent_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(parent_rows) != 32_644:
        raise AssertionError(f"expected 32,644 I-10 rows, got {len(parent_rows)}")
    if len(requests) != 538 or len({row["candidate_id"] for row in requests}) != 538:
        raise AssertionError("expected 538 unique registered CoT-fix requests")
    by_core = {str(row["prompt_core"]): row for row in requests}
    if len(by_core) != 538:
        raise AssertionError("request prompt cores are not unique")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(parent_rows):
        core = prompt_core(row)
        if core in by_core:
            grouped[core].append((index, row))
    if set(grouped) != set(by_core):
        raise AssertionError(f"parent aligned only {len(grouped)}/538 request groups")

    result: list[dict[str, Any]] = []
    for core, request in by_core.items():
        candidate = str(request["candidate_id"])
        generation = generations.get(candidate)
        if generation is None or generation.get("status") != "accepted":
            raise AssertionError(f"{candidate}: missing accepted I-18 generation")
        if generation.get("verdict") != "TRUNCATED" or not generation.get("continuation"):
            raise AssertionError(f"{candidate}: expected a non-empty TRUNCATED continuation")

        members = grouped[core]
        instructions = {str(row.get("instruction") or "") for _, row in members}
        histories = {
            json.dumps(row.get("history") or [], ensure_ascii=False, sort_keys=True)
            for _, row in members
        }
        if len(instructions) != 1 or len(histories) != 1:
            raise AssertionError(f"{candidate}: group prompt metadata drifted")

        retained: list[tuple[int, dict[str, Any], str, str]] = []
        golds: list[dict[str, Any]] = []
        seen_answers: set[str] = set()
        for index, row in members:
            thought, shell, answer = split_output(str(row["output"]))
            if "该用户最近" not in answer:
                raise AssertionError(f"{candidate}: aligned non-recommendation row")
            if answer not in seen_answers:
                golds.append(
                    {
                        "answer": answer,
                        "answer_sha256": text_sha256(answer),
                        "domain": answer_domain(answer),
                        "source_parent_index": index,
                    }
                )
                seen_answers.add(answer)
            if thought == str(request["prefix"]).strip():
                retained.append((index, row, shell, answer))

        if len(retained) != 1:
            raise AssertionError(f"{candidate}: retained original-CoT rows={len(retained)}")
        retained_index, retained_row, shell, retained_answer = retained[0]
        if not str(retained_row["input"]).rstrip().endswith("/think"):
            raise AssertionError(f"{candidate}: retained row is not /think")
        result.append(
            {
                "candidate_id": candidate,
                "request": request,
                "generation": generation,
                "retained_index": retained_index,
                "retained_row": retained_row,
                "retained_answer": retained_answer,
                "retained_domain": answer_domain(retained_answer),
                "output_shell": shell,
                "golds": golds,
            }
        )

    result.sort(key=lambda group: group["candidate_id"])
    if sum(len(group["golds"]) for group in result) != 1_836:
        raise AssertionError("expected 1,836 unique known golds across candidate groups")
    return result


def encoded_item(
    tokenizer: Any,
    *,
    candidate_id: str,
    gold_index: int,
    variant: str,
    context: str,
    answer: str,
    cutoff_len: int,
) -> dict[str, Any]:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    answer_end_ids = tokenizer.encode(context + answer, add_special_tokens=False)
    if answer_end_ids[: len(context_ids)] != context_ids:
        raise AssertionError(
            f"{candidate_id}/{gold_index}/{variant}: tokenization crosses answer boundary"
        )
    if len(answer_end_ids) > cutoff_len:
        raise AssertionError(
            f"{candidate_id}/{gold_index}/{variant}: {len(answer_end_ids)} > cutoff {cutoff_len}"
        )
    if len(answer_end_ids) == len(context_ids):
        raise AssertionError(f"{candidate_id}/{gold_index}/{variant}: empty answer tokens")
    return {
        "candidate_id": candidate_id,
        "gold_index": gold_index,
        "variant": variant,
        "ids": answer_end_ids,
        "answer_start": len(context_ids),
    }


def score_items(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    batch_token_budget: int,
    max_batch_size: int,
) -> None:
    import torch
    import torch.nn.functional as functional

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad nor eos token")

    pending = sorted(items, key=lambda item: len(item["ids"]), reverse=True)
    cursor = 0
    while cursor < len(pending):
        batch: list[dict[str, Any]] = []
        maximum = 0
        while cursor < len(pending):
            candidate = pending[cursor]
            candidate_maximum = max(maximum, len(candidate["ids"]))
            if batch and (
                len(batch) >= max_batch_size
                or candidate_maximum * (len(batch) + 1) > batch_token_budget
            ):
                break
            batch.append(candidate)
            maximum = candidate_maximum
            cursor += 1

        input_ids = torch.tensor(
            [item["ids"] + [pad_id] * (maximum - len(item["ids"])) for item in batch],
            dtype=torch.long,
            device="cuda",
        )
        attention_mask = torch.tensor(
            [[1] * len(item["ids"]) + [0] * (maximum - len(item["ids"])) for item in batch],
            dtype=torch.long,
            device="cuda",
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
        for batch_index, item in enumerate(batch):
            start = int(item["answer_start"])
            end = len(item["ids"])
            token_logps = -functional.cross_entropy(
                logits[batch_index, start - 1 : end - 1].float(),
                input_ids[batch_index, start:end],
                reduction="none",
            )
            item["sum_logp"] = float(token_logps.sum().item())
            item["mean_logp"] = float(token_logps.mean().item())
            item["answer_tokens"] = int(token_logps.numel())
        del logits, input_ids, attention_mask


def selection_decision(
    retained_domain: str,
    mean_delta: float,
    retained_delta: float,
    win_rate: float,
    repair_vs_natural_nothink: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if mean_delta <= POSITIVE_EPS:
        failures.append("group_mean_delta_not_positive")
    if retained_delta <= POSITIVE_EPS:
        failures.append("retained_gold_delta_not_positive")
    minimum_win_rate = (
        VIDEO_MIN_WIN_RATE if retained_domain == "video" else NONVIDEO_MIN_WIN_RATE
    )
    if win_rate < minimum_win_rate:
        failures.append("gold_win_rate_below_domain_threshold")
    if retained_domain == "video" and repair_vs_natural_nothink <= POSITIVE_EPS:
        failures.append("video_repair_not_better_than_natural_nothink")
    return not failures, failures


def score(args: argparse.Namespace) -> None:
    for path in (
        args.base,
        args.adapter,
        args.parent,
        args.requests,
        args.generations,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    adapter_file = args.adapter / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.limit == 0 and args.max_batch_size != 1:
        raise ValueError("formal 538-group scoring requires --max-batch-size 1")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(19260817)
    torch.cuda.manual_seed_all(19260817)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "right"
    parent_rows = read_jsonl(args.parent)
    requests = read_jsonl(args.requests)
    generations = latest_by_candidate(args.generations)
    groups = collect_groups(parent_rows, requests, generations)
    if args.limit:
        groups = groups[: args.limit]

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(
        model, args.adapter, is_trainable=False
    ).eval()

    started = time.time()
    ledger: list[dict[str, Any]] = []
    for chunk_start in range(0, len(groups), args.group_chunk_size):
        chunk = groups[chunk_start : chunk_start + args.group_chunk_size]
        items: list[dict[str, Any]] = []
        for group in chunk:
            row = group["retained_row"]
            request = group["request"]
            generation = group["generation"]
            shell = group["output_shell"]
            original_prefix = shell.format(thought=str(request["prefix"]).strip())
            repaired_prefix = shell.format(
                thought=str(request["prefix"]).strip() + str(generation["continuation"])
            )
            empty_prefix = shell.format(thought="")
            think_prompt = qwen_prompt(row, str(row["input"]))
            nothink_prompt = qwen_prompt(row, natural_nothink_input(str(row["input"])))
            contexts = {
                "original": think_prompt + original_prefix,
                "repaired": think_prompt + repaired_prefix,
                "no_think_same_prompt": think_prompt + empty_prefix,
                "no_think_natural_prompt": nothink_prompt + empty_prefix,
            }
            for gold_index, gold in enumerate(group["golds"]):
                for variant, context in contexts.items():
                    items.append(
                        encoded_item(
                            tokenizer,
                            candidate_id=group["candidate_id"],
                            gold_index=gold_index,
                            variant=variant,
                            context=context,
                            answer=gold["answer"],
                            cutoff_len=args.cutoff_len,
                        )
                    )

        score_items(
            model,
            tokenizer,
            items,
            batch_token_budget=args.batch_token_budget,
            max_batch_size=args.max_batch_size,
        )
        item_by_key = {
            (item["candidate_id"], item["gold_index"], item["variant"]): item
            for item in items
        }
        for group in chunk:
            gold_rows = []
            retained_gold_position = None
            for gold_index, gold in enumerate(group["golds"]):
                if gold["answer"] == group["retained_answer"]:
                    retained_gold_position = gold_index
                scores = {}
                answer_tokens = None
                for variant in (
                    "original",
                    "repaired",
                    "no_think_same_prompt",
                    "no_think_natural_prompt",
                ):
                    item = item_by_key[(group["candidate_id"], gold_index, variant)]
                    scores[variant] = round(float(item["mean_logp"]), 8)
                    if answer_tokens is None:
                        answer_tokens = int(item["answer_tokens"])
                    elif answer_tokens != int(item["answer_tokens"]):
                        raise AssertionError("answer token count changed across trace variants")
                gold_rows.append(
                    {
                        "answer_sha256": gold["answer_sha256"],
                        "domain": gold["domain"],
                        "source_parent_index": gold["source_parent_index"],
                        "answer_tokens": answer_tokens,
                        "mean_logp": scores,
                        "delta_repaired_minus_original": round(
                            scores["repaired"] - scores["original"], 8
                        ),
                        "delta_repaired_minus_natural_nothink": round(
                            scores["repaired"] - scores["no_think_natural_prompt"], 8
                        ),
                    }
                )
            if retained_gold_position is None:
                raise AssertionError(f"{group['candidate_id']}: retained gold missing")

            deltas = [row["delta_repaired_minus_original"] for row in gold_rows]
            natural_deltas = [
                row["delta_repaired_minus_natural_nothink"] for row in gold_rows
            ]
            mean_delta = statistics.fmean(deltas)
            retained_delta = deltas[retained_gold_position]
            win_rate = sum(delta > POSITIVE_EPS for delta in deltas) / len(deltas)
            repair_vs_natural = statistics.fmean(natural_deltas)
            selected, failures = selection_decision(
                group["retained_domain"],
                mean_delta,
                retained_delta,
                win_rate,
                repair_vs_natural,
            )
            ledger.append(
                {
                    "candidate_id": group["candidate_id"],
                    "core_sha256": group["request"]["core_sha256"],
                    "retained_parent_index": group["retained_index"],
                    "retained_domain": group["retained_domain"],
                    "unique_gold_count": len(gold_rows),
                    "selected": selected,
                    "selection_failures": failures,
                    "group_metrics": {
                        "mean_delta_repaired_minus_original": round(mean_delta, 8),
                        "retained_gold_delta_repaired_minus_original": round(
                            retained_delta, 8
                        ),
                        "gold_positive_win_rate": round(win_rate, 8),
                        "mean_delta_repaired_minus_natural_nothink": round(
                            repair_vs_natural, 8
                        ),
                    },
                    "golds": gold_rows,
                }
            )
        print(
            f"scored_groups={min(chunk_start + len(chunk), len(groups))}/{len(groups)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    ledger.sort(key=lambda row: row["candidate_id"])
    write_jsonl(args.out, ledger, overwrite=args.overwrite)
    selected = [row for row in ledger if row["selected"]]
    selected_counts = Counter(row["retained_domain"] for row in selected)
    all_counts = Counter(row["retained_domain"] for row in ledger)
    selected_nonvideo = sum(
        count for domain, count in selected_counts.items() if domain != "video"
    )
    counted_domains = sum(
        count >= MIN_SELECTED_PER_COUNTED_DOMAIN for count in selected_counts.values()
    )
    formal = args.limit == 0
    gate_checks = {
        "all_538_groups_scored": len(ledger) == 538,
        "selected_total_gte_80": len(selected) >= MIN_SELECTED_TOTAL,
        "selected_nonvideo_gte_48": selected_nonvideo >= MIN_SELECTED_NONVIDEO,
        "domains_with_at_least_8_selected_gte_3": counted_domains >= MIN_SELECTED_DOMAINS,
    }
    gate_pass = formal and all(gate_checks.values())
    summary = {
        "status": "COMPLETE_CONSTRUCTION_GATE" if formal else "SMOKE_ONLY",
        "not_an_online_score_estimate": True,
        "method": {
            "scorer": "frozen I-10 E3",
            "template": "qwen3_nothink exact ChatML literals",
            "scored_tokens": "known final-answer tokens only; chat end token excluded",
            "group_aggregation": "macro mean over exact-deduplicated known golds",
            "same_prompt_primary_comparison": "repaired /think trace minus original /think trace",
            "no_think_role": "safety reference; natural /no_think required only for video",
            "stochastic_sampling": False,
            "cutoff_len": args.cutoff_len,
            "positive_epsilon_nats_per_token": POSITIVE_EPS,
            "nonvideo_min_gold_win_rate": NONVIDEO_MIN_WIN_RATE,
            "video_min_gold_win_rate": VIDEO_MIN_WIN_RATE,
            "video_requires_better_than_natural_nothink": True,
        },
        "sources": {
            "parent_I10_dataset": {
                "path": str(args.parent.resolve()),
                "rows": len(parent_rows),
                "sha256": sha256(args.parent),
            },
            "cotfix_v2_requests": {
                "path": str(args.requests.resolve()),
                "rows": len(requests),
                "sha256": sha256(args.requests),
            },
            "cotfix_v2_generations": {
                "path": str(args.generations.resolve()),
                "sha256": sha256(args.generations),
            },
            "base_O6": str(args.base.resolve()),
            "adapter_I10_E3": {
                "path": str(args.adapter.resolve()),
                "adapter_sha256": sha256(adapter_file),
            },
            "T_rows": 0,
            "E_rows_or_prompts": 0,
        },
        "coverage": {
            "groups": len(ledger),
            "unique_golds": sum(row["unique_gold_count"] for row in ledger),
            "all_groups_by_retained_domain": dict(sorted(all_counts.items())),
            "selected_groups": len(selected),
            "selected_by_retained_domain": dict(sorted(selected_counts.items())),
            "selected_nonvideo": selected_nonvideo,
        },
        "formal_training_gate": {
            "pass": gate_pass,
            "checks": gate_checks,
            "thresholds": {
                "min_selected_total": MIN_SELECTED_TOTAL,
                "min_selected_nonvideo": MIN_SELECTED_NONVIDEO,
                "min_selected_domains": MIN_SELECTED_DOMAINS,
                "min_selected_per_counted_domain": MIN_SELECTED_PER_COUNTED_DOMAIN,
            },
        },
        "runtime": {
            "gpu_argument": str(args.gpu),
            "dtype": "bfloat16 weights / float32 log-softmax",
            "batch_token_budget": args.batch_token_budget,
            "max_batch_size": args.max_batch_size,
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
            "argv": sys.argv,
        },
        "ledger": {
            "path": str(args.out.resolve()),
            "rows": len(ledger),
            "sha256": sha256(args.out),
        },
    }
    write_json(args.summary, summary, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def canonical_training_row(row: dict[str, Any]) -> str:
    normalized = {
        "instruction": str(row.get("instruction") or ""),
        "input": str(row.get("input") or ""),
        "output": str(row.get("output") or ""),
        "history": row.get("history") or [],
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build(args: argparse.Namespace) -> None:
    from build_cotfix_v2 import token_length_audit, validate_proposal
    from build_seed_scoremax_v1 import target_token_mix, task_of

    for path in (
        args.parent,
        args.o2_teacher,
        args.requests,
        args.generations,
        args.ledger,
        args.score_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    score_summary = json.loads(args.score_summary.read_text(encoding="utf-8"))
    ledger = read_jsonl(args.ledger)
    if score_summary.get("formal_training_gate", {}).get("pass") is not True:
        raise AssertionError("answer-utility formal training gate did not pass")
    if score_summary.get("ledger", {}).get("sha256") != sha256(args.ledger):
        raise AssertionError("answer-utility ledger checksum drifted")
    if len(ledger) != 538 or len({row["candidate_id"] for row in ledger}) != 538:
        raise AssertionError("expected 538 unique utility ledger rows")
    selected = {str(row["candidate_id"]) for row in ledger if row.get("selected") is True}
    if len(selected) < MIN_SELECTED_TOTAL:
        raise AssertionError("selected set fell below the frozen construction gate")

    parent_rows = read_jsonl(args.parent)
    requests = read_jsonl(args.requests)
    request_by_core = {str(row["prompt_core"]): row for row in requests}
    request_by_id = {str(row["candidate_id"]): row for row in requests}
    generations = latest_by_candidate(args.generations)
    groups = collect_groups(parent_rows, requests, generations)
    group_by_id = {group["candidate_id"]: group for group in groups}
    if selected - set(group_by_id):
        raise AssertionError("utility ledger selected unknown candidate IDs")

    parent_fields = [
        (row["instruction"], row["input"], row.get("history") or []) for row in parent_rows
    ]
    parent_outputs = [str(row["output"]) for row in parent_rows]
    parent_answers = [split_output(str(row["output"]))[2] for row in parent_rows]
    parent_counts = Counter(task_of(row) for row in parent_rows)
    parent_mix = target_token_mix(parent_rows)

    o2_teacher_rows = read_jsonl(args.o2_teacher)
    teacher_identities = {canonical_training_row(row) for row in o2_teacher_rows}
    teacher_indices = [
        index
        for index, row in enumerate(parent_rows)
        if canonical_training_row(row) in teacher_identities
    ]
    if len(o2_teacher_rows) != 164 or len(teacher_identities) != 164:
        raise AssertionError("O2 teacher identity source must contain 164 unique rows")
    if len(teacher_indices) != 164:
        raise AssertionError("I-10 parent must contain all 164 O2 teacher rows once")

    selected_indices = {group_by_id[candidate]["retained_index"]: candidate for candidate in selected}
    output_rows: list[dict[str, Any]] = []
    continuation_lengths: list[int] = []
    for index, row in enumerate(parent_rows):
        if index not in selected_indices:
            output_rows.append(dict(row))
            continue
        candidate = selected_indices[index]
        request = request_by_id[candidate]
        generation = generations[candidate]
        errors = validate_proposal(request, generation)
        if errors:
            raise AssertionError(f"{candidate}: registered generation failed revalidation: {errors}")
        match = THINK_RE.search(str(row["output"]))
        if match is None or match.group(1).strip() != str(request["prefix"]).strip():
            raise AssertionError(f"{candidate}: retained parent trace drifted")
        inner = match.group(1)
        leading = inner[: len(inner) - len(inner.lstrip())]
        trailing = inner[len(inner.rstrip()) :]
        repaired = (
            leading
            + str(request["prefix"]).strip()
            + str(generation["continuation"])
            + trailing
        )
        new_row = dict(row)
        new_row["output"] = (
            str(row["output"])[: match.start(1)]
            + repaired
            + str(row["output"])[match.end(1) :]
        )
        output_rows.append(new_row)
        continuation_lengths.append(len(str(generation["continuation"])))

    if len(output_rows) != len(parent_rows):
        raise AssertionError("row count changed")
    field_diffs = sum(
        (row["instruction"], row["input"], row.get("history") or []) != parent_fields[index]
        for index, row in enumerate(output_rows)
    )
    answer_diffs = sum(
        split_output(str(row["output"]))[2] != parent_answers[index]
        for index, row in enumerate(output_rows)
    )
    output_diff_indices = {
        index
        for index, row in enumerate(output_rows)
        if str(row["output"]) != parent_outputs[index]
    }
    if field_diffs or answer_diffs or output_diff_indices != set(selected_indices):
        raise AssertionError(
            f"dataset invariants failed: fields={field_diffs}, answers={answer_diffs}, "
            f"output_diffs={len(output_diff_indices)}, selected={len(selected_indices)}"
        )
    teacher_diffs = sum(index in output_diff_indices for index in teacher_indices)
    if teacher_diffs:
        raise AssertionError("an O2 teacher row was modified")
    final_counts = Counter(task_of(row) for row in output_rows)
    if final_counts != parent_counts:
        raise AssertionError("task counts changed")

    write_jsonl(args.out, output_rows, overwrite=args.overwrite)
    final_mix = target_token_mix(output_rows)
    length_audit = token_length_audit(output_rows)
    if length_audit["rows_over_cutoff_16384_with_8_token_margin"]:
        raise AssertionError(f"v3 exceeds cutoff: {length_audit}")
    selected_counts = Counter(
        group_by_id[candidate]["retained_domain"] for candidate in selected
    )
    lengths = sorted(continuation_lengths)
    audit = {
        "status": "READY_FOR_FORMAL_TRAINING",
        "asset_class": (
            "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O3; "
            "M-I10 construction filter)"
        ),
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
            "v2_validator_sha256": sha256(ROOT / "scripts/data/build_cotfix_v2.py"),
        },
        "upstream": {
            "parent_I10_dataset": {
                "asset_id": "D-I10",
                "path": str(args.parent.resolve()),
                "rows": len(parent_rows),
                "sha256": sha256(args.parent),
            },
            "O2_teacher_identity_source": {
                "asset_id": "D-O2-action-teacher-v5",
                "path": str(args.o2_teacher.resolve()),
                "rows": len(o2_teacher_rows),
                "sha256": sha256(args.o2_teacher),
            },
            "cotfix_v2_requests": {
                "asset_id": "D-I18-requests",
                "path": str(args.requests.resolve()),
                "rows": len(requests),
                "sha256": sha256(args.requests),
            },
            "cotfix_v2_generations": {
                "asset_id": "D-I18-generations",
                "path": str(args.generations.resolve()),
                "sha256": sha256(args.generations),
            },
            "answer_utility_ledger": {
                "asset_id": "D-I23-selection-ledger",
                "path": str(args.ledger.resolve()),
                "rows": len(ledger),
                "sha256": sha256(args.ledger),
                "scorer_adapter_sha256": score_summary["sources"]["adapter_I10_E3"][
                    "adapter_sha256"
                ],
            },
        },
        "rows": len(output_rows),
        "row_mix": {
            "O1_parent_rows": {"rows": 32_480, "ratio": round(32_480 / 32_644, 8)},
            "O2_teacher_unique_once": {"rows": 164, "ratio": round(164 / 32_644, 8)},
            "T": {"rows": 0, "ratio": 0.0},
            "E": {"rows": 0, "ratio": 0.0},
        },
        "single_variable_change": {
            "candidate_groups_scored": len(ledger),
            "retained_CoT_rows_changed": len(selected),
            "selected_by_retained_domain": dict(sorted(selected_counts.items())),
            "all_other_rows_unchanged": len(output_rows) - len(selected),
            "continuation_chars": {
                "min": lengths[0],
                "median": statistics.median(lengths),
                "p90": lengths[round((len(lengths) - 1) * 0.9)],
                "max": lengths[-1],
            },
        },
        "invariants": {
            "row_order_preserved": True,
            "instruction_input_history_diffs": field_diffs,
            "answer_diffs": answer_diffs,
            "output_diff_count": len(output_diff_indices),
            "task_count_diffs": 0,
            "O2_teacher_rows_changed": teacher_diffs,
            "failed_checkpoint_inputs": 0,
            "T_rows": 0,
            "E_rows_or_prompts": 0,
        },
        "task_counts": dict(sorted(final_counts.items())),
        "target_token_mix_before": parent_mix,
        "target_token_mix_after": final_mix,
        "token_length_audit": length_audit,
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(output_rows),
            "sha256": sha256(args.out),
        },
    }
    write_json(args.audit, audit, overwrite=args.overwrite)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    subparsers = top.add_subparsers(dest="command", required=True)

    scorer = subparsers.add_parser("score")
    scorer.add_argument("--base", type=Path, default=DEFAULT_BASE)
    scorer.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    scorer.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    scorer.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    scorer.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    scorer.add_argument("--out", type=Path, default=DEFAULT_LEDGER)
    scorer.add_argument("--summary", type=Path, default=DEFAULT_SCORE_SUMMARY)
    scorer.add_argument("--gpu", default="0")
    scorer.add_argument("--cutoff-len", type=int, default=16_384)
    scorer.add_argument("--batch-token-budget", type=int, default=16_384)
    scorer.add_argument("--max-batch-size", type=int, default=1)
    scorer.add_argument("--group-chunk-size", type=int, default=16)
    scorer.add_argument("--limit", type=int, default=0)
    scorer.add_argument("--overwrite", action="store_true")
    scorer.set_defaults(function=score)

    builder = subparsers.add_parser("build")
    builder.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    builder.add_argument("--o2-teacher", type=Path, default=DEFAULT_O2_TEACHER)
    builder.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    builder.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    builder.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    builder.add_argument("--score-summary", type=Path, default=DEFAULT_SCORE_SUMMARY)
    builder.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    builder.add_argument("--audit", type=Path, default=DEFAULT_BUILD_AUDIT)
    builder.add_argument("--overwrite", action="store_true")
    builder.set_defaults(function=build)
    return top


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "batch_token_budget", 1) < 1:
        raise ValueError("--batch-token-budget must be positive")
    if getattr(args, "max_batch_size", 1) < 1:
        raise ValueError("--max-batch-size must be positive")
    if getattr(args, "group_chunk_size", 1) < 1:
        raise ValueError("--group-chunk-size must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
