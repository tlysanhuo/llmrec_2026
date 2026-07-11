#!/usr/bin/env python3
"""Build a grouped, token-balanced no-think dataset for stage-2 LoRA SFT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, TypeVar


ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
KEEP_KEYS = ("instruction", "input", "output", "history")
REC_BEGIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
ITEM_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
ANSWER_RE = re.compile(r"正确答案是\s*[\(（]?\s*([A-D])\s*[\)）]?")
WHITESPACE_RE = re.compile(r"\s+")
WORLD_SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
WORLD_PREFIX = "请回答以下问题：\n\n"
WORLD_SUFFIX = '\n\n请按以下格式作答："正确答案是 (在此处填写选项字母)"/no_think'

T = TypeVar("T")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if isinstance(row, list):
                row = row[0]
            if "prompt" in row and "input" not in row:
                row = {
                    "instruction": row.get("system", ""),
                    "input": row["prompt"],
                    "output": row["response"],
                    "history": row.get("history") or [],
                }
            rows.append(row)
    return rows


def clean_row(row: dict) -> dict:
    return {
        "instruction": row.get("instruction", "") or "",
        "input": row.get("input", "") or "",
        "output": row.get("output", "") or "",
        "history": row.get("history") or [],
    }


def answer_body(row: dict) -> str:
    return row.get("output", "").split("</think>")[-1].strip()


def core_input(text: str) -> str:
    text = text.rstrip()
    for suffix in ("/no_think", "/think"):
        if text.endswith(suffix):
            return text[: -len(suffix)].rstrip()
    return text


def prompt_key(row: dict) -> tuple[str, str]:
    return row.get("instruction", "") or "", core_input(row.get("input", "") or "")


def normalized_prompt(row: dict) -> str:
    return WHITESPACE_RE.sub("", core_input(row.get("input", "") or ""))


def load_dev_prompts(dev_dir: Path) -> set[str]:
    prompts = set()
    for path in sorted(dev_dir.glob("dev_*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("user"):
                    prompts.add(WHITESPACE_RE.sub("", core_input(row["user"])))
    return prompts


def to_nothink(row: dict) -> dict:
    clean = clean_row(row)
    clean["input"] = core_input(clean["input"]) + "/no_think"
    clean["output"] = "<think>\n\n</think>\n" + answer_body(clean)
    return clean


def classify_official(row: dict) -> tuple[str, str | None]:
    body = answer_body(row)
    if body.startswith("["):
        return "action", None
    if body.startswith("{") and "logic_chain" in body:
        return "topic", None
    if "该用户最近" in body:
        match = REC_BEGIN_RE.search(body)
        if match is None:
            raise ValueError("recommendation row has no domain token")
        return "recommendation", match.group(1)
    return "material", None


def largest_remainder_counts(sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(sizes.values())
    if total > population:
        raise ValueError(f"cannot sample {total} rows from {population}")
    raw = {key: total * size / population for key, size in sizes.items()}
    counts = {key: min(size, math.floor(raw[key])) for key, size in sizes.items()}
    remaining = total - sum(counts.values())
    order = sorted(sizes, key=lambda key: (raw[key] - counts[key], sizes[key], key), reverse=True)
    for key in order:
        if remaining == 0:
            break
        if counts[key] < sizes[key]:
            counts[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"failed to allocate {remaining} samples")
    return counts


def stratified_take(
    rows: list[T],
    count: int,
    stratum: Callable[[T], str],
    rng: random.Random,
) -> tuple[list[T], list[T]]:
    buckets: dict[str, list[T]] = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    allocations = largest_remainder_counts({key: len(value) for key, value in buckets.items()}, count)
    selected, remaining = [], []
    for key in sorted(buckets):
        take = allocations[key]
        selected.extend(buckets[key][:take])
        remaining.extend(buckets[key][take:])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def action_stratum(row: dict) -> str:
    values = json.loads(answer_body(row))
    size = len(values)
    if size <= 4:
        size_bin = "01-04"
    elif size <= 10:
        size_bin = "05-10"
    elif size <= 19:
        size_bin = "11-19"
    else:
        size_bin = "20+"
    domains = sorted({match.group(1) for value in values for match in [ITEM_RE.fullmatch(value)] if match})
    domain_bin = "+".join(domains) if len(domains) <= 2 else "multi3+"
    return f"{size_bin}|{domain_bin or 'text'}"


def topic_stratum(row: dict) -> str:
    value = json.loads(answer_body(row)).get("logic_chain", {})
    events = value.get("events", []) if isinstance(value, dict) else []
    size = len(events)
    if size <= 2:
        return "01-02"
    if size == 3:
        return "03"
    if size == 4:
        return "04"
    return "05+"


def answer_letter(row: dict) -> str:
    match = ANSWER_RE.search(answer_body(row))
    return match.group(1) if match else "unknown"


def select_grouped_recommendation(
    rows: list[dict],
    train_count: int,
    holdout_count: int,
    group_cap: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict], dict]:
    grouped_by_answer: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    duplicate_prompt_gold_rows = 0
    for row in rows:
        answers = grouped_by_answer[prompt_key(row)]
        answer = answer_body(row)
        if answer in answers:
            duplicate_prompt_gold_rows += 1
        else:
            answers[answer] = row

    grouped = {key: list(answer_rows.values()) for key, answer_rows in grouped_by_answer.items()}

    groups = list(grouped.items())
    for _, group_rows in groups:
        group_rows.sort(key=lambda row: answer_body(row))
        rng.shuffle(group_rows)
    rng.shuffle(groups)

    holdout, holdout_keys = [], set()
    remaining_groups = []
    for key, group_rows in groups:
        if len(holdout) < holdout_count:
            take = min(group_cap, holdout_count - len(holdout), len(group_rows))
            holdout.extend(group_rows[:take])
            holdout_keys.add(key)
        else:
            remaining_groups.append((key, group_rows))

    train, train_keys = [], set()
    for key, group_rows in remaining_groups:
        if len(train) >= train_count:
            break
        take = min(group_cap, train_count - len(train), len(group_rows))
        train.extend(group_rows[:take])
        train_keys.add(key)

    if len(train) != train_count or len(holdout) != holdout_count:
        raise RuntimeError(
            f"recommendation allocation failed: train={len(train)}/{train_count}, "
            f"holdout={len(holdout)}/{holdout_count}"
        )
    if train_keys & holdout_keys:
        raise AssertionError("recommendation prompt-group leakage")

    audit = {
        "source_rows": len(rows),
        "source_unique_prompt_gold_rows": sum(len(group_rows) for group_rows in grouped.values()),
        "dropped_duplicate_prompt_gold_rows": duplicate_prompt_gold_rows,
        "source_prompt_groups": len(grouped),
        "train_rows": len(train),
        "train_prompt_groups": len(train_keys),
        "holdout_rows": len(holdout),
        "holdout_prompt_groups": len(holdout_keys),
        "group_cap": group_cap,
    }
    return train, holdout, audit


def make_world_row(question: str, options: dict[str, str], answer: str) -> dict:
    option_text = "\n".join(f"{letter}.{options[letter].strip()}" for letter in "ABCD")
    return {
        "instruction": WORLD_SYSTEM,
        "input": WORLD_PREFIX + question.strip() + "\n" + option_text + WORLD_SUFFIX,
        "output": f"<think>\n\n</think>\n\n\n正确答案是 ({answer})",
        "history": [],
    }


def load_ceval_rows(ceval_dir: Path) -> list[tuple[str, dict]]:
    import pyarrow.parquet as pq

    rows = []
    for path in sorted(ceval_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["question", "A", "B", "C", "D", "answer"])
        for record in table.to_pylist():
            answer = str(record.get("answer", "")).strip()
            question = str(record.get("question", "")).strip()
            options = {letter: str(record.get(letter, "")) for letter in "ABCD"}
            if question and answer in "ABCD" and all(options.values()):
                rows.append(("ceval", make_world_row(question, options, answer)))
    return rows


def load_cmmlu_rows(cmmlu_zip: Path) -> list[tuple[str, dict]]:
    rows = []
    with zipfile.ZipFile(cmmlu_zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("test/") or not name.endswith(".csv"):
                continue
            with archive.open(name) as source:
                reader = csv.DictReader(io.TextIOWrapper(source, "utf-8"))
                for record in reader:
                    answer = (record.get("Answer") or record.get("answer") or "").strip()
                    question = (record.get("Question") or record.get("question") or "").strip()
                    options = {letter: (record.get(letter) or "").strip() for letter in "ABCD"}
                    if question and answer in "ABCD" and all(options.values()):
                        rows.append(("cmmlu", make_world_row(question, options, answer)))
    return rows


def load_world_candidates(
    ceval_dir: Path,
    cmmlu_zip: Path,
    eval_path: Path,
    dev_dir: Path,
) -> tuple[list[tuple[str, dict]], dict]:
    eval_prompts = {normalized_prompt(row) for row in load_jsonl(eval_path)}
    dev_prompts = load_dev_prompts(dev_dir)
    candidates: list[tuple[str, dict]] = []
    stats = Counter()

    candidates.extend(load_ceval_rows(ceval_dir))
    candidates.extend(load_cmmlu_rows(cmmlu_zip))

    deduped = []
    seen = set()
    for source, row in candidates:
        key = normalized_prompt(row)
        if key in eval_prompts:
            stats[f"drop_eval_collision:{source}"] += 1
            continue
        if key in dev_prompts:
            stats[f"drop_dev_collision:{source}"] += 1
            continue
        if key in seen:
            stats[f"drop_duplicate:{source}"] += 1
            continue
        if answer_letter(row) == "unknown":
            stats[f"drop_bad_answer:{source}"] += 1
            continue
        seen.add(key)
        deduped.append((source, row))
        stats[f"kept:{source}"] += 1

    return deduped, dict(stats)


def select_world_rows(
    candidates: list[tuple[str, dict]],
    train_count: int,
    ceval_train_count: int,
    holdout_per_public_source: int,
    rng: random.Random,
) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]], dict]:
    by_source: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for item in candidates:
        by_source[item[0]].append(item)

    ceval_train, ceval_remaining = stratified_take(
        by_source["ceval"], ceval_train_count, lambda item: answer_letter(item[1]), rng
    )
    cmmlu_train_count = train_count - len(ceval_train)
    if cmmlu_train_count <= 0:
        raise ValueError("world source quotas leave no room for CMMLU")
    cmmlu_train, cmmlu_remaining = stratified_take(
        by_source["cmmlu"], cmmlu_train_count, lambda item: answer_letter(item[1]), rng
    )
    ceval_holdout, _ = stratified_take(
        ceval_remaining, holdout_per_public_source, lambda item: answer_letter(item[1]), rng
    )
    cmmlu_holdout, _ = stratified_take(
        cmmlu_remaining, holdout_per_public_source, lambda item: answer_letter(item[1]), rng
    )
    train = ceval_train + cmmlu_train
    holdout = ceval_holdout + cmmlu_holdout
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout, {
        "available": {source: len(rows) for source, rows in sorted(by_source.items())},
        "train": dict(sorted(Counter(source for source, _ in train).items())),
        "holdout": dict(sorted(Counter(source for source, _ in holdout).items())),
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            encoded = (json.dumps(clean_row(row), ensure_ascii=False) + "\n").encode("utf-8")
            output.write(encoded.decode("utf-8"))
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest()


def token_audit(rows: list[tuple[str, dict]], tokenizer_path: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    target_tokens = Counter()
    full_tokens = Counter()
    for offset in range(0, len(rows), 256):
        batch = rows[offset : offset + 256]
        outputs = [row["output"] for _, row in batch]
        full_texts = [
            row["instruction"] + "\n" + row["input"] + "\n" + row["output"] for _, row in batch
        ]
        output_lengths = tokenizer(
            outputs, add_special_tokens=False, return_length=True, truncation=False
        )["length"]
        full_lengths = tokenizer(
            full_texts, add_special_tokens=False, return_length=True, truncation=False
        )["length"]
        for (label, _), target_length, full_length in zip(batch, output_lengths, full_lengths):
            target_tokens[label] += target_length
            full_tokens[label] += full_length

    target_total = sum(target_tokens.values())
    full_total = sum(full_tokens.values())
    return {
        "target_tokens": dict(sorted(target_tokens.items())),
        "target_token_total": target_total,
        "target_token_share": {
            key: round(value / target_total, 6) for key, value in sorted(target_tokens.items())
        },
        "raw_full_tokens": dict(sorted(full_tokens.items())),
        "raw_full_token_total": full_total,
        "approx_packed_sequences_32768": math.ceil(full_total / 32768),
        "approx_optimizer_steps_accum4": math.ceil(full_total / (32768 * 4)),
    }


def assert_no_think(rows: list[tuple[str, dict]]) -> None:
    for index, (label, row) in enumerate(rows):
        if not row["input"].rstrip().endswith("/no_think"):
            raise AssertionError(f"{label} row {index} lacks /no_think")
        if not row["output"].startswith("<think>\n\n</think>\n"):
            raise AssertionError(f"{label} row {index} has non-empty think")


def task_counts(rows: list[tuple[str, dict]]) -> dict[str, int]:
    return dict(sorted(Counter(label for label, _ in rows).items()))


def action_distribution(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(action_stratum(row) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, default=P / "data_final.jsonl")
    parser.add_argument("--ceval-dir", type=Path, default=P.parent / "offline_eval/_ceval_val")
    parser.add_argument("--cmmlu-zip", type=Path, default=P.parent / "offline_eval/_cmmlu.zip")
    parser.add_argument("--world-eval", type=Path, default=ROOT / "懂世界.jsonl")
    parser.add_argument("--dev-dir", type=Path, default=P.parent / "offline_eval")
    parser.add_argument("--out", type=Path, default=P / "data_stage2_gold_v1.jsonl")
    parser.add_argument("--holdout", type=Path, default=P / "stage2_gold_v1_holdout.jsonl")
    parser.add_argument("--audit", type=Path, default=ROOT / "logs/data/stage2_gold_v1_audit.json")
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "models/OneReason-0.8B-pretrain-competition"
    )
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--action-holdout", type=int, default=158)
    parser.add_argument("--topic-train", type=int, default=300)
    parser.add_argument("--topic-holdout", type=int, default=100)
    parser.add_argument("--rec-train-per-domain", type=int, default=1200)
    parser.add_argument("--rec-holdout-per-domain", type=int, default=50)
    parser.add_argument("--rec-group-cap", type=int, default=3)
    parser.add_argument("--world-train", type=int, default=1500)
    parser.add_argument("--world-ceval-train", type=int, default=1000)
    parser.add_argument("--world-holdout-per-public-source", type=int, default=150)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    official_buckets: dict[str, list[dict]] = defaultdict(list)
    source_counts = Counter()
    for row in load_jsonl(args.official):
        task, domain = classify_official(row)
        source_counts[task if domain is None else f"{task}:{domain}"] += 1
        if task == "action":
            official_buckets["action"].append(row)
        elif task == "topic":
            official_buckets["topic"].append(row)
        elif task == "recommendation":
            official_buckets[f"rec_{domain}"].append(row)

    expected = {
        "action": 1588,
        "material": 10384,
        "recommendation:ad": 1576,
        "recommendation:living": 1271,
        "recommendation:prod": 1489,
        "recommendation:video": 14868,
        "topic": 1304,
    }
    if dict(sorted(source_counts.items())) != expected:
        raise AssertionError(f"unexpected official source counts: {dict(source_counts)}")

    action_holdout, action_train = stratified_take(
        official_buckets["action"], args.action_holdout, action_stratum, rng
    )
    if len(action_train) != 1430:
        raise AssertionError(f"expected 1430 action train rows, found {len(action_train)}")

    topic_holdout, topic_remaining = stratified_take(
        official_buckets["topic"], args.topic_holdout, topic_stratum, rng
    )
    topic_train, _ = stratified_take(topic_remaining, args.topic_train, topic_stratum, rng)

    train_labeled: list[tuple[str, dict]] = [("action", to_nothink(row)) for row in action_train]
    holdout_labeled: list[tuple[str, dict]] = [("action", to_nothink(row)) for row in action_holdout]
    train_labeled.extend(("topic", to_nothink(row)) for row in topic_train)
    holdout_labeled.extend(("topic", to_nothink(row)) for row in topic_holdout)

    rec_audit = {}
    for domain in ("video", "prod", "ad", "living"):
        train_rows, holdout_rows, domain_audit = select_grouped_recommendation(
            official_buckets[f"rec_{domain}"],
            args.rec_train_per_domain,
            args.rec_holdout_per_domain,
            args.rec_group_cap,
            rng,
        )
        label = f"rec_{domain}"
        train_labeled.extend((label, to_nothink(row)) for row in train_rows)
        holdout_labeled.extend((label, to_nothink(row)) for row in holdout_rows)
        rec_audit[domain] = domain_audit

    world_candidates, world_source_audit = load_world_candidates(
        args.ceval_dir, args.cmmlu_zip, args.world_eval, args.dev_dir
    )
    world_train, world_holdout, world_selection_audit = select_world_rows(
        world_candidates,
        args.world_train,
        args.world_ceval_train,
        args.world_holdout_per_public_source,
        rng,
    )
    train_labeled.extend(("world", row) for _, row in world_train)
    holdout_labeled.extend(("world", row) for _, row in world_holdout)

    rng.shuffle(train_labeled)
    rng.shuffle(holdout_labeled)
    assert_no_think(train_labeled)
    assert_no_think(holdout_labeled)

    train_prompt_keys = {(label, prompt_key(row)) for label, row in train_labeled}
    holdout_prompt_keys = {(label, prompt_key(row)) for label, row in holdout_labeled}
    prompt_overlap = train_prompt_keys & holdout_prompt_keys
    if prompt_overlap:
        raise AssertionError(f"train/holdout prompt leakage: {len(prompt_overlap)}")
    dev_prompts = load_dev_prompts(args.dev_dir)
    dev_overlap = [
        (label, normalized_prompt(row))
        for label, row in train_labeled + holdout_labeled
        if normalized_prompt(row) in dev_prompts
    ]
    if dev_overlap:
        raise AssertionError(f"stage2 data overlaps offline dev prompts: {len(dev_overlap)}")

    expected_train_counts = {
        "action": 1430,
        "rec_ad": 1200,
        "rec_living": 1200,
        "rec_prod": 1200,
        "rec_video": 1200,
        "topic": 300,
        "world": 1500,
    }
    if task_counts(train_labeled) != expected_train_counts:
        raise AssertionError(f"unexpected train counts: {task_counts(train_labeled)}")

    train_count, train_md5 = write_jsonl(args.out, (row for _, row in train_labeled))
    holdout_count, holdout_md5 = write_jsonl(args.holdout, (row for _, row in holdout_labeled))

    audit = {
        "seed": args.seed,
        "sources": {
            "official": str(args.official.resolve()),
            "ceval": str(args.ceval_dir.resolve()),
            "cmmlu": str(args.cmmlu_zip.resolve()),
            "world_eval_exclusion": str(args.world_eval.resolve()),
            "offline_dev_exclusion": str(args.dev_dir.resolve()),
        },
        "source_counts": dict(sorted(source_counts.items())),
        "world_source_audit": world_source_audit,
        "world_selection_audit": world_selection_audit,
        "train": {
            "path": str(args.out.resolve()),
            "rows": train_count,
            "md5": train_md5,
            "task_counts": task_counts(train_labeled),
            "action_strata": action_distribution(action_train),
        },
        "holdout": {
            "path": str(args.holdout.resolve()),
            "rows": holdout_count,
            "md5": holdout_md5,
            "task_counts": task_counts(holdout_labeled),
            "action_strata": action_distribution(action_holdout),
        },
        "recommendation_grouping": rec_audit,
        "train_holdout_prompt_overlap": 0,
        "offline_dev_prompt_overlap": 0,
        "material_rows": 0,
        "synthetic_action_rows": 0,
        "rewritten_recommendation_gold_rows": 0,
        "all_rows_nothink": True,
        "token_audit": token_audit(train_labeled, args.tokenizer),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
