#!/usr/bin/env python3
"""Build O1-derived reward-aligned preference pairs.

Recommendation pairs keep the official chosen response byte-identical and
replace only its final item with a same-domain history item.  All official
positives for the same prompt are excluded from the negative pool.  Action
pairs keep the official chosen response byte-identical and add exactly one
chronologically placed non-gold history event to the rejected JSON array.

Action candidates are also constructed and audited, but the default formal
training output contains recommendation pairs only.  The strong E3 parent
already prefers the action gold over an added false positive, so action pairs
are retained only in the diagnostic holdout unless ``--train-tasks all`` is
explicitly requested.

No model rollout, teacher label, third-party data, or evaluation data is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "assets/derived/processed/data_seed_clean_v1.jsonl"
DEFAULT_TRAIN = ROOT / "assets/derived/processed/data_o1_reward_preference_v1_train.jsonl"
DEFAULT_HOLDOUT = ROOT / "assets/evaluation/holdout/data_o1_reward_preference_v1_holdout.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/o1_reward_preference_v1_audit.json"

MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
ITEM_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
ITEM_PART_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
DATE_RE = re.compile(r"^【(\d{4}-\d{2}-\d{2})】$")
ACTION_RE = re.compile(r"\[([^\]]+)]")
HISTORY_SPLIT_RE = re.compile(r"\n\n(?=角色任务：)")


@dataclass(frozen=True)
class Event:
    index: int
    date: str
    line: str
    token: str
    action: str
    domain: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, list):
                if len(row) != 1:
                    raise ValueError(f"unexpected list row at {path}:{line_number}")
                row = row[0]
            normalized = {
                "instruction": str(row.get("instruction", row.get("system", "")) or ""),
                "input": str(row.get("input", row.get("prompt", "")) or ""),
                "output": str(row.get("output", row.get("response", "")) or ""),
                "history": row.get("history") or [],
            }
            if normalized["history"]:
                raise AssertionError(f"non-empty history at {path}:{line_number}")
            rows.append(normalized)
    return rows


def response_parts(response: str) -> tuple[str, str, str]:
    """Return byte-preserving prefix, semantic body, and trailing whitespace."""
    marker = "</think>"
    if marker not in response:
        raise ValueError("response is missing </think>")
    head, tail = response.split(marker, 1)
    leading_count = len(tail) - len(tail.lstrip())
    trailing_count = len(tail) - len(tail.rstrip())
    body_end = len(tail) - trailing_count if trailing_count else len(tail)
    leading = tail[:leading_count]
    body = tail[leading_count:body_end]
    trailing = tail[body_end:]
    if not body:
        raise ValueError("response body is empty")
    return head + marker + leading, body, trailing


def replace_response_body(response: str, body: str) -> str:
    prefix, _, trailing = response_parts(response)
    return prefix + body + trailing


def answer_body(row: dict[str, Any]) -> str:
    return response_parts(row["output"])[1]


def task_of(row: dict[str, Any]) -> str:
    body = answer_body(row).strip()
    if body.startswith("["):
        return "action"
    if body.startswith("{") and "logic_chain" in body:
        return "topic"
    if "该用户最近" in body:
        match = DOMAIN_RE.search(body)
        return f"rec_{match.group(1) if match else 'unknown'}"
    input_has_sid = "<s_a_" in row["input"]
    output_has_sid = "<s_a_" in body
    if output_has_sid and not input_has_sid:
        return "material_desc2sid"
    if input_has_sid and not output_has_sid:
        return "material_sid2desc"
    raise ValueError(f"cannot classify row: {body[:120]!r}")


def core_prompt(row: dict[str, Any]) -> tuple[str, str]:
    return row["instruction"], MODE_SUFFIX_RE.sub("", row["input"].rstrip())


def item_parts(token: str) -> tuple[str, int, int, int]:
    match = ITEM_PART_RE.fullmatch(token)
    if not match:
        raise ValueError(f"invalid itemic token: {token!r}")
    return (
        match.group("domain"),
        int(match.group("a")),
        int(match.group("b")),
        int(match.group("c")),
    )


def split_name(seed: int, kind: str, *parts: object) -> str:
    bucket = int(stable_hash(seed, "split", kind, *parts)[:16], 16) % 10
    return "holdout" if bucket == 0 else "train"


def rec_target(row: dict[str, Any]) -> str:
    matches = ITEM_RE.findall(answer_body(row))
    if len(matches) != 1:
        raise ValueError(f"recommendation response has {len(matches)} itemic targets")
    return matches[0]


def choose_rec_negative(
    row: dict[str, Any],
    positives: set[str],
    seed: int,
    row_index: int,
) -> tuple[str, str] | None:
    target = rec_target(row)
    target_domain, target_a, target_b, _ = item_parts(target)
    last_positions: dict[str, int] = {}
    for position, token in enumerate(ITEM_RE.findall(row["input"])):
        last_positions[token] = position

    candidates = []
    for token, position in last_positions.items():
        domain, a_value, b_value, _ = item_parts(token)
        if token in positives or domain != target_domain:
            continue
        if a_value == target_a and b_value == target_b:
            tier_rank, tier = 0, "same_ab"
        elif a_value == target_a:
            tier_rank, tier = 1, "same_a"
        else:
            tier_rank, tier = 2, "same_domain"
        candidates.append(
            (
                tier_rank,
                -position,
                stable_hash(seed, "rec_negative", row_index, token),
                token,
                tier,
            )
        )
    if not candidates:
        return None
    *_, token, tier = min(candidates)
    return token, tier


def build_rec_pairs(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        if task_of(row).startswith("rec_"):
            groups[core_prompt(row)].append((row_index, row))

    pairs = []
    skipped_by_task = Counter()
    eligible_by_task = Counter()
    tier_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    groups_by_split = Counter()
    group_coverage = Counter()
    for key, grouped_rows in groups.items():
        positives = {rec_target(row) for _, row in grouped_rows}
        group_id = stable_hash("rec_group", *key)[:16]
        split = split_name(seed, "rec", *key)
        eligible_in_group = 0
        for row_index, row in grouped_rows:
            task = task_of(row)
            selected = choose_rec_negative(row, positives, seed, row_index)
            if selected is None:
                skipped_by_task[task] += 1
                continue
            negative, tier = selected
            target = rec_target(row)
            prefix, body, trailing = response_parts(row["output"])
            if body.count(target) != 1:
                raise AssertionError("recommendation answer target is not unique in body")
            rejected = prefix + body.replace(target, negative, 1) + trailing
            if rejected == row["output"] or rec_target({**row, "output": rejected}) != negative:
                raise AssertionError("invalid recommendation rejected response")
            if negative in positives:
                raise AssertionError("known recommendation positive selected as negative")
            pairs.append(
                {
                    "instruction": row["instruction"],
                    "input": row["input"],
                    "chosen": row["output"],
                    "rejected": rejected,
                    "meta": {
                        "source": "D(O1):data_seed_clean_v1",
                        "source_row": row_index,
                        "task": task,
                        "split": split,
                        "prompt_group": group_id,
                        "group_positive_count": len(positives),
                        "chosen_target": target,
                        "rejected_target": negative,
                        "negative_tier": tier,
                    },
                }
            )
            eligible_in_group += 1
            eligible_by_task[task] += 1
            tier_by_task[task][tier] += 1
        if eligible_in_group == len(grouped_rows):
            group_coverage["all_eligible"] += 1
        elif eligible_in_group:
            group_coverage["partial"] += 1
        else:
            group_coverage["none"] += 1
        if eligible_in_group:
            groups_by_split[split] += 1

    return pairs, {
        "source_rows": sum(len(group) for group in groups.values()),
        "prompt_groups": len(groups),
        "eligible_rows": len(pairs),
        "skipped_no_same_domain_negative": sum(skipped_by_task.values()),
        "eligible_by_task": dict(sorted(eligible_by_task.items())),
        "skipped_by_task": dict(sorted(skipped_by_task.items())),
        "negative_tiers_by_task": {
            task: dict(sorted(counts.items())) for task, counts in sorted(tier_by_task.items())
        },
        "group_coverage": dict(sorted(group_coverage.items())),
        "eligible_groups_by_split": dict(sorted(groups_by_split.items())),
    }


def parse_action_events(prompt: str) -> list[Event]:
    parts = HISTORY_SPLIT_RE.split(prompt, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("action prompt lacks history/task boundary")
    history_text, _ = parts
    lines = history_text.splitlines()
    if not lines or lines[0].strip() != "【用户交互历史】：":
        raise ValueError("action prompt has unexpected history header")

    events = []
    current_date = ""
    for line in lines[1:]:
        stripped = line.strip()
        date_match = DATE_RE.match(stripped)
        if date_match:
            current_date = date_match.group(1)
            continue
        token_match = ITEM_RE.search(line)
        if not token_match:
            continue
        action_match = ACTION_RE.search(line)
        domain_match = DOMAIN_RE.search(token_match.group(0))
        if not current_date or not action_match or not domain_match:
            raise ValueError("action event is missing date/action/domain")
        events.append(
            Event(
                index=len(events),
                date=current_date,
                line=line.rstrip(),
                token=token_match.group(0),
                action=action_match.group(1),
                domain=domain_match.group(1),
            )
        )
    if not events:
        raise ValueError("action prompt contains no itemic events")
    return events


def action_targets(row: dict[str, Any]) -> list[str]:
    values = json.loads(answer_body(row))
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ValueError("action target is not a non-empty string list")
    if len(values) != len(set(values)):
        raise ValueError("action target contains duplicates")
    if any(ITEM_RE.fullmatch(value) is None for value in values):
        raise ValueError("action target contains an invalid itemic token")
    return values


def align_target_events(events: list[Event], targets: list[str]) -> list[int] | None:
    aligned = []
    cursor = 0
    for target in targets:
        match = next((event.index for event in events[cursor:] if event.token == target), None)
        if match is None:
            return None
        aligned.append(match)
        cursor = match + 1
    return aligned


def choose_action_negative(
    events: list[Event], selected_indices: list[int], targets: set[str], seed: int, row_index: int
) -> tuple[Event, str] | None:
    selected = [events[index] for index in selected_indices]
    candidates = []
    for event in events:
        if event.token in targets:
            continue
        same_action_domain = any(
            event.action == target.action and event.domain == target.domain for target in selected
        )
        same_action = any(event.action == target.action for target in selected)
        same_domain = any(event.domain == target.domain for target in selected)
        if same_action_domain:
            tier_rank, tier = 0, "same_action_domain"
        elif same_action:
            tier_rank, tier = 1, "same_action"
        elif same_domain:
            tier_rank, tier = 2, "same_domain"
        else:
            tier_rank, tier = 3, "other"
        distance = min(abs(event.index - target.index) for target in selected)
        candidates.append(
            (
                tier_rank,
                distance,
                -event.index,
                stable_hash(seed, "action_negative", row_index, event.index, event.token),
                event,
                tier,
            )
        )
    if not candidates:
        return None
    *_, event, tier = min(candidates)
    return event, tier


def build_action_pairs(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = []
    stats = Counter()
    tier_counts = Counter()
    for row_index, row in enumerate(rows):
        if task_of(row) != "action":
            continue
        stats["source_rows"] += 1
        events = parse_action_events(row["input"])
        targets = action_targets(row)
        selected_indices = align_target_events(events, targets)
        if selected_indices is None:
            stats["skipped_non_monotonic"] += 1
            continue
        selected = choose_action_negative(events, selected_indices, set(targets), seed, row_index)
        if selected is None:
            stats["skipped_no_negative"] += 1
            continue
        negative, tier = selected
        rejected_events = [(index, events[index].token) for index in selected_indices]
        rejected_events.append((negative.index, negative.token))
        rejected_events.sort(key=lambda value: value[0])
        rejected_targets = [token for _, token in rejected_events]
        if len(rejected_targets) != len(targets) + 1 or len(rejected_targets) != len(set(rejected_targets)):
            raise AssertionError("action rejected target count/uniqueness failed")
        rejected = replace_response_body(row["output"], json.dumps(rejected_targets, ensure_ascii=False))
        if action_targets({**row, "output": rejected}) != rejected_targets:
            raise AssertionError("action rejected response failed round trip")
        split = split_name(seed, "action", row["instruction"], row["input"])
        action_id = stable_hash("action", row["instruction"], row["input"])[:16]
        pairs.append(
            {
                "instruction": row["instruction"],
                "input": row["input"],
                "chosen": row["output"],
                "rejected": rejected,
                "meta": {
                    "source": "D(O1):data_seed_clean_v1",
                    "source_row": row_index,
                    "task": "action",
                    "split": split,
                    "prompt_group": action_id,
                    "gold_count": len(targets),
                    "rejected_count": len(rejected_targets),
                    "rejected_extra": negative.token,
                    "negative_event_index": negative.index,
                    "negative_tier": tier,
                },
            }
        )
        stats["eligible_rows"] += 1
        stats[f"split_{split}"] += 1
        tier_counts[tier] += 1

    stats["negative_tiers"] = dict(sorted(tier_counts.items()))
    return pairs, dict(sorted(stats.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return sha256(path)


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["meta"]["task"] for row in rows).items()))


def tier_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["meta"]["negative_tier"] for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--holdout-out", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260821)
    parser.add_argument("--train-tasks", choices=("rec", "all"), default="rec")
    args = parser.parse_args()

    rows = load_jsonl(args.src)
    if len(rows) != 32_480:
        raise AssertionError(f"expected 32,480 O1-derived rows, got {len(rows)}")
    source_hash = sha256(args.src)
    if source_hash != "e526caea4a1afd8befbd5d266fb80d0378a5bf7eff90fdacd14934332d64d309":
        raise AssertionError(f"unexpected data_seed_clean_v1 hash: {source_hash}")

    rec_pairs, rec_audit = build_rec_pairs(rows, args.seed)
    action_pairs, action_audit = build_action_pairs(rows, args.seed)
    all_pairs = rec_pairs + action_pairs
    if len(rec_pairs) != 17_019 or len(action_pairs) != 1_539:
        raise AssertionError(
            f"preference eligibility signature drifted: rec={len(rec_pairs)}, action={len(action_pairs)}"
        )

    train_candidates = [row for row in all_pairs if row["meta"]["split"] == "train"]
    if args.train_tasks == "rec":
        train_rows = [row for row in train_candidates if row["meta"]["task"].startswith("rec_")]
    else:
        train_rows = train_candidates
    holdout_rows = [row for row in all_pairs if row["meta"]["split"] == "holdout"]
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed + 1).shuffle(holdout_rows)

    train_groups = {row["meta"]["prompt_group"] for row in train_rows}
    holdout_groups = {row["meta"]["prompt_group"] for row in holdout_rows}
    overlap = train_groups & holdout_groups
    if overlap:
        raise AssertionError(f"train/holdout prompt leakage: {sorted(overlap)[:5]}")
    if any(row["chosen"] == row["rejected"] for row in all_pairs):
        raise AssertionError("identical chosen/rejected response detected")

    train_hash = write_jsonl(args.train_out, train_rows)
    holdout_hash = write_jsonl(args.holdout_out, holdout_rows)
    audit = {
        "asset_class": {
            "train": "D(O1)",
            "holdout": "E(D(O1))",
        },
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "seed": args.seed,
        "upstream": {
            "asset_id": "D(O1):data_seed_clean_v1",
            "ultimate_official_asset_id": "O1",
            "path": str(args.src.resolve()),
            "rows": len(rows),
            "sha256": source_hash,
        },
        "construction": {
            "recommendation": rec_audit,
            "action": action_audit,
            "teacher_rows": 0,
            "model_rollout_rows": 0,
            "third_party_rows": 0,
            "evaluation_source_rows": 0,
        },
        "split": {
            "method": "sha256 prompt-group modulo 10; bucket 0 holdout",
            "formal_train_tasks": args.train_tasks,
            "train_candidate_rows_before_task_gate": len(train_candidates),
            "withheld_action_train_candidates": (
                sum(row["meta"]["task"] == "action" for row in train_candidates)
                if args.train_tasks == "rec"
                else 0
            ),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_ratio": round(len(train_rows) / len(all_pairs), 8),
            "holdout_ratio": round(len(holdout_rows) / len(all_pairs), 8),
            "train_prompt_groups": len(train_groups),
            "holdout_prompt_groups": len(holdout_groups),
            "prompt_group_overlap": len(overlap),
        },
        "train_mix": {
            "rows": len(train_rows),
            "task_counts": task_counts(train_rows),
            "task_ratios": {
                task: round(count / len(train_rows), 8)
                for task, count in task_counts(train_rows).items()
            },
            "negative_tiers": tier_counts(train_rows),
            "O1_derived_ratio": 1.0,
            "T_ratio": 0.0,
            "E_ratio": 0.0,
        },
        "holdout_mix": {
            "rows": len(holdout_rows),
            "task_counts": task_counts(holdout_rows),
            "negative_tiers": tier_counts(holdout_rows),
        },
        "outputs": {
            "train": {
                "path": str(args.train_out.resolve()),
                "rows": len(train_rows),
                "sha256": train_hash,
            },
            "holdout": {
                "path": str(args.holdout_out.resolve()),
                "rows": len(holdout_rows),
                "sha256": holdout_hash,
            },
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
