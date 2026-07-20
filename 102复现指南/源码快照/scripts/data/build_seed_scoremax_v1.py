#!/usr/bin/env python3
"""Build the O1-only score-max SFT mixture.

The builder preserves every official target. It removes redundant recommendation
CoT exposure, aligns topic generation with the no-think evaluation path, and
adds label-preserving action-select views with chronologically ordered hard
negatives. No new semantic labels are introduced.
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
DEFAULT_SRC = ROOT / "assets/derived/processed/data_final.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_seed_scoremax_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/seed_scoremax_v1_audit.json"
MODEL = ROOT / "models/OneReason-0.8B-pretrain-competition"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
ITEM_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
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
            rows.append(
                {
                    "instruction": str(row.get("instruction", row.get("system", "")) or ""),
                    "input": str(row.get("input", row.get("prompt", "")) or ""),
                    "output": str(row.get("output", row.get("response", "")) or ""),
                    "history": row.get("history") or [],
                }
            )
    return rows


def answer_body(row: dict[str, Any]) -> str:
    if "</think>" not in row["output"]:
        raise ValueError("response is missing </think>")
    return row["output"].split("</think>", 1)[1].lstrip("\n")


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


def to_nothink(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["input"] = MODE_SUFFIX_RE.sub("/no_think", row["input"].rstrip())
    converted["output"] = "<think>\n\n</think>\n" + answer_body(row)
    return converted


def stable_hash(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compress_recommendation_cot(rows: list[dict[str, Any]], seed: int) -> dict[str, int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if task_of(row).startswith("rec_"):
            groups[core_prompt(row)].append(index)

    converted = 0
    singleton_groups = 0
    for key, indexes in groups.items():
        if len(indexes) == 1:
            singleton_groups += 1
            continue
        think_bodies = []
        for index in indexes:
            match = THINK_RE.search(rows[index]["output"])
            if not match or not match.group(1).strip():
                raise AssertionError("O1 recommendation row unexpectedly lacks original CoT")
            think_bodies.append(match.group(1).strip())
        if len(set(think_bodies)) != 1:
            raise AssertionError("recommendation prompt group has non-identical CoTs")

        representative = min(
            indexes,
            key=lambda index: stable_hash(seed, key, answer_body(rows[index])),
        )
        for index in indexes:
            if index != representative:
                rows[index] = to_nothink(rows[index])
                converted += 1

    return {
        "prompt_groups": len(groups),
        "singleton_groups": singleton_groups,
        "duplicate_cot_converted": converted,
    }


def parse_action_events(prompt: str) -> tuple[str, list[Event], str]:
    parts = HISTORY_SPLIT_RE.split(prompt, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("action prompt lacks the history/task boundary")
    history_text, task_text = parts
    lines = history_text.splitlines()
    if not lines or lines[0].strip() != "【用户交互历史】：":
        raise ValueError("action prompt has an unexpected history header")

    events: list[Event] = []
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
        if not current_date:
            raise ValueError("action event appears before a date header")
        token = token_match.group(0)
        action_match = ACTION_RE.search(line)
        domain_match = DOMAIN_RE.search(token)
        if not action_match or not domain_match:
            raise ValueError("action event is missing action/domain metadata")
        events.append(
            Event(
                index=len(events),
                date=current_date,
                line=line.rstrip(),
                token=token,
                action=action_match.group(1),
                domain=domain_match.group(1),
            )
        )
    if not events:
        raise ValueError("action prompt contains no itemic events")
    return lines[0], events, task_text


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
        match = next(
            (event.index for event in events[cursor:] if event.token == target),
            None,
        )
        if match is None:
            return None
        aligned.append(match)
        cursor = match + 1
    return aligned


def choose_hard_negatives(
    events: list[Event],
    selected_indices: list[int],
    selected_tokens: set[str],
    ratio: int,
    seed: int,
    row_index: int,
) -> list[int]:
    selected = [events[index] for index in selected_indices]
    wanted = min(
        max(24, ratio * len(selected_indices)),
        max(0, 192 - len(selected_indices)),
    )
    candidates = [event for event in events if event.token not in selected_tokens]

    def difficulty(event: Event) -> tuple[int, int, int, str]:
        same_action = any(event.action == target.action for target in selected)
        same_domain = any(event.domain == target.domain for target in selected)
        distance = min(abs(event.index - target.index) for target in selected)
        return (
            0 if same_action else 1,
            0 if same_domain else 1,
            distance,
            stable_hash(seed, row_index, ratio, event.index, event.token),
        )

    candidates.sort(key=difficulty)
    return [event.index for event in candidates[:wanted]]


def render_action_view(
    row: dict[str, Any],
    header: str,
    events: list[Event],
    task_text: str,
    included_indices: set[int],
) -> dict[str, Any]:
    lines = [header]
    current_date = None
    included_tokens = set()
    for event in events:
        if event.index not in included_indices:
            continue
        if event.date != current_date:
            lines.append(f"【{event.date}】")
            current_date = event.date
        lines.append(event.line)
        included_tokens.add(event.token)

    targets = action_targets(row)
    if any(target not in included_tokens for target in targets):
        raise AssertionError("action view dropped a gold target")
    view = dict(row)
    view["input"] = "\n".join(lines) + "\n\n" + task_text
    if not view["input"].rstrip().endswith("/no_think"):
        raise AssertionError("action view is not /no_think")
    return view


def augment_action_rows(
    rows: list[dict[str, Any]], ratios: tuple[int, ...], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    augmented = []
    eligible = 0
    skipped_non_monotonic = 0
    view_lengths = Counter()
    for row_index, row in enumerate(rows):
        if task_of(row) != "action":
            continue
        header, events, task_text = parse_action_events(row["input"])
        targets = action_targets(row)
        selected_indices = align_target_events(events, targets)
        if selected_indices is None:
            skipped_non_monotonic += 1
            continue
        eligible += 1
        selected_tokens = set(targets)
        for ratio in ratios:
            negative_indices = choose_hard_negatives(
                events,
                selected_indices,
                selected_tokens,
                ratio,
                seed,
                row_index,
            )
            included = set(selected_indices) | set(negative_indices)
            view = render_action_view(row, header, events, task_text, included)
            augmented.append(view)
            view_lengths[f"ratio_{ratio}_min"] = min(
                view_lengths.get(f"ratio_{ratio}_min", len(included)), len(included)
            )
            view_lengths[f"ratio_{ratio}_max"] = max(
                view_lengths.get(f"ratio_{ratio}_max", 0), len(included)
            )
            view_lengths[f"ratio_{ratio}_total"] += len(included)

    means = {
        f"ratio_{ratio}_mean": round(
            view_lengths[f"ratio_{ratio}_total"] / eligible, 4
        )
        for ratio in ratios
    }
    return augmented, {
        "source_action_rows": sum(task_of(row) == "action" for row in rows),
        "eligible_monotonic_rows": eligible,
        "skipped_non_monotonic_rows": skipped_non_monotonic,
        "views_per_eligible_row": len(ratios),
        "ratios": list(ratios),
        "augmented_rows": len(augmented),
        "view_event_lengths": {**dict(sorted(view_lengths.items())), **means},
    }


def target_token_mix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    counts = Counter()
    total = 0
    for row in rows:
        count = len(tokenizer.encode(row["output"], add_special_tokens=False)) + 1
        counts[task_of(row)] += count
        total += count
    return {
        "total_including_eos": total,
        "by_task": dict(sorted(counts.items())),
        "ratio_by_task": {
            key: round(value / total, 8) for key, value in sorted(counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260817)
    parser.add_argument("--action-negative-ratios", default="3,6")
    args = parser.parse_args()

    ratios = tuple(int(value) for value in args.action_negative_ratios.split(",") if value)
    if not ratios or any(ratio < 1 for ratio in ratios):
        raise ValueError("action negative ratios must be positive integers")

    rows = load_jsonl(args.src)
    if len(rows) != 32_480:
        raise AssertionError(f"expected 32,480 O1 rows, got {len(rows)}")
    source_sha256 = sha256(args.src)
    source_counts = Counter(task_of(row) for row in rows)

    rec_stats = compress_recommendation_cot(rows, args.seed)
    topic_converted = 0
    for index, row in enumerate(rows):
        if task_of(row) == "topic" and not row["input"].rstrip().endswith("/no_think"):
            rows[index] = to_nothink(row)
            topic_converted += 1

    action_views, action_stats = augment_action_rows(rows, ratios, args.seed)
    final_rows = rows + action_views
    final_counts = Counter(task_of(row) for row in final_rows)

    if rec_stats["prompt_groups"] != 6_460 or rec_stats["duplicate_cot_converted"] != 12_744:
        raise AssertionError(f"recommendation grouping signature drifted: {rec_stats}")
    if source_counts["action"] != 1_588 or source_counts["topic"] != 1_304:
        raise AssertionError(f"user task signature drifted: {source_counts}")
    if action_stats["eligible_monotonic_rows"] != 1_539:
        raise AssertionError(f"action monotonic signature drifted: {action_stats}")
    if topic_converted != 602:
        raise AssertionError(f"topic think signature drifted: {topic_converted}")

    rng = random.Random(args.seed)
    rng.shuffle(final_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in final_rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.out)

    output_sha256 = sha256(args.out)
    audit = {
        "asset_class": "D(O1)",
        "builder": str(Path(__file__).resolve()),
        "seed": args.seed,
        "upstream": {
            "asset_id": "O1",
            "path": str(args.src.resolve()),
            "rows": len(rows),
            "sha256": source_sha256,
        },
        "transformations": {
            "recommendation": rec_stats,
            "topic_to_nothink": topic_converted,
            "action_views": action_stats,
        },
        "rows": len(final_rows),
        "row_mix": {
            "O1_preserved": {
                "rows": len(rows),
                "ratio": round(len(rows) / len(final_rows), 8),
            },
            "O1_action_views": {
                "rows": len(action_views),
                "ratio": round(len(action_views) / len(final_rows), 8),
            },
        },
        "source_task_counts": dict(sorted(source_counts.items())),
        "final_task_counts": dict(sorted(final_counts.items())),
        "target_token_mix": target_token_mix(final_rows),
        "output": str(args.out.resolve()),
        "output_sha256": output_sha256,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
