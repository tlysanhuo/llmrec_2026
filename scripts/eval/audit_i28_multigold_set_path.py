#!/usr/bin/env python3
"""Audit I-28 checkpoints with prompt-disjoint multi-gold set probability.

For every frozen gate group, this evaluator renders the exact ``/no_think``
generation prompt, appends the video-domain marker, and teacher-forces every
known non-history gold path.  Only the three autoregressive SID tokens
``s_a,s_b,s_c`` are scored; the domain marker is supplied by the evaluator.

The report compares the frozen I-23 parent with I-28 step 64 and step 128 by:

* ``logsumexp`` over all known-gold path log probabilities (set mass);
* ``logmeanexp`` (set mass normalized by the number of known golds); and
* the best individual gold-path log probability.

This is a prompt-disjoint checkpoint-selection diagnostic, not an online-score
estimate.  It reads only the E-class gate, its build audit, O6, and adapters.
It never reads, rewrites, or emits I-28 training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_PARENT = ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
DEFAULT_RUN = ROOT / "checkpoints/i28_i23_rec_multigold_proposal_retkl_v1"
DEFAULT_STEP64 = DEFAULT_RUN / "checkpoint-64"
DEFAULT_STEP128 = DEFAULT_RUN / "checkpoint-128"
DEFAULT_GATE = (
    ROOT / "assets/evaluation/holdout/data_i28_video_multigold_proposal_v1_gate.jsonl"
)
DEFAULT_BUILD_AUDIT = ROOT / "logs/data/i28_video_multigold_proposal_v1_audit.json"
OUTPUT_ROOT = ROOT / "logs/probe"

SCHEMA_GATE = "i28-video-multigold-proposal-gate-v1"
GATE_ROUTE = "gate_only"
NORMALIZED_PROMPT_SCHEMA = "i28-mode-normalized-prompt-v1"
PROPOSAL_ASSET_ID = "D(O1):data_seed_clean_v1"
EXPECTED_GROUPS = 128
VIDEO_MARKER = "<|video_begin|>"
EMPTY_THINK_PREFIX = "<think>\n\n</think>\n"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
ITEMIC_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)

GATE_KEYS = {
    "schema_version",
    "route",
    "group_id",
    "prompt_group_id",
    "normalized_prompt_sha256",
    "domain",
    "upstream_ids",
    "instruction",
    "input",
    "history",
    "gold_count",
    "non_history_gold_count",
    "all_o1_gold_count",
    "history_overlap_gold_count",
    "primary_reward_excludes_history_golds",
    "golds",
}
GOLD_KEYS = {"itemic", "itemic_sha256"}
ADAPTERS = ("parent", "step64", "step128")
METRICS = ("set_logsumexp", "normalized_logmeanexp", "best_gold_logprob")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank gate line at {line_number}")
            try:
                value = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid gate JSON at line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"gate line {line_number} is not an object")
            rows.append(value)
    return rows


def require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{context} schema drifted: missing={missing} extra={extra}")


def require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} is not a lowercase SHA256: {value!r}")
    return value


def require_plain_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} is not an integer: {value!r}")
    return value


def mode_normalized_input(value: str) -> str:
    return MODE_SUFFIX_RE.sub("", value.rstrip())


def normalized_prompt_sha256(row: dict[str, Any]) -> str:
    return text_sha256(
        canonical_json(
            {
                "schema_version": NORMALIZED_PROMPT_SCHEMA,
                "instruction": row["instruction"],
                "input_core": mode_normalized_input(row["input"]),
                "history": row["history"],
            }
        )
    )


def parse_itemic(value: str) -> tuple[int, int, int]:
    match = ITEMIC_RE.fullmatch(value)
    if match is None or match.group("domain") != "video":
        raise ValueError(f"gate gold is not an exact video itemic: {value!r}")
    abc = tuple(int(match.group(part)) for part in "abc")
    if any(number < 0 or number > 8191 for number in abc):
        raise ValueError(f"gate gold SID is outside [0,8191]: {value!r}")
    return abc  # type: ignore[return-value]


def validate_gate_rows(
    rows: list[dict[str, Any]], expected_groups: int = EXPECTED_GROUPS
) -> dict[str, Any]:
    if len(rows) != expected_groups:
        raise ValueError(f"expected {expected_groups} gate groups, got {len(rows)}")

    group_ids: set[str] = set()
    prompt_ids: set[str] = set()
    normalized_hashes: set[str] = set()
    gold_hashes: set[str] = set()
    gold_histogram: Counter[str] = Counter()
    all_gold_count = 0
    excluded_history_count = 0
    prompt_itemic_counts: list[int] = []

    for index, row in enumerate(rows):
        context = f"gate row {index}"
        require_exact_keys(row, GATE_KEYS, context)
        if row["schema_version"] != SCHEMA_GATE or row["route"] != GATE_ROUTE:
            raise ValueError(f"{context} schema/route drifted")
        if row["domain"] != "video":
            raise ValueError(f"{context} is not video")
        if row["upstream_ids"] != [PROPOSAL_ASSET_ID]:
            raise ValueError(f"{context} upstream_ids drifted: {row['upstream_ids']!r}")
        if not isinstance(row["instruction"], str) or not row["instruction"]:
            raise ValueError(f"{context} has an invalid instruction")
        if not isinstance(row["input"], str) or not row["input"].rstrip().endswith(
            "/no_think"
        ):
            raise ValueError(f"{context} input is not an explicit /no_think prompt")
        if row["history"] != []:
            raise ValueError(f"{context} history column must be empty")
        if row["primary_reward_excludes_history_golds"] is not True:
            raise ValueError(f"{context} does not exclude history golds")

        group_id = require_sha256(row["group_id"], f"{context}.group_id")
        prompt_id = require_sha256(row["prompt_group_id"], f"{context}.prompt_group_id")
        normalized_hash = require_sha256(
            row["normalized_prompt_sha256"], f"{context}.normalized_prompt_sha256"
        )
        if normalized_hash != normalized_prompt_sha256(row):
            raise ValueError(f"{context} normalized prompt hash does not reproduce")
        if group_id in group_ids or prompt_id in prompt_ids or normalized_hash in normalized_hashes:
            raise ValueError(f"{context} duplicates a gate group or prompt")
        group_ids.add(group_id)
        prompt_ids.add(prompt_id)
        normalized_hashes.add(normalized_hash)

        gold_count = require_plain_int(row["gold_count"], f"{context}.gold_count")
        non_history = require_plain_int(
            row["non_history_gold_count"], f"{context}.non_history_gold_count"
        )
        all_known = require_plain_int(
            row["all_o1_gold_count"], f"{context}.all_o1_gold_count"
        )
        excluded = require_plain_int(
            row["history_overlap_gold_count"], f"{context}.history_overlap_gold_count"
        )
        if gold_count < 2 or non_history != gold_count or all_known != gold_count + excluded:
            raise ValueError(
                f"{context} gold-count invariant drifted: "
                f"{gold_count}/{non_history}/{all_known}/{excluded}"
            )
        golds = row["golds"]
        if not isinstance(golds, list) or len(golds) != gold_count:
            raise ValueError(f"{context} gold list/count mismatch")

        prompt_text = row["instruction"] + "\n" + row["input"]
        prompt_itemics = set(ITEMIC_RE.findall(prompt_text))
        prompt_itemic_counts.append(len(prompt_itemics))
        itemics: list[str] = []
        row_hashes: set[str] = set()
        for gold_index, gold in enumerate(golds):
            gold_context = f"{context}.golds[{gold_index}]"
            if not isinstance(gold, dict):
                raise ValueError(f"{gold_context} is not an object")
            require_exact_keys(gold, GOLD_KEYS, gold_context)
            itemic = gold["itemic"]
            if not isinstance(itemic, str):
                raise ValueError(f"{gold_context}.itemic is not a string")
            parse_itemic(itemic)
            itemic_hash = require_sha256(gold["itemic_sha256"], f"{gold_context}.hash")
            if itemic_hash != text_sha256(itemic):
                raise ValueError(f"{gold_context} itemic hash does not reproduce")
            if itemic in prompt_text:
                raise ValueError(f"{gold_context} appears in the immutable prompt")
            if itemic_hash in row_hashes:
                raise ValueError(f"{gold_context} duplicates a known gold")
            row_hashes.add(itemic_hash)
            gold_hashes.add(itemic_hash)
            itemics.append(itemic)
        if itemics != sorted(itemics):
            raise ValueError(f"{context} gold list is not deterministically sorted")

        gold_histogram[str(gold_count)] += 1
        all_gold_count += all_known
        excluded_history_count += excluded

    if [row["group_id"] for row in rows] != sorted(row["group_id"] for row in rows):
        raise ValueError("gate rows are not sorted by group_id")
    return {
        "groups": len(rows),
        "non_history_gold_paths": sum(int(row["gold_count"]) for row in rows),
        "all_o1_known_gold_paths": all_gold_count,
        "history_overlap_gold_paths_excluded": excluded_history_count,
        "gold_count_histogram": dict(
            sorted(gold_histogram.items(), key=lambda item: int(item[0]))
        ),
        "unique_gold_hashes": len(gold_hashes),
        "prompt_itemic_count_min": min(prompt_itemic_counts),
        "prompt_itemic_count_max": max(prompt_itemic_counts),
    }


def validate_build_audit(
    audit: dict[str, Any], gate_path: Path, gate_sha256: str, gate_summary: dict[str, Any]
) -> dict[str, Any]:
    try:
        split = audit["split"]
        gate = audit["gate"]
        output = audit["outputs"]["gate"]
        asset_class = audit["asset_class"]["gate"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"I-28 build audit is missing a required section: {error}") from error
    overlap_keys = (
        "train_gate_normalized_prompt_overlap",
        "train_i27_normalized_prompt_overlap",
        "gate_i27_normalized_prompt_overlap",
        "complete_train_gate_normalized_prompt_overlap",
        "complete_train_i27_normalized_prompt_overlap",
    )
    overlap = {}
    for key in overlap_keys:
        value = require_plain_int(split.get(key), f"build_audit.split.{key}")
        if value != 0:
            raise ValueError(f"prompt-disjoint build gate failed: {key}={value}")
        overlap[key] = value
    if split.get("gate_groups") != EXPECTED_GROUPS:
        raise ValueError(f"build audit gate_groups drifted: {split.get('gate_groups')!r}")
    if gate.get("rows") != EXPECTED_GROUPS or gate.get("groups") != EXPECTED_GROUPS:
        raise ValueError("build audit gate row/group count drifted")
    if gate.get("training_allowed") is not False:
        raise ValueError("build audit no longer marks the E gate training-forbidden")
    if gate.get("primary_non_history_gold_count") != gate_summary["non_history_gold_paths"]:
        raise ValueError("build audit primary gold count differs from gate JSONL")
    if gate.get("all_o1_known_gold_count") != gate_summary["all_o1_known_gold_paths"]:
        raise ValueError("build audit all-known gold count differs from gate JSONL")
    if (
        gate.get("history_overlap_gold_count_excluded_from_primary_reward")
        != gate_summary["history_overlap_gold_paths_excluded"]
    ):
        raise ValueError("build audit excluded-history gold count differs from gate JSONL")
    if gate.get("primary_non_history_gold_count_histogram") != gate_summary[
        "gold_count_histogram"
    ]:
        raise ValueError("build audit gate gold histogram differs from gate JSONL")
    if output.get("rows") != EXPECTED_GROUPS or output.get("sha256") != gate_sha256:
        raise ValueError("build audit gate output hash/row count drifted")
    if Path(str(output.get("path"))).resolve() != gate_path.resolve():
        raise ValueError("build audit gate output path differs from evaluator input")
    if not isinstance(asset_class, str) or "diagnostic only" not in asset_class:
        raise ValueError("build audit does not classify gate as diagnostic-only E data")
    return overlap


def render_prompt(row: dict[str, Any]) -> str:
    if row["history"] != [] or not row["input"].rstrip().endswith("/no_think"):
        raise ValueError("gate prompt violates empty-history /no_think contract")
    query = "\n".join(value for value in (row["instruction"], row["input"]) if value)
    prompt = (
        "<|im_start|>user\n"
        + query
        + "<|im_end|>\n<|im_start|>assistant\n"
        + EMPTY_THINK_PREFIX
        + VIDEO_MARKER
    )
    if not prompt.endswith(VIDEO_MARKER):
        raise AssertionError("rendered I-28 prompt lacks the explicit video marker")
    return prompt


def encode_items(tokenizer: Any, rows: list[dict[str, Any]], cutoff_len: int) -> list[dict[str, Any]]:
    video_id = int(tokenizer.convert_tokens_to_ids(VIDEO_MARKER))
    if video_id != 176245:
        raise ValueError(f"O6 video marker ID drifted: {video_id}/176245")
    items: list[dict[str, Any]] = []
    for group_index, row in enumerate(rows):
        prompt = render_prompt(row)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids or prompt_ids[-1] != video_id:
            raise ValueError(f"group {row['group_id']} prompt does not end in video marker")
        for gold_index, gold in enumerate(row["golds"]):
            abc = parse_itemic(gold["itemic"])
            target_text = "".join(f"<s_{part}_{value}>" for part, value in zip("abc", abc))
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            if len(target_ids) != 3:
                raise ValueError(f"gold path is not exactly three atomic SID tokens: {target_ids}")
            if not (
                151669 <= target_ids[0] <= 159860
                and 159861 <= target_ids[1] <= 168052
                and 168053 <= target_ids[2] <= 176244
            ):
                raise ValueError(f"gold path token ranges drifted: {target_ids}")
            full_ids = tokenizer.encode(prompt + target_text, add_special_tokens=False)
            if full_ids != prompt_ids + target_ids:
                raise ValueError("tokenization crosses the video-marker/SID boundary")
            if len(full_ids) > cutoff_len:
                raise ValueError(
                    f"group {row['group_id']} exceeds cutoff: {len(full_ids)}/{cutoff_len}"
                )
            items.append(
                {
                    "group_index": group_index,
                    "gold_index": gold_index,
                    "group_id": row["group_id"],
                    "gold_sha256": gold["itemic_sha256"],
                    "ids": full_ids,
                    "targets": target_ids,
                    "scores": {},
                }
            )
    return items


def make_batches(
    items: list[dict[str, Any]], max_batch_size: int, batch_token_budget: int
) -> Iterable[list[dict[str, Any]]]:
    if max_batch_size <= 0 or batch_token_budget <= 0:
        raise ValueError("batch caps must be positive")
    pending = sorted(
        items,
        key=lambda item: (-len(item["ids"]), item["group_id"], item["gold_index"]),
    )
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
            if not batch and len(candidate["ids"]) > batch_token_budget:
                raise ValueError(
                    f"one encoded path exceeds batch token budget: "
                    f"{len(candidate['ids'])}/{batch_token_budget}"
                )
            batch.append(candidate)
            maximum = candidate_maximum
            cursor += 1
        if not batch:
            raise AssertionError("batch planner made no progress")
        yield batch


def score_batches(
    model: Any,
    tokenizer: Any,
    adapters: list[tuple[str, Path]],
    items: list[dict[str, Any]],
    max_batch_size: int,
    batch_token_budget: int,
) -> int:
    import torch
    import torch.nn.functional as functional

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad nor EOS token")

    batches = list(make_batches(items, max_batch_size, batch_token_budget))
    with torch.inference_mode():
        # The frozen protocol requires the complete parent pass before any
        # candidate score is computed, then step64 before step128.
        for name, _ in adapters:
            model.set_adapter(name)
            for batch_index, batch in enumerate(batches, start=1):
                maximum = max(len(item["ids"]) for item in batch)
                input_ids = torch.tensor(
                    [
                        [pad_id] * (maximum - len(item["ids"])) + item["ids"]
                        for item in batch
                    ],
                    dtype=torch.long,
                    device="cuda",
                )
                attention_mask = torch.tensor(
                    [
                        [0] * (maximum - len(item["ids"])) + [1] * len(item["ids"])
                        for item in batch
                    ],
                    dtype=torch.long,
                    device="cuda",
                )
                position_ids = attention_mask.cumsum(dim=-1) - 1
                position_ids.masked_fill_(attention_mask.eq(0), 0)
                prediction_positions = torch.arange(
                    maximum - 4, maximum - 1, device="cuda"
                )
                targets = input_ids[:, -3:]
                expected_targets = torch.tensor(
                    [item["targets"] for item in batch], dtype=torch.long, device="cuda"
                )
                if not torch.equal(targets, expected_targets):
                    raise RuntimeError(
                        "left-padded batch no longer aligns the three SID targets"
                    )

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    logits_to_keep=prediction_positions,
                ).logits
                if tuple(logits.shape[:2]) != (len(batch), 3):
                    raise RuntimeError(
                        f"partial-logit shape mismatch for {name}: {tuple(logits.shape)}"
                    )
                token_losses = functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(len(batch), 3)
                path_logps = -token_losses.sum(dim=-1)
                if not torch.isfinite(path_logps).all():
                    raise RuntimeError(f"non-finite path log probability for {name}")
                for item, value in zip(batch, path_logps.detach().cpu().tolist()):
                    item["scores"][name] = float(value)
                del logits, token_losses, path_logps
                del input_ids, attention_mask, position_ids, prediction_positions, targets
                if (
                    batch_index == 1
                    or batch_index % 20 == 0
                    or batch_index == len(batches)
                ):
                    print(
                        f"[i28-set-path] {name} batch {batch_index}/{len(batches)}",
                        flush=True,
                    )
    return len(batches)


def stable_logsumexp(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("set aggregation requires finite non-empty path scores")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def set_metrics(path_scores: list[float]) -> dict[str, float]:
    set_logprob = stable_logsumexp(path_scores)
    result = {
        "set_logsumexp": set_logprob,
        "normalized_logmeanexp": set_logprob - math.log(len(path_scores)),
        "best_gold_logprob": max(path_scores),
    }
    if result["set_logsumexp"] + 1e-12 < result["best_gold_logprob"]:
        raise AssertionError("set logsumexp is below its best member")
    if result["normalized_logmeanexp"] - 1e-12 > result["best_gold_logprob"]:
        raise AssertionError("logmeanexp exceeds the best member")
    if result["set_logsumexp"] > 1e-4:
        raise RuntimeError(f"known disjoint path mass exceeds one: {result['set_logsumexp']}")
    return result


def rounded(value: float) -> float:
    return round(float(value), 8)


def absolute_summary(group_metrics: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"groups": len(group_metrics)}
    for metric in METRICS:
        values = [row[metric] for row in group_metrics]
        summary[metric] = {
            "mean": rounded(statistics.fmean(values)),
            "median": rounded(statistics.median(values)),
            "min": rounded(min(values)),
            "max": rounded(max(values)),
        }
    return summary


def delta_summary(
    candidate: list[dict[str, float]], parent: list[dict[str, float]]
) -> dict[str, Any]:
    if len(candidate) != len(parent) or not candidate:
        raise ValueError("paired group metrics are empty or misaligned")
    result: dict[str, Any] = {"groups": len(candidate), "improvement_definition": "delta > 0"}
    for metric in METRICS:
        deltas = [left[metric] - right[metric] for left, right in zip(candidate, parent)]
        result[metric] = {
            "delta_mean": rounded(statistics.fmean(deltas)),
            "delta_median": rounded(statistics.median(deltas)),
            "improved_groups": sum(delta > 0 for delta in deltas),
            "improved_rate": rounded(sum(delta > 0 for delta in deltas) / len(deltas)),
            "unchanged_groups": sum(delta == 0 for delta in deltas),
        }
    return result


def aggregate_results(
    rows: list[dict[str, Any]], items: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if set(item["scores"]) != set(ADAPTERS):
            raise RuntimeError(
                f"missing adapter scores for {item['group_id']}: {sorted(item['scores'])}"
            )
        by_group[int(item["group_index"])].append(item)

    metrics_by_model: dict[str, list[dict[str, float]]] = {name: [] for name in ADAPTERS}
    group_report: list[dict[str, Any]] = []
    for group_index, row in enumerate(rows):
        group_items = sorted(by_group[group_index], key=lambda item: item["gold_index"])
        if len(group_items) != row["gold_count"]:
            raise RuntimeError(f"group {row['group_id']} lost a gold path during scoring")
        model_report: dict[str, Any] = {}
        for name in ADAPTERS:
            path_scores = [item["scores"][name] for item in group_items]
            metrics = set_metrics(path_scores)
            metrics_by_model[name].append(metrics)
            best_index = max(range(len(path_scores)), key=path_scores.__getitem__)
            model_report[name] = {
                **{metric: rounded(metrics[metric]) for metric in METRICS},
                "best_gold_itemic_sha256": group_items[best_index]["gold_sha256"],
            }
            if name != "parent":
                parent_metrics = metrics_by_model["parent"][-1]
                model_report[name]["delta_vs_parent"] = {
                    metric: rounded(metrics[metric] - parent_metrics[metric])
                    for metric in METRICS
                }
        group_report.append(
            {
                "group_id": row["group_id"],
                "normalized_prompt_sha256": row["normalized_prompt_sha256"],
                "gold_count": row["gold_count"],
                "models": model_report,
            }
        )

    models = {}
    for name in ADAPTERS:
        models[name] = {"absolute": absolute_summary(metrics_by_model[name])}
        if name != "parent":
            models[name]["delta_vs_parent"] = delta_summary(
                metrics_by_model[name], metrics_by_model["parent"]
            )
    return models, group_report


def adapter_identity(name: str, path: Path) -> dict[str, str]:
    model_file = path / "adapter_model.safetensors"
    config_file = path / "adapter_config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"{name} adapter is incomplete: {path}")
    return {
        "path": str(path.resolve()),
        "adapter_model_sha256": file_sha256(model_file),
        "adapter_config_sha256": file_sha256(config_file),
    }


def canonical_adapter_structure_value(key: str, value: Any) -> Any:
    """Canonicalize PEFT fields whose JSON list order has no model semantics."""
    if key in {"target_modules", "modules_to_save"} and isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} contains a non-string entry: {value!r}")
        if len(value) != len(set(value)):
            raise ValueError(f"{key} contains duplicate entries: {value!r}")
        return sorted(value)
    return value


def validate_adapter_configs(adapters: list[tuple[str, Path]]) -> None:
    structural_keys = (
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "modules_to_save",
    )
    configs = {name: load_json(path / "adapter_config.json") for name, path in adapters}
    parent = configs["parent"]
    for name in ("step64", "step128"):
        drift = {
            key: (
                canonical_adapter_structure_value(key, configs[name].get(key)),
                canonical_adapter_structure_value(key, parent.get(key)),
            )
            for key in structural_keys
            if canonical_adapter_structure_value(key, configs[name].get(key))
            != canonical_adapter_structure_value(key, parent.get(key))
        }
        if drift:
            raise ValueError(f"{name} adapter structure differs from I-23 parent: {drift}")
    if parent.get("r") != 64 or parent.get("lora_alpha") != 64:
        raise ValueError("I-28 evaluator requires the I-23 r64/alpha64 adapter family")


def validate_output_path(path: Path, overwrite: bool) -> Path:
    resolved = path.resolve()
    output_root = OUTPUT_ROOT.resolve()
    if resolved.parent != output_root or resolved.suffix != ".json":
        raise ValueError(f"report must be a direct logs/probe/*.json file: {resolved}")
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite report without --overwrite: {resolved}")
    return resolved


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_self_test() -> None:
    prompt_row = {
        "instruction": "预测下一物品",
        "input": "历史为 <|video_begin|><s_a_9><s_b_8><s_c_7>/no_think",
        "history": [],
    }
    normalized = normalized_prompt_sha256(prompt_row)
    golds = [
        "<|video_begin|><s_a_1><s_b_2><s_c_3>",
        "<|video_begin|><s_a_4><s_b_5><s_c_6>",
    ]
    row = {
        "schema_version": SCHEMA_GATE,
        "route": GATE_ROUTE,
        "group_id": "1" * 64,
        "prompt_group_id": "2" * 64,
        "normalized_prompt_sha256": normalized,
        "domain": "video",
        "upstream_ids": [PROPOSAL_ASSET_ID],
        **prompt_row,
        "gold_count": 2,
        "non_history_gold_count": 2,
        "all_o1_gold_count": 3,
        "history_overlap_gold_count": 1,
        "primary_reward_excludes_history_golds": True,
        "golds": [
            {"itemic": gold, "itemic_sha256": text_sha256(gold)} for gold in golds
        ],
    }
    summary = validate_gate_rows([row], expected_groups=1)
    assert summary["non_history_gold_paths"] == 2
    assert render_prompt(row).endswith(EMPTY_THINK_PREFIX + VIDEO_MARKER)
    assert parse_itemic(golds[0]) == (1, 2, 3)

    fake_gate = Path("/tmp/i28-self-test-gate.jsonl")
    fake_gate_sha = "3" * 64
    fake_audit = {
        "asset_class": {"gate": "E(D(O1)); diagnostic only; forbidden for training"},
        "split": {
            "gate_groups": EXPECTED_GROUPS,
            "train_gate_normalized_prompt_overlap": 0,
            "train_i27_normalized_prompt_overlap": 0,
            "gate_i27_normalized_prompt_overlap": 0,
            "complete_train_gate_normalized_prompt_overlap": 0,
            "complete_train_i27_normalized_prompt_overlap": 0,
        },
        "gate": {
            "rows": EXPECTED_GROUPS,
            "groups": EXPECTED_GROUPS,
            "primary_non_history_gold_count": summary["non_history_gold_paths"],
            "all_o1_known_gold_count": summary["all_o1_known_gold_paths"],
            "history_overlap_gold_count_excluded_from_primary_reward": summary[
                "history_overlap_gold_paths_excluded"
            ],
            "primary_non_history_gold_count_histogram": summary["gold_count_histogram"],
            "training_allowed": False,
        },
        "outputs": {
            "gate": {
                "path": str(fake_gate.resolve()),
                "rows": EXPECTED_GROUPS,
                "sha256": fake_gate_sha,
            }
        },
    }
    overlap = validate_build_audit(fake_audit, fake_gate, fake_gate_sha, summary)
    assert set(overlap.values()) == {0}
    fake_audit["split"]["gate_i27_normalized_prompt_overlap"] = 1
    try:
        validate_build_audit(fake_audit, fake_gate, fake_gate_sha, summary)
    except ValueError as error:
        assert "prompt-disjoint" in str(error)
    else:
        raise AssertionError("build-audit self-test accepted prompt overlap")

    malformed = dict(row)
    malformed["history"] = [["x", "y"]]
    try:
        validate_gate_rows([malformed], expected_groups=1)
    except ValueError as error:
        assert "history column" in str(error)
    else:
        raise AssertionError("gate schema self-test did not reject non-empty history")

    metrics = set_metrics([-1.0, -2.0])
    expected_lse = math.log(math.exp(-1.0) + math.exp(-2.0))
    assert math.isclose(metrics["set_logsumexp"], expected_lse, abs_tol=1e-12)
    assert math.isclose(
        metrics["normalized_logmeanexp"], expected_lse - math.log(2), abs_tol=1e-12
    )
    assert metrics["best_gold_logprob"] == -1.0
    parent = [set_metrics([-2.0, -3.0]), set_metrics([-4.0, -5.0])]
    candidate = [set_metrics([-1.0, -3.0]), set_metrics([-5.0, -6.0])]
    deltas = delta_summary(candidate, parent)
    assert deltas["set_logsumexp"]["improved_groups"] == 1
    assert deltas["set_logsumexp"]["improved_rate"] == 0.5

    fake_items = [
        {"ids": list(range(length)), "group_id": str(index), "gold_index": 0}
        for index, length in enumerate((10, 9, 7, 6))
    ]
    batches = list(make_batches(fake_items, max_batch_size=2, batch_token_budget=20))
    assert [len(batch) for batch in batches] == [2, 2]
    assert reject_duplicate_keys([("a", 1)]) == {"a": 1}
    try:
        reject_duplicate_keys([("a", 1), ("a", 2)])
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate-key self-test failed")
    assert canonical_adapter_structure_value(
        "target_modules", ["q_proj", "k_proj"]
    ) == canonical_adapter_structure_value("target_modules", ["k_proj", "q_proj"])
    try:
        canonical_adapter_structure_value("target_modules", ["q_proj", "q_proj"])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("adapter-structure self-test accepted duplicate target modules")
    print(
        "[i28-set-path] self-test passed: strict gate schema, non-history checks, "
        "explicit /no_think video prompt, set metrics/deltas, batch planner, and "
        "duplicate-key/adapter-structure rejection"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare I-23, I-28 step64, and I-28 step128 on the frozen "
            "prompt-disjoint multi-gold three-SID set path."
        )
    )
    parser.add_argument("--self-test", action="store_true", help="run CPU-only unit tests")
    parser.add_argument("--out", type=Path, help="report path under logs/probe/*.json")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--step64", type=Path, default=DEFAULT_STEP64)
    parser.add_argument("--step128", type=Path, default=DEFAULT_STEP128)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--build-audit", type=Path, default=DEFAULT_BUILD_AUDIT)
    parser.add_argument("--gpu", default="0", help="one GPU index or full UUID")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-token-budget", type=int, default=65536)
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=19260828)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.out is None:
        parser.error("--out is required unless --self-test is used")
    if not args.gpu or "," in args.gpu:
        parser.error("--gpu must expose exactly one GPU")
    if args.max_batch_size <= 0 or args.batch_token_budget <= 0 or args.cutoff_len <= 3:
        parser.error("batch and cutoff arguments must be positive")

    output_path = validate_output_path(args.out, args.overwrite)
    for label, path in (
        ("base", args.base),
        ("gate", args.gate),
        ("build audit", args.build_audit),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
    base_config = args.base / "config.json"
    if not base_config.is_file():
        raise FileNotFoundError(base_config)

    adapters = [("parent", args.parent), ("step64", args.step64), ("step128", args.step128)]
    if len({path.resolve() for _, path in adapters}) != len(adapters):
        raise ValueError("parent/step64/step128 adapter paths must be distinct")
    adapter_hashes = {name: adapter_identity(name, path) for name, path in adapters}
    validate_adapter_configs(adapters)

    gate_sha = file_sha256(args.gate)
    rows = load_jsonl(args.gate)
    gate_summary = validate_gate_rows(rows)
    build_audit_sha = file_sha256(args.build_audit)
    build_audit = load_json(args.build_audit)
    overlap = validate_build_audit(build_audit, args.gate, gate_sha, gate_summary)
    manifest = [
        {
            "group_id": row["group_id"],
            "normalized_prompt_sha256": row["normalized_prompt_sha256"],
            "gold_itemic_sha256": [gold["itemic_sha256"] for gold in row["golds"]],
        }
        for row in rows
    ]
    manifest_sha = text_sha256(canonical_json(manifest))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"I-28 set-path audit requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    items = encode_items(tokenizer, rows, args.cutoff_len)
    if len(items) != gate_summary["non_history_gold_paths"]:
        raise RuntimeError("encoded gold-path count differs from validated gate")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        args.parent,
        adapter_name="parent",
        is_trainable=False,
        autocast_adapter_dtype=True,
    ).eval()
    model.load_adapter(
        args.step64,
        adapter_name="step64",
        is_trainable=False,
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=True,
    )
    model.load_adapter(
        args.step128,
        adapter_name="step128",
        is_trainable=False,
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=True,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    batch_count = score_batches(
        model,
        tokenizer,
        adapters,
        items,
        args.max_batch_size,
        args.batch_token_budget,
    )
    model_summaries, group_report = aggregate_results(rows, items)
    for name, _ in adapters:
        model_summaries[name]["identity"] = adapter_hashes[name]

    report = {
        "status": "COMPLETE_NOT_AN_ONLINE_SCORE_ESTIMATE",
        "method": {
            "gate_schema": SCHEMA_GATE,
            "route": (
                "exact qwen3_nothink prompt; empty think; evaluator-supplied "
                "<|video_begin|>; teacher-force and score only s_a,s_b,s_c"
            ),
            "path_logprob": "sum of three autoregressive SID token log probabilities",
            "set_logsumexp": "logsumexp over every non-history known-gold path",
            "normalized_logmeanexp": "set_logsumexp - log(number of known gold paths)",
            "best_gold_logprob": "maximum individual known-gold path log probability",
            "stochastic_sampling": False,
            "training_data_accessed": False,
            "training_data_written": False,
            "selection_warning": (
                "Prompt-disjoint mechanism/checkpoint diagnostic only; do not map to online score."
            ),
        },
        "inputs": {
            "gate": {
                "class": "E(D(O1)); diagnostic only; training forbidden",
                "path": str(args.gate.resolve()),
                "sha256": gate_sha,
                "manifest_sha256": manifest_sha,
                **gate_summary,
            },
            "build_audit": {
                "path": str(args.build_audit.resolve()),
                "sha256": build_audit_sha,
                "validated_zero_prompt_overlaps": overlap,
            },
            "base": {
                "path": str(args.base.resolve()),
                "config_sha256": file_sha256(base_config),
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__)),
            },
        },
        "models": model_summaries,
        "groups": group_report,
        "resources": {
            "gpu_count": 1,
            "encoded_gold_paths": len(items),
            "batches": batch_count,
            "max_batch_size": args.max_batch_size,
            "batch_token_budget": args.batch_token_budget,
            "cutoff_len": args.cutoff_len,
            "elapsed_seconds": round(time.time() - started, 3),
            "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        },
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
