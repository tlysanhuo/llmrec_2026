#!/usr/bin/env python3
"""Build the I-28 group-balanced video proposal/retention dataset and gate.

I-28 is deliberately small enough that a batch-size-one, accumulation-four
run reaches every row exactly once in 128 optimizer steps.  The supervised
branch contains 64 O1 video prompt groups and exactly two deterministic known
positive SIDs per group.  Its response is a unique empty-think signature whose
body is exactly ``domain + s_a + s_b + s_c``.  The other 384 rows are unchanged
teacher-forcing rows from the registered I-12 retention mixture and are used
only for frozen-I-23 KL by the custom trainer.

The 128-group gate, the 64 supervised groups, and every prompt in the I-27
manifest are disjoint by a mode-normalized prompt hash.  I-27 source row
indices are never used for exclusion.  The gate is an E(D(O1)) diagnostic
asset and must never be loaded by a training configuration.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_o1_rec_multigold_v1 import (
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    aggregate_groups,
    canonical_json,
    file_sha256,
    read_source,
    stable_hash,
    text_sha256,
)
from build_seed_scoremax_v1 import load_jsonl, task_of


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROPOSAL_SOURCE = (
    ROOT / "assets/derived/processed/data_seed_clean_v1.jsonl"
)
DEFAULT_RETENTION_SOURCE = (
    ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
)
DEFAULT_I27_MANIFEST = (
    ROOT / "assets/derived/processed/o1_rec_multigold_v1_prompt_manifest.jsonl"
)
DEFAULT_OUT = (
    ROOT
    / "assets/derived/processed/data_i28_video_multigold_proposal_retkl_v1.jsonl"
)
DEFAULT_GATE = (
    ROOT
    / "assets/evaluation/holdout/data_i28_video_multigold_proposal_v1_gate.jsonl"
)
DEFAULT_AUDIT = ROOT / "logs/data/i28_video_multigold_proposal_v1_audit.json"
DEFAULT_TOKENIZER = ROOT / "assets/official/base_model"

PROPOSAL_ASSET_ID = "D(O1):data_seed_clean_v1"
RETENTION_ASSET_ID = (
    "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General):"
    "data_user_residual_retention_v1"
)
I27_MANIFEST_ASSET_ID = "D(O1):o1_rec_multigold_v1_prompt_manifest"
EXPECTED_RETENTION_ROWS = 6_106
EXPECTED_RETENTION_SHA256 = (
    "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0"
)
EXPECTED_I27_ROWS = 512
EXPECTED_I27_SHA256 = (
    "c75e6a326dd02da07b671787a0bbc76cc391c0ec1254a7eaed0fa1cc250d0300"
)

DEFAULT_SEED = 19_260_828
MULTIGOLD_DOMAIN = "video"
MIN_UNIQUE_GOLDS = 2
MIN_NON_HISTORY_GOLDS = 2
TRAIN_GROUPS = 64
GATE_GROUPS = 128
GOLDS_PER_TRAIN_GROUP = 2
PROPOSAL_ROWS = TRAIN_GROUPS * GOLDS_PER_TRAIN_GROUP
RETENTION_QUOTAS = {
    "material_desc2sid": 96,
    "material_sid2desc": 96,
    "action": 32,
    "topic": 32,
    "rec_prod": 32,
    "rec_ad": 32,
    "rec_living": 32,
    "world": 32,
}
RETENTION_ROWS = sum(RETENTION_QUOTAS.values())
EXPECTED_OUTPUT_ROWS = PROPOSAL_ROWS + RETENTION_ROWS

SCHEMA_TRAIN = "i28-video-multigold-proposal-retkl-v1"
SCHEMA_GATE = "i28-video-multigold-proposal-gate-v1"
SCHEMA_RETENTION_GROUP = "i28-retention-prompt-task-v1"
NORMALIZED_PROMPT_SCHEMA = "i28-mode-normalized-prompt-v1"
PROPOSAL_ROUTE = "proposal_ce"
RETENTION_ROUTE = "retention_kl"
GATE_ROUTE = "gate_only"

CORE_KEYS = {"instruction", "input", "output", "history"}
TRAIN_KEYS = CORE_KEYS | {
    "schema_version",
    "route",
    "group_id",
    "domain",
    "task",
    "upstream_ids",
}
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
ITEMIC_TEXT = (
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
STRICT_PROPOSAL_RE = re.compile(
    rf"^<think>\s*</think>\s*(?P<itemic>{ITEMIC_TEXT})\s*$", re.S
)
OPEN_THINK = "<think>"
CLOSE_THINK = "</think>"
WHITESPACE_TOKEN_IDS = {198, 220, 262, 271}

EXPECTED_RETENTION_SOURCE_TASKS = {
    "action": 1_752,
    "topic": 1_301,
    "rec_video": 565,
    "rec_prod": 565,
    "rec_ad": 565,
    "rec_living": 565,
    "material_desc2sid": 281,
    "material_sid2desc": 281,
    "world": 231,
}


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = {
        "instruction": str(raw.get("instruction", raw.get("system", "")) or ""),
        "input": str(raw.get("input", raw.get("prompt", "")) or ""),
        "output": str(raw.get("output", raw.get("response", "")) or ""),
        "history": raw.get("history") or [],
    }
    if not isinstance(row["history"], list):
        raise ValueError("history must be a list")
    return row


def mode_normalized_input(input_text: str) -> str:
    """Strip at most one terminal /think mode without changing other text."""

    return MODE_SUFFIX_RE.sub("", input_text.rstrip())


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


def strict_proposal_match(output: str) -> re.Match[str] | None:
    match = STRICT_PROPOSAL_RE.fullmatch(output)
    if match is None:
        return None
    values = (int(match.group("a")), int(match.group("b")), int(match.group("c")))
    if any(value < 0 or value > 8191 for value in values):
        return None
    return match


def has_valid_response_text_structure(output: str) -> bool:
    """Mirror the trainer's fail-closed one-think-block response contract."""

    if not output.startswith(OPEN_THINK):
        return False
    if output.count(OPEN_THINK) != 1 or output.count(CLOSE_THINK) != 1:
        return False
    close_index = output.find(CLOSE_THINK)
    if close_index < len(OPEN_THINK):
        return False
    if not output[close_index + len(CLOSE_THINK) :].strip():
        return False
    # EOS is supplied exactly once by the qwen3 template, never embedded in the
    # response string itself.
    return "<|im_end|>" not in output and "<|endoftext|>" not in output


def classify_retention_task(row: dict[str, Any]) -> str:
    """Classify the registered I-12 mix; its 231 world rows are the fallback."""

    try:
        return task_of(row)
    except ValueError:
        # The registered world rows have heterogeneous output formats, including
        # answers without a think block.  The exact expected source signature
        # below makes this fallback fail closed if any other task drifts.
        return "world"


def task_domain(task: str) -> str:
    if task.startswith("rec_"):
        return task.removeprefix("rec_")
    if task.startswith("material_"):
        return "material"
    return task


def read_i27_exclusion(path: Path) -> tuple[set[str], dict[str, Any]]:
    source_hash = file_sha256(path)
    if source_hash != EXPECTED_I27_SHA256:
        raise AssertionError(
            f"I-27 manifest SHA256 drifted: {source_hash} != {EXPECTED_I27_SHA256}"
        )

    normalized_hashes: set[str] = set()
    group_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = normalize_row(raw)
            if raw.get("domain") != MULTIGOLD_DOMAIN:
                raise AssertionError(f"non-video I-27 row at line {line_number}")
            normalized_hashes.add(normalized_prompt_sha256(row))
            group_id = str(raw.get("group_id") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", group_id):
                raise AssertionError(f"invalid I-27 group ID at line {line_number}")
            group_ids.add(group_id)

    if len(group_ids) != EXPECTED_I27_ROWS or len(normalized_hashes) != EXPECTED_I27_ROWS:
        raise AssertionError(
            "I-27 exclusion must contain 512 unique groups and normalized prompts: "
            f"{len(group_ids)}/{len(normalized_hashes)}"
        )
    return normalized_hashes, {
        "asset_id": I27_MANIFEST_ASSET_ID,
        "path": str(path.resolve()),
        "rows": len(group_ids),
        "sha256": source_hash,
        "exclusion_key": (
            "SHA256(schema, instruction, mode-normalized input, history); "
            "never source row index"
        ),
        "unique_normalized_prompt_hashes": len(normalized_hashes),
    }


def group_normalized_hash(group: dict[str, Any]) -> str:
    row = {
        "instruction": group["instruction"],
        "input": group["prompt_core"] + "/no_think",
        "output": "",
        "history": [],
    }
    return normalized_prompt_sha256(row)


def select_train_and_gate_groups(
    groups: list[dict[str, Any]], i27_hashes: set[str], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    multigold = [
        group
        for group in groups
        if group["domain"] == MULTIGOLD_DOMAIN
        and len(group["golds"]) >= MIN_UNIQUE_GOLDS
    ]
    if len(multigold) != 2_720:
        raise AssertionError(f"expected 2,720 video multigold groups, got {len(multigold)}")

    for group in multigold:
        group["normalized_prompt_sha256"] = group_normalized_hash(group)
        group["non_history_golds"] = [
            gold for gold in group["golds"] if not gold["target_in_prompt"]
        ]
    if len({group["normalized_prompt_sha256"] for group in multigold}) != len(multigold):
        raise AssertionError("video multigold groups are not unique by normalized prompt")

    i27_matches = [
        group for group in multigold if group["normalized_prompt_sha256"] in i27_hashes
    ]
    if len(i27_matches) != EXPECTED_I27_ROWS:
        raise AssertionError(
            f"I-27 normalized-prompt exclusion matched {len(i27_matches)}, expected 512"
        )

    eligible = [
        group
        for group in multigold
        if len(group["non_history_golds"]) >= MIN_NON_HISTORY_GOLDS
    ]
    rejected_for_history = [
        group
        for group in multigold
        if len(group["non_history_golds"]) < MIN_NON_HISTORY_GOLDS
    ]
    remaining = [
        group for group in eligible if group["normalized_prompt_sha256"] not in i27_hashes
    ]

    gate = sorted(
        remaining,
        key=lambda group: stable_hash(
            seed, "i28-gate-v1", group["normalized_prompt_sha256"]
        ),
    )[:GATE_GROUPS]
    gate_hashes = {group["normalized_prompt_sha256"] for group in gate}
    train_pool = [
        group for group in remaining if group["normalized_prompt_sha256"] not in gate_hashes
    ]
    train = sorted(
        train_pool,
        key=lambda group: stable_hash(
            seed, "i28-train-v1", group["normalized_prompt_sha256"]
        ),
    )[:TRAIN_GROUPS]
    train_hashes = {group["normalized_prompt_sha256"] for group in train}

    if len(gate) != GATE_GROUPS or len(train) != TRAIN_GROUPS:
        raise AssertionError("insufficient groups for frozen I-28 split")
    if train_hashes & gate_hashes or train_hashes & i27_hashes or gate_hashes & i27_hashes:
        raise AssertionError("I-28 train/gate/I-27 normalized prompts are not disjoint")

    return train, gate, {
        "selection_method": (
            "require at least two known non-history golds; exclude all I-27 prompts "
            "by mode-normalized SHA256; take gate and train by independent ascending "
            "SHA256 namespaces"
        ),
        "seed": seed,
        "video_multigold_groups_before_history_filter": len(multigold),
        "video_multigold_known_golds": sum(len(group["golds"]) for group in multigold),
        "history_overlap_known_golds": sum(
            len(group["golds"]) - len(group["non_history_golds"])
            for group in multigold
        ),
        "groups_rejected_fewer_than_two_non_history_golds": len(rejected_for_history),
        "known_golds_removed_with_rejected_groups": sum(
            len(group["golds"]) for group in rejected_for_history
        ),
        "eligible_video_multigold_groups_after_history_filter": len(eligible),
        "i27_normalized_prompt_matches_before_history_filter": len(i27_matches),
        "i27_excluded_eligible_groups": sum(
            group["normalized_prompt_sha256"] in i27_hashes for group in eligible
        ),
        "remaining_after_i27": len(remaining),
        "gate_groups": len(gate),
        "train_groups": len(train),
        "train_gate_normalized_prompt_overlap": len(train_hashes & gate_hashes),
        "train_i27_normalized_prompt_overlap": len(train_hashes & i27_hashes),
        "gate_i27_normalized_prompt_overlap": len(gate_hashes & i27_hashes),
    }


def build_proposal_rows(
    groups: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows_per_group: Counter[str] = Counter()
    gold_count_histogram: Counter[str] = Counter()
    selected_target_in_prompt = 0

    for group in groups:
        ordered_golds = sorted(
            group["non_history_golds"],
            key=lambda gold: stable_hash(
                seed, "i28-positive-gold-v1", group["group_id"], gold["itemic"]
            ),
        )
        selected = ordered_golds[:GOLDS_PER_TRAIN_GROUP]
        if len(selected) != GOLDS_PER_TRAIN_GROUP:
            raise AssertionError("selected group has fewer than two known positives")
        gold_count_histogram[str(len(group["golds"]))] += 1

        for gold in selected:
            itemic = gold["itemic"]
            output = f"<think>\n\n</think>\n{itemic}"
            match = strict_proposal_match(output)
            if match is None or match.group("domain") != MULTIGOLD_DOMAIN:
                raise AssertionError("constructed proposal response is not strict video 4-token")
            row = {
                "schema_version": SCHEMA_TRAIN,
                "route": PROPOSAL_ROUTE,
                "group_id": group["group_id"],
                "domain": MULTIGOLD_DOMAIN,
                "task": "rec_video",
                "upstream_ids": [PROPOSAL_ASSET_ID],
                "instruction": group["instruction"],
                "input": group["prompt_core"] + "/no_think",
                "output": output,
                "history": [],
            }
            if set(row) != TRAIN_KEYS:
                raise AssertionError(f"proposal schema drifted: {sorted(row)}")
            rows.append(row)
            rows_per_group[group["group_id"]] += 1
            selected_target_in_prompt += int(gold["target_in_prompt"])

    if len(rows) != PROPOSAL_ROWS:
        raise AssertionError(f"expected {PROPOSAL_ROWS} proposal rows, got {len(rows)}")
    if set(rows_per_group.values()) != {GOLDS_PER_TRAIN_GROUP}:
        raise AssertionError(f"proposal rows are not group-equal: {rows_per_group}")
    if len({row["group_id"] for row in rows}) != TRAIN_GROUPS:
        raise AssertionError("proposal row group count drifted")
    if any(not row["input"].rstrip().endswith("/no_think") for row in rows):
        raise AssertionError("proposal input is not routed through /no_think")
    if selected_target_in_prompt:
        raise AssertionError(
            f"proposal selected {selected_target_in_prompt} golds copied from prompt history"
        )

    return rows, {
        "groups": len(rows_per_group),
        "rows": len(rows),
        "rows_per_group": GOLDS_PER_TRAIN_GROUP,
        "rows_per_group_histogram": {
            str(count): sum(value == count for value in rows_per_group.values())
            for count in sorted(set(rows_per_group.values()))
        },
        "source_unique_gold_count_histogram": dict(
            sorted(gold_count_histogram.items(), key=lambda item: int(item[0]))
        ),
        "gold_selection": (
            "first two non-history golds by ascending SHA256(seed, namespace, "
            "group_id, exact itemic)"
        ),
        "strict_empty_think_four_token_responses": sum(
            strict_proposal_match(row["output"]) is not None for row in rows
        ),
        "selected_golds_also_in_immutable_prompt": selected_target_in_prompt,
        "selected_non_history_golds": len(rows),
        "group_equal": True,
    }


def build_gate_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        # Primary gate rewards never include a gold already visible in history;
        # this prevents a copy-only beam from receiving a positive signal.
        golds = [
            {
                "itemic": gold["itemic"],
                "itemic_sha256": gold["itemic_sha256"],
            }
            for gold in sorted(
                group["non_history_golds"], key=lambda value: value["itemic"]
            )
        ]
        history_overlap_count = len(group["golds"]) - len(golds)
        if len(golds) < MIN_NON_HISTORY_GOLDS:
            raise AssertionError("gate row has fewer than two non-history golds")
        rows.append(
            {
                "schema_version": SCHEMA_GATE,
                "route": GATE_ROUTE,
                "group_id": group["group_id"],
                "prompt_group_id": group["prompt_group_id"],
                "normalized_prompt_sha256": group["normalized_prompt_sha256"],
                "domain": MULTIGOLD_DOMAIN,
                "upstream_ids": [PROPOSAL_ASSET_ID],
                "instruction": group["instruction"],
                "input": group["prompt_core"] + "/no_think",
                "history": [],
                "gold_count": len(golds),
                "non_history_gold_count": len(golds),
                "all_o1_gold_count": len(group["golds"]),
                "history_overlap_gold_count": history_overlap_count,
                "primary_reward_excludes_history_golds": True,
                "golds": golds,
            }
        )
    rows.sort(key=lambda row: row["group_id"])
    if len(rows) != GATE_GROUPS or len({row["group_id"] for row in rows}) != GATE_GROUPS:
        raise AssertionError("gate row count or group uniqueness drifted")
    return rows


def load_retention_source(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_hash = file_sha256(path)
    if source_hash != EXPECTED_RETENTION_SHA256:
        raise AssertionError(
            f"retention source SHA256 drifted: {source_hash} != {EXPECTED_RETENTION_SHA256}"
        )
    rows = [normalize_row(row) for row in load_jsonl(path)]
    if len(rows) != EXPECTED_RETENTION_ROWS:
        raise AssertionError(
            f"expected {EXPECTED_RETENTION_ROWS} retention source rows, got {len(rows)}"
        )
    counts = Counter(classify_retention_task(row) for row in rows)
    if dict(counts) != EXPECTED_RETENTION_SOURCE_TASKS:
        raise AssertionError(f"retention source task signature drifted: {dict(counts)}")
    return rows, {
        "asset_id": RETENTION_ASSET_ID,
        "path": str(path.resolve()),
        "rows": len(rows),
        "sha256": source_hash,
        "task_counts": dict(sorted(counts.items())),
    }


def build_retention_rows(
    source_rows: list[dict[str, Any]],
    forbidden_prompt_hashes: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups_by_task: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: defaultdict(list) for task in RETENTION_QUOTAS
    }
    strict_signature_excluded: Counter[str] = Counter()
    response_structure_excluded: Counter[str] = Counter()
    forbidden_prompt_excluded: Counter[str] = Counter()

    for row in source_rows:
        task = classify_retention_task(row)
        if task not in RETENTION_QUOTAS:
            continue
        if not has_valid_response_text_structure(row["output"]):
            response_structure_excluded[task] += 1
            continue
        if strict_proposal_match(row["output"]) is not None:
            strict_signature_excluded[task] += 1
            continue
        prompt_hash = normalized_prompt_sha256(row)
        if prompt_hash in forbidden_prompt_hashes:
            forbidden_prompt_excluded[task] += 1
            continue
        groups_by_task[task][prompt_hash].append(row)

    selected: list[dict[str, Any]] = []
    available_groups: dict[str, int] = {}
    selected_counts: Counter[str] = Counter()
    core_field_changes = 0

    for task, quota in RETENTION_QUOTAS.items():
        prompt_groups = groups_by_task[task]
        available_groups[task] = len(prompt_groups)
        ordered_hashes = sorted(
            prompt_groups,
            key=lambda prompt_hash: stable_hash(
                seed, "i28-retention-select-v1", task, prompt_hash
            ),
        )
        if len(ordered_hashes) < quota:
            raise AssertionError(
                f"only {len(ordered_hashes)} eligible retention groups for {task}; need {quota}"
            )
        for prompt_hash in ordered_hashes[:quota]:
            candidates = prompt_groups[prompt_hash]
            source_row = min(
                candidates,
                key=lambda row: stable_hash(
                    seed, "i28-retention-representative-v1", task, canonical_json(row)
                ),
            )
            group_id = stable_hash(SCHEMA_RETENTION_GROUP, task, prompt_hash)
            output_row = {
                "schema_version": SCHEMA_TRAIN,
                "route": RETENTION_ROUTE,
                "group_id": group_id,
                "domain": task_domain(task),
                "task": task,
                "upstream_ids": [RETENTION_ASSET_ID],
                **source_row,
            }
            if set(output_row) != TRAIN_KEYS:
                raise AssertionError(f"retention schema drifted: {sorted(output_row)}")
            core_field_changes += int(
                any(output_row[key] != source_row[key] for key in CORE_KEYS)
            )
            if strict_proposal_match(output_row["output"]) is not None:
                raise AssertionError("retention row collides with strict proposal signature")
            if not has_valid_response_text_structure(output_row["output"]):
                raise AssertionError("selected retention row violates think/EOS text contract")
            selected.append(output_row)
            selected_counts[task] += 1

    if len(selected) != RETENTION_ROWS or dict(selected_counts) != RETENTION_QUOTAS:
        raise AssertionError(
            f"retention quota drifted: rows={len(selected)}, counts={dict(selected_counts)}"
        )
    if core_field_changes:
        raise AssertionError(f"modified {core_field_changes} retention teacher rows")
    if len({row["group_id"] for row in selected}) != len(selected):
        raise AssertionError("retention group IDs are not unique")

    return selected, {
        "rows": len(selected),
        "quota": dict(RETENTION_QUOTAS),
        "selected_task_counts": dict(sorted(selected_counts.items())),
        "available_prompt_groups_after_exclusions": dict(sorted(available_groups.items())),
        "strict_proposal_signature_excluded": dict(
            sorted(strict_signature_excluded.items())
        ),
        "invalid_think_or_embedded_eos_structure_excluded": dict(
            sorted(response_structure_excluded.items())
        ),
        "forbidden_i27_or_gate_prompt_excluded": dict(
            sorted(forbidden_prompt_excluded.items())
        ),
        "selected_strict_proposal_signature_rows": 0,
        "teacher_core_field_changes": core_field_changes,
        "teacher_semantics": "frozen-I23 KL only; no gold CE",
        "selection": (
            "mode-normalized prompt groups; stable hash without replacement; "
            "source instruction/input/output/history copied exactly"
        ),
    }


def _strip_whitespace_tokens(tokens: list[int]) -> list[int]:
    start = 0
    end = len(tokens)
    while start < end and tokens[start] in WHITESPACE_TOKEN_IDS:
        start += 1
    while end > start and tokens[end - 1] in WHITESPACE_TOKEN_IDS:
        end -= 1
    return tokens[start:end]


def _is_itemic_four_tokens(tokens: list[int], domain_ids: set[int]) -> bool:
    return (
        len(tokens) == 4
        and tokens[0] in domain_ids
        and 151_669 <= tokens[1] <= 159_860
        and 159_861 <= tokens[2] <= 168_052
        and 168_053 <= tokens[3] <= 176_244
    )


def validate_qwen3_response_tokens(
    rows: list[dict[str, Any]], tokenizer_path: Path, cutoff_len: int = 16_384
) -> dict[str, Any]:
    """Validate the response contract after O6 tokenization plus one template EOS."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )
    eos_id = int(tokenizer.eos_token_id)
    open_id = int(tokenizer.convert_tokens_to_ids(OPEN_THINK))
    close_id = int(tokenizer.convert_tokens_to_ids(CLOSE_THINK))
    domain_ids = {
        int(tokenizer.convert_tokens_to_ids(f"<|{domain}_begin|>"))
        for domain in ("video", "prod", "ad", "living")
    }
    video_domain_id = int(tokenizer.convert_tokens_to_ids("<|video_begin|>"))
    expected_ids = {"eos": 151_645, "open_think": 151_667, "close_think": 151_668}
    actual_ids = {"eos": eos_id, "open_think": open_id, "close_think": close_id}
    if actual_ids != expected_ids:
        raise AssertionError(f"O6 response special IDs drifted: {actual_ids}")

    route_counts: Counter[str] = Counter()
    empty_think_counts: Counter[str] = Counter()
    itemic_four_counts: Counter[str] = Counter()
    proposal_signature_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    formatted_lengths: list[int] = []
    overflow_rows = 0

    for index, row in enumerate(rows):
        response_ids = tokenizer.encode(row["output"], add_special_tokens=False)
        if not response_ids or response_ids[0] != open_id:
            raise AssertionError(f"row {index} response does not begin with <think>")
        if response_ids.count(open_id) != 1 or response_ids.count(close_id) != 1:
            raise AssertionError(f"row {index} does not contain exactly one think pair")
        if response_ids.count(eos_id) != 0:
            raise AssertionError(f"row {index} embeds EOS before template termination")

        close_index = response_ids.index(close_id)
        if close_index <= 0:
            raise AssertionError(f"row {index} has invalid think ordering")
        thought = _strip_whitespace_tokens(response_ids[1:close_index])
        answer = _strip_whitespace_tokens(response_ids[close_index + 1 :])
        if not answer:
            raise AssertionError(f"row {index} has no response body")

        # qwen3_nothink SFT appends exactly one <|im_end|> to this response.
        terminated = response_ids + [eos_id]
        if terminated.count(eos_id) != 1 or terminated[-1] != eos_id:
            raise AssertionError(f"row {index} lacks a unique terminal EOS")

        route = row["route"]
        route_counts[route] += 1
        token_lengths.append(len(terminated))
        empty_think_counts[route] += int(not thought)
        is_itemic = _is_itemic_four_tokens(answer, domain_ids)
        itemic_four_counts[route] += int(is_itemic)
        proposal_signature_counts[route] += int(not thought and is_itemic)

        if route == PROPOSAL_ROUTE:
            if thought or not is_itemic or answer[0] != video_domain_id:
                raise AssertionError(
                    f"proposal row {index} is not empty-think exact-video-four-token"
                )
        elif route == RETENTION_ROUTE:
            if not thought and is_itemic:
                raise AssertionError(
                    f"retention row {index} collides with proposal token signature"
                )
        else:
            raise AssertionError(f"unknown training route at row {index}: {route}")

        # Exact Alpaca conversion plus qwen3_nothink formatting used by the
        # registered LLaMA-Factory dataset: instruction and input are joined as
        # one user message; the assistant formatter appends <|im_end|> + LF.
        user_parts = [value for value in (row["instruction"], row["input"]) if value]
        user_content = "\n".join(user_parts)
        formatted = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{row['output']}<|im_end|>\n"
        )
        formatted_length = len(tokenizer.encode(formatted, add_special_tokens=False))
        formatted_lengths.append(formatted_length)
        overflow_rows += int(formatted_length > cutoff_len)

    if route_counts != {PROPOSAL_ROUTE: PROPOSAL_ROWS, RETENTION_ROUTE: RETENTION_ROWS}:
        raise AssertionError(f"token audit route counts drifted: {route_counts}")
    if empty_think_counts[PROPOSAL_ROUTE] != PROPOSAL_ROWS:
        raise AssertionError("not all proposal responses have empty think spans")
    if itemic_four_counts[PROPOSAL_ROUTE] != PROPOSAL_ROWS:
        raise AssertionError("not all proposal responses have exact 4-token itemic bodies")
    if proposal_signature_counts[RETENTION_ROUTE] != 0:
        raise AssertionError("retention contains empty-think exact-4-itemic response")
    if overflow_rows:
        raise AssertionError(
            f"{overflow_rows} qwen3_nothink-formatted rows exceed cutoff {cutoff_len}"
        )

    return {
        "status": "PASS",
        "tokenizer_asset_id": "O6:OneReason-0.8B-pretrain-competition",
        "tokenizer_path": str(tokenizer_path.resolve()),
        "template_semantics": (
            "qwen3_nothink response tokens with exactly one terminal tokenizer EOS appended"
        ),
        "special_token_ids": actual_ids,
        "rows_checked": len(rows),
        "route_counts": dict(sorted(route_counts.items())),
        "exactly_one_think_pair_rows": len(rows),
        "unique_terminal_eos_rows": len(rows),
        "empty_think_by_route": dict(sorted(empty_think_counts.items())),
        "exact_itemic_four_tokens_by_route": dict(sorted(itemic_four_counts.items())),
        "empty_think_exact_itemic_proposal_signature_by_route": dict(
            sorted(proposal_signature_counts.items())
        ),
        "proposal_answer_atomic_token_contract": {
            "video_domain_token_id": video_domain_id,
            "a_token_id_range_inclusive": [151_669, 159_860],
            "b_token_id_range_inclusive": [159_861, 168_052],
            "c_token_id_range_inclusive": [168_053, 176_244],
            "validated_rows": PROPOSAL_ROWS,
        },
        "response_tokens_including_eos": {
            "min": min(token_lengths),
            "max": max(token_lengths),
        },
        "qwen3_nothink_formatted_cutoff": {
            "cutoff_len": cutoff_len,
            "overflow_rows": overflow_rows,
            "min_tokens": min(formatted_lengths),
            "max_tokens": max(formatted_lengths),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(canonical_json(row) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def preflight_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    paths = list(paths)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("train, gate, and audit outputs must be distinct")
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
            )


def build(args: argparse.Namespace) -> dict[str, Any]:
    preflight_outputs((args.out, args.gate, args.audit), args.overwrite)

    proposal_hash = file_sha256(args.proposal_source)
    if proposal_hash != EXPECTED_SOURCE_SHA256:
        raise AssertionError(
            f"proposal source SHA256 drifted: {proposal_hash} != {EXPECTED_SOURCE_SHA256}"
        )
    proposal_records = read_source(args.proposal_source)
    if len(proposal_records) != EXPECTED_SOURCE_ROWS:
        raise AssertionError("proposal source row count drifted")
    groups, grouping_audit = aggregate_groups(proposal_records)

    i27_hashes, i27_audit = read_i27_exclusion(args.i27_manifest)
    train_groups, gate_groups, split_audit = select_train_and_gate_groups(
        groups, i27_hashes, args.seed
    )
    proposal_rows, proposal_audit = build_proposal_rows(train_groups, args.seed)
    gate_rows = build_gate_rows(gate_groups)

    retention_source_rows, retention_source_audit = load_retention_source(
        args.retention_source
    )
    gate_hashes = {row["normalized_prompt_sha256"] for row in gate_rows}
    retention_rows, retention_audit = build_retention_rows(
        retention_source_rows,
        forbidden_prompt_hashes=i27_hashes | gate_hashes,
        seed=args.seed,
    )

    final_rows = proposal_rows + retention_rows
    final_rows.sort(
        key=lambda row: stable_hash(
            args.seed,
            "i28-final-row-order-v1",
            row["route"],
            row["group_id"],
            row["output"],
        )
    )
    if len(final_rows) != EXPECTED_OUTPUT_ROWS:
        raise AssertionError(
            f"expected {EXPECTED_OUTPUT_ROWS} output rows, got {len(final_rows)}"
        )
    route_counts = Counter(row["route"] for row in final_rows)
    if route_counts != {PROPOSAL_ROUTE: PROPOSAL_ROWS, RETENTION_ROUTE: RETENTION_ROWS}:
        raise AssertionError(f"final route counts drifted: {route_counts}")
    strict_by_route = Counter(
        row["route"]
        for row in final_rows
        if strict_proposal_match(row["output"]) is not None
    )
    if strict_by_route != {PROPOSAL_ROUTE: PROPOSAL_ROWS}:
        raise AssertionError(f"proposal/retention signature collision: {strict_by_route}")

    train_prompt_hashes = {normalized_prompt_sha256(row) for row in final_rows}
    if train_prompt_hashes & gate_hashes or train_prompt_hashes & i27_hashes:
        raise AssertionError("complete train data overlaps gate or I-27 normalized prompts")

    tokenizer_audit = validate_qwen3_response_tokens(final_rows, args.tokenizer)

    write_jsonl(args.out, final_rows)
    write_jsonl(args.gate, gate_rows)

    audit = {
        "asset_class": {
            "train": (
                "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,"
                "O2.General)"
            ),
            "gate": "E(D(O1)); diagnostic only; forbidden for training",
        },
        "purpose": (
            "64-group video multigold proposal SFT with frozen-I23 cross-task KL "
            "retention and a separate 128-group prompt-disjoint gate"
        ),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": file_sha256(Path(__file__)),
        "seed": args.seed,
        "upstream": {
            "proposal": {
                "asset_id": PROPOSAL_ASSET_ID,
                "path": str(args.proposal_source.resolve()),
                "rows": len(proposal_records),
                "sha256": proposal_hash,
            },
            "retention": retention_source_audit,
            "i27_exclusion_manifest": i27_audit,
        },
        "source_grouping": grouping_audit,
        "split": {
            **split_audit,
            "complete_train_gate_normalized_prompt_overlap": len(
                train_prompt_hashes & gate_hashes
            ),
            "complete_train_i27_normalized_prompt_overlap": len(
                train_prompt_hashes & i27_hashes
            ),
            "exclusion_is_by_source_row_index": False,
        },
        "training_rows": {
            "rows": len(final_rows),
            "route_counts": dict(sorted(route_counts.items())),
            "row_mix": {
                PROPOSAL_ROUTE: {
                    "rows": PROPOSAL_ROWS,
                    "ratio": PROPOSAL_ROWS / EXPECTED_OUTPUT_ROWS,
                },
                RETENTION_ROUTE: {
                    "rows": RETENTION_ROWS,
                    "ratio": RETENTION_ROWS / EXPECTED_OUTPUT_ROWS,
                },
            },
            "proposal": proposal_audit,
            "retention": retention_audit,
            "strict_response_signature_by_route": dict(sorted(strict_by_route.items())),
            "qwen3_response_token_audit": tokenizer_audit,
            "schema_keys": sorted(TRAIN_KEYS),
            "llamafactory_compatible": (
                "alpaca dataset_info maps instruction/input/output/history; metadata "
                "columns are intentionally removed by the converter"
            ),
            "one_epoch_geometry": {
                "rows": EXPECTED_OUTPUT_ROWS,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "optimizer_steps": 128,
                "all_64_groups_observe_both_positive_golds": True,
            },
        },
        "gate": {
            "rows": len(gate_rows),
            "groups": len(gate_rows),
            "primary_non_history_gold_count": sum(
                row["non_history_gold_count"] for row in gate_rows
            ),
            "all_o1_known_gold_count": sum(row["all_o1_gold_count"] for row in gate_rows),
            "history_overlap_gold_count_excluded_from_primary_reward": sum(
                row["history_overlap_gold_count"] for row in gate_rows
            ),
            "primary_non_history_gold_count_histogram": dict(
                sorted(
                    Counter(str(row["gold_count"]) for row in gate_rows).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "training_allowed": False,
        },
        "forbidden_sources": {
            "T_rows": 0,
            "E_rows_in_training": 0,
            "model_or_teacher_rollout_rows": 0,
            "O3_target_metadata_rows": 0,
        },
        "outputs": {
            "train": {
                "path": str(args.out.resolve()),
                "rows": len(final_rows),
                "sha256": file_sha256(args.out),
            },
            "gate": {
                "path": str(args.gate.resolve()),
                "rows": len(gate_rows),
                "sha256": file_sha256(args.gate),
            },
        },
        "formal_training_started": False,
    }
    write_json(args.audit, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-source", type=Path, default=DEFAULT_PROPOSAL_SOURCE)
    parser.add_argument("--retention-source", type=Path, default=DEFAULT_RETENTION_SOURCE)
    parser.add_argument("--i27-manifest", type=Path, default=DEFAULT_I27_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    audit = build(parse_args())
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
