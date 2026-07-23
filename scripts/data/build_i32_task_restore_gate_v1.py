#!/usr/bin/env python3
"""Build the prompt-disjoint I-32 acceptance holdout from registered D assets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MATERIAL_SOURCE = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
RETENTION_SOURCE = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
I30_TRAIN = ROOT / "assets/derived/processed/data_i30_r96_material_teacher_retkl_v1.jsonl"
I30_DEV = ROOT / "assets/evaluation/holdout/data_i30_r96_material_teacher_gate_v1.jsonl"
OUTPUT = ROOT / "assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl"
AUDIT = ROOT / "logs/data/i32_task_restore_gate_v1_audit.json"

EXPECTED_SHA256 = {
    MATERIAL_SOURCE: "13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f",
    RETENTION_SOURCE: "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0",
    I30_TRAIN: "0df9a192976eb61eb8dd333fd59edb994d1fcad482710e1282f36dd792bfc4a4",
    I30_DEV: "dd744ee2d2f584b9bcae938cde1f5976801a9eece39aa1972b284641603f97a0",
}
MATERIAL_TASKS = ("material_desc2sid", "material_sid2desc")
QUOTAS = {
    "material_desc2sid": 128,
    "material_sid2desc": 128,
    "action": 64,
    "topic": 64,
    "rec_video": 64,
    "rec_prod": 64,
    "rec_ad": 64,
    "rec_living": 64,
    "world": 16,
}
SEED = 19260832


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", "") or ""),
        "input": str(row.get("input", "") or ""),
        "output": str(row.get("output", "") or ""),
        "history": row.get("history") or [],
    }


def prompt_digest(row: dict[str, Any]) -> str:
    payload = [row.get("instruction", ""), row.get("input", ""), row.get("history") or []]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(normalized(row)).encode()).hexdigest()


def stable_key(task: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical([SEED, "i32-acceptance", task, row_digest(row)]).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def load_classifier():
    helper_path = ROOT / "scripts/data/build_seed_scoremax_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_i32_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper.task_of


def classify(row: dict[str, Any], task_of) -> str:
    try:
        return task_of(row)
    except ValueError:
        if "<s_a_" not in canonical(normalized(row)):
            return "world"
        raise


def unique_prompt_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[prompt_digest(row)].append(normalized(row))
    selected = []
    conflicting = 0
    for digest, values in grouped.items():
        targets = {row["output"] for row in values}
        if len(targets) != 1:
            conflicting += 1
            continue
        selected.append(min(values, key=row_digest))
    return selected, conflicting


def main() -> None:
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"upstream hash drift: {path} {actual}/{expected}")
    if OUTPUT.exists() or AUDIT.exists():
        raise RuntimeError("I-32 holdout/audit already exists; refusing overwrite")

    task_of = load_classifier()
    excluded_rows = load_jsonl(I30_TRAIN) + load_jsonl(I30_DEV)
    excluded = {prompt_digest(row) for row in excluded_rows}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(MATERIAL_SOURCE):
        task = classify(row, task_of)
        if task in MATERIAL_TASKS and prompt_digest(row) not in excluded:
            buckets[task].append(row)
    for row in load_jsonl(RETENTION_SOURCE):
        task = classify(row, task_of)
        if task in QUOTAS and task not in MATERIAL_TASKS and prompt_digest(row) not in excluded:
            buckets[task].append(row)

    output = []
    available = {}
    conflicting_prompt_groups = {}
    cross_task_prompt_groups_excluded = {}
    manifests = {}
    used_prompts: set[str] = set()
    for task, quota in QUOTAS.items():
        unique, conflicting = unique_prompt_rows(buckets[task])
        ranked_all = sorted(unique, key=lambda row: stable_key(task, row))
        ranked = [row for row in ranked_all if prompt_digest(row) not in used_prompts]
        available[task] = len(ranked)
        conflicting_prompt_groups[task] = conflicting
        cross_task_prompt_groups_excluded[task] = len(ranked_all) - len(ranked)
        if len(ranked) < quota:
            raise RuntimeError(f"{task}: only {len(ranked)} fresh prompts, need {quota}")
        chosen = ranked[:quota]
        used_prompts.update(prompt_digest(row) for row in chosen)
        output.extend({**row, "route": "gate_only", "task": task} for row in chosen)
        manifests[task] = hashlib.sha256(
            "\n".join(prompt_digest(row) for row in chosen).encode()
        ).hexdigest()

    output_prompts = [prompt_digest(row) for row in output]
    if len(output_prompts) != len(set(output_prompts)):
        raise RuntimeError("I-32 holdout has duplicate prompts")
    if set(output_prompts) & excluded:
        raise RuntimeError("I-32 holdout overlaps I-30 train/development prompts")
    counts = Counter(row["task"] for row in output)
    if dict(counts) != QUOTAS:
        raise RuntimeError(f"I-32 quota mismatch: {dict(counts)}/{QUOTAS}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with output_tmp.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_tmp.replace(OUTPUT)
    audit = {
        "status": "COMPLETE_ACCEPTANCE_HOLDOUT_ONLY_NOT_TRAINING_DATA",
        "seed": SEED,
        "upstreams": {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED_SHA256},
        "rows": len(output),
        "counts": dict(counts),
        "available_after_exclusion": available,
        "conflicting_prompt_groups_excluded": conflicting_prompt_groups,
        "cross_task_prompt_groups_excluded": cross_task_prompt_groups_excluded,
        "prompt_overlap_i30_train_or_dev": 0,
        "prompt_manifest_sha256_by_task": manifests,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = AUDIT.with_suffix(AUDIT.suffix + ".tmp")
    audit_tmp.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_tmp.replace(AUDIT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
