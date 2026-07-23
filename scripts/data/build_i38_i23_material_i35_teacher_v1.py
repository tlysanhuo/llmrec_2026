#!/usr/bin/env python3
"""Build the I-38M task-conditioned teacher mix.

The policy starts from the verified I-23 r64 adapter.  Material rows are
anchors from the exact I-35 material renderer and are matched to the I-23
reference during training; retention rows are matched to the verified I-35
step548 reference.  No labels are changed and no model-generated rows are
written to the data asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "i38-i23-material-i35-teacher-retkl-v1"
SEED = 19260838

SOURCE = ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"
GATE_SOURCE = ROOT / "assets/evaluation/holdout/data_i33_r96_material_desc2sid_gate_v1.jsonl"
GATE_WORLD_SOURCE = ROOT / "assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl"
OUTPUT = ROOT / "assets/derived/processed/data_i38_i23_material_i35_teacher_retkl_v1.jsonl"
GATE_OUTPUT = ROOT / "assets/evaluation/holdout/data_i38_i23_material_i35_teacher_gate_v1.jsonl"
AUDIT = ROOT / "logs/data/i38_i23_material_i35_teacher_retkl_v1_audit.json"
GATE_AUDIT = ROOT / "logs/data/i38_i23_material_i35_teacher_gate_v1_audit.json"
REGISTERED_HOLDOUTS = (
    ROOT / "assets/evaluation/holdout/data_i30_r96_material_teacher_gate_v1.jsonl",
    ROOT / "assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl",
    ROOT / "assets/evaluation/holdout/data_i33_r96_material_desc2sid_gate_v1.jsonl",
    ROOT / "assets/evaluation/holdout/data_i34_material_beam_dev_v1.jsonl",
)

EXPECTED_SOURCE_ROWS = 2740
EXPECTED_MATERIAL_ROWS = 1370
EXPECTED_RETENTION_ROWS = 1370
GATE_MATERIAL_DESC2SID_ROWS = 128
GATE_MATERIAL_SID2DESC_ROWS = 64
GATE_RETENTION_PER_TASK = 32
GATE_WORLD_ROWS = 16
TASKS = ("action", "topic", "rec_video", "rec_prod", "rec_ad", "rec_living")

SID_RE = re.compile(r"^<\|(video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing source: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                raise RuntimeError(f"blank row at {path}:{line_no}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object row at {path}:{line_no}")
            rows.append(value)
    return rows


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def prompt_hash(row: dict[str, Any]) -> str:
    value = normalized(row)
    text = value["input"].rstrip()
    for suffix in ("/think", "/no_think"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            break
    return digest([value["instruction"], text, value["history"]])


def row_hash(row: dict[str, Any]) -> str:
    return digest(normalized(row))


def body(row: dict[str, Any]) -> str:
    output = normalized(row)["output"]
    if "</think>" not in output:
        return output.strip()
    return output.split("</think>", 1)[1].strip()


def is_material(row: dict[str, Any]) -> bool:
    value = normalized(row)
    return (
        value["input"].rstrip().endswith("/no_think")
        and value["output"].startswith("<think>\n\n</think>\n")
        and bool(SID_RE.fullmatch(body(row)))
        and body(row).startswith("<|video_begin|>")
    )


def stable_key(row: dict[str, Any]) -> str:
    return digest([SEED, prompt_hash(row), row_hash(row)])


def copy_row(row: dict[str, Any], *, route: str, task: str, source: str) -> dict[str, Any]:
    value = normalized(row)
    value.update({"schema_version": SCHEMA, "route": route, "task": task, "source_asset": source})
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    with path.open("x", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def build_training(source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    material = [r for r in source_rows if r.get("route") == "material_boundary"]
    retention = [r for r in source_rows if r.get("route") == "retention_kl"]
    if len(material) != EXPECTED_MATERIAL_ROWS or len(retention) != EXPECTED_RETENTION_ROWS:
        raise RuntimeError(f"unexpected I-35 source split: material={len(material)} retention={len(retention)}")
    rows = [copy_row(r, route="material_anchor_i23", task="material_desc2sid", source="data_i35_video_boundary_retkl_v1") for r in material]
    rows.extend(copy_row(r, route="retention_teacher_i35", task=str(r.get("task")), source="data_i35_video_boundary_retkl_v1") for r in retention)
    random.Random(SEED).shuffle(rows)
    counts = Counter(r["route"] for r in rows)
    return rows, counts


def registered_holdout_prompts() -> tuple[set[str], dict[str, dict[str, Any]]]:
    prompts: set[str] = set()
    assets: dict[str, dict[str, Any]] = {}
    for path in REGISTERED_HOLDOUTS:
        rows = load_jsonl(path)
        prompts.update(prompt_hash(row) for row in rows)
        assets[str(path.relative_to(ROOT))] = {"rows": len(rows), "sha256": sha256(path)}
    return prompts, assets


def take_unique(
    candidates: list[dict[str, Any]], count: int, occupied: set[str], label: str
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(candidates, key=stable_key):
        signature = prompt_hash(row)
        if signature in occupied:
            continue
        selected.append(row)
        occupied.add(signature)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"not enough unique {label} gate rows: {len(selected)}/{count}")
    return selected


def build_gate(train_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    train_prompts = {prompt_hash(r) for r in train_rows}
    source = load_jsonl(GATE_SOURCE)
    gate: list[dict[str, Any]] = []
    occupied = set(train_prompts)
    material_desc2sid = take_unique(
        [row for row in source if row.get("task") == "material_desc2sid"],
        GATE_MATERIAL_DESC2SID_ROWS,
        occupied,
        "material_desc2sid",
    )
    gate.extend(
        copy_row(row, route="material_gate", task="material_desc2sid", source="data_i33_r96_material_desc2sid_gate_v1")
        for row in material_desc2sid
    )
    material_sid2desc = take_unique(
        [row for row in source if row.get("task") == "material_sid2desc"],
        GATE_MATERIAL_SID2DESC_ROWS,
        occupied,
        "material_sid2desc",
    )
    gate.extend(
        copy_row(row, route="material_gate", task="material_sid2desc", source="data_i33_r96_material_desc2sid_gate_v1")
        for row in material_sid2desc
    )
    for task in TASKS:
        selected = take_unique(
            [row for row in source if row.get("task") == task],
            GATE_RETENTION_PER_TASK,
            occupied,
            task,
        )
        gate.extend(
            copy_row(r, route="retention_gate", task=task, source="data_i33_r96_material_desc2sid_gate_v1")
            for r in selected
        )
    world_source = load_jsonl(GATE_WORLD_SOURCE)
    world = take_unique(
        [row for row in world_source if row.get("task") == "world"],
        GATE_WORLD_ROWS,
        occupied,
        "world",
    )
    gate.extend(
        copy_row(row, route="retention_gate", task="world", source="data_i32_task_restore_gate_v1")
        for row in world
    )
    random.Random(SEED + 1).shuffle(gate)
    return gate, Counter(r["task"] for r in gate)


def audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite audit: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("use --build")
    for path in (OUTPUT, GATE_OUTPUT, AUDIT, GATE_AUDIT):
        if path.exists():
            raise RuntimeError(f"output already exists: {path}")

    source_rows = load_jsonl(SOURCE)
    train_rows, train_counts = build_training(source_rows)
    holdout_prompts, holdout_assets = registered_holdout_prompts()
    gate_rows, gate_counts = build_gate(train_rows)
    gate_prompts = {prompt_hash(row) for row in gate_rows}
    if len(gate_prompts) != len(gate_rows):
        raise RuntimeError("I-38 gate contains duplicate mode-normalized prompts")
    if not gate_prompts <= holdout_prompts:
        raise RuntimeError("I-38 gate contains a prompt outside its frozen registered holdout sources")
    write_jsonl(OUTPUT, train_rows)
    write_jsonl(GATE_OUTPUT, gate_rows)
    common = {
        "schema": SCHEMA,
        "seed": SEED,
        "source_assets": {
            "training_source": {"path": str(SOURCE.relative_to(ROOT)), "rows": len(source_rows), "sha256": sha256(SOURCE)},
            "gate_source": {"path": str(GATE_SOURCE.relative_to(ROOT)), "rows": sum(1 for _ in GATE_SOURCE.open(encoding="utf-8")), "sha256": sha256(GATE_SOURCE)},
            "gate_world_source": {"path": str(GATE_WORLD_SOURCE.relative_to(ROOT)), "rows": sum(1 for _ in GATE_WORLD_SOURCE.open(encoding="utf-8")), "sha256": sha256(GATE_WORLD_SOURCE)},
            "registered_holdout_blacklist": holdout_assets,
            "i23_policy_adapter": {"path": "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform", "role": "policy_start_and_material_reference"},
            "i35_teacher_adapter": {"path": "submissions/i35_r96_video_boundary_retkl_r112_step548_platform", "role": "retention_reference"},
        },
    }
    audit(AUDIT, {
        **common,
        "status": "formal_data_built_ready_for_training",
        "output": {"path": str(OUTPUT.relative_to(ROOT)), "rows": len(train_rows), "sha256": sha256(OUTPUT), "route_counts": dict(train_counts), "task_counts": dict(Counter(r["task"] for r in train_rows))},
        "mix": {"material_anchor_i23": EXPECTED_MATERIAL_ROWS, "retention_teacher_i35": EXPECTED_RETENTION_ROWS, "ratio": "1:1", "T_rows": 0, "E_rows": 0, "model_generated_rows": 0},
    })
    audit(GATE_AUDIT, {
        **common,
        "status": "frozen_training_disjoint_subset_of_registered_holdouts",
        "output": {"path": str(GATE_OUTPUT.relative_to(ROOT)), "rows": len(gate_rows), "sha256": sha256(GATE_OUTPUT), "task_counts": dict(gate_counts), "unique_mode_normalized_prompts": len(gate_prompts), "prompt_overlap_with_training": 0, "prompts_outside_registered_holdout_sources": 0},
        "use": "checkpoint mechanism and retention gate only; no training gradient and no online-score estimate",
    })
    print(json.dumps({"training": str(OUTPUT), "gate": str(GATE_OUTPUT), "training_rows": len(train_rows), "gate_rows": len(gate_rows), "training_sha256": sha256(OUTPUT), "gate_sha256": sha256(GATE_OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
