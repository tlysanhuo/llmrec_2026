#!/usr/bin/env python3
"""Build paired E-clean I-33 material control/treatment data and a fresh gate.

Both arms retain I-30's 512/1,536 route schedule and share the same E-clean
retention rows.  The control keeps 256 rows in each material direction; the
treatment uses 512 description-to-SID rows.  Material rows remain selected by
I-30's frozen teacher-advantage ledger.  Every registered prompt-bearing E or
holdout asset is excluded by exact and mode-normalized prompt hash.
"""

from __future__ import annotations

import argparse
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
I30_LEDGER = ROOT / "logs/data/i30_r96_material_teacher_selection_v1.jsonl"
I30_GATE = ROOT / "assets/evaluation/holdout/data_i30_r96_material_teacher_gate_v1.jsonl"
I32_GATE = ROOT / "assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl"
NATIVE_RETENTION_GATE = (
    ROOT / "assets/evaluation/holdout/s800_native_general_replay_retention_gate_v1.jsonl"
)
REWARD_PREFERENCE_HOLDOUT = (
    ROOT / "assets/evaluation/holdout/data_o1_reward_preference_v1_holdout.jsonl"
)
I28_VIDEO_GATE = (
    ROOT / "assets/evaluation/holdout/data_i28_video_multigold_proposal_v1_gate.jsonl"
)
OFFICIAL_WORLD_HOLDOUT = (
    ROOT / "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl"
)
I22_WORLD_HOLDOUT = ROOT / "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl"
VISIBLE_WORLD = ROOT / "assets/evaluation/visible/懂世界.jsonl"
CONTROL_OUTPUT = (
    ROOT / "assets/derived/processed/data_i33_r96_material_bidirectional_e_clean_control_v1.jsonl"
)
OUTPUT = ROOT / "assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl"
HOLDOUT = ROOT / "assets/evaluation/holdout/data_i33_r96_material_desc2sid_gate_v1.jsonl"
SELECTION = ROOT / "logs/data/i33_r96_material_desc2sid_selection_v1.jsonl"
CONTROL_SELECTION = ROOT / "logs/data/i33_r96_material_bidirectional_control_selection_v1.jsonl"
RETENTION_REPLACEMENTS = ROOT / "logs/data/i33_r96_e_clean_retention_replacements_v1.jsonl"
AUDIT = ROOT / "logs/data/i33_r96_material_desc2sid_retkl_v1_audit.json"

EXPECTED_SHA256 = {
    MATERIAL_SOURCE: "13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f",
    RETENTION_SOURCE: "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0",
    I30_TRAIN: "0df9a192976eb61eb8dd333fd59edb994d1fcad482710e1282f36dd792bfc4a4",
    I30_LEDGER: "b303a501dddb1f7ae3afef192298f349dd211cbda229eb1618a94634c82b5b3d",
    I30_GATE: "dd744ee2d2f584b9bcae938cde1f5976801a9eece39aa1972b284641603f97a0",
    I32_GATE: "f75106758792163dd33d1d52639ba507a6d9e69094d8213d5f3b0969ee272f62",
    NATIVE_RETENTION_GATE: "3206e91ac465ca4f1410e3f8a9219a60c11cbb1beb3d4eb2fa9fa69c4b89c30f",
    REWARD_PREFERENCE_HOLDOUT: "1c7292cb96d45e9d20c0b3add78d3e5a30ec7a559844217584408921f996696e",
    I28_VIDEO_GATE: "48dd7f4224e7ca9e98805d966ca901814fdb76b85471afcf1ec7d98a0c22c7e5",
    OFFICIAL_WORLD_HOLDOUT: "fb67b76d8d071799ba372185bd89cb556afef9065a1b188fb9dd86a9131e13df",
    I22_WORLD_HOLDOUT: "8aa4306f139afc0a00cacd91508de90aa9fa2cbd9942af9cdb665d895721402a",
    VISIBLE_WORLD: "d7f341e2277473ff3b5b556531370a7697471cc5e540586a53c93c6adb32b5e5",
}

E_HOLDOUT_PATHS = (
    I30_GATE,
    I32_GATE,
    NATIVE_RETENTION_GATE,
    REWARD_PREFERENCE_HOLDOUT,
    I28_VIDEO_GATE,
    OFFICIAL_WORLD_HOLDOUT,
    I22_WORLD_HOLDOUT,
    VISIBLE_WORLD,
)

TARGET_TASK = "material_desc2sid"
REVERSE_TASK = "material_sid2desc"
RETENTION_TASKS = (
    "action",
    "topic",
    "rec_video",
    "rec_prod",
    "rec_ad",
    "rec_living",
    "world",
)
TRAIN_TARGET_ROWS = 512
TRAIN_RETENTION_ROWS = 1536
HOLDOUT_QUOTAS = {
    TARGET_TASK: 256,
    REVERSE_TASK: 128,
    "action": 64,
    "topic": 64,
    "rec_video": 32,
    "rec_prod": 48,
    "rec_ad": 64,
    "rec_living": 64,
    # All 106 E-clean world prompts are required by the shared retention set.
    # World is therefore checked only on the already frozen registered gates.
    "world": 0,
}
HOLDOUT_SEED = 19260833


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
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(normalized(row)).encode()).hexdigest()


def prompt_digest(row: dict[str, Any]) -> str:
    value = normalized(row)
    payload = [value["instruction"], value["input"], value["history"]]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def mode_normalized_prompt_digest(row: dict[str, Any]) -> str:
    value = normalized(row)
    input_text = value["input"].rstrip()
    for suffix in ("/no_think", "/think"):
        if input_text.endswith(suffix):
            input_text = input_text[: -len(suffix)].rstrip()
            break
    payload = [value["instruction"], input_text, value["history"]]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def stable_key(task: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical([HOLDOUT_SEED, "i33-acceptance", task, row_digest(row)]).encode()
    ).hexdigest()


def replacement_key(task: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical(["i33-e-clean-retention-v1", task, row_digest(row)]).encode()
    ).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            values = value if isinstance(value, list) else [value]
            if not values or any(not isinstance(item, dict) for item in values):
                raise RuntimeError(f"{path}:{line_number} is not an object or object array")
            rows.extend(values)
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_classifier():
    helper_path = ROOT / "scripts/data/build_seed_scoremax_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_i33_helpers", helper_path)
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


def unique_nonconflicting(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[mode_normalized_prompt_digest(row)].append(normalized(row))
    selected = []
    conflicts = 0
    for values in grouped.values():
        if len({row["output"] for row in values}) != 1:
            conflicts += 1
            continue
        selected.append(min(values, key=row_digest))
    return selected, conflicts


def select_material_metrics(
    task: str,
    count: int,
    ledger: list[dict[str, Any]],
    source_by_digest: dict[str, dict[str, Any]],
    forbidden_prompts: set[str],
    forbidden_modes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    positive = [
        entry
        for entry in ledger
        if entry.get("task") == task
        and float(entry.get("teacher_minus_parent_mean_logp", 0.0)) > 0
        and entry.get("row_sha256") in source_by_digest
    ]
    positive.sort(
        key=lambda entry: (
            -float(entry["teacher_minus_parent_mean_logp"]),
            entry["row_sha256"],
        )
    )
    eligible = []
    used_rows: set[str] = set()
    used_prompts: set[str] = set()
    used_modes: set[str] = set()
    for entry in positive:
        source = source_by_digest[entry["row_sha256"]]
        prompt = prompt_digest(source)
        mode = mode_normalized_prompt_digest(source)
        if prompt in forbidden_prompts or mode in forbidden_modes:
            continue
        if entry["row_sha256"] in used_rows or prompt in used_prompts or mode in used_modes:
            continue
        eligible.append(entry)
        used_rows.add(entry["row_sha256"])
        used_prompts.add(prompt)
        used_modes.add(mode)
    if len(eligible) < count:
        raise RuntimeError(f"{task}: only {len(eligible)} E-clean positive rows, need {count}")
    return eligible[:count], {
        "positive": len(positive),
        "eligible_e_clean_unique": len(eligible),
        "excluded_or_deduplicated": len(positive) - len(eligible),
        "selected": count,
        "eligible_unselected": len(eligible) - count,
    }


def assign_material_positions(
    i30_rows: list[dict[str, Any]],
    positions: list[int],
    selected: list[dict[str, Any]],
    source_by_digest: dict[str, dict[str, Any]],
    task: str,
) -> tuple[dict[int, dict[str, Any]], int]:
    selected_digests = {entry["row_sha256"] for entry in selected}
    assigned: dict[int, dict[str, Any]] = {}
    used: set[str] = set()
    for position in positions:
        digest = row_digest(i30_rows[position])
        if digest in selected_digests and digest not in used:
            assigned[position] = {
                **normalized(i30_rows[position]),
                "route": "material_teacher",
                "task": task,
            }
            used.add(digest)
    remaining = [entry for entry in selected if entry["row_sha256"] not in used]
    open_positions = [position for position in positions if position not in assigned]
    if len(remaining) != len(open_positions):
        raise RuntimeError(f"{task}: material position assignment drifted")
    for position, entry in zip(open_positions, remaining):
        assigned[position] = {
            **source_by_digest[entry["row_sha256"]],
            "route": "material_teacher",
            "task": task,
        }
    return assigned, len(used)


def build_shared_retention(
    task_of,
    retention_rows: list[dict[str, Any]],
    i30_rows: list[dict[str, Any]],
    forbidden_prompts: set[str],
    forbidden_modes: set[str],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    positions_by_task: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(i30_rows):
        if row.get("route") == "retention_kl":
            positions_by_task[row["task"]].append(position)

    assigned: dict[int, dict[str, Any]] = {}
    used_prompts: set[str] = set()
    used_modes: set[str] = set()
    replacement_positions: dict[str, list[int]] = defaultdict(list)
    for task in RETENTION_TASKS:
        for position in positions_by_task[task]:
            row = normalized(i30_rows[position])
            prompt = prompt_digest(row)
            mode = mode_normalized_prompt_digest(row)
            if (
                prompt in forbidden_prompts
                or mode in forbidden_modes
                or prompt in used_prompts
                or mode in used_modes
            ):
                replacement_positions[task].append(position)
                continue
            assigned[position] = {**row, "route": "retention_kl", "task": task}
            used_prompts.add(prompt)
            used_modes.add(mode)

    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in retention_rows:
        task = classify(row, task_of)
        if task not in RETENTION_TASKS:
            continue
        value = normalized(row)
        if (
            prompt_digest(value) in forbidden_prompts
            or mode_normalized_prompt_digest(value) in forbidden_modes
        ):
            continue
        grouped[task][prompt_digest(value)][value["output"]].append(value)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflicts: Counter[str] = Counter()
    for task, prompt_groups in grouped.items():
        for outputs in prompt_groups.values():
            if len(outputs) != 1:
                conflicts[task] += 1
                continue
            candidates[task].append(min(next(iter(outputs.values())), key=row_digest))
        candidates[task].sort(key=lambda row: replacement_key(task, row))

    replacements: list[dict[str, Any]] = []
    repeated: Counter[str] = Counter()
    unique_added: Counter[str] = Counter()
    for task in RETENTION_TASKS:
        positions = list(replacement_positions[task])
        chosen_fresh = []
        planned_prompts: set[str] = set()
        planned_modes: set[str] = set()
        for row in candidates[task]:
            prompt = prompt_digest(row)
            mode = mode_normalized_prompt_digest(row)
            if (
                prompt in used_prompts
                or mode in used_modes
                or prompt in planned_prompts
                or mode in planned_modes
            ):
                continue
            chosen_fresh.append(row)
            planned_prompts.add(prompt)
            planned_modes.add(mode)
            if len(chosen_fresh) == len(positions):
                break
        chosen: list[tuple[dict[str, Any], bool]] = [(row, False) for row in chosen_fresh]
        unique_added[task] = len(chosen_fresh)
        for row in chosen_fresh:
            used_prompts.add(prompt_digest(row))
            used_modes.add(mode_normalized_prompt_digest(row))

        shortage = len(positions) - len(chosen)
        if shortage:
            if task != "world":
                raise RuntimeError(f"{task}: short {shortage} unique E-clean retention prompts")
            repeat_pool = sorted(
                [row for row in assigned.values() if row["task"] == task] + chosen_fresh,
                key=lambda row: replacement_key(task, row),
            )
            if len(repeat_pool) < shortage:
                raise RuntimeError(f"world: only {len(repeat_pool)} safe rows for {shortage} repeats")
            chosen.extend((row, True) for row in repeat_pool[:shortage])
            repeated[task] = shortage

        if len(chosen) != len(positions):
            raise RuntimeError(f"{task}: retention replacement count drifted")
        for position, (row, is_repeat) in zip(positions, chosen):
            original = i30_rows[position]
            assigned[position] = {**row, "route": "retention_kl", "task": task}
            replacements.append(
                {
                    "position": position,
                    "task": task,
                    "original_row_sha256": row_digest(original),
                    "original_prompt_sha256": prompt_digest(original),
                    "original_mode_prompt_sha256": mode_normalized_prompt_digest(original),
                    "replacement_row_sha256": row_digest(row),
                    "replacement_prompt_sha256": prompt_digest(row),
                    "replacement_mode_prompt_sha256": mode_normalized_prompt_digest(row),
                    "replacement_is_second_exposure": is_repeat,
                    "upstream_asset": "data_user_residual_retention_v1",
                }
            )

    if len(assigned) != TRAIN_RETENTION_ROWS:
        raise RuntimeError(f"I-33 assigned {len(assigned)} retention rows")
    repeated_modes = Counter(
        mode_normalized_prompt_digest(row) for row in assigned.values()
    )
    duplicate_exposures = sum(count - 1 for count in repeated_modes.values())
    if duplicate_exposures != repeated["world"]:
        raise RuntimeError(
            f"unexpected retention duplicate exposures: {duplicate_exposures}/{dict(repeated)}"
        )
    audit = {
        "original_positions_replaced": dict(Counter(row["task"] for row in replacements)),
        "unique_replacements": dict(unique_added),
        "second_exposures": dict(repeated),
        "conflicting_source_prompt_groups_excluded": dict(conflicts),
        "unique_mode_prompts": len(repeated_modes),
        "rows": len(assigned),
    }
    return assigned, replacements, audit


def build_training(
    task_of,
    material_rows: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    i30_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
):
    forbidden_prompts = {prompt_digest(row) for row in forbidden_rows}
    forbidden_modes = {mode_normalized_prompt_digest(row) for row in forbidden_rows}
    material_sources = {
        task: {
            row_digest(row): normalized(row)
            for row in material_rows
            if classify(row, task_of) == task
        }
        for task in (TARGET_TASK, REVERSE_TASK)
    }
    desc512, desc_stats = select_material_metrics(
        TARGET_TASK,
        TRAIN_TARGET_ROWS,
        ledger,
        material_sources[TARGET_TASK],
        forbidden_prompts,
        forbidden_modes,
    )
    sid256, sid_stats = select_material_metrics(
        REVERSE_TASK,
        256,
        ledger,
        material_sources[REVERSE_TASK],
        forbidden_prompts,
        forbidden_modes,
    )
    desc_positions = [
        index
        for index, row in enumerate(i30_rows)
        if row.get("route") == "material_teacher" and row.get("task") == TARGET_TASK
    ]
    sid_positions = [
        index
        for index, row in enumerate(i30_rows)
        if row.get("route") == "material_teacher" and row.get("task") == REVERSE_TASK
    ]
    if len(desc_positions) != 256 or len(sid_positions) != 256:
        raise RuntimeError("I-30 material position signature drifted")

    shared_desc, retained_desc = assign_material_positions(
        i30_rows,
        desc_positions,
        desc512[:256],
        material_sources[TARGET_TASK],
        TARGET_TASK,
    )
    control_sid, retained_sid = assign_material_positions(
        i30_rows,
        sid_positions,
        sid256,
        material_sources[REVERSE_TASK],
        REVERSE_TASK,
    )
    treatment_extra, _ = assign_material_positions(
        i30_rows,
        sid_positions,
        desc512[256:],
        material_sources[TARGET_TASK],
        TARGET_TASK,
    )
    shared_retention, replacements, retention_audit = build_shared_retention(
        task_of,
        retention_rows,
        i30_rows,
        forbidden_prompts,
        forbidden_modes,
    )

    control = [dict(row) for row in i30_rows]
    treatment = [dict(row) for row in i30_rows]
    for position, row in shared_retention.items():
        control[position] = row
        treatment[position] = dict(row)
    for position, row in shared_desc.items():
        control[position] = row
        treatment[position] = dict(row)
    for position, row in control_sid.items():
        control[position] = row
    for position, row in treatment_extra.items():
        treatment[position] = row

    expected_routes = [row["route"] for row in i30_rows]
    for name, arm in (("control", control), ("treatment", treatment)):
        if [row["route"] for row in arm] != expected_routes:
            raise RuntimeError(f"I-33 {name} changed I-30 route positions")
        if any(
            prompt_digest(row) in forbidden_prompts
            or mode_normalized_prompt_digest(row) in forbidden_modes
            for row in arm
        ):
            raise RuntimeError(f"I-33 {name} contains a registered E/holdout prompt")
        if Counter(row["route"] for row in arm) != {
            "material_teacher": TRAIN_TARGET_ROWS,
            "retention_kl": TRAIN_RETENTION_ROWS,
        }:
            raise RuntimeError(f"I-33 {name} route count drifted")
    control_counts = Counter(row["task"] for row in control)
    treatment_counts = Counter(row["task"] for row in treatment)
    if control_counts[TARGET_TASK] != 256 or control_counts[REVERSE_TASK] != 256:
        raise RuntimeError(f"I-33 control material signature drifted: {dict(control_counts)}")
    if treatment_counts[TARGET_TASK] != 512 or treatment_counts[REVERSE_TASK] != 0:
        raise RuntimeError(f"I-33 treatment material signature drifted: {dict(treatment_counts)}")
    differences = [index for index, pair in enumerate(zip(control, treatment)) if pair[0] != pair[1]]
    if differences != sid_positions:
        raise RuntimeError("I-33 paired arms differ outside the 256 frozen SID slots")

    treatment_selection = [
        {
            **entry,
            "selection_rank": rank,
            "arm": "treatment",
            "source_role": "shared_control_desc2sid" if rank <= 256 else "replaces_control_sid2desc",
        }
        for rank, entry in enumerate(desc512, start=1)
    ]
    control_selection = []
    for task, entries in ((TARGET_TASK, desc512[:256]), (REVERSE_TASK, sid256)):
        for rank, entry in enumerate(entries, start=1):
            control_selection.append(
                {**entry, "selection_rank_within_task": rank, "arm": "control", "source_role": task}
            )
    audit = {
        "material": {
            TARGET_TASK: desc_stats,
            REVERSE_TASK: sid_stats,
            "control_existing_rows_retained": {
                TARGET_TASK: retained_desc,
                REVERSE_TASK: retained_sid,
            },
        },
        "retention": retention_audit,
        "paired_different_positions": len(differences),
        "paired_retention_rows_identical": all(
            control[index] == treatment[index] for index in shared_retention
        ),
    }
    return control, treatment, control_selection, treatment_selection, replacements, audit


def build_holdout(task_of, material_rows, retention_rows, excluded_rows):
    excluded = {prompt_digest(row) for row in excluded_rows}
    excluded_modes = {mode_normalized_prompt_digest(row) for row in excluded_rows}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in material_rows:
        task = classify(row, task_of)
        if (
            task in (TARGET_TASK, REVERSE_TASK)
            and prompt_digest(row) not in excluded
            and mode_normalized_prompt_digest(row) not in excluded_modes
        ):
            buckets[task].append(row)
    for row in retention_rows:
        task = classify(row, task_of)
        if (
            task in RETENTION_TASKS
            and prompt_digest(row) not in excluded
            and mode_normalized_prompt_digest(row) not in excluded_modes
        ):
            buckets[task].append(row)

    output = []
    used: set[str] = set()
    used_modes: set[str] = set()
    audit = {}
    for task, quota in HOLDOUT_QUOTAS.items():
        unique, conflicts = unique_nonconflicting(buckets[task])
        ranked_all = sorted(unique, key=lambda row: stable_key(task, row))
        ranked = [
            row
            for row in ranked_all
            if prompt_digest(row) not in used
            and mode_normalized_prompt_digest(row) not in used_modes
        ]
        if len(ranked) < quota:
            raise RuntimeError(f"{task}: only {len(ranked)} fresh prompts, need {quota}")
        chosen = ranked[:quota]
        used.update(prompt_digest(row) for row in chosen)
        used_modes.update(mode_normalized_prompt_digest(row) for row in chosen)
        output.extend({**row, "route": "gate_only", "task": task} for row in chosen)
        audit[task] = {
            "available": len(ranked),
            "selected": quota,
            "conflicting_prompt_groups_excluded": conflicts,
            "prompt_manifest_sha256": hashlib.sha256(
                "\n".join(prompt_digest(row) for row in chosen).encode()
            ).hexdigest(),
        }
    counts = Counter(row["task"] for row in output)
    expected_counts = Counter({task: quota for task, quota in HOLDOUT_QUOTAS.items() if quota})
    if counts != expected_counts:
        raise RuntimeError(f"I-33 holdout signature drifted: {dict(counts)}")
    if {prompt_digest(row) for row in output} & excluded:
        raise RuntimeError("I-33 holdout overlaps a training or earlier holdout prompt")
    if {mode_normalized_prompt_digest(row) for row in output} & excluded_modes:
        raise RuntimeError("I-33 holdout overlaps a mode-normalized construction prompt")
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert sum(HOLDOUT_QUOTAS.values()) == 720
        assert TRAIN_TARGET_ROWS + TRAIN_RETENTION_ROWS == 2048
        print("[i33-build] self-test passed")
        return

    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"locked I-33 upstream drift: {path}")
    generated = (
        CONTROL_OUTPUT,
        OUTPUT,
        HOLDOUT,
        CONTROL_SELECTION,
        SELECTION,
        RETENTION_REPLACEMENTS,
        AUDIT,
    )
    for path in generated:
        if path.exists():
            raise RuntimeError(f"I-33 refuses to overwrite existing artifact: {path}")

    task_of = load_classifier()
    material_rows = load_jsonl(MATERIAL_SOURCE)
    retention_rows = load_jsonl(RETENTION_SOURCE)
    i30_rows = load_jsonl(I30_TRAIN)
    ledger = load_jsonl(I30_LEDGER)
    e_rows_by_path = {path: load_jsonl(path) for path in E_HOLDOUT_PATHS}
    forbidden_rows = [row for path in E_HOLDOUT_PATHS for row in e_rows_by_path[path]]

    material_by_digest = {row_digest(row): normalized(row) for row in material_rows}
    ledger_source_rows = []
    for entry in ledger:
        source_row = material_by_digest.get(entry.get("row_sha256"))
        if source_row is None:
            raise RuntimeError(f"I-30 ledger row missing from material source: {entry.get('row_sha256')}")
        ledger_source_rows.append(source_row)

    (
        control,
        treatment,
        control_selection,
        treatment_selection,
        retention_replacements,
        training_audit,
    ) = build_training(
        task_of,
        material_rows,
        retention_rows,
        i30_rows,
        ledger,
        forbidden_rows,
    )
    holdout, holdout_audit = build_holdout(
        task_of,
        material_rows,
        retention_rows,
        forbidden_rows + ledger_source_rows + i30_rows + control + treatment,
    )
    atomic_jsonl(CONTROL_OUTPUT, control)
    atomic_jsonl(OUTPUT, treatment)
    atomic_jsonl(HOLDOUT, holdout)
    atomic_jsonl(CONTROL_SELECTION, control_selection)
    atomic_jsonl(SELECTION, treatment_selection)
    atomic_jsonl(RETENTION_REPLACEMENTS, retention_replacements)

    selected_deltas = [
        float(row["teacher_minus_parent_mean_logp"]) for row in treatment_selection
    ]
    forbidden_prompts = {prompt_digest(row) for row in forbidden_rows}
    forbidden_modes = {mode_normalized_prompt_digest(row) for row in forbidden_rows}
    original_e_positions = [
        position
        for position, row in enumerate(i30_rows)
        if prompt_digest(row) in forbidden_prompts
        or mode_normalized_prompt_digest(row) in forbidden_modes
    ]
    audit = {
        "schema": "i33-r96-material-desc2sid-retkl-v1",
        "asset_class": "D(O1,O2.*; M-I23/I19-world construction filter)",
        "training_seed": 19260831,
        "holdout_seed": HOLDOUT_SEED,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "upstream": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "rows": len(load_jsonl(path))}
            for path in EXPECTED_SHA256
        },
        "e_holdout_exclusion": {
            "assets": [str(path.relative_to(ROOT)) for path in E_HOLDOUT_PATHS],
            "rows": len(forbidden_rows),
            "unique_prompt_sha256": len(forbidden_prompts),
            "unique_mode_normalized_prompt_sha256": len(forbidden_modes),
            "i30_positions_requiring_sanitation": len(original_e_positions),
            "i30_positions_requiring_sanitation_by_task": dict(
                Counter(i30_rows[position]["task"] for position in original_e_positions)
            ),
            "control_exact_or_mode_overlap": 0,
            "treatment_exact_or_mode_overlap": 0,
        },
        "paired_design": {
            "parent_experiment": "I-30",
            "control": "256 material_desc2sid + 256 material_sid2desc",
            "treatment": "512 material_desc2sid",
            "different_positions": training_audit["paired_different_positions"],
            "different_position_role": "the 256 original I-30 material_sid2desc slots",
            "shared_retention_rows": TRAIN_RETENTION_ROWS,
            "shared_retention_rows_identical": training_audit[
                "paired_retention_rows_identical"
            ],
            "material_retention_route_order_identical_to_i30": True,
            "compliance_sanitation_applied_equally_to_both_arms": True,
        },
        "material_selection": {
            "source": str(I30_LEDGER.relative_to(ROOT)),
            "rule": "descending frozen I23-minus-r96 advantage, strictly positive, then row SHA256; exclude every registered E/holdout by exact and mode-normalized prompt",
            "statistics": training_audit["material"],
            "treatment_selected": len(treatment_selection),
            "control_selected": len(control_selection),
            "teacher_minus_parent_delta_min": min(selected_deltas),
            "teacher_minus_parent_delta_mean": sum(selected_deltas) / len(selected_deltas),
            "teacher_minus_parent_delta_max": max(selected_deltas),
            "treatment_ledger": str(SELECTION.relative_to(ROOT)),
            "treatment_ledger_sha256": sha256(SELECTION),
            "control_ledger": str(CONTROL_SELECTION.relative_to(ROOT)),
            "control_ledger_sha256": sha256(CONTROL_SELECTION),
        },
        "retention_sanitation": {
            **training_audit["retention"],
            "rule": "same task and same registered retention upstream; unique E-clean prompts first; world-only deterministic second exposure when the unique clean pool is exhausted",
            "replacement_ledger": str(RETENTION_REPLACEMENTS.relative_to(ROOT)),
            "replacement_ledger_sha256": sha256(RETENTION_REPLACEMENTS),
        },
        "training_arms": {
            "control": {
                "path": str(CONTROL_OUTPUT.relative_to(ROOT)),
                "rows": len(control),
                "sha256": sha256(CONTROL_OUTPUT),
                "task_counts": dict(Counter(row["task"] for row in control)),
            },
            "treatment": {
                "path": str(OUTPUT.relative_to(ROOT)),
                "rows": len(treatment),
                "sha256": sha256(OUTPUT),
                "task_counts": dict(Counter(row["task"] for row in treatment)),
            },
            "mix": {
                "material": {"rows": TRAIN_TARGET_ROWS, "ratio": 0.25},
                "retention_kl": {"rows": TRAIN_RETENTION_ROWS, "ratio": 0.75},
            },
        },
        "holdout": {
            "path": str(HOLDOUT.relative_to(ROOT)),
            "rows": len(holdout),
            "sha256": sha256(HOLDOUT),
            "counts": dict(Counter(row["task"] for row in holdout)),
            "prompt_overlap_i30_train_all_registered_e_control_treatment": 0,
            "prompt_overlap_full_i30_construction_ledger": 0,
            "mode_normalized_prompt_overlap_all_exclusions": 0,
            "world_policy": "no new rows; use already frozen registered world gates because the shared E-clean retention consumes all 106 available unique world prompts",
            "allocation": holdout_audit,
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT.with_suffix(AUDIT.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(AUDIT)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
