#!/usr/bin/env python3
"""Freeze an I-34 material beam candidate, development, and retention pool.

This builder only freezes diagnostic pools.  It does not load a model, create a
training projection, or update the asset registry.  Every registered prompt
asset is checked by an explicit, hash-locked manifest before selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SEED = 19260834
SCHEMA_VERSION = "i34-material-beam-pool-v1"
TARGET_TASK = "material_desc2sid"

SOURCE = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
RETENTION_SOURCE = ROOT / "assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl"
I30_TRAIN = ROOT / "assets/derived/processed/data_i30_r96_material_teacher_retkl_v1.jsonl"
I33_TRAIN = ROOT / "assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl"

HOLDOUT_DIR = ROOT / "assets/evaluation/holdout"
OFFLINE_DIR = ROOT / "assets/evaluation/offline_eval"
VISIBLE_DIR = ROOT / "assets/evaluation/visible"

# These are the seven current holdout files registered in ASSETS.md.  Keeping
# the list explicit makes an unregistered addition fail closed.
E_HOLDOUT_PATHS = (
    HOLDOUT_DIR / "data_i28_video_multigold_proposal_v1_gate.jsonl",
    HOLDOUT_DIR / "data_i30_r96_material_teacher_gate_v1.jsonl",
    HOLDOUT_DIR / "data_i32_task_restore_gate_v1.jsonl",
    HOLDOUT_DIR / "data_i33_r96_material_desc2sid_gate_v1.jsonl",
    HOLDOUT_DIR / "data_o1_reward_preference_v1_holdout.jsonl",
    HOLDOUT_DIR / "official_general_world_mc_v1_holdout.jsonl",
    HOLDOUT_DIR / "s800_native_general_replay_retention_gate_v1.jsonl",
)
E_DERIVED_PATHS = (
    ROOT / "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl",
)
E_OFFLINE_PATHS = tuple(
    OFFLINE_DIR / name
    for name in (
        "dev_action.jsonl",
        "dev_mat_fresh.jsonl",
        "dev_mat_train.jsonl",
        "dev_rec_ad.jsonl",
        "dev_rec_ad_v2_exally.jsonl",
        "dev_rec_live.jsonl",
        "dev_rec_live_v2_exally.jsonl",
        "dev_rec_prod.jsonl",
        "dev_rec_prod_v2_exally.jsonl",
        "dev_rec_video.jsonl",
        "dev_rec_video_v2_exally.jsonl",
        "dev_topic.jsonl",
        "dev_world.jsonl",
    )
)
E_VISIBLE_PATHS = (VISIBLE_DIR / "懂世界.jsonl",)
E_PATHS = E_HOLDOUT_PATHS + E_DERIVED_PATHS + E_OFFLINE_PATHS + E_VISIBLE_PATHS

CANDIDATE_OUTPUT = ROOT / "logs/data/i34_material_beam_candidate_pool_v1.jsonl"
DEV_OUTPUT = ROOT / "assets/evaluation/holdout/data_i34_material_beam_dev_v1.jsonl"
RETENTION_OUTPUT = ROOT / "logs/data/i34_material_beam_retention_pool_v1.jsonl"
AUDIT_OUTPUT = ROOT / "logs/data/i34_material_beam_pool_v1_audit.json"

DEV_ROWS = 256
CANDIDATE_ROWS = 1024
RETENTION_QUOTAS = {
    "action": 55,
    "topic": 55,
    "rec_video": 55,
    "rec_prod": 55,
    "rec_ad": 55,
    "rec_living": 55,
    "world": 54,
}

SID_RE = re.compile(
    r"^<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>$"
)


# The locks cover the source, both formal material arms, and every E path.
# They are intentionally kept in the builder so a changed input cannot silently
# produce a different frozen pool.
EXPECTED_LOCKS: dict[Path, tuple[int, str]] = {
    SOURCE: (32644, "13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f"),
    I30_TRAIN: (2048, "0df9a192976eb61eb8dd333fd59edb994d1fcad482710e1282f36dd792bfc4a4"),
    I33_TRAIN: (2048, "7d6a1e4a44238a79dcb0d31384f147c02baea95cd870224e2a6815444f8470fd"),
    E_HOLDOUT_PATHS[0]: (128, "48dd7f4224e7ca9e98805d966ca901814fdb76b85471afcf1ec7d98a0c22c7e5"),
    E_HOLDOUT_PATHS[1]: (704, "dd744ee2d2f584b9bcae938cde1f5976801a9eece39aa1972b284641603f97a0"),
    E_HOLDOUT_PATHS[2]: (656, "f75106758792163dd33d1d52639ba507a6d9e69094d8213d5f3b0969ee272f62"),
    E_HOLDOUT_PATHS[3]: (720, "76acc6a39b248e4501a11e99bc889871f5208c811070485791b554538e658f99"),
    E_HOLDOUT_PATHS[4]: (1784, "1c7292cb96d45e9d20c0b3add78d3e5a30ec7a559844217584408921f996696e"),
    E_HOLDOUT_PATHS[5]: (25, "fb67b76d8d071799ba372185bd89cb556afef9065a1b188fb9dd86a9131e13df"),
    E_HOLDOUT_PATHS[6]: (256, "3206e91ac465ca4f1410e3f8a9219a60c11cbb1beb3d4eb2fa9fa69c4b89c30f"),
    E_DERIVED_PATHS[0]: (46, "8aa4306f139afc0a00cacd91508de90aa9fa2cbd9942af9cdb665d895721402a"),
    E_OFFLINE_PATHS[0]: (325, "3f99f0f2a1264edf9078880b5b1b6454c4ee169c18945d25214797aba269ff86"),
    E_OFFLINE_PATHS[1]: (542, "1152ed871edef13c75484943e7e83005b23c4e5f5cf2438d8c526504d7093fb9"),
    E_OFFLINE_PATHS[2]: (300, "266428eb9df1e45be44d68672b95ce900dd7bf40832b7add3f7a3a50cdc3ebfb"),
    E_OFFLINE_PATHS[3]: (1000, "18eade5d595a7a87883931e21dc8af112911834f55e1a5d3e68258c077399461"),
    E_OFFLINE_PATHS[4]: (1000, "314def91e51b9ec6116dac6a2aae5fa5cabfeea3ab9c0701cde47297db59c8d1"),
    E_OFFLINE_PATHS[5]: (1000, "1e53f06db0914bc1d8bf546ce0ee462b430607623e174619460734e4260d1c10"),
    E_OFFLINE_PATHS[6]: (1000, "8550f41bec1bcef2e4804fbc4857d5ea7b65f123c913ceb9d08e78125b906bc1"),
    E_OFFLINE_PATHS[7]: (1000, "578edb886e406add39863d4a8118714c921bffd5f5c2ec8a341e794905168338"),
    E_OFFLINE_PATHS[8]: (1000, "2d099d7183f053dbc700b2c96def438f2f3124418c5443044dde7f92a5b36010"),
    E_OFFLINE_PATHS[9]: (1000, "755b8537ea85145dee4153f444298f559518014e48cbdda34d67569cc3cdaca5"),
    E_OFFLINE_PATHS[10]: (1000, "d0f711631f992eb5ae1069e60eb97f8a70eb34a71b627a851e793dba735b7865"),
    E_OFFLINE_PATHS[11]: (110, "1204e407737b603ccba957401281042ac70463c9d571a87fcc423e99318c5ec5"),
    E_OFFLINE_PATHS[12]: (500, "c506c78060e4b56cb6a13467c1c510e62b86ad75b89cba72851157a90d7c2d67"),
    E_VISIBLE_PATHS[0]: (7, "d7f341e2277473ff3b5b556531370a7697471cc5e540586a53c93c6adb32b5e5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    # The first two fallbacks mirror I-33; user is added for the registered
    # offline_eval system/user schema.
    instruction = row.get("instruction")
    if instruction is None:
        instruction = row.get("system", "")
    input_text = row.get("input")
    if input_text is None:
        input_text = row.get("prompt")
    if input_text is None:
        input_text = row.get("user", "")
    output = row.get("output")
    if output is None:
        output = row.get("response", "")
    return {
        "instruction": str(instruction or ""),
        "input": str(input_text or ""),
        "output": str(output or ""),
        "history": row.get("history") or [],
    }


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(normalized(row)).encode()).hexdigest()


def prompt_digest(row: dict[str, Any]) -> str:
    value = normalized(row)
    return hashlib.sha256(
        canonical([value["instruction"], value["input"], value["history"]]).encode()
    ).hexdigest()


def mode_prompt_digest(row: dict[str, Any]) -> str:
    value = normalized(row)
    input_text = value["input"].rstrip()
    for suffix in ("/no_think", "/think"):
        if input_text.endswith(suffix):
            input_text = input_text[: -len(suffix)].rstrip()
            break
    return hashlib.sha256(
        canonical([value["instruction"], input_text, value["history"]]).encode()
    ).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(item, dict) for item in values):
                raise RuntimeError(f"{path}:{line_number} is not an object or object array")
            rows.extend(values)
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def assert_manifest() -> dict[str, dict[str, Any]]:
    expected_holdout = {path.resolve() for path in E_HOLDOUT_PATHS}
    observed_holdout = {path.resolve() for path in HOLDOUT_DIR.glob("*.jsonl") if path.is_file()}
    if observed_holdout != expected_holdout:
        missing = sorted(str(path) for path in expected_holdout - observed_holdout)
        extra = sorted(str(path) for path in observed_holdout - expected_holdout)
        raise RuntimeError(f"holdout manifest drift; missing={missing}, extra={extra}")

    expected_offline = {path.resolve() for path in E_OFFLINE_PATHS}
    observed_offline = {
        path.resolve() for path in OFFLINE_DIR.glob("*.jsonl") if path.is_file()
    }
    if observed_offline != expected_offline:
        missing = sorted(str(path) for path in expected_offline - observed_offline)
        extra = sorted(str(path) for path in observed_offline - expected_offline)
        raise RuntimeError(f"offline_eval manifest drift; missing={missing}, extra={extra}")

    expected_visible = {path.resolve() for path in E_VISIBLE_PATHS}
    observed_visible = {
        path.resolve() for path in VISIBLE_DIR.glob("*.jsonl") if path.is_file()
    }
    if observed_visible != expected_visible:
        missing = sorted(str(path) for path in expected_visible - observed_visible)
        extra = sorted(str(path) for path in observed_visible - expected_visible)
        raise RuntimeError(f"visible manifest drift; missing={missing}, extra={extra}")

    manifest: dict[str, dict[str, Any]] = {}
    for path, (expected_rows, expected_hash) in EXPECTED_LOCKS.items():
        if not path.is_file():
            raise RuntimeError(f"registered input is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 drift for {path}: {actual_hash} != {expected_hash}")
        rows = len(load_jsonl(path))
        if rows != expected_rows:
            raise RuntimeError(f"row-count drift for {path}: {rows} != {expected_rows}")
        manifest[str(path.relative_to(ROOT))] = {
            "rows": rows,
            "sha256": actual_hash,
        }
    return manifest


def load_task_classifier():
    helper_path = ROOT / "scripts/data/build_seed_scoremax_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_i34_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper.task_of


def classify(row: dict[str, Any], task_of) -> str:
    try:
        return task_of(normalized(row))
    except ValueError:
        if "<s_a_" not in canonical(normalized(row)):
            return "world"
        raise


def answer_body(row: dict[str, Any]) -> str:
    output = normalized(row)["output"]
    if "</think>" not in output:
        raise RuntimeError("material row is missing </think>")
    return output.split("</think>", 1)[1].strip()


def parse_gold_sid(row: dict[str, Any]) -> tuple[str, int, int, int, str]:
    body = answer_body(row)
    match = SID_RE.fullmatch(body)
    if match is None:
        raise RuntimeError(f"material_desc2sid row has non-single gold: {body[:160]!r}")
    domain = match.group("domain")
    a, b, c = (int(match.group(name)) for name in ("a", "b", "c"))
    token = match.group(0)
    return domain, a, b, c, token


def beam_prefix_compatible(row: dict[str, Any]) -> bool:
    """Match the beam runner's exact empty-think/no-think prefix contract."""
    value = normalized(row)
    if not value["input"].endswith("/no_think"):
        return False
    output = value["output"]
    if output.count("</think>") != 1 or not output.startswith("<think>"):
        return False
    think_prefix, _answer = output.split("</think>", 1)
    return think_prefix[len("<think>") :].strip() == ""


def stable_key(*parts: object) -> str:
    return hashlib.sha256(canonical(list(parts)).encode()).hexdigest()


def dedupe_material(rows: list[dict[str, Any]], task_of) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_rows = []
    task_counts: Counter[str] = Counter()
    for raw in rows:
        row = normalized(raw)
        task = classify(row, task_of)
        task_counts[task] += 1
        if task == TARGET_TASK:
            parse_gold_sid(row)
            target_rows.append(row)
    by_sid: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        gold = parse_gold_sid(row)
        by_sid[gold[:4]].append(row)
    unique = [min(group, key=row_digest) for group in by_sid.values()]
    unique.sort(key=lambda row: (parse_gold_sid(row)[:4], row_digest(row)))
    return unique, {
        "source_task_counts": dict(sorted(task_counts.items())),
        "target_rows_before_sid_dedupe": len(target_rows),
        "unique_full_gold_sid_rows": len(unique),
        "duplicate_full_gold_sid_rows": len(target_rows) - len(unique),
        "full_gold_sid_groups": len(by_sid),
    }


def select_group_subset(
    groups: dict[tuple[str, int, int], list[dict[str, Any]]],
    target: int,
    namespace: str,
) -> set[tuple[str, int, int]]:
    ordered = sorted(
        groups.items(), key=lambda item: stable_key(SEED, namespace, item[0])
    )
    # Parent pointers keep the exact-sum DP bounded by the requested row count.
    dp: dict[int, tuple[int, int] | None] = {0: None}
    for index, (group_key, group_rows) in enumerate(ordered):
        size = len(group_rows)
        if size > target:
            continue
        for subtotal in sorted(tuple(dp), reverse=True):
            new_total = subtotal + size
            if new_total <= target and new_total not in dp:
                dp[new_total] = (index, subtotal)
        if target in dp:
            break
    if target not in dp:
        raise RuntimeError(f"cannot form exact {namespace} quota of {target} rows by whole groups")
    chosen_indices: set[int] = set()
    subtotal = target
    while subtotal:
        parent = dp[subtotal]
        if parent is None:
            raise AssertionError("broken group subset parent chain")
        index, previous = parent
        chosen_indices.add(index)
        subtotal = previous
    return {ordered[index][0] for index in chosen_indices}


def make_material_record(
    row: dict[str, Any], role: str, rank: int, group_rank: int
) -> dict[str, Any]:
    value = normalized(row)
    domain, a, b, c, token = parse_gold_sid(value)
    return {
        **value,
        "schema_version": SCHEMA_VERSION,
        "pool_role": role,
        "task": TARGET_TASK,
        "selection_rank": rank,
        "group_rank": group_rank,
        "gold_sid": token,
        "gold_domain": domain,
        "gold_s_a": a,
        "gold_s_b": b,
        "gold_s_c": c,
        "prefix_group": f"{domain}:{a}:{b}",
        "row_sha256": row_digest(value),
        "prompt_sha256": prompt_digest(value),
        "mode_prompt_sha256": mode_prompt_digest(value),
    }


def build_material_pools(
    source_rows: list[dict[str, Any]],
    task_of,
    e_prompts: set[str],
    e_modes: set[str],
    train_prompts: set[str],
    train_modes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], set[str], set[str]]:
    unique, audit = dedupe_material(source_rows, task_of)
    eligible: list[dict[str, Any]] = []
    excluded = Counter()
    for row in unique:
        if not beam_prefix_compatible(row):
            excluded["beam_prefix_requires_exact_no_think_empty_think"] += 1
            continue
        prompt = prompt_digest(row)
        mode = mode_prompt_digest(row)
        if prompt in e_prompts or mode in e_modes:
            excluded["registered_E_exact_or_mode"] += 1
            continue
        if prompt in train_prompts or mode in train_modes:
            excluded["I30_I33_training_exact_or_mode"] += 1
            continue
        eligible.append(row)

    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        domain, a, b, _c, _token = parse_gold_sid(row)
        groups[(domain, a, b)].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (parse_gold_sid(row)[3], row_digest(row)))

    dev_groups = select_group_subset(groups, DEV_ROWS, "development")
    remaining_groups = {key: values for key, values in groups.items() if key not in dev_groups}
    candidate_groups = select_group_subset(remaining_groups, CANDIDATE_ROWS, "candidate")

    def ordered_rows(selected_groups: set[tuple[str, int, int]]) -> list[dict[str, Any]]:
        return [
            row
            for key in sorted(selected_groups, key=lambda value: stable_key(SEED, "rows", value))
            for row in groups.get(key, remaining_groups.get(key, []))
        ]

    dev_rows = ordered_rows(dev_groups)
    candidate_rows = ordered_rows(candidate_groups)
    if len(dev_rows) != DEV_ROWS or len(candidate_rows) != CANDIDATE_ROWS:
        raise AssertionError("material group selection count drift")

    dev_records = [make_material_record(row, "development", rank, rank) for rank, row in enumerate(dev_rows, 1)]
    candidate_records = [
        make_material_record(row, "candidate", rank, rank) for rank, row in enumerate(candidate_rows, 1)
    ]
    dev_prompt = {prompt_digest(row) for row in dev_rows}
    candidate_prompt = {prompt_digest(row) for row in candidate_rows}
    dev_mode = {mode_prompt_digest(row) for row in dev_rows}
    candidate_mode = {mode_prompt_digest(row) for row in candidate_rows}
    dev_sid = {parse_gold_sid(row)[:4] for row in dev_rows}
    candidate_sid = {parse_gold_sid(row)[:4] for row in candidate_rows}
    dev_group_set = {parse_gold_sid(row)[:3] for row in dev_rows}
    candidate_group_set = {parse_gold_sid(row)[:3] for row in candidate_rows}
    if dev_group_set & candidate_group_set:
        raise AssertionError("development/candidate group intersection")
    if dev_prompt & candidate_prompt or dev_mode & candidate_mode or dev_sid & candidate_sid:
        raise AssertionError("development/candidate prompt, mode, or SID intersection")
    audit.update(
        {
            "eligible_after_E_and_training_exclusion": len(eligible),
            "beam_prefix_compatible_unique_sid_rows": len(eligible)
            + excluded["registered_E_exact_or_mode"]
            + excluded["I30_I33_training_exact_or_mode"],
            "eligible_group_count": len(groups),
            "excluded_unique_sid_rows": dict(sorted(excluded.items())),
            "development_rows": len(dev_rows),
            "development_group_count": len(dev_groups),
            "candidate_rows": len(candidate_rows),
            "candidate_group_count": len(candidate_groups),
            "eligible_domain_counts": dict(
                sorted(Counter(parse_gold_sid(row)[0] for row in eligible).items())
            ),
            "development_domain_counts": dict(
                sorted(Counter(parse_gold_sid(row)[0] for row in dev_rows).items())
            ),
            "candidate_domain_counts": dict(
                sorted(Counter(parse_gold_sid(row)[0] for row in candidate_rows).items())
            ),
            "development_candidate_cross_intersection": {
                "group": len(dev_group_set & candidate_group_set),
                "prompt": len(dev_prompt & candidate_prompt),
                "mode": len(dev_mode & candidate_mode),
                "full_gold_sid": len(dev_sid & candidate_sid),
            },
        }
    )
    return candidate_records, dev_records, audit, dev_prompt | candidate_prompt, dev_mode | candidate_mode


def make_retention_record(row: dict[str, Any], task: str, rank: int) -> dict[str, Any]:
    value = normalized(row)
    return {
        **value,
        "schema_version": SCHEMA_VERSION,
        "pool_role": "retention",
        "route": "retention_reference",
        "task": task,
        "selection_rank": rank,
        "source_row_sha256": row_digest(value),
        "prompt_sha256": prompt_digest(value),
        "mode_prompt_sha256": mode_prompt_digest(value),
    }


def build_retention_pool(
    rows: list[dict[str, Any]],
    e_prompts: set[str],
    e_modes: set[str],
    material_prompts: set[str],
    material_modes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    excluded = Counter()
    for raw in rows:
        if raw.get("route") != "retention_kl":
            continue
        task = str(raw.get("task", ""))
        if task not in RETENTION_QUOTAS:
            raise RuntimeError(f"unexpected I-33 retention task: {task!r}")
        value = normalized(raw)
        prompt = prompt_digest(value)
        mode = mode_prompt_digest(value)
        if prompt in e_prompts or mode in e_modes:
            excluded["registered_E_exact_or_mode"] += 1
            continue
        if prompt in material_prompts or mode in material_modes:
            excluded["material_candidate_or_dev_exact_or_mode"] += 1
            continue
        existing = candidates[task].get(mode)
        if existing is not None and existing["output"] != value["output"]:
            raise RuntimeError(f"conflicting retention outputs for mode prompt {mode}")
        if existing is None or row_digest(value) < row_digest(existing):
            candidates[task][mode] = value

    selected: list[dict[str, Any]] = []
    selected_prompts: set[str] = set()
    selected_modes: set[str] = set()
    per_task: dict[str, int] = {}
    for task, quota in RETENTION_QUOTAS.items():
        ranked = sorted(
            candidates[task].values(),
            key=lambda row: stable_key(SEED, "retention", task, row_digest(row)),
        )
        if len(ranked) < quota:
            raise RuntimeError(f"retention task {task} has {len(ranked)} rows; need {quota}")
        chosen = ranked[:quota]
        for row in chosen:
            prompt = prompt_digest(row)
            mode = mode_prompt_digest(row)
            if prompt in selected_prompts or mode in selected_modes:
                raise AssertionError("retention task selection cross-over")
            selected_prompts.add(prompt)
            selected_modes.add(mode)
        selected.extend(make_retention_record(row, task, rank) for rank, row in enumerate(chosen, 1))
        per_task[task] = len(chosen)

    if len(selected) != sum(RETENTION_QUOTAS.values()):
        raise AssertionError("retention row count drift")
    if selected_prompts & e_prompts or selected_modes & e_modes:
        raise AssertionError("retention intersects E blacklist")
    if selected_prompts & material_prompts or selected_modes & material_modes:
        raise AssertionError("retention intersects material candidate/development")
    return selected, {
        "source": str(RETENTION_SOURCE.relative_to(ROOT)),
        "source_role": "I33 E-clean retention_kl; intentionally not treated as new CE",
        "rows_after_exclusion_and_mode_dedupe": {
            task: len(values) for task, values in sorted(candidates.items())
        },
        "excluded": dict(sorted(excluded.items())),
        "fixed_quotas": RETENTION_QUOTAS,
        "selected_by_task": per_task,
        "selected_rows": len(selected),
    }


def set_intersection_audit(
    candidate: list[dict[str, Any]], dev: list[dict[str, Any]], retention: list[dict[str, Any]], e_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]
) -> dict[str, int]:
    def sets(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
        return ({prompt_digest(row) for row in rows}, {mode_prompt_digest(row) for row in rows})

    cp, cm = sets(candidate)
    dp, dm = sets(dev)
    rp, rm = sets(retention)
    ep, em = sets(e_rows)
    tp, tm = sets(train_rows)
    return {
        "candidate_vs_E_prompt": len(cp & ep),
        "candidate_vs_E_mode": len(cm & em),
        "development_vs_E_prompt": len(dp & ep),
        "development_vs_E_mode": len(dm & em),
        "retention_vs_E_prompt": len(rp & ep),
        "retention_vs_E_mode": len(rm & em),
        "candidate_vs_training_prompt": len(cp & tp),
        "candidate_vs_training_mode": len(cm & tm),
        "development_vs_training_prompt": len(dp & tp),
        "development_vs_training_mode": len(dm & tm),
    }


def self_test() -> None:
    base = {
        "instruction": "i",
        "input": "p/think",
        "output": "<think>\n</think>\nx",
        "history": [],
    }
    changed = dict(base, input="p/no_think")
    if prompt_digest(base) == prompt_digest(changed):
        raise AssertionError("exact prompt digest unexpectedly ignores mode")
    if mode_prompt_digest(base) != mode_prompt_digest(changed):
        raise AssertionError("mode digest does not normalize suffix")
    compatible = dict(
        base,
        input="p/no_think",
        output="<think>\n\n</think>\n<|prod_begin|><s_a_1><s_b_2><s_c_3>",
    )
    incompatible = dict(
        compatible,
        output="<think>non-empty</think>\n<|prod_begin|><s_a_1><s_b_2><s_c_3>",
    )
    if not beam_prefix_compatible(compatible) or beam_prefix_compatible(incompatible):
        raise AssertionError("beam prefix contract self-test failed")
    parsed = parse_gold_sid(
        dict(base, output="<think>\n</think>\n<|prod_begin|><s_a_1><s_b_2><s_c_3>")
    )
    if parsed[:4] != ("prod", 1, 2, 3):
        raise AssertionError("SID parser self-test failed")
    synthetic = {
        ("prod", 1, 1): [dict(base, output=str(i)) for i in range(3)],
        ("prod", 2, 1): [dict(base, output=str(i)) for i in range(2)],
        ("video", 3, 1): [dict(base, output=str(i)) for i in range(1)],
    }
    picked = select_group_subset(synthetic, 3, "self-test")
    if sum(len(synthetic[key]) for key in picked) != 3:
        raise AssertionError("group subset self-test failed")
    if sum(RETENTION_QUOTAS.values()) != 384:
        raise AssertionError("retention quota self-test failed")
    print("[i34-build] self-test passed")


def build() -> dict[str, Any]:
    for output in (CANDIDATE_OUTPUT, DEV_OUTPUT, RETENTION_OUTPUT, AUDIT_OUTPUT):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite frozen output: {output}")

    manifest = assert_manifest()
    task_of = load_task_classifier()
    source_rows = load_jsonl(SOURCE)
    i30_rows = load_jsonl(I30_TRAIN)
    i33_rows = load_jsonl(I33_TRAIN)
    e_rows: list[dict[str, Any]] = []
    for path in E_PATHS:
        e_rows.extend(load_jsonl(path))
    e_prompts = {prompt_digest(row) for row in e_rows}
    e_modes = {mode_prompt_digest(row) for row in e_rows}
    train_rows = i30_rows + i33_rows
    train_prompts = {prompt_digest(row) for row in train_rows}
    train_modes = {mode_prompt_digest(row) for row in train_rows}

    candidate, dev, material_audit, material_pool_prompts, material_pool_modes = build_material_pools(
        source_rows,
        task_of,
        e_prompts,
        e_modes,
        train_prompts,
        train_modes,
    )
    retention, retention_audit = build_retention_pool(
        i33_rows,
        e_prompts,
        e_modes,
        material_pool_prompts,
        material_pool_modes,
    )

    # Validate the final records through the same normalized digest contract.
    cross = set_intersection_audit(candidate, dev, retention, e_rows, train_rows)
    if any(cross.values()):
        raise AssertionError(f"non-zero forbidden intersection: {cross}")

    atomic_jsonl(CANDIDATE_OUTPUT, candidate)
    atomic_jsonl(DEV_OUTPUT, dev)
    atomic_jsonl(RETENTION_OUTPUT, retention)
    builder_hash = sha256(Path(__file__))
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "builder": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": builder_hash,
        },
        "scope": {
            "model_loaded": False,
            "training_started": False,
            "training_projection_created": False,
            "asset_registry_updated": False,
            "docs_updated": False,
            "candidate_and_retention_are_logs_data_only": True,
        },
        "source": {
            "path": str(SOURCE.relative_to(ROOT)),
            "rows": len(source_rows),
            "sha256": sha256(SOURCE),
            "task": TARGET_TASK,
            "upstream_asset_id": "data_seed_teacher_v1",
        },
        "e_manifest": manifest,
        "e_blacklist": {
            "paths": [str(path.relative_to(ROOT)) for path in E_PATHS],
            "rows": len(e_rows),
            "unique_exact_prompts": len(e_prompts),
            "unique_mode_prompts": len(e_modes),
            "system_user_schema_supported": True,
        },
        "training_exclusion": {
            "paths": [str(path.relative_to(ROOT)) for path in (I30_TRAIN, I33_TRAIN)],
            "rows": len(train_rows),
            "unique_exact_prompts": len(train_prompts),
            "unique_mode_prompts": len(train_modes),
            "formal_training_prompt_exclusion_applies_to_material_pools": True,
            "retention_source_is_explicit_I33_E_clean_exception": True,
        },
        "material": material_audit,
        "retention": retention_audit,
        "final_cross_intersections": cross,
        "outputs": {
            "candidate": {
                "path": str(CANDIDATE_OUTPUT.relative_to(ROOT)),
                "rows": len(candidate),
                "sha256": sha256(CANDIDATE_OUTPUT),
            },
            "development": {
                "path": str(DEV_OUTPUT.relative_to(ROOT)),
                "rows": len(dev),
                "sha256": sha256(DEV_OUTPUT),
                "role": "evaluation holdout; not training input",
            },
            "retention": {
                "path": str(RETENTION_OUTPUT.relative_to(ROOT)),
                "rows": len(retention),
                "sha256": sha256(RETENTION_OUTPUT),
            },
        },
    }
    atomic_json(AUDIT_OUTPUT, audit)
    print(json.dumps(audit["outputs"], ensure_ascii=False, indent=2, sort_keys=True))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic unit checks only")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    build()


if __name__ == "__main__":
    main()
