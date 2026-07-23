#!/usr/bin/env python3
"""Build the I-39 user-overlap video AB/first-divergence beam pool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SEED = 19260839

O3 = ROOT / "assets/official/sft_aligned/baseline_caption_tag_lists.parquet"
I36 = ROOT / "assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl"
I35_FORMAL = ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"
O1_FORWARD = tuple(
    ROOT / "assets/official/seed_sft" / f"懂物料part{index}.jsonl"
    for index in range(1, 5)
)

OUTPUT = ROOT / "logs/data/i39_userab_video_beam64_pool_v1.jsonl"
DEV_OUTPUT = ROOT / "logs/data/i39_userab_video_beam64_pool_v1_dev.jsonl"
AUDIT = ROOT / "logs/data/i39_userab_video_beam64_pool_v1_audit.json"

O3_ROWS = 19_204
O3_SHA256 = "c307fe6d723ebdebc2d343de3481bdc878f6193d65456e3b933ae7f6b78b8d9d"
I36_ROWS = 16_500
I36_SHA256 = "2720746a2e8aa7804d519698ce9f2b127e9be2db1d4488e642e800a5337b692d"
I35_FORMAL_ROWS = 2_740
I35_FORMAL_SHA256 = "9c044e47d26fb7644281107a548249e49564e0f203a04795337c6a90c0927100"
O1_FORWARD_LOCKS = (
    (1_611, "d75526a52a96806088a21634b0f3c9e989d724c152a61a9682398cfbda70011d"),
    (784, "e4f152391cf344e43c8a49d4230ce8a18a10d4d7b0ae8271a3b679d6fb37f781"),
    (1_581, "713e3d81cbcc3e28e962738e8d63d791fac42c4ce54f5ffe421fcbba23179d21"),
    (1_621, "0a35f02b229e6b8e0d7e884a65bf12003d899f99914de277f88b1978959deccc"),
)

PRIMARY_AB_ROWS = 2_560
PAIRED_AB_ROWS = 512
TOTAL_ROWS = PRIMARY_AB_ROWS + PAIRED_AB_ROWS

OFFICIAL_SYSTEM = "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。"
OFFICIAL_USER_PREFIX = "请解析以下视频内容并输出对应的视频token：\n\n"
SID_RE = re.compile(
    r"^<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>$"
)
SID_FIND_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)


class ContractError(RuntimeError):
    """Raised when a registered source or derived invariant drifts."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stable_hash(*parts: Any) -> str:
    return digest([SEED, *parts])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def strip_mode(value: str) -> str:
    return re.sub(r"/(?:no_)?think\s*$", "", value.rstrip()).rstrip()


def normalized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(
            row.get("input", row.get("prompt", row.get("user", ""))) or ""
        ),
        "output": str(
            row.get("output", row.get("response", row.get("gold", ""))) or ""
        ),
        "history": row.get("history") or [],
    }


def prompt_digest(row: Mapping[str, Any], *, mode_normalized: bool = False) -> str:
    value = normalized_row(row)
    user_input = strip_mode(value["input"]) if mode_normalized else value["input"]
    return digest([value["instruction"], user_input, value["history"]])


def description_text(user_input: str) -> str:
    value = strip_mode(user_input)
    if value.startswith(OFFICIAL_USER_PREFIX):
        return normalize_text(value[len(OFFICIAL_USER_PREFIX) :])
    if "：" in value:
        return normalize_text(value.split("：", 1)[1])
    if ":\n" in value:
        return normalize_text(value.split(":\n", 1)[1])
    return normalize_text(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            values = value if isinstance(value, list) else [value]
            require(
                len(values) == 1 and isinstance(values[0], dict),
                f"expected one object at {path}:{line_number}",
            )
            rows.append(values[0])
    return rows


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_i34_e_manifest() -> tuple[tuple[Path, ...], dict[Path, tuple[int, str]]]:
    helper_path = ROOT / "scripts/data/build_i34_material_beam_pool_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_i39_e_manifest", helper_path)
    require(spec is not None and spec.loader is not None, f"cannot import {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    paths = tuple(Path(path) for path in helper.E_PATHS)
    locks = {
        Path(path): (int(rows), str(file_hash))
        for path, (rows, file_hash) in helper.EXPECTED_LOCKS.items()
        if Path(path) in paths
    }
    i34_dev = ROOT / "assets/evaluation/holdout/data_i34_material_beam_dev_v1.jsonl"
    paths = paths + (i34_dev, I35_FORMAL)
    locks[i34_dev] = (
        256,
        "fec7f5cb5dd642e83addd4d23ec1f7f0c6d3e285960a417e0520d27b6938401c",
    )
    locks[I35_FORMAL] = (I35_FORMAL_ROWS, I35_FORMAL_SHA256)
    require(len(paths) == len(set(paths)), "duplicate prompt-exclusion path")
    require(set(paths) == set(locks), "prompt-exclusion lock set is incomplete")
    return paths, locks


def verify_jsonl_lock(path: Path, expected_rows: int, expected_hash: str) -> int:
    require(path.is_file(), f"registered input is missing: {path}")
    actual_hash = sha256(path)
    require(
        actual_hash == expected_hash,
        f"registered input hash drifted: {path} {actual_hash}/{expected_hash}",
    )
    rows = sum(1 for line in path.open("rb") if line.strip())
    require(
        rows == expected_rows,
        f"registered input row count drifted: {path} {rows}/{expected_rows}",
    )
    return rows


def load_user_inventory() -> tuple[
    set[str], dict[str, Counter[str]], Counter[str], Counter[str]
]:
    verify_jsonl_lock(I36, I36_ROWS, I36_SHA256)
    user_sids: set[str] = set()
    sid_tasks: dict[str, Counter[str]] = defaultdict(Counter)
    task_rows: Counter[str] = Counter()
    event_domains: Counter[str] = Counter()
    with I36.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("i36_route") != "user_ce":
                continue
            task = str(row.get("task"))
            require(task in {"action", "topic"}, f"unexpected I36 user task at {line_number}")
            task_rows[task] += 1
            seen_in_row: set[str] = set()
            for match in SID_FIND_RE.finditer(str(row.get("input") or "")):
                sid = match.group(0)
                user_sids.add(sid)
                event_domains[match.group("domain")] += 1
                seen_in_row.add(sid)
            for sid in seen_in_row:
                sid_tasks[sid][task] += 1
    require(task_rows == Counter({"action": 4_000, "topic": 1_500}), "I36 user mix drifted")
    require(len(user_sids) == 422_317, f"I36 unique history SID count drifted: {len(user_sids)}")
    return user_sids, sid_tasks, task_rows, event_domains


def load_o1_forward_sids() -> tuple[set[str], Counter[str]]:
    sids: set[str] = set()
    domains: Counter[str] = Counter()
    total_rows = 0
    for path, (expected_rows, expected_hash) in zip(O1_FORWARD, O1_FORWARD_LOCKS):
        verify_jsonl_lock(path, expected_rows, expected_hash)
        for row in load_jsonl(path):
            value = normalized_row(row)
            matches = list(SID_FIND_RE.finditer(value["output"]))
            require(len(matches) == 1, f"O1 forward material row lacks one SID: {path}")
            match = matches[0]
            sids.add(match.group(0))
            domains[match.group("domain")] += 1
            total_rows += 1
    require(total_rows == 5_597, f"O1 forward row count drifted: {total_rows}")
    require(len(sids) == 5_585, f"O1 forward unique SID count drifted: {len(sids)}")
    return sids, domains


def load_exclusions() -> tuple[set[str], set[str], set[str], list[dict[str, Any]]]:
    paths, locks = load_i34_e_manifest()
    exact: set[str] = set()
    mode: set[str] = set()
    descriptions: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for path in paths:
        expected_rows, expected_hash = locks[path]
        verify_jsonl_lock(path, expected_rows, expected_hash)
        rows = load_jsonl(path)
        for row in rows:
            value = normalized_row(row)
            exact.add(prompt_digest(value))
            mode.add(prompt_digest(value, mode_normalized=True))
            description = description_text(value["input"])
            if description:
                descriptions.add(digest(description))
        manifest.append(
            {
                "path": str(path.relative_to(ROOT)),
                "rows": len(rows),
                "sha256": expected_hash,
            }
        )
    return exact, mode, descriptions, manifest


def valid_caption(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, "missing"
    caption = normalize_text(value)
    if len(caption) < 16:
        return None, "too_short"
    if len(caption) > 2_000:
        return None, "too_long"
    if "<|" in caption or "<s_" in caption:
        return None, "contains_semantic_token"
    if re.search(r"/(?:no_)?think", caption):
        return None, "contains_mode_control"
    return caption, None


def collect_o3_candidates(
    user_sids: set[str],
    sid_tasks: Mapping[str, Counter[str]],
    o1_sids: set[str],
    excluded_exact: set[str],
    excluded_mode: set[str],
    excluded_descriptions: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ContractError("pyarrow is required for O3 scanning") from error

    require(O3.is_file(), f"registered O3 is missing: {O3}")
    require(sha256(O3) == O3_SHA256, "registered O3 hash drifted")
    parquet = pq.ParquetFile(O3)
    require(parquet.metadata.num_rows == O3_ROWS, "registered O3 row count drifted")

    caption_owner: dict[str, str | None] = {}
    per_sid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    counters: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()

    for batch in parquet.iter_batches(
        batch_size=32,
        columns=["record_id", "sid_token_list", "caption_list", "tag_list"],
    ):
        for record in batch.to_pylist():
            record_id = int(record["record_id"])
            sids = record.get("sid_token_list") or []
            captions = record.get("caption_list") or []
            tags = record.get("tag_list") or []
            counters["o3_sid_entries"] += len(sids)
            counters["o3_caption_entries"] += len(captions)
            counters["o3_tag_entries"] += len(tags)
            for item_index, (sid_value, caption_value) in enumerate(zip(sids, captions)):
                counters["paired_entries"] += 1
                sid = str(sid_value).strip()
                match = SID_RE.fullmatch(sid)
                if match is None:
                    counters["invalid_sid"] += 1
                    continue
                domain_counts[match.group("domain")] += 1
                if match.group("domain") != "video":
                    counters["non_video"] += 1
                    continue
                if sid not in user_sids:
                    counters["not_exact_i36_history_sid"] += 1
                    continue
                counters["exact_i36_video_pairs"] += 1
                if sid in o1_sids:
                    counters["o1_forward_sid_excluded"] += 1
                    continue
                caption, rejection = valid_caption(caption_value)
                if rejection is not None:
                    counters[f"caption_{rejection}"] += 1
                    continue
                assert caption is not None
                description_hash = digest(caption)
                if description_hash in excluded_descriptions:
                    counters["registered_description_excluded"] += 1
                    continue
                row_core = {
                    "instruction": OFFICIAL_SYSTEM,
                    "input": OFFICIAL_USER_PREFIX + caption + "/no_think",
                    "output": f"<think>\n\n</think>\n{sid}",
                    "history": [],
                }
                exact_hash = prompt_digest(row_core)
                mode_hash = prompt_digest(row_core, mode_normalized=True)
                if exact_hash in excluded_exact or mode_hash in excluded_mode:
                    counters["registered_prompt_excluded"] += 1
                    continue

                owner = caption_owner.get(caption)
                if owner is None and caption not in caption_owner:
                    caption_owner[caption] = sid
                elif owner != sid:
                    caption_owner[caption] = None
                tag = tags[item_index] if item_index < len(tags) else None
                candidate = {
                    "record_id": record_id,
                    "item_index": item_index,
                    "sid": sid,
                    "caption": caption,
                    "tag": tag if isinstance(tag, str) else None,
                    "domain": "video",
                    "a": int(match.group("a")),
                    "b": int(match.group("b")),
                    "c": int(match.group("c")),
                    "description_sha256": description_hash,
                    "prompt_sha256": exact_hash,
                    "mode_prompt_sha256": mode_hash,
                    "row_sha256": digest(row_core),
                    "source_pair_sha256": digest(
                        [record_id, item_index, sid, caption, tag]
                    ),
                    "user_overlap_by_task": dict(sorted(sid_tasks[sid].items())),
                }
                current = per_sid[sid].get(caption)
                if current is None or stable_hash(
                    "source", candidate["source_pair_sha256"]
                ) < stable_hash("source", current["source_pair_sha256"]):
                    per_sid[sid][caption] = candidate

    selected_by_sid: dict[str, dict[str, Any]] = {}
    ambiguous_captions = sum(owner is None for owner in caption_owner.values())
    for sid, values in per_sid.items():
        eligible = [
            value
            for caption, value in values.items()
            if caption_owner.get(caption) == sid
        ]
        if not eligible:
            counters["sid_without_unique_caption"] += 1
            continue
        selected_by_sid[sid] = min(
            eligible,
            key=lambda value: stable_hash(
                "caption",
                value["sid"],
                value["caption"],
                value["source_pair_sha256"],
            ),
        )
    counters["ambiguous_caption_strings"] = ambiguous_captions
    counters["eligible_unique_sids"] = len(selected_by_sid)
    return selected_by_sid, {
        "counts": dict(sorted(counters.items())),
        "paired_domains": dict(sorted(domain_counts.items())),
        "unique_candidate_caption_strings": len(caption_owner),
    }


def make_pool_row(candidate: Mapping[str, Any], view_role: str) -> dict[str, Any]:
    sid = str(candidate["sid"])
    ab = f"video:{candidate['a']}:{candidate['b']}"
    row = {
        "instruction": OFFICIAL_SYSTEM,
        "input": OFFICIAL_USER_PREFIX + str(candidate["caption"]) + "/no_think",
        "output": f"<think>\n\n</think>\n{sid}",
        "history": [],
        "route": "beam_train_pool",
        "task": "material_desc2sid",
        "source_asset_id": "O3.baseline_caption_tag_lists",
        "source_record_id": int(candidate["record_id"]),
        "source_item_index": int(candidate["item_index"]),
        "source_pair_sha256": candidate["source_pair_sha256"],
        "source_tag": candidate["tag"],
        "i36_overlap_asset_id": "data_i36_i35_user_expand_retkl_v1.user_ce_history",
        "i36_user_overlap_by_task": candidate["user_overlap_by_task"],
        "view_role": view_role,
        "gold_sid": sid,
        "gold_domain": "video",
        "gold_s_a": int(candidate["a"]),
        "gold_s_b": int(candidate["b"]),
        "gold_s_c": int(candidate["c"]),
        "prefix_group": ab,
        "description_sha256": candidate["description_sha256"],
        "prompt_sha256": candidate["prompt_sha256"],
        "mode_prompt_sha256": candidate["mode_prompt_sha256"],
        "row_sha256": candidate["row_sha256"],
    }
    return row


def select_pool(selected_by_sid: Mapping[str, Mapping[str, Any]]) -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in selected_by_sid.values():
        key = f"video:{candidate['a']}:{candidate['b']}"
        grouped[key].append(candidate)
    for key in grouped:
        grouped[key].sort(
            key=lambda value: stable_hash(
                "within_ab",
                key,
                value["c"],
                value["sid"],
                value["description_sha256"],
            )
        )

    ordered_groups = sorted(grouped, key=lambda key: stable_hash("primary_ab", key))
    require(
        len(ordered_groups) >= PRIMARY_AB_ROWS,
        f"only {len(ordered_groups)} eligible AB groups; need {PRIMARY_AB_ROWS}",
    )
    primary_groups = ordered_groups[:PRIMARY_AB_ROWS]
    paired_groups = sorted(
        (key for key in primary_groups if len(grouped[key]) >= 2),
        key=lambda key: stable_hash("paired_ab", key),
    )
    require(
        len(paired_groups) >= PAIRED_AB_ROWS,
        f"only {len(paired_groups)} primary AB groups have a second C; need {PAIRED_AB_ROWS}",
    )
    paired_groups = paired_groups[:PAIRED_AB_ROWS]

    rows = [make_pool_row(grouped[key][0], "ab_primary") for key in primary_groups]
    rows.extend(make_pool_row(grouped[key][1], "same_ab_second_c") for key in paired_groups)
    rows.sort(
        key=lambda row: stable_hash(
            "row_order",
            row["prefix_group"],
            row["gold_s_c"],
            row["view_role"],
            row["description_sha256"],
        )
    )
    require(len(rows) == TOTAL_ROWS, "I39 selected pool row count drifted")
    require(
        len({row["prompt_sha256"] for row in rows}) == TOTAL_ROWS,
        "I39 selected prompts are not unique",
    )
    require(
        len({row["gold_sid"] for row in rows}) == TOTAL_ROWS,
        "I39 selected full SIDs are not unique",
    )
    require(
        len({row["prefix_group"] for row in rows}) == PRIMARY_AB_ROWS,
        "I39 selected AB coverage drifted",
    )
    paired_check: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        paired_check[row["prefix_group"]].add(int(row["gold_s_c"]))
    require(
        sum(len(values) >= 2 for values in paired_check.values()) == PAIRED_AB_ROWS,
        "I39 paired AB/C coverage drifted",
    )
    return rows, {
        "eligible_unique_sids": len(selected_by_sid),
        "eligible_unique_ab": len(grouped),
        "eligible_ab_with_multiple_c": sum(len(values) >= 2 for values in grouped.values()),
        "selected_rows": len(rows),
        "selected_unique_full_sids": len({row["gold_sid"] for row in rows}),
        "selected_unique_ab": len({row["prefix_group"] for row in rows}),
        "selected_ab_with_two_c_views": PAIRED_AB_ROWS,
        "view_roles": dict(Counter(row["view_role"] for row in rows)),
    }


def build() -> dict[str, Any]:
    for path in (OUTPUT, DEV_OUTPUT, AUDIT):
        require(not path.exists(), f"refusing to overwrite frozen output: {path}")

    user_sids, sid_tasks, user_task_rows, user_event_domains = load_user_inventory()
    o1_sids, o1_domains = load_o1_forward_sids()
    excluded_exact, excluded_mode, excluded_descriptions, exclusion_manifest = (
        load_exclusions()
    )
    selected_by_sid, o3_audit = collect_o3_candidates(
        user_sids,
        sid_tasks,
        o1_sids,
        excluded_exact,
        excluded_mode,
        excluded_descriptions,
    )
    rows, selection_audit = select_pool(selected_by_sid)

    dev_index = min(
        range(len(rows)),
        key=lambda index: stable_hash("runner_dev_split", rows[index]["row_sha256"]),
    )
    dev = [dict(rows[dev_index], route="beam_gate_pool")]
    train = rows[:dev_index] + rows[dev_index + 1 :]
    atomic_jsonl(OUTPUT, train)
    atomic_jsonl(DEV_OUTPUT, dev)

    audit: dict[str, Any] = {
        "schema_version": "i39-userab-video-beam64-pool-v1",
        "asset_class": "D-construction(O3; D-I36 history membership; O1/E exclusions)",
        "seed": SEED,
        "formal_training_generated": False,
        "selection_definition": {
            "domain": "video",
            "membership": "full SID appears in an I36 user_ce input history",
            "caption": "O3 aligned caption; unique normalized caption owner",
            "exclusions": [
                "all O1 forward material full SIDs",
                "registered E exact/mode prompts",
                "registered E normalized material descriptions",
                "I35 formal exact/mode prompts and descriptions",
            ],
            "primary_ab_rows": PRIMARY_AB_ROWS,
            "same_ab_distinct_c_second_views": PAIRED_AB_ROWS,
        },
        "upstream": {
            "O3": {
                "asset_id": "O3",
                "path": str(O3.relative_to(ROOT)),
                "rows": O3_ROWS,
                "sha256": O3_SHA256,
            },
            "I36_user_history_filter": {
                "asset_id": "data_i36_i35_user_expand_retkl_v1",
                "path": str(I36.relative_to(ROOT)),
                "rows": I36_ROWS,
                "sha256": I36_SHA256,
                "user_ce_rows": dict(sorted(user_task_rows.items())),
                "unique_history_sids": len(user_sids),
                "history_events_by_domain": dict(sorted(user_event_domains.items())),
            },
            "O1_forward_material_exclusion": [
                {
                    "asset_id": f"O1.懂物料part{index}",
                    "path": str(path.relative_to(ROOT)),
                    "rows": lock[0],
                    "sha256": lock[1],
                }
                for index, (path, lock) in enumerate(
                    zip(O1_FORWARD, O1_FORWARD_LOCKS), 1
                )
            ],
            "O1_forward_unique_sids": len(o1_sids),
            "O1_forward_rows_by_domain": dict(sorted(o1_domains.items())),
            "prompt_exclusion_manifest": exclusion_manifest,
        },
        "o3_scan": o3_audit,
        "selection": selection_audit,
        "renderer": {
            "system": OFFICIAL_SYSTEM,
            "user_prefix": OFFICIAL_USER_PREFIX,
            "mode": "/no_think",
            "assistant": "empty think plus one video SID",
        },
        "outputs": {
            "train_pool": {
                "path": str(OUTPUT.relative_to(ROOT)),
                "rows": len(train),
                "sha256": sha256(OUTPUT),
            },
            "dev_pool": {
                "path": str(DEV_OUTPUT.relative_to(ROOT)),
                "rows": len(dev),
                "sha256": sha256(DEV_OUTPUT),
            },
            "combined_rows": len(rows),
            "combined_mix": {
                "ab_primary": PRIMARY_AB_ROWS,
                "same_ab_second_c": PAIRED_AB_ROWS,
            },
        },
    }
    atomic_json(AUDIT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return audit


def self_test() -> None:
    row = {
        "instruction": OFFICIAL_SYSTEM,
        "input": OFFICIAL_USER_PREFIX + "  示例  描述  " + "/no_think",
        "output": "<think>\n\n</think>\n<|video_begin|><s_a_1><s_b_2><s_c_3>",
        "history": [],
    }
    require(description_text(row["input"]) == "示例 描述", "description normalization failed")
    require(
        prompt_digest(row) != prompt_digest(row, mode_normalized=True),
        "mode-normalized prompt hash failed",
    )
    caption, rejection = valid_caption("这是一段足够长并且不包含任何语义标识符的测试视频描述。")
    require(caption is not None and rejection is None, "valid caption rejected")
    require(SID_RE.fullmatch("<|video_begin|><s_a_1><s_b_2><s_c_3>") is not None, "SID parser failed")
    print("i39 AB/first-divergence pool self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        build()


if __name__ == "__main__":
    main()
