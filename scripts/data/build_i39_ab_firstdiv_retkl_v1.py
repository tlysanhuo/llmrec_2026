#!/usr/bin/env python3
"""Freeze the I-39 AB/first-divergence, user-microdose, and retention mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SEED = 19260839

POOL_TRAIN = ROOT / "logs/data/i39_userab_video_beam64_pool_v1.jsonl"
POOL_DEV = ROOT / "logs/data/i39_userab_video_beam64_pool_v1_dev.jsonl"
POOL_AUDIT = ROOT / "logs/data/i39_userab_video_beam64_pool_v1_audit.json"
BEAM_TRAIN = ROOT / "logs/data/i39_userab_video_beam64_beam_ledger_v1.jsonl"
BEAM_DEV = ROOT / "logs/probe/i39_userab_video_beam64_dev_ledger_v1.jsonl"
BEAM_AUDIT = ROOT / "logs/probe/i39_userab_video_beam64_audit_v1.json"
I36 = ROOT / "assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl"
RETENTION = (
    ROOT
    / "assets/derived/releases/e3_userres_r80_retkl_v3_s875"
    / "data_user_residual_retention_v1.jsonl"
)
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"

OUTPUT = ROOT / "assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1.jsonl"
SIDECAR = ROOT / "assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1_sidecar.jsonl"
GATE = ROOT / "assets/evaluation/holdout/data_i39_userab_firstdiv_gate_v1.jsonl"
AUDIT = ROOT / "logs/data/i39_i35_userab_firstdiv_retkl_v1_audit.json"

POOL_TRAIN_SHA256 = "daffaf734a62656c3de08bc92adfb4305c0c04a506068f2c2a40ddcaecc7f0a4"
POOL_DEV_SHA256 = "56a32e1a5d8271130ab2892631c1957907a8dc4dbd1a0a91239cd8092d9efbde"
POOL_AUDIT_SHA256 = "879db9723f0e51b18fefb5bf047f6cf5df3bdec16fe98872f82c80e5519fcc8d"
I36_SHA256 = "2720746a2e8aa7804d519698ce9f2b127e9be2db1d4488e642e800a5337b692d"
RETENTION_SHA256 = "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0"
PARENT_MODEL_SHA256 = "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
PARENT_CONFIG_SHA256 = "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"

MATERIAL_QUOTAS = {
    "a_firstdiv": 128,
    "b_firstdiv": 128,
    "c_firstdiv": 192,
    "full_anchor": 64,
}
USER_QUOTAS = {"action": 96, "topic": 32}
RETENTION_QUOTAS = {
    "material_desc2sid": 128,
    "material_sid2desc": 128,
    "action": 256,
    "topic": 256,
    "rec_video": 240,
    "rec_prod": 240,
    "rec_ad": 240,
    "rec_living": 240,
    "world": 192,
}
GATE_AB_GROUPS = 256
GATE_OBJECTIVE_MINIMUMS = {
    "a_firstdiv": 16,
    "b_firstdiv": 16,
    "c_firstdiv": 16,
    "full_anchor": 4,
}
GATE_DUAL_C_GROUPS_MINIMUM = 32
GATE_DUAL_C_FIRSTDIV_GROUPS_MINIMUM = 4
C_PAIR_GROUPS = 32
C_SINGLE_GROUPS = 128
TOTAL_ROWS = sum(MATERIAL_QUOTAS.values()) + sum(USER_QUOTAS.values()) + sum(
    RETENTION_QUOTAS.values()
)

MATERIAL_ROUTE = "material_firstdiv"
USER_ROUTE = "user_micro_ce"
RETENTION_ROUTE = "retention_kl"
OFFICIAL_SYSTEM = "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。"
OFFICIAL_USER_PREFIX = "请解析以下视频内容并输出对应的视频token：\n\n"
SID_RE = re.compile(
    r"^<\|video_begin\|><s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>$"
)
SID_FIND_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)


class ContractError(RuntimeError):
    """Raised when a frozen input or selection invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


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


def stable_hash(*parts: Any) -> str:
    return digest([SEED, *parts])


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def token_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", row.get("user", ""))) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def core_hash(row: Mapping[str, Any]) -> str:
    value = normalized(row)
    return digest(
        {
            "instruction": value["instruction"],
            "input": value["input"],
            "output": value["output"],
            "history": value["history"],
        }
    )


def prompt_hash(row: Mapping[str, Any], *, mode_normalized: bool = False) -> str:
    value = normalized(row)
    user_input = value["input"]
    if mode_normalized:
        user_input = re.sub(r"/(?:no_)?think\s*$", "", user_input.rstrip()).rstrip()
    return digest([value["instruction"], user_input, value["history"]])


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank row at {path}:{line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def encoded_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical(row) + "\n" for row in rows).encode("utf-8")


def encoded_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_new(payloads: Sequence[tuple[Path, bytes]]) -> None:
    paths = [path for path, _payload in payloads]
    require(len(paths) == len(set(paths)), "I39 output paths are not unique")
    existing = [str(path) for path in paths if path.exists()]
    require(not existing, "refusing to overwrite I39 outputs: " + ", ".join(existing))
    temporary: list[Path] = []
    try:
        for path, payload in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(payload)
            temporary.append(temp)
        for (path, _payload), temp in zip(payloads, temporary):
            temp.replace(path)
    finally:
        for temp in temporary:
            if temp.exists():
                temp.unlink()


def verify_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in (
        (POOL_TRAIN, POOL_TRAIN_SHA256),
        (POOL_DEV, POOL_DEV_SHA256),
        (POOL_AUDIT, POOL_AUDIT_SHA256),
        (I36, I36_SHA256),
        (RETENTION, RETENTION_SHA256),
    ):
        require(path.is_file(), f"missing registered I39 source: {path}")
        require(sha256(path) == expected, f"registered I39 source hash drifted: {path}")
    pool_audit = load_json(POOL_AUDIT)
    require(pool_audit.get("formal_training_generated") is False, "pool audit already claims formal training")
    require(pool_audit.get("outputs", {}).get("combined_rows") == 3_072, "pool count drifted")

    beam_audit = load_json(BEAM_AUDIT)
    require(beam_audit.get("status") == "complete", "I39 Beam audit is incomplete")
    require(beam_audit.get("formal_training_generated") is False, "Beam runner wrote formal data")
    require(beam_audit.get("runtime", {}).get("single_parent_request") is True, "I39 Beam used more than one parent request")
    require(beam_audit.get("runtime", {}).get("beam_width") == 64, "I39 Beam width drifted")
    parent = beam_audit.get("parent", {})
    require(parent.get("adapter_model_sha256") == PARENT_MODEL_SHA256, "I39 Beam parent drifted")
    require(parent.get("adapter_config_sha256") == PARENT_CONFIG_SHA256, "I39 Beam parent config drifted")
    outputs = beam_audit.get("outputs", {})
    for key, path in (("train_ledger", BEAM_TRAIN), ("dev_ledger", BEAM_DEV)):
        entry = outputs.get(key)
        require(isinstance(entry, dict), f"I39 Beam audit lacks {key}")
        require(Path(entry.get("path", "")).resolve() == path.resolve(), f"I39 Beam {key} path drifted")
        require(entry.get("sha256") == sha256(path), f"I39 Beam {key} hash drifted")
    return pool_audit, beam_audit


def load_pool_and_ledger() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = load_jsonl(POOL_TRAIN) + load_jsonl(POOL_DEV)
    ledger = load_jsonl(BEAM_TRAIN) + load_jsonl(BEAM_DEV)
    require(len(pool) == len(ledger) == 3_072, "I39 pool/ledger row count drifted")
    pool_by_hash = {str(row.get("row_sha256")): row for row in pool}
    ledger_by_hash = {str(row.get("row_sha256")): row for row in ledger}
    require(len(pool_by_hash) == len(ledger_by_hash) == 3_072, "I39 pool/ledger keys are not unique")
    require(set(pool_by_hash) == set(ledger_by_hash), "I39 pool and ledger key sets differ")
    for key, source in pool_by_hash.items():
        result = ledger_by_hash[key]
        require(
            result.get("source_prompt_sha256") == source.get("prompt_sha256"),
            "I39 ledger source prompt hash drifted",
        )
        require(
            result.get("source_mode_prompt_sha256") == source.get("mode_prompt_sha256"),
            "I39 ledger mode prompt hash drifted",
        )
        expected_abc = [
            str(source.get("gold_s_a")),
            str(source.get("gold_s_b")),
            str(source.get("gold_s_c")),
        ]
        require(result.get("gold_abc") == expected_abc, "I39 ledger gold ABC drifted")
        require(result.get("domain") == "video", "I39 ledger domain drifted")
        require(result.get("task") == source.get("task"), "I39 ledger task drifted")
        require(result.get("route") == source.get("route"), "I39 ledger route drifted")
    ordered_pool = sorted(pool, key=lambda row: stable_hash("pool_order", row["row_sha256"]))
    return ordered_pool, [ledger_by_hash[str(row["row_sha256"])] for row in ordered_pool]


def gate_identifiers(
    gate_rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    gate_sids = {str(row["i39_gold_sid"]) for row in gate_rows}
    gate_ab = {str(row["i39_prefix_group"]) for row in gate_rows}
    require(len(gate_ab) == GATE_AB_GROUPS, "I39 gate AB identifier count drifted")
    require(len(gate_sids) == len(gate_rows), "I39 gate SID identifiers are not unique")
    return gate_sids, gate_ab


def gate_hits(
    text: str, gate_sids: set[str], gate_ab: set[str]
) -> tuple[set[str], set[str]]:
    sid_hits: set[str] = set()
    ab_hits: set[str] = set()
    for match in SID_FIND_RE.finditer(text):
        sid = match.group(0)
        if sid in gate_sids:
            sid_hits.add(sid)
        if match.group("domain") == "video":
            ab = f"video:{match.group('a')}:{match.group('b')}"
            if ab in gate_ab:
                ab_hits.add(ab)
    return sid_hits, ab_hits


def is_gate_clean(
    row: Mapping[str, Any], gate_sids: set[str], gate_ab: set[str]
) -> bool:
    value = normalized(row)
    input_sid, input_ab = gate_hits(value["input"], gate_sids, gate_ab)
    output_sid, output_ab = gate_hits(value["output"], gate_sids, gate_ab)
    return not (input_sid or input_ab or output_sid or output_ab)


def classify(ledger: Mapping[str, Any]) -> str:
    parent = ledger.get("parent")
    require(isinstance(parent, dict), "I39 ledger parent block is missing")
    a_hit = parent.get("a_hit")
    ab_hit = parent.get("ab_hit")
    full_hit = parent.get("full_gold_hit")
    require(all(isinstance(value, bool) for value in (a_hit, ab_hit, full_hit)), "I39 ledger hit flags are invalid")
    require(not ab_hit or a_hit, "I39 ledger AB hit without A hit")
    require(not full_hit or ab_hit, "I39 ledger full hit without AB hit")
    if not a_hit:
        return "a_firstdiv"
    if not ab_hit:
        return "b_firstdiv"
    if not full_hit:
        return "c_firstdiv"
    return "full_anchor"


def focus_index(objective: str) -> int | None:
    return {"a_firstdiv": 0, "b_firstdiv": 1, "c_firstdiv": 2}.get(objective)


def focus_negatives(ledger: Mapping[str, Any], objective: str) -> list[dict[str, Any]]:
    focus = focus_index(objective)
    if focus is None:
        return []
    values = ledger.get("hard_negatives")
    require(isinstance(values, list), "I39 ledger hard_negatives is invalid")
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or value.get("first_divergence") != focus:
            continue
        sanitized = dict(value)
        compatibility_score = sanitized.pop("teacher_score", None)
        if compatibility_score is not None:
            require(
                compatibility_score == sanitized.get("parent_score"),
                "I39 single-parent ledger has a non-parent teacher_score",
            )
        result.append(sanitized)
    require(result, f"I39 {objective} row lacks a focus-matched hard negative")
    require(len(result) <= 4, f"I39 {objective} row has too many focus negatives")
    return result


def select_material(
    pool: Sequence[dict[str, Any]], ledger: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source, result in zip(pool, ledger):
        require(source.get("prefix_group") is not None, "I39 pool row lacks AB group")
        require(result.get("schema_version") == "i39-userab-video-beam64-ledger-v1", "I39 ledger schema drifted")
        require(result.get("parent_adapter_sha256") == PARENT_MODEL_SHA256, "I39 ledger parent hash drifted")
        objective = classify(result)
        negatives = focus_negatives(result, objective)
        records.append(
            {
                "pool": source,
                "ledger": result,
                "objective": objective,
                "focus_negatives": negatives,
                "ab": str(source["prefix_group"]),
                "c": int(source["gold_s_c"]),
            }
        )

    all_ab = sorted({record["ab"] for record in records}, key=lambda value: stable_hash("gate_ab", value))
    require(len(all_ab) == 2_560, "I39 pool AB count drifted before gate split")
    gate_ab = set(all_ab[:GATE_AB_GROUPS])
    gate_records = [record for record in records if record["ab"] in gate_ab]
    gate_primary = [
        record for record in gate_records if record["pool"].get("view_role") == "ab_primary"
    ]
    require(len(gate_primary) == GATE_AB_GROUPS, "I39 gate primary AB selection drifted")
    gate_objectives = Counter(record["objective"] for record in gate_records)
    for objective, minimum in GATE_OBJECTIVE_MINIMUMS.items():
        require(
            gate_objectives[objective] >= minimum,
            f"I39 frozen gate has only {gate_objectives[objective]} {objective} rows; need {minimum}",
        )
    gate_by_ab: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gate_records:
        gate_by_ab[record["ab"]].append(record)
    gate_dual_c_groups = sum(
        len({record["c"] for record in values}) >= 2 for values in gate_by_ab.values()
    )
    gate_dual_c_firstdiv_groups = sum(
        len({record["c"] for record in values if record["objective"] == "c_firstdiv"})
        >= 2
        for values in gate_by_ab.values()
    )
    require(
        gate_dual_c_groups >= GATE_DUAL_C_GROUPS_MINIMUM,
        f"I39 frozen gate has only {gate_dual_c_groups} dual-C AB groups",
    )
    require(
        gate_dual_c_firstdiv_groups >= GATE_DUAL_C_FIRSTDIV_GROUPS_MINIMUM,
        "I39 frozen gate lacks paired C-first-divergence coverage",
    )
    train_candidates = [record for record in records if record["ab"] not in gate_ab]

    by_objective_ab: dict[str, dict[str, list[dict[str, Any]]]] = {
        objective: defaultdict(list) for objective in MATERIAL_QUOTAS
    }
    for record in train_candidates:
        by_objective_ab[record["objective"]][record["ab"]].append(record)
    for objective in by_objective_ab:
        for ab in by_objective_ab[objective]:
            by_objective_ab[objective][ab].sort(
                key=lambda record: stable_hash(
                    "within_stratum",
                    objective,
                    ab,
                    record["c"],
                    record["pool"]["row_sha256"],
                )
            )

    chosen: list[dict[str, Any]] = []
    used_ab: set[str] = set()

    pair_groups = [
        ab
        for ab, values in by_objective_ab["c_firstdiv"].items()
        if len({record["c"] for record in values}) >= 2
    ]
    pair_groups.sort(key=lambda ab: stable_hash("c_pair_group", ab))
    require(len(pair_groups) >= C_PAIR_GROUPS, f"I39 has only {len(pair_groups)} C-miss pair groups")
    for ab in pair_groups[:C_PAIR_GROUPS]:
        values = by_objective_ab["c_firstdiv"][ab]
        first = values[0]
        second = next(record for record in values[1:] if record["c"] != first["c"])
        chosen.extend([first, second])
        used_ab.add(ab)

    c_single_groups = sorted(
        (
            ab
            for ab in by_objective_ab["c_firstdiv"]
            if ab not in used_ab
        ),
        key=lambda ab: stable_hash("c_single_group", ab),
    )
    require(len(c_single_groups) >= C_SINGLE_GROUPS, "I39 lacks C-miss single groups")
    for ab in c_single_groups[:C_SINGLE_GROUPS]:
        chosen.append(by_objective_ab["c_firstdiv"][ab][0])
        used_ab.add(ab)

    def choose_unique_ab(objective: str, quota: int) -> None:
        groups = sorted(
            (ab for ab in by_objective_ab[objective] if ab not in used_ab),
            key=lambda ab: stable_hash("objective_group", objective, ab),
        )
        require(len(groups) >= quota, f"I39 {objective} has {len(groups)} disjoint AB groups; need {quota}")
        for ab in groups[:quota]:
            chosen.append(by_objective_ab[objective][ab][0])
            used_ab.add(ab)

    choose_unique_ab("full_anchor", MATERIAL_QUOTAS["full_anchor"])
    choose_unique_ab("a_firstdiv", MATERIAL_QUOTAS["a_firstdiv"])
    choose_unique_ab("b_firstdiv", MATERIAL_QUOTAS["b_firstdiv"])

    counts = Counter(record["objective"] for record in chosen)
    require(counts == Counter(MATERIAL_QUOTAS), f"I39 material quota drifted: {dict(counts)}")
    require(len(chosen) == 512 and len(used_ab) == 480, "I39 material row/AB count drifted")
    c_groups: dict[str, set[int]] = defaultdict(set)
    for record in chosen:
        if record["objective"] == "c_firstdiv":
            c_groups[record["ab"]].add(record["c"])
    require(sum(len(values) >= 2 for values in c_groups.values()) == C_PAIR_GROUPS, "I39 C-pair selection drifted")

    material_rows: list[dict[str, Any]] = []
    for record in chosen:
        source = normalized(record["pool"])
        material_rows.append(
            {
                "instruction": source["instruction"],
                "input": source["input"],
                "output": source["output"],
                "history": source["history"],
                "route": MATERIAL_ROUTE,
                "task": "material_desc2sid",
                "i39_objective": record["objective"],
                "i39_source_row_sha256": record["pool"]["row_sha256"],
                "i39_prefix_group": record["ab"],
                "i39_gold_sid": record["pool"]["gold_sid"],
            }
        )

    gate_rows: list[dict[str, Any]] = []
    for record in sorted(gate_records, key=lambda value: stable_hash("gate_row", value["pool"]["row_sha256"])):
        source = normalized(record["pool"])
        gate_rows.append(
            {
                "instruction": source["instruction"],
                "input": source["input"],
                "output": source["output"],
                "history": source["history"],
                "route": "i39_material_gate",
                "task": "material_desc2sid",
                "i39_objective": record["objective"],
                "i39_source_row_sha256": record["pool"]["row_sha256"],
                "i39_prefix_group": record["ab"],
                "i39_gold_sid": record["pool"]["gold_sid"],
            }
        )

    selected_records = sorted(
        chosen,
        key=lambda record: stable_hash(
            "selected_material",
            record["objective"],
            record["ab"],
            record["pool"]["row_sha256"],
        ),
    )
    selection_audit = {
        "pre_gate_strata": dict(sorted(Counter(record["objective"] for record in records).items())),
        "gate_ab_groups": len(gate_ab),
        "gate_rows": len(gate_rows),
        "gate_by_objective": dict(sorted(gate_objectives.items())),
        "gate_dual_c_groups": gate_dual_c_groups,
        "gate_dual_c_firstdiv_groups": gate_dual_c_firstdiv_groups,
        "gate_train_ab_overlap": len(gate_ab & used_ab),
        "formal_rows": len(material_rows),
        "formal_by_objective": dict(sorted(counts.items())),
        "formal_unique_ab": len(used_ab),
        "formal_c_pair_groups": C_PAIR_GROUPS,
        "formal_c_single_groups": C_SINGLE_GROUPS,
    }
    return material_rows, gate_rows, selected_records, selection_audit


def select_user_rows(
    selected_material: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_sids = {str(record["pool"]["gold_sid"]) for record in selected_material}
    gate_sids, gate_ab = gate_identifiers(gate_rows)
    candidates: dict[str, list[tuple[int, str, dict[str, Any], set[str]]]] = defaultdict(list)
    rejected_gate = Counter()
    label_linked_candidates = Counter()
    for line_number, row in enumerate(load_jsonl(I36), 1):
        if row.get("i36_route") != "user_ce":
            continue
        task = str(row.get("task"))
        require(task in USER_QUOTAS, f"unexpected I36 user task at line {line_number}")
        value = normalized(row)
        hits = {
            match.group(0)
            for match in SID_FIND_RE.finditer(value["input"])
            if match.group(0) in selected_sids
        }
        if not hits:
            continue
        if not is_gate_clean(row, gate_sids, gate_ab):
            rejected_gate[task] += 1
            continue
        if any(sid in value["output"] for sid in hits):
            label_linked_candidates[task] += 1
        transformed = {
            "instruction": value["instruction"],
            "input": value["input"],
            "output": value["output"],
            "history": value["history"],
            "route": USER_ROUTE,
            "task": task,
            "i39_overlap_material_sids": sorted(hits),
            "i39_material_link_scope": "user_history_input",
            "i39_source_i36_line": line_number,
        }
        candidates[task].append((len(hits), core_hash(transformed), transformed, hits))

    selected: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "candidate_rows": {},
        "selected_rows": {},
        "selected_sid_links": {},
        "rejected_gate_rows": dict(rejected_gate),
        "label_linked_candidate_rows": dict(label_linked_candidates),
        "link_contract": "selected material full SID occurs in user history input; label overlap is audited but not required",
    }
    seen_prompt: set[str] = set()
    for task, quota in USER_QUOTAS.items():
        ranked = sorted(
            candidates[task],
            key=lambda item: (-item[0], stable_hash("user_row", task, item[1])),
        )
        audit["candidate_rows"][task] = len(ranked)
        chosen: list[tuple[int, str, dict[str, Any], set[str]]] = []
        for item in ranked:
            key = prompt_hash(item[2])
            if key in seen_prompt:
                continue
            seen_prompt.add(key)
            chosen.append(item)
            if len(chosen) == quota:
                break
        require(len(chosen) == quota, f"I39 user {task} rows {len(chosen)}/{quota}")
        selected.extend(item[2] for item in chosen)
        audit["selected_rows"][task] = len(chosen)
        audit["selected_sid_links"][task] = sum(item[0] for item in chosen)
    require(len(selected) == sum(USER_QUOTAS.values()), "I39 user row count drifted")
    require(
        all(is_gate_clean(row, gate_sids, gate_ab) for row in selected),
        "I39 selected user rows leak frozen gate SID/AB",
    )
    return selected, audit


def classify_retention(row: Mapping[str, Any]) -> str:
    value = normalized(row)
    body = value["output"].split("</think>", 1)[-1].strip()
    if body.startswith("["):
        return "action"
    if body.startswith("{") and "logic_chain" in body:
        return "topic"
    if "该用户最近" in body:
        for domain in ("video", "prod", "ad", "living"):
            if f"<|{domain}_begin|>" in body:
                return f"rec_{domain}"
    input_has_sid = "<s_a_" in value["input"]
    output_has_sid = "<s_a_" in body
    if output_has_sid and not input_has_sid:
        return "material_desc2sid"
    if input_has_sid and not output_has_sid:
        return "material_sid2desc"
    return "world"


def select_retention_rows(
    gate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(RETENTION)
    require(len(rows) == 6_106, "I12 retention source row count drifted")
    gate_sids, gate_ab = gate_identifiers(gate_rows)
    by_task: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    rejected_gate = Counter()
    for line_number, row in enumerate(rows, 1):
        task = classify_retention(row)
        require(task in RETENTION_QUOTAS, f"unexpected I12 retention task at line {line_number}: {task}")
        if not is_gate_clean(row, gate_sids, gate_ab):
            rejected_gate[task] += 1
            continue
        by_task[task].append((line_number, row))
    source_counts = {task: len(by_task[task]) for task in RETENTION_QUOTAS}
    expected_clean_source = {
        "material_desc2sid": 281,
        "material_sid2desc": 280,
        "action": 1_217,
        "topic": 964,
        "rec_video": 406,
        "rec_prod": 506,
        "rec_ad": 467,
        "rec_living": 482,
        "world": 231,
    }
    require(
        source_counts == expected_clean_source,
        f"I12 gate-clean source task counts drifted: {source_counts}",
    )

    representatives_by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    source_unique_counts: dict[str, int] = {}
    for task, quota in RETENTION_QUOTAS.items():
        unique_by_prompt: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for line_number, row in by_task[task]:
            unique_by_prompt[prompt_hash(row)].append((line_number, row))
        representatives = [
            min(
                values,
                key=lambda item: stable_hash(
                    "retention_duplicate",
                    task,
                    item[0],
                    core_hash(item[1]),
                ),
            )
            for values in unique_by_prompt.values()
        ]
        source_unique_counts[task] = len(representatives)
        representatives_by_task[task] = sorted(
            representatives,
            key=lambda item: stable_hash("retention", task, item[0], core_hash(item[1])),
        )
        require(len(representatives_by_task[task]) >= quota, f"I39 retention {task} lacks rows")

    selection_order = sorted(
        RETENTION_QUOTAS,
        key=lambda task: (
            source_unique_counts[task] / RETENTION_QUOTAS[task],
            task,
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    cross_task_prompt_rejections: Counter[str] = Counter()
    seen_prompt: set[str] = set()
    for task in selection_order:
        quota = RETENTION_QUOTAS[task]
        for line_number, row in representatives_by_task[task]:
            key = prompt_hash(row)
            if key in seen_prompt:
                cross_task_prompt_rejections[task] += 1
                continue
            seen_prompt.add(key)
            value = normalized(row)
            selected.append(
                {
                    "instruction": value["instruction"],
                    "input": value["input"],
                    "output": value["output"],
                    "history": value["history"],
                    "route": RETENTION_ROUTE,
                    "task": task,
                    "i39_source_i12_line": line_number,
                }
            )
            selected_counts[task] += 1
            if selected_counts[task] == quota:
                break
        require(
            selected_counts[task] == quota,
            f"I39 retention {task} rows {selected_counts[task]}/{quota} after global prompt dedupe",
        )
    require(len(selected) == sum(RETENTION_QUOTAS.values()), "I39 retention count drifted")
    require(
        len(seen_prompt) == len(selected),
        "I39 retention global prompt uniqueness drifted",
    )
    require(
        all(is_gate_clean(row, gate_sids, gate_ab) for row in selected),
        "I39 selected retention rows leak frozen gate SID/AB",
    )
    return selected, {
        "source_rows": len(rows),
        "gate_clean_source_counts": source_counts,
        "rejected_gate_rows": dict(rejected_gate),
        "source_unique_prompt_counts": source_unique_counts,
        "selected_rows": len(selected),
        "selected_counts": dict(selected_counts),
        "selection_order_most_constrained_first": selection_order,
        "cross_task_prompt_rejections": dict(cross_task_prompt_rejections),
        "selection": (
            "gate-clean stable-hash subset with exact-prompt uniqueness across all "
            "classified I12 tasks; most constrained task selected first"
        ),
    }


def encode_contract(
    material_rows: Sequence[dict[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
    user_rows: Sequence[dict[str, Any]],
    retention_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
        from llamafactory.data.template import TEMPLATES
    except ImportError as error:
        raise ContractError("LLaMA-Factory environment is required for I39 tokenization") from error

    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    record_by_source = {
        str(record["pool"]["row_sha256"]): record for record in selected_records
    }
    sidecar: list[dict[str, Any]] = []
    max_tokens = 0
    route_token_max: Counter[str] = Counter()
    all_rows = list(material_rows) + list(user_rows) + list(retention_rows)
    for line_number, row in enumerate(all_rows, 1):
        prompt_ids, response_ids = template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": row["output"]},
            ],
            row["instruction"],
            None,
        )
        total = len(prompt_ids) + len(response_ids)
        require(total <= 16_384, f"I39 cutoff overflow at logical row {line_number}: {total}")
        max_tokens = max(max_tokens, total)
        route_token_max[row["route"]] = max(route_token_max[row["route"]], total)
        entry: dict[str, Any] = {
            "schema_version": "i39-i35-userab-firstdiv-retkl-v1",
            "route": row["route"],
            "task": row["task"],
            "row_sha256": core_hash(row),
            "prompt_sha256": prompt_hash(row),
            "prompt_token_sha256": token_hash(prompt_ids),
            "prompt_token_count": len(prompt_ids),
            "response_token_sha256": token_hash(response_ids),
            "response_token_count": len(response_ids),
            "parent_adapter_sha256": PARENT_MODEL_SHA256,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
        }
        if row["route"] == MATERIAL_ROUTE:
            source_key = str(row["i39_source_row_sha256"])
            record = record_by_source.get(source_key)
            require(record is not None, "I39 material row lacks selected record")
            result = record["ledger"]
            gold_tokens = result.get("gold_tokens")
            require(
                isinstance(gold_tokens, list)
                and len(gold_tokens) == 5
                and all(isinstance(value, int) and not isinstance(value, bool) for value in gold_tokens),
                "I39 ledger gold tokens are invalid",
            )
            objective = str(record["objective"])
            focus = focus_index(objective)
            negatives = record["focus_negatives"]
            if focus is None:
                require(not negatives, "I39 anchor unexpectedly has negatives")
            else:
                require(negatives, "I39 first-divergence row lacks negatives")
                require(
                    all(value.get("first_divergence") == focus for value in negatives),
                    "I39 sidecar focus mismatch",
                )
            require(
                all("teacher_score" not in value for value in negatives),
                "I39 formal sidecar retained a compatibility teacher score",
            )
            entry.update(
                {
                    "objective": objective,
                    "focus_index": focus,
                    "source_row_sha256": source_key,
                    "source_prompt_sha256": record["pool"].get("prompt_sha256"),
                    "source_mode_prompt_sha256": record["pool"].get("mode_prompt_sha256"),
                    "domain": "video",
                    "gold_sid": record["pool"]["gold_sid"],
                    "gold_tokens": gold_tokens,
                    "gold_abc": gold_tokens[1:4],
                    "positive_tokens": [gold_tokens[1:4]],
                    "hard_negatives": negatives,
                }
            )
        sidecar.append(entry)
    require(len(sidecar) == len(all_rows) == TOTAL_ROWS, "I39 sidecar count drifted")
    require(
        len({row["prompt_token_sha256"] for row in sidecar}) == TOTAL_ROWS,
        "I39 sidecar prompt tokens are not unique",
    )
    sidecar_routes = Counter(row["route"] for row in sidecar)
    require(
        sidecar_routes
        == Counter(
            {
                MATERIAL_ROUTE: sum(MATERIAL_QUOTAS.values()),
                USER_ROUTE: sum(USER_QUOTAS.values()),
                RETENTION_ROUTE: sum(RETENTION_QUOTAS.values()),
            }
        ),
        f"I39 sidecar route count drifted: {dict(sidecar_routes)}",
    )
    return sidecar, {
        "max_total_tokens": max_tokens,
        "max_total_tokens_by_route": dict(route_token_max),
        "template": "qwen3_nothink",
        "cutoff": 16_384,
        "sidecar_rows": len(sidecar),
        "sidecar_routes": dict(sidecar_routes),
    }


def build() -> dict[str, Any]:
    for path in (OUTPUT, SIDECAR, GATE, AUDIT):
        require(not path.exists(), f"refusing to overwrite frozen I39 output: {path}")
    pool_audit, beam_audit = verify_sources()
    pool, ledger = load_pool_and_ledger()
    material_rows, gate_rows, selected_records, material_audit = select_material(pool, ledger)
    user_rows, user_audit = select_user_rows(selected_records, gate_rows)
    retention_rows, retention_audit = select_retention_rows(gate_rows)
    sidecar_rows, token_audit = encode_contract(
        material_rows, selected_records, user_rows, retention_rows
    )

    training_rows = material_rows + user_rows + retention_rows
    require(len(training_rows) == TOTAL_ROWS == 2_560, "I39 formal row count drifted")
    training_rows.sort(
        key=lambda row: stable_hash(
            "formal_order", row["route"], row["task"], core_hash(row)
        )
    )
    sidecar_rows.sort(key=lambda row: row["prompt_token_sha256"])

    route_counts = Counter(row["route"] for row in training_rows)
    expected_routes = Counter(
        {
            MATERIAL_ROUTE: sum(MATERIAL_QUOTAS.values()),
            USER_ROUTE: sum(USER_QUOTAS.values()),
            RETENTION_ROUTE: sum(RETENTION_QUOTAS.values()),
        }
    )
    require(route_counts == expected_routes, f"I39 route counts drifted: {dict(route_counts)}")
    exact_sets = {
        route: {prompt_hash(row) for row in training_rows if row["route"] == route}
        for route in expected_routes
    }
    mode_sets = {
        route: {prompt_hash(row, mode_normalized=True) for row in training_rows if row["route"] == route}
        for route in expected_routes
    }
    require(
        all(len(exact_sets[route]) == expected_routes[route] for route in expected_routes),
        "I39 formal exact prompts are not unique within route",
    )
    require(
        not exact_sets[MATERIAL_ROUTE] & exact_sets[USER_ROUTE]
        and not exact_sets[MATERIAL_ROUTE] & exact_sets[RETENTION_ROUTE]
        and not exact_sets[USER_ROUTE] & exact_sets[RETENTION_ROUTE],
        "I39 formal routes have exact prompt overlap",
    )
    require(
        not mode_sets[MATERIAL_ROUTE] & mode_sets[USER_ROUTE]
        and not mode_sets[MATERIAL_ROUTE] & mode_sets[RETENTION_ROUTE]
        and not mode_sets[USER_ROUTE] & mode_sets[RETENTION_ROUTE],
        "I39 formal routes have mode prompt overlap",
    )
    gate_exact = {prompt_hash(row) for row in gate_rows}
    gate_mode = {prompt_hash(row, mode_normalized=True) for row in gate_rows}
    require(
        not gate_exact & set().union(*exact_sets.values())
        and not gate_mode & set().union(*mode_sets.values()),
        "I39 gate overlaps formal training prompts",
    )
    gate_sids, gate_ab = gate_identifiers(gate_rows)
    material_ab = {str(row["i39_prefix_group"]) for row in material_rows}
    require(not gate_ab & material_ab, "I39 gate overlaps material training AB groups")
    gate_route_hits: Counter[str] = Counter()
    for row in training_rows:
        value = normalized(row)
        input_sid, input_ab = gate_hits(value["input"], gate_sids, gate_ab)
        output_sid, output_ab = gate_hits(value["output"], gate_sids, gate_ab)
        if input_sid or input_ab or output_sid or output_ab:
            gate_route_hits[str(row["route"])] += 1
    require(
        not gate_route_hits,
        f"I39 frozen gate SID/AB leaks across formal routes: {dict(gate_route_hits)}",
    )

    data_payload = encoded_jsonl(training_rows)
    sidecar_payload = encoded_jsonl(sidecar_rows)
    gate_payload = encoded_jsonl(gate_rows)
    audit: dict[str, Any] = {
        "schema_version": "i39-i35-userab-firstdiv-retkl-v1",
        "asset_class": "D(O3; D-I36 user overlap; D-I12 retention; M-I35 Beam64 filter)",
        "seed": SEED,
        "formal_training_generated": True,
        "upstream": {
            "pool": {
                "paths": [str(POOL_TRAIN.relative_to(ROOT)), str(POOL_DEV.relative_to(ROOT))],
                "rows": 3_072,
                "sha256": [POOL_TRAIN_SHA256, POOL_DEV_SHA256],
                "audit_path": str(POOL_AUDIT.relative_to(ROOT)),
                "audit_sha256": POOL_AUDIT_SHA256,
                "asset_ids": ["O3", "data_i36_i35_user_expand_retkl_v1", "O1/E exclusions"],
            },
            "beam": {
                "paths": [str(BEAM_TRAIN.relative_to(ROOT)), str(BEAM_DEV.relative_to(ROOT))],
                "sha256": [sha256(BEAM_TRAIN), sha256(BEAM_DEV)],
                "audit_path": str(BEAM_AUDIT.relative_to(ROOT)),
                "audit_sha256": sha256(BEAM_AUDIT),
                "parent_adapter_sha256": PARENT_MODEL_SHA256,
                "beam_width": 64,
            },
            "user": {
                "path": str(I36.relative_to(ROOT)),
                "rows": 16_500,
                "sha256": I36_SHA256,
                "source_route": "user_ce",
            },
            "retention": {
                "path": str(RETENTION.relative_to(ROOT)),
                "rows": 6_106,
                "sha256": RETENTION_SHA256,
                "source_role": "frozen I35 parent KL prompts only",
            },
            "pool_audit_snapshot_sha256": digest(pool_audit),
            "beam_audit_snapshot_sha256": digest(beam_audit),
        },
        "selection": {
            "material": material_audit,
            "user": user_audit,
            "retention": retention_audit,
        },
        "mix": {
            "total_rows": len(training_rows),
            "optimizer_steps_batch1_acc4": math.ceil(len(training_rows) / 4),
            "routes": {
                MATERIAL_ROUTE: {
                    "rows": expected_routes[MATERIAL_ROUTE],
                    "ratio": expected_routes[MATERIAL_ROUTE] / len(training_rows),
                    "by_objective": dict(MATERIAL_QUOTAS),
                },
                USER_ROUTE: {
                    "rows": expected_routes[USER_ROUTE],
                    "ratio": expected_routes[USER_ROUTE] / len(training_rows),
                    "by_task": dict(USER_QUOTAS),
                },
                RETENTION_ROUTE: {
                    "rows": expected_routes[RETENTION_ROUTE],
                    "ratio": expected_routes[RETENTION_ROUTE] / len(training_rows),
                    "by_task": dict(RETENTION_QUOTAS),
                },
            },
        },
        "leakage": {
            "exact_cross_route_prompt_overlap": 0,
            "mode_cross_route_prompt_overlap": 0,
            "gate_formal_exact_prompt_overlap": 0,
            "gate_formal_mode_prompt_overlap": 0,
            "gate_material_ab_overlap": 0,
            "gate_all_route_full_sid_overlap": 0,
            "gate_all_route_ab_overlap": 0,
            "gate_unique_ab": len(gate_ab),
            "gate_unique_full_sid": len(gate_sids),
            "formal_material_unique_ab": len(material_ab),
        },
        "tokenizer": token_audit,
        "sidecar_contract": {
            "rows": len(sidecar_rows),
            "routes": {
                MATERIAL_ROUTE: sum(MATERIAL_QUOTAS.values()),
                USER_ROUTE: sum(USER_QUOTAS.values()),
                RETENTION_ROUTE: sum(RETENTION_QUOTAS.values()),
            },
            "objectives": dict(MATERIAL_QUOTAS),
            "focus_index": {"a_firstdiv": 0, "b_firstdiv": 1, "c_firstdiv": 2, "full_anchor": None},
            "parent_adapter_sha256": PARENT_MODEL_SHA256,
            "model_candidates_are_negatives_only": True,
            "single_parent_compatibility_teacher_score_removed": True,
            "gold_source": "O3 aligned SID-caption pair",
        },
        "outputs": {
            "training_data": {
                "path": str(OUTPUT.relative_to(ROOT)),
                "rows": len(training_rows),
                "bytes": len(data_payload),
                "sha256": hashlib.sha256(data_payload).hexdigest(),
            },
            "sidecar": {
                "path": str(SIDECAR.relative_to(ROOT)),
                "rows": len(sidecar_rows),
                "bytes": len(sidecar_payload),
                "sha256": hashlib.sha256(sidecar_payload).hexdigest(),
            },
            "gate": {
                "path": str(GATE.relative_to(ROOT)),
                "rows": len(gate_rows),
                "bytes": len(gate_payload),
                "sha256": hashlib.sha256(gate_payload).hexdigest(),
            },
        },
    }
    audit_payload = encoded_json(audit)
    write_new(
        (
            (OUTPUT, data_payload),
            (SIDECAR, sidecar_payload),
            (GATE, gate_payload),
            (AUDIT, audit_payload),
        )
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return audit


def self_test() -> None:
    assert sum(MATERIAL_QUOTAS.values()) == 512
    assert sum(USER_QUOTAS.values()) == 128
    assert sum(RETENTION_QUOTAS.values()) == 1_920
    assert TOTAL_ROWS == 2_560
    assert focus_index("a_firstdiv") == 0
    assert focus_index("b_firstdiv") == 1
    assert focus_index("c_firstdiv") == 2
    assert focus_index("full_anchor") is None
    print("i39 formal builder self-test: PASS")


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
