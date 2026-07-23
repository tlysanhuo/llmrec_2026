#!/usr/bin/env python3
"""Build the I-36 user-expansion residual mix over the I-35 step548 parent.

The imported user artifact is treated as construction input, never as a
ready-to-train dataset.  This builder removes prompt demonstrations, drops
domain/SID-inconsistent history events, canonicalizes action targets, keeps
only official-shaped logic-chain nodes, excludes registered E prompts, and
adds six non-target tasks of frozen-I35 KL retention.  Action/topic rows also
receive weak parent KL in the custom trainer, so all eight score items retain
a parent anchor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "i36-i35-user-expand-retkl-v1"
SEED = 19260836

RAW_SOURCE = ROOT / "assets/derived/processed/data_user_topic10000_logic5000_v1_raw.jsonl"
I35_RETENTION = ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"
O1_USER = ROOT / "assets/official/seed_sft/懂用户.jsonl"
E_MANIFEST_SOURCE = ROOT / "logs/data/i35_video_material_beam128_pool_v1_audit.json"
O2_PID2SID = ROOT / "assets/official/hf_raw/OneReason_Pid2Sid"

OUTPUT = ROOT / "assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl"
AUDIT = ROOT / "logs/data/i36_i35_user_expand_retkl_v1_audit.json"

EXPECTED_RAW_SHA256 = "62f13962d4cfc0d4c2b591f2b9fd598d820e37ffcf5ac51433a0d5b9b8dd5ffa"
EXPECTED_I35_SHA256 = "9c044e47d26fb7644281107a548249e49564e0f203a04795337c6a90c0927100"
EXPECTED_O1_USER_SHA256 = "8fbd7d0c2b2c2fde6ac00a4980c6f8bad5721ffab87d682b3096ba66d776d2ff"
EXPECTED_E_MANIFEST_SHA256 = "b290946a509de4713fa0f5750a3e3e23c9589b8895639c51c1c7250604637d50"

ACTION_ROWS = 4000
TOPIC_ROWS = 1500
RETENTION_COUNTS = {
    "material": 2500,
    "rec_video": 2000,
    "rec_prod": 1750,
    "rec_ad": 1750,
    "rec_living": 1500,
    "world": 1500,
}

SID_PATTERN = (
    r"<\|(ad|video|prod|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)
SID_RE = re.compile(SID_PATTERN)
FULL_SID_RE = re.compile(rf"^{SID_PATTERN}$")
EVENT_RE = re.compile(
    rf"^(\s*)(\d{{2}}:\d{{2}}|--:--) \[([^]-]+)-([^]]+)\] ({SID_PATTERN})$"
)
DATE_RE = re.compile(r"^【(\d{4}-\d{2}-\d{2})】$")
LABEL_DOMAIN = {"广告": "ad", "视频": "video", "商品": "prod", "直播": "living"}
SOURCE_TASKS = {"user_topic_extract", "user_logic_chain"}


class ContractError(RuntimeError):
    """Raised when a formal construction invariant is violated."""


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return sha256_bytes(canonical(parts).encode("utf-8"))


def load_jsonl(path: Path, *, flexible: bool = False) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank row at {path}:{line_number}")
            try:
                value = json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ContractError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            values = value if flexible and isinstance(value, list) else [value]
            require(values and all(isinstance(item, dict) for item in values), f"non-object row at {path}:{line_number}")
            rows.extend(values)
    return rows


def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def prompt_digest(row: Mapping[str, Any], *, mode_normalized: bool = False) -> str:
    value = normalized(row)
    user_input = value["input"].rstrip()
    if mode_normalized:
        for suffix in ("/no_think", "/think"):
            if user_input.endswith(suffix):
                user_input = user_input[: -len(suffix)].rstrip()
                break
    return stable_hash(value["instruction"], user_input, value["history"])


def history_text(user_input: str) -> str:
    marker = "\n角色任务："
    require(marker in user_input, "user prompt lacks the role-task boundary")
    return user_input.split(marker, 1)[0].rstrip()


def history_digest(user_input: str) -> str:
    return sha256_bytes(history_text(user_input).encode("utf-8"))


def strip_demonstration(user_input: str, task: str) -> str:
    if task == "user_topic_extract":
        marker = "\n输出示例为"
    elif task == "user_logic_chain":
        marker = "\n有效逻辑链案例"
    else:
        raise ContractError(f"unknown source task: {task}")
    require(marker in user_input, f"{task} prompt lacks its demonstration marker")
    return user_input.split(marker, 1)[0].rstrip() + "/no_think"


def clean_domain_mismatches(user_input: str) -> tuple[str, int, list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for line in user_input.splitlines():
        match = EVENT_RE.fullmatch(line)
        if match is not None:
            label = match.group(3)
            sid_domain = match.group(6)
            expected = LABEL_DOMAIN.get(label)
            require(expected is not None, f"unknown event label domain: {label!r}")
            if expected != sid_domain:
                removed.append(match.group(5))
                continue
        kept.append(line)
    return "\n".join(kept), len(removed), removed


def answer_body(output: str) -> str:
    prefix = "<think>\n</think>\n"
    require(output.startswith(prefix), "source response is not canonical empty-think")
    body = output[len(prefix) :].strip()
    require(bool(body), "source response body is empty")
    return body


def sid_positions(text: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, match in enumerate(SID_RE.finditer(text)):
        positions.setdefault(match.group(0), index)
    return positions


def output_bins(counts: Sequence[int]) -> dict[str, int]:
    bins = Counter()
    for count in counts:
        if count <= 5:
            bins["01_05"] += 1
        elif count <= 10:
            bins["06_10"] += 1
        elif count <= 20:
            bins["11_20"] += 1
        elif count <= 32:
            bins["21_32"] += 1
        else:
            bins["33_56"] += 1
    return dict(sorted(bins.items()))


def parse_timeline(user_input: str) -> set[tuple[str, str]]:
    timeline: set[tuple[str, str]] = set()
    current_date: str | None = None
    for line in history_text(user_input).splitlines():
        date_match = DATE_RE.fullmatch(line)
        if date_match is not None:
            current_date = date_match.group(1)
            continue
        event_match = EVENT_RE.fullmatch(line)
        if event_match is not None:
            require(current_date is not None, "event appears before the first date")
            action = f"[{event_match.group(3)}-{event_match.group(4)}] {event_match.group(5)}"
            timeline.add((current_date, action))
    return timeline


def build_e_sets() -> tuple[set[str], set[str], list[dict[str, Any]]]:
    require(sha256_file(E_MANIFEST_SOURCE) == EXPECTED_E_MANIFEST_SHA256, "E manifest source hash drift")
    source = json.loads(E_MANIFEST_SOURCE.read_text(encoding="utf-8"))
    manifest = source.get("e_manifest")
    require(isinstance(manifest, list) and manifest, "registered E manifest is absent")
    exact: set[str] = set()
    modes: set[str] = set()
    audit: list[dict[str, Any]] = []
    for entry in manifest:
        require(isinstance(entry, dict), "invalid E manifest entry")
        path = ROOT / str(entry.get("path"))
        require(path.is_file(), f"registered E asset is missing: {path}")
        observed_hash = sha256_file(path)
        require(observed_hash == entry.get("sha256"), f"registered E asset hash drift: {path}")
        rows = load_jsonl(path, flexible=True)
        require(len(rows) == entry.get("rows"), f"registered E row count drift: {path}")
        for row in rows:
            exact.add(prompt_digest(row))
            modes.add(prompt_digest(row, mode_normalized=True))
        audit.append({"path": str(path.relative_to(ROOT)), "rows": len(rows), "sha256": observed_hash})
    return exact, modes, audit


def o1_history_inventory() -> tuple[set[str], set[str]]:
    require(sha256_file(O1_USER) == EXPECTED_O1_USER_SHA256, "O1 user source hash drift")
    rows = load_jsonl(O1_USER, flexible=True)
    require(len(rows) == 2892, f"O1 user row count drifted: {len(rows)}")
    prompts = {prompt_digest(row) for row in rows}
    histories = {history_digest(normalized(row)["input"]) for row in rows}
    return prompts, histories


def build_action_candidate(
    row: Mapping[str, Any], line_number: int, counters: Counter[str]
) -> dict[str, Any] | None:
    raw_input = str(row.get("input") or "")
    stripped = strip_demonstration(raw_input, "user_topic_extract")
    cleaned, removed_count, removed_sids = clean_domain_mismatches(stripped)
    counters["domain_mismatch_events_removed"] += removed_count
    try:
        targets = json.loads(answer_body(str(row.get("output") or "")))
    except (json.JSONDecodeError, ContractError):
        counters["invalid_output_json"] += 1
        return None
    if not isinstance(targets, list) or not targets or not all(
        isinstance(item, str) and FULL_SID_RE.fullmatch(item) for item in targets
    ):
        counters["invalid_target_array"] += 1
        return None
    deduplicated = list(dict.fromkeys(targets))
    counters["duplicate_target_occurrences_removed"] += len(targets) - len(deduplicated)
    positions = sid_positions(history_text(cleaned))
    if any(target not in positions for target in deduplicated):
        counters["target_lost_after_domain_cleanup"] += 1
        return None
    if not 1 <= len(deduplicated) <= 56:
        counters["target_count_outside_official_range"] += 1
        return None
    chronological = sorted(deduplicated, key=positions.__getitem__)
    if chronological != deduplicated:
        counters["target_arrays_reordered_chronologically"] += 1
    output = "<think>\n</think>\n" + json.dumps(chronological, ensure_ascii=False)
    return {
        "instruction": "",
        "input": cleaned,
        "output": output,
        "history": [],
        "task": "action",
        "i36_route": "user_ce",
        "i36_source": {
            "source_line": line_number,
            "source_task": "user_topic_extract",
            "source_prompt_sha256": sha256_bytes(raw_input.encode("utf-8")),
            "removed_domain_mismatch_events": removed_count,
            "removed_domain_mismatch_sids": len(set(removed_sids)),
            "target_count": len(chronological),
        },
    }


def build_topic_candidate(
    row: Mapping[str, Any], line_number: int, counters: Counter[str]
) -> dict[str, Any] | None:
    raw_input = str(row.get("input") or "")
    stripped = strip_demonstration(raw_input, "user_logic_chain")
    cleaned, removed_count, _ = clean_domain_mismatches(stripped)
    counters["domain_mismatch_events_removed"] += removed_count
    try:
        payload = json.loads(answer_body(str(row.get("output") or "")))
    except (json.JSONDecodeError, ContractError):
        counters["invalid_output_json"] += 1
        return None
    chain = payload.get("logic_chain") if isinstance(payload, dict) else None
    events = chain.get("events") if isinstance(chain, dict) else None
    name = chain.get("name") if isinstance(chain, dict) else None
    meta = row.get("meta")
    topic = meta.get("topic") if isinstance(meta, dict) else None
    if not isinstance(name, str) or name != topic or not isinstance(events, list) or not 3 <= len(events) <= 5:
        counters["invalid_logic_schema"] += 1
        return None
    timeline = parse_timeline(cleaned)
    previous_date = ""
    for event in events:
        if not isinstance(event, dict) or set(event) != {"date", "action", "logic"}:
            counters["invalid_logic_event"] += 1
            return None
        date, action, logic = event["date"], event["action"], event["logic"]
        if not all(isinstance(value, str) and value for value in (date, action, logic)):
            counters["invalid_logic_event"] += 1
            return None
        if len(SID_RE.findall(action)) != 1 or (date, action) not in timeline:
            counters["non_official_shaped_logic_action"] += 1
            return None
        if date < previous_date:
            counters["non_chronological_logic_chain"] += 1
            return None
        previous_date = date
    output = "<think>\n</think>\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    return {
        "instruction": "",
        "input": cleaned,
        "output": output,
        "history": [],
        "task": "topic",
        "i36_route": "user_ce",
        "i36_source": {
            "source_line": line_number,
            "source_task": "user_logic_chain",
            "source_prompt_sha256": sha256_bytes(raw_input.encode("utf-8")),
            "removed_domain_mismatch_events": removed_count,
            "logic_nodes": len(events),
        },
    }


def select_unique_histories(
    candidates: Sequence[dict[str, Any]], wanted: int, label: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[history_digest(row["input"])].append(row)
    representatives = []
    for key, values in grouped.items():
        values.sort(key=lambda row: stable_hash(SEED, label, key, row["input"], row["output"]))
        representatives.append(values[0])
    representatives.sort(key=lambda row: stable_hash(SEED, "select", label, row["input"], row["output"]))
    require(len(representatives) >= wanted, f"only {len(representatives)} unique {label} histories; need {wanted}")
    return representatives[:wanted], {
        "candidate_rows": len(candidates),
        "unique_histories": len(representatives),
        "selected_rows": wanted,
        "discarded_same_history_variants": len(candidates) - len(representatives),
    }


def retention_bucket(task: str) -> str | None:
    if task == "material_desc2sid":
        return "material"
    if task in RETENTION_COUNTS:
        return task
    return None


def build_retention_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(sha256_file(I35_RETENTION) == EXPECTED_I35_SHA256, "I35 retention source hash drift")
    source_rows = load_jsonl(I35_RETENTION)
    require(len(source_rows) == 2740, f"I35 retention source count drifted: {len(source_rows)}")
    buckets: dict[str, list[dict[str, Any]]] = {task: [] for task in RETENTION_COUNTS}
    ignored = Counter()
    for row in source_rows:
        task = str(row.get("task") or "")
        bucket = retention_bucket(task)
        if bucket is None:
            ignored[task] += 1
            continue
        value = normalized(row)
        require(value["input"] and value["output"] and isinstance(value["history"], list), "invalid I35 retention row")
        value["task"] = task
        buckets[bucket].append(value)

    output: list[dict[str, Any]] = []
    allocation: dict[str, Any] = {}
    for bucket, wanted in RETENTION_COUNTS.items():
        candidates = buckets[bucket]
        require(candidates, f"I35 source lacks retention bucket: {bucket}")
        candidates.sort(key=lambda row: stable_hash(SEED, "retention", bucket, row))
        for index in range(wanted):
            source = candidates[index % len(candidates)]
            value = copy.deepcopy(source)
            value["i36_route"] = "retention_kl"
            value["i36_source"] = {
                "source": "data_i35_video_boundary_retkl_v1",
                "bucket": bucket,
                "cycle": index // len(candidates),
                "source_index": index % len(candidates),
            }
            output.append(value)
        allocation[bucket] = {
            "available_unique_rows": len(candidates),
            "selected_with_deterministic_replay": wanted,
            "maximum_exposures_per_unique_row": math.ceil(wanted / len(candidates)),
        }
    return output, {
        "allocation": allocation,
        "ignored_source_tasks": dict(sorted(ignored.items())),
    }


def pack_sid(match: Sequence[str]) -> int:
    domain, a_text, b_text, c_text = match
    domain_id = {"ad": 0, "video": 1, "prod": 2, "living": 3}[domain]
    a, b, c = int(a_text), int(b_text), int(c_text)
    require(all(0 <= value < 8192 for value in (a, b, c)), "SID component outside 13-bit range")
    return (domain_id << 39) | (a << 26) | (b << 13) | c


def verify_o2_sid_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ContractError("pyarrow and numpy are required for O2 SID verification") from exc

    required = {
        pack_sid(match.groups())
        for row in rows
        for match in SID_RE.finditer(str(row.get("input") or ""))
    }
    require(required, "formal user rows contain no SID tokens")
    required_sorted = np.asarray(sorted(required), dtype=np.int64)
    found: set[int] = set()
    files = sorted(O2_PID2SID.glob("part-*.parquet"))
    require(len(files) == 198, f"O2 Pid2Sid shard count drifted: {len(files)}")
    raw_domain_ids = {"video/ad": 0, "video/video": 1, "goods": 2, "live": 3}

    for shard in files:
        table = pq.read_table(shard, columns=["domain", "sid_three"])
        domains = table.column("domain").combine_chunks()
        sid_lists = table.column("sid_three").combine_chunks()
        offsets = sid_lists.offsets.to_numpy(zero_copy_only=False)
        require(bool(np.all(np.diff(offsets) == 3)), f"non-three-part SID in {shard}")
        values = sid_lists.values.to_numpy(zero_copy_only=False).reshape(-1, 3)
        require(bool(np.all(np.isfinite(values))), f"non-finite SID component in {shard}")
        integers = values.astype(np.int64)
        require(bool(np.all(values == integers)), f"non-integral SID component in {shard}")
        domain_ids = np.full(len(table), -1, dtype=np.int64)
        for raw_domain, domain_id in raw_domain_ids.items():
            mask = pc.equal(domains, raw_domain).to_numpy(zero_copy_only=False)
            domain_ids[mask] = domain_id
        require(bool(np.all(domain_ids >= 0)), f"unknown O2 Pid2Sid domain in {shard}")
        packed = (
            (domain_ids << 39)
            | (integers[:, 0] << 26)
            | (integers[:, 1] << 13)
            | integers[:, 2]
        )
        indices = np.searchsorted(required_sorted, packed)
        mask = indices < required_sorted.size
        matched = packed[mask]
        matched_indices = indices[mask]
        matched = matched[required_sorted[matched_indices] == matched]
        found.update(np.unique(matched).tolist())
        if len(found) == len(required):
            break

    missing = sorted(required - found)
    require(not missing, f"{len(missing)} formal user SIDs are absent from O2.Pid2Sid")
    return {
        "registered_asset": "O2.OneReason_Pid2Sid",
        "registered_path": str(O2_PID2SID.relative_to(ROOT)),
        "shards_total": len(files),
        "shards_scanned_until_complete": files.index(shard) + 1,
        "unique_formal_user_sids": len(required),
        "matched_unique_sids": len(found),
        "missing_unique_sids": 0,
    }


def summarize(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    require(bool(ordered), "cannot summarize an empty sequence")
    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 6),
    }


def write_new(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite existing formal output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    require(sha256_file(RAW_SOURCE) == EXPECTED_RAW_SHA256, "imported raw source hash drift")
    raw_rows = load_jsonl(RAW_SOURCE)
    require(len(raw_rows) == 15023, f"raw source count drifted: {len(raw_rows)}")
    source_counts = Counter()
    action_counters: Counter[str] = Counter()
    topic_counters: Counter[str] = Counter()
    action_candidates: list[dict[str, Any]] = []
    topic_candidates: list[dict[str, Any]] = []
    for line_number, row in enumerate(raw_rows, 1):
        meta = row.get("meta")
        require(isinstance(meta, dict), f"raw row {line_number} lacks meta")
        task = meta.get("task")
        require(task in SOURCE_TASKS, f"raw row {line_number} has unknown task {task!r}")
        source_counts[str(task)] += 1
        if task == "user_topic_extract":
            candidate = build_action_candidate(row, line_number, action_counters)
            if candidate is not None:
                action_candidates.append(candidate)
        else:
            candidate = build_topic_candidate(row, line_number, topic_counters)
            if candidate is not None:
                topic_candidates.append(candidate)

    require(source_counts == {"user_topic_extract": 10001, "user_logic_chain": 5022}, f"raw task signature drifted: {dict(source_counts)}")
    e_exact, e_modes, e_manifest = build_e_sets()
    o1_prompts, o1_histories = o1_history_inventory()

    exclusion = Counter()
    def admissible(row: dict[str, Any]) -> bool:
        exact = prompt_digest(row)
        mode = prompt_digest(row, mode_normalized=True)
        if exact in e_exact or mode in e_modes:
            exclusion["registered_E_prompt"] += 1
            return False
        if exact in o1_prompts:
            exclusion["exact_O1_user_prompt"] += 1
            return False
        return True

    action_candidates = [row for row in action_candidates if admissible(row)]
    topic_candidates = [row for row in topic_candidates if admissible(row)]
    topic_history_inventory = {history_digest(row["input"]) for row in topic_candidates}
    action_disjoint_pool = [
        row for row in action_candidates
        if history_digest(row["input"]) not in topic_history_inventory
    ]
    action_rows, action_selection = select_unique_histories(
        action_disjoint_pool, ACTION_ROWS, "action-disjoint"
    )
    topic_rows, topic_selection = select_unique_histories(topic_candidates, TOPIC_ROWS, "topic")
    selected_user = action_rows + topic_rows
    require(len({prompt_digest(row) for row in selected_user}) == len(selected_user), "duplicate formal user prompts")

    selected_history_counts = Counter(history_digest(row["input"]) for row in selected_user)
    cross_task_history_overlap = sum(count - 1 for count in selected_history_counts.values() if count > 1)
    require(cross_task_history_overlap == 0, "formal action/topic histories overlap")

    o1_history_overlap = sum(history_digest(row["input"]) in o1_histories for row in selected_user)
    require(o1_history_overlap == 0, "formal user histories overlap O1 user histories")
    sid_audit = verify_o2_sid_inventory(selected_user)

    retention_rows, retention_audit = build_retention_rows()
    final_rows = selected_user + retention_rows
    require(len(final_rows) == 16500, f"formal mix count drifted: {len(final_rows)}")
    random.Random(SEED).shuffle(final_rows)

    route_counts = Counter(str(row.get("i36_route")) for row in final_rows)
    task_counts = Counter(str(row.get("task")) for row in final_rows)
    require(route_counts == {"user_ce": 5500, "retention_kl": 11000}, f"formal route signature drifted: {dict(route_counts)}")

    data_payload = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in final_rows
    )
    output_hash = sha256_bytes(data_payload)
    action_target_counts = [len(json.loads(answer_body(row["output"]))) for row in action_rows]
    topic_node_counts = [len(json.loads(answer_body(row["output"]))["logic_chain"]["events"]) for row in topic_rows]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_training",
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General; imported model annotations; M-I35 retention source)",
        "builder": str(Path(__file__).relative_to(ROOT)),
        "seed": SEED,
        "upstream": {
            "imported_user_annotation_artifact": {
                "path": str(RAW_SOURCE.relative_to(ROOT)),
                "rows": len(raw_rows),
                "bytes": RAW_SOURCE.stat().st_size,
                "sha256": EXPECTED_RAW_SHA256,
                "task_counts": dict(sorted(source_counts.items())),
                "generator_status": "source generator implementation not received; every formal SID is independently verified against registered O2.Pid2Sid",
            },
            "O2_Pid2Sid_lineage_check": sid_audit,
            "O1_user_shape_reference": {
                "path": str(O1_USER.relative_to(ROOT)),
                "rows": 2892,
                "sha256": EXPECTED_O1_USER_SHA256,
                "used_as_training_rows": 0,
            },
            "I35_parent_retention_source": {
                "path": str(I35_RETENTION.relative_to(ROOT)),
                "rows": 2740,
                "sha256": EXPECTED_I35_SHA256,
            },
        },
        "construction": {
            "user_supervision": {
                "rows": len(selected_user),
                "ratio": len(selected_user) / len(final_rows),
                "action": {
                    **action_selection,
                    "selected_target_counts": summarize(action_target_counts),
                    "selected_target_count_bins": output_bins(action_target_counts),
                    "transformations": dict(sorted(action_counters.items())),
                },
                "topic": {
                    **topic_selection,
                    "selected_logic_nodes": summarize(topic_node_counts),
                    "transformations": dict(sorted(topic_counters.items())),
                },
                "instruction_normalization": "generic helper system removed; official user task system is empty",
                "prompt_demonstrations_removed": True,
                "action_target_rule": "deduplicate then restore first-history-occurrence order; keep 1..56",
                "topic_target_rule": "3..5 nodes; exactly one SID per node; date/action exact in cleaned timeline",
                "action_topic_history_overlap": 0,
                "O1_user_history_overlap": o1_history_overlap,
            },
            "parent_retention": {
                "rows": len(retention_rows),
                "ratio": len(retention_rows) / len(final_rows),
                "counts": RETENTION_COUNTS,
                **retention_audit,
                "semantics": "KL-only against frozen I35 step548 parent; no gold CE",
            },
            "mix_ratio_user_to_kl_retention": "1:2",
            "registered_E_exclusion": {
                "manifest": e_manifest,
                "candidate_rows_excluded": dict(sorted(exclusion.items())),
                "formal_exact_overlap": sum(prompt_digest(row) in e_exact for row in selected_user),
                "formal_mode_overlap": sum(prompt_digest(row, mode_normalized=True) in e_modes for row in selected_user),
            },
            "forbidden_sources": {"third_party_rows": 0, "E_rows": 0},
        },
        "training_semantics": {
            "parent": "I35 step548 r112 adapter SHA256 52d945cc...2c00",
            "action_topic": "answer-body gold CE plus weak frozen-parent KL",
            "other_six_items": "frozen-parent KL only at bounded answer positions",
            "fresh_adapter": "r16/alpha16; exact concatenation with r112 produces legal r128",
        },
        "mix": {
            "total_rows": len(final_rows),
            "route_counts": dict(sorted(route_counts.items())),
            "task_counts": dict(sorted(task_counts.items())),
        },
        "output": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "rows": len(final_rows),
            "bytes": len(data_payload),
            "sha256": output_hash,
        },
    }
    audit_payload = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_new(OUTPUT, data_payload, args.overwrite)
    write_new(AUDIT, audit_payload, args.overwrite)
    print(
        f"[i36-builder] PASS rows={len(final_rows)} user/retention={len(selected_user)}/{len(retention_rows)} "
        f"action/topic={len(action_rows)}/{len(topic_rows)} unique_o2_sids={sid_audit['unique_formal_user_sids']} "
        f"sha256={output_hash}",
        flush=True,
    )


if __name__ == "__main__":
    main()
