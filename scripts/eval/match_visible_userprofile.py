#!/usr/bin/env python3
"""Test whether a visible recommendation prompt comes from an O2 UserProfile row.

The platform log exposes SID tokens while O2 UserProfile stores PID lists. This
diagnostic inverts only the SIDs in one frozen prompt through the registered
``assets/derived/index/pid2sid.parquet`` index, then reports the UserProfile row
with the largest number of *unique visible SID* matches. It does not create a
dataset or claim that a match recovers the platform gold.

Run one shard per process to keep Arrow's peak memory bounded::

    python3 scripts/eval/match_visible_userprofile.py --shard part-00000.parquet
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "logs/eval/riders_fk_lora_ep1_20260706.log"
PID2SID = ROOT / "assets/derived/index/pid2sid.parquet"
USER_PROFILE = ROOT / "assets/official/hf_raw/OneReason_UserProfile"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITEM_RE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)

DOMAIN_COLUMNS = {
    "video": ["video_history_sampled_pid_list"],
    "prod": [
        "ec_item_id_list",
        "ec_colossus_rs_item_id_list",
        "ec_good_click_item_id_list_extend",
        "ec_good_order_item_id_list_extend",
    ],
    "ad": [
        "outer_loop_history_action_pid_list_pos",
        "outer_loop_history_action_pid_list_click",
    ],
    "living": ["live_hist_author_id_list", "live_hist_live_id_list"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_prompt(log_path: Path, task: str, sample_id: int) -> str:
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    text = text.replace("\r", "\n")
    task_match = re.search(
        rf"Task \[\d/8\]:\s*{re.escape(task)}\s*\|\s*Split:\s*test",
        text,
    )
    if task_match is None:
        raise ValueError(f"task {task!r} not found in {log_path}")
    section = text[task_match.end():]
    sample_match = re.search(rf"Sample ID: {sample_id}\nInput:\n", section)
    if sample_match is None:
        raise ValueError(f"sample {sample_id} not found for {task}")
    output_start = section.find("\nOutput[0]:\n", sample_match.end())
    if output_start < 0:
        raise ValueError("sample output boundary not found")
    return section[sample_match.end():output_start].strip("\n")


def target_sids(prompt: str) -> list[tuple[str, int, int, int]]:
    values = []
    for domain, a, b, c in ITEM_RE.findall(prompt):
        item = (domain, int(a), int(b), int(c))
        if item not in values:
            values.append(item)
    return values


def invert_sids(targets: list[tuple[str, int, int, int]]):
    target_ids = {target: index for index, target in enumerate(targets)}
    target_codes = np.array(
        list({(a << 26) | (b << 13) | c for _, a, b, c in targets}),
        dtype=np.int64,
    )
    pid_to_target = {domain: {} for domain in DOMAIN_COLUMNS}
    parquet = pq.ParquetFile(PID2SID)
    for batch in parquet.iter_batches(
        columns=["key", "s_a", "s_b", "s_c"], batch_size=1_000_000
    ):
        a = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        b = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        c = batch.column(3).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        codes = (a << 26) | (b << 13) | c
        indices = np.flatnonzero(np.isin(codes, target_codes))
        if not len(indices):
            continue
        keys = batch.column(0).take(pa.array(indices)).to_pylist()
        for row_index, key in zip(indices, keys):
            domain, pid = key.split("|", 1)
            target = (domain, int(a[row_index]), int(b[row_index]), int(c[row_index]))
            if domain in pid_to_target and target in target_ids:
                pid_to_target[domain][int(pid)] = target_ids[target]
    return pid_to_target


def non_null_numpy(array: pa.Array):
    values = array.to_numpy(zero_copy_only=False)
    if values.dtype != object:
        return values.astype(np.int64, copy=False), np.ones(len(values), dtype=bool)
    valid = np.fromiter((value is not None for value in values), dtype=bool, count=len(values))
    return values[valid].astype(np.int64, copy=False), valid


def scan_shard(shard: Path, pid_to_target: dict[str, dict[int, int]]):
    needed_columns = sorted(
        {column for columns in DOMAIN_COLUMNS.values() for column in columns}
    )
    lookup = {}
    for domain, mapping in pid_to_target.items():
        pids = np.array(sorted(mapping), dtype=np.int64)
        ids = np.array([mapping[int(pid)] for pid in pids], dtype=np.int16)
        lookup[domain] = (pids, ids)

    best = {"unique_sid_matches": 0, "row_ordinal": None}
    row_offset = 0
    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(columns=needed_columns, batch_size=1000):
        encoded_pairs = []
        for domain, columns in DOMAIN_COLUMNS.items():
            pids, target_ids = lookup[domain]
            if not len(pids):
                continue
            for column in columns:
                values = batch.column(batch.schema.get_field_index(column))
                flat = values.flatten()
                parents = values.value_parent_indices().to_numpy(zero_copy_only=False)
                flat_values, valid = non_null_numpy(flat)
                parents = parents[valid]
                positions = np.searchsorted(pids, flat_values)
                matched = positions < len(pids)
                matched &= pids[np.minimum(positions, len(pids) - 1)] == flat_values
                if matched.any():
                    encoded_pairs.append(
                        parents[matched].astype(np.int64, copy=False) * 256
                        + target_ids[positions[matched]]
                    )
        if encoded_pairs:
            unique_pairs = np.unique(np.concatenate(encoded_pairs))
            counts = np.bincount(unique_pairs // 256, minlength=batch.num_rows)
            local_row = int(np.argmax(counts))
            local_count = int(counts[local_row])
            if local_count > best["unique_sid_matches"]:
                best = {
                    "unique_sid_matches": local_count,
                    "row_ordinal": row_offset + local_row,
                }
        row_offset += batch.num_rows
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--task", default="challenge_recommendation_video"
    )
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--shard", required=True)
    args = parser.parse_args()

    shard = USER_PROFILE / args.shard
    if not shard.is_file():
        raise SystemExit(f"missing registered UserProfile shard: {shard}")
    prompt = visible_prompt(args.log, args.task, args.sample_id)
    targets = target_sids(prompt)
    if not targets:
        raise SystemExit("visible prompt contains no itemic SID")
    pid_to_target = invert_sids(targets)
    best = scan_shard(shard, pid_to_target)
    result = {
        "source_log": str(args.log),
        "source_log_sha256": sha256_file(args.log),
        "task": args.task,
        "sample_id": args.sample_id,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_unique_sids": len(targets),
        "target_by_domain": dict(collections.Counter(target[0] for target in targets)),
        "candidate_pid_count": {
            domain: len(mapping) for domain, mapping in pid_to_target.items()
        },
        "shard": args.shard,
        **best,
        "interpretation": "diagnostic_only_not_gold_recovery",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
