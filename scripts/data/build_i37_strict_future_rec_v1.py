#!/usr/bin/env python3
"""Build a small strict-future video/ad residual dataset over I-35.

The CE rows use only official UserProfile fields whose target timestamp is
strictly after the rendered history.  The retention rows are copied from the
registered I-35 mixture and are never used as CE targets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = ROOT / "assets/official/hf_raw/OneReason_UserProfile"
SID_ROOT = ROOT / "assets/official/hf_raw/OneReason_Pid2Sid"
RETENTION_DATA = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"

SCHEMA_VERSION = "i37-strict-future-rec-v1"
SEED = 19260837
VIDEO_GAP_MS = 10 * 60 * 1000
FUTURE_PER_DOMAIN = 512
RETENTION_COUNTS = {
    "material_sid2desc": 128,
    "action": 128,
    "topic": 128,
    "rec_video": 131,
    "rec_prod": 131,
    "rec_ad": 131,
    "rec_living": 131,
    "world": 116,
}


def load_rec_helpers():
    path = Path(__file__).with_name("build_official_rec_v3.py")
    spec = importlib.util.spec_from_file_location("i37_rec_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = load_rec_helpers()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def clean_aligned(value: Any) -> list[Any]:
    return [item for item in as_list(value) if item is not None]


def as_ts(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def token_for(sid_maps: dict[str, dict[int, Any]], domain: str, pid: Any) -> str | None:
    return H.sid_lookup(sid_maps, domain, pid)


def history_max_ts(row: dict[str, Any], domain: str) -> int | None:
    if domain == "video":
        values = [as_ts(v) for v in as_list(row.get("video_history_ts_list"))]
    else:
        values = []
        for col in (
            "outer_loop_history_action_pid_list_pos_ts",
            "outer_loop_history_action_pid_list_click_ts",
        ):
            values.extend(as_ts(v) for v in as_list(row.get(col)))
    values = [v for v in values if v is not None]
    return max(values) if values else None


def strict_video_target(
    row: dict[str, Any], sid_maps: dict[str, dict[int, Any]], history_tokens: set[str]
) -> tuple[str, int] | None:
    last_history = history_max_ts(row, "video")
    if last_history is None:
        return None
    pids = as_list(row.get("video_sampled_pid_list"))
    ts = as_list(row.get("video_ts_list"))
    done = as_list(row.get("video_play_done_list"))
    candidates: list[tuple[int, str]] = []
    for index, pid in enumerate(pids):
        if index >= len(ts) or index >= len(done) or not H.flag(done[index]):
            continue
        stamp = as_ts(ts[index])
        if stamp is None or stamp <= last_history + VIDEO_GAP_MS:
            continue
        token = token_for(sid_maps, "video", pid)
        if token and token not in history_tokens:
            candidates.append((stamp, token))
    if not candidates:
        return None
    stamp, token = min(candidates)
    return token, stamp


def strict_ad_target(
    row: dict[str, Any], sid_maps: dict[str, dict[int, Any]], history_tokens: set[str]
) -> tuple[str, int] | None:
    last_history = history_max_ts(row, "ad")
    if last_history is None:
        return None
    pids = clean_aligned(row.get("outer_loop_deep_target_pid"))
    ts = clean_aligned(row.get("outer_loop_deep_target_pid_ts"))
    candidates: list[tuple[int, str]] = []
    for pid, raw_ts in zip(pids, ts):
        stamp = as_ts(raw_ts)
        token = token_for(sid_maps, "ad", pid)
        if stamp is not None and stamp > last_history and token and token not in history_tokens:
            candidates.append((stamp, token))
    if not candidates:
        return None
    stamp, token = min(candidates)
    return token, stamp


def future_row(
    row: dict[str, Any],
    domain: str,
    sid_maps: dict[str, dict[int, Any]],
) -> dict[str, Any] | None:
    # Filter on the target-domain history first.  Rendering all four domains
    # for every UserProfile row is needlessly expensive when most rows do not
    # contain a strict future candidate.
    if domain == "video":
        history_pids = as_list(row.get("video_history_sampled_pid_list"))
    else:
        history_pids = as_list(row.get("outer_loop_history_action_pid_list_pos"))
        history_pids += as_list(row.get("outer_loop_history_action_pid_list_click"))
    history_tokens = {
        token
        for pid in history_pids
        if (token := token_for(sid_maps, domain, pid)) is not None
    }
    target = strict_video_target(row, sid_maps, history_tokens) if domain == "video" else strict_ad_target(row, sid_maps, history_tokens)
    if target is None:
        return None
    target_token, target_ts = target
    # Render only accepted candidates with the target banned from every block.
    events = H.build_events_for_row(row, sid_maps, banned={target_token})
    if len(events[domain]) < 8:
        return None
    prompt = H.build_prompt(domain, events)
    if not prompt or target_token in prompt or len(prompt) > 16384:
        return None
    return {
        "instruction": f"你是一个推荐系统助手，擅长根据用户属性与多域历史行为预测用户的{H.CN[domain]}偏好。",
        "input": prompt,
        "output": f"<think>\n</think>\n{target_token}",
        "history": [],
        "task": f"future_{domain}",
        "i37_route": "future_ce",
        "i37_target_ts": target_ts,
        "i37_future_gap_ms": VIDEO_GAP_MS if domain == "video" else target_ts - (history_max_ts(row, "ad") or target_ts),
    }


def classify_retention(row: dict[str, Any]) -> str:
    output = str(row.get("output") or "")
    body = output.split("</think>", 1)[-1].strip()
    if body.startswith("["):
        return "action"
    if body.startswith("{") and "logic_chain" in body:
        return "topic"
    if "该用户最近" in body:
        for domain in ("video", "prod", "ad", "living"):
            if f"<|{domain}_begin|>" in body:
                return f"rec_{domain}"
    input_text = str(row.get("input") or "")
    input_has_sid = "<s_a_" in input_text
    output_has_sid = "<s_a_" in body
    if output_has_sid and not input_has_sid:
        return "material_desc2sid"
    if input_has_sid and not output_has_sid:
        return "material_sid2desc"
    return "world"


def load_retention(rng: random.Random) -> tuple[list[dict[str, Any]], Counter[str]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    with RETENTION_DATA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = classify_retention(row)
            if task not in RETENTION_COUNTS:
                continue
            row = {key: row[key] for key in ("instruction", "input", "output", "history", "task") if key in row}
            row["task"] = task
            row["i37_route"] = "retention_kl"
            row["i37_source"] = "data_user_residual_retention_v1"
            buckets.setdefault(task, []).append(row)
    selected: list[dict[str, Any]] = []
    for task, count in RETENTION_COUNTS.items():
        values = buckets.get(task, [])
        rng.shuffle(values)
        if len(values) < count:
            raise RuntimeError(f"retention bucket too small: {task} {len(values)}/{count}")
        selected.extend(values[:count])
    rng.shuffle(selected)
    return selected, Counter(classify_retention(row) for row in selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--up-dir", type=Path, default=UPSTREAM_ROOT)
    parser.add_argument("--sid-dir", type=Path, default=SID_ROOT)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--future-per-domain", type=int, default=FUTURE_PER_DOMAIN)
    args = parser.parse_args()
    if args.future_per_domain != FUTURE_PER_DOMAIN:
        raise SystemExit("I37 fixed contract requires 512 rows per future domain")
    if not RETENTION_DATA.is_file():
        raise SystemExit(f"missing registered retention source: {RETENTION_DATA}")

    print("[i37] loading registered Pid2Sid maps", flush=True)
    sid_maps = H.build_sid_maps(args.sid_dir)
    rng = random.Random(SEED)
    future: dict[str, list[dict[str, Any]]] = {"video": [], "ad": []}
    reject = Counter()
    files = sorted(args.up_dir.glob("part-*.parquet"))[: args.shards]
    if len(files) != args.shards:
        raise RuntimeError(f"I37 expected {args.shards} UserProfile shards, found {len(files)}")
    columns = [
        "video_sampled_pid_list", "video_ts_list", "video_play_done_list",
        "video_history_sampled_pid_list", "video_history_ts_list",
        "video_history_neg_feedback_list", "video_history_like_list",
        "video_history_comment_list", "video_history_forward_list",
        "video_history_collect_list", "video_history_watch_time_list",
        "video_history_play_done_list", "video_history_duration_list",
        "outer_loop_history_action_pid_list_pos", "outer_loop_history_action_pid_list_pos_ts",
        "outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts",
        "outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts",
        "ec_time_ms", "ec_good_click_item_id_list_extend", "ec_trunc_clk_lag",
        "ec_good_order_item_id_list_extend", "ec_trunc_buy_lag",
        "ec_colossus_rs_item_id_list", "ec_colossus_rs_lagv1_list",
        "ec_colossus_rs_is_click_list", "ec_colossus_rs_is_cart_list", "ec_colossus_rs_is_buy_list",
        "live_hist_author_id_list", "live_hist_timestamp_list", "live_hist_follow_author_cnt_list",
    ]
    for file_path in files:
        schema = set(pq.ParquetFile(file_path).schema_arrow.names)
        read_columns = [column for column in columns if column in schema]
        print(f"[i37] scanning {file_path.name}", flush=True)
        for batch in pq.ParquetFile(file_path).iter_batches(batch_size=256, columns=read_columns):
            for row in batch.to_pylist():
                for domain in ("video", "ad"):
                    if len(future[domain]) >= FUTURE_PER_DOMAIN:
                        continue
                    try:
                        built = future_row(row, domain, sid_maps)
                    except (KeyError, TypeError, ValueError, IndexError):
                        built = None
                    if built is None:
                        reject[f"{domain}:invalid_or_not_future"] += 1
                    else:
                        future[domain].append(built)
            if all(len(future[domain]) >= FUTURE_PER_DOMAIN for domain in future):
                break
        print(f"[i37] future buckets { {k: len(v) for k, v in future.items()} }", flush=True)
        if all(len(future[domain]) >= FUTURE_PER_DOMAIN for domain in future):
            break
    if any(len(future[domain]) < FUTURE_PER_DOMAIN for domain in future):
        raise RuntimeError(f"I37 could not fill future buckets: { {k: len(v) for k, v in future.items()} }")

    retention, retention_counts = load_retention(rng)
    rows = future["video"][:FUTURE_PER_DOMAIN] + future["ad"][:FUTURE_PER_DOMAIN] + retention
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    route_counts = Counter(row["i37_route"] for row in rows)
    task_counts = Counter(row["task"] for row in rows)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_training",
        "asset_class": "D(O2.UserProfile,O2.Pid2Sid; I12 retention source)",
        "seed": SEED,
        "upstreams": {
            "userprofile": {"path": str(args.up_dir.resolve()), "class": "O"},
            "pid2sid": {"path": str(args.sid_dir.resolve()), "class": "O"},
            "i12_retention": {"path": str(RETENTION_DATA.resolve()), "class": "D", "sha256": sha256(RETENTION_DATA)},
        },
        "builder": "scripts/data/build_i37_strict_future_rec_v1.py",
        "contract": {
            "future_gap_video_ms": VIDEO_GAP_MS,
            "future_domains": {"video": FUTURE_PER_DOMAIN, "ad": FUTURE_PER_DOMAIN},
            "retention_counts": RETENTION_COUNTS,
            "route_counts": dict(route_counts),
            "task_counts": dict(task_counts),
            "total_rows": len(rows),
        },
        "reject_counts": dict(reject),
        "output": {"path": str(args.out.resolve()), "rows": len(rows), "sha256": sha256(args.out)},
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "routes": route_counts, "tasks": task_counts, "audit": str(args.audit)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
