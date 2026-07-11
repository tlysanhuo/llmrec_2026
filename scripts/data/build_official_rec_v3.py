#!/usr/bin/env python3
"""Build official_rec_v3 from official UserProfile + Pid2Sid.

This script deliberately does not use previously generated second-round
recommendation assets.  It reconstructs recommendation SFT rows from the
official raw UserProfile fields according to the contest write-up:

- multi-domain history in every prompt
- target domain last
- video immediately before the target when target is not video
- ad before video when target is not ad/video
- official field families for video/ad/product/live histories

The default assembly replaces old video-heavy recommendation rows in the
current LoRA base with the same number of official_rec rows, keeping total
dataset size fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
HF = ROOT / "data" / "hf_full" / "data"

SID_DOMAIN = {
    "video/video": "video",
    "video/ad": "ad",
    "goods": "prod",
    "live": "living",
}

CN = {
    "video": "视频",
    "prod": "商品",
    "ad": "广告",
    "living": "直播",
}

TARGET_ASK = {
    "video": "请推断用户接下来会点击的视频。",
    "prod": "请推断用户接下来会点击的商品。",
    "ad": "请推断用户接下来会点击的广告。",
    "living": "请推断用户接下来会观看或关注的直播主播。",
}

ORDER = {
    "prod": ["living", "ad", "video", "prod"],
    "living": ["prod", "ad", "video", "living"],
    "video": ["living", "prod", "ad", "video"],
    "ad": ["living", "prod", "video", "ad"],
}

REC_ROW = re.compile(r"<\|(video|prod|living|ad)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


@dataclass(frozen=True)
class Event:
    dom: str
    token: str
    ts: int
    kind: str
    strength: int


def as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return list(v)


def flag(v: Any) -> bool:
    try:
        return float(v) > 0
    except Exception:
        return False


def to_int_ts(v: Any, fallback: int = 0) -> int:
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return int(digits[:14].ljust(14, "0"))
    return fallback


def item_token(dom: str, sid: Any) -> str:
    a, b, c = int(sid[0]), int(sid[1]), int(sid[2])
    return f"<|{dom}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def latest_unique(events: list[Event], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ev in sorted(events, key=lambda x: x.ts, reverse=True):
        if ev.token in seen:
            continue
        seen.add(ev.token)
        out.append(ev.token)
        if len(out) >= limit:
            break
    return list(reversed(out))


def render_tokens(tokens: list[str]) -> str:
    return ", ".join(tokens)


def build_sid_maps(sid_dir: Path) -> dict[str, dict[int, Any]]:
    import pyarrow.parquet as pq

    maps: dict[str, dict[int, Any]] = {d: {} for d in CN}
    files = sorted(sid_dir.glob("*.parquet"))
    for i, fp in enumerate(files, 1):
        table = pq.read_table(fp, columns=["pid", "domain", "sid_three"])
        pids = table["pid"].to_pylist()
        domains = table["domain"].to_pylist()
        sids = table["sid_three"].to_pylist()
        for pid, raw_dom, sid in zip(pids, domains, sids):
            dom = SID_DOMAIN.get(raw_dom)
            if dom and sid and len(sid) == 3:
                maps[dom][int(pid)] = sid
        if i % 25 == 0:
            print(f"[sid] loaded {i}/{len(files)} shards", flush=True)
    print("[sid] map sizes:", {k: len(v) for k, v in maps.items()}, flush=True)
    return maps


def sid_lookup(sid_maps: dict[str, dict[int, Any]], dom: str, pid: Any) -> str | None:
    try:
        sid = sid_maps[dom].get(int(pid))
    except Exception:
        return None
    return item_token(dom, sid) if sid is not None else None


def add_event(
    events: dict[str, list[Event]],
    sid_maps: dict[str, dict[int, Any]],
    dom: str,
    pid: Any,
    ts: int,
    kind: str,
    strength: int,
    banned: set[str],
) -> None:
    tok = sid_lookup(sid_maps, dom, pid)
    if not tok or tok in banned:
        return
    events[dom].append(Event(dom=dom, token=tok, ts=ts, kind=kind, strength=strength))


def video_kind(row: dict[str, Any], i: int, prefix: str) -> tuple[str, int]:
    parts: list[str] = []
    strength = 1
    if flag(as_list(row.get(f"{prefix}_neg_feedback_list"))[i]):
        parts.append("负反馈")
        strength = max(strength, 1)
    if flag(as_list(row.get(f"{prefix}_comment_list"))[i]):
        parts.append("评论")
        strength = max(strength, 4)
    if flag(as_list(row.get(f"{prefix}_forward_list"))[i]):
        parts.append("转发")
        strength = max(strength, 4)
    if flag(as_list(row.get(f"{prefix}_collect_list"))[i]):
        parts.append("收藏")
        strength = max(strength, 4)
    if flag(as_list(row.get(f"{prefix}_like_list"))[i]):
        parts.append("点赞")
        strength = max(strength, 3)

    play_done = flag(as_list(row.get(f"{prefix}_play_done_list"))[i])
    wt = as_list(row.get(f"{prefix}_watch_time_list"))[i]
    dur = as_list(row.get(f"{prefix}_duration_list"))[i]
    long_play = play_done
    try:
        long_play = long_play or (float(dur) > 0 and float(wt) / float(dur) > 0.75)
    except Exception:
        pass
    if long_play:
        parts.append("长播")
        strength = max(strength, 3)
    return ("/".join(parts) if parts else "浏览", strength)


def collect_video_events(
    row: dict[str, Any],
    sid_maps: dict[str, dict[int, Any]],
    events: dict[str, list[Event]],
    banned: set[str],
) -> None:
    pids = as_list(row.get("video_history_sampled_pid_list"))
    ts = as_list(row.get("video_history_ts_list"))
    n = min(len(pids), len(ts))
    need_cols = [
        "video_history_neg_feedback_list",
        "video_history_like_list",
        "video_history_comment_list",
        "video_history_forward_list",
        "video_history_collect_list",
        "video_history_watch_time_list",
        "video_history_play_done_list",
        "video_history_duration_list",
    ]
    if any(len(as_list(row.get(c))) < n for c in need_cols):
        n = min([n] + [len(as_list(row.get(c))) for c in need_cols])
    for i in range(n):
        kind, strength = video_kind(row, i, "video_history")
        add_event(events, sid_maps, "video", pids[i], to_int_ts(ts[i], i), kind, strength, banned)


def collect_ad_events(
    row: dict[str, Any],
    sid_maps: dict[str, dict[int, Any]],
    events: dict[str, list[Event]],
    banned: set[str],
) -> None:
    for col, ts_col, kind, strength in [
        ("outer_loop_history_action_pid_list_pos", "outer_loop_history_action_pid_list_pos_ts", "深度转化", 4),
        ("outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts", "点击", 2),
    ]:
        pids = as_list(row.get(col))
        ts = as_list(row.get(ts_col))
        n = min(len(pids), len(ts)) if ts else len(pids)
        for i in range(n):
            add_event(events, sid_maps, "ad", pids[i], to_int_ts(ts[i] if ts else i, i), kind, strength, banned)


def ec_ts(row: dict[str, Any], lag: Any, fallback: int) -> int:
    snap = row.get("ec_time_ms")
    try:
        # README calls these hour-level lags; ordering is what matters here.
        return int(snap) - int(float(lag)) * 3600 * 1000
    except Exception:
        return fallback


def collect_prod_events(
    row: dict[str, Any],
    sid_maps: dict[str, dict[int, Any]],
    events: dict[str, list[Event]],
    banned: set[str],
) -> None:
    clicks = as_list(row.get("ec_good_click_item_id_list_extend"))
    clk_lags = as_list(row.get("ec_trunc_clk_lag"))
    for i, pid in enumerate(clicks):
        lag = clk_lags[i] if i < len(clk_lags) else i
        add_event(events, sid_maps, "prod", pid, ec_ts(row, lag, i), "浏览/点击", 2, banned)

    orders = as_list(row.get("ec_good_order_item_id_list_extend"))
    buy_lags = as_list(row.get("ec_trunc_buy_lag"))
    for i, pid in enumerate(orders):
        lag = buy_lags[i] if i < len(buy_lags) else i
        add_event(events, sid_maps, "prod", pid, ec_ts(row, lag, i), "购买", 5, banned)

    col_pids = as_list(row.get("ec_colossus_rs_item_id_list"))
    col_lags = as_list(row.get("ec_colossus_rs_lagv1_list"))
    carts = as_list(row.get("ec_colossus_rs_is_cart_list"))
    buys = as_list(row.get("ec_colossus_rs_is_buy_list"))
    clicks2 = as_list(row.get("ec_colossus_rs_is_click_list"))
    n = min(len(col_pids), len(col_lags)) if col_lags else len(col_pids)
    for i in range(n):
        kind = "曝光"
        strength = 1
        if i < len(clicks2) and flag(clicks2[i]):
            kind = "浏览/点击"
            strength = 2
        if i < len(carts) and flag(carts[i]):
            kind = "加购"
            strength = 4
        if i < len(buys) and flag(buys[i]):
            kind = "购买"
            strength = 5
        if kind != "曝光":
            add_event(events, sid_maps, "prod", col_pids[i], ec_ts(row, col_lags[i] if col_lags else i, i), kind, strength, banned)


def collect_live_events(
    row: dict[str, Any],
    sid_maps: dict[str, dict[int, Any]],
    events: dict[str, list[Event]],
    banned: set[str],
) -> None:
    pids = as_list(row.get("live_hist_author_id_list"))
    ts = as_list(row.get("live_hist_timestamp_list"))
    follows = as_list(row.get("live_hist_follow_author_cnt_list"))
    n = min(len(pids), len(ts)) if ts else len(pids)
    for i in range(n):
        is_follow = i < len(follows) and flag(follows[i])
        add_event(
            events,
            sid_maps,
            "living",
            pids[i],
            to_int_ts(ts[i] if ts else i, i),
            "关注" if is_follow else "观看",
            3 if is_follow else 1,
            banned,
        )


def group_events(events: list[Event], limit_by_kind: int) -> dict[str, list[str]]:
    by_kind: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_kind[ev.kind].append(ev)
    return {k: latest_unique(v, limit_by_kind) for k, v in by_kind.items()}


def render_block(dom: str, events: list[Event]) -> str:
    if not events:
        return ""
    grouped = group_events(events, 50 if dom in {"video", "prod", "living"} else 70)
    lines: list[str] = []
    if dom == "video":
        label_map = {
            "浏览": "看过的视频有",
            "长播": "深度观看过的视频有",
            "负反馈": "有过负反馈的视频有",
        }
        for kind in sorted(grouped, key=lambda k: (0 if "长播" in k else 1, k)):
            label = label_map.get(kind, f"有过{kind}行为的视频有")
            lines.append(f"{label} {render_tokens(grouped[kind])}")
        return "用户在视频域:\n" + "；\n".join(lines) + "。"
    if dom == "ad":
        if "深度转化" in grouped:
            lines.append(f"完成过深度转化的广告有 {render_tokens(grouped['深度转化'])}")
        if "点击" in grouped:
            lines.append(f"点击过的广告有 {render_tokens(grouped['点击'])}")
        return "用户在广告域: " + "；".join(lines) + "。"
    if dom == "prod":
        if "浏览/点击" in grouped:
            lines.append(f"浏览/点击过的商品有 {render_tokens(grouped['浏览/点击'])}")
        if "加购" in grouped:
            lines.append(f"加购过的商品有 {render_tokens(grouped['加购'])}")
        if "购买" in grouped:
            lines.append(f"购买过的商品有 {render_tokens(grouped['购买'])}")
        return "用户在电商域: " + "；".join(lines) + "。"
    if dom == "living":
        if "观看" in grouped:
            lines.append(f"观看过主播 {render_tokens(grouped['观看'])}")
        if "关注" in grouped:
            lines.append(f"关注了主播 {render_tokens(grouped['关注'])}")
        return "用户在直播域: " + "；".join(lines) + "。"
    raise ValueError(dom)


def build_prompt(target_dom: str, events: dict[str, list[Event]]) -> str | None:
    blocks = []
    for dom in ORDER[target_dom]:
        block = render_block(dom, events.get(dom, []))
        if block:
            blocks.append(block)
    if len(blocks) < 2:
        return None
    return "用户多域历史行为：\n" + "\n".join(blocks) + "\n" + TARGET_ASK[target_dom] + "/no_think"


def choose_latest_token(
    sid_maps: dict[str, dict[int, Any]],
    dom: str,
    candidates: list[tuple[Any, int, int]],
    history_tokens: set[str],
) -> str | None:
    # candidates: pid, ts, strength
    for pid, _ts, _strength in sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True):
        tok = sid_lookup(sid_maps, dom, pid)
        if tok and tok not in history_tokens:
            return tok
    return None


def video_target(row: dict[str, Any], sid_maps: dict[str, dict[int, Any]], history_tokens: set[str]) -> str | None:
    pids = as_list(row.get("video_sampled_pid_list"))
    ts = as_list(row.get("video_ts_list"))
    n = min(len(pids), len(ts)) if ts else len(pids)
    cols = [
        "video_neg_feedback_list",
        "video_like_list",
        "video_comment_list",
        "video_forward_list",
        "video_collect_list",
        "video_watch_time_list",
        "video_play_done_list",
        "video_duration_list",
    ]
    if any(len(as_list(row.get(c))) < n for c in cols):
        n = min([n] + [len(as_list(row.get(c))) for c in cols])
    cand: list[tuple[Any, int, int]] = []
    for i in range(n):
        kind, strength = video_kind(row, i, "video")
        if kind == "负反馈":
            continue
        cand.append((pids[i], to_int_ts(ts[i] if ts else i, i), strength))
    return choose_latest_token(sid_maps, "video", cand, history_tokens)


def ad_target(row: dict[str, Any], sid_maps: dict[str, dict[int, Any]], history_tokens: set[str]) -> str | None:
    cand: list[tuple[Any, int, int]] = []
    for pid, ts in zip(as_list(row.get("outer_loop_deep_target_pid")), as_list(row.get("outer_loop_deep_target_pid_ts"))):
        cand.append((pid, to_int_ts(ts, 0), 6))
    for pid, ts in zip(as_list(row.get("outer_loop_history_action_pid_list_pos")), as_list(row.get("outer_loop_history_action_pid_list_pos_ts"))):
        cand.append((pid, to_int_ts(ts, 0), 4))
    for pid, ts in zip(as_list(row.get("outer_loop_history_action_pid_list_click")), as_list(row.get("outer_loop_history_action_pid_list_click_ts"))):
        cand.append((pid, to_int_ts(ts, 0), 2))
    return choose_latest_token(sid_maps, "ad", cand, history_tokens)


def prod_target(row: dict[str, Any], sid_maps: dict[str, dict[int, Any]], history_tokens: set[str]) -> str | None:
    cand: list[tuple[Any, int, int]] = []
    for i, pid in enumerate(as_list(row.get("ec_good_order_item_id_list_extend"))):
        lags = as_list(row.get("ec_trunc_buy_lag"))
        cand.append((pid, ec_ts(row, lags[i] if i < len(lags) else i, i), 5))
    pids = as_list(row.get("ec_colossus_rs_item_id_list"))
    lags = as_list(row.get("ec_colossus_rs_lagv1_list"))
    carts = as_list(row.get("ec_colossus_rs_is_cart_list"))
    buys = as_list(row.get("ec_colossus_rs_is_buy_list"))
    clicks = as_list(row.get("ec_colossus_rs_is_click_list"))
    for i, pid in enumerate(pids):
        strength = 0
        if i < len(clicks) and flag(clicks[i]):
            strength = max(strength, 2)
        if i < len(carts) and flag(carts[i]):
            strength = max(strength, 4)
        if i < len(buys) and flag(buys[i]):
            strength = max(strength, 5)
        if strength:
            cand.append((pid, ec_ts(row, lags[i] if i < len(lags) else i, i), strength))
    for i, pid in enumerate(as_list(row.get("ec_good_click_item_id_list_extend"))):
        lags = as_list(row.get("ec_trunc_clk_lag"))
        cand.append((pid, ec_ts(row, lags[i] if i < len(lags) else i, i), 2))
    return choose_latest_token(sid_maps, "prod", cand, history_tokens)


def living_target(row: dict[str, Any], sid_maps: dict[str, dict[int, Any]], history_tokens: set[str]) -> str | None:
    cand: list[tuple[Any, int, int]] = []
    pids = as_list(row.get("live_hist_author_id_list"))
    ts = as_list(row.get("live_hist_timestamp_list"))
    follows = as_list(row.get("live_hist_follow_author_cnt_list"))
    n = min(len(pids), len(ts)) if ts else len(pids)
    for i in range(n):
        strength = 3 if i < len(follows) and flag(follows[i]) else 1
        cand.append((pids[i], to_int_ts(ts[i] if ts else i, i), strength))
    return choose_latest_token(sid_maps, "living", cand, history_tokens)


TARGET_FN = {
    "video": video_target,
    "ad": ad_target,
    "prod": prod_target,
    "living": living_target,
}


def build_events_for_row(
    row: dict[str, Any],
    sid_maps: dict[str, dict[int, Any]],
    banned: set[str],
) -> dict[str, list[Event]]:
    events: dict[str, list[Event]] = {d: [] for d in CN}
    collect_video_events(row, sid_maps, events, banned)
    collect_ad_events(row, sid_maps, events, banned)
    collect_prod_events(row, sid_maps, events, banned)
    collect_live_events(row, sid_maps, events, banned)
    return events


def make_row(target_dom: str, prompt: str, gold: str) -> dict[str, Any]:
    system = f"你是一个推荐系统助手，擅长根据用户属性与多域历史行为预测用户的{CN[target_dom]}偏好。"
    return {
        "instruction": system,
        "input": prompt,
        "output": f"<think>\n</think>\n{gold}",
        "history": [],
        "meta_source": "official_userprofile_rec_v3",
        "meta_target_domain": target_dom,
    }


def build_block(args: argparse.Namespace) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    sid_maps = build_sid_maps(Path(args.sid_dir))
    quotas = {"video": args.n_video, "prod": args.n_prod, "ad": args.n_ad, "living": args.n_living}
    buckets: dict[str, list[dict[str, Any]]] = {d: [] for d in CN}
    reject = Counter()

    all_cols = [
        "ec_time_ms",
        "video_sampled_pid_list",
        "video_neg_feedback_list",
        "video_like_list",
        "video_comment_list",
        "video_forward_list",
        "video_collect_list",
        "video_watch_time_list",
        "video_play_done_list",
        "video_duration_list",
        "video_ts_list",
        "video_history_sampled_pid_list",
        "video_history_neg_feedback_list",
        "video_history_like_list",
        "video_history_comment_list",
        "video_history_forward_list",
        "video_history_collect_list",
        "video_history_watch_time_list",
        "video_history_play_done_list",
        "video_history_duration_list",
        "video_history_ts_list",
        "outer_loop_history_action_pid_list_pos",
        "outer_loop_history_action_pid_list_pos_ts",
        "outer_loop_history_action_pid_list_click",
        "outer_loop_history_action_pid_list_click_ts",
        "outer_loop_deep_target_pid",
        "outer_loop_deep_target_pid_ts",
        "ec_good_click_item_id_list_extend",
        "ec_trunc_clk_lag",
        "ec_good_order_item_id_list_extend",
        "ec_trunc_buy_lag",
        "ec_colossus_rs_item_id_list",
        "ec_colossus_rs_lagv1_list",
        "ec_colossus_rs_is_click_list",
        "ec_colossus_rs_is_cart_list",
        "ec_colossus_rs_is_buy_list",
        "live_hist_author_id_list",
        "live_hist_timestamp_list",
        "live_hist_follow_author_cnt_list",
    ]

    for fp in sorted(Path(args.up_dir).glob("*.parquet"))[: args.shards]:
        schema_cols = set(pq.ParquetFile(fp).schema_arrow.names)
        cols = [c for c in all_cols if c in schema_cols]
        print(f"[user] {fp.name}: reading {len(cols)} cols", flush=True)
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
            for row in batch.to_pylist():
                if all(len(buckets[d]) >= quotas[d] for d in quotas):
                    break
                for target_dom in quotas:
                    if len(buckets[target_dom]) >= quotas[target_dom]:
                        continue
                    # For product/live/ad fallback labels the target is a
                    # leave-one-out item from the same official history field.
                    # Select it first, then remove that exact token from the
                    # rendered target-domain history below.
                    gold = TARGET_FN[target_dom](row, sid_maps, set())
                    if not gold:
                        reject[f"{target_dom}:no_gold"] += 1
                        continue
                    events = build_events_for_row(row, sid_maps, banned={gold})
                    if len(events[target_dom]) < args.min_target_history:
                        reject[f"{target_dom}:short_target_hist"] += 1
                        continue
                    prompt = build_prompt(target_dom, events)
                    if not prompt:
                        reject[f"{target_dom}:short_multidomain"] += 1
                        continue
                    if gold in prompt:
                        reject[f"{target_dom}:leak"] += 1
                        continue
                    if len(prompt) > args.max_prompt_chars:
                        reject[f"{target_dom}:too_long"] += 1
                        continue
                    buckets[target_dom].append(make_row(target_dom, prompt, gold))
            if all(len(buckets[d]) >= quotas[d] for d in quotas):
                break
        print("[user] bucket sizes:", {k: len(v) for k, v in buckets.items()}, flush=True)
        if all(len(buckets[d]) >= quotas[d] for d in quotas):
            break

    rng = random.Random(args.seed)
    out: list[dict[str, Any]] = []
    for d, q in quotas.items():
        rng.shuffle(buckets[d])
        out.extend(buckets[d][:q])
    rng.shuffle(out)
    print("[QC] official_rec_v3 block domains:", Counter(r["meta_target_domain"] for r in out))
    print("[QC] rejects:", reject.most_common(20))
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if isinstance(r, list):
            r = r[0]
        if "prompt" in r and "instruction" not in r:
            r = {"instruction": r.get("system", ""), "input": r["prompt"], "output": r["response"], "history": []}
        r.setdefault("history", [])
        rows.append(r)
    return rows


def final_domain(row: dict[str, Any]) -> str | None:
    body = row.get("output", "").split("</think>")[-1]
    m = REC_ROW.search(body)
    return m.group(1) if m else None


def is_rec_like(row: dict[str, Any]) -> bool:
    text = row.get("instruction", "") + "\n" + row.get("input", "")
    return final_domain(row) is not None and "用户" in text and ("历史" in text or "行为" in text)


def strip_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in ("instruction", "input", "output", "history") if k in row}


def assemble_dataset(args: argparse.Namespace, block: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = load_rows(Path(args.base))
    rec_by_dom: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(base):
        if is_rec_like(row):
            rec_by_dom[final_domain(row)].append(i)

    rng = random.Random(args.seed)
    remove: set[int] = set()
    need = len(block)
    if args.remove_policy == "video_first":
        order = ["video", "prod", "ad", "living"]
        for dom in order:
            idxs = list(rec_by_dom.get(dom, []))
            rng.shuffle(idxs)
            take = min(len(idxs), need - len(remove))
            remove.update(idxs[:take])
            if len(remove) >= need:
                break
    elif args.remove_policy == "same_domain":
        block_counts = Counter(r["meta_target_domain"] for r in block)
        for dom, q in block_counts.items():
            idxs = list(rec_by_dom.get(dom, []))
            rng.shuffle(idxs)
            remove.update(idxs[: min(q, len(idxs))])
        if len(remove) < need:
            rest = [i for v in rec_by_dom.values() for i in v if i not in remove]
            rng.shuffle(rest)
            remove.update(rest[: need - len(remove)])
    else:
        raise ValueError(args.remove_policy)

    if len(remove) != need:
        raise RuntimeError(f"could only remove {len(remove)} rec rows for {need} new rows")

    kept = [row for i, row in enumerate(base) if i not in remove]
    out = [strip_meta(r) for r in kept] + [strip_meta(r) for r in block]
    rng.shuffle(out)
    print("[QC] base rows:", len(base), "removed:", len(remove), "added:", len(block), "out:", len(out))
    print("[QC] removed rec domains:", Counter(final_domain(base[i]) for i in remove))
    print("[QC] base rec domains before:", {k: len(v) for k, v in rec_by_dom.items()})
    print("[QC] out rec domains after:", Counter(final_domain(r) for r in out if is_rec_like(r)))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def register_dataset(name: str, path: Path) -> None:
    info_path = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/data/dataset_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[name] = {
        "file_name": str(path),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] registered {name} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--up_dir", default=str(HF / "OneReason_UserProfile"))
    ap.add_argument("--sid_dir", default=str(HF / "OneReason_Pid2Sid"))
    ap.add_argument("--base", default=str(P / "data_riders_fk.jsonl"))
    ap.add_argument("--block_out", default=str(P / "official_rec_v3_block.jsonl"))
    ap.add_argument("--out", default=str(P / "data_official_rec_v3.jsonl"))
    ap.add_argument("--dataset_name", default="data_official_rec_v3")
    ap.add_argument("--n_video", type=int, default=2500)
    ap.add_argument("--n_prod", type=int, default=2000)
    ap.add_argument("--n_ad", type=int, default=1750)
    ap.add_argument("--n_living", type=int, default=1750)
    ap.add_argument("--min_target_history", type=int, default=8)
    ap.add_argument("--max_prompt_chars", type=int, default=28000)
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--remove_policy", choices=["video_first", "same_domain"], default="video_first")
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()

    block_path = Path(args.block_out)
    if block_path.exists() and block_path.stat().st_size > 0:
        block = load_rows(block_path)
        print(f"[load] existing block {block_path}: {len(block)} rows")
    else:
        block = build_block(args)
        write_jsonl(block_path, [strip_meta(r) | {"meta_target_domain": r["meta_target_domain"]} for r in block])
        print(f"[OK] wrote block {block_path}: {len(block)} rows")

    out = assemble_dataset(args, block)
    out_path = Path(args.out)
    write_jsonl(out_path, out)
    print(f"[OK] wrote dataset {out_path}: {len(out)} rows")
    if args.register:
        register_dataset(args.dataset_name, out_path)


if __name__ == "__main__":
    main()
