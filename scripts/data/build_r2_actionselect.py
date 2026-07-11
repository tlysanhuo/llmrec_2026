#!/usr/bin/env python3
"""build_r2_actionselect.py — 从原始 UserProfile 构造「懂用户 R2 action-select」SFT 数据。

任务(抽取式):给定按日期排列的多域交互历史 + 一个主题,从历史中"提取所有相关的历史交互",
输出一个 JSON 数组(元素为 itemic token,100% 来自历史)。格式逐字对齐官方评测。

核心不变式:
  - 输出 token 全部 ∈ 历史(extractive,构造后强校验)
  - 显式覆盖三类难度:少量正样本 / 多个正样本 / 无相关(输出 [])
  - assistant = "<think>\n\n</think>\n" + JSON 数组 + 无多余文本

用法:
  python scripts/data/build_r2_actionselect.py --n_users 20000 --out <path> --seed 2026
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import pandas as pd

HF = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/hf_full/data"
IDX = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/index"

VIDEO_ORDER = ["关注", "转发", "评论", "收藏", "点赞"]  # + 长播/浏览 结尾

FEWSHOT = (
    '输出示例为（注意：以下案例来自其他用户，仅供参考输出格式，与上述用户交互历史无关）： '
    '["<|prod_begin|><s_a_750><s_b_2525><s_c_3393>", '
    '"<|living_begin|><s_a_6354><s_b_6678><s_c_4429>", '
    '"<|ad_begin|><s_a_7852><s_b_3625><s_c_5951>", '
    '"<|video_begin|><s_a_528><s_b_1682><s_c_7986>"]'
)


def load_index():
    sid = pd.read_parquet(f"{IDX}/pid2sid.parquet")
    sid_map = {k: (a, b, c) for k, a, b, c in zip(sid.key, sid.s_a, sid.s_b, sid.s_c)}
    tag = pd.read_parquet(f"{IDX}/pid2tag.parquet")
    tag_map = dict(zip(tag.key, tag.tag_lv3))
    cap = pd.read_parquet(f"{IDX}/pid2caption.parquet")
    cap_map = dict(zip(cap.key, cap.caption))
    return sid_map, tag_map, cap_map


def tok(domain, sid):
    a, b, c = sid
    return f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def ts_to_date(ms):
    # 毫秒时间戳 -> YYYY-MM-DD (UTC, 不依赖 Date.now)
    import datetime
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def video_label(flags):
    lab = [n for n, f in zip(VIDEO_ORDER, flags[:5]) if f]  # like/comment/... flags
    played = flags[5]
    lab.append("长播" if played else "浏览")
    return "视频-" + "/".join(lab)


def collect_events(row, sid_map, tag_map, cap_map):
    """把一个用户的多域行为汇成事件列表: [{date, ts, domain, action, token, key, tag, cap}]"""
    ev = []

    def add(domain, pid, ts, action):
        if pid is None:
            return
        key = f"{domain}|{int(pid)}"
        sid = sid_map.get(key)
        if sid is None:
            return  # pid 无 sid 映射,跳过(不硬造)
        ev.append({
            "ts": int(ts), "date": ts_to_date(ts), "domain": domain,
            "action": action, "token": tok(domain, sid), "key": key,
            "tag": tag_map.get(key, ""), "cap": cap_map.get(key, ""),
        })

    def L(v):
        try:
            return len(v) if v is not None and hasattr(v, "__len__") and not isinstance(v, str) else 0
        except Exception:
            return 0

    # video (history) — 只取最近 400 条(收尾 window 会再截到 ≤260),避免处理上千事件
    vp = row.get("video_history_sampled_pid_list")
    if L(vp):
        VCAP = 400
        off = max(0, L(vp) - VCAP)
        ts = row.get("video_history_ts_list")
        lk = row.get("video_history_like_list"); cm = row.get("video_history_comment_list")
        fw = row.get("video_history_forward_list"); cl = row.get("video_history_collect_list")
        pd_ = row.get("video_history_play_done_list")
        for i in range(off, L(vp)):
            pid = vp[i]
            def g(a, i):
                return bool(a[i]) if L(a) > i else False
            flags = [g(lk, i), g(fw, i), g(cm, i), g(cl, i), False, g(pd_, i)]
            # 顺序对齐 VIDEO_ORDER = 关注,转发,评论,收藏,点赞 ; 点赞暂无单列 -> 用 like 位放到"点赞"
            flags = [False, g(fw, i), g(cm, i), g(cl, i), g(lk, i), g(pd_, i)]
            t = ts[i] if L(ts) > i else 0
            if t:
                add("video", pid, t, video_label(flags))
    # ecom: order=购买, click=点击
    for col, act in [("ec_good_order_item_id_list_extend", "商品-购买"),
                     ("ec_good_click_item_id_list_extend", "商品-点击")]:
        lst = row.get(col)
        if L(lst):
            # 时间列不总对齐,用 ec_time_ms 兜底
            bv = row.get("ec_time_ms")
            try:
                base = int(bv) if bv is not None and bv == bv else 0  # nan check
            except Exception:
                base = 0
            for pid in lst:
                add("prod", pid, base if base else 1, act)
    # live: 关注 — 只取最近 300 条(直播历史可上万)
    la = row.get("live_hist_author_id_list")
    if L(la):
        lts = row.get("live_hist_timestamp_list")
        loff = max(0, L(la) - 300)
        for i in range(loff, L(la)):
            pid = la[i]
            t = 1
            if L(lts) > i and lts[i]:
                try:
                    import datetime
                    t = int(datetime.datetime.strptime(str(lts[i])[:10], "%Y-%m-%d").timestamp() * 1000)
                except Exception:
                    t = 1
            add("living", pid, t, "直播-关注")
    # ad: click_type
    ac = row.get("outer_loop_history_action_pid_list_click")
    if L(ac):
        ats = row.get("outer_loop_history_action_pid_list_click_ts")
        for i, pid in enumerate(ac):
            t = ats[i] if L(ats) > i else 1
            add("ad", pid, int(t) if t else 1, "广告-点击")
    return ev


def build_history_block(events):
    """按 (date) 分组渲染成官方历史块文本。events 已按 ts 排序。"""
    lines = ["【用户交互历史】："]
    cur = None
    for e in events:
        if e["date"] != cur:
            cur = e["date"]
            lines.append(f"【{cur}】")
        lines.append(f"  --:-- [{e['action']}] {e['token']}")
    return "\n".join(lines)


def build_history_annot(events):
    """caption/tag 注释版历史(仿官方构造样例:文字在前 token 在后),仅供 teacher 标注用,不进训练题面。"""
    lines = ["【用户交互历史】："]
    cur = None
    for e in events:
        if e["date"] != cur:
            cur = e["date"]
            lines.append(f"【{cur}】")
        desc = (e.get("cap") or e.get("tag") or "").replace("\n", " ")[:60]
        lines.append(f"  --:-- [{e['action']}] {desc}{e['token']}")
    return "\n".join(lines)


def tag_tokens(tag):
    return set(re.split(r"[-/、\s]+", tag)) if tag else set()


def score(cand, focus):
    """确定性相关性打分,对齐官方分布(答案中位~11)。
    以 tag 语义重叠为主,same s_a/s_b(粗类目接近)为辅,弱化纯 same_domain(否则同域全中)。"""
    ft = tag_tokens(focus["tag"]); ct = tag_tokens(cand["tag"])
    jac = len(ft & ct) / len(ft | ct) if (ft or ct) else 0.0
    # s_a/s_b 相同(粗类目接近)
    fm = re.search(r"s_a_(\d+)><s_b_(\d+)", focus["token"])
    cm = re.search(r"s_a_(\d+)><s_b_(\d+)", cand["token"])
    same_ab = 0.0
    if fm and cm:
        if fm.group(1) == cm.group(1):
            same_ab = 0.6 + (0.4 if fm.group(2) == cm.group(2) else 0.0)
    same_dom = 0.10 if cand["domain"] == focus["domain"] else 0.0
    return 0.60 * jac + 0.30 * same_ab + same_dom


def make_topic(focus, tag_map):
    """由 focus item 的 tag 生成主题串,风格对齐官方『从泛化X到聚焦Y』。"""
    t = focus["tag"]
    parts = [p for p in re.split(r"[-]", t) if p]
    if len(parts) >= 2:
        return f"从泛化{parts[0]}到聚焦{parts[-1]}的需求演化"
    if parts:
        return f"聚焦{parts[0]}的兴趣深化"
    return "特定兴趣的聚焦与演化"


def make_instruction(topic):
    return (
        "角色任务：你是一个极端严苛的用户行为数据挖掘与数据格式化专家。请基于以上用户交互"
        "历史，围绕给定主题，提取出所有相关的历史交互。\n"
        f"主题：{topic}\n"
        "输出格式要求：请直接输出JSON数组，不要输出任何额外解释。\n"
        + FEWSHOT + "/no_think"
    )


def build_sample(events, rng, tag_map):
    """从一个用户的事件列表构造一条 action-select 样本。返回 (record, meta) 或 None。"""
    # 历史窗口: 对齐种子分布(事件中位~191)。过长则取最近窗口(尾部),窗口大小随机 80..260。
    if len(events) > 260:
        win = rng.randint(80, 260)
        events = events[-win:]
    # 只保留有 tag 的事件作为可打分候选(video/ad/live 有 tag)
    scored_pool = [e for e in events if e["tag"]]
    if len(events) < 8 or len(scored_pool) < 3:
        return None
    focus = rng.choice(scored_pool)
    topic = make_topic(focus, tag_map)
    n = len(events)
    scores = []
    for i, e in enumerate(events):
        s = score(e, focus)
        rec = 0.05 * (i / max(n - 1, 1))  # recency: 越靠后越近(弱)
        scores.append((s + rec, i))
    thr = 0.45
    cand = [(s, i) for s, i in scores if s >= thr and events[i]["token"] != focus["token"]]
    cand.sort(reverse=True)
    # 上限对齐种子(max~56,中位~11):取 top-K, K 随机 3..40
    kcap = rng.randint(3, 40)
    cand = cand[:kcap]
    # dedup tokens (保序,按分数高->低)
    seen = set(); pos = []
    for _, i in cand:
        t = events[i]["token"]
        if t not in seen:
            seen.add(t); pos.append(t)
    hist_tokens = set(e["token"] for e in events)
    # extractive 强校验
    assert all(p in hist_tokens for p in pos), "positive not in history!"
    ans = json.dumps(pos, ensure_ascii=False)
    instr = make_instruction(topic)
    hist = build_history_block(events)
    record = {
        "instruction": "",
        "input": hist + "\n\n" + instr,
        "output": "<think>\n\n</think>\n" + ans,
        "history": [],
        "_hist_annot": build_history_annot(events),
    }
    return record, {"n_hist": n, "n_pos": len(pos), "topic": topic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--empty_ratio", type=float, default=0.12, help="强制无相关(空[])样本比例")
    ap.add_argument("--shard_offset", type=int, default=0, help="从第 N 个 UserProfile 分片开始扫(dev 用,避免与 train 用户重叠)")
    ap.add_argument("--max_users_scan", type=int, default=0, help="最多扫描多少用户(0=按需)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("[load] index...", file=sys.stderr)
    sid_map, tag_map, cap_map = load_index()
    print(f"[load] sid {len(sid_map):,} tag {len(tag_map):,} cap {len(cap_map):,}", file=sys.stderr)

    shards = sorted(glob.glob(f"{HF}/OneReason_UserProfile/*.parquet"))
    shards = shards[args.shard_offset:]  # dev 从后段分片起,避免与 train 用户重叠
    # 只读构造需要的列(大幅提速: 避免 63 列 Series 构造)
    NEEDED = [
        "video_history_sampled_pid_list", "video_history_ts_list",
        "video_history_like_list", "video_history_comment_list",
        "video_history_forward_list", "video_history_collect_list",
        "video_history_play_done_list",
        "ec_good_order_item_id_list_extend", "ec_good_click_item_id_list_extend", "ec_time_ms",
        "live_hist_author_id_list", "live_hist_timestamp_list",
        "outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts",
    ]
    out_recs = []
    stats = {"n_pos_dist": [], "n_hist": [], "empty": 0, "skip_short": 0, "extractive_ok": 0}
    scanned = 0
    for sh in shards:
        df = pd.read_parquet(sh, columns=[c for c in NEEDED if c])
        cols = {c: df[c].to_numpy() for c in df.columns}
        nrows = len(df)
        del df
        for i in range(nrows):
            if len(out_recs) >= args.n_users:
                break
            scanned += 1
            row = {c: cols[c][i] for c in cols}  # 轻量 dict,不构造 Series
            ev = collect_events(row, sid_map, tag_map, cap_map)
            ev.sort(key=lambda e: e["ts"])
            if len(ev) < 8:
                stats["skip_short"] += 1
                continue
            res = build_sample(ev, rng, tag_map)
            if res is None:
                stats["skip_short"] += 1
                continue
            rec, meta = res
            # 强制一部分空样本(把 output 改成 [])
            if rng.random() < args.empty_ratio:
                rec["output"] = "<think>\n\n</think>\n[]"
                stats["empty"] += 1
            else:
                if meta["n_pos"] == 0:
                    continue  # 非空样本要求至少 1 正样本
            stats["n_pos_dist"].append(meta["n_pos"])
            stats["n_hist"].append(meta["n_hist"])
            stats["extractive_ok"] += 1
            out_recs.append(rec)
        print(f"  [scan] shard done, total {len(out_recs)}/{args.n_users}", file=sys.stderr)
        if len(out_recs) >= args.n_users:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics as st
    npos = stats["n_pos_dist"]
    print(f"[done] wrote {len(out_recs)} samples -> {args.out}", file=sys.stderr)
    print(f"  scanned users: {scanned}, skip_short: {stats['skip_short']}, empty([]): {stats['empty']}", file=sys.stderr)
    if npos:
        print(f"  n_pos: min{min(npos)} median{int(st.median(npos))} max{max(npos)} mean{st.mean(npos):.1f}", file=sys.stderr)
        print(f"  n_hist: median{int(st.median(stats['n_hist']))}", file=sys.stderr)


if __name__ == "__main__":
    main()
