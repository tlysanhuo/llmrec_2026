#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_p3_quote_stop.py — P3「SID 原子复制 + 停止训练(quote-and-stop)」数据集。

════════════════════════════════════════════════════════════════════════
设计依据(docs/eval_log_mining_20260705.md §D / §2-P3)
────────────────────────────────────────────────────────────────────────
对治的评测灾难(9 份评测日志一致):
  D1  action_select 复读循环打满 4096 token(29-34k 字符,重复率>90%)
  D2  去重后引用 SID 仍变异/捏造(rebal_world Q1: 20 条中 19 条捏造)
  C1  topic_gen 引用 SID 变异(三 token 换 1 个子 token),think 引用保真仅 86-92%

任务形态(quote-and-stop):
  输入 = 一段含 20-80 个 SID 的真实用户 timeline(格式逐字仿评测/种子)
         + 一个可程序判定的筛选指令(按域/按行为类型/按时间段/组合,措辞轮换)
  输出 = 满足条件的 SID 子集,逐字引用(含域 begin token,三 token 原子不许变异),
         JSON 数组闭合,答完即停,数组内零重复(gold 按时间线首现序去重)

与种子 action_select(data/懂用户.jsonl,1588 条)的对齐点:
  - instruction="" / input=timeline+"\n\n"+角色任务块 / output='<think>\n</think>\n'+JSON / history=[]
  - timeline 渲染逐字对齐:【用户交互历史】：/【YYYY-MM-DD】/两空格+时刻+[域-行为]+SID
    时刻规则(种子实测):商品、直播 = --:--;视频、广告 = HH:MM
  - 角色任务/输出格式要求措辞取自种子与评测原文并轮换 paraphrase(防背模板)
  - prompt 尾 '/no_think' ⇔ 输出空 think('<think>\n</think>\n'),全库不变量
与种子的差异点(有意为之):
  - 种子的"主题"是语义筛选(需 teacher 判定,gold 无法程序复核);本数据用
    「筛选条件」= 域/行为/日期等程序可判定谓词 → gold 100% 查表生成,零 teacher,
    可全量程序复核。训练目标是①上下文 SID 的原子复制②JSON 闭合即停,
    不是主题理解——主题理解由种子样本自己承担。
  - gold 去重(种子 gold 亦无重复元素):显式反 D1 复读。

数据源(全官方,无泄漏):
  OneReason_UserProfile/*.parquet  用户多域行为(pid+时间+行为标志)
  OneReason_Pid2Sid/*.parquet      pid→sid 三元组;domain 归一化:
                                   video/video→video, video/ad→ad, goods→prod, live→living
  (ec_trunc_clk_lag / ec_trunc_buy_lag = 相对 ec_time_ms 锚点的"天"偏移,用于还原商品事件日期)

预登记预测(训练后):
  - action_select F1 从全员≈0 起跳(唯一趴地子项;复读与捏造均被此数据直接对治)
  - topic_gen / rec think 的历史 SID 引用保真 86-92% → ≈100%(原子复制技能跨任务迁移)
  - 零维度冲突预期:不与 grounding/rec 的"生成新 SID"能力打架(本任务只教"抄现场的")
  - 若无效则证:复制失真源于解码超参而非能力缺口

用法:
  python scripts/data/build_p3_quote_stop.py build          # 训练 3000 + 验证 150
  python scripts/data/build_p3_quote_stop.py verify <file>  # 从 prompt 文本独立复核 gold
seed=2026 固定(val 用 2027 + 独立分片 part-00009,用户零重叠)。
════════════════════════════════════════════════════════════════════════
"""
import argparse
import datetime
import glob
import json
import os
import random
import re
import sys
from collections import Counter

import pandas as pd

ROOT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
HF = f"{ROOT}/data/hf_full/data"
OUT_TRAIN = f"{ROOT}/data/processed/p3_quote_stop.jsonl"
OUT_VAL = f"{ROOT}/data/processed/p3_quote_stop_val.jsonl"

TZ = datetime.timezone(datetime.timedelta(hours=8))  # 北京时区,与源数据一致性自洽即可

DOMAIN_MAP = {"video/video": "video", "video/ad": "ad", "goods": "prod", "live": "living"}
DOM_ZH = {"prod": "商品", "living": "直播", "video": "视频", "ad": "广告"}
ZH_DOM = {v: k for k, v in DOM_ZH.items()}
VIDEO_ORDER = ["关注", "转发", "评论", "收藏", "点赞"]  # 种子实测的组合顺序,尾接 长播|浏览

FEWSHOT = (
    "输出示例为（注意：以下案例来自其他用户，仅供参考输出格式，与上述用户交互历史无关）： "
    '["<|prod_begin|><s_a_750><s_b_2525><s_c_3393>", '
    '"<|living_begin|><s_a_6354><s_b_6678><s_c_4429>", '
    '"<|ad_begin|><s_a_7852><s_b_3625><s_c_5951>", '
    '"<|video_begin|><s_a_528><s_b_1682><s_c_7986>"]'
)

# 角色任务措辞轮换(1/2 取自种子与评测原文骨架;其余 paraphrase,部分显式点出
# 逐字/原样/答完即止 = quote-and-stop 的行为学提示)
ROLES = [
    "角色任务：你是一个极端严苛的用户行为数据挖掘与数据格式化专家。请基于以上用户交互历史，按下方筛选条件提取出所有满足条件的历史交互。",
    "角色任务：你需要以极端严苛的用户行为数据挖掘专家及数据格式化专家的身份，基于以上交互历史，筛选并提取出全部满足下方条件的历史交互行为。",
    "任务：请从上述用户交互历史中，找出满足下方筛选条件的每一条交互，并逐字原样引用其 SID，不得改动其中任何一个 token。",
    "角色任务：你是一位严谨的用户行为数据审计员。请扫描以上交互历史，摘录满足下方筛选条件的交互 SID；引用必须与历史记录逐字一致。",
    "你是一名数据格式化专家。请依据以上用户交互历史，按下方条件筛选交互记录；SID 必须从历史中原样复制，输出完毕立即停止。",
    "角色任务：请仔细核对以上用户交互历史，提取所有满足下方筛选条件的交互 SID。同一 SID 只保留首次出现，答完即止，不要输出任何重复内容。",
]
FORMATS = [
    "输出格式要求：请直接输出JSON数组，不要输出任何额外解释。",  # 评测原文
    "输出格式要求：请仅以包含 SID 的 JSON 数组形式返回结果，切勿输出任何额外的解释说明或无关字符。",  # 种子原文
]

SID_RE = re.compile(r"<\|(\w+)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
EVENT_LINE_RE = re.compile(r"^  (--:--|\d\d:\d\d) \[([^\]]+)\] (<\|\w+_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>)$")
DATE_LINE_RE = re.compile(r"^【(\d{4}-\d{2}-\d{2})】$")


# ──────────────────────────── Pass 1: 用户事件收集 ────────────────────────────

NEEDED_COLS = [
    "ec_good_click_item_id_list_extend", "ec_trunc_clk_lag",
    "ec_good_order_item_id_list_extend", "ec_trunc_buy_lag", "ec_time_ms",
    "video_history_sampled_pid_list", "video_history_ts_list",
    "video_history_like_list", "video_history_comment_list",
    "video_history_forward_list", "video_history_collect_list",
    "video_history_play_done_list", "video_history_watch_time_list",
    "live_hist_author_id_list", "live_hist_timestamp_list",
    "outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts",
    "outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts",
]


def _L(v):
    try:
        return len(v) if v is not None and hasattr(v, "__len__") and not isinstance(v, str) else 0
    except Exception:
        return 0


def _ok_ts(ms):
    return 1640966400000 <= ms <= 1798732800000  # 2022-01-01 .. 2026-12-31


def video_label(like, fwd, cm, col, long_play):
    parts = []
    for name, flag in zip(VIDEO_ORDER, [False, fwd, cm, col, like]):
        if flag:
            parts.append(name)
    parts.append("长播" if long_play else "浏览")
    return "视频-" + "/".join(parts)


def collect_raw_events(row):
    """→ [(dom, pid, ts_ms, action)];ts 用于排序与渲染(HH:MM 仅 video/ad)。"""
    ev = []

    def add(dom, pid, ts, action):
        if pid is not None and ts and _ok_ts(int(ts)):
            ev.append((dom, int(pid), int(ts), action))

    # prod: lag = 相对 anchor 的天数偏移(向过去)
    anchor = row.get("ec_time_ms")
    try:
        anchor = int(anchor) if anchor is not None and anchor == anchor else 0
    except Exception:
        anchor = 0
    if anchor:
        for pid_col, lag_col, act in [
            ("ec_good_click_item_id_list_extend", "ec_trunc_clk_lag", "商品-点击"),
            ("ec_good_order_item_id_list_extend", "ec_trunc_buy_lag", "商品-购买"),
        ]:
            pids, lags = row.get(pid_col), row.get(lag_col)
            n = min(_L(pids), _L(lags))
            for i in range(max(0, n - 250), n):
                add("prod", pids[i], anchor - int(lags[i]) * 86400000, act)
    # video
    vp = row.get("video_history_sampled_pid_list")
    if _L(vp):
        ts = row.get("video_history_ts_list")
        lk, cm = row.get("video_history_like_list"), row.get("video_history_comment_list")
        fw, cl = row.get("video_history_forward_list"), row.get("video_history_collect_list")
        pdn, wt = row.get("video_history_play_done_list"), row.get("video_history_watch_time_list")

        def g(a, i):
            return bool(a[i]) if _L(a) > i else False

        n = _L(vp)
        for i in range(max(0, n - 250), n):
            if _L(ts) <= i or not ts[i]:
                continue
            long_play = g(pdn, i) or (_L(wt) > i and wt[i] is not None and wt[i] >= 30)
            add("video", vp[i], ts[i], video_label(g(lk, i), g(fw, i), g(cm, i), g(cl, i), long_play))
    # living
    la, lts = row.get("live_hist_author_id_list"), row.get("live_hist_timestamp_list")
    if _L(la):
        n = _L(la)
        for i in range(max(0, n - 120), n):
            if _L(lts) <= i or not lts[i]:
                continue
            s = str(lts[i])[:10].replace("-", "")[:8]
            try:
                t = int(datetime.datetime.strptime(s, "%Y%m%d").replace(tzinfo=TZ).timestamp() * 1000)
            except Exception:
                continue
            add("living", la[i], t, "直播-关注")
    # ad
    for pid_col, ts_col, act in [
        ("outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts", "广告-点击"),
        ("outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts", "广告-深度转化"),
    ]:
        pids, ts = row.get(pid_col), row.get(ts_col)
        n = min(_L(pids), _L(ts))
        for i in range(n):
            add("ad", pids[i], ts[i], act)
    ev.sort(key=lambda e: e[2])
    return ev


def scan_users(shards, max_users, min_raw=25):
    users = []
    for sh in shards:
        df = pd.read_parquet(sh, columns=NEEDED_COLS)
        cols = {c: df[c].to_numpy() for c in df.columns}
        nrows = len(df)
        del df
        for i in range(nrows):
            row = {c: cols[c][i] for c in cols}
            ev = collect_raw_events(row)
            if len(ev) >= min_raw:
                users.append(ev)
            if len(users) >= max_users:
                return users
        print(f"  [pass1] {os.path.basename(sh)} done, users={len(users)}", file=sys.stderr)
    return users


# ──────────────────────────── Pass 2: pid→sid join ────────────────────────────

def build_sid_map(needed_keys):
    """needed_keys: set[(dom,pid)] → dict[(dom,pid)] = (a,b,c)"""
    needed_pids = {p for _, p in needed_keys}
    sid_map = {}
    shards = sorted(glob.glob(f"{HF}/OneReason_Pid2Sid/*.parquet"))
    for j, sh in enumerate(shards):
        df = pd.read_parquet(sh, columns=["pid", "domain", "sid_three"])
        df = df[df["pid"].isin(needed_pids) & df["sid_three"].notna()]
        for p, d, s in zip(df["pid"], df["domain"], df["sid_three"]):
            k = (DOMAIN_MAP.get(d, d), int(p))
            if k in needed_keys and k not in sid_map:
                sid_map[k] = (int(s[0]), int(s[1]), int(s[2]))
        if (j + 1) % 40 == 0:
            print(f"  [pass2] {j+1}/{len(shards)} shards, mapped={len(sid_map)}", file=sys.stderr)
    return sid_map


# ──────────────────────────── 渲染与条件系统 ────────────────────────────

def render_events(raw_events, sid_map):
    """→ [{date, hhmm, action, token, dom, ts}](只保留 join 成功的)。"""
    out = []
    for dom, pid, ts, action in raw_events:
        sid = sid_map.get((dom, pid))
        if sid is None:
            continue
        dt = datetime.datetime.fromtimestamp(ts / 1000, TZ)
        hhmm = dt.strftime("%H:%M") if dom in ("video", "ad") else "--:--"
        out.append({
            "date": dt.strftime("%Y-%m-%d"), "hhmm": hhmm, "action": action, "dom": dom,
            "token": f"<|{dom}_begin|><s_a_{sid[0]}><s_b_{sid[1]}><s_c_{sid[2]}>", "ts": ts,
        })
    return out


def render_timeline(events):
    lines = ["【用户交互历史】："]
    cur = None
    for e in events:
        if e["date"] != cur:
            cur = e["date"]
            lines.append(f"【{cur}】")
        lines.append(f"  {e['hhmm']} [{e['action']}] {e['token']}")
    return "\n".join(lines)


# 条件 = (text, predicate)。text 模板与 verify 端 parse_condition 一一对应(机器可复核)。
def cond_domain(dom):
    zh = DOM_ZH[dom]
    return (f"【{zh}】域的全部交互（即所有 [{zh}-*] 行为）", lambda e: e["dom"] == dom)


def cond_action_exact(label):
    return (f"行为类型为 [{label}] 的全部交互", lambda e: e["action"] == label)


def cond_video_contains(word):
    return (f"行为标注中包含「{word}」的全部视频交互",
            lambda e: e["dom"] == "video" and word in e["action"].split("视频-", 1)[-1].split("/"))


def cond_range(d1, d2):
    return (f"发生日期在【{d1}】至【{d2}】之间（含两端）的全部交互", lambda e: d1 <= e["date"] <= d2)


def cond_month(ym):
    y, m = ym.split("-")
    return (f"发生在 {y}年{m}月 的全部交互", lambda e: e["date"][:7] == ym)


def cond_year(y):
    return (f"发生在 {y}年 的全部交互", lambda e: e["date"][:4] == y)


def cond_dom_month(dom, ym):
    y, m = ym.split("-")
    zh = DOM_ZH[dom]
    return (f"发生在 {y}年{m}月 的全部【{zh}】域交互", lambda e: e["dom"] == dom and e["date"][:7] == ym)


def cond_action_range(label, d1, d2):
    return (f"发生日期在【{d1}】至【{d2}】之间（含两端）、且行为类型为 [{label}] 的全部交互",
            lambda e: d1 <= e["date"] <= d2 and e["action"] == label)


def parse_condition(text):
    """verify 端:由筛选条件文本独立还原谓词(与上面模板一一镜像)。"""
    m = re.fullmatch(r"【([^】]+)】域的全部交互（即所有 \[[^\]]+-\*\] 行为）", text)
    if m:
        return cond_domain(ZH_DOM[m.group(1)])[1]
    m = re.fullmatch(r"行为类型为 \[([^\]]+)\] 的全部交互", text)
    if m:
        return cond_action_exact(m.group(1))[1]
    m = re.fullmatch(r"行为标注中包含「([^」]+)」的全部视频交互", text)
    if m:
        return cond_video_contains(m.group(1))[1]
    m = re.fullmatch(r"发生日期在【([\d-]+)】至【([\d-]+)】之间（含两端）的全部交互", text)
    if m:
        return cond_range(m.group(1), m.group(2))[1]
    m = re.fullmatch(r"发生日期在【([\d-]+)】至【([\d-]+)】之间（含两端）、且行为类型为 \[([^\]]+)\] 的全部交互", text)
    if m:
        return cond_action_range(m.group(3), m.group(1), m.group(2))[1]
    m = re.fullmatch(r"发生在 (\d{4})年(\d{2})月 的全部交互", text)
    if m:
        return cond_month(f"{m.group(1)}-{m.group(2)}")[1]
    m = re.fullmatch(r"发生在 (\d{4})年 的全部交互", text)
    if m:
        return cond_year(m.group(1))[1]
    m = re.fullmatch(r"发生在 (\d{4})年(\d{2})月 的全部【([^】]+)】域交互", text)
    if m:
        return cond_dom_month(ZH_DOM[m.group(3)], f"{m.group(1)}-{m.group(2)}")[1]
    raise ValueError(f"unparsable condition: {text}")


def gold_of(events, pred):
    """首现序去重(quote-and-stop:数组内零重复)。"""
    seen, out = set(), []
    for e in events:
        if pred(e) and e["token"] not in seen:
            seen.add(e["token"])
            out.append(e["token"])
    return out


# ──────────────────────────── 条件调度(分层配额) ────────────────────────────

STRATA = (
    [("dom", d, 225) for d in ("prod", "living", "video", "ad")]
    + [("act", a, q) for a, q in [
        ("商品-点击", 135), ("商品-购买", 135), ("广告-点击", 135), ("直播-关注", 115),
        ("广告-深度转化", 90), ("v含长播", 100), ("v含点赞", 100), ("视频-浏览", 45), ("v含收藏", 45)]]
    + [("time", "range", 350), ("time", "month", 300), ("time", "year", 100)]
    + [("combo", "dom_month", 150), ("combo", "act_range", 150)]
    + [("empty", "empty", 150)]
)
ALL_EXACT_ACTIONS = ["商品-点击", "商品-购买", "广告-点击", "直播-关注", "广告-深度转化"]


def n_match(events, pred):
    return sum(1 for e in events if pred(e))


def try_make(kind, key, events, rng):
    """→ (cond_text, gold) 或 None。非空条件约束:1≤匹配事件≤0.9*len 且 gold≤45。"""
    w = len(events)

    def accept(c):
        text, pred = c
        nm = n_match(events, pred)
        if not (1 <= nm <= max(1, int(0.9 * w))):
            return None
        g = gold_of(events, pred)
        if not (1 <= len(g) <= 45):
            return None
        return text, g

    dates = sorted({e["date"] for e in events})
    months = sorted({d[:7] for d in dates})
    years = sorted({d[:4] for d in dates})
    if kind == "dom":
        return accept(cond_domain(key)) if any(e["dom"] == key for e in events) else None
    if kind == "act":
        if key.startswith("v含"):
            return accept(cond_video_contains(key[2:]))
        return accept(cond_action_exact(key))
    if kind == "time":
        for _ in range(6):
            if key == "range" and len(dates) >= 4:
                i = rng.randrange(0, len(dates) - 1)
                j = rng.randrange(i + 1, min(len(dates), i + 1 + max(2, len(dates) // 3)))
                r = accept(cond_range(dates[i], dates[j]))
            elif key == "month" and months:
                r = accept(cond_month(rng.choice(months)))
            elif key == "year" and len(years) >= 2:
                r = accept(cond_year(rng.choice(years)))
            else:
                return None
            if r:
                return r
        return None
    if kind == "combo":
        for _ in range(6):
            if key == "dom_month" and months:
                r = accept(cond_dom_month(rng.choice([e["dom"] for e in events]), rng.choice(months)))
            elif key == "act_range" and len(dates) >= 4:
                acts = [e["action"] for e in events if not e["action"].startswith("视频")]
                if not acts:
                    return None
                i = rng.randrange(0, len(dates) - 1)
                j = rng.randrange(i + 1, len(dates))
                r = accept(cond_action_range(rng.choice(acts), dates[i], dates[j]))
            else:
                return None
            if r:
                return r
        return None
    if kind == "empty":
        # 可从 prompt 独立证明为空的条件:缺席域 > 缺席精确行为 > 跨度内缺席月份
        absent_dom = [d for d in ("prod", "living", "video", "ad") if all(e["dom"] != d for e in events)]
        if absent_dom:
            return cond_domain(rng.choice(absent_dom))[0], []
        absent_act = [a for a in ALL_EXACT_ACTIONS if all(e["action"] != a for e in events)]
        if absent_act:
            return cond_action_exact(rng.choice(absent_act))[0], []
        lo, hi = months[0], months[-1]
        y, m = int(lo[:4]), int(lo[5:7])
        span = []
        while f"{y:04d}-{m:02d}" <= hi:
            span.append(f"{y:04d}-{m:02d}")
            m += 1
            if m > 12:
                y, m = y + 1, 1
        gap = [x for x in span if x not in months]
        if gap:
            return cond_month(rng.choice(gap))[0], []
        return None
    return None


# ──────────────────────────── 样本组装 ────────────────────────────

def assemble(events, cond_text, gold, rng):
    role = rng.choice(ROLES)
    fmt = rng.choice(FORMATS)
    instr = role + "\n" + f"筛选条件：{cond_text}" + "\n" + fmt
    if rng.random() < 0.4:
        instr += "\n" + FEWSHOT
    instr += "/no_think"
    return {
        "instruction": "",
        "input": render_timeline(events) + "\n\n" + instr,
        "output": "<think>\n</think>\n" + json.dumps(gold, ensure_ascii=False),
        "history": [],
    }


def build_split(user_shards, n_target, seed, out_path):
    rng = random.Random(seed)
    print(f"[pass1] scanning users for {out_path} ...", file=sys.stderr)
    users = scan_users(user_shards, max_users=int(n_target * 8))
    print(f"[pass1] {len(users)} candidate users", file=sys.stderr)

    needed = {(dom, pid) for ev in users for dom, pid, _, _ in ev}
    print(f"[pass2] joining {len(needed):,} keys against Pid2Sid ...", file=sys.stderr)
    sid_map = build_sid_map(needed)
    print(f"[pass2] mapped {len(sid_map):,}/{len(needed):,}", file=sys.stderr)

    quotas = {(k, key): q for k, key, q in STRATA}
    scale = n_target / sum(q for _, _, q in STRATA)
    quotas = {k: max(1, round(v * scale)) for k, v in quotas.items()}
    recs, stats = [], Counter()
    seen_inputs = set()
    for ev_raw in users:
        if len(recs) >= n_target:
            break
        events = render_events(ev_raw, sid_map)
        # video 稀释:防止时间线被近期密集 video 淹没(种子/评测时间线是多域混合的)
        nv = [e for e in events if e["dom"] != "video"]
        vv = [e for e in events if e["dom"] == "video"]
        cap = max(10, 2 * len(nv))
        if len(vv) > cap:
            step = len(vv) / cap
            vv = [vv[int(i * step)] for i in range(cap)]
            events = sorted(nv + vv, key=lambda e: e["ts"])
        if len(events) < 20:
            stats["skip_short_after_join"] += 1
            continue
        w = rng.randint(20, 80)
        start = rng.randint(0, max(0, len(events) - w))
        win = events[start:start + w]
        if len({e["date"] for e in win}) < 3:
            win = events[-80:]  # 日期太集中则扩到最大尾窗
        if len(win) < 20 or len({e["date"] for e in win}) < 2:
            stats["skip_window"] += 1
            continue
        events = win
        made = None
        for (kind, key), left in sorted(quotas.items(), key=lambda kv: -kv[1]):
            if left <= 0:
                continue
            made = try_make(kind, key, events, rng)
            if made:
                quotas[(kind, key)] -= 1
                stats[f"stratum:{kind}:{key}"] += 1
                break
        if not made:
            # 兜底:配额层全部不可构造/耗尽时,任取一个可构造条件(保证总量;分布如实上报)
            fb = ([("dom", d) for d in {e["dom"] for e in events}]
                  + [("time", "range"), ("time", "month")])
            rng.shuffle(fb)
            for kind, key in fb:
                made = try_make(kind, key, events, rng)
                if made:
                    stats[f"fallback:{kind}:{key}"] += 1
                    break
        if not made:
            stats["skip_no_condition"] += 1
            continue
        rec = assemble(events, made[0], made[1], rng)
        if rec["input"] in seen_inputs:
            stats["skip_dup_input"] += 1
            continue
        seen_inputs.add(rec["input"])
        recs.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(recs)} -> {out_path}", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}", file=sys.stderr)
    return recs


# ──────────────────────────── verify:从 prompt 独立复核 ────────────────────────────

def parse_prompt(inp):
    """→ (events, cond_text);只依赖 prompt 文本,不依赖构建内部状态。
    时间线块内无空行,首个 \n\n 即时间线/指令分界(few-shot 中的 SID 不会被误采)。"""
    body, _instr = inp.split("\n\n", 1)
    events, cur = [], None
    for line in body.split("\n"):
        m = DATE_LINE_RE.match(line)
        if m:
            cur = m.group(1)
            continue
        m = EVENT_LINE_RE.match(line)
        if m:
            hhmm, action, token = m.groups()
            dom = SID_RE.match(token).group(1)
            events.append({"date": cur, "hhmm": hhmm, "action": action, "token": token, "dom": dom})
    mc = re.search(r"筛选条件：(.+)", inp)
    return events, mc.group(1).strip()


def verify(path, n_bytecheck=None):
    n = ok_cond = ok_quote = ok_fmt = 0
    fails = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        n += 1
        inp, out = r["input"], r["output"]
        # 不变量:/no_think ⇔ 空 think
        fmt_ok = inp.rstrip().endswith("/no_think") and out.startswith("<think>\n</think>\n")
        arr = json.loads(out.split("</think>\n", 1)[1])
        fmt_ok = fmt_ok and isinstance(arr, list) and len(arr) == len(set(arr))
        ok_fmt += fmt_ok
        # 逐字节引用一致性:gold 元素 ∈ 时间线 token 集合(且是 input 子串)
        events, cond_text = parse_prompt(inp)
        toks = {e["token"] for e in events}
        q_ok = all(t in toks and t in inp for t in arr)
        ok_quote += q_ok
        # 程序复核筛选条件正确率:由条件文本独立重算 gold
        pred = parse_condition(cond_text)
        seen, expect = set(), []
        for e in events:
            if pred(e) and e["token"] not in seen:
                seen.add(e["token"])
                expect.append(e["token"])
        c_ok = expect == arr
        ok_cond += c_ok
        if not (fmt_ok and q_ok and c_ok) and len(fails) < 5:
            fails.append((i, fmt_ok, q_ok, c_ok, cond_text))
    print(f"[verify] {path}")
    print(f"  n={n}  format_invariant_ok={ok_fmt}  quote_exact_ok={ok_quote}  condition_recompute_ok={ok_cond}")
    if fails:
        print("  FAILS:", fails)
    return ok_fmt == n and ok_quote == n and ok_cond == n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("path", nargs="?", default=OUT_TRAIN)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=150)
    args = ap.parse_args()
    if args.cmd == "verify":
        sys.exit(0 if verify(args.path) else 1)
    shards = sorted(glob.glob(f"{HF}/OneReason_UserProfile/*.parquet"))
    build_split(shards[:9], args.n_train, 2026, OUT_TRAIN)   # 训练:分片 0-8
    build_split(shards[9:], args.n_val, 2027, OUT_VAL)       # 验证:分片 9(用户零重叠)


if __name__ == "__main__":
    main()
