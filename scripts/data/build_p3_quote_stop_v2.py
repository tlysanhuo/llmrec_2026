#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_p3_quote_stop_v2.py — P3 v2:quote-and-stop + B线三修正(2026-07-10)

在 v1(build_p3_quote_stop.py,全管线复用)之上新增,证据=docs/action_termination_eda.md §3.5:
  M1 [搜索]文本行注入:评测两用户历史 18%/45% 为搜索明文行,种子/P3 训练历史 0 覆盖(OOD 缺口);
     查询词从官方 Caption/Tag parquet 取时间线内 item 的类目叶子合成(语义与该用户自洽,全官方材料)。
     时间条件的 gold 自然混入查询文本 = 评测 few-shot 所示的混合数组形态;另设「搜索选取」条件层。
  M2 长 k 层:gold 20-45 的样本显式配额(评测 gold k 中位 11-15、p75=19;欠输出吃 recall)。
  M3 few-shot 对齐评测五元版(第 5 元为文本项"大静儿在北京"同款形态)。
不做:槽级近重复合成(向历史注入非官方 SID 有 pstack 型风险,搁置)。
用法:
  python scripts/data/build_p3_quote_stop_v2.py build   # p3_v2_extra 2600 + val 150
  python scripts/data/build_p3_quote_stop_v2.py verify data/processed/p3_v2_extra.jsonl
seed=2126/2127;对 v1 p3_quote_stop.jsonl 的 input 去重。
"""
import importlib.util
import json
import random
import re
import sys
from collections import Counter

ROOT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
spec = importlib.util.spec_from_file_location("p3v1", f"{ROOT}/scripts/data/build_p3_quote_stop.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)

OUT_TRAIN = f"{ROOT}/data/processed/p3_v2_extra.jsonl"
OUT_VAL = f"{ROOT}/data/processed/p3_v2_extra_val.jsonl"
CAP_PARQUET = f"{ROOT}/assets/official/sft_aligned/baseline_caption_tag_lists.parquet"

FEWSHOT_V2 = (
    "输出示例为（注意：以下案例来自其他用户，仅供参考输出格式，与上述用户交互历史无关）： "
    '["<|prod_begin|><s_a_750><s_b_2525><s_c_3393>", '
    '"<|living_begin|><s_a_6354><s_b_6678><s_c_4429>", '
    '"<|ad_begin|><s_a_7852><s_b_3625><s_c_5951>", '
    '"<|video_begin|><s_a_528><s_b_1682><s_c_7986>", '
    '"大静儿在北京"]'
)
Q_TPL = ["{x}", "{x}推荐", "{x}测评", "{x}怎么选", "{x}教程", "{x}多少钱", "便宜的{x}", "{x}排行榜"]
SEARCH_COND = "【搜索】行为的全部查询词（逐字原样引用查询文本，按出现顺序，去重）"


def load_sid2leaf():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(CAP_PARQUET)
    m = {}
    for batch in pf.iter_batches(columns=["sid_token_list", "tag_list"], batch_size=2000):
        for sl, tl in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            for s, t in zip(sl, tl):
                if t and s not in m:
                    leaf = t.split("-")[-1].strip()
                    if 2 <= len(leaf) <= 10:
                        m[s] = leaf
    return m


def inject_search(win, rng, sid2leaf):
    """向时间窗注入 2..min(25,35%) 条搜索行;查询词取窗内 item 类目叶子。返回新窗(按 ts 序)。"""
    leaves = []
    for e in win:
        lf = sid2leaf.get(e["token"])
        if lf:
            leaves.append((e["ts"], e["date"], lf))
    if not leaves:
        return win, 0
    n = rng.randint(2, max(3, min(25, int(0.35 * len(win)))))
    rows, used = [], set()
    for _ in range(n * 3):
        if len(rows) >= n:
            break
        ts, date, lf = rng.choice(leaves)
        q = rng.choice(Q_TPL).format(x=lf)
        if q in used:
            continue
        used.add(q)
        rows.append({"date": date, "hhmm": "--:--", "action": "搜索-搜索", "dom": "search",
                     "token": q, "ts": ts + rng.randint(-3600_000, 3600_000)})
    out = sorted(win + rows, key=lambda e: e["ts"])
    return out, len(rows)


def cond_search():
    return (SEARCH_COND, lambda e: e["dom"] == "search")


def parse_condition_v2(text):
    if text == SEARCH_COND:
        return cond_search()[1]
    return v1.parse_condition(text)


STRATA_V2 = (
    [("dom", d, 90) for d in ("prod", "living", "video", "ad")]          # 360 常规(含搜索干扰行)
    + [("act", a, 60) for a in ("商品-点击", "商品-购买", "广告-点击", "直播-关注")]  # 240
    + [("time", "range", 330), ("time", "month", 250)]                     # 580 时间条件(gold 混文本项)
    + [("search", "search", 420)]                                          # 420 搜索选取
    + [("longk", "dom", 300), ("longk", "time", 500)]                      # 800 长k层(20-45)
    + [("empty", "empty", 130)]                                            # 130 空答案
)  # ≈2530 名义,×scale 到 n_target


def try_make_v2(kind, key, events, rng):
    if kind == "search":
        g = v1.gold_of(events, cond_search()[1])
        return (SEARCH_COND, g) if 2 <= len(g) <= 25 else None
    if kind == "longk":
        for _ in range(8):
            if key == "dom":
                doms = [e["dom"] for e in events if e["dom"] != "search"]
                if not doms:
                    return None
                c = v1.cond_domain(rng.choice(doms))
            else:
                dates = sorted({e["date"] for e in events})
                if len(dates) < 4:
                    return None
                i = rng.randrange(0, len(dates) - 2)
                j = rng.randrange(i + 2, len(dates))
                c = v1.cond_range(dates[i], dates[j])
            g = v1.gold_of(events, c[1])
            if 20 <= len(g) <= 45:
                return c[0], g
        return None
    return v1.try_make(kind, key, events, rng)


def assemble_v2(events, cond_text, gold, rng):
    role = rng.choice(v1.ROLES)
    fmt = rng.choice(v1.FORMATS)
    instr = role + "\n" + f"筛选条件：{cond_text}" + "\n" + fmt
    if rng.random() < 0.4:
        instr += "\n" + FEWSHOT_V2
    instr += "/no_think"
    return {"instruction": "", "input": v1.render_timeline(events) + "\n\n" + instr,
            "output": "<think>\n</think>\n" + json.dumps(gold, ensure_ascii=False), "history": []}


def build_split_v2(user_shards, n_target, seed, out_path, sid2leaf, existing_inputs):
    rng = random.Random(seed)
    users = v1.scan_users(user_shards, max_users=int(n_target * 8))
    print(f"[pass1] {len(users)} users", file=sys.stderr)
    needed = {(dom, pid) for ev in users for dom, pid, _, _ in ev}
    sid_map = v1.build_sid_map(needed)
    print(f"[pass2] mapped {len(sid_map):,}/{len(needed):,}", file=sys.stderr)

    quotas = {(k, key): q for k, key, q in STRATA_V2}
    scale = n_target / sum(q for _, _, q in STRATA_V2)
    quotas = {k: max(1, round(q * scale)) for k, q in quotas.items()}
    recs, stats, seen = [], Counter(), set(existing_inputs)
    for ev_raw in users:
        if len(recs) >= n_target:
            break
        events = v1.render_events(ev_raw, sid_map)
        nv = [e for e in events if e["dom"] != "video"]
        vv = [e for e in events if e["dom"] == "video"]
        cap = max(10, 2 * len(nv))
        if len(vv) > cap:
            step = len(vv) / cap
            vv = [vv[int(i * step)] for i in range(cap)]
            events = sorted(nv + vv, key=lambda e: e["ts"])
        if len(events) < 20:
            continue
        w = rng.randint(20, 80)
        start = rng.randint(0, max(0, len(events) - w))
        win = events[start:start + w]
        if len({e["date"] for e in win}) < 3:
            win = events[-80:]
        if len(win) < 20 or len({e["date"] for e in win}) < 2:
            continue
        # 70% 样本注入搜索行(搜索层样本必注入)
        planned = sorted(quotas.items(), key=lambda kv: -kv[1])
        want_search = rng.random() < 0.70
        if want_search or any(k == ("search", "search") and q > 0 for k, q in planned[:1]):
            win, ns = inject_search(win, rng, sid2leaf)
            if ns:
                stats["search_rows_injected"] += ns
                stats["samples_with_search"] += 1
        made = None
        for (kind, key), left in planned:
            if left <= 0:
                continue
            made = try_make_v2(kind, key, win, rng)
            if made:
                quotas[(kind, key)] -= 1
                stats[f"stratum:{kind}:{key}"] += 1
                break
        if not made:
            continue
        rec = assemble_v2(win, made[0], made[1], rng)
        if rec["input"] in seen:
            stats["skip_dup"] += 1
            continue
        seen.add(rec["input"])
        recs.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(recs)} -> {out_path}", file=sys.stderr)
    for k, c in sorted(stats.items()):
        print(f"    {k}: {c}", file=sys.stderr)
    return recs


SEARCH_LINE_RE = re.compile(r"^  --:-- \[搜索-搜索\] (.+)$")


def parse_prompt_v2(inp):
    body, _ = inp.split("\n\n", 1)
    events, cur = [], None
    for line in body.split("\n"):
        m = v1.DATE_LINE_RE.match(line)
        if m:
            cur = m.group(1)
            continue
        m = v1.EVENT_LINE_RE.match(line)
        if m:
            hhmm, action, token = m.groups()
            events.append({"date": cur, "hhmm": hhmm, "action": action, "token": token,
                           "dom": v1.SID_RE.match(token).group(1)})
            continue
        m = SEARCH_LINE_RE.match(line)
        if m:
            events.append({"date": cur, "hhmm": "--:--", "action": "搜索-搜索",
                           "token": m.group(1), "dom": "search"})
    mc = re.search(r"筛选条件：(.+)", inp)
    return events, mc.group(1).strip()


def verify_v2(path):
    n = ok = 0
    fails = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        n += 1
        inp, out = r["input"], r["output"]
        fmt = inp.rstrip().endswith("/no_think") and out.startswith("<think>\n</think>\n")
        arr = json.loads(out.split("</think>\n", 1)[1])
        fmt = fmt and isinstance(arr, list) and len(arr) == len(set(arr))
        events, cond_text = parse_prompt_v2(inp)
        toks = {e["token"] for e in events}
        quote = all(t in toks and t in inp for t in arr)
        pred = parse_condition_v2(cond_text)
        seen_g, expect = set(), []
        for e in events:
            if pred(e) and e["token"] not in seen_g:
                seen_g.add(e["token"])
                expect.append(e["token"])
        cond = expect == arr
        good = fmt and quote and cond
        ok += good
        if not good and len(fails) < 5:
            fails.append((i, fmt, quote, cond, cond_text[:40]))
    print(f"[verify] {path}\n  n={n} all_ok={ok}")
    if fails:
        print("  FAILS:", fails)
    return ok == n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("path", nargs="?", default=OUT_TRAIN)
    ap.add_argument("--n_train", type=int, default=2600)
    ap.add_argument("--n_val", type=int, default=150)
    args = ap.parse_args()
    if args.cmd == "verify":
        sys.exit(0 if verify_v2(args.path) else 1)
    print("[cap] loading sid->tag leaf map ...", file=sys.stderr)
    sid2leaf = load_sid2leaf()
    print(f"[cap] {len(sid2leaf):,} sids with tag leaf", file=sys.stderr)
    existing = set()
    for line in open(f"{ROOT}/data/processed/p3_quote_stop.jsonl", encoding="utf-8"):
        existing.add(json.loads(line)["input"])
    import glob as g
    shards = sorted(g.glob(f"{v1.HF}/OneReason_UserProfile/*.parquet"))
    build_split_v2(shards[:9], args.n_train, 2126, OUT_TRAIN, sid2leaf, existing)
    build_split_v2(shards[9:], args.n_val, 2127, OUT_VAL, sid2leaf, existing)


if __name__ == "__main__":
    main()
