#!/usr/bin/env python3
"""build_base_v3.py — base_v3 底座(2026-07-05):pstack 底座手术 + 种子 think 质量三刀。

第一步·复刻既有手术(逐字节复用 build_pstack_v2.py 的底座段逻辑):
  data_rebal_world.jsonl 29019 → 精确去重(-4339) → 同(instruction,input)组内 think
  逐字相同→留1条filled其余转nothink(4226条) → 24680条。

第二步·think 质量手术(docs/seed_think_audit_20260705.md 定案的三个刀口):
  刀1 夭折think剥除(懂推荐行):think 以非句末符收尾(、/(/裸汉字/逗号/反引号/SID
      token 等)且 len<800 → 剥think转nothink。
      判据标定(在 data_final 全量懂推荐 19204 行重测):
        本判据 1495行/425唯一think/命中长度p50=283
        审计口径 1467行/439唯一think/p50=289 —— 行数+1.9%,唯一-3.2%,视为同刀。
  刀2 物料 part5-7 泛化think剥除(SID→desc 方向物料行):filled think 全部剥除。
      审计口径 2,390 行(全库 SID→desc filled-think 总量),本脚本按结构判定
      (物料instruction + input含SID + output正文无SID)。
  刀3 物料 part1-4 元叙述think剥除(desc→SID 方向物料行):think 以"提取"开头
      且 desc侧字2-gram覆盖率<0.3(cov=|2g(think)∩2g(desc)|/|2g(desc)|)。
      标定:data_final 全量命中 278 ≈ 审计"~280行"口径(该定义下"提取"开头即
      必然<0.3;think侧覆盖率定义只得193,与口径不符,弃用)。

不变量:/no_think⇔空think;剥除后答案部分(</think>之后)逐字节不动;
       world_zh 行无 /think|/no_think 后缀、自带filled think,三刀均不触碰。
用法: python scripts/data/build_base_v3.py
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
SEED = 2026
THINK = re.compile(r"<think>(.*?)</think>", re.S)
SID = "<s_a_"
# 句末合法收尾符(标定于 data_final 懂推荐 think 实测分布;'.'/','/':'等为截断证据)
FINAL_OK = set("。！？!?…”\"」』)）】》]")
LEN_CAP = 800  # 审计"长度显著短于正常"(截断p50=289 ≪ 正常p50=1396)的操作化


def load_alpaca(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def classify(r):
    """结构分类:rec=懂推荐 / mat_s2d=物料SID→desc / mat_d2s=物料desc→SID / other。"""
    ins = r["instruction"]
    if not ins:
        return "other"  # 懂用户(空instruction) 或 world_zh
    body = r["output"].split("</think>")[-1]
    if "token" in ins or "标识" in ins:  # 物料 instruction 全部含 token/标识
        i_sid, o_sid = SID in r["input"], SID in body
        if i_sid and not o_sid:
            return "mat_s2d"
        if o_sid and not i_sid:
            return "mat_d2s"
        return "other"
    return "rec"  # 6 种懂推荐 system prompt 均不含 token/标识


def get_think(r):
    m = THINK.search(r["output"])
    return m.group(1).strip() if m and m.group(1).strip() else None


def grams2(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def desc_of(inp):
    inp = inp.rstrip()
    for suf in ("/think", "/no_think"):
        if inp.endswith(suf):
            return inp[: -len(suf)]
    return inp


def is_aborted(t):
    """刀1判据:非句末符收尾 且 长度显著短。"""
    return t[-1] not in FINAL_OK and len(t) < LEN_CAP


def is_meta(t, inp):
    """刀3判据:'提取'开头 且 desc侧2-gram覆盖率<0.3。"""
    if not t.startswith("提取"):
        return False
    gd = grams2(desc_of(inp))
    return len(grams2(t) & gd) / max(1, len(gd)) < 0.3


def strip_think(r):
    """剥think转nothink——与 build_pstack_v2 底座段转换逻辑逐字节相同。"""
    assert r["input"].rstrip().endswith("/think"), "剥除对象必须是 /think 行"
    tail = r["output"].split("</think>", 1)[1]
    return {"instruction": r["instruction"],
            "input": r["input"][: r["input"].rfind("/think")] + "/no_think",
            "output": "<think>\n</think>" + tail,
            "history": r.get("history", [])}


def main():
    rng = random.Random(SEED)

    # ---- 第一步 A. 底座手术(与 build_pstack_v2.py 逐字节同逻辑) ----
    base = load_alpaca(P / "data_rebal_world.jsonl")
    assert len(base) == 29019
    seen, uniq = set(), []
    for r in base:
        k = (r["instruction"], r["input"], r["output"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    groups = defaultdict(list)
    for r in uniq:
        groups[(r["instruction"], r["input"])].append(r)
    out_a, n_conv = [], 0
    for key, rows in groups.items():
        if len(rows) > 1 and key[1].rstrip().endswith("/think"):
            thinks = set()
            ok = True
            for r in rows:
                m = THINK.search(r["output"])
                if not m or not m.group(1).strip():
                    ok = False
                    break
                thinks.add(m.group(1))
            if ok and len(thinks) == 1:
                out_a.append(rows[0])
                for r in rows[1:]:
                    tail = r["output"].split("</think>", 1)[1]
                    out_a.append({"instruction": r["instruction"],
                                  "input": r["input"][: r["input"].rfind("/think")] + "/no_think",
                                  "output": "<think>\n</think>" + tail,
                                  "history": r.get("history", [])})
                    n_conv += 1
                continue
        out_a.extend(rows)
    assert len(uniq) == 24680 and len(out_a) == 24680 and n_conv == 4226, \
        f"底座手术对不上: uniq={len(uniq)} out={len(out_a)} conv={n_conv}"

    # ---- 第二步 B. think 质量三刀 ----
    hits = Counter()
    out, audit_rows = [], []
    for r in out_a:
        cls = classify(r)
        t = get_think(r)
        knife = None
        if t is not None and r["input"].rstrip().endswith("/think"):
            if cls == "rec" and is_aborted(t):
                knife = "k1_aborted_rec"
            elif cls == "mat_s2d":
                knife = "k2_mat_sid2desc"
            elif cls == "mat_d2s" and is_meta(t, r["input"]):
                knife = "k3_mat_meta"
        if knife:
            hits[knife] += 1
            s = strip_think(r)
            assert s["output"].split("</think>", 1)[1] == r["output"].split("</think>", 1)[1]
            audit_rows.append((r, s, knife))
            out.append(s)
        else:
            out.append(r)

    # ---- QC ----
    assert len(out) == 24680, "行数不守恒"
    n_bad_inv = n_nosuffix = 0
    for r in out:
        inp = r["input"].rstrip()
        filled = get_think(r) is not None
        if inp.endswith("/no_think"):
            if filled:
                n_bad_inv += 1
        elif inp.endswith("/think"):
            if not filled:
                n_bad_inv += 1
        else:
            n_nosuffix += 1  # world_zh 行(无模式后缀,不受不变量约束)
            if not filled:
                n_bad_inv += 1
    assert n_bad_inv == 0, f"不变量违例 {n_bad_inv}"

    # 剥除行答案 diff=0(独立重查一遍)
    n_diff = sum(1 for r, s, _ in audit_rows
                 if r["output"].split("</think>", 1)[1] != s["output"].split("</think>", 1)[1])
    assert n_diff == 0

    # 对账:刀1 判据在 data_final 全量懂推荐上的读数(审计口径 1467/19204=7.64%)
    final_rec = [r for r in load_alpaca(P / "data_final.jsonl") if classify(r) == "rec"]
    fr_hits = [t for t in (get_think(r) for r in final_rec) if t and is_aborted(t)]
    base_rec = [r for r in out_a if classify(r) == "rec"]
    base_rec_filled = [r for r in base_rec if get_think(r) and r["input"].rstrip().endswith("/think")]

    # ---- 写盘(与 pstack 同风格:seed=2026 shuffle) ----
    rng.shuffle(out)
    dst = P / "base_v3.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_strip = sum(hits.values())
    print(f"[OK] {dst}: {len(out)} 条 (行数守恒 24680)")
    print(f"底座手术: 29019 → 去重 {len(uniq)} → 组内think去重转nothink {n_conv} 条")
    print(f"刀1 夭折think(懂推荐): {hits['k1_aborted_rec']} 条"
          f" / 底座filled懂推荐 {len(base_rec_filled)} 行 = {hits['k1_aborted_rec']/len(base_rec_filled):.2%}")
    print(f"   对账: 同判据在 data_final 全量懂推荐 {len(final_rec)} 行命中 {len(fr_hits)}"
          f" ({len(fr_hits)/len(final_rec):.2%}), 唯一think {len(set(fr_hits))}"
          f"; 审计口径 1467 (7.64%), 唯一 439")
    print(f"刀2 物料SID→desc泛化think: {hits['k2_mat_sid2desc']} 条 (审计全库口径 2,390)")
    print(f"刀3 物料desc→SID元叙述think: {hits['k3_mat_meta']} 条"
          f" (同判据 data_final 全量命中 278 ≈ 审计~280口径)")
    print(f"think剥除合计: {total_strip} 条; base_v3 nothink 总量 = {n_conv} + {total_strip} + 原生")
    print(f"不变量: /no_think⇔空think 违例 0; 无后缀行(world_zh) {n_nosuffix}; 剥除行答案diff 0")

    # 随机抽 20 条剥除行供人工复查
    print("\n===== 随机抽检 20 条剥除行(前=原think尾部 | 后=转换头部) =====")
    for r, s, kn in random.Random(7).sample(audit_rows, min(20, len(audit_rows))):
        t = get_think(r)
        print(f"[{kn}] len={len(t)} think尾: ...{t[-60:]!r}")
        print(f"      new input尾: ...{s['input'][-30:]!r} | new output头: {s['output'][:40]!r}")

    return hits


if __name__ == "__main__":
    main()
