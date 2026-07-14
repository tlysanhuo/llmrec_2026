#!/usr/bin/env python3
"""build_global_v1_assembly.py — P4 总装(2026-07-07,ideas/archive/global_recipe_v1.md 的落地)。

合流全部块 → data/processed/data_global_v1.jsonl,并输出对账表。
质检:①剥内部元数据键;②rec/mat 答案侧 itemic 三元组完整性(T2红线:<s_a 必须跟全 b/c);
③空输出/超长剔除;④按(input前120字+output前80字)全局去重。
"""
import json, re, random
from pathlib import Path

P = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed")
B = P / "blocks_v1"
rng = random.Random(20260707)
KEEP = ("instruction", "input", "output", "history")
BAD_TRIPLE = re.compile(r"<s_a_\d+>(?!<s_b_\d+><s_c_\d+>)")

def load(fp, tag):
    rows = []
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l)
        rows.append({k: r.get(k, [] if k == "history" else "") for k in KEEP})
    return tag, rows

SOURCES = [
    ("rec_video(P3三桶)", P / "p3_video_cot.jsonl"),
    ("rec_prod(P3三桶)",  P / "p3_prod_cot.jsonl"),
    ("rec_ad(直出80%)",   B / "block_rec_ad.jsonl"),
    ("rec_live(直出80%)", B / "block_rec_live.jsonl"),
    ("mat(种子原样)",     B / "block_mat.jsonl"),
    ("world(锚件)",       B / "block_world.jsonl"),
    ("general(推理地基)", B / "block_general.jsonl"),
    ("nonrec(直通)",      B / "block_nonrec.jsonl"),
    ("user_orig(直通)",   B / "block_user_orig.jsonl"),
    ("U2候选链",          B / "block_u2_mc.jsonl"),
]
U1_FILES = ["r2_gold_local", "r2_gold_g1", "r2_gold_g2", "r2_gold_v4"]
U3_FILES = ["u3_topic_a", "u3_topic_b"]

all_rows, ledger = [], []
for tag, fp in SOURCES:
    t, rows = load(fp, tag)
    ledger.append((tag, len(rows))); all_rows += rows
u1, seen1 = [], set()
for f in U1_FILES:
    fp = P / f"{f}.jsonl"
    if not fp.exists(): continue
    for l in open(fp):
        r = json.loads(l)
        k = r.get("_src_idx")
        if k in seen1: continue
        seen1.add(k)
        u1.append({kk: r.get(kk, [] if kk == "history" else "") for kk in KEEP})
ledger.append(("U1行为选择", len(u1))); all_rows += u1
u3 = []
for f in U3_FILES:
    fp = P / f"{f}.jsonl"
    if not fp.exists(): continue
    for l in open(fp):
        r = json.loads(l)
        u3.append({kk: r.get(kk, [] if kk == "history" else "") for kk in KEEP})
ledger.append(("U3主题链", len(u3))); all_rows += u3

# 全局质检
out, seen, drop_dup = [], set(), 0
drop_triple = drop_len = 0
for r in all_rows:
    o = r["output"]
    if not o or len(r["input"]) + len(o) > 120000:
        drop_len += 1; continue
    ans = o.split("</think>")[-1]
    if BAD_TRIPLE.search(ans):
        drop_triple += 1; continue
    k = (r["input"][:120], o[:80])
    if k in seen:
        drop_dup += 1; continue
    seen.add(k); out.append(r)
rng.shuffle(out)

dst = P / "data_global_v1.jsonl"
with open(dst, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("== 对账表 ==")
for tag, n in ledger:
    print(f"  {tag:22s} {n:>6}")
print(f"  合计输入 {sum(n for _, n in ledger)} | 剔重{drop_dup} 剔断裂三元组{drop_triple} 剔超长{drop_len}")
print(f"== 成品 {len(out)} 条 → {dst}")
