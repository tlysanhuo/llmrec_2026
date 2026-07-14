#!/usr/bin/env python3
"""build_rebal_pstack.py — rebal_pstack_ep3 训练数据(2026-07-05,用户批准"go")。

战略:放弃Frinkleko家族底座(其唯一优势ad已证不可复现),回到我方0.9009底座做手术+靶向增量。
组成:
  A. data_rebal_world.jsonl 29019 手术:
     ①精确去重(其自带4339条纯重复,audit已量化) ②think去重转nothink(同(instruction,input)组内
     think逐字相同→留1条filled,其余 input尾 /think→/no_think、output=空think+原答案;
     机制=Frinkleko变换字节级复刻,治"CoT与答案解耦"毒点+免费产协议直出样本)
  B. + p3_quote_stop 3000(action_select全员≈0空地;预期+0.02~0.04)
  C. + p1_bucket_discrim 4679(物料头部同桶s_c分辨;全参3ep下embedding可动,机制通路成立)
  D. + p2_tail_cover 子集4000(尾部s_a覆盖;控物料总占比防挤占,seed采样)
  E. + p4_evolution 60(演化链首批,欠产待扩)
  F. + world_mc_clean 238(评测逐字模板MC;world已饱和,守格式不图分)
超参配套 configs/history/rebal_pstack_ep3.yaml = lr2e-5/3ep/全参(唯一登顶0.9009且物料上过2453的配方)。
预登记(07-05):action(±0.005噪声)≥0.095⇒P3生效;物料(零噪声)>0.2146⇒P1/P2×全参生效,
  ≥0.2453⇒历史新高;topic≥0.048⇒P4/P3迁移;ad不指望(0.05±);挤占检查=live/prod/video对0.9009锚点。
用法: python scripts/data/build_rebal_pstack.py
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
SEED = 2026
THINK = re.compile(r"<think>(.*?)</think>", re.S)

def load_alpaca(name):
    return [json.loads(l) for l in open(P / name, encoding="utf-8")]

def main():
    rng = random.Random(SEED)

    # A. 底座手术
    base = load_alpaca("data_rebal_world.jsonl")
    assert len(base) == 29019
    # ①精确去重
    seen, uniq = set(), []
    for r in base:
        k = (r["instruction"], r["input"], r["output"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    n_dedup = len(base) - len(uniq)

    # ②think去重转nothink
    groups = defaultdict(list)
    for r in uniq:
        groups[(r["instruction"], r["input"])].append(r)
    out_a, n_conv, n_groups = [], 0, 0
    for key, rows in groups.items():
        if len(rows) > 1:
            thinks = set()
            ok = True
            for r in rows:
                m = THINK.search(r["output"])
                if not m or not m.group(1).strip():
                    ok = False
                    break
                thinks.add(m.group(1))
            if ok and len(thinks) == 1 and key[1].rstrip().endswith("/think"):
                n_groups += 1
                out_a.append(rows[0])  # 留1条filled
                for r in rows[1:]:
                    tail = r["output"].split("</think>", 1)[1]
                    out_a.append({"instruction": r["instruction"],
                                  "input": r["input"][: r["input"].rfind("/think")] + "/no_think",
                                  "output": "<think>\n</think>" + tail,
                                  "history": r.get("history", [])})
                    n_conv += 1
                continue
        out_a.extend(rows)

    # B-E. 靶向增量(均为alpaca)
    p3 = load_alpaca("p3_quote_stop.jsonl")
    p1 = load_alpaca("p1_bucket_discrim.jsonl")
    p2_all = load_alpaca("p2_tail_cover.jsonl")
    p2 = rng.sample(p2_all, 4000)
    p4 = load_alpaca("p4_evolution.jsonl")

    # F. world_mc_clean 平台格式→alpaca
    wmc = []
    for l in open(P / "world_mc_clean.jsonl", encoding="utf-8"):
        d = json.loads(l)[0]
        wmc.append({"instruction": d["system"], "input": d["prompt"],
                    "output": d["response"], "history": []})

    out = out_a + p3 + p1 + p2 + p4 + wmc
    rng.shuffle(out)
    dst = P / "rebal_pstack.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {dst}: {len(out)} 条")
    print(f"  底座: 29019 -去重{n_dedup} → think去重转换 {n_conv}条/{n_groups}组 → {len(out_a)}")
    print(f"  增量: p3 {len(p3)} + p1 {len(p1)} + p2 {len(p2)}/{len(p2_all)} + p4 {len(p4)} + world_mc {len(wmc)}")

if __name__ == "__main__":
    main()
