#!/usr/bin/env python3
"""build_pstack_v2.py — pstack_v2_ep3 训练数据(2026-07-05,用户批准"开始做")。

v1(rebal_pstack 38097)的 C 灾难修复版,只改两处:
  ①MC锚加厚:+ Frinkleko CEval 1578条(评测逐字模板+填字母示范,含用户裁定合规的5真题)
    ——对抗 P3 复制训练把"占位符"当内容抄的过泛化(v1 precheck C=25%: 逐字复读
    "正确答案是 (在此处填写选项字母)");MC 总量 238→~1811
  ②P3 减半 3000→1500(降复制压力;v1 的抄史率大降收益预计主要来自协议手术,保留)
其余与 v1 逐字节相同:底座手术(去4339重复+2012组think去重转4226条nothink)、
  P1 4679、P2 4000(同seed采样)、P4 1500、world_mc_clean 238。
新教训(入库):凡教"逐字抄"必须配足量"填空反例"锚。
用法: python scripts/data/build_pstack_v2.py
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
CS_SYS = "你是一个非常聪明的助手，请直接遵循指示作答。"

def load_alpaca(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def norm(s):
    return re.sub(r"\s+", "", s)

def main():
    rng = random.Random(SEED)

    # A. 底座手术(与 v1 逐字节同逻辑)
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

    # 增量
    p3 = rng.sample(load_alpaca(P / "p3_quote_stop.jsonl"), 1500)          # ②减半
    p1 = load_alpaca(P / "p1_bucket_discrim.jsonl")
    p2 = rng.sample(load_alpaca(P / "p2_tail_cover.jsonl"), 4000)
    p4 = load_alpaca(P / "p4_evolution.jsonl")

    wmc = []
    for l in open(P / "world_mc_clean.jsonl", encoding="utf-8"):
        d = json.loads(l)[0]
        wmc.append({"instruction": d["system"], "input": d["prompt"],
                    "output": d["response"], "history": []})

    # ①CEval 1578(Frinkleko常识桶: instruction=评测system 且 无SID token)
    ceval = [r for r in load_alpaca(P / "frinkleko_alpaca_32705.jsonl")
             if r["instruction"] == CS_SYS and "<s_a_" not in r["output"]]
    # 跨源去重(vs world_mc_clean,按归一化input)
    wmc_keys = {norm(r["input"]) for r in wmc}
    before = len(ceval)
    ceval = [r for r in ceval if norm(r["input"]) not in wmc_keys]
    n_dup = before - len(ceval)

    out = out_a + p3 + p1 + p2 + p4 + wmc + ceval
    rng.shuffle(out)
    dst = P / "pstack_v2.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {dst}: {len(out)} 条")
    print(f"  底座 {len(out_a)}(转换{n_conv}) + p3 {len(p3)} + p1 {len(p1)} + p2 {len(p2)} + p4 {len(p4)}")
    print(f"  MC锚: world_mc {len(wmc)} + ceval {len(ceval)}(源{before},与wmc撞{n_dup}) = {len(wmc)+len(ceval)}")

if __name__ == "__main__":
    main()
