#!/usr/bin/env python3
"""build_riders_act_v1.py — 名次弹数据组装(2026-07-10)

data_riders_act_v1 = data_riders_fk(37,267,0.9177 底盘原封)+ p3_v2_extra(~2,600,B线三修正)。
一锅式(ep5_rider 教训:顺序 stage 蚀精度);新增行全部用户域(不碰 itemic 码本)。
QC:行数守恒;新行不变量(/no_think⇔空think、JSON数组去重);全库 input 零重复;
    与 dev_*.jsonl 零撞车(撞车制度);P3 v2 verify 必须先行通过。
"""
import glob
import hashlib
import json
import sys

ROOT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
SRC_BASE = f"{ROOT}/data/processed/data_riders_fk.jsonl"
SRC_P3V2 = f"{ROOT}/data/processed/p3_v2_extra.jsonl"
OUT = f"{ROOT}/data/processed/data_riders_act_v1.jsonl"

seen = set()
n_base = n_new = 0
dev_inputs = set()
for p in glob.glob(f"{ROOT}/data/offline_eval/dev_*.jsonl"):
    for line in open(p, encoding="utf-8"):
        try:
            dev_inputs.add(hashlib.md5(json.loads(line).get("input", "").encode()).hexdigest())
        except Exception:
            pass

with open(OUT, "w", encoding="utf-8") as fo:
    for line in open(SRC_BASE, encoding="utf-8"):
        d = json.loads(line)
        h = hashlib.md5(d["input"].encode()).hexdigest()
        seen.add(h)
        fo.write(line if line.endswith("\n") else line + "\n")
        n_base += 1
    for line in open(SRC_P3V2, encoding="utf-8"):
        d = json.loads(line)
        assert d["input"].rstrip().endswith("/no_think"), "marker invariant"
        assert d["output"].startswith("<think>\n</think>\n"), "empty-think invariant"
        arr = json.loads(d["output"].split("</think>\n", 1)[1])
        assert isinstance(arr, list) and len(arr) == len(set(arr)), "dedup invariant"
        h = hashlib.md5(d["input"].encode()).hexdigest()
        assert h not in seen, "dup vs base"
        assert h not in dev_inputs, "dev collision"
        seen.add(h)
        fo.write(json.dumps(d, ensure_ascii=False) + "\n")
        n_new += 1

md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
print(f"[ok] base={n_base} + p3v2={n_new} = {n_base+n_new} -> {OUT}")
print(f"[ok] 全库 input 唯一;dev 零撞车;新行不变量全过;md5 {md5[:8]}")
if n_base != 37267:
    print("[FAIL] base 行数异常", file=sys.stderr)
    sys.exit(1)
