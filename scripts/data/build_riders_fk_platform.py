#!/usr/bin/env python3
"""build_riders_fk_platform.py — data_riders_fk(alpaca)→ 平台/官方种子格式(2026-07-09)

目的:把当前最高分底盘 data_riders_fk.jsonl(37267 行,riders_fk_lora_ep1=0.9177 的训练数据)
转成官方种子原始格式(每行 JSON list:[{system, prompt, response}]),供平台"模型定制-精调"
上传数据集用(平台训练臂:同数据同超参,单变量=平台 focal+token加权 loss)。

映射(逐字段,零内容改动):instruction→system(可为空,Frinkleko 0.9107 数据同款)、
input→prompt、output→response;history 全部为 [](断言校验)后丢弃。
不做 shuffle(源文件构建时已 shuffle,保持逐行同序可复现)。

QC:行数守恒 37267;逐行回读反向重建 alpaca 三字段与源字节级相等;统计 /no_think 尾标记
覆盖与空 system 行数。
"""
import json
import hashlib
import sys

SRC = "data/processed/data_riders_fk.jsonl"
DST = "data/processed/riders_fk_platform.jsonl"

n = 0
empty_sys = 0
nothink_tail = 0
with open(SRC) as fin, open(DST, "w") as fout:
    for line in fin:
        d = json.loads(line)
        assert d["history"] == [], f"row {n}: nonempty history"
        rec = [{"system": d["instruction"], "prompt": d["input"], "response": d["output"]}]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if not d["instruction"]:
            empty_sys += 1
        if d["input"].rstrip().endswith("/no_think"):
            nothink_tail += 1

# 回读校验:反向重建与源逐行相等
m = 0
with open(SRC) as fsrc, open(DST) as fdst:
    for ls, ld in zip(fsrc, fdst):
        s = json.loads(ls)
        t = json.loads(ld)[0]
        assert t["system"] == s["instruction"] and t["prompt"] == s["input"] and t["response"] == s["output"], f"row {m} mismatch"
        m += 1
assert m == n, "row count mismatch on verify"

md5 = hashlib.md5(open(DST, "rb").read()).hexdigest()
print(f"[ok] {DST}: {n} rows (src=dst 守恒), 空system {empty_sys}, /no_think尾 {nothink_tail}/{n}")
print(f"[ok] 回读逐行字节级校验通过 ({m} 行)")
print(f"[ok] md5 {md5}")
if n != 37267:
    print("[FAIL] 行数 != 37267", file=sys.stderr)
    sys.exit(1)
