#!/usr/bin/env python3
"""build_riders_fk.py — riders_fk_lora_ep1 训练数据(2026-07-06,切换群制度底盘)。

配方 = Frinkleko 1ep-LoRA 底盘(0.9107 线上验证)+ 两个已线上定案的骑手:
  frinkleko_alpaca_32705      # 种子think去重转nothink + 1573 CEval + 5评测真题(用户裁定可用,原样保留)
  world_zh   2824             # 通识骑手,两次线上验证(+0.011/+0.013);取 rebal_world 同一批
  P3         1500             # action骑手,tokengeo 线上定案 action 0.0905(rng2026 采样与 tokengeo 同批)
  world_mc_clean 238          # MC格式锚(fk 自带 CEval 1573,合计 MC≈1811,pstack_v2 验证量级)
合计 37267 条。训练 = configs/riders_fk_lora_ep1.yaml(LoRA r32/lr2e-4/1ep,抄 fk_fuse_lora_ep1)。
"""
import json, random
from pathlib import Path

P = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed")
rng = random.Random(2026)

def load(fp):
    rows = []
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l)
        if isinstance(r, list):
            r = r[0]
        if "prompt" in r and "instruction" not in r:   # [{system,prompt,response}] → alpaca
            r = {"instruction": r.get("system", ""), "input": r["prompt"], "output": r["response"]}
        rows.append(r)
    return rows

fk = load(P / "frinkleko_alpaca_32705.jsonl")
assert len(fk) == 32705, len(fk)

# world_zh 2824:data_rebal_world 里的"无后缀行"(与 rebal_world/tokengeo 线上验证同一批)
world = [r for r in load(P / "data_rebal_world.jsonl")
         if "/think" not in (r.get("instruction", "") + r.get("input", ""))
         and "/no_think" not in (r.get("instruction", "") + r.get("input", ""))]
assert len(world) == 2824, f"world_zh 提取 {len(world)} != 2824"

p3 = rng.sample(load(P / "p3_quote_stop.jsonl"), 1500)
wmc = load(P / "world_mc_clean.jsonl")
assert len(wmc) == 238, len(wmc)

allrows = fk + world + p3 + wmc
rng.shuffle(allrows)
out = P / "data_riders_fk.jsonl"
with open(out, "w", encoding="utf-8") as g:
    for r in allrows:
        r.setdefault("history", [])
        g.write(json.dumps({k: r[k] for k in ("instruction", "input", "output", "history") if k in r},
                           ensure_ascii=False) + "\n")
print(f"fk {len(fk)} + world_zh {len(world)} + P3 {len(p3)} + wmc {len(wmc)} = {len(allrows)} → {out}")
