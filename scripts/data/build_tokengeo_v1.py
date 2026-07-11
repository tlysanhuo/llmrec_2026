#!/usr/bin/env python3
"""build_tokengeo_v1.py — 合成弹 tokengeo_v1 训练数据(2026-07-06,用户批准"开始做")。

配方 = 纯种子全量(物料8题底盘) + 三个已验证/低风险增量 + Token粒度包(本发的科学载荷):
  data_final 32480               # 种子全量,video不砍(保物料8题)
  world_zh   2824                # 通识,已验证+0.011(从 data_rebal_world 提取同一批2824条)
  P3         1500                # action专项(pstack_v2同seed同量,已验证0.0808)
  world_mc_clean 238 + CEval 1578  # MC格式锚(pstack_v2验证:治P3复制训练的C崩)
  token_pack 3377                # Token粒度四形态(论文§4.2.1,码本几何)
预登记:物料>8题⇒Token包实锤;world≈0.142;action≈0.07;详见 experiment_log。
"""
import json, random, re
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

seed = load(P / "data_final.jsonl")

# world_zh 2824:data_rebal_world 里的"无后缀行"(prompt 不含 /think 或 /no_think)
world = [r for r in load(P / "data_rebal_world.jsonl")
         if "/think" not in (r.get("instruction", "") + r.get("input", ""))
         and "/no_think" not in (r.get("instruction", "") + r.get("input", ""))]
assert len(world) == 2824, f"world_zh 提取 {len(world)} != 2824"

p3 = rng.sample(load(P / "p3_quote_stop.jsonl"), 1500)

wmc = load(P / "world_mc_clean.jsonl")
norm = lambda s: re.sub(r"\s+", "", s or "")
wmc_keys = {norm(r.get("input", "")) for r in wmc}
SID = re.compile(r"<s_[abc]_\d+>")
fk = load(P / "frinkleko_alpaca_32705.jsonl")
ceval = [r for r in fk if "遵循指示作答" in r.get("instruction", "") and not SID.search(json.dumps(r, ensure_ascii=False))]
ceval = [r for r in ceval if norm(r.get("input", "")) not in wmc_keys]
print(f"CEval 桶: {len(ceval)}")

tok = load(P / "token_pack.jsonl")

allrows = seed + world + p3 + wmc + ceval + tok
rng.shuffle(allrows)
out = P / "data_tokengeo_v1.jsonl"
with open(out, "w", encoding="utf-8") as g:
    for r in allrows:
        r.setdefault("history", [])
        g.write(json.dumps({k: r[k] for k in ("instruction", "input", "output", "history") if k in r},
                           ensure_ascii=False) + "\n")
print(f"seed {len(seed)} + world_zh {len(world)} + P3 {len(p3)} + wmc {len(wmc)} + ceval {len(ceval)} + token {len(tok)}"
      f" = {len(allrows)} → {out}")
