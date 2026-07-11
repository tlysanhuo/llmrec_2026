#!/usr/bin/env python3
"""build_p0_blocks.py — 全局配方v1 的 P0 零成本块(2026-07-07,依据 docs/global_recipe_v1.md)。

产出 data/processed/blocks_v1/:
  block_rec_ad.jsonl     ad 全量,CoT行80%转nothink直出(Frinkleko字节机制),20%保留CoT
  block_rec_live.jsonl   living 同政策
  block_rec_video.jsonl  video 采样9000原样(待P3重写CoT;本步只降采样去冗余)
  block_rec_prod.jsonl   prod 全量原样(待P3)
  block_mat.jsonl        物料desc2token 1814 原样
  block_world.jsonl      world_zh2824(rebal验证批)+CEval1578+world_mc238
  block_general.jsonl    world_zh池剩余部分采样3000(与2824无重叠)
"""
import json, random, re
from pathlib import Path

P = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed")
OUT = P / "blocks_v1"
OUT.mkdir(exist_ok=True)
rng = random.Random(20260707)
BEGIN = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
SID = re.compile(r"<s_[abc]_\d+>")


def load(fp):
    rows = []
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l)
        if isinstance(r, list):
            r = r[0]
        if "prompt" in r and "instruction" not in r:
            r = {"instruction": r.get("system", ""), "input": r["prompt"], "output": r["response"]}
        rows.append(r)
    return rows


def classify(r):
    ins = r.get("instruction", "")
    if "token" in ins and ("生成助手" in ins or "输出匹配" in ins or "输出对应" in ins):
        return "mat_desc2token"
    body = r["output"].split("</think>")[-1]
    if body.strip().startswith("["):
        return "user_select"
    m = BEGIN.search(body)
    if m:
        return f"rec_{m.group(1)}"
    return "non_rec"


def to_nothink(r):
    """Frinkleko 字节机制:prompt /think→/no_think,think 清空,答案不动。"""
    n = dict(r)
    n["input"] = re.sub(r"/think\s*$", "/no_think", n["input"])
    body = n["output"].split("</think>", 1)[-1].lstrip("\n")
    n["output"] = "<think>\n\n</think>\n" + body
    return n


def uncot_policy(rows, keep_cot=0.2):
    out = []
    for r in rows:
        has_cot = len(r["output"].split("</think>")[0]) > 30
        if has_cot and rng.random() > keep_cot:
            out.append(to_nothink(r))
        else:
            out.append(dict(r))
    return out


def dump(name, rows):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            r.setdefault("history", [])
            f.write(json.dumps({k: r[k] for k in ("instruction", "input", "output", "history") if k in r},
                               ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}")


seed = load(P / "data_final.jsonl")
buckets = {}
for r in seed:
    buckets.setdefault(classify(r), []).append(r)
print("种子分类:", {k: len(v) for k, v in sorted(buckets.items())})

dump("block_rec_ad.jsonl", uncot_policy(buckets["rec_ad"]))
dump("block_rec_live.jsonl", uncot_policy(buckets["rec_living"]))
vid = buckets["rec_video"][:]
rng.shuffle(vid)
dump("block_rec_video.jsonl", vid[:9000])
dump("block_rec_prod.jsonl", buckets["rec_prod"])
dump("block_mat.jsonl", buckets["mat_desc2token"])

# world 块:与 tokengeo 同源提取
world_pool = load(P / "data_rebal_world.jsonl")
wz = [r for r in world_pool
      if "/think" not in (r.get("instruction", "") + r.get("input", ""))
      and "/no_think" not in (r.get("instruction", "") + r.get("input", ""))]
assert len(wz) == 2824
wmc = load(P / "world_mc_clean.jsonl")
norm = lambda s: re.sub(r"\s+", "", s or "")
wmc_keys = {norm(r.get("input", "")) for r in wmc}
fk = load(P / "frinkleko_alpaca_32705.jsonl")
ceval = [r for r in fk if "遵循指示作答" in r.get("instruction", "")
         and not SID.search(json.dumps(r, ensure_ascii=False))
         and norm(r.get("input", "")) not in wmc_keys]
dump("block_world.jsonl", wz + ceval + wmc)

# general 块:world_zh 全池(16237)扣掉已用 2824,采 3000
wz_all = load(P / "world_zh.jsonl")
used = {norm(r.get("input", ""))[:120] for r in wz}
rest = [r for r in wz_all if norm(r.get("input", ""))[:120] not in used]
rng.shuffle(rest)
dump("block_general.jsonl", rest[:3000])
print("[P0 done]", OUT)
