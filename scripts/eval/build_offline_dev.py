#!/usr/bin/env python3
"""build_offline_dev.py — 离线评测台 v3 的 dev 集构建(一次性,幂等)。

产物 → data/offline_eval/:
  dev_mat_fresh.jsonl   543  队友 mat_probe_fresh 重排为评测模板(圈外性核验:与 fresh_mat 3000 去重叠)
  dev_mat_train.jsonl   300  block_mat 抽样(圈内记忆化对照,不判决)
  dev_rec_{video,prod,ad,live}.jsonl 各1000  rec_loo_v2 分域抽样
  dev_action.jsonl      ~325 r2_gold_v4 + r2_gold_local
  dev_topic.jsonl       110  u3_heldout
  dev_world.jsonl       ~500 CEval-val 下载(hf-mirror),剔除训练 MC 锚 1816 题干,套评测逐字模板
统一 schema: {"system": str|""(空=无system块), "user": str, "gold": ..., "meta": {...}}
设计记录:docs/offline_eval.md §2
"""
import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request
import zipfile

PROJ = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
P = f"{PROJ}/data/processed"
OUT = f"{PROJ}/data/offline_eval"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(2026)
SID = re.compile(r"<\|(video|ad|prod|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
ITEM = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")

MAT_SYS = {  # 评测同款任务形(系统提示为同义改写族,取种子高频款)
    "video": "你是一个精准的视频属性提取器，能够根据提供的视频描述生成唯一的视频标识符。",
    "prod": "你是一名专业的电商商品token生成助手，请根据商品描述生成匹配的商品token。",
    "ad": "你是一名广告token生成助手，需要根据广告内容描述生成最匹配的广告token。",
    "living": "你是一名专业的主播token生成助手，请根据主播的形象、内容与风格描述生成匹配的主播token。",
}
WORLD_SYS = "你是一个非常聪明的助手，请直接遵循指示作答。"
WORLD_USER = '请回答以下问题：\n\n{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\n\n请按以下格式作答："正确答案是 (在此处填写选项字母)"'


def wl(name, rows):
    fp = f"{OUT}/{name}"
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.md5(open(fp, "rb").read()).hexdigest()[:8]
    print(f"  {name}: {len(rows)} 条  md5={h}")


def parse_gold(output):
    m = SID.search(output.split("</think>")[-1])
    return {"dom": m.group(1), "abc": [m.group(2), m.group(3), m.group(4)]} if m else None


# ---------- mat ----------
def build_mat():
    # 圈外性核验:fresh_mat 3000(入过 ally_map/rebal_fresh 训练)的 gold SID 集
    trained_sids = set()
    for l in open(f"{P}/fresh_mat.jsonl"):
        g = parse_gold(json.loads(l)["output"])
        if g:
            trained_sids.add((g["dom"], *g["abc"]))
    rows, dropped = [], 0
    for l in open(f"{P}/mat_probe_fresh.jsonl"):
        r = json.loads(l)
        m = SID.search(r["gold"])
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3), m.group(4))
        if key in trained_sids:
            dropped += 1
            continue
        rows.append({"system": MAT_SYS[m.group(1)], "user": r["prompt"],
                     "gold": {"dom": m.group(1), "abc": list(key[1:])}, "meta": {"src": "mat_probe_fresh"}})
    print(f"  [mat_fresh] 与 fresh_mat(已入训) SID 重叠剔除 {dropped} 条")
    wl("dev_mat_fresh.jsonl", rows)

    mats = []
    for l in open(f"{P}/blocks_v1/block_mat.jsonl"):
        r = json.loads(l)
        g = parse_gold(r["output"])
        if g:
            mats.append({"system": r["instruction"], "user": r["input"], "gold": g, "meta": {"src": "block_mat"}})
    rng.shuffle(mats)
    wl("dev_mat_train.jsonl", mats[:300])


# ---------- rec ----------
def build_rec():
    by_dom = {}
    for l in open(f"{P}/rec_loo_v2.jsonl"):
        r = json.loads(l)
        g = parse_gold(r["output"])
        if g:
            by_dom.setdefault(g["dom"], []).append(
                {"system": r["instruction"], "user": r["input"], "gold": g, "meta": {"src": "rec_loo_v2"}})
    name_map = {"video": "video", "prod": "prod", "ad": "ad", "living": "live"}
    for dom, items in sorted(by_dom.items()):
        rng.shuffle(items)
        wl(f"dev_rec_{name_map[dom]}.jsonl", items[:1000])


# ---------- action / topic ----------
def build_action_topic():
    rows, seen = [], set()
    for f in ["r2_gold_v4", "r2_gold_local"]:
        for l in open(f"{P}/{f}.jsonl"):
            r = json.loads(l)
            k = r["input"][:300]
            if k in seen:
                continue
            seen.add(k)
            gold = ITEM.findall(r["output"].split("</think>")[-1])
            if not gold:
                continue
            rows.append({"system": "", "user": r["input"], "gold": sorted(set(gold)), "meta": {"src": f}})
    wl("dev_action.jsonl", rows)

    rows = []
    for l in open(f"{P}/u3_heldout.jsonl"):
        r = json.loads(l)
        try:
            events = json.loads(r["output"].split("</think>")[-1])["logic_chain"]["events"]
        except Exception:
            continue
        rows.append({"system": "", "user": r["input"],
                     "gold": [{"action": e.get("action", ""), "logic": e.get("logic", "")} for e in events],
                     "meta": {"src": "u3_heldout"}})
    wl("dev_topic.jsonl", rows)


# ---------- world ----------
def norm_stem(t):
    return re.sub(r"\s+", "", t)[:60]


def build_world(n_target=500):
    # 训练 MC 锚题干集(block_world 内 1816 条 + block_u2_mc,全部剔除)
    trained = set()
    for f in ["blocks_v1/block_world.jsonl", "blocks_v1/block_u2_mc.jsonl"]:
        for l in open(f"{P}/{f}"):
            r = json.loads(l)
            m = re.search(r"请回答以下问题：\s*(.+?)\nA[\.、]", r["input"], re.S)
            trained.add(norm_stem(m.group(1) if m else r["input"]))
    # CEval 仓库=52 学科目录 × parquet;逐科下 val parquet(带本地缓存)
    import subprocess
    import pandas as pd
    cache_dir = f"{OUT}/_ceval_val"
    os.makedirs(cache_dir, exist_ok=True)
    api = "https://hf-mirror.com/api/datasets/ceval/ceval-exam/tree/main"
    tree = json.loads(subprocess.run(["curl", "-sL", api], capture_output=True, text=True, check=True).stdout)
    subjects = [f["path"] for f in tree if f["type"] == "directory"]
    rows = []
    for sub in sorted(subjects):
        fp = f"{cache_dir}/{sub}.parquet"
        if not os.path.exists(fp):
            url = f"https://hf-mirror.com/datasets/ceval/ceval-exam/resolve/main/{sub}/val-00000-of-00001.parquet"
            subprocess.run(["curl", "-sL", "-o", fp, url], check=True)
        try:
            df = pd.read_parquet(fp)
        except Exception:
            os.remove(fp)
            continue
        for rec in df.to_dict("records"):
            q = str(rec.get("question", "")).strip()
            ans = str(rec.get("answer", "")).strip()
            if not q or ans not in "ABCD" or norm_stem(q) in trained:
                continue
            rows.append({"system": WORLD_SYS,
                         "user": WORLD_USER.format(q=q, a=str(rec["A"]).strip(), b=str(rec["B"]).strip(),
                                                   c=str(rec["C"]).strip(), d=str(rec["D"]).strip()),
                         "gold": ans, "meta": {"src": f"ceval_val/{sub}"}})
    print(f"  [world] CEval val 圈外候选 {len(rows)} 条(已剔训练锚重合;CEval 基本被 MC 锚烧光属预期)")
    # 主力源:CMMLU(独立数据集,test/*.csv in zip),补足圈外量
    zfp = f"{OUT}/_cmmlu.zip"
    if not os.path.exists(zfp):
        subprocess.run(["curl", "-sL", "-o", zfp,
                        "https://hf-mirror.com/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip"], check=True)
    import csv
    n_cm_drop = 0
    with zipfile.ZipFile(zfp) as z:
        for name in sorted(z.namelist()):
            if "test/" not in name or not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                for rec in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
                    q = (rec.get("Question") or rec.get("question") or "").strip()
                    ans = (rec.get("Answer") or rec.get("answer") or "").strip()
                    if not q or ans not in "ABCD":
                        continue
                    if norm_stem(q) in trained:
                        n_cm_drop += 1
                        continue
                    rows.append({"system": WORLD_SYS,
                                 "user": WORLD_USER.format(q=q, a=str(rec["A"]).strip(), b=str(rec["B"]).strip(),
                                                           c=str(rec["C"]).strip(), d=str(rec["D"]).strip()),
                                 "gold": ans, "meta": {"src": f"cmmlu/{os.path.basename(name)}"}})
    print(f"  [world] +CMMLU test(与训练锚重合剔 {n_cm_drop});合计候选 {len(rows)} 条")
    rng.shuffle(rows)
    wl("dev_world.jsonl", rows[:n_target])


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "mat"):
        build_mat()
    if only in ("all", "rec"):
        build_rec()
    if only in ("all", "action_topic"):
        build_action_topic()
    if only in ("all", "world"):
        try:
            build_world()
        except Exception as e:
            print(f"  [world] 构建失败(可单独重试 python3 {sys.argv[0]} world): {e}")
    print("done ->", OUT)
