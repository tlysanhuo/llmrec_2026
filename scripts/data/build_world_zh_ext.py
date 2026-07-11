#!/usr/bin/env python3
"""build_world_zh_ext.py — world_zh 的合规官方源独立增补(2026-07-05)。

源: data/hf_full/data/OneReason_General(158 shard / 152,005 条,100% stepfun_general,官方发布)。
定位: build_world_zh.py 清洗口径**排除掉**的两层高价值中文数据,单独成文件,
      不改动既有 world_zh.jsonl(其有 566 行历史过滤血统问题,保持只读)。

两层口径(与 build_world_zh.py 的排除条件互补):
  层① multiturn_first_qa — 非单轮(len(messages)!=2 或角色结构非 user+assistant),
      但全对话主体中文(cn_ratio>0.5)。
      拼接策略(写明): 取第一对相邻 user→assistant(跳过开头 system),history=[]。
      理由: 首轮 QA 自包含、无上下文依赖;后续轮次常有话题漂移/依赖前文,不取。
      对该 QA 对再施质量过滤: 对内 cn_ratio>=0.5 且 50<=len(output)<=8000。
  层② long_answer — 单轮主体中文(cn_ratio>=0.5),assistant 长度 4000<len<=8000
      (world_zh 上限 4000 截掉的长回答,8000 封顶防超长挤爆 cutoff_len)。

附加过滤(两层共用):
  - 身份污染: output 含 阶跃星辰/StepFun/我是Step 的样本剔除(教模型自称 Step 有害)。
  - 去重: 归一化 prompt(去全部空白)——文件内去重 + 与 world_zh.jsonl 16237 条零重叠(硬校验)。

输出:
  data/processed/world_zh_ext.jsonl        alpaca,与 world_zh 同 4 键 {instruction,input,output,history}
  data/processed/world_zh_ext.jsonl.meta.jsonl  逐行对齐 {uuid,layer,shard} 溯源
固定 seed=2026 末端 shuffle;扫描顺序 sorted(shard),全流程确定可复现(平台 Q7)。

用法: python scripts/data/build_world_zh_ext.py
"""
import argparse
import glob
import json
import random
import re

ROOT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
CJK = re.compile(r"[一-鿿]")
IDENTITY = re.compile(r"阶跃星辰|StepFun|我是\s*\**Step\b", re.I)


def cn_ratio(s):
    if not s:
        return 0.0
    return len(CJK.findall(s)) / len(s)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def norm(s):
    return re.sub(r"\s+", "", s)


def first_qa(msgs):
    """取第一对相邻 user→assistant(跳过开头 system);找不到返回 None。"""
    for i in range(len(msgs) - 1):
        if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
            u = extract_text(msgs[i]["content"]).strip()
            a = extract_text(msgs[i + 1]["content"]).strip()
            if u and a:
                return u, a
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=f"{ROOT}/data/hf_full/data/OneReason_General")
    ap.add_argument("--world_zh", default=f"{ROOT}/data/processed/world_zh.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/data/processed/world_zh_ext.jsonl")
    ap.add_argument("--cn_ratio", type=float, default=0.5)
    ap.add_argument("--long_min", type=int, default=4000)
    ap.add_argument("--long_max", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    # 既有 world_zh 的归一化 prompt 集合(零重叠硬约束)
    base_prompts = set()
    for line in open(args.world_zh, encoding="utf-8"):
        base_prompts.add(norm(json.loads(line)["input"]))
    print(f"world_zh 基线 prompt 集: {len(base_prompts)}", flush=True)

    fs = sorted(glob.glob(f"{args.src_dir}/*.parquet"))
    print(f"扫描 {len(fs)} shard ...", flush=True)

    seen = set()
    kept, meta = [], []
    stat = {"total": 0, "multi_zh_seen": 0, "long_zh_seen": 0,
            "multi_kept": 0, "long_kept": 0,
            "drop_no_qa": 0, "drop_pair_quality": 0, "drop_identity": 0,
            "drop_dup_internal": 0, "drop_dup_world_zh": 0}
    for fi, f in enumerate(fs):
        shard = f.rsplit("/", 1)[-1]
        t = pq.read_table(f, columns=["uuid", "messages"])
        for uid, m in zip(t["uuid"].to_pylist(), t["messages"].to_pylist()):
            stat["total"] += 1
            try:
                msgs = json.loads(m)
            except Exception:
                continue
            single = (len(msgs) == 2 and msgs[0].get("role") == "user"
                      and msgs[1].get("role") == "assistant")
            if single:
                u = extract_text(msgs[0]["content"]).strip()
                a = extract_text(msgs[1]["content"]).strip()
                if not u or not a or cn_ratio(u + a) < args.cn_ratio:
                    continue
                if not (args.long_min < len(a) <= args.long_max):
                    continue  # <=4000 已在 world_zh;>8000 不要
                stat["long_zh_seen"] += 1
                layer = "long_answer"
            else:
                full = " ".join(extract_text(x.get("content")) for x in msgs)
                if cn_ratio(full) <= args.cn_ratio:
                    continue
                stat["multi_zh_seen"] += 1
                qa = first_qa(msgs)
                if qa is None:
                    stat["drop_no_qa"] += 1
                    continue
                u, a = qa
                if cn_ratio(u + a) < args.cn_ratio or not (50 <= len(a) <= args.long_max):
                    stat["drop_pair_quality"] += 1
                    continue
                layer = "multiturn_first_qa"
            if IDENTITY.search(a) or IDENTITY.search(u):
                stat["drop_identity"] += 1
                continue
            k = norm(u)
            if k in seen:
                stat["drop_dup_internal"] += 1
                continue
            if k in base_prompts:
                stat["drop_dup_world_zh"] += 1
                continue
            seen.add(k)
            kept.append({"instruction": "", "input": u, "output": a, "history": []})
            meta.append({"uuid": uid, "layer": layer, "shard": shard})
            stat["multi_kept" if layer == "multiturn_first_qa" else "long_kept"] += 1
        if (fi + 1) % 40 == 0:
            print(f"  {fi+1}/{len(fs)} shard, 已保留 {len(kept)}", flush=True)

    idx = list(range(len(kept)))
    random.Random(args.seed).shuffle(idx)
    kept = [kept[i] for i in idx]
    meta = [meta[i] for i in idx]

    # 硬校验: 与 world_zh 零重叠
    overlap = {norm(r["input"]) for r in kept} & base_prompts
    assert not overlap, f"与 world_zh 重叠 {len(overlap)} 条,违反零重叠约束"

    with open(args.out, "w", encoding="utf-8") as g:
        for r in kept:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".meta.jsonl", "w", encoding="utf-8") as g:
        for r in meta:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(stat, ensure_ascii=False, indent=2))
    print(f"[OK] {args.out}: {len(kept)} 条 (层① {stat['multi_kept']} + 层② {stat['long_kept']}),与 world_zh 零重叠已断言", flush=True)


if __name__ == "__main__":
    main()
