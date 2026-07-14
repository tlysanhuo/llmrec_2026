#!/usr/bin/env python3
"""build_world_zh.py — 从官方 General 数据洗出中文通识 SFT 数据(懂世界维度)。

背景:懂世界是确定性得分点(momo实测混入+0.024,全群在洗)。官方原始素材
`hf_full/data/OneReason_General`(158 shard/15.8万条,15%主体中文),我们本地就有。
论文§5.6:通用数据做过清洗;群里选手也在洗中文子集(克西2500/滿栀3000条)。

清洗规则(固定seed,可复现,满足平台Q7复现审核):
  1. 只保留 messages 可解析、恰好 user+assistant 单轮
  2. 主体中文:user+assistant 合并文本中文字符占比 > CN_RATIO(默认0.5,比探查的0.3更严保质量)
  3. 长度过滤:assistant 文本在 [MIN_LEN, MAX_LEN](默认50-4000字符,去掉过短/过长)
  4. 去重:按 user 文本 md5
  5. 保留原文 CoT(<think>...</think>)——官方数据带CoT,不剥(recipe3剥CoT已证灾难)
输出 alpaca 格式:{instruction:"", input:user, output:assistant(含think), history:[]}
  → 与种子 data_final.jsonl 同格式,可直接混入训练。

用法:
  python scripts/data/build_world_zh.py --out /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/world_zh.jsonl \
      --cn_ratio 0.5 --min_len 50 --max_len 4000 [--max_out 20000]
"""
import argparse
import glob
import hashlib
import json
import re

CJK = re.compile(r"[一-鿿]")


def cn_ratio(s):
    if not s:
        return 0.0
    return len(CJK.findall(s)) / len(s)


def extract_text(content):
    """content 可能是 str 或 [{'type':'text','text':...}]。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_General")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cn_ratio", type=float, default=0.5)
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=4000)
    ap.add_argument("--max_out", type=int, default=0, help="0=不限;>0 固定seed采样这么多条")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    fs = sorted(glob.glob(f"{args.src_dir}/*.parquet"))
    print(f"扫描 {len(fs)} shard ...", flush=True)

    seen = set()
    kept = []
    n_total = 0
    for fi, f in enumerate(fs):
        t = pq.read_table(f, columns=["messages"])
        for m in t["messages"].to_pylist():
            n_total += 1
            try:
                msgs = json.loads(m)
            except Exception:
                continue
            if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
                continue
            u = extract_text(msgs[0]["content"]).strip()
            a = extract_text(msgs[1]["content"]).strip()
            if not u or not a:
                continue
            if cn_ratio(u + a) < args.cn_ratio:
                continue
            if not (args.min_len <= len(a) <= args.max_len):
                continue
            h = hashlib.md5(u.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kept.append({"instruction": "", "input": u, "output": a, "history": []})
        if (fi + 1) % 20 == 0:
            print(f"  {fi+1}/{len(fs)} shard, 已保留 {len(kept)}", flush=True)

    if args.max_out and len(kept) > args.max_out:
        import random
        random.Random(args.seed).shuffle(kept)
        kept = kept[: args.max_out]
        print(f"采样至 {args.max_out} 条(seed={args.seed})", flush=True)

    with open(args.out, "w") as g:
        for r in kept:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成:扫描 {n_total} 条 → 保留 {len(kept)} 条 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
