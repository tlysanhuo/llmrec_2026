#!/usr/bin/env python3
"""build_item_index.py — 从官方 17GB 原始数据建 pid -> sid / caption / tag 全局索引。

产出(存运行区 assets/derived/index/):
  pid2sid.parquet   : columns [key, s_a, s_b, s_c]   key = f"{domain}|{pid}"
  pid2caption.parquet: columns [key, caption]
  pid2tag.parquet   : columns [key, tag_lv3]

domain 归一化到 itemic-token 前缀域:
  video/video -> video ; video/ad -> ad ; goods -> prod ; live -> living
sid_three = [s_a, s_b, s_c] (float -> int)。

用法:
  python scripts/data/build_item_index.py            # 全量
  python scripts/data/build_item_index.py --limit 5  # 每类只扫 5 分片(快速冒烟)
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

HF = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw"
OUT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/derived/index"

# 原始 domain -> itemic-token 前缀域
DOMAIN_MAP = {
    "video/video": "video",
    "video/ad": "ad",
    "goods": "prod",
    "live": "living",
}


def norm_domain(d):
    return DOMAIN_MAP.get(d, d)


def make_key(domain_col, pid_col):
    # 归一化域 + pid 组成联合键
    return norm_domain(domain_col) + "|" + str(int(pid_col))


def build_sid(shards, out_path):
    frames = []
    t0 = time.time()
    for i, f in enumerate(shards):
        df = pd.read_parquet(f, columns=["pid", "domain", "sid_three"])
        df = df[df["sid_three"].notna()].copy()
        df["key"] = [norm_domain(d) + "|" + str(int(p)) for d, p in zip(df["domain"], df["pid"])]
        sid = df["sid_three"].tolist()
        df["s_a"] = [int(x[0]) for x in sid]
        df["s_b"] = [int(x[1]) for x in sid]
        df["s_c"] = [int(x[2]) for x in sid]
        frames.append(df[["key", "s_a", "s_b", "s_c"]])
        if (i + 1) % 20 == 0:
            print(f"  [sid] {i+1}/{len(shards)} shards, {time.time()-t0:.0f}s", file=sys.stderr)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("key")
    out.to_parquet(out_path, index=False)
    print(f"[sid] {len(out):,} unique keys -> {out_path}", file=sys.stderr)
    return out


def build_kv(shards, val_col, out_path, tag):
    frames = []
    t0 = time.time()
    for i, f in enumerate(shards):
        df = pd.read_parquet(f, columns=["pid", "domain", val_col])
        df = df[df[val_col].notna()].copy()
        df["key"] = [norm_domain(d) + "|" + str(int(p)) for d, p in zip(df["domain"], df["pid"])]
        frames.append(df[["key", val_col]])
        if (i + 1) % 20 == 0:
            print(f"  [{tag}] {i+1}/{len(shards)} shards, {time.time()-t0:.0f}s", file=sys.stderr)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("key")
    out.to_parquet(out_path, index=False)
    print(f"[{tag}] {len(out):,} unique keys -> {out_path}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每类只扫前 N 分片(0=全量)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    def shards(name):
        fs = sorted(glob.glob(f"{HF}/{name}/*.parquet"))
        return fs[: args.limit] if args.limit else fs

    build_sid(shards("OneReason_Pid2Sid"), f"{OUT}/pid2sid.parquet")
    build_kv(shards("OneReason_Pid2Caption"), "caption", f"{OUT}/pid2caption.parquet", "cap")
    build_kv(shards("OneReason_Pid2Tag"), "tag_lv3", f"{OUT}/pid2tag.parquet", "tag")
    print("[done] index built at", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
