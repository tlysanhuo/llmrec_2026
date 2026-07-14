#!/usr/bin/env python3
"""build_fresh_mat.py — 从 Pid2Caption⋈Pid2Sid 构造【训练集外】新鲜物料 desc→SID 样本。

动机(2026-07-04,rebal_mat_ep3=0.8454 教训):重复上采样=毒药,物料要"多样曝光"。
Pid2Caption(2106万)⋈Pid2Sid(3591万)= 免费的 desc→SID 对,查表即得,零 teacher。
样本 prompt 措辞逐字对齐种子物料样本(四域各自的官方 instruction 模板)。

清洗规则(队友17G调研 §2.2.3/§2.3):
  - goods: caption<10字符=short_noise(6.28%)过滤;caption 是"商品标题"形态
  - video/ad: 空串(0.1-0.2%)过滤;>500字硬截断(离群)
  - live: caption 是 taglist(100%),按队友建议 I 转"关键词序列"形态
  - 去重:同 caption 多 pid 只保留 1 个(队友建议 M,goods 8.78% 行共享 caption)
  - 与种子物料样本去重:种子 desc2token 的 SID 集合排除(保证"训练集外")

用法:
  python scripts/data/build_fresh_mat.py --out data/processed/fresh_mat.jsonl \
      [--n_per_dom 1500 --shards_cap 8 --shards_sid 30 --seed 2026]
"""
import argparse
import ast
import glob
import json
import random
import re

# 与种子物料样本逐字一致的 instruction(2026-07-04 从 data_final 抽取核对)
INS = {
    "live":  "你是一名专业的主播token生成助手，请根据主播的形象、内容与风格描述生成匹配的主播token。",
    "goods": "你是一名专业的电商商品token生成助手，请根据商品描述生成匹配的商品token。",
    "video/video": "你是一名专业的短视频token生成助手，请根据短视频的画面、主体、动作、场景与风格描述生成匹配的短视频token。",
    "video/ad": "你擅长根据广告内容、风格和主题描述，输出对应的广告token。",
}
LEAD = {
    "live":  "下面是一段主播描述，请返回匹配的主播token：",
    "goods": "请根据以下商品描述生成匹配的商品token：",
    "video/video": "请分析这段短视频内容，并生成对应的短视频token：",
    "video/ad": "根据以下广告描述还原其token，只输出目标广告token：",
}
BEGIN = {"live": "living", "goods": "prod", "video/video": "video", "video/ad": "ad"}


def clean_caption(dom, cap):
    if not cap or not cap.strip():
        return None
    cap = cap.strip()
    if dom == "live":
        # taglist → 关键词串
        try:
            tags = ast.literal_eval(cap)
            if not isinstance(tags, list) or len(tags) < 3:
                return None
            return "、".join(str(t) for t in tags[:10])
        except Exception:
            return None
    if len(cap) < 10:
        return None
    if cap.startswith("[") and cap.endswith("]"):
        return None
    return cap[:500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_Pid2Caption")
    ap.add_argument("--sid_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_Pid2Sid")
    ap.add_argument("--seed_data", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/data_final.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_dom", type=int, default=1500)
    ap.add_argument("--shards_cap", type=int, default=8)
    ap.add_argument("--shards_sid", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    import pyarrow.parquet as pq

    # 种子物料样本的 SID 集合(排除,保证训练集外)
    seed_sids = set()
    SIDPAT = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
    for line in open(args.seed_data):
        r = json.loads(line)
        ins = r.get("instruction", "")
        if "token" in ins and ("生成助手" in ins or "输出匹配" in ins or "输出对应" in ins):
            m = SIDPAT.search(r["output"])
            if m:
                seed_sids.add((m.group(1), m.group(2), m.group(3)))
    print(f"种子物料 SID 排除集: {len(seed_sids)}")

    # pid→sid(抽 shard 建局部索引,双向都是随机 shard,交上多少用多少)
    pid2sid = {}
    sid_files = sorted(glob.glob(f"{args.sid_dir}/*.parquet"))
    rng.shuffle(sid_files)
    for f in sid_files[: args.shards_sid]:
        t = pq.read_table(f)
        for p, d, s in zip(t["pid"].to_pylist(), t["domain"].to_pylist(), t["sid_three"].to_pylist()):
            if d in INS and s:
                try:
                    v = ast.literal_eval(s) if isinstance(s, str) else s
                    pid2sid[p] = (d, tuple(str(int(float(x))) for x in v))
                except Exception:
                    pass
    print(f"pid2sid 局部索引: {len(pid2sid):,}")

    buckets = {d: [] for d in INS}
    seen_caption = set()
    seen_sid = set()
    cap_files = sorted(glob.glob(f"{args.cap_dir}/*.parquet"))
    rng.shuffle(cap_files)
    for f in cap_files[: args.shards_cap]:
        t = pq.read_table(f)
        for p, d, cap in zip(t["pid"].to_pylist(), t["domain"].to_pylist(), t["caption"].to_pylist()):
            if d not in INS or len(buckets[d]) >= args.n_per_dom * 2:
                continue
            got = pid2sid.get(p)
            if not got or got[0] != d:
                continue
            sid = got[1]
            if sid in seed_sids or sid in seen_sid:
                continue  # 训练集外 + SID 去重(多样曝光)
            desc = clean_caption(d, cap)
            if not desc:
                continue
            ck = hash(desc)
            if ck in seen_caption:
                continue  # caption 去重(队友建议M)
            seen_caption.add(ck)
            seen_sid.add(sid)
            dom = BEGIN[d]
            gold = f"<|{dom}_begin|><s_a_{sid[0]}><s_b_{sid[1]}><s_c_{sid[2]}>"
            buckets[d].append({
                "instruction": INS[d],
                "input": LEAD[d] + desc,
                "output": f"<think>\n</think>\n{gold}",
                "history": [],
            })

    out = []
    for d, v in buckets.items():
        rng.shuffle(v)
        out.extend(v[: args.n_per_dom])
        print(f"  {d}: 候选{len(v)} → 取{min(len(v), args.n_per_dom)}")
    rng.shuffle(out)
    with open(args.out, "w") as g:
        for r in out:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"共 {len(out)} 条 → {args.out}")


if __name__ == "__main__":
    main()
