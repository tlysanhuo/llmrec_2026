#!/usr/bin/env python3
"""build_p1_bucket_discrim.py — P1「同桶精细分辨」数据集(2026-07-05)。

设计依据(docs/eval_log_mining_20260705.md A1/A3 + P1 设计稿):
  懂物料评测头部题的瓶颈不是 s_a——模型能找对 (s_a,s_b) 桶,但头部桶内有数百~上万个
  真实 item(证据:video 桶 (5254,6442) 全库 1627 个、(2367,6234) 8117 个;A3 热门 hub
  s_b_6234 兜底),beam64 在桶内退化为 s_c 级抽签。对治 = 同一 (domain,s_a,s_b) 桶内
  喂 15-25 个 caption 互异的真实 item,强迫模型把 desc 的细粒度差异绑定到 s_c 上。
  沿用 07-04 铁律:上采样必须造新样本不许纯重复——本数据全部来自训练集外新 item
  (排除种子懂物料与 fresh_mat 已有 SID),是"多样曝光"在桶内维度的定向加密。

构造:
  - 全量扫 OneReason_Pid2Sid(198 shards, 3591 万行)聚合 (domain,s_a,s_b) 桶;
    domain 归一化 live→living, goods→prod, video/video→video, video/ad→ad。
  - 头部桶 = 桶内真实 item 数 ≥50;按桶大小加权、无放回采样 ~250 个训练桶
    (配额 video 100 / prod 80 / ad 45 / living 25,按可用性回退),
    必含证据桶 (video,2367,6234)、(video,5254,6442) 与 prod 最大桶。
  - val 桶与 train 桶完全不相交(独立再采 ~10 桶,150 条)。
  - 每桶 15-25 个互不相同 item:caption 清洗逐字复用 build_fresh_mat.py
    (live taglist→关键词、<10字过滤、500字截断、同caption全局去重),
    桶内再按 caption 首 20 字符去重;SID 三元组全局唯一。
  - prompt 逐字复用种子对应域 instruction 模板(与 build_fresh_mat.py 同一组,
    07-04 已从 data_final 逐字核对);nothink 直出:prompt 尾 /no_think + 空 think +
    域 begin + 三 SID token——与评测 grounding 的 beam 直出通路一致,且满足全库
    不变量 /no_think⇔空think。
  - 排除集:data/懂物料part1-7.jsonl 的全部 SID(防背原文)∪ fresh_mat.jsonl 的
    6000 SID(防重复);seed=2026。

预登记预测(必须先写后验,物料维 beam 确定性=零噪声硬读数):
  - 若 s_c 在评测中有语义:懂物料 +0.02~0.04(头部题从 1/桶大小 抽签 → 桶内可分辨)。
  - 若无效(±0.005 内):反向定案 s_c 在评测 grounding 中近似无语义,P1 路线关闭,
    资源转 P2(尾部 s_a 覆盖)。
  - 风险预案:纯 nothink 物料样本占比升高可能轻微影响 rec 四域 think 通路——本数据
    只应与主配方混合(≤10%),不单独成方。

用法:
  python scripts/data/build_p1_bucket_discrim.py \
      [--out data/processed/p1_bucket_discrim.jsonl --seed 2026]
产出:
  - {out}                 训练集 ~5000 条(alpaca 格式,与 fresh_mat 同构)
  - {out%.jsonl}_val.jsonl  val 150 条(独立桶)
  - {out}.meta.jsonl      每条样本的 pid/domain/bucket/sid 侧写(供质检全量查表复核)
"""
import argparse
import ast
import collections
import glob
import json
import random
import re

import numpy as np

# ---- 与种子物料样本逐字一致的模板(build_fresh_mat.py 同源,07-04 已核对) ----
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
RAW2NORM = {"live": "living", "goods": "prod", "video/video": "video", "video/ad": "ad"}
NORM2RAW = {v: k for k, v in RAW2NORM.items()}
SIDPAT = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
K = 8404  # 编码基数,> 码本最大索引


def clean_caption(raw_dom, cap):
    """逐字复用 build_fresh_mat.py 的清洗逻辑。"""
    if not cap or not cap.strip():
        return None
    cap = cap.strip()
    if raw_dom == "live":
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


def load_exclusion(seed_glob, fresh_path):
    excl = set()
    n_seed = 0
    for fp in sorted(glob.glob(seed_glob)):
        for line in open(fp):
            row = json.loads(line)[0]
            m = SIDPAT.search(row["response"])
            if m:
                excl.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                n_seed += 1
    n_after_seed = len(excl)
    for line in open(fresh_path):
        r = json.loads(line)
        m = SIDPAT.search(r["output"])
        if m:
            excl.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    print(f"排除集: 种子懂物料 {n_seed} 行→{n_after_seed} SID;+fresh_mat 后共 {len(excl)} SID")
    return excl


def main():
    ap = argparse.ArgumentParser()
    base = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
    ap.add_argument("--sid_dir", default=f"{base}/data/hf_full/data/OneReason_Pid2Sid")
    ap.add_argument("--cap_dir", default=f"{base}/data/hf_full/data/OneReason_Pid2Caption")
    ap.add_argument("--seed_mat_glob", default=f"{base}/data/懂物料part*.jsonl")
    ap.add_argument("--fresh_mat", default=f"{base}/data/processed/fresh_mat.jsonl")
    ap.add_argument("--out", default=f"{base}/data/processed/p1_bucket_discrim.jsonl")
    ap.add_argument("--min_bucket", type=int, default=50)
    ap.add_argument("--n_keep_per_bucket", type=int, default=300)  # 候选池(caption覆盖率~59%)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    import pyarrow.parquet as pq

    DOMS = {"live": 0, "goods": 1, "video/video": 2, "video/ad": 3}
    INV = {0: "living", 1: "prod", 2: "video", 3: "ad"}

    # ---------- Pass 1: 全量聚合 (domain,s_a,s_b) 桶 ----------
    print("Pass1: 全量扫 Pid2Sid 聚合桶 ...")
    agg = collections.Counter()
    sid_files = sorted(glob.glob(f"{args.sid_dir}/*.parquet"))
    for f in sid_files:
        t = pq.read_table(f, columns=["domain", "sid_three"])
        dom = t["domain"].to_numpy(zero_copy_only=False)
        col = t["sid_three"].combine_chunks()
        offs = np.diff(col.offsets.to_numpy())
        vals = col.values.to_numpy(zero_copy_only=False)
        ok = offs == 3
        v = vals[np.repeat(ok, offs)].reshape(-1, 3).astype(np.int64)
        d = np.array([DOMS.get(x, -1) for x in dom[ok]], dtype=np.int64)
        m = d >= 0
        key = d[m] * K * K + v[m, 0] * K + v[m, 1]
        u, c = np.unique(key, return_counts=True)
        for kk, cc in zip(u.tolist(), c.tolist()):
            agg[kk] += cc
    print(f"  桶总数 {len(agg):,},item 总数 {sum(agg.values()):,}")

    head = {d: [] for d in INV.values()}  # dom -> [(key,count)]
    for k, c in agg.items():
        if c >= args.min_bucket:
            head[INV[k // (K * K)]].append((k, c))
    for d, v in head.items():
        v.sort(key=lambda x: -x[1])
        print(f"  头部桶(≥{args.min_bucket}) {d}: {len(v)} 个,覆盖 item {sum(x[1] for x in v):,}")

    # ---------- 选桶:必含证据桶 + 各域配额按桶大小加权无放回 ----------
    def enc(dom, sa, sb):
        return DOMS[NORM2RAW[dom]] * K * K + sa * K + sb

    forced = []
    for dom, sa, sb in [("video", 2367, 6234), ("video", 5254, 6442)]:
        kk = enc(dom, sa, sb)
        if agg.get(kk, 0) >= args.min_bucket:
            forced.append(kk)
            print(f"  证据桶纳入: ({dom},{sa},{sb}) size={agg[kk]}")
        else:
            print(f"  ⚠️ 证据桶缺席: ({dom},{sa},{sb}) size={agg.get(kk,0)}")
    if head["prod"]:
        forced.append(head["prod"][0][0])  # prod 最大桶
        kk = head["prod"][0][0]
        print(f"  prod 最大桶纳入: sa={kk % (K*K) // K}, sb={kk % K}, size={agg[kk]}")

    QUOTA_TRAIN = {"video": 100, "prod": 80, "ad": 45, "living": 25}   # ≈250
    QUOTA_VAL = {"video": 4, "prod": 3, "ad": 2, "living": 1}          # ≈10 桶×15 = 150

    def weighted_pick(cands, n, exclude):
        pool = [(k, c) for k, c in cands if k not in exclude]
        n = min(n, len(pool))
        if n == 0:
            return []
        keys = np.array([k for k, _ in pool])
        w = np.array([c for _, c in pool], dtype=np.float64)
        idx = nprng.choice(len(pool), size=n, replace=False, p=w / w.sum())
        return keys[idx].tolist()

    train_buckets, val_buckets = set(forced), set()
    for d in ["video", "prod", "ad", "living"]:
        already = sum(1 for k in forced if INV[k // (K * K)] == d)
        train_buckets.update(weighted_pick(head[d], QUOTA_TRAIN[d] - already, train_buckets))
    for d in ["video", "prod", "ad", "living"]:
        val_buckets.update(weighted_pick(head[d], QUOTA_VAL[d], train_buckets | val_buckets))
    print(f"  train 桶 {len(train_buckets)},val 桶 {len(val_buckets)}(不相交:{not (train_buckets & val_buckets)})")

    # ---------- Pass 2: 收集选中桶的 (pid, sid) 候选,蓄水池采样 ----------
    print("Pass2: 收集选中桶的 pid 候选 ...")
    wanted = train_buckets | val_buckets
    cand = collections.defaultdict(list)  # key -> [(pid, sa,sb,sc)]
    n_seen = collections.Counter()
    for f in sid_files:
        t = pq.read_table(f)
        pids = t["pid"].to_pylist()
        doms = t["domain"].to_pylist()
        sids = t["sid_three"].to_pylist()
        for p, dm, s in zip(pids, doms, sids):
            di = DOMS.get(dm, -1)
            if di < 0 or not s or len(s) != 3:
                continue
            sa, sb, sc = (int(x) for x in s)
            kk = di * K * K + sa * K + sb
            if kk not in wanted:
                continue
            n_seen[kk] += 1
            lst = cand[kk]
            if len(lst) < args.n_keep_per_bucket:
                lst.append((p, sa, sb, sc))
            else:  # reservoir
                j = rng.randrange(n_seen[kk])
                if j < args.n_keep_per_bucket:
                    lst[j] = (p, sa, sb, sc)
    pid_want = {p: kk for kk, lst in cand.items() for (p, _, _, _) in lst}
    print(f"  候选 pid {len(pid_want):,}(桶 {len(cand)})")

    # ---------- Pass 3: 扫 Pid2Caption 取 caption ----------
    print("Pass3: 扫 Pid2Caption ...")
    pid_cap = {}
    for f in sorted(glob.glob(f"{args.cap_dir}/*.parquet")):
        t = pq.read_table(f)
        for p, dm, cp in zip(t["pid"].to_pylist(), t["domain"].to_pylist(), t["caption"].to_pylist()):
            if p in pid_want and p not in pid_cap and dm in DOMS:
                # caption 行域须与桶域一致
                if DOMS[dm] == pid_want[p] // (K * K):
                    pid_cap[p] = (dm, cp)
    print(f"  命中 caption {len(pid_cap):,}/{len(pid_want):,}")

    # ---------- 组装样本 ----------
    excl = load_exclusion(args.seed_mat_glob, args.fresh_mat)
    seen_sid, seen_cap_global = set(), set()

    def build_split(buckets, tag):
        rows, metas = [], []
        per_bucket = collections.Counter()
        for kk in sorted(buckets):
            dom_i = kk // (K * K)
            raw = NORM2RAW[INV[dom_i]]
            target = rng.randint(15, 25)
            lst = list(cand.get(kk, []))
            rng.shuffle(lst)
            seen_cap20 = set()
            for (p, sa, sb, sc) in lst:
                if per_bucket[kk] >= target:
                    break
                if (sa, sb, sc) in excl or (sa, sb, sc) in seen_sid:
                    continue
                got = pid_cap.get(p)
                if not got:
                    continue
                desc = clean_caption(raw, got[1])
                if not desc:
                    continue
                h = hash(desc)
                if h in seen_cap_global or desc[:20] in seen_cap20:
                    continue
                seen_cap_global.add(h)
                seen_cap20.add(desc[:20])
                seen_sid.add((sa, sb, sc))
                gold = f"<|{BEGIN[raw]}_begin|><s_a_{sa}><s_b_{sb}><s_c_{sc}>"
                rows.append({
                    "instruction": INS[raw],
                    "input": LEAD[raw] + desc + "/no_think",
                    "output": f"<think>\n</think>\n{gold}",
                    "history": [],
                })
                metas.append({"split": tag, "pid": p, "domain": INV[dom_i],
                              "bucket": [sa, sb], "sid": [sa, sb, sc],
                              "bucket_size": agg[kk]})
                per_bucket[kk] += 1
        return rows, metas, per_bucket

    train_rows, train_meta, pb_train = build_split(train_buckets, "train")
    # val 只取 150 条
    val_rows, val_meta, pb_val = build_split(val_buckets, "val")
    order = list(range(len(val_rows)))
    rng.shuffle(order)
    order = order[:150]
    val_rows = [val_rows[i] for i in order]
    val_meta = [val_meta[i] for i in order]

    idx = list(range(len(train_rows)))
    rng.shuffle(idx)
    train_rows = [train_rows[i] for i in idx]
    train_meta = [train_meta[i] for i in idx]

    val_path = args.out.replace(".jsonl", "_val.jsonl")
    with open(args.out, "w") as g:
        for r in train_rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as g:
        for r in val_rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out + ".meta.jsonl", "w") as g:
        for r in train_meta + val_meta:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")

    dom_dist = collections.Counter(m["domain"] for m in train_meta)
    print(f"train {len(train_rows)} 条 → {args.out}")
    print(f"  域分布: {dict(dom_dist)}")
    print(f"  实际使用桶数 train={len(pb_train)}, 每桶样本数 min={min(pb_train.values())}, "
          f"max={max(pb_train.values())}, mean={sum(pb_train.values())/len(pb_train):.1f}")
    print(f"val {len(val_rows)} 条({len(set(tuple(m['bucket'])+(m['domain'],) for m in val_meta))} 桶)→ {val_path}")
    print(f"meta → {args.out}.meta.jsonl")


if __name__ == "__main__":
    main()
