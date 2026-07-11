#!/usr/bin/env python3
"""build_p2_tail_cover.py — P2「尾部 s_a 覆盖」数据集(2026-07-05)。

设计依据(docs/eval_log_mining_20260705.md A2/A4 + P2 设计稿):
  懂物料评测尾部题的失败模式是 s_a 撒网发散——beam64 的 distinct s_a 达 22-28 个,
  模型对尾部语义没有稳定的 desc→s_a 映射(A2);长尾内容题(剧集,真实邻域 s_a=2832)
  模型答案与真实邻域零交集(A4)。对治 = 对"训练曝光度 0-1 次"的每个 (domain,s_a)
  补 2-3 个 caption 互异的真实 item 的 desc→SID 样本,把尾部 s_a 从"从未见过输出侧"
  变成"至少见过 2-3 个多样描述"。沿用 07-04 铁律:上采样必须造新样本不许纯重复——
  本数据全部来自训练集外新 item,是"多样曝光"在 s_a 维度的定向扩边。

训练曝光度定义(P1/P2 互补不重叠的关键):
  exposure(domain, s_a) = fk_fuse.jsonl(41529 条,当前主力配方底料)output 字段中
  <|{dom}_begin|><s_a_N> 的出现次数 + p1_bucket_discrim(train+val)gold 的同型计数。
  P1 训练的头部桶 s_a 每桶出现 15-25 次 ⇒ 曝光度远超 1,自动被 P2 排除;P2 只做
  曝光 0-1 次的尾部 ⇒ 两者按定义零重叠。
  (fk_fuse 输入侧历史 SID 不计:曝光度衡量的是"输出侧被监督过",与生成通路对齐。)

07-05 预扫描事实(本脚本 Pass1 会复算并打印):
  s_a 空间 video 5777 / prod 4279 / ad 2971 / living 802;
  尾部(曝光≤1) video 3961 / prod 1802 / ad 2147 / living 464;
  其中源表内 item≥2 的可用尾部 video 3448 / prod 1741 / ad 1800 / living 343。
  证据 s_a(全部 video 域,已从 parsed_logs.json 9 模型 grounding 样例核实):
  2832(A4 剧集题真实邻域)、1383/606/5983(A2 尾部题 beam 撒网命中)、
  287/1173/1509/7328(A2/A4 原始报告提及)——若在源表存在必纳入 train。

构造:
  - Pass1 全量扫 OneReason_Pid2Sid(199 shards, 3591 万行)聚合 (domain,s_a) 计数;
    domain 归一化 live→living, goods→prod, video/video→video, video/ad→ad。
  - 尾部 s_a = exposure≤1 且源表 item≥2(证据 s_a 豁免 item≥2,存在即纳入)。
  - val s_a 池与 train s_a 池完全不相交(先抽 val 池,train 绝不触碰);val 150 条。
  - 每个入选 s_a 采 2-3 个 caption 互异 item(优先不同 s_b,扩大子空间覆盖);
    caption 清洗逐字复用 build_fresh_mat.py(live taglist→关键词、<10字过滤、
    500字截断、同caption全局去重),s_a 内再按 caption 首 20 字符去重。
  - 域配额(train)video 2900 / prod 2300 / ad 1400 / living 500 ≈ 7100,
    依据 = 各域尾部规模(video/ad 尾部最大)×评测四域均等权重 ⇒ video/prod 为主、
    ad 居中、living 按可用性(尾部仅 343 个可用 s_a)取小头。
  - prompt 逐字复用种子对应域 instruction 模板(与 build_fresh_mat.py /
    build_p1_bucket_discrim.py 同一组);nothink 直出:prompt 尾 /no_think +
    空 think + 域 begin + 三 SID token——与评测 grounding beam 直出通路一致,
    且满足全库不变量 /no_think⇔空think。三兄弟(fresh_mat/P1/P2)格式完全同构。
  - 排除集:data/懂物料part1-7.jsonl 全部 SID ∪ fresh_mat.jsonl 6000 SID ∪
    p1_bucket_discrim(train+val)4829 SID;SID 三元组全局唯一;seed=2026。

预登记预测(必须先写后验;物料维 beam 确定性=零噪声硬读数):
  - 机制假设:seed_ep3 物料 +0.09 的来源是"多样曝光"(子空间多样曝光量函数,
    07-04 修正版结论)——P2 是该机制的主动放大:把曝光从"配方顺带覆盖的 7280 个
    (dom,s_a)"定向扩到 ~2800 个纯尾部 s_a,直接对治 A2 发散题。
  - 预测:单独混入主配方(≤15%)训 1ep,懂物料尾部题 beam 聚焦(distinct s_a
    22-28 → 显著收窄);与 P1 叠加时物料向 0.245+(v6 水位)推进。
  - 若无效(±0.005 内):反向定案 = 尾部 s_a 的 desc→s_a 映射无法在 1ep 内建立,
    P2 路线关闭,转"多 epoch 只过物料数据"方案(物料子集多轮+其余单轮)。
  - 风险预案:同 P1——纯 nothink 物料样本占比升高可能轻微影响 rec 四域 think
    通路,本数据只应与主配方混合,不单独成方。

用法:
  python scripts/data/build_p2_tail_cover.py \
      [--out data/processed/p2_tail_cover.jsonl --seed 2026]
产出:
  - {out}                  训练集 ~7000 条(alpaca 格式,与 fresh_mat/P1 同构)
  - {out%.jsonl}_val.jsonl val 150 条(独立 s_a,零交集)
  - {out}.meta.jsonl       每条样本 pid/domain/s_a/sid/exposure/sa_item_count 侧写
"""
import argparse
import ast
import collections
import glob
import json
import random
import re

import numpy as np

# ---- 与种子物料样本逐字一致的模板(fresh_mat/P1 同源,07-04 已核对) ----
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
RAW2NORM = dict(BEGIN)
NORM2RAW = {v: k for k, v in RAW2NORM.items()}
SIDPAT = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
EXPOPAT = re.compile(r"<\|(living|prod|video|ad)_begin\|><s_a_(\d+)>")
K = 8404  # 编码基数,> 码本最大索引

# A2/A4 证据 s_a(全部 video 域,07-05 从 parsed_logs.json 核实)
EVIDENCE_SA = [2832, 1383, 606, 5983, 287, 1173, 1509, 7328]

# train 域配额(依据:各域尾部规模 × 评测四域均等权重,video/prod 为主)
QUOTA_TRAIN = {"video": 2900, "prod": 2300, "ad": 1400, "living": 500}
# val s_a 池大小(train 绝不触碰,零交集)
VAL_POOL = {"video": 40, "prod": 30, "ad": 20, "living": 12}
VAL_TARGET = 150


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


def load_exposure(fk_path, p1_path, p1_val_path):
    """输出侧曝光度:fk_fuse output + P1(train+val) gold。"""
    expo = collections.Counter()
    n_fk = 0
    for line in open(fk_path):
        r = json.loads(line)
        for dm, sa in EXPOPAT.findall(r.get("output", "")):
            expo[(dm, int(sa))] += 1
            n_fk += 1
    n_p1 = 0
    for fp in (p1_path, p1_val_path):
        for line in open(fp):
            r = json.loads(line)
            for dm, sa in EXPOPAT.findall(r["output"]):
                expo[(dm, int(sa))] += 1
                n_p1 += 1
    print(f"曝光度: fk_fuse 输出侧 s_a {n_fk} 次 + P1 gold {n_p1} 次,"
          f"distinct (dom,s_a) {len(expo)}")
    return expo


def load_exclusion(seed_glob, fresh_path, p1_path, p1_val_path):
    """SID 排除集:种子懂物料 ∪ fresh_mat ∪ P1(train+val)。"""
    excl = set()
    n_seed = 0
    for fp in sorted(glob.glob(seed_glob)):
        for line in open(fp):
            row = json.loads(line)[0]
            m = SIDPAT.search(row["response"])
            if m:
                excl.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                n_seed += 1
    n1 = len(excl)
    for line in open(fresh_path):
        r = json.loads(line)
        m = SIDPAT.search(r["output"])
        if m:
            excl.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    n2 = len(excl)
    for fp in (p1_path, p1_val_path):
        for line in open(fp):
            r = json.loads(line)
            m = SIDPAT.search(r["output"])
            if m:
                excl.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    print(f"排除集: 种子懂物料 {n_seed} 行→{n1} SID;+fresh_mat→{n2};"
          f"+P1(train+val)→{len(excl)} SID")
    return excl


def main():
    ap = argparse.ArgumentParser()
    base = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
    ap.add_argument("--sid_dir", default=f"{base}/data/hf_full/data/OneReason_Pid2Sid")
    ap.add_argument("--cap_dir", default=f"{base}/data/hf_full/data/OneReason_Pid2Caption")
    ap.add_argument("--fk_fuse", default=f"{base}/data/processed/fk_fuse.jsonl")
    ap.add_argument("--p1", default=f"{base}/data/processed/p1_bucket_discrim.jsonl")
    ap.add_argument("--p1_val", default=f"{base}/data/processed/p1_bucket_discrim_val.jsonl")
    ap.add_argument("--seed_mat_glob", default=f"{base}/data/懂物料part*.jsonl")
    ap.add_argument("--fresh_mat", default=f"{base}/data/processed/fresh_mat.jsonl")
    ap.add_argument("--out", default=f"{base}/data/processed/p2_tail_cover.jsonl")
    ap.add_argument("--n_keep_per_sa", type=int, default=12)   # caption覆盖率~59%,3的4倍冗余
    ap.add_argument("--n_keep_evidence", type=int, default=80)
    ap.add_argument("--max_expo", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    import pyarrow.parquet as pq

    DOMS = {"live": 0, "goods": 1, "video/video": 2, "video/ad": 3}
    INV = {0: "living", 1: "prod", 2: "video", 3: "ad"}

    expo = load_exposure(args.fk_fuse, args.p1, args.p1_val)

    # ---------- Pass 1: 全量聚合 (domain,s_a) item 计数 ----------
    print("Pass1: 全量扫 Pid2Sid 聚合 (domain,s_a) ...")
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
        key = d[m] * K + v[m, 0]
        u, c = np.unique(key, return_counts=True)
        for kk, cc in zip(u.tolist(), c.tolist()):
            agg[kk] += cc
    print(f"  (dom,s_a) 总数 {len(agg):,},item 总数 {sum(agg.values()):,}")

    # ---------- 尾部 s_a 判定 ----------
    tail = {d: [] for d in INV.values()}   # dom -> [sa,...] 曝光≤1 且 item≥2
    for kk, c in agg.items():
        dm, sa = INV[kk // K], kk % K
        if expo.get((dm, sa), 0) <= args.max_expo and c >= 2:
            tail[dm].append(sa)
    for d in ["video", "prod", "ad", "living"]:
        space = sum(1 for kk in agg if INV[kk // K] == d)
        n_tail_all = sum(1 for kk, c in agg.items()
                         if INV[kk // K] == d and expo.get((d, kk % K), 0) <= args.max_expo)
        print(f"  {d}: s_a 空间 {space},尾部(曝光≤{args.max_expo}) {n_tail_all},"
              f"其中 item≥2 可用 {len(tail[d])}")

    # 证据 s_a:存在即纳入(豁免 item≥2)
    evidence_in = []
    for sa in EVIDENCE_SA:
        kk = DOMS["video/video"] * K + sa
        if kk in agg:
            evidence_in.append(sa)
            e = expo.get(("video", sa), 0)
            print(f"  证据 s_a 纳入: (video,{sa}) items={agg[kk]} 曝光={e}"
                  + (" [非尾部,仍强制纳入]" if e > args.max_expo else ""))
            if sa not in tail["video"] and e <= args.max_expo:
                tail["video"].append(sa)   # item==1 的尾部证据 s_a
        else:
            print(f"  ⚠️ 证据 s_a 缺席源表: (video,{sa})")

    # ---------- val / train s_a 池(完全不相交) ----------
    val_pool, train_pool = {}, {}
    for d in ["video", "prod", "ad", "living"]:
        cands = sorted(set(tail[d]) - (set(evidence_in) if d == "video" else set()))
        rng.shuffle(cands)
        val_pool[d] = set(cands[: VAL_POOL[d]])
        train_pool[d] = cands[VAL_POOL[d]:]
        if d == "video":  # 证据 s_a 一律进 train,且排最前
            train_pool[d] = evidence_in + train_pool[d]
    assert all(not (val_pool[d] & set(train_pool[d])) for d in val_pool)

    # ---------- Pass 2: 收集尾部 s_a 的 (pid,sid) 候选,蓄水池采样 ----------
    print("Pass2: 收集尾部 s_a 的 pid 候选 ...")
    wanted_keys = set()
    for d in INV.values():
        di = DOMS[NORM2RAW[d]]
        for sa in tail[d]:
            wanted_keys.add(di * K + sa)
    for sa in evidence_in:
        wanted_keys.add(DOMS["video/video"] * K + sa)
    ev_keys = {DOMS["video/video"] * K + sa for sa in evidence_in}
    wk = np.array(sorted(wanted_keys), dtype=np.int64)
    cand = collections.defaultdict(list)   # key -> [(pid,sa,sb,sc)]
    n_seen = collections.Counter()
    for f in sid_files:
        t = pq.read_table(f)
        pids = np.array(t["pid"].to_pylist(), dtype=object)
        dom = t["domain"].to_numpy(zero_copy_only=False)
        col = t["sid_three"].combine_chunks()
        offs = np.diff(col.offsets.to_numpy())
        vals = col.values.to_numpy(zero_copy_only=False)
        ok = offs == 3
        v = vals[np.repeat(ok, offs)].reshape(-1, 3).astype(np.int64)
        d_ = np.array([DOMS.get(x, -1) for x in dom[ok]], dtype=np.int64)
        p_ = pids[ok]
        m = d_ >= 0
        key = d_[m] * K + v[m, 0]
        hit = np.isin(key, wk)
        for p, kk, (sa, sb, sc) in zip(p_[m][hit], key[hit].tolist(), v[m][hit].tolist()):
            cap_n = args.n_keep_evidence if kk in ev_keys else args.n_keep_per_sa
            n_seen[kk] += 1
            lst = cand[kk]
            if len(lst) < cap_n:
                lst.append((p, sa, sb, sc))
            else:  # reservoir
                j = rng.randrange(n_seen[kk])
                if j < cap_n:
                    lst[j] = (p, sa, sb, sc)
    pid_want = {}
    for kk, lst in cand.items():
        for (p, _, _, _) in lst:
            pid_want[p] = kk
    print(f"  候选 pid {len(pid_want):,}(s_a {len(cand):,})")

    # ---------- Pass 3: 扫 Pid2Caption 取 caption ----------
    print("Pass3: 扫 Pid2Caption ...")
    pid_cap = {}
    for f in sorted(glob.glob(f"{args.cap_dir}/*.parquet")):
        t = pq.read_table(f)
        for p, dm, cp in zip(t["pid"].to_pylist(), t["domain"].to_pylist(),
                             t["caption"].to_pylist()):
            if p in pid_want and p not in pid_cap and dm in DOMS:
                if DOMS[dm] == pid_want[p] // K:   # caption 行域须与 s_a 域一致
                    pid_cap[p] = cp
    print(f"  命中 caption {len(pid_cap):,}/{len(pid_want):,}")

    # ---------- 组装样本 ----------
    excl = load_exclusion(args.seed_mat_glob, args.fresh_mat, args.p1, args.p1_val)
    seen_sid, seen_cap_global = set(), set()

    def build_sa(dom, sa, tag, min_keep=2):
        """一个 s_a 出 2-3 条(证据 s_a 允许 1 条);返回 rows, metas。"""
        di = DOMS[NORM2RAW[dom]]
        kk = di * K + sa
        raw = NORM2RAW[dom]
        target = rng.randint(2, 3)
        lst = list(cand.get(kk, []))
        rng.shuffle(lst)
        # 优先不同 s_b:先按"s_b 首现"排前
        seen_sb, first, rest = set(), [], []
        for it in lst:
            (first if it[2] not in seen_sb else rest).append(it)
            seen_sb.add(it[2])
        rows, metas = [], []
        loc_caps, loc_cap20, loc_sids = set(), set(), set()
        for (p, sa_, sb, sc) in first + rest:
            if len(rows) >= target:
                break
            sid = (sa_, sb, sc)
            if sid in excl or sid in seen_sid or sid in loc_sids:
                continue
            cp = pid_cap.get(p)
            if cp is None:
                continue
            desc = clean_caption(raw, cp)
            if not desc:
                continue
            h = hash(desc)
            if h in seen_cap_global or h in loc_caps or desc[:20] in loc_cap20:
                continue
            loc_caps.add(h)
            loc_cap20.add(desc[:20])
            loc_sids.add(sid)
            rows.append({
                "instruction": INS[raw],
                "input": LEAD[raw] + desc + "/no_think",
                "output": f"<think>\n</think>\n<|{BEGIN[raw]}_begin|>"
                          f"<s_a_{sa_}><s_b_{sb}><s_c_{sc}>",
                "history": [],
            })
            metas.append({"split": tag, "pid": p, "domain": dom, "s_a": sa_,
                          "sid": [sa_, sb, sc], "exposure": expo.get((dom, sa_), 0),
                          "sa_item_count": agg[kk]})
        if len(rows) < min_keep:
            return [], []   # 整 s_a 弃用,全局集合不落任何登记
        seen_sid.update(loc_sids)
        seen_cap_global.update(loc_caps)
        return rows, metas

    # val 先建(锁定 s_a 与 SID),按域轮转填到 150
    val_rows, val_meta = [], []
    val_iters = {d: iter(sorted(val_pool[d], key=lambda x: rng.random()))
                 for d in val_pool}
    active = ["video", "prod", "ad", "living"]
    while len(val_rows) < VAL_TARGET and active:
        for d in list(active):
            if len(val_rows) >= VAL_TARGET:
                break
            try:
                sa = next(val_iters[d])
            except StopIteration:
                active.remove(d)
                continue
            rows, metas = build_sa(d, sa, "val")
            val_rows.extend(rows)
            val_meta.extend(metas)
    val_rows, val_meta = val_rows[:VAL_TARGET], val_meta[:VAL_TARGET]

    # train:证据 s_a 先行(video),再按池序填到各域配额
    train_rows, train_meta = [], []
    for d in ["video", "prod", "ad", "living"]:
        pool = train_pool[d]
        if d != "video":
            pool = sorted(pool, key=lambda x: rng.random())
        else:  # 证据在前,其余打乱
            head_n = len(evidence_in)
            rest = pool[head_n:]
            rng.shuffle(rest)
            pool = pool[:head_n] + rest
        got = 0
        for sa in pool:
            if got >= QUOTA_TRAIN[d]:
                break
            is_ev = (d == "video" and sa in evidence_in)
            rows, metas = build_sa(d, sa, "train", min_keep=1 if is_ev else 2)
            train_rows.extend(rows)
            train_meta.extend(metas)
            got += len(rows)
        print(f"  train {d}: {got} 条 / 配额 {QUOTA_TRAIN[d]}"
              f"(s_a 覆盖 {len(set(m['s_a'] for m in train_meta if m['domain']==d))})")

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
    per_sa = collections.Counter((m["domain"], m["s_a"]) for m in train_meta)
    dist = collections.Counter(per_sa.values())
    e0 = sum(1 for m in train_meta if m["exposure"] == 0)
    print(f"train {len(train_rows)} 条 → {args.out}")
    print(f"  域分布: {dict(dom_dist)}")
    print(f"  s_a 覆盖 {len(per_sa)},每 s_a 样本数分布 {dict(sorted(dist.items()))}")
    print(f"  曝光=0 样本 {e0},曝光=1 样本 {len(train_meta)-e0}"
          f"(证据 s_a 可为更高曝光)")
    tr_sa = set((m["domain"], m["s_a"]) for m in train_meta)
    va_sa = set((m["domain"], m["s_a"]) for m in val_meta)
    print(f"val {len(val_rows)} 条({len(va_sa)} s_a,与 train s_a 交集 "
          f"{len(tr_sa & va_sa)})→ {val_path}")
    print(f"meta → {args.out}.meta.jsonl")


if __name__ == "__main__":
    main()
