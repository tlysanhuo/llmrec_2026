#!/usr/bin/env python3
"""build_token_pack.py — Token 粒度码本几何数据(OneReason §4.2.1 四形态严格复刻,规则化总结替代 LLM teacher)。

论文依据(docs/reference/onereason_data_method.md):
  T1 Compositional Prefix Semantic Prediction (1.29%):前缀对 <a,b> → 共享前缀 item 的共同语义
  T2 Prefix Itemic Token Grounding        (0.55%):共同语义描述 → 前缀对(反向)
  T3 Part-to-Whole Semantic Prediction    (0.31%):逐 sub-token 语义 → 整体粗化 caption(两步式)
  T4 Single-Token Semantic Prediction     (0.35%):单 s_a token → 语义
配套:Capacity-Aware Coarse-Graining(论文 line 149-166,过细 caption 强迫幻觉):
  (i) 去实例噪声(日期/型号/价格) (ii) 连续属性粗化 (iii) 保内容骨架
与 LLM teacher 的偏离(如实声明):共同语义=桶内 caption 的 jieba TF-IDF 关键词 + 模板成句。

用法: python scripts/data/build_token_pack.py --out data/processed/token_pack.jsonl \
        [--n_total 5000 --shards 30 --seed 2026]
v2(2026-07-06):所有 input 追加 " /no_think"——对齐种子铁约定"空think⇔/no_think后缀"
(v1 缺后缀,tokengeo_v1_ep3 已带 v1 上线,读数出来后此差异一并归因)。
"""
import argparse, ast, collections, glob, json, math, random, re

DOM_CN = {"live": "直播", "goods": "电商", "video/video": "短视频", "video/ad": "广告"}
BEGIN = {"live": "living", "goods": "prod", "video/video": "video", "video/ad": "ad"}
ITEM_CN = {"live": "主播", "goods": "商品", "video/video": "短视频", "video/ad": "广告"}
# 域配额(评测四域均衡+video_ad共享子空间侧重)
DOM_SHARE = {"video/video": 0.32, "goods": 0.28, "video/ad": 0.22, "live": 0.18}
# 四形态占比(Table 25 内部比例 1.29:0.55:0.31:0.35)
FORM_SHARE = {"T1": 0.516, "T2": 0.22, "T3": 0.124, "T4": 0.14}

STOP = set("的了在与和及或为是有一款个各这那你我他它们其中进行提供支持适用于本商品视频直播广告主播内容"
           "主要相关系列新款正品包邮批发厂家直销特价优惠限时爆款热卖推荐精选高品质")
NOISE_RE = [re.compile(p) for p in [
    r"\d{4}[-/年]\d{1,2}[-/月]?\d{0,2}日?",          # 日期
    r"[A-Za-z0-9]{2,}[-_][A-Za-z0-9-_]{2,}",          # 型号
    r"[¥￥$]?\d+(\.\d+)?(元|块|折|克|g|kg|ml|L|寸|cm|mm|码|号)",  # 价格/规格
    r"【[^】]{0,20}】|\[[^\]]{0,20}\]",                # 促销括号
    r"(第\d+|\d+)(集|期|季|话)",                       # 集数
]]

def coarse(cap):
    """Capacity-Aware Coarse-Graining:去实例噪声,保内容骨架"""
    for pat in NOISE_RE:
        cap = pat.sub("", cap)
    cap = re.sub(r"\s+", " ", cap).strip("，。、,. ")
    return cap[:120]

def clean_caption(dom, cap):
    if not cap or not isinstance(cap, str) or not cap.strip():
        return None
    cap = cap.strip()
    if dom == "live":
        try:
            tags = ast.literal_eval(cap)
            if not isinstance(tags, list) or len(tags) < 3:
                return None
            return "、".join(str(t) for t in tags[:10])
        except Exception:
            return None
    if len(cap) < 10 or (cap.startswith("[") and cap.endswith("]")):
        return None
    return cap[:300]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_Pid2Caption")
    ap.add_argument("--sid_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_Pid2Sid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_total", type=int, default=5000)
    ap.add_argument("--shards", type=int, default=30)
    ap.add_argument("--min_bucket", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    import pyarrow.parquet as pq
    import jieba
    jieba.initialize()

    # 1) pid→(dom,sid)
    pid2sid = {}
    sid_files = sorted(glob.glob(f"{args.sid_dir}/*.parquet")); rng.shuffle(sid_files)
    for f in sid_files[: args.shards]:
        t = pq.read_table(f, columns=["pid", "domain", "sid_three"])
        for p, d, s in zip(t["pid"].to_pylist(), t["domain"].to_pylist(), t["sid_three"].to_pylist()):
            if d in DOM_CN and s:
                try:
                    v = ast.literal_eval(s) if isinstance(s, str) else s
                    pid2sid[p] = (d, tuple(str(int(float(x))) for x in v))
                except Exception:
                    pass
    print(f"pid2sid: {len(pid2sid):,}")

    # 2) (dom,a,b)桶 与 (dom,a)桶 收集 coarse caption
    ab_bucket = collections.defaultdict(list)
    a_bucket = collections.defaultdict(list)
    seen_cap = set()
    cap_files = sorted(glob.glob(f"{args.cap_dir}/*.parquet")); rng.shuffle(cap_files)
    for f in cap_files[: args.shards]:
        t = pq.read_table(f, columns=["pid", "domain", "caption"])
        for p, d, cap in zip(t["pid"].to_pylist(), t["domain"].to_pylist(), t["caption"].to_pylist()):
            got = pid2sid.get(p)
            if not got or got[0] != d:
                continue
            desc = clean_caption(d, cap)
            if not desc:
                continue
            ck = hash((d, desc))
            if ck in seen_cap:
                continue          # 域内 caption 去重(消歧,审计缺陷1)
            seen_cap.add(ck)
            desc = coarse(desc)
            if len(desc) < 8:
                continue
            sid = got[1]
            if len(ab_bucket[(d, sid[0], sid[1])]) < 40:
                ab_bucket[(d, sid[0], sid[1])].append((desc, sid[2]))
            if len(a_bucket[(d, sid[0])]) < 60:
                a_bucket[(d, sid[0])].append(desc)
    print(f"(a,b)桶: {len(ab_bucket):,}  (a)桶: {len(a_bucket):,}")

    # 3) 全局 DF(TF-IDF 用)
    df = collections.Counter()
    ndoc = 0
    for caps in a_bucket.values():
        for c in caps[:20]:
            ndoc += 1
            for w in set(jieba.lcut(c)):
                if len(w) >= 2 and w not in STOP and not w.isdigit():
                    df[w] += 1

    def keywords(caps, k=8):
        tf = collections.Counter()
        for c in caps:
            for w in jieba.lcut(c):
                if len(w) >= 2 and w not in STOP and not w.isdigit():
                    tf[w] += 1
        scored = [(w, n * math.log(ndoc / (1 + df.get(w, 0)))) for w, n in tf.items() if n >= 2]
        scored.sort(key=lambda x: -x[1])
        return [w for w, _ in scored[:k]]

    def summarize(dom, kws):
        head = "、".join(kws[:4]); tail = "、".join(kws[4:8])
        s = f"该组合代表{DOM_CN[dom]}域中以{head}为核心的{ITEM_CN[dom]}内容"
        if tail:
            s += f"，常见元素包括{tail}"
        return s + "。"

    # 4) 生成四形态
    out = {"T1": [], "T2": [], "T3": [], "T4": []}
    ab_keys = [k for k, v in ab_bucket.items() if len(v) >= args.min_bucket]
    a_keys = [k for k, v in a_bucket.items() if len(v) >= args.min_bucket]
    rng.shuffle(ab_keys); rng.shuffle(a_keys)
    quota = {d: {f: int(args.n_total * DOM_SHARE[d] * FORM_SHARE[f]) for f in FORM_SHARE} for d in DOM_SHARE}

    for (d, a, b) in ab_keys:
        caps = ab_bucket[(d, a, b)]
        kws = keywords([c for c, _ in caps])
        if len(kws) < 3:
            continue
        sem = summarize(d, kws)
        pre = f"<s_a_{a}><s_b_{b}>"
        dom = BEGIN[d]
        if len(out["T1"]) < sum(quota[x]["T1"] for x in quota) and quota[d]["T1"] > 0:
            quota[d]["T1"] -= 1
            out["T1"].append({
                "instruction": f"你是一名{ITEM_CN[d]}语义标识分析专家，能够解释{ITEM_CN[d]}token前缀组合的含义。",
                "input": f"{DOM_CN[d]}域中，token前缀组合 {pre} 表示什么？ /no_think",
                "output": f"<think>\n</think>\n{sem}",
                "history": []})
        elif quota[d]["T2"] > 0:
            quota[d]["T2"] -= 1
            out["T2"].append({
                "instruction": f"你是一名{ITEM_CN[d]}语义标识分析专家，能根据语义描述反推对应的token前缀组合。",
                "input": f"在{DOM_CN[d]}域中，哪个token前缀组合表示：{sem}只输出目标前缀。 /no_think",
                "output": f"<think>\n</think>\n<|{dom}_begin|>{pre}",
                "history": []})
        elif quota[d]["T3"] > 0:
            quota[d]["T3"] -= 1
            desc, c = caps[0]
            akw = "、".join(keywords(a_bucket[(d, a)], 4)) or "该大类内容"
            bkw = "、".join(kws[:4])
            out["T3"].append({
                "instruction": f"你是一名{ITEM_CN[d]}语义标识分析专家，请逐层解释{ITEM_CN[d]}token的语义并综合成整体描述。",
                "input": f"请逐个解释 <|{dom}_begin|><s_a_{a}><s_b_{b}><s_c_{c}> 各层token的语义，再综合描述该{ITEM_CN[d]}。 /no_think",
                "output": (f"<think>\n</think>\n<s_a_{a}>表示{DOM_CN[d]}域的大类语义：{akw}；"
                           f"<s_a_{a}><s_b_{b}>细化为：{bkw}；<s_c_{c}>定位到具体{ITEM_CN[d]}。\n"
                           f"综合描述：{desc}"),
                "history": []})

    for (d, a) in a_keys:
        if quota[d]["T4"] <= 0:
            continue
        kws = keywords(a_bucket[(d, a)], 6)
        if len(kws) < 3:
            continue
        quota[d]["T4"] -= 1
        out["T4"].append({
            "instruction": f"你是一名{ITEM_CN[d]}语义标识分析专家，能够解释单个token的大类语义。",
            "input": f"{DOM_CN[d]}域中，token <s_a_{a}> 代表哪类内容？ /no_think",
            "output": f"<think>\n</think>\n<s_a_{a}>代表{DOM_CN[d]}域中与{('、'.join(kws[:5]))}相关的大类内容。",
            "history": []})

    allrows = [r for v in out.values() for r in v]
    rng.shuffle(allrows)
    with open(args.out, "w") as g:
        for r in allrows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {f: len(v) for f, v in out.items()}
    domc = collections.Counter()
    for v in out.values():
        for r in v:
            for d in DOM_CN:
                if DOM_CN[d] + "域" in r["input"]:
                    domc[d] += 1
                    break
    print(f"共 {len(allrows)} 条 → {args.out}\n形态: {stats}\n域: {dict(domc)}")

if __name__ == "__main__":
    main()
