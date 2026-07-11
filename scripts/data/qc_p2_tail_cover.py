#!/usr/bin/env python3
"""qc_p2_tail_cover.py — P2 数据集全量质检(2026-07-05)。

检查项(全部给出数字):
  1. 行数/JSON 可解析
  2. 格式不变量零违例:instruction∈4模板、input=LEAD+desc+/no_think、
     output=^<think>\n</think>\n<|dom_begin|><s_a><s_b><s_c>$、history=[]、
     begin 域与 instruction 域一致、input 无 SID 泄漏
  3. desc↔SID 全量查表复核:meta 的 (pid,sid,domain) 在 Pid2Sid 中逐一存在,
     且 pid 的 caption 清洗后 == 样本 desc(扫 Pid2Caption 复核)
  4. SID 与 种子懂物料∪fresh_mat∪P1(train+val) 零重叠;train∪val 内部 SID 全局唯一
  5. 每 s_a 样本数分布(2-3 约束,证据 s_a 允许 1)
  6. 各域 s_a 覆盖数;train/val s_a 零交集
  7. 曝光度复核:train 非证据 s_a 的 exposure≤1 零违例
"""
import ast
import collections
import glob
import json
import re

base = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
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
NORM2RAW = {v: k for k, v in BEGIN.items()}
OUTPAT = re.compile(r"^<think>\n</think>\n<\|(living|prod|video|ad)_begin\|>"
                    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$")
SIDPAT = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
EXPOPAT = re.compile(r"<\|(living|prod|video|ad)_begin\|><s_a_(\d+)>")
EVIDENCE_SA = {2832, 1383, 606, 5983, 287, 1173, 1509, 7328}


def clean_caption(raw_dom, cap):
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


def load(fp):
    return [json.loads(l) for l in open(fp)]


train = load(f"{base}/data/processed/p2_tail_cover.jsonl")
val = load(f"{base}/data/processed/p2_tail_cover_val.jsonl")
meta = load(f"{base}/data/processed/p2_tail_cover.jsonl.meta.jsonl")
mtr = [m for m in meta if m["split"] == "train"]
mva = [m for m in meta if m["split"] == "val"]
print(f"[1] 行数: train={len(train)} val={len(val)} meta={len(meta)}"
      f"(meta train={len(mtr)} val={len(mva)});JSON 全部可解析")
assert len(mtr) == len(train)
# meta val 可能比 val 文件多(150 截断) -> 对齐
mva = mva[: len(val)]

# ---- 2. 格式不变量 ----
viol = collections.Counter()
descs = []
for rows, ms, tag in ((train, mtr, "train"), (val, mva, "val")):
    for r, m in zip(rows, ms):
        raw = NORM2RAW[m["domain"]]
        if r["instruction"] != INS[raw]:
            viol["instruction"] += 1
        if not r["input"].startswith(LEAD[raw]) or not r["input"].endswith("/no_think"):
            viol["input_frame"] += 1
        mm = OUTPAT.match(r["output"])
        if not mm:
            viol["output_pat"] += 1
            continue
        if mm.group(1) != m["domain"]:
            viol["begin_dom"] += 1
        if [int(mm.group(2)), int(mm.group(3)), int(mm.group(4))] != m["sid"]:
            viol["sid_meta_mismatch"] += 1
        if r["history"] != []:
            viol["history"] += 1
        desc = r["input"][len(LEAD[raw]):-len("/no_think")]
        if SIDPAT.search(desc) or "<|" in desc:
            viol["sid_leak_in_input"] += 1
        descs.append((m["pid"], raw, desc, m["sid"], m["domain"]))
print(f"[2] 格式不变量违例: {dict(viol) if viol else '0(全项零违例)'}")

# ---- 4. SID 唯一性 + 排除集零重叠 ----
all_sids = [tuple(m["sid"]) + (m["domain"],) for m in mtr + mva]
print(f"[4a] SID(含域)全局唯一: {len(all_sids)} 条,唯一 {len(set(all_sids))},"
      f"重复 {len(all_sids) - len(set(all_sids))}")
sid_only = set(tuple(m["sid"]) for m in mtr + mva)
excl = set()
for fp in sorted(glob.glob(f"{base}/data/懂物料part*.jsonl")):
    for line in open(fp):
        row = json.loads(line)[0]
        mm = SIDPAT.search(row["response"])
        if mm:
            excl.add(tuple(int(x) for x in mm.groups()))
n_seed = len(excl)
for line in open(f"{base}/data/processed/fresh_mat.jsonl"):
    mm = SIDPAT.search(json.loads(line)["output"])
    if mm:
        excl.add(tuple(int(x) for x in mm.groups()))
n_fresh = len(excl)
for fp in (f"{base}/data/processed/p1_bucket_discrim.jsonl",
           f"{base}/data/processed/p1_bucket_discrim_val.jsonl"):
    for line in open(fp):
        mm = SIDPAT.search(json.loads(line)["output"])
        if mm:
            excl.add(tuple(int(x) for x in mm.groups()))
print(f"[4b] 排除集: 种子{n_seed}→+fresh_mat {n_fresh}→+P1 {len(excl)};"
      f"与 P2 交集 = {len(sid_only & excl)}")

# ---- 5/6. s_a 分布与覆盖 ----
per_sa_tr = collections.Counter((m["domain"], m["s_a"]) for m in mtr)
per_sa_va = collections.Counter((m["domain"], m["s_a"]) for m in mva)
dist = collections.Counter(per_sa_tr.values())
bad_count = {k: v for k, v in per_sa_tr.items()
             if not (2 <= v <= 3) and not (k[0] == "video" and k[1] in EVIDENCE_SA)}
print(f"[5] train 每 s_a 样本数分布: {dict(sorted(dist.items()))};"
      f"非证据 s_a 越界(非2-3): {len(bad_count)}")
ev_cov = sorted(sa for (d, sa) in per_sa_tr if d == "video" and sa in EVIDENCE_SA)
print(f"    证据 s_a 实际覆盖: {ev_cov}(样本数 "
      f"{[per_sa_tr[('video', s)] for s in ev_cov]})")
cov_tr = collections.Counter(d for (d, _) in per_sa_tr)
cov_va = collections.Counter(d for (d, _) in per_sa_va)
dom_tr = collections.Counter(m["domain"] for m in mtr)
dom_va = collections.Counter(m["domain"] for m in mva)
print(f"[6] train 域样本: {dict(dom_tr)};s_a 覆盖: {dict(cov_tr)}")
print(f"    val   域样本: {dict(dom_va)};s_a 覆盖: {dict(cov_va)};"
      f"train∩val s_a = {len(set(per_sa_tr) & set(per_sa_va))}")

# ---- 7. 曝光度复核 ----
expo = collections.Counter()
for line in open(f"{base}/data/processed/fk_fuse.jsonl"):
    for dm, sa in EXPOPAT.findall(json.loads(line).get("output", "")):
        expo[(dm, int(sa))] += 1
for fp in (f"{base}/data/processed/p1_bucket_discrim.jsonl",
           f"{base}/data/processed/p1_bucket_discrim_val.jsonl"):
    for line in open(fp):
        for dm, sa in EXPOPAT.findall(json.loads(line)["output"]):
            expo[(dm, int(sa))] += 1
bad_expo = [k for k in per_sa_tr
            if expo.get(k, 0) > 1 and not (k[0] == "video" and k[1] in EVIDENCE_SA)]
bad_expo_va = [k for k in per_sa_va if expo.get(k, 0) > 1]
e0 = sum(1 for k in per_sa_tr if expo.get(k, 0) == 0)
print(f"[7] train 非证据 s_a 曝光>1 违例: {len(bad_expo)};val 曝光>1: {len(bad_expo_va)}")
print(f"    train s_a 曝光=0: {e0} / 曝光=1: {len(per_sa_tr) - e0 - len(ev_cov)}"
      f"(另证据 {len(ev_cov)})")

# ---- 3. desc↔SID 全量查表复核 ----
print("[3] 全量查表复核(扫 Pid2Sid + Pid2Caption)...")
import pyarrow.parquet as pq
DOMS = {"live": 0, "goods": 1, "video/video": 2, "video/ad": 3}
need = {}   # pid -> (raw_dom, sid, desc)
for pid, raw, desc, sid, dom in descs:
    need[pid] = (raw, tuple(sid), desc)
assert len(need) == len(descs), "pid 重复!"
sid_ok, cap_ok = set(), set()
for f in sorted(glob.glob(f"{base}/data/hf_full/data/OneReason_Pid2Sid/*.parquet")):
    t = pq.read_table(f)
    for p, dm, s in zip(t["pid"].to_pylist(), t["domain"].to_pylist(),
                        t["sid_three"].to_pylist()):
        got = need.get(p)
        if got and dm == got[0] and s and len(s) == 3 \
                and tuple(int(x) for x in s) == got[1]:
            sid_ok.add(p)
for f in sorted(glob.glob(f"{base}/data/hf_full/data/OneReason_Pid2Caption/*.parquet")):
    t = pq.read_table(f)
    for p, dm, cp in zip(t["pid"].to_pylist(), t["domain"].to_pylist(),
                         t["caption"].to_pylist()):
        got = need.get(p)
        if got and dm == got[0] and p not in cap_ok:
            if clean_caption(got[0], cp) == got[2]:
                cap_ok.add(p)
print(f"    (pid,domain,sid) 查表命中: {len(sid_ok)}/{len(need)};"
      f"desc 复核一致: {len(cap_ok)}/{len(need)}")
miss = [p for p in need if p not in sid_ok or p not in cap_ok]
print(f"    复核失败 pid 数: {len(miss)}" + (f" 例: {miss[:5]}" if miss else ""))
