#!/usr/bin/env python3
"""build_world_mc_official.py — 官方 General 源选择题(懂世界评测模板)增补(2026-07-05)。

源: /tmp/mc_candidates.jsonl(821 条候选,07-04 HF 审计产出;字段 uuid/shard/type/lang/ans)
    → 只取 ans 已机械提取成功(A-D)的 245 条(作答型 84 + 模糊型 161;zh42/en196/other7),
    按 uuid 回 data/hf_full/data/OneReason_General 的 parquet 取全文。
若候选文件缺失: --reextract 按同口径直接从 General 重扫(题干含 A-D 选项行 +
    assistant 正文(剥 think)可正则唯一提取答案字母)。

格式: 逐字对齐评测模板(与 world_mc_clean/懂世界.jsonl 字节一致):
    system   = '你是一个非常聪明的助手，请直接遵循指示作答。'
    prompt   = '请回答以下问题：\n\n{题干+选项}\n\n请按以下格式作答："正确答案是 (在此处填写选项字母)"/no_think'
    response = '<think>\n\n</think>\n\n\n正确答案是 ({字母})'
存储为 alpaca 平铺(可直接注册 dataset_info 训练),字段映射与 build_pstack_v2.py
混入 world_mc_clean 的转换逐字一致: instruction=system, input=prompt, output=response。
额外标注字段(不参与训练,供配比取舍): lang(zh/en/other), uuid, src_type。

清洗: ①题干剥已知 boilerplate 前缀(单选/多选题引导语);②答案字母必须 A-D;
      ③归一化题干去重: 文件内 + vs world_mc_clean.jsonl(238) + vs 评测样例 懂世界.jsonl(防真题泄漏)。

用法: python scripts/data/build_world_mc_official.py [--candidates /tmp/mc_candidates.jsonl] [--reextract]
"""
import argparse
import collections
import glob
import json
import re

ROOT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
SYS = "你是一个非常聪明的助手，请直接遵循指示作答。"
P_HEAD = "请回答以下问题：\n\n"
P_TAIL = "\n\n请按以下格式作答：\"正确答案是 (在此处填写选项字母)\"/no_think"
WRAP_RE = re.compile(r'^请回答以下问题：\n\n(.*)\n\n请按以下格式作答："正确答案是 \(在此处填写选项字母\)"/no_think$', re.S)

# 题干已知 boilerplate 前缀(机械、白名单式,不做语义改写)
BOILER = [
    re.compile(r"^以下是一道单项选择题[:：]\s*"),
    re.compile(r"^这是一道单项选择题[。:：]\s*"),
    re.compile(r"^这是一道多选题[。:：]\s*"),
    re.compile(r"^请给出多选题的答案和解析[。:：]?\s*"),
    re.compile(r"^\(Multiple-Choice\):\*\*\s*"),
]
# 答案机械提取(仅 --reextract 回退模式使用;候选文件存在时以其 ans 为准)
ANS_PATS = [
    re.compile(r"正确答案是\s*[\(（]?\s*([A-D])\s*[\)）]?"),
    re.compile(r"(?:正确)?答案(?:选项)?[是为选：:\s]*\**[\(（]?\s*([A-D])\b"),
    re.compile(r"answer\s*(?:is|:)?\s*\**[\(（]?\s*([A-D])\b", re.I),
    re.compile(r"\\boxed\{\s*\(?([A-D])\)?"),
    re.compile(r"故选\s*\**[\(（]?\s*([A-D])\b"),
    re.compile(r"^\s*\(?([A-D])\)?[\.\s]*$", re.M),
]
OPT_LINE = re.compile(r"(?m)^\s*[\(（]?([A-E])[\)）\.、．:：]\s")
CJK = re.compile(r"[一-鿿]")


def norm(s):
    return re.sub(r"\s+", "", s)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def strip_boiler(stem):
    stem = stem.strip()
    changed = True
    while changed:
        changed = False
        for p in BOILER:
            new = p.sub("", stem)
            if new != stem:
                stem, changed = new.lstrip(), True
    return stem


def extract_ans(a_texts):
    for a in a_texts:
        body = re.sub(r"<think>.*?</think>", "", a, flags=re.S)
        for p in ANS_PATS:
            m = p.search(body)
            if m:
                return m.group(1)
    return None


def lang_of(u):
    r = len(CJK.findall(u)) / max(len(u), 1)
    if r > 0.3:
        return "zh"
    if re.search(r"[a-zA-Z]", u) and r < 0.05:
        # 简易判定: 拉丁为主且几乎无中文 → en;否则 other
        latin = len(re.findall(r"[a-zA-Z]", u)) / max(len(u), 1)
        return "en" if latin > 0.3 else "other"
    return "other"


def load_from_candidates(path, src_dir):
    cands = [json.loads(l) for l in open(path, encoding="utf-8")]
    sel = {c["uuid"]: c for c in cands if c.get("ans") in ("A", "B", "C", "D")}
    print(f"候选 {len(cands)} 条 → ans∈A-D 的 {len(sel)} 条", flush=True)
    byshard = collections.defaultdict(set)
    for c in sel.values():
        byshard[c["shard"]].add(c["uuid"])
    import pyarrow.parquet as pq
    rows = []
    for shard in sorted(byshard):
        t = pq.read_table(f"{src_dir}/{shard}", columns=["uuid", "messages"])
        for uid, m in zip(t["uuid"].to_pylist(), t["messages"].to_pylist()):
            if uid not in byshard[shard]:
                continue
            msgs = json.loads(m)
            users = [extract_text(x["content"]).strip() for x in msgs if x.get("role") == "user"]
            if not users:
                continue
            c = sel[uid]
            rows.append({"uuid": uid, "user": users[0], "ans": c["ans"],
                         "lang": c["lang"], "src_type": c["type"]})
    return rows


def reextract(src_dir):
    """回退口径: 全量扫 General,题干含≥4条 A-D 选项行 + 答案可正则唯一提取。"""
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(f"{src_dir}/*.parquet")):
        t = pq.read_table(f, columns=["uuid", "messages"])
        for uid, m in zip(t["uuid"].to_pylist(), t["messages"].to_pylist()):
            try:
                msgs = json.loads(m)
            except Exception:
                continue
            users = [extract_text(x["content"]).strip() for x in msgs if x.get("role") == "user"]
            asst = [extract_text(x["content"]).strip() for x in msgs if x.get("role") == "assistant"]
            if not users or not asst:
                continue
            u = users[0]
            opts = set(OPT_LINE.findall(u))
            if not {"A", "B", "C", "D"} <= opts:
                continue
            ans = extract_ans(asst)
            if ans is None:
                continue
            rows.append({"uuid": uid, "user": u, "ans": ans,
                         "lang": lang_of(u), "src_type": "reextract"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="/tmp/mc_candidates.jsonl")
    ap.add_argument("--src_dir", default=f"{ROOT}/data/hf_full/data/OneReason_General")
    ap.add_argument("--mc_clean", default=f"{ROOT}/data/processed/world_mc_clean.jsonl")
    ap.add_argument("--eval_sample", default=f"{ROOT}/懂世界.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/data/processed/world_mc_official.jsonl")
    ap.add_argument("--reextract", action="store_true")
    args = ap.parse_args()

    import os
    if args.reextract or not os.path.exists(args.candidates):
        print("候选文件缺失或指定 --reextract,按口径全量重扫", flush=True)
        rows = reextract(args.src_dir)
    else:
        rows = load_from_candidates(args.candidates, args.src_dir)
    rows.sort(key=lambda r: r["uuid"])  # 确定性排序

    # 既有题干集合(零重叠约束): world_mc_clean + 评测样例真题
    def unwrap(x):
        return x[0] if isinstance(x, list) else x
    base_stems = set()
    for line in open(args.mc_clean, encoding="utf-8"):
        d = unwrap(json.loads(line))
        m = WRAP_RE.match(d["prompt"])
        base_stems.add(norm(m.group(1) if m else d["prompt"]))
    eval_stems = set()
    for line in open(args.eval_sample, encoding="utf-8"):
        d = unwrap(json.loads(line))
        m = WRAP_RE.match(d["prompt"])
        eval_stems.add(norm(m.group(1) if m else d["prompt"]))
        eval_stems.add(norm(d["prompt"]))

    out, seen = [], set()
    drop = {"重复_内部": 0, "撞world_mc_clean": 0, "撞评测样例": 0, "空题干": 0}
    n_opt_e = 0
    for r in rows:
        stem = strip_boiler(r["user"])
        if not stem:
            drop["空题干"] += 1
            continue
        k = norm(stem)
        if k in seen:
            drop["重复_内部"] += 1
            continue
        if k in base_stems:
            drop["撞world_mc_clean"] += 1
            continue
        if k in eval_stems:
            drop["撞评测样例"] += 1
            continue
        seen.add(k)
        if "E" in set(OPT_LINE.findall(stem)):
            n_opt_e += 1
        prompt = P_HEAD + stem + P_TAIL
        resp = f"<think>\n\n</think>\n\n\n正确答案是 ({r['ans']})"
        out.append({"instruction": SYS, "input": prompt, "output": resp, "history": [],
                    "lang": r["lang"], "uuid": r["uuid"], "src_type": r["src_type"]})

    # 硬校验: 格式不变量
    for r in out:
        assert r["instruction"] == SYS
        assert WRAP_RE.match(r["input"]), r["uuid"]
        assert re.fullmatch(r"<think>\n\n</think>\n\n\n正确答案是 \([A-D]\)", r["output"]), r["uuid"]
    assert not ({norm(WRAP_RE.match(r["input"]).group(1)) for r in out} & base_stems)

    with open(args.out, "w", encoding="utf-8") as g:
        for r in out:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    langs = collections.Counter(r["lang"] for r in out)
    print(f"[OK] {args.out}: {len(out)} 条  lang={dict(langs)}  含E选项题 {n_opt_e} 条(答案仍为A-D)")
    print(f"[OK] 丢弃: {drop}")


if __name__ == "__main__":
    main()
