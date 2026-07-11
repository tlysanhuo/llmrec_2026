#!/usr/bin/env python3
"""build_u2_mc.py — U2 候选链选择题(2026-07-07,P1 步,官方 PPT SFT-CoT 样例页格式)。

原料:U1 成品(teacher 主题+gold+注释版历史)。
构造:gold 按时间序切成 1-2 条"支撑链",另拼 1 条无关行为"干扰链";
      题面=历史节选(带文字)+演化目标+候选 A/B/C;答案=[支撑链字母];短 think(/think 惯例)。
质检:链内条目均∈历史;干扰链与 gold 零交集;caption 缺失的条目不用于链描述。
"""
import json, random, re, glob
from pathlib import Path

P = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed")
OUT = P / "blocks_v1" / "block_u2_mc.jsonl"
rng = random.Random(20260707)
ITEM = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def parse_annot(annot):
    """注释历史 → {token: (date, action, cap)}(按出现序)。"""
    info, order, cur = {}, [], None
    for line in annot.split("\n"):
        m = re.match(r"【(\d{4}-\d{2}-\d{2})】", line.strip())
        if m:
            cur = m.group(1); continue
        m = re.match(r"--:-- \[([^\]]+)\] (.*)$", line.strip())
        if not m:
            continue
        act, rest = m.group(1), m.group(2)
        t = ITEM.search(rest)
        if not t:
            continue
        cap = rest[: t.start()].strip()
        tok = t.group(0)
        if tok not in info:
            info[tok] = (cur or "----", act, cap)
            order.append(tok)
    return info, order


def snip(cap, n=14):
    return re.sub(r"[,。;、\s].*", "", cap)[:n] if cap else ""


def chain_text(toks, info):
    parts = []
    for t in toks:
        d, act, cap = info[t]
        s = snip(cap)
        if not s:
            return None
        parts.append(f"{act.split('-')[0]}「{s}」")
    return " → ".join(parts)


def build(row):
    theme = row["_theme"]
    gold = ITEM.findall(row["output"].split("</think>")[-1])
    info, order = parse_annot(row.get("_hist_annot", ""))
    gold = [t for t in order if t in set(gold)]          # 按时间序
    non = [t for t in order if t not in set(gold) and info[t][2]]
    gold_cap = [t for t in gold if info.get(t, ("", "", ""))[2]]
    if len(gold_cap) < 3 or len(non) < 3:
        return None
    # 支撑链 1-2 条(各 3 步),干扰链 1 条
    chains, answers = [], []
    c1 = gold_cap[:3]
    chains.append(c1); answers.append(True)
    if len(gold_cap) >= 6:
        chains.append(gold_cap[3:6]); answers.append(True)
    dis = rng.sample(non, 3)
    dis.sort(key=lambda t: order.index(t))
    chains.append(dis); answers.append(False)
    texts = [chain_text(c, info) for c in chains]
    if any(t is None for t in texts):
        return None
    idx = list(range(len(chains))); rng.shuffle(idx)
    letters = "ABCD"
    opts, gold_letters, dis_letter = [], [], None
    for pos, ci in enumerate(idx):
        opts.append(f"{letters[pos]}. {texts[ci]}")
        if answers[ci]:
            gold_letters.append(letters[pos])
        else:
            dis_letter = letters[pos]
    # 历史节选:链内条目的注释行(时间序)
    involved = {t for c in chains for t in c}
    lines = []
    for t in order:
        if t in involved:
            d, act, cap = info[t]
            lines.append(f"{d} [{act}] {snip(cap,20)}{t}")
    q = ("请根据用户交互历史和候选兴趣演化链，选择哪些历史行为支撑有效的兴趣演化。\n"
         "用户交互历史（节选）：\n" + "\n".join(lines) +
         f"\n候选兴趣演化目标：{theme}\n候选行为：\n" + "\n".join(opts) +
         "\n请输出支撑该演化目标的候选字母列表。/think")
    th_parts = []
    for pos, ci in enumerate(idx):
        if answers[ci]:
            th_parts.append(f"候选 {letters[pos]} 的行为链沿时间递进,与「{theme}」的演化方向一致,构成有效支撑。")
    th_parts.append(f"候选 {dis_letter} 中的行为与该主题缺乏递进关联,不能支撑此演化目标。")
    out = "<think>\n" + "\n".join(th_parts) + "\n</think>\n答案: [" + ", ".join(sorted(gold_letters)) + "]"
    return {"instruction": "", "input": q, "output": out, "history": []}


rows = []
for f in ["r2_gold_local", "r2_gold_g1", "r2_gold_g2", "r2_gold_v4"]:
    fp = P / f"{f}.jsonl"
    if fp.exists():
        rows += [json.loads(l) for l in open(fp)]
seen, made = set(), []
for r in rows:
    k = r.get("_src_idx")
    if k in seen:
        continue
    seen.add(k)
    s = build(r)
    if s:
        made.append(s)
with open(OUT, "w", encoding="utf-8") as f:
    for r in made:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"U2 候选链选择题: {len(made)} 条(原料 {len(seen)})→ {OUT}")
