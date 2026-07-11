#!/usr/bin/env python3
"""label_action_gold.py — 按官方产线用 LLM 重标 action_select 的 gold(2026-07-07,v3)。

背景(07-07 审计定案):
  - 评测/种子 gold = API 标注,种子密度 median=11、空gold=0%;规则版 median=3、空18% → 召回系统性偏低。
  - 评测与种子的懂用户全部走 /no_think 空 think ⇒ 本管线只产 nothink 样本,不写 CoT。
流程:R2 规则版样本(prompt 格式已证与评测对齐)→ teacher 重选相关交互 → 三道质检
  (SID∈历史硬校验 / JSON合法 / 与规则gold交叉:规则gold应基本⊂teacher gold,重叠率<40%丢弃)。
用法:
  python scripts/data/label_action_gold.py --src data/processed/r2_base_v2.jsonl \
      --out data/processed/r2_gold_v3.jsonl --n 20 --workers 8
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ENV = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/configs/deepseek_api.env"
cfg = dict(l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.startswith("#"))
KEY, BASE, MODEL = cfg["YUNWU_API_KEY"], cfg["YUNWU_BASE_URL"], cfg["YUNWU_MODEL"]
ITEM = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")

TEACHER_SYS = (
    "你是快手的用户行为数据挖掘专家。给你一位用户的交互历史和一个兴趣演化主题,"
    "请提取出与该主题相关的历史交互条目。要求:"
    "①历史中绝大多数交互与主题无关,必须严格甄别:只选确实处于主题演化脉络上的交互"
    "(内容触达、兴趣深化、比价筛选、下单转化等环节);"
    "②同一环节的强相关同类交互都要保留,但主题之外的一律不选;"
    "③典型答案在 5-25 条之间,超过 30 条几乎必然是选多了,需要重新收紧;"
    "④每个条目必须是历史中出现过的完整 itemic 标识,一字不改;"
    "⑤只输出一个 JSON 数组,元素为完整 itemic 标识字符串(含 <|xx_begin|> 前缀),不要任何解释。"
)

# few-shot 校准锚:一条真实官方标注样本(题面+标准答案),锁定选取密度与口径
_FS = json.load(open("/tmp/fewshot_seed.json"))
FEWSHOT_MSGS = [
    {"role": "user", "content": _FS["q"]},
    {"role": "assistant", "content": _FS["a"]},
]


def call_teacher(prompt_text):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": TEACHER_SYS},
            *FEWSHOT_MSGS,
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}/v1/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {KEY}"}, timeout=240)
            r.raise_for_status()
            j = r.json()
            content = (j["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise RuntimeError("empty content")
            return content, j.get("usage", {})
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def parse_gold(text, hist_items):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None, "无JSON数组"
    try:
        arr = json.loads(m.group(0))
    except Exception:
        arr = ITEM.findall(m.group(0))  # 容错:直接抽 itemic
    picked, seen = [], set()
    for x in arr:
        if not isinstance(x, str):
            continue
        mm = ITEM.search(x)
        if not mm:
            continue
        s = mm.group(0)
        if s in hist_items and s not in seen:
            seen.add(s)
            picked.append(s)
    return picked, None


def process(i, row):
    # 剥掉尾部输出示例之外的原题(整题给 teacher,含主题与格式要求)
    q = row["input"]
    hist_items = set(ITEM.findall(q.split("角色任务")[0]))  # 只认历史区的条目
    text, usage = call_teacher(q)
    picked, err = parse_gold(text, hist_items)
    if err:
        return i, None, err, usage
    if not picked:
        return i, None, "空gold", usage
    if len(picked) > 35 or len(picked) > 0.3 * max(len(hist_items), 1):
        return i, None, f"选取过密n={len(picked)}/hist={len(hist_items)}", usage
    rule_gold = set(ITEM.findall(row["output"].split("</think>")[-1]))
    overlap = len(rule_gold & set(picked)) / max(len(rule_gold), 1) if rule_gold else 1.0
    if rule_gold and overlap < 0.4:
        return i, None, f"与规则gold重叠{overlap:.0%}", usage
    new = dict(row)
    new["output"] = "<think>\n\n</think>\n" + json.dumps(picked, ensure_ascii=False)
    new["_n_gold"] = len(picked)
    new["_overlap"] = round(overlap, 2)
    return i, new, None, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src)]
    if a.n:
        rows = rows[: a.n]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["_src_idx"] for l in open(a.out)}
        print(f"resume: {len(done)} done")
    ok = fail = tin = tout = 0
    golds = []
    with open(a.out, "a", encoding="utf-8") as g, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, i, r): i for i, r in enumerate(rows) if i not in done}
        for f in as_completed(futs):
            try:
                i, new, err, usage = f.result()
            except Exception as e:
                fail += 1
                print(f"[{futs[f]}] EXC {type(e).__name__}: {str(e)[:80]}")
                continue
            tin += usage.get("prompt_tokens", 0)
            tout += usage.get("completion_tokens", 0)
            if err:
                fail += 1
                print(f"[{i}] QC-FAIL {err}")
            else:
                ok += 1
                golds.append(new["_n_gold"])
                new["_src_idx"] = i
                g.write(json.dumps(new, ensure_ascii=False) + "\n")
                g.flush()
            if (ok + fail) % 100 == 0:
                print(f"progress ok={ok} fail={fail} tok={tin}/{tout}")
    if golds:
        golds.sort()
        print(f"gold条数: median={golds[len(golds)//2]} mean={sum(golds)/len(golds):.1f}")
    print(f"DONE ok={ok} fail={fail} | tokens in={tin} out={tout}")


if __name__ == "__main__":
    main()
