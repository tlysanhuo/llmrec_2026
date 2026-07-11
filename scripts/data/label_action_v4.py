#!/usr/bin/env python3
"""label_action_v4.py — 官方两步产线完整复刻:teacher 从历史抽主题+选支撑交互(2026-07-07)。

v3 试点验尸:规则主题与历史真实语义脉络不符(teacher 零重叠/空选),必须让 teacher 自己抽主题。
一次调用产 {"theme":…, "items":[…]},few-shot 用真实种子样本锚定密度(种子 median=11)与措辞。
产出:R2 prompt 里的主题行替换为 teacher 主题,gold=teacher items,nothink 空 think。
质检:items∈历史硬校验 / 密度∈[3,35]且≤30%历史 / 主题长度8-40字 / JSON合法。
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ENV = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/configs/deepseek_api.env"
cfg = dict(l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.startswith("#"))
KEY, BASE, MODEL = cfg["YUNWU_API_KEY"], cfg["YUNWU_BASE_URL"], cfg["YUNWU_MODEL"]
import os as _os
BASE = _os.environ.get("TEACHER_BASE", BASE)
MODEL = _os.environ.get("TEACHER_MODEL", MODEL)
KEY = _os.environ.get("TEACHER_KEY", KEY)
NOTHINK = _os.environ.get("TEACHER_NOTHINK", "") == "1"
STRICT = _os.environ.get("TEACHER_STRICT", "") == "1"
ITEM = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")

TEACHER_SYS = (
    "你是快手的用户兴趣演化挖掘专家。给你一位用户的交互历史,请:"
    "①从历史中发现一条最清晰的兴趣演化脉络,命名为主题(措辞风格如「从泛化X到聚焦Y的…演化/决策」,8-25字);"
    "②选出支撑这条演化的全部历史交互(内容触达、兴趣深化、筛选比较、转化下单等环节;"
    "同环节强相关的同类交互都保留,主题之外的一律不选;典型 5-25 条,超过 30 条必是选多);"
    "③条目必须是历史中出现过的完整 itemic 标识,一字不改。"
    '只输出一个 JSON 对象:{"theme": "主题", "items": ["<|xx_begin|><s_a_..><s_b_..><s_c_..>", ...]},不要任何解释。'
)

_FS = json.load(open("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/fewshot_v2.json"))
FEWSHOT = [
    {"role": "user", "content": _FS["hist"]},
    {"role": "assistant", "content": json.dumps({"theme": _FS["theme"], "items": _FS["gold"]}, ensure_ascii=False)},
]


def call_teacher(hist_text):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": TEACHER_SYS}, *FEWSHOT,
                         {"role": "user", "content": hist_text}],
            "temperature": 0.2 if STRICT else 0.4, "max_tokens": 4000}
    if STRICT:
        body["messages"][0]["content"] += (
            "。硬性约束:items 不得超过 20 条——只选与主题最直接相关的核心交互,"
            "同类交互超过 4 条时只保留最有代表性的 4 条;宁可少选,严禁超长列表")
    if NOTHINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}/v1/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {KEY}"}, timeout=240)
            r.raise_for_status()
            content = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise RuntimeError("empty content")
            return content, r.json().get("usage", {})
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def process(i, row):
    q = row["input"]
    hist_teacher = row.get("_hist_annot") or q.split("角色任务")[0].strip()
    hist_items = set(ITEM.findall(q.split("角色任务")[0]))
    text, usage = call_teacher(hist_teacher)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return i, None, "无JSON|" + repr(text[:150]), usage
    try:
        obj = json.loads(m.group(0))
        theme, items = str(obj.get("theme", "")).strip(), obj.get("items", [])
    except Exception:
        return i, None, "JSON解析失败", usage
    if not (8 <= len(theme) <= 40):
        return i, None, f"主题长度{len(theme)}", usage
    if re.search(r"\d{3,}", theme):
        return i, None, f"主题含裸ID:{theme}", usage
    picked, seen = [], set()
    for x in items:
        mm = ITEM.search(x) if isinstance(x, str) else None
        if mm and mm.group(0) in hist_items and mm.group(0) not in seen:
            seen.add(mm.group(0)); picked.append(mm.group(0))
    if not (3 <= len(picked) <= 35) or len(picked) > 0.3 * max(len(hist_items), 1):
        return i, None, f"密度异常n={len(picked)}/hist={len(hist_items)}", usage
    # 用 teacher 主题替换原 prompt 的主题行
    new_input = re.sub(r"(主题[:：]).*", r"\g<1>" + theme, q, count=1)
    if theme not in new_input:
        return i, None, "主题替换失败", usage
    new = dict(row)
    new["input"] = new_input
    new["output"] = "<think>\n\n</think>\n" + json.dumps(picked, ensure_ascii=False)
    new["_n_gold"], new["_theme"] = len(picked), theme
    return i, new, None, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src)]
    rows = list(enumerate(rows))[a.start : (a.start + a.n) if a.n else None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["_src_idx"] for l in open(a.out)}
        print(f"resume: {len(done)} done")
    ok = fail = tin = tout = 0; golds = []
    with open(a.out, "a", encoding="utf-8") as g, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, i, r): i for i, r in rows if i not in done}
        for f in as_completed(futs):
            try:
                i, new, err, usage = f.result()
            except Exception as e:
                fail += 1; print(f"[{futs[f]}] EXC {str(e)[:60]}"); continue
            tin += usage.get("prompt_tokens", 0); tout += usage.get("completion_tokens", 0)
            if err:
                fail += 1; print(f"[{i}] QC-FAIL {err}")
            else:
                ok += 1; golds.append(new["_n_gold"]); new["_src_idx"] = i
                g.write(json.dumps(new, ensure_ascii=False) + "\n"); g.flush()
            if (ok + fail) % 100 == 0:
                print(f"progress ok={ok} fail={fail} tok={tin}/{tout}")
    if golds:
        golds.sort(); print(f"gold: median={golds[len(golds)//2]} mean={sum(golds)/len(golds):.1f}")
    print(f"DONE ok={ok} fail={fail} | tokens in={tin} out={tout}")


if __name__ == "__main__":
    main()
