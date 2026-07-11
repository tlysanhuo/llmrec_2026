#!/usr/bin/env python3
"""build_u3_topic.py — U3 主题生成/逻辑链块(2026-07-07,评测形态对齐 PPT p15-17)。

原理:种子全库仅 6 条 topic 题=全场空地;评测计分=action 最优有序匹配+logic 文本 ROUGE。
产线:teacher 读注释版历史 → 输出 {name, events:[{date, token, act, logic}]}(3-5步,时间升序);
成品:input=token历史+官方角色任务/规则/格式段(逐字取自种子)+主题+/no_think;
      output=空think+logic_chain JSON,action="[行为] token"(严格对应Timeline),logic=「关键词:说明」体。
质检:事件3-5;token∈历史且行为/日期与该行一致;logic 15-70字含冒号;JSON schema 严格。
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

GEN = os.environ.get("TEACHER_GEN", "http://127.0.0.1:8123")
MODEL = "qwen3-8b"
ITEM = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")

# 官方角色任务/规则/格式段(逐字复用种子 topic 题,保 ROUGE 口径)
SEED_TOPIC = None
for l in open("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/blocks_v1/block_nonrec.jsonl"):
    r = json.loads(l)
    if "logic_chain" in r.get("input", ""):
        SEED_TOPIC = r["input"]
        break
HEAD_END = SEED_TOPIC.index("角色任务")
RULES = SEED_TOPIC[HEAD_END: SEED_TOPIC.index("请面向以下主题")]
FMT = SEED_TOPIC[SEED_TOPIC.index("输出格式"):].rsplit("/think", 1)[0]

TEACHER_SYS = (
    "你是快手的用户兴趣演化挖掘专家。给你一位用户的带文字注释的交互历史,请提取一条高质量行为逻辑链:"
    "①链长 3-5 步,按时间升序;后一步与前一步必须构成「场景需求补全/兴趣因果递进/需求深度细化」之一,"
    "严禁并列罗列;②给链命名(措辞如「从X到Y的…演化/决策链」);"
    "③每步给 logic:格式「逻辑关键词:一句说明」(如「初始触发:基于生理不适的泛化求助。」),15-60字,"
    "说明须以该步交互内容为依据;④每步给出该交互的完整 itemic 标识与行为类型,一字不改。"
    '只输出 JSON:{"name":"...","events":[{"date":"YYYY-MM-DD","act":"视频-长播","token":"<|video_begin|><s_a_1><s_b_2><s_c_3>","logic":"..."}]}'
)


def call_teacher(hist):
    body = {"model": MODEL, "temperature": 0.5, "max_tokens": 900,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": TEACHER_SYS},
                         {"role": "user", "content": hist}]}
    for a in range(3):
        try:
            r = requests.post(f"{GEN}/v1/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            c = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if c:
                return c
        except Exception:
            time.sleep(2 * (a + 1))
    raise RuntimeError("call failed")


def hist_index(q):
    """token→(date, act) 按 Timeline 行解析(token历史版)。"""
    idx, cur = {}, None
    for line in q.split("\n"):
        m = re.match(r"【(\d{4}-\d{2}-\d{2})】", line.strip())
        if m:
            cur = m.group(1); continue
        m = re.match(r"--:-- \[([^\]]+)\] (<\|.+)$", line.strip())
        if m and cur:
            t = ITEM.search(m.group(2))
            if t and t.group(0) not in idx:
                idx[t.group(0)] = (cur, m.group(1))
    return idx


def process(i, row):
    q = row["input"]
    hist_block = q.split("\n\n")[0]
    idx = hist_index(hist_block)
    if len(idx) < 15:
        return i, None, "历史过短"
    text = call_teacher(row.get("_hist_annot") or hist_block)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return i, None, "无JSON"
    try:
        j = json.loads(m.group(0))
        name, evs = str(j["name"]).strip(), j["events"]
    except Exception:
        return i, None, "JSON解析失败"
    if not (8 <= len(name) <= 40) or re.search(r"\d{3,}", name):
        return i, None, f"主题不合格:{name[:20]}"
    if not (3 <= len(evs) <= 5):
        return i, None, f"链长{len(evs)}"
    events, last = [], ""
    for e in evs:
        t = ITEM.search(str(e.get("token", "")))
        lg = str(e.get("logic", "")).strip()
        if not t or t.group(0) not in idx:
            return i, None, "token不在历史"
        d, act = idx[t.group(0)]
        if str(e.get("date", "")) != d:
            d = d  # 以历史为准
        if d < last:
            return i, None, "时间乱序"
        last = d
        if not (15 <= len(lg) <= 70) or (":" not in lg and ":" not in lg):
            return i, None, f"logic不合格:{lg[:20]}"
        events.append({"date": d, "action": f"[{act}] {t.group(0)}", "logic": lg})
    new_input = (hist_block + "\n\n" + RULES + "请面向以下主题提取行为逻辑链:\n主题:" + name +
                 "\n\n" + FMT + "/no_think")
    out = {"logic_chain": {"name": name, "events": events}}
    rec = {"instruction": row.get("instruction", ""), "input": new_input,
           "output": "<think>\n\n</think>\n" + json.dumps(out, ensure_ascii=False), "history": []}
    rec["_n_ev"] = len(events)
    return i, rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    rows = list(enumerate(json.loads(l) for l in open(a.src)))
    rows = rows[a.start: (a.start + a.n) if a.n else None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["_src_idx"] for l in open(a.out)}
        print(f"resume: {len(done)}", flush=True)
    ok = fail = 0
    with open(a.out, "a", encoding="utf-8") as g, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, i, r): i for i, r in rows if i not in done}
        for f in as_completed(futs):
            try:
                i, new, err = f.result()
            except Exception as e:
                fail += 1; print(f"[{futs[f]}] EXC {str(e)[:60]}", flush=True); continue
            if err:
                fail += 1; print(f"[{i}] QC-FAIL {err}", flush=True)
            else:
                ok += 1; new["_src_idx"] = i
                g.write(json.dumps(new, ensure_ascii=False) + "\n"); g.flush()
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
