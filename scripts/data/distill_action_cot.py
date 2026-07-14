#!/usr/bin/env python3
"""distill_action_cot.py — 用 DeepSeek v4 flash 为 R2 action-select 样本蒸馏推理链(2026-07-06)。

原理:R2 金标免费(规则构建,提取式),teacher 只写"为什么选这些/排除那些"的推理过程。
产出:每条 R2 样本 → /think 变体(output = <think>\n{CoT}\n</think>\n{金标JSON},prompt 尾缀改 /think)。
质检:金标字节不变;CoT 中出现的 SID 必须 ∈ 历史;长度窗口;无 markdown 围栏。

用法:
  python scripts/data/distill_action_cot.py --src /tmp/r2_smoke.jsonl --out data/processed/r2_cot_pilot.jsonl --n 8
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ENV = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/configs/secrets/deepseek_api.env"
cfg = dict(l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.startswith("#"))
KEY, BASE = cfg["DEEPSEEK_API_KEY"], cfg["DEEPSEEK_BASE_URL"]
MODEL = "deepseek-v4-flash"
SID = re.compile(r"<s_[abc]_\d+>")

TEACHER_SYS = (
    "你是快手推荐系统的资深算法专家。给你一道「从用户交互历史中提取与主题相关条目」的题目和它的正确答案,"
    "请写出一段第一人称的推理过程,模拟模型自己解题时的思考。要求:"
    "①先一句话解读主题的演化脉络;②按时间顺序梳理历史中与主题相关的交互,选中的条目要引用其完整 itemic 标识"
    "(如 <|prod_begin|><s_a_123><s_b_456><s_c_789>)并说明关联逻辑;③点名 1-2 类看似相关但应排除的交互及排除理由;"
    "④若答案为空数组,说明为什么历史中没有相关条目。"
    "长度 150-450 字,纯中文叙述,不用列表符号,不用 markdown,不要出现「答案」「金标」「题目给出」等字眼,"
    "结尾自然收束(如「综上,相关交互如下」)。只输出推理过程本身。"
)


def call_teacher(prompt_text, gold):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": TEACHER_SYS},
            {"role": "user", "content": f"题目:\n{prompt_text}\n\n正确答案:\n{gold}"},
        ],
        "temperature": 0.6,
        "max_tokens": 2500,   # v4-flash 是推理模型,内部思考也占 completion 配额,给足余量
    }
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {KEY}"}, timeout=180)
            r.raise_for_status()
            j = r.json()
            content = (j["choices"][0]["message"].get("content") or "").strip()
            if not content:            # 思考吃光配额 → 加码重试
                body["max_tokens"] = 4000
                raise RuntimeError("empty content")
            return content, j.get("usage", {})
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def qc(cot, hist_sids):
    if not (100 <= len(cot) <= 900):
        return f"len={len(cot)}"
    if "```" in cot or "答案" in cot or "金标" in cot:
        return "禁词/围栏"
    bad = [s for s in SID.findall(cot) if s not in hist_sids]
    if bad:
        return f"SID越界x{len(bad)}"
    return None


def process(i, row):
    prompt = row["input"]
    gold = row["output"].split("</think>\n", 1)[1]  # 剥掉空 think
    hist_sids = set(SID.findall(prompt))
    cot, usage = call_teacher(prompt, gold)
    err = qc(cot, hist_sids)
    if err:
        return i, None, err, usage
    new = dict(row)
    new["input"] = re.sub(r"/no_think\s*$", "/think", prompt)
    new["output"] = f"<think>\n{cot}\n</think>\n{gold}"
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
    done_ids = set()
    if os.path.exists(a.out):  # 断点续跑
        done_ids = {json.loads(l)["_src_idx"] for l in open(a.out)}
        print(f"resume: {len(done_ids)} already done")

    tot_in = tot_out = ok = fail = 0
    with open(a.out, "a", encoding="utf-8") as g, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, i, r): i for i, r in enumerate(rows) if i not in done_ids}
        for f in as_completed(futs):
            try:
                i, new, err, usage = f.result()
            except Exception as e:
                fail += 1
                print(f"[{futs[f]}] EXC {type(e).__name__}: {e}")
                continue
            tot_in += usage.get("prompt_tokens", 0)
            tot_out += usage.get("completion_tokens", 0)
            if err:
                fail += 1
                print(f"[{i}] QC-FAIL {err}")
            else:
                ok += 1
                new["_src_idx"] = i
                g.write(json.dumps(new, ensure_ascii=False) + "\n")
                g.flush()
            if (ok + fail) % 50 == 0:
                print(f"progress {ok+fail}/{len(futs)} ok={ok} fail={fail} tok={tot_in}/{tot_out}")
    print(f"DONE ok={ok} fail={fail} | tokens in={tot_in} out={tot_out}")


if __name__ == "__main__":
    main()
