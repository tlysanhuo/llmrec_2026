#!/usr/bin/env python3
"""build_p3_cot.py — 懂推荐 CoT 重写(2026-07-07,官方产线:PPT p32 拒绝采样 + 沙龙页 I-A-D + p20 双路启示)。

流程(每条 video/prod 样本):
  1) 语义对照表:题面历史 token → caption(sid2cap 反查,teacher 不看 gold);
  2) teacher(GPU0)不见 gold 生成 K=2 条压缩 I-A-D CoT(归纳1句→溯因1-2句→演绎1-2句,100-200字);
  3) judge(GPU1)看 gold 的语义(caption)打分:rel(演绎方向与 gold 相关 0-5)+ copy(是否史内复述);
  4) 取 rel 最高且 rel>=3 且 !copy 的 CoT,组装 <think>CoT</think>+原答案;失败样本原样保留计数。
质检:CoT 100-450字、不得含任何 itemic 记号、无"答案/gold"字样。
用法: TEACHER_GEN=http://127.0.0.1:8123 TEACHER_JUDGE=http://127.0.0.1:8124 \
  python scripts/data/build_p3_cot.py --src data/processed/blocks_v1/block_rec_video.jsonl \
    --out data/processed/p3_video_cot.jsonl --n 20 --workers 8
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

GEN = os.environ.get("TEACHER_GEN", "http://127.0.0.1:8123")
JUD = os.environ.get("TEACHER_JUDGE", "http://127.0.0.1:8124")
MODEL = "qwen3-8b"
ITEM = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
DOM = {"video": "video", "prod": "prod", "ad": "ad", "living": "living"}

print("[load] sid2cap ...", flush=True)
S2C = json.load(open("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/sid2cap.json"))
print(f"[load] {len(S2C):,}", flush=True)


def cap_of(m):
    d = DOM[m.group(1)]
    return S2C.get(f"{d}|{m.group(2)}|{m.group(3)}|{m.group(4)}")


GEN_SYS = (
    "你是快手推荐系统的思考引擎。给你一道「预测用户接下来感兴趣物料」的题目和历史物料的语义对照表,"
    "写一段推理(将作为模型的思考过程):"
    "①归纳(1句):用户历史的核心兴趣点;"
    "②溯因(1-2句):这些兴趣背后的原因/深层需求——这是主导段;"
    "③演绎(1-2句):由原因推出用户接下来可能产生的新兴趣方向,自然收束。"
    "硬性要求:100-200字;不得出现任何形如 <|xx|> 或 <s_a_..> 的记号;"
    "不要复述罗列历史条目,演绎要往前走一步而非重复已看内容;不用列表,纯中文叙述。只输出这段推理。"
)

JUD_SYS = (
    "你是数据质检裁判。给你:一段推荐思考(CoT)、用户实际接下来交互的物料语义(gold)。"
    "评两项:rel=CoT 演绎出的兴趣方向与 gold 语义的相关度(0-5,5=方向精准覆盖);"
    "copy=CoT 是否只是复述历史物料而没有向前推理(true/false)。"
    '只输出 JSON:{"rel": n, "copy": true/false}'
)


def call(base, sys, user, max_tokens, temperature):
    body = {"model": MODEL, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    for a in range(3):
        try:
            r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            c = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if c:
                return c
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    raise RuntimeError("call failed")


def process(i, row):
    q = row["input"]
    ans = row["output"].split("</think>")[-1].lstrip("\n")
    hist_caps, gold_caps = [], []
    ans_start = len(q)
    for m in ITEM.finditer(q):
        c = cap_of(m)
        if c:
            hist_caps.append(c)
    for m in ITEM.finditer(ans):
        c = cap_of(m)
        if c:
            gold_caps.append(c)
    if len(hist_caps) < 10 or not gold_caps:
        new = dict(row); new["_rel"] = -1   # 语义不可查⇒保留原样(原CoT)
        return i, new, None
    table = "\n".join(f"- {c}" for c in hist_caps[:80])
    gen_user = f"题目:\n{q[:6000]}\n\n历史物料语义对照表:\n{table}\n\n请写这段推理。"
    body = {"model": MODEL, "n": 4, "temperature": 0.85, "max_tokens": 500,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": GEN_SYS}, {"role": "user", "content": gen_user}]}
    cots = []
    for a in range(3):
        try:
            r = requests.post(f"{GEN}/v1/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            cots = [(ch["message"].get("content") or "").strip() for ch in r.json()["choices"]]
            break
        except Exception:
            time.sleep(2 * (a + 1))
    cots = [c for c in cots if 100 <= len(c) <= 450 and "<s_" not in c and "<|" not in c
            and "答案" not in c and "gold" not in c.lower()]
    if not cots:
        return i, None, "生成全废"
    gold_txt = "\n".join(f"- {c}" for c in gold_caps[:10])
    best, best_rel = None, -1
    for c in cots:
        try:
            v = call(JUD, JUD_SYS, f"CoT:\n{c}\n\ngold 物料语义:\n{gold_txt}", 120, 0.0)
            j = json.loads(re.search(r"\{.*\}", v, re.S).group(0))
            rel, cp = int(j.get("rel", 0)), bool(j.get("copy", False))
        except Exception:
            continue
        if not cp and rel > best_rel:
            best, best_rel = c, rel
    if best is None or best_rel < 3:
        # 良率即分拣:judge拒绝 ⇒ exploit型样本,转nothink直出(P0同款字节机制)
        new = dict(row)
        body = new["output"].split("</think>", 1)[-1].lstrip("\n")
        new["input"] = re.sub(r"/think\s*$", "/no_think", new["input"])
        new["output"] = "<think>\n\n</think>\n" + body
        new["_rel"] = 0
        return i, new, None
    new = dict(row)
    new["output"] = f"<think>\n{best}\n</think>\n{ans}"
    new["_rel"] = best_rel
    return i, new, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    rows = list(enumerate(json.loads(l) for l in open(a.src)))
    rows = rows[a.start: (a.start + a.n) if a.n else None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["_src_idx"] for l in open(a.out)}
        print(f"resume: {len(done)}")
    ok = fail = 0; rels = []
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
                ok += 1; rels.append(new["_rel"]); new["_src_idx"] = i
                g.write(json.dumps(new, ensure_ascii=False) + "\n"); g.flush()
            if (ok + fail) % 100 == 0:
                print(f"progress ok={ok} fail={fail}", flush=True)
    if rels:
        import collections
        print("rel分布:", dict(collections.Counter(rels)))
    print(f"DONE ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
