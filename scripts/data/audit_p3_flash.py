#!/usr/bin/env python3
"""audit_p3_flash.py — flash 复核 P3 裁决(2026-07-07):抽 300 条已过检 CoT,flash 重判,量 8B 自评偏好。"""
import json, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

cfg = dict(l.strip().split("=", 1) for l in open("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/configs/secrets/deepseek_api.env") if "=" in l and not l.startswith("#"))
KEY, BASE, MODEL = cfg["YUNWU_API_KEY"], cfg["YUNWU_BASE_URL"], "deepseek-v4-flash"
ITEM = re.compile(r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
JUD_SYS = ("你是数据质检裁判。给你:一段推荐思考(CoT)、用户实际接下来交互的物料语义(gold)。"
           "评两项:rel=CoT 演绎出的兴趣方向与 gold 语义的相关度(0-5,5=方向精准覆盖);"
           'copy=CoT 是否只是复述历史物料而没有向前推理。只输出 JSON:{"rel": n, "copy": true/false}')

print("[load] sid2cap...", flush=True)
S2C = json.load(open("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/sid2cap.json"))
rng = random.Random(2026)
pool = []
for f in ["p3_video_cot", "p3_prod_cot"]:
    rows = [json.loads(l) for l in open(f"/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/{f}.jsonl")]
    acc = [r for r in rows if r["_rel"] >= 3]
    rng.shuffle(acc)
    pool += [(f, r) for r in acc[:150]]
print("抽检池:", len(pool), flush=True)

def gold_caps(r):
    ans = r["output"].split("</think>")[-1]
    return [S2C.get(f"{m.group(1)}|{m.group(2)}|{m.group(3)}|{m.group(4)}") for m in ITEM.finditer(ans)]

def audit(item):
    f, r = item
    t = r["output"]
    cot = t[t.find("<think>") + 8: t.find("</think>")].strip()
    gc = [c for c in gold_caps(r) if c][:10]
    if not gc:
        return None
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 400,
            "messages": [{"role": "system", "content": JUD_SYS},
                         {"role": "user", "content": "CoT:\n" + cot + "\n\ngold 物料语义:\n" + "\n".join("- " + c for c in gc)}]}
    for a in range(3):
        try:
            resp = requests.post(f"{BASE}/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {KEY}"}, timeout=180)
            resp.raise_for_status()
            c = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            j = json.loads(re.search(r"\{.*\}", c, re.S).group(0))
            return f, r["_rel"], int(j.get("rel", -9)), bool(j.get("copy", False))
        except Exception:
            time.sleep(2 * (a + 1))
    return None

res = []
with ThreadPoolExecutor(10) as ex:
    futs = [ex.submit(audit, it) for it in pool]
    for fu in as_completed(futs):
        v = fu.result()
        if v:
            res.append(v)
            if len(res) % 50 == 0:
                print(f"已复核 {len(res)}", flush=True)
import collections
agree = sum(1 for _, l8, lf, cp in res if lf >= 3 and not cp)
低判 = sum(1 for _, l8, lf, cp in res if lf < 3)
判抄 = sum(1 for _, l8, lf, cp in res if cp)
print(f"复核 {len(res)} 条 | flash认可(rel≥3且非抄) {agree} ({agree/len(res):.0%}) | flash判rel<3 {低判} | flash判copy {判抄}")
d = collections.Counter((l8, lf) for _, l8, lf, cp in res)
print("8B分↔flash分 联表(前12):", dict(sorted(d.items(), key=lambda x: -x[1])[:12]))
