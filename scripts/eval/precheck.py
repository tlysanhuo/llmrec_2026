#!/usr/bin/env python3
"""precheck.py — 上传前灾难体检(不是分数预测器)。

定位(2026-07-03,proxy套件删除后的替代物,职责收窄):
  ✗ 不预测线上分数(proxy已被5个真分证伪并删除)
  ✓ 只检查"灾难性行为"——已让我们浪费配额的三种死法:
    A. 采样解码复读崩溃(recipe2 0.7692 的死因: 温度采样下单item重复500+次到截断)
    B. itemic 三元组结构断裂(recipe2 伴生: <s_a_750>allenges<s_c_..> 混入英文子词)
    C. 任务格式丢失(recipe3 证伪死因: 选择题输出"该视频为…"而非单字母)
  输出 PASS/FAIL + 证据样本。FAIL=别传,省一次配额;PASS=行为正常,值得传。

输入样本直接从官方种子数据抽(不自造dev——那是旧proxy的错误来源之一)。
解码严格对齐线上: action_select=采样(temp0.6/top_p0.95/top_k20,max4096截为1024加速),
                 物料/推荐=空think前缀+贪婪(结构检查不需要beam), 选择题=贪婪短生成。

用法: python scripts/eval/precheck.py --model <ckpt或包目录> --gpu 3 [--n 30]
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

ITEM = re.compile(r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
# 结构断裂: begin后没跟s_a / s_a后没跟s_b / s_b后没跟s_c(允许答案在s_c后正常结束)
BROKEN = re.compile(r"<\|(?:prod|video|ad|living)_begin\|>(?!<s_a_)|<s_a_\d+>(?!<s_b_)|<s_b_\d+>(?!<s_c_)")
LETTER = re.compile(r"^[\s]*[ABCD]\b")

SEED_PATH = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/data_final.jsonl"
# 选择题体检用固定的常识题(不依赖已删的CMMLU dev), 8题足够看格式是否存活
MC_QUESTIONS = [
    ("中国的首都是哪个城市?\nA.上海\nB.北京\nC.广州\nD.深圳", "B"),
    ("一年有多少个月?\nA.10\nB.11\nC.12\nD.13", "C"),
    ("水的化学式是什么?\nA.CO2\nB.H2O\nC.O2\nD.NaCl", "B"),
    ("《红楼梦》的作者是谁?\nA.罗贯中\nB.施耐庵\nC.曹雪芹\nD.吴承恩", "C"),
    ("太阳从哪个方向升起?\nA.东\nB.南\nC.西\nD.北", "A"),
    ("人体最大的器官是什么?\nA.心脏\nB.肝脏\nC.皮肤\nD.大脑", "C"),
    ("光速大约是每秒多少公里?\nA.3万\nB.30万\nC.300万\nD.3000万", "B"),
    ("春节是农历几月初一?\nA.十二月\nB.二月\nC.三月\nD.正月", "D"),
]
MC_TPL = "请回答以下问题：\n\n{q}\n\n请按以下格式作答：\"正确答案是 (在此处填写选项字母)\"/no_think"
MC_SYS = "你是一个非常聪明的助手，请直接遵循指示作答。"


def load_seed_samples(n, rng):
    """从种子数据抽 action_select(JSON输出) 和 rec(单item输出) 两类输入。"""
    sel, rec = [], []
    with open(SEED_PATH) as f:
        recs = [json.loads(l) for l in f]
    rng.shuffle(recs)
    for r in recs:
        body = r["output"].split("</think>")[-1].strip()
        if body.startswith("[") and len(sel) < n:
            sel.append(r)
        elif ITEM.search(body) and len(ITEM.findall(body)) == 1 and len(rec) < n:
            rec.append(r)
        if len(sel) >= n and len(rec) >= n:
            break
    return sel, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True).cuda().eval()

    def chat(sys_p, usr):
        msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": usr}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen(prompts, max_new, sample, bs=8):
        outs = []
        kw = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if sample else dict(do_sample=False)
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            inp = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
            for j in range(len(chunk)):
                outs.append(tok.decode(o[j][inp.input_ids.shape[1]:], skip_special_tokens=True))
            print(f"    {min(i+bs,len(prompts))}/{len(prompts)}", file=sys.stderr)
        return outs

    sel, rec = load_seed_samples(args.n, rng)
    report = {}
    evidence = []

    # ---- A. 采样解码复读(action_select, recipe2死因) ----
    print("[A] 采样复读检查 ...", file=sys.stderr)
    prompts = [chat(r.get("instruction", ""), r["input"]) for r in sel]
    outs = gen(prompts, max_new=1024, sample=True)
    crash = 0
    for o in outs:
        toks = ITEM.findall(o.split("</think>")[-1])
        if toks:
            _, cnt = Counter(toks).most_common(1)[0]
            if cnt >= 20:
                crash += 1
                if len(evidence) < 2:
                    evidence.append(("A复读", o[-150:]))
    report["A_采样复读崩溃率"] = (crash / max(len(outs), 1), crash, len(outs), 0.15)

    # ---- B. itemic结构断裂(采样输出 + 贪婪rec输出 一起查) ----
    print("[B] itemic结构检查 ...", file=sys.stderr)
    rec_prompts = []
    for r in rec:
        body = r["output"].split("</think>")[-1]
        dom = re.search(r"<\|(prod|video|ad|living)_begin\|>", body).group(1)
        rec_prompts.append(chat(r.get("instruction", ""), r["input"]) + f"<think>\n</think>\n<|{dom}_begin|>")
    rec_outs = gen(rec_prompts, max_new=10, sample=False)
    broken = 0
    all_texts = outs + rec_outs
    for o in all_texts:
        # 排除文本换行假阳性: 只在无换行紧邻处判断
        hits = [m for m in BROKEN.finditer(o) if "\n" not in o[m.end():m.end() + 3]]
        # 末尾截断(生成被max_new截断的不算断裂)
        hits = [m for m in hits if m.end() < len(o) - 12]
        if hits:
            broken += 1
            if len(evidence) < 4:
                evidence.append(("B断裂", o[max(0, hits[0].start() - 15):hits[0].end() + 25]))
    report["B_itemic结构断裂率"] = (broken / max(len(all_texts), 1), broken, len(all_texts), 0.10)

    # ---- C. 选择题格式(recipe3死因) ----
    print("[C] 选择题格式检查 ...", file=sys.stderr)
    mc_prompts = [chat(MC_SYS, MC_TPL.format(q=q)) for q, _ in MC_QUESTIONS]
    mc_outs = gen(mc_prompts, max_new=24, sample=False)
    fmt_ok = 0
    correct = 0
    for o, (_, gold) in zip(mc_outs, MC_QUESTIONS):
        body = o.split("</think>")[-1]
        m = re.search(r"[ABCD]", body)
        if m:
            fmt_ok += 1
            if m.group(0) == gold:
                correct += 1
        elif len(evidence) < 6:
            evidence.append(("C格式丢失", body[:100]))
    report["C_选择题格式存活率"] = (fmt_ok / len(MC_QUESTIONS), fmt_ok, len(MC_QUESTIONS), None)
    report["C_附带_简单题答对"] = (correct / len(MC_QUESTIONS), correct, len(MC_QUESTIONS), None)

    # ---- 判定 ----
    print("\n" + "=" * 60)
    print(f"体检对象: {args.model}")
    fail = False
    for k, (rate, num, den, thresh) in report.items():
        if k.startswith("A") or k.startswith("B"):
            verdict = "FAIL" if rate > thresh else "ok"
            if rate > thresh:
                fail = True
            print(f"  {k}: {num}/{den} = {rate:.1%}  (阈值{thresh:.0%})  [{verdict}]")
        elif "格式存活" in k:
            verdict = "FAIL" if rate < 0.75 else "ok"
            if rate < 0.75:
                fail = True
            print(f"  {k}: {num}/{den} = {rate:.1%}  (阈值≥75%)  [{verdict}]")
        else:
            print(f"  {k}: {num}/{den} = {rate:.1%}  (参考)")
    print("-" * 60)
    print(f"结论: {'❌ FAIL — 存在灾难行为, 不建议花配额上传' if fail else '✅ PASS — 无灾难行为, 可上传(不代表分数会涨)'}")
    if evidence:
        print("\n证据样本:")
        for tag, e in evidence:
            print(f"  [{tag}] ...{e!r}")


if __name__ == "__main__":
    main()
