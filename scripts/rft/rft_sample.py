#!/usr/bin/env python3
"""rft_sample.py — RFT 第一步:拒绝采样(vLLM 高温采样 K 次,命中 gold 的样本→回灌数据)。

原理(OneReason RL / roadmap §5 的 teacher-free 简化):
  评测是 Pass@64——模型采样命中率 >> greedy。把"采样碰得到 gold"的题变成
  "SFT 教材"(prompt→gold 直出),把运气蒸馏成能力。产出天然是【新样本】,
  绕开重复上采样毒药(rebal_mat_ep3=0.8454 的教训)。

流程:
  1. 读 LOO v2(gold 已知,gold∉历史保证预测型)
  2. 每题采样 K 次(temp 高探索): nothink 直出 3 token 形态(塌因分析:直通路是短板,
     RFT 优先修它;thinking 通路已相对健康)
  3. 命中 gold 的题 → 产出回灌样本(output=gold 直出);未命中的题记录 miss(难题池,后续用)
  4. 附带产出统计:每域 sample-pass@K(这就是"训练集外新鲜数据"上的真实泛化读数,
     替代已证伪的 offline_probe v1)

用法:
  $V/miniconda3/envs/verl_v071/bin/python scripts/rft/rft_sample.py \
      --model checkpoints/rebal_world_ep3 --loo data/processed/rec_loo_v2.jsonl \
      --gpu 3 --k 64 --n_per_dom 2000 --out_dir data/processed/rft_round1
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

SID = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
DOM_TOKEN = {"video": "<|video_begin|>", "ad": "<|ad_begin|>", "prod": "<|prod_begin|>", "living": "<|living_begin|>"}
GOLD = re.compile(r"<\|(video|ad|prod|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loo", required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n_per_dom", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    import random
    rng = random.Random(args.seed)
    from vllm import LLM, SamplingParams

    # ---- 读 LOO,按域配额 ----
    by_dom = defaultdict(list)
    for line in open(args.loo):
        r = json.loads(line)
        m = GOLD.search(r["output"].split("</think>")[-1])
        if m:
            by_dom[m.group(1)].append((r, (m.group(2), m.group(3), m.group(4))))
    items = []
    for dom, v in sorted(by_dom.items()):
        rng.shuffle(v)
        items.extend([(dom, r, g) for r, g in v[: args.n_per_dom]])
    print(f"采样题目: {Counter(d for d, _, _ in items)}", file=sys.stderr)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=40960,
              gpu_memory_utilization=0.85, enforce_eager=True, seed=42,
              enable_prefix_caching=True, trust_remote_code=True)

    # nothink 直出形态(与评测直通路一致): prompt 尾接空think + 域前缀,采样3 token
    prompts = []
    for dom, r, g in items:
        p = (f"<|im_start|>system\n{r['instruction']}<|im_end|>\n"
             f"<|im_start|>user\n{r['input']}<|im_end|>\n"
             f"<|im_start|>assistant\n<think>\n\n</think>\n{DOM_TOKEN[dom]}")
        prompts.append(p)
    sp = SamplingParams(n=args.k, max_tokens=3, temperature=args.temperature,
                        top_p=0.95, top_k=50, seed=42)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"采样完成 {time.time()-t0:.0f}s", file=sys.stderr)

    # ---- 筛选:命中 gold → 回灌样本 ----
    hits, misses = [], []
    stats = defaultdict(lambda: [0, 0])  # dom -> [n, hit]
    for (dom, r, gold), o in zip(items, outs):
        cands = []
        for seq in o.outputs:
            toks = re.findall(r"<s_([abc])_(\d+)>", seq.text)
            if len(toks) == 3 and [t[0] for t in toks] == ["a", "b", "c"]:
                cands.append(tuple(t[1] for t in toks))
        stats[dom][0] += 1
        if gold in set(cands):
            stats[dom][1] += 1
            hits.append({
                "instruction": r["instruction"], "input": r["input"],
                "output": r["output"],  # gold 直出(unCoT)
                "history": [],
                "meta_dom": dom,
                "meta_gold_rank": cands.index(gold),  # 第几次采样才命中(难度信号)
            })
        else:
            misses.append({"instruction": r["instruction"], "input": r["input"],
                           "output": r["output"], "history": [], "meta_dom": dom})

    report = {"model": args.model, "k": args.k, "temperature": args.temperature}
    for dom, (n, h) in sorted(stats.items()):
        report[f"sample_pass@{args.k}_{dom}"] = round(h / n, 4)
        print(f"[{dom}] sample-pass@{args.k}: {h}/{n} = {h/n:.3f}", file=sys.stderr)
    json.dump(report, open(f"{args.out_dir}/stats.json", "w"), indent=1)
    with open(f"{args.out_dir}/rft_hits.jsonl", "w") as g:
        for r in hits:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out_dir}/rft_misses.jsonl", "w") as g:
        for r in misses:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hits {len(hits)} / misses {len(misses)} -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
