#!/usr/bin/env python3
"""offline_probe.py — 机制对齐的离线分辨探针(NOT a score predictor)。

★定位(2026-07-03 建,吸取旧 proxy 全删的教训):
  ✓ 分维度回答"ckpt A vs B 谁更好"——机制与平台评测 1:1 对齐(beam 宽度/两通路/prompt格式)
  ✗ 不预测线上绝对分,不做上传决策依据(唯一裁决=平台真分)
  ✗ LOO gold 与平台评测集分布不同 → 绝对值无意义,只看版本间 Δ

维度:
  rec  — 推荐四域 Pass@K:两阶段(采样thinking→beam32×3tok) + nothink直通(beam32×3tok),
         合并64候选查 gold 命中。附:直通路抄史率/新候选数/去重数(塌因分析验证过的行为指标)。
         数据=rec_loo.jsonl(LOO,四域各3000,默认每域抽 --n_rec 条)。
  mat  — 物料 desc→token Pass@64:beam64×3tok(平台 grounding 机制)。
         数据=data_final 中 mat_desc2token 类样本(默认抽 --n_mat 条)。

prompt 格式 1:1 复刻平台日志(qwen3_soft_switch 展开形):
  <|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
  nothink 通路: 追加 "<think>\n\n</think>\n" 空think前缀(与日志一致);
  thinking 通路: assistant 起始直接采样(平台 Auto Thinking Enabled 对应),
                 采出的 thinking + "</think>\n" 后接 beam。
  LOO 数据 input 内含 "/no_think" 尾缀,与平台一致保留。

用法:
  $V/miniconda3/envs/verl_v071/bin/python scripts/eval/offline_probe.py \
      --model checkpoints/xxx --gpu 3 [--dims rec,mat] [--n_rec 150] [--n_mat 300] \
      [--tag xxx] [--wandb]
输出: logs/probe/<tag>_<date>.json + 控制台表;--wandb 时记入 llmrec-2026 项目 run=probe_<tag>。
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime

PROJ = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
LOO = f"{PROJ}/data/processed/rec_loo.jsonl"
SEED = f"{PROJ}/data/processed/data_final.jsonl"
SID = re.compile(r"<\|(video|ad|prod|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
DOM_TOKEN = {"video": "<|video_begin|>", "ad": "<|ad_begin|>", "prod": "<|prod_begin|>", "living": "<|living_begin|>"}


def build_prompt(instruction, user_input, mode):
    """mode: 'nothink' = 空think前缀(直通路/物料); 'think' = 让模型自己生成thinking"""
    p = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    if mode == "nothink":
        p += "<think>\n\n</think>\n"
    return p


def parse_gold(output):
    m = SID.search(output.split("</think>")[-1])
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--dims", default="rec,mat")
    ap.add_argument("--n_rec", type=int, default=150, help="每域抽样条数")
    ap.add_argument("--n_mat", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tag = args.tag or os.path.basename(args.model.rstrip("/"))

    import random
    rng = random.Random(args.seed)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=40960,
              gpu_memory_utilization=0.85, enforce_eager=True, seed=42,
              enable_prefix_caching=True, trust_remote_code=True,
              max_logprobs=130)  # vLLM beam_search 内部需要 2×beam_width 个 logprobs(beam64→128)
    report = {"model": args.model, "tag": tag, "date": datetime.now().isoformat()[:19]}

    def beam_decode(prompts, width):
        """beam search width × 3 token, 返回每条 prompt 的候选列表 [(a,b,c)...]
        ⚠️ vLLM beam_search 的 seq.text 含 prompt 前缀——只解析新生成部分"""
        params = BeamSearchParams(beam_width=width, max_tokens=3)
        outs = llm.beam_search([{"prompt": p} for p in prompts], params)
        res = []
        for p, o in zip(prompts, outs):
            cands = []
            for seq in o.sequences:
                gen = seq.text[len(p):] if seq.text.startswith(p) else seq.text
                toks = re.findall(r"<s_([abc])_(\d+)>", gen)
                if len(toks) == 3 and [t[0] for t in toks] == ["a", "b", "c"]:
                    cands.append(tuple(t[1] for t in toks))
                else:
                    cands.append(None)
            res.append(cands)
        return res

    # ============ rec 维 ============
    if "rec" in args.dims:
        by_dom = {}
        for line in open(LOO):
            r = json.loads(line)
            g = parse_gold(r["output"])
            if g:
                by_dom.setdefault(g[0], []).append((r, g))
        rec_out = {}
        for dom, items in sorted(by_dom.items()):
            rng.shuffle(items)
            items = items[: args.n_rec]
            # --- 通路1: nothink 直通 beam32,prompt 尾接域前缀 ---
            direct_prompts = [build_prompt(r["instruction"], r["input"], "nothink") + DOM_TOKEN[dom] for r, _ in items]
            direct = beam_decode(direct_prompts, 32)
            # --- 通路2: thinking 采样 → 接 beam32 ---
            think_prompts = [build_prompt(r["instruction"], r["input"], "think") for r, _ in items]
            sp = SamplingParams(n=1, max_tokens=1024, temperature=0.6, top_p=0.95, top_k=20, seed=42,
                                stop=["</think>"])
            thinks = llm.generate(think_prompts, sp)
            stage2_prompts = [tp + t.outputs[0].text + "</think>\n" + DOM_TOKEN[dom]
                              for tp, t in zip(think_prompts, thinks)]
            staged = beam_decode(stage2_prompts, 32)
            # --- 统计 ---
            hit64 = hit_d = hit_t = 0
            copy_d = new_d = tot_d = 0
            for (r, g), dc, tc in zip(items, direct, staged):
                gold = g[1:]
                dset = set(c for c in dc if c)
                tset = set(c for c in tc if c)
                hit_d += gold in dset
                hit_t += gold in tset
                hit64 += gold in (dset | tset)
                hist = set(m[1:] for m in SID.findall(r["input"]) if m[0] == dom)
                for c in dset:
                    tot_d += 1
                    if c in hist: copy_d += 1
                    else: new_d += 1
            n = len(items)
            rec_out[dom] = {
                "n": n, "pass@64": round(hit64 / n, 4),
                "pass@32_direct": round(hit_d / n, 4), "pass@32_think": round(hit_t / n, 4),
                "direct_copy_rate": round(copy_d / max(tot_d, 1), 4),
                "direct_new_cands_per_q": round(new_d / n, 2),
            }
            print(f"[rec/{dom}] {rec_out[dom]}", file=sys.stderr)
        report["rec"] = rec_out

    # ============ mat 维 ============
    if "mat" in args.dims:
        mats = []
        for line in open(SEED):
            r = json.loads(line)
            ins = r.get("instruction", "")
            if "token" in ins and ("生成助手" in ins or "输出匹配" in ins or "输出对应" in ins):
                g = parse_gold(r["output"])
                if g: mats.append((r, g))
        rng.shuffle(mats)
        mats = mats[: args.n_mat]
        prompts = [build_prompt(r["instruction"], r["input"], "nothink") + DOM_TOKEN[g[0]] for r, g in mats]
        cands = beam_decode(prompts, 64)
        hit = 0
        sa_hit = 0
        for (r, g), cc in zip(mats, cands):
            gold = g[1:]
            cset = set(c for c in cc if c)
            hit += gold in cset
            sa_hit += gold[0] in set(c[0] for c in cset)
        n = len(mats)
        report["mat"] = {"n": n, "pass@64": round(hit / n, 4), "sa_pass@64": round(sa_hit / n, 4)}
        print(f"[mat] {report['mat']}", file=sys.stderr)

    report["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(f"{PROJ}/logs/probe", exist_ok=True)
    out_path = f"{PROJ}/logs/probe/{tag}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"saved -> {out_path}", file=sys.stderr)

    if args.wandb:
        import wandb
        run = wandb.init(project="llmrec-2026", name=f"probe_{tag}", job_type="probe", reinit=True)
        flat = {}
        for dom, d in report.get("rec", {}).items():
            for k, v in d.items():
                if k != "n": flat[f"probe/rec_{dom}/{k}"] = v
        for k, v in report.get("mat", {}).items():
            if k != "n": flat[f"probe/mat/{k}"] = v
        wandb.log(flat)
        run.finish()


if __name__ == "__main__":
    main()
