#!/usr/bin/env python3
"""build_nothink.py — 把种子数据的 CoT 剥掉,产出 no-think 版训练集。

依据(2026-07-02 判断):
- 种子数据 73.3% itemic 信号在 CoT 里(引用历史),答案段只占 12.6% token;
  剥 CoT = 损失 100% 集中在答案 → "预测新item"信号等效放大 ~8x。
- 心定 LoRA+关thinking=0.85+;克西(榜一0.9351)"加强itemic训练"最可能即此。
- LF `enable_thinking:false` 的正则 (<think>\n...\n</think>\n\n) 在本数据命中 0/32480
  (数据格式是 <think>内容</think>\n),必须在数据层剥。
- 空 think 统一为数据中官方 no_think 样本的既有格式: "<think>\n</think>\n"。

确定性变换,无随机。用法:
  python scripts/data/build_nothink.py \
    --src /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/data_final.jsonl \
    --dst /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/data_nothink.jsonl
"""
import argparse
import json
import re

THINK = re.compile(r"^<think>.*?</think>\n?", re.DOTALL)
EMPTY = "<think>\n</think>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    n = stripped = already = odd = 0
    with open(args.src) as f, open(args.dst, "w") as g:
        for line in f:
            r = json.loads(line)
            o = r["output"]
            m = THINK.match(o)
            if m:
                body = o[m.end():]
                if m.group(0) == EMPTY or m.group(0) == "<think>\n</think>":
                    already += 1
                else:
                    stripped += 1
                r["output"] = EMPTY + body.lstrip("\n")
            else:
                odd += 1  # 无 think 开头,原样保留
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"total={n} stripped={stripped} already_empty={already} no_think_tag={odd}")


if __name__ == "__main__":
    main()
