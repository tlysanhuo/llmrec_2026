#!/usr/bin/env python3
"""build_fk_fuse.py — 构建 fk_fuse_lora_ep1 训练数据(2026-07-04,用户批准的融合实验)。

融合 = Frinkleko 0.9107 公开数据(HF Frinkleko/kuaishou-llmrec-sft-baseline-0.91,
纯官方种子重组: think去重转nothink直出 + 1573 CEval选择题 + 5条评测日志真题[用户裁定:
官方允许用评测日志,合规]) × 我方已验证增益(world_zh 通识 +0.0115 线上验证;
fresh_mat 6000 条训练集外新物料样本,攻"物料=多样曝光量"阶梯)。

输入(全部既有产物,各自可复现):
  1. data/processed/frinkleko_alpaca_32705.jsonl — 官方 convert_jsonl.py 转换自
     assets/third_party/frinkleko_sft_091/train.jsonl(不 shuffle,保持其 seed=42 原序)
  2. data/processed/world_zh.jsonl      — 16237 条中文通识池,采样 N_WORLD 条(seed 固定)
  3. data/processed/fresh_mat.jsonl     — 6000 条全部并入;修正:input 补 '/no_think' 尾缀
     (其 output 为空think直出,按全库不变量 /no_think⇔空think;原文件漏了尾缀)

输出: data/processed/fk_fuse.jsonl (预期 32705+2824+6000=41529 条, seed=2026 shuffle)
用法: python scripts/data/build_fk_fuse.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
SEED = 2026
N_WORLD = 2824  # 与 rebal_world_ep3 同量级(9.7%档,线上验证 +0.0115)

def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    fk = load(P / "frinkleko_alpaca_32705.jsonl")
    world = load(P / "world_zh.jsonl")
    fresh = load(P / "fresh_mat.jsonl")
    assert len(fk) == 32705, len(fk)
    assert len(world) == 16237, len(world)
    assert len(fresh) == 6000, len(fresh)

    rng = random.Random(SEED)
    world_pick = rng.sample(world, N_WORLD)

    fixed = 0
    for r in fresh:
        if not r["input"].rstrip().endswith(("/think", "/no_think")):
            r["input"] = r["input"] + "/no_think"
            fixed += 1

    out = fk + world_pick + fresh
    rng.shuffle(out)

    dst = P / "fk_fuse.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {dst}: {len(out)} rows = fk {len(fk)} + world {len(world_pick)} + fresh {len(fresh)}")
    print(f"[OK] fresh_mat 补 /no_think 尾缀: {fixed}/6000")

if __name__ == "__main__":
    main()
