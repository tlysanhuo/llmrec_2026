#!/usr/bin/env python3
"""config_diff.py — 训练前强制预检:新配置 vs 已证锚配置逐字段对账。

事故背景(2026-07-05):从 focal 模板复制配置时无意识继承了 enable_liger_kernel=false
(focal 为物化 logits 特意关闭),导致 07-04 后五个实验全部在数值受损的训练器上跑,
烧掉两发配额(fk_lora_embed 0.8672 / pstack_v2 0.8265),多项结论被混杂污染。

规则(铁律):任何新训练配置在启动前必须跑本脚本对账锚配置;
每个差异字段必须能在配置头注释里找到明确理由,找不到=不许启动。

用法: python scripts/train/config_diff.py configs/<新配置>.yaml [configs/history/rebal_world_ep3.yaml]
退出码: 0=关键字段全部一致或差异已知;1=存在关键字段差异(逐条打印,人工确认后方可启动)
"""
import sys
import yaml

ANCHOR = "configs/history/rebal_world_ep3.yaml"   # 当前线上最高分(0.9009)的完整配方
# 关键字段:改了就可能改变训练物理,必须显式论证
CRITICAL = [
    "enable_liger_kernel", "bf16", "pure_bf16", "flash_attn",
    "template", "cutoff_len", "packing", "neat_packing",
    "dataset", "dataset_dir", "val_size",
    "learning_rate", "num_train_epochs", "lr_scheduler_type", "warmup_ratio",
    "weight_decay", "per_device_train_batch_size", "gradient_accumulation_steps",
    "finetuning_type", "lora_rank", "lora_alpha", "lora_dropout", "lora_target",
    "max_grad_norm", "seed", "model_name_or_path", "stage",
]

def load(p):
    with open(p) as f:
        return yaml.safe_load(f)

def main():
    new_p = sys.argv[1]
    anchor_p = sys.argv[2] if len(sys.argv) > 2 else ANCHOR
    new, anchor = load(new_p), load(anchor_p)
    diffs = []
    for k in CRITICAL:
        a, n = anchor.get(k, "<未设(默认)>"), new.get(k, "<未设(默认)>")
        if a != n:
            diffs.append((k, a, n))
    print(f"锚: {anchor_p}\n新: {new_p}")
    if not diffs:
        print("✓ 关键字段与锚配置完全一致")
        sys.exit(0)
    print(f"⚠️ {len(diffs)} 个关键字段差异——每一条都必须有配置头注释里的明确理由:")
    for k, a, n in diffs:
        print(f"  {k}: 锚={a}  →  新={n}")
    sys.exit(1)

if __name__ == "__main__":
    main()
