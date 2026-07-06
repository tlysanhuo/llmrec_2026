#!/usr/bin/env python3
"""train_itemic_weighted.py — LLaMA-Factory 启动器:itemic token 损失加权 SFT。

背景:榜一指路"模型对 itemic 不友好,加强训练";官方平台文档明示"自定义 loss
(商品 token 加权)"。本启动器不改 LF 源码,在进程内给 CustomSeq2SeqTrainer 注入
compute_loss_func:对 labels 落在 itemic token 区间的位置乘以权重 W_ITEMIC。

itemic token id 布局(OneReason-0.8B, added_tokens.json 已核实,三段连续):
  s_a: 151669-159860   s_b: 159861-168052   s_c: 168053-176244
  domain begin/end: 176245-176252
  → itemic 全区 = [151669, 176252](含 think 标记 151667/151668 之前的区间不含)

用法(和 llamafactory-cli train 相同,yaml 照常):
  ITEMIC_W=5.0 python scripts/train/train_itemic_weighted.py configs/recipe2_itemic_w5.yaml
环境变量:
  ITEMIC_W      itemic 区 loss 权重(默认 5.0)
  ITEMIC_LO/HI  区间覆盖(默认 151669/176252)

实现要点:
- 与 HF Trainer 的 compute_loss_func 协议一致:f(outputs, labels, num_items_in_batch)。
- 加权平均分母用 sum(weights)(非加权 token 数),保证 loss 量纲稳定、
  等价于"itemic token 出现 W 次"的过采样语义。
- packing/neat_packing 兼容:mask 只看 labels,与 attention 无关。
- ★CE 分块 + gradient checkpoint:packing 序列 32768 × vocab 176253 的 fp32 CE
  中间量(log_softmax 等)若全保留 ≈ 46GB;checkpoint 后反向逐块重算,峰值 ~几GB。
- ★配套 yaml 必须 enable_liger_kernel: false —— Liger 的 fused-linear-CE 不物化
  logits(outputs.logits 为 None),与自定义 loss 冲突。
"""
import os
import sys

import torch
import torch.utils.checkpoint as ckpt

ITEMIC_LO = int(os.environ.get("ITEMIC_LO", 151669))
ITEMIC_HI = int(os.environ.get("ITEMIC_HI", 176252))
ITEMIC_W = float(os.environ.get("ITEMIC_W", 5.0))
CE_CHUNK = int(os.environ.get("CE_CHUNK", 4096))


def _chunk_weighted_ce(logits_chunk: "torch.Tensor", labels_chunk: "torch.Tensor"):
    """returns (sum(per_tok*w), sum(w)) for one chunk. fp32 CE, recomputed in backward."""
    per_tok = torch.nn.functional.cross_entropy(
        logits_chunk.float(), labels_chunk, ignore_index=-100, reduction="none"
    )
    w = torch.ones_like(per_tok)
    itemic = (labels_chunk >= ITEMIC_LO) & (labels_chunk <= ITEMIC_HI)
    w = torch.where(itemic, torch.full_like(w, ITEMIC_W), w)
    w = w * (labels_chunk != -100).float()
    return (per_tok * w).sum(), w.sum()


def itemic_weighted_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits
    if logits is None:
        raise RuntimeError(
            "outputs.logits is None — 关掉 enable_liger_kernel(fused CE 不物化 logits)"
        )

    vocab_size = logits.size(-1)
    # shift: predict token t from prefix < t (与 dft_loss_func 相同约定)
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)
    flat_logits = logits.view(-1, vocab_size)

    if not (shift_labels != -100).any():
        return flat_logits.sum() * 0.0

    num = flat_logits.new_zeros(())
    den = flat_logits.new_zeros(())
    for s in range(0, flat_logits.size(0), CE_CHUNK):
        e = s + CE_CHUNK
        n, d = ckpt.checkpoint(
            _chunk_weighted_ce, flat_logits[s:e], shift_labels[s:e], use_reentrant=False
        )
        num = num + n
        den = den + d

    return num / den.clamp_min(1.0)


def main():
    # 注入:LF sft workflow 实例化 CustomSeq2SeqTrainer 后,super().__init__ 已把
    # self.compute_loss_func 置 None(除非 dft/eaft)。包一层 __init__ 后再覆盖。
    from llamafactory.train.sft import trainer as sft_trainer

    orig_init = sft_trainer.CustomSeq2SeqTrainer.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.compute_loss_func = itemic_weighted_loss
        print(
            f"[itemic-weighted] compute_loss_func injected: "
            f"W={ITEMIC_W} range=[{ITEMIC_LO},{ITEMIC_HI}]",
            flush=True,
        )

    sft_trainer.CustomSeq2SeqTrainer.__init__ = patched_init

    from llamafactory.train.tuner import run_exp

    # llamafactory-cli train <yaml> 等价:run_exp 直接读 sys.argv[1:]
    run_exp()


if __name__ == "__main__":
    main()
