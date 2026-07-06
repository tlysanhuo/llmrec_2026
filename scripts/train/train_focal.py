#!/usr/bin/env python3
"""train_focal.py — LLaMA-Factory 启动器:focal loss SFT(难样本聚焦)。

背景(2026-07-04):选手对账发现平台训练服务 = focal loss + token 加权,本地 LF = 普通 CE;
平台文档明写"自定义 loss:商品 token 加权、难样本聚焦"。focal 是跷跷板的对症机制:
自动给已学好的头部 token(video 高频模式)降权、给难 token(ad 新预测/物料长尾 s_a)提权
——我们用配比手动模拟的正是它。与 recipe2 全区均匀加权(0.7692 翻车)机制不同:
focal 按 per-token 难度自适应,不是静态区间乘数。

focal loss: FL(p_t) = -(1-p_t)^γ · log(p_t)
  γ=0 退化为 CE;γ=2 是经典值。p_t 高(学得好)→ 权重≈0;p_t 低(难)→ 权重≈1。

复用 train_itemic_weighted.py 的注入框架(compute_loss_func 协议/CE分块checkpoint/
Liger 必须关)。可选叠加 itemic 温和加权(FOCAL_ITEMIC_W,默认 1.0=不加)。

用法:
  FOCAL_GAMMA=2.0 python scripts/train/train_focal.py configs/<name>.yaml
环境变量:
  FOCAL_GAMMA     焦点参数(默认 2.0)
  FOCAL_ITEMIC_W  itemic 区额外权重(默认 1.0 不加;若加建议 ≤2,recipe2 教训)
  CE_CHUNK        CE 分块大小(默认 4096)

配套 yaml 硬要求:enable_liger_kernel: false(fused CE 不物化 logits)。
"""
import os

import torch
import torch.utils.checkpoint as ckpt

FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", 2.0))
ITEMIC_W = float(os.environ.get("FOCAL_ITEMIC_W", 1.0))
ITEMIC_LO, ITEMIC_HI = 151669, 176252
CE_CHUNK = int(os.environ.get("CE_CHUNK", 4096))


def _chunk_focal(logits_chunk: "torch.Tensor", labels_chunk: "torch.Tensor"):
    """returns (sum(focal_tok*w), sum(valid)) for one chunk. fp32, recomputed in backward."""
    ce = torch.nn.functional.cross_entropy(
        logits_chunk.float(), labels_chunk, ignore_index=-100, reduction="none"
    )
    p_t = torch.exp(-ce)  # p_t = softmax prob of gold token
    focal = (1.0 - p_t).pow(FOCAL_GAMMA) * ce
    w = torch.ones_like(focal)
    if ITEMIC_W != 1.0:
        itemic = (labels_chunk >= ITEMIC_LO) & (labels_chunk <= ITEMIC_HI)
        w = torch.where(itemic, torch.full_like(w, ITEMIC_W), w)
    valid = (labels_chunk != -100).float()
    w = w * valid
    return (focal * w).sum(), valid.sum()


def focal_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits
    if logits is None:
        raise RuntimeError("outputs.logits is None — 关掉 enable_liger_kernel")

    vocab_size = logits.size(-1)
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
            _chunk_focal, flat_logits[s:e], shift_labels[s:e], use_reentrant=False
        )
        num = num + n
        den = den + d
    # ★HF Trainer 约定(dft_loss_func 同款,γ=0 冒烟 10.54≈CE×accum4 抓出的bug):
    # 传了 num_items_in_batch(整个梯度累积批的有效token数)时必须用它作分母,
    # Trainer 不再除 accum steps;自归一(den)会把 loss 放大 accum 倍=偷偷 lr×4。
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(num.device)
        return num / num_items_in_batch
    return num / den.clamp_min(1.0)


def main():
    from llamafactory.train.sft import trainer as sft_trainer

    orig_init = sft_trainer.CustomSeq2SeqTrainer.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.compute_loss_func = focal_loss
        print(
            f"[focal] compute_loss_func injected: gamma={FOCAL_GAMMA} itemic_w={ITEMIC_W}",
            flush=True,
        )

    sft_trainer.CustomSeq2SeqTrainer.__init__ = patched_init

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
