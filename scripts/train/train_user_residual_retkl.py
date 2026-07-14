#!/usr/bin/env python3
"""Train a small user residual while distilling non-user behavior from E3.

The configured parent adapter is merged into the frozen base by LLaMA-Factory
(`create_new_adapter: true`). The newly initialized adapter is the only
trainable component. During each loss call, temporarily disabling that adapter
produces the exact parent behavior without loading a second model.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
CLOSE_THINK_ID = 151668
EOS_ID = 151645
ACTION_START_IDS = {58, 1183}  # `[` and `["`
TOPIC_START_IDS = {90, 4913}  # `{` and `{"`
WHITESPACE_IDS = {198, 220, 262, 271}

USER_PARENT_KL = float(os.environ.get("USERRES_USER_KL", "0.05"))
RETENTION_KL_WEIGHT = float(os.environ.get("USERRES_RETENTION_KL", "2.0"))
TERMINAL_MULTIPLIER = float(os.environ.get("USERRES_TERMINAL_MULTIPLIER", "2.0"))
LOGIT_CHUNK = int(os.environ.get("USERRES_LOGIT_CHUNK", "16"))


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("user-residual trainer requires per_device_train_batch_size=1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("batch has no supervised response tokens")
    start = int(positions[0])
    end = int(positions[-1]) + 1
    expected = torch.arange(start, end, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("packing must be disabled: found disjoint target spans")
    if start == 0:
        raise RuntimeError("response starts at token zero; no causal prediction position")
    return start, end


def body_start(tokens: list[int]) -> int:
    try:
        index = tokens.index(CLOSE_THINK_ID) + 1
    except ValueError as error:
        raise RuntimeError("assistant response is missing </think>") from error
    while index < len(tokens) and tokens[index] in WHITESPACE_IDS:
        index += 1
    if index >= len(tokens):
        raise RuntimeError("assistant response has no body after </think>")
    return index


def task_and_weights(targets: torch.Tensor) -> tuple[str, torch.Tensor]:
    tokens = targets.detach().cpu().tolist()
    weights = torch.ones_like(targets, dtype=torch.float32)
    if CLOSE_THINK_ID not in tokens:
        # Some registered O2.General retention answers have no think wrapper.
        # User rows are builder-validated to contain </think>, so this route
        # cannot silently turn truncated user supervision into retention.
        return "retention", weights
    start = body_start(tokens)
    first = tokens[start]
    if first in ACTION_START_IDS:
        task = "action"
    elif first in TOPIC_START_IDS:
        task = "topic"
    else:
        return "retention", weights

    # LLaMA-Factory's ChatML formatter supervises a newline after <|im_end|>.
    # Walk backward through that formatting suffix so the extra weight lands on
    # the actual JSON/list terminator and EOS, never on trailing whitespace.
    content_end = len(tokens)
    while content_end > start and tokens[content_end - 1] in WHITESPACE_IDS:
        content_end -= 1
    if content_end > start and tokens[content_end - 1] == EOS_ID:
        weights[content_end - 1] = TERMINAL_MULTIPLIER
        content_end -= 1
        while content_end > start and tokens[content_end - 1] in WHITESPACE_IDS:
            content_end -= 1
    if content_end <= start:
        raise RuntimeError(f"{task} response has no content")
    weights[content_end - 1] = TERMINAL_MULTIPLIER
    return task, weights


def weighted_ce(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    denominator = weights.sum().to(logits.device)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        losses = F.cross_entropy(
            logits[0, start:end].float(),
            targets[start:end].to(logits.device),
            reduction="none",
        )
        total = total + (losses * weights[start:end].to(logits.device)).sum()
    return total / denominator


def forward_kl(policy_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != ref_logits.shape:
        raise RuntimeError(
            f"policy/reference logit shape mismatch: {policy_logits.shape}/{ref_logits.shape}"
        )
    token_count = policy_logits.size(0) * policy_logits.size(1)
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, policy_logits.size(1))
        policy = policy_logits[:, start:end].float()
        reference = ref_logits[:, start:end].float()
        total = total + F.kl_div(
            F.log_softmax(policy, dim=-1),
            F.softmax(reference, dim=-1),
            reduction="sum",
        )
    return total / token_count


def residual_retention_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    start, end = target_span(labels)
    targets = labels[0, start:end]
    task, token_weights = task_and_weights(targets)
    prediction_positions = torch.arange(
        start - 1, end - 1, device=inputs["input_ids"].device, dtype=torch.long
    )

    unwrapped = self.accelerator.unwrap_model(model)
    disable_adapter = getattr(unwrapped, "disable_adapter", None)
    if disable_adapter is None:
        raise RuntimeError("expected a PEFT model with disable_adapter()")

    with torch.no_grad(), disable_adapter():
        ref_outputs = model(**inputs, logits_to_keep=prediction_positions)
        ref_logits = ref_outputs.logits.detach()

    outputs = model(**inputs, logits_to_keep=prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != targets.numel():
        raise RuntimeError(
            f"partial logits/targets mismatch: {policy_logits.size(1)}/{targets.numel()}"
        )

    kl = forward_kl(policy_logits, ref_logits)
    if task == "retention":
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl
    else:
        ce = weighted_ce(policy_logits, targets, token_weights)
        loss = ce + USER_PARENT_KL * kl

    call_count = getattr(residual_retention_loss, "call_count", 0) + 1
    residual_retention_loss.call_count = call_count
    counts = getattr(residual_retention_loss, "task_counts", Counter())
    counts[task] += 1
    residual_retention_loss.task_counts = counts
    if call_count <= 8 or call_count % 200 == 0:
        print(
            "[user-residual] "
            f"microbatch={call_count} task={task} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.6f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    torch.manual_seed(17)
    vocab = 31
    policy = torch.randn(1, 9, vocab, requires_grad=True)
    reference = torch.randn(1, 9, vocab)
    targets = torch.randint(0, vocab, (9,))
    weights = torch.linspace(1.0, 2.0, 9)

    direct_ce = (
        F.cross_entropy(policy[0], targets, reduction="none") * weights
    ).sum() / weights.sum()
    chunked_ce = weighted_ce(policy, targets, weights)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)

    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / 9
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)

    labels = torch.tensor(
        [
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                151667,
                CLOSE_THINK_ID,
                198,
                58,
                7,
                60,
                EOS_ID,
                198,
            ]
        ],
        dtype=torch.long,
    )
    assert target_span(labels) == (2, 10)
    task, terminal_weights = task_and_weights(labels[0, 2:10])
    assert task == "action"
    assert terminal_weights[-3].item() == TERMINAL_MULTIPLIER  # closing token
    assert terminal_weights[-2].item() == TERMINAL_MULTIPLIER  # <|im_end|>
    assert terminal_weights[-1].item() == 1.0  # formatter newline
    retention_task, _ = task_and_weights(torch.tensor([7, 8, EOS_ID]))
    assert retention_task == "retention"
    (chunked_ce + chunked_kl).backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    print(
        "[user-residual] self-test passed: partial-target indexing, chunked CE/KL, "
        "task routing, terminal weights, and gradients are consistent"
    )


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        return residual_retention_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
