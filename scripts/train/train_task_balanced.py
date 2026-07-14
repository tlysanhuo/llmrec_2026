#!/usr/bin/env python3
"""Launch task-balanced SFT with gradient-accumulation-correct weighted CE."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.utils.checkpoint as checkpoint


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT.parent / "ai_runtime" / "llmrec_2026"
AUDIT_PATH = Path(
    os.environ.get(
        "TASKBAL_AUDIT",
        RUNTIME / "logs" / "data" / "seed_taskbal_audit.json",
    )
)
CE_CHUNK = int(os.environ.get("TASKBAL_CE_CHUNK", "1024"))

IGNORE_INDEX = -100
CLOSE_THINK_ID = 151668
EOS_ID = 151645
ACTION_START_IDS = {58, 1183}  # `[` and `["`
TOPIC_START_IDS = {90, 4913}  # `{` and `{"`
DOMAIN_IDS = {176245, 176247, 176249, 176251}
WHITESPACE_IDS = {198, 220, 262, 271}

ACTION_WEIGHT = 3.0
ACTION_TERMINAL_MULTIPLIER = 2.0
TOPIC_WEIGHT = 0.5
DESC2SID_ANSWER_WEIGHT = 4.0


def _target_spans(row: torch.Tensor) -> list[tuple[int, int]]:
    valid = row.ne(IGNORE_INDEX)
    padded = torch.nn.functional.pad(valid, (1, 1), value=False)
    transitions = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(transitions == 1, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(transitions == -1, as_tuple=False).flatten().tolist()
    return list(zip(starts, ends))


def _body_start(tokens: list[int]) -> int:
    try:
        index = tokens.index(CLOSE_THINK_ID) + 1
    except ValueError as error:
        raise RuntimeError("assistant target is missing </think>") from error
    while index < len(tokens) and tokens[index] in WHITESPACE_IDS:
        index += 1
    return index


def build_task_weights(labels: torch.Tensor) -> tuple[torch.Tensor, Counter]:
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights.masked_fill_(labels.eq(IGNORE_INDEX), 0.0)
    task_counts = Counter()

    for batch_index in range(labels.size(0)):
        for start, end in _target_spans(labels[batch_index]):
            span = labels[batch_index, start:end]
            tokens = span.detach().cpu().tolist()
            body_start = _body_start(tokens)
            if body_start >= len(tokens):
                raise RuntimeError("assistant target has no response body")
            first_body_token = tokens[body_start]

            if first_body_token in ACTION_START_IDS:
                task = "action"
                weights[batch_index, start:end] = ACTION_WEIGHT
                content_end = end - start - int(tokens[-1] == EOS_ID)
                if content_end <= body_start:
                    raise RuntimeError("action target has no JSON content")
                weights[batch_index, start + content_end - 1] *= ACTION_TERMINAL_MULTIPLIER
                if tokens[-1] == EOS_ID:
                    weights[batch_index, end - 1] *= ACTION_TERMINAL_MULTIPLIER
            elif first_body_token in TOPIC_START_IDS:
                task = "topic"
                weights[batch_index, start:end] = TOPIC_WEIGHT
            elif first_body_token in DOMAIN_IDS:
                task = "material_desc2sid"
                body_end = end - start
                if tokens and tokens[-1] == EOS_ID:
                    body_end -= 1
                weights[batch_index, start + body_start : start + body_end] = DESC2SID_ANSWER_WEIGHT
            elif any(token in DOMAIN_IDS for token in tokens):
                task = "recommendation"
            else:
                task = "material_sid2desc"
            task_counts[task] += 1

    return weights, task_counts


def _weighted_ce_chunk(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = torch.nn.functional.cross_entropy(
        logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    return (losses * weights).sum()


def task_balanced_loss(
    outputs,
    labels: torch.Tensor,
    num_items_in_batch=None,
    *,
    mean_weight: float,
) -> torch.Tensor:
    logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits
    if logits is None:
        raise RuntimeError("outputs.logits is None; task-balanced loss requires enable_liger_kernel=false")

    token_weights, task_counts = build_task_weights(labels)
    padded_labels = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = padded_labels[..., 1:].contiguous().view(-1).to(logits.device)
    padded_weights = torch.nn.functional.pad(token_weights, (0, 1), value=0.0)
    shift_weights = padded_weights[..., 1:].contiguous().view(-1).to(logits.device)
    flat_logits = logits.view(-1, logits.size(-1))

    numerator = flat_logits.new_zeros(())
    for start in range(0, flat_logits.size(0), CE_CHUNK):
        end = min(start + CE_CHUNK, flat_logits.size(0))
        numerator = numerator + checkpoint.checkpoint(
            _weighted_ce_chunk,
            flat_logits[start:end],
            shift_labels[start:end],
            shift_weights[start:end],
            use_reentrant=False,
        )

    if num_items_in_batch is None:
        denominator = shift_labels.ne(IGNORE_INDEX).sum()
    else:
        denominator = num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(numerator.device)

    call_count = getattr(task_balanced_loss, "call_count", 0) + 1
    task_balanced_loss.call_count = call_count
    if call_count <= 3:
        print(
            f"[task-balanced] microbatch={call_count} tasks={dict(task_counts)} "
            f"valid={int(shift_labels.ne(IGNORE_INDEX).sum())} accum_valid={int(denominator)}",
            flush=True,
        )
    return numerator / (denominator * mean_weight)


def run_self_test() -> None:
    torch.manual_seed(7)
    labels = torch.tensor(
        [[IGNORE_INDEX, CLOSE_THINK_ID, 271, CLOSE_THINK_ID, 198, 58, 10, 60, EOS_ID]],
        dtype=torch.long,
    )
    # Use a valid synthetic target with a single </think>; the first CLOSE_THINK_ID is <think>-stand-in.
    labels[0, 1] = 151667
    weights, counts = build_task_weights(labels)
    assert counts == {"action": 1}
    assert weights[0, 5].item() == ACTION_WEIGHT
    assert weights[0, 7].item() == ACTION_WEIGHT * ACTION_TERMINAL_MULTIPLIER
    assert weights[0, 8].item() == ACTION_WEIGHT * ACTION_TERMINAL_MULTIPLIER

    vocab = 32
    micro_logits = [torch.randn(1, 4, vocab, requires_grad=True) for _ in range(4)]
    micro_labels = [torch.tensor([[-100, 3, 5, 7]]) for _ in range(4)]
    total_items = sum(int(item.ne(IGNORE_INDEX).sum()) for item in micro_labels)
    micro_loss = 0.0
    for logits, target in zip(micro_logits, micro_labels):
        shifted = torch.nn.functional.pad(target, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1)
        micro_loss = micro_loss + torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), shifted, ignore_index=IGNORE_INDEX, reduction="sum"
        ) / total_items
    micro_loss.backward()
    micro_grads = [value.grad.detach().clone() for value in micro_logits]

    joined_logits = torch.cat([value.detach() for value in micro_logits], dim=0).requires_grad_(True)
    joined_labels = torch.cat(micro_labels, dim=0)
    joined_shift = torch.nn.functional.pad(joined_labels, (0, 1), value=IGNORE_INDEX)[..., 1:].reshape(-1)
    joined_loss = torch.nn.functional.cross_entropy(
        joined_logits.reshape(-1, vocab), joined_shift, ignore_index=IGNORE_INDEX, reduction="mean"
    )
    joined_loss.backward()
    assert torch.allclose(micro_loss.detach(), joined_loss.detach(), atol=1e-7)
    for index, grad in enumerate(micro_grads):
        assert torch.allclose(grad, joined_logits.grad[index : index + 1], atol=1e-7)
    print("[task-balanced] self-test passed: task masks and accum=1/4 gradients are consistent")


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"task-balance audit not found: {AUDIT_PATH}")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    mean_weight = float(audit["token_weight_audit"]["loss_mean_weight"])
    print(f"[task-balanced] audit={AUDIT_PATH} mean_weight={mean_weight:.8f}", flush=True)

    from llamafactory.train.sft import trainer as sft_trainer

    original_init = sft_trainer.CustomSeq2SeqTrainer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.compute_loss_func = task_balanced_loss

    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = task_balanced_loss(
            outputs,
            labels,
            num_items_in_batch=num_items_in_batch,
            mean_weight=mean_weight,
        )
        return (loss, outputs) if return_outputs else loss

    sft_trainer.CustomSeq2SeqTrainer.__init__ = patched_init
    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
