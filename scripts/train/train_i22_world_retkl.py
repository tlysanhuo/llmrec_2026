#!/usr/bin/env python3
"""I-22 trainer: world answer-token CE inside a frozen-I13 trust region."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train.train_i20_positive_retkl import (
    IGNORE_INDEX,
    _single_adapter_name,
    body_start,
    capped_positions,
    forward_kl,
    target_span,
)
from scripts.train.train_user_residual_retkl import weighted_ce


REFERENCE_ADAPTER = os.environ.get(
    "I22_REFERENCE_ADAPTER",
    str(ROOT / "submissions/e3_userres_r80_retkl_v3_s875_platform"),
)
REFERENCE_NAME = "i22_frozen_reference"
WORLD_KL_WEIGHT = float(os.environ.get("I22_WORLD_KL", "2.0"))
RETENTION_KL_WEIGHT = float(os.environ.get("I22_RETENTION_KL", "50.0"))
RETENTION_MAX_TOKENS = int(os.environ.get("I22_RETENTION_MAX_TOKENS", "128"))

# O6 tokenizer encodes `正确答案是 (` as these four tokens.  The builder makes
# every CE row use this exact response prefix; all other rows are KL-only.
WORLD_PREFIX_IDS = (88991, 102349, 20412, 320)
CLOSE_PAREN_ID = 8
THINK_END_ID = 151668
WHITESPACE_IDS = {198, 271, 220}


def content_end(tokens: list[int], start: int) -> int:
    end = len(tokens)
    while end > start and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    return end


def route_and_positions(labels: torch.Tensor) -> tuple[str, torch.Tensor, torch.Tensor]:
    start, end = target_span(labels)
    targets = labels[0, start:end]
    tokens = targets.detach().cpu().tolist()
    body = body_start(tokens) if THINK_END_ID in tokens else 0

    if tuple(tokens[body : body + len(WORLD_PREFIX_IDS)]) == WORLD_PREFIX_IDS:
        label_start = body + len(WORLD_PREFIX_IDS)
        try:
            label_end = tokens.index(CLOSE_PAREN_ID, label_start)
        except ValueError as exc:
            raise RuntimeError("canonical world target has no closing parenthesis") from exc
        if label_end <= label_start:
            raise RuntimeError("canonical world target has an empty answer")
        relative = torch.arange(label_start, label_end, device=labels.device, dtype=torch.long)
        return "world", relative + start, torch.ones(relative.numel(), device=labels.device)

    stop = content_end(tokens, body)
    relative = capped_positions(body, stop, RETENTION_MAX_TOKENS, labels.device)
    return "retention", relative + start, torch.ones(relative.numel(), device=labels.device)


def ensure_reference(self, model) -> tuple[object, str]:
    state = getattr(self, "_i22_reference_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state["policy_name"]

    policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name == REFERENCE_NAME:
        raise RuntimeError("policy adapter name collides with I-22 reference name")
    if not Path(REFERENCE_ADAPTER).is_dir():
        raise RuntimeError(f"missing frozen I-13 reference adapter: {REFERENCE_ADAPTER}")

    initial_trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    device = next(unwrapped.parameters()).device
    unwrapped.load_adapter(
        REFERENCE_ADAPTER,
        adapter_name=REFERENCE_NAME,
        is_trainable=False,
        torch_device=str(device),
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=True,
    )
    unwrapped.set_adapter(policy_name)
    unwrapped.set_requires_grad(policy_name, True)
    unwrapped.set_requires_grad(REFERENCE_NAME, False)
    final_trainable = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
    if final_trainable != initial_trainable:
        raise RuntimeError(
            "trainable parameter count changed after reference load: "
            f"{initial_trainable}/{final_trainable}"
        )
    self._i22_reference_state = {"policy_name": policy_name, "fingerprint_checked": False}
    print(
        "[i22] frozen reference loaded: "
        f"path={REFERENCE_ADAPTER} policy={policy_name} reference={REFERENCE_NAME} "
        f"trainable={initial_trainable:,}",
        flush=True,
    )
    return unwrapped, policy_name


def set_active_adapter(unwrapped, name: str, trainable: bool) -> None:
    unwrapped.set_adapter(name)
    unwrapped.set_requires_grad(name, trainable)
    for other in unwrapped.peft_config:
        if other != name:
            unwrapped.set_requires_grad(other, False)


def paired_forward(self, model, inputs, prediction_positions):
    unwrapped, policy_name = ensure_reference(self, model)
    model.train()
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    set_active_adapter(unwrapped, REFERENCE_NAME, False)
    with torch.no_grad():
        reference = model(**inputs, logits_to_keep=prediction_positions).logits.detach()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    set_active_adapter(unwrapped, policy_name, True)
    outputs = model(**inputs, logits_to_keep=prediction_positions)

    state = self._i22_reference_state
    if not state["fingerprint_checked"]:
        max_abs = float((outputs.logits.detach().float() - reference.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"initial policy/reference logits differ: max_abs={max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i22] initial policy/reference fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return outputs, reference


def i22_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    route, positions, weights = route_and_positions(labels)
    prediction_positions = positions - 1
    targets = labels[0, positions]
    outputs, reference = paired_forward(self, model, inputs, prediction_positions)
    logits = outputs.logits
    if logits.size(1) != targets.numel():
        raise RuntimeError(f"partial logits/targets mismatch: {logits.size(1)}/{targets.numel()}")

    kl = forward_kl(logits, reference)
    if route == "world":
        ce = weighted_ce(logits, targets, weights)
        loss = ce + WORLD_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl

    count = getattr(i22_loss, "call_count", 0) + 1
    i22_loss.call_count = count
    counts = getattr(i22_loss, "route_counts", Counter())
    counts[route] += 1
    i22_loss.route_counts = counts
    if count <= 12 or count % 100 == 0:
        print(
            "[i22] "
            f"microbatch={count} route={route} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    world = torch.tensor(
        [prefix + [151667, 271, 151668, 198, *WORLD_PREFIX_IDS, 33, 34, 8, 151645, 198]],
        dtype=torch.long,
    )
    route, positions, weights = route_and_positions(world)
    assert route == "world"
    assert world[0, positions].tolist() == [33, 34]
    assert weights.tolist() == [1.0, 1.0]

    retention = torch.tensor(
        [prefix + [151667, 271, 151668, 198] + list(range(1000, 1200)) + [151645, 198]],
        dtype=torch.long,
    )
    route, positions, weights = route_and_positions(retention)
    assert route == "retention" and positions.numel() == RETENTION_MAX_TOKENS
    assert weights.numel() == RETENTION_MAX_TOKENS
    print("[i22] self-test passed: answer-letter CE and capped retention KL")


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from llamafactory.train.sft import trainer as sft_trainer

    original = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original(self, model, inputs, *args, **kwargs)
        return i22_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
