#!/usr/bin/env python3
"""I-24 trainer: action-body CE inside a frozen-I23 trust region.

The trainable policy is the original I-23 r64 adapter, updated in place.  The
same adapter is loaded once more under a frozen PEFT name and is used as the
reference on every microbatch.  Reference and policy forwards replay matching
CPU/CUDA RNG states, so LoRA dropout does not create spurious KL drift.

Only action JSON/list bodies receive gold CE.  Topic and all non-user rows in
the registered 6,106-row I-12 mixture are KL-only.  KL is evaluated on the
sorted union of 96 uniformly sampled full-trace tokens, 48 uniformly sampled
answer-body tokens, and the last 16 content tokens (at most 160 unique tokens).
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
CLOSE_THINK_ID = 151668
EOS_ID = 151645
ACTION_START_IDS = {58, 1183}  # `[` and `["`
WHITESPACE_IDS = {198, 220, 262, 271}

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ADAPTER = os.environ.get(
    "I24_REFERENCE_ADAPTER",
    str(ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"),
)
REFERENCE_NAME = "i24_frozen_i23_reference"
ACTION_KL_WEIGHT = float(os.environ.get("I24_ACTION_KL", "5.0"))
RETENTION_KL_WEIGHT = float(os.environ.get("I24_RETENTION_KL", "50.0"))
TRACE_KL_TOKENS = int(os.environ.get("I24_TRACE_KL_TOKENS", "96"))
BODY_KL_TOKENS = int(os.environ.get("I24_BODY_KL_TOKENS", "48"))
TAIL_KL_TOKENS = int(os.environ.get("I24_TAIL_KL_TOKENS", "16"))
MAX_KL_TOKENS = int(os.environ.get("I24_MAX_KL_TOKENS", "160"))
TERMINAL_MULTIPLIER = float(os.environ.get("I24_TERMINAL_MULTIPLIER", "2.0"))
LOGIT_CHUNK = int(os.environ.get("I24_LOGIT_CHUNK", "8"))


def validate_hyperparameters() -> None:
    positive = {
        "ACTION_KL_WEIGHT": ACTION_KL_WEIGHT,
        "RETENTION_KL_WEIGHT": RETENTION_KL_WEIGHT,
        "TRACE_KL_TOKENS": TRACE_KL_TOKENS,
        "BODY_KL_TOKENS": BODY_KL_TOKENS,
        "TAIL_KL_TOKENS": TAIL_KL_TOKENS,
        "MAX_KL_TOKENS": MAX_KL_TOKENS,
        "TERMINAL_MULTIPLIER": TERMINAL_MULTIPLIER,
        "LOGIT_CHUNK": LOGIT_CHUNK,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise RuntimeError(f"I-24 hyperparameters must be positive: {invalid}")
    if TRACE_KL_TOKENS + BODY_KL_TOKENS + TAIL_KL_TOKENS > MAX_KL_TOKENS:
        raise RuntimeError(
            "I-24 KL component caps exceed MAX_KL_TOKENS: "
            f"{TRACE_KL_TOKENS}+{BODY_KL_TOKENS}+{TAIL_KL_TOKENS}>"
            f"{MAX_KL_TOKENS}"
        )


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I-24 requires per_device_train_batch_size=1")
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


def answer_body_start(tokens: list[int]) -> int:
    try:
        index = tokens.index(CLOSE_THINK_ID) + 1
    except ValueError as error:
        raise RuntimeError("action response is missing </think>") from error
    while index < len(tokens) and tokens[index] in WHITESPACE_IDS:
        index += 1
    if index >= len(tokens):
        raise RuntimeError("assistant response has no body after </think>")
    return index


def response_content_end(tokens: list[int], minimum: int = 0) -> int:
    """Return the exclusive end after removing formatter-only trailing space."""

    end = len(tokens)
    while end > minimum and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    if end <= minimum:
        raise RuntimeError("assistant response has no non-whitespace content")
    return end


def uniformly_capped_positions(
    start: int, end: int, cap: int, device: torch.device
) -> torch.Tensor:
    if not (0 <= start < end):
        raise RuntimeError(f"invalid token range for KL sampling: {start}:{end}")
    count = end - start
    if count <= cap:
        return torch.arange(start, end, device=device, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap, device=device).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("uniform KL cap produced duplicate token positions")
    return relative + start


def kl_relative_positions(
    tokens: list[int], body: int, content_end: int, device: torch.device
) -> torch.Tensor:
    trace = uniformly_capped_positions(0, content_end, TRACE_KL_TOKENS, device)
    body_sample = uniformly_capped_positions(body, content_end, BODY_KL_TOKENS, device)
    tail_start = max(0, content_end - TAIL_KL_TOKENS)
    tail = torch.arange(tail_start, content_end, device=device, dtype=torch.long)
    positions = torch.cat((trace, body_sample, tail)).unique(sorted=True)
    if positions.numel() > MAX_KL_TOKENS:
        raise RuntimeError(
            f"KL union exceeds cap: {positions.numel()} > {MAX_KL_TOKENS}"
        )
    if positions.numel() == 0 or int(positions[0]) < 0 or int(positions[-1]) >= content_end:
        raise RuntimeError("KL union contains an invalid token position")
    return positions


def action_terminal_weights(tokens: list[int], body: int, content_end: int) -> torch.Tensor:
    if tokens[content_end - 1] != EOS_ID:
        raise RuntimeError("action response does not end in the expected EOS token")
    weights = torch.ones(content_end - body, dtype=torch.float32)
    weights[-1] = TERMINAL_MULTIPLIER

    terminal = content_end - 1
    while terminal > body and tokens[terminal - 1] in WHITESPACE_IDS:
        terminal -= 1
    if terminal <= body:
        raise RuntimeError("action response has no JSON/list terminal before EOS")
    weights[terminal - 1 - body] = TERMINAL_MULTIPLIER
    return weights


def route_and_positions(
    labels: torch.Tensor,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return route, optional CE positions/weights, and KL positions."""

    start, end = target_span(labels)
    targets = labels[0, start:end]
    tokens = targets.detach().cpu().tolist()
    content_end = response_content_end(tokens)

    if CLOSE_THINK_ID in tokens:
        body = answer_body_start(tokens)
    else:
        body = 0

    is_action = body < content_end and tokens[body] in ACTION_START_IDS
    if is_action:
        ce_relative = torch.arange(body, content_end, device=labels.device, dtype=torch.long)
        ce_weights = action_terminal_weights(tokens, body, content_end).to(labels.device)
        route = "action"
    else:
        ce_relative = torch.empty(0, device=labels.device, dtype=torch.long)
        ce_weights = torch.empty(0, device=labels.device, dtype=torch.float32)
        route = "retention"

    kl_relative = kl_relative_positions(tokens, body, content_end, labels.device)
    return route, ce_relative + start, ce_weights, kl_relative + start


def weighted_ce(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(0) != 1:
        raise RuntimeError(f"unexpected CE logit shape: {tuple(logits.shape)}")
    if logits.size(1) != targets.numel() or targets.numel() != weights.numel():
        raise RuntimeError(
            "CE logits/targets/weights mismatch: "
            f"{logits.size(1)}/{targets.numel()}/{weights.numel()}"
        )
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


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(
            "policy/reference logit shape mismatch: "
            f"{tuple(policy_logits.shape)}/{tuple(reference_logits.shape)}"
        )
    token_count = policy_logits.size(0) * policy_logits.size(1)
    if token_count <= 0:
        raise RuntimeError("KL received no token logits")
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, policy_logits.size(1))
        policy = policy_logits[:, start:end].float()
        reference = reference_logits[:, start:end].float()
        total = total + F.kl_div(
            F.log_softmax(policy, dim=-1),
            F.softmax(reference, dim=-1),
            reduction="sum",
        )
    return total / token_count


def _single_adapter_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError(f"expected one active PEFT adapter, got {value!r}")


def ensure_reference(self, model) -> tuple[object, str]:
    state = getattr(self, "_i24_reference_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state["policy_name"]

    policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name == REFERENCE_NAME:
        raise RuntimeError("policy adapter name collides with I-24 reference name")
    if not Path(REFERENCE_ADAPTER).is_dir():
        raise RuntimeError(f"missing frozen I-23 reference adapter: {REFERENCE_ADAPTER}")

    initial_trainable = sum(param.numel() for param in unwrapped.parameters() if param.requires_grad)
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
    final_trainable = sum(param.numel() for param in unwrapped.parameters() if param.requires_grad)
    if final_trainable != initial_trainable:
        raise RuntimeError(
            "trainable parameter count changed after reference load: "
            f"{initial_trainable}/{final_trainable}"
        )
    self._i24_reference_state = {
        "policy_name": policy_name,
        "fingerprint_checked": False,
    }
    print(
        "[i24] frozen reference loaded: "
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


def paired_forward(self, model, inputs, prediction_positions: torch.Tensor):
    unwrapped, policy_name = ensure_reference(self, model)
    model.train()

    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    set_active_adapter(unwrapped, REFERENCE_NAME, False)
    with torch.no_grad():
        reference_logits = model(
            **inputs, logits_to_keep=prediction_positions
        ).logits.detach()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    set_active_adapter(unwrapped, policy_name, True)
    outputs = model(**inputs, logits_to_keep=prediction_positions)

    state = self._i24_reference_state
    if not state["fingerprint_checked"]:
        max_abs = float(
            (outputs.logits.detach().float() - reference_logits.float()).abs().max()
        )
        if max_abs > 1e-4:
            raise RuntimeError(f"initial policy/reference logits differ: max_abs={max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(
            f"[i24] initial policy/reference fingerprint PASS: max_abs={max_abs:.8f}",
            flush=True,
        )
    return outputs, reference_logits


def indices_in_union(union: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    if subset.numel() == 0:
        return torch.empty(0, device=union.device, dtype=torch.long)
    indices = torch.searchsorted(union, subset)
    if int(indices.max()) >= union.numel() or not torch.equal(union[indices], subset):
        raise RuntimeError("selected token positions are not contained in forward union")
    return indices


def i24_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    route, ce_positions, ce_weights, kl_positions = route_and_positions(labels)
    selected_positions = torch.cat((ce_positions, kl_positions)).unique(sorted=True)
    if selected_positions.numel() == 0:
        raise RuntimeError("I-24 selected no response tokens")
    prediction_positions = selected_positions - 1

    outputs, reference_logits = paired_forward(self, model, inputs, prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != selected_positions.numel():
        raise RuntimeError(
            "partial logits/position mismatch: "
            f"{policy_logits.size(1)}/{selected_positions.numel()}"
        )

    kl_indices = indices_in_union(selected_positions, kl_positions)
    kl = forward_kl(policy_logits[:, kl_indices], reference_logits[:, kl_indices])
    if route == "action":
        ce_indices = indices_in_union(selected_positions, ce_positions)
        ce_targets = labels[0, ce_positions]
        ce = weighted_ce(policy_logits[:, ce_indices], ce_targets, ce_weights)
        loss = ce + ACTION_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl

    count = getattr(i24_loss, "call_count", 0) + 1
    i24_loss.call_count = count
    counts = getattr(i24_loss, "route_counts", Counter())
    counts[route] += 1
    i24_loss.route_counts = counts
    if count <= 12 or count % 100 == 0:
        print(
            "[i24] "
            f"microbatch={count} route={route} ce_tokens={ce_positions.numel()} "
            f"kl_tokens={kl_positions.numel()} ce={float(ce.detach()):.6f} "
            f"kl={float(kl.detach()):.8f} loss={float(loss.detach()):.6f} "
            f"counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    validate_hyperparameters()
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    action_tokens = (
        [151667, CLOSE_THINK_ID, 198, 58]
        + list(range(1_000, 1_220))
        + [60, EOS_ID, 198]
    )
    action = torch.tensor([prefix + action_tokens], dtype=torch.long)
    route, ce_positions, ce_weights, kl_positions = route_and_positions(action)
    assert route == "action"
    assert action[0, ce_positions][0].item() == 58
    assert action[0, ce_positions][-1].item() == EOS_ID
    assert ce_weights[-2].item() == TERMINAL_MULTIPLIER
    assert ce_weights[-1].item() == TERMINAL_MULTIPLIER
    assert 0 < kl_positions.numel() <= MAX_KL_TOKENS
    target_start, _ = target_span(action)
    content_end = response_content_end(action_tokens)
    tail = torch.arange(
        target_start + content_end - TAIL_KL_TOKENS,
        target_start + content_end,
        dtype=torch.long,
    )
    assert torch.equal(kl_positions[torch.searchsorted(kl_positions, tail)], tail)

    topic = torch.tensor(
        [prefix + [151667, CLOSE_THINK_ID, 198, 90, 7, 92, EOS_ID, 198]],
        dtype=torch.long,
    )
    route, ce_positions, ce_weights, topic_kl = route_and_positions(topic)
    assert route == "retention" and ce_positions.numel() == 0 and ce_weights.numel() == 0
    assert topic_kl.numel() > 0

    no_think = torch.tensor([prefix + [7, 8, EOS_ID, 198]], dtype=torch.long)
    route, _, _, no_think_kl = route_and_positions(no_think)
    assert route == "retention" and no_think_kl.numel() == 3

    torch.manual_seed(17)
    policy = torch.randn(1, 17, 31, requires_grad=True)
    reference = torch.randn(1, 17, 31)
    targets = torch.randint(0, 31, (17,))
    weights = torch.linspace(1.0, 2.0, 17)
    direct_ce = (
        F.cross_entropy(policy[0].float(), targets, reduction="none") * weights
    ).sum() / weights.sum()
    chunked_ce = weighted_ce(policy, targets, weights)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)
    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / policy.size(1)
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)

    union = torch.tensor([2, 4, 6, 8, 10])
    subset = torch.tensor([4, 8, 10])
    assert indices_in_union(union, subset).tolist() == [1, 3, 4]
    (chunked_ce + chunked_kl).backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    print(
        "[i24] self-test passed: action-only body CE, terminal/EOS weights, "
        "96/48/16 KL union, matched indexing, chunked CE/KL, and gradients"
    )


def main() -> None:
    validate_hyperparameters()
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from llamafactory.train.sft import trainer as sft_trainer

    original = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original(self, model, inputs, *args, **kwargs)
        return i24_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
