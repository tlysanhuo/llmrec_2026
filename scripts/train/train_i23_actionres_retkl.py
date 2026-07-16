#!/usr/bin/env python3
"""I-25 trainer: an isolated r16 action residual over merged I-23.

LLaMA-Factory merges the configured I-23 adapter into O6 before creating a
fresh r16 adapter (``create_new_adapter: true``).  The new adapter is the only
trainable component.  Temporarily calling ``disable_adapter()`` therefore
returns the exact merged-I23 parent without loading a second model.

Only tokens in an action answer body after ``</think>`` receive gold CE.
Action reasoning tokens are protected by parent KL, and topic plus every
non-action row are parent-KL-only.  The registered dataset is consumed for one
complete 6,106-microbatch epoch; the final route signature must be exactly
1,752 action rows and 4,354 retention rows.
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
WHITESPACE_IDS = {198, 220, 262, 271}

ACTION_PARENT_KL = float(os.environ.get("I25_ACTION_KL", "0.05"))
RETENTION_KL_WEIGHT = float(os.environ.get("I25_RETENTION_KL", "2.0"))
TERMINAL_MULTIPLIER = float(os.environ.get("I25_TERMINAL_MULTIPLIER", "2.0"))
LOGIT_CHUNK = int(os.environ.get("I25_LOGIT_CHUNK", "16"))
EXPECTED_ROWS = 6_106
EXPECTED_ROUTES = {"action": 1_752, "retention": 4_354}
EXPECTED_RANK = 16
EXPECTED_ALPHA = 16


def validate_hyperparameters() -> None:
    if ACTION_PARENT_KL < 0:
        raise RuntimeError(f"I25_ACTION_KL must be non-negative: {ACTION_PARENT_KL}")
    positive = {
        "I25_RETENTION_KL": RETENTION_KL_WEIGHT,
        "I25_TERMINAL_MULTIPLIER": TERMINAL_MULTIPLIER,
        "I25_LOGIT_CHUNK": LOGIT_CHUNK,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise RuntimeError(f"I-25 hyperparameters must be positive: {invalid}")
    if RETENTION_KL_WEIGHT <= ACTION_PARENT_KL:
        raise RuntimeError(
            "I-25 retention KL must be stronger than the weak action KL: "
            f"{RETENTION_KL_WEIGHT} <= {ACTION_PARENT_KL}"
        )


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I-25 requires per_device_train_batch_size=1")
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
    """Remove formatter-only trailing whitespace but retain the EOS token."""

    end = len(tokens)
    while end > minimum and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    if end <= minimum:
        raise RuntimeError("assistant response has no non-whitespace content")
    return end


def action_terminal_weights(
    tokens: list[int], body: int, content_end: int
) -> torch.Tensor:
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


def route_response(
    targets: torch.Tensor,
) -> tuple[str, int, int, torch.Tensor]:
    """Return route and action-only CE slice relative to the response span."""

    tokens = targets.detach().cpu().tolist()
    content_end = response_content_end(tokens)
    if CLOSE_THINK_ID not in tokens:
        return "retention", 0, 0, torch.empty(0, dtype=torch.float32)

    body = answer_body_start(tokens)
    if body < content_end and tokens[body] in ACTION_START_IDS:
        weights = action_terminal_weights(tokens, body, content_end)
        return "action", body, content_end, weights
    return "retention", 0, 0, torch.empty(0, dtype=torch.float32)


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
    raise RuntimeError(f"expected one active residual adapter, got {value!r}")


def ensure_residual_contract(self, model):
    state = getattr(self, "_i25_residual_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state

    peft_config = getattr(unwrapped, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        raise RuntimeError(
            "I-25 expects exactly one fresh PEFT adapter after merging I-23; "
            f"got {list(peft_config) if isinstance(peft_config, dict) else peft_config!r}"
        )
    adapter_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    config = peft_config[adapter_name]
    rank = int(getattr(config, "r", -1))
    alpha = int(getattr(config, "lora_alpha", -1))
    if (rank, alpha) != (EXPECTED_RANK, EXPECTED_ALPHA):
        raise RuntimeError(
            f"I-25 expected r16/alpha16 residual, got r{rank}/alpha{alpha}"
        )
    disable_adapter = getattr(unwrapped, "disable_adapter", None)
    if disable_adapter is None:
        raise RuntimeError("expected a PEFT model with disable_adapter()")

    trainable = [(name, param.numel()) for name, param in unwrapped.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        unexpected = [name for name, _ in trainable if "lora_" not in name]
        raise RuntimeError(f"non-LoRA or empty I-25 trainable set: {unexpected}")
    state = {
        "adapter_name": adapter_name,
        "fingerprint_checked": False,
        "trainable_parameters": sum(size for _, size in trainable),
    }
    self._i25_residual_state = state
    print(
        "[i25] residual contract PASS: "
        f"adapter={adapter_name} rank={rank} alpha={alpha} "
        f"trainable={state['trainable_parameters']:,}; "
        "disable_adapter() is merged-I23",
        flush=True,
    )
    return unwrapped, state


def paired_forward(self, model, inputs, prediction_positions: torch.Tensor):
    unwrapped, state = ensure_residual_contract(self, model)
    disable_adapter = unwrapped.disable_adapter

    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    with torch.no_grad(), disable_adapter():
        reference_logits = model(
            **inputs, logits_to_keep=prediction_positions
        ).logits.detach()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    outputs = model(**inputs, logits_to_keep=prediction_positions)

    if not state["fingerprint_checked"]:
        max_abs = float(
            (outputs.logits.detach().float() - reference_logits.float()).abs().max()
        )
        if max_abs > 1e-4:
            raise RuntimeError(
                "fresh residual does not reproduce disabled merged-I23 parent: "
                f"max_abs={max_abs:.8f}"
            )
        state["fingerprint_checked"] = True
        print(
            "[i25] initial residual-disabled parent fingerprint PASS: "
            f"max_abs={max_abs:.8f}",
            flush=True,
        )
    return outputs, reference_logits


def validate_route_progress(count: int, counts: Counter[str]) -> None:
    if count > EXPECTED_ROWS:
        raise RuntimeError(
            f"I-25 exceeded its registered one-epoch row count: {count}>{EXPECTED_ROWS}"
        )
    if count == EXPECTED_ROWS and dict(counts) != EXPECTED_ROUTES:
        raise RuntimeError(
            "I-25 final route signature mismatch: "
            f"got {dict(counts)}, expected {EXPECTED_ROUTES}"
        )


def i25_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    start, end = target_span(labels)
    targets = labels[0, start:end]
    route, ce_start, ce_end, ce_weights = route_response(targets)
    prediction_positions = torch.arange(
        start - 1, end - 1, device=inputs["input_ids"].device, dtype=torch.long
    )

    outputs, reference_logits = paired_forward(self, model, inputs, prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != targets.numel():
        raise RuntimeError(
            "partial logits/targets mismatch: "
            f"{policy_logits.size(1)}/{targets.numel()}"
        )
    kl = forward_kl(policy_logits, reference_logits)

    if route == "action":
        ce_targets = targets[ce_start:ce_end]
        ce = weighted_ce(
            policy_logits[:, ce_start:ce_end], ce_targets, ce_weights
        )
        loss = ce + ACTION_PARENT_KL * kl
        ce_tokens = ce_targets.numel()
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl
        ce_tokens = 0

    count = getattr(i25_loss, "call_count", 0) + 1
    i25_loss.call_count = count
    counts = getattr(i25_loss, "route_counts", Counter())
    counts[route] += 1
    i25_loss.route_counts = counts
    validate_route_progress(count, counts)
    if count <= 12 or count % 100 == 0 or count == EXPECTED_ROWS:
        print(
            "[i25] "
            f"microbatch={count}/{EXPECTED_ROWS} route={route} "
            f"response_tokens={targets.numel()} ce_tokens={ce_tokens} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    validate_hyperparameters()
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    thought = [151667, 700, 701, CLOSE_THINK_ID, 198]
    action_tokens = thought + [58, 7, 8, 60, EOS_ID, 198]
    action = torch.tensor([prefix + action_tokens], dtype=torch.long)
    start, end = target_span(action)
    targets = action[0, start:end]
    route, ce_start, ce_end, weights = route_response(targets)
    assert route == "action"
    assert targets[ce_start].item() == 58
    assert targets[ce_start - 1].item() in WHITESPACE_IDS
    assert CLOSE_THINK_ID not in targets[ce_start:ce_end].tolist()
    assert targets[ce_end - 1].item() == EOS_ID
    assert weights[-2].item() == TERMINAL_MULTIPLIER
    assert weights[-1].item() == TERMINAL_MULTIPLIER

    topic = torch.tensor(
        [prefix + thought + [90, 7, 92, EOS_ID, 198]], dtype=torch.long
    )
    topic_start, topic_end = target_span(topic)
    topic_route, topic_ce_start, topic_ce_end, topic_weights = route_response(
        topic[0, topic_start:topic_end]
    )
    assert topic_route == "retention"
    assert (topic_ce_start, topic_ce_end, topic_weights.numel()) == (0, 0, 0)

    no_think = torch.tensor([prefix + [7, 8, EOS_ID, 198]], dtype=torch.long)
    no_start, no_end = target_span(no_think)
    no_route, _, _, no_weights = route_response(no_think[0, no_start:no_end])
    assert no_route == "retention" and no_weights.numel() == 0

    torch.manual_seed(17)
    policy = torch.randn(1, 19, 31, requires_grad=True)
    reference = torch.randn(1, 19, 31)
    ce_targets = torch.randint(0, 31, (7,))
    ce_weights = torch.linspace(1.0, 2.0, 7)
    direct_ce = (
        F.cross_entropy(policy[0, 5:12].float(), ce_targets, reduction="none")
        * ce_weights
    ).sum() / ce_weights.sum()
    chunked_ce = weighted_ce(policy[:, 5:12], ce_targets, ce_weights)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)

    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / policy.size(1)
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)
    (chunked_ce + chunked_kl).backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()

    signature = Counter({"action": 1_752, "retention": 4_354})
    validate_route_progress(EXPECTED_ROWS, signature)
    try:
        validate_route_progress(
            EXPECTED_ROWS, Counter({"action": 1_751, "retention": 4_355})
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("route-signature guard did not reject a bad epoch")

    print(
        "[i25] self-test passed: action-only post-think CE, topic/non-action "
        "KL routing, terminal/EOS weights, body-CE/full-response-KL indexing, "
        "gradients, and 6106-row route guard"
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
        return i25_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
