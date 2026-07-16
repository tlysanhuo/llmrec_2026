#!/usr/bin/env python3
"""I-20 trainer: product/ad positive answer CE with a frozen I-13 trust region.

The policy is the trainable I-13 r80 adapter loaded by LLaMA-Factory.  On the
first microbatch, the same adapter is loaded under a second frozen PEFT name.
Switching adapter names gives an exact initial-policy reference without a
second base model.  Matching RNG states make LoRA dropout identical for the
reference and policy forwards, so retention KL measures parameter drift rather
than dropout noise.

Only the final domain marker and three SID tokens receive gold CE on product/ad
rows.  Every other row is KL-only retention data.
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
WHITESPACE_IDS = {198, 220, 262, 271}

A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244
PROD_DOMAIN_ID = 176247
AD_DOMAIN_ID = 176251
POSITIVE_DOMAIN_IDS = {PROD_DOMAIN_ID: "prod", AD_DOMAIN_ID: "ad"}
# Tokenization of the O1 recommendation-answer signature `该用户最近`.
# Material and action targets may also contain prod/ad itemic tokens, so the
# domain marker alone is not a safe loss-route discriminator.
RECOMMENDATION_PREFIX_IDS = [75882, 20002, 104044]

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ADAPTER = os.environ.get(
    "I20_REFERENCE_ADAPTER",
    str(ROOT / "submissions/e3_userres_r80_retkl_v3_s875_platform"),
)
REFERENCE_NAME = "i20_frozen_reference"
POSITIVE_KL_WEIGHT = float(os.environ.get("I20_POSITIVE_KL", "0.20"))
RETENTION_KL_WEIGHT = float(os.environ.get("I20_RETENTION_KL", "4.0"))
RETENTION_MAX_TOKENS = int(os.environ.get("I20_RETENTION_MAX_TOKENS", "128"))
LOGIT_CHUNK = int(os.environ.get("I20_LOGIT_CHUNK", "8"))


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I-20 requires per_device_train_batch_size=1")
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
    if CLOSE_THINK_ID not in tokens:
        return 0
    index = tokens.index(CLOSE_THINK_ID) + 1
    while index < len(tokens) and tokens[index] in WHITESPACE_IDS:
        index += 1
    if index >= len(tokens):
        raise RuntimeError("assistant response has no body after </think>")
    return index


def capped_positions(start: int, end: int, cap: int, device: torch.device) -> torch.Tensor:
    if end <= start:
        raise RuntimeError(f"invalid retention target range: {start}:{end}")
    count = end - start
    if count <= cap:
        return torch.arange(start, end, device=device, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap, device=device).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("uniform retention cap produced duplicate positions")
    return relative + start


def route_and_positions(labels: torch.Tensor) -> tuple[str, torch.Tensor]:
    start, end = target_span(labels)
    targets = labels[0, start:end]
    tokens = targets.detach().cpu().tolist()
    body = body_start(tokens)

    has_recommendation_prefix = any(
        tokens[index : index + len(RECOMMENDATION_PREFIX_IDS)] == RECOMMENDATION_PREFIX_IDS
        for index in range(body, len(tokens) - len(RECOMMENDATION_PREFIX_IDS) + 1)
    )
    matches: list[tuple[int, str]] = []
    for index in range(body, len(tokens) - 3):
        domain = POSITIVE_DOMAIN_IDS.get(tokens[index])
        if domain is None:
            continue
        if not (
            A_LO <= tokens[index + 1] <= A_HI
            and B_LO <= tokens[index + 2] <= B_HI
            and C_LO <= tokens[index + 3] <= C_HI
        ):
            raise RuntimeError(f"broken {domain} itemic target after domain marker")
        matches.append((index, domain))

    if has_recommendation_prefix and matches:
        if len(matches) != 1:
            raise RuntimeError(f"positive row contains {len(matches)} final itemic targets")
        index, domain = matches[0]
        selected = torch.arange(
            start + index, start + index + 4, device=labels.device, dtype=torch.long
        )
        return f"positive_{domain}", selected

    content_end = len(tokens)
    while content_end > body and tokens[content_end - 1] in WHITESPACE_IDS:
        content_end -= 1
    relative = capped_positions(body, content_end, RETENTION_MAX_TOKENS, labels.device)
    return "retention", relative + start


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


def _single_adapter_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError(f"expected one active PEFT adapter, got {value!r}")


def ensure_reference(self, model) -> tuple[object, str]:
    state = getattr(self, "_i20_reference_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state["policy_name"]

    policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name == REFERENCE_NAME:
        raise RuntimeError("policy adapter name collides with I-20 reference name")
    if not Path(REFERENCE_ADAPTER).is_dir():
        raise RuntimeError(f"missing frozen I-13 reference adapter: {REFERENCE_ADAPTER}")

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
            f"trainable parameter count changed after reference load: "
            f"{initial_trainable}/{final_trainable}"
        )
    self._i20_reference_state = {
        "policy_name": policy_name,
        "initial_trainable": initial_trainable,
        "fingerprint_checked": False,
    }
    print(
        "[i20] frozen reference loaded: "
        f"path={REFERENCE_ADAPTER} policy={policy_name} reference={REFERENCE_NAME} "
        f"trainable={initial_trainable:,}",
        flush=True,
    )
    return unwrapped, policy_name


def set_active_adapter(unwrapped, name: str, trainable: bool) -> None:
    unwrapped.set_adapter(name)
    unwrapped.set_requires_grad(name, trainable)
    other = REFERENCE_NAME if name != REFERENCE_NAME else None
    if other is not None and other in unwrapped.peft_config:
        unwrapped.set_requires_grad(other, False)


def paired_forward(self, model, inputs: dict[str, torch.Tensor], prediction_positions: torch.Tensor):
    unwrapped, policy_name = ensure_reference(self, model)
    model.train()

    cpu_rng = torch.get_rng_state()
    cuda_device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(cuda_device) if cuda_device.type == "cuda" else None

    set_active_adapter(unwrapped, REFERENCE_NAME, trainable=False)
    with torch.no_grad():
        ref_outputs = model(**inputs, logits_to_keep=prediction_positions)
        ref_logits = ref_outputs.logits.detach()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, cuda_device)
    set_active_adapter(unwrapped, policy_name, trainable=True)
    outputs = model(**inputs, logits_to_keep=prediction_positions)

    state = self._i20_reference_state
    if not state["fingerprint_checked"]:
        max_abs = float((outputs.logits.detach().float() - ref_logits.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"initial policy/reference logits differ: max_abs={max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i20] initial policy/reference fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return outputs, ref_logits


def i20_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    route, selected_positions = route_and_positions(labels)
    prediction_positions = selected_positions - 1
    targets = labels[0, selected_positions]

    outputs, ref_logits = paired_forward(self, model, inputs, prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != targets.numel():
        raise RuntimeError(
            f"partial logits/targets mismatch: {policy_logits.size(1)}/{targets.numel()}"
        )

    kl = forward_kl(policy_logits, ref_logits)
    if route.startswith("positive_"):
        ce = F.cross_entropy(policy_logits[0].float(), targets.to(policy_logits.device))
        loss = ce + POSITIVE_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl

    call_count = getattr(i20_loss, "call_count", 0) + 1
    i20_loss.call_count = call_count
    counts = getattr(i20_loss, "route_counts", Counter())
    counts[route] += 1
    i20_loss.route_counts = counts
    if call_count <= 12 or call_count % 100 == 0:
        print(
            "[i20] "
            f"microbatch={call_count} route={route} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    positive = torch.tensor(
        [
            prefix
            + [
                151667,
                CLOSE_THINK_ID,
                198,
                *RECOMMENDATION_PREFIX_IDS,
                32664,
                PROD_DOMAIN_ID,
                A_LO,
                B_LO,
                C_LO,
                EOS_ID,
                198,
            ]
        ],
        dtype=torch.long,
    )
    route, positions = route_and_positions(positive)
    assert route == "positive_prod"
    assert positive[0, positions].tolist() == [PROD_DOMAIN_ID, A_LO, B_LO, C_LO]

    retention_tokens = [151667, CLOSE_THINK_ID, 198] + list(range(1_000, 1_200)) + [EOS_ID, 198]
    retention = torch.tensor([prefix + retention_tokens], dtype=torch.long)
    route, positions = route_and_positions(retention)
    assert route == "retention" and positions.numel() == RETENTION_MAX_TOKENS
    assert positions.unique().numel() == RETENTION_MAX_TOKENS

    torch.manual_seed(17)
    policy = torch.randn(1, 11, 31, requires_grad=True)
    reference = torch.randn(1, 11, 31)
    direct = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / 11
    chunked = forward_kl(policy, reference)
    assert torch.allclose(direct, chunked, atol=1e-6)
    chunked.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    print(
        "[i20] self-test passed: route detection, final four-token CE mask, "
        "uniform retention cap, chunked KL, and gradients are consistent"
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
        return i20_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
