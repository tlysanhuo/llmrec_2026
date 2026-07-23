#!/usr/bin/env python3
"""Recommendation RFT-lite trainer with a frozen-parent trust region.

The trainable policy adapter and the frozen reference adapter must start from
the same checkpoint.  As in the I-20 trainer, the reference is loaded under a
second PEFT adapter name.  Reference and policy forwards use identical RNG
states so LoRA dropout does not create spurious KL.

Routing is derived only from the supervised response tokens; the prompt is
never inspected or modified.  The mixed-data builder must maintain this
fail-closed contract:

* an RFT positive has exactly one non-empty ``<think>...</think>`` block, then
  (apart from whitespace) exactly one ``domain + s_a + s_b + s_c`` answer;
* every KL-only retention row has an empty think block.

Positive rows receive CE on the complete actual response (CoT, answer, and
EOS) plus weak frozen-parent KL.  Retention rows receive frozen-parent KL only.
Packing and per-device batches larger than one are deliberately rejected.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
OPEN_THINK_ID = 151667
CLOSE_THINK_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}

A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244
DOMAIN_IDS = {
    176245: "video",
    176247: "prod",
    176249: "living",
    176251: "ad",
}

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ADAPTER = os.environ.get(
    "REC_RFT_REFERENCE_ADAPTER",
    str(ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"),
)
REFERENCE_NAME = "rec_rft_frozen_reference"
POSITIVE_KL_WEIGHT = float(os.environ.get("REC_RFT_POSITIVE_KL", "0.20"))
RETENTION_KL_WEIGHT = float(os.environ.get("REC_RFT_RETENTION_KL", "4.0"))
RETENTION_MAX_TOKENS = int(os.environ.get("REC_RFT_RETENTION_MAX_TOKENS", "128"))
LOGIT_CHUNK = int(os.environ.get("REC_RFT_LOGIT_CHUNK", "8"))

if POSITIVE_KL_WEIGHT < 0 or RETENTION_KL_WEIGHT < 0:
    raise ValueError("REC_RFT KL weights must be non-negative")
if RETENTION_MAX_TOKENS <= 0:
    raise ValueError("REC_RFT_RETENTION_MAX_TOKENS must be positive")
if LOGIT_CHUNK <= 0:
    raise ValueError("REC_RFT_LOGIT_CHUNK must be positive")


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    """Return the single contiguous response span in a batch-size-one row."""

    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("REC-RFT requires per_device_train_batch_size=1")
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


def response_layout(tokens: list[int]) -> tuple[int, int, bool]:
    """Validate think/EOS structure and return body, content end, CoT presence."""

    content_end = len(tokens)
    while content_end > 0 and tokens[content_end - 1] in WHITESPACE_IDS:
        content_end -= 1
    if content_end == 0:
        raise RuntimeError("assistant response contains only whitespace")

    eos_positions = [index for index, token in enumerate(tokens[:content_end]) if token == EOS_ID]
    if eos_positions and (len(eos_positions) != 1 or eos_positions[0] != content_end - 1):
        raise RuntimeError("EOS must occur at most once and only at the response end")

    if tokens.count(OPEN_THINK_ID) != 1 or tokens.count(CLOSE_THINK_ID) != 1:
        raise RuntimeError("response must contain exactly one <think> and one </think>")
    if tokens[0] != OPEN_THINK_ID:
        raise RuntimeError("response must begin with <think>")
    close_index = tokens.index(CLOSE_THINK_ID)
    if close_index <= 0:
        raise RuntimeError("</think> occurs before a valid think span")

    thought = tokens[1:close_index]
    has_actual_cot = any(token not in WHITESPACE_IDS for token in thought)

    body = close_index + 1
    while body < content_end and tokens[body] in WHITESPACE_IDS:
        body += 1
    if body >= content_end or (body == content_end - 1 and tokens[body] == EOS_ID):
        raise RuntimeError("assistant response has no body after </think>")
    return body, content_end, has_actual_cot


def positive_domain(tokens: list[int], body: int, content_end: int) -> str:
    """Validate the exact four-token RFT answer and return its domain."""

    answer_end = content_end
    if tokens[answer_end - 1] == EOS_ID:
        answer_end -= 1
    while answer_end > body and tokens[answer_end - 1] in WHITESPACE_IDS:
        answer_end -= 1
    answer = tokens[body:answer_end]
    if len(answer) != 4:
        raise RuntimeError(
            "non-empty-think row must contain exactly domain+s_a+s_b+s_c after </think>"
        )

    domain = DOMAIN_IDS.get(answer[0])
    if domain is None:
        raise RuntimeError(f"unknown recommendation domain token: {answer[0]}")
    if not (A_LO <= answer[1] <= A_HI):
        raise RuntimeError(f"broken {domain} s_a token: {answer[1]}")
    if not (B_LO <= answer[2] <= B_HI):
        raise RuntimeError(f"broken {domain} s_b token: {answer[2]}")
    if not (C_LO <= answer[3] <= C_HI):
        raise RuntimeError(f"broken {domain} s_c token: {answer[3]}")
    return domain


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
    """Route solely from response labels and select CE/KL prediction targets."""

    start, end = target_span(labels)
    tokens = labels[0, start:end].detach().cpu().tolist()
    body, content_end, has_actual_cot = response_layout(tokens)

    if has_actual_cot:
        domain = positive_domain(tokens, body, content_end)
        # Full actual response CE: <think>, CoT, </think>, answer, and EOS.
        selected = torch.arange(
            start, start + content_end, device=labels.device, dtype=torch.long
        )
        return f"positive_{domain}", selected

    # Builder contract: every empty-think row is retention, even if its answer
    # happens to have recommendation item tokens.  No gold CE is applied here.
    relative = capped_positions(body, content_end, RETENTION_MAX_TOKENS, labels.device)
    return "retention", relative + start


def token_ce(policy_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean token CE without materializing a full-response fp32 logits copy."""

    if policy_logits.ndim != 3 or policy_logits.size(0) != 1:
        raise RuntimeError(f"expected [1,tokens,vocab] logits, got {policy_logits.shape}")
    if targets.ndim != 1 or policy_logits.size(1) != targets.numel():
        raise RuntimeError(
            f"logit/target shape mismatch: {policy_logits.shape}/{targets.shape}"
        )
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        total = total + F.cross_entropy(
            policy_logits[0, start:end].float(),
            targets[start:end].to(policy_logits.device),
            reduction="sum",
        )
    return total / targets.numel()


def forward_kl(policy_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Mean forward KL(reference || policy), evaluated in small fp32 chunks."""

    if policy_logits.shape != ref_logits.shape:
        raise RuntimeError(
            f"policy/reference logit shape mismatch: {policy_logits.shape}/{ref_logits.shape}"
        )
    if policy_logits.ndim != 3 or policy_logits.size(0) != 1 or policy_logits.size(1) == 0:
        raise RuntimeError(f"expected non-empty [1,tokens,vocab] logits, got {policy_logits.shape}")
    token_count = policy_logits.size(1)
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, token_count, LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, token_count)
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
    state = getattr(self, "_rec_rft_reference_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state["policy_name"]

    policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name == REFERENCE_NAME:
        raise RuntimeError("policy adapter name collides with REC-RFT reference name")
    if not Path(REFERENCE_ADAPTER).is_dir():
        raise RuntimeError(f"missing frozen REC-RFT reference adapter: {REFERENCE_ADAPTER}")

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

    self._rec_rft_reference_state = {
        "policy_name": policy_name,
        "initial_trainable": initial_trainable,
        "fingerprint_checked": False,
    }
    print(
        "[rec-rft] frozen reference loaded: "
        f"path={REFERENCE_ADAPTER} policy={policy_name} reference={REFERENCE_NAME} "
        f"trainable={initial_trainable:,}",
        flush=True,
    )
    return unwrapped, policy_name


def set_active_adapter(unwrapped, name: str, policy_name: str) -> None:
    unwrapped.set_adapter(name)
    unwrapped.set_requires_grad(policy_name, name == policy_name)
    unwrapped.set_requires_grad(REFERENCE_NAME, False)


def paired_forward(self, model, inputs: dict[str, torch.Tensor], prediction_positions: torch.Tensor):
    unwrapped, policy_name = ensure_reference(self, model)
    model.train()

    cpu_rng = torch.get_rng_state()
    cuda_device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(cuda_device) if cuda_device.type == "cuda" else None

    try:
        set_active_adapter(unwrapped, REFERENCE_NAME, policy_name)
        with torch.no_grad():
            ref_outputs = model(**inputs, logits_to_keep=prediction_positions)
            ref_logits = ref_outputs.logits.detach()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, cuda_device)
        set_active_adapter(unwrapped, policy_name, policy_name)

    outputs = model(**inputs, logits_to_keep=prediction_positions)

    state = self._rec_rft_reference_state
    if not state["fingerprint_checked"]:
        max_abs = float((outputs.logits.detach().float() - ref_logits.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"initial policy/reference logits differ: max_abs={max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(
            f"[rec-rft] initial policy/reference fingerprint PASS: max_abs={max_abs:.8f}",
            flush=True,
        )
    return outputs, ref_logits


def rec_rft_loss(self, model, inputs, return_outputs=False, **kwargs):
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
        ce = token_ce(policy_logits, targets)
        loss = ce + POSITIVE_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl

    call_count = getattr(rec_rft_loss, "call_count", 0) + 1
    rec_rft_loss.call_count = call_count
    counts = getattr(rec_rft_loss, "route_counts", Counter())
    counts[route] += 1
    rec_rft_loss.route_counts = counts
    if call_count <= 12 or call_count % 100 == 0:
        print(
            "[rec-rft] "
            f"microbatch={call_count} route={route} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def _expect_runtime_error(fn, text: str) -> None:
    try:
        fn()
    except RuntimeError as error:
        if text not in str(error):
            raise AssertionError(f"unexpected error: {error}") from error
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def run_self_test() -> None:
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    thought = [OPEN_THINK_ID, 198, 700, 701, 198, CLOSE_THINK_ID, 198]

    for domain_id, domain in DOMAIN_IDS.items():
        response = thought + [domain_id, A_LO, B_LO, C_LO, EOS_ID, 198]
        labels = torch.tensor([prefix + response], dtype=torch.long)
        route, positions = route_and_positions(labels)
        assert route == f"positive_{domain}"
        assert positions[0].item() == len(prefix)
        assert labels[0, positions].tolist() == response[:-1]
        assert CLOSE_THINK_ID in labels[0, positions].tolist()
        assert labels[0, positions][-5:].tolist() == [domain_id, A_LO, B_LO, C_LO, EOS_ID]

    retention_response = [
        OPEN_THINK_ID,
        198,
        CLOSE_THINK_ID,
        198,
        58,
        10,
        60,
        EOS_ID,
        198,
    ]
    retention = torch.tensor([prefix + retention_response], dtype=torch.long)
    route, positions = route_and_positions(retention)
    assert route == "retention"
    assert retention[0, positions].tolist() == [58, 10, 60, EOS_ID]
    assert OPEN_THINK_ID not in retention[0, positions].tolist()

    long_retention = torch.tensor(
        [
            prefix
            + [OPEN_THINK_ID, CLOSE_THINK_ID, 198]
            + [1_000] * (RETENTION_MAX_TOKENS + 17)
            + [EOS_ID]
        ],
        dtype=torch.long,
    )
    route, positions = route_and_positions(long_retention)
    assert route == "retention" and positions.numel() == RETENTION_MAX_TOKENS
    assert positions.unique().numel() == RETENTION_MAX_TOKENS

    malformed = torch.tensor(
        [prefix + thought + [176245, A_LO, B_LO, C_LO, 42, EOS_ID]], dtype=torch.long
    )
    _expect_runtime_error(
        lambda: route_and_positions(malformed), "exactly domain+s_a+s_b+s_c"
    )
    bad_code = torch.tensor(
        [prefix + thought + [176245, A_LO, B_LO, C_HI + 1, EOS_ID]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(bad_code), "broken video s_c token")
    missing_think = torch.tensor(
        [prefix + [176245, A_LO, B_LO, C_LO, EOS_ID]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(missing_think), "exactly one <think>")

    torch.manual_seed(17)
    ce_logits = torch.randn(1, 11, 31, requires_grad=True)
    ce_targets = torch.randint(0, 31, (11,))
    direct_ce = F.cross_entropy(ce_logits[0].float(), ce_targets)
    chunked_ce = token_ce(ce_logits, ce_targets)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)
    chunked_ce.backward()
    assert ce_logits.grad is not None and torch.isfinite(ce_logits.grad).all()

    policy = torch.randn(1, 11, 31, requires_grad=True)
    reference = torch.randn(1, 11, 31)
    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / 11
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)
    chunked_kl.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()

    print(
        "[rec-rft] self-test passed: four-domain response-only routing, full-response "
        "positive CE mask, empty-think retention, fail-closed validation, capped KL, "
        "chunked CE/KL, and gradients are consistent"
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
        if args:
            if len(args) != 1 or "return_outputs" in kwargs:
                raise RuntimeError("unexpected positional CustomSeq2SeqTrainer.compute_loss arguments")
            kwargs["return_outputs"] = args[0]
        return rec_rft_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
