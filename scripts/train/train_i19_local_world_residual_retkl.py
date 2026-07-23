#!/usr/bin/env python3
"""Train a fresh r16 world residual over the retained local s800 parent.

World rows receive full-response CE plus weak parent KL.  The balanced
retention rows receive parent KL only.  Routing is token-locked to the
canonical world answer prefix; unknown or malformed targets fail closed.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
THINK_END_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}
# qwen2 tokenizes the three newlines before the canonical world answer as
# token 1406, followed by the stable `正确答案是 (` sequence.
WORLD_PREFIX_IDS = (1406, 88991, 102349, 20412, 320)
WORLD_KL_WEIGHT = float(os.environ.get("I19_LOCAL_WORLD_KL", "0.05"))
RETENTION_KL_WEIGHT = float(os.environ.get("I19_LOCAL_RETENTION_KL", "2.0"))
LOGIT_CHUNK = int(os.environ.get("I19_LOCAL_LOGIT_CHUNK", "8"))
EXPECTED_MICROBATCHES = 3146
EXPECTED_WORLD = 1573
EXPECTED_RETENTION = 1573
EXPECTED_GRADIENT_ACCUMULATION = 4
EXPECTED_OPTIMIZER_STEPS = 787
EXPECTED_RANK = 16
EXPECTED_ALPHA = 16


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I19 local trainer requires batch size 1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("batch has no supervised response tokens")
    start = int(positions[0])
    end = int(positions[-1]) + 1
    expected = torch.arange(start, end, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("packing/history target spans are forbidden")
    if start == 0:
        raise RuntimeError("response starts at token zero")
    return start, end


def body_start(tokens: list[int]) -> int:
    try:
        index = tokens.index(THINK_END_ID) + 1
    except ValueError as error:
        raise RuntimeError("response is missing </think>") from error
    while index < len(tokens) and tokens[index] in WHITESPACE_IDS:
        index += 1
    if index >= len(tokens):
        raise RuntimeError("response has no body after </think>")
    return index


def route_targets(targets: torch.Tensor) -> str:
    tokens = targets.detach().cpu().tolist()
    start = body_start(tokens)
    if tuple(tokens[start : start + len(WORLD_PREFIX_IDS)]) == WORLD_PREFIX_IDS:
        return "world"
    if "正确答案是" in "".join(map(str, tokens[start : start + 12])):
        raise RuntimeError("world-like target has an unexpected token prefix")
    return "retention"


def token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(0) != 1 or logits.size(1) != targets.numel():
        raise RuntimeError(f"CE logits/targets mismatch: {tuple(logits.shape)}/{targets.shape}")
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        total = total + F.cross_entropy(
            logits[0, start:end].float(), targets[start:end].to(logits.device), reduction="sum"
        )
    return total / targets.numel()


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(f"policy/reference shape mismatch: {policy_logits.shape}/{reference_logits.shape}")
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, policy_logits.size(1))
        total = total + F.kl_div(
            F.log_softmax(policy_logits[:, start:end].float(), dim=-1),
            F.softmax(reference_logits[:, start:end].float(), dim=-1),
            reduction="sum",
        )
    return total / policy_logits.size(1)


def _single_adapter_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError(f"expected one active adapter, got {value!r}")


def ensure_contract(trainer: Any, model: Any) -> dict[str, Any]:
    state = getattr(trainer, "_i19_local_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        return state
    args = trainer.args
    observed = {
        "batch": int(args.per_device_train_batch_size),
        "accum": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
    }
    expected = {"batch": 1, "accum": 4, "max_steps": 787, "world_size": 1}
    if observed != expected:
        raise RuntimeError(f"trainer contract drifted: {observed}/{expected}")
    report_to = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in report_to:
        raise RuntimeError(f"I19 local training requires W&B: {report_to!r}")
    peft_config = getattr(unwrapped, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        raise RuntimeError("expected one fresh PEFT residual")
    adapter_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    config = peft_config[adapter_name]
    rank_alpha = (int(getattr(config, "r", -1)), int(getattr(config, "lora_alpha", -1)))
    if rank_alpha != (EXPECTED_RANK, EXPECTED_ALPHA):
        raise RuntimeError(f"expected r16/alpha16 residual, got {rank_alpha}")
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("fresh residual lacks disable_adapter()")
    trainable = [name for name, p in unwrapped.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("trainable set contains non-LoRA or is empty")
    state = {"adapter_name": adapter_name, "fingerprint_checked": False}
    trainer._i19_local_state = state
    print(f"[i19-local] residual contract PASS: r16/alpha16 trainable={len(trainable)} tensors", flush=True)
    return state


def paired_forward(trainer: Any, model: Any, inputs: dict[str, torch.Tensor], positions: torch.Tensor):
    state = ensure_contract(trainer, model)
    unwrapped = trainer.accelerator.unwrap_model(model)
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    with torch.no_grad(), unwrapped.disable_adapter():
        reference = model(**inputs, logits_to_keep=positions).logits.detach()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    outputs = model(**inputs, logits_to_keep=positions)
    if not state["fingerprint_checked"]:
        max_abs = float((outputs.logits.detach().float() - reference.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"fresh residual parent fingerprint failed: {max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i19-local] step-0 parent fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return outputs, reference


def record_route(route: str) -> tuple[int, Counter[str]]:
    count = getattr(i19_loss, "call_count", 0) + 1
    counts = Counter(getattr(i19_loss, "route_counts", Counter()))
    counts[route] += 1
    if count > EXPECTED_MICROBATCHES or counts["world"] > EXPECTED_WORLD or counts["retention"] > EXPECTED_RETENTION:
        raise RuntimeError(f"route count exceeded contract: {count}/{dict(counts)}")
    remaining = EXPECTED_MICROBATCHES - count
    if counts["world"] + remaining < EXPECTED_WORLD or counts["retention"] + remaining < EXPECTED_RETENTION:
        raise RuntimeError(f"remaining rows cannot satisfy route contract: {dict(counts)}")
    if count == EXPECTED_MICROBATCHES and counts != {"world": EXPECTED_WORLD, "retention": EXPECTED_RETENTION}:
        raise RuntimeError(f"final route mismatch: {dict(counts)}")
    i19_loss.call_count = count
    i19_loss.route_counts = counts
    return count, counts


def i19_loss(trainer: Any, model: Any, inputs: dict[str, torch.Tensor], return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    start, end = target_span(labels)
    targets = labels[0, start:end]
    route = route_targets(targets)
    count, counts = record_route(route)
    positions = torch.arange(start - 1, end - 1, device=labels.device, dtype=torch.long)
    outputs, reference = paired_forward(trainer, model, inputs, positions)
    logits = outputs.logits
    kl = forward_kl(logits, reference)
    if route == "world":
        ce = token_ce(logits, targets)
        loss = ce + WORLD_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl
    if count <= 8 or count % 200 == 0 or count == EXPECTED_MICROBATCHES:
        print(f"[i19-local] microbatch={count}/{EXPECTED_MICROBATCHES} route={route} tokens={targets.numel()} ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} loss={float(loss.detach()):.6f} counts={dict(counts)}", flush=True)
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    labels = torch.tensor([[-100, -100, 151667, 271, 151668, *WORLD_PREFIX_IDS, 33, 34, 8, EOS_ID, 198]], dtype=torch.long)
    assert target_span(labels) == (2, labels.size(1))
    assert route_targets(labels[0, 2:]) == "world"
    retention = torch.tensor([[-100, -100, 151667, 271, 151668, 198, 1, 2, 3, EOS_ID, 198]], dtype=torch.long)
    assert route_targets(retention[0, 2:]) == "retention"
    torch.manual_seed(41)
    logits = torch.randn(1, 13, 29, requires_grad=True)
    targets = torch.randint(0, 29, (13,))
    assert torch.allclose(token_ce(logits, targets), F.cross_entropy(logits[0].float(), targets), atol=1e-6)
    reference = torch.randn_like(logits)
    (token_ce(logits, targets) + forward_kl(logits, reference)).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print("[i19-local] self-test passed", flush=True)


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    from llamafactory.train.sft import trainer as sft_trainer
    original = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original(self, model, inputs, *args, **kwargs)
        return i19_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    main()
