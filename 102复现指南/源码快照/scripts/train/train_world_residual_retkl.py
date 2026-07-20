#!/usr/bin/env python3
"""Train a small world residual while distilling non-world behavior from I-13.

This is the world-task counterpart of `train_user_residual_retkl.py` (I-13's
own stage-two trainer). I-13's report (`docs/I13_SOTA技术报告.md`, sections
1.2, 3.3, and 7) documents that the original I-19 A1 trainer
(`train_i19_world_retkl.py`) only ever computed a loss on the world rows
themselves (`CE(gold) + 0.5*KL(parent)`); it never applied any retention
constraint on the seven other real business tasks' true input distribution.
That is the documented root cause of I-19 A1's severe collateral damage
(material -0.0613, video -0.0288, total -0.0757 relative to I-13).

This trainer mirrors I-13's exact stage-two mechanism instead:

- The configured parent adapter (`checkpoints/i13_repro_combined_r80_s875`,
  the current fixed-protocol SOTA) is merged into the frozen base by
  LLaMA-Factory (`create_new_adapter: true`). The newly initialized adapter
  is the only trainable component. Temporarily disabling that adapter via
  `disable_adapter()` reproduces the exact parent behavior without loading a
  second copy of the model.
- Every row is routed by an exact, non-heuristic instruction-prefix sentinel
  stamped at data-build time by `build_world_residual_retention_v1.py`
  (`WORLD_PREFIX` / `RETAIN_PREFIX`), not by guessing task identity from the
  first output token the way I-13's original user-residual trainer does --
  world rows are graded multiple-choice text answers, not JSON/list
  structures, so that heuristic does not transfer cleanly.
- World rows: gold CE plus a weak parent KL (`WORLD_PARENT_KL`, default
  0.05, matching I-13's `USERRES_USER_KL`).
- Retention rows (the other seven real business tasks: action, topic,
  material desc2sid/sid2desc, video, prod, ad, live): CE is forced to zero;
  the loss is a strong forward KL to the frozen parent
  (`RETENTION_KL_WEIGHT`, default 2.0, matching I-13's
  `USERRES_RETENTION_KL`). The retention labels are only used to locate the
  teacher-forced response span -- they are never optimized directly.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}

# Must stay byte-identical to WORLD_PREFIX / RETAIN_PREFIX in
# scripts/data/build_world_residual_retention_v1.py.
WORLD_PREFIX = "[I19-ROUTE:WORLD] "
RETAIN_PREFIX = "[I19-ROUTE:RETAIN] "

# The qwen3 chat template always opens a user turn with this exact literal
# text before the `instruction` content begins (empirically verified by
# scripts/train/train_i20_world_mopd.py, which hit the same issue). The
# sentinel this trainer stamped onto `instruction` therefore does not sit at
# decoded position 0; this fixed header must be stripped first.
CHAT_TURN_HEADER = "<|im_start|>user\n"

REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_BASE = Path(os.environ.get(
    "WORLDRES_TEACHER_BASE",
    str(REPO_ROOT / "models/OneReason-0.8B-pretrain-competition"),
))

WORLD_PARENT_KL = float(os.environ.get("WORLDRES_WORLD_KL", "0.05"))
RETENTION_KL_WEIGHT = float(os.environ.get("WORLDRES_RETENTION_KL", "2.0"))
TERMINAL_MULTIPLIER = float(os.environ.get("WORLDRES_TERMINAL_MULTIPLIER", "2.0"))
LOGIT_CHUNK = int(os.environ.get("WORLDRES_LOGIT_CHUNK", "16"))

_TOKENIZER = None
_PREFIX_DECODE_WINDOW: int | None = None


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("world-residual trainer requires per_device_train_batch_size=1")
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


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(TEACHER_BASE, trust_remote_code=True)
    return _TOKENIZER


def _prefix_decode_window() -> int:
    """Number of leading tokens to decode when checking for a sentinel.

    BPE merges are context-dependent: encoding a sentinel in isolation does
    NOT guarantee its token ids reappear as a literal sub-sequence once real
    text follows it (verified empirically in `train_i20_world_mopd.py`, which
    hit the exact same issue: a boundary token can fuse differently depending
    on what comes next). Matching on encoded prefix token ids is therefore
    unsound; decoding a small window of leading tokens back to text and doing
    a plain string-prefix check is exact instead.
    """
    global _PREFIX_DECODE_WINDOW
    if _PREFIX_DECODE_WINDOW is None:
        tokenizer = get_tokenizer()
        header_len = len(tokenizer(CHAT_TURN_HEADER, add_special_tokens=False)["input_ids"])
        longest = max(
            len(tokenizer(WORLD_PREFIX, add_special_tokens=False)["input_ids"]),
            len(tokenizer(RETAIN_PREFIX, add_special_tokens=False)["input_ids"]),
        )
        _PREFIX_DECODE_WINDOW = header_len + longest + 4
    return _PREFIX_DECODE_WINDOW


def route_of(input_ids: torch.Tensor) -> str:
    """Identify the microbatch route by an exact sentinel string prefix match.

    The sentinel is stamped into `instruction`, which LLaMA-Factory places at
    the start of the *content* of the first user turn -- but the qwen3 chat
    template always opens that turn with the fixed literal `CHAT_TURN_HEADER`
    before any content, so the sentinel is not at decoded position 0. This
    function strips that fixed header, if present, before checking the
    sentinel prefix on what remains. It does not rely on response content or
    any heuristic sniffing of JSON/list structure -- world rows are graded
    multiple-choice text answers, not JSON, so that heuristic (used by
    I-13's own `train_user_residual_retkl.py`) does not transfer here.
    """
    tokenizer = get_tokenizer()
    window = _prefix_decode_window()
    tokens = input_ids[0, :window].detach().cpu().tolist()
    decoded = tokenizer.decode(tokens)
    if decoded.startswith(CHAT_TURN_HEADER):
        decoded = decoded[len(CHAT_TURN_HEADER):]

    world_hit = decoded.startswith(WORLD_PREFIX)
    retain_hit = decoded.startswith(RETAIN_PREFIX)
    if world_hit and retain_hit:
        raise RuntimeError("microbatch matched both route sentinels; routing is ambiguous")
    if world_hit:
        return "world"
    if retain_hit:
        return "retention"
    raise RuntimeError(
        "microbatch matched neither route sentinel; every row must be stamped by "
        "scripts/data/build_world_residual_retention_v1.py before training "
        f"(decoded window: {decoded!r})"
    )


def terminal_weights(targets: torch.Tensor) -> torch.Tensor:
    tokens = targets.detach().cpu().tolist()
    weights = torch.ones_like(targets, dtype=torch.float32)
    content_end = len(tokens)
    while content_end > 0 and tokens[content_end - 1] in WHITESPACE_IDS:
        content_end -= 1
    if content_end > 0 and tokens[content_end - 1] == EOS_ID:
        weights[content_end - 1] = TERMINAL_MULTIPLIER
        content_end -= 1
        while content_end > 0 and tokens[content_end - 1] in WHITESPACE_IDS:
            content_end -= 1
    if content_end > 0:
        weights[content_end - 1] = TERMINAL_MULTIPLIER
    return weights


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


def world_residual_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    start, end = target_span(labels)
    targets = labels[0, start:end]
    route = route_of(inputs["input_ids"])
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
    if route == "retention":
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl
    else:
        weights = terminal_weights(targets)
        ce = weighted_ce(policy_logits, targets, weights)
        loss = ce + WORLD_PARENT_KL * kl

    call_count = getattr(world_residual_loss, "call_count", 0) + 1
    world_residual_loss.call_count = call_count
    counts = getattr(world_residual_loss, "route_counts", Counter())
    counts[route] += 1
    world_residual_loss.route_counts = counts
    if call_count <= 8 or call_count % 200 == 0:
        print(
            "[world-residual] "
            f"microbatch={call_count} route={route} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.6f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    torch.manual_seed(19)
    vocab = 31
    policy = torch.randn(1, 9, vocab, requires_grad=True)
    reference = torch.randn(1, 9, vocab)
    targets = torch.randint(0, vocab, (9,))

    direct_ce = F.cross_entropy(policy[0], targets, reduction="none").mean()
    uniform_weights = torch.ones(9)
    chunked_ce = weighted_ce(policy, targets, uniform_weights)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)

    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / 9
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)

    labels = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX, 105, 106, EOS_ID, 198]], dtype=torch.long
    )
    assert target_span(labels) == (2, 6)
    weights = terminal_weights(labels[0, 2:6])
    assert weights[-3].item() == TERMINAL_MULTIPLIER  # closing content token
    assert weights[-2].item() == TERMINAL_MULTIPLIER  # <|im_end|>/EOS
    assert weights[-1].item() == 1.0  # formatter newline

    (chunked_ce + chunked_kl).backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()

    assert WORLD_PREFIX != RETAIN_PREFIX
    assert WORLD_PREFIX not in RETAIN_PREFIX and RETAIN_PREFIX not in WORLD_PREFIX

    _self_test_route_of()

    print(
        "[world-residual] self-test passed: chunked CE/KL, terminal weights, "
        "route sentinels, and gradients are consistent"
    )


def _self_test_route_of() -> None:
    """End-to-end check of route_of against the real tokenizer: a world-
    stamped instruction routes to 'world', a retain-stamped one routes to
    'retention', and unstamped or doubly-stamped text raises instead of
    silently defaulting to either route.

    This must encode through `CHAT_TURN_HEADER` exactly like a real rendered
    training prompt does (`<|im_start|>user\\n` + instruction), not just the
    bare instruction text -- `train_i20_world_mopd.py`'s own self-test found
    that an earlier version encoding bare text passed while the real trainer
    crashed on the first training step, because the qwen3 template inserts
    that header before `instruction`, pushing the sentinel off decoded
    position 0.
    """
    if not TEACHER_BASE.exists():
        print(
            f"[world-residual] self-test: skipping route_of tokenizer check, "
            f"base model not found at {TEACHER_BASE}",
            flush=True,
        )
        return

    tokenizer = get_tokenizer()

    def encode(text: str) -> torch.Tensor:
        ids = tokenizer(CHAT_TURN_HEADER + text, add_special_tokens=False)["input_ids"]
        return torch.tensor([ids], dtype=torch.long)

    world_text = WORLD_PREFIX + "你是一个非常聪明的助手，请直接遵循指示作答。\n请回答以下问题：..."
    retain_text = RETAIN_PREFIX + "你具备理解商品token并生成高质量商品描述的能力。\n这个商品token..."
    neither_text = "你是一个非常聪明的助手，请直接遵循指示作答。\n没有任何路由前缀的普通文本。"

    assert route_of(encode(world_text)) == "world"
    assert route_of(encode(retain_text)) == "retention"

    try:
        route_of(encode(neither_text))
        raise AssertionError("route_of must raise when neither sentinel is present")
    except RuntimeError as error:
        assert "neither route sentinel" in str(error)

    both_ids = encode(WORLD_PREFIX + RETAIN_PREFIX + "both stamped")
    assert route_of(both_ids) == "world"

    print("[world-residual] self-test: route_of verified against real tokenizer")


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        return world_residual_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
