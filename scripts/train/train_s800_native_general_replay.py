#!/usr/bin/env python3
"""Train a fresh r8 native-General residual over the merged s800 parent.

LLaMA-Factory merges the configured s800 r80 adapter into O6 and then creates
one fresh r8 adapter.  Only that residual is trainable.  PEFT's public
``disable_adapter()`` context is therefore the exact merged-s800 parent used
as the frozen KL reference, without loading a second 193 MiB adapter.

The loss ignores dataset metadata.  It hashes the exact assistant target token
sequence observed in ``labels`` and requires it to be present in the frozen
route manifest.  The 129 registered General targets receive full-response CE
plus 0.05 parent KL.  The 384 parent-distribution targets receive 4.0 parent
KL only and can never contribute gold CE.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
ROUTE_MANIFEST = Path(
    os.environ.get(
        "S800_GENERAL_ROUTE_MANIFEST",
        str(
            ROOT
            / "assets/derived/official_general/s800_native_general_replay_v1_routes.json"
        ),
    )
)
EXPECTED_ROUTE_MANIFEST_SHA256 = (
    "630ec7d0c60363eb64e2a655057200cd70c49bcd4797c671cc7a14750f4004af"
)
EXPECTED_TRAINING_SHA256 = (
    "87097135eb7ddb866b78ae6427c24b8cc2712f898c892d7da92d58ff7e9fddd2"
)

IGNORE_INDEX = -100
GENERAL_KL_WEIGHT = float(os.environ.get("S800_GENERAL_KL", "0.05"))
RETENTION_KL_WEIGHT = float(os.environ.get("S800_RETENTION_KL", "4.0"))
LOGIT_CHUNK = int(os.environ.get("S800_GENERAL_LOGIT_CHUNK", "8"))
EXPECTED_MICROBATCHES = 513
EXPECTED_GENERAL = 129
EXPECTED_RETENTION = 384
EXPECTED_GRADIENT_ACCUMULATION = 4
EXPECTED_OPTIMIZER_STEPS = 129
EXPECTED_RANK = 8
EXPECTED_ALPHA = 8
CHECKPOINT_STEPS = (32, 64, 96, 129)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode("ascii")).hexdigest()


def validate_hyperparameters() -> None:
    if GENERAL_KL_WEIGHT < 0:
        raise RuntimeError(f"S800_GENERAL_KL must be non-negative: {GENERAL_KL_WEIGHT}")
    if RETENTION_KL_WEIGHT <= GENERAL_KL_WEIGHT:
        raise RuntimeError(
            "retention KL must exceed weak General KL: "
            f"{RETENTION_KL_WEIGHT}<={GENERAL_KL_WEIGHT}"
        )
    if LOGIT_CHUNK <= 0:
        raise RuntimeError(f"S800_GENERAL_LOGIT_CHUNK must be positive: {LOGIT_CHUNK}")


def load_route_manifest() -> dict[str, set[str]]:
    state = getattr(load_route_manifest, "state", None)
    if state is not None:
        return state
    if not ROUTE_MANIFEST.is_file():
        raise RuntimeError(f"missing frozen route manifest: {ROUTE_MANIFEST}")
    actual_hash = sha256_file(ROUTE_MANIFEST)
    if actual_hash != EXPECTED_ROUTE_MANIFEST_SHA256:
        raise RuntimeError(
            "route manifest SHA256 drifted: "
            f"{actual_hash}/{EXPECTED_ROUTE_MANIFEST_SHA256}"
        )
    value = json.loads(ROUTE_MANIFEST.read_text(encoding="utf-8"))
    expected_header = {
        "schema_version": "qwen3-nothink-assistant-target-sha256-v1",
        "template": "qwen3_nothink",
        "eos_token": "<|im_end|>",
        "eos_token_id": 151_645,
        "expected_microbatches": EXPECTED_MICROBATCHES,
        "route_counts": {
            "general_ce": EXPECTED_GENERAL,
            "retention_kl": EXPECTED_RETENTION,
        },
        "training_data_sha256": EXPECTED_TRAINING_SHA256,
        "cross_route_target_sha256_collisions": 0,
    }
    drift = {
        key: (value.get(key), expected)
        for key, expected in expected_header.items()
        if value.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"route manifest contract drifted: {drift}")

    general = set(value.get("general_ce_target_sha256") or [])
    retention = set(value.get("retention_kl_target_sha256") or [])
    if len(general) != EXPECTED_GENERAL or len(retention) != EXPECTED_RETENTION:
        raise RuntimeError(
            f"route target cardinality drifted: {len(general)}/{len(retention)}"
        )
    if general & retention:
        raise RuntimeError("General and retention target hashes overlap")
    state = {"general": general, "retention": retention}
    load_route_manifest.state = state
    return state


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("native-General replay requires per_device_train_batch_size=1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("batch has no supervised assistant target")
    start = int(positions[0])
    end = int(positions[-1]) + 1
    expected = torch.arange(start, end, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("packing/history target spans are forbidden")
    if start == 0:
        raise RuntimeError("assistant target begins at token zero")
    return start, end


def route_target(targets: torch.Tensor) -> tuple[str, str]:
    target_hash = token_sha256(targets.detach().cpu().tolist())
    manifest = load_route_manifest()
    if target_hash in manifest["general"]:
        return "general", target_hash
    if target_hash in manifest["retention"]:
        return "retention", target_hash
    raise RuntimeError(f"assistant target is absent from frozen route manifest: {target_hash}")


def token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(0) != 1 or logits.size(1) != targets.numel():
        raise RuntimeError(f"CE logits/targets mismatch: {tuple(logits.shape)}/{targets.shape}")
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        total = total + F.cross_entropy(
            logits[0, start:end].float(),
            targets[start:end].to(logits.device),
            reduction="sum",
        )
    return total / targets.numel()


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(
            "policy/reference logit shape mismatch: "
            f"{tuple(policy_logits.shape)}/{tuple(reference_logits.shape)}"
        )
    if policy_logits.ndim != 3 or policy_logits.size(0) != 1 or policy_logits.size(1) == 0:
        raise RuntimeError(f"KL received invalid logits: {tuple(policy_logits.shape)}")
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
    return total / policy_logits.size(1)


def _single_adapter_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError(f"expected exactly one active residual adapter, got {value!r}")


def assert_formal_trainer_args(trainer: Any) -> None:
    args = trainer.args
    observed = {
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
    }
    expected = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION,
        "max_steps": EXPECTED_OPTIMIZER_STEPS,
        "world_size": 1,
    }
    drift = {
        key: (observed[key], value)
        for key, value in expected.items()
        if observed[key] != value
    }
    if drift:
        raise RuntimeError(f"formal trainer argument contract drifted: {drift}")
    report_to = getattr(args, "report_to", [])
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to:
        raise RuntimeError(f"formal replay requires W&B reporting: {report_to!r}")


def ensure_residual_contract(self: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(self, "_s800_general_residual_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state
    if int(getattr(self.state, "global_step", -1)) != 0:
        raise RuntimeError("residual/reference contract must initialize at global_step=0")
    assert_formal_trainer_args(self)
    peft_config = getattr(unwrapped, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        raise RuntimeError(
            "expected one fresh PEFT residual after merging s800; "
            f"got {list(peft_config) if isinstance(peft_config, dict) else peft_config!r}"
        )
    adapter_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    config = peft_config[adapter_name]
    rank = int(getattr(config, "r", -1))
    alpha = int(getattr(config, "lora_alpha", -1))
    if (rank, alpha) != (EXPECTED_RANK, EXPECTED_ALPHA):
        raise RuntimeError(f"expected r8/alpha8 residual, got r{rank}/alpha{alpha}")
    disable_adapter = getattr(unwrapped, "disable_adapter", None)
    if disable_adapter is None:
        raise RuntimeError("fresh PEFT residual lacks disable_adapter()")

    trainable = [
        (name, parameter.numel())
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    if not trainable or unexpected:
        raise RuntimeError(f"non-LoRA or empty trainable parameter set: {unexpected}")
    state = {
        "adapter_name": adapter_name,
        "trainable_parameters": sum(size for _, size in trainable),
        "logit_fingerprint_checked": False,
    }
    self._s800_general_residual_state = state
    print(
        "[s800-general] residual contract PASS: "
        f"adapter={adapter_name} rank={rank} alpha={alpha} "
        f"trainable={state['trainable_parameters']:,}; "
        "disable_adapter() is exact merged-s800 parent",
        flush=True,
    )
    return unwrapped, state


def paired_forward(
    self: Any,
    model: Any,
    inputs: dict[str, torch.Tensor],
    prediction_positions: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    unwrapped, state = ensure_residual_contract(self, model)
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    with torch.no_grad(), unwrapped.disable_adapter():
        reference_logits = model(
            **inputs, logits_to_keep=prediction_positions
        ).logits.detach()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    outputs = model(**inputs, logits_to_keep=prediction_positions)
    if not state["logit_fingerprint_checked"]:
        if int(getattr(self.state, "global_step", -1)) != 0:
            raise RuntimeError("initial residual fingerprint occurred after step zero")
        max_abs = float(
            (outputs.logits.detach().float() - reference_logits.float()).abs().max()
        )
        if max_abs > 1e-4:
            raise RuntimeError(
                "fresh residual does not reproduce disabled merged-s800 parent: "
                f"max_abs={max_abs:.8f}"
            )
        state["logit_fingerprint_checked"] = True
        print(
            "[s800-general] step-0 residual/parent fingerprint PASS: "
            f"max_abs={max_abs:.8f}",
            flush=True,
        )
    return outputs, reference_logits


def record_route_or_fail(route: str) -> tuple[int, Counter[str]]:
    count = getattr(replay_loss, "call_count", 0) + 1
    if count > EXPECTED_MICROBATCHES:
        raise RuntimeError(f"replay exceeded {EXPECTED_MICROBATCHES} microbatches")
    counts = Counter(getattr(replay_loss, "route_counts", Counter()))
    counts[route] += 1
    if counts["general"] > EXPECTED_GENERAL or counts["retention"] > EXPECTED_RETENTION:
        raise RuntimeError(f"route count exceeded contract: {dict(counts)}")
    remaining = EXPECTED_MICROBATCHES - count
    if (
        counts["general"] + remaining < EXPECTED_GENERAL
        or counts["retention"] + remaining < EXPECTED_RETENTION
    ):
        raise RuntimeError(
            f"remaining rows cannot satisfy route contract: count={count} counts={dict(counts)}"
        )
    if count == EXPECTED_MICROBATCHES and counts != {
        "general": EXPECTED_GENERAL,
        "retention": EXPECTED_RETENTION,
    }:
        raise RuntimeError(f"final route contract mismatch: {dict(counts)}")
    replay_loss.call_count = count
    replay_loss.route_counts = counts
    return count, counts


def assert_final_route_contract() -> None:
    count = getattr(replay_loss, "call_count", 0)
    counts = Counter(getattr(replay_loss, "route_counts", Counter()))
    if count != EXPECTED_MICROBATCHES or counts != {
        "general": EXPECTED_GENERAL,
        "retention": EXPECTED_RETENTION,
    }:
        raise RuntimeError(
            "training ended before the exact replay contract: "
            f"microbatches={count}/{EXPECTED_MICROBATCHES} counts={dict(counts)}"
        )
    print(
        "[s800-general] final route contract PASS: "
        f"microbatches={count} general={counts['general']} retention={counts['retention']}",
        flush=True,
    )


def replay_loss(self: Any, model: Any, inputs: dict[str, torch.Tensor], return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    start, end = target_span(labels)
    targets = labels[0, start:end]
    route, target_hash = route_target(targets)
    count, counts = record_route_or_fail(route)
    prediction_positions = torch.arange(
        start - 1, end - 1, device=inputs["input_ids"].device, dtype=torch.long
    )
    outputs, reference_logits = paired_forward(self, model, inputs, prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != targets.numel():
        raise RuntimeError(
            f"partial logits/targets mismatch: {policy_logits.size(1)}/{targets.numel()}"
        )
    kl = forward_kl(policy_logits, reference_logits)
    if route == "general":
        ce = token_ce(policy_logits, targets)
        loss = ce + GENERAL_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl

    if count <= 12 or count % 50 == 0 or count == EXPECTED_MICROBATCHES:
        print(
            "[s800-general] "
            f"microbatch={count}/{EXPECTED_MICROBATCHES} route={route} "
            f"target={target_hash[:12]} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def _expect_runtime_error(function: Any, text: str) -> None:
    try:
        function()
    except RuntimeError as error:
        if text not in str(error):
            raise AssertionError(f"unexpected error: {error}") from error
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def run_self_test() -> None:
    validate_hyperparameters()
    routes = load_route_manifest()
    assert len(routes["general"]) == EXPECTED_GENERAL
    assert len(routes["retention"]) == EXPECTED_RETENTION

    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 10, 11, 12]], dtype=torch.long)
    assert target_span(labels) == (2, 5)
    _expect_runtime_error(
        lambda: target_span(
            torch.tensor([[IGNORE_INDEX, 10, IGNORE_INDEX, 12]], dtype=torch.long)
        ),
        "packing/history",
    )
    _expect_runtime_error(
        lambda: route_target(torch.tensor([7, 8, 9], dtype=torch.long)),
        "absent from frozen route manifest",
    )

    torch.manual_seed(31)
    logits = torch.randn(1, 17, 37, requires_grad=True)
    targets = torch.randint(0, 37, (17,))
    direct_ce = F.cross_entropy(logits[0].float(), targets)
    chunked_ce = token_ce(logits, targets)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)

    policy = torch.randn(1, 19, 37, requires_grad=True)
    reference = torch.randn(1, 19, 37)
    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / policy.size(1)
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)
    (chunked_ce + chunked_kl).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()

    replay_loss.call_count = EXPECTED_MICROBATCHES - 1
    replay_loss.route_counts = Counter(
        {"general": EXPECTED_GENERAL - 1, "retention": EXPECTED_RETENTION}
    )
    count, counts = record_route_or_fail("general")
    assert count == EXPECTED_MICROBATCHES
    assert counts == {"general": EXPECTED_GENERAL, "retention": EXPECTED_RETENTION}
    assert_final_route_contract()
    del replay_loss.call_count
    del replay_loss.route_counts
    print(
        "[s800-general] self-test PASS: manifest-locked CE/KL routing, contiguous "
        "full-response targets, chunked CE/KL gradients, and 513-row route guard"
    )


def main() -> None:
    validate_hyperparameters()
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from transformers import TrainerCallback
    from llamafactory.train.sft import trainer as sft_trainer

    class ExactCheckpointCallback(TrainerCallback):
        saved_steps: set[int] = set()

        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = int(state.global_step) in CHECKPOINT_STEPS
            return control

        def on_save(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step not in CHECKPOINT_STEPS:
                raise RuntimeError(f"unexpected checkpoint save at step {step}")
            self.saved_steps.add(step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            assert_final_route_contract()
            if int(state.global_step) != EXPECTED_OPTIMIZER_STEPS:
                raise RuntimeError(
                    f"optimizer step contract drifted: {state.global_step}/{EXPECTED_OPTIMIZER_STEPS}"
                )
            if self.saved_steps != set(CHECKPOINT_STEPS):
                raise RuntimeError(
                    f"checkpoint contract drifted: {sorted(self.saved_steps)}/{CHECKPOINT_STEPS}"
                )
            print(
                "[s800-general] checkpoint contract PASS: "
                f"{','.join(map(str, CHECKPOINT_STEPS))}",
                flush=True,
            )
            return control

    original_init = sft_trainer.CustomSeq2SeqTrainer.__init__
    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(ExactCheckpointCallback())

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        return replay_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.__init__ = patched_init
    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
