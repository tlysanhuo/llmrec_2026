#!/usr/bin/env python3
"""I-28 multi-gold proposal trainer with a frozen-I23 trust region.

The policy adapter is loaded through the training stack and upcast before this
trainer runs.  The frozen reference adapter is then initialized from that
already-loaded policy by an exact per-tensor copy.  Loading the same on-disk
checkpoint a second time is not sufficient: PEFT can round that second load
through the base-model dtype before autocasting it back to fp32.  Proposal/
retention routing is derived solely from the supervised assistant response;
dataset metadata is deliberately ignored by the loss.

Fail-closed response contract (batch size one, packing disabled):

* every response has exactly one ``<think>...</think>`` block and a terminal
  EOS (formatter whitespace after EOS is tolerated);
* a proposal has an empty think block and exactly
  ``domain + s_a + s_b + s_c + EOS`` after ``</think>``;
* every other structurally valid response is retention-only.  The builder must
  therefore audit that no intended retention row has the proposal signature.

Proposal rows receive CE only on the four item tokens and EOS, plus weak
forward KL to frozen I-23 on those same five positions.  Retention rows receive
only frozen-I23 KL on at most 128 uniformly sampled answer-body positions.
Input embeddings and the language-model head must remain frozen.
Every adapter activation is checked through PEFT's public model-status API,
and every checkpoint/root save explicitly selects the policy adapter; the
in-memory frozen reference is never serialized.
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
    "I28_REFERENCE_ADAPTER",
    str(ROOT / "checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/checkpoint-1995"),
)
REFERENCE_NAME = "i28_frozen_i23_reference"
PROPOSAL_KL_WEIGHT = float(os.environ.get("I28_PROPOSAL_KL", "0.20"))
RETENTION_KL_WEIGHT = float(os.environ.get("I28_RETENTION_KL", "4.0"))
RETENTION_MAX_TOKENS = int(os.environ.get("I28_RETENTION_MAX_TOKENS", "128"))
LOGIT_CHUNK = int(os.environ.get("I28_LOGIT_CHUNK", "8"))
EXPECTED_MICROBATCHES = int(os.environ.get("I28_EXPECTED_MICROBATCHES", "512"))
EXPECTED_PROPOSALS = int(os.environ.get("I28_EXPECTED_PROPOSALS", "128"))
EXPECTED_RETENTIONS = int(os.environ.get("I28_EXPECTED_RETENTIONS", "384"))
EXPECTED_PROPOSAL_DOMAIN = os.environ.get("I28_EXPECTED_PROPOSAL_DOMAIN", "video")
EXPECTED_GRADIENT_ACCUMULATION = 4
EXPECTED_OPTIMIZER_STEPS = 128
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"


def validate_hyperparameters() -> None:
    if PROPOSAL_KL_WEIGHT < 0 or RETENTION_KL_WEIGHT < 0:
        raise RuntimeError("I-28 KL weights must be non-negative")
    if RETENTION_MAX_TOKENS <= 0:
        raise RuntimeError("I28_RETENTION_MAX_TOKENS must be positive")
    if LOGIT_CHUNK <= 0:
        raise RuntimeError("I28_LOGIT_CHUNK must be positive")
    expected = (EXPECTED_MICROBATCHES, EXPECTED_PROPOSALS, EXPECTED_RETENTIONS)
    if any(value <= 0 for value in expected):
        raise RuntimeError(f"I-28 expected route counts must be positive: {expected}")
    if EXPECTED_PROPOSALS + EXPECTED_RETENTIONS != EXPECTED_MICROBATCHES:
        raise RuntimeError(
            "I-28 expected proposal+retention counts must equal expected microbatches"
        )
    if EXPECTED_PROPOSAL_DOMAIN not in DOMAIN_IDS.values():
        raise RuntimeError(
            f"unknown I28_EXPECTED_PROPOSAL_DOMAIN={EXPECTED_PROPOSAL_DOMAIN!r}"
        )


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    """Return the one contiguous response span for a batch-size-one row."""

    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I-28 requires per_device_train_batch_size=1")
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
    """Validate response framing and return body start/end and empty-think flag."""

    content_end = len(tokens)
    while content_end > 0 and tokens[content_end - 1] in WHITESPACE_IDS:
        content_end -= 1
    if content_end == 0:
        raise RuntimeError("assistant response contains only whitespace")
    if tokens[content_end - 1] != EOS_ID:
        raise RuntimeError("assistant response must terminate with EOS")
    eos_positions = [index for index, token in enumerate(tokens[:content_end]) if token == EOS_ID]
    if eos_positions != [content_end - 1]:
        raise RuntimeError("EOS must occur exactly once at the response end")

    if tokens.count(OPEN_THINK_ID) != 1 or tokens.count(CLOSE_THINK_ID) != 1:
        raise RuntimeError("response must contain exactly one <think> and one </think>")
    if tokens[0] != OPEN_THINK_ID:
        raise RuntimeError("response must begin with <think>")
    close_index = tokens.index(CLOSE_THINK_ID)
    if close_index <= 0:
        raise RuntimeError("</think> occurs before a valid think span")

    thought = tokens[1:close_index]
    empty_think = not any(token not in WHITESPACE_IDS for token in thought)
    body = close_index + 1
    while body < content_end and tokens[body] in WHITESPACE_IDS:
        body += 1
    if body >= content_end - 1:
        raise RuntimeError("assistant response has no body before EOS")
    return body, content_end, empty_think


def validate_proposal_answer(answer: list[int]) -> str:
    """Validate exactly domain+s_a+s_b+s_c and return the domain name."""

    if len(answer) != 4:
        raise RuntimeError("empty-think itemic row must be exactly domain+s_a+s_b+s_c")
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


def uniformly_capped_positions(
    start: int, end: int, cap: int, device: torch.device
) -> torch.Tensor:
    if not (0 <= start < end):
        raise RuntimeError(f"invalid retention target range: {start}:{end}")
    count = end - start
    if count <= cap:
        return torch.arange(start, end, device=device, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap, device=device).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("uniform retention cap produced duplicate positions")
    return relative + start


def route_and_positions(labels: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Route from response tokens only and select CE/KL target positions."""

    start, end = target_span(labels)
    tokens = labels[0, start:end].detach().cpu().tolist()
    body, content_end, empty_think = response_layout(tokens)
    answer = tokens[body : content_end - 1]  # EOS is validated separately.

    if empty_think:
        looks_itemic = bool(answer) and (
            answer[0] in DOMAIN_IDS
            or (
                len(answer) == 4
                and A_LO <= answer[1] <= A_HI
                and B_LO <= answer[2] <= B_HI
                and C_LO <= answer[3] <= C_HI
            )
        )
        if looks_itemic:
            domain = validate_proposal_answer(answer)
            if domain != EXPECTED_PROPOSAL_DOMAIN:
                raise RuntimeError(
                    "I-28 proposal domain mismatch: "
                    f"expected={EXPECTED_PROPOSAL_DOMAIN} actual={domain}"
                )
            selected = torch.arange(
                start + body,
                start + content_end,
                device=labels.device,
                dtype=torch.long,
            )
            if selected.numel() != 5 or labels[0, selected[-1]].item() != EOS_ID:
                raise RuntimeError("proposal CE mask must be domain+s_a+s_b+s_c+EOS")
            return f"proposal_{domain}", selected

    # Metadata is not consulted here.  Any structurally valid response without
    # the exact empty-think itemic signature is frozen-parent retention.
    relative = uniformly_capped_positions(
        body, content_end, RETENTION_MAX_TOKENS, labels.device
    )
    return "retention", relative + start


def token_ce(policy_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(
            "policy/reference logit shape mismatch: "
            f"{tuple(policy_logits.shape)}/{tuple(reference_logits.shape)}"
        )
    if policy_logits.ndim != 3 or policy_logits.size(0) != 1 or policy_logits.size(1) == 0:
        raise RuntimeError(f"expected non-empty [1,tokens,vocab] logits, got {policy_logits.shape}")
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
    raise RuntimeError(f"expected one active PEFT adapter, got {value!r}")


def assert_frozen_embeddings_and_head(unwrapped) -> None:
    """Fail if any input/output embedding or head parameter is trainable."""

    forbidden_fragments = (
        "embed_tokens",
        "word_embeddings",
        "tok_embeddings",
        "lm_head",
        "output_layer",
    )
    bad_names = [
        name
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad and any(fragment in name for fragment in forbidden_fragments)
    ]
    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(unwrapped, getter_name, None)
        module = getter() if callable(getter) else None
        if module is None:
            continue
        bad_names.extend(
            f"{getter_name}:{name}"
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    if bad_names:
        raise RuntimeError(
            "I-28 requires frozen input embeddings and LM head; trainable="
            + ",".join(sorted(set(bad_names))[:16])
        )


def adapter_parameters(unwrapped, adapter_name: str) -> dict[str, torch.nn.Parameter]:
    """Collect one adapter's parameters under adapter-name-independent keys."""

    marker = f".{adapter_name}."
    result: dict[str, torch.nn.Parameter] = {}
    for name, parameter in unwrapped.named_parameters():
        if marker in name:
            canonical = name.replace(marker, ".__ADAPTER__.", 1)
            if canonical in result:
                raise RuntimeError(
                    f"duplicate canonical adapter parameter for {adapter_name!r}: {canonical}"
                )
            result[canonical] = parameter
    if not result:
        raise RuntimeError(f"no adapter tensors found for {adapter_name!r}")
    return result


def assert_matching_adapter_parameter_sets(
    policy: dict[str, torch.nn.Parameter],
    reference: dict[str, torch.nn.Parameter],
) -> None:
    """Require identical parameter names, shapes, and dtypes before a copy."""

    if policy.keys() != reference.keys():
        missing_ref = sorted(policy.keys() - reference.keys())[:8]
        missing_policy = sorted(reference.keys() - policy.keys())[:8]
        raise RuntimeError(
            "policy/reference adapter tensor sets differ: "
            f"missing_reference={missing_ref} missing_policy={missing_policy}"
        )
    for name in policy:
        left, right = policy[name], reference[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                "policy/reference adapter tensor metadata differ: "
                f"tensor={name} shape={tuple(left.shape)}/{tuple(right.shape)} "
                f"dtype={left.dtype}/{right.dtype}"
            )
        if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
            raise RuntimeError(
                f"policy/reference adapter tensors unexpectedly share storage: {name}"
            )


def copy_policy_adapter_to_reference(unwrapped, policy_name: str) -> int:
    """Copy the post-load, post-upcast policy into the reference tensor by tensor."""

    policy = adapter_parameters(unwrapped, policy_name)
    reference = adapter_parameters(unwrapped, REFERENCE_NAME)
    assert_matching_adapter_parameter_sets(policy, reference)
    non_trainable_policy = sorted(name for name, value in policy.items() if not value.requires_grad)
    if non_trainable_policy:
        raise RuntimeError(
            "policy adapter must remain trainable before reference synchronization: "
            + ",".join(non_trainable_policy[:8])
        )
    with torch.no_grad():
        for name in sorted(policy):
            reference[name].copy_(policy[name])
    return len(policy)


def assert_frozen_reference_adapter(unwrapped) -> None:
    trainable = sorted(
        name
        for name, parameter in adapter_parameters(unwrapped, REFERENCE_NAME).items()
        if parameter.requires_grad
    )
    if trainable:
        raise RuntimeError(
            "I-28 reference adapter must be frozen: " + ",".join(trainable[:8])
        )


def assert_identical_adapter_weights(unwrapped, policy_name: str) -> None:
    """Require bit-identical policy/reference adapter tensors at step zero."""

    policy = adapter_parameters(unwrapped, policy_name)
    reference = adapter_parameters(unwrapped, REFERENCE_NAME)
    assert_matching_adapter_parameter_sets(policy, reference)
    for name in policy:
        left, right = policy[name], reference[name]
        if not torch.equal(left.detach(), right.detach()):
            max_abs = float((left.float() - right.float()).abs().max())
            raise RuntimeError(
                "step-0 policy/reference adapter mismatch: "
                f"tensor={name} shape={tuple(left.shape)}/{tuple(right.shape)} "
                f"dtype={left.dtype}/{right.dtype} max_abs={max_abs:.8g}"
            )


def assert_exact_policy_trainable_parameters(
    unwrapped, policy_name: str
) -> frozenset[int]:
    """Require the global trainable set to be exactly the policy adapter."""

    policy = adapter_parameters(unwrapped, policy_name)
    expected_ids = frozenset(id(parameter) for parameter in policy.values())
    trainable_by_id = {
        id(parameter): name
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    actual_ids = frozenset(trainable_by_id)
    if actual_ids != expected_ids:
        policy_name_by_id = {id(parameter): name for name, parameter in policy.items()}
        unexpected = sorted(trainable_by_id[value] for value in actual_ids - expected_ids)[:8]
        missing = sorted(policy_name_by_id[value] for value in expected_ids - actual_ids)[:8]
        raise RuntimeError(
            "global requires_grad parameter IDs must equal the policy adapter exactly: "
            f"unexpected={unexpected} missing={missing} "
            f"actual={len(actual_ids)} expected={len(expected_ids)}"
        )
    return expected_ids


def assert_optimizer_policy_only(trainer, unwrapped, policy_name: str) -> bool:
    """If an optimizer exists, require its unique parameters to be policy-only."""

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return False
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        raise RuntimeError("I-28 optimizer exposes no stable param_groups list")
    optimizer_parameters = [
        parameter for group in param_groups for parameter in group.get("params", [])
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("I-28 optimizer contains duplicate parameter IDs")

    policy = adapter_parameters(unwrapped, policy_name)
    policy_ids = {id(parameter) for parameter in policy.values()}
    actual_ids = set(optimizer_ids)
    reference_ids: set[int] = set()
    peft_config = getattr(unwrapped, "peft_config", {})
    if REFERENCE_NAME in peft_config:
        reference_ids = {
            id(parameter)
            for parameter in adapter_parameters(unwrapped, REFERENCE_NAME).values()
        }
    reference_overlap = actual_ids & reference_ids
    if actual_ids != policy_ids or reference_overlap:
        names = {id(parameter): name for name, parameter in unwrapped.named_parameters()}
        unexpected = sorted(names.get(value, f"unknown:{value}") for value in actual_ids - policy_ids)[:8]
        missing = sorted(names.get(value, f"unknown:{value}") for value in policy_ids - actual_ids)[:8]
        overlap = sorted(names.get(value, f"unknown:{value}") for value in reference_overlap)[:8]
        raise RuntimeError(
            "I-28 optimizer parameter IDs must equal policy adapter IDs exactly: "
            f"unexpected={unexpected} missing={missing} reference_overlap={overlap}"
        )
    return True


def assert_adapter_runtime_state(unwrapped, expected_name: str, policy_name: str) -> None:
    """Use PEFT's public status API to validate every adapter activation."""

    get_status = getattr(unwrapped, "get_model_status", None)
    if not callable(get_status):
        raise RuntimeError("I-28 requires PEFT get_model_status for adapter-state checks")
    status = get_status()
    if status.enabled is not True:
        raise RuntimeError(f"I-28 adapters are not uniformly enabled: {status.enabled!r}")
    if status.active_adapters != [expected_name]:
        raise RuntimeError(
            "I-28 active adapter mismatch: "
            f"expected={[expected_name]!r} actual={status.active_adapters!r}"
        )
    if status.merged_adapters != []:
        raise RuntimeError(f"I-28 adapters must remain unmerged: {status.merged_adapters!r}")
    expected_available = sorted([policy_name, REFERENCE_NAME])
    if status.available_adapters != expected_available:
        raise RuntimeError(
            "I-28 available adapter set drifted: "
            f"expected={expected_available!r} actual={status.available_adapters!r}"
        )
    expected_requires_grad = {
        policy_name: expected_name == policy_name,
        REFERENCE_NAME: False,
    }
    if status.requires_grad != expected_requires_grad:
        raise RuntimeError(
            "I-28 adapter requires_grad status drifted: "
            f"expected={expected_requires_grad!r} actual={status.requires_grad!r}"
        )


def assert_no_reference_save_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    forbidden = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if any(REFERENCE_NAME in part for part in path.relative_to(output_dir).parts)
    )
    if forbidden:
        raise RuntimeError(
            "I-28 save directory contains frozen-reference artifacts: "
            + ",".join(forbidden[:8])
        )


def assert_policy_only_save_artifacts(output_dir: Path, safe_serialization: bool) -> None:
    """Require a top-level policy adapter and no frozen-reference artifacts."""

    expected_weights = (
        ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME
    )
    required = [output_dir / ADAPTER_CONFIG_NAME, output_dir / expected_weights]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"I-28 policy-only save is incomplete: missing={missing}")
    unexpected_weights = (
        output_dir
        / (ADAPTER_WEIGHTS_NAME if safe_serialization else ADAPTER_SAFE_WEIGHTS_NAME)
    )
    if unexpected_weights.exists():
        raise RuntimeError(f"I-28 save has a stale alternate adapter payload: {unexpected_weights}")
    config_text = (output_dir / ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
    if REFERENCE_NAME in config_text:
        raise RuntimeError("I-28 frozen-reference name leaked into saved adapter config")
    if safe_serialization:
        from safetensors import safe_open

        with safe_open(str(output_dir / expected_weights), framework="pt", device="cpu") as source:
            saved_keys = list(source.keys())
    else:
        saved_state = torch.load(
            output_dir / expected_weights, map_location="cpu", weights_only=True
        )
        saved_keys = list(saved_state)
    if not saved_keys:
        raise RuntimeError("I-28 saved policy adapter has no tensors")
    leaked_keys = sorted(key for key in saved_keys if REFERENCE_NAME in key)
    if leaked_keys:
        raise RuntimeError(
            "I-28 frozen-reference tensors leaked into saved policy adapter: "
            + ",".join(leaked_keys[:8])
        )
    assert_no_reference_save_artifacts(output_dir)


def i28_policy_only_save_pretrained(
    original_save_pretrained,
    peft_model,
    save_directory: str,
    safe_serialization: bool = True,
    selected_adapters=None,
    save_embedding_layers="auto",
    is_main_process: bool = True,
    path_initial_model_for_weight_conversion=None,
    **kwargs,
):
    """Force any dual-I28 PEFT save to serialize only the policy adapter."""

    if REFERENCE_NAME not in getattr(peft_model, "peft_config", {}):
        return original_save_pretrained(
            peft_model,
            save_directory,
            safe_serialization=safe_serialization,
            selected_adapters=selected_adapters,
            save_embedding_layers=save_embedding_layers,
            is_main_process=is_main_process,
            path_initial_model_for_weight_conversion=path_initial_model_for_weight_conversion,
            **kwargs,
        )
    if not is_main_process:
        raise RuntimeError("I-28 policy-only save must run on the main process")
    policy_names = sorted(
        name for name in peft_model.peft_config if name != REFERENCE_NAME
    )
    if policy_names != ["default"]:
        raise RuntimeError(
            f"I-28 save requires one top-level default policy adapter: {policy_names!r}"
        )
    policy_name = policy_names[0]
    if selected_adapters not in (None, [policy_name]):
        raise RuntimeError(
            "I-28 caller requested an unsafe adapter save selection: "
            f"{selected_adapters!r}"
        )
    assert_adapter_runtime_state(peft_model, policy_name, policy_name)
    assert_exact_policy_trainable_parameters(peft_model, policy_name)

    output_dir = Path(save_directory)
    assert_no_reference_save_artifacts(output_dir)
    stale_top_level = [
        str(output_dir / name)
        for name in (ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME)
        if (output_dir / name).exists()
    ]
    if stale_top_level:
        raise RuntimeError(
            "I-28 refuses to overwrite an existing top-level adapter payload: "
            + ",".join(stale_top_level)
        )

    result = original_save_pretrained(
        peft_model,
        save_directory,
        safe_serialization=safe_serialization,
        selected_adapters=[policy_name],
        save_embedding_layers=save_embedding_layers,
        is_main_process=is_main_process,
        path_initial_model_for_weight_conversion=path_initial_model_for_weight_conversion,
        **kwargs,
    )
    assert_policy_only_save_artifacts(output_dir, safe_serialization)
    assert_adapter_runtime_state(peft_model, policy_name, policy_name)
    return result


def assert_formal_trainer_args(trainer) -> None:
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
        raise RuntimeError(f"I-28 formal trainer-argument contract drifted: {drift}")
    report_to = getattr(args, "report_to", [])
    if isinstance(report_to, str):
        report_to = [report_to]
    if "wandb" not in report_to:
        raise RuntimeError(f"I-28 formal training requires W&B reporting: {report_to!r}")


def ensure_reference(self, model) -> tuple[object, str]:
    state = getattr(self, "_i28_reference_state", None)
    unwrapped = self.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state["policy_name"]

    if int(getattr(self.state, "global_step", -1)) != 0:
        raise RuntimeError("I-28 reference must be initialized at global_step=0; resume is forbidden")
    assert_formal_trainer_args(self)
    policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name == REFERENCE_NAME:
        raise RuntimeError("policy adapter name collides with I-28 reference name")
    reference_path = Path(REFERENCE_ADAPTER).resolve()
    if not reference_path.is_dir():
        raise RuntimeError(f"missing frozen I-23 reference adapter: {reference_path}")

    assert_frozen_embeddings_and_head(unwrapped)
    initial_policy_parameter_ids = assert_exact_policy_trainable_parameters(
        unwrapped, policy_name
    )
    initial_trainable = sum(parameter.numel() for parameter in unwrapped.parameters() if parameter.requires_grad)
    device = next(unwrapped.parameters()).device
    unwrapped.load_adapter(
        str(reference_path),
        adapter_name=REFERENCE_NAME,
        is_trainable=False,
        torch_device=str(device),
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=True,
    )
    unwrapped.set_requires_grad(policy_name, True)
    copied_tensors = copy_policy_adapter_to_reference(unwrapped, policy_name)
    unwrapped.set_requires_grad(REFERENCE_NAME, False)
    set_active_adapter(unwrapped, policy_name, policy_name)
    assert_frozen_embeddings_and_head(unwrapped)
    final_policy_parameter_ids = assert_exact_policy_trainable_parameters(
        unwrapped, policy_name
    )
    if final_policy_parameter_ids != initial_policy_parameter_ids:
        raise RuntimeError("I-28 policy parameter IDs changed while loading the reference")
    optimizer_checked = assert_optimizer_policy_only(self, unwrapped, policy_name)
    optimizer_policy_ids = getattr(self, "_i28_optimizer_policy_ids", None)
    if optimizer_policy_ids is not None and optimizer_policy_ids != final_policy_parameter_ids:
        raise RuntimeError("I-28 optimizer-time policy parameter IDs changed before step zero")
    final_trainable = sum(parameter.numel() for parameter in unwrapped.parameters() if parameter.requires_grad)
    if final_trainable != initial_trainable:
        raise RuntimeError(
            "trainable parameter count changed after reference load: "
            f"{initial_trainable}/{final_trainable}"
        )
    assert_identical_adapter_weights(unwrapped, policy_name)

    self._i28_reference_state = {
        "policy_name": policy_name,
        "policy_parameter_ids": final_policy_parameter_ids,
        "initial_trainable": initial_trainable,
        "optimizer_checked": optimizer_checked,
        "logit_fingerprint_checked": False,
    }
    print(
        "[i28] step-0 adapter fingerprint PASS; frozen reference synchronized: "
        f"path={reference_path} policy={policy_name} reference={REFERENCE_NAME} "
        f"copied_from_post_upcast_policy={copied_tensors} "
        f"trainable={initial_trainable:,} optimizer_checked={optimizer_checked}; "
        "embeddings/head frozen",
        flush=True,
    )
    return unwrapped, policy_name


def set_active_adapter(unwrapped, name: str, policy_name: str) -> None:
    unwrapped.set_adapter(name)
    unwrapped.set_requires_grad(policy_name, name == policy_name)
    unwrapped.set_requires_grad(REFERENCE_NAME, False)
    assert_adapter_runtime_state(unwrapped, name, policy_name)
    assert_frozen_reference_adapter(unwrapped)
    assert_frozen_embeddings_and_head(unwrapped)


def paired_forward(self, model, inputs: dict[str, torch.Tensor], prediction_positions: torch.Tensor):
    unwrapped, policy_name = ensure_reference(self, model)
    model.train()

    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        set_active_adapter(unwrapped, REFERENCE_NAME, policy_name)
        with torch.no_grad():
            reference_logits = model(
                **inputs, logits_to_keep=prediction_positions
            ).logits.detach()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        set_active_adapter(unwrapped, policy_name, policy_name)

    outputs = model(**inputs, logits_to_keep=prediction_positions)
    state = self._i28_reference_state
    if not state["logit_fingerprint_checked"]:
        if int(getattr(self.state, "global_step", -1)) != 0:
            raise RuntimeError("initial policy/reference logit fingerprint occurred after step zero")
        max_abs = float(
            (outputs.logits.detach().float() - reference_logits.float()).abs().max()
        )
        if max_abs > 1e-4:
            raise RuntimeError(f"initial policy/reference logits differ: max_abs={max_abs:.8f}")
        state["logit_fingerprint_checked"] = True
        print(
            f"[i28] step-0 policy/reference logit fingerprint PASS: max_abs={max_abs:.8f}",
            flush=True,
        )
    return outputs, reference_logits


def record_route_or_fail(route: str) -> tuple[int, Counter]:
    """Record the formal 512-row route contract before doing a forward pass."""

    count = getattr(i28_loss, "call_count", 0) + 1
    if count > EXPECTED_MICROBATCHES:
        raise RuntimeError(
            f"I-28 exceeded {EXPECTED_MICROBATCHES} formal microbatches: {count}"
        )
    counts = Counter(getattr(i28_loss, "route_counts", Counter()))
    counts[route] += 1
    proposal_count = sum(
        value for name, value in counts.items() if name.startswith("proposal_")
    )
    retention_count = counts["retention"]
    unexpected = {
        name: value
        for name, value in counts.items()
        if name not in {f"proposal_{EXPECTED_PROPOSAL_DOMAIN}", "retention"}
    }
    if unexpected:
        raise RuntimeError(f"I-28 observed unexpected loss routes: {unexpected}")
    if proposal_count > EXPECTED_PROPOSALS or retention_count > EXPECTED_RETENTIONS:
        raise RuntimeError(
            "I-28 route count exceeded contract before epoch end: "
            f"proposal={proposal_count}/{EXPECTED_PROPOSALS} "
            f"retention={retention_count}/{EXPECTED_RETENTIONS}"
        )
    remaining = EXPECTED_MICROBATCHES - count
    if (
        proposal_count + remaining < EXPECTED_PROPOSALS
        or retention_count + remaining < EXPECTED_RETENTIONS
    ):
        raise RuntimeError(
            "I-28 remaining rows cannot satisfy the route contract: "
            f"seen={count} proposal={proposal_count} retention={retention_count} "
            f"remaining={remaining}"
        )
    if count == EXPECTED_MICROBATCHES and (
        proposal_count != EXPECTED_PROPOSALS or retention_count != EXPECTED_RETENTIONS
    ):
        raise RuntimeError(
            "I-28 final route contract mismatch: "
            f"proposal={proposal_count}/{EXPECTED_PROPOSALS} "
            f"retention={retention_count}/{EXPECTED_RETENTIONS}"
        )
    i28_loss.call_count = count
    i28_loss.route_counts = counts
    return count, counts


def assert_final_route_contract() -> None:
    count = getattr(i28_loss, "call_count", 0)
    counts = Counter(getattr(i28_loss, "route_counts", Counter()))
    proposal_count = sum(
        value for name, value in counts.items() if name.startswith("proposal_")
    )
    retention_count = counts["retention"]
    if (
        count != EXPECTED_MICROBATCHES
        or proposal_count != EXPECTED_PROPOSALS
        or retention_count != EXPECTED_RETENTIONS
    ):
        raise RuntimeError(
            "I-28 training ended before satisfying the exact route contract: "
            f"microbatches={count}/{EXPECTED_MICROBATCHES} "
            f"proposal={proposal_count}/{EXPECTED_PROPOSALS} "
            f"retention={retention_count}/{EXPECTED_RETENTIONS}"
        )
    print(
        "[i28] final route contract PASS: "
        f"microbatches={count} proposal_{EXPECTED_PROPOSAL_DOMAIN}={proposal_count} "
        f"retention={retention_count}",
        flush=True,
    )


def i28_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    route, selected_positions = route_and_positions(labels)
    count, counts = record_route_or_fail(route)
    prediction_positions = selected_positions - 1
    targets = labels[0, selected_positions]

    outputs, reference_logits = paired_forward(self, model, inputs, prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != targets.numel():
        raise RuntimeError(
            f"partial logits/targets mismatch: {policy_logits.size(1)}/{targets.numel()}"
        )
    kl = forward_kl(policy_logits, reference_logits)
    if route.startswith("proposal_"):
        if targets.numel() != 5 or targets[-1].item() != EOS_ID:
            raise RuntimeError("proposal loss received a non-five-token target")
        ce = token_ce(policy_logits, targets)
        loss = ce + PROPOSAL_KL_WEIGHT * kl
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_KL_WEIGHT * kl

    if count <= 12 or count % 100 == 0:
        print(
            "[i28] "
            f"microbatch={count} route={route} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} kl={float(kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def _expect_runtime_error(function, text: str) -> None:
    try:
        function()
    except RuntimeError as error:
        if text not in str(error):
            raise AssertionError(f"unexpected error: {error}") from error
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def run_self_test() -> None:
    validate_hyperparameters()
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    empty_think = [OPEN_THINK_ID, 198, CLOSE_THINK_ID, 198]

    for domain_id, domain in DOMAIN_IDS.items():
        response = empty_think + [domain_id, A_LO, B_LO, C_LO, EOS_ID, 198]
        labels = torch.tensor([prefix + response], dtype=torch.long)
        if domain == EXPECTED_PROPOSAL_DOMAIN:
            route, positions = route_and_positions(labels)
            assert route == f"proposal_{domain}"
            assert labels[0, positions].tolist() == [domain_id, A_LO, B_LO, C_LO, EOS_ID]
        else:
            _expect_runtime_error(
                lambda labels=labels: route_and_positions(labels), "proposal domain mismatch"
            )

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

    cot_retention = torch.tensor(
        [prefix + [OPEN_THINK_ID, 700, CLOSE_THINK_ID, 198, 176245, A_LO, B_LO, C_LO, EOS_ID]],
        dtype=torch.long,
    )
    route, positions = route_and_positions(cot_retention)
    assert route == "retention" and positions.numel() == 5

    long_retention = torch.tensor(
        [prefix + empty_think + [1_000] * (RETENTION_MAX_TOKENS + 17) + [EOS_ID]],
        dtype=torch.long,
    )
    route, positions = route_and_positions(long_retention)
    assert route == "retention" and positions.numel() == RETENTION_MAX_TOKENS
    assert positions.unique().numel() == RETENTION_MAX_TOKENS

    malformed = torch.tensor(
        [prefix + empty_think + [176245, A_LO, B_LO, C_LO, 42, EOS_ID]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(malformed), "exactly domain+s_a+s_b+s_c")
    bad_code = torch.tensor(
        [prefix + empty_think + [176245, A_LO, B_LO, C_HI + 1, EOS_ID]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(bad_code), "broken video s_c token")
    missing_eos = torch.tensor(
        [prefix + empty_think + [176245, A_LO, B_LO, C_LO]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(missing_eos), "terminate with EOS")
    missing_think = torch.tensor(
        [prefix + [176245, A_LO, B_LO, C_LO, EOS_ID]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(missing_think), "exactly one <think>")
    packed = torch.tensor(
        [[IGNORE_INDEX, 10, 11, IGNORE_INDEX, 12, 13]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(packed), "packing must be disabled")
    contiguous_packed = torch.tensor(
        [
            prefix
            + empty_think
            + [176245, A_LO, B_LO, C_LO, EOS_ID]
            + empty_think
            + [176245, A_LO, B_LO, C_LO, EOS_ID]
        ],
        dtype=torch.long,
    )
    _expect_runtime_error(
        lambda: route_and_positions(contiguous_packed), "EOS must occur exactly once"
    )
    batch_two = torch.tensor(
        [[IGNORE_INDEX, 10], [IGNORE_INDEX, 10]], dtype=torch.long
    )
    _expect_runtime_error(lambda: route_and_positions(batch_two), "batch_size=1")

    torch.manual_seed(17)
    ce_logits = torch.randn(1, 5, 31, requires_grad=True)
    ce_targets = torch.randint(0, 31, (5,))
    direct_ce = F.cross_entropy(ce_logits[0].float(), ce_targets)
    chunked_ce = token_ce(ce_logits, ce_targets)
    assert torch.allclose(direct_ce, chunked_ce, atol=1e-6)

    policy = torch.randn(1, 11, 31, requires_grad=True)
    reference = torch.randn(1, 11, 31)
    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(reference.float(), dim=-1),
        reduction="sum",
    ) / 11
    chunked_kl = forward_kl(policy, reference)
    assert torch.allclose(direct_kl, chunked_kl, atol=1e-6)
    (chunked_ce + chunked_kl).backward()
    assert ce_logits.grad is not None and torch.isfinite(ce_logits.grad).all()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()

    class FakePeftModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(5, 3)
            self.lm_head = torch.nn.Linear(3, 5, bias=False)
            self.lora_A = torch.nn.ModuleDict(
                {
                    "default": torch.nn.Linear(3, 2, bias=False),
                    REFERENCE_NAME: torch.nn.Linear(3, 2, bias=False),
                }
            )
            self.peft_config = {"default": object(), REFERENCE_NAME: object()}
            self.active_adapter = "default"
            self.adapters_enabled = True
            self.merged_adapters: list[str] = []
            self.embed_tokens.requires_grad_(False)
            self.lm_head.requires_grad_(False)
            self.lora_A[REFERENCE_NAME].load_state_dict(
                self.lora_A["default"].state_dict()
            )
            self.lora_A[REFERENCE_NAME].requires_grad_(True)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def set_adapter(self, adapter_name: str) -> None:
            if adapter_name not in self.lora_A:
                raise RuntimeError(f"missing fake adapter: {adapter_name}")
            self.active_adapter = adapter_name
            for name, module in self.lora_A.items():
                module.requires_grad_(name == adapter_name)

        def set_requires_grad(self, adapter_name: str, value: bool) -> None:
            self.lora_A[adapter_name].requires_grad_(value)

        def get_model_status(self):
            class FakeStatus:
                pass

            status = FakeStatus()
            status.enabled = self.adapters_enabled
            status.active_adapters = [self.active_adapter]
            status.merged_adapters = list(self.merged_adapters)
            status.available_adapters = sorted(self.peft_config)
            status.requires_grad = {
                name: all(parameter.requires_grad for parameter in module.parameters())
                for name, module in self.lora_A.items()
            }
            return status

    fake = FakePeftModel()
    assert_frozen_embeddings_and_head(fake)
    assert_identical_adapter_weights(fake, "default")
    with torch.no_grad():
        fake.lora_A[REFERENCE_NAME].weight[0, 0].add_(1.0)
    _expect_runtime_error(
        lambda: assert_identical_adapter_weights(fake, "default"),
        "adapter mismatch",
    )
    policy_before_sync = fake.lora_A["default"].weight.detach().clone()
    expected_trainable_after_freeze = fake.lora_A["default"].weight.numel()
    copied_tensors = copy_policy_adapter_to_reference(fake, "default")
    assert copied_tensors == 1
    fake.lora_A[REFERENCE_NAME].requires_grad_(False)
    set_active_adapter(fake, "default", "default")
    assert_frozen_reference_adapter(fake)
    assert fake.lora_A["default"].weight.requires_grad
    assert sum(parameter.numel() for parameter in fake.parameters() if parameter.requires_grad) == (
        expected_trainable_after_freeze
    )
    assert torch.equal(fake.lora_A["default"].weight.detach(), policy_before_sync)
    assert_identical_adapter_weights(fake, "default")
    policy_parameter_ids = assert_exact_policy_trainable_parameters(fake, "default")
    assert policy_parameter_ids == {id(fake.lora_A["default"].weight)}

    class FakeOptimizerTrainer:
        pass

    fake_optimizer_trainer = FakeOptimizerTrainer()
    fake_optimizer_trainer.optimizer = torch.optim.AdamW(
        fake.lora_A["default"].parameters(), lr=1e-4
    )
    assert assert_optimizer_policy_only(fake_optimizer_trainer, fake, "default")
    bad_optimizer_trainer = FakeOptimizerTrainer()
    bad_optimizer_trainer.optimizer = torch.optim.AdamW(
        [fake.lora_A["default"].weight, fake.lora_A[REFERENCE_NAME].weight],
        lr=1e-4,
    )
    _expect_runtime_error(
        lambda: assert_optimizer_policy_only(bad_optimizer_trainer, fake, "default"),
        "reference_overlap",
    )

    set_active_adapter(fake, REFERENCE_NAME, "default")
    assert not any(parameter.requires_grad for parameter in fake.parameters())
    set_active_adapter(fake, "default", "default")
    fake.adapters_enabled = False
    _expect_runtime_error(
        lambda: assert_adapter_runtime_state(fake, "default", "default"),
        "not uniformly enabled",
    )
    fake.adapters_enabled = True
    fake.merged_adapters = ["default"]
    _expect_runtime_error(
        lambda: assert_adapter_runtime_state(fake, "default", "default"),
        "must remain unmerged",
    )
    fake.merged_adapters = []
    assert_adapter_runtime_state(fake, "default", "default")

    dtype_fake = FakePeftModel()
    dtype_fake.lora_A[REFERENCE_NAME].weight.data = (
        dtype_fake.lora_A[REFERENCE_NAME].weight.data.to(torch.bfloat16)
    )
    _expect_runtime_error(
        lambda: copy_policy_adapter_to_reference(dtype_fake, "default"),
        "tensor metadata differ",
    )

    import tempfile

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    tiny_config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=16,
        n_ctx=16,
        vocab_size=32,
    )
    tiny_policy_config = LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    tiny_peft = get_peft_model(GPT2LMHeadModel(tiny_config), tiny_policy_config)
    tiny_peft.add_adapter(
        REFERENCE_NAME,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        ),
    )
    tiny_peft.set_requires_grad("default", True)
    copy_policy_adapter_to_reference(tiny_peft, "default")
    tiny_peft.set_requires_grad(REFERENCE_NAME, False)
    set_active_adapter(tiny_peft, "default", "default")
    tiny_policy_ids = assert_exact_policy_trainable_parameters(tiny_peft, "default")
    tiny_trainer = FakeOptimizerTrainer()
    tiny_trainer.optimizer = torch.optim.AdamW(
        adapter_parameters(tiny_peft, "default").values(), lr=1e-4
    )
    assert assert_optimizer_policy_only(tiny_trainer, tiny_peft, "default")

    with tempfile.TemporaryDirectory(prefix="i28_policy_only_save_") as temp_dir:
        save_root = Path(temp_dir)
        checkpoint_dir = save_root / "checkpoint-64"
        original_cpu_peft_save = PeftModel.save_pretrained

        def patched_cpu_peft_save(peft_model, save_directory, *args, **kwargs):
            return i28_policy_only_save_pretrained(
                original_cpu_peft_save,
                peft_model,
                save_directory,
                *args,
                **kwargs,
            )

        PeftModel.save_pretrained = patched_cpu_peft_save
        try:
            tiny_peft.save_pretrained(str(checkpoint_dir), safe_serialization=True)
        finally:
            PeftModel.save_pretrained = original_cpu_peft_save
        assert_policy_only_save_artifacts(checkpoint_dir, True)
        i28_policy_only_save_pretrained(
            PeftModel.save_pretrained,
            tiny_peft,
            str(save_root),
            safe_serialization=True,
        )
        assert_policy_only_save_artifacts(save_root, True)
        assert_no_reference_save_artifacts(save_root)
        _expect_runtime_error(
            lambda: i28_policy_only_save_pretrained(
                PeftModel.save_pretrained,
                tiny_peft,
                str(save_root / "unsafe"),
                selected_adapters=[REFERENCE_NAME],
            ),
            "unsafe adapter save selection",
        )
    assert tiny_policy_ids == assert_exact_policy_trainable_parameters(
        tiny_peft, "default"
    )

    fake.lm_head.weight.requires_grad_(True)
    _expect_runtime_error(
        lambda: assert_exact_policy_trainable_parameters(fake, "default"),
        "requires_grad parameter IDs",
    )
    _expect_runtime_error(
        lambda: assert_frozen_embeddings_and_head(fake), "frozen input embeddings"
    )

    class FakeArgs:
        per_device_train_batch_size = 1
        gradient_accumulation_steps = EXPECTED_GRADIENT_ACCUMULATION
        max_steps = EXPECTED_OPTIMIZER_STEPS
        world_size = 1
        report_to = ["wandb"]

    class FakeTrainer:
        args = FakeArgs()

    assert_formal_trainer_args(FakeTrainer())
    FakeTrainer.args.per_device_train_batch_size = 2
    _expect_runtime_error(
        lambda: assert_formal_trainer_args(FakeTrainer()), "argument contract drifted"
    )
    FakeTrainer.args.per_device_train_batch_size = 1

    old_calls = getattr(i28_loss, "call_count", None)
    old_counts = getattr(i28_loss, "route_counts", None)
    try:
        i28_loss.call_count = EXPECTED_MICROBATCHES - 1
        i28_loss.route_counts = Counter(
            {
                f"proposal_{EXPECTED_PROPOSAL_DOMAIN}": EXPECTED_PROPOSALS,
                "retention": EXPECTED_RETENTIONS - 1,
            }
        )
        count, counts = record_route_or_fail("retention")
        assert count == EXPECTED_MICROBATCHES
        assert counts[f"proposal_{EXPECTED_PROPOSAL_DOMAIN}"] == EXPECTED_PROPOSALS
        assert_final_route_contract()
        _expect_runtime_error(
            lambda: record_route_or_fail("retention"), "exceeded"
        )

        i28_loss.call_count = EXPECTED_MICROBATCHES - 1
        i28_loss.route_counts = Counter(
            {
                f"proposal_{EXPECTED_PROPOSAL_DOMAIN}": EXPECTED_PROPOSALS - 1,
                "retention": EXPECTED_RETENTIONS,
            }
        )
        _expect_runtime_error(
            lambda: record_route_or_fail("retention"), "exceeded contract"
        )
        i28_loss.call_count = EXPECTED_MICROBATCHES - 1
        i28_loss.route_counts = Counter(
            {
                f"proposal_{EXPECTED_PROPOSAL_DOMAIN}": EXPECTED_PROPOSALS - 1,
                "retention": EXPECTED_RETENTIONS,
            }
        )
        _expect_runtime_error(
            assert_final_route_contract, "ended before satisfying"
        )
    finally:
        if old_calls is None:
            if hasattr(i28_loss, "call_count"):
                delattr(i28_loss, "call_count")
        else:
            i28_loss.call_count = old_calls
        if old_counts is None:
            if hasattr(i28_loss, "route_counts"):
                delattr(i28_loss, "route_counts")
        else:
            i28_loss.route_counts = old_counts

    print(
        "[i28] self-test passed: response-only four-domain routing, exact "
        "domain+s_a+s_b+s_c+EOS CE mask, retention-only capped KL, malformed/packed/"
        "batch-two fail-closed checks, strict policy-to-reference tensor synchronization/freeze, "
        "active/enabled/unmerged and optimizer-ID gates, CPU PEFT policy-only checkpoint/root "
        "save gate, exact final route-count gate, and chunked CE/KL gradients"
    )


def main() -> None:
    validate_hyperparameters()
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from peft import PeftModel
    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss
    original_create_optimizer = sft_trainer.CustomSeq2SeqTrainer.create_optimizer
    original_trainer_save = sft_trainer.CustomSeq2SeqTrainer._save
    original_peft_save_pretrained = PeftModel.save_pretrained

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        if args:
            if len(args) != 1 or "return_outputs" in kwargs:
                raise RuntimeError(
                    "unexpected positional CustomSeq2SeqTrainer.compute_loss arguments"
                )
            kwargs["return_outputs"] = args[0]
        return i28_loss(self, model, inputs, **kwargs)

    def patched_create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        unwrapped = self.accelerator.unwrap_model(self.model)
        policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
        policy_parameter_ids = assert_exact_policy_trainable_parameters(
            unwrapped, policy_name
        )
        if not assert_optimizer_policy_only(self, unwrapped, policy_name):
            raise RuntimeError("I-28 optimizer was not available after create_optimizer")
        self._i28_optimizer_policy_ids = policy_parameter_ids
        print(
            "[i28] optimizer policy-only parameter-ID gate PASS: "
            f"policy={policy_name} tensors={len(policy_parameter_ids)}",
            flush=True,
        )
        return optimizer

    def patched_trainer_save(self, output_dir=None, state_dict=None):
        state = getattr(self, "_i28_reference_state", None)
        if state is None:
            raise RuntimeError("I-28 refuses to save before frozen-reference initialization")
        unwrapped = self.accelerator.unwrap_model(self.model)
        policy_name = state["policy_name"]
        set_active_adapter(unwrapped, policy_name, policy_name)
        policy_parameter_ids = assert_exact_policy_trainable_parameters(
            unwrapped, policy_name
        )
        if policy_parameter_ids != state["policy_parameter_ids"]:
            raise RuntimeError("I-28 policy parameter IDs changed before save")
        if not assert_optimizer_policy_only(self, unwrapped, policy_name):
            raise RuntimeError("I-28 optimizer disappeared before save")
        result = original_trainer_save(self, output_dir=output_dir, state_dict=state_dict)
        saved_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        safe_serialization = bool(getattr(self.args, "save_safetensors", True))
        assert_policy_only_save_artifacts(saved_dir, safe_serialization)
        print(
            "[i28] policy-only checkpoint save PASS: "
            f"path={saved_dir} policy={policy_name} reference_artifacts=0",
            flush=True,
        )
        return result

    def patched_peft_save_pretrained(
        peft_model,
        save_directory,
        safe_serialization=True,
        selected_adapters=None,
        save_embedding_layers="auto",
        is_main_process=True,
        path_initial_model_for_weight_conversion=None,
        **kwargs,
    ):
        return i28_policy_only_save_pretrained(
            original_peft_save_pretrained,
            peft_model,
            save_directory,
            safe_serialization=safe_serialization,
            selected_adapters=selected_adapters,
            save_embedding_layers=save_embedding_layers,
            is_main_process=is_main_process,
            path_initial_model_for_weight_conversion=path_initial_model_for_weight_conversion,
            **kwargs,
        )

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss
    sft_trainer.CustomSeq2SeqTrainer.create_optimizer = patched_create_optimizer
    sft_trainer.CustomSeq2SeqTrainer._save = patched_trainer_save
    PeftModel.save_pretrained = patched_peft_save_pretrained
    from llamafactory.train.tuner import run_exp

    run_exp()
    assert_final_route_contract()


if __name__ == "__main__":
    main()
