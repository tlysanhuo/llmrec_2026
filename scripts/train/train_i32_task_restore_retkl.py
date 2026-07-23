#!/usr/bin/env python3
"""Train a fresh r8 task-restoration residual over the I31 lambda-0.10 start.

Material answer bodies receive only frozen-I23 KL. Seven-task retention rows
receive only frozen-r96 KL. LLaMA-Factory merges the exact r160 interpolation
before creating the fresh r8, which is the only trainable component.
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
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
START_ADAPTER = ROOT / "checkpoints/i31_r96_i23_exact_interp_r160_l010"
PARENT_ADAPTER = ROOT / "submissions/i19_world_external_r96_s875_platform"
TEACHER_ADAPTER = ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
TRAINING_DATA = ROOT / "assets/derived/processed/data_i30_r96_material_teacher_retkl_v1.jsonl"
START_SHA256 = "5ad2e7898f5ac52d286e634f2d38067b59a45c6b0d6be7d61d2af54a9ebde64c"
PARENT_SHA256 = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
TEACHER_SHA256 = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"

IGNORE_INDEX = -100
CLOSE_THINK_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}
ACTION_START_IDS = {58, 1183}
TOPIC_START_IDS = {90, 4913}  # `{` and `{"`
DOMAIN_IDS = {176245, 176247, 176249, 176251}
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244
RECOMMENDATION_PREFIX_IDS = [75882, 20002, 104044]
WORLD_PREFIX_IDS = [1406, 88991, 102349, 20412, 320]

MATERIAL_TEACHER_KL = float(os.environ.get("I32_MATERIAL_TEACHER_KL", "4.0"))
RETENTION_PARENT_KL = float(os.environ.get("I32_RETENTION_PARENT_KL", "8.0"))
RETENTION_MAX_POSITIONS = int(os.environ.get("I32_RETENTION_MAX_POSITIONS", "96"))
LOGIT_CHUNK = int(os.environ.get("I32_LOGIT_CHUNK", "8"))

EXPECTED_ROWS = 2048
EXPECTED_ROUTES = {"material": 512, "retention": 1536}
EXPECTED_RANK_ALPHA = (8, 8)
EXPECTED_TRAINER = {"batch": 1, "accum": 4, "max_steps": 512, "world_size": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I-32 requires per-device batch size 1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("I-32 batch has no response targets")
    start, end = int(positions[0]), int(positions[-1]) + 1
    if start == 0:
        raise RuntimeError("I-32 response starts at token zero")
    expected = torch.arange(start, end, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("I-32 forbids packing and disjoint response spans")
    return start, end


def body_bounds(tokens: list[int]) -> tuple[int, int] | None:
    try:
        start = tokens.index(CLOSE_THINK_ID) + 1
    except ValueError:
        return None
    while start < len(tokens) and tokens[start] in WHITESPACE_IDS:
        start += 1
    end = len(tokens)
    while end > start and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    if start >= end or tokens[end - 1] != EOS_ID:
        raise RuntimeError("I-32 response body is empty or missing EOS")
    return start, end


def contains_prefix(tokens: list[int], prefix: list[int], start: int, end: int) -> bool:
    return any(tokens[index : index + len(prefix)] == prefix for index in range(start, end - len(prefix) + 1))


def has_itemic(tokens: list[int]) -> bool:
    return any(
        tokens[index] in DOMAIN_IDS
        and A_LO <= tokens[index + 1] <= A_HI
        and B_LO <= tokens[index + 2] <= B_HI
        and C_LO <= tokens[index + 3] <= C_HI
        for index in range(0, len(tokens) - 3)
    )


def route_response(targets: torch.Tensor, prompt_tokens: list[int]) -> tuple[str, int, int]:
    tokens = targets.detach().cpu().tolist()
    bounds = body_bounds(tokens)
    if bounds is None:
        end = len(tokens)
        while end > 0 and tokens[end - 1] in WHITESPACE_IDS:
            end -= 1
        if end <= 0:
            raise RuntimeError("I-32 legacy retention response is empty")
        return "retention", 0, end
    start, end = bounds
    body = tokens[start:end]
    if len(body) >= 4 and (
        body[0] in DOMAIN_IDS
        and A_LO <= body[1] <= A_HI
        and B_LO <= body[2] <= B_HI
        and C_LO <= body[3] <= C_HI
    ):
        return "material", start, end
    if body[0] in ACTION_START_IDS:
        return "retention", start, end
    if body[0] in TOPIC_START_IDS:
        return "retention", start, end
    if contains_prefix(tokens, RECOMMENDATION_PREFIX_IDS, start, end):
        return "retention", start, end
    if body[: len(WORLD_PREFIX_IDS)] == WORLD_PREFIX_IDS:
        return "retention", start, end
    if has_itemic(prompt_tokens):
        return "material", start, end
    return "retention", start, end


def capped_positions(start: int, end: int, cap: int, device: torch.device) -> torch.Tensor:
    count = end - start
    if count <= 0:
        raise RuntimeError("I-32 cannot select positions from an empty body")
    if count <= cap:
        return torch.arange(start, end, device=device, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap, device=device).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("I-32 retention position cap produced duplicates")
    return relative + start


def token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != (1, targets.numel()):
        raise RuntimeError(f"I-32 CE shape mismatch: {tuple(logits.shape)}/{targets.shape}")
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        total += F.cross_entropy(logits[0, start:end].float(), targets[start:end], reduction="sum")
    return total / targets.numel()


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(f"I-32 KL shape mismatch: {policy_logits.shape}/{reference_logits.shape}")
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, policy_logits.size(1))
        total += F.kl_div(
            F.log_softmax(policy_logits[:, start:end].float(), dim=-1),
            F.softmax(reference_logits[:, start:end].float(), dim=-1),
            reduction="sum",
        )
    return total / policy_logits.size(1)


def single_adapter_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError(f"I-32 expected one active residual adapter, got {value!r}")


def ensure_models(trainer: Any, model: Any):
    state = getattr(trainer, "_i32_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state

    args = trainer.args
    observed = {
        "batch": int(args.per_device_train_batch_size),
        "accum": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
    }
    if observed != EXPECTED_TRAINER:
        raise RuntimeError(f"I-32 trainer contract drifted: {observed}/{EXPECTED_TRAINER}")
    report_to = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in report_to:
        raise RuntimeError(f"I-32 requires W&B online reporting: {report_to!r}")

    peft_config = getattr(unwrapped, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        raise RuntimeError("I-32 expected exactly one fresh residual after merging r160 start")
    adapter_name = single_adapter_name(getattr(unwrapped, "active_adapter", None))
    config = peft_config[adapter_name]
    rank_alpha = (int(getattr(config, "r", -1)), int(getattr(config, "lora_alpha", -1)))
    if rank_alpha != EXPECTED_RANK_ALPHA:
        raise RuntimeError(f"I-32 expected r8/alpha8, got {rank_alpha}")
    trainable = [name for name, parameter in unwrapped.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("I-32 trainable set is empty or contains non-LoRA parameters")
    if sha256(START_ADAPTER / "adapter_model.safetensors") != START_SHA256:
        raise RuntimeError("I-32 lambda-0.10 start hash drifted")
    if sha256(PARENT_ADAPTER / "adapter_model.safetensors") != PARENT_SHA256:
        raise RuntimeError("I-32 frozen r96 parent hash drifted")
    if sha256(TEACHER_ADAPTER / "adapter_model.safetensors") != TEACHER_SHA256:
        raise RuntimeError("I-32 frozen I-23 teacher hash drifted")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = next(unwrapped.parameters()).device
    def load_reference(adapter_path: Path, adapter_name: str):
        reference_base = AutoModelForCausalLM.from_pretrained(
            BASE,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        reference = PeftModel.from_pretrained(
            reference_base,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
        ).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    parent = load_reference(PARENT_ADAPTER, "r96_parent")
    teacher = load_reference(TEACHER_ADAPTER, "i23_teacher")
    state = {
        "adapter_name": adapter_name,
        "parent": parent,
        "teacher": teacher,
        "fingerprint_checked": False,
    }
    trainer._i32_state = state
    print(
        f"[i32] contract PASS: merged lambda-0.10 r160 + fresh r8/alpha8; "
        f"trainable_tensors={len(trainable)}; frozen r96 and I-23 references loaded",
        flush=True,
    )
    return unwrapped, state


def policy_with_start_fingerprint(trainer, model, inputs, positions):
    unwrapped, state = ensure_models(trainer, model)
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    with torch.no_grad(), unwrapped.disable_adapter():
        start_logits = model(**inputs, logits_to_keep=positions).logits.detach()
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    outputs = model(**inputs, logits_to_keep=positions)
    if not state["fingerprint_checked"]:
        max_abs = float((outputs.logits.detach().float() - start_logits.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"I-32 step-0 lambda-0.10 fingerprint failed: {max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i32] step-0 lambda-0.10 fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return outputs, state


def record_route(route: str) -> tuple[int, Counter[str]]:
    count = getattr(i32_loss, "call_count", 0) + 1
    counts = Counter(getattr(i32_loss, "route_counts", Counter()))
    counts[route] += 1
    if count > EXPECTED_ROWS or counts[route] > EXPECTED_ROUTES[route]:
        raise RuntimeError(f"I-32 route count exceeded contract: {count}/{dict(counts)}")
    remaining = EXPECTED_ROWS - count
    for name, expected in EXPECTED_ROUTES.items():
        if counts[name] + remaining < expected:
            raise RuntimeError(f"I-32 remaining rows cannot satisfy {name}: {dict(counts)}")
    if count == EXPECTED_ROWS and dict(counts) != EXPECTED_ROUTES:
        raise RuntimeError(f"I-32 final route mismatch: {dict(counts)}/{EXPECTED_ROUTES}")
    i32_loss.call_count = count
    i32_loss.route_counts = counts
    return count, counts


def i32_loss(trainer, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    response_start, response_end = target_span(labels)
    response_targets = labels[0, response_start:response_end]
    prompt_tokens = inputs["input_ids"][0, :response_start].detach().cpu().tolist()
    route, body_start, body_end = route_response(response_targets, prompt_tokens)
    if route == "material":
        selected = torch.arange(
            response_start + body_start,
            response_start + body_end,
            device=labels.device,
            dtype=torch.long,
        )
    else:
        relative = capped_positions(body_start, body_end, RETENTION_MAX_POSITIONS, labels.device)
        selected = relative + response_start
    targets = labels[0, selected]
    positions = selected - 1
    outputs, state = policy_with_start_fingerprint(trainer, model, inputs, positions)
    policy_logits = outputs.logits
    if route == "material":
        with torch.no_grad():
            teacher_logits = state["teacher"](**inputs, logits_to_keep=positions).logits.detach()
        teacher_kl = forward_kl(policy_logits, teacher_logits)
        parent_kl = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = MATERIAL_TEACHER_KL * teacher_kl
    else:
        with torch.no_grad():
            parent_logits = state["parent"](**inputs, logits_to_keep=positions).logits.detach()
        parent_kl = forward_kl(policy_logits, parent_logits)
        teacher_kl = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_PARENT_KL * parent_kl
    count, counts = record_route(route)
    if count <= 8 or count % 128 == 0 or count == EXPECTED_ROWS:
        print(
            f"[i32] microbatch={count}/{EXPECTED_ROWS} route={route} "
            f"tokens={targets.numel()} "
            f"teacher_kl={float(teacher_kl.detach()):.8f} "
            f"parent_kl={float(parent_kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    material = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX, 151667, 271, CLOSE_THINK_ID, 198, 176251, 151669, 159861, 168053, EOS_ID, 198]],
        dtype=torch.long,
    )
    start, end = target_span(material)
    assert route_response(material[0, start:end], [101, 102])[0] == "material"
    action = torch.tensor(
        [[IGNORE_INDEX, 151667, 271, CLOSE_THINK_ID, 198, 58, 10, 60, EOS_ID, 198]],
        dtype=torch.long,
    )
    start, end = target_span(action)
    assert route_response(action[0, start:end], [101, 102])[0] == "retention"
    natural_material = torch.tensor(
        [[IGNORE_INDEX, 151667, 271, CLOSE_THINK_ID, 198, 3001, 3002, EOS_ID, 198]],
        dtype=torch.long,
    )
    start, end = target_span(natural_material)
    material_prompt = [101, 176247, 151669, 159861, 168053, 102]
    assert route_response(natural_material[0, start:end], material_prompt)[0] == "material"
    assert route_response(natural_material[0, start:end], [101, 102])[0] == "retention"
    torch.manual_seed(13)
    logits = torch.randn(1, 17, 37, requires_grad=True)
    reference = torch.randn_like(logits)
    targets = torch.randint(0, 37, (17,))
    direct = F.cross_entropy(logits[0].float(), targets)
    assert torch.allclose(token_ce(logits, targets), direct, atol=1e-6)
    (token_ce(logits, targets) + forward_kl(logits, reference)).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert EXPECTED_ROUTES == {"material": 512, "retention": 1536}
    print("[i32] self-test passed")


def run_data_preflight() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    maximum = 0
    with TRAINING_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            user = "\n".join(
                value for value in (row["instruction"], row["input"]) if value
            )
            prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            response = f"{row['output']}<|im_end|>\n"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            route, body_start, body_end = route_response(
                torch.tensor(response_ids), prompt_ids
            )
            expected = "material" if row["route"] == "material_teacher" else "retention"
            if route != expected:
                raise RuntimeError(
                    f"I-32 preflight route mismatch at line {line_number}: "
                    f"{route}/{expected} task={row['task']}"
                )
            if body_end <= body_start:
                raise RuntimeError(f"I-32 empty scored body at line {line_number}")
            length = len(prompt_ids) + len(response_ids)
            if length > 16384:
                raise RuntimeError(f"I-32 cutoff overflow at line {line_number}: {length}")
            maximum = max(maximum, length)
            counts[route] += 1
            task_counts[row["task"]] += 1
    if dict(counts) != EXPECTED_ROUTES:
        raise RuntimeError(f"I-32 preflight route signature mismatch: {dict(counts)}")
    print(
        f"[i32] data preflight passed: rows={sum(counts.values())} "
        f"routes={dict(counts)} tasks={dict(task_counts)} max_tokens={maximum}",
        flush=True,
    )


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if "--data-preflight" in sys.argv:
        run_data_preflight()
        return
    from llamafactory.train.sft import trainer as sft_trainer

    original = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original(self, model, inputs, *args, **kwargs)
        return i32_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
