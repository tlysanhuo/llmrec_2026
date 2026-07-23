#!/usr/bin/env python3
"""Train the I-38M task-conditioned teacher residual.

The policy starts from the verified I-23 r64 adapter. Material responses are
anchored to that exact merged start through ``disable_adapter()``. Every other
task is distilled from the frozen I-35 step548 r112 adapter. Only the fresh
r16 residual is trainable; gold CE is intentionally zero on both routes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
START_ADAPTER = ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
RETENTION_TEACHER = ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"
TRAINING_DATA = ROOT / "assets/derived/processed/data_i38_i23_material_i35_teacher_retkl_v1.jsonl"
OUTPUT_DIR = ROOT / "checkpoints/i38_i23_material_i35_teacher_retkl_r16_v1"

BASE_CONFIG_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
START_ADAPTER_SHA256 = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
START_CONFIG_SHA256 = "b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7"
RETENTION_TEACHER_SHA256 = "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
RETENTION_TEACHER_CONFIG_SHA256 = "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"
TRAINING_DATA_SHA256 = "5d8ca1a6fa9190841187543559ead1d497d48a50b082382c9fa8501add928d58"

IGNORE_INDEX = -100
CLOSE_THINK_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}
DOMAIN_IDS = {176245, 176247, 176249, 176251}
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244

MATERIAL_START_KL = float(os.environ.get("I38_MATERIAL_START_KL", "8.0"))
RETENTION_TEACHER_KL = float(os.environ.get("I38_RETENTION_TEACHER_KL", "8.0"))
RETENTION_MAX_POSITIONS = int(os.environ.get("I38_RETENTION_MAX_POSITIONS", "128"))
LOGIT_CHUNK = int(os.environ.get("I38_LOGIT_CHUNK", "8"))

EXPECTED_ROWS = 2740
EXPECTED_ROUTES = {"material": 1370, "retention": 1370}
EXPECTED_TASKS = {
    "material_desc2sid": 1370,
    "action": 207,
    "topic": 206,
    "rec_video": 206,
    "rec_prod": 207,
    "rec_ad": 206,
    "rec_living": 207,
    "world": 131,
}
EXPECTED_RANK_ALPHA = (16, 16)
EXPECTED_TRAINER = {
    "batch": 1,
    "accum": 4,
    "max_steps": 685,
    "world_size": 1,
    "seed": 19260838,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_static_contract() -> None:
    expected = {
        BASE / "config.json": BASE_CONFIG_SHA256,
        START_ADAPTER / "adapter_model.safetensors": START_ADAPTER_SHA256,
        START_ADAPTER / "adapter_config.json": START_CONFIG_SHA256,
        RETENTION_TEACHER / "adapter_model.safetensors": RETENTION_TEACHER_SHA256,
        RETENTION_TEACHER / "adapter_config.json": RETENTION_TEACHER_CONFIG_SHA256,
        TRAINING_DATA: TRAINING_DATA_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file():
            raise RuntimeError(f"I-38 locked artifact is missing: {path}")
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f"I-38 locked artifact drifted: {path} {actual}/{expected_hash}")


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I-38 requires per-device batch size 1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("I-38 batch has no response targets")
    start, end = int(positions[0]), int(positions[-1]) + 1
    if start == 0:
        raise RuntimeError("I-38 response starts at token zero")
    expected = torch.arange(start, end, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("I-38 forbids packing and disjoint response spans")
    return start, end


def scored_bounds(tokens: list[int]) -> tuple[int, int]:
    try:
        start = tokens.index(CLOSE_THINK_ID) + 1
    except ValueError:
        start = 0
    while start < len(tokens) and tokens[start] in WHITESPACE_IDS:
        start += 1
    end = len(tokens)
    while end > start and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    if start >= end:
        raise RuntimeError("I-38 response body is empty")
    if tokens[end - 1] != EOS_ID:
        raise RuntimeError("I-38 response body is missing final EOS")
    return start, end


def route_response(targets: torch.Tensor) -> tuple[str, int, int]:
    tokens = targets.detach().cpu().tolist()
    start, end = scored_bounds(tokens)
    body = tokens[start:end]
    if (
        len(body) == 5
        and body[0] in DOMAIN_IDS
        and A_LO <= body[1] <= A_HI
        and B_LO <= body[2] <= B_HI
        and C_LO <= body[3] <= C_HI
        and body[4] == EOS_ID
    ):
        return "material", start, end
    return "retention", start, end


def capped_positions(start: int, end: int, cap: int, device: torch.device) -> torch.Tensor:
    count = end - start
    if count <= 0:
        raise RuntimeError("I-38 cannot select positions from an empty response")
    if count <= cap:
        return torch.arange(start, end, device=device, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap, device=device).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("I-38 position cap produced duplicate offsets")
    return relative + start


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape or policy_logits.ndim != 3:
        raise RuntimeError(
            f"I-38 KL shape mismatch: {tuple(policy_logits.shape)}/{tuple(reference_logits.shape)}"
        )
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
    raise RuntimeError(f"I-38 expected one active residual adapter, got {value!r}")


def selected_logits(model: Any, inputs: Mapping[str, Any], positions: torch.Tensor) -> Any:
    try:
        return model(**inputs, logits_to_keep=positions)
    except TypeError as error:
        raise RuntimeError("I-38 model must support logits_to_keep") from error


def ensure_models(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i38_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        return unwrapped, state

    args = trainer.args
    observed = {
        "batch": int(args.per_device_train_batch_size),
        "accum": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
        "seed": int(args.seed),
    }
    if observed != EXPECTED_TRAINER:
        raise RuntimeError(f"I-38 trainer contract drifted: {observed}/{EXPECTED_TRAINER}")
    if (
        not math.isclose(float(args.learning_rate), 5.0e-6, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(args.warmup_ratio), 0.03, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(args.weight_decay), 0.001, rel_tol=0.0, abs_tol=1e-12)
        or not bool(args.bf16)
        or not str(args.lr_scheduler_type).lower().endswith("cosine")
    ):
        raise RuntimeError("I-38 optimizer/scheduler/bf16 contract drifted")
    if bool(getattr(args, "packing", False)):
        raise RuntimeError("I-38 requires packing=false")
    if Path(str(args.output_dir)).resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"I-38 output directory drifted: {args.output_dir}")
    report_to = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in report_to or os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("I-38 requires W&B online reporting")
    verify_static_contract()

    peft_config = getattr(unwrapped, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        raise RuntimeError("I-38 expected one fresh residual after merging the I-23 start")
    adapter_name = single_adapter_name(getattr(unwrapped, "active_adapter", None))
    config = peft_config[adapter_name]
    rank_alpha = (int(getattr(config, "r", -1)), int(getattr(config, "lora_alpha", -1)))
    if rank_alpha != EXPECTED_RANK_ALPHA:
        raise RuntimeError(f"I-38 expected r16/alpha16, got {rank_alpha}")
    trainable = [name for name, parameter in unwrapped.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("I-38 trainable state is empty or contains non-LoRA parameters")
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("I-38 requires disable_adapter for the exact I-23 start anchor")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = next(unwrapped.parameters()).device
    teacher_base = AutoModelForCausalLM.from_pretrained(
        BASE,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    retention_teacher = PeftModel.from_pretrained(
        teacher_base,
        RETENTION_TEACHER,
        adapter_name="i35_step548_teacher",
        is_trainable=False,
        low_cpu_mem_usage=True,
    ).eval()
    for parameter in retention_teacher.parameters():
        parameter.requires_grad_(False)
    state = {
        "adapter_name": adapter_name,
        "retention_teacher": retention_teacher,
        "fingerprint_checked": False,
    }
    trainer._i38_state = state
    print(
        f"[i38] contract PASS: merged I-23 r64 + fresh r16/alpha16; "
        f"trainable_tensors={len(trainable)}; frozen I-35 step548 teacher loaded",
        flush=True,
    )
    return unwrapped, state


def policy_and_start(trainer: Any, model: Any, inputs: Mapping[str, Any], positions: torch.Tensor):
    unwrapped, state = ensure_models(trainer, model)
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        with torch.no_grad(), unwrapped.disable_adapter():
            start_logits = selected_logits(model, inputs, positions).logits.detach()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    outputs = selected_logits(model, inputs, positions)
    if not state["fingerprint_checked"]:
        max_abs = float((outputs.logits.detach().float() - start_logits.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"I-38 step-0 I-23 fingerprint failed: {max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i38] step-0 I-23 fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return outputs, start_logits, state


def record_route(route: str) -> tuple[int, Counter[str]]:
    count = int(getattr(i38_loss, "call_count", 0)) + 1
    counts = Counter(getattr(i38_loss, "route_counts", Counter()))
    counts[route] += 1
    if count > EXPECTED_ROWS or counts[route] > EXPECTED_ROUTES[route]:
        raise RuntimeError(f"I-38 route count exceeded contract: {count}/{dict(counts)}")
    remaining = EXPECTED_ROWS - count
    for name, expected in EXPECTED_ROUTES.items():
        if counts[name] + remaining < expected:
            raise RuntimeError(f"I-38 remaining rows cannot satisfy {name}: {dict(counts)}")
    if count == EXPECTED_ROWS and dict(counts) != EXPECTED_ROUTES:
        raise RuntimeError(f"I-38 final route mismatch: {dict(counts)}/{EXPECTED_ROUTES}")
    i38_loss.call_count = count
    i38_loss.route_counts = counts
    return count, counts


def i38_loss(trainer: Any, model: Any, inputs: dict[str, torch.Tensor], return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    response_start, response_end = target_span(labels)
    response_targets = labels[0, response_start:response_end]
    route, body_start, body_end = route_response(response_targets)
    if route == "material":
        relative = torch.arange(body_start, body_end, device=labels.device, dtype=torch.long)
    else:
        relative = capped_positions(body_start, body_end, RETENTION_MAX_POSITIONS, labels.device)
    selected = relative + response_start
    positions = selected - 1
    outputs, start_logits, state = policy_and_start(trainer, model, inputs, positions)
    policy_logits = outputs.logits
    if route == "material":
        material_kl = forward_kl(policy_logits, start_logits)
        retention_kl = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = MATERIAL_START_KL * material_kl
    else:
        with torch.no_grad():
            teacher_logits = selected_logits(
                state["retention_teacher"], inputs, positions
            ).logits.detach()
        retention_kl = forward_kl(policy_logits, teacher_logits)
        material_kl = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_TEACHER_KL * retention_kl
    count, counts = record_route(route)
    if count <= 8 or count % 128 == 0 or count == EXPECTED_ROWS:
        print(
            f"[i38] microbatch={count}/{EXPECTED_ROWS} route={route} tokens={selected.numel()} "
            f"material_start_kl={float(material_kl.detach()):.8f} "
            f"retention_i35_kl={float(retention_kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_self_test() -> None:
    material = torch.tensor(
        [151667, 271, CLOSE_THINK_ID, 198, 176251, A_LO, B_LO, C_LO, EOS_ID, 198],
        dtype=torch.long,
    )
    route, start, end = route_response(material)
    assert route == "material" and material[start:end].tolist() == [176251, A_LO, B_LO, C_LO, EOS_ID]
    retention = torch.tensor(
        [151667, 271, CLOSE_THINK_ID, 198, 58, 10, 60, EOS_ID, 198], dtype=torch.long
    )
    assert route_response(retention)[0] == "retention"
    legacy = torch.tensor([151667, 1234, EOS_ID, 198], dtype=torch.long)
    assert route_response(legacy)[0] == "retention"
    torch.manual_seed(38)
    logits = torch.randn(1, 17, 41, requires_grad=True)
    reference = torch.randn_like(logits)
    loss = forward_kl(logits, reference)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None and torch.isfinite(logits.grad).all()
    assert EXPECTED_ROUTES == {"material": 1370, "retention": 1370}
    print("[i38] self-test PASS: routing, bounded KL, and gradients", flush=True)


def run_data_preflight() -> None:
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    verify_static_contract()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    routes: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    maximum = 0
    with TRAINING_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"I-38 blank data row at line {line_number}")
            row = json.loads(line)
            prompt_ids, response_ids = template.encode_oneturn(
                tokenizer,
                [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ],
                row["instruction"],
                None,
            )
            route, body_start, body_end = route_response(torch.tensor(response_ids))
            declared = row.get("route")
            expected_route = (
                "material" if declared == "material_anchor_i23" else
                "retention" if declared == "retention_teacher_i35" else None
            )
            if route != expected_route:
                raise RuntimeError(
                    f"I-38 route mismatch at line {line_number}: {route}/{expected_route}"
                )
            if body_end <= body_start:
                raise RuntimeError(f"I-38 empty scored body at line {line_number}")
            length = len(prompt_ids) + len(response_ids)
            if length > 16384:
                raise RuntimeError(f"I-38 cutoff overflow at line {line_number}: {length}")
            if row.get("source_asset") != "data_i35_video_boundary_retkl_v1":
                raise RuntimeError(f"I-38 unregistered source marker at line {line_number}")
            maximum = max(maximum, length)
            routes[route] += 1
            tasks[str(row.get("task"))] += 1
    if sum(routes.values()) != EXPECTED_ROWS or dict(routes) != EXPECTED_ROUTES:
        raise RuntimeError(f"I-38 route counts drifted: {dict(routes)}/{EXPECTED_ROUTES}")
    if dict(tasks) != EXPECTED_TASKS:
        raise RuntimeError(f"I-38 task counts drifted: {dict(tasks)}/{EXPECTED_TASKS}")
    print(
        f"[i38] data preflight PASS: rows={sum(routes.values())} routes={dict(routes)} "
        f"tasks={dict(tasks)} max_tokens={maximum} sha256={sha256(TRAINING_DATA)}",
        flush=True,
    )


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if "--data-preflight" in sys.argv:
        run_data_preflight()
        return
    verify_static_contract()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"I-38 refuses to overwrite its output directory: {OUTPUT_DIR}")

    from llamafactory.train.sft import trainer as sft_trainer

    original = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original(self, model, inputs, *args, **kwargs)
        if args:
            raise RuntimeError(f"I-38 unexpected positional compute_loss args: {args!r}")
        return i38_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()
