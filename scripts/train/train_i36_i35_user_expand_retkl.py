#!/usr/bin/env python3
"""Train a fresh r16 user residual over the verified I-35 step548 r112.

Action/topic rows optimize only the answer body with weak frozen-parent KL.
Material, recommendation, and world rows optimize frozen-parent KL only at a
bounded set of answer positions.  The configured r112 parent is merged before
LLaMA-Factory creates the one trainable r16 policy adapter.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT_ADAPTER = Path(
    os.environ.get(
        "I36_PARENT_ADAPTER",
        str(ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"),
    )
)
TRAINING_DATA = Path(
    os.environ.get(
        "I36_TRAINING_DATA",
        str(ROOT / "assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl"),
    )
)
AUDIT = Path(
    os.environ.get(
        "I36_AUDIT",
        str(ROOT / "logs/data/i36_i35_user_expand_retkl_v1_audit.json"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "I36_OUTPUT_DIR",
        str(ROOT / "checkpoints/i36_i35_user_expand_retkl_r16_v1"),
    )
)

SCHEMA_VERSION = "i36-i35-user-expand-retkl-v1"
BASE_CONFIG_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
PARENT_ADAPTER_SHA256 = "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
PARENT_CONFIG_SHA256 = "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"

IGNORE_INDEX = -100
CLOSE_THINK_ID = 151668
EOS_ID = 151645
ACTION_START_IDS = {58, 1183}
TOPIC_START_IDS = {90, 4913}
WHITESPACE_IDS = {198, 220, 262, 271}

EXPECTED_BATCH = 1
EXPECTED_ACCUMULATION = 4
EXPECTED_WORLD_SIZE = 1
EXPECTED_RANK = 16
EXPECTED_ALPHA = 16
EXPECTED_ROWS = 16500
EXPECTED_STEPS = 4125
EXPECTED_SAVE_STEPS = 2063
EXPECTED_TASK_COUNTS = {
    "action": 4000,
    "topic": 1500,
    "material_desc2sid": 2500,
    "rec_video": 2000,
    "rec_prod": 1750,
    "rec_ad": 1750,
    "rec_living": 1500,
    "world": 1500,
}

USER_PARENT_KL = 0.10
RETENTION_PARENT_KL = 4.0
RETENTION_MAX_POSITIONS = 128
TERMINAL_MULTIPLIER = 2.0
LOGIT_CHUNK = 8

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"


def _load_i34_helpers() -> Any:
    path = Path(__file__).with_name("train_i34_material_beam_margin_retkl.py")
    spec = importlib.util.spec_from_file_location("llmrec_i34_helpers_for_i36", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


I34 = _load_i34_helpers()


@dataclass(frozen=True)
class FormalContract:
    total_rows: int
    optimizer_steps: int
    route_counts: dict[str, int]
    task_counts: dict[str, int]
    data_sha256: str
    seed: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"I36 {field} must be a positive integer")
    return int(value)


def load_contract() -> FormalContract:
    if not AUDIT.is_file():
        raise RuntimeError(f"I36 formal audit is missing: {AUDIT}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"I36 schema drifted: {audit.get('schema_version')!r}")
    if audit.get("status") != "ready_for_training":
        raise RuntimeError(f"I36 audit is not trainable: {audit.get('status')!r}")
    mix = audit.get("mix")
    output = audit.get("output")
    if not isinstance(mix, dict) or not isinstance(output, dict):
        raise RuntimeError("I36 audit lacks mix/output contracts")
    total_rows = _positive_int(mix.get("total_rows"), "mix.total_rows")
    route_counts = {str(key): _positive_int(value, f"route.{key}") for key, value in mix.get("route_counts", {}).items()}
    task_counts = {str(key): _positive_int(value, f"task.{key}") for key, value in mix.get("task_counts", {}).items()}
    if total_rows != EXPECTED_ROWS or route_counts != {"retention_kl": 11000, "user_ce": 5500}:
        raise RuntimeError(f"I36 formal route contract drifted: {total_rows}/{route_counts}")
    if task_counts != EXPECTED_TASK_COUNTS:
        raise RuntimeError(f"I36 formal task contract drifted: {task_counts}")
    output_path = ROOT / str(output.get("path"))
    if output_path.resolve() != TRAINING_DATA.resolve():
        raise RuntimeError(f"I36 audit output path drifted: {output_path}/{TRAINING_DATA}")
    if _positive_int(output.get("rows"), "output.rows") != total_rows:
        raise RuntimeError("I36 output row count disagrees with mix")
    data_hash = str(output.get("sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", data_hash) is None:
        raise RuntimeError("I36 output SHA256 is invalid")
    seed = audit.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 19260836:
        raise RuntimeError(f"I36 seed drifted: {seed!r}")
    optimizer_steps = math.ceil(total_rows / (EXPECTED_BATCH * EXPECTED_ACCUMULATION))
    if optimizer_steps != EXPECTED_STEPS:
        raise RuntimeError(f"I36 optimizer-step contract drifted: {optimizer_steps}")
    return FormalContract(total_rows, optimizer_steps, route_counts, task_counts, data_hash, seed)


def verify_static_contract(*, require_data: bool) -> FormalContract | None:
    base_config = BASE / "config.json"
    if not base_config.is_file() or sha256(base_config) != BASE_CONFIG_SHA256:
        raise RuntimeError("I36 O6 base config is missing or hash-drifted")
    parent_weights = PARENT_ADAPTER / ADAPTER_SAFE_WEIGHTS_NAME
    parent_config_path = PARENT_ADAPTER / ADAPTER_CONFIG_NAME
    if not parent_weights.is_file() or sha256(parent_weights) != PARENT_ADAPTER_SHA256:
        raise RuntimeError("I36 I35-step548 parent weights are missing or hash-drifted")
    if not parent_config_path.is_file() or sha256(parent_config_path) != PARENT_CONFIG_SHA256:
        raise RuntimeError("I36 I35-step548 parent config is missing or hash-drifted")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    I34.assert_lora_config_contract(parent_config, 112, 112, "I36 parent")
    if not require_data:
        return None
    contract = load_contract()
    if not TRAINING_DATA.is_file() or sha256(TRAINING_DATA) != contract.data_sha256:
        raise RuntimeError("I36 formal training data is missing or hash-drifted")
    return contract


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    if labels.ndim != 2 or labels.size(0) != 1:
        raise RuntimeError("I36 requires per_device_train_batch_size=1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("I36 batch has no supervised response tokens")
    start = int(positions[0])
    end = int(positions[-1]) + 1
    if start == 0 or not torch.equal(
        positions, torch.arange(start, end, device=positions.device)
    ):
        raise RuntimeError("I36 requires a contiguous response span and packing=False")
    return start, end


def body_bounds(targets: torch.Tensor) -> tuple[int, int]:
    tokens = targets.detach().cpu().tolist()
    if CLOSE_THINK_ID in tokens:
        start = tokens.index(CLOSE_THINK_ID) + 1
        while start < len(tokens) and tokens[start] in WHITESPACE_IDS:
            start += 1
    else:
        start = 0
    end = len(tokens)
    while end > start and tokens[end - 1] in WHITESPACE_IDS:
        end -= 1
    if start >= end:
        raise RuntimeError("I36 response has no answer body")
    return start, end


def task_from_targets(targets: torch.Tensor) -> tuple[str, int, int]:
    start, end = body_bounds(targets)
    first = int(targets[start])
    if first in ACTION_START_IDS:
        return "action", start, end
    if first in TOPIC_START_IDS:
        return "topic", start, end
    return "retention", start, end


def uniformly_bounded_indices(start: int, end: int, maximum: int) -> torch.Tensor:
    count = end - start
    if count <= maximum:
        return torch.arange(start, end, dtype=torch.long)
    values = torch.linspace(start, end - 1, maximum, dtype=torch.float64).round().long()
    values = torch.unique_consecutive(values)
    if values.numel() != maximum:
        raise RuntimeError("I36 bounded-index construction lost positions")
    return values


def terminal_weights(targets: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(targets, dtype=torch.float32)
    content_end = targets.numel()
    if content_end and int(targets[content_end - 1]) == EOS_ID:
        weights[content_end - 1] = TERMINAL_MULTIPLIER
        content_end -= 1
    if content_end:
        weights[content_end - 1] = TERMINAL_MULTIPLIER
    return weights


def weighted_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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


def forward_kl(policy_logits: torch.Tensor, parent_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != parent_logits.shape:
        raise RuntimeError(f"I36 policy/parent logit mismatch: {policy_logits.shape}/{parent_logits.shape}")
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    tokens = policy_logits.size(1)
    for start in range(0, tokens, LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, tokens)
        policy = policy_logits[:, start:end].float()
        parent = parent_logits[:, start:end].float()
        total = total + F.kl_div(
            F.log_softmax(policy, dim=-1),
            F.softmax(parent, dim=-1),
            reduction="sum",
        )
    return total / tokens


def assert_formal_trainer_args(trainer: Any, contract: FormalContract) -> None:
    args = trainer.args
    observed = {
        "batch": int(args.per_device_train_batch_size),
        "accum": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
    }
    expected = {
        "batch": EXPECTED_BATCH,
        "accum": EXPECTED_ACCUMULATION,
        "max_steps": contract.optimizer_steps,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if observed != expected:
        raise RuntimeError(f"I36 trainer contract drifted: {observed}/{expected}")
    checks = {
        "learning_rate": (float(args.learning_rate), 5.0e-6),
        "warmup_ratio": (float(args.warmup_ratio), 0.03),
        "weight_decay": (float(args.weight_decay), 0.001),
    }
    for name, (observed_value, expected_value) in checks.items():
        if not math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"I36 {name} drifted: {observed_value}/{expected_value}")
    scheduler = getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)
    save_strategy = getattr(args.save_strategy, "value", args.save_strategy)
    if str(scheduler) != "cosine" or str(save_strategy) != "steps":
        raise RuntimeError("I36 requires cosine scheduling and step checkpoints")
    if int(args.save_steps) != EXPECTED_SAVE_STEPS or not bool(args.save_only_model):
        raise RuntimeError("I36 checkpoint cadence or adapter-only save drifted")
    if bool(getattr(args, "packing", False)) or not bool(args.bf16):
        raise RuntimeError("I36 requires packing=False and BF16")
    if int(args.seed) != contract.seed:
        raise RuntimeError(f"I36 trainer seed drifted: {args.seed}/{contract.seed}")
    if Path(str(args.output_dir)).resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"I36 output path drifted: {args.output_dir}/{OUTPUT_DIR}")
    reports = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in reports or os.environ.get("WANDB_MODE", "online").lower() != "online":
        raise RuntimeError("I36 requires online W&B")


def ensure_runtime(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i36_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        I34.assert_adapter_runtime_state(unwrapped, state["policy_name"])
        I34.assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        return unwrapped, state
    contract = load_contract()
    assert_formal_trainer_args(trainer, contract)
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError("I36 expects one fresh adapter over the merged r112 parent")
    policy_name = I34._single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name not in configs:
        raise RuntimeError("I36 active policy adapter is not configured")
    I34.assert_lora_config_contract(configs[policy_name], EXPECTED_RANK, EXPECTED_ALPHA, "I36 fresh policy")
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("I36 requires PEFT disable_adapter for the merged parent")
    I34.assert_frozen_embeddings_and_head(unwrapped)
    parameter_ids = I34.assert_exact_policy_trainable_parameters(unwrapped, policy_name)
    I34.assert_adapter_runtime_state(unwrapped, policy_name)
    state = {
        "policy_name": policy_name,
        "policy_parameter_ids": parameter_ids,
        "contract": contract,
        "fingerprint_checked": False,
    }
    trainer._i36_state = state
    print(
        f"[i36] contract PASS: merged r112 + fresh r16; rows={contract.total_rows} "
        f"steps={contract.optimizer_steps}",
        flush=True,
    )
    return unwrapped, state


def selected_logits(model: Any, inputs: Mapping[str, Any], positions: torch.Tensor) -> Any:
    try:
        return model(**inputs, logits_to_keep=positions)
    except TypeError as exc:
        raise RuntimeError("I36 model must support logits_to_keep") from exc


def paired_parent_policy(
    trainer: Any, model: Any, inputs: Mapping[str, Any], positions: torch.Tensor
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    unwrapped, state = ensure_runtime(trainer, model)
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        with torch.no_grad(), unwrapped.disable_adapter():
            parent_outputs = selected_logits(model, inputs, positions)
            parent_logits = parent_outputs.logits.detach()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    policy_outputs = selected_logits(model, inputs, positions)
    if not state["fingerprint_checked"]:
        maximum = float((policy_outputs.logits.detach().float() - parent_logits.float()).abs().max())
        if maximum > 1e-4:
            raise RuntimeError(f"I36 step-0 parent fingerprint failed: max_abs={maximum:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i36] step-0 parent fingerprint PASS: max_abs={maximum:.8f}", flush=True)
    return policy_outputs, parent_logits, state


def reset_counters() -> None:
    for name in ("call_count", "route_counts"):
        if hasattr(i36_loss, name):
            delattr(i36_loss, name)


def record_route(task: str, contract: FormalContract) -> tuple[int, Counter[str]]:
    count = int(getattr(i36_loss, "call_count", 0)) + 1
    routes = Counter(getattr(i36_loss, "route_counts", Counter()))
    routes[task] += 1
    expected = Counter({"action": 4000, "topic": 1500, "retention": 11000})
    if count > contract.total_rows or routes[task] > expected[task]:
        raise RuntimeError(f"I36 route count exceeded contract: {count}/{dict(routes)}")
    remaining = contract.total_rows - count
    for name, wanted in expected.items():
        if routes[name] + remaining < wanted:
            raise RuntimeError(f"I36 can no longer satisfy route {name}")
    i36_loss.call_count = count
    i36_loss.route_counts = routes
    return count, routes


def i36_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    target_start, target_end = target_span(labels)
    full_targets = labels[0, target_start:target_end]
    task, body_start, body_end = task_from_targets(full_targets)
    if task == "retention":
        selected_relative = uniformly_bounded_indices(body_start, body_end, RETENTION_MAX_POSITIONS)
    else:
        selected_relative = torch.arange(body_start, body_end, dtype=torch.long)
    selected_targets = full_targets[selected_relative.to(full_targets.device)]
    prediction_positions = selected_relative.to(inputs["input_ids"].device) + target_start - 1
    outputs, parent_logits, state = paired_parent_policy(self, model, inputs, prediction_positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != selected_targets.numel():
        raise RuntimeError(f"I36 logits/targets mismatch: {policy_logits.size(1)}/{selected_targets.numel()}")
    parent_kl = forward_kl(policy_logits, parent_logits)
    if task == "retention":
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_PARENT_KL * parent_kl
    else:
        weights = terminal_weights(selected_targets)
        ce = weighted_ce(policy_logits, selected_targets, weights)
        loss = ce + USER_PARENT_KL * parent_kl
    count, routes = record_route(task, state["contract"])
    if count <= 8 or count % 250 == 0 or count == state["contract"].total_rows:
        print(
            f"[i36] microbatch={count}/{state['contract'].total_rows} route={task} "
            f"tokens={selected_targets.numel()} ce={float(ce.detach()):.6f} "
            f"parent_kl={float(parent_kl.detach()):.8f} loss={float(loss.detach()):.6f} "
            f"routes={dict(routes)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def textual_route(row: Mapping[str, Any]) -> str:
    output = str(row.get("output") or "")
    marker = "</think>"
    body = output.split(marker, 1)[1].strip() if marker in output else output.strip()
    if body.startswith("["):
        return "action"
    if body.startswith("{"):
        return "topic"
    return "retention"


def run_data_preflight() -> FormalContract:
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    contract = verify_static_contract(require_data=True)
    assert contract is not None
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True, use_fast=True)
    template = TEMPLATES["qwen3_nothink"]
    routes = Counter()
    tasks = Counter()
    maximum_tokens = 0
    with TRAINING_DATA.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"I36 blank data row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"I36 non-object row at line {line_number}")
            route = textual_route(row)
            declared = row.get("i36_route")
            expected_declared = "user_ce" if route in {"action", "topic"} else "retention_kl"
            if declared != expected_declared:
                raise RuntimeError(f"I36 route drift at line {line_number}: {declared}/{route}")
            task = str(row.get("task") or "")
            if route in {"action", "topic"} and task != route:
                raise RuntimeError(f"I36 user task drift at line {line_number}: {task}/{route}")
            if route == "retention" and task in {"action", "topic"}:
                raise RuntimeError(f"I36 KL-only row collides with user task at line {line_number}")
            messages = [
                {"role": "user", "content": row.get("input")},
                {"role": "assistant", "content": row.get("output")},
            ]
            prompt_ids, response_ids = template.encode_oneturn(
                tokenizer, messages, row.get("instruction"), None
            )
            total_tokens = len(prompt_ids) + len(response_ids)
            if total_tokens > 16384:
                raise RuntimeError(f"I36 cutoff overflow at line {line_number}: {total_tokens}")
            response_tensor = torch.tensor(response_ids, dtype=torch.long)
            token_route, _, _ = task_from_targets(response_tensor)
            if token_route != route:
                raise RuntimeError(f"I36 token/text route drift at line {line_number}: {token_route}/{route}")
            maximum_tokens = max(maximum_tokens, total_tokens)
            routes[route] += 1
            tasks[task] += 1
    expected_routes = Counter({"action": 4000, "topic": 1500, "retention": 11000})
    if routes != expected_routes or tasks != Counter(contract.task_counts):
        raise RuntimeError(f"I36 preflight signature drifted: routes={dict(routes)} tasks={dict(tasks)}")
    print(
        f"[i36] data preflight PASS: rows={contract.total_rows} routes={dict(routes)} "
        f"tasks={dict(tasks)} steps={contract.optimizer_steps} max_tokens={maximum_tokens} "
        f"data_sha256={contract.data_sha256}",
        flush=True,
    )
    return contract


def run_self_test() -> None:
    torch.manual_seed(36)
    vocab = 41
    policy = torch.randn(1, 17, vocab, requires_grad=True)
    parent = torch.randn(1, 17, vocab)
    targets = torch.randint(0, vocab, (17,))
    weights = torch.linspace(1.0, 2.0, 17)
    direct_ce = (F.cross_entropy(policy[0], targets, reduction="none") * weights).sum() / weights.sum()
    chunked_ce = weighted_ce(policy, targets, weights)
    if not torch.allclose(direct_ce, chunked_ce, atol=1e-6):
        raise AssertionError("I36 chunked CE self-test failed")
    direct_kl = F.kl_div(
        F.log_softmax(policy.float(), dim=-1),
        F.softmax(parent.float(), dim=-1),
        reduction="sum",
    ) / 17
    chunked_kl = forward_kl(policy, parent)
    if not torch.allclose(direct_kl, chunked_kl, atol=1e-6):
        raise AssertionError("I36 chunked KL self-test failed")
    action = torch.tensor([151667, 198, CLOSE_THINK_ID, 198, 58, 100, 60, EOS_ID, 198])
    topic = torch.tensor([151667, 198, CLOSE_THINK_ID, 198, 90, 100, 92, EOS_ID, 198])
    retention = torch.tensor([151667, 198, CLOSE_THINK_ID, 198, 176245, 100, EOS_ID, 198])
    if task_from_targets(action)[0] != "action" or task_from_targets(topic)[0] != "topic" or task_from_targets(retention)[0] != "retention":
        raise AssertionError("I36 route self-test failed")
    bounded = uniformly_bounded_indices(0, 1000, 128)
    if bounded.numel() != 128 or int(bounded[0]) != 0 or int(bounded[-1]) != 999:
        raise AssertionError("I36 bounded-position self-test failed")
    (chunked_ce + chunked_kl).backward()
    if policy.grad is None or not torch.isfinite(policy.grad).all():
        raise AssertionError("I36 gradient self-test failed")
    print("[i36] self-test PASS: routing, answer-body CE, bounded parent KL, and gradients", flush=True)


def assert_final_contract(contract: FormalContract) -> None:
    count = int(getattr(i36_loss, "call_count", 0))
    routes = Counter(getattr(i36_loss, "route_counts", Counter()))
    expected = Counter({"action": 4000, "topic": 1500, "retention": 11000})
    if count != contract.total_rows or routes != expected:
        raise RuntimeError(f"I36 final route contract failed: {count}/{dict(routes)}")


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if "--data-preflight" in sys.argv:
        run_data_preflight()
        return

    contract = verify_static_contract(require_data=True)
    assert contract is not None
    if OUTPUT_DIR.exists() and any(
        (OUTPUT_DIR / name).exists()
        for name in (ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME)
    ):
        raise RuntimeError(f"I36 refuses to overwrite existing adapter output: {OUTPUT_DIR}")

    from peft import PeftModel
    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss
    original_create_optimizer = sft_trainer.CustomSeq2SeqTrainer.create_optimizer
    original_trainer_save = sft_trainer.CustomSeq2SeqTrainer._save
    original_peft_save = PeftModel.save_pretrained

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        if args:
            if len(args) != 1 or "return_outputs" in kwargs:
                raise RuntimeError("I36 unexpected compute_loss positional arguments")
            kwargs["return_outputs"] = args[0]
        return i36_loss(self, model, inputs, **kwargs)

    def patched_create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        unwrapped, state = ensure_runtime(self, self.model)
        policy_ids = I34.assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        if not I34.assert_optimizer_policy_only(self, unwrapped, state["policy_name"]):
            raise RuntimeError("I36 optimizer is not policy-only")
        self._i36_optimizer_policy_ids = policy_ids
        print(f"[i36] optimizer policy-only PASS: tensors={len(policy_ids)}", flush=True)
        return optimizer

    def patched_trainer_save(self, output_dir=None, state_dict=None):
        unwrapped, state = ensure_runtime(self, self.model)
        policy_ids = I34.assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        if policy_ids != state["policy_parameter_ids"]:
            raise RuntimeError("I36 policy parameters changed before save")
        if not I34.assert_optimizer_policy_only(self, unwrapped, state["policy_name"]):
            raise RuntimeError("I36 optimizer disappeared before save")
        result = original_trainer_save(self, output_dir=output_dir, state_dict=state_dict)
        saved = Path(output_dir if output_dir is not None else self.args.output_dir)
        I34.assert_policy_only_save_artifacts(saved, bool(getattr(self.args, "save_safetensors", True)))
        return result

    def patched_peft_save(peft_model, save_directory, *args, selected_adapters=None, **kwargs):
        return I34.policy_only_save_pretrained(
            original_peft_save,
            peft_model,
            save_directory,
            *args,
            selected_adapters=selected_adapters,
            **kwargs,
        )

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss
    sft_trainer.CustomSeq2SeqTrainer.create_optimizer = patched_create_optimizer
    sft_trainer.CustomSeq2SeqTrainer._save = patched_trainer_save
    PeftModel.save_pretrained = patched_peft_save

    from llamafactory.train.tuner import run_exp

    reset_counters()
    run_exp()
    assert_final_contract(contract)
    print(f"[i36] training PASS: rows={contract.total_rows} steps={contract.optimizer_steps}", flush=True)


if __name__ == "__main__":
    main()
