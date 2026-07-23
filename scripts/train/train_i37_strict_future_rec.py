#!/usr/bin/env python3
"""Train a guarded r8 strict-future recommendation residual over I-35 step548."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]


def load_i36_runtime() -> Any:
    path = Path(__file__).with_name("train_i36_i35_user_expand_retkl.py")
    spec = importlib.util.spec_from_file_location("llmrec_i36_runtime_for_i37", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


I36 = load_i36_runtime()

TRAINING_DATA = Path(
    os.environ.get(
        "I37_TRAINING_DATA",
        str(ROOT / "assets/derived/processed/data_i37_strict_future_rec_v1.jsonl"),
    )
)
AUDIT = Path(
    os.environ.get(
        "I37_AUDIT",
        str(ROOT / "logs/data/i37_strict_future_rec_v1_audit.json"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "I37_OUTPUT_DIR",
        str(ROOT / "checkpoints/i37_strict_future_rec_r8_v1"),
    )
)

SCHEMA_VERSION = "i37-strict-future-rec-v1"
EXPECTED_ROWS = 2048
EXPECTED_STEPS = 512
EXPECTED_SAVE_STEPS = 256
EXPECTED_ROUTE_COUNTS = {"future_ce": 1024, "retention_kl": 1024}
EXPECTED_TASK_COUNTS = {
    "future_video": 512,
    "future_ad": 512,
    "material_sid2desc": 128,
    "action": 128,
    "topic": 128,
    "rec_video": 131,
    "rec_prod": 131,
    "rec_ad": 131,
    "rec_living": 131,
    "world": 116,
}
FUTURE_DOMAIN_IDS = {176245, 176251}
FUTURE_BODY_RE = re.compile(
    r"^<\|(video|ad)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>$"
)

FUTURE_CE_WEIGHT = 0.10
FUTURE_PARENT_KL = 16.0
RETENTION_PARENT_KL = 16.0
RETENTION_MAX_POSITIONS = 128

# Reuse the audited I-36 parent/policy plumbing, but replace every experiment
# contract that can affect data, rank, steps, output, or loss routing.
I36.TRAINING_DATA = TRAINING_DATA
I36.AUDIT = AUDIT
I36.OUTPUT_DIR = OUTPUT_DIR
I36.SCHEMA_VERSION = SCHEMA_VERSION
I36.EXPECTED_RANK = 8
I36.EXPECTED_ALPHA = 8
I36.EXPECTED_ROWS = EXPECTED_ROWS
I36.EXPECTED_STEPS = EXPECTED_STEPS
I36.EXPECTED_SAVE_STEPS = EXPECTED_SAVE_STEPS
I36.EXPECTED_TASK_COUNTS = EXPECTED_TASK_COUNTS


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"I37 {field} must be a positive integer")
    return int(value)


def load_contract() -> Any:
    if not AUDIT.is_file():
        raise RuntimeError(f"I37 formal audit is missing: {AUDIT}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"I37 schema drifted: {audit.get('schema_version')!r}")
    if audit.get("status") != "ready_for_training":
        raise RuntimeError(f"I37 audit is not trainable: {audit.get('status')!r}")
    contract = audit.get("contract")
    output = audit.get("output")
    if not isinstance(contract, dict) or not isinstance(output, dict):
        raise RuntimeError("I37 audit lacks contract/output")
    rows = positive_int(contract.get("total_rows"), "contract.total_rows")
    routes = {
        str(key): positive_int(value, f"route.{key}")
        for key, value in contract.get("route_counts", {}).items()
    }
    tasks = {
        str(key): positive_int(value, f"task.{key}")
        for key, value in contract.get("task_counts", {}).items()
    }
    if rows != EXPECTED_ROWS or routes != EXPECTED_ROUTE_COUNTS:
        raise RuntimeError(f"I37 route contract drifted: {rows}/{routes}")
    if tasks != EXPECTED_TASK_COUNTS:
        raise RuntimeError(f"I37 task contract drifted: {tasks}")
    output_path = Path(str(output.get("path") or ""))
    if output_path.resolve() != TRAINING_DATA.resolve():
        raise RuntimeError(f"I37 audit output path drifted: {output_path}/{TRAINING_DATA}")
    if positive_int(output.get("rows"), "output.rows") != rows:
        raise RuntimeError("I37 output rows disagree with contract")
    data_hash = str(output.get("sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", data_hash) is None:
        raise RuntimeError("I37 output SHA256 is invalid")
    seed = audit.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 19260837:
        raise RuntimeError(f"I37 seed drifted: {seed!r}")
    steps = math.ceil(rows / (I36.EXPECTED_BATCH * I36.EXPECTED_ACCUMULATION))
    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"I37 optimizer-step contract drifted: {steps}")
    return I36.FormalContract(rows, steps, routes, tasks, data_hash, seed)


def ensure_runtime(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i37_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        I36.I34.assert_adapter_runtime_state(unwrapped, state["policy_name"])
        I36.I34.assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        return unwrapped, state
    contract = load_contract()
    I36.assert_formal_trainer_args(trainer, contract)
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError("I37 expects one fresh adapter over the merged r112 parent")
    policy_name = I36.I34._single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name not in configs:
        raise RuntimeError("I37 active policy adapter is not configured")
    I36.I34.assert_lora_config_contract(configs[policy_name], 8, 8, "I37 fresh policy")
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("I37 requires PEFT disable_adapter for the merged parent")
    I36.I34.assert_frozen_embeddings_and_head(unwrapped)
    parameter_ids = I36.I34.assert_exact_policy_trainable_parameters(unwrapped, policy_name)
    I36.I34.assert_adapter_runtime_state(unwrapped, policy_name)
    state = {
        "policy_name": policy_name,
        "policy_parameter_ids": parameter_ids,
        "contract": contract,
        "fingerprint_checked": False,
    }
    trainer._i37_state = state
    print(
        f"[i37] contract PASS: merged I35 r112 + fresh r8; "
        f"rows={contract.total_rows} steps={contract.optimizer_steps}",
        flush=True,
    )
    return unwrapped, state


def response_route(targets: torch.Tensor) -> tuple[str, int, int]:
    start, end = I36.body_bounds(targets)
    first = int(targets[start])
    return ("future_ce" if first in FUTURE_DOMAIN_IDS else "retention_kl"), start, end


def record_route(route: str, contract: Any) -> tuple[int, Counter[str]]:
    count = int(getattr(i37_loss, "call_count", 0)) + 1
    routes = Counter(getattr(i37_loss, "route_counts", Counter()))
    routes[route] += 1
    expected = Counter(EXPECTED_ROUTE_COUNTS)
    if count > contract.total_rows or routes[route] > expected[route]:
        raise RuntimeError(f"I37 route count exceeded contract: {count}/{dict(routes)}")
    remaining = contract.total_rows - count
    for name, wanted in expected.items():
        if routes[name] + remaining < wanted:
            raise RuntimeError(f"I37 can no longer satisfy route {name}")
    i37_loss.call_count = count
    i37_loss.route_counts = routes
    return count, routes


def i37_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.pop("labels")
    target_start, target_end = I36.target_span(labels)
    full_targets = labels[0, target_start:target_end]
    route, body_start, body_end = response_route(full_targets)
    if route == "future_ce":
        selected_relative = torch.arange(body_start, body_end, dtype=torch.long)
    else:
        selected_relative = I36.uniformly_bounded_indices(
            body_start, body_end, RETENTION_MAX_POSITIONS
        )
    selected_targets = full_targets[selected_relative.to(full_targets.device)]
    positions = selected_relative.to(inputs["input_ids"].device) + target_start - 1
    outputs, parent_logits, state = I36.paired_parent_policy(self, model, inputs, positions)
    policy_logits = outputs.logits
    if policy_logits.size(1) != selected_targets.numel():
        raise RuntimeError(
            f"I37 logits/targets mismatch: {policy_logits.size(1)}/{selected_targets.numel()}"
        )
    parent_kl = I36.forward_kl(policy_logits, parent_logits)
    if route == "future_ce":
        weights = I36.terminal_weights(selected_targets)
        ce = I36.weighted_ce(policy_logits, selected_targets, weights)
        loss = FUTURE_CE_WEIGHT * ce + FUTURE_PARENT_KL * parent_kl
    else:
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_PARENT_KL * parent_kl
    count, routes = record_route(route, state["contract"])
    if count <= 8 or count % 128 == 0 or count == state["contract"].total_rows:
        print(
            f"[i37] microbatch={count}/{state['contract'].total_rows} route={route} "
            f"tokens={selected_targets.numel()} ce={float(ce.detach()):.6f} "
            f"parent_kl={float(parent_kl.detach()):.8f} loss={float(loss.detach()):.6f} "
            f"routes={dict(routes)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def textual_route(row: Mapping[str, Any]) -> str:
    output = str(row.get("output") or "")
    body = output.split("</think>", 1)[-1].strip()
    return "future_ce" if FUTURE_BODY_RE.fullmatch(body) else "retention_kl"


def run_data_preflight() -> Any:
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    contract = I36.verify_static_contract(require_data=True)
    if contract is None:
        raise RuntimeError("I37 contract unexpectedly missing")
    tokenizer = AutoTokenizer.from_pretrained(
        I36.BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    routes: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    maximum_tokens = 0
    prompt_hashes: set[str] = set()
    with TRAINING_DATA.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"I37 blank row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"I37 non-object row at line {line_number}")
            route = textual_route(row)
            if row.get("i37_route") != route:
                raise RuntimeError(f"I37 route drift at line {line_number}")
            task = str(row.get("task") or "")
            if route == "future_ce":
                expected_task = f"future_{FUTURE_BODY_RE.fullmatch(str(row['output']).split('</think>', 1)[-1].strip()).group(1)}"
                if task != expected_task:
                    raise RuntimeError(f"I37 future task drift at line {line_number}")
                if str(row["output"]).split("</think>", 1)[-1].strip() in str(row.get("input") or ""):
                    raise RuntimeError(f"I37 future target leaked at line {line_number}")
            elif task.startswith("future_"):
                raise RuntimeError(f"I37 future row reached retention route at line {line_number}")
            prompt_key = I36.sha256_bytes(
                (str(row.get("instruction") or "") + "\0" + str(row.get("input") or "")).encode("utf-8")
            ) if hasattr(I36, "sha256_bytes") else None
            if prompt_key is not None:
                if prompt_key in prompt_hashes:
                    raise RuntimeError(f"I37 duplicate prompt at line {line_number}")
                prompt_hashes.add(prompt_key)
            messages = [
                {"role": "user", "content": row.get("input")},
                {"role": "assistant", "content": row.get("output")},
            ]
            prompt_ids, response_ids = template.encode_oneturn(
                tokenizer, messages, row.get("instruction"), None
            )
            total_tokens = len(prompt_ids) + len(response_ids)
            if total_tokens > 16384:
                raise RuntimeError(f"I37 cutoff overflow at line {line_number}: {total_tokens}")
            token_route, _, _ = response_route(torch.tensor(response_ids, dtype=torch.long))
            if token_route != route:
                raise RuntimeError(
                    f"I37 token/text route drift at line {line_number}: {token_route}/{route}"
                )
            maximum_tokens = max(maximum_tokens, total_tokens)
            routes[route] += 1
            tasks[task] += 1
    if routes != Counter(EXPECTED_ROUTE_COUNTS) or tasks != Counter(EXPECTED_TASK_COUNTS):
        raise RuntimeError(f"I37 preflight drifted: routes={dict(routes)} tasks={dict(tasks)}")
    print(
        f"[i37] data preflight PASS: rows={contract.total_rows} routes={dict(routes)} "
        f"tasks={dict(tasks)} steps={contract.optimizer_steps} max_tokens={maximum_tokens} "
        f"data_sha256={contract.data_sha256}",
        flush=True,
    )
    return contract


def run_self_test() -> None:
    torch.manual_seed(37)
    future = torch.tensor(
        [151667, 198, I36.CLOSE_THINK_ID, 198, 176245, 151670, 159863, 168056, I36.EOS_ID]
    )
    retention = torch.tensor(
        [151667, 198, I36.CLOSE_THINK_ID, 198, 108386, 100, I36.EOS_ID]
    )
    if response_route(future)[0] != "future_ce" or response_route(retention)[0] != "retention_kl":
        raise AssertionError("I37 route self-test failed")
    policy = torch.randn(1, 17, 41, requires_grad=True)
    parent = torch.randn(1, 17, 41)
    targets = torch.randint(0, 41, (17,))
    weights = torch.linspace(1.0, 2.0, 17)
    loss = I36.weighted_ce(policy, targets, weights) + I36.forward_kl(policy, parent)
    loss.backward()
    if policy.grad is None or not torch.isfinite(policy.grad).all():
        raise AssertionError("I37 gradient self-test failed")
    print("[i37] self-test PASS: future routing, guarded CE/KL, and gradients", flush=True)


def assert_final_contract(contract: Any) -> None:
    count = int(getattr(i37_loss, "call_count", 0))
    routes = Counter(getattr(i37_loss, "route_counts", Counter()))
    if count != contract.total_rows or routes != Counter(EXPECTED_ROUTE_COUNTS):
        raise RuntimeError(f"I37 final route contract failed: {count}/{dict(routes)}")


def assert_policy_only_save_artifacts_i37(output_dir: Path, safe_serialization: bool) -> None:
    """Validate the fresh r8 payload without I34's hard-coded r16 constant."""
    expected_weights = I36.ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else I36.ADAPTER_WEIGHTS_NAME
    config_path = output_dir / I36.ADAPTER_CONFIG_NAME
    weights_path = output_dir / expected_weights
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"I37 policy-only save is incomplete: {output_dir}")
    alternate = output_dir / (I36.ADAPTER_WEIGHTS_NAME if safe_serialization else I36.ADAPTER_SAFE_WEIGHTS_NAME)
    if alternate.exists():
        raise RuntimeError(f"I37 save has a stale alternate adapter payload: {alternate}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or int(config.get("r", -1)) != 8 or int(config.get("lora_alpha", -1)) != 8:
        raise RuntimeError("I37 saved adapter rank/alpha drifted")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern"):
        if config.get(key) not in (None, {}, []):
            raise RuntimeError(f"I37 saved adapter contains unsupported {key}")
    if safe_serialization:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            keys = list(source.keys())
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        keys = list(state)
    if not keys or any("embed" in key or "lm_head" in key or "reference" in key.lower() for key in keys):
        raise RuntimeError("I37 saved adapter contains non-policy tensors")
    if any("lora_" not in key.lower() for key in keys):
        raise RuntimeError("I37 saved adapter contains a non-LoRA tensor")


I36.load_contract = load_contract
I36.ensure_runtime = ensure_runtime
I36.i36_loss = i37_loss
I36.run_data_preflight = run_data_preflight
I36.run_self_test = run_self_test
I36.assert_final_contract = assert_final_contract
I36.I34.assert_policy_only_save_artifacts = assert_policy_only_save_artifacts_i37


if __name__ == "__main__":
    I36.main()
