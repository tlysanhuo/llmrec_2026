#!/usr/bin/env python3
"""Directly continue every trainable tensor of the I-35 step548 r112 adapter.

Unlike I-36, this trainer does not merge I-35 and does not create a fresh
residual.  The adapter loaded by LLaMA-Factory is the trainable policy itself.
At optimizer creation time, a second copy of the same adapter is loaded under
the name ``i40_reference`` on the same frozen base model.  Adapter switching
therefore provides exact frozen-I35 reference logits without duplicating the
0.8B base model.  Only the original policy adapter is in the optimizer and
only that adapter is written to checkpoints.
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
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]


def load_i36_helpers() -> Any:
    path = Path(__file__).with_name("train_i36_i35_user_expand_retkl.py")
    spec = importlib.util.spec_from_file_location("llmrec_i36_helpers_for_i40", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


I36 = load_i36_helpers()

BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT_ADAPTER = Path(
    os.environ.get(
        "I40_PARENT_ADAPTER",
        str(ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"),
    )
)
TRAINING_DATA = Path(
    os.environ.get(
        "I40_TRAINING_DATA",
        str(
            ROOT
            / "assets/derived/processed/data_i40_i35_direct_user_continue_v1.jsonl"
        ),
    )
)
SIDECAR = Path(
    os.environ.get(
        "I40_SIDECAR",
        str(
            ROOT
            / "assets/derived/processed/data_i40_i35_direct_user_continue_v1_sidecar.jsonl"
        ),
    )
)
AUDIT = Path(
    os.environ.get(
        "I40_AUDIT",
        str(ROOT / "logs/data/i40_i35_direct_user_continue_v1_audit.json"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "I40_OUTPUT_DIR",
        str(ROOT / "checkpoints/i40_i35_direct_user_continue_r112_v1"),
    )
)

SCHEMA_VERSION = "i40-i35-direct-user-continue-r112-v1"
REFERENCE_ADAPTER_NAME = "i40_reference"
SEED = 19260840
EXPECTED_ROWS = 8_240
EXPECTED_STEPS = 2_060
EXPECTED_SAVE_STEPS = 515
EXPECTED_RANK = 112
EXPECTED_ALPHA = 112
EXPECTED_TENSORS = 392
EXPECTED_TRAINABLE_PARAMETERS = 70_647_808
EXPECTED_ROUTES = {"user_ce": 5_500, "retention_kl": 2_740}
EXPECTED_ROUTE_TASKS = {
    "user_ce": {"action": 4_000, "topic": 1_500},
    "retention_kl": {
        "material_desc2sid": 1_370,
        "action": 207,
        "topic": 206,
        "rec_video": 206,
        "rec_prod": 207,
        "rec_ad": 206,
        "rec_living": 207,
        "world": 131,
    },
}
EXPECTED_TASKS = {
    "material_desc2sid": 1_370,
    "action": 4_207,
    "topic": 1_706,
    "rec_video": 206,
    "rec_prod": 207,
    "rec_ad": 206,
    "rec_living": 207,
    "world": 131,
}

BASE_CONFIG_SHA256 = (
    "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
)
PARENT_ADAPTER_SHA256 = (
    "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
)
PARENT_CONFIG_SHA256 = (
    "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"
)
USER_CE_WEIGHT = 0.05
USER_PARENT_KL = 16.0
RETENTION_PARENT_KL = 16.0
MAX_ANSWER_POSITIONS = 128
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FormalContract:
    total_rows: int
    optimizer_steps: int
    route_counts: dict[str, int]
    route_task_counts: dict[str, dict[str, int]]
    task_counts: dict[str, int]
    data_sha256: str
    sidecar_sha256: str
    sidecar_rows: int
    sidecar_unique_keys: int
    seed: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def token_hash(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def routing_token_hash(prompt_ids: Sequence[int], response_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        canonical(
            {
                "prompt_token_sha256": token_hash(prompt_ids),
                "response_token_sha256": token_hash(response_ids),
            }
        ).encode("utf-8")
    ).hexdigest()


def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(
            row.get("input", row.get("prompt", row.get("user", ""))) or ""
        ),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def core_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(normalized(row)).encode("utf-8")).hexdigest()


def valid_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise RuntimeError(f"I40 {field} must be a lowercase SHA256")
    return value


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"I40 {field} must be a positive integer")
    return int(value)


def load_contract() -> FormalContract:
    if not AUDIT.is_file():
        raise RuntimeError(f"I40 formal audit is missing: {AUDIT}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("formal_training_generated") is not True
        or audit.get("status") != "FORMAL_DATA_FROZEN_TRAINING_AUTHORIZED"
    ):
        raise RuntimeError("I40 formal audit does not authorize training")
    if audit.get("seed") != SEED:
        raise RuntimeError(f"I40 seed drifted: {audit.get('seed')!r}")
    if audit.get("intersections") != {
        "user_retention_exact_prompt": 0,
        "user_retention_mode_prompt": 0,
        "forbidden_E_rows": 0,
        "third_party_rows": 0,
    }:
        raise RuntimeError("I40 formal intersection contract drifted")

    mix = audit.get("mix")
    outputs = audit.get("outputs")
    tokenization = audit.get("tokenization")
    sidecar_contract = audit.get("sidecar_contract")
    if not all(
        isinstance(value, dict)
        for value in (mix, outputs, tokenization, sidecar_contract)
    ):
        raise RuntimeError("I40 audit lacks mix/output/tokenization contracts")
    total_rows = positive_int(mix.get("total_rows"), "mix.total_rows")
    optimizer_steps = positive_int(
        mix.get("optimizer_steps_batch1_acc4"), "mix.optimizer_steps"
    )
    routes = mix.get("routes")
    if not isinstance(routes, dict):
        raise RuntimeError("I40 audit routes are missing")
    route_counts = {
        str(route): positive_int(entry.get("rows"), f"route.{route}.rows")
        for route, entry in routes.items()
        if isinstance(entry, dict)
    }
    route_tasks = {
        str(route): {
            str(task): positive_int(count, f"route.{route}.task.{task}")
            for task, count in dict(entry.get("by_task") or {}).items()
        }
        for route, entry in routes.items()
        if isinstance(entry, dict)
    }
    task_counts = {
        str(task): positive_int(count, f"task.{task}")
        for task, count in dict(mix.get("aggregate_task_counts") or {}).items()
    }
    if (
        total_rows != EXPECTED_ROWS
        or optimizer_steps != EXPECTED_STEPS
        or route_counts != EXPECTED_ROUTES
        or route_tasks != EXPECTED_ROUTE_TASKS
        or task_counts != EXPECTED_TASKS
    ):
        raise RuntimeError(
            "I40 formal mix drifted: "
            f"rows={total_rows} steps={optimizer_steps} routes={route_counts} "
            f"route_tasks={route_tasks} tasks={task_counts}"
        )
    if (
        routes["user_ce"].get("objective")
        != "0.05 weighted answer CE + 16.0 I35 parent KL"
        or routes["retention_kl"].get("objective")
        != "16.0 I35 parent KL only; max 128 answer positions"
        or routes["retention_kl"].get("old_i35_objective_reused") is not False
    ):
        raise RuntimeError("I40 loss semantics drifted in formal audit")

    data_entry = outputs.get("training_data")
    sidecar_entry = outputs.get("sidecar")
    if not isinstance(data_entry, dict) or not isinstance(sidecar_entry, dict):
        raise RuntimeError("I40 output entries are missing")
    expected_data_path = Path(str(data_entry.get("path") or ""))
    expected_sidecar_path = Path(str(sidecar_entry.get("path") or ""))
    if not expected_data_path.is_absolute():
        expected_data_path = ROOT / expected_data_path
    if not expected_sidecar_path.is_absolute():
        expected_sidecar_path = ROOT / expected_sidecar_path
    if (
        expected_data_path.resolve() != TRAINING_DATA.resolve()
        or expected_sidecar_path.resolve() != SIDECAR.resolve()
    ):
        raise RuntimeError("I40 formal output path drifted")
    data_rows = positive_int(data_entry.get("rows"), "outputs.training_data.rows")
    sidecar_rows = positive_int(sidecar_entry.get("rows"), "outputs.sidecar.rows")
    data_hash = valid_sha(data_entry.get("sha256"), "training_data.sha256")
    sidecar_hash = valid_sha(sidecar_entry.get("sha256"), "sidecar.sha256")
    unique_keys = positive_int(
        tokenization.get("unique_routing_token_hashes"),
        "tokenization.unique_routing_token_hashes",
    )
    if data_rows != EXPECTED_ROWS or sidecar_rows != EXPECTED_ROWS:
        raise RuntimeError("I40 data and sidecar must each contain 8,240 rows")
    if (
        tokenization.get("rows") != EXPECTED_ROWS
        or tokenization.get("duplicate_routing_token_exposures") != 25
        or tokenization.get("maximum_routing_token_exposure") != 2
        or tokenization.get("maximum_qwen3_nothink_tokens") != 8_864
        or unique_keys != 8_215
    ):
        raise RuntimeError("I40 tokenization/exposure contract drifted")
    if sidecar_contract.get("routes") != EXPECTED_ROUTES:
        raise RuntimeError("I40 sidecar route contract drifted")
    return FormalContract(
        total_rows=total_rows,
        optimizer_steps=optimizer_steps,
        route_counts=route_counts,
        route_task_counts=route_tasks,
        task_counts=task_counts,
        data_sha256=data_hash,
        sidecar_sha256=sidecar_hash,
        sidecar_rows=sidecar_rows,
        sidecar_unique_keys=unique_keys,
        seed=SEED,
    )


def verify_static_contract(require_data: bool = True) -> FormalContract | None:
    checks = {
        BASE / "config.json": BASE_CONFIG_SHA256,
        PARENT_ADAPTER / I36.ADAPTER_SAFE_WEIGHTS_NAME: PARENT_ADAPTER_SHA256,
        PARENT_ADAPTER / I36.ADAPTER_CONFIG_NAME: PARENT_CONFIG_SHA256,
    }
    for path, wanted in checks.items():
        if not path.is_file() or sha256(path) != wanted:
            raise RuntimeError(f"I40 frozen model artifact is missing or drifted: {path}")
    parent_config = json.loads(
        (PARENT_ADAPTER / I36.ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
    )
    I36.I34.assert_lora_config_contract(
        parent_config, EXPECTED_RANK, EXPECTED_ALPHA, "I40 I35 policy"
    )
    if not require_data:
        return None
    contract = load_contract()
    if not TRAINING_DATA.is_file() or sha256(TRAINING_DATA) != contract.data_sha256:
        raise RuntimeError("I40 training data is missing or hash-drifted")
    if not SIDECAR.is_file() or sha256(SIDECAR) != contract.sidecar_sha256:
        raise RuntimeError("I40 routing sidecar is missing or hash-drifted")
    return contract


def load_sidecar(
    expected_hash: str,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    if not SIDECAR.is_file() or sha256(SIDECAR) != expected_hash:
        raise RuntimeError("I40 sidecar is missing or hash-drifted")
    entries: dict[str, dict[str, Any]] = {}
    exposures: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    route_tasks: Counter[str] = Counter()
    with SIDECAR.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"I40 blank sidecar row at line {line_number}")
            entry = json.loads(line)
            if not isinstance(entry, dict) or entry.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"I40 sidecar schema drift at line {line_number}")
            key = valid_sha(entry.get("routing_token_sha256"), "routing_token_sha256")
            valid_sha(entry.get("prompt_token_sha256"), "prompt_token_sha256")
            valid_sha(entry.get("response_token_sha256"), "response_token_sha256")
            valid_sha(entry.get("row_sha256"), "row_sha256")
            valid_sha(entry.get("prompt_sha256"), "prompt_sha256")
            if (
                entry.get("parent_adapter_sha256") != PARENT_ADAPTER_SHA256
                or entry.get("parent_config_sha256") != PARENT_CONFIG_SHA256
            ):
                raise RuntimeError(f"I40 parent identity drift at sidecar line {line_number}")
            route = str(entry.get("route"))
            task = str(entry.get("task"))
            if task not in EXPECTED_ROUTE_TASKS.get(route, {}):
                raise RuntimeError(f"I40 sidecar route/task drift at line {line_number}")
            if route == "retention_kl":
                source = entry.get("source")
                if (
                    not isinstance(source, dict)
                    or source.get("asset") != "data_i35_video_boundary_retkl_v1"
                    or source.get("objective_reuse") is not False
                    or source.get("semantics")
                    != "KL-only against frozen I35 step548"
                ):
                    raise RuntimeError(
                        f"I40 retention source semantics drift at line {line_number}"
                    )
            elif route == "user_ce":
                source = entry.get("source")
                if (
                    not isinstance(source, dict)
                    or source.get("asset")
                    != "data_i36_i35_user_expand_retkl_v1"
                    or source.get("source_route") != "user_ce"
                ):
                    raise RuntimeError(
                        f"I40 user source semantics drift at line {line_number}"
                    )
            if key in entries:
                previous = entries[key]
                for field in (
                    "prompt_token_sha256",
                    "response_token_sha256",
                    "row_sha256",
                    "prompt_sha256",
                    "route",
                    "task",
                ):
                    if previous.get(field) != entry.get(field):
                        raise RuntimeError(
                            f"I40 duplicate routing key disagrees at line {line_number}"
                        )
            else:
                entries[key] = entry
            exposures[key] += 1
            if exposures[key] > 2:
                raise RuntimeError(f"I40 sidecar exposure exceeds two: {key}")
            routes[route] += 1
            route_tasks[f"{route}:{task}"] += 1
    expected_route_tasks = Counter(
        {
            f"{route}:{task}": count
            for route, tasks in EXPECTED_ROUTE_TASKS.items()
            for task, count in tasks.items()
        }
    )
    if (
        sum(exposures.values()) != EXPECTED_ROWS
        or len(entries) != 8_215
        or sum(count - 1 for count in exposures.values()) != 25
        or routes != Counter(EXPECTED_ROUTES)
        or route_tasks != expected_route_tasks
    ):
        raise RuntimeError(
            f"I40 sidecar multiset drifted: rows={sum(exposures.values())} "
            f"keys={len(entries)} routes={dict(routes)} route_tasks={dict(route_tasks)}"
        )
    return entries, exposures


def assert_formal_trainer_args(trainer: Any, contract: FormalContract) -> None:
    args = trainer.args
    observed = {
        "batch": int(args.per_device_train_batch_size),
        "accum": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
    }
    expected = {
        "batch": 1,
        "accum": 4,
        "max_steps": contract.optimizer_steps,
        "world_size": 1,
    }
    if observed != expected:
        raise RuntimeError(f"I40 trainer dimensions drifted: {observed}/{expected}")
    checks = {
        "learning_rate": (float(args.learning_rate), 5.0e-7),
        "warmup_ratio": (float(args.warmup_ratio), 0.03),
        "weight_decay": (float(args.weight_decay), 0.001),
        "max_grad_norm": (float(args.max_grad_norm), 0.5),
    }
    for name, (observed_value, expected_value) in checks.items():
        if not math.isclose(
            observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"I40 {name} drifted: {observed_value}/{expected_value}"
            )
    scheduler = getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)
    save_strategy = getattr(args.save_strategy, "value", args.save_strategy)
    if str(scheduler) != "cosine" or str(save_strategy) != "steps":
        raise RuntimeError("I40 requires cosine scheduling and step checkpoints")
    if (
        int(args.save_steps) != EXPECTED_SAVE_STEPS
        or int(args.save_total_limit) != 4
        or not bool(args.save_only_model)
    ):
        raise RuntimeError("I40 checkpoint cadence/model-only contract drifted")
    if (
        bool(getattr(args, "packing", False))
        or not bool(args.bf16)
        or int(args.seed) != SEED
    ):
        raise RuntimeError("I40 packing/BF16/seed contract drifted")
    if Path(str(args.output_dir)).resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"I40 output path drifted: {args.output_dir}/{OUTPUT_DIR}")
    reports = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in reports or os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("I40 requires online W&B")


def adapter_parameter_contract(
    unwrapped: Any, adapter_name: str, *, require_trainable: bool
) -> tuple[dict[str, torch.nn.Parameter], frozenset[int]]:
    params = I36.I34.adapter_parameters(unwrapped, adapter_name)
    if len(params) != EXPECTED_TENSORS:
        raise RuntimeError(
            f"I40 adapter {adapter_name!r} tensor count drifted: {len(params)}"
        )
    total = sum(parameter.numel() for parameter in params.values())
    if total != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"I40 adapter {adapter_name!r} parameter count drifted: {total}"
        )
    trainable = frozenset(
        id(parameter) for parameter in params.values() if parameter.requires_grad
    )
    expected = frozenset(id(parameter) for parameter in params.values())
    if require_trainable and trainable != expected:
        raise RuntimeError(f"I40 policy adapter {adapter_name!r} is not fully trainable")
    if not require_trainable and trainable:
        raise RuntimeError(f"I40 reference adapter {adapter_name!r} is trainable")
    return params, expected


def activate_policy(unwrapped: Any, policy_name: str) -> None:
    unwrapped.set_adapter(policy_name)
    unwrapped.set_requires_grad(REFERENCE_ADAPTER_NAME, requires_grad=False)
    I36.I34.assert_frozen_embeddings_and_head(unwrapped)
    adapter_parameter_contract(unwrapped, policy_name, require_trainable=True)
    adapter_parameter_contract(
        unwrapped, REFERENCE_ADAPTER_NAME, require_trainable=False
    )
    I36.I34.assert_exact_policy_trainable_parameters(unwrapped, policy_name)


def activate_reference(unwrapped: Any, policy_name: str) -> None:
    unwrapped.set_adapter(REFERENCE_ADAPTER_NAME)
    unwrapped.set_requires_grad(REFERENCE_ADAPTER_NAME, requires_grad=False)
    _policy, policy_ids = adapter_parameter_contract(
        unwrapped, policy_name, require_trainable=False
    )
    _reference, reference_ids = adapter_parameter_contract(
        unwrapped, REFERENCE_ADAPTER_NAME, require_trainable=False
    )
    if policy_ids & reference_ids:
        raise RuntimeError("I40 policy/reference adapter parameter identity overlaps")
    if any(parameter.requires_grad for parameter in unwrapped.parameters()):
        raise RuntimeError("I40 reference forward must have no trainable parameters")


def assert_runtime_state(unwrapped: Any, policy_name: str) -> None:
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or set(configs) != {
        policy_name,
        REFERENCE_ADAPTER_NAME,
    }:
        raise RuntimeError(f"I40 adapter set drifted: {list(configs or {})}")
    I36.I34.assert_lora_config_contract(
        configs[policy_name], EXPECTED_RANK, EXPECTED_ALPHA, "I40 policy"
    )
    I36.I34.assert_lora_config_contract(
        configs[REFERENCE_ADAPTER_NAME],
        EXPECTED_RANK,
        EXPECTED_ALPHA,
        "I40 reference",
    )
    active = I36.I34._single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if active != policy_name:
        raise RuntimeError(f"I40 active policy drifted: {active!r}/{policy_name!r}")
    activate_policy(unwrapped, policy_name)


def ensure_runtime(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i40_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        assert_runtime_state(unwrapped, state["policy_name"])
        return unwrapped, state

    contract = load_contract()
    assert_formal_trainer_args(trainer, contract)
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError(
            "I40 must start with exactly one resumed I35 policy adapter"
        )
    policy_name = I36.I34._single_adapter_name(
        getattr(unwrapped, "active_adapter", None)
    )
    if policy_name not in configs or policy_name == REFERENCE_ADAPTER_NAME:
        raise RuntimeError("I40 resumed policy adapter is not active/configured")
    I36.I34.assert_lora_config_contract(
        configs[policy_name], EXPECTED_RANK, EXPECTED_ALPHA, "I40 resumed I35 policy"
    )
    I36.I34.assert_frozen_embeddings_and_head(unwrapped)
    policy_params, policy_ids = adapter_parameter_contract(
        unwrapped, policy_name, require_trainable=True
    )
    I36.I34.assert_exact_policy_trainable_parameters(unwrapped, policy_name)

    unwrapped.load_adapter(
        str(PARENT_ADAPTER),
        adapter_name=REFERENCE_ADAPTER_NAME,
        is_trainable=False,
    )
    unwrapped.train()
    activate_policy(unwrapped, policy_name)
    reference_params, _ = adapter_parameter_contract(
        unwrapped, REFERENCE_ADAPTER_NAME, require_trainable=False
    )
    if set(policy_params) != set(reference_params):
        raise RuntimeError("I40 policy/reference tensor keys differ")
    if int(getattr(trainer.state, "global_step", 0)) != 0:
        raise RuntimeError("I40 reference must be initialized before optimizer step 1")
    maximum_before_copy = 0.0
    for key in policy_params:
        difference = (
            policy_params[key].detach().float()
            - reference_params[key].detach().float()
        ).abs()
        maximum_before_copy = max(maximum_before_copy, float(difference.max()))

    # LLaMA-Factory resumes the trainable adapter in the model compute dtype and
    # then upcasts it to FP32, while PEFT's later frozen-adapter load can preserve
    # the source FP32 values directly.  Both loads originate from the same
    # registered I-35 file, but the first route may already contain the exact
    # BF16 round-trip that defines the runtime policy.  Snapshot that actual
    # step-0 policy into the frozen reference before any optimizer step so the
    # KL anchor is bit-identical to the model being continued.
    with torch.no_grad():
        for key in policy_params:
            reference_params[key].copy_(policy_params[key].detach())
    maximum_after_copy = 0.0
    for key in policy_params:
        difference = (
            policy_params[key].detach().float()
            - reference_params[key].detach().float()
        ).abs()
        maximum_after_copy = max(maximum_after_copy, float(difference.max()))
    if maximum_after_copy != 0.0:
        raise RuntimeError(
            "I40 exact policy snapshot into reference failed: "
            f"max_abs={maximum_after_copy}"
        )
    sidecar, sidecar_exposures = load_sidecar(contract.sidecar_sha256)
    state = {
        "policy_name": policy_name,
        "policy_parameter_ids": policy_ids,
        "contract": contract,
        "sidecar": sidecar,
        "sidecar_exposures": sidecar_exposures,
        "fingerprint_checked": False,
    }
    trainer._i40_state = state
    print(
        f"[i40] contract PASS: direct trainable I35 r112 + frozen adapter-copy "
        f"reference; policy_tensors={len(policy_ids)} "
        f"policy_parameters={EXPECTED_TRAINABLE_PARAMETERS} "
        f"load_path_max_abs={maximum_before_copy:.8f} "
        f"snapshot_max_abs={maximum_after_copy:.8f} "
        f"rows={contract.total_rows} steps={contract.optimizer_steps}",
        flush=True,
    )
    return unwrapped, state


def selected_logits(model: Any, inputs: Mapping[str, Any], positions: torch.Tensor) -> Any:
    try:
        return model(**inputs, logits_to_keep=positions)
    except TypeError as exc:
        raise RuntimeError("I40 model must support logits_to_keep") from exc


def paired_reference_policy(
    trainer: Any,
    model: Any,
    inputs: Mapping[str, Any],
    positions: torch.Tensor,
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    unwrapped, state = ensure_runtime(trainer, model)
    policy_name = state["policy_name"]
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        activate_reference(unwrapped, policy_name)
        with torch.no_grad():
            reference_outputs = selected_logits(model, inputs, positions)
            reference_logits = reference_outputs.logits.detach()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        activate_policy(unwrapped, policy_name)
    policy_outputs = selected_logits(model, inputs, positions)
    if not state["fingerprint_checked"]:
        maximum = float(
            (
                policy_outputs.logits.detach().float() - reference_logits.float()
            )
            .abs()
            .max()
        )
        if maximum > 1e-4:
            raise RuntimeError(
                f"I40 step-0 I35 reference fingerprint failed: max_abs={maximum:.8f}"
            )
        state["fingerprint_checked"] = True
        print(
            f"[i40] step-0 I35 reference fingerprint PASS: max_abs={maximum:.8f}",
            flush=True,
        )
    return policy_outputs, reference_logits, state


def record_route(
    route: str, task: str, contract: FormalContract
) -> tuple[int, Counter[str], Counter[str]]:
    count = int(getattr(i40_loss, "call_count", 0)) + 1
    routes = Counter(getattr(i40_loss, "route_counts", Counter()))
    route_tasks = Counter(getattr(i40_loss, "route_task_counts", Counter()))
    routes[route] += 1
    route_tasks[f"{route}:{task}"] += 1
    expected_routes = Counter(contract.route_counts)
    expected_route_tasks = Counter(
        {
            f"{route_name}:{task_name}": value
            for route_name, tasks in contract.route_task_counts.items()
            for task_name, value in tasks.items()
        }
    )
    if count > contract.total_rows:
        raise RuntimeError("I40 observed too many microbatches")
    remaining = contract.total_rows - count
    for observed, expected, label in (
        (routes, expected_routes, "route"),
        (route_tasks, expected_route_tasks, "route-task"),
    ):
        for name, value in observed.items():
            if value > expected[name]:
                raise RuntimeError(f"I40 {label} count exceeded {name}: {value}")
        for name, value in expected.items():
            if observed[name] + remaining < value:
                raise RuntimeError(f"I40 cannot satisfy remaining {label} {name}")
    i40_loss.call_count = count
    i40_loss.route_counts = routes
    i40_loss.route_task_counts = route_tasks
    return count, routes, route_tasks


def i40_loss(
    trainer: Any,
    model: Any,
    inputs: dict[str, Any],
    return_outputs: bool = False,
    **kwargs: Any,
) -> Any:
    del kwargs
    labels = inputs.pop("labels")
    target_start, target_end = I36.target_span(labels)
    response_targets = labels[0, target_start:target_end]
    prompt_ids = inputs["input_ids"][0, :target_start].detach().cpu().tolist()
    response_ids = response_targets.detach().cpu().tolist()
    key = routing_token_hash(prompt_ids, response_ids)
    _unwrapped, state = ensure_runtime(trainer, model)
    entry = state["sidecar"].get(key)
    if entry is None:
        raise RuntimeError(f"I40 runtime row is absent from sidecar: {key}")
    if (
        token_hash(prompt_ids) != entry["prompt_token_sha256"]
        or token_hash(response_ids) != entry["response_token_sha256"]
    ):
        raise RuntimeError(f"I40 runtime token signature drifted: {key}")
    route = str(entry["route"])
    task = str(entry["task"])

    body_start, body_end = I36.body_bounds(response_targets)
    relative = I36.uniformly_bounded_indices(
        body_start, body_end, MAX_ANSWER_POSITIONS
    )
    selected = relative.to(labels.device) + target_start
    positions = selected - 1
    targets = labels[0, selected]
    outputs, reference_logits, _ = paired_reference_policy(
        trainer, model, inputs, positions
    )
    policy_logits = outputs.logits
    parent_kl = I36.forward_kl(policy_logits, reference_logits)
    if route == "user_ce":
        if task not in {"action", "topic"}:
            raise RuntimeError(f"I40 user route has invalid task: {task}")
        weights = I36.terminal_weights(targets)
        ce = I36.weighted_ce(policy_logits, targets, weights)
        loss = USER_CE_WEIGHT * ce + USER_PARENT_KL * parent_kl
    elif route == "retention_kl":
        ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_PARENT_KL * parent_kl
    else:
        raise RuntimeError(f"I40 unknown runtime route: {route}")
    count, routes, _route_tasks = record_route(route, task, state["contract"])
    if count <= 8 or count % 200 == 0 or count == state["contract"].total_rows:
        print(
            f"[i40] microbatch={count}/{state['contract'].total_rows} "
            f"route={route} task={task} tokens={targets.numel()} "
            f"ce={float(ce.detach()):.6f} "
            f"parent_kl={float(parent_kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} routes={dict(routes)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_data_preflight() -> FormalContract:
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    contract = verify_static_contract(require_data=True)
    assert contract is not None
    sidecar, sidecar_exposures = load_sidecar(contract.sidecar_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    routes: Counter[str] = Counter()
    route_tasks: Counter[str] = Counter()
    observed_exposures: Counter[str] = Counter()
    maximum_tokens = 0
    with TRAINING_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"I40 blank data row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"I40 non-object data row at line {line_number}")
            prompt_ids, response_ids = template.encode_oneturn(
                tokenizer,
                [
                    {"role": "user", "content": row.get("input")},
                    {"role": "assistant", "content": row.get("output")},
                ],
                row.get("instruction"),
                None,
            )
            total = len(prompt_ids) + len(response_ids)
            if total > 16_384:
                raise RuntimeError(f"I40 cutoff overflow at line {line_number}: {total}")
            key = routing_token_hash(prompt_ids, response_ids)
            entry = sidecar.get(key)
            if entry is None:
                raise RuntimeError(f"I40 data row missing from sidecar at line {line_number}")
            if (
                core_hash(row) != entry["row_sha256"]
                or token_hash(prompt_ids) != entry["prompt_token_sha256"]
                or token_hash(response_ids) != entry["response_token_sha256"]
            ):
                raise RuntimeError(f"I40 data/sidecar hash drift at line {line_number}")
            route = str(row.get("route"))
            task = str(row.get("task"))
            if route != entry["route"] or task != entry["task"]:
                raise RuntimeError(f"I40 data/sidecar route drift at line {line_number}")
            response_tensor = torch.tensor(response_ids, dtype=torch.long)
            I36.body_bounds(response_tensor)
            routes[route] += 1
            route_tasks[f"{route}:{task}"] += 1
            observed_exposures[key] += 1
            maximum_tokens = max(maximum_tokens, total)
    expected_route_tasks = Counter(
        {
            f"{route}:{task}": count
            for route, tasks in EXPECTED_ROUTE_TASKS.items()
            for task, count in tasks.items()
        }
    )
    if (
        routes != Counter(EXPECTED_ROUTES)
        or route_tasks != expected_route_tasks
        or observed_exposures != sidecar_exposures
        or maximum_tokens != 8_864
    ):
        raise RuntimeError(
            f"I40 data preflight drifted: routes={dict(routes)} "
            f"route_tasks={dict(route_tasks)} max_tokens={maximum_tokens}"
        )
    print(
        f"[i40] data preflight PASS: rows={contract.total_rows} "
        f"unique_routes={len(observed_exposures)} duplicate_exposures=25 "
        f"routes={dict(routes)} steps={contract.optimizer_steps} "
        f"max_tokens={maximum_tokens} data_sha256={contract.data_sha256} "
        f"sidecar_sha256={contract.sidecar_sha256}",
        flush=True,
    )
    return contract


def run_self_test() -> None:
    torch.manual_seed(40)
    policy = torch.randn(1, 4, 31, requires_grad=True)
    reference = torch.randn(1, 4, 31)
    targets = torch.tensor([1, 2, 3, 4])
    weights = I36.terminal_weights(targets)
    loss = (
        USER_CE_WEIGHT * I36.weighted_ce(policy, targets, weights)
        + USER_PARENT_KL * I36.forward_kl(policy, reference)
    )
    loss.backward()
    if policy.grad is None or not torch.isfinite(policy.grad).all():
        raise AssertionError("I40 CE/KL gradient self-test failed")
    if routing_token_hash([1, 2], [3, 4]) == routing_token_hash([1], [2, 3, 4]):
        raise AssertionError("I40 routing hash boundary self-test failed")
    aggregate: Counter[str] = Counter()
    for tasks in EXPECTED_ROUTE_TASKS.values():
        aggregate.update(tasks)
    if dict(aggregate) != EXPECTED_TASKS:
        raise AssertionError("I40 task aggregation self-test failed")
    print(
        "[i40] self-test PASS: bounded user CE, parent KL, multiset routing, and gradients",
        flush=True,
    )


def reset_counters() -> None:
    for name in ("call_count", "route_counts", "route_task_counts"):
        if hasattr(i40_loss, name):
            delattr(i40_loss, name)


def assert_final_contract(contract: FormalContract) -> None:
    count = int(getattr(i40_loss, "call_count", 0))
    routes = Counter(getattr(i40_loss, "route_counts", Counter()))
    route_tasks = Counter(getattr(i40_loss, "route_task_counts", Counter()))
    expected_route_tasks = Counter(
        {
            f"{route}:{task}": count
            for route, tasks in contract.route_task_counts.items()
            for task, count in tasks.items()
        }
    )
    if (
        count != contract.total_rows
        or routes != Counter(contract.route_counts)
        or route_tasks != expected_route_tasks
    ):
        raise RuntimeError(
            f"I40 final route contract failed: calls={count}/{contract.total_rows} "
            f"routes={dict(routes)} route_tasks={dict(route_tasks)}"
        )


def assert_policy_only_save(
    output_dir: Path, safe_serialization: bool
) -> None:
    expected_weights = (
        I36.ADAPTER_SAFE_WEIGHTS_NAME
        if safe_serialization
        else I36.ADAPTER_WEIGHTS_NAME
    )
    config_path = output_dir / I36.ADAPTER_CONFIG_NAME
    weights_path = output_dir / expected_weights
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"I40 policy-only save is incomplete: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    I36.I34.assert_lora_config_contract(
        config, EXPECTED_RANK, EXPECTED_ALPHA, "I40 saved policy"
    )
    reference_payloads = list(output_dir.glob(f"**/{REFERENCE_ADAPTER_NAME}/adapter_*"))
    if reference_payloads:
        raise RuntimeError(f"I40 checkpoint leaked reference adapter: {reference_payloads}")
    if safe_serialization:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            keys = list(source.keys())
            total = sum(source.get_tensor(key).numel() for key in keys)
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        keys = list(state)
        total = sum(value.numel() for value in state.values())
    if (
        len(keys) != EXPECTED_TENSORS
        or total != EXPECTED_TRAINABLE_PARAMETERS
        or any("reference" in key.lower() for key in keys)
        or any("lora_" not in key.lower() for key in keys)
    ):
        raise RuntimeError(
            f"I40 saved payload drifted: tensors={len(keys)} parameters={total}"
        )


def policy_only_save_pretrained(
    original_save: Any,
    peft_model: Any,
    save_directory: str,
    *args: Any,
    selected_adapters: Any = None,
    **kwargs: Any,
) -> Any:
    configs = getattr(peft_model, "peft_config", None)
    if not isinstance(configs, dict) or set(configs) != {
        "default",
        REFERENCE_ADAPTER_NAME,
    }:
        raise RuntimeError(f"I40 save adapter set drifted: {list(configs or {})}")
    policy_name = "default"
    if selected_adapters not in (None, [policy_name]):
        raise RuntimeError(f"I40 unsafe adapter save selection: {selected_adapters!r}")
    activate_policy(peft_model, policy_name)
    output_dir = Path(save_directory)
    stale = [
        output_dir / name
        for name in (
            I36.ADAPTER_CONFIG_NAME,
            I36.ADAPTER_SAFE_WEIGHTS_NAME,
            I36.ADAPTER_WEIGHTS_NAME,
        )
        if (output_dir / name).exists()
    ]
    if stale:
        raise RuntimeError(f"I40 refuses to overwrite adapter payload: {stale}")
    kwargs["selected_adapters"] = [policy_name]
    result = original_save(peft_model, save_directory, *args, **kwargs)
    assert_policy_only_save(output_dir, bool(kwargs.get("safe_serialization", True)))
    return result


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
        for name in (
            I36.ADAPTER_CONFIG_NAME,
            I36.ADAPTER_SAFE_WEIGHTS_NAME,
            I36.ADAPTER_WEIGHTS_NAME,
        )
    ):
        raise RuntimeError(f"I40 refuses to overwrite adapter output: {OUTPUT_DIR}")

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
                raise RuntimeError("I40 unexpected compute_loss positional arguments")
            kwargs["return_outputs"] = args[0]
        return i40_loss(self, model, inputs, **kwargs)

    def patched_create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        unwrapped, state = ensure_runtime(self, self.model)
        if not I36.I34.assert_optimizer_policy_only(
            self, unwrapped, state["policy_name"]
        ):
            raise RuntimeError("I40 optimizer is not policy-only")
        print(
            f"[i40] optimizer policy-only PASS: tensors={EXPECTED_TENSORS} "
            f"parameters={EXPECTED_TRAINABLE_PARAMETERS}",
            flush=True,
        )
        return optimizer

    def patched_trainer_save(self, output_dir=None, state_dict=None):
        unwrapped, state = ensure_runtime(self, self.model)
        activate_policy(unwrapped, state["policy_name"])
        if not I36.I34.assert_optimizer_policy_only(
            self, unwrapped, state["policy_name"]
        ):
            raise RuntimeError("I40 optimizer disappeared before save")
        result = original_trainer_save(
            self, output_dir=output_dir, state_dict=state_dict
        )
        saved = Path(output_dir if output_dir is not None else self.args.output_dir)
        assert_policy_only_save(
            saved, bool(getattr(self.args, "save_safetensors", True))
        )
        return result

    def patched_peft_save(
        peft_model, save_directory, *args, selected_adapters=None, **kwargs
    ):
        return policy_only_save_pretrained(
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
    print(
        f"[i40] training PASS: direct r112 rows={contract.total_rows} "
        f"steps={contract.optimizer_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
