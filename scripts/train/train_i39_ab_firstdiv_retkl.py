#!/usr/bin/env python3
"""Train the single guarded I-39 AB/first-divergence r8 residual."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def load_i36_runtime() -> Any:
    path = Path(__file__).with_name("train_i36_i35_user_expand_retkl.py")
    spec = importlib.util.spec_from_file_location("llmrec_i36_runtime_for_i39", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


I36 = load_i36_runtime()

PARENT_ADAPTER = Path(
    os.environ.get(
        "I39_PARENT_ADAPTER",
        str(ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"),
    )
)
TRAINING_DATA = Path(
    os.environ.get(
        "I39_TRAINING_DATA",
        str(ROOT / "assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1.jsonl"),
    )
)
SIDECAR = Path(
    os.environ.get(
        "I39_SIDECAR",
        str(
            ROOT
            / "assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1_sidecar.jsonl"
        ),
    )
)
AUDIT = Path(
    os.environ.get(
        "I39_AUDIT",
        str(ROOT / "logs/data/i39_i35_userab_firstdiv_retkl_v1_audit.json"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "I39_OUTPUT_DIR",
        str(ROOT / "checkpoints/i39_i35_userab_firstdiv_retkl_r8_v1"),
    )
)

SCHEMA_VERSION = "i39-i35-userab-firstdiv-retkl-v1"
SEED = 19260839
EXPECTED_ROWS = 2_560
EXPECTED_STEPS = 640
EXPECTED_RANK = 8
EXPECTED_ALPHA = 8
EXPECTED_ROUTES = {
    "material_firstdiv": 512,
    "user_micro_ce": 128,
    "retention_kl": 1_920,
}
EXPECTED_OBJECTIVES = {
    "a_firstdiv": 128,
    "b_firstdiv": 128,
    "c_firstdiv": 192,
    "full_anchor": 64,
}
EXPECTED_ROUTE_TASKS = {
    "material_firstdiv": {"material_desc2sid": 512},
    "user_micro_ce": {"action": 96, "topic": 32},
    "retention_kl": {
        "material_desc2sid": 128,
        "material_sid2desc": 128,
        "action": 256,
        "topic": 256,
        "rec_video": 240,
        "rec_prod": 240,
        "rec_ad": 240,
        "rec_living": 240,
        "world": 192,
    },
}
EXPECTED_TASK_COUNTS = {
    "material_desc2sid": 640,
    "material_sid2desc": 128,
    "action": 352,
    "topic": 288,
    "rec_video": 240,
    "rec_prod": 240,
    "rec_ad": 240,
    "rec_living": 240,
    "world": 192,
}

VIDEO_DOMAIN_ID = 176245
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244
MATERIAL_MARGIN = 0.10
MATERIAL_MARGIN_WEIGHT = 0.50
MATERIAL_FOCUS_CE_WEIGHT = 0.02
MATERIAL_PARENT_KL = 8.0
ANCHOR_PARENT_KL = 16.0
USER_CE_WEIGHT = 0.05
USER_PARENT_KL = 16.0
RETENTION_PARENT_KL = 16.0
MAX_ANSWER_POSITIONS = 128


@dataclass(frozen=True)
class FormalContract:
    total_rows: int
    optimizer_steps: int
    route_counts: dict[str, int]
    task_counts: dict[str, int]
    data_sha256: str
    seed: int
    sidecar_sha256: str
    sidecar_rows: int
    objective_counts: dict[str, int]
    route_task_counts: dict[str, dict[str, int]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def core_hash(row: Mapping[str, Any]) -> str:
    normalized = {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", row.get("user", ""))) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }
    return hashlib.sha256(
        canonical(normalized).encode("utf-8")
    ).hexdigest()


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"I39 {field} must be a positive integer")
    return int(value)


def valid_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"I39 {field} must be a lowercase SHA256")
    return value


def resolve_output(entry: Mapping[str, Any], expected: Path, field: str) -> tuple[int, str]:
    path = Path(str(entry.get("path") or ""))
    if not path.is_absolute():
        path = ROOT / path
    if path.resolve() != expected.resolve():
        raise RuntimeError(f"I39 {field} path drifted: {path}/{expected}")
    return positive_int(entry.get("rows"), f"{field}.rows"), valid_sha(
        entry.get("sha256"), f"{field}.sha256"
    )


def load_contract() -> FormalContract:
    if not AUDIT.is_file():
        raise RuntimeError(f"I39 formal audit is missing: {AUDIT}")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"I39 schema drifted: {audit.get('schema_version')!r}")
    if audit.get("formal_training_generated") is not True:
        raise RuntimeError("I39 audit does not authorize formal training")
    if audit.get("seed") != SEED:
        raise RuntimeError(f"I39 seed drifted: {audit.get('seed')!r}")
    mix = audit.get("mix")
    outputs = audit.get("outputs")
    sidecar_contract = audit.get("sidecar_contract")
    if not isinstance(mix, dict) or not isinstance(outputs, dict):
        raise RuntimeError("I39 audit lacks mix/outputs")
    if not isinstance(sidecar_contract, dict):
        raise RuntimeError("I39 audit lacks sidecar contract")
    total_rows = positive_int(mix.get("total_rows"), "mix.total_rows")
    routes_raw = mix.get("routes")
    if not isinstance(routes_raw, dict):
        raise RuntimeError("I39 audit mix.routes is missing")
    route_counts = {
        str(route): positive_int(value.get("rows"), f"mix.routes.{route}.rows")
        for route, value in routes_raw.items()
        if isinstance(value, dict)
    }
    if total_rows != EXPECTED_ROWS or route_counts != EXPECTED_ROUTES:
        raise RuntimeError(f"I39 route contract drifted: {total_rows}/{route_counts}")
    objectives = dict(routes_raw["material_firstdiv"].get("by_objective") or {})
    user_tasks = dict(routes_raw["user_micro_ce"].get("by_task") or {})
    retention_tasks = dict(routes_raw["retention_kl"].get("by_task") or {})
    if objectives != EXPECTED_OBJECTIVES:
        raise RuntimeError(f"I39 objective contract drifted: {objectives}")
    route_tasks = {
        "material_firstdiv": {"material_desc2sid": route_counts["material_firstdiv"]},
        "user_micro_ce": user_tasks,
        "retention_kl": retention_tasks,
    }
    if route_tasks != EXPECTED_ROUTE_TASKS:
        raise RuntimeError(f"I39 route-task contract drifted: {route_tasks}")
    task_counts: Counter[str] = Counter()
    for values in route_tasks.values():
        task_counts.update({str(key): int(value) for key, value in values.items()})
    data_rows, data_hash = resolve_output(
        outputs.get("training_data") or {}, TRAINING_DATA, "training_data"
    )
    sidecar_rows, sidecar_hash = resolve_output(
        outputs.get("sidecar") or {}, SIDECAR, "sidecar"
    )
    if data_rows != total_rows or sidecar_rows != total_rows:
        raise RuntimeError("I39 data/sidecar must each cover all formal rows")
    if sidecar_contract.get("routes") != EXPECTED_ROUTES:
        raise RuntimeError("I39 sidecar route contract drifted")
    if sidecar_contract.get("objectives") != EXPECTED_OBJECTIVES:
        raise RuntimeError("I39 sidecar objective contract drifted")
    steps = math.ceil(total_rows / (I36.EXPECTED_BATCH * I36.EXPECTED_ACCUMULATION))
    if steps != EXPECTED_STEPS or mix.get("optimizer_steps_batch1_acc4") != steps:
        raise RuntimeError(f"I39 optimizer-step contract drifted: {steps}")
    return FormalContract(
        total_rows=total_rows,
        optimizer_steps=steps,
        route_counts=route_counts,
        task_counts=dict(task_counts),
        data_sha256=data_hash,
        seed=SEED,
        sidecar_sha256=sidecar_hash,
        sidecar_rows=sidecar_rows,
        objective_counts={str(key): int(value) for key, value in objectives.items()},
        route_task_counts={
            route: {str(key): int(value) for key, value in values.items()}
            for route, values in route_tasks.items()
        },
    )


def validate_gold_abc(tokens: Sequence[int], field: str) -> None:
    if len(tokens) != 3:
        raise RuntimeError(f"I39 {field} must contain three tokens")
    ranges = ((A_LO, A_HI), (B_LO, B_HI), (C_LO, C_HI))
    for index, (token, (lower, upper)) in enumerate(zip(tokens, ranges)):
        if isinstance(token, bool) or not isinstance(token, int) or not lower <= token <= upper:
            raise RuntimeError(f"I39 {field}[{index}] is outside its SID range")


def objective_focus(objective: str) -> int | None:
    return {"a_firstdiv": 0, "b_firstdiv": 1, "c_firstdiv": 2}.get(objective)


def load_sidecar(expected_hash: str) -> dict[str, dict[str, Any]]:
    if not SIDECAR.is_file() or sha256(SIDECAR) != expected_hash:
        raise RuntimeError("I39 full routing sidecar is missing or hash-drifted")
    entries: dict[str, dict[str, Any]] = {}
    routes: Counter[str] = Counter()
    objectives: Counter[str] = Counter()
    route_tasks: Counter[str] = Counter()
    with SIDECAR.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"I39 sidecar has a blank row at line {line_number}")
            entry = json.loads(line)
            if not isinstance(entry, dict) or entry.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"I39 sidecar schema drift at line {line_number}")
            key = valid_sha(entry.get("prompt_token_sha256"), "prompt_token_sha256")
            if key in entries:
                raise RuntimeError(f"I39 duplicate sidecar prompt token hash: {key}")
            valid_sha(entry.get("response_token_sha256"), "response_token_sha256")
            valid_sha(entry.get("row_sha256"), "row_sha256")
            valid_sha(entry.get("prompt_sha256"), "prompt_sha256")
            if entry.get("parent_adapter_sha256") != I36.PARENT_ADAPTER_SHA256:
                raise RuntimeError(f"I39 sidecar parent hash drift at line {line_number}")
            if entry.get("parent_config_sha256") != I36.PARENT_CONFIG_SHA256:
                raise RuntimeError(f"I39 sidecar parent config drift at line {line_number}")
            route = str(entry.get("route"))
            task = str(entry.get("task"))
            if task not in EXPECTED_ROUTE_TASKS.get(route, {}):
                raise RuntimeError(f"I39 sidecar route/task drift at line {line_number}")
            if route == "material_firstdiv":
                objective = str(entry.get("objective"))
                if objective not in EXPECTED_OBJECTIVES:
                    raise RuntimeError(f"I39 material objective drift at line {line_number}")
                focus = objective_focus(objective)
                if entry.get("focus_index") != focus:
                    raise RuntimeError(f"I39 material focus drift at line {line_number}")
                gold_tokens = entry.get("gold_tokens")
                if (
                    not isinstance(gold_tokens, list)
                    or len(gold_tokens) != 5
                    or gold_tokens[0] != VIDEO_DOMAIN_ID
                    or gold_tokens[-1] != I36.EOS_ID
                ):
                    raise RuntimeError(f"I39 material gold body drift at line {line_number}")
                gold_abc = list(gold_tokens[1:4])
                validate_gold_abc(gold_abc, "gold_abc")
                if entry.get("gold_abc") != gold_abc:
                    raise RuntimeError(f"I39 material gold aliases disagree at line {line_number}")
                negatives = entry.get("hard_negatives")
                if not isinstance(negatives, list):
                    raise RuntimeError(f"I39 hard negatives are invalid at line {line_number}")
                if focus is None and negatives:
                    raise RuntimeError(f"I39 full anchor has negatives at line {line_number}")
                if focus is not None and not negatives:
                    raise RuntimeError(f"I39 first-divergence row lacks negatives at line {line_number}")
                seen: set[tuple[int, int, int]] = set()
                for negative in negatives:
                    if not isinstance(negative, dict) or "teacher_score" in negative:
                        raise RuntimeError(f"I39 sidecar retained teacher metadata at line {line_number}")
                    tokens = negative.get("tokens")
                    if not isinstance(tokens, list):
                        raise RuntimeError(f"I39 negative tokens drift at line {line_number}")
                    validate_gold_abc(tokens, "hard_negative.tokens")
                    token_key = tuple(tokens)
                    if token_key in seen or list(tokens) == gold_abc:
                        raise RuntimeError(f"I39 duplicate/gold negative at line {line_number}")
                    seen.add(token_key)
                    divergence = I36.I34.first_divergence(gold_abc, list(tokens))
                    if divergence != focus or negative.get("first_divergence") != focus:
                        raise RuntimeError(f"I39 negative focus mismatch at line {line_number}")
                if len(negatives) > 4:
                    raise RuntimeError(f"I39 material row has too many focus negatives at line {line_number}")
                objectives[objective] += 1
            elif any(
                field in entry
                for field in (
                    "objective",
                    "focus_index",
                    "gold_tokens",
                    "gold_abc",
                    "hard_negatives",
                )
            ):
                raise RuntimeError(f"I39 nonmaterial sidecar has material fields at line {line_number}")
            routes[route] += 1
            route_tasks[f"{route}:{task}"] += 1
            entries[key] = entry
    expected_route_tasks = Counter(
        {
            f"{route}:{task}": count
            for route, tasks in EXPECTED_ROUTE_TASKS.items()
            for task, count in tasks.items()
        }
    )
    if len(entries) != EXPECTED_ROWS or routes != Counter(EXPECTED_ROUTES):
        raise RuntimeError(f"I39 sidecar route counts drifted: {len(entries)}/{dict(routes)}")
    if objectives != Counter(EXPECTED_OBJECTIVES):
        raise RuntimeError(f"I39 sidecar objectives drifted: {dict(objectives)}")
    if route_tasks != expected_route_tasks:
        raise RuntimeError(f"I39 sidecar route tasks drifted: {dict(route_tasks)}")
    return entries


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
        raise RuntimeError(f"I39 trainer contract drifted: {observed}/{expected}")
    checks = {
        "learning_rate": (float(args.learning_rate), 5.0e-6),
        "warmup_ratio": (float(args.warmup_ratio), 0.03),
        "weight_decay": (float(args.weight_decay), 0.001),
    }
    for name, (observed_value, expected_value) in checks.items():
        if not math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"I39 {name} drifted: {observed_value}/{expected_value}")
    scheduler = getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type)
    save_strategy = getattr(args.save_strategy, "value", args.save_strategy)
    if str(scheduler) != "cosine" or str(save_strategy) != "no":
        raise RuntimeError("I39 requires cosine scheduling and no intermediate checkpoints")
    if not bool(args.save_only_model) or bool(getattr(args, "packing", False)):
        raise RuntimeError("I39 requires adapter-only saving and packing=False")
    if not bool(args.bf16) or int(args.seed) != contract.seed:
        raise RuntimeError("I39 BF16/seed contract drifted")
    cutoff = getattr(args, "cutoff_len", getattr(args, "generation_max_length", -1))
    if int(cutoff) != 16_384:
        raise RuntimeError("I39 cutoff must be 16384")
    if Path(str(args.output_dir)).resolve() != OUTPUT_DIR.resolve():
        raise RuntimeError(f"I39 output path drifted: {args.output_dir}/{OUTPUT_DIR}")
    reports = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in reports or os.environ.get("WANDB_MODE", "online").lower() != "online":
        raise RuntimeError("I39 requires online W&B")


def ensure_runtime(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i39_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        I36.I34.assert_adapter_runtime_state(unwrapped, state["policy_name"])
        I36.I34.assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        return unwrapped, state
    contract = load_contract()
    assert_formal_trainer_args(trainer, contract)
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError("I39 expects one fresh adapter over the merged I35 r112 parent")
    policy_name = I36.I34._single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name not in configs:
        raise RuntimeError("I39 active policy adapter is not configured")
    I36.I34.assert_lora_config_contract(
        configs[policy_name], EXPECTED_RANK, EXPECTED_ALPHA, "I39 fresh policy"
    )
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("I39 requires PEFT disable_adapter for the merged parent")
    I36.I34.assert_frozen_embeddings_and_head(unwrapped)
    parameter_ids = I36.I34.assert_exact_policy_trainable_parameters(
        unwrapped, policy_name
    )
    I36.I34.assert_adapter_runtime_state(unwrapped, policy_name)
    sidecar = load_sidecar(contract.sidecar_sha256)
    state = {
        "policy_name": policy_name,
        "policy_parameter_ids": parameter_ids,
        "contract": contract,
        "sidecar": sidecar,
        "fingerprint_checked": False,
    }
    trainer._i39_state = state
    print(
        f"[i39] contract PASS: merged I35 r112 + fresh r8; "
        f"rows={contract.total_rows} steps={contract.optimizer_steps}",
        flush=True,
    )
    return unwrapped, state


def material_bounds(targets: torch.Tensor) -> tuple[int, int]:
    start, end, empty_think = I36.I34.response_body_bounds(
        targets.detach().cpu().tolist()
    )
    body = targets[start:end].detach().cpu().tolist()
    if not empty_think or len(body) != 5:
        raise RuntimeError("I39 material response must be empty-think video+A+B+C+EOS")
    if body[0] != VIDEO_DOMAIN_ID or body[-1] != I36.EOS_ID:
        raise RuntimeError("I39 material domain/EOS drifted")
    validate_gold_abc(body[1:4], "material body")
    return start, end


def focus_margin_loss(
    policy_logits: torch.Tensor,
    gold_abc: Sequence[int],
    negatives: Sequence[Mapping[str, Any]],
    focus: int,
) -> torch.Tensor:
    if policy_logits.shape[:2] != (1, 3) or focus not in (0, 1, 2):
        raise RuntimeError("I39 focus-margin shape/index drifted")
    log_probs = F.log_softmax(policy_logits.float(), dim=-1)[0, focus]
    losses: list[torch.Tensor] = []
    for negative in negatives:
        tokens = [int(value) for value in negative["tokens"]]
        if negative.get("first_divergence") != focus:
            raise RuntimeError("I39 runtime negative focus drifted")
        gap = log_probs[int(gold_abc[focus])] - log_probs[tokens[focus]]
        losses.append(F.softplus(MATERIAL_MARGIN - gap))
    if not losses:
        raise RuntimeError("I39 focus margin has no negatives")
    return torch.stack(losses).mean()


def record_route(
    route: str, task: str, objective: str | None, contract: FormalContract
) -> tuple[int, Counter[str], Counter[str], Counter[str]]:
    count = int(getattr(i39_loss, "call_count", 0)) + 1
    routes = Counter(getattr(i39_loss, "route_counts", Counter()))
    objectives = Counter(getattr(i39_loss, "objective_counts", Counter()))
    route_tasks = Counter(getattr(i39_loss, "route_task_counts", Counter()))
    routes[route] += 1
    route_tasks[f"{route}:{task}"] += 1
    if objective is not None:
        objectives[objective] += 1
    expected_routes = Counter(contract.route_counts)
    expected_objectives = Counter(contract.objective_counts)
    expected_route_tasks = Counter(
        {
            f"{route_name}:{task_name}": value
            for route_name, tasks in contract.route_task_counts.items()
            for task_name, value in tasks.items()
        }
    )
    if count > contract.total_rows:
        raise RuntimeError("I39 observed too many microbatches")
    for observed, expected, label in (
        (routes, expected_routes, "route"),
        (objectives, expected_objectives, "objective"),
        (route_tasks, expected_route_tasks, "route-task"),
    ):
        for name, value in observed.items():
            if value > expected[name]:
                raise RuntimeError(f"I39 {label} count exceeded {name}: {value}")
        remaining = contract.total_rows - count
        for name, value in expected.items():
            if observed[name] + remaining < value:
                raise RuntimeError(f"I39 cannot satisfy remaining {label} {name}")
    i39_loss.call_count = count
    i39_loss.route_counts = routes
    i39_loss.objective_counts = objectives
    i39_loss.route_task_counts = route_tasks
    return count, routes, objectives, route_tasks


def i39_loss(self, model, inputs, return_outputs=False, **kwargs):
    del kwargs
    labels = inputs.pop("labels")
    target_start, target_end = I36.target_span(labels)
    response_targets = labels[0, target_start:target_end]
    _unwrapped, state = ensure_runtime(self, model)
    prompt_ids = inputs["input_ids"][0, :target_start].detach().cpu().tolist()
    key = I36.I34.prompt_token_sha256(prompt_ids)
    entry = state["sidecar"].get(key)
    if entry is None:
        raise RuntimeError(f"I39 runtime prompt is absent from sidecar: {key}")
    response_list = response_targets.detach().cpu().tolist()
    if token_hash(response_list) != entry["response_token_sha256"]:
        raise RuntimeError(f"I39 runtime response drifted for prompt {key}")
    route = str(entry["route"])
    task = str(entry["task"])
    objective: str | None = None
    if route == "material_firstdiv":
        body_start, _body_end = material_bounds(response_targets)
        absolute = target_start + body_start
        positions = torch.arange(
            absolute, absolute + 3, device=labels.device, dtype=torch.long
        )
        targets = labels[0, positions + 1]
        gold_abc = [int(value) for value in entry["gold_abc"]]
        if targets.detach().cpu().tolist() != gold_abc:
            raise RuntimeError("I39 material runtime gold does not match sidecar")
        outputs, parent_logits, _ = I36.paired_parent_policy(
            self, model, inputs, positions
        )
        policy_logits = outputs.logits
        parent_kl = I36.forward_kl(policy_logits, parent_logits)
        objective = str(entry["objective"])
        focus = objective_focus(objective)
        if focus is None:
            margin = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
            ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
            loss = ANCHOR_PARENT_KL * parent_kl
        else:
            margin = focus_margin_loss(
                policy_logits, gold_abc, entry["hard_negatives"], focus
            )
            ce = F.cross_entropy(
                policy_logits[0, focus].float(),
                targets[focus].to(policy_logits.device).long(),
            )
            loss = (
                MATERIAL_MARGIN_WEIGHT * margin
                + MATERIAL_FOCUS_CE_WEIGHT * ce
                + MATERIAL_PARENT_KL * parent_kl
            )
    else:
        body_start, body_end = I36.body_bounds(response_targets)
        relative = I36.uniformly_bounded_indices(
            body_start, body_end, MAX_ANSWER_POSITIONS
        )
        selected = relative.to(labels.device) + target_start
        positions = selected - 1
        targets = labels[0, selected]
        outputs, parent_logits, _ = I36.paired_parent_policy(
            self, model, inputs, positions
        )
        policy_logits = outputs.logits
        parent_kl = I36.forward_kl(policy_logits, parent_logits)
        margin = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        if route == "user_micro_ce":
            weights = I36.terminal_weights(targets)
            ce = I36.weighted_ce(policy_logits, targets, weights)
            loss = USER_CE_WEIGHT * ce + USER_PARENT_KL * parent_kl
        elif route == "retention_kl":
            ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
            loss = RETENTION_PARENT_KL * parent_kl
        else:
            raise RuntimeError(f"I39 unknown runtime route: {route}")
    count, routes, objectives, _route_tasks = record_route(
        route, task, objective, state["contract"]
    )
    if count <= 8 or count % 128 == 0 or count == state["contract"].total_rows:
        print(
            f"[i39] microbatch={count}/{state['contract'].total_rows} "
            f"route={route} task={task} objective={objective or '-'} "
            f"tokens={targets.numel()} margin={float(margin.detach()):.6f} "
            f"ce={float(ce.detach()):.6f} parent_kl={float(parent_kl.detach()):.8f} "
            f"loss={float(loss.detach()):.6f} routes={dict(routes)} "
            f"objectives={dict(objectives)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def run_data_preflight() -> FormalContract:
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    contract = I36.verify_static_contract(require_data=True)
    if contract is None:
        raise RuntimeError("I39 contract unexpectedly missing")
    sidecar = load_sidecar(contract.sidecar_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        I36.BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    routes: Counter[str] = Counter()
    objectives: Counter[str] = Counter()
    route_tasks: Counter[str] = Counter()
    observed_keys: set[str] = set()
    maximum_tokens = 0
    with TRAINING_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"I39 data has a blank row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"I39 data line {line_number} is not an object")
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
                raise RuntimeError(f"I39 cutoff overflow at line {line_number}: {total}")
            key = I36.I34.prompt_token_sha256(prompt_ids)
            if key in observed_keys or key not in sidecar:
                raise RuntimeError(f"I39 duplicate/missing sidecar key at line {line_number}")
            observed_keys.add(key)
            entry = sidecar[key]
            if token_hash(response_ids) != entry["response_token_sha256"]:
                raise RuntimeError(f"I39 response token drift at line {line_number}")
            if core_hash(row) != entry["row_sha256"]:
                raise RuntimeError(f"I39 row hash drift at line {line_number}")
            route = str(row.get("route"))
            task = str(row.get("task"))
            if route != entry["route"] or task != entry["task"]:
                raise RuntimeError(f"I39 row/sidecar route drift at line {line_number}")
            response_tensor = torch.tensor(response_ids, dtype=torch.long)
            if route == "material_firstdiv":
                body_start, body_end = material_bounds(response_tensor)
                if response_ids[body_start:body_end] != entry["gold_tokens"]:
                    raise RuntimeError(f"I39 material gold drift at line {line_number}")
                objective = str(entry["objective"])
                objectives[objective] += 1
            elif route in ("user_micro_ce", "retention_kl"):
                I36.body_bounds(response_tensor)
                if route == "user_micro_ce" and task not in ("action", "topic"):
                    raise RuntimeError(f"I39 user task drift at line {line_number}")
            else:
                raise RuntimeError(f"I39 unknown data route at line {line_number}")
            routes[route] += 1
            route_tasks[f"{route}:{task}"] += 1
            maximum_tokens = max(maximum_tokens, total)
    expected_route_tasks = Counter(
        {
            f"{route}:{task}": count
            for route, tasks in EXPECTED_ROUTE_TASKS.items()
            for task, count in tasks.items()
        }
    )
    if observed_keys != set(sidecar):
        raise RuntimeError("I39 data/sidecar prompt-token sets are not a bijection")
    if routes != Counter(EXPECTED_ROUTES):
        raise RuntimeError(f"I39 data routes drifted: {dict(routes)}")
    if objectives != Counter(EXPECTED_OBJECTIVES):
        raise RuntimeError(f"I39 data objectives drifted: {dict(objectives)}")
    if route_tasks != expected_route_tasks:
        raise RuntimeError(f"I39 data route-tasks drifted: {dict(route_tasks)}")
    print(
        f"[i39] data preflight PASS: rows={contract.total_rows} "
        f"routes={dict(routes)} objectives={dict(objectives)} "
        f"steps={contract.optimizer_steps} max_tokens={maximum_tokens} "
        f"data_sha256={contract.data_sha256} sidecar_sha256={contract.sidecar_sha256}",
        flush=True,
    )
    return contract


def run_self_test() -> None:
    torch.manual_seed(39)
    policy = torch.randn(1, 3, 17, requires_grad=True)
    parent = torch.randn(1, 3, 17)
    negatives = [{"tokens": [4, 2, 3], "first_divergence": 0}]
    margin = focus_margin_loss(policy, [1, 2, 3], negatives, 0)
    loss = margin + I36.forward_kl(policy, parent)
    loss.backward()
    if policy.grad is None or not torch.isfinite(policy.grad).all():
        raise AssertionError("I39 margin/KL gradient self-test failed")
    if objective_focus("a_firstdiv") != 0 or objective_focus("full_anchor") is not None:
        raise AssertionError("I39 objective-focus self-test failed")
    aggregated: Counter[str] = Counter()
    for task_counts in EXPECTED_ROUTE_TASKS.values():
        aggregated.update(task_counts)
    if dict(aggregated) != EXPECTED_TASK_COUNTS:
        raise AssertionError("I39 aggregate task-count self-test failed")
    if token_hash([1, 2, 3]) != token_hash([1, 2, 3]):
        raise AssertionError("I39 token hash self-test failed")
    print(
        "[i39] self-test PASS: token-hash routing, focused margin, parent KL, and gradients",
        flush=True,
    )


def assert_final_contract(contract: FormalContract) -> None:
    count = int(getattr(i39_loss, "call_count", 0))
    routes = Counter(getattr(i39_loss, "route_counts", Counter()))
    objectives = Counter(getattr(i39_loss, "objective_counts", Counter()))
    route_tasks = Counter(getattr(i39_loss, "route_task_counts", Counter()))
    expected_route_tasks = Counter(
        {
            f"{route}:{task}": value
            for route, tasks in contract.route_task_counts.items()
            for task, value in tasks.items()
        }
    )
    if (
        count != contract.total_rows
        or routes != Counter(contract.route_counts)
        or objectives != Counter(contract.objective_counts)
        or route_tasks != expected_route_tasks
    ):
        raise RuntimeError(
            f"I39 final route contract failed: calls={count}/{contract.total_rows} "
            f"routes={dict(routes)} objectives={dict(objectives)} "
            f"route_tasks={dict(route_tasks)}"
        )


def assert_policy_only_save_artifacts_i39(
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
        raise RuntimeError(f"I39 policy-only save is incomplete: {output_dir}")
    alternate = output_dir / (
        I36.ADAPTER_WEIGHTS_NAME
        if safe_serialization
        else I36.ADAPTER_SAFE_WEIGHTS_NAME
    )
    if alternate.exists():
        raise RuntimeError(f"I39 save has a stale alternate adapter payload: {alternate}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("peft_type") != "LORA"
        or int(config.get("r", -1)) != EXPECTED_RANK
        or int(config.get("lora_alpha", -1)) != EXPECTED_ALPHA
    ):
        raise RuntimeError("I39 saved adapter rank/alpha drifted")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern"):
        if config.get(key) not in (None, {}, []):
            raise RuntimeError(f"I39 saved adapter contains unsupported {key}")
    if safe_serialization:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            keys = list(source.keys())
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        keys = list(state)
    if not keys or any(
        "embed" in key or "lm_head" in key or "reference" in key.lower()
        for key in keys
    ):
        raise RuntimeError("I39 saved adapter contains non-policy tensors")
    if any("lora_" not in key.lower() for key in keys):
        raise RuntimeError("I39 saved adapter contains a non-LoRA tensor")


I36.PARENT_ADAPTER = PARENT_ADAPTER
I36.TRAINING_DATA = TRAINING_DATA
I36.AUDIT = AUDIT
I36.OUTPUT_DIR = OUTPUT_DIR
I36.SCHEMA_VERSION = SCHEMA_VERSION
I36.EXPECTED_RANK = EXPECTED_RANK
I36.EXPECTED_ALPHA = EXPECTED_ALPHA
I36.EXPECTED_ROWS = EXPECTED_ROWS
I36.EXPECTED_STEPS = EXPECTED_STEPS
I36.EXPECTED_TASK_COUNTS = EXPECTED_TASK_COUNTS
I36.load_contract = load_contract
I36.assert_formal_trainer_args = assert_formal_trainer_args
I36.ensure_runtime = ensure_runtime
I36.i36_loss = i39_loss
I36.run_data_preflight = run_data_preflight
I36.run_self_test = run_self_test
I36.assert_final_contract = assert_final_contract
I36.I34.assert_policy_only_save_artifacts = assert_policy_only_save_artifacts_i39


if __name__ == "__main__":
    I36.main()
