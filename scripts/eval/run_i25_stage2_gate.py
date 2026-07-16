#!/usr/bin/env python3
"""Locked, sequential executor for the preregistered I-25 Stage-2 gate.

The default mode is a read-only dry run.  ``--prepare`` is deliberately
separate and may run only after Stage 1 has frozen exactly one checkpoint.  It
emits a no-argument launcher that evaluates the locked scales in ascending
order, materialising only the current scale and stopping at the first full
pass.  This file never packages or uploads a candidate.

Internal modes are used by the emitted launcher.  They revalidate the gate,
the immutable Stage-1 ticket, the executor hash, and the contiguous decision
prefix on every invocation.  A failed scale is deleted before its decision is
published.  A passing scale is retained only after all D/E, generation,
composition, and merged-structure requirements have passed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
GATE_REL = Path("configs/evaluation/i23_actionres_r16_checkpoint_gate.json")
GATE_SHA256 = "53b5b375630ba2255dada2fd04d8fc4cd1b694e3afde10079ef492135dcef212"
STAGE1_TICKET = ROOT / "logs/probe/i25_stage1/frozen-stage1-action.json"
STAGE1_REPORT_ROOT = ROOT / "logs/probe/i25_stage1"
STAGE1_PREPARE_MANIFEST = ROOT / "logs/model/i25_stage1_scale1_prepare_manifest.json"

SCALES = (0.25, 0.375, 0.5, 0.625, 0.75)
SCALE_TAG = {
    0.25: "s0250",
    0.375: "s0375",
    0.5: "s0500",
    0.625: "s0625",
    0.75: "s0750",
}
STAGE1_STEPS = (250, 500, 750, 1000, 1250, 1527)

REPORT_ROOT = ROOT / "logs/probe/i25_stage2"
CANDIDATE_ROOT = ROOT / "checkpoints/i25_stage2_scale_candidates"
PREPARE_MANIFEST = ROOT / "logs/model/i25_stage2_prepare_manifest.json"
COMMAND_LAUNCHER = ROOT / "logs/model/i25_stage2_gpu_commands.sh"
PREPARE_PENDING = ROOT / "logs/model/.i25_stage2_prepare.pending.json"
PARENT_GENERATION = REPORT_ROOT / "parent-action-generation.json"
PARENT_GENERATION_LOG = REPORT_ROOT / "parent-action-generation.log"
FROZEN_STAGE2 = REPORT_ROOT / "frozen-stage2-full.json"
FROZEN_STAGE2_PENDING = REPORT_ROOT / ".frozen-stage2-full.pending.json"
REJECTED_STAGE2 = REPORT_ROOT / "stage2-rejected.json"

MODEL_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
DATA_FINAL_SHA256 = "995efe62ee98b44c28175a444849e51c3ebbbc9c1b9e83fc83146f09024f8f71"
NORMAL_PYTHON = Path(
    "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/"
    "llmrec_2026/LLaMA-Factory/.venv/bin/python3"
)
VLLM_PYTHON = Path(
    "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/"
    "miniconda3/envs/verl_v071/bin/python"
)
EXPORT_CLI = Path(
    "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/"
    "llmrec_2026/LLaMA-Factory/.venv/bin/llamafactory-cli"
)

EXIT_SCALE_FAILED = 3
EXIT_ALREADY_FAILED = 5
EXIT_ALREADY_SELECTED = 6
EXIT_ALREADY_REJECTED = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def json_document(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse JSON staging path: {temporary}")
    temporary.write_text(json_document(value), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse text staging path: {temporary}")
    temporary.write_text(value, encoding="utf-8")
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def rooted(value: str | Path) -> Path:
    return (ROOT / Path(value)).resolve(strict=False)


def lexical_repo_path(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        absolute.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay lexically inside the repository: {absolute}") from error
    return absolute


def record_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return lexical_repo_path(path, label)


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 drifted: expected {expected}, got {actual}")
    return actual


def require_executable(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"missing executable {label}: {path}")


def require_lower_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} is not a lowercase SHA256: {value!r}")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is boolean, not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite: {number}")
    return number


def exact_stable_action_manifest(path: Path, count: int = 32) -> str:
    hashes = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            packed = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            hashes.append(hashlib.sha256(packed.encode()).hexdigest())
    if len(hashes) < count:
        raise RuntimeError(f"stable action source has only {len(hashes)} rows")
    selected = sorted(hashes)[:count]
    return hashlib.sha256(
        "".join(value + "\n" for value in selected).encode()
    ).hexdigest()


def static_contract() -> tuple[dict[str, Any], dict[str, Path | str]]:
    gate_path = rooted(GATE_REL)
    require_hash(gate_path, GATE_SHA256, "I-25 preregistered gate")
    gate = load_json(gate_path)
    if gate.get("status") != "PREREGISTERED_BEFORE_I25_FORMAL_LAUNCH":
        raise RuntimeError(f"I-25 gate is inactive: {gate.get('status')!r}")

    stage2 = gate["stage_2_residual_scale_selection"]
    if tuple(stage2.get("scale_order", ())) != SCALES:
        raise RuntimeError("Stage-2 scale order drifted")
    if stage2.get("combined_adapter_spec", {}).get("rank") != 80:
        raise RuntimeError("Stage-2 combined rank drifted")
    if stage2.get("combined_adapter_spec", {}).get("alpha") != 80:
        raise RuntimeError("Stage-2 combined alpha drifted")
    if tuple(gate["stage_1_training_checkpoint_selection"].get("candidate_order", ())) != STAGE1_STEPS:
        raise RuntimeError("Stage-1 checkpoint order drifted")

    locked = gate["locked_audit_protocols"]
    path_records = {
        "combine": locked["composition"],
        "d_tool": locked["d_residual_mechanism_and_retention"],
        "e_user_tool": locked["e_action_topic_path"],
        "e_rec_tool": locked["e_recommendation_path"],
        "d_material_tool": locked["d_material_exact_row_holdout"],
        "generation_tool": {
            "tool": locked["e_action_generation"]["generator"],
            "tool_sha256": locked["e_action_generation"]["generator_sha256"],
        },
        "generation_checker": {
            "tool": locked["e_action_generation"]["paired_checker"],
            "tool_sha256": locked["e_action_generation"]["paired_checker_sha256"],
        },
        "precheck": locked["structure"],
    }
    paths: dict[str, Path | str] = {"gate": gate_path}
    for name, record in path_records.items():
        path = rooted(record["tool"])
        require_hash(path, record["tool_sha256"], f"locked {name}")
        paths[name] = path

    base = rooted(locked["base"]["path"])
    require_hash(base / "config.json", locked["base"]["config_sha256"], "O6 config")
    paths["base"] = base

    parent = rooted(gate["parent"]["adapter"])
    require_hash(parent / MODEL_NAME, gate["parent"]["adapter_sha256"], "I-23 adapter")
    require_hash(
        parent / CONFIG_NAME,
        gate["parent"]["adapter_config_sha256"],
        "I-23 adapter config",
    )
    paths["parent"] = parent

    d_record = locked["d_residual_mechanism_and_retention"]
    for name, key, hash_key in (
        ("dataset", "dataset", "dataset_sha256"),
        ("retention_source", "retention_source", "retention_source_sha256"),
        ("retention_exclude", "exact_row_exclude", "exact_row_exclude_sha256"),
    ):
        path = rooted(d_record[key])
        require_hash(path, d_record[hash_key], f"locked {name}")
        paths[name] = path

    dev = ROOT / "assets/evaluation/offline_eval"
    e_user = locked["e_action_topic_path"]
    require_hash(dev / "dev_action.jsonl", e_user["action_source_sha256"], "E action source")
    require_hash(dev / "dev_topic.jsonl", e_user["topic_source_sha256"], "E topic source")
    e_rec = locked["e_recommendation_path"]
    for domain in ("video", "prod", "ad", "live"):
        require_hash(
            dev / f"dev_rec_{domain}.jsonl",
            e_rec["source_sha256"][domain],
            f"E recommendation source {domain}",
        )
    paths["dev"] = dev
    paths["stable_action_manifest_sha256"] = exact_stable_action_manifest(
        dev / "dev_action.jsonl"
    )

    parent_precheck_log = rooted(locked["structure"]["parent_log"])
    require_hash(
        parent_precheck_log,
        locked["structure"]["parent_log_sha256"],
        "I-23 parent precheck log",
    )
    paths["parent_precheck_log"] = parent_precheck_log
    precheck_source = ROOT / "assets/derived/processed/data_final.jsonl"
    require_hash(
        precheck_source,
        DATA_FINAL_SHA256,
        "registered D(O1) precheck source data_final",
    )
    paths["precheck_source"] = precheck_source

    require_executable(NORMAL_PYTHON, "normal audit Python")
    require_executable(VLLM_PYTHON, "vLLM Python")
    require_executable(EXPORT_CLI, "LLaMA-Factory export CLI")
    if shutil.which("flock") is None:
        raise FileNotFoundError("flock is required for the Stage-2 single-run lock")
    paths["normal_python"] = NORMAL_PYTHON
    paths["vllm_python"] = VLLM_PYTHON
    paths["export_cli"] = EXPORT_CLI
    return gate, paths


def lora_config(path: Path, rank: int, alpha: int, label: str) -> dict[str, Any]:
    config = load_json(path)
    if config.get("peft_type") != "LORA":
        raise RuntimeError(f"{label} is not LoRA")
    if int(config.get("r", -1)) != rank or int(config.get("lora_alpha", -1)) != alpha:
        raise RuntimeError(
            f"{label} expected r{rank}/alpha{alpha}, got "
            f"r{config.get('r')}/alpha{config.get('lora_alpha')}"
        )
    if config.get("bias") != "none" or config.get("modules_to_save") not in (None, []):
        raise RuntimeError(f"{label} contains non-LoRA trainable state")
    if config.get("use_dora") or config.get("use_rslora"):
        raise RuntimeError(f"{label} uses an unsupported LoRA variant")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise RuntimeError(f"{label} uses per-module rank/alpha patterns")
    if config.get("fan_in_fan_out") is not False:
        raise RuntimeError(f"{label} must use fan_in_fan_out=false")
    return config


def load_stage1_ticket(
    gate: dict[str, Any], paths: dict[str, Path | str], required: bool
) -> dict[str, Any] | None:
    if not STAGE1_TICKET.is_file():
        if required:
            raise RuntimeError(
                "Stage 2 is closed: missing locked Stage-1 ticket "
                f"{STAGE1_TICKET}"
            )
        return None
    ticket = load_json(STAGE1_TICKET)
    if ticket.get("status") != "PASS_FROZEN_STAGE1_CHECKPOINT":
        raise RuntimeError(f"Stage-1 ticket is not a pass: {ticket.get('status')!r}")
    if ticket.get("gate_sha256") != GATE_SHA256 or ticket.get("pass") is not True:
        raise RuntimeError("Stage-1 ticket gate identity/pass bit drifted")
    if ticket.get("stage2_opened") is not False:
        raise RuntimeError("Stage-1 ticket unexpectedly claims Stage 2 was already opened")
    if ticket.get("selection_fields") != "action only":
        raise RuntimeError("Stage-1 ticket was not selected using action-only fields")
    if ticket.get("selection_axis") != "checkpoint only at residual multiplier 1.0":
        raise RuntimeError("Stage-1 ticket checkpoint axis drifted")
    if not ticket.get("checks") or not all(ticket["checks"].values()):
        raise RuntimeError("Stage-1 ticket contains a failed action check")

    frozen = ticket.get("frozen_checkpoint", {})
    step = frozen.get("step")
    if step not in STAGE1_STEPS or ticket.get("step") != step:
        raise RuntimeError(f"invalid frozen Stage-1 step: {step!r}")
    residual_sha = require_lower_sha(
        frozen.get("residual_adapter_sha256"), "Stage-1 residual SHA256"
    )
    combined_sha = require_lower_sha(
        frozen.get("scale_1_combined_adapter_sha256"),
        "Stage-1 scale-1 combined SHA256",
    )

    checkpoint = rooted(gate["checkpoint_axis"]["expected_paths"][str(step)])
    require_hash(checkpoint / MODEL_NAME, residual_sha, "frozen Stage-1 residual")
    lora_config(checkpoint / CONFIG_NAME, 16, 16, "frozen Stage-1 residual")
    if not STAGE1_PREPARE_MANIFEST.is_file():
        raise RuntimeError(
            "Stage-1 ticket has no retained CPU prepare manifest: "
            f"{STAGE1_PREPARE_MANIFEST}"
        )
    stage1_manifest = load_json(STAGE1_PREPARE_MANIFEST)
    if (
        stage1_manifest.get("status")
        != "CPU_PREPARATION_COMPLETE_GPU_NOT_RUN_NO_SELECTION"
        or stage1_manifest.get("gate", {}).get("sha256") != GATE_SHA256
        or stage1_manifest.get("contract", {}).get("candidate_order")
        != list(STAGE1_STEPS)
        or stage1_manifest.get("contract", {}).get("residual_multiplier") != 1.0
        or stage1_manifest.get("training_completion", {}).get("global_step") != 1527
    ):
        raise RuntimeError("Stage-1 CPU prepare manifest contract drifted")
    stage1_candidates = stage1_manifest.get("candidates")
    if (
        not isinstance(stage1_candidates, list)
        or [record.get("step") for record in stage1_candidates] != list(STAGE1_STEPS)
    ):
        raise RuntimeError("Stage-1 CPU prepare candidate order drifted")
    selected_manifest_record = next(
        record for record in stage1_candidates if record.get("step") == step
    )
    if (
        selected_manifest_record.get("residual", {}).get("path") != str(checkpoint)
        or selected_manifest_record.get("residual", {}).get("adapter_sha256")
        != residual_sha
    ):
        raise RuntimeError("Stage-1 prepare manifest residual identity drifted")
    residual_config_sha = require_lower_sha(
        selected_manifest_record.get("residual", {}).get("config_sha256"),
        "Stage-1 prepared residual config SHA256",
    )
    require_hash(
        checkpoint / CONFIG_NAME,
        residual_config_sha,
        "Stage-1 prepared residual config",
    )

    decision_path = STAGE1_REPORT_ROOT / f"step-{step}-action-decision.json"
    expected_decision_sha = require_lower_sha(
        ticket.get("decision_report_sha256"), "Stage-1 decision report SHA256"
    )
    require_hash(decision_path, expected_decision_sha, "Stage-1 decision report")
    decision = load_json(decision_path)
    for key in ("status", "step", "gate_sha256", "selection_axis", "selection_fields", "checks", "pass"):
        if decision.get(key) != ticket.get(key):
            raise RuntimeError(f"Stage-1 frozen ticket differs from decision at {key}")

    for record_name in ("d_action_report", "e_action_report"):
        if decision.get(record_name) != ticket.get(record_name):
            raise RuntimeError(f"Stage-1 {record_name} record drifted")
        record = ticket[record_name]
        report_path = record_path(record["path"], f"Stage-1 {record_name}")
        require_hash(report_path, record["sha256"], f"Stage-1 {record_name}")

    d_report = load_json(record_path(ticket["d_action_report"]["path"], "Stage-1 D report"))
    if d_report.get("step") != step or d_report.get("scale") != 1.0:
        raise RuntimeError("Stage-1 D action projection identity drifted")
    if d_report.get("artifacts", {}).get("residual_adapter_sha256") != residual_sha:
        raise RuntimeError("Stage-1 D report residual hash differs from frozen ticket")
    e_report = load_json(record_path(ticket["e_action_report"]["path"], "Stage-1 E report"))
    candidate_name = f"step-{step}"
    if (
        e_report.get("models", {}).get(candidate_name, {}).get("adapter_sha256")
        != combined_sha
    ):
        raise RuntimeError("Stage-1 E report combined hash differs from frozen ticket")
    if (
        e_report.get("models", {}).get("parent", {}).get("adapter_sha256")
        != gate["parent"]["adapter_sha256"]
    ):
        raise RuntimeError("Stage-1 E report parent hash drifted")

    for earlier in STAGE1_STEPS:
        earlier_path = STAGE1_REPORT_ROOT / f"step-{earlier}-action-decision.json"
        if earlier < step:
            if not earlier_path.is_file():
                raise RuntimeError(f"missing earlier Stage-1 decision for step {earlier}")
            earlier_decision = load_json(earlier_path)
            if (
                earlier_decision.get("status")
                != "FAIL_CONTINUE_ASCENDING_CHECKPOINT_AXIS"
                or earlier_decision.get("step") != earlier
                or earlier_decision.get("gate_sha256") != GATE_SHA256
                or earlier_decision.get("selection_axis")
                != "checkpoint only at residual multiplier 1.0"
                or earlier_decision.get("selection_fields") != "action only"
                or earlier_decision.get("pass") is not False
                or not isinstance(earlier_decision.get("checks"), dict)
                or all(earlier_decision["checks"].values())
            ):
                raise RuntimeError(f"earlier Stage-1 step {earlier} was not a recorded failure")
            for record_name in ("d_action_report", "e_action_report"):
                record = earlier_decision.get(record_name, {})
                report_path = record_path(
                    record.get("path", ""),
                    f"earlier Stage-1 step {earlier} {record_name}",
                )
                require_hash(
                    report_path,
                    require_lower_sha(
                        record.get("sha256"),
                        f"earlier Stage-1 step {earlier} {record_name} SHA256",
                    ),
                    f"earlier Stage-1 step {earlier} {record_name}",
                )
        elif earlier > step and earlier_path.exists():
            raise RuntimeError(f"later Stage-1 decision exists after first pass: step {earlier}")

    return {
        "path": STAGE1_TICKET,
        "sha256": sha256(STAGE1_TICKET),
        "step": step,
        "checkpoint": checkpoint,
        "residual_adapter_sha256": residual_sha,
        "residual_config_sha256": residual_config_sha,
        "scale_1_combined_adapter_sha256": combined_sha,
        "prepare_manifest": STAGE1_PREPARE_MANIFEST,
        "prepare_manifest_sha256": sha256(STAGE1_PREPARE_MANIFEST),
        "decision_path": decision_path,
        "decision_sha256": expected_decision_sha,
        "ticket": ticket,
    }


def candidate_dir(scale: float) -> Path:
    return CANDIDATE_ROOT / SCALE_TAG[scale]


def scale_report_dir(scale: float) -> Path:
    return REPORT_ROOT / SCALE_TAG[scale]


def validate_running_dir(path: Path, scale: float, must_exist: bool = True) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    expected_prefix = f".running-{SCALE_TAG[scale]}."
    if absolute.parent != REPORT_ROOT or not absolute.name.startswith(expected_prefix):
        raise ValueError(
            f"current-scale staging directory must match "
            f"{REPORT_ROOT}/{expected_prefix}*: {absolute}"
        )
    if must_exist and not absolute.is_dir():
        raise FileNotFoundError(absolute)
    return absolute


def validate_merged_temp_path(path: Path, scale: float, must_exist: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    expected_prefix = f".merged-{SCALE_TAG[scale]}."
    if absolute.parent != REPORT_ROOT or not absolute.name.startswith(expected_prefix):
        raise ValueError(
            f"merged precheck path must match "
            f"{REPORT_ROOT}/{expected_prefix}*: {absolute}"
        )
    if must_exist and not absolute.is_dir():
        raise FileNotFoundError(absolute)
    if not must_exist and (absolute.exists() or absolute.is_symlink()):
        raise FileExistsError(absolute)
    return absolute


def validate_published_decision(
    decision: dict[str, Any], scale: float, stage1: dict[str, Any]
) -> bool:
    if decision.get("gate_sha256") != GATE_SHA256:
        raise RuntimeError(f"published {SCALE_TAG[scale]} decision gate hash drifted")
    if decision.get("stage1_ticket_sha256") != stage1["sha256"]:
        raise RuntimeError(f"published {SCALE_TAG[scale]} Stage-1 ticket hash drifted")
    if decision.get("executor_sha256") != sha256(SELF):
        raise RuntimeError(f"published {SCALE_TAG[scale]} executor hash drifted")
    if decision.get("frozen_stage1_step") != stage1["step"]:
        raise RuntimeError(f"published {SCALE_TAG[scale]} Stage-1 step drifted")
    if decision.get("frozen_residual_adapter_sha256") != stage1["residual_adapter_sha256"]:
        raise RuntimeError(f"published {SCALE_TAG[scale]} residual hash drifted")
    if finite(decision.get("scale"), "published decision scale") != scale:
        raise RuntimeError(f"published decision scale drifted for {SCALE_TAG[scale]}")
    passed = decision.get("pass")
    if passed not in (True, False):
        raise RuntimeError(f"published {SCALE_TAG[scale]} decision has no boolean pass")
    expected_status = (
        "PASS_FROZEN_STAGE2_SCALE"
        if passed
        else "FAIL_CONTINUE_ASCENDING_SCALE_AXIS"
    )
    if decision.get("status") != expected_status:
        raise RuntimeError(f"published {SCALE_TAG[scale]} status/pass mismatch")
    checks = decision.get("checks")
    if not isinstance(checks, dict) or len(checks) != 59:
        raise RuntimeError(f"published {SCALE_TAG[scale]} decision has no checks")
    if bool(all(checks.values())) != passed:
        raise RuntimeError(f"published {SCALE_TAG[scale]} checks/pass mismatch")
    candidate = decision.get("candidate", {})
    if (
        candidate.get("path") != str(candidate_dir(scale))
        or require_lower_sha(
            candidate.get("adapter_sha256"),
            f"published {SCALE_TAG[scale]} candidate adapter SHA256",
        )
        != candidate.get("adapter_sha256")
        or require_lower_sha(
            candidate.get("config_sha256"),
            f"published {SCALE_TAG[scale]} candidate config SHA256",
        )
        != candidate.get("config_sha256")
        or candidate.get("rank") != 80
        or candidate.get("alpha") != 80
    ):
        raise RuntimeError(f"published {SCALE_TAG[scale]} candidate record drifted")
    reports = decision.get("reports")
    expected_report_names = set(RAW_FILES) | {"parent_generation"}
    if not isinstance(reports, dict) or set(reports) != expected_report_names:
        raise RuntimeError(f"published {SCALE_TAG[scale]} report set drifted")
    for name, filename in RAW_FILES.items():
        record = reports[name]
        expected_path = scale_report_dir(scale) / filename
        if record.get("path") != str(expected_path):
            raise RuntimeError(
                f"published {SCALE_TAG[scale]} report path drifted for {name}"
            )
        require_hash(
            expected_path,
            require_lower_sha(
                record.get("sha256"),
                f"published {SCALE_TAG[scale]} {name} SHA256",
            ),
            f"published {SCALE_TAG[scale]} {name}",
        )
        if record.get("bytes") != expected_path.stat().st_size:
            raise RuntimeError(
                f"published {SCALE_TAG[scale]} report byte count drifted for {name}"
            )
    parent_record = reports["parent_generation"]
    if parent_record.get("path") != str(PARENT_GENERATION):
        raise RuntimeError(f"published {SCALE_TAG[scale]} parent report path drifted")
    require_hash(
        PARENT_GENERATION,
        require_lower_sha(
            parent_record.get("sha256"),
            f"published {SCALE_TAG[scale]} parent generation SHA256",
        ),
        f"published {SCALE_TAG[scale]} parent generation",
    )
    if parent_record.get("bytes") != PARENT_GENERATION.stat().st_size:
        raise RuntimeError(
            f"published {SCALE_TAG[scale]} parent generation byte count drifted"
        )
    composition = load_json(scale_report_dir(scale) / RAW_FILES["composition"])
    if (
        composition.get("combined", {}).get("adapter_sha256")
        != candidate.get("adapter_sha256")
        or composition.get("combined", {}).get("config_sha256")
        != candidate.get("config_sha256")
    ):
        raise RuntimeError(f"published {SCALE_TAG[scale]} composition/candidate drifted")
    if decision.get("packaging_authorized") is not False or decision.get("upload_authorized") is not False:
        raise RuntimeError(f"published {SCALE_TAG[scale]} improperly authorizes release")
    if decision.get("generation_checker_boundary_override") != (
        "The locked I-24 comparison helper labels max_repeat_p95 as diagnostic; "
        "I-25 independently hard-gates max_repeat_p95_delta <= 0."
    ):
        raise RuntimeError(
            f"published {SCALE_TAG[scale]} generation boundary contract drifted"
        )
    return passed


def sequence_state(stage1: dict[str, Any]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    gap_seen = False
    passing_scale: float | None = None
    for scale in SCALES:
        directory = scale_report_dir(scale)
        decision_path = directory / "gate-decision.json"
        if decision_path.is_file():
            if gap_seen:
                raise RuntimeError("Stage-2 decisions are not a contiguous ascending prefix")
            decision = load_json(decision_path)
            passed = validate_published_decision(decision, scale, stage1)
            decisions.append(
                {
                    "scale": scale,
                    "path": decision_path,
                    "sha256": sha256(decision_path),
                    "pass": passed,
                    "decision": decision,
                }
            )
            if passed:
                if passing_scale is not None:
                    raise RuntimeError("multiple Stage-2 passing decisions exist")
                passing_scale = scale
                gap_seen = True
            elif candidate_dir(scale).exists() or candidate_dir(scale).is_symlink():
                raise RuntimeError(
                    f"failed Stage-2 candidate was not deleted: {candidate_dir(scale)}"
                )
        elif directory.exists() or directory.is_symlink():
            raise RuntimeError(f"incomplete published Stage-2 directory: {directory}")
        else:
            gap_seen = True

    if passing_scale is not None:
        if not FROZEN_STAGE2.is_file():
            raise RuntimeError("passing Stage-2 decision exists without frozen result")
        passing_record = next(
            record for record in decisions if record["scale"] == passing_scale
        )
        passing_decision = passing_record["decision"]
        frozen = load_json(FROZEN_STAGE2)
        if frozen.get("status") != "PASS_FROZEN_STAGE2_SCALE":
            raise RuntimeError("frozen Stage-2 result status drifted")
        if finite(frozen.get("scale"), "frozen Stage-2 scale") != passing_scale:
            raise RuntimeError("frozen Stage-2 scale differs from passing decision")
        if frozen.get("stage1_ticket_sha256") != stage1["sha256"]:
            raise RuntimeError("frozen Stage-2 ticket identity drifted")
        for key, value in passing_decision.items():
            if frozen.get(key) != value:
                raise RuntimeError(
                    f"frozen Stage-2 result differs from passing decision at {key}"
                )
        if frozen.get("decision_report") != {
            "path": str(passing_record["path"]),
            "sha256": passing_record["sha256"],
        }:
            raise RuntimeError("frozen Stage-2 decision report identity drifted")
        if frozen.get("selected_candidate") != passing_decision.get("candidate"):
            raise RuntimeError("frozen Stage-2 selected candidate record drifted")
        if frozen.get("first_full_pass_scale") != passing_scale:
            raise RuntimeError("frozen Stage-2 first-pass scale drifted")
        if frozen.get("future_scales_evaluated") is not False:
            raise RuntimeError("frozen Stage-2 result claims future-scale evaluation")
        selected = candidate_dir(passing_scale)
        expected_sha = require_lower_sha(
            frozen.get("selected_candidate", {}).get("adapter_sha256"),
            "frozen Stage-2 adapter SHA256",
        )
        require_hash(selected / MODEL_NAME, expected_sha, "frozen Stage-2 candidate")
        require_hash(
            selected / CONFIG_NAME,
            frozen["selected_candidate"]["config_sha256"],
            "frozen Stage-2 candidate config",
        )
        lora_config(selected / CONFIG_NAME, 80, 80, "frozen Stage-2 candidate")
        if frozen.get("selected_candidate", {}).get("path") != str(selected):
            raise RuntimeError("frozen Stage-2 selected path drifted")
    elif FROZEN_STAGE2.exists():
        raise RuntimeError("frozen Stage-2 result exists without a passing decision")

    all_failed = len(decisions) == len(SCALES) and not any(item["pass"] for item in decisions)
    if REJECTED_STAGE2.exists() and not all_failed:
        raise RuntimeError("Stage-2 rejection exists before all scales failed")
    if all_failed and FROZEN_STAGE2.exists():
        raise RuntimeError("Stage-2 cannot be both rejected and selected")
    if REJECTED_STAGE2.exists():
        rejected = load_json(REJECTED_STAGE2)
        expected_decisions = [
            {
                "scale": item["scale"],
                "path": str(item["path"]),
                "sha256": item["sha256"],
                "pass": item["pass"],
            }
            for item in decisions
        ]
        if (
            rejected.get("status") != "REJECTED_NO_LOCKED_SCALE_PASSED"
            or rejected.get("gate_sha256") != GATE_SHA256
            or rejected.get("stage1_ticket_sha256") != stage1["sha256"]
            or rejected.get("executor_sha256") != sha256(SELF)
            or rejected.get("frozen_stage1_step") != stage1["step"]
            or rejected.get("frozen_residual_adapter_sha256")
            != stage1["residual_adapter_sha256"]
            or rejected.get("scale_order") != list(SCALES)
            or rejected.get("decisions") != expected_decisions
            or rejected.get("packaging_authorized") is not False
            or rejected.get("upload_authorized") is not False
        ):
            raise RuntimeError("frozen Stage-2 rejection record drifted")

    return {
        "decisions": decisions,
        "passing_scale": passing_scale,
        "all_failed": all_failed,
        "next_scale": (
            None
            if passing_scale is not None or len(decisions) == len(SCALES)
            else SCALES[len(decisions)]
        ),
    }


def assert_scale_ready(stage1: dict[str, Any], scale: float, allow_candidate: bool) -> None:
    state = sequence_state(stage1)
    if state["passing_scale"] is not None:
        raise RuntimeError(f"Stage 2 already selected scale {state['passing_scale']}")
    if REJECTED_STAGE2.exists():
        raise RuntimeError("Stage 2 is already frozen as locally rejected")
    index = SCALES.index(scale)
    decisions = state["decisions"]
    if len(decisions) != index:
        raise RuntimeError(
            f"scale {scale:g} is not next; locked prefix length is {len(decisions)}"
        )
    if any(item["pass"] for item in decisions):
        raise RuntimeError("a previous Stage-2 scale already passed")
    current = candidate_dir(scale)
    if allow_candidate:
        if not (current / MODEL_NAME).is_file() or not (current / CONFIG_NAME).is_file():
            raise RuntimeError(f"current scale candidate is missing: {current}")
    elif current.exists() or current.is_symlink():
        raise RuntimeError(f"current scale candidate already exists: {current}")
    for later in SCALES[index + 1 :]:
        if candidate_dir(later).exists() or candidate_dir(later).is_symlink():
            raise RuntimeError(f"future scale candidate exists prematurely: {later:g}")


def check_executor_hash(expected: str | None, required: bool) -> str:
    actual = sha256(SELF)
    if required:
        expected = require_lower_sha(expected, "expected executor SHA256")
        if actual != expected:
            raise RuntimeError(f"Stage-2 executor drifted: expected {expected}, got {actual}")
    return actual


def require_run_lock() -> None:
    lock_path = REPORT_ROOT / ".stage2.lock"
    try:
        descriptor = os.fstat(9)
        lock = lock_path.stat()
    except (OSError, FileNotFoundError) as error:
        raise RuntimeError(
            "internal Stage-2 mode requires inherited fd9 from the no-argument launcher"
        ) from error
    if (descriptor.st_dev, descriptor.st_ino) != (lock.st_dev, lock.st_ino):
        raise RuntimeError("inherited fd9 does not identify the Stage-2 run lock")
    try:
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("inherited fd9 does not hold the Stage-2 run lock") from error


def validate_prepare_manifest(
    expected_sha256: str | None,
    stage1: dict[str, Any],
    executor_sha: str,
) -> dict[str, Any]:
    expected = require_lower_sha(
        expected_sha256, "expected Stage-2 prepare manifest SHA256"
    )
    require_hash(PREPARE_MANIFEST, expected, "Stage-2 prepare manifest")
    manifest = load_json(PREPARE_MANIFEST)
    if (
        manifest.get("status")
        != "STAGE2_CPU_PREPARATION_ONLY_GPU_NOT_RUN_SCALE_AXIS_NOT_OPENED"
        or manifest.get("gate", {}).get("sha256") != GATE_SHA256
        or manifest.get("stage1", {}).get("ticket_sha256") != stage1["sha256"]
        or manifest.get("stage1", {}).get("frozen_checkpoint_step")
        != stage1["step"]
        or manifest.get("stage1", {}).get("frozen_residual_adapter_sha256")
        != stage1["residual_adapter_sha256"]
        or manifest.get("stage1", {}).get("prepare_manifest")
        != str(stage1["prepare_manifest"])
        or manifest.get("stage1", {}).get("prepare_manifest_sha256")
        != stage1["prepare_manifest_sha256"]
        or manifest.get("contract", {}).get("scale_order") != list(SCALES)
        or manifest.get("contract", {}).get("stop_first_full_pass") is not True
        or manifest.get("contract", {}).get("checkpoint_reselection") is not False
        or manifest.get("contract", {}).get("additional_or_interpolated_scale")
        is not False
        or manifest.get("contract", {}).get("merged_precheck_temp_scope")
        != str(REPORT_ROOT)
        or manifest.get("contract", {}).get("global_tmp_cleanup") is not False
        or manifest.get("contract", {}).get("prepare_transaction_marker")
        != str(PREPARE_PENDING)
        or manifest.get("contract", {}).get("launcher_refuses_while_pending")
        is not True
        or manifest.get("stage2_opened") is not False
    ):
        raise RuntimeError("Stage-2 prepare manifest contract drifted")
    locked_files = manifest.get("locked_files")
    if not isinstance(locked_files, dict) or not locked_files:
        raise RuntimeError("Stage-2 prepare manifest has no locked-file table")
    for name, record in locked_files.items():
        path = Path(str(record.get("path", "")))
        observed = require_lower_sha(
            record.get("sha256"), f"prepare manifest {name} SHA256"
        )
        require_hash(path, observed, f"prepare manifest locked file {name}")
    if locked_files.get("executor", {}).get("sha256") != executor_sha:
        raise RuntimeError("prepare manifest executor hash drifted")
    if locked_files.get("stage1_ticket", {}).get("sha256") != stage1["sha256"]:
        raise RuntimeError("prepare manifest Stage-1 ticket hash drifted")
    if (
        locked_files.get("stage1_prepare_manifest", {}).get("sha256")
        != stage1["prepare_manifest_sha256"]
    ):
        raise RuntimeError("prepare manifest Stage-1 prepare-manifest hash drifted")
    return manifest


def tensor_identity_audit(
    parent_dir: Path, residual_dir: Path, combined_dir: Path, scale: float
) -> dict[str, Any]:
    from safetensors.torch import load_file

    parent_config = lora_config(parent_dir / CONFIG_NAME, 64, 64, "I-23 parent")
    residual_config = lora_config(residual_dir / CONFIG_NAME, 16, 16, "I-25 residual")
    combined_config = lora_config(combined_dir / CONFIG_NAME, 80, 80, "I-25 composition")
    target_sets = [
        set(parent_config["target_modules"]),
        set(residual_config["target_modules"]),
        set(combined_config["target_modules"]),
    ]
    if not (target_sets[0] == target_sets[1] == target_sets[2]):
        raise RuntimeError("parent/residual/combined target-module sets differ")

    parent = load_file(parent_dir / MODEL_NAME, device="cpu")
    residual = load_file(residual_dir / MODEL_NAME, device="cpu")
    combined = load_file(combined_dir / MODEL_NAME, device="cpu")
    if not (set(parent) == set(residual) == set(combined)):
        raise RuntimeError("parent/residual/combined tensor-key sets differ")
    if len(combined) != 392:
        raise RuntimeError(f"combined tensor count is {len(combined)}, expected 392")

    maximum = 0.0
    per_tensor = []
    for key in sorted(parent):
        parent_tensor = parent[key]
        residual_tensor = residual[key]
        combined_tensor = combined[key]
        if key.endswith("lora_A.weight"):
            expected_shape = (
                parent_tensor.shape[0] + residual_tensor.shape[0],
                *parent_tensor.shape[1:],
            )
            if tuple(combined_tensor.shape) != tuple(expected_shape):
                raise RuntimeError(f"combined A shape mismatch for {key}")
            parent_piece = combined_tensor[: parent_tensor.shape[0]]
            residual_piece = combined_tensor[parent_tensor.shape[0] :]
            parent_expected = parent_tensor
            residual_expected = residual_tensor
        elif key.endswith("lora_B.weight"):
            expected_shape = (
                *parent_tensor.shape[:-1],
                parent_tensor.shape[-1] + residual_tensor.shape[-1],
            )
            if tuple(combined_tensor.shape) != tuple(expected_shape):
                raise RuntimeError(f"combined B shape mismatch for {key}")
            parent_piece = combined_tensor[..., : parent_tensor.shape[-1]]
            residual_piece = combined_tensor[..., parent_tensor.shape[-1] :]
            parent_expected = parent_tensor
            residual_expected = residual_tensor * scale
        else:
            raise RuntimeError(f"unexpected adapter tensor key: {key}")
        parent_error = float((parent_piece.float() - parent_expected.float()).abs().max())
        residual_error = float((residual_piece.float() - residual_expected.float()).abs().max())
        tensor_max = max(parent_error, residual_error)
        if not math.isfinite(tensor_max):
            raise RuntimeError(f"non-finite factor identity error for {key}")
        maximum = max(maximum, tensor_max)
        per_tensor.append(
            {
                "key": key,
                "parent_factor_max_abs": parent_error,
                "scaled_residual_factor_max_abs": residual_error,
                "max_abs": tensor_max,
            }
        )
    if maximum > 1e-6:
        raise RuntimeError(f"tensor-by-tensor additive identity failed: {maximum}")
    del parent, residual, combined
    return {
        "identity": f"delta_combined = delta_I23 + {scale:g} * delta_frozen_I25_residual",
        "proof": (
            "A factors are exact row concatenations; parent B factors are exact and "
            "residual B factors are multiplied by the locked scale. All adapters have "
            "alpha/r=1, so the block-factor identity proves the additive delta identity."
        ),
        "rank": 80,
        "alpha": 80,
        "tensor_count": len(per_tensor),
        "target_module_sets_match": True,
        "tensor_key_sets_match": True,
        "tensor_by_tensor_additive_identity_max_abs": maximum,
        "threshold": 1e-6,
        "pass": True,
        "per_tensor": per_tensor,
    }


def compose_scale(
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    stage1: dict[str, Any],
    scale: float,
    composition_out: Path,
    executor_sha: str,
) -> None:
    assert_scale_ready(stage1, scale, allow_candidate=False)
    composition_out = Path(os.path.abspath(os.fspath(composition_out)))
    running_dir = validate_running_dir(composition_out.parent, scale)
    if composition_out != running_dir / RAW_FILES["composition"]:
        raise ValueError("composition report path is not the locked current-scale path")
    if composition_out.exists() or composition_out.is_symlink():
        raise FileExistsError(composition_out)
    output = candidate_dir(scale)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                str(NORMAL_PYTHON),
                str(paths["combine"]),
                str(paths["parent"]),
                str(stage1["checkpoint"]),
                str(output),
                "--residual-scale",
                f"{scale:g}",
            ],
            check=True,
            text=True,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        raw = json.loads(completed.stdout)
        identity = tensor_identity_audit(
            paths["parent"], stage1["checkpoint"], output, scale
        )
        combined_sha = sha256(output / MODEL_NAME)
        combined_config_sha = sha256(output / CONFIG_NAME)
        if raw.get("combined", {}).get("adapter_sha256") != combined_sha:
            raise RuntimeError("composition tool output adapter hash mismatch")
        if raw.get("combined", {}).get("config_sha256") != combined_config_sha:
            raise RuntimeError("composition tool output config hash mismatch")
        record = {
            "status": "CURRENT_SCALE_CPU_COMPOSITION_COMPLETE_NOT_SELECTED",
            "gate_sha256": GATE_SHA256,
            "stage1_ticket_sha256": stage1["sha256"],
            "executor_sha256": executor_sha,
            "frozen_stage1_step": stage1["step"],
            "scale": scale,
            "parent": {
                "path": str(paths["parent"]),
                "adapter_sha256": gate["parent"]["adapter_sha256"],
                "config_sha256": gate["parent"]["adapter_config_sha256"],
            },
            "residual": {
                "path": str(stage1["checkpoint"]),
                "adapter_sha256": stage1["residual_adapter_sha256"],
                "config_sha256": stage1["residual_config_sha256"],
            },
            "combined": {
                "path": str(output),
                "adapter_sha256": combined_sha,
                "config_sha256": combined_config_sha,
                "adapter_bytes": (output / MODEL_NAME).stat().st_size,
            },
            "locked_composition_tool": {
                "path": str(paths["combine"]),
                "sha256": sha256(paths["combine"]),
                "stdout": raw,
            },
            "identity_audit": identity,
            "integrity_checks": {
                "parent_adapter_sha256_matches_locked_I23": True,
                "residual_adapter_sha256_matches_frozen_Stage1": True,
                "identity_formula_matches_locked_scale": True,
            },
            "selected": False,
        }
        atomic_json(composition_out, record)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "scale": scale,
                    "combined_adapter_sha256": combined_sha,
                }
            )
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def load_composition(
    composition_path: Path,
    gate: dict[str, Any],
    stage1: dict[str, Any],
    scale: float,
    executor_sha: str,
) -> tuple[dict[str, Any], str]:
    composition = load_json(composition_path)
    if composition.get("gate_sha256") != GATE_SHA256:
        raise RuntimeError("composition gate hash drifted")
    if composition.get("stage1_ticket_sha256") != stage1["sha256"]:
        raise RuntimeError("composition Stage-1 ticket hash drifted")
    if composition.get("executor_sha256") != executor_sha:
        raise RuntimeError("composition executor hash drifted")
    if composition.get("frozen_stage1_step") != stage1["step"]:
        raise RuntimeError("composition frozen checkpoint step drifted")
    if finite(composition.get("scale"), "composition scale") != scale:
        raise RuntimeError("composition scale drifted")
    expected_parent = rooted(gate["parent"]["adapter"])
    parent_record = composition.get("parent", {})
    if (
        parent_record.get("path") != str(expected_parent)
        or parent_record.get("adapter_sha256") != gate["parent"]["adapter_sha256"]
        or parent_record.get("config_sha256")
        != gate["parent"]["adapter_config_sha256"]
    ):
        raise RuntimeError("composition parent identity drifted")
    residual_record = composition.get("residual", {})
    if (
        residual_record.get("path") != str(stage1["checkpoint"])
        or residual_record.get("adapter_sha256")
        != stage1["residual_adapter_sha256"]
        or residual_record.get("config_sha256")
        != stage1["residual_config_sha256"]
    ):
        raise RuntimeError("composition frozen residual identity drifted")
    locked_tool = composition.get("locked_composition_tool", {})
    expected_tool = rooted(
        gate["locked_audit_protocols"]["composition"]["tool"]
    )
    if (
        locked_tool.get("path") != str(expected_tool)
        or locked_tool.get("sha256")
        != gate["locked_audit_protocols"]["composition"]["tool_sha256"]
    ):
        raise RuntimeError("composition tool identity drifted")
    combined = composition.get("combined", {})
    current = candidate_dir(scale)
    if combined.get("path") != str(current):
        raise RuntimeError("composition candidate path drifted")
    require_hash(
        current / MODEL_NAME,
        combined.get("adapter_sha256"),
        "current Stage-2 combined adapter",
    )
    require_hash(
        current / CONFIG_NAME,
        combined.get("config_sha256"),
        "current Stage-2 combined config",
    )
    stdout = locked_tool.get("stdout", {})
    expected_formula = (
        f"delta_combined = delta_I23 + {scale:g} * delta_frozen_I25_residual"
    )
    if (
        stdout.get("parent", {}).get("adapter_sha256")
        != gate["parent"]["adapter_sha256"]
        or stdout.get("residual", {}).get("adapter_sha256")
        != stage1["residual_adapter_sha256"]
        or finite(stdout.get("residual", {}).get("multiplier"), "composition stdout scale")
        != scale
        or stdout.get("combined", {}).get("adapter_sha256")
        != combined.get("adapter_sha256")
        or stdout.get("combined", {}).get("config_sha256")
        != combined.get("config_sha256")
        or stdout.get("identity")
        != f"delta_combined = delta_parent + {scale:g} * delta_residual"
    ):
        raise RuntimeError("composition tool stdout identity drifted")
    integrity_checks = composition.get("integrity_checks")
    if integrity_checks != {
        "parent_adapter_sha256_matches_locked_I23": True,
        "residual_adapter_sha256_matches_frozen_Stage1": True,
        "identity_formula_matches_locked_scale": True,
    }:
        raise RuntimeError("composition explicit integrity checks drifted")
    identity = composition.get("identity_audit", {})
    if identity.get("pass") is not True:
        raise RuntimeError("composition identity audit is not a pass")
    if finite(
        identity.get("tensor_by_tensor_additive_identity_max_abs"),
        "composition identity max_abs",
    ) > 1e-6:
        raise RuntimeError("composition identity max_abs exceeds the locked threshold")
    if identity.get("rank") != 80 or identity.get("alpha") != 80:
        raise RuntimeError("composition rank/alpha drifted")
    if identity.get("tensor_count") != 392:
        raise RuntimeError("composition tensor count drifted")
    if identity.get("target_module_sets_match") is not True:
        raise RuntimeError("composition target-module set mismatch")
    if identity.get("tensor_key_sets_match") is not True:
        raise RuntimeError("composition tensor-key set mismatch")
    if identity.get("identity") != expected_formula:
        raise RuntimeError("composition identity formula drifted")
    recomputed_identity = tensor_identity_audit(
        expected_parent, stage1["checkpoint"], current, scale
    )
    for key in (
        "identity",
        "rank",
        "alpha",
        "tensor_count",
        "target_module_sets_match",
        "tensor_key_sets_match",
        "tensor_by_tensor_additive_identity_max_abs",
        "pass",
    ):
        if identity.get(key) != recomputed_identity.get(key):
            raise RuntimeError(f"stored/recomputed composition identity differs at {key}")
    composition["identity_audit"] = recomputed_identity
    composition["integrity_checks"] = integrity_checks
    return composition, combined["adapter_sha256"]


EXPECTED_ACTION_SAMPLING = {
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


def validate_generation_report(
    report: dict[str, Any],
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    adapter: Path,
    adapter_sha: str,
    adapter_config_sha: str,
    rank: int,
    label: str,
) -> dict[str, float]:
    if Path(str(report.get("model", ""))).resolve() != Path(paths["base"]).resolve():
        raise RuntimeError(f"{label} generation base-model path drifted")
    if report.get("protocol_version") != "offline-eval-v4-platform-params":
        raise RuntimeError(f"{label} generation protocol drifted")
    if report.get("sampling", {}).get("action_topic") != EXPECTED_ACTION_SAMPLING:
        raise RuntimeError(f"{label} action sampling drifted")
    if report.get("think_suffix") != "keep":
        raise RuntimeError(f"{label} think suffix drifted")
    selection = report.get("action_selection", {})
    if selection.get("method") != "canonical_json_sha256_ascending":
        raise RuntimeError(f"{label} stable action selection method drifted")
    if selection.get("requested") != 32:
        raise RuntimeError(f"{label} stable action row count drifted")
    if selection.get("source_sha256") != gate["locked_audit_protocols"]["e_action_topic_path"]["action_source_sha256"]:
        raise RuntimeError(f"{label} action source hash drifted")
    if selection.get("manifest_sha256") != paths["stable_action_manifest_sha256"]:
        raise RuntimeError(f"{label} action selection manifest drifted")
    metadata = report.get("adapter", {})
    if metadata.get("path") != str(adapter.resolve()):
        raise RuntimeError(f"{label} adapter path drifted")
    if metadata.get("adapter_sha256") != adapter_sha:
        raise RuntimeError(f"{label} adapter hash drifted")
    if metadata.get("config_sha256") != adapter_config_sha:
        raise RuntimeError(f"{label} adapter config hash drifted")
    if metadata.get("rank") != rank:
        raise RuntimeError(f"{label} adapter rank drifted")
    action = report.get("action", {})
    if action.get("n") != 32:
        raise RuntimeError(f"{label} generation action.n drifted")
    return {
        "f1": finite(action.get("f1_unrounded", action.get("f1")), f"{label} f1"),
        "json_ok": finite(action.get("json_ok_count"), f"{label} json count") / 32,
        "trunc_rate": finite(action.get("trunc_count"), f"{label} trunc count") / 32,
        "max_repeat_p95": finite(action.get("max_repeat_p95"), f"{label} max repeat p95"),
    }


def accept_or_validate_parent_generation(
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    input_path: Path | None,
) -> None:
    if input_path is not None:
        input_path = Path(os.path.abspath(os.fspath(input_path)))
        expected_input = REPORT_ROOT / ".parent-action-generation.running.json"
        if input_path != expected_input:
            raise ValueError(
                f"parent generation staging path must be {expected_input}: {input_path}"
            )
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if PARENT_GENERATION.exists() or PARENT_GENERATION.is_symlink():
            raise FileExistsError(PARENT_GENERATION)
        report = load_json(input_path)
    else:
        if not PARENT_GENERATION.is_file():
            raise FileNotFoundError(PARENT_GENERATION)
        report = load_json(PARENT_GENERATION)
    validate_generation_report(
        report,
        gate,
        paths,
        paths["parent"],
        gate["parent"]["adapter_sha256"],
        gate["parent"]["adapter_config_sha256"],
        64,
        "parent",
    )
    if input_path is not None:
        os.replace(input_path, PARENT_GENERATION)
    print(
        json.dumps(
            {
                "status": "PARENT_GENERATION_REPORT_LOCKED",
                "path": str(PARENT_GENERATION),
                "sha256": sha256(PARENT_GENERATION),
            }
        )
    )


def write_merge_config(
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    stage1: dict[str, Any],
    scale: float,
    composition_path: Path,
    merged_dir: Path,
    output: Path,
    executor_sha: str,
) -> None:
    assert_scale_ready(stage1, scale, allow_candidate=True)
    composition_path = Path(os.path.abspath(os.fspath(composition_path)))
    output = Path(os.path.abspath(os.fspath(output)))
    running_dir = validate_running_dir(composition_path.parent, scale)
    if composition_path != running_dir / RAW_FILES["composition"]:
        raise ValueError("composition input is not the locked current-scale path")
    if output != running_dir / RAW_FILES["merge_config"]:
        raise ValueError("merge config output is not the locked current-scale path")
    load_composition(composition_path, gate, stage1, scale, executor_sha)
    merged_dir = validate_merged_temp_path(merged_dir, scale, must_exist=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = {
        "model_name_or_path": str(paths["base"]),
        "adapter_name_or_path": str(candidate_dir(scale)),
        "template": "qwen3_nothink",
        "trust_remote_code": True,
        "export_dir": str(merged_dir),
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    atomic_text(output, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED_CPU_MERGE_CONFIG_WRITTEN", "scale": scale}))


def assert_identity(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_d_report(
    report: dict[str, Any],
    gate: dict[str, Any],
    stage1: dict[str, Any],
    scale: float,
) -> dict[str, dict[str, Any]]:
    locked = gate["locked_audit_protocols"]["d_residual_mechanism_and_retention"]
    assert_identity(report.get("scales") == [scale], "D report scale drifted")
    assert_identity(
        report.get("requested") == {"action": 128, "topic": 128, "retention": 0},
        "D report requested rows drifted",
    )
    expected_tasks = [
        "material_desc2sid",
        "material_sid2desc",
        "rec_video",
        "rec_prod",
        "rec_ad",
        "rec_living",
    ]
    assert_identity(
        list(report.get("heldout_retention", {})) == expected_tasks,
        "D held-out task order/set drifted",
    )
    for task in expected_tasks:
        assert_identity(
            report["heldout_retention"][task].get("selected") == 96,
            f"D held-out {task} row count drifted",
        )
    artifacts = report.get("artifacts", {})
    expected_hashes = {
        "parent_adapter_sha256": gate["parent"]["adapter_sha256"],
        "residual_adapter_sha256": stage1["residual_adapter_sha256"],
        "dataset_sha256": locked["dataset_sha256"],
        "retention_source_sha256": locked["retention_source_sha256"],
        "retention_exclude_sha256": locked["exact_row_exclude_sha256"],
    }
    for key, expected in expected_hashes.items():
        assert_identity(artifacts.get(key) == expected, f"D report {key} drifted")
    metrics = report.get("metrics_by_scale", {}).get(f"{scale:g}")
    assert_identity(isinstance(metrics, dict), "D report has no locked scale metrics")
    assert_identity(set(metrics) == set(["action", "topic", *expected_tasks]), "D metric task set drifted")
    for task in ("action", "topic"):
        for metric in ("kl", "top1_agreement", "ce_delta"):
            assert_identity(
                metrics[task].get(metric, {}).get("n") == 128,
                f"D {task}/{metric} n drifted",
            )
    for task in expected_tasks:
        for metric in ("kl", "top1_agreement"):
            assert_identity(
                metrics[task].get(metric, {}).get("n") == 96,
                f"D {task}/{metric} n drifted",
            )
    return metrics


def validate_e_user_report(
    report: dict[str, Any], gate: dict[str, Any], candidate_sha: str, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    locked = gate["locked_audit_protocols"]["e_action_topic_path"]
    assert_identity(report.get("status") == "COMPLETE_NOT_A_SCORE_ESTIMATE", "E user status drifted")
    method = report.get("method", {})
    assert_identity(method.get("per_task_requested") == 32, "E user row count drifted")
    assert_identity(method.get("skipped_over_cutoff") == 0, "E user skipped rows")
    source = report.get("source", {}).get("file_sha256", {})
    assert_identity(source.get("action") == locked["action_source_sha256"], "E action source drifted")
    assert_identity(source.get("topic") == locked["topic_source_sha256"], "E topic source drifted")
    models = report.get("models", {})
    assert_identity(set(models) == {"parent", name}, "E user model set drifted")
    assert_identity(models["parent"].get("adapter_sha256") == gate["parent"]["adapter_sha256"], "E user parent hash drifted")
    assert_identity(models[name].get("adapter_sha256") == candidate_sha, "E user candidate hash drifted")
    action = models[name].get("delta_vs_parent", {}).get("action", {})
    topic = models[name].get("delta_vs_parent", {}).get("topic", {})
    assert_identity(action.get("n") == 32, "E action n drifted")
    assert_identity(topic.get("n") == 32, "E topic n drifted")
    return action, topic


def validate_material_report(
    report: dict[str, Any], gate: dict[str, Any], candidate_sha: str
) -> dict[str, Any]:
    locked = gate["locked_audit_protocols"]["d_material_exact_row_holdout"]
    assert_identity(report.get("status") == "COMPLETE_NOT_A_SCORE_ESTIMATE", "material status drifted")
    method = report.get("method", {})
    assert_identity(method.get("seed") == locked["seed"], "material seed drifted")
    assert_identity(method.get("per_task") == 96, "material row count drifted")
    assert_identity(method.get("tasks") == locked["directions"], "material directions drifted")
    holdout = report.get("holdout", {})
    assert_identity(
        holdout.get("combined_selected_manifest_sha256")
        == locked["combined_selected_manifest_sha256"],
        "material selected manifest drifted",
    )
    for task in locked["directions"]:
        assert_identity(holdout.get(task, {}).get("selected") == 96, f"material {task} n drifted")
    artifacts = report.get("artifacts", {})
    base_record = gate["locked_audit_protocols"]["base"]
    d_record = gate["locked_audit_protocols"]["d_residual_mechanism_and_retention"]
    assert_identity(
        artifacts.get("base_config_sha256") == base_record["config_sha256"],
        "material base config hash drifted",
    )
    assert_identity(artifacts.get("parent_adapter_sha256") == gate["parent"]["adapter_sha256"], "material parent hash drifted")
    assert_identity(artifacts.get("candidate_adapter_sha256") == candidate_sha, "material candidate hash drifted")
    assert_identity(
        artifacts.get("source_sha256") == d_record["retention_source_sha256"],
        "material source hash drifted",
    )
    assert_identity(
        artifacts.get("exact_row_exclude_sha256")
        == d_record["exact_row_exclude_sha256"],
        "material exclusion hash drifted",
    )
    metrics = report.get("metrics_by_material_direction", {})
    assert_identity(set(metrics) == set(locked["directions"]), "material metric directions drifted")
    for task in locked["directions"]:
        assert_identity(metrics[task].get("n") == 96, f"material {task} metric n drifted")
    return metrics


def validate_e_rec_report(
    report: dict[str, Any], gate: dict[str, Any], candidate_sha: str, name: str
) -> dict[str, Any]:
    locked = gate["locked_audit_protocols"]["e_recommendation_path"]
    assert_identity(report.get("status") == "COMPLETE_NOT_A_SCORE_ESTIMATE", "E rec status drifted")
    method = report.get("method", {})
    assert_identity(method.get("per_domain") == 64, "E rec rows drifted")
    assert_identity(method.get("batch_size") == 4, "E rec batch size drifted")
    assert_identity(method.get("sample_manifest_sha256") == locked["sample_manifest_sha256"], "E rec manifest drifted")
    assert_identity(report.get("source", {}).get("file_sha256") == locked["source_sha256"], "E rec source hashes drifted")
    models = report.get("models", {})
    assert_identity(set(models) == {"parent", name}, "E rec model set drifted")
    assert_identity(models["parent"].get("adapter_sha256") == gate["parent"]["adapter_sha256"], "E rec parent hash drifted")
    assert_identity(models[name].get("adapter_sha256") == candidate_sha, "E rec candidate hash drifted")
    by_domain = models[name].get("delta_vs_parent", {}).get("by_domain", {})
    assert_identity(set(by_domain) == {"video", "prod", "ad", "live"}, "E rec domain set drifted")
    for domain, metrics in by_domain.items():
        assert_identity(metrics.get("n") == 64, f"E rec {domain} n drifted")
    return by_domain


def parse_precheck(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "action_repeat_collapse": r"A_采样复读崩溃率_仅诊断:\s*(\d+)/(\d+)",
        "itemic_breakage": r"B_itemic结构断裂率:\s*(\d+)/(\d+)",
        "choice_format_survival": r"C_选择题格式存活率_仅诊断:\s*(\d+)/(\d+)",
        "choice_placeholder": r"C_附带_占位符复读:\s*(\d+)/(\d+)",
    }
    parsed: dict[str, int] = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if len(matches) != 1:
            raise RuntimeError(f"precheck log has {len(matches)} matches for {name}")
        numerator, denominator = (int(value) for value in matches[0])
        parsed[name] = numerator
        parsed[f"{name}_rows"] = denominator
    expected_rows = {
        "action_repeat_collapse_rows": 30,
        "itemic_breakage_rows": 60,
        "choice_format_survival_rows": 8,
        "choice_placeholder_rows": 8,
    }
    for key, expected in expected_rows.items():
        if parsed[key] != expected:
            raise RuntimeError(f"precheck {key} drifted: {parsed[key]}/{expected}")
    return parsed


def threshold_checks(
    d_metrics: dict[str, Any],
    e_action: dict[str, Any],
    e_topic: dict[str, Any],
    material: dict[str, Any],
    e_rec: dict[str, Any],
    generation_delta: dict[str, float],
    generation_identity: dict[str, Any],
    composition: dict[str, Any],
    structure: dict[str, int],
    skipped_too_long: int,
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "d_all_requested_rows_scored_and_skipped_over_cutoff_is_0": skipped_too_long == 0,
        "d_action_weighted_ce_delta_mean_le_neg_0_005": finite(d_metrics["action"]["ce_delta"]["mean"], "D action CE mean") <= -0.005,
        "d_action_weighted_ce_delta_median_le_0": finite(d_metrics["action"]["ce_delta"]["median"], "D action CE median") <= 0.0,
        "e_action_gold_sum_logp_delta_mean_ge_0": finite(e_action["gold_sum_logp_delta_mean"], "E action logp delta") >= 0.0,
        "e_action_gold_sum_logp_improved_rate_ge_0_55": finite(e_action["gold_sum_logp_improved_rate"], "E action improved rate") >= 0.55,
        "e_action_top1_agreement_delta_mean_ge_0": finite(e_action["top1_agreement_delta_mean"], "E action top1 delta") >= 0.0,
        "e_action_parent_to_candidate_kl_mean_le_0_02": finite(e_action["parent_to_candidate_kl_mean"], "E action KL") <= 0.02,
        "d_topic_weighted_ce_delta_mean_le_0_005": finite(d_metrics["topic"]["ce_delta"]["mean"], "D topic CE mean") <= 0.005,
        "d_topic_parent_to_candidate_kl_mean_le_0_005": finite(d_metrics["topic"]["kl"]["mean"], "D topic KL") <= 0.005,
        "d_topic_parent_top1_agreement_mean_ge_0_99": finite(d_metrics["topic"]["top1_agreement"]["mean"], "D topic top1") >= 0.99,
        "e_topic_gold_sum_logp_delta_mean_ge_neg_0_01": finite(e_topic["gold_sum_logp_delta_mean"], "E topic logp delta") >= -0.01,
        "e_topic_parent_to_candidate_kl_mean_le_0_005": finite(e_topic["parent_to_candidate_kl_mean"], "E topic KL") <= 0.005,
        "e_topic_top1_agreement_delta_mean_ge_0": finite(e_topic["top1_agreement_delta_mean"], "E topic top1 delta") >= 0.0,
        "generation_paired_protocol_identity_all_true": bool(generation_identity) and all(generation_identity.values()),
        "generation_f1_delta_ge_0": generation_delta["f1"] >= 0.0,
        "generation_json_ok_delta_ge_0": generation_delta["json_ok"] >= 0.0,
        "generation_trunc_rate_delta_le_0": generation_delta["trunc_rate"] <= 0.0,
        "generation_max_repeat_p95_delta_le_0": generation_delta["max_repeat_p95"] <= 0.0,
        "composition_parent_adapter_sha256_matches_locked_I23": composition.get("integrity_checks", {}).get("parent_adapter_sha256_matches_locked_I23") is True,
        "composition_residual_adapter_sha256_matches_frozen_Stage1": composition.get("integrity_checks", {}).get("residual_adapter_sha256_matches_frozen_Stage1") is True,
        "composition_identity_formula_matches_locked_scale": composition.get("integrity_checks", {}).get("identity_formula_matches_locked_scale") is True,
        "composition_rank_is_80": composition["identity_audit"].get("rank") == 80,
        "composition_alpha_is_80": composition["identity_audit"].get("alpha") == 80,
        "composition_tensor_count_is_392": composition["identity_audit"].get("tensor_count") == 392,
        "composition_target_module_sets_match": composition["identity_audit"].get("target_module_sets_match") is True,
        "composition_tensor_key_sets_match": composition["identity_audit"].get("tensor_key_sets_match") is True,
        "composition_tensor_identity_max_abs_le_1e_6": finite(composition["identity_audit"].get("tensor_by_tensor_additive_identity_max_abs"), "composition max abs") <= 1e-6,
        "structure_itemic_breakage_count_le_0": structure["itemic_breakage"] <= 0,
        "structure_action_repeat_collapse_count_le_3": structure["action_repeat_collapse"] <= 3,
        "structure_choice_placeholder_count_le_0": structure["choice_placeholder"] <= 0,
        "structure_choice_format_survival_count_ge_7": structure["choice_format_survival"] >= 7,
    }
    for task in ("material_desc2sid", "material_sid2desc"):
        checks[f"d_{task}_kl_mean_le_0_005"] = finite(d_metrics[task]["kl"]["mean"], f"D {task} KL") <= 0.005
        checks[f"d_{task}_top1_agreement_mean_ge_0_99"] = finite(d_metrics[task]["top1_agreement"]["mean"], f"D {task} top1") >= 0.99
        checks[f"composed_{task}_kl_mean_le_0_005"] = finite(material[task]["parent_to_candidate_kl_mean"], f"composed {task} KL") <= 0.005
        checks[f"composed_{task}_top1_agreement_mean_ge_0_99"] = finite(material[task]["top1_agreement_mean"], f"composed {task} top1") >= 0.99
    domain_map = {
        "video": "rec_video",
        "prod": "rec_prod",
        "ad": "rec_ad",
        "live": "rec_living",
    }
    for domain, d_task in domain_map.items():
        checks[f"d_{domain}_heldout_kl_mean_le_0_005"] = finite(d_metrics[d_task]["kl"]["mean"], f"D {domain} KL") <= 0.005
        checks[f"d_{domain}_heldout_top1_agreement_mean_ge_0_99"] = finite(d_metrics[d_task]["top1_agreement"]["mean"], f"D {domain} top1") >= 0.99
        checks[f"e_{domain}_gold_sum_logp_delta_mean_ge_neg_0_03"] = finite(e_rec[domain]["gold_sum_logp_delta_mean"], f"E {domain} logp delta") >= -0.03
        checks[f"e_{domain}_parent_to_candidate_kl_mean_le_0_005"] = finite(e_rec[domain]["parent_to_candidate_kl_mean"], f"E {domain} KL") <= 0.005
        checks[f"e_{domain}_all_token_rank_le_64_delta_ge_0"] = finite(e_rec[domain]["all_rank_le_64_delta"], f"E {domain} rank64 delta") >= 0.0
    return checks


RAW_FILES = {
    "composition": "composition.json",
    "d_residual": "d-residual.json",
    "e_action_topic": "e-action-topic.json",
    "d_material_composed": "d-material-composed.json",
    "e_recommendation": "e-recommendation.json",
    "candidate_generation": "action-generation-candidate.json",
    "generation_compare": "action-generation-compare.json",
    "merge_config": "merge.yaml",
    "merge_export_log": "merge-export.log",
    "precheck_log": "precheck.log",
}


def report_records(directory: Path, public: Path) -> dict[str, Any]:
    records = {}
    for name, filename in RAW_FILES.items():
        source = directory / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing current-scale report {name}: {source}")
        records[name] = {
            "path": str(public / filename),
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        }
    records["parent_generation"] = {
        "path": str(PARENT_GENERATION),
        "sha256": sha256(PARENT_GENERATION),
        "bytes": PARENT_GENERATION.stat().st_size,
    }
    return records


def remove_directory_strict(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to remove symlink as {label}: {path}")
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"{label} is not a directory: {path}")
        shutil.rmtree(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"failed to remove {label}: {path}")


def remove_file_strict(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to remove symlink as {label}: {path}")
    if path.exists():
        if not path.is_file():
            raise RuntimeError(f"{label} is not a regular file: {path}")
        path.unlink()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"failed to remove {label}: {path}")


def passing_decision_from_frozen(frozen: dict[str, Any]) -> dict[str, Any]:
    frozen_only = {
        "decision_report",
        "selected_candidate",
        "first_full_pass_scale",
        "future_scales_evaluated",
        "compliance",
    }
    return {key: value for key, value in frozen.items() if key not in frozen_only}


def validate_pending_frozen_record(
    pending: dict[str, Any], stage1: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    scale = finite(pending.get("scale"), "pending frozen Stage-2 scale")
    if scale not in SCALES:
        raise RuntimeError(f"pending frozen Stage-2 scale is not locked: {scale:g}")
    candidate = pending.get("candidate")
    checks = pending.get("checks")
    if (
        pending.get("status") != "PASS_FROZEN_STAGE2_SCALE"
        or pending.get("pass") is not True
        or pending.get("gate_sha256") != GATE_SHA256
        or pending.get("stage1_ticket_sha256") != stage1["sha256"]
        or pending.get("executor_sha256") != sha256(SELF)
        or pending.get("frozen_stage1_step") != stage1["step"]
        or pending.get("frozen_residual_adapter_sha256")
        != stage1["residual_adapter_sha256"]
        or not isinstance(checks, dict)
        or len(checks) != 59
        or not all(value is True for value in checks.values())
        or not isinstance(candidate, dict)
        or candidate.get("path") != str(candidate_dir(scale))
        or candidate.get("rank") != 80
        or candidate.get("alpha") != 80
        or pending.get("selected_candidate") != candidate
        or pending.get("first_full_pass_scale") != scale
        or pending.get("future_scales_evaluated") is not False
        or pending.get("packaging_authorized") is not False
        or pending.get("upload_authorized") is not False
    ):
        raise RuntimeError("pending frozen Stage-2 record contract drifted")
    require_lower_sha(
        candidate.get("adapter_sha256"), "pending Stage-2 candidate adapter SHA256"
    )
    require_lower_sha(
        candidate.get("config_sha256"), "pending Stage-2 candidate config SHA256"
    )
    decision_report = pending.get("decision_report")
    expected_decision_path = scale_report_dir(scale) / "gate-decision.json"
    if (
        not isinstance(decision_report, dict)
        or decision_report.get("path") != str(expected_decision_path)
    ):
        raise RuntimeError("pending frozen Stage-2 decision path drifted")
    require_lower_sha(
        decision_report.get("sha256"), "pending Stage-2 decision report SHA256"
    )
    return scale, passing_decision_from_frozen(pending)


def validate_pending_candidate(pending: dict[str, Any], scale: float) -> None:
    candidate = pending["candidate"]
    selected = candidate_dir(scale)
    require_hash(
        selected / MODEL_NAME,
        candidate["adapter_sha256"],
        "pending Stage-2 candidate",
    )
    require_hash(
        selected / CONFIG_NAME,
        candidate["config_sha256"],
        "pending Stage-2 candidate config",
    )
    lora_config(selected / CONFIG_NAME, 80, 80, "pending Stage-2 candidate")


def validate_failed_prefix_before_pending(
    stage1: dict[str, Any], pending_scale: float
) -> None:
    pending_index = SCALES.index(pending_scale)
    for scale in SCALES[:pending_index]:
        decision_path = scale_report_dir(scale) / "gate-decision.json"
        if not decision_path.is_file():
            raise RuntimeError(
                f"pending Stage-2 pass has a gap before scale {pending_scale:g}"
            )
        if validate_published_decision(load_json(decision_path), scale, stage1):
            raise RuntimeError("pending Stage-2 pass follows an earlier passing scale")
        if candidate_dir(scale).exists() or candidate_dir(scale).is_symlink():
            raise RuntimeError(
                f"failed Stage-2 candidate remains before pending pass: {scale:g}"
            )
    for scale in SCALES[pending_index + 1 :]:
        if scale_report_dir(scale).exists() or scale_report_dir(scale).is_symlink():
            raise RuntimeError("a future Stage-2 report exists after a pending pass")
        if candidate_dir(scale).exists() or candidate_dir(scale).is_symlink():
            raise RuntimeError("a future Stage-2 candidate exists after a pending pass")


def running_directories(scale: float) -> list[Path]:
    matches = sorted(REPORT_ROOT.glob(f".running-{SCALE_TAG[scale]}.*"))
    return [validate_running_dir(path, scale) for path in matches]


def merged_directories(scale: float) -> list[Path]:
    matches = sorted(REPORT_ROOT.glob(f".merged-{SCALE_TAG[scale]}.*"))
    return [validate_merged_temp_path(path, scale, must_exist=True) for path in matches]


def validate_staged_passing_decision(
    raw_dir: Path,
    pending: dict[str, Any],
    decision: dict[str, Any],
    scale: float,
) -> None:
    raw_decision = raw_dir / "gate-decision.json"
    if load_json(raw_decision) != decision:
        raise RuntimeError("staged passing decision differs from pending frozen record")
    if sha256(raw_decision) != pending["decision_report"]["sha256"]:
        raise RuntimeError("staged passing decision hash differs from pending record")
    reports = decision.get("reports")
    if not isinstance(reports, dict) or set(reports) != set(RAW_FILES) | {
        "parent_generation"
    }:
        raise RuntimeError("staged passing decision report set drifted")
    for name, filename in RAW_FILES.items():
        source = raw_dir / filename
        record = reports[name]
        require_hash(
            source,
            require_lower_sha(
                record.get("sha256"), f"staged passing {name} SHA256"
            ),
            f"staged passing report {name}",
        )
        if (
            record.get("path") != str(scale_report_dir(scale) / filename)
            or record.get("bytes") != source.stat().st_size
        ):
            raise RuntimeError(f"staged passing report identity drifted for {name}")
    parent = reports["parent_generation"]
    require_hash(
        PARENT_GENERATION,
        require_lower_sha(
            parent.get("sha256"), "staged passing parent generation SHA256"
        ),
        "staged passing parent generation",
    )
    if (
        parent.get("path") != str(PARENT_GENERATION)
        or parent.get("bytes") != PARENT_GENERATION.stat().st_size
    ):
        raise RuntimeError("staged passing parent generation identity drifted")


def cleanup_unpublished_runtime_artifacts(
    stage1: dict[str, Any], state: dict[str, Any] | None = None
) -> dict[str, Any]:
    if state is None:
        state = sequence_state(stage1)
    if state["passing_scale"] is not None or state["all_failed"]:
        return {"candidate_directories": 0, "raw_directories": 0, "merged_directories": 0, "files": 0}
    next_scale = state["next_scale"]
    assert next_scale is not None
    next_index = SCALES.index(next_scale)
    removed = {
        "candidate_directories": 0,
        "raw_directories": 0,
        "merged_directories": 0,
        "files": 0,
    }
    for scale in SCALES[next_index:]:
        path = candidate_dir(scale)
        if path.exists() or path.is_symlink():
            remove_directory_strict(path, f"stale unpublished {SCALE_TAG[scale]} candidate")
            removed["candidate_directories"] += 1
    for scale in SCALES:
        for path in running_directories(scale):
            remove_directory_strict(path, f"stale unpublished {SCALE_TAG[scale]} reports")
            removed["raw_directories"] += 1
        for path in merged_directories(scale):
            remove_directory_strict(path, f"stale unpublished {SCALE_TAG[scale]} merge")
            removed["merged_directories"] += 1
    parent_staging = REPORT_ROOT / ".parent-action-generation.running.json"
    if parent_staging.exists() or parent_staging.is_symlink():
        remove_file_strict(parent_staging, "stale parent-generation staging file")
        removed["files"] += 1
    return removed


def reconcile_pending_commit(stage1: dict[str, Any]) -> dict[str, Any]:
    if not (FROZEN_STAGE2_PENDING.exists() or FROZEN_STAGE2_PENDING.is_symlink()):
        state = sequence_state(stage1)
        return {
            "recovery": "NONE",
            "cleanup": cleanup_unpublished_runtime_artifacts(stage1, state),
            "state": sequence_state(stage1),
        }
    if FROZEN_STAGE2_PENDING.is_symlink() or not FROZEN_STAGE2_PENDING.is_file():
        raise RuntimeError("pending frozen Stage-2 record is not a regular file")
    pending = load_json(FROZEN_STAGE2_PENDING)
    scale, decision = validate_pending_frozen_record(pending, stage1)
    validate_failed_prefix_before_pending(stage1, scale)
    public = scale_report_dir(scale)
    decision_path = public / "gate-decision.json"

    if FROZEN_STAGE2.exists() or FROZEN_STAGE2.is_symlink():
        if FROZEN_STAGE2.is_symlink() or not FROZEN_STAGE2.is_file():
            raise RuntimeError("frozen Stage-2 result is not a regular file")
        if load_json(FROZEN_STAGE2) != pending:
            raise RuntimeError("pending and committed Stage-2 frozen records differ")
        validate_pending_candidate(pending, scale)
        if not decision_path.is_file():
            raise RuntimeError("committed Stage-2 result has no published pass")
        published = load_json(decision_path)
        if (
            published != decision
            or sha256(decision_path) != pending["decision_report"]["sha256"]
            or not validate_published_decision(published, scale, stage1)
        ):
            raise RuntimeError("committed Stage-2 result differs from its published pass")
        remove_file_strict(FROZEN_STAGE2_PENDING, "redundant pending frozen result")
        return {
            "recovery": "REMOVED_REDUNDANT_PENDING_RECORD",
            "cleanup": {},
            "state": sequence_state(stage1),
        }

    if decision_path.is_file():
        validate_pending_candidate(pending, scale)
        published = load_json(decision_path)
        if published != decision:
            raise RuntimeError("published pass differs from pending frozen decision")
        if sha256(decision_path) != pending["decision_report"]["sha256"]:
            raise RuntimeError("published pass hash differs from pending frozen record")
        if not validate_published_decision(published, scale, stage1):
            raise RuntimeError("pending commit points to a nonpassing decision")
        os.replace(FROZEN_STAGE2_PENDING, FROZEN_STAGE2)
        return {
            "recovery": "COMMITTED_PUBLISHED_PASS",
            "cleanup": {},
            "state": sequence_state(stage1),
        }
    if public.exists() or public.is_symlink():
        raise RuntimeError("pending pass has an incomplete public report directory")

    raw_matches = running_directories(scale)
    if len(raw_matches) > 1:
        raise RuntimeError(
            "pending unpublished pass has multiple staged report directories"
        )
    selected = candidate_dir(scale)
    candidate_complete = (
        (selected / MODEL_NAME).is_file() and (selected / CONFIG_NAME).is_file()
    )
    if len(raw_matches) == 1 and candidate_complete:
        validate_pending_candidate(pending, scale)
        validate_staged_passing_decision(raw_matches[0], pending, decision, scale)
    state = sequence_state(stage1)
    cleanup = cleanup_unpublished_runtime_artifacts(stage1, state)
    remove_file_strict(FROZEN_STAGE2_PENDING, "rolled-back pending frozen result")
    return {
        "recovery": (
            "ROLLED_BACK_VERIFIED_UNPUBLISHED_PASS"
            if len(raw_matches) == 1 and candidate_complete
            else "ROLLED_BACK_PARTIALLY_CLEANED_UNPUBLISHED_PASS"
        ),
        "cleanup": cleanup,
        "state": sequence_state(stage1),
    }


def validate_merge_artifacts(
    merge_config_path: Path,
    merged_dir: Path,
    paths: dict[str, Path | str],
    scale: float,
) -> dict[str, Any]:
    expected = {
        "model_name_or_path": str(paths["base"]),
        "adapter_name_or_path": str(candidate_dir(scale)),
        "template": "qwen3_nothink",
        "trust_remote_code": True,
        "export_dir": str(merged_dir),
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    if load_json(merge_config_path) != expected:
        raise RuntimeError("CPU merge config drifted")
    if not (merged_dir / "config.json").is_file():
        raise FileNotFoundError("merged precheck model has no config.json")
    weights = sorted(
        path for path in merged_dir.glob("*.safetensors") if path.is_file()
    )
    if not weights:
        raise FileNotFoundError("merged precheck model has no safetensors weights")
    if (merged_dir / MODEL_NAME).exists():
        raise RuntimeError("merged precheck model still contains an adapter-only weight file")
    return {
        "path": str(merged_dir),
        "config_sha256": sha256(merged_dir / "config.json"),
        "weight_files": [path.name for path in weights],
        "weight_file_count": len(weights),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "merge_config_sha256": sha256(merge_config_path),
    }


def judge_scale(
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    stage1: dict[str, Any],
    scale: float,
    raw_dir: Path,
    merged_dir: Path,
    executor_sha: str,
) -> bool:
    assert_scale_ready(stage1, scale, allow_candidate=True)
    raw_dir = validate_running_dir(raw_dir, scale)
    merged_dir = validate_merged_temp_path(merged_dir, scale, must_exist=True)
    public = scale_report_dir(scale)
    if public.exists() or public.is_symlink():
        raise FileExistsError(public)
    if not PARENT_GENERATION.is_file():
        raise FileNotFoundError(PARENT_GENERATION)

    composition_path = raw_dir / RAW_FILES["composition"]
    composition, candidate_sha = load_composition(
        composition_path, gate, stage1, scale, executor_sha
    )
    candidate_config_sha = composition["combined"]["config_sha256"]
    name = SCALE_TAG[scale]

    d_raw = load_json(raw_dir / RAW_FILES["d_residual"])
    d_metrics = validate_d_report(d_raw, gate, stage1, scale)
    e_user_raw = load_json(raw_dir / RAW_FILES["e_action_topic"])
    e_action, e_topic = validate_e_user_report(e_user_raw, gate, candidate_sha, name)
    material_raw = load_json(raw_dir / RAW_FILES["d_material_composed"])
    material = validate_material_report(material_raw, gate, candidate_sha)
    e_rec_raw = load_json(raw_dir / RAW_FILES["e_recommendation"])
    e_rec = validate_e_rec_report(e_rec_raw, gate, candidate_sha, name)

    parent_generation_raw = load_json(PARENT_GENERATION)
    parent_generation_metrics = validate_generation_report(
        parent_generation_raw,
        gate,
        paths,
        paths["parent"],
        gate["parent"]["adapter_sha256"],
        gate["parent"]["adapter_config_sha256"],
        64,
        "parent",
    )
    candidate_generation_path = raw_dir / RAW_FILES["candidate_generation"]
    candidate_generation_raw = load_json(candidate_generation_path)
    candidate_generation_metrics = validate_generation_report(
        candidate_generation_raw,
        gate,
        paths,
        candidate_dir(scale),
        candidate_sha,
        candidate_config_sha,
        80,
        "candidate",
    )
    generation_delta = {
        key: candidate_generation_metrics[key] - parent_generation_metrics[key]
        for key in parent_generation_metrics
    }
    compare_path = raw_dir / RAW_FILES["generation_compare"]
    compare_raw = load_json(compare_path)
    assert_identity(compare_raw.get("status") == "COMPLETE_NOT_A_SCORE_ESTIMATE", "generation compare status drifted")
    assert_identity(Path(compare_raw.get("parent_report", "")).resolve() == PARENT_GENERATION.resolve(), "generation compare parent path drifted")
    assert_identity(Path(compare_raw.get("candidate_report", "")).resolve() == candidate_generation_path.resolve(), "generation compare candidate path drifted")
    generation_identity = compare_raw.get("identity_checks", {})
    expected_generation_identity = {
        "protocol",
        "action_sampling",
        "think_suffix",
        "selection_method",
        "selection_manifest",
        "selection_source",
        "row_count",
    }
    assert_identity(
        isinstance(generation_identity, dict)
        and set(generation_identity) == expected_generation_identity
        and all(isinstance(value, bool) for value in generation_identity.values()),
        "generation identity checks missing or malformed",
    )
    checker_delta = compare_raw.get("metrics", {}).get("delta", {})
    for key in ("f1", "json_ok", "trunc_rate", "max_repeat_p95"):
        assert_identity(
            abs(finite(checker_delta.get(key), f"checker generation delta {key}") - generation_delta[key]) <= 1e-12,
            f"generation checker delta drifted for {key}",
        )

    merge_record = validate_merge_artifacts(
        raw_dir / RAW_FILES["merge_config"], merged_dir, paths, scale
    )
    structure = parse_precheck(raw_dir / RAW_FILES["precheck_log"])
    checks = threshold_checks(
        d_metrics,
        e_action,
        e_topic,
        material,
        e_rec,
        generation_delta,
        generation_identity,
        composition,
        structure,
        int(d_raw.get("skipped_too_long", -1)),
    )
    passed = all(checks.values())
    records = report_records(raw_dir, public)
    decision = {
        "status": (
            "PASS_FROZEN_STAGE2_SCALE"
            if passed
            else "FAIL_CONTINUE_ASCENDING_SCALE_AXIS"
        ),
        "gate_sha256": GATE_SHA256,
        "stage1_ticket_sha256": stage1["sha256"],
        "executor_sha256": executor_sha,
        "frozen_stage1_step": stage1["step"],
        "frozen_residual_adapter_sha256": stage1["residual_adapter_sha256"],
        "scale": scale,
        "selection_axis": "locked residual scale only on the frozen Stage-1 checkpoint",
        "selection_rule": "ascending scale order; stop at the first full pass",
        "candidate": {
            "path": str(candidate_dir(scale)),
            "adapter_sha256": candidate_sha,
            "config_sha256": candidate_config_sha,
            "rank": 80,
            "alpha": 80,
        },
        "checks": checks,
        "pass": passed,
        "reports": records,
        "generation_exact_metrics": {
            "parent": parent_generation_metrics,
            "candidate": candidate_generation_metrics,
            "delta": generation_delta,
        },
        "merged_precheck_artifact": merge_record,
        "structure_counts": structure,
        "generation_checker_boundary_override": (
            "The locked I-24 comparison helper labels max_repeat_p95 as diagnostic; "
            "I-25 independently hard-gates max_repeat_p95_delta <= 0."
        ),
        "online_score_estimate": None,
        "packaging_authorized": False,
        "upload_authorized": False,
    }
    raw_decision_path = raw_dir / "gate-decision.json"
    atomic_json(raw_decision_path, decision)
    decision_path = public / "gate-decision.json"
    remove_directory_strict(merged_dir, "current-scale merged precheck model")
    if passed:
        frozen = {
            **decision,
            "decision_report": {
                "path": str(decision_path),
                "sha256": sha256(raw_decision_path),
            },
            "selected_candidate": decision["candidate"],
            "first_full_pass_scale": scale,
            "future_scales_evaluated": False,
            "compliance": "PARAMETER_CONCATENATION_OR_MODEL_FUSION_GRAY_AREA",
        }
        if FROZEN_STAGE2_PENDING.exists() or FROZEN_STAGE2_PENDING.is_symlink():
            raise FileExistsError(FROZEN_STAGE2_PENDING)
        atomic_json(FROZEN_STAGE2_PENDING, frozen)
        try:
            os.replace(raw_dir, public)
            os.replace(FROZEN_STAGE2_PENDING, FROZEN_STAGE2)
        except BaseException:
            if public.is_dir() and not raw_dir.exists():
                os.replace(public, raw_dir)
            if FROZEN_STAGE2_PENDING.exists():
                FROZEN_STAGE2_PENDING.unlink()
            if FROZEN_STAGE2.exists():
                FROZEN_STAGE2.unlink()
            raise
    else:
        remove_directory_strict(
            candidate_dir(scale), "failed current-scale combined candidate"
        )
        os.replace(raw_dir, public)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "scale": scale,
                "pass": passed,
                "failed_check_count": sum(not value for value in checks.values()),
            }
        )
    )
    return passed


def finalize_rejection(stage1: dict[str, Any], executor_sha: str) -> None:
    state = sequence_state(stage1)
    if not state["all_failed"]:
        raise RuntimeError("cannot reject Stage 2 before all five locked scales fail")
    if FROZEN_STAGE2.exists():
        raise RuntimeError("cannot reject Stage 2 after a passing scale")
    if REJECTED_STAGE2.exists() or REJECTED_STAGE2.is_symlink():
        raise FileExistsError(REJECTED_STAGE2)
    leftovers = [str(path) for path in CANDIDATE_ROOT.glob("*") if path.exists()]
    if leftovers:
        raise RuntimeError(f"nonselected Stage-2 candidates were not cleaned: {leftovers}")
    result = {
        "status": "REJECTED_NO_LOCKED_SCALE_PASSED",
        "gate_sha256": GATE_SHA256,
        "stage1_ticket_sha256": stage1["sha256"],
        "executor_sha256": executor_sha,
        "frozen_stage1_step": stage1["step"],
        "frozen_residual_adapter_sha256": stage1["residual_adapter_sha256"],
        "scale_order": list(SCALES),
        "decisions": [
            {
                "scale": item["scale"],
                "path": str(item["path"]),
                "sha256": item["sha256"],
                "pass": item["pass"],
            }
            for item in state["decisions"]
        ],
        "checkpoint_reselection_forbidden": True,
        "new_scale_forbidden": True,
        "packaging_authorized": False,
        "upload_authorized": False,
    }
    atomic_json(REJECTED_STAGE2, result)
    print(json.dumps({"status": result["status"], "evaluated_scales": len(SCALES)}))


def internal_common(
    expected_executor_sha: str | None,
    expected_manifest_sha: str | None,
) -> tuple[dict[str, Any], dict[str, Path | str], dict[str, Any], str]:
    require_run_lock()
    executor_sha = check_executor_hash(expected_executor_sha, required=True)
    gate, paths = static_contract()
    stage1 = load_stage1_ticket(gate, paths, required=True)
    assert stage1 is not None
    validate_prepare_manifest(expected_manifest_sha, stage1, executor_sha)
    return gate, paths, stage1, executor_sha


def manifest_record(paths: dict[str, Path | str], stage1: dict[str, Any], executor_sha: str, gpu: str) -> dict[str, Any]:
    locked_files = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
        if isinstance(path, Path) and path.is_file()
    }
    locked_files["executor"] = {"path": str(SELF), "sha256": executor_sha}
    locked_files["stage1_ticket"] = {"path": str(STAGE1_TICKET), "sha256": stage1["sha256"]}
    locked_files["stage1_prepare_manifest"] = {
        "path": str(stage1["prepare_manifest"]),
        "sha256": stage1["prepare_manifest_sha256"],
    }
    return {
        "status": "STAGE2_CPU_PREPARATION_ONLY_GPU_NOT_RUN_SCALE_AXIS_NOT_OPENED",
        "gate": {"path": str(paths["gate"]), "sha256": GATE_SHA256},
        "stage1": {
            "ticket": str(STAGE1_TICKET),
            "ticket_sha256": stage1["sha256"],
            "frozen_checkpoint_step": stage1["step"],
            "frozen_residual_adapter_sha256": stage1["residual_adapter_sha256"],
            "prepare_manifest": str(stage1["prepare_manifest"]),
            "prepare_manifest_sha256": stage1["prepare_manifest_sha256"],
        },
        "contract": {
            "scale_order": list(SCALES),
            "stop_first_full_pass": True,
            "future_scale_not_materialized_before_prior_decision": True,
            "failed_candidate_deleted_before_decision_publication": True,
            "passing_candidate_retained_only_after_all_checks": True,
            "checkpoint_reselection": False,
            "additional_or_interpolated_scale": False,
            "merged_precheck_temp_scope": str(REPORT_ROOT),
            "global_tmp_cleanup": False,
            "prepare_transaction_marker": str(PREPARE_PENDING),
            "launcher_refuses_while_pending": True,
            "packaging": False,
            "upload": False,
        },
        "gpu_launcher": {
            "path": str(COMMAND_LAUNCHER),
            "arguments": [],
            "gpu": gpu,
            "single_process_lock": str(REPORT_ROOT / ".stage2.lock"),
            "normal_python": str(NORMAL_PYTHON),
            "vllm_python": str(VLLM_PYTHON),
            "cpu_export_cli": str(EXPORT_CLI),
        },
        "outputs": {
            "report_root": str(REPORT_ROOT),
            "scale_candidate_root": str(CANDIDATE_ROOT),
            "frozen_result": str(FROZEN_STAGE2),
            "rejected_result": str(REJECTED_STAGE2),
        },
        "locked_files": locked_files,
        "stable_action_manifest_sha256": paths["stable_action_manifest_sha256"],
        "stage2_opened": False,
    }


def command_for_scale(
    scale: float,
    gpu: str,
    executor_sha: str,
    manifest_sha: str,
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    stage1: dict[str, Any],
) -> list[str]:
    tag = SCALE_TAG[scale]
    candidate = candidate_dir(scale)
    name_arg = f"{tag}={candidate}"
    common = (
        f"--expected-executor-sha256 {q(executor_sha)} "
        f"--expected-manifest-sha256 {q(manifest_sha)}"
    )
    d_tasks = "material_desc2sid,material_sid2desc,rec_video,rec_prod,rec_ad,rec_living"
    return [
        f"echo '[i25-stage2] evaluating locked scale {scale:g}'",
        f"raw_dir=$(mktemp -d {q(str(REPORT_ROOT / ('.running-' + tag + '.XXXXXX')))})",
        "chmod 700 \"$raw_dir\"",
        f"merged_dir=$(mktemp -d {q(str(REPORT_ROOT / ('.merged-' + tag + '.XXXXXX')))})",
        "chmod 700 \"$merged_dir\"",
        "rmdir -- \"$merged_dir\"",
        f"candidate_dir={q(candidate)}",
        f"{q(NORMAL_PYTHON)} {q(SELF)} --compose-scale --scale {scale:g} --composition-out \"$raw_dir/composition.json\" {common} >\"$raw_dir/compose.stdout.log\" 2>&1",
        " ".join(
            [
                q(NORMAL_PYTHON), q(paths["d_tool"]),
                "--base", q(paths["base"]), "--parent", q(paths["parent"]),
                "--residual", q(stage1["checkpoint"]), "--dataset", q(paths["dataset"]),
                "--retention-source", q(paths["retention_source"]),
                "--retention-exclude", q(paths["retention_exclude"]),
                "--retention-tasks", q(d_tasks), "--retention-per-task", "96",
                "--output", '"$raw_dir/d-residual.json"', "--gpu", q(gpu),
                "--action-n", "128", "--topic-n", "128", "--retention-n", "0",
                "--cutoff", "16384", "--seed", "19260821", "--scales", f"{scale:g}",
                '>"$raw_dir/d-residual.stdout.log" 2>&1',
            ]
        ),
        " ".join(
            [
                q(NORMAL_PYTHON), q(paths["e_user_tool"]),
                "--base", q(paths["base"]), "--parent", q(paths["parent"]),
                "--dev", q(paths["dev"]), "--candidate", q(name_arg),
                "--out", '"$raw_dir/e-action-topic.json"', "--gpu", q(gpu),
                "--per-task", "32", "--cutoff", "16384",
                '>"$raw_dir/e-action-topic.stdout.log" 2>&1',
            ]
        ),
        " ".join(
            [
                q(NORMAL_PYTHON), q(paths["d_material_tool"]),
                "--base", q(paths["base"]), "--parent", q(paths["parent"]),
                "--candidate", q(candidate), "--source", q(paths["retention_source"]),
                "--exclude", q(paths["retention_exclude"]),
                "--seed", "19260827", "--per-task", "96", "--cutoff", "16384",
                "--out", '"$raw_dir/d-material-composed.json"', "--gpu", q(gpu),
                '>"$raw_dir/d-material-composed.stdout.log" 2>&1',
            ]
        ),
        " ".join(
            [
                q(NORMAL_PYTHON), q(paths["e_rec_tool"]),
                "--base", q(paths["base"]), "--parent", q(paths["parent"]),
                "--dev", q(paths["dev"]), "--candidate", q(name_arg),
                "--out", '"$raw_dir/e-recommendation.json"', "--gpu", q(gpu),
                "--per-domain", "64", "--batch-size", "4",
                '>"$raw_dir/e-recommendation.stdout.log" 2>&1',
            ]
        ),
        " ".join(
            [
                q(VLLM_PYTHON), q(paths["generation_tool"]),
                "--model", q(paths["base"]), "--adapter", q(candidate),
                "--gpu", q(gpu), "--dims", "action", "--stable_action_rows", "32",
                "--think_suffix", "keep", "--tag", q(f"i25-{tag}"),
                "--out", '"$raw_dir/action-generation-candidate.json"',
                '>"$raw_dir/action-generation-candidate.stdout.log" 2>&1',
            ]
        ),
        " ".join(
            [
                q(NORMAL_PYTHON), q(paths["generation_checker"]),
                "--parent", q(PARENT_GENERATION),
                "--candidate", '"$raw_dir/action-generation-candidate.json"',
                "--out", '"$raw_dir/action-generation-compare.json"',
                '>"$raw_dir/action-generation-compare.stdout.log" 2>&1',
            ]
        ),
        f"{q(NORMAL_PYTHON)} {q(SELF)} --write-merge-config --scale {scale:g} --composition-out \"$raw_dir/composition.json\" --merged-dir \"$merged_dir\" --merge-config-out \"$raw_dir/merge.yaml\" {common} >\"$raw_dir/merge-config.stdout.log\" 2>&1",
        f"CUDA_VISIBLE_DEVICES='' {q(EXPORT_CLI)} export \"$raw_dir/merge.yaml\" >\"$raw_dir/merge-export.log\" 2>&1",
        " ".join(
            [
                q(NORMAL_PYTHON), q(paths["precheck"]),
                "--model", '"$merged_dir"', "--gpu", q(gpu), "--n", "30",
                "--seed", "2026", "--batch-size", "8",
                '>"$raw_dir/precheck.log" 2>&1',
            ]
        ),
        "set +e",
        f"{q(NORMAL_PYTHON)} {q(SELF)} --judge-scale --scale {scale:g} --raw-dir \"$raw_dir\" --merged-dir \"$merged_dir\" {common}",
        "decision_rc=$?",
        "set -e",
        "if [[ $decision_rc -eq 0 ]]; then",
        "  raw_dir=",
        "  merged_dir=",
        "  candidate_dir=",
        f"  echo '[i25-stage2] froze first fully passing scale: {scale:g}'",
        "  exit 0",
        "fi",
        f"if [[ $decision_rc -ne {EXIT_SCALE_FAILED} ]]; then",
        f"  echo '[i25-stage2] judge failed unexpectedly at scale {scale:g}' >&2",
        "  exit \"$decision_rc\"",
        "fi",
        "raw_dir=",
        "merged_dir=",
        "candidate_dir=",
        "",
    ]


def guarded_scale_lines(
    scale: float, status_command: str, scale_commands: list[str]
) -> list[str]:
    lines = [
        "set +e",
        status_command,
        "status_rc=$?",
        "set -e",
        f"if [[ $status_rc -eq {EXIT_ALREADY_FAILED} ]]; then",
        f"  echo '[i25-stage2] scale {scale:g} already failed; skipping this fixed block'",
        f"elif [[ $status_rc -eq {EXIT_ALREADY_SELECTED} ]]; then",
        "  exit 0",
        f"elif [[ $status_rc -eq {EXIT_ALREADY_REJECTED} ]]; then",
        "  exit 4",
        "elif [[ $status_rc -ne 0 ]]; then",
        "  exit \"$status_rc\"",
        "else",
    ]
    lines.extend(f"  {line}" if line else "" for line in scale_commands)
    lines.extend(["fi", ""])
    return lines


def command_launcher_text(
    gpu: str,
    executor_sha: str,
    manifest_sha: str,
    gate: dict[str, Any],
    paths: dict[str, Path | str],
    stage1: dict[str, Any],
) -> str:
    common = (
        f"--expected-executor-sha256 {q(executor_sha)} "
        f"--expected-manifest-sha256 {q(manifest_sha)}"
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# No arguments are accepted. Checkpoint, scales, order, and thresholds are fixed.",
        "if [[ $# -ne 0 ]]; then",
        "  echo '[i25-stage2] this launcher accepts no arguments' >&2",
        "  exit 2",
        "fi",
        f"if [[ -e {q(PREPARE_PENDING)} || -L {q(PREPARE_PENDING)} ]]; then",
        "  echo '[i25-stage2] CPU preparation transaction is incomplete' >&2",
        "  exit 1",
        "fi",
        f"mkdir -p {q(REPORT_ROOT)}",
        f"exec 9>{q(REPORT_ROOT / '.stage2.lock')}",
        "if ! flock -n 9; then",
        "  echo '[i25-stage2] another Stage-2 executor holds the lock' >&2",
        "  exit 1",
        "fi",
        "raw_dir=",
        "merged_dir=",
        "candidate_dir=",
        "parent_tmp=",
        "cleanup_current() {",
        (
            f"  if [[ -e {q(FROZEN_STAGE2_PENDING)} || -L {q(FROZEN_STAGE2_PENDING)} "
            f"|| -e {q(FROZEN_STAGE2)} || -L {q(FROZEN_STAGE2)} ]]; then"
        ),
        "    echo '[i25-stage2] preserving pending/frozen artifacts for locked recovery' >&2",
        "    return",
        "  fi",
        "  if [[ -n \"${merged_dir:-}\" && -d \"$merged_dir\" ]]; then rm -rf -- \"$merged_dir\"; fi",
        "  if [[ -n \"${candidate_dir:-}\" && -d \"$candidate_dir\" ]]; then rm -rf -- \"$candidate_dir\"; fi",
        "  if [[ -n \"${raw_dir:-}\" && -d \"$raw_dir\" ]]; then rm -rf -- \"$raw_dir\"; fi",
        "  if [[ -n \"${parent_tmp:-}\" && -f \"$parent_tmp\" ]]; then rm -f -- \"$parent_tmp\"; fi",
        "}",
        "trap cleanup_current EXIT",
        "",
        f"{q(NORMAL_PYTHON)} {q(SELF)} --runtime-preflight {common}",
        f"if [[ ! -f {q(PARENT_GENERATION)} ]]; then",
        f"  parent_tmp={q(str(REPORT_ROOT / '.parent-action-generation.running.json'))}",
        "  if [[ -e \"$parent_tmp\" ]]; then echo '[i25-stage2] stale parent generation staging file' >&2; exit 1; fi",
        "  " + " ".join(
            [
                q(VLLM_PYTHON), q(paths["generation_tool"]),
                "--model", q(paths["base"]), "--adapter", q(paths["parent"]),
                "--gpu", q(gpu), "--dims", "action", "--stable_action_rows", "32",
                "--think_suffix", "keep", "--tag", "i25-parent",
                "--out", '"$parent_tmp"',
                f">{q(PARENT_GENERATION_LOG)} 2>&1",
            ]
        ),
        f"  {q(NORMAL_PYTHON)} {q(SELF)} --accept-parent-generation --parent-generation-input \"$parent_tmp\" {common}",
        "  parent_tmp=",
        "else",
        f"  {q(NORMAL_PYTHON)} {q(SELF)} --accept-parent-generation {common}",
        "fi",
        "",
    ]
    for scale in SCALES:
        status_command = (
            f"{q(NORMAL_PYTHON)} {q(SELF)} --scale-status "
            f"--scale {scale:g} {common}"
        )
        scale_commands = command_for_scale(
            scale, gpu, executor_sha, manifest_sha, gate, paths, stage1
        )
        lines.extend(guarded_scale_lines(scale, status_command, scale_commands))
    lines.extend(
        [
            f"{q(NORMAL_PYTHON)} {q(SELF)} --finalize-rejection {common}",
            "echo '[i25-stage2] all locked scales failed; I-25 rejected locally' >&2",
            "exit 4",
            "",
        ]
    )
    return "\n".join(lines)


def dry_run(
    gate: dict[str, Any], paths: dict[str, Path | str], stage1: dict[str, Any] | None
) -> None:
    result = {
        "status": "DRY_RUN_NO_FILES_WRITTEN_NO_GPU_STAGE2_NOT_OPENED",
        "gate_sha256": GATE_SHA256,
        "scale_order": list(SCALES),
        "stage1_ticket": (
            {
                "available": True,
                "path": str(STAGE1_TICKET),
                "sha256": stage1["sha256"],
                "frozen_step": stage1["step"],
                "residual_adapter_sha256": stage1["residual_adapter_sha256"],
            }
            if stage1 is not None
            else {
                "available": False,
                "path": str(STAGE1_TICKET),
                "prepare_would_refuse": True,
            }
        ),
        "execution_contract": {
            "compose_only_current_scale": True,
            "ascending_only": True,
            "stop_first_full_pass": True,
            "delete_failed_candidate_before_publishing_decision": True,
            "do_not_return_to_checkpoint_axis": True,
            "no_new_or_interpolated_scale": True,
            "future_scale_results_exposed": False,
            "merged_precheck_temp_scope": str(REPORT_ROOT),
            "global_tmp_cleanup": False,
            "packaging": False,
            "upload": False,
        },
        "locked_tools": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
            if name in {
                "combine", "d_tool", "e_user_tool", "e_rec_tool",
                "d_material_tool", "generation_tool", "generation_checker", "precheck",
            }
        },
        "prepare_manifest": str(PREPARE_MANIFEST),
        "prepare_transaction_marker": str(PREPARE_PENDING),
        "gpu_launcher": str(COMMAND_LAUNCHER),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def prepare_transaction_record(
    stage1: dict[str, Any],
    executor_sha: str,
    gpu: str,
    manifest_sha: str,
    launcher_sha: str,
) -> dict[str, Any]:
    creator_pid = os.getpid()
    return {
        "status": "STAGE2_PREPARE_TRANSACTION_PENDING",
        "gate_sha256": GATE_SHA256,
        "stage1_ticket_sha256": stage1["sha256"],
        "executor_sha256": executor_sha,
        "creator_pid": creator_pid,
        "gpu": gpu,
        "outputs": {
            "manifest": {
                "path": str(PREPARE_MANIFEST),
                "sha256": manifest_sha,
            },
            "launcher": {
                "path": str(COMMAND_LAUNCHER),
                "sha256": launcher_sha,
            },
        },
        "staging_files": [
            str(
                PREPARE_MANIFEST.parent
                / f".{PREPARE_MANIFEST.name}.tmp-{creator_pid}"
            ),
            str(
                COMMAND_LAUNCHER.parent
                / f".{COMMAND_LAUNCHER.name}.tmp-{creator_pid}"
            ),
        ],
        "report_root": str(REPORT_ROOT),
        "candidate_root": str(CANDIDATE_ROOT),
        "launcher_refuses_while_pending": True,
        "packaging_authorized": False,
        "upload_authorized": False,
    }


def recover_interrupted_prepare() -> dict[str, Any]:
    if PREPARE_PENDING.is_symlink() or not PREPARE_PENDING.is_file():
        raise RuntimeError(
            f"prepare transaction marker is not a regular file: {PREPARE_PENDING}"
        )
    transaction = load_json(PREPARE_PENDING)
    if set(transaction) != {
        "status",
        "gate_sha256",
        "stage1_ticket_sha256",
        "executor_sha256",
        "creator_pid",
        "gpu",
        "outputs",
        "staging_files",
        "report_root",
        "candidate_root",
        "launcher_refuses_while_pending",
        "packaging_authorized",
        "upload_authorized",
    }:
        raise RuntimeError("prepare transaction marker schema drifted")
    if (
        transaction.get("status") != "STAGE2_PREPARE_TRANSACTION_PENDING"
        or transaction.get("gate_sha256") != GATE_SHA256
        or transaction.get("report_root") != str(REPORT_ROOT)
        or transaction.get("candidate_root") != str(CANDIDATE_ROOT)
        or transaction.get("launcher_refuses_while_pending") is not True
        or transaction.get("packaging_authorized") is not False
        or transaction.get("upload_authorized") is not False
    ):
        raise RuntimeError("prepare transaction marker contract drifted")
    require_lower_sha(
        transaction.get("stage1_ticket_sha256"),
        "prepare transaction Stage-1 ticket SHA256",
    )
    require_lower_sha(
        transaction.get("executor_sha256"),
        "prepare transaction executor SHA256",
    )
    creator_pid = transaction.get("creator_pid")
    if not isinstance(creator_pid, int) or isinstance(creator_pid, bool) or creator_pid <= 0:
        raise RuntimeError("prepare transaction creator PID drifted")
    gpu = transaction.get("gpu")
    if not isinstance(gpu, str) or not gpu or "," in gpu or any(
        character.isspace() for character in gpu
    ):
        raise RuntimeError("prepare transaction GPU identity drifted")
    outputs = transaction.get("outputs")
    expected_outputs = {
        "manifest": PREPARE_MANIFEST,
        "launcher": COMMAND_LAUNCHER,
    }
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        raise RuntimeError("prepare transaction output set drifted")
    expected_staging_files = [
        PREPARE_MANIFEST.parent / f".{PREPARE_MANIFEST.name}.tmp-{creator_pid}",
        COMMAND_LAUNCHER.parent / f".{COMMAND_LAUNCHER.name}.tmp-{creator_pid}",
    ]
    if transaction.get("staging_files") != [
        str(path) for path in expected_staging_files
    ]:
        raise RuntimeError("prepare transaction staging-file paths drifted")

    existing_outputs: list[tuple[Path, str]] = []
    for name, path in expected_outputs.items():
        record = outputs[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"prepare transaction {name} record drifted")
        expected_sha = require_lower_sha(
            record.get("sha256"), f"prepare transaction {name} SHA256"
        )
        if record.get("path") != str(path):
            raise RuntimeError(f"prepare transaction {name} path drifted")
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"refusing to recover unexpected prepare {name}: {path}"
                )
            require_hash(path, expected_sha, f"interrupted prepare {name}")
            existing_outputs.append((path, name))

    if CANDIDATE_ROOT.exists() or CANDIDATE_ROOT.is_symlink():
        raise RuntimeError(
            "interrupted CPU preparation unexpectedly materialized Stage-2 candidates"
        )
    report_root_present = REPORT_ROOT.exists() or REPORT_ROOT.is_symlink()
    if report_root_present:
        if REPORT_ROOT.is_symlink() or not REPORT_ROOT.is_dir():
            raise RuntimeError("interrupted prepare report root is not a directory")
        if any(REPORT_ROOT.iterdir()):
            raise RuntimeError(
                "interrupted prepare report root is nonempty; refusing automatic rollback"
            )

    existing_staging_files: list[Path] = []
    for path in expected_staging_files:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"refusing to remove unexpected prepare staging file: {path}"
                )
            existing_staging_files.append(path)

    for path, name in existing_outputs:
        remove_file_strict(path, f"interrupted prepare {name}")
    for path in existing_staging_files:
        remove_file_strict(path, "interrupted prepare staging file")
    if report_root_present:
        REPORT_ROOT.rmdir()
    remove_file_strict(PREPARE_PENDING, "prepare transaction marker")
    return {
        "status": "RECOVERED_VERIFIED_INTERRUPTED_STAGE2_PREPARE",
        "removed_outputs": [name for _, name in existing_outputs],
        "removed_staging_files": [str(path) for path in existing_staging_files],
        "removed_empty_report_root": report_root_present,
    }


def prepare(gpu: str) -> None:
    executor_sha = sha256(SELF)
    gate, paths = static_contract()
    stage1 = load_stage1_ticket(gate, paths, required=True)
    assert stage1 is not None
    recovered: dict[str, Any] | None = None
    if PREPARE_PENDING.exists() or PREPARE_PENDING.is_symlink():
        recovered = recover_interrupted_prepare()
    sequence_state(stage1)
    for path, label in (
        (PREPARE_MANIFEST, "Stage-2 prepare manifest"),
        (COMMAND_LAUNCHER, "Stage-2 GPU launcher"),
        (REPORT_ROOT, "Stage-2 report root"),
        (CANDIDATE_ROOT, "Stage-2 candidate root"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {label}: {path}")
    manifest = manifest_record(paths, stage1, executor_sha, gpu)
    manifest_text = json_document(manifest)
    manifest_sha = text_sha256(manifest_text)
    launcher = command_launcher_text(
        gpu, executor_sha, manifest_sha, gate, paths, stage1
    )
    launcher_sha = text_sha256(launcher)
    transaction = prepare_transaction_record(
        stage1, executor_sha, gpu, manifest_sha, launcher_sha
    )
    try:
        atomic_json(PREPARE_PENDING, transaction)
        REPORT_ROOT.mkdir(parents=True)
        atomic_json(PREPARE_MANIFEST, manifest)
        atomic_text(COMMAND_LAUNCHER, launcher, mode=0o755)
        require_hash(PREPARE_MANIFEST, manifest_sha, "published Stage-2 manifest")
        require_hash(COMMAND_LAUNCHER, launcher_sha, "published Stage-2 launcher")
        remove_file_strict(PREPARE_PENDING, "completed prepare transaction marker")
    except BaseException as error:
        cleanup_errors: list[str] = []
        marker_temporary = (
            PREPARE_PENDING.parent / f".{PREPARE_PENDING.name}.tmp-{os.getpid()}"
        )
        if marker_temporary.exists() or marker_temporary.is_symlink():
            try:
                remove_file_strict(
                    marker_temporary, "failed-prepare transaction staging file"
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if PREPARE_PENDING.exists() or PREPARE_PENDING.is_symlink():
            try:
                recover_interrupted_prepare()
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if cleanup_errors:
            raise RuntimeError(
                "Stage-2 prepare failed and rollback was incomplete: "
                + "; ".join(cleanup_errors)
            ) from error
        raise
    output = {
        **manifest,
        "prepare_manifest_sha256": manifest_sha,
        "gpu_launcher_sha256": launcher_sha,
        "invocation": str(COMMAND_LAUNCHER),
        "recovered_interrupted_prepare": recovered,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def self_test() -> None:
    assert list(SCALE_TAG) == list(SCALES)
    assert [SCALE_TAG[value] for value in SCALES] == [
        "s0250", "s0375", "s0500", "s0625", "s0750"
    ]
    sample = "\n".join(
        [
            "A_采样复读崩溃率_仅诊断: 3/30 = 10.0%",
            "B_itemic结构断裂率: 0/60 = 0.0%",
            "C_选择题格式存活率_仅诊断: 7/8 = 87.5%",
            "C_附带_占位符复读: 0/8 = 0.0%",
        ]
    )
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "precheck.log"
        path.write_text(sample, encoding="utf-8")
        parsed = parse_precheck(path)
    assert parsed["action_repeat_collapse"] == 3
    assert parsed["itemic_breakage"] == 0
    assert parsed["choice_format_survival"] == 7
    assert parsed["choice_placeholder"] == 0
    assert finite(0.0, "zero") == 0.0
    try:
        finite(float("nan"), "nan")
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite guard failed")
    distribution = lambda mean, median=0.0: {
        "n": 128,
        "mean": mean,
        "median": median,
        "p95": mean,
        "max": mean,
    }
    d_metrics = {
        "action": {
            "ce_delta": distribution(-0.005, 0.0),
            "kl": distribution(0.02),
            "top1_agreement": distribution(0.99),
        },
        "topic": {
            "ce_delta": distribution(0.005, 0.0),
            "kl": distribution(0.005),
            "top1_agreement": distribution(0.99),
        },
    }
    for task in (
        "material_desc2sid",
        "material_sid2desc",
        "rec_video",
        "rec_prod",
        "rec_ad",
        "rec_living",
    ):
        d_metrics[task] = {
            "kl": distribution(0.005),
            "top1_agreement": distribution(0.99),
        }
    e_action = {
        "gold_sum_logp_delta_mean": 0.0,
        "gold_sum_logp_improved_rate": 0.55,
        "top1_agreement_delta_mean": 0.0,
        "parent_to_candidate_kl_mean": 0.02,
    }
    e_topic = {
        "gold_sum_logp_delta_mean": -0.01,
        "parent_to_candidate_kl_mean": 0.005,
        "top1_agreement_delta_mean": 0.0,
    }
    material = {
        task: {
            "parent_to_candidate_kl_mean": 0.005,
            "top1_agreement_mean": 0.99,
        }
        for task in ("material_desc2sid", "material_sid2desc")
    }
    e_rec = {
        domain: {
            "gold_sum_logp_delta_mean": -0.03,
            "parent_to_candidate_kl_mean": 0.005,
            "all_rank_le_64_delta": 0.0,
        }
        for domain in ("video", "prod", "ad", "live")
    }
    composition = {
        "integrity_checks": {
            "parent_adapter_sha256_matches_locked_I23": True,
            "residual_adapter_sha256_matches_frozen_Stage1": True,
            "identity_formula_matches_locked_scale": True,
        },
        "identity_audit": {
            "rank": 80,
            "alpha": 80,
            "tensor_count": 392,
            "target_module_sets_match": True,
            "tensor_key_sets_match": True,
            "tensor_by_tensor_additive_identity_max_abs": 1e-6,
        }
    }
    boundary_checks = threshold_checks(
        d_metrics,
        e_action,
        e_topic,
        material,
        e_rec,
        {"f1": 0.0, "json_ok": 0.0, "trunc_rate": 0.0, "max_repeat_p95": 0.0},
        {"protocol": True},
        composition,
        {
            "itemic_breakage": 0,
            "action_repeat_collapse": 3,
            "choice_placeholder": 0,
            "choice_format_survival": 7,
        },
        0,
    )
    assert len(boundary_checks) == 59
    assert all(boundary_checks.values())
    generation_regression = threshold_checks(
        d_metrics,
        e_action,
        e_topic,
        material,
        e_rec,
        {"f1": 0.0, "json_ok": 0.0, "trunc_rate": 0.0, "max_repeat_p95": 1.0},
        {"protocol": True},
        composition,
        {
            "itemic_breakage": 0,
            "action_repeat_collapse": 3,
            "choice_placeholder": 0,
            "choice_format_survival": 7,
        },
        0,
    )
    assert generation_regression["generation_max_repeat_p95_delta_le_0"] is False
    assert sum(not value for value in generation_regression.values()) == 1
    print("[i25-stage2] static self-test passed; no repository files written; no GPU used")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "I-25 Stage-2 locked sequential gate. Default is a read-only dry run; "
            "the emitted launcher is the only intended GPU entry point."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--runtime-preflight", action="store_true")
    modes.add_argument("--accept-parent-generation", action="store_true")
    modes.add_argument("--scale-status", action="store_true")
    modes.add_argument("--compose-scale", action="store_true")
    modes.add_argument("--write-merge-config", action="store_true")
    modes.add_argument("--judge-scale", action="store_true")
    modes.add_argument("--finalize-rejection", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--scale", type=float, choices=SCALES)
    parser.add_argument("--composition-out", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--merged-dir", type=Path)
    parser.add_argument("--merge-config-out", type=Path)
    parser.add_argument("--parent-generation-input", type=Path)
    parser.add_argument("--expected-executor-sha256")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    internal = any(
        (
            args.runtime_preflight,
            args.accept_parent_generation,
            args.scale_status,
            args.compose_scale,
            args.write_merge_config,
            args.judge_scale,
            args.finalize_rejection,
        )
    )
    if internal:
        gate, paths, stage1, executor_sha = internal_common(
            args.expected_executor_sha256,
            args.expected_manifest_sha256,
        )
        if args.runtime_preflight:
            recovery = reconcile_pending_commit(stage1)
            state = recovery.pop("state")
            print(
                json.dumps(
                    {
                        "status": "RUNTIME_PREFLIGHT_PASS",
                        "stage2_opened": bool(state["decisions"]),
                        "terminal": (
                            state["passing_scale"] is not None
                            or state["all_failed"]
                            or REJECTED_STAGE2.exists()
                        ),
                        **recovery,
                    }
                )
            )
            return
        if args.accept_parent_generation:
            accept_or_validate_parent_generation(
                gate, paths, args.parent_generation_input
            )
            return
        if args.scale_status:
            if args.scale is None:
                parser.error("--scale-status requires --scale")
            state = sequence_state(stage1)
            if state["passing_scale"] is not None:
                print(json.dumps({"status": "ALREADY_SELECTED", "scale": state["passing_scale"]}))
                raise SystemExit(EXIT_ALREADY_SELECTED)
            if REJECTED_STAGE2.exists():
                print(json.dumps({"status": "ALREADY_REJECTED"}))
                raise SystemExit(EXIT_ALREADY_REJECTED)
            index = SCALES.index(args.scale)
            if len(state["decisions"]) > index:
                record = state["decisions"][index]
                if record["scale"] != args.scale or record["pass"]:
                    raise RuntimeError("published scale prefix is inconsistent")
                print(json.dumps({"status": "ALREADY_FAILED", "scale": args.scale}))
                raise SystemExit(EXIT_ALREADY_FAILED)
            assert_scale_ready(stage1, args.scale, allow_candidate=False)
            print(json.dumps({"status": "READY", "scale": args.scale}))
            return
        if args.compose_scale:
            if args.scale is None or args.composition_out is None:
                parser.error("--compose-scale requires --scale and --composition-out")
            compose_scale(
                gate, paths, stage1, args.scale, args.composition_out, executor_sha
            )
            return
        if args.write_merge_config:
            if any(
                value is None
                for value in (
                    args.scale,
                    args.composition_out,
                    args.merged_dir,
                    args.merge_config_out,
                )
            ):
                parser.error(
                    "--write-merge-config requires --scale, --composition-out, "
                    "--merged-dir, and --merge-config-out"
                )
            write_merge_config(
                gate,
                paths,
                stage1,
                args.scale,
                args.composition_out,
                args.merged_dir,
                args.merge_config_out,
                executor_sha,
            )
            return
        if args.judge_scale:
            if args.scale is None or args.raw_dir is None or args.merged_dir is None:
                parser.error("--judge-scale requires --scale, --raw-dir, and --merged-dir")
            passed = judge_scale(
                gate,
                paths,
                stage1,
                args.scale,
                args.raw_dir,
                args.merged_dir,
                executor_sha,
            )
            raise SystemExit(0 if passed else EXIT_SCALE_FAILED)
        if args.finalize_rejection:
            finalize_rejection(stage1, executor_sha)
            return
        raise AssertionError("unhandled internal mode")

    if not args.gpu or "," in args.gpu or any(character.isspace() for character in args.gpu):
        parser.error("--gpu must name exactly one GPU id or UUID")
    if args.prepare:
        prepare(args.gpu)
        return
    gate, paths = static_contract()
    stage1 = load_stage1_ticket(gate, paths, required=False)
    dry_run(gate, paths, stage1)


if __name__ == "__main__":
    main()
