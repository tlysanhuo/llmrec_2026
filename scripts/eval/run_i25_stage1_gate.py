#!/usr/bin/env python3
"""Prepare I-25 Stage-1 scale-1 candidates without running a GPU audit.

The preregistered Stage-1 axis is checkpoint-only.  This helper therefore:

* accepts only the locked 250,500,750,1000,1250,1527 order;
* combines I-23 r64 with each fresh r16 residual at multiplier 1.0;
* verifies every saved LoRA factor tensor and the resulting r80/alpha80 spec;
* emits a manifest and a locked ascending GPU command launcher; and
* projects shared-tool output to action-only fields, stopping at the first
  Stage-1 pass without exposing Stage-2 scales.

With no flag (or with ``--dry-run``), it performs static checks and writes
nothing.  ``--prepare`` remains CPU-only and refuses to run until formal
training has completed successfully through step 1527.  The generated launcher
is the only GPU entry point; the helper's internal projection mode is CPU-only.
"""

from __future__ import annotations

import argparse
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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GATE_REL = Path("configs/evaluation/i23_actionres_r16_checkpoint_gate.json")
GATE_SHA256 = "53b5b375630ba2255dada2fd04d8fc4cd1b694e3afde10079ef492135dcef212"
ORDER = (250, 500, 750, 1000, 1250, 1527)
PARENT_REL = Path("submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform")
DEFAULT_OUTPUT_ROOT = ROOT / "checkpoints/i25_stage1_scale1_temporary"
DEFAULT_MANIFEST = ROOT / "logs/model/i25_stage1_scale1_prepare_manifest.json"
DEFAULT_COMMANDS = ROOT / "logs/model/i25_stage1_scale1_gpu_commands.sh"
DEFAULT_REPORT_ROOT = ROOT / "logs/probe/i25_stage1"
DEFAULT_PYTHON = Path(
    "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/"
    "llmrec_2026/LLaMA-Factory/.venv/bin/python3"
)
MODEL_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"


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


def rooted(relative: str | Path) -> Path:
    path = ROOT / Path(relative)
    return path.resolve(strict=False)


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 drifted: expected {expected}, got {actual}")
    return actual


def ensure_repo_path(path: Path, label: str) -> Path:
    # Keep the repository's registered compatibility links (checkpoints/logs)
    # lexical.  Path.resolve() follows them into ai_runtime and would reject a
    # legitimate registered destination even though the user supplied a path
    # below this checkout.
    resolved = Path(os.path.abspath(os.fspath(path)))
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository: {resolved}") from error
    return resolved


def static_contract() -> tuple[dict[str, Any], dict[str, Path]]:
    gate_path = rooted(GATE_REL)
    require_hash(gate_path, GATE_SHA256, "I-25 preregistered gate")
    gate = load_json(gate_path)
    if gate.get("status") != "PREREGISTERED_BEFORE_I25_FORMAL_LAUNCH":
        raise RuntimeError(f"I-25 gate is inactive: {gate.get('status')!r}")

    axis = gate["checkpoint_axis"]
    stage1 = gate["stage_1_training_checkpoint_selection"]
    if tuple(axis.get("candidates_in_ascending_order", ())) != ORDER:
        raise RuntimeError("checkpoint-axis order drifted")
    if tuple(stage1.get("candidate_order", ())) != ORDER:
        raise RuntimeError("Stage-1 candidate order drifted")
    if float(stage1.get("evaluation_multiplier", -1)) != 1.0:
        raise RuntimeError("Stage-1 is no longer locked to residual multiplier 1.0")
    expected_paths = axis.get("expected_paths", {})
    expected_mapping = {
        str(step): f"checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-{step}"
        for step in ORDER
    }
    if expected_paths != expected_mapping:
        raise RuntimeError("Stage-1 checkpoint paths drifted")

    config_record = gate["formal_training_artifacts"]["config"]
    config_path = rooted(config_record["path"])
    require_hash(config_path, config_record["sha256"], "I-25 training config")
    trainer_record = gate["formal_training_artifacts"]["trainer"]
    trainer_path = rooted(trainer_record["path"])
    require_hash(trainer_path, trainer_record["sha256"], "I-25 trainer")
    for launcher_name in ("online_launcher", "detached_launcher"):
        launcher_path = rooted(gate["formal_training_artifacts"][launcher_name]["path"])
        if not launcher_path.is_file():
            raise FileNotFoundError(f"missing I-25 {launcher_name}: {launcher_path}")

    locked = gate["locked_audit_protocols"]
    combine_record = locked["composition"]
    combine_path = rooted(combine_record["tool"])
    require_hash(combine_path, combine_record["tool_sha256"], "LoRA composition tool")

    d_record = locked["d_residual_mechanism_and_retention"]
    d_tool = rooted(d_record["tool"])
    require_hash(d_tool, d_record["tool_sha256"], "Stage-1 D audit tool")
    dataset = rooted(d_record["dataset"])
    retention_source = rooted(d_record["retention_source"])
    retention_exclude = rooted(d_record["exact_row_exclude"])
    require_hash(dataset, d_record["dataset_sha256"], "Stage-1 D mechanism dataset")
    require_hash(
        retention_source,
        d_record["retention_source_sha256"],
        "Stage-1 held-out retention source",
    )
    require_hash(
        retention_exclude,
        d_record["exact_row_exclude_sha256"],
        "Stage-1 exact-row exclusion dataset",
    )

    e_record = locked["e_action_topic_path"]
    e_tool = rooted(e_record["tool"])
    require_hash(e_tool, e_record["tool_sha256"], "Stage-1 E action audit tool")
    dev = ROOT / "assets/evaluation/offline_eval"
    require_hash(
        dev / "dev_action.jsonl",
        e_record["action_source_sha256"],
        "Stage-1 E action source",
    )
    require_hash(
        dev / "dev_topic.jsonl",
        e_record["topic_source_sha256"],
        "Stage-1 E topic source",
    )

    base = rooted(locked["base"]["path"])
    require_hash(base / "config.json", locked["base"]["config_sha256"], "O6 config")

    parent = rooted(gate["parent"]["adapter"])
    if parent != rooted(PARENT_REL):
        raise RuntimeError(f"I-25 parent path drifted: {parent}")
    require_hash(
        parent / MODEL_NAME,
        gate["parent"]["adapter_sha256"],
        "I-23 parent adapter",
    )
    require_hash(
        parent / CONFIG_NAME,
        gate["parent"]["adapter_config_sha256"],
        "I-23 parent adapter config",
    )

    return gate, {
        "gate": gate_path,
        "config": config_path,
        "trainer": trainer_path,
        "combine": combine_path,
        "d_tool": d_tool,
        "e_tool": e_tool,
        "dataset": dataset,
        "retention_source": retention_source,
        "retention_exclude": retention_exclude,
        "dev": dev,
        "base": base,
        "parent": parent,
    }


def checkpoint_state(gate: dict[str, Any]) -> list[dict[str, Any]]:
    states = []
    expected = gate["checkpoint_axis"]["expected_paths"]
    for step in ORDER:
        path = rooted(expected[str(step)])
        model = path / MODEL_NAME
        config = path / CONFIG_NAME
        ready = model.is_file() and config.is_file()
        states.append(
            {
                "step": step,
                "path": str(path),
                "ready": ready,
                "missing": [
                    str(item)
                    for item in (model, config)
                    if not item.is_file()
                ],
                **(
                    {
                        "adapter_bytes": model.stat().st_size,
                        "adapter_sha256": sha256(model),
                        "config_sha256": sha256(config),
                    }
                    if ready
                    else {}
                ),
            }
        )
    return states


def verify_training_complete(gate: dict[str, Any], states: list[dict[str, Any]]) -> dict[str, Any]:
    if not all(record["ready"] for record in states):
        missing_steps = [record["step"] for record in states if not record["ready"]]
        raise RuntimeError(
            "formal training is incomplete; refusing Stage-1 preparation; "
            f"missing checkpoint steps: {missing_steps}"
        )

    output_root = rooted(gate["checkpoint_axis"]["output_root"])
    trainer_state_path = output_root / "trainer_state.json"
    trainer_state = load_json(trainer_state_path)
    if trainer_state.get("global_step") != 1527:
        raise RuntimeError(
            f"formal training did not finish step 1527: {trainer_state.get('global_step')}"
        )

    final_dir = rooted(gate["checkpoint_axis"]["expected_paths"]["1527"])
    for filename in (MODEL_NAME, CONFIG_NAME):
        root_file = output_root / filename
        final_file = final_dir / filename
        if not root_file.is_file() or root_file.read_bytes() != final_file.read_bytes():
            raise RuntimeError(f"checkpoint-1527 is not byte-identical to root {filename}")

    exit_code_path = ROOT / "logs/train/i23_actionres_r16_ansretkl_ep1.exit_code"
    if not exit_code_path.is_file() or exit_code_path.read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("detached I-25 exit code is missing or non-zero")

    log_path = ROOT / "logs/train/i23_actionres_r16_ansretkl_ep1.log"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    bad_patterns = {
        "traceback": r"traceback \(most recent call last\)",
        "oom": r"(?:cuda )?out of memory|outofmemoryerror|\boom\b",
        "nan_or_inf": r"(?<![\w.])(?:nan|inf)(?![\w.])",
        "parent_fingerprint_error": r"parent fingerprint error",
        "route_error": r"route error|routing mismatch",
    }
    found = [name for name, pattern in bad_patterns.items() if re.search(pattern, log_text, re.I)]
    if found:
        raise RuntimeError(f"formal training log contains forbidden signatures: {found}")
    if "route=action" not in log_text or "route=retention" not in log_text:
        raise RuntimeError("formal training log does not prove both action and retention routes")

    forbidden_names = {"optimizer.pt", "scheduler.pt", "scaler.pt"}
    forbidden = []
    for path in output_root.rglob("*"):
        if path.is_file() and (
            path.name in forbidden_names
            or (path.name.startswith("rng_state") and path.suffix == ".pth")
        ):
            forbidden.append(str(path))
    if forbidden:
        raise RuntimeError(f"I-25 retained forbidden training state: {forbidden[:5]}")

    return {
        "trainer_state": str(trainer_state_path),
        "global_step": trainer_state["global_step"],
        "exit_code_path": str(exit_code_path),
        "training_log": str(log_path),
        "training_log_sha256": sha256(log_path),
        "routes_observed": ["action", "retention"],
        "checkpoint_1527_root_byte_identical": True,
        "forbidden_training_state_count": 0,
    }


def validate_lora_config(path: Path, rank: int, alpha: int, label: str) -> dict[str, Any]:
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
    if config.get("fan_in_fan_out") is not False:
        raise RuntimeError(f"{label} must use fan_in_fan_out=false")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise RuntimeError(f"{label} uses rank/alpha patterns")
    return config


def tensor_identity_audit(parent_dir: Path, residual_dir: Path, combined_dir: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    parent_config = validate_lora_config(parent_dir / CONFIG_NAME, 64, 64, "I-23 parent")
    residual_config = validate_lora_config(
        residual_dir / CONFIG_NAME, 16, 16, "I-25 residual"
    )
    combined_config = validate_lora_config(
        combined_dir / CONFIG_NAME, 80, 80, "I-25 Stage-1 composition"
    )
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

    per_tensor = []
    maximum = 0.0
    for key in sorted(parent):
        p_tensor = parent[key]
        r_tensor = residual[key]
        c_tensor = combined[key]
        if key.endswith("lora_A.weight"):
            expected_shape = (p_tensor.shape[0] + r_tensor.shape[0], *p_tensor.shape[1:])
            if tuple(c_tensor.shape) != tuple(expected_shape):
                raise RuntimeError(f"combined A shape mismatch for {key}: {c_tensor.shape}")
            pieces = (c_tensor[: p_tensor.shape[0]], c_tensor[p_tensor.shape[0] :])
        elif key.endswith("lora_B.weight"):
            expected_shape = (*p_tensor.shape[:-1], p_tensor.shape[-1] + r_tensor.shape[-1])
            if tuple(c_tensor.shape) != tuple(expected_shape):
                raise RuntimeError(f"combined B shape mismatch for {key}: {c_tensor.shape}")
            pieces = (c_tensor[..., : p_tensor.shape[-1]], c_tensor[..., p_tensor.shape[-1] :])
        else:
            raise RuntimeError(f"unexpected adapter tensor key: {key}")
        parent_max = float((pieces[0].float() - p_tensor.float()).abs().max())
        residual_max = float((pieces[1].float() - r_tensor.float()).abs().max())
        tensor_max = max(parent_max, residual_max)
        if not math.isfinite(tensor_max):
            raise RuntimeError(f"non-finite identity error for {key}")
        maximum = max(maximum, tensor_max)
        per_tensor.append(
            {
                "key": key,
                "parent_factor_max_abs": parent_max,
                "residual_factor_max_abs": residual_max,
                "max_abs": tensor_max,
            }
        )

    if maximum > 1e-6:
        raise RuntimeError(f"tensor-by-tensor additive identity failed: max_abs={maximum}")
    del parent, residual, combined
    return {
        "identity": "delta_combined = delta_I23 + 1.0 * delta_I25_residual",
        "proof": (
            "every combined A tensor is the exact row concatenation and every "
            "combined B tensor is the exact column concatenation; all three "
            "adapters have alpha/r=1, so factor-block identity implies additive identity"
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


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def commands_for_step(
    step: int,
    combined_dir: Path,
    report_root: Path,
    python_bin: Path,
    gpu: str,
    paths: dict[str, Path],
    d_audit_output: str | Path | None = None,
    e_audit_output: str | Path | None = None,
) -> dict[str, str]:
    d_output = report_root / f"step-{step}-d-action.json"
    e_output = report_root / f"step-{step}-e-action.json"
    d_audit_output = d_output if d_audit_output is None else d_audit_output
    e_audit_output = e_output if e_audit_output is None else e_audit_output
    d_command = " ".join(
        [
            q(python_bin),
            q(paths["d_tool"]),
            "--base", q(paths["base"]),
            "--parent", q(paths["parent"]),
            "--residual", q(rooted(f"checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-{step}")),
            "--dataset", q(paths["dataset"]),
            "--retention-source", q(paths["retention_source"]),
            "--retention-exclude", q(paths["retention_exclude"]),
            "--retention-tasks", q("material_desc2sid,material_sid2desc,rec_video,rec_prod,rec_ad,rec_living"),
            "--retention-per-task", "96",
            "--output", q(d_audit_output),
            "--gpu", q(gpu),
            "--action-n", "128",
            "--topic-n", "128",
            "--retention-n", "0",
            "--cutoff", "16384",
            "--seed", "19260821",
            "--scales", "1.0",
        ]
    )
    e_command = " ".join(
        [
            q(python_bin),
            q(paths["e_tool"]),
            "--base", q(paths["base"]),
            "--parent", q(paths["parent"]),
            "--dev", q(paths["dev"]),
            "--candidate", q(f"step-{step}={combined_dir}"),
            "--out", q(e_audit_output),
            "--gpu", q(gpu),
            "--per-task", "32",
            "--cutoff", "16384",
        ]
    )
    return {
        "d_action_mechanism": d_command,
        "e_action_gold_path": e_command,
        "d_report": str(d_output),
        "e_report": str(e_output),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to reuse JSON staging path: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def project_action_reports(
    gate: dict[str, Any],
    step: int,
    d_raw_path: Path,
    e_raw_path: Path,
    d_action_path: Path,
    e_action_path: Path,
    decision_path: Path,
    frozen_path: Path,
    expected_residual_sha256: str,
    expected_combined_sha256: str,
) -> bool:
    """Project shared-tool reports to Stage-1 action fields and judge one step."""

    if step not in ORDER:
        raise ValueError(f"invalid Stage-1 step: {step}")
    d_raw = load_json(d_raw_path)
    e_raw = load_json(e_raw_path)
    scale_metrics = d_raw.get("metrics_by_scale", {}).get("1", {})
    d_action = scale_metrics.get("action")
    if not isinstance(d_action, dict):
        raise RuntimeError("D report is missing scale=1 action metrics")
    candidate_name = f"step-{step}"
    e_models = e_raw.get("models", {})
    if set(e_models) != {"parent", candidate_name}:
        raise RuntimeError(f"E report model set drifted: {sorted(e_models)}")
    e_parent_action = e_models["parent"].get("absolute", {}).get("action")
    e_candidate_absolute = e_models[candidate_name].get("absolute", {}).get("action")
    e_candidate_delta = (
        e_models[candidate_name].get("delta_vs_parent", {}).get("action")
    )
    if not all(
        isinstance(value, dict)
        for value in (e_parent_action, e_candidate_absolute, e_candidate_delta)
    ):
        raise RuntimeError("E report is missing action-only model fields")

    parent_sha256 = gate["parent"]["adapter_sha256"]
    d_artifacts = d_raw.get("artifacts", {})
    e_source = e_raw.get("source", {})
    checks = {
        "step_is_on_locked_axis": step in ORDER,
        "d_scale_is_exactly_1": d_raw.get("scales") == [1.0],
        "d_action_requested_is_128": d_raw.get("requested", {}).get("action") == 128,
        "d_skipped_over_cutoff_is_0": d_raw.get("skipped_too_long") == 0,
        "d_parent_hash_matches": d_artifacts.get("parent_adapter_sha256") == parent_sha256,
        "d_residual_hash_matches": (
            d_artifacts.get("residual_adapter_sha256") == expected_residual_sha256
        ),
        "d_action_ce_n_is_128": d_action.get("ce_delta", {}).get("n") == 128,
        "d_action_ce_delta_mean_le_neg_0_01": (
            d_action.get("ce_delta", {}).get("mean", math.inf) <= -0.01
        ),
        "d_action_ce_delta_median_le_0": (
            d_action.get("ce_delta", {}).get("median", math.inf) <= 0.0
        ),
        "d_action_kl_n_is_128": d_action.get("kl", {}).get("n") == 128,
        "d_action_kl_mean_le_0_05": d_action.get("kl", {}).get("mean", math.inf) <= 0.05,
        "e_per_task_is_32": e_raw.get("method", {}).get("per_task_requested") == 32,
        "e_action_source_hash_matches": (
            e_source.get("file_sha256", {}).get("action")
            == gate["locked_audit_protocols"]["e_action_topic_path"]["action_source_sha256"]
        ),
        "e_parent_hash_matches": (
            e_models["parent"].get("adapter_sha256") == parent_sha256
        ),
        "e_combined_hash_matches": (
            e_models[candidate_name].get("adapter_sha256") == expected_combined_sha256
        ),
        "e_action_n_is_32": e_candidate_delta.get("n") == 32,
        "e_action_gold_sum_logp_delta_mean_ge_0": (
            e_candidate_delta.get("gold_sum_logp_delta_mean", -math.inf) >= 0.0
        ),
        "e_action_gold_sum_logp_improved_rate_ge_0_55": (
            e_candidate_delta.get("gold_sum_logp_improved_rate", -math.inf) >= 0.55
        ),
        "e_action_top1_agreement_delta_mean_ge_0": (
            e_candidate_delta.get("top1_agreement_delta_mean", -math.inf) >= 0.0
        ),
        "e_action_parent_to_candidate_kl_mean_le_0_02": (
            e_candidate_delta.get("parent_to_candidate_kl_mean", math.inf) <= 0.02
        ),
    }
    passed = all(checks.values())

    d_projection = {
        "status": "STAGE1_ACTION_ONLY_PROJECTION_NOT_A_SCORE_ESTIMATE",
        "step": step,
        "gate_sha256": GATE_SHA256,
        "source_class": "D",
        "scale": 1.0,
        "requested_action_rows": d_raw.get("requested", {}).get("action"),
        "skipped_over_cutoff": d_raw.get("skipped_too_long"),
        "artifacts": {
            key: d_artifacts.get(key)
            for key in (
                "base",
                "parent",
                "parent_adapter_sha256",
                "residual",
                "residual_adapter_sha256",
                "dataset",
                "dataset_sha256",
            )
        },
        "action": d_action,
        "masked_fields": ["topic", "material", "recommendation"],
    }
    e_projection = {
        "status": "STAGE1_ACTION_ONLY_PROJECTION_NOT_A_SCORE_ESTIMATE",
        "step": step,
        "gate_sha256": GATE_SHA256,
        "source_class": "E",
        "method": {
            "route": e_raw.get("method", {}).get("route"),
            "action_rows_requested": e_raw.get("method", {}).get("per_task_requested"),
        },
        "source": {
            "action_file_sha256": e_source.get("file_sha256", {}).get("action")
        },
        "models": {
            "parent": {
                "adapter_sha256": e_models["parent"].get("adapter_sha256"),
                "action_absolute": e_parent_action,
            },
            candidate_name: {
                "adapter_sha256": e_models[candidate_name].get("adapter_sha256"),
                "action_absolute": e_candidate_absolute,
                "action_delta_vs_parent": e_candidate_delta,
            },
        },
        "masked_fields": ["topic"],
    }
    output_paths = (d_action_path, e_action_path, decision_path)
    for path in output_paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite Stage-1 action result: {path}")
    if passed and (frozen_path.exists() or frozen_path.is_symlink()):
        raise FileExistsError(f"Stage-1 checkpoint is already frozen: {frozen_path}")
    atomic_json(d_action_path, d_projection)
    atomic_json(e_action_path, e_projection)
    decision = {
        "status": (
            "PASS_FROZEN_STAGE1_CHECKPOINT"
            if passed
            else "FAIL_CONTINUE_ASCENDING_CHECKPOINT_AXIS"
        ),
        "step": step,
        "gate_sha256": GATE_SHA256,
        "selection_axis": "checkpoint only at residual multiplier 1.0",
        "selection_fields": "action only",
        "d_action_report": {
            "path": str(d_action_path),
            "sha256": sha256(d_action_path),
        },
        "e_action_report": {
            "path": str(e_action_path),
            "sha256": sha256(e_action_path),
        },
        "checks": checks,
        "pass": passed,
        "stage2_opened": False,
    }
    atomic_json(decision_path, decision)
    if passed:
        frozen = dict(decision)
        frozen["frozen_checkpoint"] = {
            "step": step,
            "residual_adapter_sha256": expected_residual_sha256,
            "scale_1_combined_adapter_sha256": expected_combined_sha256,
        }
        frozen["decision_report_sha256"] = sha256(decision_path)
        atomic_json(frozen_path, frozen)
    print(
        json.dumps(
            {"step": step, "pass": passed, "status": decision["status"]},
            ensure_ascii=False,
        )
    )
    return passed


def command_launcher(
    candidates: list[dict[str, Any]],
    output_root: Path,
    report_root: Path,
    python_bin: Path,
    gpu: str,
    paths: dict[str, Path],
) -> str:
    by_step = {int(candidate["step"]): candidate for candidate in candidates}
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# No arguments: the checkpoint axis and stop-first-pass rule are immutable.",
        "# Shared audit tools emit incidental protection fields only inside a private",
        "# ephemeral directory. The projector publishes and judges action fields only.",
        "raw_dir=",
        "cleanup_raw() {",
        "  if [[ -n \"${raw_dir:-}\" && -d \"$raw_dir\" ]]; then",
        "    rm -f -- \"$raw_dir/d.json\" \"$raw_dir/e.json\" \"$raw_dir/d.stdout\" \"$raw_dir/e.stdout\"",
        "    rmdir -- \"$raw_dir\" 2>/dev/null || true",
        "  fi",
        "}",
        "trap cleanup_raw EXIT",
        "",
    ]
    for step in ORDER:
        candidate = by_step[step]
        raw_commands = commands_for_step(
            step,
            output_root / f"step-{step}-scale-1",
            report_root,
            python_bin,
            gpu,
            paths,
            d_audit_output="__D_RAW__",
            e_audit_output="__E_RAW__",
        )
        d_command = raw_commands["d_action_mechanism"].replace(
            "__D_RAW__", '"$raw_dir/d.json"'
        )
        e_command = raw_commands["e_action_gold_path"].replace(
            "__E_RAW__", '"$raw_dir/e.json"'
        )
        decision_path = report_root / f"step-{step}-action-decision.json"
        frozen_path = report_root / "frozen-stage1-action.json"
        project_command = " ".join(
            [
                q(python_bin),
                q(Path(__file__).resolve()),
                "--project-action-reports",
                "--step", str(step),
                "--d-raw", '"$raw_dir/d.json"',
                "--e-raw", '"$raw_dir/e.json"',
                "--d-action-out", q(Path(raw_commands["d_report"])),
                "--e-action-out", q(Path(raw_commands["e_report"])),
                "--decision-out", q(decision_path),
                "--frozen-out", q(frozen_path),
                "--expected-residual-sha256",
                q(candidate["residual"]["adapter_sha256"]),
                "--expected-combined-sha256",
                q(candidate["temporary_combined"]["adapter_sha256"]),
            ]
        )
        lines.extend(
            [
                f"echo '[i25-stage1] evaluating locked step {step}'",
                f"raw_dir=$(mktemp -d \"${{TMPDIR:-/tmp}}/i25-stage1-{step}.XXXXXX\")",
                f"if ! {d_command} >\"$raw_dir/d.stdout\" 2>&1; then",
                f"  echo '[i25-stage1] D action audit failed at step {step}; raw report removed' >&2",
                "  exit 1",
                "fi",
                f"if ! {e_command} >\"$raw_dir/e.stdout\" 2>&1; then",
                f"  echo '[i25-stage1] E action audit failed at step {step}; raw report removed' >&2",
                "  exit 1",
                "fi",
                "set +e",
                project_command,
                "decision_rc=$?",
                "set -e",
                "cleanup_raw",
                "raw_dir=",
                "if [[ $decision_rc -eq 0 ]]; then",
                f"  echo '[i25-stage1] froze first passing checkpoint: step {step}'",
                "  exit 0",
                "fi",
                "if [[ $decision_rc -ne 3 ]]; then",
                f"  echo '[i25-stage1] action projector failed at step {step}' >&2",
                "  exit \"$decision_rc\"",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            "echo '[i25-stage1] no checkpoint passed the locked action gate' >&2",
            "exit 4",
            "",
        ]
    )
    return "\n".join(lines)


def estimate_bytes(parent: Path, candidate_count: int) -> int:
    # All adapters share tensor keys/dtype. r80 has 80/64 of the r64 payload;
    # the small overestimate also covers safetensors headers and JSON manifests.
    return math.ceil((parent / MODEL_NAME).stat().st_size * 80 / 64) * candidate_count + 2**20


def dry_run(
    gate: dict[str, Any],
    paths: dict[str, Path],
    output_root: Path,
    report_root: Path,
    python_bin: Path,
    gpu: str,
) -> None:
    states = checkpoint_state(gate)
    result = {
        "status": "DRY_RUN_NO_FILES_WRITTEN_NO_GPU",
        "gate_sha256": GATE_SHA256,
        "candidate_order": list(ORDER),
        "residual_multiplier": 1.0,
        "training_complete": all(record["ready"] for record in states),
        "checkpoints": states,
        "temporary_output_root": str(output_root),
        "estimated_temporary_bytes": estimate_bytes(paths["parent"], len(ORDER)),
        "estimated_temporary_gib": round(
            estimate_bytes(paths["parent"], len(ORDER)) / 2**30, 4
        ),
        "stage1_selection_inputs": "action fields only; incidental topic/retention fields masked",
        "stage2_scales_exposed": False,
        "gpu_launcher_plan": {
            "gpu": gpu,
            "python_bin": str(python_bin),
            "order_is_mechanically_fixed": list(ORDER),
            "stop_at_first_action_pass": True,
            "shared_tool_raw_reports": "private mktemp; deleted before operator-visible output",
            "published_reports": "action-only D/E projections and per-step action decision",
            "launcher_not_emitted_until_prepare": True,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def prepare(
    gate: dict[str, Any],
    paths: dict[str, Path],
    output_root: Path,
    manifest_path: Path,
    commands_path: Path,
    report_root: Path,
    python_bin: Path,
    gpu: str,
) -> None:
    # The composition tools request CPU tensors explicitly. Hiding every GPU
    # as a second guard makes accidental allocator use impossible in this
    # preparation phase, including inside the locked subprocess.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    states = checkpoint_state(gate)
    completion = verify_training_complete(gate, states)
    for path, label in (
        (output_root, "temporary output root"),
        (manifest_path, "manifest"),
        (commands_path, "GPU command launcher"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {label}: {path}")

    required = estimate_bytes(paths["parent"], len(ORDER))
    free = shutil.disk_usage(output_root.parent).free
    if free < required:
        raise RuntimeError(f"insufficient disk for temporary r80 candidates: {free} < {required}")

    staging = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to reuse staging directory: {staging}")
    staging.mkdir(parents=True)
    candidates = []
    try:
        for record in states:
            step = int(record["step"])
            residual_dir = Path(record["path"])
            staged_candidate = staging / f"step-{step}-scale-1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(paths["combine"]),
                    str(paths["parent"]),
                    str(residual_dir),
                    str(staged_candidate),
                    "--residual-scale",
                    "1.0",
                ],
                check=True,
                text=True,
                capture_output=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            combine_stdout = json.loads(completed.stdout)
            identity = tensor_identity_audit(paths["parent"], residual_dir, staged_candidate)
            final_candidate = output_root / staged_candidate.name
            candidates.append(
                {
                    "step": step,
                    "residual": {
                        "path": str(residual_dir),
                        "adapter_sha256": sha256(residual_dir / MODEL_NAME),
                        "config_sha256": sha256(residual_dir / CONFIG_NAME),
                    },
                    "temporary_combined": {
                        "path": str(final_candidate),
                        "adapter_bytes": (staged_candidate / MODEL_NAME).stat().st_size,
                        "adapter_sha256": sha256(staged_candidate / MODEL_NAME),
                        "config_sha256": sha256(staged_candidate / CONFIG_NAME),
                        "combine_tool_result": {
                            "rank": combine_stdout["combined"]["rank"],
                            "alpha": combine_stdout["combined"]["alpha"],
                            "tensor_count": combine_stdout["combined"]["tensor_count"],
                            "identity": combine_stdout["identity"],
                        },
                    },
                    "composition_identity": identity,
                }
            )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    manifest = {
        "status": "CPU_PREPARATION_COMPLETE_GPU_NOT_RUN_NO_SELECTION",
        "gate": {"path": str(paths["gate"]), "sha256": GATE_SHA256},
        "contract": {
            "candidate_order": list(ORDER),
            "residual_multiplier": 1.0,
            "selection_inputs": "action only",
            "incidental_fields_masked": ["topic", "material", "recommendation", "generation", "structure"],
            "stage2_scales_exposed": False,
            "checkpoint_selected": None,
        },
        "training_completion": completion,
        "locked_inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "config": paths["config"],
                "trainer": paths["trainer"],
                "combine_tool": paths["combine"],
                "d_action_tool": paths["d_tool"],
                "e_action_tool": paths["e_tool"],
                "base_config": paths["base"] / "config.json",
                "parent_adapter": paths["parent"] / MODEL_NAME,
                "parent_config": paths["parent"] / CONFIG_NAME,
                "d_dataset": paths["dataset"],
                "retention_source": paths["retention_source"],
                "retention_exclude": paths["retention_exclude"],
                "e_action_source": paths["dev"] / "dev_action.jsonl",
                "e_topic_source": paths["dev"] / "dev_topic.jsonl",
            }.items()
        },
        "temporary_bytes": sum(
            candidate["temporary_combined"]["adapter_bytes"] for candidate in candidates
        ),
        "candidates": candidates,
        "gpu_commands_not_run": {
            "launcher": str(commands_path),
            "invocation": q(commands_path),
            "arguments": [],
            "fixed_order": list(ORDER),
            "stop_first_action_pass": True,
            "raw_incidental_reports_are_ephemeral_and_deleted": True,
        },
        "cleanup_after_stage1_is_resolved_by_pass_or_full_rejection": {
            "delete": str(output_root),
            "keep": [str(manifest_path), str(commands_path), str(report_root)],
            "never_delete": "the six original r16 training checkpoints or their audit reports",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    commands_path.write_text(
        command_launcher(
            candidates,
            output_root,
            report_root,
            python_bin,
            gpu,
            paths,
        ),
        encoding="utf-8",
    )
    commands_path.chmod(0o755)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only I-25 Stage-1 scale-1 preparation and action-only report "
            "projection. This helper never runs a GPU audit or opens Stage 2."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate locked inputs and print the plan without writing files (default)",
    )
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="after successful step 1527, create CPU-composed temporary r80 candidates",
    )
    mode.add_argument(
        "--project-action-reports",
        action="store_true",
        help=(
            "internal CPU-only mode used by the emitted locked-order launcher; "
            "publish/judge action fields and mask incidental shared-tool fields"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--commands-out", type=Path, default=DEFAULT_COMMANDS)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python executable recorded in, but not used to run, the emitted GPU commands",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="single GPU id recorded in emitted commands; no GPU is run by this helper",
    )
    parser.add_argument("--step", type=int, choices=ORDER)
    parser.add_argument("--d-raw", type=Path)
    parser.add_argument("--e-raw", type=Path)
    parser.add_argument("--d-action-out", type=Path)
    parser.add_argument("--e-action-out", type=Path)
    parser.add_argument("--decision-out", type=Path)
    parser.add_argument("--frozen-out", type=Path)
    parser.add_argument("--expected-residual-sha256")
    parser.add_argument("--expected-combined-sha256")
    args = parser.parse_args()

    gate, paths = static_contract()
    if args.project_action_reports:
        required = {
            "--step": args.step,
            "--d-raw": args.d_raw,
            "--e-raw": args.e_raw,
            "--d-action-out": args.d_action_out,
            "--e-action-out": args.e_action_out,
            "--decision-out": args.decision_out,
            "--frozen-out": args.frozen_out,
            "--expected-residual-sha256": args.expected_residual_sha256,
            "--expected-combined-sha256": args.expected_combined_sha256,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if missing:
            parser.error(f"--project-action-reports is missing: {', '.join(missing)}")
        for label, value in (
            ("--expected-residual-sha256", args.expected_residual_sha256),
            ("--expected-combined-sha256", args.expected_combined_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                parser.error(f"{label} must be a lowercase SHA256")
        d_action_path = ensure_repo_path(args.d_action_out, "--d-action-out")
        e_action_path = ensure_repo_path(args.e_action_out, "--e-action-out")
        decision_path = ensure_repo_path(args.decision_out, "--decision-out")
        frozen_path = ensure_repo_path(args.frozen_out, "--frozen-out")
        passed = project_action_reports(
            gate,
            args.step,
            args.d_raw.resolve(strict=True),
            args.e_raw.resolve(strict=True),
            d_action_path,
            e_action_path,
            decision_path,
            frozen_path,
            args.expected_residual_sha256,
            args.expected_combined_sha256,
        )
        raise SystemExit(0 if passed else 3)

    if not args.gpu or "," in args.gpu:
        parser.error("--gpu must name exactly one GPU id or UUID")
    output_root = ensure_repo_path(args.output_root, "--output-root")
    manifest_path = ensure_repo_path(args.manifest, "--manifest")
    commands_path = ensure_repo_path(args.commands_out, "--commands-out")
    report_root = ensure_repo_path(args.report_root, "--report-root")
    # Preserve a virtualenv launcher path instead of resolving its interpreter
    # symlink; executing the resolved base binary would lose the venv package
    # context needed by the GPU audit tools.
    python_bin = Path(os.path.abspath(os.fspath(args.python_bin)))
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise FileNotFoundError(f"GPU-command Python executable is unavailable: {python_bin}")

    if args.prepare:
        prepare(
            gate,
            paths,
            output_root,
            manifest_path,
            commands_path,
            report_root,
            python_bin,
            args.gpu,
        )
    else:
        dry_run(gate, paths, output_root, report_root, python_bin, args.gpu)


if __name__ == "__main__":
    main()
