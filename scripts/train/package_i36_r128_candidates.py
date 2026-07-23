#!/usr/bin/env python3
"""Build and verify the two pre-registered I-36 r128 submission packages."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"
RUN = ROOT / "checkpoints/i36_i35_user_expand_retkl_r16_v1"
LOG_DIR = ROOT / "logs/package"
MAX_UPLOAD_BYTES = 400_000_000
EXPECTED_PARENT_ADAPTER = "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
EXPECTED_PARENT_CONFIG = "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"
EXPECTED_TENSORS = 392
EXPECTED_TARGETS = {
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
}
CANDIDATES = {
    "step2063": {
        "residual": RUN / "checkpoint-2063",
        "output": ROOT / "submissions/i36_i35_user_expand_retkl_r128_step2063_platform",
        "audit": LOG_DIR / "i36_i35_user_expand_retkl_r128_step2063.json",
        "global_step": 2063,
    },
    "step4125": {
        "residual": RUN,
        "output": ROOT / "submissions/i36_i35_user_expand_retkl_r128_step4125_platform",
        "audit": LOG_DIR / "i36_i35_user_expand_retkl_r128_step4125.json",
        "global_step": 4125,
    },
}


def _load_combiner() -> Any:
    path = Path(__file__).with_name("combine_lora_adapters.py")
    spec = importlib.util.spec_from_file_location("llmrec_i36_combiner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMBINER = _load_combiner()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def assert_adapter_config(path: Path, *, rank: int, alpha: int, label: str) -> dict[str, Any]:
    config = load_json(path / "adapter_config.json")
    checks = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "r": rank,
        "lora_alpha": alpha,
        "use_dora": False,
        "use_rslora": False,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise RuntimeError(f"{label} config drift for {key}: {config.get(key)!r}/{expected!r}")
    if config.get("modules_to_save") not in (None, []):
        raise RuntimeError(f"{label} contains modules_to_save")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise RuntimeError(f"{label} contains rank/alpha patterns")
    if set(config.get("target_modules") or []) != EXPECTED_TARGETS:
        raise RuntimeError(f"{label} target modules drifted")
    return config


def assert_parent() -> dict[str, torch.Tensor]:
    weights = PARENT / "adapter_model.safetensors"
    config = PARENT / "adapter_config.json"
    if sha256(weights) != EXPECTED_PARENT_ADAPTER or sha256(config) != EXPECTED_PARENT_CONFIG:
        raise RuntimeError("I-36 parent package is missing or hash-drifted")
    assert_adapter_config(PARENT, rank=112, alpha=112, label="I-36 parent")
    tensors = load_file(weights, device="cpu")
    if len(tensors) != EXPECTED_TENSORS:
        raise RuntimeError(f"I-36 parent tensor count drifted: {len(tensors)}")
    return tensors


def assert_residual(path: Path, global_step: int) -> dict[str, torch.Tensor]:
    weights = path / "adapter_model.safetensors"
    state_path = path / "trainer_state.json"
    if not weights.is_file() or not state_path.is_file():
        raise RuntimeError(f"I-36 step{global_step} residual is incomplete: {path}")
    assert_adapter_config(path, rank=16, alpha=16, label=f"I-36 step{global_step} residual")
    state = load_json(state_path)
    if int(state.get("global_step", -1)) != global_step:
        raise RuntimeError(
            f"I-36 residual trainer step drifted: {state.get('global_step')!r}/{global_step}"
        )
    tensors = load_file(weights, device="cpu")
    if len(tensors) != EXPECTED_TENSORS:
        raise RuntimeError(f"I-36 residual tensor count drifted: {len(tensors)}")
    return tensors


def verify_exact_additivity(
    parent: dict[str, torch.Tensor],
    residual: dict[str, torch.Tensor],
    output: Path,
) -> dict[str, Any]:
    files = sorted(path.name for path in output.iterdir() if path.is_file())
    if files != ["adapter_config.json", "adapter_model.safetensors"]:
        raise RuntimeError(f"I-36 package is not strict two-file: {files}")
    assert_adapter_config(output, rank=128, alpha=128, label="I-36 combined")
    combined = load_file(output / "adapter_model.safetensors", device="cpu")
    if set(combined) != set(parent) or set(combined) != set(residual):
        raise RuntimeError("I-36 combined tensor keys drifted")
    for key in sorted(combined):
        if key.endswith("lora_A.weight"):
            split_dim = 0
        elif key.endswith("lora_B.weight"):
            split_dim = 1
        else:
            raise RuntimeError(f"I-36 unexpected adapter tensor: {key}")
        parent_rank = parent[key].shape[split_dim]
        parent_slice, residual_slice = torch.split(
            combined[key], [parent_rank, residual[key].shape[split_dim]], dim=split_dim
        )
        if not torch.equal(parent_slice, parent[key]) or not torch.equal(residual_slice, residual[key]):
            raise RuntimeError(f"I-36 exact additivity failed: {key}")
    model_path = output / "adapter_model.safetensors"
    config_path = output / "adapter_config.json"
    total_bytes = model_path.stat().st_size + config_path.stat().st_size
    if total_bytes >= MAX_UPLOAD_BYTES:
        raise RuntimeError(f"I-36 package exceeds platform upload limit: {total_bytes}")
    return {
        "strict_files": files,
        "tensor_count": len(combined),
        "exact_additivity": "PASS_ALL_TENSORS",
        "total_bytes": total_bytes,
        "upload_limit_bytes": MAX_UPLOAD_BYTES,
        "adapter_sha256": sha256(model_path),
        "config_sha256": sha256(config_path),
    }


def package_candidate(name: str, parent: dict[str, torch.Tensor]) -> dict[str, Any]:
    candidate = CANDIDATES[name]
    residual_path = candidate["residual"]
    output = candidate["output"]
    audit_path = candidate["audit"]
    if output.exists() or audit_path.exists():
        raise RuntimeError(f"I-36 refuses to overwrite existing {name} package or audit")
    residual = assert_residual(residual_path, candidate["global_step"])
    combine_audit = COMBINER.combine(PARENT, residual_path, output)
    verification = verify_exact_additivity(parent, residual, output)
    result = {
        "schema_version": "i36-r128-package-v1",
        "candidate": name,
        "global_step": candidate["global_step"],
        "status": "PASS",
        "combine": combine_audit,
        "verification": verification,
        "residual_config_sha256": sha256(residual_path / "adapter_config.json"),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(audit_path)
    print(
        f"[i36-package] {name} PASS: r128 tensors={verification['tensor_count']} "
        f"bytes={verification['total_bytes']} adapter={verification['adapter_sha256']}",
        flush=True,
    )
    return result


def run_self_test() -> None:
    COMBINER.run_self_test()
    if set(CANDIDATES) != {"step2063", "step4125"}:
        raise AssertionError("I-36 candidate set drifted")
    if 112 + 16 != 128 or MAX_UPLOAD_BYTES != 400_000_000:
        raise AssertionError("I-36 package constants drifted")
    print("[i36-package] self-test PASS: fixed two-candidate r128 contract", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=("all", "step2063", "step4125"),
        default="all",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    parent = assert_parent()
    names = tuple(CANDIDATES) if args.candidate == "all" else (args.candidate,)
    for name in names:
        package_candidate(name, parent)


if __name__ == "__main__":
    main()
