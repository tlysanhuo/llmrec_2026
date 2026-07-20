#!/usr/bin/env python3
"""Exactly concatenate two additive LoRA adapters into one higher-rank adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_config(config: dict[str, Any], label: str) -> None:
    if config.get("peft_type") != "LORA":
        raise ValueError(f"{label} is not a LoRA adapter")
    for field in ("use_dora", "use_rslora"):
        if config.get(field):
            raise ValueError(f"{label} uses unsupported {field}")
    if config.get("bias") != "none" or config.get("modules_to_save") not in (None, []):
        raise ValueError(f"{label} contains non-LoRA trainable state")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{label} uses per-module rank/alpha patterns")


def combine(
    parent_dir: Path,
    residual_dir: Path,
    output_dir: Path,
    residual_multiplier: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(residual_multiplier) or residual_multiplier < 0:
        raise ValueError("residual multiplier must be finite and non-negative")
    parent_config = load_config(parent_dir / "adapter_config.json")
    residual_config = load_config(residual_dir / "adapter_config.json")
    validate_config(parent_config, "parent")
    validate_config(residual_config, "residual")

    parent_targets = set(parent_config["target_modules"])
    residual_targets = set(residual_config["target_modules"])
    if parent_targets != residual_targets:
        raise ValueError("parent/residual target modules differ")

    parent_rank = int(parent_config["r"])
    residual_rank = int(residual_config["r"])
    combined_rank = parent_rank + residual_rank
    parent_scale = float(parent_config["lora_alpha"]) / parent_rank
    residual_scale = float(residual_config["lora_alpha"]) / residual_rank
    combined_alpha = combined_rank
    combined_scale = combined_alpha / combined_rank

    parent_path = parent_dir / "adapter_model.safetensors"
    residual_path = residual_dir / "adapter_model.safetensors"
    parent = load_file(parent_path, device="cpu")
    residual = load_file(residual_path, device="cpu")
    if set(parent) != set(residual):
        missing_parent = sorted(set(residual) - set(parent))[:5]
        missing_residual = sorted(set(parent) - set(residual))[:5]
        raise ValueError(
            "adapter tensor keys differ: "
            f"only_residual={missing_parent}, only_parent={missing_residual}"
        )

    combined: dict[str, torch.Tensor] = {}
    for key in sorted(parent):
        parent_tensor = parent[key]
        residual_tensor = residual[key]
        if key.endswith("lora_A.weight"):
            if parent_tensor.shape[1:] != residual_tensor.shape[1:]:
                raise ValueError(f"A shape mismatch for {key}")
            combined[key] = torch.cat((parent_tensor, residual_tensor), dim=0).contiguous()
        elif key.endswith("lora_B.weight"):
            if parent_tensor.shape[:-1] != residual_tensor.shape[:-1]:
                raise ValueError(f"B shape mismatch for {key}")
            combined[key] = torch.cat(
                (
                    parent_tensor * (parent_scale / combined_scale),
                    residual_tensor
                    * (residual_scale * residual_multiplier / combined_scale),
                ),
                dim=1,
            ).contiguous()
        else:
            raise ValueError(f"unexpected adapter tensor: {key}")

    output_config = dict(parent_config)
    output_config.update(
        {
            "r": combined_rank,
            "lora_alpha": combined_alpha,
            "inference_mode": True,
            "target_modules": sorted(parent_targets),
        }
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    model_tmp = output_dir / "adapter_model.safetensors.tmp"
    save_file(combined, model_tmp, metadata={"format": "pt"})
    model_tmp.replace(output_dir / "adapter_model.safetensors")
    config_tmp = output_dir / "adapter_config.json.tmp"
    config_tmp.write_text(
        json.dumps(output_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_tmp.replace(output_dir / "adapter_config.json")

    return {
        "parent": {
            "path": str(parent_dir.resolve()),
            "rank": parent_rank,
            "alpha": parent_config["lora_alpha"],
            "adapter_sha256": sha256(parent_path),
        },
        "residual": {
            "path": str(residual_dir.resolve()),
            "rank": residual_rank,
            "alpha": residual_config["lora_alpha"],
            "multiplier": residual_multiplier,
            "adapter_sha256": sha256(residual_path),
        },
        "combined": {
            "path": str(output_dir.resolve()),
            "rank": combined_rank,
            "alpha": combined_alpha,
            "tensor_count": len(combined),
            "adapter_sha256": sha256(output_dir / "adapter_model.safetensors"),
            "config_sha256": sha256(output_dir / "adapter_config.json"),
        },
        "identity": (
            "delta_combined = delta_parent + "
            f"{residual_multiplier:g} * delta_residual"
        ),
    }


def run_self_test() -> None:
    torch.manual_seed(23)
    in_features, out_features = 7, 5
    rank_a, rank_b = 3, 2
    alpha_a, alpha_b = 6, 1
    residual_multiplier = 0.625
    a_a = torch.randn(rank_a, in_features)
    b_a = torch.randn(out_features, rank_a)
    a_b = torch.randn(rank_b, in_features)
    b_b = torch.randn(out_features, rank_b)
    a = torch.cat((a_a, a_b), dim=0)
    b = torch.cat(
        (
            b_a * (alpha_a / rank_a),
            b_b * (alpha_b / rank_b) * residual_multiplier,
        ),
        dim=1,
    )
    expected = (alpha_a / rank_a) * (b_a @ a_a) + residual_multiplier * (
        alpha_b / rank_b
    ) * (b_b @ a_b)
    actual = b @ a  # combined alpha/r is one
    assert torch.allclose(expected, actual, atol=1e-6)
    print("[combine-lora] self-test passed: concatenated adapter is exactly additive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path, nargs="?")
    parser.add_argument("residual", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not all((args.parent, args.residual, args.output)):
        parser.error("parent, residual, and output are required")
    result = combine(
        args.parent,
        args.residual,
        args.output,
        residual_multiplier=args.residual_scale,
    )
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
