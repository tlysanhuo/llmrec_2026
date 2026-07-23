#!/usr/bin/env python3
"""Compress a uniform-rank LoRA adapter with per-module truncated SVD."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    if value.get("peft_type") != "LORA" or value.get("task_type") != "CAUSAL_LM":
        raise ValueError("source is not a causal-LM LoRA adapter")
    if value.get("use_dora") or value.get("use_rslora"):
        raise ValueError("DoRA and rsLoRA are unsupported")
    if value.get("modules_to_save") not in (None, []):
        raise ValueError("source contains modules_to_save")
    if value.get("rank_pattern") or value.get("alpha_pattern"):
        raise ValueError("non-uniform source rank is unsupported")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--preserve-prefix-rank", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    source_model = source / "adapter_model.safetensors"
    source_config = source / "adapter_config.json"
    if not source_model.is_file() or not source_config.is_file():
        raise FileNotFoundError("source must contain adapter_model.safetensors and adapter_config.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if args.audit.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.audit}")

    config = load_config(source_config)
    source_rank = int(config["r"])
    if not 1 <= args.rank < source_rank:
        raise ValueError(f"target rank must be in [1, {source_rank - 1}]")
    if not 0 <= args.preserve_prefix_rank < args.rank:
        raise ValueError("preserved prefix rank must be in [0, target rank - 1]")
    compressed_rank = args.rank - args.preserve_prefix_rank
    source_scale = float(config["lora_alpha"]) / source_rank
    target_alpha = args.rank
    target_scale = target_alpha / args.rank
    destination_dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    output_tensors: dict[str, torch.Tensor] = {}
    retained_energy: list[float] = []
    actual_relative_errors: list[float] = []
    total_source_energy = 0.0
    total_residual_energy = 0.0
    total_spectral_energy = 0.0
    total_spectral_kept = 0.0
    with safe_open(source_model, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {"format": "pt"}
        a_keys = sorted(key for key in keys if key.endswith("lora_A.weight"))
        b_keys = sorted(key for key in keys if key.endswith("lora_B.weight"))
        if len(a_keys) * 2 != len(keys) or len(a_keys) != len(b_keys):
            raise ValueError("source must contain paired LoRA A/B tensors only")

        for index, a_key in enumerate(a_keys, start=1):
            b_key = a_key.removesuffix("lora_A.weight") + "lora_B.weight"
            if b_key not in keys:
                raise ValueError(f"missing paired tensor: {b_key}")
            a = handle.get_tensor(a_key).float().to(device)
            b = handle.get_tensor(b_key).float().to(device)
            if a.shape[0] != source_rank or b.shape[1] != source_rank:
                raise ValueError(f"rank mismatch for {a_key}: A={tuple(a.shape)} B={tuple(b.shape)}")

            prefix = args.preserve_prefix_rank
            a_prefix, a_tail = a[:prefix], a[prefix:]
            b_prefix, b_tail = b[:, :prefix], b[:, prefix:]
            qb, rb = torch.linalg.qr(b_tail, mode="reduced")
            qa, ra = torch.linalg.qr(a_tail.T, mode="reduced")
            core = (source_scale / target_scale) * (rb @ ra.T)
            u, singular, vh = torch.linalg.svd(core, full_matrices=False)
            singular_kept = singular[:compressed_rank]
            root = singular_kept.sqrt()
            b_tail_new = ((qb @ u[:, :compressed_rank]) * root.unsqueeze(0)).contiguous()
            a_tail_new = (root.unsqueeze(1) * (vh[:compressed_rank] @ qa.T)).contiguous()
            b_new = torch.cat(
                (b_prefix * (source_scale / target_scale), b_tail_new), dim=1
            ).contiguous()
            a_new = torch.cat((a_prefix, a_tail_new), dim=0).contiguous()

            source_delta = source_scale * (b @ a)
            b_saved = b_new.to(destination_dtype)
            a_saved = a_new.to(destination_dtype)
            reconstructed = target_scale * (b_saved.float() @ a_saved.float())
            source_energy = float(source_delta.square().sum())
            residual_energy = float((source_delta - reconstructed).square().sum())
            spectral_energy = float(singular.square().sum())
            spectral_kept = float(singular_kept.square().sum())
            retained_energy.append(spectral_kept / spectral_energy if spectral_energy else 1.0)
            actual_relative_errors.append(
                (residual_energy / source_energy) ** 0.5 if source_energy else 0.0
            )
            total_source_energy += source_energy
            total_residual_energy += residual_energy
            total_spectral_energy += spectral_energy
            total_spectral_kept += spectral_kept
            output_tensors[a_key] = a_saved.cpu()
            output_tensors[b_key] = b_saved.cpu()
            del a, b, qa, qb, ra, rb, core, u, singular, vh
            del a_prefix, a_tail, b_prefix, b_tail, a_tail_new, b_tail_new
            del source_delta, reconstructed, a_new, b_new, a_saved, b_saved
            if index % 28 == 0 or index == len(a_keys):
                print(f"[compress-lora] {index}/{len(a_keys)}", flush=True)

    output_config = dict(config)
    output_config.update(
        {
            "r": args.rank,
            "lora_alpha": target_alpha,
            "inference_mode": True,
            "rank_pattern": {},
            "alpha_pattern": {},
        }
    )
    temporary = output.with_name(output.name + ".tmp")
    temporary.mkdir(parents=True)
    try:
        save_file(output_tensors, temporary / "adapter_model.safetensors", metadata=metadata)
        (temporary / "adapter_config.json").write_text(
            json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_files = sorted(path.name for path in temporary.iterdir() if path.is_file())
        if output_files != ["adapter_config.json", "adapter_model.safetensors"]:
            raise RuntimeError(f"unexpected output file set: {output_files}")
        with safe_open(
            temporary / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as handle:
            output_keys = list(handle.keys())
            output_dtypes = sorted({str(handle.get_tensor(key).dtype) for key in output_keys})
        if output_keys != keys or output_dtypes != [str(destination_dtype)]:
            raise RuntimeError("output tensor identity or dtype validation failed")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    output_model = output / "adapter_model.safetensors"
    output_config_path = output / "adapter_config.json"
    report = {
        "status": "COMPLETE_PER_MODULE_TRUNCATED_SVD_LORA_COMPRESSION",
        "source": str(source),
        "source_rank": source_rank,
        "source_alpha": int(config["lora_alpha"]),
        "source_adapter_sha256": sha256(source_model),
        "output": str(output),
        "output_rank": args.rank,
        "output_alpha": target_alpha,
        "preserved_prefix_rank": args.preserve_prefix_rank,
        "compressed_tail_source_rank": source_rank - args.preserve_prefix_rank,
        "compressed_tail_output_rank": compressed_rank,
        "output_dtype": str(destination_dtype),
        "output_adapter_bytes": output_model.stat().st_size,
        "output_adapter_sha256": sha256(output_model),
        "output_config_bytes": output_config_path.stat().st_size,
        "output_config_sha256": sha256(output_config_path),
        "tensor_count": len(output_tensors),
        "matrix_count": len(retained_energy),
        "retained_spectral_energy_min": min(retained_energy),
        "retained_spectral_energy_mean": statistics.fmean(retained_energy),
        "retained_spectral_energy_global": total_spectral_kept / total_spectral_energy,
        "actual_relative_frobenius_error_max": max(actual_relative_errors),
        "actual_relative_frobenius_error_global": (
            total_residual_energy / total_source_energy
        )
        ** 0.5,
        "package_bytes": output_model.stat().st_size + output_config_path.stat().st_size,
        "under_400000000_bytes": (
            output_model.stat().st_size + output_config_path.stat().st_size < 400_000_000
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = args.audit.with_suffix(args.audit.suffix + ".tmp")
    audit_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    audit_tmp.replace(args.audit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
