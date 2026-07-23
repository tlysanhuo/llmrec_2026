#!/usr/bin/env python3
"""Cast a strict two-file LoRA adapter to a smaller storage dtype."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


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
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
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
    destination_dtype = DTYPES[args.dtype]
    tensors: dict[str, torch.Tensor] = {}
    source_dtypes: set[str] = set()
    max_abs_error = 0.0
    with safe_open(source_model, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {"format": "pt"}
        for key in keys:
            if ".lora_A." not in key and ".lora_B." not in key:
                raise ValueError(f"non-LoRA tensor found: {key}")
            tensor = handle.get_tensor(key)
            if not tensor.is_floating_point():
                raise ValueError(f"non-floating tensor found: {key} ({tensor.dtype})")
            source_dtypes.add(str(tensor.dtype))
            cast = tensor.to(destination_dtype).contiguous()
            max_abs_error = max(
                max_abs_error,
                float((tensor.float() - cast.float()).abs().max()),
            )
            tensors[key] = cast

    temporary = output.with_name(output.name + ".tmp")
    temporary.mkdir(parents=True)
    try:
        save_file(tensors, temporary / "adapter_model.safetensors", metadata=metadata)
        shutil.copy2(source_config, temporary / "adapter_config.json")
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
    output_config = output / "adapter_config.json"
    report = {
        "status": "COMPLETE_STRICT_TWO_FILE_LORA_DTYPE_CAST",
        "source": str(source),
        "source_adapter_bytes": source_model.stat().st_size,
        "source_adapter_sha256": sha256(source_model),
        "source_dtypes": sorted(source_dtypes),
        "output": str(output),
        "output_adapter_bytes": output_model.stat().st_size,
        "output_adapter_sha256": sha256(output_model),
        "output_config_bytes": output_config.stat().st_size,
        "output_config_sha256": sha256(output_config),
        "output_dtype": str(destination_dtype),
        "tensor_count": len(keys),
        "rank": int(config["r"]),
        "alpha": int(config["lora_alpha"]),
        "max_abs_cast_error": max_abs_error,
        "package_bytes": output_model.stat().st_size + output_config.stat().st_size,
        "under_400000000_bytes": (
            output_model.stat().st_size + output_config.stat().st_size < 400_000_000
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = args.audit.with_suffix(args.audit.suffix + ".tmp")
    audit_tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    audit_tmp.replace(args.audit)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
