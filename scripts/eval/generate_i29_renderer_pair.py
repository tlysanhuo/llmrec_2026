#!/usr/bin/env python3
"""Generate the three missing cells of the I29 renderer calibration.

The fixed comparison is a 2 x 2 crossing of the I23 and s800 adapters with
the legacy ``instruction -> system`` renderer and the canonical
``instruction + input -> user`` renderer.  The existing s800 x legacy cell is
hash-locked and reused; this program generates only:

* s800 x canonical-user
* I23 x legacy-system
* I23 x canonical-user

The program deliberately has no label-ledger argument and never opens one.
It consumes only the prompt manifest, model artifacts, and the label-free old
rollout used as the fourth cell.  All decoding parameters are constants copied
from the frozen I27 rollout config.  vLLM is imported only in ``--run`` mode.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE = PROJECT_ROOT / "models/OneReason-0.8B-pretrain-competition"
DEFAULT_I23 = (
    PROJECT_ROOT / "submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform"
)
DEFAULT_S800 = PROJECT_ROOT / "submissions/e3_userres_r80_retkl_v3_s800_platform"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "assets/derived/processed/o1_rec_multigold_v1_prompt_manifest.jsonl"
)
DEFAULT_OLD_S800_LEGACY = (
    PROJECT_ROOT / "assets/derived/processed/o1_rec_multigold_v1_rollouts.jsonl"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "logs/probe"
LEGACY_GENERATOR_PATH = PROJECT_ROOT / "scripts/rft/generate_rec_rft_rollouts.py"

FIRST_N = 16
EXPECTED_VLLM_VERSION = "0.12.0"
EXPECTED_LEGACY_GENERATOR_SHA256 = (
    "668a7e09b1460bb57e80baa6bcbfc28af50e2d99352784294805b1a4c5fa8c0d"
)
EXPECTED_MANIFEST_SHA256 = (
    "c75e6a326dd02da07b671787a0bbc76cc391c0ec1254a7eaed0fa1cc250d0300"
)
EXPECTED_MANIFEST_FIRST16_SHA256 = (
    "b3d7300f57847ce1a5e83e8ca438e167df00d80f720034278d3ce0280c2f5a57"
)
EXPECTED_OLD_ROLLOUT_SHA256 = (
    "c3dfe9bc5a2dbfb3161a9aa0d241692b1ad0616e7f954e096e9dd5caf4198fac"
)
EXPECTED_OLD_ROLLOUT_FIRST16_SHA256 = (
    "53df39ccaaa0946e32bd6e16951feed25b7a6bb73a0e7eb5af94f4fd7e04884f"
)
EXPECTED_OLD_CONFIG_SHA256 = (
    "c72fce0cf5b2c7b54d7bf632a33bd97d6b2501a298a3c11b6d5df1bfb47b7513"
)
EXPECTED_BASE_ARTIFACT_SHA256 = (
    "431cc7546a1813ed21a184974a1ac739139b7bdc4643d04e521d066f6ad20652"
)
EXPECTED_I23_ARTIFACT_SHA256 = (
    "7c193b8db334fe23a2cc74774b8adbee15ce6ba0a260b3afd3fefbbe3cbbb4f1"
)
EXPECTED_S800_ARTIFACT_SHA256 = (
    "ed5366a6c38a3e4da3c90970d243bd1b0f86fe7aad3ea08074fd7f32c2633c51"
)

SEED = 19260829
REASONING_SAMPLES = 4
ITEM_CANDIDATES = 8
MAX_REASONING_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
BATCH_PROMPTS = 16
BEAM_BATCH_PROMPTS = 16
GPU_MEMORY_UTILIZATION = 0.25
MAX_MODEL_LEN = 4096
DTYPE = "bfloat16"

CELL_SPECS = (
    {
        "name": "s800_canonical",
        "adapter": "s800",
        "renderer": "canonical_user",
        "filename": "i29_renderer_s800_canonical_n16.jsonl",
    },
    {
        "name": "i23_legacy",
        "adapter": "i23",
        "renderer": "legacy_system",
        "filename": "i29_renderer_i23_legacy_n16.jsonl",
    },
    {
        "name": "i23_canonical",
        "adapter": "i23",
        "renderer": "canonical_user",
        "filename": "i29_renderer_i23_canonical_n16.jsonl",
    },
)

_LEGACY_MODULE: ModuleType | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def load_legacy_module() -> ModuleType:
    """Load the frozen I27 implementation without importing vLLM."""
    global _LEGACY_MODULE
    if _LEGACY_MODULE is not None:
        return _LEGACY_MODULE
    if not LEGACY_GENERATOR_PATH.is_file():
        raise FileNotFoundError(LEGACY_GENERATOR_PATH)
    require_hash(
        LEGACY_GENERATOR_PATH,
        EXPECTED_LEGACY_GENERATOR_SHA256,
        "frozen I27 generator",
    )
    spec = importlib.util.spec_from_file_location(
        "_i29_frozen_i27_generator", LEGACY_GENERATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {LEGACY_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LEGACY_MODULE = module
    return module


def build_canonical_user_prompt(record: dict[str, Any]) -> str:
    """Render LLaMA-Factory's Alpaca instruction/input semantics exactly."""
    query = record["input"]
    if record["instruction"]:
        query = f"{record['instruction']}\n{query}"
    return (
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@contextlib.contextmanager
def renderer_override(
    legacy: ModuleType, renderer: Callable[[dict[str, Any]], str]
) -> Iterator[None]:
    """Select a renderer for frozen ``generate_batch`` and always restore it."""
    original = legacy.build_reasoning_prompt
    legacy.build_reasoning_prompt = renderer
    try:
        yield
    finally:
        legacy.build_reasoning_prompt = original


def first_lines(path: Path, count: int) -> tuple[bytes, list[str]]:
    raw_parts: list[bytes] = []
    decoded: list[str] = []
    with path.open("rb") as handle:
        for index in range(count):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path} has fewer than {count} rows")
            if not raw.strip():
                raise ValueError(f"blank JSONL row at {path}:{index + 1}")
            raw_parts.append(raw)
            decoded.append(raw.decode("utf-8"))
    return b"".join(raw_parts), decoded


def fixed_decode_args() -> SimpleNamespace:
    """Namespace consumed by the frozen I27 ``generate_batch`` function."""
    return SimpleNamespace(
        reasoning_samples=REASONING_SAMPLES,
        item_candidates=ITEM_CANDIDATES,
        max_reasoning_tokens=MAX_REASONING_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        seed=SEED,
        batch_prompts=BATCH_PROMPTS,
        beam_batch_prompts=BEAM_BATCH_PROMPTS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
    )


def output_paths(out_dir: Path) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for spec in CELL_SPECS:
        rollout = out_dir / str(spec["filename"])
        result[str(spec["name"])] = {
            "rollout": rollout,
            "config": rollout.with_suffix(rollout.suffix + ".config.json"),
            "metadata": rollout.with_suffix(rollout.suffix + ".meta.json"),
        }
    return result


def verify_old_decode_lock(config: dict[str, Any]) -> None:
    expected_reasoning = {
        "global_seed": SEED,
        "max_tokens": MAX_REASONING_TOKENS,
        "samples": REASONING_SAMPLES,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "stop": ["</think>"],
        "per_prompt_seed": "(global_seed + manifest.rollout_seed) mod 2**31",
    }
    expected_item = {
        "beam_width": ITEM_CANDIDATES,
        "ignore_eos": False,
        "length_penalty": 1.0,
        "max_tokens": 3,
        "temperature": 0.0,
    }
    expected_runtime = {
        "batch_prompts": BATCH_PROMPTS,
        "beam_batch_prompts": BEAM_BATCH_PROMPTS,
        "dtype": DTYPE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_logprobs": 2 * ITEM_CANDIDATES,
        "max_model_len": MAX_MODEL_LEN,
        "single_gpu": True,
        "vllm_version": EXPECTED_VLLM_VERSION,
    }
    if config.get("reasoning_sampling") != expected_reasoning:
        raise ValueError("old rollout reasoning decode does not match the I29 lock")
    if config.get("item_beam") != expected_item:
        raise ValueError("old rollout item beam does not match the I29 lock")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("old rollout runtime is missing")
    runtime_without_device = {key: value for key, value in runtime.items() if key != "gpu"}
    if runtime_without_device != expected_runtime:
        raise ValueError("old rollout runtime does not match the I29 lock")
    if config.get("script", {}).get("sha256") != EXPECTED_LEGACY_GENERATOR_SHA256:
        raise ValueError("old rollout was not made by the frozen I27 generator")
    if config.get("manifest", {}).get("sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("old rollout manifest identity mismatch")
    if config.get("base", {}).get("artifact_sha256") != EXPECTED_BASE_ARTIFACT_SHA256:
        raise ValueError("old rollout base model identity mismatch")
    if config.get("adapter", {}).get("artifact_sha256") != EXPECTED_S800_ARTIFACT_SHA256:
        raise ValueError("old rollout s800 adapter identity mismatch")


def load_fixed_records(
    legacy: ModuleType, manifest_path: Path
) -> list[dict[str, Any]]:
    raw, lines = first_lines(manifest_path, FIRST_N)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_MANIFEST_FIRST16_SHA256:
        raise ValueError(
            "first-16 manifest SHA256 mismatch: "
            f"expected {EXPECTED_MANIFEST_FIRST16_SHA256}, got {actual}"
        )
    records: list[dict[str, Any]] = []
    for offset, line in enumerate(lines, 1):
        record = legacy.validate_manifest_record(
            legacy.strict_json_loads(line), f"{manifest_path}:{offset}"
        )
        if record["domain"] != "video":
            raise ValueError(f"fixed row {offset} is not in the video domain")
        records.append(record)
    if len({record["group_id"] for record in records}) != FIRST_N:
        raise ValueError("fixed first-16 manifest contains duplicate group IDs")
    return records


def verify_old_fourth_cell(
    legacy: ModuleType,
    old_rollout_path: Path,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    require_hash(old_rollout_path, EXPECTED_OLD_ROLLOUT_SHA256, "old s800 legacy rollout")
    config_path = old_rollout_path.with_suffix(old_rollout_path.suffix + ".config.json")
    require_hash(config_path, EXPECTED_OLD_CONFIG_SHA256, "old s800 legacy config")
    config = legacy.strict_json_loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("old rollout config must be an object")
    verify_old_decode_lock(config)

    raw, lines = first_lines(old_rollout_path, FIRST_N)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_OLD_ROLLOUT_FIRST16_SHA256:
        raise ValueError(
            "old rollout first-16 SHA256 mismatch: "
            f"expected {EXPECTED_OLD_ROLLOUT_FIRST16_SHA256}, got {actual}"
        )
    generator = {
        "config_sha256": legacy.canonical_sha256(config),
        "base_sha256": EXPECTED_BASE_ARTIFACT_SHA256,
        "adapter_sha256": EXPECTED_S800_ARTIFACT_SHA256,
        "seed": SEED,
    }
    for index, (line, record) in enumerate(zip(lines, records), 1):
        legacy.validate_rollout_row(
            legacy.strict_json_loads(line),
            record,
            generator,
            REASONING_SAMPLES,
            ITEM_CANDIDATES,
            f"{old_rollout_path}:{index}",
        )
    return {
        "path": str(old_rollout_path.resolve()),
        "sha256": EXPECTED_OLD_ROLLOUT_SHA256,
        "first16_sha256": EXPECTED_OLD_ROLLOUT_FIRST16_SHA256,
        "config_path": str(config_path.resolve()),
        "config_sha256": EXPECTED_OLD_CONFIG_SHA256,
        "reused_rows": FIRST_N,
        "renderer": "legacy_system",
        "adapter": "s800",
    }


def assert_no_output_collisions(paths: dict[str, dict[str, Path]]) -> None:
    collisions = [
        path
        for cell_paths in paths.values()
        for path in cell_paths.values()
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite I29 outputs: {collisions}")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    legacy = load_legacy_module()
    base_path = Path(args.base)
    i23_path = Path(args.i23_adapter)
    s800_path = Path(args.s800_adapter)
    manifest_path = Path(args.manifest)
    old_rollout_path = Path(args.legacy_s800_rollout)
    out_dir = Path(args.out_dir)

    for path, directory in (
        (base_path, True),
        (i23_path, True),
        (s800_path, True),
        (manifest_path, False),
        (old_rollout_path, False),
    ):
        legacy.verify_input(path, directory=directory)
    legacy.verify_volume_path(out_dir)
    require_hash(manifest_path, EXPECTED_MANIFEST_SHA256, "prompt manifest")

    records = load_fixed_records(legacy, manifest_path)
    reused_cell = verify_old_fourth_cell(legacy, old_rollout_path, records)

    base = legacy.runtime_artifact_fingerprint(base_path, adapter=False)
    i23 = legacy.runtime_artifact_fingerprint(i23_path, adapter=True)
    s800 = legacy.runtime_artifact_fingerprint(s800_path, adapter=True)
    expected_artifacts = (
        ("base", base, EXPECTED_BASE_ARTIFACT_SHA256),
        ("I23", i23, EXPECTED_I23_ARTIFACT_SHA256),
        ("s800", s800, EXPECTED_S800_ARTIFACT_SHA256),
    )
    for label, fingerprint, expected in expected_artifacts:
        if fingerprint["artifact_sha256"] != expected:
            raise ValueError(
                f"{label} runtime artifact mismatch: expected {expected}, "
                f"got {fingerprint['artifact_sha256']}"
            )
    i23_rank = legacy.adapter_rank(i23_path.resolve())
    s800_rank = legacy.adapter_rank(s800_path.resolve())
    if (i23_rank, s800_rank) != (64, 80):
        raise ValueError(f"unexpected adapter ranks: I23={i23_rank}, s800={s800_rank}")

    paths = output_paths(out_dir)
    assert_no_output_collisions(paths)
    compact_identity = [
        {
            "group_id": record["group_id"],
            "prompt_sha256": record["prompt_sha256"],
            "rollout_seed": record["rollout_seed"],
        }
        for record in records
    ]
    report = {
        "status": "ready",
        "mode": "cpu_preflight_only",
        "fixed_group_count": FIRST_N,
        "fixed_group_identity_sha256": legacy.canonical_sha256(compact_identity),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": EXPECTED_MANIFEST_SHA256,
            "first16_sha256": EXPECTED_MANIFEST_FIRST16_SHA256,
        },
        "base_artifact_sha256": base["artifact_sha256"],
        "adapters": {
            "i23": {"artifact_sha256": i23["artifact_sha256"], "rank": i23_rank},
            "s800": {
                "artifact_sha256": s800["artifact_sha256"],
                "rank": s800_rank,
            },
        },
        "reused_cell": reused_cell,
        "pending_cells": [spec["name"] for spec in CELL_SPECS],
        "outputs": {
            name: {key: str(path.resolve()) for key, path in cell_paths.items()}
            for name, cell_paths in paths.items()
        },
        "decode": decode_lock(),
    }
    return {
        "legacy": legacy,
        "records": records,
        "base": base,
        "i23": i23,
        "s800": s800,
        "i23_rank": i23_rank,
        "s800_rank": s800_rank,
        "paths": paths,
        "report": report,
    }


def decode_lock() -> dict[str, Any]:
    return {
        "reasoning": {
            "samples": REASONING_SAMPLES,
            "max_tokens": MAX_REASONING_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "stop": ["</think>"],
            "seed": SEED,
            "per_prompt_seed": "(seed + rollout_seed) mod 2**31",
        },
        "item_beam": {
            "beam_width": ITEM_CANDIDATES,
            "max_tokens": 3,
            "temperature": 0.0,
            "ignore_eos": False,
            "length_penalty": 1.0,
        },
        "runtime": {
            "batch_prompts": BATCH_PROMPTS,
            "beam_batch_prompts": BEAM_BATCH_PROMPTS,
            "dtype": DTYPE,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "max_logprobs": 2 * ITEM_CANDIDATES,
            "vllm_version": EXPECTED_VLLM_VERSION,
        },
    }


def cell_config(
    spec: dict[str, str],
    output_path: Path,
    manifest_path: Path,
    old_rollout_path: Path,
    base: dict[str, Any],
    adapter: dict[str, Any],
    adapter_rank: int,
    vllm_max_lora_rank: int,
) -> dict[str, Any]:
    renderer = spec["renderer"]
    renderer_detail = (
        {
            "name": "canonical_user",
            "messages": ["user"],
            "query": "instruction + newline + input",
        }
        if renderer == "canonical_user"
        else {
            "name": "legacy_system",
            "messages": ["system", "user"],
            "query": "system=instruction; user=input",
        }
    )
    return {
        "schema_version": "i29-renderer-calibration-cell-config-v1",
        "protocol": "i29-i23-s800-renderer-calibration-n16-v1",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "frozen_generator": {
            "path": str(LEGACY_GENERATOR_PATH.resolve()),
            "sha256": EXPECTED_LEGACY_GENERATOR_SHA256,
            "reuse": "generate_batch with scoped renderer override",
        },
        "cell": {
            "name": spec["name"],
            "adapter_name": spec["adapter"],
            "renderer": renderer_detail,
        },
        "base": base,
        "adapter": {
            **adapter,
            "rank": adapter_rank,
            "vllm_max_lora_rank": vllm_max_lora_rank,
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": EXPECTED_MANIFEST_SHA256,
            "selection": "physical first 16 JSONL rows",
            "selected_rows": FIRST_N,
            "first16_sha256": EXPECTED_MANIFEST_FIRST16_SHA256,
            "contains_labels": False,
        },
        "reused_fourth_cell": {
            "path": str(old_rollout_path.resolve()),
            "sha256": EXPECTED_OLD_ROLLOUT_SHA256,
            "first16_sha256": EXPECTED_OLD_ROLLOUT_FIRST16_SHA256,
            "cell": "s800_legacy",
        },
        "decode": decode_lock(),
        "output": str(output_path.resolve()),
    }


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    ).encode("utf-8")


def publish_exclusive(path: Path, payload: bytes) -> None:
    """Atomically publish complete bytes and fail if the destination exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2), unlike replace(2), is atomic and refuses an existing target.
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> None:
    if not args.gpu or not args.gpu.strip() or "," in args.gpu:
        raise ValueError("--run requires exactly one --gpu device index or UUID")

    # Complete every CPU/hash/output check before CUDA visibility or vLLM import.
    state = preflight(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import BeamSearchParams

    if str(vllm_version) != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"vLLM version mismatch: expected {EXPECTED_VLLM_VERSION}, got {vllm_version}"
        )

    legacy: ModuleType = state["legacy"]
    records: list[dict[str, Any]] = state["records"]
    paths: dict[str, dict[str, Path]] = state["paths"]
    i23_path = Path(args.i23_adapter).resolve()
    s800_path = Path(args.s800_adapter).resolve()
    max_lora_rank = max(
        legacy.vllm_max_lora_rank(state["i23_rank"]),
        legacy.vllm_max_lora_rank(state["s800_rank"]),
    )
    model = LLM(
        model=str(Path(args.base).resolve()),
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        seed=SEED,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_logprobs=2 * ITEM_CANDIDATES,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
    )
    requests = {
        "i23": LoRARequest("i29_i23", 1, lora_path=str(i23_path)),
        "s800": LoRARequest("i29_s800", 2, lora_path=str(s800_path)),
    }
    adapters = {"i23": state["i23"], "s800": state["s800"]}
    ranks = {"i23": state["i23_rank"], "s800": state["s800_rank"]}
    renderers: dict[str, Callable[[dict[str, Any]], str]] = {
        "legacy_system": legacy.build_reasoning_prompt,
        "canonical_user": build_canonical_user_prompt,
    }
    decode_args = fixed_decode_args()
    generated: dict[str, dict[str, Any]] = {}

    # Keep all three small cells in memory.  No formal output is visible unless
    # every generation cell has completed and passed frozen row validation.
    for spec_value in CELL_SPECS:
        spec = {key: str(value) for key, value in spec_value.items()}
        name = spec["name"]
        adapter_name = spec["adapter"]
        rollout_path = paths[name]["rollout"]
        config = cell_config(
            spec,
            rollout_path,
            Path(args.manifest),
            Path(args.legacy_s800_rollout),
            state["base"],
            adapters[adapter_name],
            ranks[adapter_name],
            max_lora_rank,
        )
        generator = legacy.generator_identity(
            config, state["base"], adapters[adapter_name], SEED
        )
        with renderer_override(legacy, renderers[spec["renderer"]]):
            rows = legacy.generate_batch(
                model,
                SamplingParams,
                BeamSearchParams,
                requests[adapter_name],
                records,
                decode_args,
                generator,
            )
        if len(rows) != FIRST_N:
            raise RuntimeError(f"{name} produced {len(rows)} rows, expected {FIRST_N}")
        generated[name] = {"config": config, "generator": generator, "rows": rows}
        print(f"generated in memory: {name} ({len(rows)} rows)", flush=True)

    assert_no_output_collisions(paths)
    for spec_value in CELL_SPECS:
        name = str(spec_value["name"])
        cell = generated[name]
        rollout_payload = jsonl_bytes(cell["rows"])
        config_payload = json_bytes(cell["config"])
        metadata = {
            "schema_version": "i29-renderer-calibration-cell-metadata-v1",
            "status": "complete",
            "cell": name,
            "config_sha256": cell["generator"]["config_sha256"],
            "rollout_sha256": hashlib.sha256(rollout_payload).hexdigest(),
            "rows": FIRST_N,
            "traces": FIRST_N * REASONING_SAMPLES,
            "candidates": FIRST_N * REASONING_SAMPLES * ITEM_CANDIDATES,
        }
        publish_exclusive(paths[name]["config"], config_payload)
        publish_exclusive(paths[name]["rollout"], rollout_payload)
        publish_exclusive(paths[name]["metadata"], json_bytes(metadata))
        print(
            f"published: {paths[name]['rollout']} "
            f"sha256={metadata['rollout_sha256']}",
            flush=True,
        )


def self_test(parser: argparse.ArgumentParser) -> None:
    legacy = load_legacy_module()
    record = {
        "schema_version": legacy.MANIFEST_SCHEMA,
        "group_id": "a" * 64,
        "instruction": "instruction",
        "input": "input /think",
        "history": [],
        "domain": "video",
        "prompt_sha256": legacy.canonical_sha256(
            {"history": [], "input": "input /think", "instruction": "instruction"}
        ),
        "rollout_seed": 17,
    }
    legacy.validate_manifest_record(record, "I29 self-test")
    assert build_canonical_user_prompt(record) == (
        "<|im_start|>user\ninstruction\ninput /think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    legacy_prompt = legacy.build_reasoning_prompt(record)
    assert legacy_prompt == (
        "<|im_start|>system\ninstruction<|im_end|>\n"
        "<|im_start|>user\ninput /think<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    original = legacy.build_reasoning_prompt
    try:
        with renderer_override(legacy, build_canonical_user_prompt):
            assert legacy.build_reasoning_prompt(record) == build_canonical_user_prompt(record)
            raise RuntimeError("exercise restoration")
    except RuntimeError as exc:
        assert str(exc) == "exercise restoration"
    assert legacy.build_reasoning_prompt is original

    all_options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    forbidden_term = "go" + "ld"
    assert not any(forbidden_term in option.lower() for option in all_options)
    args = fixed_decode_args()
    assert (args.seed, args.reasoning_samples, args.item_candidates) == (
        19260829,
        4,
        8,
    )
    assert len(output_paths(Path("/x"))) == 3
    with tempfile.TemporaryDirectory(prefix="i29-generator-selftest-") as directory:
        target = Path(directory) / "atomic.jsonl"
        publish_exclusive(target, b"complete\n")
        assert target.read_bytes() == b"complete\n"
        try:
            publish_exclusive(target, b"must-not-overwrite\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive publisher overwrote an existing target")
        assert target.read_bytes() == b"complete\n"
    print("I29 generator self-test passed (CPU only; vLLM not imported)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate three missing I29 I23/s800 renderer-calibration cells."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true", help="run CPU unit tests")
    modes.add_argument(
        "--preflight", action="store_true", help="hash-check all inputs without vLLM"
    )
    modes.add_argument("--run", action="store_true", help="run the fixed GPU generation")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--i23-adapter", default=str(DEFAULT_I23))
    parser.add_argument("--s800-adapter", default=str(DEFAULT_S800))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--legacy-s800-rollout", default=str(DEFAULT_OLD_S800_LEGACY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--gpu", help="single CUDA device index/UUID; required only with --run"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run and args.gpu is not None:
        parser.error("--gpu is accepted only with --run")
    if args.self_test:
        self_test(parser)
        return
    if args.preflight:
        state = preflight(args)
        print(json.dumps(state["report"], ensure_ascii=False, indent=2, sort_keys=True))
        return
    if not args.gpu:
        parser.error("--run requires --gpu")
    run(args)


if __name__ == "__main__":
    main()
