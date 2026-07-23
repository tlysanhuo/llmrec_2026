#!/usr/bin/env python3
"""Generate one-parent Beam64 AB/first-divergence ledgers for I-39.

The runner reuses the frozen I-34 renderer and beam parser, but loads only the
I-35 step548 adapter.  It intentionally does not run an I-23 or second
teacher request: the ledger describes the parent's A, AB, and full-ABC
coverage and the parent's first-divergence negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPL_PATH = ROOT / "scripts/eval/generate_i34_material_beam_gap_v1.py"
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT = ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"
TRAIN_INPUT = ROOT / "logs/data/i39_userab_video_beam64_pool_v1.jsonl"
DEV_INPUT = ROOT / "logs/data/i39_userab_video_beam64_pool_v1_dev.jsonl"
TRAIN_OUTPUT = ROOT / "logs/data/i39_userab_video_beam64_beam_ledger_v1.jsonl"
DEV_OUTPUT = ROOT / "logs/probe/i39_userab_video_beam64_dev_ledger_v1.jsonl"
AUDIT_OUTPUT = ROOT / "logs/probe/i39_userab_video_beam64_audit_v1.json"

TRAIN_INPUT_SHA256 = "daffaf734a62656c3de08bc92adfb4305c0c04a506068f2c2a40ddcaecc7f0a4"
DEV_INPUT_SHA256 = "56a32e1a5d8271130ab2892631c1957907a8dc4dbd1a0a91239cd8092d9efbde"
BASE_ARTIFACT_SHA256 = "431cc7546a1813ed21a184974a1ac739139b7bdc4643d04e521d066f6ad20652"
PARENT_ARTIFACT_SHA256 = "fd98574e13585eca935570b1209eb518a204c903ce92f7b2a6eee701030546bd"
PARENT_MODEL_SHA256 = "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
PARENT_CONFIG_SHA256 = "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"
EXPECTED_VLLM_VERSION = "0.12.0"
BEAM_WIDTH = 64
SEED = 42


def load_impl() -> Any:
    spec = importlib.util.spec_from_file_location("llmrec_i39_beam_impl", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {IMPL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BEAM_WIDTH = BEAM_WIDTH
    module.AUDIT_SCHEMA_VERSION = "i39-userab-video-beam64-audit-v1"
    module.SCHEMA_VERSION = "i39-userab-video-beam64-ledger-v1"
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash drifted: {actual}/{expected}")


def compact_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "abc": list(candidate["abc"]),
            "cum_logprob": candidate.get("cum_logprob"),
            "rank": int(candidate["rank"]),
            "text": candidate.get("text", ""),
            "token_count": int(candidate.get("token_count", 0)),
            "finish_reason": candidate.get("finish_reason"),
            "stop_reason": candidate.get("stop_reason"),
        }
        for candidate in candidates
    ]


def best_rank(
    candidates: Sequence[Mapping[str, Any]], predicate: Any
) -> int | None:
    ranks = [
        int(candidate["rank"]) + 1
        for candidate in candidates
        if predicate(tuple(str(value) for value in candidate["abc"]))
    ]
    return min(ranks) if ranks else None


def make_ledger_row(
    module: Any,
    row: Any,
    token_meta: Mapping[str, Any],
    result: Mapping[str, Any],
    token_ids_by_abc: Mapping[tuple[str, str, str], list[int]],
    tokenizer: Any,
    parent_model_sha256: str,
) -> dict[str, Any]:
    candidates = result["candidates"]
    gold = tuple(row.gold_abc)
    a_hit = any(str(item["abc"][0]) == gold[0] for item in candidates)
    ab_hit = any(tuple(str(value) for value in item["abc"][:2]) == gold[:2] for item in candidates)
    full_hit = any(tuple(str(value) for value in item["abc"]) == gold for item in candidates)
    all_negatives = module.hard_negative_rows(
        row.gold_abc,
        candidates,
        candidates,
        token_ids_by_abc,
        tokenizer=tokenizer,
        max_total=None,
        max_per_divergence=None,
    )
    negatives = module.hard_negative_rows(
        row.gold_abc,
        candidates,
        candidates,
        token_ids_by_abc,
        tokenizer=tokenizer,
        max_total=12,
        max_per_divergence=4,
    )
    divergence_counts: dict[str, int] = {}
    for negative in all_negatives:
        key = str(int(negative["first_divergence"]))
        divergence_counts[key] = divergence_counts.get(key, 0) + 1
    return {
        "schema_version": "i39-userab-video-beam64-ledger-v1",
        "task": "material_desc2sid",
        "route": row.raw["route"],
        "row_sha256": row.row_sha256,
        "prompt_sha256": row.prompt_sha256,
        "source_prompt_sha256": row.source_prompt_sha256,
        "source_mode_prompt_sha256": row.source_mode_prompt_sha256,
        "prompt_token_sha256": token_meta["prompt_token_sha256"],
        "prompt_token_count": token_meta["prompt_token_count"],
        "renderer_prompt_sha256": row.renderer_prompt_sha256,
        "domain": row.domain,
        "gold_abc": list(row.gold_abc),
        "gold_tokens": list(token_meta["gold_tokens"]),
        "positive_tokens": [list(value) for value in token_meta["positive_tokens"]],
        "parent_adapter_sha256": parent_model_sha256,
        "parent": {
            "name": "i35_step548_r112",
            "full_gold_hit": full_hit,
            "a_hit": a_hit,
            "ab_hit": ab_hit,
            "gold_rank_1based": best_rank(candidates, lambda key: key == gold),
            "best_a_rank_1based": best_rank(candidates, lambda key: key[0] == gold[0]),
            "best_ab_rank_1based": best_rank(candidates, lambda key: key[:2] == gold[:2]),
            "valid_candidates": compact_candidates(candidates),
            "invalid_count": len(result["invalid_ranks"]),
            "invalid_ranks": list(result["invalid_ranks"]),
            "beam_count": int(result["beam_count"]),
        },
        "ab_first_divergence": {
            "definition": "A/AB/full coverage from the single I35 step548 Beam64",
            "a_hit": a_hit,
            "ab_hit": ab_hit,
            "full_abc_hit": full_hit,
            "first_divergence_counts": divergence_counts,
            "hard_negative_pool_count": len(all_negatives),
            "hard_negative_count": len(negatives),
            "hard_negative_dropped_count": len(all_negatives) - len(negatives),
            "trainer_ready": bool(not full_hit and negatives),
        },
        "hard_negatives": negatives,
        "formal_training_generated": False,
    }


def load_and_check(module: Any, path: Path, route: str, expected_hash: str) -> Any:
    pool = module.load_pool(path, route, None, require_native_no_think=True, require_empty_think=True)
    if pool.file_sha256 != expected_hash:
        raise RuntimeError(f"pool hash drifted: {pool.file_sha256}/{expected_hash}")
    return pool


def preflight() -> dict[str, Any]:
    module = load_impl()
    train = load_and_check(module, TRAIN_INPUT, module.TRAIN_ROUTE, TRAIN_INPUT_SHA256)
    dev = load_and_check(module, DEV_INPUT, module.DEV_ROUTE, DEV_INPUT_SHA256)
    base_artifact = module.artifact_fingerprint(BASE, adapter=False)
    parent_artifact = module.artifact_fingerprint(PARENT, adapter=True)
    if base_artifact["artifact_sha256"] != BASE_ARTIFACT_SHA256:
        raise RuntimeError("I39 base artifact fingerprint drifted")
    if parent_artifact["artifact_sha256"] != PARENT_ARTIFACT_SHA256:
        raise RuntimeError("I39 parent artifact fingerprint drifted")
    if module.adapter_rank(PARENT) != 112:
        raise RuntimeError("I39 parent rank must be 112")
    if next(item["sha256"] for item in parent_artifact["files"] if item["path"] == "adapter_model.safetensors") != PARENT_MODEL_SHA256:
        raise RuntimeError("I39 parent adapter hash drifted")
    if next(item["sha256"] for item in parent_artifact["files"] if item["path"] == "adapter_config.json") != PARENT_CONFIG_SHA256:
        raise RuntimeError("I39 parent config hash drifted")
    return {
        "train": {"path": str(TRAIN_INPUT), "rows": len(train.rows), "sha256": train.file_sha256},
        "dev": {"path": str(DEV_INPUT), "rows": len(dev.rows), "sha256": dev.file_sha256},
        "base_artifact": BASE_ARTIFACT_SHA256,
        "parent_artifact": PARENT_ARTIFACT_SHA256,
        "parent_adapter_sha256": PARENT_MODEL_SHA256,
        "parent_rank": 112,
        "beam_width": BEAM_WIDTH,
        "single_parent_request": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    module = load_impl()
    if "," in args.gpu or not args.gpu.strip():
        raise RuntimeError("--gpu must name exactly one CUDA device")
    train = load_and_check(module, TRAIN_INPUT, module.TRAIN_ROUTE, TRAIN_INPUT_SHA256)
    dev = load_and_check(module, DEV_INPUT, module.DEV_ROUTE, DEV_INPUT_SHA256)
    base_artifact = module.artifact_fingerprint(BASE, adapter=False)
    parent_artifact = module.artifact_fingerprint(PARENT, adapter=True)
    if base_artifact["artifact_sha256"] != BASE_ARTIFACT_SHA256:
        raise RuntimeError("I39 base artifact fingerprint drifted")
    if parent_artifact["artifact_sha256"] != PARENT_ARTIFACT_SHA256:
        raise RuntimeError("I39 parent artifact fingerprint drifted")
    if module.adapter_rank(PARENT) != 112:
        raise RuntimeError("I39 parent rank must be 112")
    parent_model_sha = next(
        item["sha256"]
        for item in parent_artifact["files"]
        if item["path"] == "adapter_model.safetensors"
    )
    if parent_model_sha != PARENT_MODEL_SHA256:
        raise RuntimeError("I39 parent adapter hash drifted")
    parent_config_sha = next(
        item["sha256"]
        for item in parent_artifact["files"]
        if item["path"] == "adapter_config.json"
    )
    if parent_config_sha != PARENT_CONFIG_SHA256:
        raise RuntimeError("I39 parent config hash drifted")

    output_paths = (TRAIN_OUTPUT, DEV_OUTPUT, AUDIT_OUTPUT)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise RuntimeError("I39 output paths must be distinct")
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise RuntimeError("I39 refuses to overwrite outputs: " + ", ".join(existing))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import vllm
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    version = str(getattr(vllm, "__version__", "unknown"))
    if version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(f"vLLM version mismatch: {version}/{EXPECTED_VLLM_VERSION}")
    llm = LLM(
        model=str(BASE),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=SEED,
        enable_prefix_caching=True,
        trust_remote_code=True,
        max_logprobs=130,
        enable_lora=True,
        max_lora_rank=128,
    )
    tokenizer = llm.get_tokenizer()
    request = LoRARequest("i39_i35_parent", 1, lora_path=str(PARENT))
    all_rows = list(train.rows) + list(dev.rows)
    token_meta = {
        row.row_sha256: module.row_token_metadata(row, tokenizer)
        for row in all_rows
    }
    token_ids_by_abc = module.build_token_id_lookup(tokenizer, all_rows)

    def generate(pool: Any) -> list[dict[str, Any]]:
        prompts = [row.renderer_prompt for row in pool.rows]
        domains = [row.domain for row in pool.rows]
        generated = module.run_adapter_beam(
            llm, tokenizer, prompts, request, domains, args.batch_size
        )
        for result in generated:
            for candidate in result["candidates"]:
                key = tuple(str(value) for value in candidate["abc"])
                if key not in token_ids_by_abc:
                    token_ids_by_abc[key] = module.convert_token_ids(
                        tokenizer,
                        [f"<s_a_{key[0]}>", f"<s_b_{key[1]}>", f"<s_c_{key[2]}>"],
                    )
        return [
            make_ledger_row(
                module,
                row,
                token_meta[row.row_sha256],
                result,
                token_ids_by_abc,
                tokenizer,
                parent_model_sha,
            )
            for row, result in zip(pool.rows, generated)
        ]

    train_ledger = generate(train)
    dev_ledger = generate(dev)
    module.atomic_write_jsonl(TRAIN_OUTPUT, train_ledger, overwrite=args.overwrite)
    module.atomic_write_jsonl(DEV_OUTPUT, dev_ledger, overwrite=args.overwrite)

    combined = train_ledger + dev_ledger
    counts = {
        "rows": len(combined),
        "a_hit": sum(int(row["parent"]["a_hit"]) for row in combined),
        "ab_hit": sum(int(row["parent"]["ab_hit"]) for row in combined),
        "full_abc_hit": sum(int(row["parent"]["full_gold_hit"]) for row in combined),
        "ab_hit_c_miss": sum(
            int(row["parent"]["ab_hit"] and not row["parent"]["full_gold_hit"])
            for row in combined
        ),
        "a_miss": sum(int(not row["parent"]["a_hit"]) for row in combined),
        "invalid_candidates": sum(int(row["parent"]["invalid_count"]) for row in combined),
        "trainer_ready": sum(int(row["ab_first_divergence"]["trainer_ready"]) for row in combined),
    }
    audit = {
        "schema_version": "i39-userab-video-beam64-audit-v1",
        "status": "complete",
        "formal_training_generated": False,
        "selection_definition": {
            "a_hit": "any candidate with gold A in Beam64",
            "ab_hit": "any candidate with gold A and B in Beam64",
            "full_abc_hit": "exact gold triple in Beam64",
            "first_divergence": "parent candidate first differing SID position",
        },
        "inputs": {
            "train": {"path": str(TRAIN_INPUT), "rows": len(train.rows), "sha256": train.file_sha256},
            "dev": {"path": str(DEV_INPUT), "rows": len(dev.rows), "sha256": dev.file_sha256},
        },
        "parent": {
            "name": "i35_step548_r112",
            "path": str(PARENT),
            "artifact_sha256": PARENT_ARTIFACT_SHA256,
            "adapter_model_sha256": parent_model_sha,
            "adapter_config_sha256": parent_config_sha,
            "rank": 112,
        },
        "base_artifact_sha256": BASE_ARTIFACT_SHA256,
        "runtime": {
            "gpu": args.gpu,
            "single_parent_request": True,
            "vllm_version": version,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": SEED,
            "beam_width": BEAM_WIDTH,
            "max_tokens": 3,
            "renderer": "I34 offline_eval native /no_think renderer + fixed video prefix",
            "batch_size": args.batch_size,
        },
        "counts": counts,
        "outputs": {
            "train_ledger": {"path": str(TRAIN_OUTPUT), "rows": len(train_ledger), "sha256": file_sha256(TRAIN_OUTPUT)},
            "dev_ledger": {"path": str(DEV_OUTPUT), "rows": len(dev_ledger), "sha256": file_sha256(DEV_OUTPUT)},
        },
        "script_sha256": file_sha256(Path(__file__).resolve()),
    }
    module.atomic_write_json(AUDIT_OUTPUT, audit, overwrite=args.overwrite)
    audit["audit_sha256"] = file_sha256(AUDIT_OUTPUT)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        load_impl()._self_test()
        print("i39 single-parent Beam64 self-test: PASS")
        return
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.batch_size < 1 or not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("invalid batch size or GPU memory utilization")
    run(args)


if __name__ == "__main__":
    main()
