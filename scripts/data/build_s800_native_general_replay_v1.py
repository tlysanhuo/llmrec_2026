#!/usr/bin/env python3
"""Build the small native-General replay mix over the current s800 parent.

The positive branch contains every one of the 129 task-fit-reviewed official
native General rows exactly once.  The trust-region branch contains 48 rows
from each of the eight existing parent tasks (two material directions, two
user tasks, and four recommendation domains), selected deterministically and
copied without changing the four trainer-visible Alpaca fields.

The JSONL ``route`` metadata is for audit only because LLaMA-Factory discards
unregistered columns.  Runtime routing is therefore locked by a second asset:
the SHA256 of the exact qwen3_nothink assistant target-token sequence for each
row.  The custom trainer accepts CE only for hashes registered as
``general_ce`` and fails on every unknown target hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from build_seed_scoremax_v1 import task_of


ROOT = Path(__file__).resolve().parents[2]
PERSONAL_ROOT = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51")
DEFAULT_GENERAL = (
    ROOT / "assets/derived/processed/data_official_general_native_static_reviewed_v1.jsonl"
)
DEFAULT_GENERAL_LINEAGE = (
    ROOT
    / "assets/derived/official_general/official_general_native_static_reviewed_v1_lineage.jsonl"
)
DEFAULT_RETENTION = (
    ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
)
DEFAULT_TOKENIZER = ROOT / "assets/official/base_model"
DEFAULT_OUT = (
    ROOT / "assets/derived/processed/data_s800_native_general_replay_v1.jsonl"
)
DEFAULT_ROUTE_MANIFEST = (
    ROOT / "assets/derived/official_general/s800_native_general_replay_v1_routes.json"
)
DEFAULT_AUDIT = ROOT / "logs/data/s800_native_general_replay_v1_audit.json"

GENERAL_SHA256 = "867f109366142f7e6ce19133129fc12f3dc75f0b4d811e298b5edfba5aa16237"
GENERAL_LINEAGE_SHA256 = "adf174f32e4e485128672c87d3a26037e8c418da9297721f7ae85a02c0e690e4"
RETENTION_SHA256 = "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0"
EXPECTED_GENERAL_ROWS = 129
EXPECTED_RETENTION_SOURCE_ROWS = 6_106
DEFAULT_SEED = 19_260_831
CUTOFF_LEN = 16_384

GENERAL_ROUTE = "general_ce"
RETENTION_ROUTE = "retention_kl"
SCHEMA_VERSION = "s800-native-general-replay-v1"
ROUTE_MANIFEST_SCHEMA = "qwen3-nothink-assistant-target-sha256-v1"
CORE_KEYS = ("instruction", "input", "output", "history")
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")

RETENTION_QUOTAS = {
    "material_desc2sid": 48,
    "material_sid2desc": 48,
    "action": 48,
    "topic": 48,
    "rec_video": 48,
    "rec_prod": 48,
    "rec_ad": 48,
    "rec_living": 48,
}
EXPECTED_RETENTION_ROWS = sum(RETENTION_QUOTAS.values())
EXPECTED_ROWS = EXPECTED_GENERAL_ROWS + EXPECTED_RETENTION_ROWS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode("ascii")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            row = {
                "instruction": str(raw.get("instruction", "") or ""),
                "input": str(raw.get("input", "") or ""),
                "output": str(raw.get("output", "") or ""),
                "history": raw.get("history") or [],
            }
            if not row["input"] or not row["output"] or not isinstance(row["history"], list):
                raise ValueError(f"invalid Alpaca row at {path}:{line_number}")
            rows.append(row)
    return rows


def normalized_prompt_key(row: dict[str, Any]) -> str:
    return canonical_json(
        {
            "instruction": row["instruction"],
            "input_core": MODE_SUFFIX_RE.sub("", row["input"].rstrip()),
            "history": row["history"],
        }
    )


def valid_response_structure(output: str) -> bool:
    return (
        output.startswith("<think>")
        and output.count("<think>") == 1
        and output.count("</think>") == 1
        and output.index("</think>") > 0
        and bool(output.split("</think>", 1)[1].strip())
        and "<|im_end|>" not in output
        and "<|endoftext|>" not in output
    )


def build_general_rows(
    source: list[dict[str, Any]], lineage: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(source) != EXPECTED_GENERAL_ROWS or len(lineage) != EXPECTED_GENERAL_ROWS:
        raise AssertionError(f"General source signature drifted: {len(source)}/{len(lineage)}")

    rows: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for index, (row, meta) in enumerate(zip(source, lineage), start=1):
        record_id = str(meta.get("record_id") or "")
        review = meta.get("task_fit_review") or {}
        if not record_id or record_id in record_ids:
            raise AssertionError(f"missing or duplicate General record_id at row {index}")
        if review.get("status") != "pass" or review.get("factual_gold_certified") is not False:
            raise AssertionError(f"General review contract drifted at row {index}")
        if meta.get("mode") != "think" or not row["input"].rstrip().endswith("/think"):
            raise AssertionError(f"General native /think route drifted at row {index}")
        if not valid_response_structure(row["output"]):
            raise AssertionError(f"General response structure drifted at row {index}")
        record_ids.add(record_id)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route": GENERAL_ROUTE,
                "task": "general",
                "domain": str(meta.get("domain") or "unknown"),
                "record_id": record_id,
                "upstream_ids": ["D-reviewed(O2.General)-native-static-SFT"],
                **row,
            }
        )

    return rows, {
        "rows": len(rows),
        "unique_record_ids": len(record_ids),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "dose": "every reviewed row exactly once",
        "source_response_role": "official_native_sft_supervision_not_independent_factual_gold",
    }


def build_retention_rows(
    source: list[dict[str, Any]], general_prompt_keys: set[str], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(source) != EXPECTED_RETENTION_SOURCE_ROWS:
        raise AssertionError(f"retention source row count drifted: {len(source)}")

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: defaultdict(list) for task in RETENTION_QUOTAS
    }
    invalid_structure: Counter[str] = Counter()
    general_prompt_overlap: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in source:
        try:
            task = task_of(row)
        except ValueError:
            source_counts["world"] += 1
            continue
        source_counts[task] += 1
        if task not in groups:
            continue
        if not valid_response_structure(row["output"]):
            invalid_structure[task] += 1
            continue
        prompt_key = normalized_prompt_key(row)
        if prompt_key in general_prompt_keys:
            general_prompt_overlap[task] += 1
            continue
        groups[task][prompt_key].append(row)

    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    selected_counts: Counter[str] = Counter()
    for task, quota in RETENTION_QUOTAS.items():
        task_groups = groups[task]
        available[task] = len(task_groups)
        ordered_keys = sorted(
            task_groups,
            key=lambda prompt_key: stable_hash(
                seed, "s800-general-retention-group-v1", task, prompt_key
            ),
        )
        if len(ordered_keys) < quota:
            raise AssertionError(f"only {len(ordered_keys)} eligible {task} groups; need {quota}")
        for prompt_key in ordered_keys[:quota]:
            candidates = task_groups[prompt_key]
            row = min(
                candidates,
                key=lambda value: stable_hash(
                    seed, "s800-general-retention-row-v1", task, canonical_json(value)
                ),
            )
            output = {
                "schema_version": SCHEMA_VERSION,
                "route": RETENTION_ROUTE,
                "task": task,
                "domain": task.removeprefix("rec_").split("_", 1)[0],
                "record_id": stable_hash(SCHEMA_VERSION, task, prompt_key),
                "upstream_ids": ["data_user_residual_retention_v1"],
                **row,
            }
            if any(output[key] != row[key] for key in CORE_KEYS):
                raise AssertionError("retention teacher row was modified")
            selected.append(output)
            selected_counts[task] += 1

    if len(selected) != EXPECTED_RETENTION_ROWS or dict(selected_counts) != RETENTION_QUOTAS:
        raise AssertionError(f"retention quota drifted: {len(selected)}/{dict(selected_counts)}")
    if len({row["record_id"] for row in selected}) != len(selected):
        raise AssertionError("retention record IDs are not unique")
    return selected, {
        "rows": len(selected),
        "quota": dict(RETENTION_QUOTAS),
        "selected_task_counts": dict(sorted(selected_counts.items())),
        "source_task_counts": dict(sorted(source_counts.items())),
        "available_prompt_groups": dict(sorted(available.items())),
        "invalid_response_structure_excluded": dict(sorted(invalid_structure.items())),
        "general_prompt_overlap_excluded": dict(sorted(general_prompt_overlap.items())),
        "teacher_core_field_changes": 0,
        "teacher_semantics": "s800-parent KL only; no gold CE",
        "selection": "stable hash without replacement over mode-normalized prompt groups",
    }


def qwen3_target_ids(tokenizer: Any, output: str) -> list[int]:
    if tokenizer.eos_token != "<|im_end|>" or int(tokenizer.eos_token_id) != 151_645:
        raise AssertionError(
            f"qwen3_nothink EOS drifted: {tokenizer.eos_token!r}/{tokenizer.eos_token_id}"
        )
    return tokenizer.encode(output + tokenizer.eos_token + "\n", add_special_tokens=False)


def qwen3_formatted_text(row: dict[str, Any], tokenizer: Any) -> str:
    parts: list[str] = []
    for old_prompt, old_response in row["history"]:
        parts.append(
            f"<|im_start|>user\n{old_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{old_response}{tokenizer.eos_token}\n"
        )
    user_content = "\n".join(
        value for value in (row["instruction"], row["input"]) if value
    )
    parts.append(
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['output']}{tokenizer.eos_token}\n"
    )
    return "".join(parts)


def build_route_manifest_and_token_audit(
    rows: list[dict[str, Any]], tokenizer_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.eos_token != "<|im_end|>":
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    route_hashes: dict[str, list[str]] = {GENERAL_ROUTE: [], RETENTION_ROUTE: []}
    route_counts: Counter[str] = Counter()
    target_lengths: list[int] = []
    formatted_lengths: list[int] = []
    all_hash_routes: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        route = row["route"]
        if route not in route_hashes:
            raise AssertionError(f"unknown route at row {index}: {route}")
        target_ids = qwen3_target_ids(tokenizer, row["output"])
        target_hash = token_sha256(target_ids)
        route_hashes[route].append(target_hash)
        all_hash_routes[target_hash].add(route)
        route_counts[route] += 1
        target_lengths.append(len(target_ids))
        formatted_len = len(
            tokenizer.encode(qwen3_formatted_text(row, tokenizer), add_special_tokens=False)
        )
        formatted_lengths.append(formatted_len)
        if formatted_len > CUTOFF_LEN:
            raise AssertionError(
                f"row {index} exceeds cutoff: {formatted_len}>{CUTOFF_LEN}"
            )

    collisions = sorted(
        value for value, routes in all_hash_routes.items() if len(routes) > 1
    )
    if collisions:
        raise AssertionError(f"target hashes collide across CE/KL routes: {collisions[:3]}")
    if route_counts != {GENERAL_ROUTE: EXPECTED_GENERAL_ROWS, RETENTION_ROUTE: EXPECTED_RETENTION_ROWS}:
        raise AssertionError(f"route count drifted during token audit: {route_counts}")
    general_hashes = sorted(set(route_hashes[GENERAL_ROUTE]))
    if len(general_hashes) != EXPECTED_GENERAL_ROWS:
        raise AssertionError("General target-token hashes are not unique")

    manifest = {
        "schema_version": ROUTE_MANIFEST_SCHEMA,
        "hash_encoding": "sha256(ascii comma-separated decimal target token IDs)",
        "template": "qwen3_nothink",
        "assistant_target": "tokenizer.encode(output + '<|im_end|>\\n', add_special_tokens=False)",
        "eos_token": tokenizer.eos_token,
        "eos_token_id": int(tokenizer.eos_token_id),
        "expected_microbatches": EXPECTED_ROWS,
        "route_counts": dict(sorted(route_counts.items())),
        "general_ce_target_sha256": general_hashes,
        "retention_kl_target_sha256": sorted(set(route_hashes[RETENTION_ROUTE])),
        "retention_unique_target_sha256_count": len(set(route_hashes[RETENTION_ROUTE])),
        "cross_route_target_sha256_collisions": 0,
    }
    audit = {
        "status": "PASS",
        "tokenizer_path": str(tokenizer_path.resolve()),
        "template": "qwen3_nothink",
        "rows_checked": len(rows),
        "route_counts": dict(sorted(route_counts.items())),
        "general_unique_target_hashes": len(general_hashes),
        "retention_unique_target_hashes": len(set(route_hashes[RETENTION_ROUTE])),
        "cross_route_target_hash_collisions": 0,
        "assistant_target_tokens": {
            "min": min(target_lengths),
            "max": max(target_lengths),
        },
        "formatted_tokens": {
            "cutoff_len": CUTOFF_LEN,
            "overflow_rows": 0,
            "min": min(formatted_lengths),
            "max": max(formatted_lengths),
        },
    }
    return manifest, audit


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(canonical_json(row) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general", type=Path, default=DEFAULT_GENERAL)
    parser.add_argument("--general-lineage", type=Path, default=DEFAULT_GENERAL_LINEAGE)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not PERSONAL_ROOT.is_mount() or not PERSONAL_ROOT.is_dir():
        raise RuntimeError(f"personal volume is not mounted: {PERSONAL_ROOT}")
    outputs = (args.out, args.route_manifest, args.audit)
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing replay asset")
    expected_hashes = {
        args.general: GENERAL_SHA256,
        args.general_lineage: GENERAL_LINEAGE_SHA256,
        args.retention: RETENTION_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"upstream hash drifted for {path}: {actual}/{expected}")

    general_source = load_jsonl(args.general)
    with args.general_lineage.open(encoding="utf-8") as source:
        lineage = [json.loads(line) for line in source if line.strip()]
    retention_source = load_jsonl(args.retention)
    general_rows, general_audit = build_general_rows(general_source, lineage)
    general_prompt_keys = {normalized_prompt_key(row) for row in general_rows}
    if len(general_prompt_keys) != EXPECTED_GENERAL_ROWS:
        raise AssertionError("General prompts are not unique after mode normalization")
    retention_rows, retention_audit = build_retention_rows(
        retention_source, general_prompt_keys, args.seed
    )

    rows = general_rows + retention_rows
    random.Random(args.seed).shuffle(rows)
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(f"formal mix row count drifted: {len(rows)}/{EXPECTED_ROWS}")
    manifest, token_audit = build_route_manifest_and_token_audit(rows, args.tokenizer)

    atomic_jsonl(args.out, rows)
    manifest["training_data_sha256"] = sha256_file(args.out)
    atomic_json(args.route_manifest, manifest)
    audit = {
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General)",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": SCHEMA_VERSION,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__)),
        "seed": args.seed,
        "parent": {
            "name": "e3_userres_r80_retkl_v3_s800",
            "role": "unique policy parent and disabled-adapter KL reference",
        },
        "upstream": {
            "reviewed_general": {
                "path": str(args.general.resolve()),
                "rows": len(general_source),
                "sha256": GENERAL_SHA256,
            },
            "reviewed_general_lineage": {
                "path": str(args.general_lineage.resolve()),
                "rows": len(lineage),
                "sha256": GENERAL_LINEAGE_SHA256,
            },
            "parent_distribution": {
                "path": str(args.retention.resolve()),
                "rows": len(retention_source),
                "sha256": RETENTION_SHA256,
            },
        },
        "training_rows": {
            "rows": len(rows),
            "route_counts": dict(sorted(Counter(row["route"] for row in rows).items())),
            "general": general_audit,
            "retention": retention_audit,
            "qwen3_token_audit": token_audit,
        },
        "loss_semantics": {
            "general_ce": "full assistant target CE plus 0.05x exact s800-parent KL",
            "retention_kl": "4.0x exact s800-parent KL only; gold target receives no CE",
        },
        "forbidden_sources": {
            "T_rows": 0,
            "E_rows_in_training": 0,
            "model_rollout_rows": 0,
            "unreviewed_270_candidate_rows": 0,
            "legacy_world_zh_rows": 0,
            "world_zh_ext_rows": 0,
            "math_mc_patch_rows": 0,
        },
        "outputs": {
            "training": {
                "path": str(args.out.resolve()),
                "rows": len(rows),
                "bytes": args.out.stat().st_size,
                "sha256": sha256_file(args.out),
            },
            "route_manifest": {
                "path": str(args.route_manifest.resolve()),
                "bytes": args.route_manifest.stat().st_size,
                "sha256": sha256_file(args.route_manifest),
            },
        },
        "formal_training_started": False,
    }
    atomic_json(args.audit, audit)
    print(json.dumps(audit["training_rows"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[OK] training: {args.out}")
    print(f"[OK] route manifest: {args.route_manifest}")
    print(f"[OK] audit: {args.audit}")


if __name__ == "__main__":
    main()
