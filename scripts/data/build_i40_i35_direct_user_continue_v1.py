#!/usr/bin/env python3
"""Build the guarded I-40 direct-I35-r112 continuation mix.

The user-supervision half of this experiment is not rebuilt from the imported
15,023-row construction source.  It is copied byte-for-byte from the already
audited I-36 formal asset and contains exactly 4,000 action plus 1,500 topic
rows.  Every one of the 2,740 I-35 formal rows is retained exactly once, but
its old r96-relative boundary/preserve objective is deliberately discarded:
all of those prompts are KL-only anchors against the frozen I-35 step548
reference during I-40.

The builder is fail closed.  It locks every upstream checksum, requires
prompt-disjoint user and retention routes, creates a full token-hash routing
sidecar, and refuses to overwrite formal outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "i40-i35-direct-user-continue-r112-v1"
SEED = 19260840

I36_DATA = ROOT / "assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl"
I36_AUDIT = ROOT / "logs/data/i36_i35_user_expand_retkl_v1_audit.json"
I35_DATA = ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"
I35_AUDIT = ROOT / "logs/data/i35_video_boundary_retkl_v1_audit.json"
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT = ROOT / "submissions/i35_r96_video_boundary_retkl_r112_step548_platform"

OUTPUT = ROOT / "assets/derived/processed/data_i40_i35_direct_user_continue_v1.jsonl"
SIDECAR = (
    ROOT
    / "assets/derived/processed/data_i40_i35_direct_user_continue_v1_sidecar.jsonl"
)
AUDIT = ROOT / "logs/data/i40_i35_direct_user_continue_v1_audit.json"

EXPECTED_I36_DATA_SHA256 = (
    "2720746a2e8aa7804d519698ce9f2b127e9be2db1d4488e642e800a5337b692d"
)
EXPECTED_I36_AUDIT_SHA256 = (
    "eb426018525f9e3e1d682e1c89e5ca3dc8963b0a57c104911e1197324e464240"
)
EXPECTED_I35_DATA_SHA256 = (
    "9c044e47d26fb7644281107a548249e49564e0f203a04795337c6a90c0927100"
)
EXPECTED_I35_AUDIT_SHA256 = (
    "7f72ebeeeb3718bc21ccb6c4831a02aa5407ee4da1dd7dba6a01541ad8b63ad1"
)
PARENT_ADAPTER_SHA256 = (
    "52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00"
)
PARENT_CONFIG_SHA256 = (
    "4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996"
)
BASE_CONFIG_SHA256 = (
    "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
)

USER_COUNTS = {"action": 4_000, "topic": 1_500}
RETENTION_COUNTS = {
    "material_desc2sid": 1_370,
    "action": 207,
    "topic": 206,
    "rec_video": 206,
    "rec_prod": 207,
    "rec_ad": 206,
    "rec_living": 207,
    "world": 131,
}
ROUTE_COUNTS = {"user_ce": 5_500, "retention_kl": 2_740}
TOTAL_ROWS = sum(ROUTE_COUNTS.values())
OPTIMIZER_STEPS_BATCH1_ACC4 = 2_060
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised when an I-40 construction invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return sha256_bytes(canonical([SEED, *parts]).encode("utf-8"))


def token_hash(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(value)) for value in ids).encode("ascii")
    return sha256_bytes(payload)


def routing_token_hash(prompt_ids: Sequence[int], response_ids: Sequence[int]) -> str:
    return sha256_bytes(
        canonical(
            {
                "prompt_token_sha256": token_hash(prompt_ids),
                "response_token_sha256": token_hash(response_ids),
            }
        ).encode("utf-8")
    )


def normalized(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def core_hash(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical(normalized(row)).encode("utf-8"))


def prompt_hash(row: Mapping[str, Any], *, mode_normalized: bool = False) -> str:
    value = normalized(row)
    user_input = value["input"].rstrip()
    if mode_normalized:
        user_input = re.sub(r"/(?:no_)?think\s*$", "", user_input).rstrip()
    return sha256_bytes(
        canonical([value["instruction"], user_input, value["history"]]).encode(
            "utf-8"
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank row at {path}:{line_number}")
            value = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
            require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def verify_upstreams() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {
        I36_DATA: EXPECTED_I36_DATA_SHA256,
        I36_AUDIT: EXPECTED_I36_AUDIT_SHA256,
        I35_DATA: EXPECTED_I35_DATA_SHA256,
        I35_AUDIT: EXPECTED_I35_AUDIT_SHA256,
        PARENT / "adapter_model.safetensors": PARENT_ADAPTER_SHA256,
        PARENT / "adapter_config.json": PARENT_CONFIG_SHA256,
        BASE / "config.json": BASE_CONFIG_SHA256,
    }
    for path, wanted in expected.items():
        require(path.is_file(), f"missing frozen upstream: {path}")
        observed = sha256_file(path)
        require(observed == wanted, f"upstream hash drift: {path}: {observed}/{wanted}")

    i36_audit = load_json(I36_AUDIT)
    i35_audit = load_json(I35_AUDIT)
    require(
        i36_audit.get("construction", {})
        .get("forbidden_sources", {})
        .get("E_rows")
        == 0,
        "I36 audit no longer proves E_rows=0",
    )
    require(
        i36_audit.get("construction", {})
        .get("forbidden_sources", {})
        .get("third_party_rows")
        == 0,
        "I36 audit no longer proves third_party_rows=0",
    )
    require(
        i35_audit.get("forbidden_sources", {}).get("E_rows") == 0
        and i35_audit.get("forbidden_sources", {}).get("third_party_rows") == 0,
        "I35 audit no longer proves E/T-free formal data",
    )

    i36_rows = load_jsonl(I36_DATA)
    i35_rows = load_jsonl(I35_DATA)
    require(len(i36_rows) == 16_500, f"I36 row count drifted: {len(i36_rows)}")
    require(len(i35_rows) == 2_740, f"I35 row count drifted: {len(i35_rows)}")
    return i36_rows, i35_rows


def select_formal_rows(
    i36_rows: Sequence[Mapping[str, Any]],
    i35_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    user_rows: list[dict[str, Any]] = []
    user_tasks: Counter[str] = Counter()
    for source_index, row in enumerate(i36_rows):
        if row.get("i36_route") != "user_ce":
            continue
        task = str(row.get("task") or "")
        require(task in USER_COUNTS, f"unexpected I36 user task: {task!r}")
        value = normalized(row)
        require(value["input"] and value["output"], "empty I36 user prompt/response")
        require(value["history"] == [], "I40 requires flattened empty history")
        value.update(
            {
                "route": "user_ce",
                "task": task,
                "i40_source": {
                    "asset": "data_i36_i35_user_expand_retkl_v1",
                    "source_index": source_index,
                    "source_route": "user_ce",
                    "source_row_sha256": core_hash(row),
                },
            }
        )
        user_rows.append(value)
        user_tasks[task] += 1
    require(dict(user_tasks) == USER_COUNTS, f"user task counts drifted: {dict(user_tasks)}")

    retention_rows: list[dict[str, Any]] = []
    retention_tasks: Counter[str] = Counter()
    old_routes: Counter[str] = Counter()
    for source_index, row in enumerate(i35_rows):
        task = str(row.get("task") or "")
        require(task in RETENTION_COUNTS, f"unexpected I35 task: {task!r}")
        old_route = str(row.get("route") or "")
        require(
            old_route in {"material_boundary", "retention_kl"},
            f"unexpected I35 source route: {old_route!r}",
        )
        value = normalized(row)
        require(value["input"] and value["output"], "empty I35 prompt/response")
        require(value["history"] == [], "I40 requires flattened empty history")
        value.update(
            {
                "route": "retention_kl",
                "task": task,
                "i40_source": {
                    "asset": "data_i35_video_boundary_retkl_v1",
                    "source_index": source_index,
                    "source_route": old_route,
                    "source_row_sha256": core_hash(row),
                    "objective_reuse": False,
                    "semantics": "KL-only against frozen I35 step548",
                },
            }
        )
        retention_rows.append(value)
        retention_tasks[task] += 1
        old_routes[old_route] += 1
    require(
        dict(retention_tasks) == RETENTION_COUNTS,
        f"retention task counts drifted: {dict(retention_tasks)}",
    )
    require(
        dict(old_routes) == {"material_boundary": 1_370, "retention_kl": 1_370},
        f"I35 source routes drifted: {dict(old_routes)}",
    )

    user_exact = {prompt_hash(row) for row in user_rows}
    user_mode = {prompt_hash(row, mode_normalized=True) for row in user_rows}
    retention_exact = {prompt_hash(row) for row in retention_rows}
    retention_mode = {
        prompt_hash(row, mode_normalized=True) for row in retention_rows
    }
    require(not (user_exact & retention_exact), "user/retention exact prompt overlap")
    require(not (user_mode & retention_mode), "user/retention mode prompt overlap")

    rows = [*user_rows, *retention_rows]
    row_hash_counts = Counter(core_hash(row) for row in rows)
    require(
        max(row_hash_counts.values()) <= 2,
        "formal row exposure exceeds the inherited maximum of two",
    )
    rows.sort(key=lambda row: stable_hash("formal_shuffle", core_hash(row), row["route"]))
    require(len(rows) == TOTAL_ROWS, f"formal total drifted: {len(rows)}")
    routes = Counter(str(row["route"]) for row in rows)
    require(dict(routes) == ROUTE_COUNTS, f"formal routes drifted: {dict(routes)}")
    return rows, {
        "user_tasks": dict(user_tasks),
        "retention_tasks": dict(retention_tasks),
        "i35_source_routes": dict(old_routes),
        "user_exact_prompts": len(user_exact),
        "user_mode_prompts": len(user_mode),
        "retention_exact_prompts": len(retention_exact),
        "retention_mode_prompts": len(retention_mode),
        "unique_normalized_rows": len(row_hash_counts),
        "duplicate_row_exposures": len(rows) - len(row_hash_counts),
        "maximum_normalized_row_exposure": max(row_hash_counts.values()),
        "user_retention_exact_overlap": 0,
        "user_retention_mode_overlap": 0,
    }


def tokenize_sidecar(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    sidecar: list[dict[str, Any]] = []
    prompt_token_hashes: set[str] = set()
    routing_token_hashes: Counter[str] = Counter()
    route_tasks: Counter[str] = Counter()
    maximum_tokens = 0
    for row_number, row in enumerate(rows, 1):
        value = normalized(row)
        prompt_ids, response_ids = template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": value["input"]},
                {"role": "assistant", "content": value["output"]},
            ],
            value["instruction"],
            None,
        )
        total = len(prompt_ids) + len(response_ids)
        require(total <= 16_384, f"cutoff overflow at formal row {row_number}: {total}")
        prompt_token_sha256 = token_hash(prompt_ids)
        prompt_token_hashes.add(prompt_token_sha256)
        response_token_sha256 = token_hash(response_ids)
        routing_token_sha256 = routing_token_hash(prompt_ids, response_ids)
        routing_token_hashes[routing_token_sha256] += 1
        require(
            routing_token_hashes[routing_token_sha256] <= 2,
            f"tokenized prompt/response exposure exceeds two at row {row_number}",
        )
        route = str(row["route"])
        task = str(row["task"])
        route_tasks[f"{route}:{task}"] += 1
        sidecar.append(
            {
                "schema_version": SCHEMA_VERSION,
                "routing_token_sha256": routing_token_sha256,
                "prompt_token_sha256": prompt_token_sha256,
                "response_token_sha256": response_token_sha256,
                "row_sha256": core_hash(row),
                "prompt_sha256": prompt_hash(row),
                "route": route,
                "task": task,
                "source": row["i40_source"],
                "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
                "parent_config_sha256": PARENT_CONFIG_SHA256,
            }
        )
        maximum_tokens = max(maximum_tokens, total)
    require(len(sidecar) == TOTAL_ROWS, "sidecar row count drifted")
    return sidecar, {
        "rows": len(sidecar),
        "unique_prompt_token_hashes": len(prompt_token_hashes),
        "unique_routing_token_hashes": len(routing_token_hashes),
        "duplicate_routing_token_exposures": len(sidecar)
        - len(routing_token_hashes),
        "maximum_routing_token_exposure": max(routing_token_hashes.values()),
        "maximum_qwen3_nothink_tokens": maximum_tokens,
        "cutoff": 16_384,
        "route_tasks": dict(sorted(route_tasks.items())),
    }


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"refusing to overwrite formal output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    require(not any(path.exists() for path in (OUTPUT, SIDECAR, AUDIT)), "I40 outputs already exist")
    require(
        TOTAL_ROWS == 8_240
        and TOTAL_ROWS % 4 == 0
        and TOTAL_ROWS // 4 == OPTIMIZER_STEPS_BATCH1_ACC4,
        "static row/step contract drifted",
    )
    if args.self_test:
        require(sha256_bytes(b"i40") == sha256_bytes(b"i40"), "SHA self-test failed")
        print("[i40-builder] self-test PASS", flush=True)
        return

    i36_rows, i35_rows = verify_upstreams()
    rows, selection = select_formal_rows(i36_rows, i35_rows)
    sidecar, tokenization = tokenize_sidecar(rows)
    write_jsonl_atomic(OUTPUT, rows)
    write_jsonl_atomic(SIDECAR, sidecar)

    aggregate_tasks: Counter[str] = Counter(str(row["task"]) for row in rows)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "FORMAL_DATA_FROZEN_TRAINING_AUTHORIZED",
        "formal_training_generated": True,
        "asset_class": (
            "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,"
            "O2.Pid2Tag,O2.General; imported model annotations; M-I35-step548)"
        ),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256_file(Path(__file__)),
        "seed": SEED,
        "upstreams": {
            "i36_formal_data": {
                "path": str(I36_DATA.relative_to(ROOT)),
                "rows": 16_500,
                "sha256": EXPECTED_I36_DATA_SHA256,
                "use": "copy only 5,500 audited user_ce rows",
            },
            "i36_audit": {
                "path": str(I36_AUDIT.relative_to(ROOT)),
                "sha256": EXPECTED_I36_AUDIT_SHA256,
            },
            "i35_formal_data": {
                "path": str(I35_DATA.relative_to(ROOT)),
                "rows": 2_740,
                "sha256": EXPECTED_I35_DATA_SHA256,
                "use": "copy every row once; discard old objective; I35 KL-only",
            },
            "i35_audit": {
                "path": str(I35_AUDIT.relative_to(ROOT)),
                "sha256": EXPECTED_I35_AUDIT_SHA256,
            },
            "base_config": {
                "path": str((BASE / "config.json").relative_to(ROOT)),
                "sha256": BASE_CONFIG_SHA256,
            },
            "i35_step548_reference": {
                "path": str(PARENT.relative_to(ROOT)),
                "adapter_sha256": PARENT_ADAPTER_SHA256,
                "config_sha256": PARENT_CONFIG_SHA256,
            },
        },
        "mix": {
            "total_rows": TOTAL_ROWS,
            "optimizer_steps_batch1_acc4": OPTIMIZER_STEPS_BATCH1_ACC4,
            "fixed_seed_hash_shuffle": True,
            "routes": {
                "user_ce": {
                    "rows": ROUTE_COUNTS["user_ce"],
                    "ratio": ROUTE_COUNTS["user_ce"] / TOTAL_ROWS,
                    "by_task": USER_COUNTS,
                    "objective": "0.05 weighted answer CE + 16.0 I35 parent KL",
                },
                "retention_kl": {
                    "rows": ROUTE_COUNTS["retention_kl"],
                    "ratio": ROUTE_COUNTS["retention_kl"] / TOTAL_ROWS,
                    "by_task": RETENTION_COUNTS,
                    "objective": "16.0 I35 parent KL only; max 128 answer positions",
                    "old_i35_objective_reused": False,
                },
            },
            "aggregate_task_counts": dict(sorted(aggregate_tasks.items())),
        },
        "selection": selection,
        "tokenization": tokenization,
        "intersections": {
            "user_retention_exact_prompt": 0,
            "user_retention_mode_prompt": 0,
            "forbidden_E_rows": 0,
            "third_party_rows": 0,
        },
        "sidecar_contract": {
            "schema_version": SCHEMA_VERSION,
            "rows": TOTAL_ROWS,
            "routes": ROUTE_COUNTS,
            "route_tasks": tokenization["route_tasks"],
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
        },
        "outputs": {
            "training_data": {
                "path": str(OUTPUT.relative_to(ROOT)),
                "rows": TOTAL_ROWS,
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256_file(OUTPUT),
            },
            "sidecar": {
                "path": str(SIDECAR.relative_to(ROOT)),
                "rows": TOTAL_ROWS,
                "bytes": SIDECAR.stat().st_size,
                "sha256": sha256_file(SIDECAR),
            },
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
