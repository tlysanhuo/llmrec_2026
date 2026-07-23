#!/usr/bin/env python3
"""Frozen teacher-forced mechanism gate for the unique I-39 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs/evaluation/i39_i35_ab_firstdiv_material_checkpoint_gate_v1.json"
)
CONFIG_SHA256 = "89737747e68161a68607d06fdd8767cbcae9a31373252bdc7e6a764df32ed504"
CANDIDATE = (
    ROOT
    / "submissions/i39_i35_userab_firstdiv_retkl_r120_step640_platform"
)
RESIDUAL = ROOT / "checkpoints/i39_i35_userab_firstdiv_retkl_r8_v1"
PACKAGE_AUDIT = (
    ROOT
    / "logs/package/i39_i35_userab_firstdiv_retkl_r120_step640_audit.json"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SID = re.compile(
    r"^<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)
FOCUS = {"a_firstdiv": 0, "b_firstdiv": 1, "c_firstdiv": 2}
VIDEO_ID = 176245
A_RANGE = (151669, 159860)
B_RANGE = (159861, 168052)
C_RANGE = (168053, 176244)
CLOSE_THINK_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def locked_path(entry: Mapping[str, Any], *, key: str = "path") -> Path:
    path = Path(str(entry.get(key) or ""))
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    require(len(rows) == expected_rows, f"row count drifted: {path}: {len(rows)}")
    return rows


def mode_prompt_hash(row: Mapping[str, Any]) -> str:
    user_input = re.sub(
        r"/(?:no_)?think\s*$", "", str(row["input"]).rstrip()
    ).rstrip()
    return digest([row["instruction"], user_input, row["history"]])


def classify(ledger: Mapping[str, Any]) -> str:
    parent = ledger.get("parent")
    require(isinstance(parent, dict), "ledger parent metadata is missing")
    if not parent.get("a_hit"):
        return "a_firstdiv"
    if not parent.get("ab_hit"):
        return "b_firstdiv"
    if not parent.get("full_gold_hit"):
        return "c_firstdiv"
    return "full_anchor"


def first_divergence(gold: Sequence[int], negative: Sequence[int]) -> int:
    require(len(gold) == len(negative) == 3, "ABC triples must have length three")
    differences = [
        index for index, (left, right) in enumerate(zip(gold, negative)) if left != right
    ]
    require(bool(differences), "hard negative is identical to gold")
    return differences[0]


def validate_abc(tokens: Any, field: str) -> list[int]:
    require(
        isinstance(tokens, list)
        and len(tokens) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) for value in tokens),
        f"{field} is not a three-token integer ABC",
    )
    result = [int(value) for value in tokens]
    for value, (lower, upper) in zip(result, (A_RANGE, B_RANGE, C_RANGE)):
        require(lower <= value <= upper, f"{field} token is outside its SID range")
    return result


def load_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(CONFIG.is_file() and sha256(CONFIG) == CONFIG_SHA256, "I39 gate config drifted")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(
        config.get("status") == "RESULT_PREDEFINED_HASH_FROZEN_BEFORE_FORMAL_TRAINING",
        "I39 gate is not result-predefined",
    )

    base = config["immutable_base"]
    parent = config["parent"]
    holdout = config["frozen_holdout"]
    ledgers = config["frozen_beam_ledger"]
    locks = (
        (locked_path(base) / "config.json", base["config_sha256"]),
        (locked_path(parent) / "adapter_model.safetensors", parent["adapter_sha256"]),
        (
            locked_path(parent) / "adapter_config.json",
            parent["adapter_config_sha256"],
        ),
        (locked_path(holdout), holdout["sha256"]),
        (locked_path(ledgers["train"]), ledgers["train"]["sha256"]),
        (locked_path(ledgers["dev"]), ledgers["dev"]["sha256"]),
        (locked_path(ledgers["audit"]), ledgers["audit"]["sha256"]),
    )
    for path, expected in locks:
        require(path.is_file() and sha256(path) == expected, f"locked input drifted: {path}")

    beam_audit = json.loads(locked_path(ledgers["audit"]).read_text(encoding="utf-8"))
    require(
        beam_audit.get("schema_version") == ledgers["audit"]["schema_version"]
        and beam_audit.get("status") == "complete",
        "I39 Beam audit schema/status drifted",
    )
    require(
        beam_audit.get("runtime", {}).get("beam_width") == ledgers["audit"]["beam_width"]
        and beam_audit.get("runtime", {}).get("single_parent_request") is True,
        "I39 Beam runtime contract drifted",
    )
    require(
        beam_audit.get("parent", {}).get("adapter_model_sha256")
        == parent["adapter_sha256"],
        "I39 Beam parent drifted",
    )
    for name in ("train", "dev"):
        audit_key = f"{name}_ledger"
        require(
            beam_audit.get("outputs", {}).get(audit_key, {}).get("sha256")
            == ledgers[name]["sha256"],
            f"I39 Beam audit {audit_key} hash drifted",
        )

    ledger_rows: list[dict[str, Any]] = []
    for name in ("train", "dev"):
        ledger_rows.extend(
            load_jsonl(locked_path(ledgers[name]), int(ledgers[name]["rows"]))
        )
    ledger_by_key: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        key = str(row.get("row_sha256") or "")
        require(HEX64.fullmatch(key) is not None and key not in ledger_by_key, "bad ledger key")
        require(
            row.get("schema_version") == "i39-userab-video-beam64-ledger-v1"
            and row.get("parent_adapter_sha256") == parent["adapter_sha256"],
            f"I39 ledger schema/parent drifted: {key}",
        )
        ledger_by_key[key] = row
    require(len(ledger_by_key) == 3072, "I39 combined ledger key count drifted")

    gate_rows = load_jsonl(locked_path(holdout), int(holdout["rows"]))
    joined: list[dict[str, Any]] = []
    objective_counts: Counter[str] = Counter()
    by_ab: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gate in gate_rows:
        require(
            gate.get("route") == holdout["route"] and gate.get("task") == holdout["task"],
            "I39 holdout route/task drifted",
        )
        key = str(gate.get("i39_source_row_sha256") or "")
        ledger = ledger_by_key.get(key)
        require(ledger is not None, f"I39 holdout key is absent from Beam ledger: {key}")
        core = {
            "instruction": str(gate.get("instruction") or ""),
            "input": str(gate.get("input") or ""),
            "output": str(gate.get("output") or ""),
            "history": gate.get("history") or [],
        }
        require(digest(core) == key, f"I39 holdout core hash drifted: {key}")
        require(
            ledger.get("source_prompt_sha256")
            == digest([core["instruction"], core["input"], core["history"]])
            and ledger.get("source_mode_prompt_sha256") == mode_prompt_hash(core),
            f"I39 holdout/ledger prompt hash drifted: {key}",
        )
        match = SID.fullmatch(str(gate.get("i39_gold_sid") or ""))
        require(match is not None, f"I39 gate gold SID is invalid: {key}")
        sid_abc = list(match.groups())
        require(ledger.get("gold_abc") == sid_abc, f"I39 gate/ledger gold SID drifted: {key}")
        gold_tokens = ledger.get("gold_tokens")
        require(
            isinstance(gold_tokens, list)
            and len(gold_tokens) == 5
            and gold_tokens[0] == VIDEO_ID
            and gold_tokens[-1] == EOS_ID,
            f"I39 ledger gold body drifted: {key}",
        )
        gold_abc = validate_abc(list(gold_tokens[1:4]), "gold_abc")
        objective = str(gate.get("i39_objective") or "")
        require(objective == classify(ledger), f"I39 objective drifted: {key}")
        focus = FOCUS.get(objective)
        negatives: list[dict[str, Any]] = []
        values = ledger.get("hard_negatives")
        require(isinstance(values, list), f"I39 hard negatives are invalid: {key}")
        for value in values:
            require(isinstance(value, dict), f"I39 hard negative is not an object: {key}")
            if "teacher_score" in value:
                require(
                    value["teacher_score"] == value.get("parent_score"),
                    f"I39 compatibility teacher score is not the parent score: {key}",
                )
            if focus is None or value.get("first_divergence") != focus:
                continue
            tokens = validate_abc(value.get("tokens"), "hard_negative.tokens")
            require(
                first_divergence(gold_abc, tokens) == focus,
                f"I39 hard negative focus drifted: {key}",
            )
            negatives.append({"tokens": tokens, "first_divergence": focus})
        if focus is not None:
            require(1 <= len(negatives) <= 4, f"I39 focus-negative count drifted: {key}")
        prefix = str(gate.get("i39_prefix_group") or "")
        require(prefix == f"video:{sid_abc[0]}:{sid_abc[1]}", f"I39 AB key drifted: {key}")
        record = {
            "gate": gate,
            "ledger": ledger,
            "source_row_sha256": key,
            "objective": objective,
            "focus": focus,
            "gold_abc": gold_abc,
            "negatives": negatives,
            "ab": prefix,
        }
        joined.append(record)
        objective_counts[objective] += 1
        by_ab[prefix].append(record)

    require(
        dict(objective_counts) == holdout["by_objective"],
        f"I39 holdout strata drifted: {dict(objective_counts)}",
    )
    require(len(by_ab) == holdout["unique_ab_groups"], "I39 gate AB count drifted")
    require(
        sum(len(values) > 1 for values in by_ab.values())
        == holdout["dual_view_ab_groups"],
        "I39 dual-view AB count drifted",
    )
    require(
        sum(sum(row["objective"] == "c_firstdiv" for row in values) >= 2 for values in by_ab.values())
        == holdout["dual_c_firstdiv_ab_groups"],
        "I39 dual-C-firstdiv AB count drifted",
    )
    return config, joined


def response_body(response_ids: Sequence[int]) -> tuple[int, int]:
    try:
        start = list(response_ids).index(CLOSE_THINK_ID) + 1
    except ValueError as error:
        raise RuntimeError("I39 gate response lacks </think>") from error
    while start < len(response_ids) and response_ids[start] in WHITESPACE_IDS:
        start += 1
    end = len(response_ids)
    while end > start and response_ids[end - 1] in WHITESPACE_IDS:
        end -= 1
    require(
        end - start == 5
        and response_ids[start] == VIDEO_ID
        and response_ids[end - 1] == EOS_ID,
        "I39 gate response is not empty-think video+A+B+C+EOS",
    )
    return start, end


def parent_to_candidate_kl(parent_logits: Any, candidate_logits: Any) -> float:
    import torch.nn.functional as functional

    require(parent_logits.shape == candidate_logits.shape, "I39 gate KL shape mismatch")
    parent_logp = functional.log_softmax(parent_logits.float(), dim=-1)
    candidate_logp = functional.log_softmax(candidate_logits.float(), dim=-1)
    value = float((parent_logp.exp() * (parent_logp - candidate_logp)).sum() / parent_logits.size(1))
    require(math.isfinite(value), "I39 gate produced non-finite KL")
    return value


def selected_logp(logits: Any, targets: Any) -> Any:
    import torch.nn.functional as functional

    return functional.log_softmax(logits.float(), dim=-1)[0].gather(
        -1, targets.view(-1, 1)
    ).squeeze(-1)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(rows), "cannot summarize an empty I39 stratum")
    token_count = 3 * len(rows)
    report: dict[str, Any] = {
        "rows": len(rows),
        "abc_tokens": token_count,
        "parent_gold_logp_mean": statistics.fmean(row["parent_gold_logp"] for row in rows),
        "candidate_gold_logp_mean": statistics.fmean(
            row["candidate_gold_logp"] for row in rows
        ),
        "gold_logp_mean_delta_vs_parent": statistics.fmean(
            row["gold_logp_delta"] for row in rows
        ),
        "gold_logp_improved_row_rate": statistics.fmean(
            row["gold_logp_delta"] > 0.0 for row in rows
        ),
        "parent_to_candidate_forward_kl_mean": statistics.fmean(
            row["parent_kl"] for row in rows
        ),
        "candidate_parent_top1_agreement": sum(row["top1_matches"] for row in rows)
        / token_count,
        "parent_gold_top1_token_rate": sum(row["parent_gold_top1"] for row in rows)
        / token_count,
        "candidate_gold_top1_token_rate": sum(
            row["candidate_gold_top1"] for row in rows
        )
        / token_count,
        "gold_top1_token_correct_delta": sum(
            row["candidate_gold_top1"] - row["parent_gold_top1"] for row in rows
        ),
    }
    focus_rows = [row for row in rows if row.get("focus") is not None]
    if focus_rows:
        report.update(
            {
                "focus_rows": len(focus_rows),
                "parent_focus_gold_logp_mean": statistics.fmean(
                    row["parent_focus_gold_logp"] for row in focus_rows
                ),
                "candidate_focus_gold_logp_mean": statistics.fmean(
                    row["candidate_focus_gold_logp"] for row in focus_rows
                ),
                "focus_gold_logp_mean_delta_vs_parent": statistics.fmean(
                    row["focus_gold_logp_delta"] for row in focus_rows
                ),
                "parent_focus_margin_mean": statistics.fmean(
                    row["parent_focus_margin"] for row in focus_rows
                ),
                "candidate_focus_margin_mean": statistics.fmean(
                    row["candidate_focus_margin"] for row in focus_rows
                ),
                "focus_margin_mean_delta_vs_parent": statistics.fmean(
                    row["focus_margin_delta"] for row in focus_rows
                ),
                "focus_margin_improved_row_rate": statistics.fmean(
                    row["focus_margin_delta"] > 0.0 for row in focus_rows
                ),
            }
        )
    return report


def dual_view_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_ab: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ab[str(row["ab"])].append(row)
    groups = [values for values in by_ab.values() if len(values) > 1]
    focus_groups = [
        [row for row in values if row.get("focus") is not None]
        for values in groups
    ]
    focus_groups = [values for values in focus_groups if values]
    return {
        "groups": len(groups),
        "rows": sum(len(values) for values in groups),
        "both_c_firstdiv_groups": sum(
            sum(row["objective"] == "c_firstdiv" for row in values) >= 2
            for values in groups
        ),
        "group_mean_gold_logp_delta_vs_parent": statistics.fmean(
            statistics.fmean(row["gold_logp_delta"] for row in values)
            for values in groups
        ),
        "group_mean_parent_to_candidate_forward_kl": statistics.fmean(
            statistics.fmean(row["parent_kl"] for row in values) for values in groups
        ),
        "group_mean_candidate_parent_top1_agreement": statistics.fmean(
            sum(row["top1_matches"] for row in values) / (3 * len(values))
            for values in groups
        ),
        "focus_group_mean_margin_delta_vs_parent": statistics.fmean(
            statistics.fmean(row["focus_margin_delta"] for row in values)
            for values in focus_groups
        ),
        "focus_margin_improved_group_rate": statistics.fmean(
            statistics.fmean(row["focus_margin_delta"] for row in values) > 0.0
            for values in focus_groups
        ),
    }


def hard_checks(
    config: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    by_objective: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    thresholds = config["hard_pass_thresholds"]
    firstdiv = thresholds["each_firstdiv_stratum"]
    checks: dict[str, bool] = {}
    for objective in firstdiv["strata"]:
        report = by_objective[objective]
        checks[f"{objective}_focus_margin_delta"] = (
            report["focus_margin_mean_delta_vs_parent"]
            >= firstdiv["focus_gold_vs_hard_negative_margin_mean_delta_vs_parent_min"]
        )
        checks[f"{objective}_focus_margin_improved_rate"] = (
            report["focus_margin_improved_row_rate"]
            >= firstdiv["focus_margin_improved_row_rate_min"]
        )
        checks[f"{objective}_focus_gold_logp_delta"] = (
            report["focus_gold_logp_mean_delta_vs_parent"]
            >= firstdiv["focus_gold_logp_mean_delta_vs_parent_min"]
        )
    anchor = thresholds["full_anchor"]
    anchor_report = by_objective["full_anchor"]
    checks["full_anchor_top1_agreement"] = (
        anchor_report["candidate_parent_top1_agreement"]
        >= anchor["candidate_parent_top1_agreement_min"]
    )
    checks["full_anchor_parent_kl"] = (
        anchor_report["parent_to_candidate_forward_kl_mean"]
        <= anchor["parent_to_candidate_forward_kl_mean_max"]
    )
    checks["full_anchor_gold_logp_delta"] = (
        anchor_report["gold_logp_mean_delta_vs_parent"]
        >= anchor["gold_logp_mean_delta_vs_parent_min"]
    )
    checks["aggregate_parent_kl"] = (
        aggregate["parent_to_candidate_forward_kl_mean"]
        <= thresholds["aggregate_all_rows"]["parent_to_candidate_forward_kl_mean_max"]
    )
    return checks


def validate_candidate(config: Mapping[str, Any], candidate: Path) -> dict[str, Any]:
    candidate = candidate.resolve()
    require(
        candidate == CANDIDATE.resolve(),
        f"I39 gate accepts only the unique full package: {CANDIDATE}",
    )
    require(candidate.is_dir(), f"I39 candidate directory is missing: {candidate}")
    required = set(config["candidate_rule"]["required_files"])
    observed = {path.name for path in candidate.iterdir() if path.is_file()}
    require(observed == required, f"I39 candidate must be an exact two-file package: {observed}")
    candidate_config = candidate / "adapter_config.json"
    value = json.loads(candidate_config.read_text(encoding="utf-8"))
    rank_alpha = [int(value.get("r", -1)), int(value.get("lora_alpha", -1))]
    require(
        rank_alpha == config["candidate_rule"]["required_rank_alpha"],
        f"I39 candidate rank/alpha drifted: {rank_alpha}",
    )
    require(PACKAGE_AUDIT.is_file(), f"I39 package audit is missing: {PACKAGE_AUDIT}")
    package_audit = json.loads(PACKAGE_AUDIT.read_text(encoding="utf-8"))
    parent = package_audit.get("parent") or {}
    residual = package_audit.get("residual") or {}
    combined = package_audit.get("combined") or {}
    parent_config = config["parent"]
    require(
        Path(str(parent.get("path") or "")).resolve()
        == locked_path(parent_config).resolve()
        and [parent.get("rank"), parent.get("alpha")]
        == parent_config["rank_alpha"]
        and parent.get("multiplier") == 1.0
        and parent.get("adapter_sha256") == parent_config["adapter_sha256"],
        "I39 package audit parent provenance drifted",
    )
    require(
        Path(str(residual.get("path") or "")).resolve() == RESIDUAL.resolve()
        and [residual.get("rank"), residual.get("alpha")] == [8, 8]
        and residual.get("multiplier") == 1.0,
        "I39 package audit residual provenance drifted",
    )
    residual_model = RESIDUAL / "adapter_model.safetensors"
    residual_config = RESIDUAL / "adapter_config.json"
    require(
        residual_model.is_file()
        and residual_config.is_file()
        and sha256(residual_model) == residual.get("adapter_sha256"),
        "I39 package audit does not bind the final residual",
    )
    residual_config_value = json.loads(residual_config.read_text(encoding="utf-8"))
    require(
        [
            int(residual_config_value.get("r", -1)),
            int(residual_config_value.get("lora_alpha", -1)),
        ]
        == [8, 8],
        "I39 final residual rank/alpha drifted",
    )
    adapter_sha = sha256(candidate / "adapter_model.safetensors")
    config_sha = sha256(candidate_config)
    require(
        Path(str(combined.get("path") or "")).resolve() == candidate
        and [combined.get("rank"), combined.get("alpha")] == rank_alpha
        and combined.get("tensor_count") == 392
        and combined.get("adapter_sha256") == adapter_sha
        and combined.get("config_sha256") == config_sha
        and package_audit.get("identity")
        == "delta_combined = 1 * delta_parent + 1 * delta_residual",
        "I39 package audit does not bind the exact full candidate",
    )
    return {
        "path": str(candidate),
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": config_sha,
        "rank_alpha": rank_alpha,
        "residual_adapter_sha256": residual["adapter_sha256"],
        "package_audit_path": str(PACKAGE_AUDIT),
        "package_audit_sha256": sha256(PACKAGE_AUDIT),
        "exact_additivity_verified": True,
    }


def run_self_test() -> None:
    import torch

    torch.manual_seed(39)
    parent = torch.randn(1, 3, 31)
    candidate = parent.clone()
    require(parent_to_candidate_kl(parent, candidate) < 1e-7, "identity KL failed")
    candidate[0, 1, 7] += 1.0
    require(parent_to_candidate_kl(parent, candidate) > 0.0, "positive KL failed")
    assert classify({"parent": {"a_hit": False, "ab_hit": False, "full_gold_hit": False}}) == "a_firstdiv"
    assert classify({"parent": {"a_hit": True, "ab_hit": True, "full_gold_hit": True}}) == "full_anchor"
    print("[i39-gate] self-test PASS", flush=True)


def evaluate(
    config: dict[str, Any],
    joined: list[dict[str, Any]],
    candidate_path: Path,
    out: Path,
    gpu: str,
) -> None:
    require(re.fullmatch(r"\d+", gpu) is not None, "--gpu must name exactly one numeric GPU")
    candidate_meta = validate_candidate(config, candidate_path)
    require(not out.exists(), f"I39 gate refuses to overwrite: {out}")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    base = locked_path(config["immutable_base"])
    parent_path = locked_path(config["parent"])
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    model = AutoModelForCausalLM.from_pretrained(
        base,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(
        model,
        parent_path,
        adapter_name="parent",
        is_trainable=False,
        low_cpu_mem_usage=True,
    ).eval()
    model.load_adapter(
        candidate_path.resolve(),
        adapter_name="candidate",
        is_trainable=False,
        low_cpu_mem_usage=True,
    )

    paired: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for index, record in enumerate(joined, 1):
            row = record["gate"]
            prompt_ids, response_ids = template.encode_oneturn(
                tokenizer,
                [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ],
                row["instruction"],
                None,
            )
            body_start, _body_end = response_body(response_ids)
            require(
                list(response_ids[body_start + 1 : body_start + 4])
                == record["gold_abc"],
                "I39 runtime gold tokens disagree with ledger",
            )
            positions = (
                torch.arange(body_start, body_start + 3, device="cuda")
                + len(prompt_ids)
            )
            targets = torch.tensor(record["gold_abc"], device="cuda")
            input_ids = torch.tensor(
                [prompt_ids + response_ids], device="cuda", dtype=torch.long
            )
            attention_mask = torch.ones_like(input_ids)
            logits: dict[str, Any] = {}
            for name in ("parent", "candidate"):
                model.set_adapter(name)
                logits[name] = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=positions,
                ).logits.detach()
                require(
                    tuple(logits[name].shape[:2]) == (1, 3),
                    "I39 runtime did not return three ABC logits",
                )
            parent_logp = selected_logp(logits["parent"], targets)
            candidate_logp = selected_logp(logits["candidate"], targets)
            parent_top1 = logits["parent"].argmax(-1)[0]
            candidate_top1 = logits["candidate"].argmax(-1)[0]
            values: dict[str, Any] = {
                "source_row_sha256": record["source_row_sha256"],
                "objective": record["objective"],
                "ab": record["ab"],
                "focus": record["focus"],
                "parent_gold_logp": float(parent_logp.mean()),
                "candidate_gold_logp": float(candidate_logp.mean()),
                "gold_logp_delta": float(candidate_logp.mean() - parent_logp.mean()),
                "parent_kl": parent_to_candidate_kl(
                    logits["parent"], logits["candidate"]
                ),
                "top1_matches": int(candidate_top1.eq(parent_top1).sum()),
                "parent_gold_top1": int(parent_top1.eq(targets).sum()),
                "candidate_gold_top1": int(candidate_top1.eq(targets).sum()),
            }
            focus = record["focus"]
            if focus is not None:
                negative_tokens = torch.tensor(
                    [negative["tokens"][focus] for negative in record["negatives"]],
                    device="cuda",
                )
                parent_full = torch.log_softmax(logits["parent"].float(), dim=-1)[0, focus]
                candidate_full = torch.log_softmax(logits["candidate"].float(), dim=-1)[0, focus]
                parent_margin = (
                    parent_full[targets[focus]] - parent_full[negative_tokens]
                ).mean()
                candidate_margin = (
                    candidate_full[targets[focus]] - candidate_full[negative_tokens]
                ).mean()
                values.update(
                    {
                        "parent_focus_gold_logp": float(parent_logp[focus]),
                        "candidate_focus_gold_logp": float(candidate_logp[focus]),
                        "focus_gold_logp_delta": float(
                            candidate_logp[focus] - parent_logp[focus]
                        ),
                        "parent_focus_margin": float(parent_margin),
                        "candidate_focus_margin": float(candidate_margin),
                        "focus_margin_delta": float(candidate_margin - parent_margin),
                    }
                )
            paired.append(values)
            if index % 32 == 0 or index == len(joined):
                print(
                    f"[i39-gate] {index}/{len(joined)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    aggregate = summarize(paired)
    by_objective = {
        objective: summarize([row for row in paired if row["objective"] == objective])
        for objective in (*FOCUS, "full_anchor")
    }
    checks = hard_checks(config, aggregate, by_objective)
    checks["error_count"] = 0 <= config["hard_pass_thresholds"]["error_count_max"]
    post_candidate_meta = validate_candidate(config, candidate_path)
    require(
        post_candidate_meta == candidate_meta,
        "I39 candidate or package audit changed during evaluation",
    )
    post_config, post_joined = load_contract()
    require(
        post_config == config
        and [row["source_row_sha256"] for row in post_joined]
        == [row["source_row_sha256"] for row in joined],
        "I39 frozen gate inputs changed during evaluation",
    )
    report = {
        "status": "COMPLETE_NOT_AN_ONLINE_SCORE_ESTIMATE",
        "gate_config": {
            "path": str(CONFIG),
            "sha256": CONFIG_SHA256,
            "result_predefined": True,
        },
        "locked_inputs": {
            "holdout": config["frozen_holdout"],
            "beam_ledger": config["frozen_beam_ledger"],
            "parent": config["parent"],
        },
        "candidate": candidate_meta,
        "aggregate_all_rows": aggregate,
        "by_objective": by_objective,
        "dual_view_ab_groups": dual_view_summary(paired),
        "paired_row_metrics_sha256": digest(paired),
        "error_count": 0,
        "hard_checks": checks,
        "teacher_forced_pass": all(checks.values()),
        "next_action": config["next_action"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(out)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    config, joined = load_contract()
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASS_NO_MODEL_EVALUATION",
                    "config_sha256": CONFIG_SHA256,
                    "rows": len(joined),
                    "by_objective": dict(
                        Counter(row["objective"] for row in joined)
                    ),
                    "unique_ab_groups": len({row["ab"] for row in joined}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.candidate is None or args.out is None:
        parser.error("--candidate and --out are required unless using --self-test/--preflight")
    evaluate(config, joined, args.candidate, args.out, args.gpu)


if __name__ == "__main__":
    main()
