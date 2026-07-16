#!/usr/bin/env python3
"""Compare paired action-1024 diagnostic reports without making a score claim.

This deliberately does not reuse the frozen I-24/I-25 v4 gate: that gate is
locked to the historical 4096-token probe, while action-1024 is only a local
structural diagnostic.  The output records paired behavior deltas and leaves
the online score direction undecided.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL = "offline-eval-action1024-diagnostic-v1"
EXPECTED_DIMS = ["action"]
EXPECTED_SAMPLING = {
    "max_tokens": 1024,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    action = report["action"]
    rows = action.get("rows", [])
    n = int(action["n"])
    if n <= 0 or len(rows) != n:
        raise ValueError("action rows must be present and match action.n")
    return {
        "n": n,
        "f1": float(action.get("f1_unrounded", action["f1"])),
        "json_ok_count": int(action["json_ok_count"]),
        "trunc_count": int(action["trunc_count"]),
        "generated_tokens_mean": statistics.mean(float(row["generated_tokens"]) for row in rows),
        "duplicate_items_mean": statistics.mean(float(row["duplicate_items"]) for row in rows),
        "max_repeat_p95": int(action["max_repeat_p95"]),
        "repeat_ge_20_count": sum(int(row["max_repeat"]) >= 20 for row in rows),
    }


def paired_deltas(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    parent_rows = parent["action"]["rows"]
    candidate_rows = candidate["action"]["rows"]
    deltas = [
        {
            "row_sha256": p["row_sha256"],
            "f1_delta": float(c["f1"]) - float(p["f1"]),
            "json_transition": f"{int(bool(p['json_ok']))}->{int(bool(c['json_ok']))}",
            "trunc_transition": f"{int(bool(p['truncated']))}->{int(bool(c['truncated']))}",
            "generated_tokens_delta": int(c["generated_tokens"]) - int(p["generated_tokens"]),
            "duplicate_items_delta": int(c["duplicate_items"]) - int(p["duplicate_items"]),
            "max_repeat_delta": int(c["max_repeat"]) - int(p["max_repeat"]),
        }
        for p, c in zip(parent_rows, candidate_rows)
    ]
    return {
        "f1_improved_rows": sum(row["f1_delta"] > 0 for row in deltas),
        "f1_degraded_rows": sum(row["f1_delta"] < 0 for row in deltas),
        "json_gained_rows": sum(row["json_transition"] == "0->1" for row in deltas),
        "json_lost_rows": sum(row["json_transition"] == "1->0" for row in deltas),
        "trunc_recovered_rows": sum(row["trunc_transition"] == "1->0" for row in deltas),
        "new_trunc_rows": sum(row["trunc_transition"] == "0->1" for row in deltas),
        "repeat_ge_20_gained_rows": sum(
            int(c["max_repeat"]) >= 20 and int(p["max_repeat"]) < 20
            for p, c in zip(parent_rows, candidate_rows)
        ),
        "repeat_ge_20_recovered_rows": sum(
            int(p["max_repeat"]) >= 20 and int(c["max_repeat"]) < 20
            for p, c in zip(parent_rows, candidate_rows)
        ),
        "rows": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    parent = load(args.parent)
    candidate = load(args.candidate)
    parent_selection = parent.get("action_selection", {})
    candidate_selection = candidate.get("action_selection", {})
    parent_rows = parent.get("action", {}).get("rows", [])
    candidate_rows = candidate.get("action", {}).get("rows", [])

    identity_checks = {
        "protocol": parent.get("protocol_version") == candidate.get("protocol_version") == EXPECTED_PROTOCOL,
        "action_sampling": (
            parent.get("sampling", {}).get("action_topic")
            == candidate.get("sampling", {}).get("action_topic")
            == EXPECTED_SAMPLING
        ),
        "action_only": parent.get("evaluated_dims") == candidate.get("evaluated_dims") == EXPECTED_DIMS,
        "not_full_platform_mirror": (
            parent.get("full_current_platform_mirror") is False
            and candidate.get("full_current_platform_mirror") is False
        ),
        "base_model": bool(parent.get("model")) and parent.get("model") == candidate.get("model"),
        "generation_seed": (
            parent.get("generation_seed") == candidate.get("generation_seed") == 42
        ),
        "think_suffix": parent.get("think_suffix") == candidate.get("think_suffix") == "keep",
        "selection_method": (
            parent_selection.get("method")
            == candidate_selection.get("method")
            == "canonical_json_sha256_ascending"
        ),
        "selection_manifest": (
            bool(parent_selection.get("manifest_sha256"))
            and parent_selection.get("manifest_sha256")
            == candidate_selection.get("manifest_sha256")
        ),
        "selection_source": (
            bool(parent_selection.get("source_sha256"))
            and parent_selection.get("source_sha256")
            == candidate_selection.get("source_sha256")
        ),
        "row_order": (
            [row.get("row_sha256") for row in parent_rows]
            == [row.get("row_sha256") for row in candidate_rows]
            == parent_selection.get("selected_row_sha256")
            == candidate_selection.get("selected_row_sha256")
        ),
        "row_count": (
            len(parent_rows)
            == len(candidate_rows)
            == int(parent.get("action", {}).get("n", -1))
            == int(candidate.get("action", {}).get("n", -2))
        ),
    }
    if not all(identity_checks.values()):
        failed = [name for name, passed in identity_checks.items() if not passed]
        raise ValueError(f"paired action1024 protocol mismatch: {', '.join(failed)}")

    parent_metrics = exact_metrics(parent)
    candidate_metrics = exact_metrics(candidate)
    aggregate_delta = {
        key: candidate_metrics[key] - parent_metrics[key]
        for key in parent_metrics
        if key != "n"
    }
    result = {
        "status": "COMPLETE_DIAGNOSTIC_NOT_A_SCORE_ESTIMATE",
        "score_direction": "ABSTAIN",
        "hard_gate_pass": None,
        "parent_report": str(args.parent.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "identity_checks": identity_checks,
        "metrics": {
            "parent": parent_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_parent": aggregate_delta,
        },
        "paired": paired_deltas(parent, candidate),
        "interpretation_boundary": (
            "These local action metrics are uncalibrated for leaderboard direction. "
            "Use them only to identify obvious format, truncation, or repetition collapse."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
