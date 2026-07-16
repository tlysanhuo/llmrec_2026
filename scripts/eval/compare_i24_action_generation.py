#!/usr/bin/env python3
"""Compare paired I-24 action-generation reports against the frozen numeric gate.

This CPU-only checker does not score or generate examples.  It verifies that
both offline_eval reports used the same stable 32-row manifest and decoding
protocol before applying the three preregistered numeric generation deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL = "offline-eval-v4-platform-params"
EXPECTED_ROWS = 32
EXPECTED_SELECTION = "canonical_json_sha256_ascending"
EXPECTED_SAMPLING = {
    "max_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_metrics(report: dict[str, Any]) -> dict[str, float]:
    action = report["action"]
    n = int(action["n"])
    if n <= 0:
        raise ValueError("action.n must be positive")
    return {
        "f1": float(action.get("f1_unrounded", action["f1"])),
        "json_ok": float(action.get("json_ok_count", action["json_ok"] * n)) / n,
        "trunc_rate": float(action.get("trunc_count", action["trunc_rate"] * n)) / n,
        "max_repeat_p95": float(action["max_repeat_p95"]),
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
    parent_action = parent.get("action", {})
    candidate_action = candidate.get("action", {})

    identity_checks = {
        "protocol": parent.get("protocol_version") == candidate.get("protocol_version") == EXPECTED_PROTOCOL,
        "action_sampling": (
            parent.get("sampling", {}).get("action_topic")
            == candidate.get("sampling", {}).get("action_topic")
            == EXPECTED_SAMPLING
        ),
        "think_suffix": parent.get("think_suffix") == candidate.get("think_suffix") == "keep",
        "selection_method": (
            parent_selection.get("method")
            == candidate_selection.get("method")
            == EXPECTED_SELECTION
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
        "row_count": parent_action.get("n") == candidate_action.get("n") == EXPECTED_ROWS,
    }
    if not all(identity_checks.values()):
        failed = [name for name, passed in identity_checks.items() if not passed]
        raise ValueError(f"paired generation protocol mismatch: {', '.join(failed)}")

    parent_metrics = exact_metrics(parent)
    candidate_metrics = exact_metrics(candidate)
    deltas = {
        name: candidate_metrics[name] - parent_metrics[name]
        for name in parent_metrics
    }
    numeric_checks = {
        "f1_delta_min_0": deltas["f1"] >= 0.0,
        "json_ok_delta_min_0": deltas["json_ok"] >= 0.0,
        "trunc_rate_delta_max_0": deltas["trunc_rate"] <= 0.0,
    }
    report = {
        "status": "COMPLETE_NOT_A_SCORE_ESTIMATE",
        "parent_report": str(args.parent.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "identity_checks": identity_checks,
        "metrics": {"parent": parent_metrics, "candidate": candidate_metrics, "delta": deltas},
        "numeric_checks": numeric_checks,
        "all_preregistered_numeric_generation_requirements_pass": all(numeric_checks.values()),
        "diagnostic_boundary": (
            "max_repeat_p95 is recorded but has no numeric hard threshold in the frozen gate; "
            "placeholder/format collapse and itemic breakage require the separate structural precheck."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
