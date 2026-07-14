#!/usr/bin/env python3
"""Audit and apply the selective checkpoint score-direction gate.

This tool deliberately separates three outcomes:

* UP/DOWN: a calibrated direction claim.
* ABSTAIN: insufficient calibrated evidence.
* REJECT: a structural safety fuse fired; this is not a score prediction.

Certification is prospective. Retrospective probes and multiple pairs made from
the same checkpoints must not be counted as independent successes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/evaluation/score_direction_gate_v1.json"
DEFAULT_LEDGER = ROOT / "configs/evaluation/score_direction_ledger_v1.jsonl"
DIRECTION_DECISIONS = {"UP", "DOWN"}
ALL_DECISIONS = DIRECTION_DECISIONS | {"ABSTAIN", "REJECT"}


class GateError(ValueError):
    """Raised for a malformed or statistically invalid gate input."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GateError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise GateError(f"expected an object at {path}:{line_number}")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def parse_utc(value: str, field: str) -> datetime:
    if not isinstance(value, str):
        raise GateError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError(f"invalid timestamp for {field}: {value}") from exc
    if parsed.tzinfo is None:
        raise GateError(f"{field} must include a timezone: {value}")
    return parsed


def clopper_pearson_lower(successes: int, trials: int, confidence: float) -> float | None:
    """One-sided exact binomial lower confidence bound."""
    if trials == 0:
        return None
    if not 0 <= successes <= trials:
        raise GateError("successes must be between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise GateError("confidence must lie in (0, 1)")
    if successes == 0:
        return 0.0
    try:
        from scipy.stats import beta
    except ImportError as exc:
        raise GateError("scipy is required for the exact confidence bound") from exc
    alpha = 1.0 - confidence
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def minimum_trials(max_errors: int, target: float, confidence: float) -> tuple[int, float]:
    if max_errors < 0:
        raise GateError("max_errors must be non-negative")
    for trials in range(max_errors + 1, 100000):
        successes = trials - max_errors
        lower = clopper_pearson_lower(successes, trials, confidence)
        if lower is not None and lower >= target:
            return trials, lower
    raise GateError("sample-size search did not converge")


def validate_ledger_row(row: dict[str, Any]) -> None:
    line = row.get("_line_number", "?")
    required = [
        "experiment_id",
        "baseline_id",
        "family_id",
        "independence_unit",
        "protocol_version",
        "prospective",
        "eligible",
        "decision_at_utc",
        "decision",
    ]
    missing = [field for field in required if field not in row]
    if missing:
        raise GateError(f"ledger line {line} missing fields: {', '.join(missing)}")
    if row["decision"] not in ALL_DECISIONS:
        raise GateError(f"ledger line {line} has invalid decision {row['decision']!r}")
    decision_at = parse_utc(row["decision_at_utc"], f"line {line} decision_at_utc")
    outcome = row.get("outcome")
    if outcome is not None:
        if not isinstance(outcome, dict):
            raise GateError(f"ledger line {line} outcome must be an object or null")
        for field in ("observed_at_utc", "candidate_score", "baseline_score"):
            if field not in outcome:
                raise GateError(f"ledger line {line} outcome missing {field}")
        observed_at = parse_utc(outcome["observed_at_utc"], f"line {line} observed_at_utc")
        if decision_at >= observed_at:
            raise GateError(f"ledger line {line} decision is not earlier than its outcome")
        for field in ("candidate_score", "baseline_score"):
            value = outcome[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise GateError(f"ledger line {line} outcome {field} must be finite")


def direction_is_correct(decision: str, delta: float, threshold: float = 0.0) -> bool:
    if decision == "UP":
        return delta > threshold
    if decision == "DOWN":
        return delta < -threshold
    raise GateError(f"{decision} is not a direction decision")


def audit(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        validate_ledger_row(row)

    target = config["target"]
    protocol = config["protocol_version"]
    confidence = float(target["one_sided_confidence"])
    target_accuracy = float(target["conditional_accuracy"])
    practical_delta = float(target["practical_delta"])
    minimum_coverage = float(target["minimum_coverage"])

    eligible_outcomes = [
        row
        for row in rows
        if row["protocol_version"] == protocol
        and row["prospective"] is True
        and row["eligible"] is True
        and row.get("outcome") is not None
    ]
    direction_rows = [row for row in eligible_outcomes if row["decision"] in DIRECTION_DECISIONS]

    units = [row["independence_unit"] for row in direction_rows]
    duplicate_units = sorted(unit for unit, count in Counter(units).items() if count > 1)
    if duplicate_units:
        raise GateError(
            "direction decisions reuse independence units: " + ", ".join(duplicate_units)
        )
    families = [row["family_id"] for row in direction_rows]
    duplicate_families = sorted(family for family, count in Counter(families).items() if count > 1)
    if duplicate_families:
        raise GateError(
            "direction decisions reuse experiment-family clusters: "
            + ", ".join(duplicate_families)
        )

    class_results: dict[str, Any] = {}
    for decision in ("UP", "DOWN"):
        selected = [row for row in direction_rows if row["decision"] == decision]
        strict_successes = 0
        practical_successes = 0
        outcomes = []
        for row in selected:
            outcome = row["outcome"]
            delta = float(outcome["candidate_score"]) - float(outcome["baseline_score"])
            strict = direction_is_correct(decision, delta)
            practical = direction_is_correct(decision, delta, practical_delta)
            strict_successes += strict
            practical_successes += practical
            outcomes.append(
                {
                    "experiment_id": row["experiment_id"],
                    "independence_unit": row["independence_unit"],
                    "delta": round(delta, 6),
                    "strict_correct": strict,
                    "practical_correct": practical,
                }
            )
        trials = len(selected)
        lower = clopper_pearson_lower(strict_successes, trials, confidence)
        class_results[decision] = {
            "trials": trials,
            "strict_successes": strict_successes,
            "strict_accuracy": strict_successes / trials if trials else None,
            "strict_lower_confidence_bound": lower,
            "practical_successes": practical_successes,
            "practical_accuracy": practical_successes / trials if trials else None,
            "certified": lower is not None and lower >= target_accuracy,
            "outcomes": outcomes,
        }

    coverage = len(direction_rows) / len(eligible_outcomes) if eligible_outcomes else 0.0
    required_classes = target.get("required_classes", ["UP", "DOWN"])
    certified = (
        coverage >= minimum_coverage
        and all(class_results[name]["certified"] for name in required_classes)
    )
    requirements = []
    for errors in range(3):
        trials, lower = minimum_trials(errors, target_accuracy, confidence)
        requirements.append(
            {"max_errors": errors, "minimum_trials": trials, "lower_bound": lower}
        )
    zero_error_per_class = requirements[0]["minimum_trials"]
    minimum_eligible_zero_error = math.ceil(
        zero_error_per_class * len(target.get("required_classes", ["UP", "DOWN"]))
        / minimum_coverage
    )

    return {
        "schema_version": config["schema_version"],
        "protocol_version": protocol,
        "estimand": target["estimand"],
        "status": "CERTIFIED" if certified else "NOT_CERTIFIED",
        "target_accuracy": target_accuracy,
        "one_sided_confidence": confidence,
        "practical_delta": practical_delta,
        "ledger_rows": len(rows),
        "eligible_outcomes": len(eligible_outcomes),
        "direction_decisions": len(direction_rows),
        "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "classes": class_results,
        "sample_requirements": requirements,
        "minimum_eligible_outcomes_zero_error_both_classes": minimum_eligible_zero_error,
    }


def load_frozen_calibration(evidence: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    reference = evidence.get("calibration_artifact")
    if not isinstance(reference, dict):
        return None, ["frozen calibration artifact is unavailable"]
    relpath = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(relpath, str) or not isinstance(expected_hash, str):
        return None, ["calibration artifact path or SHA256 is missing"]
    path = (ROOT / relpath).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return None, ["calibration artifact must be inside the repository"]
    if not path.is_file():
        return None, ["calibration artifact does not exist"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        return None, ["calibration artifact SHA256 mismatch"]
    try:
        artifact = load_json(path)
    except (OSError, json.JSONDecodeError, GateError) as exc:
        return None, [f"invalid calibration artifact: {exc}"]
    if artifact.get("artifact_sha256") not in (None, actual_hash):
        return None, ["calibration artifact self-hash mismatch"]
    return artifact, []


def apply_gate(config: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    hard_gate = evidence.get("hard_gate", {})
    if hard_gate.get("reject") is True:
        return {
            "experiment_id": evidence.get("experiment_id"),
            "decision": "REJECT",
            "score_direction_claim": False,
            "reasons": hard_gate.get("reasons", ["structural safety fuse fired"]),
        }

    if evidence.get("protocol_version") != config["protocol_version"]:
        reasons.append("protocol version mismatch")
    benchmark_id = config["benchmark"]["benchmark_id"]
    if evidence.get("benchmark_id") != benchmark_id:
        reasons.append("benchmark id mismatch")

    selection = config["selection"]
    artifact, artifact_errors = load_frozen_calibration(evidence)
    reasons.extend(artifact_errors)
    calibration = artifact or {}
    if artifact is not None:
        if calibration.get("protocol_version") != config["protocol_version"]:
            reasons.append("calibration artifact protocol mismatch")
        if calibration.get("benchmark_id") != benchmark_id:
            reasons.append("calibration artifact benchmark mismatch")
        if calibration.get("status") != "CERTIFIED":
            reasons.append("direction calibration artifact is not certified")
        supported = calibration.get("supported_families", [])
        if selection.get("require_supported_family") and evidence.get("family_id") not in supported:
            reasons.append("experiment family is not supported by calibration")
        if selection.get("require_drift_check") and calibration.get("drift_check_passed") is not True:
            reasons.append("benchmark drift check did not pass")

    coverage = calibration.get("estimated_coverage")
    minimum_coverage = float(config["target"]["minimum_coverage"])
    if not isinstance(coverage, (int, float)) or coverage < minimum_coverage:
        reasons.append("calibrated coverage is missing or below the preregistered minimum")

    prediction = calibration.get("predictions", {}).get(evidence.get("experiment_id"), {})
    p_up = prediction.get("p_up")
    p_down = prediction.get("p_down")
    valid_probability = lambda value: isinstance(value, (int, float)) and 0.0 <= value <= 1.0
    if not valid_probability(p_up) or not valid_probability(p_down):
        reasons.append("calibrated direction probabilities are unavailable")

    if reasons:
        return {
            "experiment_id": evidence.get("experiment_id"),
            "decision": "ABSTAIN",
            "score_direction_claim": False,
            "reasons": reasons,
        }

    up_threshold = float(selection["up_probability_threshold"])
    down_threshold = float(selection["down_probability_threshold"])
    up = p_up >= up_threshold
    down = p_down >= down_threshold
    if up and down:
        raise GateError("both UP and DOWN probabilities exceed their decision thresholds")
    if up:
        decision = "UP"
    elif down:
        decision = "DOWN"
    else:
        decision = "ABSTAIN"
    return {
        "experiment_id": evidence.get("experiment_id"),
        "decision": decision,
        "score_direction_claim": decision in DIRECTION_DECISIONS,
        "p_up": p_up,
        "p_down": p_down,
        "reasons": [] if decision in DIRECTION_DECISIONS else ["neither direction reached 90%"],
    }


def markdown_audit(report: dict[str, Any]) -> str:
    lines = [
        f"# Score direction gate: {report['status']}",
        "",
        f"- Protocol: `{report['protocol_version']}`",
        f"- Eligible outcomes: {report['eligible_outcomes']}",
        f"- Direction decisions: {report['direction_decisions']}",
        f"- Coverage: {report['coverage']:.1%} (required {report['minimum_coverage']:.1%})",
        "",
        "| Class | Trials | Correct | Accuracy | One-sided lower bound | Certified |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in ("UP", "DOWN"):
        result = report["classes"][name]
        accuracy = "—" if result["strict_accuracy"] is None else f"{result['strict_accuracy']:.1%}"
        lower = (
            "—"
            if result["strict_lower_confidence_bound"] is None
            else f"{result['strict_lower_confidence_bound']:.1%}"
        )
        lines.append(
            f"| {name} | {result['trials']} | {result['strict_successes']} | "
            f"{accuracy} | {lower} | {'yes' if result['certified'] else 'no'} |"
        )
    lines += [
        "",
        "Minimum independent prospective decisions per class at the configured target:",
        "",
        "| Allowed errors | Required trials | Lower bound |",
        "|---:|---:|---:|",
    ]
    for row in report["sample_requirements"]:
        lines.append(
            f"| {row['max_errors']} | {row['minimum_trials']} | {row['lower_bound']:.3f} |"
        )
    lines += [
        "",
        "With both UP and DOWN required and 30% minimum coverage, the zero-error case needs at least "
        f"{report['minimum_eligible_outcomes_zero_error_both_classes']} eligible outcomes.",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit_parser = sub.add_parser("audit", help="audit prospective direction accuracy")
    audit_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    audit_parser.add_argument("--format", choices=("json", "markdown"), default="markdown")

    decide_parser = sub.add_parser("decide", help="apply the frozen selective decision rule")
    decide_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    decide_parser.add_argument("--evidence", type=Path, required=True)

    req_parser = sub.add_parser("requirements", help="show exact sample-size requirements")
    req_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    req_parser.add_argument("--max-errors", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_json(args.config)
        if args.command == "audit":
            report = audit(config, load_jsonl(args.ledger))
            print(json.dumps(report, ensure_ascii=False, indent=2) if args.format == "json" else markdown_audit(report))
        elif args.command == "decide":
            result = apply_gate(config, load_json(args.evidence))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            target = config["target"]
            rows = []
            for errors in range(args.max_errors + 1):
                trials, lower = minimum_trials(
                    errors,
                    float(target["conditional_accuracy"]),
                    float(target["one_sided_confidence"]),
                )
                rows.append({"max_errors": errors, "minimum_trials": trials, "lower_bound": lower})
            print(json.dumps(rows, ensure_ascii=False, indent=2))
    except (GateError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
