import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/eval/score_direction_gate.py"
SPEC = importlib.util.spec_from_file_location("score_direction_gate", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


class ScoreDirectionGateTest(unittest.TestCase):
    def setUp(self):
        self.config = gate.load_json(ROOT / "configs/evaluation/score_direction_gate_v1.json")

    def test_exact_sample_requirements(self):
        self.assertEqual(gate.minimum_trials(0, 0.9, 0.95)[0], 29)
        self.assertEqual(gate.minimum_trials(1, 0.9, 0.95)[0], 46)
        self.assertEqual(gate.minimum_trials(2, 0.9, 0.95)[0], 61)

    def test_current_candidate_abstains(self):
        evidence = gate.load_json(
            ROOT / "configs/evaluation/riders_fk_clean_r64_e1_direction_evidence.json"
        )
        result = gate.apply_gate(self.config, evidence)
        self.assertEqual(result["decision"], "ABSTAIN")
        self.assertFalse(result["score_direction_claim"])

    def test_hard_failure_is_not_direction_claim(self):
        evidence = {
            "experiment_id": "broken",
            "protocol_version": "score-direction-v1",
            "benchmark_id": "llmrec-2026-visible-v2",
            "hard_gate": {"reject": True, "reasons": ["broken itemic"]},
        }
        result = gate.apply_gate(self.config, evidence)
        self.assertEqual(result["decision"], "REJECT")
        self.assertFalse(result["score_direction_claim"])

    def test_self_reported_probability_cannot_bypass_artifact(self):
        evidence = {
            "experiment_id": "self-reported",
            "family_id": "new-family",
            "protocol_version": "score-direction-v1",
            "benchmark_id": "llmrec-2026-visible-v2",
            "hard_gate": {"reject": False},
            "calibration": {
                "status": "certified",
                "family_supported": True,
                "drift_check_passed": True,
                "estimated_coverage": 1.0,
                "p_up": 0.999,
                "p_down": 0.001,
            },
        }
        result = gate.apply_gate(self.config, evidence)
        self.assertEqual(result["decision"], "ABSTAIN")
        self.assertIn("frozen calibration artifact is unavailable", result["reasons"])

    def test_audit_counts_eligible_abstention_but_not_legacy_or_rejected_rows(self):
        rows = gate.load_jsonl(ROOT / "configs/evaluation/score_direction_ledger_v1.jsonl")
        result = gate.audit(self.config, rows)
        self.assertEqual(result["status"], "NOT_CERTIFIED")
        self.assertEqual(result["eligible_outcomes"], 1)
        self.assertEqual(result["direction_decisions"], 0)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["minimum_eligible_outcomes_zero_error_both_classes"], 194)

    def test_same_family_cannot_be_renamed_into_two_independent_units(self):
        base = {
            "baseline_id": "base",
            "family_id": "shared-family",
            "protocol_version": "score-direction-v1",
            "prospective": True,
            "eligible": True,
            "decision_at_utc": "2026-07-01T00:00:00Z",
            "decision": "UP",
            "outcome": {
                "observed_at_utc": "2026-07-02T00:00:00Z",
                "candidate_score": 1.0,
                "baseline_score": 0.9,
            },
        }
        rows = [
            {**base, "experiment_id": "one", "independence_unit": "renamed-one"},
            {**base, "experiment_id": "two", "independence_unit": "renamed-two"},
        ]
        with self.assertRaisesRegex(gate.GateError, "experiment-family clusters"):
            gate.audit(self.config, rows)


if __name__ == "__main__":
    unittest.main()
