#!/usr/bin/env python3
"""Re-run the I-38 gate with the intended KL direction.

The frozen I-38 evaluator calls ``F.kl_div(log_softmax(candidate),
softmax(reference))``. PyTorch interprets that as ``KL(reference ||
candidate)``, although the report labels it candidate-to-reference. This
wrapper keeps the original frozen evaluator and thresholds untouched while
reusing its locked data/model loop with the intended ``KL(candidate ||
reference)`` calculation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATH = ROOT / "scripts/eval/audit_i38_i23_material_i35_teacher_gate.py"


def load_legacy():
    spec = importlib.util.spec_from_file_location("i38_legacy_gate", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen evaluator: {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_to_reference_kl(candidate, reference) -> float:
    import torch
    import torch.nn.functional as functional

    if candidate.shape != reference.shape:
        raise RuntimeError("I-38 corrected gate KL shape mismatch")
    total = torch.zeros((), device=candidate.device, dtype=torch.float32)
    for start in range(0, candidate.size(1), 8):
        end = min(start + 8, candidate.size(1))
        log_candidate = functional.log_softmax(candidate[:, start:end].float(), dim=-1)
        log_reference = functional.log_softmax(reference[:, start:end].float(), dim=-1)
        candidate_prob = log_candidate.exp()
        total += (candidate_prob * (log_candidate - log_reference)).sum()
    value = float(total / candidate.size(1))
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError("non-finite I-38 corrected gate KL")
    return value


def main() -> None:
    legacy = load_legacy()
    legacy.forward_kl = candidate_to_reference_kl
    legacy.main()


if __name__ == "__main__":
    main()
