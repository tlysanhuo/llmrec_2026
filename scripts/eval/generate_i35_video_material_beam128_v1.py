#!/usr/bin/env python3
"""Run the audited I-34 beam ledger implementation at Beam128 for I-35."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "scripts/eval/generate_i34_material_beam_gap_v1.py"
MAX_LOGPROBS = 258


def _requested_gpu() -> str:
    try:
        return sys.argv[sys.argv.index("--gpu") + 1]
    except (ValueError, IndexError):
        return "0"


def main() -> int:
    spec = importlib.util.spec_from_file_location("llmrec_i35_beam_impl", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise ImportError(IMPLEMENTATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if "--self-test" not in sys.argv:
        module.BEAM_WIDTH = 128
    module.AUDIT_SCHEMA_VERSION = "i35-video-material-beam128-audit-v1"
    if "--self-test" in sys.argv or "--preflight" in sys.argv:
        return int(module.main())

    # I34's Beam64 runner caps vLLM at 130 logprobs. Beam128 requests 256,
    # so override only the I35 construction without changing the closed I34 code.
    os.environ["CUDA_VISIBLE_DEVICES"] = _requested_gpu()
    import vllm

    original_llm = vllm.LLM

    def i35_llm(*args: object, **kwargs: object) -> object:
        kwargs["max_logprobs"] = MAX_LOGPROBS
        return original_llm(*args, **kwargs)

    vllm.LLM = i35_llm
    try:
        return int(module.main())
    finally:
        vllm.LLM = original_llm


if __name__ == "__main__":
    raise SystemExit(main())
