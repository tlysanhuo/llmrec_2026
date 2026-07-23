#!/usr/bin/env python3
"""Score the frozen s800 parent on the permanent official-General holdout.

This is a bounded teacher-forced answer-letter diagnostic, not generation and
not an online-score estimator.  It never writes gradients or model artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
ADAPTER = ROOT / "submissions/e3_userres_r80_retkl_v3_s800_platform"
HOLDOUT = ROOT / "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl"
TRAIN = ROOT / "assets/derived/processed/data_official_general_world_mc_v1.jsonl"
SPLIT_AUDIT = ROOT / "logs/data/official_general_world_mc_v1_split_audit.json"
OUT = ROOT / "logs/probe/official_general_world_mc_v1_s800_baseline.json"
LEDGER = ROOT / "logs/probe/official_general_world_mc_v1_s800_baseline_rows.jsonl"

EXPECTED = {
    "base_config": "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4",
    "adapter": "bb86eb8af0efd3560b7b7c8440f3830627e9255f4fcc2265b9274a27668f63c6",
    "adapter_config": "e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0",
    "holdout": "fb67b76d8d071799ba372185bd89cb556afef9065a1b188fb9dd86a9131e13df",
    "train": "f8cccd1f2302704adca10f4f75c5b348e67592b419e912062f93f2572962e79f",
}
EXPECTED_HOLDOUT_ROWS = 25
EXPECTED_SCORING_HOLDOUT_ROWS = 23
EXPECTED_TRAIN_ROWS = 68
EXPECTED_RETIRED_IDS = {
    "d4e27d1a9b8a22b59db0025a62c36e67d2264489b96e1848df4c66942f7a32ae",
    "9f327cd1a9a39706ef559c7c76f6b442c326bc20040cfb7f9e3cb8758f5315a7",
}
ANSWER_LETTERS = "ABCD"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(stable_json(row) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def prompt_prefix(row: dict[str, Any]) -> tuple[str, str]:
    output = str(row["output"])
    answer = str(row["clean"]["answer_letter"])
    if answer not in ANSWER_LETTERS or not output.endswith(f"({answer})"):
        raise ValueError(f"non-canonical holdout answer: {row.get('record_id')}")
    assistant_prefix = output[: -len(answer + ")")]
    prompt = (
        f"<|im_start|>system\n{row['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{row['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )
    return prompt, answer


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "abcd_accuracy": round(mean([float(row["abcd_correct"]) for row in rows]), 8),
        "all_vocab_accuracy": round(mean([float(row["all_vocab_correct"]) for row in rows]), 8),
        "gold_logp_mean": round(mean([float(row["gold_logp"]) for row in rows]), 8),
        "gold_abcd_probability_mean": round(
            mean([float(row["gold_abcd_probability"]) for row in rows]), 8
        ),
        "gold_vs_best_wrong_margin_mean": round(
            mean([float(row["gold_vs_best_wrong_margin"]) for row in rows]), 8
        ),
        "gold_rank_median": round(
            statistics.median([float(row["gold_rank_all_vocab"]) for row in rows]), 4
        ),
        "prediction_distribution": dict(
            sorted(Counter(str(row["abcd_prediction"]) for row in rows).items())
        ),
        "answer_distribution": dict(sorted(Counter(str(row["gold_answer"]) for row in rows).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = {
        "base": BASE,
        "adapter": ADAPTER,
        "holdout": HOLDOUT,
        "train": TRAIN,
    }
    for name, path in (("base", args.base), ("adapter", args.adapter), ("holdout", args.holdout), ("train", args.train)):
        if path.resolve() != canonical[name].resolve():
            raise RuntimeError(f"{name} must use canonical path: {canonical[name]}")
    checks = {
        "base_config": sha256(args.base / "config.json"),
        "adapter": sha256(args.adapter / "adapter_model.safetensors"),
        "adapter_config": sha256(args.adapter / "adapter_config.json"),
        "holdout": sha256(args.holdout),
        "train": sha256(args.train),
    }
    if checks != EXPECTED:
        raise RuntimeError(f"input fingerprint drifted: {checks} != {EXPECTED}")

    rows = read_jsonl(args.holdout)
    train_rows = read_jsonl(args.train)
    if len(rows) != EXPECTED_HOLDOUT_ROWS or len(train_rows) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"row signature drifted: holdout={len(rows)} train={len(train_rows)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate holdout record_id")
    if {row["input"] for row in rows} & {row["input"] for row in train_rows}:
        raise RuntimeError("holdout prompt leaked into trainer projection")
    scoring_rows = [
        row for row in rows if row.get("split", {}).get("evaluation_eligible") is True
    ]
    retired_rows = [
        row for row in rows if row.get("split", {}).get("evaluation_eligible") is False
    ]
    if len(scoring_rows) != EXPECTED_SCORING_HOLDOUT_ROWS:
        raise RuntimeError(
            f"scoring row signature drifted: {len(scoring_rows)} != "
            f"{EXPECTED_SCORING_HOLDOUT_ROWS}"
        )
    if {str(row["record_id"]) for row in retired_rows} != EXPECTED_RETIRED_IDS:
        raise RuntimeError("retired holdout ID signature drifted")
    if any(
        row.get("split", {}).get("checkpoint_selection_eligible") is not False
        or row.get("split", {}).get("contamination", {}).get("contamination_reason")
        != "current_parent_exact"
        for row in retired_rows
    ):
        raise RuntimeError("retired holdout metadata is incomplete")
    examples = [(row, *prompt_prefix(row)) for row in scoring_rows]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, use_fast=True)
    letter_token_ids: dict[str, int] = {}
    for letter in ANSWER_LETTERS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(f"answer letter is not one token: {letter} -> {token_ids}")
        letter_token_ids[letter] = token_ids[0]
    if len(set(letter_token_ids.values())) != 4:
        raise RuntimeError("answer letter token IDs collide")

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    started = time.time()
    ledger: list[dict[str, Any]] = []
    answer_ids = torch.tensor([letter_token_ids[x] for x in ANSWER_LETTERS], device="cuda")
    with torch.inference_mode():
        for index, (row, prompt, answer) in enumerate(examples, 1):
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)], device="cuda"
            )
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            abcd_logits = logits.index_select(0, answer_ids)
            abcd_probs = F.softmax(abcd_logits, dim=-1)
            gold_index = ANSWER_LETTERS.index(answer)
            wrong = torch.cat((abcd_logits[:gold_index], abcd_logits[gold_index + 1 :]))
            all_prediction_id = int(logits.argmax())
            abcd_prediction = ANSWER_LETTERS[int(abcd_logits.argmax())]
            gold_token_id = letter_token_ids[answer]
            gold_rank = int((logits > logits[gold_token_id]).sum()) + 1
            ledger.append(
                {
                    "record_id": row["record_id"],
                    "split_cohort": row["split"]["cohort"],
                    "split_topic": row["split"]["topic"],
                    "upstream_asset": row["lineage"].get("asset_id", "unknown"),
                    "gold_answer": answer,
                    "abcd_prediction": abcd_prediction,
                    "all_vocab_prediction_token_id": all_prediction_id,
                    "all_vocab_prediction_text": tokenizer.decode([all_prediction_id]),
                    "abcd_correct": abcd_prediction == answer,
                    "all_vocab_correct": all_prediction_id == gold_token_id,
                    "gold_logp": float(log_probs[gold_token_id]),
                    "gold_abcd_probability": float(abcd_probs[gold_index]),
                    "gold_vs_best_wrong_margin": float(abcd_logits[gold_index] - wrong.max()),
                    "gold_rank_all_vocab": gold_rank,
                }
            )
            if index % 8 == 0:
                print(f"[world-v1-baseline] {index}/{len(examples)}", flush=True)

    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_upstream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_cohort[str(row["split_cohort"])].append(row)
        by_topic[str(row["split_topic"])].append(row)
        by_upstream[str(row["upstream_asset"])].append(row)
    atomic_jsonl(args.ledger, ledger)
    report = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "COMPLETE_BASELINE_NOT_ONLINE_SCORE_ESTIMATE",
        "method": {
            "model": "current fixed-protocol main submission s800 (online displays 1.0037/1.0048)",
            "teacher_forced": True,
            "generation": False,
            "stochastic_sampling": False,
            "route": "single A-D answer token after canonical empty-think prefix",
            "selection_warning": (
                "23 scoring-eligible rows from a 25-row permanent-E asset; two exposed "
                "current-parent duplicates are retired; not an online score prediction"
            ),
        },
        "inputs": {
            **checks,
            "base": str(args.base.resolve()),
            "adapter_path": str(args.adapter.resolve()),
            "holdout_path": str(args.holdout.resolve()),
            "train_path": str(args.train.resolve()),
            "permanent_holdout_rows": len(rows),
            "scoring_holdout_rows": len(scoring_rows),
            "retired_parent_contaminated_rows": len(retired_rows),
            "retired_record_ids": sorted(EXPECTED_RETIRED_IDS),
            "train_rows": len(train_rows),
            "train_holdout_prompt_overlap": 0,
        },
        "tokenization": {"answer_letter_token_ids": letter_token_ids},
        "overall": summarize(ledger),
        "by_review_cohort": {key: summarize(value) for key, value in sorted(by_cohort.items())},
        "by_topic": {key: summarize(value) for key, value in sorted(by_topic.items())},
        "by_upstream_asset": {key: summarize(value) for key, value in sorted(by_upstream.items())},
        "output_ledger": {
            "path": str(args.ledger.resolve()),
            "rows": len(ledger),
            "bytes": args.ledger.stat().st_size,
            "sha256": sha256(args.ledger),
            "asset_role": "E diagnostic; never gradient data",
        },
        "resources": {
            "gpu_count": 1,
            "elapsed_seconds": round(time.time() - started, 3),
            "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        },
        "provenance": {
            "evaluator": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "split_audit": str(SPLIT_AUDIT.resolve()),
            "split_audit_sha256": sha256(SPLIT_AUDIT),
        },
    }
    atomic_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
