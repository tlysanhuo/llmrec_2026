#!/usr/bin/env python3
"""Paired mechanism and held-out retention audit for an I-19 world residual.

This is the world-task counterpart of `audit_user_residual_delta.py` (I-13's
own residual-scale Pareto auditor). It is not a score predictor. It measures,
for a range of candidate residual scales, whether the trained world residual
(`train_world_residual_retkl.py` / `build_world_residual_retention_v1.py`)
still lowers teacher-forced world loss relative to the frozen I-13 parent
while the parent's behavior on the other eight real business tasks
(action/topic/material_desc2sid/material_sid2desc/rec_video/rec_prod/rec_ad/
rec_living) stays close under a paired forward-KL/top-1-agreement probe.

Two important asymmetries versus I-13's own auditor:

- World coverage. The world-residual's supervision branch is the *entire*
  1,573-row Frinkleko `_clean` bucket (see
  `build_world_residual_retention_v1.py`); there is no disjoint held-out slice
  of world rows left over. World rows are therefore audited in-training
  (sampled straight from the exact training mixture,
  `data_world_residual_retention_v1.jsonl`, filtered by the `[I19-ROUTE:WORLD]`
  sentinel) -- this measures how much signal the residual learned, not
  out-of-distribution generalization. This mirrors how I-13's own auditor
  treats its action/topic supervision rows (also sampled from the training
  mixture, not held out).
- Retention coverage. The eight non-world task buckets are still properly
  held out: candidates are drawn from the same 32,644-row
  `data_seed_teacher_v1.jsonl` used to train the I-13 parent, with the 1,573
  rows actually selected into the residual's retention branch
  (`[I19-ROUTE:RETAIN]` rows in the training mixture) excluded first.

Routing does not use I-13's own "first output token" JSON/list heuristic --
world rows are graded multiple-choice text, not JSON -- it instead reuses the
exact sentinel-prefix routing already verified by
`train_world_residual_retkl.py` (stripping the qwen3 chat header, then
matching `WORLD_PREFIX`/`RETAIN_PREFIX`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DATA_DIR = REPO_ROOT / "scripts" / "data"
if str(SCRIPTS_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DATA_DIR))


IM_END = "<|im_end|>"
DEFAULT_RETENTION_TASKS = (
    "action",
    "topic",
    "material_desc2sid",
    "material_sid2desc",
    "rec_video",
    "rec_prod",
    "rec_ad",
    "rec_living",
)

# Must stay byte-identical to WORLD_PREFIX / RETAIN_PREFIX in
# scripts/data/build_world_residual_retention_v1.py and
# scripts/train/train_world_residual_retkl.py.
WORLD_PREFIX = "[I19-ROUTE:WORLD] "
RETAIN_PREFIX = "[I19-ROUTE:RETAIN] "


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("instruction", "")),
        str(row.get("input", "")),
        str(row.get("output", "")),
    )


def unstamped_key(row: dict[str, object]) -> tuple[str, str, str]:
    """Row identity with any route sentinel stripped from `instruction`.

    Used to match training-mixture rows (sentinel-stamped) back to their
    original, unstamped `data_seed_teacher_v1.jsonl` rows so the retention
    held-out pool can exclude exactly the rows actually used for training.
    """
    instruction = str(row.get("instruction", ""))
    if instruction.startswith(WORLD_PREFIX):
        instruction = instruction[len(WORLD_PREFIX):]
    elif instruction.startswith(RETAIN_PREFIX):
        instruction = instruction[len(RETAIN_PREFIX):]
    return instruction, str(row.get("input", "")), str(row.get("output", ""))


def parse_scales(raw: str) -> list[float]:
    scales = []
    for part in raw.split(","):
        scale = float(part.strip())
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("scales must be finite and non-negative")
        if scale not in scales:
            scales.append(scale)
    if not scales:
        raise ValueError("at least one scale is required")
    return scales


def scale_key(scale: float) -> str:
    return f"{scale:g}"


def render(row: dict[str, object]) -> tuple[str, str]:
    instruction = str(row.get("instruction", ""))
    query = str(row.get("input", ""))
    user = "\n".join(part for part in (instruction, query) if part)
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{row['output']}{IM_END}\n"
    return prompt, response


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def load_training_mixture(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split the exact world-residual training mixture by route sentinel."""
    world_rows: list[dict[str, object]] = []
    retain_rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = str(row.get("instruction", ""))
            if instruction.startswith(WORLD_PREFIX):
                world_rows.append(row)
            elif instruction.startswith(RETAIN_PREFIX):
                retain_rows.append(row)
            else:
                raise RuntimeError(
                    f"training-mixture row matched neither route sentinel: "
                    f"{instruction[:80]!r}"
                )
    return world_rows, retain_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument(
        "--train-mixture",
        type=Path,
        required=True,
        help=(
            "the exact sentinel-stamped training file produced by "
            "build_world_residual_retention_v1.py "
            "(data_world_residual_retention_v1.jsonl); world rows are "
            "sampled from its [I19-ROUTE:WORLD] rows (in-training audit, "
            "since the residual's world branch has no disjoint held-out "
            "slice), and its [I19-ROUTE:RETAIN] rows are excluded from the "
            "retention held-out pool below"
        ),
    )
    parser.add_argument(
        "--retention-source",
        type=Path,
        required=True,
        help=(
            "the 32,644-row data_seed_teacher_v1.jsonl used to train the "
            "I-13 parent; held-out retention candidates are drawn from here "
            "after excluding rows already selected into --train-mixture"
        ),
    )
    parser.add_argument(
        "--retention-tasks", default=",".join(DEFAULT_RETENTION_TASKS)
    )
    parser.add_argument("--retention-per-task", type=int, default=32)
    parser.add_argument("--world-n", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cutoff", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=19260821)
    parser.add_argument("--scales", default="1.0")
    args = parser.parse_args()
    scales = parse_scales(args.scales)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from build_seed_scoremax_v1 import load_jsonl, stable_hash, task_of

    from scripts.train.train_world_residual_retkl import forward_kl, weighted_ce

    def uniform_weights(targets: "torch.Tensor") -> "torch.Tensor":
        return torch.ones_like(targets, dtype=torch.float32)

    world_rows, retain_rows_in_mixture = load_training_mixture(args.train_mixture)
    if not world_rows:
        raise RuntimeError(f"no [I19-ROUTE:WORLD] rows found in {args.train_mixture}")
    if not retain_rows_in_mixture:
        raise RuntimeError(f"no [I19-ROUTE:RETAIN] rows found in {args.train_mixture}")

    rng = random.Random(args.seed)
    rng.shuffle(world_rows)
    if len(world_rows) < args.world_n:
        raise RuntimeError(
            f"requested {args.world_n} world rows, training mixture only has "
            f"{len(world_rows)}"
        )
    world_selected = world_rows[: args.world_n]

    retention_tasks = tuple(
        task.strip() for task in args.retention_tasks.split(",") if task.strip()
    )
    if not retention_tasks:
        raise ValueError("retention tasks cannot be empty")

    excluded = {unstamped_key(row) for row in retain_rows_in_mixture}
    source_buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in load_jsonl(args.retention_source):
        task = task_of(row)
        if task in retention_tasks and row_key(row) not in excluded:
            source_buckets[task].append(row)

    heldout_audit: dict[str, Any] = {}
    selected: list[tuple[str, dict[str, object]]] = [
        ("world", row) for row in world_selected
    ]
    for task in retention_tasks:
        candidates = source_buckets[task]
        candidates.sort(
            key=lambda row: stable_hash(
                args.seed, "world-scale-heldout", task, *row_key(row)
            )
        )
        if len(candidates) < args.retention_per_task:
            raise RuntimeError(
                f"only {len(candidates)} held-out {task} rows, "
                f"need {args.retention_per_task}"
            )
        chosen = candidates[: args.retention_per_task]
        selected.extend((task, row) for row in chosen)
        heldout_audit[task] = {
            "available_after_exclusion": len(candidates),
            "selected": len(chosen),
        }

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda()
    parent_wrapper = PeftModel.from_pretrained(base, args.parent, is_trainable=False)
    merged_parent = parent_wrapper.merge_and_unload(safe_merge=True)
    model = PeftModel.from_pretrained(merged_parent, args.residual, is_trainable=False)
    model.eval()

    lora_layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    if not lora_layers:
        raise RuntimeError("residual model contains no LoRA layers")

    def set_residual_scale(scale: float) -> None:
        for layer in lora_layers:
            for adapter in layer.active_adapters:
                layer.set_scale(adapter, scale)

    by_scale: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped_too_long = 0
    with torch.no_grad():
        for index, (expected_task, row) in enumerate(selected, start=1):
            prompt, response = render(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(response, add_special_tokens=False)
            input_ids = prompt_ids + target_ids
            if len(input_ids) > args.cutoff:
                skipped_too_long += 1
                continue

            targets = torch.tensor(target_ids, dtype=torch.long, device="cuda")
            weights = uniform_weights(targets)
            positions = torch.arange(
                len(prompt_ids) - 1,
                len(input_ids) - 1,
                dtype=torch.long,
                device="cuda",
            )
            batch = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long, device="cuda"),
                "attention_mask": torch.ones(
                    (1, len(input_ids)), dtype=torch.long, device="cuda"
                ),
                "logits_to_keep": positions,
            }
            disable_adapter = getattr(model, "disable_adapter", None)
            if disable_adapter is None:
                raise RuntimeError("expected a PEFT residual with disable_adapter()")
            context = disable_adapter()
            with context:
                parent_logits = model(**batch).logits.detach()
            parent_ce = None
            if expected_task == "world":
                parent_ce = float(weighted_ce(parent_logits, targets, weights))
            for scale in scales:
                set_residual_scale(scale)
                candidate_logits = model(**batch).logits.detach()
                if candidate_logits.size(1) != len(target_ids):
                    raise RuntimeError("partial logit length does not match target length")
                metrics = by_scale[scale_key(scale)][expected_task]
                metrics["kl"].append(float(forward_kl(candidate_logits, parent_logits)))
                metrics["top1_agreement"].append(
                    float(
                        candidate_logits.argmax(dim=-1)
                        .eq(parent_logits.argmax(dim=-1))
                        .float()
                        .mean()
                    )
                )
                if parent_ce is not None:
                    candidate_ce = float(weighted_ce(candidate_logits, targets, weights))
                    metrics["parent_ce"].append(parent_ce)
                    metrics["candidate_ce"].append(candidate_ce)
                    metrics["ce_delta"].append(candidate_ce - parent_ce)
                del candidate_logits

            set_residual_scale(1.0)
            del batch, parent_logits, targets
            if index % 16 == 0 or index == len(selected):
                print(f"[world-paired-audit] {index}/{len(selected)}", flush=True)

    result: dict[str, object] = {
        "scope": (
            "in-training world mechanism probe (world branch has no disjoint "
            "held-out slice) plus held-out parent-retention audit on the "
            "other eight real business tasks; not an online score predictor"
        ),
        "seed": args.seed,
        "scales": scales,
        "requested": {"world": args.world_n, **{
            task: args.retention_per_task for task in retention_tasks
        }},
        "heldout_retention": heldout_audit,
        "skipped_too_long": skipped_too_long,
        "artifacts": {
            "base": str(args.base.resolve()),
            "parent": str(args.parent.resolve()),
            "parent_adapter_sha256": sha256(args.parent / "adapter_model.safetensors"),
            "residual": str(args.residual.resolve()),
            "residual_adapter_sha256": sha256(args.residual / "adapter_model.safetensors"),
            "train_mixture": str(args.train_mixture.resolve()),
            "train_mixture_sha256": sha256(args.train_mixture),
            "retention_source": str(args.retention_source.resolve()),
            "retention_source_sha256": sha256(args.retention_source),
        },
        "metrics_by_scale": {},
    }
    metrics_by_scale = result["metrics_by_scale"]
    assert isinstance(metrics_by_scale, dict)
    for scale, task_values in by_scale.items():
        metrics_by_scale[scale] = {
            task: {name: summarize(values) for name, values in task_metrics.items()}
            for task, task_metrics in task_values.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
