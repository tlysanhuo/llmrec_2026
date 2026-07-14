#!/usr/bin/env python3
"""Replay the visible action-select prompts printed in a platform eval log."""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITEM_RE = re.compile(
    r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)


def parse_visible_prompts(log_path: Path, limit: int) -> list[dict]:
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    task_match = re.search(
        r"Task \[\d/8\]:\s*challenge_evolution_action_select\s*\|\s*Split:\s*test",
        text,
    )
    if task_match is None:
        raise ValueError(f"action-select task not found in {log_path}")
    samples_start = text.find("Sample ID:", task_match.end())
    next_task = re.search(
        r"Task \[\d/8\]:\s*challenge_evolution_topic_gen\s*\|\s*Split:\s*test",
        text[samples_start:],
    )
    if samples_start < 0 or next_task is None:
        raise ValueError(f"action-select sample boundaries not found in {log_path}")
    section = text[samples_start:samples_start + next_task.start()]

    matches = list(re.finditer(r"Sample ID: (\d+)\nInput:\n", section))
    prompts = []
    for match in matches[:limit]:
        block_end = section.find("\nOutput[0]:\n", match.end())
        if block_end < 0:
            continue
        prompt = section[match.end():block_end].strip("\n")
        if "【用户交互历史】" not in prompt or "请回答以下问题" in prompt:
            raise ValueError(
                f"sample {match.group(1)} is not an action-select prompt in {log_path}"
            )
        prompts.append({"sample_id": int(match.group(1)), "prompt": prompt})
    expected = min(limit, 5)
    if len(prompts) != expected or len({item["sample_id"] for item in prompts}) != expected:
        raise ValueError(
            f"expected {expected} unique action prompts, parsed {len(prompts)} from {log_path}"
        )
    return prompts


def parse_array(output: str):
    body = output.split("</think>", 1)[-1]
    start = body.find("[")
    end = body.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def analyze_output(prompt: str, output: str, generated_tokens: int, max_new_tokens: int):
    body = output.split("</think>", 1)[-1]
    values = parse_array(output)
    json_strings = None if values is None else [value for value in values if isinstance(value, str)]
    occurrences = ITEM_RE.findall(body)
    positions = [prompt.find(value) for value in occurrences]
    quoted = [position >= 0 for position in positions]
    ordered_positions = [position for position in positions if position >= 0]
    ordered = None
    if ordered_positions:
        ordered = all(
            left <= right for left, right in zip(ordered_positions, ordered_positions[1:])
        )
    counts = Counter(occurrences)
    return {
        "json_array": values is not None,
        "json_value_count": None if json_strings is None else len(json_strings),
        "json_item_count": None
        if json_strings is None
        else sum(ITEM_RE.fullmatch(value) is not None for value in json_strings),
        "item_occurrences": len(occurrences),
        "unique_items": len(counts),
        "duplicate_item_occurrences": len(occurrences) - len(counts),
        "max_repeat": max(counts.values(), default=0),
        "quoted_from_prompt": sum(quoted),
        "quote_rate": sum(quoted) / len(occurrences) if occurrences else None,
        "quoted_in_history_order": ordered,
        "generated_tokens": generated_tokens,
        "hit_token_limit": generated_tokens >= max_new_tokens,
        "output_chars": len(output),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    prompts = parse_visible_prompts(Path(args.log), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda().eval()

    results = []
    for record in prompts:
        encoded = tokenizer(record["prompt"], return_tensors="pt").to(model.device)
        started = time.monotonic()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - started
        new_ids = generated[0, encoded.input_ids.shape[1]:]
        output = tokenizer.decode(new_ids, skip_special_tokens=True)
        metrics = analyze_output(
            record["prompt"], output, len(new_ids), args.max_new_tokens
        )
        metrics["sample_id"] = record["sample_id"]
        metrics["prompt_chars"] = len(record["prompt"])
        metrics["prompt_md5"] = hashlib.md5(record["prompt"].encode()).hexdigest()
        metrics["seconds"] = round(elapsed, 3)
        metrics["output_preview"] = output[:500]
        metrics["raw_output"] = output
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

    summary = {
        "parser_version": "action-visible-v3-occurrence-safe",
        "task": "challenge_evolution_action_select",
        "model": str(Path(args.model).resolve()),
        "source_log": str(Path(args.log).resolve()),
        "prompt_source": "platform-log-display-reconstructed",
        "backend": "transformers-flash_attention_2",
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "samples": results,
        "totals": {
            "json_ok": sum(item["json_array"] for item in results),
            "token_limit": sum(item["hit_token_limit"] for item in results),
            "all_quoted": sum(item["quote_rate"] == 1.0 for item in results),
            "all_ordered": sum(item["quoted_in_history_order"] is True for item in results),
            "item_occurrences": sum(item["item_occurrences"] for item in results),
            "unique_items_across_samples_sum": sum(item["unique_items"] for item in results),
            "generated_tokens": sum(item["generated_tokens"] for item in results),
            "seconds": round(sum(item["seconds"] for item in results), 3),
        },
    }
    output_path = Path(args.output) if args.output else Path("logs/probe") / (
        f"visible_action_{Path(args.model).name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
