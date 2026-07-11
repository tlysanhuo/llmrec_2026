#!/usr/bin/env python3
"""Replay the visible action-select prompts printed in a platform eval log."""

import argparse
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
        r"Task \[\d/8\]: challenge_evolution_action_select", text
    )
    if task_match is None:
        raise ValueError(f"action-select task not found in {log_path}")
    task_start = task_match.start()
    samples_start = text.find("Sample ID:", task_start)
    task_end = text.find("Task [3/8]", samples_start)
    section = text[samples_start:task_end if task_end >= 0 else None]

    matches = list(re.finditer(r"Sample ID: (\d+)\nInput:\n", section))
    prompts = []
    for match in matches[:limit]:
        block_end = section.find("\nOutput[0]:\n", match.end())
        if block_end < 0:
            continue
        prompt = section[match.end():block_end].strip("\n")
        prompts.append({"sample_id": int(match.group(1)), "prompt": prompt})
    if not prompts:
        raise ValueError(f"no visible action prompts parsed from {log_path}")
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
    values = parse_array(output)
    strings = [value for value in values or [] if isinstance(value, str)]
    positions = [prompt.find(value) for value in strings]
    quoted = [position >= 0 for position in positions]
    ordered_positions = [position for position in positions if position >= 0]
    ordered = all(
        left <= right for left, right in zip(ordered_positions, ordered_positions[1:])
    )
    counts = Counter(strings)
    item_values = [value for value in strings if ITEM_RE.fullmatch(value)]
    return {
        "json_array": values is not None,
        "n_values": len(strings),
        "n_unique": len(counts),
        "duplicate_values": len(strings) - len(counts),
        "max_repeat": max(counts.values(), default=0),
        "quoted_from_prompt": sum(quoted),
        "quote_rate": sum(quoted) / len(strings) if strings else 0.0,
        "quoted_in_history_order": ordered,
        "item_values": len(item_values),
        "text_values": len(strings) - len(item_values),
        "generated_tokens": generated_tokens,
        "hit_token_limit": generated_tokens >= max_new_tokens,
        "output_chars": len(output),
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
        metrics["seconds"] = round(elapsed, 3)
        metrics["output_preview"] = output[:500]
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

    summary = {
        "model": str(Path(args.model).resolve()),
        "source_log": str(Path(args.log).resolve()),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "samples": results,
        "totals": {
            "json_ok": sum(item["json_array"] for item in results),
            "token_limit": sum(item["hit_token_limit"] for item in results),
            "all_quoted": sum(item["quote_rate"] == 1.0 for item in results),
            "all_ordered": sum(item["quoted_in_history_order"] for item in results),
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
