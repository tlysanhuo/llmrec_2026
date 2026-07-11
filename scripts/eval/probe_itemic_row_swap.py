#!/usr/bin/env python3
"""Probe itemic embedding/output-row transplants between two checkpoints."""

import argparse
import json
import os
import re
from pathlib import Path


def rebuild_visible_prompt(log_path: Path) -> tuple[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    start = text.find("Task [4/8]")
    end = text.find("Task [5/8]", start)
    section = text[start:end]
    blocks = re.split(r"Sample ID: (\d+)", section)
    body = blocks[blocks.index("3") + 1]
    raw_input = body[body.find("Input:") + 6:body.find("Output[0]:")]
    system = re.search(
        r"<\|im_start\|>system\n(.*?)<\|im_end", raw_input, re.S
    ).group(1).replace("\n", "")
    user = re.search(
        r"<\|im_start\|>user\n(.*?)/no_think", raw_input, re.S
    ).group(1)
    _, _, description = user.partition("：\n\n")
    description = "\n\n".join(
        part.replace("\n", "") for part in description.split("\n\n")
    )
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n请解析以下视频内容并输出对应的视频token：\n\n"
        f"{description}/no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<|video_begin|>"
    )
    return prompt, description


def beam_stats(tokenizer, generated, prompt_length: int):
    sequences = [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=False)
        for row in generated
    ]
    locked = sum(sequence.startswith("<s_a_2391>") for sequence in sequences)
    joined = "".join(sequences)
    fanout = len(set(re.findall(r"<s_a_2391><s_b_(\d+)>", joined)))
    top_a = {}
    for sequence in sequences:
        match = re.match(r"<s_a_(\d+)>", sequence)
        if match:
            token = match.group(1)
            top_a[token] = top_a.get(token, 0) + 1
    return {
        "locked_a2391": locked,
        "fanout_b_under_a2391": fanout,
        "top_a": sorted(top_a.items(), key=lambda item: -item[1])[:5],
        "sequences": sequences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--donor", required=True)
    parser.add_argument("--log", default="logs/eval/riders_fk_lora_ep1_20260706.log")
    parser.add_argument("--gpu", default="3")
    parser.add_argument(
        "--modes", default="control,lm_head,embed,both",
        help="Comma-separated subset of control,lm_head,embed,both",
    )
    parser.add_argument("--output", default="logs/probe/itemic_row_swap.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda().eval()

    first_itemic_id = tokenizer.convert_tokens_to_ids("<s_a_0>")
    final_domain_id = max(
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<|video_begin|>", "<|prod_begin|>", "<|living_begin|>", "<|ad_begin|>")
    )
    row_slice = slice(first_itemic_id, final_domain_id + 1)
    original_head = model.lm_head.weight[row_slice].detach().clone()
    original_embed = model.model.embed_tokens.weight[row_slice].detach().clone()

    donor_path = Path(args.donor) / "model.safetensors"
    with safe_open(donor_path, framework="pt", device="cpu") as handle:
        donor_head = handle.get_slice("lm_head.weight")[row_slice].to(
            model.device, dtype=model.lm_head.weight.dtype
        )
        donor_embed = handle.get_slice("model.embed_tokens.weight")[row_slice].to(
            model.device, dtype=model.model.embed_tokens.weight.dtype
        )

    prompt, description = rebuild_visible_prompt(Path(args.log))
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        if not mode:
            continue
        with torch.no_grad():
            model.lm_head.weight[row_slice].copy_(
                donor_head if mode in {"lm_head", "both"} else original_head
            )
            model.model.embed_tokens.weight[row_slice].copy_(
                donor_embed if mode in {"embed", "both"} else original_embed
            )
            generated = model.generate(
                **encoded,
                num_beams=64,
                num_return_sequences=64,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        results[mode] = beam_stats(tokenizer, generated, encoded.input_ids.shape[1])
        printable = {key: value for key, value in results[mode].items() if key != "sequences"}
        print(f"{mode}: {json.dumps(printable, ensure_ascii=False)}")

    control = set(results.get("control", {}).get("sequences", []))
    for mode, result in results.items():
        sequences = set(result["sequences"])
        union = control | sequences
        result["jaccard_vs_control"] = len(control & sequences) / len(union) if union else 1.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": str(Path(args.model).resolve()),
                "donor": str(Path(args.donor).resolve()),
                "itemic_rows": [first_itemic_id, final_domain_id],
                "description_chars": len(description),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
