#!/usr/bin/env python3
"""Generate unfiltered action-select responses for later preference scoring.

This is a GLM Platform Training Tasks entry-point, not a local dev-machine job.
All sampling choices are required CLI arguments so the script does not silently
select rollout or decoding parameters.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


DEFAULT_VOLUME = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--samples-per-prompt", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", required=True, type=float)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-prompts", required=True, type=int)
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def verify_volume_path(path: Path):
    volume = Path(os.environ.get("PERSONAL_VOLUME_ROOT", str(DEFAULT_VOLUME))).resolve()
    subprocess.run(["mountpoint", "-q", str(volume)], check=True)
    if not os.access(volume, os.W_OK):
        raise PermissionError(f"personal volume is not writable: {volume}")
    resolved = path.resolve()
    if resolved != volume and volume not in resolved.parents:
        raise ValueError(f"path must be on the personal volume: {resolved}")


def verify_input(path: Path, *, directory: bool):
    verify_volume_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if directory and not path.is_dir():
        raise NotADirectoryError(path)
    if not directory and not path.is_file():
        raise ValueError(f"expected a file: {path}")


def load_records(path: Path, num_shards: int, shard_index: int):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard selection")
    records = []
    action_ids = set()
    for index, line in enumerate(path.open(encoding="utf-8")):
        if index % num_shards == shard_index:
            record = json.loads(line)
            action_id = record.get("action_id")
            if not isinstance(action_id, str):
                raise ValueError(f"missing action_id at {path}:{index + 1}")
            if action_id in action_ids:
                raise ValueError(f"duplicate action_id in gold shard: {action_id}")
            action_ids.add(action_id)
            records.append(record)
    return records


def build_prompt(record: dict) -> str:
    sections = []
    instruction = record.get("instruction", "")
    if instruction:
        sections.append(f"<|im_start|>system\n{instruction}<|im_end|>\n")
    sections.append(
        f"<|im_start|>user\n{record['input']}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )
    return "".join(sections)


def existing_ids(path: Path, expected_samples: int) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        row = json.loads(line)
        action_id = row.get("action_id")
        if not isinstance(action_id, str):
            raise ValueError(f"invalid existing row at {path}:{line_number}")
        if action_id in ids:
            raise ValueError(f"duplicate existing action_id: {action_id}")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != expected_samples:
            raise ValueError(f"incomplete existing row at {path}:{line_number}")
        if any(
            not isinstance(candidate, dict)
            or not isinstance(candidate.get("text"), str)
            for candidate in outputs
        ):
            raise ValueError(f"invalid candidate at {path}:{line_number}")
        ids.add(action_id)
    return ids


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollout_config(args, output_path: Path) -> dict:
    return {
        "model": str(Path(args.model).resolve()),
        "gold": str(Path(args.gold).resolve()),
        "gold_sha256": file_sha256(Path(args.gold)),
        "output": str(output_path.resolve()),
        "samples_per_prompt": args.samples_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "batch_prompts": args.batch_prompts,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }


def write_metadata(output_path: Path, config: dict, expected_samples: int):
    metadata = {
        **config,
        "rows_in_file": len(existing_ids(output_path, expected_samples)),
        "sha256": file_sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if args.samples_per_prompt < 1 or args.batch_prompts < 1:
        raise ValueError("sample and batch counts must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("--top-k must be -1 or positive")
    if min(
        args.max_new_tokens,
        args.max_model_len,
        args.tensor_parallel_size,
        args.num_shards,
    ) < 1:
        raise ValueError("token, parallel, and shard counts must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    if args.max_new_tokens >= args.max_model_len:
        raise ValueError("--max-new-tokens must be smaller than --max-model-len")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1)")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_path = Path(args.model)
    gold_path = Path(args.gold)
    output_path = Path(args.out)
    verify_input(model_path, directory=True)
    verify_input(gold_path, directory=False)
    if not (model_path / "config.json").is_file():
        raise ValueError(f"model directory has no config.json: {model_path}")
    verify_volume_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output_path}")
    if args.resume and not output_path.exists():
        raise FileNotFoundError(f"cannot resume missing output: {output_path}")

    config_path = output_path.with_suffix(output_path.suffix + ".config.json")
    current_config = rollout_config(args, output_path)
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"cannot resume without rollout config: {config_path}")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != current_config:
            raise ValueError("resume arguments do not match the saved rollout config")
    else:
        config_path.write_text(
            json.dumps(current_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    completed = (
        existing_ids(output_path, args.samples_per_prompt) if args.resume else set()
    )
    shard_records = load_records(gold_path, args.num_shards, args.shard_index)
    shard_ids = {record["action_id"] for record in shard_records}
    unknown_completed = completed - shard_ids
    if unknown_completed:
        raise ValueError(f"existing output has IDs outside this shard: {sorted(unknown_completed)[:5]}")
    records = [record for record in shard_records if record["action_id"] not in completed]
    if not records:
        print("no pending prompts")
        if not output_path.exists():
            output_path.touch()
        write_metadata(output_path, current_config, args.samples_per_prompt)
        return

    from vllm import LLM, SamplingParams

    model = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=args.seed,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling = SamplingParams(
        n=args.samples_per_prompt,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    mode = "a" if args.resume else "w"
    generated_count = 0
    with output_path.open(mode, encoding="utf-8") as output:
        for offset in range(0, len(records), args.batch_prompts):
            batch = records[offset:offset + args.batch_prompts]
            responses = model.generate([build_prompt(record) for record in batch], sampling)
            if len(responses) != len(batch):
                raise RuntimeError("vLLM returned a different number of prompt responses")
            for record, response in zip(batch, responses):
                candidates = [
                    {
                        "text": candidate.text,
                        "finish_reason": candidate.finish_reason,
                        "stop_reason": candidate.stop_reason,
                        "token_count": len(candidate.token_ids),
                    }
                    for candidate in response.outputs
                ]
                if len(candidates) != args.samples_per_prompt:
                    raise RuntimeError("vLLM returned an incomplete candidate set")
                output.write(
                    json.dumps(
                        {"action_id": record["action_id"], "outputs": candidates},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                generated_count += 1
            output.flush()
            print(f"generated {generated_count}/{len(records)} pending prompts", flush=True)

    write_metadata(output_path, current_config, args.samples_per_prompt)


if __name__ == "__main__":
    main()
