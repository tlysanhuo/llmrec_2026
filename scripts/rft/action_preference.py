#!/usr/bin/env python3
"""Extract official action gold, score rollouts, and build DPO pairs."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ITEM_RE = re.compile(
    r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
SEARCH_RE = re.compile(r"^\s+--:-- \[(?:搜索|搜索-搜索)\]\s+(.+?)\s*$", re.M)
EMPTY_THINK_RE = re.compile(r"^<think>\s*</think>\s*", re.S)


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_body(response: str) -> str:
    if "</think>" in response:
        return response.split("</think>", 1)[1].strip()
    return response.strip()


def parse_array(response: str):
    try:
        value = json.loads(response_body(response))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


def history_text(prompt: str) -> str:
    cut_points = [
        position
        for marker in ("\n角色任务", "\n任务：", "\n主题：", "\n输出格式要求")
        if (position := prompt.find(marker)) >= 0
    ]
    return prompt[: min(cut_points)] if cut_points else prompt


def history_values(prompt: str) -> list[str]:
    text = history_text(prompt)
    values = [(match.start(), match.group(0)) for match in ITEM_RE.finditer(text)]
    values.extend((match.start(), match.group(1)) for match in SEARCH_RE.finditer(text))
    values.sort(key=lambda item: item[0])
    return [value for _, value in values]


def is_ordered_subsequence(predicted: list[str], history: list[str]) -> bool:
    cursor = 0
    for value in predicted:
        try:
            offset = history[cursor:].index(value)
        except ValueError:
            return False
        cursor += offset + 1
    return True


def set_f1(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_response(response: str, gold: list[str], history: list[str]) -> dict:
    parsed = parse_array(response)
    predicted = parsed or []
    precision, recall, f1 = set_f1(predicted, gold)
    counts = Counter(predicted)
    predicted_set = set(predicted)
    gold_set = set(gold)
    history_set = set(history)
    quoted = sum(value in history_set for value in predicted)
    duplicate_count = len(predicted) - len(counts)
    return {
        "json_ok": parsed is not None,
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "set_f1": round(f1, 8),
        "true_positive_count": len(predicted_set & gold_set),
        "false_positive_count": len(predicted_set - gold_set),
        "false_negative_count": len(gold_set - predicted_set),
        "predicted_count": len(predicted),
        "unique_count": len(counts),
        "gold_count": len(gold),
        "duplicate_count": duplicate_count,
        "max_repeat": max(counts.values(), default=0),
        "history_quote_rate": (
            round(quoted / len(predicted), 8)
            if predicted
            else float(parsed is not None)
        ),
        "history_ordered": (
            is_ordered_subsequence(predicted, history) if parsed is not None else False
        ),
        "gold_history_ordered": is_ordered_subsequence(gold, history),
        "sequence_exact": predicted == gold,
        "output_chars": len(response),
    }


def stable_action_id(source_index: int, instruction: str, prompt: str) -> str:
    digest = hashlib.sha256((instruction + "\n" + prompt).encode("utf-8")).hexdigest()[:12]
    return f"action-{source_index:05d}-{digest}"


def percentile(values: list[float], fraction: float):
    if not values:
        return 0
    index = round((len(values) - 1) * fraction)
    return sorted(values)[index]


def extract_gold(args):
    source_path = Path(args.src)
    output_path = Path(args.out)
    audit_path = Path(args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for source_index, line in enumerate(source_path.open(encoding="utf-8")):
        row = json.loads(line)
        gold = parse_array(row["output"])
        if gold is None:
            continue
        history = history_values(row["input"])
        if not gold:
            raise AssertionError(f"empty gold at source row {source_index}")
        if len(gold) != len(set(gold)):
            raise AssertionError(f"duplicate gold at source row {source_index}")
        if not set(gold).issubset(set(history)):
            raise AssertionError(f"gold outside history at source row {source_index}")
        records.append(
            {
                "action_id": stable_action_id(
                    source_index, row.get("instruction", ""), row["input"]
                ),
                "source_index": source_index,
                "instruction": row.get("instruction", ""),
                "input": row["input"],
                "chosen": row["output"],
                "gold_values": gold,
                "history_values": history,
                "gold_count": len(gold),
                "gold_history_ordered": is_ordered_subsequence(gold, history),
            }
        )

    if len(records) != 1588:
        raise AssertionError(f"expected 1588 action rows, got {len(records)}")
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = [record["gold_count"] for record in records]
    audit = {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": len(records),
        "empty_think": sum(bool(EMPTY_THINK_RE.match(record["chosen"])) for record in records),
        "gold_inside_history": len(records),
        "gold_no_duplicates": len(records),
        "gold_history_ordered": sum(record["gold_history_ordered"] for record in records),
        "gold_count": {
            "min": min(counts),
            "p25": percentile(counts, 0.25),
            "median": percentile(counts, 0.5),
            "p75": percentile(counts, 0.75),
            "p90": percentile(counts, 0.9),
            "max": max(counts),
        },
        "sha256": file_hash(output_path, "sha256"),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def load_gold(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        action_id = record["action_id"]
        if action_id in records:
            raise AssertionError(f"duplicate action_id: {action_id}")
        records[action_id] = record
    return records


def candidate_record(value) -> dict:
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, dict):
        for key in ("text", "output", "response"):
            if isinstance(value.get(key), str):
                record = dict(value)
                record["text"] = value[key]
                return record
    raise ValueError(f"unsupported candidate: {type(value).__name__}")


def load_generations(paths: list[Path]) -> dict[str, list[dict]]:
    generations = defaultdict(list)
    for path in paths:
        for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
            row = json.loads(line)
            action_id = row.get("action_id") or row.get("id")
            if not isinstance(action_id, str):
                raise ValueError(f"missing action_id at {path}:{line_number}")
            values = row.get("outputs")
            if values is None:
                values = [row.get("output")]
            if not isinstance(values, list):
                raise ValueError(f"outputs must be a list at {path}:{line_number}")
            generations[action_id].extend(candidate_record(value) for value in values)
    return generations


def normalize_response(output: str) -> str:
    stripped = output.strip()
    if stripped.startswith("<think>"):
        return stripped
    return f"<think>\n</think>\n{stripped}"


def error_types(metrics: dict, generation: dict) -> list[str]:
    errors = []
    if not metrics["json_ok"]:
        errors.append("invalid_json")
    if metrics["duplicate_count"]:
        errors.append("duplicate")
    if metrics["json_ok"] and metrics["history_quote_rate"] < 1.0:
        errors.append("history_external")
    if (
        metrics["json_ok"]
        and metrics["history_quote_rate"] == 1.0
        and metrics["gold_history_ordered"]
        and not metrics["history_ordered"]
    ):
        errors.append("unordered")
    if metrics["false_negative_count"]:
        errors.append("underselect")
    if metrics["false_positive_count"]:
        errors.append("overselect")
    if generation.get("finish_reason") == "length":
        errors.append("token_limit")
    return errors or ["clean"]


def scored_candidates(gold_records: dict[str, dict], generations: dict[str, list[dict]]):
    unknown = sorted(set(generations) - set(gold_records))
    if unknown:
        raise ValueError(f"generation file has unknown action ids: {unknown[:5]}")
    for action_id, outputs in generations.items():
        gold_record = gold_records[action_id]
        seen = set()
        for candidate_index, generation in enumerate(outputs):
            normalized = normalize_response(generation["text"])
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            metrics = score_response(
                normalized,
                gold_record["gold_values"],
                gold_record["history_values"],
            )
            yield {
                "action_id": action_id,
                "candidate_index": candidate_index,
                "output": normalized,
                "metrics": metrics,
                "generation": {
                    key: value
                    for key, value in generation.items()
                    if key not in {"text", "output", "response"}
                },
                "error_types": error_types(metrics, generation),
            }


def require_complete_generations(gold_records: dict, generations: dict):
    missing = sorted(set(gold_records) - set(generations))
    if missing:
        raise ValueError(f"generation files are missing action ids: {missing[:5]}")


def score_generations(args):
    gold_records = load_gold(Path(args.gold))
    generations = load_generations([Path(path) for path in args.generations])
    if args.require_complete:
        require_complete_generations(gold_records, generations)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    candidate_count = 0
    set_f1_sum = 0.0
    json_count = 0
    full_quote_count = 0
    no_duplicate_count = 0
    token_limit_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for row in scored_candidates(gold_records, generations):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics = row["metrics"]
            summary.update(row["error_types"])
            candidate_count += 1
            set_f1_sum += metrics["set_f1"]
            json_count += metrics["json_ok"]
            full_quote_count += (
                metrics["json_ok"] and metrics["history_quote_rate"] == 1.0
            )
            no_duplicate_count += (
                metrics["json_ok"] and metrics["duplicate_count"] == 0
            )
            token_limit_count += "token_limit" in row["error_types"]
    denominator = max(candidate_count, 1)
    generation_samples = sum(len(outputs) for outputs in generations.values())
    aggregate = {
        "generation_samples": generation_samples,
        "unique_candidates": candidate_count,
        "duplicate_sample_rate": round(
            (generation_samples - candidate_count) / max(generation_samples, 1), 8
        ),
        "mean_set_f1": round(set_f1_sum / denominator, 8),
        "json_rate": round(json_count / denominator, 8),
        "full_quote_rate": round(full_quote_count / denominator, 8),
        "no_duplicate_rate": round(no_duplicate_count / denominator, 8),
        "token_limit_rate": round(token_limit_count / denominator, 8),
        "error_types": dict(summary),
    }
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def hard_negative_key(row: dict):
    metrics = row["metrics"]
    return (
        metrics["set_f1"],
        int(metrics["json_ok"]),
        metrics["history_quote_rate"],
        -metrics["duplicate_count"],
        int(metrics["history_ordered"]),
        -abs(metrics["predicted_count"] - metrics["gold_count"]),
    )


def build_pairs(args):
    if args.max_rejected_per_prompt < 1:
        raise ValueError("--max-rejected-per-prompt must be positive")
    gold_records = load_gold(Path(args.gold))
    generation_paths = [Path(path) for path in args.generations]
    generations = load_generations(generation_paths)
    if args.require_complete:
        require_complete_generations(gold_records, generations)
    grouped = defaultdict(list)
    for row in scored_candidates(gold_records, generations):
        metrics = row["metrics"]
        clean = (
            metrics["set_f1"] == 1.0
            and metrics["json_ok"]
            and metrics["history_quote_rate"] == 1.0
            and metrics["duplicate_count"] == 0
            and (not metrics["gold_history_ordered"] or metrics["history_ordered"])
        )
        if not clean:
            grouped[row["action_id"]].append(row)

    output_path = Path(args.out)
    audit_path = Path(args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    for action_id, rows in grouped.items():
        gold_record = gold_records[action_id]
        rows.sort(key=hard_negative_key, reverse=True)
        for row in rows[: args.max_rejected_per_prompt]:
            if row["output"] == gold_record["chosen"]:
                raise AssertionError(f"identical chosen/rejected pair: {action_id}")
            pairs.append(
                {
                    "instruction": gold_record["instruction"],
                    "input": gold_record["input"],
                    "chosen": gold_record["chosen"],
                    "rejected": row["output"],
                    "meta": {
                        "action_id": action_id,
                        "candidate_index": row["candidate_index"],
                        "metrics": row["metrics"],
                        "generation": row["generation"],
                        "error_types": row["error_types"],
                    },
                }
            )
    pairs.sort(key=lambda row: (row["meta"]["action_id"], row["meta"]["candidate_index"]))
    with output_path.open("w", encoding="utf-8") as output:
        for pair in pairs:
            output.write(json.dumps(pair, ensure_ascii=False) + "\n")

    error_summary = Counter(
        error for pair in pairs for error in pair["meta"]["error_types"]
    )
    selected_f1 = [pair["meta"]["metrics"]["set_f1"] for pair in pairs]
    audit = {
        "gold": str(Path(args.gold).resolve()),
        "generations": [str(path.resolve()) for path in generation_paths],
        "output": str(output_path.resolve()),
        "gold_prompts": len(gold_records),
        "prompts_with_generations": len(generations),
        "missing_gold_prompts": len(set(gold_records) - set(generations)),
        "prompts_with_negative": len(grouped),
        "prompts_without_negative": len(generations) - len(grouped),
        "negative_prompt_rate": round(len(grouped) / max(len(generations), 1), 8),
        "pairs": len(pairs),
        "max_rejected_per_prompt": args.max_rejected_per_prompt,
        "selected_set_f1": {
            "mean": round(sum(selected_f1) / max(len(selected_f1), 1), 8),
            "min": min(selected_f1, default=0.0),
            "p25": percentile(selected_f1, 0.25),
            "median": percentile(selected_f1, 0.5),
            "p75": percentile(selected_f1, 0.75),
            "p90": percentile(selected_f1, 0.9),
            "max": max(selected_f1, default=0.0),
        },
        "error_types": dict(error_summary),
        "sha256": file_hash(output_path, "sha256"),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--src", default="data/processed/data_final.jsonl")
    extract.add_argument("--out", default="data/processed/action_gold_v1.jsonl")
    extract.add_argument("--audit", default="logs/data/action_gold_v1_audit.json")
    extract.set_defaults(func=extract_gold)

    score = subparsers.add_parser("score")
    score.add_argument("--gold", default="data/processed/action_gold_v1.jsonl")
    score.add_argument("--generations", required=True, nargs="+")
    score.add_argument("--out", required=True)
    score.add_argument("--require-complete", action="store_true")
    score.set_defaults(func=score_generations)

    build = subparsers.add_parser("build")
    build.add_argument("--gold", default="data/processed/action_gold_v1.jsonl")
    build.add_argument("--generations", required=True, nargs="+")
    build.add_argument("--out", required=True)
    build.add_argument("--audit", required=True)
    build.add_argument("--max-rejected-per-prompt", required=True, type=int)
    build.add_argument("--require-complete", action="store_true")
    build.set_defaults(func=build_pairs)
    return parser


def main():
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
