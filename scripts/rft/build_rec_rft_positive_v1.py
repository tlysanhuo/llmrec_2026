#!/usr/bin/env python3
"""Build positive-only recommendation RFT rows from prompt-only rollouts.

Inputs are physically separated:

* the prompt manifest supplies only instruction/input text and stable IDs;
* the gold ledger supplies the complete O1 set of valid targets per
  (prompt-group, domain) unit;
* the rollout file supplies actually sampled reasoning and itemic candidates.

A group contributes at most one training row, and only when a structurally
valid generated trace has an actually sampled candidate that exactly matches
one member of the full gold set.  Misses, invalid candidates, prefix matches,
and other partial signals never become training examples or reward terms.
The accepted response is exactly ``<think>{generated CoT}</think>\n{hit}``;
no original-gold CoT, answer prose, or private marker is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT / "assets/derived/processed/o1_rec_multigold_v1_prompt_manifest.jsonl"
)
DEFAULT_GOLD = ROOT / "assets/derived/processed/o1_rec_multigold_v1_gold_ledger.jsonl"
DEFAULT_ROLLOUTS = ROOT / "assets/derived/processed/o1_rec_multigold_v1_rollouts.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_o1_rec_rft_positive_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/o1_rec_rft_positive_v1_audit.json"

SCHEMA_MANIFEST = "o1-rec-prompt-manifest-v1"
SCHEMA_GOLD = "o1-rec-gold-ledger-v1"
SCHEMA_ROLLOUTS = "o1-rec-rollouts-v1"
EXPECTED_GROUPS = 512
EXPECTED_TRACES_PER_GROUP = 4
EXPECTED_CANDIDATES_PER_TRACE = 8
MAX_REASONING_TOKENS = 1_024
MIN_ACCEPTED_YIELD_GATE = 128
MIN_VALID_CANDIDATE_RATE = 0.95

HEX64_RE = re.compile(r"[0-9a-f]{64}")
ITEM_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
ITEM_FRAGMENT_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|>|<s_[abc]_\d+>"
)
CHAT_CONTROL_RE = re.compile(r"<\|im_(?:start|end)\|>")

MANIFEST_KEYS = {
    "schema_version",
    "group_id",
    "instruction",
    "input",
    "history",
    "domain",
    "prompt_sha256",
    "rollout_seed",
}
GOLD_KEYS = {
    "schema_version",
    "group_id",
    "prompt_sha256",
    "domain",
    "prompt_group_id",
    "source_prompt_group_size",
    "source_group_size",
    "gold_count",
    "golds",
    "original_thought_sha256s",
    "original_thought_stripped_sha256s",
}
GOLD_ENTRY_KEYS = {
    "itemic",
    "itemic_sha256",
    "answer",
    "output_prefix",
    "output_suffix",
    "output_shell_sha256",
    "source_row_indices",
    "source_row_sha256s",
    "target_in_prompt",
}
ROLLOUT_KEYS = {
    "schema_version",
    "group_id",
    "prompt_sha256",
    "domain",
    "generator",
    "traces",
}
GENERATOR_KEYS = {
    "config_sha256",
    "base_sha256",
    "adapter_sha256",
    "seed",
}
TRACE_KEYS = {
    "trace_id",
    "reasoning_index",
    "thought",
    "reasoning",
    "candidates",
}
REASONING_KEYS = {
    "text",
    "raw_text",
    "finish_reason",
    "stop_reason",
    "token_count",
    "seed",
}
CANDIDATE_KEYS = {
    "text",
    "item",
    "valid",
    "finish_reason",
    "stop_reason",
    "token_count",
    "cumulative_logprob",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: Any) -> str:
    return text_sha256(canonical_json(list(parts)))


def make_prompt_sha256(instruction: str, input_text: str) -> str:
    return text_sha256(
        canonical_json(
            {"history": [], "input": input_text, "instruction": instruction}
        )
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} schema drift: missing={missing}, extra={extra}")


def require_hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
        raise ValueError(f"{context} is not a lowercase 64-hex SHA256")
    return value


def require_optional_reason(value: Any, context: str) -> None:
    """Match vLLM's public finish/stop reason surface without coercion."""
    if value is not None and not (
        isinstance(value, (str, int)) and not isinstance(value, bool)
    ):
        raise ValueError(f"{context} must be string, integer, or null")


def parse_item(value: str, context: str) -> tuple[str, tuple[int, int, int]]:
    match = ITEM_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{context} is not one exact itemic: {value!r}")
    components = (int(match.group("a")), int(match.group("b")), int(match.group("c")))
    if any(component < 0 or component > 8191 for component in components):
        raise ValueError(f"{context} itemic component outside 0..8191")
    return match.group("domain"), components


def validate_manifest(rows: list[dict[str, Any]], expected_groups: int | None) -> dict[str, dict[str, Any]]:
    if expected_groups is not None and len(rows) != expected_groups:
        raise AssertionError(f"expected {expected_groups} manifest rows, got {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        context = f"manifest[{index}]"
        require_exact_keys(row, MANIFEST_KEYS, context)
        if row["schema_version"] != SCHEMA_MANIFEST:
            raise ValueError(f"{context} schema_version mismatch")
        group_id = require_hex64(row["group_id"], f"{context}.group_id")
        require_hex64(row["prompt_sha256"], f"{context}.prompt_sha256")
        if group_id in by_id:
            raise ValueError(f"duplicate manifest group_id: {group_id}")
        if row["domain"] not in {"video", "prod", "ad", "living"}:
            raise ValueError(f"{context} invalid domain")
        if not isinstance(row["instruction"], str) or not isinstance(row["input"], str):
            raise ValueError(f"{context} prompt fields must be strings")
        if row["history"] != []:
            raise ValueError(f"{context} history must be empty")
        if not row["input"].rstrip().endswith("/think"):
            raise ValueError(f"{context} is not a /think rollout prompt")
        if make_prompt_sha256(row["instruction"], row["input"]) != row["prompt_sha256"]:
            raise ValueError(f"{context} prompt_sha256 does not reproduce")
        if isinstance(row["rollout_seed"], bool) or not isinstance(row["rollout_seed"], int):
            raise ValueError(f"{context} rollout_seed must be int")
        by_id[group_id] = row
    return by_id


def validate_gold(rows: list[dict[str, Any]], expected_groups: int | None) -> dict[str, dict[str, Any]]:
    if expected_groups is not None and len(rows) != expected_groups:
        raise AssertionError(f"expected {expected_groups} gold rows, got {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        context = f"gold[{index}]"
        require_exact_keys(row, GOLD_KEYS, context)
        if row["schema_version"] != SCHEMA_GOLD:
            raise ValueError(f"{context} schema_version mismatch")
        group_id = require_hex64(row["group_id"], f"{context}.group_id")
        require_hex64(row["prompt_sha256"], f"{context}.prompt_sha256")
        require_hex64(row["prompt_group_id"], f"{context}.prompt_group_id")
        if group_id in by_id:
            raise ValueError(f"duplicate gold group_id: {group_id}")
        domain = row["domain"]
        if domain not in {"video", "prod", "ad", "living"}:
            raise ValueError(f"{context} invalid domain")
        if not isinstance(row["golds"], list) or not row["golds"]:
            raise ValueError(f"{context}.golds must be a non-empty list")
        if row["gold_count"] != len(row["golds"]):
            raise ValueError(f"{context}.gold_count mismatch")
        items: set[str] = set()
        for gold_index, gold in enumerate(row["golds"]):
            gold_context = f"{context}.golds[{gold_index}]"
            if not isinstance(gold, dict):
                raise ValueError(f"{gold_context} must be object")
            require_exact_keys(gold, GOLD_ENTRY_KEYS, gold_context)
            item = gold["itemic"]
            item_domain, _ = parse_item(item, f"{gold_context}.itemic")
            if item_domain != domain:
                raise ValueError(f"{gold_context} crosses requested domain")
            if text_sha256(item) != gold["itemic_sha256"]:
                raise ValueError(f"{gold_context}.itemic_sha256 mismatch")
            if item in items:
                raise ValueError(f"{context} contains duplicate gold itemic")
            items.add(item)
            if not isinstance(gold["target_in_prompt"], bool):
                raise ValueError(f"{gold_context}.target_in_prompt must be bool")
            if not isinstance(gold["source_row_indices"], list) or not gold[
                "source_row_indices"
            ]:
                raise ValueError(f"{gold_context} lacks source row provenance")
            if not isinstance(gold["source_row_sha256s"], list) or not gold[
                "source_row_sha256s"
            ]:
                raise ValueError(f"{gold_context} lacks source hash provenance")
            for digest in gold["source_row_sha256s"]:
                require_hex64(digest, f"{gold_context}.source_row_sha256s")
            expected_shell = gold["output_prefix"] + "{thought}" + gold["output_suffix"]
            if text_sha256(expected_shell) != gold["output_shell_sha256"]:
                raise ValueError(f"{gold_context}.output_shell_sha256 mismatch")
        for field in ("original_thought_sha256s", "original_thought_stripped_sha256s"):
            if not isinstance(row[field], list):
                raise ValueError(f"{context}.{field} must be list")
            for digest in row[field]:
                require_hex64(digest, f"{context}.{field}")
        by_id[group_id] = row
    return by_id


def validate_generator(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be object")
    require_exact_keys(value, GENERATOR_KEYS, context)
    for field in ("config_sha256", "base_sha256", "adapter_sha256"):
        require_hex64(value[field], f"{context}.{field}")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError(f"{context}.seed must be int")
    return value


def validate_candidate(value: Any, domain: str, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be object")
    require_exact_keys(value, CANDIDATE_KEYS, context)
    if not isinstance(value["text"], str):
        raise ValueError(f"{context}.text must be string")
    if not isinstance(value["valid"], bool):
        raise ValueError(f"{context}.valid must be bool")
    require_optional_reason(value["finish_reason"], f"{context}.finish_reason")
    require_optional_reason(value["stop_reason"], f"{context}.stop_reason")
    if isinstance(value["token_count"], bool) or not isinstance(value["token_count"], int):
        raise ValueError(f"{context}.token_count must be int")
    if value["token_count"] < 0:
        raise ValueError(f"{context}.token_count must be non-negative")
    logprob = value["cumulative_logprob"]
    if logprob is not None:
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            raise ValueError(f"{context}.cumulative_logprob must be numeric or null")
        if not math.isfinite(float(logprob)):
            raise ValueError(f"{context}.cumulative_logprob must be finite")

    if value["valid"]:
        if not isinstance(value["item"], str):
            raise ValueError(f"{context}.item must be string when valid")
        if value["text"] != value["item"]:
            raise ValueError(
                f"{context} valid text must be the exact full generated itemic"
            )
        candidate_domain, _ = parse_item(value["item"], f"{context}.item")
        if candidate_domain != domain:
            raise ValueError(f"{context} valid candidate crosses requested domain")
    elif value["item"] is not None:
        raise ValueError(f"{context}.item must be null when invalid")


def validate_trace(value: Any, domain: str, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be object")
    require_exact_keys(value, TRACE_KEYS, context)
    if not isinstance(value["trace_id"], str) or not value["trace_id"]:
        raise ValueError(f"{context}.trace_id must be a non-empty string")
    if isinstance(value["reasoning_index"], bool) or not isinstance(
        value["reasoning_index"], int
    ):
        raise ValueError(f"{context}.reasoning_index must be int")
    if not isinstance(value["thought"], str):
        raise ValueError(f"{context}.thought must be string")
    reasoning = value["reasoning"]
    if not isinstance(reasoning, dict):
        raise ValueError(f"{context}.reasoning must be object")
    require_exact_keys(reasoning, REASONING_KEYS, f"{context}.reasoning")
    if reasoning["text"] != value["thought"]:
        raise ValueError(f"{context} reasoning.text must equal thought exactly")
    if not isinstance(reasoning["raw_text"], str):
        raise ValueError(f"{context}.reasoning.raw_text must be string")
    if value["thought"] not in reasoning["raw_text"]:
        raise ValueError(f"{context}.reasoning.raw_text does not contain thought")
    require_optional_reason(reasoning["finish_reason"], f"{context}.reasoning.finish_reason")
    require_optional_reason(reasoning["stop_reason"], f"{context}.reasoning.stop_reason")
    for field in ("token_count", "seed"):
        if isinstance(reasoning[field], bool) or not isinstance(reasoning[field], int):
            raise ValueError(f"{context}.reasoning.{field} must be int")
    if reasoning["token_count"] < 0:
        raise ValueError(f"{context}.reasoning.token_count must be non-negative")
    if not isinstance(value["candidates"], list):
        raise ValueError(f"{context}.candidates must be list")
    if len(value["candidates"]) != EXPECTED_CANDIDATES_PER_TRACE:
        raise ValueError(
            f"{context} expected {EXPECTED_CANDIDATES_PER_TRACE} candidates, "
            f"got {len(value['candidates'])}"
        )
    for candidate_index, candidate in enumerate(value["candidates"]):
        validate_candidate(candidate, domain, f"{context}.candidates[{candidate_index}]")


def validate_rollouts(
    rows: list[dict[str, Any]], expected_groups: int | None
) -> dict[str, dict[str, Any]]:
    if expected_groups is not None and len(rows) != expected_groups:
        raise AssertionError(f"expected {expected_groups} rollout rows, got {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    generator_signatures: set[str] = set()
    for index, row in enumerate(rows):
        context = f"rollouts[{index}]"
        require_exact_keys(row, ROLLOUT_KEYS, context)
        if row["schema_version"] != SCHEMA_ROLLOUTS:
            raise ValueError(f"{context} schema_version mismatch")
        group_id = require_hex64(row["group_id"], f"{context}.group_id")
        require_hex64(row["prompt_sha256"], f"{context}.prompt_sha256")
        if group_id in by_id:
            raise ValueError(f"duplicate rollout group_id: {group_id}")
        if row["domain"] not in {"video", "prod", "ad", "living"}:
            raise ValueError(f"{context} invalid domain")
        generator = validate_generator(row["generator"], f"{context}.generator")
        generator_signatures.add(canonical_json(generator))
        if not isinstance(row["traces"], list):
            raise ValueError(f"{context}.traces must be list")
        if len(row["traces"]) != EXPECTED_TRACES_PER_GROUP:
            raise ValueError(
                f"{context} expected {EXPECTED_TRACES_PER_GROUP} traces, "
                f"got {len(row['traces'])}"
            )
        indices: list[int] = []
        trace_ids: set[str] = set()
        for trace_index, trace in enumerate(row["traces"]):
            validate_trace(trace, row["domain"], f"{context}.traces[{trace_index}]")
            indices.append(trace["reasoning_index"])
            if trace["trace_id"] in trace_ids:
                raise ValueError(f"{context} duplicate trace_id")
            trace_ids.add(trace["trace_id"])
        if sorted(indices) != list(range(EXPECTED_TRACES_PER_GROUP)):
            raise ValueError(f"{context} reasoning_index must be exactly 0..3")
        by_id[group_id] = row
    if len(generator_signatures) != 1:
        raise ValueError("rollout rows do not share one frozen generator identity")
    return by_id


def thought_itemics_are_grounded(thought: str, prompt_input: str) -> bool:
    matches = list(ITEM_RE.finditer(thought))
    masked = list(thought)
    for match in matches:
        masked[match.start() : match.end()] = " " * (match.end() - match.start())
    residue = "".join(masked)
    if ITEM_FRAGMENT_RE.search(residue):
        return False
    # Also catch incomplete fragments that do not reach a closing ``>`` and
    # therefore would not be recognized by ITEM_FRAGMENT_RE.
    if any(
        marker in residue
        for marker in (
            "<|video_begin|>",
            "<|prod_begin|>",
            "<|ad_begin|>",
            "<|living_begin|>",
            "<s_a_",
            "<s_b_",
            "<s_c_",
        )
    ):
        return False
    prompt_items = {match.group(0) for match in ITEM_RE.finditer(prompt_input)}
    return all(match.group(0) in prompt_items for match in matches)


def trace_qc_reasons(
    trace: dict[str, Any], manifest: dict[str, Any], gold: dict[str, Any]
) -> list[str]:
    thought = trace["thought"]
    reasons: list[str] = []
    if not thought.strip():
        reasons.append("empty_thought")
    if "<think>" in thought or "</think>" in thought:
        reasons.append("nested_think_tag")
    if CHAT_CONTROL_RE.search(thought):
        reasons.append("chat_control_token")
    if trace["reasoning"]["token_count"] <= 0:
        reasons.append("nonpositive_reasoning_token_count")
    if trace["reasoning"]["token_count"] > MAX_REASONING_TOKENS:
        reasons.append("reasoning_too_long")
    # Do not turn a max-token/EOS-truncated rollout into apparently complete
    # supervision merely because the generator can append an artificial close
    # tag before the item beam.  The sampled trace itself must have reached the
    # explicit </think> stop string.
    if not (
        trace["reasoning"]["finish_reason"] == "stop"
        and trace["reasoning"]["stop_reason"] == "</think>"
    ):
        reasons.append("reasoning_not_closed_by_think_stop")
    gold_items = {entry["itemic"] for entry in gold["golds"]}
    if any(item in thought for item in gold_items):
        reasons.append("gold_item_in_thought")
    if not thought_itemics_are_grounded(thought, manifest["input"]):
        reasons.append("ungrounded_or_malformed_thought_itemic")
    if text_sha256(thought) in set(gold["original_thought_sha256s"]):
        reasons.append("exact_original_o1_thought")
    if text_sha256(thought.strip()) in set(gold["original_thought_stripped_sha256s"]):
        reasons.append("stripped_original_o1_thought")
    return sorted(set(reasons))


def first_a_diversity(items: Iterable[str]) -> int:
    values: set[int] = set()
    for item in items:
        _, components = parse_item(item, "candidate diversity item")
        values.add(components[0])
    return len(values)


def choose_positive_trace(
    rollout: dict[str, Any], manifest: dict[str, Any], gold: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    gold_items = {entry["itemic"] for entry in gold["golds"]}
    passing: list[dict[str, Any]] = []
    trace_rejections: Counter[str] = Counter()
    valid_candidate_count = 0
    exact_hit_candidate_count = 0

    for trace in rollout["traces"]:
        valid_items = [
            candidate["item"] for candidate in trace["candidates"] if candidate["valid"]
        ]
        valid_candidate_count += len(valid_items)
        hit_candidates = [
            (candidate_index, candidate["item"])
            for candidate_index, candidate in enumerate(trace["candidates"])
            if candidate["valid"] and candidate["item"] in gold_items
        ]
        exact_hit_candidate_count += len(hit_candidates)
        reasons = trace_qc_reasons(trace, manifest, gold)
        if not hit_candidates:
            reasons.append("no_exact_set_hit")
        if reasons:
            trace_rejections.update(set(reasons))
            continue
        distinct_valid_items = set(valid_items)
        distinct_hit_items = {item for _, item in hit_candidates}
        passing.append(
            {
                "trace": trace,
                "hit_candidates": hit_candidates,
                "distinct_hit_count": len(distinct_hit_items),
                "distinct_valid_count": len(distinct_valid_items),
                "first_a_diversity": first_a_diversity(distinct_valid_items),
                "valid_count": len(valid_items),
            }
        )

    passing.sort(
        key=lambda entry: (
            -entry["distinct_hit_count"],
            -entry["first_a_diversity"],
            -entry["distinct_valid_count"],
            -entry["valid_count"],
            entry["trace"]["reasoning"]["token_count"],
            stable_hash(rollout["group_id"], entry["trace"]["trace_id"]),
        )
    )
    selected: dict[str, Any] | None = None
    if passing:
        best = passing[0]
        candidate_index, item = best["hit_candidates"][0]
        selected = {
            **best,
            "candidate_index": candidate_index,
            "item": item,
        }
    diagnostics = {
        "trace_rejections": dict(sorted(trace_rejections.items())),
        "valid_candidate_count": valid_candidate_count,
        "exact_hit_candidate_count": exact_hit_candidate_count,
        "passing_positive_traces": len(passing),
    }
    return selected, diagnostics


def build_positive_rows(
    manifest_by_id: dict[str, dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    rollout_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = set(manifest_by_id)
    if manifest_ids != set(gold_by_id) or manifest_ids != set(rollout_by_id):
        raise AssertionError(
            "manifest/gold/rollout group sets differ: "
            f"manifest={len(manifest_ids)}, gold={len(gold_by_id)}, "
            f"rollout={len(rollout_by_id)}"
        )

    output_rows: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    group_rejections: Counter[str] = Counter()
    trace_rejections: Counter[str] = Counter()
    domain_groups: Counter[str] = Counter()
    domain_accepted: Counter[str] = Counter()
    total_traces = 0
    total_candidates = 0
    valid_candidates = 0
    exact_hit_candidates = 0

    for group_id in sorted(manifest_ids):
        manifest = manifest_by_id[group_id]
        gold = gold_by_id[group_id]
        rollout = rollout_by_id[group_id]
        if not (
            manifest["prompt_sha256"]
            == gold["prompt_sha256"]
            == rollout["prompt_sha256"]
        ):
            raise AssertionError(f"{group_id}: prompt SHA mismatch across partitions")
        if not (manifest["domain"] == gold["domain"] == rollout["domain"]):
            raise AssertionError(f"{group_id}: domain mismatch across partitions")
        domain = manifest["domain"]
        domain_groups[domain] += 1
        total_traces += len(rollout["traces"])
        total_candidates += sum(len(trace["candidates"]) for trace in rollout["traces"])

        selected, diagnostics = choose_positive_trace(rollout, manifest, gold)
        valid_candidates += diagnostics["valid_candidate_count"]
        exact_hit_candidates += diagnostics["exact_hit_candidate_count"]
        trace_rejections.update(diagnostics["trace_rejections"])
        if selected is None:
            if diagnostics["exact_hit_candidate_count"] == 0:
                group_rejections["no_exact_set_hit"] += 1
            else:
                group_rejections["all_exact_hit_traces_failed_qc"] += 1
            continue

        trace = selected["trace"]
        item = selected["item"]
        actual_candidate = trace["candidates"][selected["candidate_index"]]
        if not actual_candidate["valid"] or actual_candidate["text"] != item:
            raise AssertionError(f"{group_id}: selected item is not an actual valid candidate")
        if item not in {entry["itemic"] for entry in gold["golds"]}:
            raise AssertionError(f"{group_id}: selected item is not in full gold set")
        output = f"<think>{trace['thought']}</think>\n{item}"
        if not trace["thought"].strip() or not output.startswith("<think>"):
            raise AssertionError(f"{group_id}: accepted output lacks generated thought")
        if output.count("<think>") != 1 or output.count("</think>") != 1:
            raise AssertionError(f"{group_id}: accepted output think structure drifted")

        training_row = {
            "instruction": manifest["instruction"],
            "input": manifest["input"],
            "output": output,
            "history": [],
        }
        output_rows.append(training_row)
        domain_accepted[domain] += 1
        accepted_records.append(
            {
                "group_id": group_id,
                "prompt_sha256": manifest["prompt_sha256"],
                "domain": domain,
                "trace_id": trace["trace_id"],
                "reasoning_index": trace["reasoning_index"],
                "candidate_index": selected["candidate_index"],
                "actual_hit_item_sha256": text_sha256(item),
                "output_sha256": text_sha256(canonical_json(training_row)),
                "gold_set_size": gold["gold_count"],
                "passing_positive_traces": diagnostics["passing_positive_traces"],
                "distinct_hit_count": selected["distinct_hit_count"],
                "distinct_valid_count": selected["distinct_valid_count"],
                "first_a_diversity": selected["first_a_diversity"],
                "valid_candidate_count": selected["valid_count"],
                "reasoning_token_count": trace["reasoning"]["token_count"],
            }
        )

    if len(output_rows) != len(accepted_records):
        raise AssertionError("accepted output/provenance count mismatch")
    if len({record["group_id"] for record in accepted_records}) != len(accepted_records):
        raise AssertionError("more than one accepted row was emitted for a group")
    return output_rows, {
        "source_groups": len(manifest_ids),
        "domain_source_groups": dict(sorted(domain_groups.items())),
        "traces": total_traces,
        "candidates": total_candidates,
        "valid_candidates": valid_candidates,
        "exact_set_hit_candidates": exact_hit_candidates,
        "accepted_unique_groups": len(output_rows),
        "acceptance_rate": round(len(output_rows) / len(manifest_ids), 8)
        if manifest_ids
        else 0.0,
        "domain_accepted_groups": dict(sorted(domain_accepted.items())),
        "group_rejections": dict(sorted(group_rejections.items())),
        "trace_rejections": dict(sorted(trace_rejections.items())),
        "accepted_output_qc_violations": {
            "gold_item_in_thought": 0,
            "original_O1_thought": 0,
            "unexpected_control_string_leakage": 0,
        },
        "accepted_records": accepted_records,
    }


def preflight_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("accepted data and audit paths must be distinct")
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing output(s): "
                + ", ".join(str(path) for path in existing)
            )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(canonical_json(row) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build(args: argparse.Namespace, expected_groups: int | None = EXPECTED_GROUPS) -> dict[str, Any]:
    preflight_outputs((args.out, args.audit), args.overwrite)
    manifest_rows = read_jsonl(args.manifest)
    gold_rows = read_jsonl(args.gold_ledger)
    rollout_rows = read_jsonl(args.rollouts)
    manifest_by_id = validate_manifest(manifest_rows, expected_groups)
    gold_by_id = validate_gold(gold_rows, expected_groups)
    rollout_by_id = validate_rollouts(rollout_rows, expected_groups)
    output_rows, construction = build_positive_rows(
        manifest_by_id, gold_by_id, rollout_by_id
    )
    write_jsonl(args.out, output_rows)

    generator = next(iter(rollout_by_id.values()))["generator"] if rollout_by_id else None
    accepted = construction["accepted_unique_groups"]
    yield_gate_pass = accepted >= MIN_ACCEPTED_YIELD_GATE
    valid_candidate_rate = (
        construction["valid_candidates"] / construction["candidates"]
        if construction["candidates"]
        else 0.0
    )
    valid_gate_pass = valid_candidate_rate >= MIN_VALID_CANDIDATE_RATE
    total_output_characters = sum(len(row["output"]) for row in output_rows)
    audit = {
        "asset_class": "D(O1; M-teacher rollout)",
        "purpose": "strict exact-set-hit positive-only recommendation RFT-lite",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": file_sha256(Path(__file__)),
        "upstream": {
            "prompt_manifest": {
                "path": str(args.manifest.resolve()),
                "rows": len(manifest_rows),
                "sha256": file_sha256(args.manifest),
                "schema_version": SCHEMA_MANIFEST,
            },
            "gold_ledger": {
                "path": str(args.gold_ledger.resolve()),
                "rows": len(gold_rows),
                "sha256": file_sha256(args.gold_ledger),
                "schema_version": SCHEMA_GOLD,
            },
            "rollouts": {
                "path": str(args.rollouts.resolve()),
                "rows": len(rollout_rows),
                "sha256": file_sha256(args.rollouts),
                "schema_version": SCHEMA_ROLLOUTS,
                "generator": generator,
            },
        },
        "construction": construction,
        "reward_contract": {
            "positive_condition": "actual valid generated candidate is an exact member of full O1 group-domain gold set",
            "partial_reward_terms": [],
            "misses_used_as_negatives": False,
            "invalid_candidates_used_for_training": False,
            "rows_per_group_cap": 1,
            "selection_uses_logprob": False,
        },
        "output_contract": {
            "format": "<think>{actual_generated_nonempty_CoT}</think>\\n{actual_exact_hit_itemic}",
            "original_O1_CoT_written": False,
            "answer_prose_written": False,
            "private_markers_written": False,
            "all_selected_targets_are_actual_candidates": True,
            "all_selected_targets_are_full_set_exact_hits": True,
            "unique_group_rows": True,
        },
        "mix": {
            "rows": len(output_rows),
            "domain_rows": construction["domain_accepted_groups"],
            "RFT_positive_ratio": 1.0 if output_rows else 0.0,
            "O1_derived_ratio": 1.0 if output_rows else 0.0,
            "T_ratio": 0.0,
            "E_ratio": 0.0,
            "partial_or_negative_ratio": 0.0,
            "supervised_output_characters": total_output_characters,
        },
        "yield_gate": {
            "minimum_accepted_unique_groups": MIN_ACCEPTED_YIELD_GATE,
            "observed": accepted,
            "status": "PASS" if yield_gate_pass else "FAIL",
            "gate_does_not_relax_exact_hit_or_QC": True,
        },
        "valid_candidate_gate": {
            "minimum_valid_candidates_over_total_candidates": MIN_VALID_CANDIDATE_RATE,
            "valid_candidates": construction["valid_candidates"],
            "total_candidates": construction["candidates"],
            "observed_rate": round(valid_candidate_rate, 8),
            "status": "PASS" if valid_gate_pass else "FAIL",
        },
        "forbidden_sources": {
            "O2_rows": 0,
            "O3_rows_or_target_metadata": 0,
            "T_rows": 0,
            "E_rows_prompts_answers_or_logs": 0,
        },
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(output_rows),
            "sha256": file_sha256(args.out),
        },
        "formal_training_ready": False,
        "formal_training_blockers": [
            "register derived asset and mix in docs/reference/ASSETS.md",
            "register experiment config/ledger and approved parent role",
        ]
        + ([] if yield_gate_pass else ["positive-only yield gate failed"])
        + ([] if valid_gate_pass else ["valid-candidate-rate gate failed"]),
    }
    write_json(args.audit, audit)
    return audit


def synthetic_item(domain: str, a: int, b: int, c: int) -> str:
    return f"<|{domain}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def synthetic_partitions() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    generator = {
        "config_sha256": "1" * 64,
        "base_sha256": "2" * 64,
        "adapter_sha256": "3" * 64,
        "seed": 123,
    }
    for index in range(3):
        domain = "video"
        instruction = "synthetic recommendation instruction"
        history_item = synthetic_item(domain, 100 + index, 200 + index, 300 + index)
        input_text = f"synthetic history {history_item} /think"
        prompt_sha = make_prompt_sha256(instruction, input_text)
        group_id = text_sha256(f"synthetic-group-{index}")
        prompt_group_id = text_sha256(f"synthetic-prompt-group-{index}")
        gold_item = synthetic_item(domain, 10 + index, 20 + index, 30 + index)
        miss_item = synthetic_item(domain, 10 + index, 20 + index, 99)
        manifest_rows.append(
            {
                "schema_version": SCHEMA_MANIFEST,
                "group_id": group_id,
                "instruction": instruction,
                "input": input_text,
                "history": [],
                "domain": domain,
                "prompt_sha256": prompt_sha,
                "rollout_seed": 1000 + index,
            }
        )
        original_thought = f"original thought {index}"
        gold_rows.append(
            {
                "schema_version": SCHEMA_GOLD,
                "group_id": group_id,
                "prompt_sha256": prompt_sha,
                "domain": domain,
                "prompt_group_id": prompt_group_id,
                "source_prompt_group_size": 2,
                "source_group_size": 2,
                "gold_count": 1,
                "golds": [
                    {
                        "itemic": gold_item,
                        "itemic_sha256": text_sha256(gold_item),
                        "answer": gold_item,
                        "output_prefix": "<think>",
                        "output_suffix": f"</think>\n{gold_item}",
                        "output_shell_sha256": text_sha256(
                            f"<think>{{thought}}</think>\n{gold_item}"
                        ),
                        "source_row_indices": [index],
                        "source_row_sha256s": [text_sha256(f"source-{index}")],
                        "target_in_prompt": False,
                    }
                ],
                "original_thought_sha256s": [text_sha256(original_thought)],
                "original_thought_stripped_sha256s": [text_sha256(original_thought)],
            }
        )

        traces: list[dict[str, Any]] = []
        for reasoning_index in range(EXPECTED_TRACES_PER_GROUP):
            if index == 0 and reasoning_index == 0:
                thought = f"new grounded reasoning with {history_item}"
                candidate_item = gold_item
            elif index == 1:
                # Prefix-related miss only; it must never be accepted.
                thought = f"new grounded miss reasoning with {history_item}"
                candidate_item = miss_item
            elif index == 2 and reasoning_index == 0:
                # Exact hit exists, but the thought leaks the target and must fail QC.
                thought = f"leaked target {gold_item}"
                candidate_item = gold_item
            else:
                thought = f"other generated reasoning {index}-{reasoning_index}"
                candidate_item = miss_item
            candidates = []
            for candidate_index in range(EXPECTED_CANDIDATES_PER_TRACE):
                item = candidate_item if candidate_index == 0 else miss_item
                candidates.append(
                    {
                        "text": item,
                        "item": item,
                        "valid": True,
                        "finish_reason": "length",
                        "stop_reason": 7 if candidate_index == 0 else None,
                        "token_count": 3,
                        "cumulative_logprob": -1.0 - candidate_index,
                    }
                )
            traces.append(
                {
                    "trace_id": f"trace-{index}-{reasoning_index}",
                    "reasoning_index": reasoning_index,
                    "thought": thought,
                    "reasoning": {
                        "text": thought,
                        "raw_text": thought + "</think>",
                        "finish_reason": "stop",
                        "stop_reason": "</think>",
                        "token_count": 12,
                        "seed": 2000 + reasoning_index,
                    },
                    "candidates": candidates,
                }
            )
        rollout_rows.append(
            {
                "schema_version": SCHEMA_ROLLOUTS,
                "group_id": group_id,
                "prompt_sha256": prompt_sha,
                "domain": domain,
                "generator": generator,
                "traces": traces,
            }
        )
    return manifest_rows, gold_rows, rollout_rows


def self_test() -> dict[str, Any]:
    manifest_rows, gold_rows, rollout_rows = synthetic_partitions()
    manifest_by_id = validate_manifest(manifest_rows, expected_groups=3)
    gold_by_id = validate_gold(gold_rows, expected_groups=3)
    rollout_by_id = validate_rollouts(rollout_rows, expected_groups=3)
    output_rows, construction = build_positive_rows(
        manifest_by_id, gold_by_id, rollout_by_id
    )
    if len(output_rows) != 1:
        raise AssertionError(f"expected exactly one synthetic acceptance: {construction}")
    accepted = output_rows[0]
    if not accepted["output"].startswith("<think>new grounded reasoning"):
        raise AssertionError("accepted row did not preserve actual generated thought")
    if "该用户最近" in accepted["output"] or "meta" in accepted:
        raise AssertionError("accepted row contains answer prose or a private marker")
    if construction["group_rejections"].get("no_exact_set_hit") != 1:
        raise AssertionError("prefix-related miss was not rejected as an exact-set miss")
    if construction["group_rejections"].get("all_exact_hit_traces_failed_qc") != 1:
        raise AssertionError("target-in-thought exact hit did not fail QC")

    truncated_trace = dict(rollout_rows[0]["traces"][0])
    truncated_trace["reasoning"] = dict(truncated_trace["reasoning"])
    truncated_trace["reasoning"]["finish_reason"] = "length"
    truncated_trace["reasoning"]["stop_reason"] = None
    if "reasoning_not_closed_by_think_stop" not in trace_qc_reasons(
        truncated_trace, manifest_rows[0], gold_rows[0]
    ):
        raise AssertionError("truncated reasoning was not rejected")

    with tempfile.TemporaryDirectory(prefix="rec_rft_positive_v1_") as directory:
        temp_root = Path(directory)
        outputs = (temp_root / "accepted.jsonl", temp_root / "audit.json")
        preflight_outputs(outputs, overwrite=False)
        write_jsonl(outputs[0], output_rows)
        write_json(outputs[1], {"ok": True})
        try:
            preflight_outputs(outputs, overwrite=False)
        except FileExistsError:
            overwrite_guard = True
        else:
            raise AssertionError("overwrite guard did not reject existing outputs")

    result = {
        "status": "PASS",
        "synthetic_groups": 3,
        "accepted_groups": 1,
        "partial_prefix_miss_rejected": True,
        "gold_in_thought_rejected": True,
        "truncated_reasoning_rejected": True,
        "actual_generated_thought_and_hit_preserved": True,
        "one_row_per_group": True,
        "overwrite_guard": overwrite_guard,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold-ledger", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    audit = build(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
