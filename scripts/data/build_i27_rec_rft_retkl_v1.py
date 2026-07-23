#!/usr/bin/env python3
"""Build the I-27 recommendation-RFT and frozen-I23 retention mixture.

The positive input is a separately accepted JSONL.  Every accepted row must
contain one non-empty actual think trace followed by exactly one four-token
recommendation answer.  Positive rows are copied unchanged, exactly once.

The only retention source is the registered ``data_user_residual_retention_v1``
asset.  Three retention occurrences are selected per positive.  Every selected
row is converted to the canonical empty-think form and is intended solely for
frozen-I23 KL, including rows that were supervised action/topic examples in
the older I-12 mixture.  No prompt-side private route marker is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_seed_scoremax_v1 import MODE_SUFFIX_RE, task_of


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETENTION = ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl"
DEFAULT_OUT = ROOT / "assets/derived/processed/data_i27_rec_rft_retkl_v1.jsonl"
DEFAULT_AUDIT = ROOT / "logs/data/i27_rec_rft_retkl_v1_audit.json"

REGISTERED_RETENTION_ROWS = 6_106
REGISTERED_RETENTION_SHA256 = (
    "bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0"
)
CORE_KEYS = {"instruction", "input", "output", "history"}
CANONICAL_EMPTY_PREFIX = "<think></think>\n"

RETENTION_TASK_ORDER = (
    "material_desc2sid",
    "material_sid2desc",
    "action",
    "topic",
    "rec_video",
    "rec_prod",
    "rec_ad",
    "rec_living",
    "world",
)
EXPECTED_RETENTION_TASK_COUNTS = {
    "action": 1_752,
    "topic": 1_301,
    "material_desc2sid": 281,
    "material_sid2desc": 281,
    "rec_video": 565,
    "rec_prod": 565,
    "rec_ad": 565,
    "rec_living": 565,
    "world": 231,
}

ITEM_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rows_multiset_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(row_key(row) for row in rows):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_row(raw: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    row = {
        "instruction": str(raw.get("instruction", raw.get("system", "")) or ""),
        "input": str(raw.get("input", raw.get("prompt", "")) or ""),
        "output": str(raw.get("output", raw.get("response", "")) or ""),
        "history": raw.get("history") or [],
    }
    if not isinstance(row["history"], list):
        raise ValueError(f"history must be a list at {path}:{line_number}")
    if not row["input"] or not row["output"]:
        raise ValueError(f"empty input/output at {path}:{line_number}")
    return row


def load_jsonl(path: Path, *, strict_positive: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix != ".jsonl":
        raise ValueError(f"expected a .jsonl input: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            if strict_positive:
                if set(raw) != CORE_KEYS:
                    raise ValueError(
                        f"accepted positive schema must be exactly {sorted(CORE_KEYS)} "
                        f"at {path}:{line_number}; got {sorted(raw)}"
                    )
                if not all(isinstance(raw[key], str) for key in ("instruction", "input", "output")):
                    raise ValueError(f"positive text fields must be strings at {path}:{line_number}")
                if not isinstance(raw["history"], list):
                    raise ValueError(f"positive history must be a list at {path}:{line_number}")
                if not raw["input"] or not raw["output"]:
                    raise ValueError(f"empty positive input/output at {path}:{line_number}")
                rows.append(dict(raw))
            else:
                rows.append(normalize_row(raw, path, line_number))
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def positive_task(row: dict[str, Any]) -> str:
    """Validate a fail-closed actual-CoT positive and return its rec task."""

    output = row["output"]
    if output.count("<think>") != 1 or output.count("</think>") != 1:
        raise ValueError("positive must contain exactly one <think> and one </think>")
    if not output.startswith("<think>"):
        raise ValueError("positive response must begin with <think>")
    close_index = output.index("</think>")
    thought = output[len("<think>") : close_index]
    if not thought.strip():
        raise ValueError("accepted positive must have a non-empty actual CoT")

    answer = output[close_index + len("</think>") :].strip()
    match = ITEM_RE.fullmatch(answer)
    if match is None:
        raise ValueError(
            "positive answer must be exactly one domain+s_a+s_b+s_c item after </think>"
        )
    indexes = [int(match.group(name)) for name in ("a", "b", "c")]
    if any(not 0 <= value < 8_192 for value in indexes):
        raise ValueError(f"positive item code is outside 0..8191: {indexes}")
    return f"rec_{match.group('domain')}"


def validate_positives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        duplicates = len(keys) - len(set(keys))
        raise ValueError(f"accepted positive JSONL contains {duplicates} exact duplicate rows")
    task_counts = Counter(positive_task(row) for row in rows)
    return {
        "rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "exact_duplicate_rows": 0,
        "nonempty_think_rows": len(rows),
        "empty_think_rows": 0,
        "multiset_sha256": rows_multiset_sha256(rows),
    }


def classify_retention(row: dict[str, Any]) -> str:
    try:
        task = task_of(row)
    except (ValueError, json.JSONDecodeError):
        # The registered source has exactly 231 O2.General rows that do not all
        # carry the competition </think>/task signature.  The immutable source
        # hash and exact task-count assertion below make this fallback closed.
        task = "world"
    if task not in EXPECTED_RETENTION_TASK_COUNTS:
        raise ValueError(f"unexpected retention task: {task}")
    return task


def validate_registered_retention(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    if len(rows) != REGISTERED_RETENTION_ROWS:
        raise ValueError(
            f"registered retention row count drifted: {len(rows)}/{REGISTERED_RETENTION_ROWS}"
        )
    tasks = [classify_retention(row) for row in rows]
    counts = dict(sorted(Counter(tasks).items()))
    expected = dict(sorted(EXPECTED_RETENTION_TASK_COUNTS.items()))
    if counts != expected:
        raise ValueError(f"registered retention task signature drifted: {counts}/{expected}")
    return tasks, counts


def response_body_for_retention(output: str) -> str:
    close_count = output.count("</think>")
    if close_count > 1:
        raise ValueError("retention source response contains multiple </think> tags")
    if close_count == 1:
        body = output.split("</think>", 1)[1].lstrip("\r\n")
    else:
        open_count = output.count("<think>")
        if open_count == 0:
            body = output
        elif open_count == 1 and output.startswith("<think>"):
            # 101 registered O2.General rows lack </think>; some of them start
            # with an unmatched <think>.  Retention is KL-only, so preserve all
            # original content after that route token as the canonical body.
            body = output[len("<think>") :].lstrip("\r\n")
        else:
            raise ValueError("cannot canonicalize ambiguous unterminated retention think tags")
    if not body.strip():
        raise ValueError("retention source has an empty answer body")
    if "<think>" in body or "</think>" in body:
        raise ValueError("retention answer body contains a route-confusing think tag")
    return body


def canonicalize_retention(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    converted = dict(row)
    original_input = converted["input"]
    converted["input"] = MODE_SUFFIX_RE.sub("/no_think", original_input.rstrip())
    converted["output"] = CANONICAL_EMPTY_PREFIX + response_body_for_retention(row["output"])
    if converted["output"].count("<think>") != 1 or converted["output"].count("</think>") != 1:
        raise AssertionError("canonical retention output has an ambiguous think route")
    if not converted["output"].startswith(CANONICAL_EMPTY_PREFIX):
        raise AssertionError("retention output is not canonical empty-think")
    return converted, converted["input"] != original_input


def source_order_key(
    seed: int, phase: str, index: int, row: dict[str, Any], task: str
) -> tuple[str, int]:
    return stable_hash(seed, phase, task, index, row_key(row)), index


def select_retention_indices(
    rows: list[dict[str, Any]], tasks: list[str], wanted: int, seed: int
) -> tuple[list[int], dict[str, Any]]:
    if len(rows) != len(tasks) or not rows:
        raise ValueError("retention rows/tasks are empty or misaligned")
    if wanted < len(RETENTION_TASK_ORDER):
        raise ValueError(
            f"need at least {len(RETENTION_TASK_ORDER)} retention occurrences "
            "to cover every registered task"
        )

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, task in enumerate(tasks):
        buckets[task].append(index)
    missing = [task for task in RETENTION_TASK_ORDER if not buckets[task]]
    if missing:
        raise ValueError(f"retention source lacks required tasks: {missing}")

    ordered = sorted(
        range(len(rows)),
        key=lambda index: source_order_key(seed, "global", index, rows[index], tasks[index]),
    )
    if wanted <= len(rows):
        coverage = [
            min(
                buckets[task],
                key=lambda index: source_order_key(
                    seed, "coverage", index, rows[index], tasks[index]
                ),
            )
            for task in RETENTION_TASK_ORDER
        ]
        coverage_set = set(coverage)
        selected = coverage + [
            index for index in ordered if index not in coverage_set
        ][: wanted - len(coverage)]
        selection_mode = "stable_hash_without_replacement_with_task_coverage"
        full_cycles = 0
        cycle_remainder = wanted
    else:
        full_cycles, cycle_remainder = divmod(wanted, len(rows))
        selected = []
        for _ in range(full_cycles):
            selected.extend(ordered)
        selected.extend(ordered[:cycle_remainder])
        selection_mode = "stable_hash_balanced_full_cycles"

    repeats = Counter(selected)
    all_repeat_counts = [repeats.get(index, 0) for index in range(len(rows))]
    repeat_min = min(all_repeat_counts)
    repeat_max = max(all_repeat_counts)
    if repeat_max - repeat_min > 1:
        raise AssertionError(
            f"source-row repeat imbalance exceeds one: {repeat_min}/{repeat_max}"
        )

    selected_task_counts = Counter(tasks[index] for index in selected)
    uncovered = [task for task in RETENTION_TASK_ORDER if selected_task_counts[task] == 0]
    if uncovered:
        raise AssertionError(f"selected retention lost task coverage: {uncovered}")

    unique_task_counts = Counter(tasks[index] for index in repeats)
    return selected, {
        "selection": selection_mode,
        "wanted": wanted,
        "selected_occurrences": len(selected),
        "selected_unique_source_rows": len(repeats),
        "full_cycles": full_cycles,
        "cycle_remainder": cycle_remainder,
        "source_repeat_min": repeat_min,
        "source_repeat_max": repeat_max,
        "source_repeat_difference": repeat_max - repeat_min,
        "source_repeat_histogram": {
            str(count): occurrences
            for count, occurrences in sorted(Counter(all_repeat_counts).items())
        },
        "selected_task_counts": dict(sorted(selected_task_counts.items())),
        "selected_unique_source_task_counts": dict(sorted(unique_task_counts.items())),
        "required_task_coverage": list(RETENTION_TASK_ORDER),
        "missing_tasks": [],
    }


def is_empty_think(row: dict[str, Any]) -> bool:
    output = row["output"]
    if output.count("<think>") != 1 or output.count("</think>") != 1:
        return False
    if not output.startswith("<think>"):
        return False
    close_index = output.index("</think>")
    return not output[len("<think>") : close_index].strip()


def build_mix(
    positives: list[dict[str, Any]],
    retention_source: list[dict[str, Any]],
    retention_tasks: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive_audit = validate_positives(positives)
    retention_wanted = 3 * len(positives)
    selected_indices, selection_audit = select_retention_indices(
        retention_source, retention_tasks, retention_wanted, seed
    )

    retention_rows: list[dict[str, Any]] = []
    input_changes = 0
    source_role_counts = Counter()
    for index in selected_indices:
        converted, input_changed = canonicalize_retention(retention_source[index])
        retention_rows.append(converted)
        input_changes += int(input_changed)
        source_role_counts[
            "original_i12_supervised" if retention_tasks[index] in {"action", "topic"}
            else "original_i12_retention"
        ] += 1

    if len(retention_rows) != retention_wanted:
        raise AssertionError("retention occurrence count drifted during canonicalization")
    if not all(is_empty_think(row) for row in retention_rows):
        raise AssertionError("not every selected retention row is canonical empty-think")

    positive_copies = [dict(row) for row in positives]
    if rows_multiset_sha256(positive_copies) != positive_audit["multiset_sha256"]:
        raise AssertionError("positive rows changed before mixing")
    if any(is_empty_think(row) for row in positive_copies):
        raise AssertionError("a positive row was converted to empty-think")

    final_rows = positive_copies + retention_rows
    random.Random(seed).shuffle(final_rows)
    final_positives = [row for row in final_rows if not is_empty_think(row)]
    final_retention = [row for row in final_rows if is_empty_think(row)]
    if rows_multiset_sha256(final_positives) != positive_audit["multiset_sha256"]:
        raise AssertionError("positive rows were changed, lost, or duplicated in the final mix")
    if len(final_positives) != len(positives) or len(final_retention) != retention_wanted:
        raise AssertionError("final 1:3 route ratio is not exact")

    total = len(final_rows)
    mix_audit = {
        "positive": positive_audit,
        "retention": {
            **selection_audit,
            "canonical_empty_think_rows": len(retention_rows),
            "nonempty_think_rows": 0,
            "input_mode_or_trailing_space_changes": input_changes,
            "source_role_counts": dict(sorted(source_role_counts.items())),
            "original_i12_supervised_rows_receiving_gold_ce": 0,
        },
        "total_rows": total,
        "positive_rows": len(positives),
        "retention_rows": retention_wanted,
        "positive_to_retention": f"{len(positives)}:{retention_wanted}",
        "positive_ratio": len(positives) / total,
        "retention_ratio": retention_wanted / total,
        "positive_rows_changed": 0,
        "positive_rows_converted_to_empty_think": 0,
    }
    return final_rows, mix_audit


def guard_output_paths(out: Path, audit: Path, overwrite: bool) -> None:
    if out.resolve() == audit.resolve():
        raise ValueError("output JSONL and audit JSON must be different paths")
    existing = [path for path in (out, audit) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s) without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    stale = [path.with_suffix(path.suffix + ".tmp") for path in (out, audit)]
    stale = [path for path in stale if path.exists()]
    if stale:
        raise FileExistsError(
            "refusing to replace stale temporary output(s): "
            + ", ".join(str(path) for path in stale)
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def make_test_row(input_text: str, output: str) -> dict[str, Any]:
    return {"instruction": "", "input": input_text, "output": output, "history": []}


def _expect_error(fn, text: str) -> None:
    try:
        fn()
    except (ValueError, FileExistsError) as error:
        if text not in str(error):
            raise AssertionError(f"unexpected error: {error}") from error
    else:
        raise AssertionError(f"expected an error containing {text!r}")


def run_self_test() -> None:
    items = {
        "video": "<|video_begin|><s_a_1><s_b_2><s_c_3>",
        "prod": "<|prod_begin|><s_a_4><s_b_5><s_c_6>",
        "ad": "<|ad_begin|><s_a_7><s_b_8><s_c_9>",
        "living": "<|living_begin|><s_a_10><s_b_11><s_c_12>",
    }
    positives = [
        make_test_row(
            f"positive-{domain}/think",
            f"<think>actual reasoning for {domain}</think>\n{item}",
        )
        for domain, item in items.items()
    ]
    positive_snapshot = rows_multiset_sha256(positives)

    retention_by_task = {
        "material_desc2sid": make_test_row(
            "describe material/think", f"<think>old</think>\n{items['video']}"
        ),
        "material_sid2desc": make_test_row(
            f"describe {items['video']}/think", "<think>old</think>\na video description"
        ),
        "action": make_test_row(
            "choose actions/think", f'<think>old</think>\n["{items["video"]}"]'
        ),
        "topic": make_test_row(
            "topic/think", '<think>old</think>\n{"logic_chain":{"events":[]}}'
        ),
        "rec_video": make_test_row(
            "recommend/think", f"<think>old</think>\n该用户最近可能喜欢{items['video']}"
        ),
        "rec_prod": make_test_row(
            "recommend/think", f"<think>old</think>\n该用户最近可能喜欢{items['prod']}"
        ),
        "rec_ad": make_test_row(
            "recommend/think", f"<think>old</think>\n该用户最近可能喜欢{items['ad']}"
        ),
        "rec_living": make_test_row(
            "recommend/think", f"<think>old</think>\n该用户最近可能喜欢{items['living']}"
        ),
        "world": make_test_row("world question", "reasoning\n正确答案是 (A)"),
    }
    retention_source = [retention_by_task[task] for task in RETENTION_TASK_ORDER]
    classified = [classify_retention(row) for row in retention_source]
    assert classified == list(RETENTION_TASK_ORDER)

    selected_a, selection_a = select_retention_indices(
        retention_source, classified, 3 * len(positives), 27
    )
    selected_b, selection_b = select_retention_indices(
        retention_source, classified, 3 * len(positives), 27
    )
    assert selected_a == selected_b and selection_a == selection_b
    assert selection_a["source_repeat_min"] == 1
    assert selection_a["source_repeat_max"] == 2
    assert selection_a["source_repeat_difference"] == 1
    assert not selection_a["missing_tasks"]

    final_rows, mix = build_mix(positives, retention_source, classified, seed=27)
    assert len(final_rows) == 16
    assert mix["positive_rows"] == 4 and mix["retention_rows"] == 12
    assert mix["positive_ratio"] == 0.25 and mix["retention_ratio"] == 0.75
    assert rows_multiset_sha256([row for row in final_rows if not is_empty_think(row)]) == (
        positive_snapshot
    )
    assert all(
        row["output"].startswith(CANONICAL_EMPTY_PREFIX)
        for row in final_rows
        if is_empty_think(row)
    )

    empty_positive = make_test_row(
        "bad/think", f"<think></think>\n{items['video']}"
    )
    _expect_error(lambda: positive_task(empty_positive), "non-empty actual CoT")
    extra_answer = make_test_row(
        "bad/think", f"<think>reason</think>\nanswer: {items['video']}"
    )
    _expect_error(lambda: positive_task(extra_answer), "exactly one domain")

    unmatched_world, _ = canonicalize_retention(
        make_test_row("world question", "<think>reasoning\n正确答案是 (B)")
    )
    assert unmatched_world["output"] == (
        CANONICAL_EMPTY_PREFIX + "reasoning\n正确答案是 (B)"
    )
    assert is_empty_think(unmatched_world)

    with tempfile.TemporaryDirectory(prefix="i27_builder_selftest_") as directory:
        temp = Path(directory)
        out = temp / "mix.jsonl"
        audit = temp / "audit.json"
        guard_output_paths(out, audit, overwrite=False)
        write_jsonl(out, final_rows)
        write_json(audit, {"rows": len(final_rows), "sha256": sha256(out)})
        assert len(load_jsonl(out, strict_positive=False)) == len(final_rows)
        _expect_error(
            lambda: guard_output_paths(out, audit, overwrite=False),
            "refusing to overwrite",
        )

    print(
        "[i27-builder] self-test passed: strict positives, four domains, exact 1:3 mix, "
        "stable balanced cycling, nine-task retention coverage, canonical empty-think, "
        "positive immutability, audit writers, and overwrite guard are consistent"
    )


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive",
        type=Path,
        required=True,
        help="strict accepted-positive JSONL; no default is allowed",
    )
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=19260828)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    resolved_inputs = {args.positive.resolve(), args.retention.resolve()}
    resolved_outputs = {args.out.resolve(), args.audit.resolve()}
    overlap = resolved_inputs & resolved_outputs
    if overlap:
        raise ValueError(f"input/output path collision: {sorted(str(path) for path in overlap)}")
    guard_output_paths(args.out, args.audit, args.overwrite)

    retention_hash = sha256(args.retention)
    if retention_hash != REGISTERED_RETENTION_SHA256:
        raise ValueError(
            "retention input is not the registered data_user_residual_retention_v1: "
            f"{retention_hash}/{REGISTERED_RETENTION_SHA256}"
        )

    positives = load_jsonl(args.positive, strict_positive=True)
    retention_source = load_jsonl(args.retention, strict_positive=False)
    retention_tasks, source_task_counts = validate_registered_retention(retention_source)
    final_rows, mix_audit = build_mix(
        positives, retention_source, retention_tasks, args.seed
    )

    write_jsonl(args.out, final_rows)
    output_hash = sha256(args.out)
    audit = {
        "asset_class": (
            "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,"
            "O2.General; M-s800 rollout)"
        ),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "seed": args.seed,
        "upstream": {
            "accepted_positive_jsonl": {
                "asset_id": "data_o1_rec_rft_positive_v1",
                "path": str(args.positive.resolve()),
                "rows": len(positives),
                "sha256": sha256(args.positive),
                "task_counts": mix_audit["positive"]["task_counts"],
                "role": "actual-CoT recommendation positive; copied unchanged once",
            },
            "registered_i12_mix": {
                "asset_id": "data_user_residual_retention_v1",
                "path": str(args.retention.resolve()),
                "rows": len(retention_source),
                "sha256": retention_hash,
                "task_counts": source_task_counts,
                "role": "frozen-I23 KL retention only after canonical empty-think conversion",
            },
        },
        "training_semantics": {
            "positive_rows": "complete actual CoT+answer CE plus weak frozen-I23 KL",
            "retention_rows": "frozen-I23 KL only; no source gold CE",
            "original_i12_action_topic_gold_ce_rows": 0,
            "private_prompt_route_markers_added": 0,
        },
        "mix": mix_audit,
        "forbidden_training_sources": {
            "T_rows": 0,
            "E_rows": 0,
            "O3_rows": 0,
        },
        "invariants": {
            "positive_rows_preserved_exactly_once": True,
            "positive_rows_changed": 0,
            "positive_rows_converted_to_empty_think": 0,
            "all_retention_rows_canonical_empty_think": True,
            "all_retention_rows_kl_only": True,
            "retention_source_repeat_difference_le_one": True,
            "retention_task_coverage_complete": True,
        },
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(final_rows),
            "sha256": output_hash,
        },
    }
    write_json(args.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
