#!/usr/bin/env python3
"""Build a prompt-only O1 recommendation smoke manifest and a separate gold ledger.

The registered ``data_seed_clean_v1`` source contains 19,204 recommendation
rows but only 6,460 prompt groups.  Rows in one group may carry different
valid targets.  This builder preserves that set-valued supervision while
physically separating what a rollout process may read from the gold used by a
later scorer.

The frozen v1 smoke selection is 512 video groups with at least two distinct
known O1 targets.  Selection is by a stable SHA-256 ordering, never by model
output or evaluation data.  The prompt manifest contains no response, target,
gold, source-row, or gold-count field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "assets/derived/processed/data_seed_clean_v1.jsonl"
DEFAULT_MANIFEST = (
    ROOT / "assets/derived/processed/o1_rec_multigold_v1_prompt_manifest.jsonl"
)
DEFAULT_GOLD_LEDGER = (
    ROOT / "assets/derived/processed/o1_rec_multigold_v1_gold_ledger.jsonl"
)
DEFAULT_AUDIT = ROOT / "logs/data/o1_rec_multigold_v1_audit.json"

SOURCE_ASSET_ID = "D(O1):data_seed_clean_v1"
EXPECTED_SOURCE_SHA256 = (
    "e526caea4a1afd8befbd5d266fb80d0378a5bf7eff90fdacd14934332d64d309"
)
EXPECTED_SOURCE_ROWS = 32_480
EXPECTED_REC_ROWS = 19_204
EXPECTED_PROMPT_GROUPS = 6_460
EXPECTED_DOMAIN_ROWS = {
    "video": 14_868,
    "prod": 1_489,
    "ad": 1_576,
    "living": 1_271,
}

SCHEMA_PROMPT_GROUP = "o1-rec-prompt-group-v1"
SCHEMA_GROUP = "o1-rec-group-domain-v1"
SCHEMA_MANIFEST = "o1-rec-prompt-manifest-v1"
SCHEMA_GOLD = "o1-rec-gold-ledger-v1"
SELECTION_NAME = "video-multigold-smoke-v1"
SELECTION_DOMAIN = "video"
SELECTION_MIN_GOLDS = 2
SELECTION_SIZE = 512
DEFAULT_SEED = 19_260_826

MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ITEM_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
HEX64_RE = re.compile(r"[0-9a-f]{64}")

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
MANIFEST_FORBIDDEN_KEYS = {
    "answer",
    "assistant",
    "completion",
    "gold",
    "golds",
    "gold_count",
    "label",
    "output",
    "response",
    "source_row_index",
    "source_row_indices",
    "target",
    "targets",
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


def normalize_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = {
        "instruction": str(raw.get("instruction", raw.get("system", "")) or ""),
        "input": str(raw.get("input", raw.get("prompt", "")) or ""),
        "output": str(raw.get("output", raw.get("response", "")) or ""),
        "history": raw.get("history") or [],
    }
    if row["history"]:
        raise AssertionError("O1 clean source unexpectedly contains non-empty history")
    return row


def read_source(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for source_index, line in enumerate(source):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"expected object at {path}:{source_index + 1}")
            records.append(
                {
                    "source_index": source_index,
                    "source_row_sha256": text_sha256(line.rstrip("\n")),
                    "row": normalize_row(raw),
                }
            )
    return records


def prompt_core(input_text: str) -> str:
    stripped = input_text.rstrip()
    core, replacements = MODE_SUFFIX_RE.subn("", stripped)
    if replacements != 1:
        raise ValueError(f"recommendation prompt lacks one mode suffix: {stripped[-80:]!r}")
    return core


def split_output(output: str) -> tuple[str, str, str, str]:
    matches = list(THINK_RE.finditer(output))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one think block, got {len(matches)}")
    match = matches[0]
    suffix = output[match.end() :]
    answer = suffix.lstrip()
    if not answer:
        raise ValueError("assistant output has no final answer")
    return match.group(1), output[: match.start(1)], output[match.end(1) :], answer


def rec_target(answer: str) -> tuple[str, str] | None:
    if "该用户最近" not in answer:
        return None
    matches = list(ITEM_RE.finditer(answer))
    if len(matches) != 1:
        raise ValueError(f"recommendation answer has {len(matches)} itemic targets")
    match = matches[0]
    values = (int(match.group("a")), int(match.group("b")), int(match.group("c")))
    if any(value < 0 or value > 8191 for value in values):
        raise ValueError(f"itemic component outside 0..8191: {match.group(0)}")
    return match.group("domain"), match.group(0)


def make_prompt_group_id(instruction: str, core: str) -> str:
    return stable_hash(SCHEMA_PROMPT_GROUP, instruction, core)


def make_group_id(prompt_group_id: str, domain: str) -> str:
    """Return the rollout unit ID for one prompt and one requested domain."""
    return stable_hash(SCHEMA_GROUP, prompt_group_id, domain)


def make_prompt_sha256(instruction: str, input_text: str) -> str:
    # This formula is shared verbatim with the rollout generator.
    return text_sha256(
        canonical_json(
            {"history": [], "input": input_text, "instruction": instruction}
        )
    )


def aggregate_groups(
    records: list[dict[str, Any]],
    *,
    expected_source_rows: int | None = EXPECTED_SOURCE_ROWS,
    expected_rec_rows: int | None = EXPECTED_REC_ROWS,
    expected_prompt_groups: int | None = EXPECTED_PROMPT_GROUPS,
    expected_domain_rows: dict[str, int] | None = EXPECTED_DOMAIN_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_source_rows is not None and len(records) != expected_source_rows:
        raise AssertionError(
            f"source rows drifted: expected {expected_source_rows}, got {len(records)}"
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    domain_rows: Counter[str] = Counter()
    rec_rows = 0
    for record in records:
        row = record["row"]
        thought, output_prefix, output_suffix, answer = split_output(row["output"])
        parsed = rec_target(answer)
        if parsed is None:
            continue
        domain, itemic = parsed
        core = prompt_core(row["input"])
        grouped[(row["instruction"], core)].append(
            {
                **record,
                "thought": thought,
                "output_prefix": output_prefix,
                "output_suffix": output_suffix,
                "answer": answer,
                "domain": domain,
                "itemic": itemic,
            }
        )
        rec_rows += 1
        domain_rows[domain] += 1

    if expected_rec_rows is not None and rec_rows != expected_rec_rows:
        raise AssertionError(
            f"recommendation rows drifted: expected {expected_rec_rows}, got {rec_rows}"
        )
    if expected_prompt_groups is not None and len(grouped) != expected_prompt_groups:
        raise AssertionError(
            f"prompt groups drifted: expected {expected_prompt_groups}, got {len(grouped)}"
        )
    if expected_domain_rows is not None and dict(domain_rows) != expected_domain_rows:
        raise AssertionError(
            f"recommendation domain rows drifted: {dict(sorted(domain_rows.items()))}"
        )

    groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    duplicate_gold_rows = 0
    for (instruction, core), members in grouped.items():
        prompt_group_id = make_prompt_group_id(instruction, core)
        input_text = core + "/think"
        prompt_sha = make_prompt_sha256(instruction, input_text)
        original_thoughts = {member["thought"] for member in members if member["thought"]}
        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for member in members:
            by_domain[member["domain"]].append(member)

        # One O1 prompt may have valid answers in more than one domain.  The
        # official evaluator asks one domain at a time, so the rollout unit is
        # (prompt group, requested domain), while the 6,460 count remains the
        # underlying prompt-group count.
        for domain, domain_members in sorted(by_domain.items()):
            group_id = make_group_id(prompt_group_id, domain)
            if group_id in seen_group_ids:
                raise AssertionError(f"SHA-256 group-domain collision: {group_id}")
            seen_group_ids.add(group_id)

            by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for member in domain_members:
                by_gold[member["itemic"]].append(member)
            duplicate_gold_rows += len(domain_members) - len(by_gold)

            gold_entries: list[dict[str, Any]] = []
            for itemic, occurrences in sorted(by_gold.items()):
                representative = min(occurrences, key=lambda item: item["source_index"])
                output_shell = (
                    representative["output_prefix"]
                    + "{thought}"
                    + representative["output_suffix"]
                )
                gold_entries.append(
                    {
                        "itemic": itemic,
                        "itemic_sha256": text_sha256(itemic),
                        "answer": representative["answer"],
                        "output_prefix": representative["output_prefix"],
                        "output_suffix": representative["output_suffix"],
                        "output_shell_sha256": text_sha256(output_shell),
                        "source_row_indices": sorted(
                            occurrence["source_index"] for occurrence in occurrences
                        ),
                        "source_row_sha256s": sorted(
                            occurrence["source_row_sha256"] for occurrence in occurrences
                        ),
                        "target_in_prompt": itemic in core,
                    }
                )

            groups.append(
                {
                    "group_id": group_id,
                    "prompt_group_id": prompt_group_id,
                    "instruction": instruction,
                    "prompt_core": core,
                    "input": input_text,
                    "prompt_sha256": prompt_sha,
                    "domain": domain,
                    "source_prompt_group_size": len(members),
                    "source_group_size": len(domain_members),
                    "golds": gold_entries,
                    "original_thought_sha256s": sorted(
                        text_sha256(thought) for thought in original_thoughts
                    ),
                    "original_thought_stripped_sha256s": sorted(
                        {text_sha256(thought.strip()) for thought in original_thoughts}
                    ),
                }
            )

    groups.sort(key=lambda group: group["group_id"])
    group_domains = Counter(group["domain"] for group in groups)
    prompt_group_size_histogram = Counter(str(len(members)) for members in grouped.values())
    group_size_histogram = Counter(str(group["source_group_size"]) for group in groups)
    gold_count_histogram = Counter(str(len(group["golds"])) for group in groups)
    multigold_domains = Counter(
        group["domain"] for group in groups if len(group["golds"]) >= 2
    )
    return groups, {
        "source_rows": len(records),
        "recommendation_rows": rec_rows,
        "prompt_groups": len(grouped),
        "prompt_domain_units": len(groups),
        "domain_rows": dict(sorted(domain_rows.items())),
        "domain_prompt_groups": dict(sorted(group_domains.items())),
        "prompt_group_size_histogram": dict(
            sorted(prompt_group_size_histogram.items(), key=lambda item: int(item[0]))
        ),
        "prompt_domain_unit_size_histogram": dict(
            sorted(group_size_histogram.items(), key=lambda item: int(item[0]))
        ),
        "unique_gold_count_histogram": dict(
            sorted(gold_count_histogram.items(), key=lambda item: int(item[0]))
        ),
        "multigold_prompt_groups_by_domain": dict(sorted(multigold_domains.items())),
        "duplicate_gold_source_rows": duplicate_gold_rows,
    }


def select_smoke_groups(
    groups: list[dict[str, Any]],
    *,
    seed: int,
    sample_size: int = SELECTION_SIZE,
    domain: str = SELECTION_DOMAIN,
    min_golds: int = SELECTION_MIN_GOLDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        group
        for group in groups
        if group["domain"] == domain and len(group["golds"]) >= min_golds
    ]
    eligible.sort(
        key=lambda group: stable_hash(seed, SELECTION_NAME, group["group_id"])
    )
    if len(eligible) < sample_size:
        raise AssertionError(
            f"only {len(eligible)} eligible {domain} multigold groups; need {sample_size}"
        )
    selected = eligible[:sample_size]
    selected_gold_hist = Counter(str(len(group["golds"])) for group in selected)
    return selected, {
        "name": SELECTION_NAME,
        "method": "ascending SHA256(seed, selection_name, full_group_id)",
        "seed": seed,
        "domain": domain,
        "minimum_unique_golds": min_golds,
        "eligible_groups": len(eligible),
        "selected_groups": len(selected),
        "selected_unique_gold_count_histogram": dict(
            sorted(selected_gold_hist.items(), key=lambda item: int(item[0]))
        ),
    }


def build_artifacts(
    selected: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    gold_ledger: list[dict[str, Any]] = []
    target_in_prompt = 0
    for group in selected:
        rollout_seed = int(
            stable_hash(seed, "rollout-seed", group["group_id"])[:8], 16
        ) & 0x7FFF_FFFF
        prompt_row = {
            "schema_version": SCHEMA_MANIFEST,
            "group_id": group["group_id"],
            "instruction": group["instruction"],
            "input": group["input"],
            "history": [],
            "domain": group["domain"],
            "prompt_sha256": group["prompt_sha256"],
            "rollout_seed": rollout_seed,
        }
        if set(prompt_row) != MANIFEST_KEYS:
            raise AssertionError(f"prompt manifest schema drifted: {sorted(prompt_row)}")
        if set(prompt_row) & MANIFEST_FORBIDDEN_KEYS:
            raise AssertionError("prompt manifest contains a label-bearing field")
        if not HEX64_RE.fullmatch(prompt_row["group_id"]):
            raise AssertionError("group_id is not a full SHA-256")
        if make_prompt_sha256(prompt_row["instruction"], prompt_row["input"]) != prompt_row[
            "prompt_sha256"
        ]:
            raise AssertionError("prompt hash does not reproduce")

        gold_row = {
            "schema_version": SCHEMA_GOLD,
            "group_id": group["group_id"],
            "prompt_sha256": group["prompt_sha256"],
            "domain": group["domain"],
            "prompt_group_id": group["prompt_group_id"],
            "source_prompt_group_size": group["source_prompt_group_size"],
            "source_group_size": group["source_group_size"],
            "gold_count": len(group["golds"]),
            "golds": group["golds"],
            "original_thought_sha256s": group["original_thought_sha256s"],
            "original_thought_stripped_sha256s": group[
                "original_thought_stripped_sha256s"
            ],
        }
        target_in_prompt += sum(gold["target_in_prompt"] for gold in group["golds"])
        manifest.append(prompt_row)
        gold_ledger.append(gold_row)

    manifest.sort(key=lambda row: row["group_id"])
    gold_ledger.sort(key=lambda row: row["group_id"])
    manifest_ids = [row["group_id"] for row in manifest]
    gold_ids = [row["group_id"] for row in gold_ledger]
    if manifest_ids != gold_ids or len(set(manifest_ids)) != len(manifest_ids):
        raise AssertionError("prompt/gold physical partitions do not join one-to-one")
    if any("instruction" in row or "input" in row or "history" in row for row in gold_ledger):
        raise AssertionError("gold ledger unexpectedly contains a rollout prompt")

    separation = {
        "manifest_exact_keys": sorted(MANIFEST_KEYS),
        "manifest_forbidden_label_fields": sorted(MANIFEST_FORBIDDEN_KEYS),
        "manifest_rows_with_forbidden_label_fields": 0,
        "gold_ledger_rows_with_prompt_text_fields": 0,
        "prompt_and_gold_group_id_match": True,
        "prompt_and_gold_prompt_sha256_match": all(
            prompt["prompt_sha256"] == gold["prompt_sha256"]
            for prompt, gold in zip(manifest, gold_ledger)
        ),
        "known_golds_also_present_in_prompt_history": target_in_prompt,
        "note": (
            "A known O1 target may already occur in the immutable source prompt history; "
            "the manifest nevertheless contains no explicit label field."
        ),
    }
    return manifest, gold_ledger, separation


def preflight_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("manifest, gold ledger, and audit paths must be distinct")
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    preflight_outputs((args.manifest, args.gold_ledger, args.audit), args.overwrite)
    source_hash = file_sha256(args.source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise AssertionError(
            f"registered source SHA256 drifted: {source_hash} != {EXPECTED_SOURCE_SHA256}"
        )
    records = read_source(args.source)
    groups, grouping_audit = aggregate_groups(records)
    selected, selection_audit = select_smoke_groups(groups, seed=args.seed)
    manifest, gold_ledger, separation_audit = build_artifacts(selected, args.seed)

    write_jsonl(args.manifest, manifest)
    write_jsonl(args.gold_ledger, gold_ledger)
    audit = {
        "asset_class": {
            "prompt_manifest": "D(O1) construction prompt-only manifest",
            "gold_ledger": "D(O1) construction gold ledger",
        },
        "purpose": "512-group video multigold positive-only RFT yield smoke",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": file_sha256(Path(__file__)),
        "seed": args.seed,
        "upstream": {
            "asset_id": SOURCE_ASSET_ID,
            "path": str(args.source.resolve()),
            "rows": len(records),
            "sha256": source_hash,
        },
        "grouping": grouping_audit,
        "selection": selection_audit,
        "physical_separation": separation_audit,
        "forbidden_sources": {
            "O2_rows": 0,
            "O3_rows_or_metadata": 0,
            "T_rows": 0,
            "E_rows_prompts_answers_or_logs": 0,
            "model_rollout_rows": 0,
        },
        "outputs": {
            "prompt_manifest": {
                "path": str(args.manifest.resolve()),
                "rows": len(manifest),
                "sha256": file_sha256(args.manifest),
                "schema_version": SCHEMA_MANIFEST,
            },
            "gold_ledger": {
                "path": str(args.gold_ledger.resolve()),
                "rows": len(gold_ledger),
                "sha256": file_sha256(args.gold_ledger),
                "schema_version": SCHEMA_GOLD,
            },
        },
        "formal_training_authorized": False,
    }
    write_json(args.audit, audit)
    return audit


def synthetic_row(
    core: str, domain: str, item: tuple[int, int, int], think: str
) -> tuple[dict[str, Any], str]:
    itemic = (
        f"<|{domain}_begin|><s_a_{item[0]}><s_b_{item[1]}><s_c_{item[2]}>"
    )
    row = {
        "instruction": "synthetic recommendation instruction",
        "input": core + ("/think" if think else "/no_think"),
        "output": f"<think>{think}</think>\n该用户最近喜欢的视频有: {itemic}",
        "history": [],
    }
    raw = canonical_json(row)
    return row, text_sha256(raw)


def self_test() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    specifications = [
        ("history-A ", "video", (1, 2, 3), "trace A"),
        ("history-A ", "video", (4, 5, 6), ""),
        ("history-B ", "video", (7, 8, 9), "trace B"),
        ("history-B ", "video", (10, 11, 12), "trace B"),
        ("history-C ", "video", (13, 14, 15), "trace C"),
        ("history-D ", "prod", (16, 17, 18), "trace D"),
        ("history-D ", "prod", (19, 20, 21), "trace D"),
    ]
    for index, specification in enumerate(specifications):
        row, row_hash = synthetic_row(*specification)
        records.append(
            {"source_index": index, "source_row_sha256": row_hash, "row": row}
        )
    groups, grouping = aggregate_groups(
        records,
        expected_source_rows=None,
        expected_rec_rows=None,
        expected_prompt_groups=None,
        expected_domain_rows=None,
    )
    if grouping["prompt_groups"] != 4 or grouping["recommendation_rows"] != 7:
        raise AssertionError(f"synthetic grouping failed: {grouping}")
    selected, selection = select_smoke_groups(
        groups, seed=123, sample_size=2, domain="video", min_golds=2
    )
    manifest, gold, separation = build_artifacts(selected, 123)
    if len(manifest) != 2 or len(gold) != 2:
        raise AssertionError("synthetic smoke selection did not produce two groups")
    if any(set(row) != MANIFEST_KEYS for row in manifest):
        raise AssertionError("synthetic prompt manifest leaked schema fields")
    if not separation["prompt_and_gold_group_id_match"]:
        raise AssertionError("synthetic prompt/gold join failed")

    with tempfile.TemporaryDirectory(prefix="o1_rec_multigold_v1_") as directory:
        temp_root = Path(directory)
        outputs = (
            temp_root / "manifest.jsonl",
            temp_root / "gold.jsonl",
            temp_root / "audit.json",
        )
        preflight_outputs(outputs, overwrite=False)
        write_jsonl(outputs[0], manifest)
        write_jsonl(outputs[1], gold)
        write_json(outputs[2], {"ok": True})
        try:
            preflight_outputs(outputs, overwrite=False)
        except FileExistsError:
            overwrite_guard = True
        else:
            raise AssertionError("overwrite guard did not reject existing outputs")

    result = {
        "status": "PASS",
        "synthetic_source_rows": len(records),
        "synthetic_prompt_groups": len(groups),
        "selected_groups": selection["selected_groups"],
        "manifest_gold_physical_join": True,
        "overwrite_guard": overwrite_guard,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold-ledger", type=Path, default=DEFAULT_GOLD_LEDGER)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
