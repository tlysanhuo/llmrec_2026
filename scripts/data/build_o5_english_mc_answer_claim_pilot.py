#!/usr/bin/env python3
"""Build a quarantined O5 English A-D answer-claim translation pilot.

This program intentionally does *not* create training data.  O5 usually has no
independent answer label for the rows considered here; the assistant response
is therefore recorded only as ``source_answer_claim``.  Translation, two blind
reviews, adjudication, and a second leakage check are required before any row
can be promoted to a reviewed candidate.

Only a frozen 32-shard pilot is scanned.  The shard selector reproduces an
earlier registered audit whose separator is the two literal ASCII bytes
backslash and zero (``b"\\0"``), not a NUL byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_official_general_world_clean as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
O5_DIR = ROOT / "assets/official/general_sft"
OUT = ROOT / "assets/derived/official_general/o5_en_mc_answer_claim_pilot.jsonl"
AUDIT = ROOT / "logs/data/o5_en_mc_answer_claim_pilot_audit.json"

O5_REVISION = "4b8e43913aeb8e6c66b9253df4ab64ecc77dfd6c"
RULESET_VERSION = "o5-en-mc-answer-claim-pilot-20260718-v1"
SHARD_SEED = b"20260711"
SHARD_ASSET = b"O5"
LITERAL_BACKSLASH_ZERO = bytes((92, 48))
EXPECTED_FILES = 301
EXPECTED_ROWS = 2_555_706
EXPECTED_SELECTED_FILES = 32
EXPECTED_SELECTED_ROWS = 280_957

ALLOWED_QUARANTINE_SOURCES = frozenset({"R1-Distill-SFT", "Infinity_Instruct"})
DEFAULT_SOURCE_CAPS = {"R1-Distill-SFT": 50, "Infinity_Instruct": 200}
DEFAULT_TOPIC_CAPS = {"math_logic": 50}
DEFAULT_ANSWER_CAP = 125
DEFAULT_TOTAL_CAP = 500

# Labels must be uppercase, on their own lines, and occur in A-D order.
OPTION_LINE = re.compile(
    r"^\s*(?:(?P<bare>[A-H])\s*[.):]|\((?P<paren>[A-H])\)|"
    r"\[(?P<bracket>[A-H])\]\.?)[ \t]*(?P<text>\S.*?)\s*$"
)
LOWER_OPTION_LINE = re.compile(
    r"^\s*(?:[a-h]\s*[.):]|\([a-h]\)|\[[a-h]\]\.?)\s*\S",
    re.IGNORECASE,
)
LOWER_ONLY_OPTION_LINE = re.compile(
    r"^\s*(?:[a-h]\s*[.):]|\([a-h]\)|\[[a-h]\]\.?)\s*\S.*$"
)
INLINE_ABCD = re.compile(
    r"(?is)(?:^|\s)A\s*[.):].+?\s+B\s*[.):].+?\s+C\s*[.):].+?\s+D\s*[.):]"
)
MULTISELECT = re.compile(
    r"\b(?:select|choose|mark|check)\s+(?:all|two|three|four|[2-9])\b|"
    r"\ball\s+that\s+apply\b|\bmore\s+than\s+one\s+(?:answer|option)\b|"
    r"\bone\s+or\s+more\s+(?:answers?|options?)\b|"
    r"\bmore\s+than\s+one\s+correct\s+(?:answer|option)\b|"
    r"\bmultiple\s+(?:correct\s+)?(?:answers?|options?)\b|"
    r"\bstatements?\s+are\s+correct\b|\banswers?\s+are\s+correct\b",
    re.IGNORECASE,
)
ALL_NONE = re.compile(
    r"\b(?:all|none)\s+of\s+the\s+above\b|"
    r"\bboth\s+[A-D]\s+and\s+[A-D]\b|\bneither\s+[A-D]\s+nor\s+[A-D]\b|"
    r"\b[A-D]\s*(?:,|and|or|&)\s*[A-D]\b",
    re.IGNORECASE,
)
NEGATIVE_STEM = re.compile(
    r"\bNOT\b|\bEXCEPT\b|\bincorrect\b|\bfalse\b|\bleast\s+likely\b",
    re.IGNORECASE,
)
PROMPT_ANSWER_LEAK = re.compile(
    r"\b(?:final\s+answer|correct\s+answer|answer\s+key)\s*(?::|=|is)\s*"
    r"[([]?[A-D][])]?",
    re.IGNORECASE,
)

BENCHMARK = re.compile(
    r"\b(?:c[-_ ]?eval|cmmlu|mmlu|arc(?:[-_ ]challenge)?|race|hellaswag|"
    r"winogrande|openbookqa|commonsenseqa|truthfulqa|ag[-_ ]?news|gsm8k|"
    r"math[-_ ]?training[-_ ]?set|aops|aime|amc|leetcode|taco|humaneval|mbpp)\b",
    re.IGNORECASE,
)
READING_CONTEXT = re.compile(
    r"\b(?:read\s+the\s+(?:following\s+)?(?:passage|article|story|text)|"
    r"according\s+to\s+the\s+(?:passage|article|story|text)|"
    r"best\s+summari[sz]es|most\s+appropriate\s+title|appropriate\s+title|"
    r"choose\s+the\s+next\s+sentence|premise\s*(?:and|/)\s*hypothesis|"
    r"based\s+on\s+the\s+(?:passage|article|story|text)|"
    r"read\s+this\s+fact|now\s+answer\s+the\s+question|"
    r"what\s+is\s+(?:this|the)\s+text\s+about|"
    r"which\s+topic\s+is\s+(?:this|the)\s+(?:article|text|story)\s+about|"
    r"select\s+the\s+topic\s+that\s+this\s+(?:is\s+)?about|"
    r"which\s+topic\s+is\s+this\s+article\s+about|"
    r"given\s+the\s+fact\b.{0,240}\banswer\s+to\s+the\s+question\s+or\s+completion)\b",
    re.IGNORECASE,
)
MEDICAL = re.compile(
    r"\b(?:patient|diagnos(?:is|e)|treat(?:ment|ing)|drug\s+dose|dosage|"
    r"prescription|surgery|clinical|symptom|cancer|tumou?r|pregnan(?:t|cy)|"
    r"vaccine|physician|nurs(?:e|ing))\b",
    re.IGNORECASE,
)
LEGAL_POLITICAL = re.compile(
    r"\b(?:legal(?:ly)?|statute|court|liability|lawsuit|criminal|civil\s+law|"
    r"securities\s+law|constitutional|election|voter|political\s+party|"
    r"president|prime\s+minister|government\s+policy)\b",
    re.IGNORECASE,
)
FINANCE = re.compile(
    r"\b(?:investment\s+advice|stock\s+price|share\s+price|portfolio|"
    r"cryptocurrency|mortgage\s+rate|interest\s+rate\s+today|market\s+forecast)\b",
    re.IGNORECASE,
)
TIME_SENSITIVE = re.compile(
    r"\b(?:current|latest|today(?:'s)?|this\s+(?:week|month|year)|now|"
    r"most\s+recent|real[- ]time|incumbent)\s+"
    r"(?:president|prime\s+minister|ceo|price|ranking|population|record|rate)\b|"
    r"\bwho\s+is\s+the\s+current\b",
    re.IGNORECASE,
)
SUBJECTIVE = re.compile(
    r"\b(?:in\s+your\s+opinion|do\s+you\s+think|most\s+beautiful|"
    r"greatest\s+ever|best\s+tasting|favorite|favourite|most\s+enjoyable|"
    r"what\s+should\s+someone\s+prefer)\b",
    re.IGNORECASE,
)
PROMPT_INJECTION = re.compile(
    r"\bignore\s+(?:all\s+)?previous\b|\bsystem\s+prompt\b|"
    r"\bdo\s+not\s+answer\s+the\s+question\b|\breveal\s+your\s+instructions\b",
    re.IGNORECASE,
)
IDENTITY = re.compile(
    r"\b(?:ChatGPT|Claude|Gemini|Copilot|Llama|Qwen|StepFun)\b|"
    r"\b(?:your|model)\s+identity\b",
    re.IGNORECASE,
)
TOOL_MEDIA = re.compile(
    r"<\|(?:tool|image|video)|</?(?:tool_call|tool_response|image|video)>|"
    r"\[(?:image|img|figure|table)\s*=|\b(?:shown|pictured)\s+(?:above|below)\b|"
    r"\b(?:image|figure|chart|diagram|table)\s+(?:above|below)\b|"
    r"https?://\S+",
    re.IGNORECASE,
)

UNCERTAIN = re.compile(
    r"\b(?:probably|possibly|maybe|perhaps|I\s+think|I\s+guess|may\s+be|"
    r"might\s+be|not\s+sure|uncertain)\b|"
    r"\b(?:answer|option|choice)\s+(?:is\s+)?[A-D]\s+(?:or|/)\s+[A-D]\b|"
    r"\bfinal\s+answer\s*:\s*[A-D]\s+(?:or|/)\s+[A-D]\b",
    re.IGNORECASE,
)
ASSERTION_PATTERNS = (
    re.compile(
        r"\b(?:final\s+answer|(?:the\s+)?correct\s+answer|answer)\s*"
        r"(?:is\s*)?[:=]?\s*\(?\s*([A-D])\s*\)?",
        re.IGNORECASE,
    ),
    re.compile(r"\b([A-D])\s+is\s+(?:the\s+)?correct\b", re.IGNORECASE),
    re.compile(r"\bcorrect\s+(?:choice|option)\s+(?:is\s*)?[:=]?\s*([A-D])\b", re.IGNORECASE),
)
FINAL_ASSERTIONS = (
    re.compile(
        r"\b(?:final\s+answer|(?:the\s+)?correct\s+answer|answer)\s*"
        r"(?:is\s*)?[:=]?\s*\(?\s*([A-D])\s*\)?\s*[.!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\b([A-D])\s+is\s+(?:the\s+)?correct\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[([A-D])\]\s*[.!]?\s*$"),
    re.compile(r"^\s*\(?([A-D])\)?\s*[.!]?\s*$"),
)

MATH_HINT = re.compile(
    r"(?:\\(?:frac|sqrt|sin|cos|tan|log|sum|int|theta|alpha|beta|pi|boxed)\b|"
    r"\$|\^|\b(?:equation|function|probability|integer|polynomial|triangle|"
    r"coordinate|matrix|derivative|integral|geometry|algebra|ratio|angle|"
    r"calculate|compute|value\s+of|number\s+of\s+(?:ways|arrangements))\b|"
    r"\d\s*[+\-*/=<>]\s*\d)",
    re.IGNORECASE,
)


ParsedMC = base.ParsedMC


def _semantic(value: str) -> str:
    return base.semantic_normalize(value)


def _option_label(match: re.Match[str]) -> str:
    return next(value for value in (match.group("bare"), match.group("paren"), match.group("bracket")) if value)


def parse_english_mc_prompt(text: str) -> tuple[ParsedMC | None, list[str]]:
    """Parse an uppercase, multiline, exactly-four-option English MC prompt."""

    value = base.normalize_raw(text).strip()
    lines = value.splitlines()
    matches: list[tuple[int, re.Match[str]]] = []
    lowercase_lines = 0
    for index, line in enumerate(lines):
        match = OPTION_LINE.fullmatch(line)
        if match:
            matches.append((index, match))
        elif LOWER_ONLY_OPTION_LINE.fullmatch(line):
            lowercase_lines += 1

    reasons: list[str] = []
    if not matches:
        if lowercase_lines >= 2:
            reasons.append("mc_lowercase_options")
        elif INLINE_ABCD.search(value):
            reasons.append("mc_inline_options")
        else:
            reasons.append("mc_options_unparsed")
        return None, reasons

    labels = [_option_label(match) for _index, match in matches]
    if any(label in "EFGH" for label in labels):
        reasons.append("mc_extra_option")
    if labels != list("ABCD"):
        reasons.append("mc_labels_not_exact_abcd")
    indices = [index for index, _match in matches]
    if indices and indices != list(range(indices[0], indices[0] + len(indices))):
        reasons.append("mc_option_lines_not_contiguous")
    if matches and any(line.strip() for line in lines[matches[-1][0] + 1 :]):
        reasons.append("mc_postscript")
    if reasons:
        return None, sorted(set(reasons))

    first_index = matches[0][0]
    question = "\n".join(lines[:first_index]).strip()
    if not question:
        reasons.append("mc_empty_stem")
    if INLINE_ABCD.search(question):
        reasons.append("mc_inline_options")
    if MULTISELECT.search(value):
        reasons.append("mc_multiselect")
    options = {label: match.group("text").strip() for label, (_index, match) in zip("ABCD", matches)}
    normalized_options = [_semantic(options[label]) for label in "ABCD"]
    if any(not item for item in normalized_options):
        reasons.append("mc_empty_option")
    if len(set(normalized_options)) != 4:
        reasons.append("mc_duplicate_option")
    if any(ALL_NONE.search(options[label]) for label in "ABCD"):
        reasons.append("mc_all_none_or_combination")
    if NEGATIVE_STEM.search(question):
        reasons.append("mc_negated_stem")
    if PROMPT_ANSWER_LEAK.search(question):
        reasons.append("mc_prompt_answer_leak")
    if reasons:
        return None, sorted(set(reasons))
    return ParsedMC(question=question, options=options, postscript=""), []


def _script_name(char: str) -> str:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return "unknown"
    for script in ("LATIN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC"):
        if script in name:
            return script.casefold()
    return "other"


def is_strict_en(text: str) -> tuple[bool, dict[str, Any]]:
    """Return a fail-closed English-script decision and auditable counts."""

    counts = Counter()
    for char in text:
        if char.isalpha():
            counts[_script_name(char)] += 1
    latin = counts["latin"]
    total_alpha = sum(counts.values())
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)
    forbidden = sum(counts[key] for key in ("cjk", "hiragana", "katakana", "hangul", "cyrillic", "arabic"))
    ratio = latin / max(total_alpha, 1)
    stats = {
        "latin_letters": latin,
        "alphabetic_letters": total_alpha,
        "latin_alpha_ratio": ratio,
        "english_word_count": len(words),
        "forbidden_script_letters": forbidden,
        "script_counts": dict(sorted(counts.items())),
    }
    passed = latin >= 40 and len(words) >= 6 and ratio >= 0.98 and forbidden == 0
    return passed, stats


def _boxed_spans(text: str) -> list[tuple[str | None, str, int, int]]:
    """Parse balanced ``\\boxed{...}`` spans and an optional leading A-D."""

    spans: list[tuple[str | None, str, int, int]] = []
    marker = "\\boxed{"
    start = 0
    while True:
        found = text.find(marker, start)
        if found < 0:
            break
        depth = 1
        cursor = found + len(marker)
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            spans.append((None, text[found:], found, len(text)))
            break
        content = text[found + len(marker) : cursor - 1]
        label_match = re.match(
            r"\s*(?:\\(?:text|mathrm|mathbf)\s*\{\s*)?([A-D])(?:\s*:)?(?:\s*\})?",
            content,
        )
        label = label_match.group(1) if label_match else None
        spans.append((label, text[found:cursor], found, cursor))
        start = cursor
    return spans


def _strong_assertions(text: str) -> list[tuple[str, str, int, int]]:
    assertions: list[tuple[str, str, int, int]] = []
    for pattern in ASSERTION_PATTERNS:
        for match in pattern.finditer(text):
            assertions.append((match.group(1).upper(), match.group(0), match.start(), match.end()))
    for label, evidence, start, end in _boxed_spans(text):
        if label is not None:
            assertions.append((label, evidence, start, end))
    for match in re.finditer(r"(?m)^\s*\[([A-D])\]\s*[.!]?\s*$", text):
        assertions.append((match.group(1), match.group(0), match.start(), match.end()))
    assertions.sort(key=lambda item: (item[2], item[3], item[0]))
    return assertions


def _final_assertion(last_line: str) -> tuple[str | None, str | None]:
    for pattern in FINAL_ASSERTIONS:
        match = pattern.search(last_line)
        if match:
            return match.group(1).upper(), match.group(0)
    boxes = _boxed_spans(last_line)
    for label, evidence, _start, end in reversed(boxes):
        if label is None:
            continue
        tail = last_line[end:]
        if re.fullmatch(r"[\s$\\\[\]().,!;:]*", tail):
            return label, evidence
    return None, None


def parse_english_mc_answer_claim(text: str) -> tuple[str | None, str | None, list[str]]:
    """Extract a conflict-safe final A-D claim; never infer from answer text."""

    normalized = base.normalize_raw(text).strip()
    final_text, think_status = base.split_reasoning(normalized)
    if final_text is None:
        return None, None, [think_status]
    reasons: list[str] = []
    if UNCERTAIN.search(final_text):
        reasons.append("answer_uncertain")
    assertions = _strong_assertions(final_text)
    if not assertions:
        if re.search(r"\[[A-D]\]", final_text):
            reasons.append("answer_not_final")
        elif "\\boxed{" in final_text:
            reasons.append("answer_no_letter")
        else:
            reasons.append("answer_unparsed")
        return None, None, sorted(set(reasons))
    labels = {item[0] for item in assertions}
    if len(labels) != 1:
        reasons.append("answer_conflict")
    last_line = next((line.strip() for line in reversed(final_text.splitlines()) if line.strip()), "")
    final_label, final_evidence = _final_assertion(last_line)
    if final_label is None:
        reasons.append("answer_not_final")
    elif len(labels) == 1 and final_label not in labels:
        reasons.append("answer_conflict")
    if reasons:
        return None, None, sorted(set(reasons))
    label = next(iter(labels))
    return label, final_evidence or assertions[-1][1], []


def _decode_metadata(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.casefold() in {"null", "none", "nan"}:
            return None
        try:
            return json.loads(stripped)
        except Exception:
            return value
    return value


def extract_metadata_label(metadata: Any) -> tuple[str | None, list[str]]:
    """Extract only explicit independent A-D metadata labels."""

    decoded = _decode_metadata(metadata)
    candidates: list[str] = []
    keys = {"answer", "correct_answer", "correctanswer", "label", "gold", "answer_key", "target"}

    def visit(value: Any, key: str | None = None) -> None:
        value = _decode_metadata(value)
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key).casefold())
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key in keys and isinstance(value, str):
            match = re.fullmatch(r"\s*[([]?\s*([A-D])\s*[])]?\s*[.!]?\s*", value)
            if match:
                candidates.append(match.group(1))

    visit(decoded)
    unique = sorted(set(candidates))
    if len(unique) > 1:
        return None, ["metadata_answer_conflict"]
    return (unique[0] if unique else None), []


def source_policy_reasons(prompt: str, lineage: Mapping[str, Any]) -> list[str]:
    """Return quarantine source, benchmark, context, and safety violations."""

    reasons: list[str] = []
    if lineage.get("asset_id") != "O5":
        reasons.append("source_asset_not_o5")
    source = str(lineage.get("source", ""))
    context = prompt + "\n" + base.stable_json(dict(lineage))
    if BENCHMARK.search(context):
        reasons.append("source_benchmark_family")
    if source not in ALLOWED_QUARANTINE_SOURCES:
        reasons.append("source_not_quarantine_allowlist")
    if READING_CONTEXT.search(prompt):
        reasons.append("reading_comprehension_context")
    if MEDICAL.search(prompt):
        reasons.append("high_risk_medical")
    if LEGAL_POLITICAL.search(prompt):
        reasons.append("high_risk_legal_or_political")
    if FINANCE.search(prompt):
        reasons.append("risk_finance")
    if TIME_SENSITIVE.search(prompt):
        reasons.append("risk_time_sensitive")
    if SUBJECTIVE.search(prompt):
        reasons.append("risk_subjective")
    if PROMPT_INJECTION.search(prompt):
        reasons.append("risk_prompt_injection")
    if IDENTITY.search(prompt):
        reasons.append("risk_identity")
    if TOOL_MEDIA.search(prompt):
        reasons.append("risk_tool_or_media")
    return sorted(set(reasons))


def _shard_rank(path: Path) -> tuple[bytes, str]:
    payload = (
        SHARD_SEED
        + LITERAL_BACKSLASH_ZERO
        + SHARD_ASSET
        + LITERAL_BACKSLASH_ZERO
        + path.name.encode("utf-8")
    )
    return hashlib.sha256(payload).digest(), path.name


def select_fixed_shards(paths: Sequence[Path], count: int = EXPECTED_SELECTED_FILES) -> list[Path]:
    if count < 0 or count > len(paths):
        raise ValueError(f"invalid shard count {count} for {len(paths)} paths")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("duplicate shard filenames")
    return sorted(sorted(paths, key=_shard_rank)[:count], key=lambda path: path.name)


def stable_stratified_sample(
    rows: Sequence[dict[str, Any]],
    *,
    quotas: Mapping[str, int],
    seed: int | str,
    stratum_key: str = "source_bucket",
    id_key: str = "record_id",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generic deterministic quota sampler used by tests and future reviews."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        record_id = str(row[id_key])
        if record_id in seen_ids:
            raise ValueError(f"duplicate id: {record_id}")
        seen_ids.add(record_id)
        groups[str(row[stratum_key])].append(row)
    selected: list[dict[str, Any]] = []
    shortfall: dict[str, int] = {}
    for stratum in sorted(quotas):
        quota = int(quotas[stratum])
        if quota < 0:
            raise ValueError(f"negative quota for {stratum}")
        ranked = sorted(
            groups.get(stratum, []),
            key=lambda row: (
                hashlib.sha256(f"{seed}\0{stratum}\0{row[id_key]}".encode()).hexdigest(),
                str(row[id_key]),
            ),
        )
        selected.extend(ranked[:quota])
        missing = quota - min(quota, len(ranked))
        if missing:
            shortfall[stratum] = missing
    selected.sort(key=lambda row: str(row[id_key]))
    requested = sum(int(value) for value in quotas.values())
    audit = {
        "requested_rows": requested,
        "selected_rows": len(selected),
        "shortfall_rows": requested - len(selected),
        "shortfall_by_stratum": dict(sorted(shortfall.items())),
        "dirty_backfill_rows": 0,
    }
    return selected, audit


def topic_bucket(parsed: ParsedMC, source: str) -> str:
    # The audited R1 English A-D pool in the frozen shards is overwhelmingly
    # mathematics.  Treating the whole source as math is deliberately
    # conservative and prevents a weak regex from bypassing the 50-row cap.
    if source == "R1-Distill-SFT":
        return "math_logic"
    value = parsed.question + "\n" + "\n".join(parsed.options.values())
    if MATH_HINT.search(value):
        return "math_logic"
    if re.search(r"\b(?:history|century|empire|war|dynasty|philosoph|literature|artist|music)\b", value, re.I):
        return "history_culture"
    if re.search(r"\b(?:country|capital|river|mountain|ocean|climate|environment|continent)\b", value, re.I):
        return "geography_environment"
    if re.search(r"\b(?:biology|chemistry|physics|element|planet|cell|species|energy)\b", value, re.I):
        return "natural_science"
    if re.search(r"\b(?:computer|software|internet|algorithm|technology|device)\b", value, re.I):
        return "computing_technology"
    return "other_general"


def _media_present(value: Any) -> bool:
    decoded = _decode_metadata(value)
    return decoded not in (None, "", [], {})


def _language_reasons(parsed: ParsedMC) -> tuple[list[str], dict[str, Any]]:
    rendered = parsed.question + "\n" + "\n".join(parsed.options.values())
    passed, stats = is_strict_en(rendered)
    reasons: list[str] = []
    if not passed:
        reasons.append("non_strict_english")
    stem_words = len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", parsed.question))
    option_words = {
        label: len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", parsed.options[label]))
        for label in "ABCD"
    }
    stats["stem_word_count"] = stem_words
    stats["option_word_counts"] = option_words
    if not 6 <= stem_words <= 80:
        reasons.append("stem_word_count_out_of_range")
    if any(count > 40 for count in option_words.values()):
        reasons.append("option_too_long")
    return sorted(set(reasons)), stats


def _selection_hash(record_id: str) -> str:
    payload = f"{O5_REVISION}\0{RULESET_VERSION}\0{record_id}"
    return base.hash_text(payload)


def _select_with_caps(rows: Sequence[dict[str, Any]], total_cap: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (row["quality"]["selection_hash"], row["record_id"]))
    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    cap_rejections: Counter[str] = Counter()
    for row in ranked:
        source = row["lineage"]["source"]
        topic = row["quality"]["topic"]
        answer = row["source_answer_claim"]["letter"]
        if len(selected) >= total_cap:
            cap_rejections["total_cap"] += 1
            continue
        if source_counts[source] >= DEFAULT_SOURCE_CAPS[source]:
            cap_rejections[f"source_cap:{source}"] += 1
            continue
        if topic in DEFAULT_TOPIC_CAPS and topic_counts[topic] >= DEFAULT_TOPIC_CAPS[topic]:
            cap_rejections[f"topic_cap:{topic}"] += 1
            continue
        if answer_counts[answer] >= DEFAULT_ANSWER_CAP:
            cap_rejections[f"answer_cap:{answer}"] += 1
            continue
        selected.append(row)
        source_counts[source] += 1
        topic_counts[topic] += 1
        answer_counts[answer] += 1
    selected.sort(key=lambda row: row["record_id"])
    return selected, {
        "eligible_rows": len(rows),
        "selected_rows": len(selected),
        "requested_max_rows": total_cap,
        "shortfall_to_max": max(total_cap - len(selected), 0),
        "source_counts": dict(sorted(source_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "answer_counts": dict(sorted(answer_counts.items())),
        "cap_rejections": dict(sorted(cap_rejections.items())),
    }


def _load_reviewed_index() -> tuple[base.LeakageIndex, int]:
    path = ROOT / "assets/derived/official_general/world_mc_human_reviewed_safe.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    index = base.LeakageIndex()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            clean = row["clean"]
            prompt = clean["question"] + "\n" + "\n".join(
                f"{label}. {clean['options'][label]}" for label in "ABCD"
            )
            index.add(prompt, path.name, include_near=True)
            count += 1
    if count != 29:
        raise AssertionError(f"reviewed world signature drifted: {count} != 29")
    return index, count


def _dedupe(rows: Sequence[dict[str, Any]], stats: Counter[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["quality"]["semantic"]["option_invariant_hash"]].append(row)
    kept: list[dict[str, Any]] = []
    for group in groups.values():
        claims = {row["source_answer_claim"]["letter"] for row in group}
        if len(claims) != 1:
            stats["drop:dedupe_answer_conflict"] += len(group)
            continue
        group = sorted(group, key=lambda row: (row["quality"]["selection_hash"], row["record_id"]))
        kept.append(group[0])
        stats["drop:semantic_duplicate"] += len(group) - 1
    return kept


def _counter_nested(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    if field == "source":
        counter = Counter(row["lineage"]["source"] for row in rows)
    elif field == "answer":
        counter = Counter(row["source_answer_claim"]["letter"] for row in rows)
    elif field == "topic":
        counter = Counter(row["quality"]["topic"] for row in rows)
    else:
        raise ValueError(field)
    return dict(sorted(counter.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.o5_dir.resolve() != O5_DIR.resolve():
        raise RuntimeError(f"O5 must use registered canonical asset: {O5_DIR}")
    base.ensure_safe_paths((args.out, args.audit))
    files = sorted(args.o5_dir.glob("*.parquet"))
    if len(files) != EXPECTED_FILES:
        raise AssertionError(f"O5 shard signature drifted: {len(files)} != {EXPECTED_FILES}")

    import pyarrow.parquet as pq

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if total_rows != EXPECTED_ROWS:
        raise AssertionError(f"O5 row signature drifted: {total_rows} != {EXPECTED_ROWS}")
    selected_files = select_fixed_shards(files)
    selected_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in selected_files)
    if selected_rows != EXPECTED_SELECTED_ROWS:
        raise AssertionError(
            f"selected O5 row signature drifted: {selected_rows} != {EXPECTED_SELECTED_ROWS}"
        )

    print("[blacklist] loading E prompts and current parent prompts", flush=True)
    eval_index = base.load_eval_index()
    train_index, train_counts = base.load_train_index()
    reviewed_index, reviewed_count = _load_reviewed_index()
    print(
        f"[blacklist] eval={sum(eval_index.source_counts.values())} "
        f"parent={sum(train_counts.values())} reviewed={reviewed_count}",
        flush=True,
    )

    stats: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    columns = ["source", "messages", "metadata", "uuid", "label", "images", "videos"]
    for file_number, path in enumerate(selected_files, 1):
        file_survivors = 0
        for row_group, row_index, row in base.parquet_rows(path, columns):
            stats["rows_scanned"] += 1
            source = str(row.get("source") or "")
            stats[f"source_seen:{source}"] += 1
            if source not in ALLOWED_QUARANTINE_SOURCES:
                stats["drop:source_not_quarantine_allowlist"] += 1
                continue
            parsed_messages = base.parse_messages(row.get("messages"))
            if parsed_messages is None:
                stats["drop:messages_not_single_plain_round"] += 1
                continue
            _roles, user, assistant = parsed_messages
            if _media_present(row.get("images")) or _media_present(row.get("videos")):
                stats["drop:media_columns_present"] += 1
                continue
            record_id, lineage = base.source_locator(
                "O5",
                O5_REVISION,
                args.o5_dir,
                path,
                row_group,
                row_index,
                source,
                row.get("uuid"),
                row.get("messages"),
            )
            policy_lineage = dict(lineage)
            policy_lineage["metadata"] = row.get("metadata")
            policy_reasons = source_policy_reasons(user, policy_lineage)
            if policy_reasons:
                for reason in policy_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            parsed, parse_reasons = parse_english_mc_prompt(user)
            if parsed is None:
                for reason in parse_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            language_reasons, language = _language_reasons(parsed)
            if language_reasons:
                for reason in language_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            claim, evidence, answer_reasons = parse_english_mc_answer_claim(assistant)
            if claim is None:
                for reason in answer_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            metadata_value: Any = row.get("metadata")
            if row.get("label") is not None:
                metadata_value = {"metadata": metadata_value, "label": str(row.get("label"))}
            metadata_label, metadata_reasons = extract_metadata_label(metadata_value)
            if metadata_reasons:
                for reason in metadata_reasons:
                    stats[f"drop:{reason}"] += 1
                continue
            if metadata_label is not None and metadata_label != claim:
                stats["drop:metadata_claim_mismatch"] += 1
                continue
            eval_hit, eval_modes = eval_index.match(user, parsed)
            train_hit, train_modes = train_index.match(user, parsed)
            reviewed_hit, reviewed_modes = reviewed_index.match(user, parsed)
            if eval_hit or train_hit or reviewed_hit:
                if eval_hit:
                    stats["drop:eval_overlap"] += 1
                if train_hit:
                    stats["drop:parent_overlap"] += 1
                if reviewed_hit:
                    stats["drop:reviewed_overlap"] += 1
                continue
            semantic = base.mc_semantic_keys(parsed, claim)
            selection_hash = _selection_hash(record_id)
            topic = topic_bucket(parsed, source)
            candidate = {
                "record_id": record_id,
                "task_type": "world_mc_translation_candidate",
                "lineage": lineage,
                "original": {
                    "language": "en",
                    "question": parsed.question,
                    "options": parsed.options,
                },
                "source_answer_claim": {
                    "status": "source_assistant_claim_only_not_gold",
                    "letter": claim,
                    "answer_text": parsed.options[claim],
                    "evidence": evidence,
                    "independent_metadata_label": metadata_label is not None,
                    "metadata_label": metadata_label,
                },
                "quality": {
                    "language": language,
                    "topic": topic,
                    "semantic": semantic,
                    "selection_hash": selection_hash,
                    "original_eval_overlap_modes": eval_modes,
                    "original_parent_overlap_modes": train_modes,
                    "original_reviewed_overlap_modes": reviewed_modes,
                },
                "translation": {"status": "pending_answer_blind_translation"},
                "review": {
                    "status": "pending_two_answer_blind_reviews",
                    "gold_status": "unresolved",
                },
                "builder": {
                    "ruleset_version": RULESET_VERSION,
                    "builder_sha256": None,
                    "build_fingerprint": None,
                },
            }
            candidates.append(candidate)
            stats["pre_dedupe_eligible"] += 1
            stats[f"pre_dedupe_topic:{topic}"] += 1
            file_survivors += 1
        print(
            f"[scan {file_number:02d}/{len(selected_files)}] {path.name} "
            f"rows={pq.ParquetFile(path).metadata.num_rows} eligible={file_survivors}",
            flush=True,
        )

    if stats["rows_scanned"] != EXPECTED_SELECTED_ROWS:
        raise AssertionError(
            f"streamed row signature drifted: {stats['rows_scanned']} != {EXPECTED_SELECTED_ROWS}"
        )
    deduped = _dedupe(candidates, stats)
    selected, selection_audit = _select_with_caps(deduped, min(args.max_candidates, DEFAULT_TOTAL_CAP))

    builder_sha = base.sha256_file(Path(__file__))
    build_fingerprint = base.hash_text(base.stable_json({
        "builder_sha256": builder_sha,
        "ruleset_version": RULESET_VERSION,
        "o5_revision": O5_REVISION,
        "selected_shards": [path.name for path in selected_files],
        "max_candidates": min(args.max_candidates, DEFAULT_TOTAL_CAP),
        "source_caps": DEFAULT_SOURCE_CAPS,
        "topic_caps": DEFAULT_TOPIC_CAPS,
        "answer_cap": DEFAULT_ANSWER_CAP,
    }))
    for row in selected:
        row["builder"]["builder_sha256"] = builder_sha
        row["builder"]["build_fingerprint"] = build_fingerprint

    # Final fail-closed structural validation before atomic publication.
    ids: set[str] = set()
    invariants: set[str] = set()
    for row in selected:
        assert row["record_id"] not in ids
        assert row["quality"]["semantic"]["option_invariant_hash"] not in invariants
        assert list(row["original"]["options"]) == list("ABCD")
        assert row["source_answer_claim"]["letter"] in "ABCD"
        assert row["source_answer_claim"]["status"].endswith("not_gold")
        assert row["translation"]["status"].startswith("pending")
        assert row["review"]["gold_status"] == "unresolved"
        ids.add(row["record_id"])
        invariants.add(row["quality"]["semantic"]["option_invariant_hash"])

    base.atomic_jsonl(args.out, selected)
    output_sha = base.sha256_file(args.out)
    audit = {
        "asset_class": "D-quarantine-candidate(O5); NOT TRAINING DATA",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "ruleset_version": RULESET_VERSION,
        "upstream": {
            "asset_id": "O5",
            "revision": O5_REVISION,
            "path": str(args.o5_dir.resolve()),
            "parquet_files": len(files),
            "rows": total_rows,
        },
        "frozen_pilot": {
            "selector_seed_ascii": SHARD_SEED.decode(),
            "selector_asset_ascii": SHARD_ASSET.decode(),
            "separator_hex": LITERAL_BACKSLASH_ZERO.hex(),
            "separator_is_literal_backslash_zero_not_nul": True,
            "selected_files": [path.name for path in selected_files],
            "selected_file_count": len(selected_files),
            "selected_rows": selected_rows,
        },
        "policy": {
            "answer_semantics": "assistant output is source_answer_claim, never source gold",
            "allowed_quarantine_sources": sorted(ALLOWED_QUARANTINE_SOURCES),
            "source_caps": DEFAULT_SOURCE_CAPS,
            "topic_caps": DEFAULT_TOPIC_CAPS,
            "answer_cap": DEFAULT_ANSWER_CAP,
            "max_candidates": min(args.max_candidates, DEFAULT_TOTAL_CAP),
            "r1_forced_topic": "math_logic",
            "dirty_backfill": False,
        },
        "blacklist": {
            "eval_prompt_instances": sum(eval_index.source_counts.values()),
            "current_parent_prompt_instances": dict(sorted(train_counts.items())),
            "reviewed_world_candidates": reviewed_count,
            "selection_uses_prompt_text_only": True,
        },
        "filter_counts": dict(sorted(stats.items())),
        "pre_selection": {
            "deduped_eligible_rows": len(deduped),
            "source_counts": _counter_nested(deduped, "source"),
            "topic_counts": _counter_nested(deduped, "topic"),
            "answer_counts": _counter_nested(deduped, "answer"),
        },
        "selection": selection_audit,
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(selected),
            "bytes": args.out.stat().st_size,
            "sha256": output_sha,
        },
        "release_gate": {
            "candidate_only": True,
            "translation_projection_created": False,
            "training_projection_created": False,
            "gold_labels_created": False,
            "required_next_steps": [
                "create a physically answer-blind translation packet",
                "translate without source_answer_claim access",
                "run translated Chinese E/parent/reviewed leakage checks",
                "collect two independent blind solutions",
                "only then reveal and adjudicate source_answer_claim",
            ],
        },
    }
    base.atomic_json(args.audit, audit)
    print(
        f"[done] selected={len(selected)} output={args.out} sha256={output_sha}",
        flush=True,
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o5-dir", type=Path, default=O5_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_TOTAL_CAP)
    args = parser.parse_args()
    if not 0 <= args.max_candidates <= DEFAULT_TOTAL_CAP:
        parser.error(f"--max-candidates must be in [0,{DEFAULT_TOTAL_CAP}]")
    if args.out.resolve(strict=False) == args.audit.resolve(strict=False):
        parser.error("--out and --audit must differ")
    return args


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
