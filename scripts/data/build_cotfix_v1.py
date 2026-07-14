#!/usr/bin/env python3
"""Build a surgical repair of sentence-truncated official recommendation CoTs.

The pipeline has three explicit phases:
  prepare  - find high-confidence broken CoTs and prepare auditable requests;
  generate - deterministically close the broken sentence;
  build    - validate generations and replace CoT suffixes without changing answers.

The repair never reads target metadata and never introduces a new SID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT.parent / "ai_runtime" / "llmrec_2026"
SEED = RUNTIME / "data/processed/data_final.jsonl"
REQUESTS = RUNTIME / "logs/data/cotfix_v1_requests.jsonl"
GENERATIONS = RUNTIME / "logs/data/cotfix_v1_generations.jsonl"
OUTPUT = RUNTIME / "data/processed/data_seed_cotfix_v1.jsonl"
AUDIT = RUNTIME / "logs/data/cotfix_v1_audit.json"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
SID_RE = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
TERMINAL_CHARS = set("。！？!?…”\"」』)）】》]")
FORBIDDEN_CONTINUATION_TEXT = (
    "<think>",
    "</think>",
    "正确答案",
    "该用户最近喜欢的视频有:",
    "该用户最近喜欢的商品有:",
    "该用户最近点击的广告有:",
    "该用户最近首次打赏了主播:",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def seed_groups(rows: list[dict]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        body = row["output"].split("</think>", 1)[-1]
        if "该用户最近" not in body:
            continue
        core = MODE_SUFFIX_RE.sub("", row["input"].rstrip())
        groups[core].append(index)
    return dict(groups)


def candidate_kind(think: str) -> str | None:
    # All audited failures are very short relative to the normal p50=1396.
    # The hard rule reproduces 425 high-confidence unique failures; 800-899
    # character rows are intentionally left unchanged in this conservative v1.
    if len(think) < 800 and think[-1] not in TERMINAL_CHARS:
        return "hard"
    return None


def prepare(args) -> None:
    rows = load_jsonl(args.seed)
    groups = seed_groups(rows)
    candidates = {}
    for core, indexes in groups.items():
        thinks = {
            match.group(1).strip()
            for index in indexes
            if (match := THINK_RE.search(rows[index]["output"])) and match.group(1).strip()
        }
        if len(thinks) != 1:
            continue
        think = next(iter(thinks))
        kind = candidate_kind(think)
        if kind:
            candidates[core] = {"think": think, "kind": kind, "row_count": len(indexes)}

    requests = []
    for core, candidate in sorted(candidates.items(), key=lambda item: hashlib.sha1(item[0].encode()).hexdigest()):
        history_sids = unique_in_order(SID_RE.findall(core))
        candidate_id = hashlib.sha1((core + "\0" + candidate["think"]).encode()).hexdigest()[:20]
        requests.append(
            {
                "candidate_id": candidate_id,
                "core_sha1": hashlib.sha1(core.encode()).hexdigest(),
                "kind": candidate["kind"],
                "row_count": candidate["row_count"],
                "prefix": candidate["think"],
                "prefix_chars": len(candidate["think"]),
                "history_sids": history_sids,
            }
        )
    write_jsonl(args.requests, requests)
    counts = Counter(request["kind"] for request in requests)
    print(json.dumps({"requests": len(requests), "kinds": counts, "path": str(args.requests)}, ensure_ascii=False, default=dict))

def unmatched_open_count(text: str, left: str, right: str) -> int:
    depth = 0
    for char in text:
        if char == left:
            depth += 1
        elif char == right and depth:
            depth -= 1
    return depth


def bracket_errors(prefix: str, continuation: str) -> list[str]:
    errors = []
    for left, right in (("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")):
        before = unmatched_open_count(prefix, left, right)
        after = unmatched_open_count(prefix + continuation, left, right)
        if before > 0 and after != 0:
            errors.append(f"failed to close {left}{right}: {before}->{after}")
    if prefix.count("`") % 2 and (prefix + continuation).count("`") % 2:
        errors.append("failed to close backtick")
    return errors


def validate_decision(request: dict, decision: dict) -> list[str]:
    errors = []
    verdict = str(decision.get("verdict", "")).upper()
    continuation = decision.get("continuation", "")
    if verdict not in {"TRUNCATED", "KEEP"}:
        return ["invalid verdict"]
    if not isinstance(continuation, str):
        return ["continuation is not a string"]
    if request["kind"] == "hard" and verdict != "TRUNCATED":
        errors.append("hard candidate returned KEEP")
    if verdict == "KEEP":
        if continuation.strip():
            errors.append("KEEP has nonempty continuation")
        return errors

    continuation = continuation.rstrip()
    combined = request["prefix"] + continuation
    if len(continuation) < 10:
        errors.append("continuation too short")
    if len(continuation) > 260:
        errors.append("continuation too long")
    if "\n" in continuation:
        errors.append("continuation contains newline")
    if len(combined) > 1200:
        errors.append("combined text longer than 1200")
    if not combined or combined[-1] not in TERMINAL_CHARS:
        errors.append("combined text has no terminal punctuation")
    if any(token in continuation for token in FORBIDDEN_CONTINUATION_TEXT):
        errors.append("forbidden answer/tag text")
    if "<s_" in continuation or "<|" in continuation:
        errors.append("continuation contains newly generated token syntax")
    if "、)" in continuation or "、）" in continuation or "如的" in continuation:
        errors.append("continuation contains malformed list syntax")
    max_overlap = min(80, len(request["prefix"]), len(continuation))
    overlap = max(
        (size for size in range(1, max_overlap + 1) if request["prefix"].endswith(continuation[:size])),
        default=0,
    )
    if overlap >= 6:
        errors.append(f"continuation repeats {overlap} prefix-tail characters")
    if re.search(r"(?:-[^-\s]+){2,}\(\d+\)", continuation):
        errors.append("continuation copied tag-frequency paths")
    history = set(request["history_sids"])
    introduced = set(SID_RE.findall(continuation))
    if not introduced <= history:
        errors.append(f"introduced non-history SID: {sorted(introduced - history)[:3]}")
    errors.extend(bracket_errors(request["prefix"], continuation))
    return errors


def deterministic_decision(request: dict) -> dict:
    prefix = request["prefix"]
    headings = re.findall(r"(?m)^\s*\d+[.、]\s*\*\*([^*]+)\*\*", prefix)
    heading = headings[-1].strip("：: ") if headings else "这一兴趣"
    selector = int(request["candidate_id"][:8], 16)

    if re.search(r"(?m)^\s*\d+\.$", prefix):
        numbered_closures = (
            " **其他稳定兴趣**：用户还会持续关注与既有偏好相关的内容。",
            " **补充兴趣方向**：用户对既有偏好的相近内容也保持一定关注。",
            " **相关延伸内容**：用户还会浏览与主要兴趣相衔接的内容。",
            " **持续关注方向**：用户对符合既有偏好的内容仍有稳定需求。",
            " **次要兴趣内容**：除核心方向外，用户也会关注相近主题。",
            " **兴趣补充**：用户的其他互动仍主要围绕既有偏好展开。",
        )
        continuation = numbered_closures[selector % len(numbered_closures)]
        return {"verdict": "TRUNCATED", "continuation": continuation, "confidence": 1.0}

    fillers = {
        "如": ("前述同类内容等", "相关同类内容等", "其他同类内容等"),
        "(": ("包括其他同类内容等", "涵盖相关同类行为等", "涉及其他相近内容等"),
        "（": ("包括其他同类内容等", "涵盖相关同类行为等", "涉及其他相近内容等"),
        "、": ("其他同类内容等", "相关同类行为等", "其余相近内容等"),
        ",": ("其他同类内容等", "相关同类行为等", "其余相近内容等"),
        "，": ("其他同类内容等", "相关同类行为等", "其余相近内容等"),
        ">": ("等", "等", "等"),
        "`": ("相关同类内容`", "其他相近内容`", "与当前兴趣相关的内容`"),
        ".": ("其他同类内容等", "相关行为记录等", "其余相近内容等"),
        "发": ("等深度互动", "等相关行为", "等高价值互动"),
    }
    end = prefix[-1]
    if end not in fillers:
        raise ValueError(f"unsupported deterministic ending: {end!r}")
    continuation = fillers[end][selector % len(fillers[end])]

    ascii_open = unmatched_open_count(prefix, "(", ")")
    chinese_open = unmatched_open_count(prefix, "（", "）")
    if prefix.count("`") % 2 and continuation.count("`") % 2 == 0:
        continuation += "`"
    if ascii_open > 0:
        continuation += ")" * ascii_open
    if chinese_open > 0:
        continuation += "）" * chinese_open

    clauses = (
        f"，进一步印证了用户对“{heading}”方向的持续关注。",
        f"，说明“{heading}”在用户历史兴趣中具有较高权重。",
        f"，反映出用户在“{heading}”相关内容上的稳定偏好。",
        f"，体现了用户围绕“{heading}”产生的较深交互。",
        f"，显示用户对“{heading}”相关内容保持明确兴趣。",
        f"，表明“{heading}”是用户较为稳定的兴趣方向。",
    )
    continuation += clauses[(selector // len(fillers[end])) % len(clauses)]
    return {"verdict": "TRUNCATED", "continuation": continuation, "confidence": 1.0}


def generate(args) -> None:
    requests = load_jsonl(args.requests)
    if args.limit:
        requests = requests[: args.limit]
    final = []
    for request in requests:
        try:
            decision = deterministic_decision(request)
            errors = validate_decision(request, decision)
            raw = json.dumps(decision, ensure_ascii=False)
        except Exception as error:
            decision, errors, raw = None, [str(error)], ""
        final.append(
            {
                "candidate_id": request["candidate_id"],
                "raw": raw,
                "decision": decision,
                "errors": errors,
                "method": "deterministic",
            }
        )

    final.sort(key=lambda record: record["candidate_id"])
    write_jsonl(args.generations, final)
    counts = Counter(
        "invalid" if record["errors"] else record["decision"]["verdict"].upper()
        for record in final
    )
    print(json.dumps({"generated": len(final), "decisions": counts, "path": str(args.generations)}, ensure_ascii=False, default=dict))


def build(args) -> None:
    rows = load_jsonl(args.seed)
    requests = {record["candidate_id"]: record for record in load_jsonl(args.requests)}
    generations = {record["candidate_id"]: record for record in load_jsonl(args.generations)}
    groups = seed_groups(rows)

    decisions_by_key = {}
    invalid = {}
    for candidate_id, request in requests.items():
        generation = generations.get(candidate_id)
        if not generation:
            invalid[candidate_id] = ["missing generation"]
            continue
        errors = generation.get("errors") or []
        decision = generation.get("decision")
        if not errors and decision:
            errors = validate_decision(request, decision)
        if errors:
            invalid[candidate_id] = errors
            continue
        decisions_by_key[(request["core_sha1"], request["prefix"])] = decision

    modified_rows = 0
    modified_groups = set()
    modified_domains = Counter()
    continuations = []
    original_rows = [dict(row) for row in rows]
    for core, indexes in groups.items():
        core_sha1 = hashlib.sha1(core.encode()).hexdigest()
        for index in indexes:
            row = rows[index]
            match = THINK_RE.search(row["output"])
            if not match or not match.group(1).strip():
                continue
            prefix = match.group(1).strip()
            decision = decisions_by_key.get((core_sha1, prefix))
            if not decision or decision["verdict"].upper() != "TRUNCATED":
                continue
            continuation = decision["continuation"].rstrip()
            raw_body = match.group(1)
            trailing = raw_body[len(raw_body.rstrip()) :]
            new_body = raw_body.rstrip() + continuation + trailing
            row["output"] = row["output"][: match.start(1)] + new_body + row["output"][match.end(1) :]
            modified_rows += 1
            modified_groups.add((core_sha1, prefix))
            continuations.append(continuation)
            answer = row["output"].split("</think>", 1)[-1]
            domain = DOMAIN_RE.search(answer)
            modified_domains[domain.group(1) if domain else "unknown"] += 1

    # Full-dataset invariants: only the contents of nonempty think blocks may differ.
    answer_diffs = prompt_diffs = history_diffs = mode_diffs = sid_violations = 0
    for before, after in zip(original_rows, rows):
        prompt_diffs += before["instruction"] != after["instruction"] or before["input"] != after["input"]
        history_diffs += before.get("history", []) != after.get("history", [])
        before_match, after_match = THINK_RE.search(before["output"]), THINK_RE.search(after["output"])
        before_answer = before["output"][before_match.end() :] if before_match else before["output"]
        after_answer = after["output"][after_match.end() :] if after_match else after["output"]
        answer_diffs += before_answer != after_answer
        mode_diffs += before["input"].rstrip().endswith("/think") != after["input"].rstrip().endswith("/think")
        if before["output"] != after["output"]:
            history_sids = set(SID_RE.findall(MODE_SUFFIX_RE.sub("", before["input"].rstrip())))
            added_sids = set(SID_RE.findall(after_match.group(1))) - set(SID_RE.findall(before_match.group(1)))
            sid_violations += not added_sids <= history_sids

    if not modified_rows:
        raise AssertionError("no rows were modified")
    if answer_diffs or prompt_diffs or history_diffs or mode_diffs or sid_violations:
        raise AssertionError(
            f"invariant failure: answer={answer_diffs} prompt={prompt_diffs} history={history_diffs} "
            f"mode={mode_diffs} sid={sid_violations}"
        )
    if invalid:
        raise AssertionError(f"{len(invalid)} invalid/missing repairs remain; examples={list(invalid.items())[:3]}")

    write_jsonl(args.output, rows)
    continuation_counts = Counter(continuations)
    prefix20_counts = Counter(text[:20] for text in continuations)
    audit = {
        "source": str(args.seed.resolve()),
        "output": str(args.output.resolve()),
        "rows": len(rows),
        "candidate_groups": len(requests),
        "candidate_kinds": dict(Counter(request["kind"] for request in requests.values())),
        "repair_methods": dict(Counter(generation.get("method", "unknown") for generation in generations.values())),
        "repair_decisions": dict(
            Counter(generation["decision"]["verdict"].upper() for generation in generations.values() if not generation.get("errors") and generation.get("decision"))
        ),
        "modified_groups": len(modified_groups),
        "modified_rows": modified_rows,
        "modified_rows_by_target_domain": dict(modified_domains),
        "continuation_chars": {
            "min": min(map(len, continuations)),
            "median": sorted(map(len, continuations))[len(continuations) // 2],
            "max": max(map(len, continuations)),
        },
        "continuation_diversity": {
            "unique": len(continuation_counts),
            "max_exact_duplicate": max(continuation_counts.values()),
            "unique_first_20_chars": len(prefix20_counts),
            "max_shared_first_20_chars": max(prefix20_counts.values()),
        },
        "invariants": {
            "answer_diffs": answer_diffs,
            "prompt_diffs": prompt_diffs,
            "history_diffs": history_diffs,
            "mode_diffs": mode_diffs,
            "sid_violations": sid_violations,
            "invalid_generations": len(invalid),
        },
        "md5": md5(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "generate", "build"))
    parser.add_argument("--seed", type=Path, default=SEED)
    parser.add_argument("--requests", type=Path, default=REQUESTS)
    parser.add_argument("--generations", type=Path, default=GENERATIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    {"prepare": prepare, "generate": generate, "build": build}[args.phase](args)


if __name__ == "__main__":
    main()
