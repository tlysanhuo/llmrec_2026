#!/usr/bin/env python3
"""Distill high-precision action-select labels from O2-derived histories.

The generator only sees stable event IDs and semantic annotations. Itemic tokens
are mapped back programmatically, so the teacher cannot invent or corrupt SIDs.
An independent judge filters the proposed evolution chain before it becomes SFT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "assets/derived/processed/r2_base_v3.jsonl"
DEFAULT_ENV = ROOT / "configs/secrets/deepseek_api.env"
DEFAULT_EXCLUDES = (
    ROOT / "assets/derived/processed/r2_gold_v4.jsonl",
    ROOT / "assets/derived/processed/r2_gold_g1.jsonl",
    ROOT / "assets/derived/processed/r2_gold_g2.jsonl",
    ROOT / "assets/derived/processed/r2_gold_local.jsonl",
)

ITEM_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
DOMAIN_RE = re.compile(r"<\|(?P<domain>video|prod|ad|living)_begin\|>")
DATE_RE = re.compile(r"^【(?P<date>\d{4}-\d{2}-\d{2})】$")
ACTION_RE = re.compile(r"\[(?P<action>[^\]]+)]")
EVENT_ID_RE = re.compile(r"^E(?P<number>\d{4})$")

GENERATOR_SYSTEM = """你是推荐系统用户兴趣演化数据的高级标注员。输入是严格按时间排序的行为事件，每条只有稳定事件号 E0001...、日期、行为类型与物料语义描述。

任务：找出历史中证据最完整、最有区分度的一条兴趣演化主题，并选择支撑该主题的核心事件。

硬规则：
1. 只能引用输入中存在的事件号，不得输出 SID、物料 token 或虚构事实。
2. event_ids 必须保持原始时间顺序、互不重复，选 5-18 条且不超过全部事件的 30%；典型优质答案为 7-15 条。
3. 主题必须具体而非“泛娱乐/日常生活”等万能主题；要覆盖该主题下所有直接相关的触达、深化、比较、纠偏与转化证据，不能只挑少数代表项造成召回不足。
4. 链条必须体现至少两阶段的兴趣状态变化，例如接触→深化、泛化→聚焦、比较→决策、需求→转化；同类内容机械堆叠不算演化。
5. 尊重行为强度、时间密度、纠偏/收敛信号；不要臆测用户心理，也不要因为同域就全选。
6. logic 只概括可由行为证据支持的状态，2-5 步；每个入选事件应被某一步覆盖。重复出现但确实强化同一具体兴趣的不同物料应保留，完全相同 SID 或弱相关内容不保留。

只输出严格 JSON 对象，不要 Markdown 和额外文字：
{"theme":"8-40字的具体演化主题","event_ids":["E0003","E0011","E0027","E0042","E0068"],"logic":[{"step":1,"event_ids":["E0003","E0011"],"state":"证据支持的初始兴趣状态"},{"step":2,"event_ids":["E0027","E0042","E0068"],"state":"证据支持的深化或收敛状态"}],"confidence":0.0}"""

JUDGE_SYSTEM = """你是独立的推荐行为标注质检员。你会看到完整的按时序事件和一个候选兴趣演化链。判断它是否适合作为 action-select SFT 金标。

逐项检查：
- order_valid：事件保持原始时序且无重复；
- evidence_grounded：主题和状态均可从物料描述与行为直接得到，无读心或虚构；
- theme_specific：主题有区分度，不是可套多数历史的万能主题；
- evolution_valid：至少两阶段，存在认知/需求的增量、纠偏或收敛，而非相似内容堆叠；
- selected_relevant：每个入选事件都直接支持主题；
- selection_complete：已覆盖同一具体主题下所有直接证据，尤其是重复强化、比较、纠偏和转化事件；既不能只挑代表项造成低召回，也不能为追求数量加入弱相关噪声；
- action_strength_used：强行为与时间密度被合理利用，而非忽略交互强弱；
- evidence_closed：logic 的每一步均由引用事件闭环支撑。

只拒绝会把模型教错的实质问题，不因措辞风格做苛刻拒绝。score 为 1-5；只有证据链完整、可直接进入高质量 SFT 的样本才给 5。只输出严格 JSON：
{"accept":true,"score":4,"checks":{"order_valid":true,"evidence_grounded":true,"theme_specific":true,"evolution_valid":true,"selected_relevant":true,"selection_complete":true,"action_strength_used":true,"evidence_closed":true},"reason":"一句话说明"}"""

REQUIRED_JUDGE_CHECKS = (
    "order_valid",
    "evidence_grounded",
    "theme_specific",
    "evolution_valid",
    "selected_relevant",
    "selection_complete",
    "action_strength_used",
    "evidence_closed",
)


@dataclass(frozen=True)
class Event:
    event_id: str
    index: int
    date: str
    action: str
    domain: str
    token: str
    description: str


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for raw in source:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("response has no valid JSON object")


def history_part(text: str) -> str:
    return text.split("\n\n角色任务", 1)[0].strip()


def parse_event_lines(text: str) -> list[dict[str, str]]:
    current_date = "unknown"
    parsed: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        date_match = DATE_RE.match(line)
        if date_match:
            current_date = date_match.group("date")
            continue
        token_match = ITEM_RE.search(line)
        if not token_match:
            continue
        action_match = ACTION_RE.search(line)
        prefix_end = action_match.end() if action_match else 0
        description = line[prefix_end : token_match.start()].strip()
        domain_match = DOMAIN_RE.search(token_match.group(0))
        parsed.append(
            {
                "date": current_date,
                "action": action_match.group("action") if action_match else "未知行为",
                "domain": domain_match.group("domain") if domain_match else "unknown",
                "token": token_match.group(0),
                "description": re.sub(r"\s+", " ", description),
            }
        )
    return parsed


def build_events(row: dict[str, Any], description_chars: int) -> list[Event]:
    plain = parse_event_lines(history_part(str(row.get("input", ""))))
    annotated = parse_event_lines(str(row.get("_hist_annot", "")))
    if not plain:
        raise ValueError("student history has no events")
    if len(plain) != len(annotated):
        raise ValueError(f"plain/annotated event count mismatch: {len(plain)}/{len(annotated)}")
    events = []
    for index, (base, rich) in enumerate(zip(plain, annotated), start=1):
        if base["token"] != rich["token"]:
            raise ValueError(f"plain/annotated token mismatch at event {index}")
        description = rich["description"] or "无可用文本描述"
        events.append(
            Event(
                event_id=f"E{index:04d}",
                index=index,
                date=base["date"],
                action=base["action"],
                domain=base["domain"],
                token=base["token"],
                description=description[:description_chars],
            )
        )
    return events


def render_timeline(events: list[Event]) -> str:
    lines = [f"事件总数：{len(events)}"]
    for event in events:
        lines.append(
            f"[{event.event_id}] {event.date} [{event.domain}/{event.action}] {event.description}"
        )
    return "\n".join(lines)


class ChatClient:
    def __init__(self, base_url: str, api_key: str, timeout: int, retries: int):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        last_error = "unknown API error"
        for attempt in range(self.retries):
            try:
                response = self._session().post(
                    self.url,
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                payload = response.json()
                content = payload["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise RuntimeError("empty assistant content")
                usage = payload.get("usage") or {}
                return content.strip(), {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                }
            except Exception as error:  # requests and proxy payload failures
                last_error = str(error)
                if attempt + 1 < self.retries:
                    time.sleep(min(3 * (attempt + 1), 12))
        raise RuntimeError(last_error)


def normalize_proposal(
    proposal: dict[str, Any], events: list[Event]
) -> tuple[dict[str, Any], list[Event]]:
    theme = str(proposal.get("theme") or "").strip()
    if not 8 <= len(theme) <= 40:
        raise ValueError(f"theme length is {len(theme)}, expected 8-40")
    if re.search(r"E\d{4}", theme) or ITEM_RE.search(theme) or re.search(r"\d{3,}", theme):
        raise ValueError("theme contains an event/item/raw numeric ID")

    raw_ids = proposal.get("event_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(x, str) for x in raw_ids):
        raise ValueError("event_ids is not a string list")
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("event_ids contains duplicates")
    event_by_id = {event.event_id: event for event in events}
    if any(event_id not in event_by_id for event_id in raw_ids):
        raise ValueError("event_ids contains an ID absent from history")
    picked = [event_by_id[event_id] for event_id in raw_ids]
    if [event.index for event in picked] != sorted(event.index for event in picked):
        raise ValueError("event_ids is not in chronological order")

    max_selected = min(18, int(len(events) * 0.30))
    if not 5 <= len(picked) <= max_selected:
        raise ValueError(
            f"selected density is {len(picked)}/{len(events)}, allowed 5-{max_selected}"
        )
    if len({event.token for event in picked}) < 5:
        raise ValueError("fewer than five unique selected itemic tokens")

    logic = proposal.get("logic")
    if not isinstance(logic, list) or not 2 <= len(logic) <= 5:
        raise ValueError("logic must contain 2-5 steps")
    covered: list[str] = []
    normalized_logic = []
    for expected_step, raw_step in enumerate(logic, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError("logic step is not an object")
        step_ids = raw_step.get("event_ids")
        state = str(raw_step.get("state") or "").strip()
        if not isinstance(step_ids, list) or not step_ids or not state:
            raise ValueError("logic step lacks event_ids or state")
        if any(event_id not in raw_ids for event_id in step_ids):
            raise ValueError("logic cites an event outside event_ids")
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("logic step repeats an event ID")
        covered.extend(step_ids)
        normalized_logic.append(
            {"step": expected_step, "event_ids": step_ids, "state": state[:100]}
        )
    if len(covered) != len(set(covered)) or set(covered) != set(raw_ids):
        raise ValueError("logic must cover every selected event exactly once")

    try:
        confidence = float(proposal.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    normalized = {
        "theme": theme,
        "event_ids": raw_ids,
        "logic": normalized_logic,
        "confidence": max(0.0, min(confidence, 1.0)),
    }
    return normalized, picked


def validate_judgement(
    judgement: dict[str, Any], min_score: int
) -> tuple[bool, str]:
    checks = judgement.get("checks")
    try:
        score = int(judgement.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    missing_or_false = []
    if not isinstance(checks, dict):
        missing_or_false = list(REQUIRED_JUDGE_CHECKS)
    else:
        missing_or_false = [name for name in REQUIRED_JUDGE_CHECKS if checks.get(name) is not True]
    accepted = judgement.get("accept") is True and score >= min_score and not missing_or_false
    reason = str(judgement.get("reason") or "").strip()
    if missing_or_false:
        reason = f"failed checks={','.join(missing_or_false)}; {reason}".strip()
    if score < min_score:
        reason = f"score={score}; {reason}".strip()
    return accepted, reason[:500]


def build_training_row(row: dict[str, Any], theme: str, picked: list[Event]) -> dict[str, Any]:
    source_input = str(row.get("input", ""))
    new_input, substitutions = re.subn(
        r"(主题[:：])[^\n]*", lambda match: match.group(1) + theme, source_input, count=1
    )
    if substitutions != 1:
        raise ValueError("failed to replace topic in student prompt")
    tokens = []
    seen = set()
    for event in picked:
        if event.token not in seen:
            seen.add(event.token)
            tokens.append(event.token)
    return {
        "instruction": str(row.get("instruction") or ""),
        "input": new_input,
        "output": "<think>\n\n</think>\n" + json.dumps(tokens, ensure_ascii=False),
        "history": row.get("history") or [],
    }


def usage_sum(*parts: dict[str, int]) -> dict[str, int]:
    return {
        key: sum(part.get(key, 0) for part in parts)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def process_row(
    src_idx: int,
    row: dict[str, Any],
    client: ChatClient,
    generator_model: str,
    judge_model: str,
    description_chars: int,
    repair_attempts: int,
    min_judge_score: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    aggregate_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        events = build_events(row, description_chars)
    except Exception as error:
        return None, {
            "src_idx": src_idx,
            "status": "rejected",
            "stage": "parse_history",
            "reason": str(error),
            "usage": aggregate_usage,
        }

    timeline = render_timeline(events)
    feedback = ""
    last_stage = "generator"
    last_reason = "no attempt"
    proposal: dict[str, Any] | None = None
    judgement: dict[str, Any] | None = None

    for attempt in range(1, repair_attempts + 2):
        generator_user = timeline
        if feedback:
            generator_user += (
                "\n\n上一次候选未通过质检。请重新独立选择更可靠的链条，并修复以下实质问题："
                + feedback[:800]
            )
        try:
            raw_proposal, generator_usage = client.call(
                generator_model, GENERATOR_SYSTEM, generator_user, max_tokens=1800
            )
            aggregate_usage = usage_sum(aggregate_usage, generator_usage)
            proposal = extract_json_object(raw_proposal)
            normalized, picked = normalize_proposal(proposal, events)
        except Exception as error:
            last_stage = "generator_or_hard_qc"
            last_reason = str(error)
            feedback = last_reason
            continue

        judge_user = (
            timeline
            + "\n\n候选链：\n"
            + json.dumps(normalized, ensure_ascii=False)
        )
        try:
            raw_judgement, judge_usage = client.call(
                judge_model, JUDGE_SYSTEM, judge_user, max_tokens=1000
            )
            aggregate_usage = usage_sum(aggregate_usage, judge_usage)
            judgement = extract_json_object(raw_judgement)
            accepted, reason = validate_judgement(judgement, min_judge_score)
        except Exception as error:
            last_stage = "judge"
            last_reason = str(error)
            feedback = last_reason
            continue

        if accepted:
            try:
                training_row = build_training_row(row, normalized["theme"], picked)
            except Exception as error:
                last_stage = "build_training_row"
                last_reason = str(error)
                feedback = last_reason
                continue
            return training_row, {
                "src_idx": src_idx,
                "status": "accepted",
                "attempts": attempt,
                "n_history": len(events),
                "n_selected": len(json.loads(training_row["output"].split("</think>\n", 1)[1])),
                "proposal": normalized,
                "judgement": judgement,
                "usage": aggregate_usage,
            }
        last_stage = "judge_qc"
        last_reason = reason or "judge rejected candidate"
        feedback = last_reason

    return None, {
        "src_idx": src_idx,
        "status": "rejected",
        "stage": last_stage,
        "reason": last_reason[:800],
        "attempts": repair_attempts + 1,
        "n_history": len(events),
        "proposal": proposal,
        "judgement": judgement,
        "usage": aggregate_usage,
    }


def load_excluded_indices(paths: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"registered exclusion asset is missing: {path}")
        for row in read_jsonl(path):
            if "_src_idx" in row:
                excluded.add(int(row["_src_idx"]))
    return excluded


def append_jsonl(handle: Any, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--generator-model", default="gpt-5.6-sol-max")
    parser.add_argument("--judge-model", default="gpt-5.6-terra-max")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--max-accepted", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19260817)
    parser.add_argument("--description-chars", type=int, default=96)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--min-judge-score", type=int, default=5, choices=(4, 5))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        help="JSONL with _src_idx to exclude; defaults to all registered action eval sources",
    )
    args = parser.parse_args()

    if args.workers < 1 or args.repair_attempts < 0:
        parser.error("workers must be >=1 and repair-attempts must be >=0")
    if args.generator_model == args.judge_model:
        parser.error("generator and judge models must be independent variants")

    env = load_env(args.env_file)
    base_url = os.environ.get("TEACHER_BASE", env.get("YUNWU_BASE_URL", ""))
    api_key = os.environ.get("TEACHER_KEY", env.get("YUNWU_API_KEY", ""))
    if not base_url or not api_key:
        raise RuntimeError("missing YUNWU_BASE_URL/YUNWU_API_KEY")

    rows = read_jsonl(args.src)
    exclude_paths = args.exclude if args.exclude is not None else list(DEFAULT_EXCLUDES)
    excluded = load_excluded_indices(exclude_paths)

    completed: set[int] = set()
    previously_accepted = 0
    if args.audit_log.exists():
        for audit in read_jsonl(args.audit_log):
            if audit.get("status") == "accepted":
                previously_accepted += 1
                completed.add(int(audit["src_idx"]))
            elif not args.retry_rejected:
                completed.add(int(audit["src_idx"]))

    candidates = [
        index
        for index in range(args.start, len(rows))
        if index not in excluded and index not in completed
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    if args.n:
        candidates = candidates[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.audit_log.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    client = ChatClient(base_url, api_key, args.timeout, args.retries)
    attempted = accepted = rejected = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    started = time.time()

    print(
        f"source={len(rows)} excluded_eval={len(excluded)} pending={len(candidates)} "
        f"generator={args.generator_model} judge={args.judge_model}",
        flush=True,
    )
    with args.out.open("a", encoding="utf-8") as output, args.audit_log.open(
        "a", encoding="utf-8"
    ) as audit_output, ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(candidates)
        futures = {}

        def submit_one() -> bool:
            try:
                src_idx = next(iterator)
            except StopIteration:
                return False
            future = pool.submit(
                process_row,
                src_idx,
                rows[src_idx],
                client,
                args.generator_model,
                args.judge_model,
                args.description_chars,
                args.repair_attempts,
                args.min_judge_score,
            )
            futures[future] = src_idx
            return True

        for _ in range(args.workers):
            if not submit_one():
                break

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                src_idx = futures.pop(future)
                attempted += 1
                try:
                    training_row, audit = future.result()
                except Exception as error:
                    training_row = None
                    audit = {
                        "src_idx": src_idx,
                        "status": "rejected",
                        "stage": "uncaught",
                        "reason": str(error),
                        "usage": {},
                    }
                audit["generator_model"] = args.generator_model
                audit["judge_model"] = args.judge_model
                append_jsonl(audit_output, audit)
                usage = usage_sum(usage, audit.get("usage") or {})
                if training_row is not None:
                    append_jsonl(output, training_row)
                    accepted += 1
                else:
                    rejected += 1

                if attempted % 10 == 0 or training_row is not None:
                    rate = accepted / attempted if attempted else 0.0
                    print(
                        f"progress attempted={attempted} accepted={accepted} rejected={rejected} "
                        f"accept_rate={rate:.1%} tokens={usage['prompt_tokens']}/{usage['completion_tokens']}",
                        flush=True,
                    )

                target_reached = args.max_accepted and (
                    previously_accepted + accepted >= args.max_accepted
                )
                if not target_reached:
                    submit_one()

            if args.max_accepted and previously_accepted + accepted >= args.max_accepted:
                for future in futures:
                    future.cancel()
                futures.clear()

    elapsed = time.time() - started
    result = {
        "source": str(args.src.resolve()),
        "source_rows": len(rows),
        "source_sha256": sha256(args.src),
        "excluded_eval_indices": len(excluded),
        "exclude_sources": [str(path.resolve()) for path in exclude_paths],
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "seed": args.seed,
        "attempted_this_run": attempted,
        "accepted_this_run": accepted,
        "accepted_before_run": previously_accepted,
        "accepted_total": previously_accepted + accepted,
        "rejected_this_run": rejected,
        "accept_rate_this_run": accepted / attempted if attempted else 0.0,
        "output": str(args.out.resolve()),
        "output_rows": sum(1 for _ in args.out.open(encoding="utf-8")),
        "output_sha256": sha256(args.out),
        "audit_log": str(args.audit_log.resolve()),
        "usage_this_run": usage,
        "elapsed_seconds": round(elapsed, 3),
    }
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
