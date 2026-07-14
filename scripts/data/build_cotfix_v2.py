#!/usr/bin/env python3
"""Build a semantically repaired recommendation-CoT dataset for I-18.

The pipeline is deliberately split into three auditable stages:

``prepare``
    Find recommendation CoTs in D(O1) that end in a syntactically non-terminal
    state and join *history-side only* Caption/Tag evidence from O3.  Target
    answers and target metadata are never written to the request file.

``generate``
    Ask an independent generator/judge pair to continue only the missing tail.
    Every added SID must already occur in the student prompt.  A rejected first
    draft receives one repair attempt and is judged again.  The stage is
    resumable and keeps model, token-usage, and verdict provenance.

``build``
    Patch the one retained CoT row per recommendation prompt group in the
    registered I-10 training set.  Row order, prompts, answers, non-CoT rows,
    and the 164 O2-derived teacher rows remain byte/logically unchanged.

This is not the old deterministic cotfix_v1 closure.  No generic templated
suffix is used, and the failed v1 checkpoint is never an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_O1 = ROOT / "assets/derived/processed/data_final.jsonl"
DEFAULT_O3 = ROOT / "assets/official/sft_aligned/baseline_caption_tag_lists.parquet"
DEFAULT_PARENT = ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl"
DEFAULT_O2_TEACHER = ROOT / "assets/derived/processed/action_distill_v5.jsonl"
DEFAULT_REQUESTS = ROOT / "logs/data/cotfix_v2_requests.jsonl"
DEFAULT_PREPARE_SUMMARY = ROOT / "logs/data/cotfix_v2_prepare_summary.json"
DEFAULT_GENERATIONS = ROOT / "logs/data/cotfix_v2_generations.jsonl"
DEFAULT_GENERATION_AUDIT = ROOT / "logs/data/cotfix_v2_generation_audit.jsonl"
DEFAULT_GENERATION_SUMMARY = ROOT / "logs/data/cotfix_v2_generation_summary.json"
DEFAULT_OUTPUT = ROOT / "assets/derived/processed/data_seed_teacher_cotfix_v2.jsonl"
DEFAULT_BUILD_AUDIT = ROOT / "logs/data/seed_teacher_cotfix_v2_audit.json"
DEFAULT_ENV = ROOT / "configs/secrets/deepseek_api.env"
MODEL = ROOT / "models/OneReason-0.8B-pretrain-competition"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
ITEM_RE = re.compile(
    r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
DOMAIN_RE = re.compile(r"<\|(video|prod|ad|living)_begin\|>")
TERMINAL_CHARS = set("。！？!?…”\"」』)）】》]")
FORBIDDEN_TEXT = (
    "<think>",
    "</think>",
    "正确答案",
    "目标答案",
    "gold",
    "其他同类内容等",
    "相关同类内容等",
    "其余相近内容等",
)
REQUIRED_JUDGE_CHECKS = (
    "verdict_correct",
    "coherent",
    "history_grounded",
    "no_sid_repeat",
    "domain_consistent",
    "no_target_leak",
    "syntax_closed",
    "not_generic",
    "minimal_tail_only",
)

# Words that describe the reporting scaffold rather than the unfinished
# recommendation clause.  Removing them only for evidence ranking prevents a
# generic Caption ("用户观看相关视频内容") from outranking an exact entity or
# category match such as "洗衣液", "儿童玩具", or "手机清理".
FOCUS_GENERIC_PHRASES = (
    "用户",
    "兴趣",
    "相关",
    "内容",
    "行为",
    "表现",
    "开始",
    "浏览",
    "转发",
    "点击",
    "购买",
    "此外",
    "例如",
    "其中",
    "进行",
    "体现",
    "反映",
    "显示",
    "视频",
    "商品",
    "广告",
    "直播",
    "主题",
    "可能",
    "近期",
    "这一",
    "模式",
    "结合",
    "同时",
    "多次",
    "收藏",
    "深度",
    "互动",
    "点赞",
    "持续",
    "关注",
    "复购",
    "稳定",
    "核心",
)
FOCUS_STOP_UNIGRAMS = set("的了和与及是在对为如等类有中上其")
FOCUS_QUERY_EXPANSIONS = {
    "保健品": "营养健康传统滋补膳食营养",
    "食品生鲜": "速食干货饮料水果蔬菜肉类粮油调味",
    "烹饪": "家常菜做法制作料理美食",
    "仿妆": "妆容化妆模仿妆容",
    "服饰": "女装男装穿搭衣服",
    "鞋履": "女鞋男鞋棉鞋休闲鞋",
    "鞋服": "女鞋女装男鞋男装服饰穿搭",
    "美妆": "彩妆化妆品护肤粉底眼妆美甲",
    "个护产品": "个护清洁洗面奶沐浴露身体护理",
    "衍生品模型": "游戏周边武器模型合金摆件",
}

GENERATOR_SYSTEM = """你是推荐系统训练数据修复专家。输入是一段被上游生成长度限制截断的中文 CoT 前缀，以及只来自学生题面历史的 Caption/Tag 证据。

任务是只续写缺失尾部，绝不重写、复述或修改前缀。

硬规则：
1. 不得接触、猜测或暗示目标答案；证据区不含目标答案。
2. 只能引用证据区中标为“可续写”的历史 SID，必须从证据行 SID 的第一个字符开始逐字完整复制；禁止输出含有 domain 字样的占位符。标为“已引用”的 SID 仅供理解，禁止再次输出。
3. 优先完成当前未闭合的短语、括号、反引号、枚举或编号段；只补必要的 1-3 句自然收束。
4. 续写必须结合当前段落的具体主题和历史证据，拒绝“其他同类内容等、进一步印证偏好”之类空泛套话。
5. 若前缀以“如”“(”“（”结尾，至少补入一个证据区的完整 SID；若以“、”结尾，续写必须先补入一个完整 SID，再闭合列表。
6. 括号应先闭合、句号应放在闭括号之后，禁止“。)”或“。）”；最终以自然中文句号、问号或叹号结束。
7. 若前缀已经以“2.”、“3.”等编号标记结束，直接续写该条的标题或正文，禁止再次输出相同编号。
8. 不要重新总结整篇 CoT，不要开启与当前尾段无关的新主题，不要输出 Markdown 代码块。
9. 如果原文其实是完整自然结尾，返回 KEEP；否则返回 TRUNCATED。

只输出严格 JSON：
{"verdict":"TRUNCATED或KEEP","continuation":"只含新增文本；KEEP时为空串","confidence":0.0}"""

JUDGE_SYSTEM = """你是独立的推荐训练数据修复质检员。你只能使用原截断前缀和历史侧证据，不能使用目标答案。

逐项检查候选 verdict/continuation：
- verdict_correct：TRUNCATED/KEEP 判断正确；
- coherent：拼接后语法和语义自然，紧接原文；
- history_grounded：新增 SID 和具体事实均由历史证据支持；
- no_sid_repeat：没有再次输出前缀已经引用过的 SID；
- domain_consistent：新增示例 SID 与当前未闭合尾段的优先域一致；
- no_target_leak：没有目标答案、目标元数据或猜答案措辞；
- syntax_closed：当前未闭合的括号、反引号、枚举和句子已正确收束；
- not_generic：不是空泛模板补句；
- minimal_tail_only：只修尾部，没有重写前文或扩写无关新段。

只有全部检查为 true 且质量可直接进入 SFT 时，accept=true、score=5。只输出严格 JSON：
{"accept":true,"score":5,"checks":{"verdict_correct":true,"coherent":true,"history_grounded":true,"no_sid_repeat":true,"domain_consistent":true,"no_target_leak":true,"syntax_closed":true,"not_generic":true,"minimal_tail_only":true},"reason":"一句话"}"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def canonical_training_row(row: dict[str, Any]) -> str:
    normalized = {
        "instruction": str(row.get("instruction") or ""),
        "input": str(row.get("input") or ""),
        "output": str(row.get("output") or ""),
        "history": row.get("history") or [],
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


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


def normalize_space(text: Any, limit: int = 0) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:limit] if limit else value


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(content_text(item) for item in value)
    if isinstance(value, dict):
        return content_text(value.get("text", value.get("content", "")))
    return "" if value is None else str(value)


def recommendation_row(row: dict[str, Any]) -> bool:
    if "</think>" not in str(row.get("output", "")):
        return False
    return "该用户最近" in row["output"].split("</think>", 1)[1]


def prompt_core(row: dict[str, Any]) -> str:
    return MODE_SUFFIX_RE.sub("", str(row.get("input", "")).rstrip())


def candidate_id(core: str, prefix: str) -> str:
    return hashlib.sha256((core + "\0" + prefix).encode("utf-8")).hexdigest()[:24]


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def collect_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if recommendation_row(row):
            groups[prompt_core(row)].append(index)

    requests: list[dict[str, Any]] = []
    for core, indexes in groups.items():
        thinks = set()
        for index in indexes:
            match = THINK_RE.search(rows[index]["output"])
            if match and match.group(1).strip():
                thinks.add(match.group(1).strip())
        if len(thinks) != 1:
            raise AssertionError(f"recommendation group has {len(thinks)} CoTs")
        prefix = next(iter(thinks))
        if prefix[-1] in TERMINAL_CHARS:
            continue
        history_sids = unique_in_order(ITEM_RE.findall(core))
        requests.append(
            {
                "candidate_id": candidate_id(core, prefix),
                "core_sha256": hashlib.sha256(core.encode("utf-8")).hexdigest(),
                "prompt_core": core,
                "prefix": prefix,
                "prefix_chars": len(prefix),
                "source_group_rows": len(indexes),
                "history_sids": history_sids,
            }
        )

    requests.sort(key=lambda row: row["candidate_id"])
    lengths = sorted(int(row["prefix_chars"]) for row in requests)
    summary = {
        "recommendation_rows": sum(len(indexes) for indexes in groups.values()),
        "prompt_groups": len(groups),
        "candidate_groups": len(requests),
        "candidate_source_rows": sum(int(row["source_group_rows"]) for row in requests),
        "prefix_chars": {
            "min": lengths[0],
            "median": statistics.median(lengths),
            "p90": lengths[round((len(lengths) - 1) * 0.9)],
            "max": lengths[-1],
        },
        "ending_chars": dict(sorted(Counter(row["prefix"][-1] for row in requests).items())),
    }
    return requests, summary


def attach_o3_history_evidence(
    requests_rows: list[dict[str, Any]], o3_path: Path
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    wanted = {row["prompt_core"] for row in requests_rows}
    by_core = {row["prompt_core"]: row for row in requests_rows}
    matched: set[str] = set()
    parquet = pq.ParquetFile(o3_path)
    rows_seen = 0
    for batch in parquet.iter_batches(
        columns=["record_id", "messages", "sid_token_list", "caption_list", "tag_list"],
        batch_size=128,
    ):
        for aligned in batch.to_pylist():
            rows_seen += 1
            messages = json.loads(aligned["messages"])
            user = next(
                (
                    content_text(message.get("content"))
                    for message in messages
                    if message.get("role") == "user"
                ),
                "",
            )
            if user not in wanted or user in matched:
                continue

            request = by_core[user]
            history_order = request["history_sids"]
            history_set = set(history_order)
            best: dict[str, tuple[str, str]] = {}
            for sid, caption, tag in zip(
                aligned["sid_token_list"], aligned["caption_list"], aligned["tag_list"]
            ):
                if sid not in history_set:
                    continue
                candidate = (normalize_space(caption, 320), normalize_space(tag, 160))
                if sid not in best or sum(map(len, candidate)) > sum(map(len, best[sid])):
                    best[sid] = candidate

            evidence = []
            for position, sid in enumerate(history_order):
                caption, tag = best.get(sid, ("", ""))
                domain_match = DOMAIN_RE.search(sid)
                evidence.append(
                    {
                        "position": position,
                        "sid": sid,
                        "domain": domain_match.group(1) if domain_match else "unknown",
                        "caption": caption,
                        "tag": tag,
                    }
                )
            request["history_evidence"] = evidence
            request["o3_record_id"] = int(aligned["record_id"])
            matched.add(user)

    missing = wanted - matched
    if rows_seen != 19_204:
        raise AssertionError(f"expected 19,204 O3 rows, got {rows_seen}")
    if missing:
        raise AssertionError(f"O3 did not align {len(missing)} candidate prompt groups")

    mapped_counts = [
        sum(bool(item["caption"] or item["tag"]) for item in row["history_evidence"])
        for row in requests_rows
    ]
    return {
        "o3_rows_scanned": rows_seen,
        "candidate_groups_aligned": len(matched),
        "history_semantic_positions": {
            "total": sum(mapped_counts),
            "min_per_group": min(mapped_counts),
            "median_per_group": statistics.median(mapped_counts),
            "max_per_group": max(mapped_counts),
        },
    }


def prepare(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.o1)
    if len(rows) != 32_480:
        raise AssertionError(f"expected 32,480 D(O1) rows, got {len(rows)}")
    requests_rows, candidate_summary = collect_candidates(rows)
    if candidate_summary["prompt_groups"] != 6_460:
        raise AssertionError(f"recommendation group signature drifted: {candidate_summary}")
    if candidate_summary["candidate_groups"] != 538:
        raise AssertionError(f"expected 538 non-terminal groups: {candidate_summary}")
    if candidate_summary["candidate_source_rows"] != 1_845:
        raise AssertionError(f"expected 1,845 source rows: {candidate_summary}")

    o3_summary = attach_o3_history_evidence(requests_rows, args.o3)
    write_jsonl(args.requests, requests_rows)
    summary = {
        "stage": "prepare",
        "asset_class": "construction_intermediate_D(O1,O3)",
        "upstream": {
            "O1_D_format": {
                "path": str(args.o1.resolve()),
                "rows": len(rows),
                "sha256": sha256(args.o1),
            },
            "O3": {
                "path": str(args.o3.resolve()),
                "rows": 19_204,
                "sha256": sha256(args.o3),
            },
        },
        "candidate_audit": candidate_summary,
        "history_only_evidence": o3_summary,
        "target_answer_or_metadata_in_requests": 0,
        "requests": {
            "path": str(args.requests.resolve()),
            "rows": len(requests_rows),
            "sha256": sha256(args.requests),
        },
    }
    args.prepare_summary.parent.mkdir(parents=True, exist_ok=True)
    args.prepare_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def infer_tail_domains(prefix: str) -> list[str]:
    # Keep the window local to the unfinished clause.  A wider window lets an
    # unrelated SID/domain from the previous paragraph override the tail.
    tail = prefix[-350:]
    weak_tail = tail[-160:]
    strong_patterns = {
        "video": (r"短视频域", r"视频域"),
        "prod": (r"电商域", r"商品域"),
        "ad": (r"广告域",),
        "living": (r"直播域",),
    }
    weak_patterns = {
        "video": (r"短视频", r"视频"),
        "prod": (
            r"购物",
            r"商品",
            r"购买",
            r"加购",
            r"保健品",
            r"食品",
            r"生鲜",
            r"服饰",
            r"鞋履",
            r"鞋服",
            r"美妆",
            r"个护",
            r"沐浴露",
            r"玩具",
            r"护肤品",
            r"私护",
        ),
        "ad": (r"广告",),
        "living": (r"直播", r"主播"),
    }
    ranked: list[tuple[int, str]] = []
    for domain in strong_patterns:
        position = -1
        for expression in strong_patterns[domain]:
            for match in re.finditer(expression, tail):
                position = max(position, match.start())
        weak_offset = len(tail) - len(weak_tail)
        for expression in weak_patterns[domain]:
            for match in re.finditer(expression, weak_tail):
                position = max(position, weak_offset + match.start())
        for match in re.finditer(rf"<\|{domain}_begin\|>", tail):
            position = max(position, match.start())
        if position >= 0:
            ranked.append((position, domain))
    ranked.sort(reverse=True)
    return [domain for _, domain in ranked]


def semantic_bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", ITEM_RE.sub("", text))
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def unfinished_focus(prefix: str) -> str:
    """Return the local clause that the continuation must complete."""

    tail = prefix[-260:]
    cuts = []
    for delimiter in ("\n", "。", "！", "？", "；", "：", "此外，"):
        position = tail.rfind(delimiter)
        if position >= 0:
            cuts.append(position + len(delimiter))
    focus = tail[max(cuts) :] if cuts else tail[-110:]
    # In a partially written example list, the last populated item is a better
    # semantic anchor than the whole preceding sentence.  Do not cut when the
    # source literally ends at the separator because that separator itself is
    # what must be repaired.
    separator = focus.rfind("、")
    before_separator = focus[:separator].rstrip(" `") if separator >= 0 else ""
    previous_example_ends_here = bool(
        separator >= 0 and ITEM_RE.search(before_separator[-100:])
    )
    if (
        previous_example_ends_here
        and len(focus[separator + 1 :].strip()) >= 4
    ):
        focus = focus[separator + 1 :]
    return focus[-110:]


def focus_ngrams(text: str, *, expand_query: bool = False) -> set[tuple[int, str]]:
    cleaned = ITEM_RE.sub("", text)
    if expand_query:
        additions = [
            expansion
            for trigger, expansion in FOCUS_QUERY_EXPANSIONS.items()
            if trigger in cleaned
        ]
        if additions:
            cleaned += " " + " ".join(additions)
    for phrase in FOCUS_GENERIC_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    runs = re.findall(r"[\u4e00-\u9fff]+", cleaned)
    return {
        (size, run[index : index + size])
        for run in runs
        for size in (1, 2, 3, 4)
        for index in range(max(0, len(run) - size + 1))
        if size != 1 or run[index] not in FOCUS_STOP_UNIGRAMS
    }


def select_history_evidence(request: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    prefix = request["prefix"]
    # Rank by the unfinished clause first.  A single broad window can be
    # dominated by the previous numbered paragraph (for example, game-related
    # terms just before a dangling laundry-detergent example).  The wider tail
    # remains a secondary tie-breaker so short/generic final clauses still have
    # enough context.
    tail = prefix[-500:]
    local_tail = prefix[-120:]
    tail_domains = infer_tail_domains(prefix)
    primary = tail_domains[0] if tail_domains else None
    prefix_sids = set(ITEM_RE.findall(prefix))
    tail_sids = set(ITEM_RE.findall(tail))
    local_features = semantic_bigrams(local_tail)
    context_features = semantic_bigrams(tail)

    # Score the final unfinished clause with character 2-4 grams and an IDF
    # term computed inside this prompt history.  Longer, rarer matches dominate
    # generic report vocabulary, while the existing broad-overlap scores remain
    # deterministic fallbacks when the clause itself carries little semantics.
    focus_features = focus_ngrams(unfinished_focus(prefix), expand_query=True)
    evidence_focus_features = {
        item["sid"]: focus_ngrams(item["caption"] + item["tag"])
        for item in request["history_evidence"]
    }
    document_frequency = Counter(
        feature
        for features in evidence_focus_features.values()
        for feature in features
    )
    document_count = len(evidence_focus_features)

    ranked_unseen_primary = []
    ranked_unseen_other = []
    ranked_seen_context = []
    for item in request["history_evidence"]:
        sid = item["sid"]
        has_semantics = bool(item["caption"] or item["tag"])
        semantic_features = semantic_bigrams(item["caption"] + item["tag"])
        local_overlap = len(local_features & semantic_features)
        context_overlap = len(context_features & semantic_features)
        focus_overlap = focus_features & evidence_focus_features[sid]
        focus_score = sum(
            (0.25 if size == 1 else (size - 1) ** 2)
            * (math.log((document_count + 1) / (document_frequency[feature] + 1)) + 1)
            for size, feature in focus_overlap
        )
        rank = (
            -focus_score,
            -local_overlap,
            -context_overlap,
            0 if has_semantics else 1,
            int(item["position"]),
        )
        if sid in prefix_sids:
            if sid in tail_sids:
                ranked_seen_context.append((rank, item))
        elif primary and item["domain"] == primary:
            ranked_unseen_primary.append((rank, item))
        else:
            ranked_unseen_other.append((rank, item))

    for values in (ranked_unseen_primary, ranked_unseen_other, ranked_seen_context):
        values.sort(key=lambda pair: pair[0])

    chosen: list[tuple[dict[str, Any], str]] = []
    # Preserve a broad enough same-domain pool for long histories: the best
    # literal match can otherwise sit just outside a narrow top-12 list even
    # when it is the only semantically exact completion (for example, a
    # motorcycle video after a long game-heavy paragraph).
    primary_quota = min(18, limit - 2) if primary else 0
    chosen.extend((item, "可续写") for _, item in ranked_unseen_primary[:primary_quota])
    remaining = limit - len(chosen) - min(2, len(ranked_seen_context))
    chosen.extend((item, "可续写") for _, item in ranked_unseen_other[: max(0, remaining)])
    chosen.extend((item, "已引用，仅供理解，禁止再次输出") for _, item in ranked_seen_context[:2])

    if len(chosen) < limit:
        selected_ids = {item["sid"] for item, _ in chosen}
        leftovers = ranked_unseen_primary[primary_quota:] + ranked_unseen_other[max(0, remaining):]
        leftovers.sort(key=lambda pair: pair[0])
        for _, item in leftovers:
            if item["sid"] in selected_ids:
                continue
            chosen.append((item, "可续写"))
            selected_ids.add(item["sid"])
            if len(chosen) == limit:
                break

    selected = [{**item, "evidence_role": role} for item, role in chosen[:limit]]
    return selected


def render_evidence(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        caption = normalize_space(item.get("caption"), 140) or "无可用Caption"
        tag = normalize_space(item.get("tag"), 100) or "无可用Tag"
        lines.append(
            f"- [{item['evidence_role']}] {item['sid']} | Tag={tag} | Caption={caption}"
        )
    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("response contains no JSON object")


def usage_add(total: dict[str, int], current: dict[str, Any]) -> dict[str, int]:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int(current.get(key) or 0)
    return total


class ChatClient:
    def __init__(self, base_url: str, api_key: str, timeout: int, retries: int):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, dict[str, Any]]:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error = "unknown API error"
        for attempt in range(self.retries):
            try:
                response = self.session().post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    payload = response.text[:400]
                    if "balance" in payload.lower() or "余额" in payload:
                        payload = "insufficient balance"
                    raise RuntimeError(f"HTTP {response.status_code}: {payload}")
                payload = response.json()
                content = payload["choices"][0]["message"].get("content") or ""
                if not content.strip():
                    raise RuntimeError("empty assistant content")
                return content.strip(), payload.get("usage") or {}
            except Exception as error:
                last_error = str(error)
                if attempt + 1 < self.retries:
                    time.sleep(min(3 * (attempt + 1), 12))
        raise RuntimeError(last_error)


def unmatched_family(text: str, left: str, right: str) -> tuple[int, int]:
    """Count a bracket family while accepting mixed ASCII/CJK glyphs."""

    depth = 0
    unexpected_close = 0
    for char in text:
        if char in left:
            depth += 1
        elif char in right:
            if depth:
                depth -= 1
            else:
                unexpected_close += 1
    return depth, unexpected_close


def normalize_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    verdict = str(raw.get("verdict") or "").strip().upper()
    continuation = raw.get("continuation", "")
    if not isinstance(continuation, str):
        raise ValueError("continuation is not a string")
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "verdict": verdict,
        "continuation": continuation.strip("\n").rstrip(),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def validate_proposal(request: dict[str, Any], proposal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    verdict = proposal["verdict"]
    continuation = proposal["continuation"]
    if verdict not in {"TRUNCATED", "KEEP"}:
        return ["invalid verdict"]
    if verdict == "KEEP":
        if continuation:
            errors.append("KEEP has a non-empty continuation")
        return errors

    prefix = request["prefix"]
    combined = prefix + continuation
    if not 2 <= len(continuation) <= 600:
        errors.append(f"continuation length {len(continuation)} outside 2-600")
    if len(combined) > 2_700:
        errors.append("combined CoT exceeds 2,700 characters")
    if not re.search(r"[。！？][”\"」』]?$", combined):
        errors.append("combined CoT lacks natural Chinese terminal punctuation")
    if re.search(r"[。！？][)）\]】]", continuation):
        errors.append("sentence punctuation appears before a closing bracket")
    list_close_pattern = r"、\s*[)）\]】]"
    if len(re.findall(list_close_pattern, combined)) > len(
        re.findall(list_close_pattern, prefix)
    ):
        errors.append("list separator is followed directly by a closing bracket")
    lowered = continuation.lower()
    for forbidden in FORBIDDEN_TEXT:
        if forbidden.lower() in lowered:
            errors.append(f"forbidden text: {forbidden}")

    history_sids = set(request["history_sids"])
    introduced_sids = set(ITEM_RE.findall(continuation))
    if not introduced_sids <= history_sids:
        errors.append(f"introduced non-history SIDs: {sorted(introduced_sids-history_sids)[:3]}")
    stripped_item_syntax = ITEM_RE.sub("", continuation)
    if "<|" in stripped_item_syntax or "<s_" in stripped_item_syntax:
        errors.append("continuation contains malformed item-token syntax")
    prefix_sids = set(ITEM_RE.findall(prefix))
    repeated_sids = introduced_sids & prefix_sids
    if repeated_sids:
        errors.append(f"repeated prefix SIDs: {sorted(repeated_sids)[:3]}")
    tail_domains = infer_tail_domains(prefix)
    if tail_domains and introduced_sids:
        primary = tail_domains[0]
        wrong_domains = {
            match.group(1)
            for sid in introduced_sids
            if (match := DOMAIN_RE.search(sid)) and match.group(1) != primary
        }
        if wrong_domains:
            errors.append(
                f"history SID domain mismatches current tail: expected {primary}, got {sorted(wrong_domains)}"
            )

    for left, right, label in (("(（", ")）", "parenthesis"), ("[【", "]】", "bracket")):
        prefix_depth, prefix_unexpected = unmatched_family(prefix, left, right)
        combined_depth, combined_unexpected = unmatched_family(combined, left, right)
        if combined_depth:
            errors.append(f"failed to close {label}: {prefix_depth}->{combined_depth}")
        if combined_unexpected > prefix_unexpected:
            errors.append(
                f"introduced unexpected closing {label}: {prefix_unexpected}->{combined_unexpected}"
            )
    if prefix.count("`") % 2 and combined.count("`") % 2:
        errors.append("failed to close backtick")

    max_overlap = min(120, len(prefix), len(continuation))
    overlap = max(
        (size for size in range(1, max_overlap + 1) if prefix.endswith(continuation[:size])),
        default=0,
    )
    if overlap >= 12:
        errors.append(f"continuation repeats {overlap} prefix-tail characters")
    if prefix.endswith(("如", "(", "（", "`")) and not ITEM_RE.search(continuation):
        errors.append("open example clause did not add a grounded history SID")
    if prefix.endswith("、") and ITEM_RE.match(continuation.lstrip()) is None:
        errors.append("dangling list separator was not followed by a history SID")
    prefix_number = re.search(r"(?:^|\s)(\d+)[.、．]\s*$", prefix)
    if prefix_number and re.match(
        rf"\s*{re.escape(prefix_number.group(1))}[.、．]", continuation
    ):
        errors.append("continuation repeats the dangling numbered-list marker")
    return errors


def generator_user(
    request: dict[str, Any], evidence_text: str, previous: dict[str, Any] | None, feedback: str
) -> str:
    text = (
        f"截断候选前缀：\n{request['prefix']}\n\n"
        f"必须紧接的局部尾句：{unfinished_focus(request['prefix'])}\n\n"
        f"当前尾段优先域：{(infer_tail_domains(request['prefix']) or ['未明确'])[0]}\n\n"
        f"仅供续写的历史侧证据（不含目标答案）：\n{evidence_text}\n\n"
    )
    if previous is not None:
        text += (
            "上一版候选：\n"
            + json.dumps(previous, ensure_ascii=False)
            + f"\n独立质检反馈：{feedback}\n请据此重新生成合格尾部。"
        )
    else:
        text += "请判断并只续写缺失尾部。"
    return text


def judge_user(request: dict[str, Any], proposal: dict[str, Any], evidence_text: str) -> str:
    return (
        f"原前缀：\n{request['prefix']}\n\n"
        f"必须紧接的局部尾句：{unfinished_focus(request['prefix'])}\n\n"
        f"候选：\n{json.dumps(proposal, ensure_ascii=False)}\n\n"
        f"历史侧证据（不含目标答案）：\n{evidence_text}"
    )


def judge_accepted(raw: dict[str, Any]) -> tuple[bool, str]:
    checks = raw.get("checks") if isinstance(raw.get("checks"), dict) else {}
    score = int(raw.get("score") or 0)
    accepted = bool(raw.get("accept")) and score == 5 and all(
        checks.get(key) is True for key in REQUIRED_JUDGE_CHECKS
    )
    reason = normalize_space(raw.get("reason"), 500)
    if not accepted and not reason:
        reason = f"judge rejected: score={score}, checks={checks}"
    return accepted, reason


def process_request(
    request: dict[str, Any],
    client: ChatClient,
    generator_model: str,
    judge_model: str,
    repair_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = select_history_evidence(request)
    evidence_text = render_evidence(selected)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempts = []
    previous = None
    feedback = ""

    for attempt_index in range(repair_attempts + 1):
        generator_text, generator_usage = client.call(
            generator_model,
            GENERATOR_SYSTEM,
            generator_user(request, evidence_text, previous, feedback),
            max_tokens=650,
            temperature=0.2,
        )
        usage_add(usage, generator_usage)
        attempt: dict[str, Any] = {
            "attempt": attempt_index + 1,
            "generator_raw": generator_text,
            "generator_usage": generator_usage,
        }
        try:
            proposal = normalize_proposal(extract_json_object(generator_text))
            programmatic_errors = validate_proposal(request, proposal)
        except Exception as error:
            proposal = None
            programmatic_errors = [str(error)]
        attempt["proposal"] = proposal
        attempt["programmatic_errors"] = programmatic_errors
        if programmatic_errors:
            feedback = "; ".join(programmatic_errors)
            previous = proposal
            attempts.append(attempt)
            continue

        judge_text, judge_usage = client.call(
            judge_model,
            JUDGE_SYSTEM,
            judge_user(request, proposal, evidence_text),
            max_tokens=500,
            temperature=0.0,
        )
        usage_add(usage, judge_usage)
        attempt["judge_raw"] = judge_text
        attempt["judge_usage"] = judge_usage
        try:
            judge = extract_json_object(judge_text)
            accepted, feedback = judge_accepted(judge)
        except Exception as error:
            judge = None
            accepted = False
            feedback = str(error)
        attempt["judge"] = judge
        attempt["accepted"] = accepted
        attempts.append(attempt)
        if accepted:
            generation = {
                "candidate_id": request["candidate_id"],
                "status": "accepted",
                **proposal,
                "generator_model": generator_model,
                "judge_model": judge_model,
                "attempts": attempt_index + 1,
                "usage": usage,
            }
            audit = {
                "candidate_id": request["candidate_id"],
                "status": "accepted",
                "selected_history_evidence": selected,
                "attempt_records": attempts,
                "usage": usage,
            }
            return generation, audit
        previous = proposal

    generation = {
        "candidate_id": request["candidate_id"],
        "status": "rejected",
        "reason": feedback or "all attempts rejected",
        "generator_model": generator_model,
        "judge_model": judge_model,
        "attempts": len(attempts),
        "usage": usage,
    }
    audit = {
        "candidate_id": request["candidate_id"],
        "status": "rejected",
        "selected_history_evidence": selected,
        "attempt_records": attempts,
        "reason": generation["reason"],
        "usage": usage,
    }
    return generation, audit


def latest_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in read_jsonl(path):
            latest[str(row["candidate_id"])] = row
    return latest


def generate(args: argparse.Namespace) -> None:
    requests_rows = read_jsonl(args.requests)
    if len(requests_rows) != 538:
        raise AssertionError(f"expected 538 prepared requests, got {len(requests_rows)}")
    if len({row["candidate_id"] for row in requests_rows}) != len(requests_rows):
        raise AssertionError("duplicate candidate IDs in requests")

    previous = latest_by_candidate(args.generations)
    pending = []
    for request in requests_rows:
        old = previous.get(request["candidate_id"])
        if old and old.get("status") == "accepted":
            current_errors = validate_proposal(request, old)
            if not (args.retry_invalid_accepted and current_errors):
                continue
        if old and old.get("status") == "rejected" and not args.retry_rejected:
            continue
        pending.append(request)
    if args.start:
        pending = pending[args.start :]
    if args.n:
        pending = pending[: args.n]

    env = load_env(args.env_file)
    base_url = os.environ.get("TEACHER_BASE", env.get("YUNWU_BASE_URL", ""))
    api_key = os.environ.get("TEACHER_KEY", env.get("YUNWU_API_KEY", ""))
    if not base_url or not api_key:
        raise RuntimeError("missing teacher endpoint/key")
    if args.generator_model == args.judge_model:
        raise ValueError("generator and judge model variants must differ")

    client = ChatClient(base_url, api_key, args.timeout, args.retries)
    args.generations.parent.mkdir(parents=True, exist_ok=True)
    args.generation_audit.parent.mkdir(parents=True, exist_ok=True)
    attempted = accepted = rejected = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    started = time.time()
    print(
        f"prepared={len(requests_rows)} pending={len(pending)} "
        f"generator={args.generator_model} judge={args.judge_model}",
        flush=True,
    )

    with args.generations.open("a", encoding="utf-8") as output, args.generation_audit.open(
        "a", encoding="utf-8"
    ) as audit_output, ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(pending)
        futures = {}

        def submit_one() -> bool:
            try:
                request = next(iterator)
            except StopIteration:
                return False
            future = pool.submit(
                process_request,
                request,
                client,
                args.generator_model,
                args.judge_model,
                args.repair_attempts,
            )
            futures[future] = request
            return True

        for _ in range(args.workers):
            if not submit_one():
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                request = futures.pop(future)
                attempted += 1
                try:
                    generation, audit = future.result()
                except Exception as error:
                    generation = {
                        "candidate_id": request["candidate_id"],
                        "status": "rejected",
                        "reason": str(error),
                        "generator_model": args.generator_model,
                        "judge_model": args.judge_model,
                        "attempts": 0,
                        "usage": {},
                    }
                    audit = {**generation, "stage": "uncaught"}
                append_jsonl(output, generation)
                append_jsonl(audit_output, audit)
                usage_add(usage, generation.get("usage") or {})
                if generation["status"] == "accepted":
                    accepted += 1
                else:
                    rejected += 1
                if attempted % 10 == 0 or generation["status"] == "rejected":
                    print(
                        f"progress attempted={attempted} accepted={accepted} rejected={rejected} "
                        f"tokens={usage['prompt_tokens']}/{usage['completion_tokens']}",
                        flush=True,
                    )
                submit_one()

    latest = latest_by_candidate(args.generations)
    accepted_total = sum(
        latest.get(row["candidate_id"], {}).get("status") == "accepted"
        for row in requests_rows
    )
    rejected_total = sum(
        latest.get(row["candidate_id"], {}).get("status") == "rejected"
        for row in requests_rows
    )
    summary = {
        "stage": "generate",
        "requests": len(requests_rows),
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "repair_attempts": args.repair_attempts,
        "attempted_this_run": attempted,
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "accepted_total_latest": accepted_total,
        "rejected_total_latest": rejected_total,
        "unresolved_total": len(requests_rows) - accepted_total - rejected_total,
        "usage_this_run": usage,
        "elapsed_seconds": round(time.time() - started, 3),
        "generations": {
            "path": str(args.generations.resolve()),
            "sha256": sha256(args.generations),
        },
        "audit_log": {
            "path": str(args.generation_audit.resolve()),
            "sha256": sha256(args.generation_audit),
        },
    }
    args.generation_summary.parent.mkdir(parents=True, exist_ok=True)
    args.generation_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def token_length_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    lengths = []
    output_lengths = []
    for start in range(0, len(rows), 128):
        batch = rows[start : start + 128]
        full_text = [
            row["instruction"] + "\n" + row["input"] + "\n" + row["output"] for row in batch
        ]
        outputs = [row["output"] for row in batch]
        lengths.extend(
            tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_length=True,
            )["length"]
        )
        output_lengths.extend(
            tokenizer(
                outputs,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_length=True,
            )["length"]
        )
    ordered = sorted(lengths)
    return {
        "approx_full_tokens_max_without_chat_overhead": max(lengths),
        "approx_full_tokens_p99_without_chat_overhead": ordered[round((len(ordered) - 1) * 0.99)],
        "output_tokens_max": max(output_lengths),
        "rows_over_16000_with_8_token_margin": sum(length + 8 > 16_000 for length in lengths),
        "rows_over_cutoff_16384_with_8_token_margin": sum(
            length + 8 > 16_384 for length in lengths
        ),
    }


def build(args: argparse.Namespace) -> None:
    from build_seed_scoremax_v1 import target_token_mix, task_of

    requests_rows = read_jsonl(args.requests)
    if len(requests_rows) != 538:
        raise AssertionError(f"expected 538 requests, got {len(requests_rows)}")
    request_by_id = {row["candidate_id"]: row for row in requests_rows}
    latest = latest_by_candidate(args.generations)
    missing = [cid for cid in request_by_id if latest.get(cid, {}).get("status") != "accepted"]
    if missing:
        raise AssertionError(f"cannot build with {len(missing)} non-accepted candidates")
    latest_audits = latest_by_candidate(args.generation_audit)
    judge_failures = []
    for cid in request_by_id:
        audit_row = latest_audits.get(cid)
        if not audit_row or audit_row.get("status") != "accepted":
            judge_failures.append(f"{cid}: missing accepted audit")
            continue
        winning = [
            record
            for record in audit_row.get("attempt_records", [])
            if record.get("accepted") is True
        ]
        if len(winning) != 1:
            judge_failures.append(f"{cid}: accepted-attempt count={len(winning)}")
            continue
        judge_ok, judge_reason = judge_accepted(winning[0].get("judge") or {})
        if not judge_ok:
            judge_failures.append(f"{cid}: {judge_reason}")
            continue
        proposal = winning[0].get("proposal") or {}
        generation = latest[cid]
        for field in ("verdict", "continuation", "confidence"):
            if proposal.get(field) != generation.get(field):
                judge_failures.append(f"{cid}: audit/generation mismatch in {field}")
                break
    if judge_failures:
        raise AssertionError(
            f"independent-judge provenance failed for {len(judge_failures)} candidates: "
            f"{judge_failures[:3]}"
        )

    parent_rows = read_jsonl(args.parent)
    if len(parent_rows) != 32_644:
        raise AssertionError(f"expected 32,644 parent rows, got {len(parent_rows)}")
    parent_outputs = [row["output"] for row in parent_rows]
    parent_fields = [
        (row["instruction"], row["input"], row.get("history") or []) for row in parent_rows
    ]
    parent_answers = [
        row["output"].split("</think>", 1)[1] if "</think>" in row["output"] else ""
        for row in parent_rows
    ]
    parent_counts = Counter(task_of(row) for row in parent_rows)
    parent_mix = target_token_mix(parent_rows)
    o2_teacher_rows = read_jsonl(args.o2_teacher)
    o2_teacher_identities = {canonical_training_row(row) for row in o2_teacher_rows}
    if len(o2_teacher_rows) != 164 or len(o2_teacher_identities) != 164:
        raise AssertionError("registered O2 teacher source must contain 164 unique rows")
    o2_teacher_indices = [
        index
        for index, row in enumerate(parent_rows)
        if canonical_training_row(row) in o2_teacher_identities
    ]
    if len(o2_teacher_indices) != 164:
        raise AssertionError(
            f"parent must contain each registered O2 teacher once, matched={len(o2_teacher_indices)}"
        )

    seen = Counter()
    changed = 0
    verdict_counts = Counter()
    continuation_lengths = []
    output_rows = []
    for row in parent_rows:
        new_row = dict(row)
        if recommendation_row(row):
            match = THINK_RE.search(row["output"])
            if match and match.group(1).strip():
                prefix = match.group(1).strip()
                cid = candidate_id(prompt_core(row), prefix)
                if cid in request_by_id:
                    generation = latest[cid]
                    seen[cid] += 1
                    verdict = generation["verdict"]
                    verdict_counts[verdict] += 1
                    if verdict == "TRUNCATED":
                        continuation = generation["continuation"]
                        errors = validate_proposal(request_by_id[cid], generation)
                        if errors:
                            raise AssertionError(f"accepted generation failed revalidation {cid}: {errors}")
                        inner = match.group(1)
                        leading = inner[: len(inner) - len(inner.lstrip())]
                        trailing = inner[len(inner.rstrip()) :]
                        repaired = leading + prefix + continuation + trailing
                        new_row["output"] = (
                            row["output"][: match.start(1)]
                            + repaired
                            + row["output"][match.end(1) :]
                        )
                        continuation_lengths.append(len(continuation))
                        changed += 1
        output_rows.append(new_row)

    if set(seen) != set(request_by_id):
        missing_parent = set(request_by_id) - set(seen)
        raise AssertionError(f"parent lacks {len(missing_parent)} prepared candidate groups")
    if any(count != 1 for count in seen.values()):
        raise AssertionError(f"parent should retain exactly one CoT per group: {seen.most_common(3)}")

    non_output_diffs = 0
    answer_diffs = 0
    unexpected_output_diffs = 0
    for index, row in enumerate(output_rows):
        if (row["instruction"], row["input"], row.get("history") or []) != parent_fields[index]:
            non_output_diffs += 1
        answer = row["output"].split("</think>", 1)[1] if "</think>" in row["output"] else ""
        answer_diffs += answer != parent_answers[index]
        if row["output"] != parent_outputs[index] and not (
            recommendation_row(row) and THINK_RE.search(row["output"])
        ):
            unexpected_output_diffs += 1
    if non_output_diffs or answer_diffs or unexpected_output_diffs:
        raise AssertionError(
            f"invariant failure fields={non_output_diffs} answers={answer_diffs} "
            f"unexpected_outputs={unexpected_output_diffs}"
        )
    output_diff_count = sum(
        row["output"] != parent_outputs[index] for index, row in enumerate(output_rows)
    )
    o2_teacher_output_diffs = sum(
        output_rows[index]["output"] != parent_outputs[index]
        for index in o2_teacher_indices
    )
    if output_diff_count != changed or o2_teacher_output_diffs:
        raise AssertionError(
            f"output-diff invariant failed diffs={output_diff_count} changed={changed} "
            f"O2_teacher_diffs={o2_teacher_output_diffs}"
        )

    final_counts = Counter(task_of(row) for row in output_rows)
    if final_counts != parent_counts:
        raise AssertionError("task counts changed")
    write_jsonl(args.out, output_rows)
    final_mix = target_token_mix(output_rows)
    length_audit = token_length_audit(output_rows)
    if length_audit["rows_over_cutoff_16384_with_8_token_margin"]:
        raise AssertionError(f"new dataset exceeds cutoff: {length_audit}")

    lengths = sorted(continuation_lengths)
    audit = {
        "asset_class": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O3)",
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__)),
        "upstream": {
            "parent_I10_dataset": {
                "path": str(args.parent.resolve()),
                "rows": len(parent_rows),
                "sha256": sha256(args.parent),
                "asset_lineage": "D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)",
            },
            "O2_teacher_identity_source": {
                "path": str(args.o2_teacher.resolve()),
                "rows": len(o2_teacher_rows),
                "sha256": sha256(args.o2_teacher),
                "matched_parent_rows": len(o2_teacher_indices),
            },
            "requests_D_O1_O3": {
                "path": str(args.requests.resolve()),
                "rows": len(requests_rows),
                "sha256": sha256(args.requests),
                "target_answer_or_metadata_rows": 0,
            },
            "teacher_generations": {
                "path": str(args.generations.resolve()),
                "sha256": sha256(args.generations),
                "generator_model": next(iter(latest.values()))["generator_model"],
                "judge_model": next(iter(latest.values()))["judge_model"],
                "accepted_latest": len(requests_rows),
            },
            "independent_judge_audit": {
                "path": str(args.generation_audit.resolve()),
                "sha256": sha256(args.generation_audit),
                "accepted_latest_with_score5_all_checks_true": len(requests_rows),
            },
            "O3_role": "history-side evidence only; target SID metadata excluded",
        },
        "rows": len(output_rows),
        "row_mix": {
            "O1_parent_rows": {"rows": 32_480, "ratio": round(32_480 / 32_644, 8)},
            "O2_teacher_unique_once": {"rows": 164, "ratio": round(164 / 32_644, 8)},
            "T": {"rows": 0, "ratio": 0.0},
            "E": {"rows": 0, "ratio": 0.0},
        },
        "single_variable_change": {
            "candidate_groups_resolved": len(requests_rows),
            "verdicts": dict(sorted(verdict_counts.items())),
            "retained_CoT_rows_changed": changed,
            "all_other_rows_unchanged": len(output_rows) - changed,
            "continuation_chars": {
                "min": lengths[0] if lengths else 0,
                "median": statistics.median(lengths) if lengths else 0,
                "p90": lengths[round((len(lengths) - 1) * 0.9)] if lengths else 0,
                "max": lengths[-1] if lengths else 0,
            },
        },
        "invariants": {
            "row_count_preserved": len(output_rows),
            "row_order_preserved": True,
            "instruction_input_history_diffs": non_output_diffs,
            "answer_diffs": answer_diffs,
            "output_diff_count": output_diff_count,
            "task_count_diffs": 0,
            "O2_teacher_rows_changed": o2_teacher_output_diffs,
            "failed_checkpoint_inputs": 0,
        },
        "task_counts": dict(sorted(final_counts.items())),
        "target_token_mix_before": parent_mix,
        "target_token_mix_after": final_mix,
        "token_length_audit": length_audit,
        "output": str(args.out.resolve()),
        "output_sha256": sha256(args.out),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    subparsers = top.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--o1", type=Path, default=DEFAULT_O1)
    prepare_parser.add_argument("--o3", type=Path, default=DEFAULT_O3)
    prepare_parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    prepare_parser.add_argument(
        "--prepare-summary", type=Path, default=DEFAULT_PREPARE_SUMMARY
    )
    prepare_parser.set_defaults(function=prepare)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    generate_parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    generate_parser.add_argument(
        "--generation-audit", type=Path, default=DEFAULT_GENERATION_AUDIT
    )
    generate_parser.add_argument(
        "--generation-summary", type=Path, default=DEFAULT_GENERATION_SUMMARY
    )
    generate_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    generate_parser.add_argument("--generator-model", default="gpt-5.6-sol-max")
    generate_parser.add_argument("--judge-model", default="gpt-5.6-terra-max")
    generate_parser.add_argument("--workers", type=int, default=4)
    generate_parser.add_argument("--start", type=int, default=0)
    generate_parser.add_argument("--n", type=int, default=0)
    generate_parser.add_argument("--repair-attempts", type=int, default=1)
    generate_parser.add_argument("--timeout", type=int, default=300)
    generate_parser.add_argument("--retries", type=int, default=4)
    generate_parser.add_argument("--retry-rejected", action="store_true")
    generate_parser.add_argument("--retry-invalid-accepted", action="store_true")
    generate_parser.set_defaults(function=generate)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    build_parser.add_argument("--o2-teacher", type=Path, default=DEFAULT_O2_TEACHER)
    build_parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    build_parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    build_parser.add_argument(
        "--generation-audit", type=Path, default=DEFAULT_GENERATION_AUDIT
    )
    build_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--audit", type=Path, default=DEFAULT_BUILD_AUDIT)
    build_parser.set_defaults(function=build)
    return top


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("workers must be >=1")
    if getattr(args, "repair_attempts", 0) < 0:
        raise ValueError("repair-attempts must be >=0")
    args.function(args)


if __name__ == "__main__":
    main()
