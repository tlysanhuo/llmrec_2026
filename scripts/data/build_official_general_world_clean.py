#!/usr/bin/env python3
"""Build reviewable world-knowledge candidates from registered official General data.

This builder deliberately stops before creating a training dataset.  It scans O2
OneReason_General for strict Chinese multiple-choice and short factual-QA
candidates, and scans O5 General-SFT only for strict Chinese multiple-choice
candidates.  Every surviving row retains source lineage and starts with
``review.status=pending``.  A separate human review is required before any row
can be projected into trainer format.

The legacy 231-row world file is never modified or used as a source.  Evaluation
and current-parent prompts are used only as text blacklists; their labels and
model outputs are never read for candidate selection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
PERSONAL_ROOT = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51")
RUNTIME_ROOT = PERSONAL_ROOT / "ai_runtime"

O2_DIR = ROOT / "assets/official/hf_raw/OneReason_General"
O5_DIR = ROOT / "assets/official/general_sft"
TOKENIZER_DIR = ROOT / "assets/official/base_model"

OUT_DIR = ROOT / "assets/derived/official_general"
DEFAULT_MC_OUT = OUT_DIR / "world_mc_strict_candidates.jsonl"
DEFAULT_QA_OUT = OUT_DIR / "general_zh_short_candidates.jsonl"
DEFAULT_REJECT_OUT = OUT_DIR / "world_clean_near_rejections.jsonl"
DEFAULT_AUDIT_OUT = ROOT / "logs/data/official_general_world_clean_audit.json"

SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
WORLD_HEAD = "请回答以下问题：\n\n"
WORLD_TAIL = "\n\n请按以下格式作答：\"正确答案是 (在此处填写选项字母)\"/no_think"
RULESET_VERSION = "official-general-world-clean-20260718-v2"

ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")
CHAT_MARKER = re.compile(r"<\|im_(?:start|end)\|>[^\n]*")
MODE_SUFFIX = re.compile(r"\s*/(?:no_)?think\s*$", re.IGNORECASE)
WORLD_PREFIX = re.compile(
    r"^\s*请回答以下问题(?:（.*?）|\(.*?\))?\s*[:：]?\s*", re.DOTALL
)
WORLD_SUFFIX = re.compile(r"\s*请按以下格式作答\s*[:：].*$", re.DOTALL)

# Anchored, uppercase-only option labels.  Lowercase a-h is intentionally not
# accepted because it frequently occurs as normal English prose.
OPTION_LINE = re.compile(
    r"(?m)^[ \t]*[（(\[]?([A-H])[）)\]]?[ \t]*[.．、:：][ \t]*(\S.*?)?[ \t]*$"
)
# Deliberately broader than OPTION_LINE and used only to prevent malformed or
# lowercase choice questions from leaking into the short-QA route.  It is not
# an acceptance parser.
BROAD_OPTION_MARKER = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[（(\[]([A-H])[）)\]]|([A-H])[ \t]*[)）.．、:：])[ \t]*(?=\S)"
)
MULTISELECT = re.compile(
    r"多项选择|多选题|多选|不定项|可多选|(?:选择|选出|勾选)所有|所有适用|"
    r"全部正确|有几项|答案不唯一|可(?:以)?选择多个|正确(?:的)?有哪几项|"
    r"哪些|哪几(?:个|项)?|"
    r"至少(?:有)?\s*(?:一|1|[二两三四五六七八九]|[2-9])\s*(?:个|项)?\s*(?:选项|答案)?|"
    r"(?:一个以上|超过一个|不只一个|不止一个|一个或多个)|"
    r"(?:请|须|需|可)?(?:选|选择|选出|勾选)\s*(?:[二两三四五六七八九]|[2-9])\s*(?:个|项)?|"
    r"有\s*(?:[二两三四五六七八九]|[2-9])\s*(?:个|项)?\s*(?:正确\s*)?(?:答案|选项)|"
    r"有\s*(?:[二两三四五六七八九]|[2-9])\s*(?:个|项)?\s*(?:答案|选项)?\s*(?:是)?\s*正确|"
    r"正确(?:的|答案|选项)?\s*有\s*(?:[二两三四五六七八九]|[2-9])\s*(?:个|项)?|"
    r"(?:[二两三四五六七八九]|[2-9])\s*(?:个|项)?\s*(?:正确\s*)?(?:答案|选项)(?:\s*是)?\s*正确|"
    r"(?:多个|多项|若干项?).{0,8}(?:正确|答案|选项)|"
    r"(?:正确|答案|选项).{0,8}(?:多个|多项|若干项?)|"
    r"all\s+that\s+apply|(?:select|choose)\s+all|"
    r"select\s+(?:two|three|four|[2-9])|more\s+than\s+one\s+(?:answer|option)|"
    r"one\s+or\s+more\s+(?:answers?|options?)|multiple\s+(?:answers?|options?)",
    re.IGNORECASE,
)
SINGLE_BOILERPLATE = [
    re.compile(r"^\s*以下是一道单项选择题\s*[:：。]?\s*"),
    re.compile(r"^\s*这是一道单项选择题\s*[:：。]?\s*"),
    re.compile(r"^\s*请给出(?:这道)?单选题的答案(?:和|以及)?(?:解析|推理过程)?\s*[。:：]?\s*"),
    re.compile(r"^\s*题目\s*[:：]\s*"),
]
MC_META_RESIDUE = re.compile(
    r"(?:^|[,，。;；:：\s])(?:请)?(?:给出|提供)(?:这道)?(?:单选题(?:的)?)?答案|"
    r"解析过程|推理过程|^\s*单选(?:题)?\s*[:：]"
)
OPTION_META_RESIDUE = re.compile(
    r"(?:[。！!?，,；;:：]|\s)(?:请\s*(?:给出|选择|回答|求出|按|一步步)|"
    r"给出(?:答案|解析|理由|推理)|选择正确答案|一步步思考|解决这个问题)"
)
PROMPT_GOLD_LEAK = re.compile(
    r"(?:"
    r"(?:正确|参考|标准)(?:答案|选项)\s*(?:是|为|应为|选|[:：=])?\s*[（(\[]?\s*[A-D]\s*[）)\]]?|"
    r"答案\s*(?:是|为|应为|选|[:：=])?\s*[（(\[]?\s*[A-D]\s*[）)\]]?|"
    r"选项\s*(?:是|为|[:：=])\s*[（(\[]?\s*[A-D]\s*[）)\]]?|"
    r"本题\s*(?:应)?选\s*[（(\[]?\s*[A-D]\s*[）)\]]?|"
    r"(?:正确的\s*是|故\s*选)\s*[（(\[]?\s*[A-D]\s*[）)\]]?|"
    r"[（(\[]?\s*[A-D]\s*[）)\]]?\s*(?:是|为)\s*(?:正确|参考|标准)?(?:答案|选项)|"
    r"(?:correct\s+)?(?:answer|ans|key)\s*(?:is|[:：=])\s*[（(\[]?\s*[A-D]\s*[）)\]]?|"
    r"[（(\[]?\s*[A-D]\s*[）)\]]?\s+is\s+(?:the\s+)?correct\s+(?:answer|option)|"
    r"解析\s*[:：]"
    r")",
    re.IGNORECASE,
)
OPTION_GOLD_MARKER = re.compile(
    r"^\s*\*|[（(\[【]\s*(?:正确|错误|答案)[^）)\]】]*[）)\]】]|"
    r"(?:^|[\s（(\[【])[✓✔✗✘](?=$|[\s）)\]】。；;，,])|"
    r"[（(\[【]\s*[√×]\s*[）)\]】]|\S\s+[√×]\s*$"
)
OPTION_META_BOUNDARY = frozenset("。！!?，,；;:：（(）)[]【】 \t")
ALLOWED_MC_POSTSCRIPT = re.compile(
    r"^(?:请)?(?:选择|给出|回答|求出)(?:正确|合适|最佳)?(?:的)?(?:答案|选项|问题)"
    r"(?:并(?:给出|提供)(?:解释|理由|解析))?[。.!！?？]?$|^解决这个问题[。.!！]?$"
)

THINK_TAG = re.compile(r"</?(?:think|analysis)>", re.IGNORECASE)
THINK_SPAN = re.compile(r"<(think|analysis)>.*?</\1>", re.IGNORECASE | re.DOTALL)
SPECIAL_THINK_OPEN = "<|begin_of_thought|>"
SPECIAL_THINK_CLOSE = "<|end_of_thought|>"
SPECIAL_SOLUTION_OPEN = "<|begin_of_solution|>"
SPECIAL_SOLUTION_CLOSE = "<|end_of_solution|>"
UNKNOWN_REASON_MARKER = re.compile(r"<\|[^>]*(?:thought|solution|analysis)[^>]*\|>", re.IGNORECASE)
TOOL_OR_MEDIA = re.compile(
    r"<\|(?:tool|image|video)|</?(?:tool_call|tool_response|image|video)>|"
    r"data:image/|https?://\S+\.(?:png|jpe?g|gif|webp)",
    re.IGNORECASE,
)

# Strong answer assertions are Chinese-anchored and uppercase-only.  Avoiding
# re.IGNORECASE on [A-H] prevents ordinary English words from becoming labels.
ANSWER_ASSERTION = re.compile(
    r"(?:最终|正确|参考)?答案(?:应该|应当)?(?:是|为|选|应为)?\s*[:：]?\s*"
    r"[（(\[]?\s*([A-Z](?:\s*[,，、/和及]?\s*[A-Z])*)\s*[）)\]]?"
)
CHOOSE_ASSERTION = re.compile(
    r"(?:故|因此|所以)?\s*(?:应|故)?选\s*[（(\[]?\s*"
    r"([A-Z](?:\s*[,，、/和及]?\s*[A-Z])*)\s*[）)\]]?"
)
BARE_ANSWER = re.compile(r"^[ \t]*[（(\[]?([A-D])[）)\]]?[。.]?[ \t]*$")
FINAL_ANSWER_ASSERTION = re.compile(
    r"^\s*(?:(?:综上(?:所述)?|因此|所以|故)[,，。:：]?\s*)*"
    r"(?:(?:最终|正确|参考)?答案(?:应该|应当)?(?:是|为|选|应为)?|实际应为)"
    r"\s*[:：]?\s*[（(\[]?\s*"
    r"([A-Z](?:\s*[,，、/和及]?\s*[A-Z])*)\s*[）)\]]?\s*[。.!！]?\s*$"
)
FINAL_CHOOSE_ASSERTION = re.compile(
    r"^\s*(?:(?:综上(?:所述)?|因此|所以|故)[,，。:：]?\s*)*"
    r"(?:应|故)?选\s*[（(\[]?\s*"
    r"([A-Z](?:\s*[,，、/和及]?\s*[A-Z])*)\s*[）)\]]?\s*[。.!！]?\s*$"
)
NEGATED_FINAL_ANSWER = re.compile(
    r"错误答案|不正确.{0,4}答案|排除.{0,6}答案|不应(?:该)?选|不能选|不可选|"
    r"答案不(?:是|为)|(?:不能|不要|不(?:应当|应该|应|该)?|切勿|无需)\s*选"
)
UNCERTAIN_ANSWER = re.compile(
    r"猜测.{0,4}答案|不确定.{0,4}答案|假设.{0,4}答案|"
    r"(?:可能|也许|或许|猜|暂且|大概)\s*选"
)

IDENTITY = re.compile(
    r"阶跃星辰|StepFun|我是\s*\**Step\b|通义千问|文心一言|讯飞星火|"
    r"Kimi|Claude|ChatGPT|Llama|Copilot|我的开发(?:者|商)|模型身份",
    re.IGNORECASE,
)
PROMPT_INJECTION = re.compile(
    r"忽略(?:之前|以上|前面)|系统提示|system\s+prompt|ignore\s+(?:all\s+)?previous|"
    r"不要回答问题|只(?:能|需)输出|必须包含(?:关键词|字样)|禁止包含(?:关键词|字样)|"
    r"输出格式为\s*(?:JSON|XML)|你的回答必须|Do\s+not\s+include\s+keywords",
    re.IGNORECASE,
)
TIME_SENSITIVE = re.compile(
    r"(?:截至|截止|当前|目前|当今|迄今|现任|最新|今天|昨天|明天|今年|明年|"
    r"本月|本周|实时|现在是星期几|今日价格|"
    r"(?:总统|首相|总理|CEO|首席执行官)是谁|人口最多|首富|"
    r"(?:市值|票房|销量)最高|最新排名|世界纪录)"
)
HIGH_RISK = re.compile(
    r"患者|病人|诊断|治疗|用药|剂量|处方|手术|临床|症状|癌症|肿瘤|"
    r"心梗|脑梗|妊娠|孕妇|药物|某某丸|某某汤|执业医师|护理|"
    r"依照.*法|现行.*法|法律咨询|刑法|民法|行政法|诉讼|法院|检察院|"
    r"犯罪|构成.*罪|承担赔偿|著作权法|土地承包法|公司法|证券法|知识产权|"
    r"宪法|宪政|上市(?:交易|监管)|选民资格|"
    r"疾病|传染病|丙肝|白痢|螨病|兽医|疫苗|病毒感染|"
    r"总书记|国家主席|社会主义|苏维埃|中国共产党|政治文明|精准扶贫|一带一路|"
    r"两岸|台湾"
)
SUBJECTIVE = re.compile(
    r"你认为|您认为|你觉得|您觉得|最适合|最喜欢|感受|评价.*味道|哪种音乐|"
    r"更好看|更幸福|最合理(?:的布局)?|最好|最佳|最能|应该优先|应优先|"
    r"优先(?:开发|选择)|提升.*满意度"
)

QA_REFERENCE = re.compile(
    r"根据(?:以下|上述|上文|材料|文本|情景|前提|图片)|请(?:仔细)?阅读|"
    r"这张图片|附件|文段|选项\s*[:：]|文本\s*\d|前提\s*[:：]|假设\s*[:：]|"
    r"回答上面的问题|以下两个句子|给定一个数组|文本中|对话中|本段文本|"
    r"基于对话|根据提供的(?:文本|内容|信息)|前提.*假设"
)
QA_WRONG_TASK = re.compile(
    r"写|撰写|生成|创作|润色|改写|翻译|总结|提取|分析|制定|设计|建议|"
    r"代码|编程|数组|JSON|XML|文章|故事|报告|计划|方案|食谱|列出|"
    r"扮演|角色|字数|不少于|至少.*句|证明|计算|求解|方程|概率|平均|"
    r"几何|三角形|最小公倍数|完全平方|多少(?:岁|分钟|公里|美元|本|个|条|人|天)|"
    r"几小时|百分比|命中率|速度|年龄|预算|置信区间|逻辑推理|是否蕴含|"
    r"描述|抽取|程序|字符串|输出结果|如何(?:创建|设置|使用|操作)|"
    r"不超过.*句|提示用户|询问用户|短句|前提.*假设"
)
QA_FACTUAL = re.compile(
    r"是什么|指什么|是谁|由谁|哪个|哪一|哪些|何时|为何|为什么|"
    r"有什么作用|作用之一|用途|原理|定义|区别|包括什么|包括哪些|"
    r"由什么.*(?:制成|加工|组成)|需要经过哪|分为哪|如何形成|如何产生"
)
QA_BAD_ANSWER = re.compile(
    r"无法(?:确定|直接|回答)|没有提供|未提供|请提供(?:文本|对话|内容)|"
    r"信息.{0,12}(?:有限|不足)|没有足够信息|无法从.{0,16}(?:得出|判断|确定)|"
    r"不知道|不清楚|不确定|无法核实|可能是|大概是|也许是|抱歉"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(stable_json(row) + "\n")
    temp.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def normalize_raw(value: str) -> str:
    value = unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = ZERO_WIDTH.sub("", value)
    return value.rstrip()


def mode_normalize(value: str) -> str:
    value = normalize_raw(value)
    value = CHAT_MARKER.sub(" ", value)
    value = MODE_SUFFIX.sub("", value)
    return " ".join(value.casefold().split())


def semantic_normalize(value: str) -> str:
    value = normalize_raw(value).casefold()
    return "".join(
        char for char in value
        if not (char.isspace() or unicodedata.category(char)[0] in "PZC")
    )


def hash_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8", "surrogatepass"))


def language_stats(value: str) -> dict[str, Any]:
    counts = Counter()
    for char in value:
        code = ord(char)
        if 0x3400 <= code <= 0x9FFF or 0x20000 <= code <= 0x3134F:
            counts["han"] += 1
        elif 0x3040 <= code <= 0x30FF:
            counts["kana"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            counts["hangul"] += 1
        elif char.isascii() and char.isalpha():
            counts["latin"] += 1
    han = counts["han"]
    latin = counts["latin"]
    return {
        "han": han,
        "latin": latin,
        "kana": counts["kana"],
        "hangul": counts["hangul"],
        "han_latin_ratio": han / max(han + latin, 1),
    }


def is_strict_zh(value: str, *, min_han: int) -> tuple[bool, dict[str, Any]]:
    stats = language_stats(value)
    passed = (
        stats["han"] >= min_han
        and stats["han_latin_ratio"] >= 0.60
        and stats["kana"] == 0
        and stats["hangul"] == 0
    )
    return passed, stats


def extract_plain_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                return None
            if item.get("type") not in (None, "text"):
                return None
            text = item.get("text", item.get("content"))
            if not isinstance(text, str):
                return None
            parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict) and content.get("type") in (None, "text"):
        text = content.get("text", content.get("content"))
        return text if isinstance(text, str) else None
    return None


def parse_messages(raw: Any) -> tuple[list[str], str, str] | None:
    try:
        messages = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        return None
    roles = [str(item.get("role", "")) for item in messages]
    if roles != ["user", "assistant"]:
        return None
    user = extract_plain_text(messages[0].get("content"))
    assistant = extract_plain_text(messages[1].get("content"))
    if user is None or assistant is None:
        return None
    user, assistant = user.strip(), assistant.strip()
    if not user or not assistant:
        return None
    return roles, user, assistant


def split_reasoning(assistant: str) -> tuple[str | None, str]:
    tags = [match.group(0).casefold() for match in THINK_TAG.finditer(assistant)]
    opens = sum(not tag.startswith("</") for tag in tags)
    closes = sum(tag.startswith("</") for tag in tags)
    if opens != closes:
        return None, "malformed_think"
    special_open = assistant.count(SPECIAL_THINK_OPEN)
    special_close = assistant.count(SPECIAL_THINK_CLOSE)
    if special_open != special_close:
        return None, "malformed_think"
    stripped = THINK_SPAN.sub("\n", assistant)
    while SPECIAL_THINK_OPEN in stripped and SPECIAL_THINK_CLOSE in stripped:
        start = stripped.find(SPECIAL_THINK_OPEN)
        end = stripped.find(SPECIAL_THINK_CLOSE, start + len(SPECIAL_THINK_OPEN))
        if end < 0:
            return None, "malformed_think"
        stripped = stripped[:start] + "\n" + stripped[end + len(SPECIAL_THINK_CLOSE) :]
    if THINK_TAG.search(stripped) or SPECIAL_THINK_OPEN in stripped or SPECIAL_THINK_CLOSE in stripped:
        return None, "malformed_think"

    solution_open = stripped.count(SPECIAL_SOLUTION_OPEN)
    solution_close = stripped.count(SPECIAL_SOLUTION_CLOSE)
    if solution_open > 1 or solution_close > 1 or solution_close > solution_open:
        return None, "malformed_solution"
    if solution_open:
        before, final = stripped.split(SPECIAL_SOLUTION_OPEN, 1)
        if before.strip():
            return None, "malformed_solution"
        if solution_close:
            final, after = final.split(SPECIAL_SOLUTION_CLOSE, 1)
            if after.strip():
                return None, "malformed_solution"
    else:
        final = stripped
    if UNKNOWN_REASON_MARKER.search(final):
        return None, "malformed_solution"
    final = final.strip()
    if not final:
        return None, "no_final"
    if special_open:
        return final, "closed_special_think"
    if tags:
        return final, "closed_think"
    return final, "no_think"


def strip_world_wrapper(prompt: str) -> str:
    core = normalize_raw(prompt)
    core = MODE_SUFFIX.sub("", core)
    core = WORLD_PREFIX.sub("", core)
    core = WORLD_SUFFIX.sub("", core)
    return core.strip()


def broad_option_labels(prompt: str) -> set[str]:
    """Return broad A-H-like labels for routing only, never acceptance."""
    core = strip_world_wrapper(prompt)
    return {
        (match.group(1) or match.group(2)).upper()
        for match in BROAD_OPTION_MARKER.finditer(core)
    }


def strip_single_choice_boilerplate(question: str) -> str:
    question = question.strip()
    while True:
        before = question
        for pattern in SINGLE_BOILERPLATE:
            question = pattern.sub("", question).strip()
        if question == before:
            return question


def find_broad_option_sequences(core: str) -> list[list[re.Match[str]]]:
    """Find every ordered A-B-C[-D] run, restarting on each A marker."""
    sequence: list[re.Match[str]] = []
    completed: list[list[re.Match[str]]] = []
    expected = "A"
    for match in BROAD_OPTION_MARKER.finditer(core):
        label = (match.group(1) or match.group(2)).upper()
        if label == "A":
            sequence = [match]
            expected = "B"
            continue
        if not sequence or label != expected:
            continue
        sequence.append(match)
        if label == "C":
            completed.append(list(sequence))
        if label == "D":
            if completed and completed[-1][0].start() == sequence[0].start():
                completed[-1] = list(sequence)
            expected = "E"
            continue
        expected = chr(ord(label) + 1)
    return completed


def find_broad_abcd_sequence(core: str) -> list[re.Match[str]]:
    """Compatibility helper returning the last broad option run."""
    sequences = find_broad_option_sequences(core)
    return sequences[-1] if sequences else []


def extract_broad_mc_stems(prompt: str) -> list[str]:
    """Extract every plausible MC stem for fail-closed leakage indexing."""
    core = strip_world_wrapper(prompt)
    stems: list[str] = []
    seen: set[str] = set()
    for sequence in find_broad_option_sequences(core):
        stem = strip_single_choice_boilerplate(core[: sequence[0].start()])
        normalized = semantic_normalize(stem)
        if stem and normalized and normalized not in seen:
            stems.append(stem)
            seen.add(normalized)
    return stems


def extract_broad_mc_stem(prompt: str) -> str | None:
    """Extract an MC stem for leakage indexing, never candidate acceptance."""
    stems = extract_broad_mc_stems(prompt)
    return stems[-1] if stems else None


def option_has_multiselect_tail(value: str) -> bool:
    for match in MULTISELECT.finditer(value):
        if match.start() == 0 or value[match.start() - 1] in OPTION_META_BOUNDARY:
            return True
    return False


@dataclass
class ParsedMC:
    question: str
    options: dict[str, str]
    postscript: str


def parse_mc_prompt(prompt: str) -> tuple[ParsedMC | None, list[str]]:
    reasons: list[str] = []
    core = strip_world_wrapper(prompt)
    matches = list(OPTION_LINE.finditer(core))
    multi_scope = core
    if matches:
        multi_scope = core[: matches[0].start()] + "\n" + core[matches[-1].end() :]
    if MULTISELECT.search(multi_scope):
        reasons.append("mc_multiselect")
    labels = [match.group(1) for match in matches]
    if any(label in "EFGH" for label in labels):
        reasons.append("mc_extra_option")
    if labels != list("ABCD"):
        reasons.append("mc_labels")
        return None, reasons
    question = strip_single_choice_boilerplate(core[: matches[0].start()])
    if MC_META_RESIDUE.search(question):
        reasons.append("mc_meta_residue")
    if PROMPT_GOLD_LEAK.search(question):
        reasons.append("mc_prompt_gold_leak")
    options = {label: (match.group(2) or "").strip() for label, match in zip(labels, matches)}
    if any(OPTION_META_RESIDUE.search(value) for value in options.values()):
        reasons.append("mc_option_meta")
    if any(option_has_multiselect_tail(value) for value in options.values()):
        reasons.append("mc_option_meta")
    if any(PROMPT_GOLD_LEAK.search(value) for value in options.values()):
        reasons.append("mc_option_gold_leak")
    if any(OPTION_GOLD_MARKER.search(value) for value in options.values()):
        reasons.append("mc_option_gold_marker")
    if not question or any(not value for value in options.values()):
        reasons.append("mc_empty_part")
    option_norms = [semantic_normalize(options[label]) for label in "ABCD"]
    if any(not value for value in option_norms) or len(set(option_norms)) != 4:
        reasons.append("mc_duplicate_option")
    postscript = core[matches[-1].end() :].strip()
    if postscript and not ALLOWED_MC_POSTSCRIPT.fullmatch(postscript):
        reasons.append("mc_postscript")
    if reasons:
        return None, sorted(set(reasons))
    return ParsedMC(question=question, options=options, postscript=postscript), []


def answer_assertions(value: str) -> list[tuple[str, str, int, int]]:
    hits: list[tuple[int, str, str, int]] = []
    for pattern in (ANSWER_ASSERTION, CHOOSE_ASSERTION):
        for match in pattern.finditer(value):
            letters = "".join(re.findall(r"[A-Z]", match.group(1)))
            if letters:
                hits.append((match.start(), letters, match.group(0)[:160], match.end()))
    for offset, line in enumerate(value.splitlines()):
        match = BARE_ANSWER.fullmatch(line)
        if match:
            hits.append((len(value) + offset, match.group(1), line.strip(), len(value)))
    return [(letters, evidence, start, end) for start, letters, evidence, end in sorted(hits)]


def parse_mc_answer(assistant: str) -> tuple[str | None, str | None, str, list[str]]:
    final, think_status = split_reasoning(assistant)
    if final is None:
        return None, None, think_status, [think_status]
    outside = answer_assertions(final)
    if not outside:
        return None, None, think_status, ["answer_unparsed"]
    reasons: list[str] = []
    if any(len(letters) != 1 for letters, _evidence, _start, _end in outside):
        reasons.append("answer_not_single")
    if any(letters not in set("ABCD") for letters, _evidence, _start, _end in outside):
        reasons.append("answer_not_abcd")
    unique = {letters for letters, _evidence, _start, _end in outside}
    if len(unique) != 1:
        reasons.append("answer_conflict")
    last_line = next((line.strip() for line in reversed(final.splitlines()) if line.strip()), "")
    final_assertion = (
        FINAL_ANSWER_ASSERTION.fullmatch(last_line)
        or FINAL_CHOOSE_ASSERTION.fullmatch(last_line)
    )
    bare = BARE_ANSWER.fullmatch(last_line)
    final_letters = None
    if final_assertion is not None:
        final_letters = "".join(re.findall(r"[A-Z]", final_assertion.group(1)))
    elif bare is not None:
        final_letters = bare.group(1)
    if final_letters is None:
        reasons.append("answer_not_final")
    elif final_letters != outside[-1][0]:
        reasons.append("answer_conflict")
    all_strong = answer_assertions(assistant)
    if any(letters != outside[-1][0] for letters, _evidence, _start, _end in all_strong):
        reasons.append("answer_conflict")
    if NEGATED_FINAL_ANSWER.search(assistant):
        reasons.append("answer_negated")
    if UNCERTAIN_ANSWER.search(assistant):
        reasons.append("answer_uncertain")
    if reasons:
        return None, None, think_status, sorted(set(reasons))
    return outside[-1][0], outside[-1][1], think_status, []


def mc_semantic_keys(parsed: ParsedMC, answer: str) -> dict[str, str]:
    question = semantic_normalize(parsed.question)
    option_norm = {label: semantic_normalize(parsed.options[label]) for label in "ABCD"}
    ordered = question + "\0" + "\0".join(option_norm[label] for label in "ABCD")
    invariant = question + "\0" + "\0".join(sorted(option_norm.values()))
    answer_text = option_norm[answer]
    return {
        "stem_norm": question,
        "stem_hash": hash_text(question),
        "ordered_qa_hash": hash_text(ordered),
        "option_invariant_hash": hash_text(invariant),
        "semantic_key": hash_text(invariant + "\0" + answer_text),
        "answer_text_norm": answer_text,
    }


def char_ngrams(value: str, n: int = 5) -> set[str]:
    if len(value) < n:
        return {value} if value else set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


@dataclass
class LeakageIndex:
    raw_hashes: set[str] = field(default_factory=set)
    mode_hashes: set[str] = field(default_factory=set)
    semantic_hashes: set[str] = field(default_factory=set)
    stem_hashes: set[str] = field(default_factory=set)
    ordered_hashes: set[str] = field(default_factory=set)
    invariant_hashes: set[str] = field(default_factory=set)
    semantic_texts: list[str] = field(default_factory=list)
    gram_index: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    near_seen: set[str] = field(default_factory=set)
    source_counts: Counter = field(default_factory=Counter)

    def add(self, prompt: str, source: str, *, include_near: bool = True) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            return
        raw = normalize_raw(prompt)
        mode = mode_normalize(prompt)
        core = strip_world_wrapper(prompt)
        semantic = semantic_normalize(core)
        self.raw_hashes.add(hash_text(raw))
        self.mode_hashes.add(hash_text(mode))
        if semantic:
            self.semantic_hashes.add(hash_text(semantic))
            # Near-duplicate search is useful only for bounded world-like text.
            # Exact hashes still cover every prompt, including very long hidden
            # recommendation prompts from platform logs.  Deduplicating here
            # avoids rebuilding the same 5-gram sets across repeated eval logs.
            semantic_hash = hash_text(semantic)
            if include_near and 24 <= len(semantic) <= 4096 and semantic_hash not in self.near_seen:
                self.near_seen.add(semantic_hash)
                index = len(self.semantic_texts)
                self.semantic_texts.append(semantic)
                for gram in char_ngrams(semantic):
                    self.gram_index[gram].add(index)
        parsed, _ = parse_mc_prompt(prompt)
        broad_stems = extract_broad_mc_stems(prompt)
        if parsed is not None:
            broad_stems.append(parsed.question)
        for broad_stem in broad_stems:
            self.stem_hashes.add(hash_text(semantic_normalize(broad_stem)))
        if parsed is not None:
            keys = mc_semantic_keys(parsed, "A")
            self.ordered_hashes.add(keys["ordered_qa_hash"])
            self.invariant_hashes.add(keys["option_invariant_hash"])
        self.source_counts[source] += 1

    def match(self, prompt: str, parsed: ParsedMC | None = None) -> tuple[bool, list[str]]:
        matches: list[str] = []
        raw = normalize_raw(prompt)
        mode = mode_normalize(prompt)
        core = strip_world_wrapper(prompt)
        semantic = semantic_normalize(core)
        if hash_text(raw) in self.raw_hashes:
            matches.append("raw_exact")
        if hash_text(mode) in self.mode_hashes:
            matches.append("mode_exact")
        if semantic and hash_text(semantic) in self.semantic_hashes:
            matches.append("core_exact")
        # A benchmark MC may otherwise leak through the short-QA route after
        # its options are removed.
        if semantic and hash_text(semantic) in self.stem_hashes:
            matches.append("stem_text_exact")
        if parsed is not None:
            keys = mc_semantic_keys(parsed, "A")
            # And the inverse: a benchmark stored as pure QA must block a
            # candidate MC that adds A-D options around the same stem.
            if keys["stem_hash"] in self.semantic_hashes:
                matches.append("indexed_prompt_as_stem_exact")
            if keys["stem_hash"] in self.stem_hashes:
                matches.append("stem_exact")
            if keys["ordered_qa_hash"] in self.ordered_hashes:
                matches.append("ordered_exact")
            if keys["option_invariant_hash"] in self.invariant_hashes:
                matches.append("option_invariant_exact")
        if not matches and len(semantic) >= 24:
            grams = char_ngrams(semantic)
            possible: Counter[int] = Counter()
            for gram in grams:
                possible.update(self.gram_index.get(gram, ()))
            for index, intersection in possible.most_common(32):
                other = self.semantic_texts[index]
                length_ratio = min(len(semantic), len(other)) / max(len(semantic), len(other))
                if length_ratio < 0.80:
                    continue
                union = len(grams) + len(char_ngrams(other)) - intersection
                if union and intersection / union >= 0.90:
                    matches.append("near_duplicate")
                    break
                if semantic in other or other in semantic:
                    matches.append("containment")
                    break
        return bool(matches), sorted(set(matches))


def row_prompts(raw: Any) -> Iterator[str]:
    if isinstance(raw, list):
        for item in raw:
            yield from row_prompts(item)
        return
    if not isinstance(raw, dict):
        return
    for key in ("prompt", "input", "question", "user"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            yield value
    messages = raw.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = None
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                text = extract_plain_text(message.get("content"))
                if text:
                    yield text


def load_jsonl_prompts(path: Path, index: LeakageIndex, source: str) -> int:
    count = 0
    if not path.exists():
        raise FileNotFoundError(f"required blacklist is missing: {path}")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid blacklist JSON: {path}:{line_number}") from exc
            prompts = list(row_prompts(raw))
            for prompt in prompts:
                index.add(prompt, source)
                count += 1
    return count


def add_structured_mc(index: LeakageIndex, question: str, options: dict[str, str], source: str) -> None:
    if not question.strip() or any(not options.get(label, "").strip() for label in "ABCD"):
        return
    prompt = question.strip() + "\n" + "\n".join(f"{label}.{options[label]}" for label in "ABCD")
    index.add(prompt, source)


def load_eval_index(*, exclude_paths: Iterable[Path] = ()) -> LeakageIndex:
    """Load the central evaluation blacklist, optionally omitting owned outputs.

    ``exclude_paths`` exists for idempotent builders whose own permanent
    holdout is discovered by the central glob after the first successful run.
    Callers remain responsible for freezing and auditing that owned holdout;
    this option must not be used to suppress any unrelated evaluation asset.
    """
    index = LeakageIndex()
    excluded = {path.resolve() for path in exclude_paths}
    structured = [
        ROOT / "assets/evaluation/visible/懂世界.jsonl",
        ROOT / "assets/evaluation/offline_eval/dev_world.jsonl",
        ROOT / "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl",
    ]
    structured.extend(sorted((ROOT / "assets/evaluation/holdout").glob("*.jsonl")))
    structured = [path for path in structured if path.resolve() not in excluded]
    for path in structured:
        if not path.exists():
            raise FileNotFoundError(f"required evaluation blacklist is missing: {path}")
        load_jsonl_prompts(path, index, str(path.relative_to(ROOT)))

    # Fixed precheck MC prompts are evaluation fixtures even though they are not
    # online questions.  AST parsing avoids importing a GPU-oriented module.
    precheck = ROOT / "scripts/eval/precheck.py"
    tree = ast.parse(precheck.read_text(encoding="utf-8"))
    precheck_found = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MC_QUESTIONS" for target in node.targets
        ):
            precheck_found += 1
            for question, _ignored_gold in ast.literal_eval(node.value):
                index.add(question, "scripts/eval/precheck.py:MC_QUESTIONS")
    if precheck_found != 1:
        raise AssertionError(
            f"expected exactly one MC_QUESTIONS assignment in scripts/eval/precheck.py; got {precheck_found}"
        )

    # CEval val and CMMLU dev/test are local selection/evaluation families.  We
    # read only question/options, never their answer columns.
    import pyarrow.parquet as pq

    ceval_dir = ROOT / "assets/evaluation/offline_eval/_ceval_val"
    ceval_paths = sorted(ceval_dir.glob("*.parquet"))
    if len(ceval_paths) != 52:
        raise AssertionError(f"CEval val file signature drifted: {len(ceval_paths)} != 52")
    for path in ceval_paths:
        table = pq.read_table(path, columns=["question", "A", "B", "C", "D"])
        for row in table.to_pylist():
            add_structured_mc(index, str(row["question"]), {label: str(row[label]) for label in "ABCD"}, "ceval_val")

    cmmlu_zip = ROOT / "assets/evaluation/offline_eval/_cmmlu.zip"
    if not cmmlu_zip.exists():
        raise FileNotFoundError(f"required CMMLU blacklist is missing: {cmmlu_zip}")
    cmmlu_files = 0
    with zipfile.ZipFile(cmmlu_zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".csv") or not (
                name.startswith(("dev/", "test/")) or "/dev/" in name or "/test/" in name
            ):
                continue
            cmmlu_files += 1
            with archive.open(name) as binary:
                rows = csv.DictReader(line.decode("utf-8-sig") for line in binary)
                for row in rows:
                    question = row.get("Question") or row.get("question") or ""
                    options = {label: row.get(label, "") for label in "ABCD"}
                    add_structured_mc(index, question, options, "cmmlu_dev_test")
    if cmmlu_files != 134:
        raise AssertionError(f"CMMLU dev/test file signature drifted: {cmmlu_files} != 134")

    # Platform logs contain hidden E prompts.  Store hashes/signatures only.
    eval_logs = sorted((ROOT / "logs/eval").glob("*.log"))
    if not eval_logs:
        raise AssertionError("no platform eval logs found for prompt blacklist")
    for path in eval_logs:
        with path.open(encoding="utf-8", errors="replace") as handle:
            collecting = False
            block: list[str] = []
            for line in handle:
                marker = line.strip()
                if marker == "<|im_start|>user":
                    if block:
                        index.add("".join(block), "logs/eval")
                    collecting, block = True, []
                    continue
                if collecting and marker.startswith("<|im_"):
                    if block:
                        index.add("".join(block), "logs/eval")
                    collecting, block = False, []
                    continue
                if collecting:
                    block.append(line)
            if block:
                index.add("".join(block), "logs/eval")
    expected_exact = {
        "assets/evaluation/visible/懂世界.jsonl": 7,
        "assets/evaluation/offline_eval/dev_world.jsonl": 500,
        "assets/derived/processed/data_i22_world_retkl_v1_holdout.jsonl": 46,
        "assets/evaluation/holdout/data_i28_video_multigold_proposal_v1_gate.jsonl": 128,
        "assets/evaluation/holdout/data_o1_reward_preference_v1_holdout.jsonl": 1784,
        "scripts/eval/precheck.py:MC_QUESTIONS": 8,
        "ceval_val": 1346,
        "cmmlu_dev_test": 11917,
        "logs/eval": 1580,
    }
    drift = {
        source: (index.source_counts[source], expected)
        for source, expected in expected_exact.items()
        if index.source_counts[source] != expected
    }
    if drift:
        raise AssertionError(f"evaluation blacklist row signature drifted: {drift}")
    return index


def load_train_index() -> tuple[LeakageIndex, Counter]:
    index = LeakageIndex()
    counts = Counter()
    paths = [
        ROOT / "assets/derived/processed/data_seed_teacher_v1.jsonl",
        ROOT / "assets/derived/processed/data_user_residual_retention_v1.jsonl",
    ]
    expected = {
        "data_seed_teacher_v1.jsonl": 32644,
        "data_user_residual_retention_v1.jsonl": 6106,
    }
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"required parent-training blacklist is missing: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except Exception as exc:
                    raise ValueError(f"invalid parent JSON: {path}:{line_number}") from exc
                prompts = list(row_prompts(raw))
                if not prompts:
                    raise ValueError(f"parent row has no user prompt: {path}:{line_number}")
                for prompt in prompts:
                    index.add(prompt, path.name, include_near=False)
                    counts[path.name] += 1
        if counts[path.name] != expected[path.name]:
            raise AssertionError(
                f"parent prompt signature drifted: {path.name}={counts[path.name]} "
                f"expected={expected[path.name]}"
            )
    return index, counts


def source_locator(
    asset_id: str,
    revision: str,
    source_root: Path,
    path: Path,
    row_group: int,
    row_index: int,
    source: str,
    uuid: Any,
    raw_messages: Any,
) -> tuple[str, dict[str, Any]]:
    raw_string = raw_messages if isinstance(raw_messages, str) else stable_json(raw_messages)
    raw_hash = hash_text(raw_string)
    relative = str(path.relative_to(source_root))
    payload = f"{asset_id}\0{revision}\0{relative}\0{row_group}\0{row_index}\0{uuid}\0{raw_hash}"
    record_id = hash_text(payload)
    return record_id, {
        "asset_id": asset_id,
        "asset_revision": revision,
        "source": source,
        "shard": path.name,
        "row_group": row_group,
        "row_index": row_index,
        "uuid": uuid,
        "raw_messages_sha256": raw_hash,
    }


def risk_reasons(user: str, assistant: str) -> list[str]:
    value = user + "\n" + assistant
    reasons = []
    if IDENTITY.search(value):
        reasons.append("identity")
    if PROMPT_INJECTION.search(value):
        reasons.append("prompt_injection")
    if TIME_SENSITIVE.search(value):
        reasons.append("time_sensitive")
    if HIGH_RISK.search(user):
        reasons.append("high_risk")
    if SUBJECTIVE.search(user):
        reasons.append("subjective")
    if TOOL_OR_MEDIA.search(value):
        reasons.append("non_text_or_tool")
    return sorted(set(reasons))


def candidate_record(
    record_id: str,
    task_type: str,
    lineage: dict[str, Any],
    user: str,
    assistant: str,
    clean: dict[str, Any],
    language: dict[str, Any],
    think_status: str,
    normalized_prompt_hash: str,
    semantic_key: str | None = None,
) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "language_stats": language,
        "think_status": think_status,
        "hard_flags": [],
        "review_flags": [],
        "normalized_prompt_hash": normalized_prompt_hash,
    }
    if semantic_key is not None:
        quality["semantic_key"] = semantic_key
    return {
        "record_id": record_id,
        "task_type": task_type,
        "lineage": lineage,
        "raw": {"roles": ["user", "assistant"], "user": user, "assistant": assistant},
        "clean": clean,
        "quality": quality,
        "review": {
            "status": "pending",
            "reviewers": [],
            "factual_correct": None,
            "unambiguous": None,
            "reason_codes": [],
            "evidence": [],
        },
        "builder": {"ruleset_version": RULESET_VERSION},
    }


def rejection_record(
    record_id: str,
    task_type: str,
    lineage: dict[str, Any],
    user: str,
    assistant: str,
    reasons: Iterable[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "task_type": task_type,
        "lineage": lineage,
        "prompt": user,
        "answer_excerpt": assistant[-600:],
        "reason_codes": sorted(set(reasons)),
        "builder": {"ruleset_version": RULESET_VERSION},
    }


def evaluate_mc(
    *,
    record_id: str,
    lineage: dict[str, Any],
    user: str,
    assistant: str,
    eval_index: LeakageIndex,
    train_index: LeakageIndex,
) -> tuple[dict[str, Any] | None, list[str]]:
    parsed, reasons = parse_mc_prompt(user)
    if parsed is None:
        return None, reasons
    zh_ok, stats = is_strict_zh(parsed.question + "\n" + "\n".join(parsed.options.values()), min_han=20)
    question_zh_ok, _question_stats = is_strict_zh(parsed.question, min_han=4)
    if not zh_ok or not question_zh_ok:
        reasons.append("non_zh")
    reasons.extend(risk_reasons(user, assistant))
    answer, evidence, think_status, answer_reasons = parse_mc_answer(assistant)
    reasons.extend(answer_reasons)
    if answer is None:
        return None, sorted(set(reasons))
    eval_hit, eval_modes = eval_index.match(user, parsed)
    if eval_hit:
        reasons.append("eval_overlap")
        reasons.extend(f"eval_{mode}" for mode in eval_modes)
    train_hit, _train_modes = train_index.match(user, parsed)
    if train_hit:
        reasons.append("train_overlap")
    if reasons:
        return None, sorted(set(reasons))
    keys = mc_semantic_keys(parsed, answer)
    record = candidate_record(
        record_id,
        "world_mc",
        lineage,
        user,
        assistant,
        {
            "question": parsed.question,
            "options": parsed.options,
            "answer_letter": answer,
            "answer_text": parsed.options[answer],
            "answer_evidence": evidence,
        },
        stats,
        think_status,
        hash_text(mode_normalize(user)),
        keys["semantic_key"],
    )
    record["quality"].update({
        "stem_hash": keys["stem_hash"],
        "ordered_qa_hash": keys["ordered_qa_hash"],
        "option_invariant_hash": keys["option_invariant_hash"],
        "answer_text_norm": keys["answer_text_norm"],
    })
    return record, []


def evaluate_qa(
    *,
    record_id: str,
    lineage: dict[str, Any],
    user: str,
    assistant: str,
    eval_index: LeakageIndex,
    train_index: LeakageIndex,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons = risk_reasons(user, assistant)
    if len(broad_option_labels(user)) >= 3:
        reasons.append("wrong_task")
    user_ok, user_stats = is_strict_zh(user, min_han=12)
    if not user_ok:
        reasons.append("non_zh")
    if not (8 <= len(user) <= 300):
        reasons.append("prompt_length")
    final, think_status = split_reasoning(assistant)
    if final is None:
        reasons.append(think_status)
        return None, sorted(set(reasons))
    final = re.sub(r"^\s*<answer>\s*|\s*</answer>\s*$", "", final, flags=re.IGNORECASE | re.DOTALL).strip()
    answer_ok, answer_stats = is_strict_zh(final, min_han=4)
    if not answer_ok:
        reasons.append("answer_non_zh")
    if not (4 <= len(final) <= 500):
        reasons.append("answer_length")
    if QA_REFERENCE.search(user):
        reasons.append("context_dependent")
    if QA_BAD_ANSWER.search(final):
        reasons.append("missing_context_answer")
    if QA_WRONG_TASK.search(user):
        reasons.append("wrong_task")
    if not QA_FACTUAL.search(user):
        reasons.append("not_factual_qa")
    if "我" in user or "你" in user or "您" in user:
        reasons.append("personalized")
    eval_hit, eval_modes = eval_index.match(user)
    if eval_hit:
        reasons.append("eval_overlap")
        reasons.extend(f"eval_{mode}" for mode in eval_modes)
    normalized_hash = hash_text(mode_normalize(user))
    train_hit, _train_modes = train_index.match(user)
    if train_hit:
        reasons.append("train_overlap")
    if reasons:
        return None, sorted(set(reasons))
    return candidate_record(
        record_id,
        "general_zh_short",
        lineage,
        user,
        assistant,
        {"prompt": user, "answer": final},
        {"prompt": user_stats, "answer": answer_stats},
        think_status,
        normalized_hash,
    ), []


def parquet_rows(path: Path, columns: list[str]) -> Iterator[tuple[int, int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    absolute_index = 0
    for row_group in range(parquet.num_row_groups):
        for batch in parquet.iter_batches(row_groups=[row_group], columns=columns, batch_size=512):
            for row in batch.to_pylist():
                yield row_group, absolute_index, row
                absolute_index += 1


def scan_asset(
    *,
    asset_id: str,
    revision: str,
    root: Path,
    include_qa: bool,
    eval_index: LeakageIndex,
    train_index: LeakageIndex,
    stats: Counter,
    expected_files: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mc_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    files = sorted(root.glob("*.parquet"))
    if len(files) != expected_files:
        raise AssertionError(
            f"{asset_id} parquet signature drifted: {len(files)} != {expected_files}"
        )
    import pyarrow.parquet as pq

    required_columns = {"uuid", "messages", "source"}
    for path in files:
        columns = set(pq.ParquetFile(path).schema.names)
        missing = required_columns - columns
        if missing:
            raise AssertionError(f"{asset_id} missing columns {sorted(missing)} in {path.name}")
    for file_index, path in enumerate(files, start=1):
        columns = ["uuid", "messages", "source"]
        for row_group, row_index, row in parquet_rows(path, columns):
            stats[f"{asset_id}:rows"] += 1
            raw_messages = row.get("messages")
            source = str(row.get("source") or "")
            record_id, lineage = source_locator(
                asset_id,
                revision,
                root,
                path,
                row_group,
                row_index,
                source,
                row.get("uuid"),
                raw_messages,
            )
            parsed_messages = parse_messages(raw_messages)
            if parsed_messages is None:
                stats[f"{asset_id}:invalid_roles_or_text"] += 1
                stats[f"{asset_id}:route:invalid_roles_or_text"] += 1
                continue
            _roles, user, assistant = parsed_messages
            if len(assistant) > 12_000:
                stats[f"{asset_id}:assistant_too_long"] += 1
                stats[f"{asset_id}:route:assistant_too_long"] += 1
                continue

            core = strip_world_wrapper(user)
            broad_mc = len(broad_option_labels(core)) >= 3
            if broad_mc:
                stats[f"{asset_id}:broad_mc"] += 1
                stats[f"{asset_id}:route:broad_mc"] += 1
                record, reasons = evaluate_mc(
                    record_id=record_id,
                    lineage=lineage,
                    user=user,
                    assistant=assistant,
                    eval_index=eval_index,
                    train_index=train_index,
                )
                if record is not None:
                    mc_rows.append(record)
                    stats[f"{asset_id}:mc_candidate"] += 1
                else:
                    stats.update(f"drop_mc:{reason}" for reason in reasons)
                    # Retain only Chinese-ish near-candidates, not tens of
                    # thousands of unrelated English option lists.
                    if language_stats(core)["han"] >= 8:
                        rejections.append(rejection_record(record_id, "world_mc", lineage, user, assistant, reasons))

            elif include_qa:
                final, _think_status = split_reasoning(assistant)
                qa_near = (
                    final is not None
                    and 8 <= len(user) <= 500
                    and 1 <= len(final) <= 800
                    and language_stats(user)["han"] >= 8
                )
                if qa_near:
                    stats[f"{asset_id}:qa_near"] += 1
                    stats[f"{asset_id}:route:qa_near"] += 1
                    record, reasons = evaluate_qa(
                        record_id=record_id,
                        lineage=lineage,
                        user=user,
                        assistant=assistant,
                        eval_index=eval_index,
                        train_index=train_index,
                    )
                    if record is not None:
                        qa_rows.append(record)
                        stats[f"{asset_id}:qa_candidate"] += 1
                    else:
                        stats.update(f"drop_qa:{reason}" for reason in reasons)
                        rejections.append(rejection_record(record_id, "general_zh_short", lineage, user, assistant, reasons))
                else:
                    stats[f"{asset_id}:route:not_candidate"] += 1
            else:
                stats[f"{asset_id}:route:not_candidate"] += 1
        print(
            f"[{asset_id}] {file_index}/{len(files)} files; "
            f"mc={len(mc_rows)} qa={len(qa_rows)} near_reject={len(rejections)}",
            flush=True,
        )
    route_total = sum(
        count for key, count in stats.items() if key.startswith(f"{asset_id}:route:")
    )
    if route_total != stats[f"{asset_id}:rows"]:
        raise AssertionError(
            f"{asset_id} routing is not exhaustive: routes={route_total} rows={stats[f'{asset_id}:rows']}"
        )
    return mc_rows, qa_rows, rejections


def dedupe_candidates(
    rows: list[dict[str, Any]], task_type: str, stats: Counter
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key_field = "option_invariant_hash" if task_type == "world_mc" else "normalized_prompt_hash"
    for row in rows:
        groups[row["quality"][key_field]].append(row)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["record_id"])
        if len(group) == 1:
            kept.append(group[0])
            continue
        if task_type == "world_mc":
            answers = {row["quality"]["answer_text_norm"] for row in group}
        else:
            answers = {semantic_normalize(row["clean"]["answer"]) for row in group}
        reason = "duplicate_conflict" if len(answers) > 1 else "duplicate"
        stats[f"dedupe:{task_type}:{reason}"] += len(group) - (reason == "duplicate")
        if reason == "duplicate":
            kept.append(group[0])
            duplicate_rows = group[1:]
        else:
            duplicate_rows = group
        for row in duplicate_rows:
            rejected.append(
                rejection_record(
                    row["record_id"],
                    task_type,
                    row["lineage"],
                    row["raw"]["user"],
                    row["raw"]["assistant"],
                    [reason],
                )
            )
    return kept, rejected


def token_gate(
    mc_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    tokenizer_dir: Path,
    max_tokens: int,
    stats: Counter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, local_files_only=True)
    kept_mc: list[dict[str, Any]] = []
    kept_qa: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in mc_rows + qa_rows:
        if row["task_type"] == "world_mc":
            clean = row["clean"]
            question = clean["question"] + "\n" + "\n".join(
                f"{label}.{clean['options'][label]}" for label in "ABCD"
            )
            prompt = WORLD_HEAD + question + WORLD_TAIL
            response = f"<think>\n\n</think>\n正确答案是 ({clean['answer_letter']})"
        else:
            prompt = row["clean"]["prompt"].rstrip() + "/no_think"
            response = f"<think>\n\n</think>\n{row['clean']['answer']}"
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        answer_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise AssertionError("unexpected batched chat template output")
            token_ids = token_ids[0]
        if not isinstance(token_ids, list) or any(not isinstance(item, int) for item in token_ids):
            raise TypeError(f"unexpected chat template token type: {type(rendered)!r}")
        total_tokens = len(token_ids)
        row["quality"].update({
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": total_tokens,
            "tokenizer_id": "O6:OneReason-0.8B-pretrain-competition",
        })
        if total_tokens > max_tokens:
            stats[f"drop_{row['task_type']}:token_limit"] += 1
            rejected.append(
                rejection_record(
                    row["record_id"], row["task_type"], row["lineage"],
                    row["raw"]["user"], row["raw"]["assistant"], ["token_limit"]
                )
            )
        elif row["task_type"] == "world_mc":
            kept_mc.append(row)
        else:
            kept_qa.append(row)
    return kept_mc, kept_qa, rejected


def validate_output(rows: list[dict[str, Any]], task_type: str, max_tokens: int) -> None:
    ids = set()
    prompt_hashes = set()
    semantic_keys = set()
    for row in rows:
        assert row["task_type"] == task_type
        assert row["review"]["status"] == "pending"
        assert not row["quality"]["hard_flags"]
        assert row["quality"]["total_tokens"] <= max_tokens
        assert row["record_id"] not in ids
        assert row["quality"]["normalized_prompt_hash"] not in prompt_hashes
        ids.add(row["record_id"])
        prompt_hashes.add(row["quality"]["normalized_prompt_hash"])
        if task_type == "world_mc":
            clean = row["clean"]
            assert list(clean["options"]) == list("ABCD")
            assert clean["answer_letter"] in "ABCD"
            assert row["quality"]["semantic_key"] not in semantic_keys
            semantic_keys.add(row["quality"]["semantic_key"])


def ensure_safe_paths(paths: Iterable[Path]) -> None:
    if not PERSONAL_ROOT.is_mount():
        raise RuntimeError(f"personal volume is not mounted: {PERSONAL_ROOT}")
    if not os.access(PERSONAL_ROOT, os.W_OK):
        raise RuntimeError(f"personal volume is not writable: {PERSONAL_ROOT}")
    paths = tuple(paths)
    resolved_outputs = [path.resolve(strict=False) for path in paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise RuntimeError("output paths must be pairwise distinct")
    for path in paths:
        resolved_parent = path.parent.resolve()
        if not resolved_parent.is_relative_to(PERSONAL_ROOT):
            raise RuntimeError(f"output must be on personal volume: {path} -> {resolved_parent}")
        stable_official = ROOT / "assets/official"
        runtime_official = (RUNTIME_ROOT / "llmrec_2026/data/official").resolve()
        resolved_target = path.resolve(strict=False)
        if (
            resolved_target.is_relative_to(stable_official)
            or resolved_target.is_relative_to(runtime_official)
        ):
            raise RuntimeError(f"refusing to write inside official assets: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o2-dir", type=Path, default=O2_DIR)
    parser.add_argument("--o5-dir", type=Path, default=O5_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--mc-out", type=Path, default=DEFAULT_MC_OUT)
    parser.add_argument("--qa-out", type=Path, default=DEFAULT_QA_OUT)
    parser.add_argument("--reject-out", type=Path, default=DEFAULT_REJECT_OUT)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--qa-limit", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--skip-o5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = (args.mc_out, args.qa_out, args.reject_out, args.audit_out)
    ensure_safe_paths(outputs)
    canonical_inputs = (
        (args.o2_dir, O2_DIR, "O2.General"),
        (args.tokenizer, TOKENIZER_DIR, "O6 tokenizer"),
    )
    if not args.skip_o5:
        canonical_inputs += ((args.o5_dir, O5_DIR, "O5"),)
    for supplied, canonical, label in canonical_inputs:
        if supplied.resolve() != canonical.resolve():
            raise RuntimeError(
                f"{label} must use the registered canonical asset: {canonical}; got {supplied}"
            )
    for source in (args.o2_dir, args.tokenizer):
        if not source.exists():
            raise FileNotFoundError(source)
    if not args.skip_o5 and not args.o5_dir.exists():
        raise FileNotFoundError(args.o5_dir)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = Counter()
    print("[blacklist] loading structured E prompts, benchmark families and eval-log user blocks", flush=True)
    eval_index = load_eval_index()
    train_index, train_counts = load_train_index()
    print(
        f"[blacklist] eval prompts={sum(eval_index.source_counts.values())} "
        f"train prompts={sum(train_counts.values())}",
        flush=True,
    )

    o2_mc, o2_qa, rejections = scan_asset(
        asset_id="O2.General",
        revision="registry-snapshot-20260717",
        root=args.o2_dir,
        include_qa=True,
        eval_index=eval_index,
        train_index=train_index,
        stats=stats,
        expected_files=158,
    )
    stage_counts = {
        "pre_dedupe_mc": len(o2_mc),
        "pre_dedupe_qa": len(o2_qa),
    }
    mc_rows = o2_mc
    qa_rows = o2_qa
    if not args.skip_o5:
        o5_mc, _unused_qa, o5_rejections = scan_asset(
            asset_id="O5",
            revision="4b8e43913aeb8e6c66b9253df4ab64ecc77dfd6c",
            root=args.o5_dir,
            include_qa=False,
            eval_index=eval_index,
            train_index=train_index,
            stats=stats,
            expected_files=301,
        )
        mc_rows.extend(o5_mc)
        rejections.extend(o5_rejections)
        stage_counts["pre_dedupe_mc"] += len(o5_mc)

    mc_rows, dup_mc = dedupe_candidates(mc_rows, "world_mc", stats)
    qa_rows, dup_qa = dedupe_candidates(qa_rows, "general_zh_short", stats)
    rejections.extend(dup_mc)
    rejections.extend(dup_qa)
    stage_counts.update({"post_dedupe_mc": len(mc_rows), "post_dedupe_qa": len(qa_rows)})

    # QA is a review pool, not a training sample.  A stable record-id cap avoids
    # loading a large, low-density review burden while preserving reproducibility.
    qa_rows.sort(key=lambda row: row["record_id"])
    if args.qa_limit >= 0 and len(qa_rows) > args.qa_limit:
        for row in qa_rows[args.qa_limit :]:
            rejections.append(
                rejection_record(
                    row["record_id"], row["task_type"], row["lineage"],
                    row["raw"]["user"], row["raw"]["assistant"], ["review_pool_cap"]
                )
            )
        stats["drop_qa:review_pool_cap"] += len(qa_rows) - args.qa_limit
        qa_rows = qa_rows[: args.qa_limit]
    stage_counts.update({"post_cap_mc": len(mc_rows), "post_cap_qa": len(qa_rows)})

    mc_rows, qa_rows, token_rejections = token_gate(
        mc_rows, qa_rows, args.tokenizer, args.max_tokens, stats
    )
    rejections.extend(token_rejections)
    stage_counts.update({"post_token_mc": len(mc_rows), "post_token_qa": len(qa_rows)})
    builder_sha = sha256_file(Path(__file__))
    build_fingerprint = hash_text(stable_json({
        "builder_sha256": builder_sha,
        "ruleset_version": RULESET_VERSION,
        "o2_revision": "registry-snapshot-20260717",
        "o5_revision": None if args.skip_o5 else "4b8e43913aeb8e6c66b9253df4ab64ecc77dfd6c",
        "qa_limit": args.qa_limit,
        "max_tokens": args.max_tokens,
    }))
    for row in mc_rows + qa_rows + rejections:
        row["builder"].update({
            "builder_sha256": builder_sha,
            "build_fingerprint": build_fingerprint,
        })
    mc_rows.sort(key=lambda row: row["record_id"])
    qa_rows.sort(key=lambda row: row["record_id"])
    rejections.sort(key=lambda row: (row["task_type"], row["record_id"], row["reason_codes"]))
    validate_output(mc_rows, "world_mc", args.max_tokens)
    validate_output(qa_rows, "general_zh_short", args.max_tokens)

    atomic_jsonl(args.mc_out, mc_rows)
    atomic_jsonl(args.qa_out, qa_rows)
    atomic_jsonl(args.reject_out, rejections)
    audit = {
        "asset_class": (
            "D-candidate(O2.General); NOT TRAINING DATA"
            if args.skip_o5
            else "D-candidate(O2.General,O5); NOT TRAINING DATA"
        ),
        "created_at_utc": created_at,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "ruleset_version": RULESET_VERSION,
        "parameters": {
            "qa_limit": args.qa_limit,
            "max_tokens": args.max_tokens,
            "skip_o5": args.skip_o5,
            "batch_size": 512,
            "workers": 1,
            "tokenizer_path": str(args.tokenizer.resolve()),
            "tokenizer_id": "O6:OneReason-0.8B-pretrain-competition",
        },
        "upstream": {
            "O2.General": {
                "registry_path": str(args.o2_dir.resolve()),
                "revision": "registry-snapshot-20260717",
                "parquet_files": len(list(args.o2_dir.glob("*.parquet"))),
            },
            "O5": None if args.skip_o5 else {
                "registry_path": str(args.o5_dir.resolve()),
                "revision": "4b8e43913aeb8e6c66b9253df4ab64ecc77dfd6c",
                "parquet_files": len(list(args.o5_dir.glob("*.parquet"))),
            },
        },
        "blacklist": {
            "policy": "only prompt text/signatures affect selection; parsed gold/output fields are discarded and never used",
            "eval_prompt_instances": sum(eval_index.source_counts.values()),
            "eval_sources": dict(sorted(eval_index.source_counts.items())),
            "current_parent_prompt_instances": dict(sorted(train_counts.items())),
            "current_parent_mode_hashes": len(train_index.mode_hashes),
            "current_parent_semantic_hashes": len(train_index.semantic_hashes),
            "current_parent_stem_hashes": len(train_index.stem_hashes),
            "current_parent_option_invariant_hashes": len(train_index.invariant_hashes),
            "raw_hashes": len(eval_index.raw_hashes),
            "mode_hashes": len(eval_index.mode_hashes),
            "semantic_hashes": len(eval_index.semantic_hashes),
            "stem_hashes": len(eval_index.stem_hashes),
            "option_invariant_hashes": len(eval_index.invariant_hashes),
        },
        "filter_counts": dict(sorted(stats.items())),
        "stage_counts": stage_counts,
        "outputs": {},
        "release_gate": {
            "training_projection_created": False,
            "required_next_step": "independent human review; only review.status=pass may be projected",
            "pending_mc": len(mc_rows),
            "pending_qa": len(qa_rows),
        },
    }
    for name, path, count in (
        ("mc_candidates", args.mc_out, len(mc_rows)),
        ("qa_candidates", args.qa_out, len(qa_rows)),
        ("near_rejections", args.reject_out, len(rejections)),
    ):
        audit["outputs"][name] = {
            "path": str(path.resolve()),
            "rows": count,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_json(args.audit_out, audit)
    print(json.dumps(audit["release_gate"], ensure_ascii=False, sort_keys=True), flush=True)
    print(f"[OK] audit: {args.audit_out}", flush=True)


if __name__ == "__main__":
    main()
