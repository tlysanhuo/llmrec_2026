#!/usr/bin/env python3
"""Recover an answer-blind O5 Infinity non-math MC prompt-review packet.

The only candidate source is the frozen ``world_clean_near_rejections``
ledger.  This builder streams that ledger once and considers only
``Infinity_Instruct`` rows rejected solely as ``mc_labels``.  It recovers
native A-D prompt structure, applies conservative prompt-only risk gates, and
checks every survivor against prior review assets, central evaluation prompts,
and current parent-training prompts.

The output is deliberately *not* training data.  It contains question text,
four native choices, lineage, and mechanical review state only.  Source
assistant text and every answer/gold-like field are prohibited recursively.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_official_general_world_clean as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/derived/official_general/world_clean_near_rejections.jsonl"
OUTPUT = (
    ROOT
    / "assets/derived/official_general/o5_infinity_nonmath_mc_format_recovery_prompt_packet.jsonl"
)
AUDIT = ROOT / "logs/data/o5_infinity_nonmath_mc_format_recovery_prompt_packet_audit.json"

EXPECTED_INPUT_SHA256 = "f3c26856eaceef7381166a67d1cc74eeb51e2e3ff0faec18c841e9745c17302f"
EXPECTED_INPUT_ROWS = 8_803
EXPECTED_INFINITY_ROWS = 245
EXPECTED_TARGET_ROWS = 116
EXPECTED_EARLIER_MANUAL_FORMAT_ROWS = 118
SOURCE_NAME = "Infinity_Instruct"
TARGET_REASON_CODES = ("mc_labels",)
EARLIER_MANUAL_FORMAT_CODES = {("answer_not_final",), ("answer_unparsed",)}
RULESET_VERSION = "o5-infinity-nonmath-mc-prompt-recovery-20260718-v1"

# Prompt-bearing assets are also indexed semantically.  Review ledgers are
# frozen separately so that every prior prompt-review decision is in the ID
# exclusion union even if a later projection contains only the passing subset.
FROZEN_EXCLUSIONS: dict[str, dict[str, Any]] = {
    "strict_mc_candidates": {
        "path": ROOT / "assets/derived/official_general/world_mc_strict_candidates.jsonl",
        "rows": 5,
        "sha256": "b25aefe29cebb236ed339d1675c122349151ded4462bd9ad19469a12a08b3b75",
        "index_prompts": True,
    },
    "legacy_manual_review_passes": {
        "path": ROOT / "assets/derived/official_general/world_mc_human_reviewed_safe.jsonl",
        "rows": 29,
        "sha256": "75a4249824272ab3afe8255a18ad0a8e13e6d9ccf1e10e69233185082dc1fd74",
        "index_prompts": True,
    },
    "o5_prompt_review_ledger": {
        "path": ROOT / "logs/data/o5_en_mc_prompt_review_ledger.jsonl",
        "rows": 51,
        "sha256": "54487b8d0c1d8bd4d32c0bcdcec0b7d25e418435cde5730b47fc38fee481bb1a",
        "index_prompts": False,
    },
    "o2_prompt_review_ledger": {
        "path": ROOT / "logs/data/o2_en_mc_prompt_review_ledger.jsonl",
        "rows": 38,
        "sha256": "f1219999893a9c4ab6e12779e98543dfab1e72c43e5a0180ed6353fb2b045cea",
        "index_prompts": False,
    },
    "o5_prompt_review_source": {
        "path": ROOT / "assets/derived/official_general/o5_en_mc_answer_claim_pilot.jsonl",
        "rows": 51,
        "sha256": "21c43445b2cb23ca6e46cd5c812b0ae883ab174a763fce3209baef5d76533db3",
        "index_prompts": True,
    },
    "o2_prompt_review_source": {
        "path": ROOT / "assets/derived/official_general/o2_en_mc_answer_claim_pilot.jsonl",
        "rows": 38,
        "sha256": "6a78709e887feb0ae034c2392e9d3117abe28ff566365f65c5623dd87a1ab1cf",
        "index_prompts": True,
    },
    "final_reviewed_train": {
        "path": ROOT
        / "assets/derived/official_general/official_general_world_mc_v1_train_reviewed.jsonl",
        "rows": 68,
        "sha256": "c86705ba3aea45c7b810be6b8ebad45ded5d48687cb5276362578d2e2d5ac52d",
        "index_prompts": True,
    },
    "permanent_holdout": {
        "path": ROOT / "assets/evaluation/holdout/official_general_world_mc_v1_holdout.jsonl",
        "rows": 25,
        "sha256": "fb67b76d8d071799ba372185bd89cb556afef9065a1b188fb9dd86a9131e13df",
        "index_prompts": True,
    },
}

# The recovery parser is intentionally narrow.  It permits inline choices and
# lowercase choice markers because these are precisely the formatting misses
# in the upstream strict-line parser, but it still requires one ordered A-D
# run and rejects every E-H marker.
OPTION_MARKER = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[（(\[]\s*([A-Ha-h])\s*[）)\]]\s*[.．、:：]?|"
    r"([A-Ha-h])\s*[.．、:：)）]"
    r")\s*"
)

MEDICAL = re.compile(
    r"患者|病人|医学|医药|中医|疾病|病原|治疗|用药|药物|药品|临床|症状|"
    r"癌症|肿瘤|冠心病|心血管|血压|胆固醇|糖尿病|膝反射|血细胞|护理|"
    r"妊娠|孕妇|疫苗|传染病|病毒|吸烟|烟草|二手烟|饮食营养|供热比例"
)
LEGAL = re.compile(
    r"法律|法治|依法|法典|法案|法规|条例|立法|执法|司法|宪法|民法|刑法|"
    r"诉讼|法院|检察院|公民权|法律义务|追究责任"
)
POLITICAL = re.compile(
    r"政治|政府|政党|共产党|国民党|社会主义|资本主义|总统|首相|总理|"
    r"国王|女王|君主|议会|外交|一国两制|民族区域自治|改革开放|革命|"
    r"扶贫|一带一路|两岸|台湾|联合国|战争|会战|抗战|军队|军民|侵略|"
    r"国共|民族英雄|专制|丞相|基本路线|政策扶持"
)
FINANCIAL = re.compile(
    r"金融|财经|股票|证券|基金|投资|贷款|利率|汇率|保险|银行|上市|市值|"
    r"收益|利润|财政|资本|GDP|国内生产总值|经济增长|市场价格|成本|"
    r"进出口|进口|出口|贸易|电商|收入阶段|稿费"
)
TIME_SENSITIVE = re.compile(
    r"截至|截止|当前|目前|当今|迄今|现任|最新|今天|昨天|明天|今年|明年|"
    r"本月|本周|实时|近年来|近年|现在"
)
READING_COMPREHENSION = re.compile(
    r"请(?:根据|阅读)|阅读(?:下面|以下)|根据(?:文章|材料|文段|段落|文本|上文|下文|所给)|"
    r"材料(?:一|二|三|反映|所述)|文段|据此(?:完成|判断)|基于(?:文章|段落|文本)|"
    r"Hypothesis|paragraph|text\s+below",
    re.IGNORECASE,
)
MATH_OR_COMPUTATION = re.compile(
    r"数学|计算|求解|方程|函数|几何|三角形|概率|数列|偶数|倍数|小数|"
    r"最小(?:的)?数|最大公约数|最小公倍数|面积|周长|体积|圆的|平分|百分比|"
    r"\\frac|\\sqrt|\$|<sup>|\bpi\b|π|"
    r"\d\s*(?:\+|×|÷|\*|/|=|<|>|\^)\s*\d",
    re.IGNORECASE,
)
CODE_OR_TRANSFORMATION = re.compile(
    r"编写|修改.{0,20}代码|代码来|程序|算法|JavaScript|Swift|Python|"
    r"聚类|分类(?:为|，可以)|prompt\s+generator|Midjourney|```",
    re.IGNORECASE,
)
LANGUAGE_FORM_TASK = re.compile(
    r"填入(?:空白|空格|句子)|填空|正确的(?:单词|词语|拼写)|拼写形式|"
    r"选择.{0,12}(?:翻译|句子)|使(?:句子|文本)有意义"
)
NEGATED_STEM = re.compile(
    r"错误(?:的|的是|的一项|选项)?|不正确|不属于|不包括|不能|不可能|"
    r"下列.{0,12}不|哪一项.{0,8}(?:错|不)"
)
SUBJECTIVE_OR_NORMATIVE = re.compile(
    r"你认为|你觉得|最配|最能代表你的|你的性格|我们应该|你应该|您应该|"
    r"随心所欲|个人未来|发脾气|不愉快|面对他人|社交场合|职业.*选择|"
    r"认识正确|道德|奉献精神|我们要|应当|应该"
)
MULTISELECT_OR_COMBINATION = re.compile(
    r"[①②③④⑤⑥⑦⑧⑨⑩]|多项选择|多选|不定项|有几项|哪些|哪几项|"
    r"所有适用|选择.{0,3}(?:两|二|三|四|2|3|4)项|常见(?:品种|类型)"
)
ALL_NONE_OR_CROSS_REFERENCE = re.compile(
    r"以上(?:都|均|全部|皆|没有|不)|(?:都|均|全部)不正确|"
    r"(?:A|B|C|D)\s*(?:和|与|及|、)\s*(?:A|B|C|D)",
    re.IGNORECASE,
)
TRAILING_OR_GENERATION_META = re.compile(
    r"回答上面|详细回答|解决这个问题|解释原因|然后解释|请记住|"
    r"I\s+think\s+the\s+answer\s+is|请给出.{0,8}(?:答案|解析|理由)",
    re.IGNORECASE,
)
NUMERIC_CHOICE = re.compile(r"^[+\-]?\d+(?:\.\d+)?(?:%|％)?$")
NUMERIC_EXTREMUM = re.compile(r"最小|最大|最低|最高")

MAX_QUESTION_CHARS = 500
MAX_CHOICE_CHARS = 180
MIN_HAN = 12

FORBIDDEN_PACKET_KEY_FRAGMENTS = frozenset(
    {"answer", "label", "assistant", "response", "gold", "evidence", "output"}
)
LINEAGE_KEYS = (
    "asset_id",
    "asset_revision",
    "source",
    "shard",
    "row_group",
    "row_index",
    "uuid",
    "raw_messages_sha256",
)


@dataclass(frozen=True)
class RecoveredMC:
    question: str
    options: dict[str, str]
    marker_forms: tuple[str, ...]


def _marker_choice(match: re.Match[str]) -> str:
    return (match.group(1) or match.group(2)).upper()


def _marker_form(match: re.Match[str]) -> str:
    token = match.group(0).strip()
    if token.startswith(("(", "（", "[")):
        return "bracketed"
    if token.endswith((")", "）")):
        return "close_parenthesis"
    if token.endswith((":", "：")):
        return "colon"
    if token.endswith(("、",)):
        return "enumeration_comma"
    return "dot"


def parse_format_near_mc(prompt: str) -> tuple[RecoveredMC | None, list[str]]:
    """Parse exactly one native A-D run without consulting any response."""

    core = base.normalize_raw(prompt).strip()
    matches = list(OPTION_MARKER.finditer(core))
    choices = [_marker_choice(match) for match in matches]
    reasons: list[str] = []
    if any(choice in "EFGH" for choice in choices):
        reasons.append("extra_choice_eh")
    if len(matches) != 4:
        reasons.append("choice_count_not_four")
    if choices != list("ABCD"):
        reasons.append("choice_sequence_not_abcd")
    if reasons:
        return None, sorted(set(reasons))

    question = base.strip_single_choice_boilerplate(core[: matches[0].start()].strip())
    options: dict[str, str] = {}
    for index, choice in enumerate("ABCD"):
        end = matches[index + 1].start() if index < 3 else len(core)
        options[choice] = core[matches[index].end() : end].strip()

    if not question:
        reasons.append("empty_question")
    if any(not options[choice] for choice in "ABCD"):
        reasons.append("empty_choice")
    normalized = [base.semantic_normalize(options[choice]) for choice in "ABCD"]
    if any(not value for value in normalized) or len(set(normalized)) != 4:
        reasons.append("duplicate_or_empty_choice")
    if len(question) > MAX_QUESTION_CHARS:
        reasons.append("question_too_long")
    if any(len(options[choice]) > MAX_CHOICE_CHARS for choice in "ABCD"):
        reasons.append("choice_too_long")
    if reasons:
        return None, sorted(set(reasons))
    return (
        RecoveredMC(
            question=question,
            options=options,
            marker_forms=tuple(_marker_form(match) for match in matches),
        ),
        [],
    )


def prompt_policy_reasons(prompt: str, parsed: RecoveredMC | None) -> tuple[list[str], dict[str, Any]]:
    """Return conservative prompt-only exclusions and auditable language stats."""

    normalized_prompt = base.normalize_raw(prompt).strip()
    question = parsed.question if parsed is not None else normalized_prompt
    combined = (
        question + "\n" + "\n".join(parsed.options[choice] for choice in "ABCD")
        if parsed is not None
        else normalized_prompt
    )
    reasons: list[str] = []
    strict_zh, language = base.is_strict_zh(combined, min_han=MIN_HAN)
    if not strict_zh:
        reasons.append("not_strict_zh")
    if MULTISELECT_OR_COMBINATION.search(combined) or base.MULTISELECT.search(combined):
        reasons.append("multiselect_or_combination")
    if READING_COMPREHENSION.search(question):
        reasons.append("reading_comprehension")
    if MATH_OR_COMPUTATION.search(combined):
        reasons.append("math_or_computation")
    if parsed is not None and NUMERIC_EXTREMUM.search(question) and all(
        NUMERIC_CHOICE.fullmatch(parsed.options[choice].strip()) for choice in "ABCD"
    ):
        reasons.append("math_or_computation")
    if CODE_OR_TRANSFORMATION.search(combined):
        reasons.append("code_or_transformation")
    if LANGUAGE_FORM_TASK.search(question):
        reasons.append("language_form_task")
    if MEDICAL.search(combined) or base.HIGH_RISK.search(combined):
        reasons.append("medical_or_other_high_risk")
    if LEGAL.search(combined):
        reasons.append("legal")
    if POLITICAL.search(combined):
        reasons.append("political")
    if FINANCIAL.search(combined):
        reasons.append("financial")
    if TIME_SENSITIVE.search(question) or base.TIME_SENSITIVE.search(question):
        reasons.append("time_sensitive")
    if NEGATED_STEM.search(question):
        reasons.append("negated_stem")
    if SUBJECTIVE_OR_NORMATIVE.search(combined):
        reasons.append("subjective_or_normative")
    choice_scope = (
        "\n".join(parsed.options[choice] for choice in "ABCD")
        if parsed is not None
        else combined
    )
    if ALL_NONE_OR_CROSS_REFERENCE.search(choice_scope):
        reasons.append("all_none_or_cross_reference")
    if TRAILING_OR_GENERATION_META.search(combined):
        reasons.append("trailing_or_generation_meta")
    if base.PROMPT_GOLD_LEAK.search(combined):
        reasons.append("prompt_target_leak")
    if parsed is not None and any(
        base.OPTION_GOLD_MARKER.search(parsed.options[choice]) for choice in "ABCD"
    ):
        reasons.append("choice_target_marker")
    return sorted(set(reasons)), language


def canonical_prompt(parsed: RecoveredMC) -> str:
    return parsed.question + "\n" + "\n".join(
        f"{choice}. {parsed.options[choice]}" for choice in "ABCD"
    )


def _read_small_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid JSON: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _index_prompt_row(index: base.LeakageIndex, row: dict[str, Any], source: str) -> bool:
    clean = row.get("clean")
    if isinstance(clean, dict):
        question = clean.get("question")
        options = clean.get("options")
        if isinstance(question, str) and isinstance(options, dict):
            projected = {choice: options.get(choice) for choice in "ABCD"}
            if all(isinstance(projected[choice], str) for choice in "ABCD"):
                base.add_structured_mc(index, question, projected, source)  # type: ignore[arg-type]
                return True
    original = row.get("original")
    if isinstance(original, dict):
        question = original.get("question")
        options = original.get("options")
        if isinstance(question, str) and isinstance(options, dict):
            projected = {choice: options.get(choice) for choice in "ABCD"}
            if all(isinstance(projected[choice], str) for choice in "ABCD"):
                base.add_structured_mc(index, question, projected, source)  # type: ignore[arg-type]
                return True
    top_input = row.get("input")
    if isinstance(top_input, str) and top_input.strip():
        index.add(top_input, source)
        return True
    return False


def load_frozen_exclusions() -> tuple[set[str], base.LeakageIndex, dict[str, Any]]:
    excluded_ids: set[str] = set()
    prompt_index = base.LeakageIndex()
    manifest: dict[str, Any] = {}
    for name, spec in FROZEN_EXCLUSIONS.items():
        path = spec["path"]
        actual_sha = base.sha256_file(path)
        if actual_sha != spec["sha256"]:
            raise AssertionError(f"frozen exclusion drifted: {name}: {actual_sha} != {spec['sha256']}")
        rows = _read_small_jsonl(path)
        if len(rows) != spec["rows"]:
            raise AssertionError(f"frozen exclusion row count drifted: {name}: {len(rows)} != {spec['rows']}")
        indexed = 0
        for row in rows:
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"frozen exclusion has no record_id: {name}")
            excluded_ids.add(record_id)
            if spec["index_prompts"] and _index_prompt_row(prompt_index, row, name):
                indexed += 1
        if spec["index_prompts"] and indexed != len(rows):
            raise AssertionError(f"not every prompt-bearing exclusion was indexed: {name}: {indexed}/{len(rows)}")
        manifest[name] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": actual_sha,
            "prompt_rows_indexed": indexed,
        }
    return excluded_ids, prompt_index, manifest


def _all_key_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}" if prefix else key_string
            paths.append(path)
            paths.extend(_all_key_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_all_key_paths(child, f"{prefix}[{index}]"))
    return paths


def forbidden_packet_key_paths(value: Any) -> list[str]:
    hits = []
    for path in _all_key_paths(value):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0].casefold()
        if any(fragment in key for fragment in FORBIDDEN_PACKET_KEY_FRAGMENTS):
            hits.append(path)
    return sorted(set(hits))


def _project_lineage(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("candidate has no lineage object")
    missing = [key for key in LINEAGE_KEYS if key not in raw]
    if missing:
        raise ValueError(f"candidate lineage is missing keys: {missing}")
    projected = {key: raw[key] for key in LINEAGE_KEYS}
    if projected["source"] != SOURCE_NAME:
        raise ValueError(f"unexpected candidate source: {projected['source']}")
    return projected


def _match_modes(
    index: base.LeakageIndex, prompt: str, parsed: RecoveredMC
) -> list[str]:
    base_parsed = base.ParsedMC(question=parsed.question, options=parsed.options, postscript="")
    matched, modes = index.match(prompt, base_parsed)
    return modes if matched else []


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.input.resolve() != INPUT.resolve():
        raise RuntimeError(f"input must be the frozen canonical ledger: {INPUT}")
    base.ensure_safe_paths((args.output, args.audit))
    input_sha = base.sha256_file(args.input)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise AssertionError(f"near-rejection ledger drifted: {input_sha} != {EXPECTED_INPUT_SHA256}")

    excluded_ids, reviewed_index, exclusion_manifest = load_frozen_exclusions()
    print("[blacklist] loading central E and current-parent prompt indexes", flush=True)
    eval_index = base.load_eval_index()
    train_index, train_counts = base.load_train_index()

    scan = Counter()
    source_reason_counts: Counter[str] = Counter()
    earlier_manual_ids: set[str] = set()
    target_rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            scan["rows"] += 1
            try:
                raw = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid input JSON: {args.input}:{line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"input row is not an object: {args.input}:{line_number}")
            reason_codes_raw = raw.get("reason_codes")
            if not isinstance(reason_codes_raw, list) or not all(
                isinstance(code, str) for code in reason_codes_raw
            ):
                raise ValueError(f"invalid reason_codes: {args.input}:{line_number}")
            reason_codes = tuple(reason_codes_raw)
            record_id = raw.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"invalid record_id: {args.input}:{line_number}")
            if reason_codes in EARLIER_MANUAL_FORMAT_CODES:
                earlier_manual_ids.add(record_id)

            lineage = raw.get("lineage")
            source = lineage.get("source") if isinstance(lineage, dict) else None
            if source != SOURCE_NAME:
                continue
            scan["infinity_rows"] += 1
            source_reason_counts["+".join(reason_codes)] += 1
            if reason_codes != TARGET_REASON_CODES:
                continue
            scan["target_rows"] += 1
            prompt = raw.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"target row has no prompt: {args.input}:{line_number}")
            # Project immediately.  In particular, never retain or inspect the
            # ledger's answer_excerpt payload.
            target_rows.append(
                {
                    "record_id": record_id,
                    "lineage": _project_lineage(lineage),
                    "prompt": prompt,
                }
            )

    if scan["rows"] != EXPECTED_INPUT_ROWS:
        raise AssertionError(f"input row count drifted: {scan['rows']} != {EXPECTED_INPUT_ROWS}")
    if scan["infinity_rows"] != EXPECTED_INFINITY_ROWS:
        raise AssertionError(
            f"Infinity row count drifted: {scan['infinity_rows']} != {EXPECTED_INFINITY_ROWS}"
        )
    if scan["target_rows"] != EXPECTED_TARGET_ROWS:
        raise AssertionError(f"target row count drifted: {scan['target_rows']} != {EXPECTED_TARGET_ROWS}")
    if len(earlier_manual_ids) != EXPECTED_EARLIER_MANUAL_FORMAT_ROWS:
        raise AssertionError(
            f"earlier manual-format cohort drifted: {len(earlier_manual_ids)} "
            f"!= {EXPECTED_EARLIER_MANUAL_FORMAT_ROWS}"
        )
    if len({row["record_id"] for row in target_rows}) != len(target_rows):
        raise AssertionError("duplicate target record_id")

    builder_sha = base.sha256_file(Path(__file__))
    filter_spec = {
        "max_question_chars": MAX_QUESTION_CHARS,
        "max_choice_chars": MAX_CHOICE_CHARS,
        "min_han": MIN_HAN,
        "patterns": {
            "all_none_or_cross_reference": ALL_NONE_OR_CROSS_REFERENCE.pattern,
            "code_or_transformation": CODE_OR_TRANSFORMATION.pattern,
            "financial": FINANCIAL.pattern,
            "legal": LEGAL.pattern,
            "language_form_task": LANGUAGE_FORM_TASK.pattern,
            "math_or_computation": MATH_OR_COMPUTATION.pattern,
            "medical": MEDICAL.pattern,
            "multiselect_or_combination": MULTISELECT_OR_COMBINATION.pattern,
            "negated_stem": NEGATED_STEM.pattern,
            "option_marker": OPTION_MARKER.pattern,
            "political": POLITICAL.pattern,
            "reading_comprehension": READING_COMPREHENSION.pattern,
            "subjective_or_normative": SUBJECTIVE_OR_NORMATIVE.pattern,
            "time_sensitive": TIME_SENSITIVE.pattern,
            "trailing_or_generation_meta": TRAILING_OR_GENERATION_META.pattern,
        },
    }
    build_fingerprint = base.hash_text(
        base.stable_json(
            {
                "builder_sha256": builder_sha,
                "exclusions": {
                    name: {"rows": value["rows"], "sha256": value["sha256"]}
                    for name, value in exclusion_manifest.items()
                },
                "filter_spec": filter_spec,
                "input_sha256": input_sha,
                "ruleset_version": RULESET_VERSION,
            }
        )
    )

    rejection_rows = Counter()
    rejection_hits = Counter()
    eval_modes = Counter()
    train_modes = Counter()
    reviewed_modes = Counter()
    packet_modes = Counter()
    accepted_index = base.LeakageIndex()
    packet: list[dict[str, Any]] = []

    for candidate in sorted(target_rows, key=lambda row: row["record_id"]):
        parsed, parse_reasons = parse_format_near_mc(candidate["prompt"])
        policy_reasons, language = prompt_policy_reasons(candidate["prompt"], parsed)
        reasons = set(parse_reasons + policy_reasons)
        record_id = candidate["record_id"]
        if record_id in excluded_ids:
            reasons.add("prior_review_id")
        if record_id in earlier_manual_ids:
            reasons.add("earlier_manual_format_id")

        canonical = ""
        if parsed is not None:
            canonical = canonical_prompt(parsed)
            modes = _match_modes(reviewed_index, canonical, parsed)
            if modes:
                reasons.add("prior_review_semantic_overlap")
                reviewed_modes.update(modes)
            modes = _match_modes(eval_index, canonical, parsed)
            if modes:
                reasons.add("evaluation_overlap")
                eval_modes.update(modes)
            modes = _match_modes(train_index, canonical, parsed)
            if modes:
                reasons.add("parent_training_overlap")
                train_modes.update(modes)
            modes = _match_modes(accepted_index, canonical, parsed)
            if modes:
                reasons.add("within_packet_duplicate")
                packet_modes.update(modes)

        if reasons:
            rejection_rows[sorted(reasons)[0]] += 1
            rejection_hits.update(reasons)
            continue
        if parsed is None:
            raise AssertionError("accepted row has no recovered prompt")

        accepted_index.add(canonical, "current_packet")
        prompt_hash = base.hash_text(
            parsed.question + "\0" + "\0".join(parsed.options[choice] for choice in "ABCD")
        )
        packet.append(
            {
                "record_id": record_id,
                "task_type": "world_mc_prompt_only_format_recovery_candidate",
                "lineage": candidate["lineage"],
                "prompt": {
                    "language": "zh",
                    "question": parsed.question,
                    "options": parsed.options,
                    "prompt_sha256": prompt_hash,
                },
                "mechanical_checks": {
                    "choice_count": 4,
                    "choice_set": "ABCD",
                    "native_choice_text_preserved": True,
                    "format_recovery_only": True,
                    "choice_marker_forms": list(parsed.marker_forms),
                    "strict_zh": True,
                    "language_stats": language,
                    "risk_policy_clear": True,
                    "evaluation_blacklist_clear": True,
                    "parent_training_blacklist_clear": True,
                    "prior_review_blacklist_clear": True,
                },
                "candidate_state": {
                    "status": "pending_human_prompt_and_factual_review",
                    "training_eligible": False,
                    "factuality_verified": False,
                    "distractors_synthesized": False,
                },
                "builder": {
                    "ruleset_version": RULESET_VERSION,
                    "builder_sha256": builder_sha,
                    "build_fingerprint": build_fingerprint,
                },
            }
        )

    packet.sort(key=lambda row: row["record_id"])
    forbidden_hits = forbidden_packet_key_paths(packet)
    if forbidden_hits:
        raise AssertionError(f"prompt-only packet contains forbidden key paths: {forbidden_hits}")
    if len({row["record_id"] for row in packet}) != len(packet):
        raise AssertionError("duplicate output record_id")
    if any(row["record_id"] in excluded_ids | earlier_manual_ids for row in packet):
        raise AssertionError("prior-reviewed ID leaked into packet")

    base.atomic_jsonl(args.output, packet)
    output_sha = base.sha256_file(args.output)
    audit = {
        "asset_class": "D-prompt-only-candidate(O5.Infinity_Instruct); NOT TRAINING DATA",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ruleset_version": RULESET_VERSION,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": build_fingerprint,
        "input": {
            "path": str(args.input.resolve()),
            "bytes": args.input.stat().st_size,
            "rows": scan["rows"],
            "sha256": input_sha,
            "streaming_passes": 1,
            "source_rows": scan["infinity_rows"],
            "target_reason_codes": list(TARGET_REASON_CODES),
            "target_rows": scan["target_rows"],
            "source_reason_distribution": _counter_dict(source_reason_counts),
        },
        "exclusions": {
            "frozen_assets": exclusion_manifest,
            "frozen_asset_union_ids": len(excluded_ids),
            "earlier_manual_format_rows_discovered_in_input": len(earlier_manual_ids),
            "central_e_prompt_rows": sum(eval_index.source_counts.values()),
            "central_e_source_counts": _counter_dict(eval_index.source_counts),
            "current_parent_prompt_rows": sum(train_counts.values()),
            "current_parent_source_counts": _counter_dict(train_counts),
        },
        "filter_spec": filter_spec,
        "filtering": {
            "target_rows": len(target_rows),
            "accepted_rows": len(packet),
            "rejected_rows": len(target_rows) - len(packet),
            "primary_rejection_counts": _counter_dict(rejection_rows),
            "all_rejection_hit_counts": _counter_dict(rejection_hits),
            "evaluation_overlap_modes": _counter_dict(eval_modes),
            "parent_training_overlap_modes": _counter_dict(train_modes),
            "prior_review_overlap_modes": _counter_dict(reviewed_modes),
            "within_packet_overlap_modes": _counter_dict(packet_modes),
        },
        "prompt_only_isolation": {
            "forbidden_packet_key_fragments": sorted(FORBIDDEN_PACKET_KEY_FRAGMENTS),
            "forbidden_key_path_hits": forbidden_hits,
            "source_assistant_text_copied": False,
            "source_answer_payload_copied": False,
            "source_metadata_target_copied": False,
            "distractors_created_or_modified": False,
        },
        "output": {
            "path": str(args.output.resolve()),
            "rows": len(packet),
            "bytes": args.output.stat().st_size,
            "sha256": output_sha,
            "record_ids_sha256": base.hash_text("\n".join(row["record_id"] for row in packet)),
            "training_eligible_rows": 0,
            "requires_human_prompt_and_factual_review": len(packet),
        },
        "decision": "PROMPT_REVIEW_PACKET_ONLY__NO_TRAINING_PROJECTION",
    }
    base.atomic_json(args.audit, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def main() -> None:
    audit = build(parse_args())
    print(base.stable_json(audit))


if __name__ == "__main__":
    main()
