#!/usr/bin/env python3
"""Full read-only EDA for registered official assets O4/O5.

This is the exhaustive companion to scripts/data/eda_general_official.py.
It scans every registered Parquet row, writes only outside assets/official,
and separates exact counts from the tokenizer-based estimates in that script.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import unicodedata
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
ASSETS = {
    "O4": ROOT / "assets/official/general_pretrain",
    "O5": ROOT / "assets/official/general_sft",
}
SOURCES = [
    "reasoning_v1_20m", "OpenMathReasoning", "NuminaMath-QwQ-CoT-5M",
    "OpenCodeReasoning_new", "KodCode_V1_SFT_R1",
    "Chinese-Reasoning-Distil-Data", "medical-o1-reasoning-SFT",
    "Bespoke-Stratos-17k", "R1-Distill-SFT", "Infinity_Instruct",
    "OpenCoderReasoning", "Chinese-Reasoning-Distil-Data-think",
    "Reasoning_Multi_subject_RLVR", "Reasoning_KodCode_V1_SFT_R1",
    "DeepMath103K", "medical-o1-reasoning-SFT-think",
]
SRC_ID = {s: i for i, s in enumerate(SOURCES)}

THINK_OPEN = re.compile(r"<think>|<\|begin_of_thought\|>|<analysis>", re.I)
THINK_CLOSE = re.compile(r"</think>|<\|end_of_thought\|>|</analysis>", re.I)
THINK_SPAN = re.compile(
    r"<think>.*?</think>|<\|begin_of_thought\|>.*?<\|end_of_thought\|>|"
    r"<analysis>.*?</analysis>", re.I | re.S,
)
ANSWER_TAG = re.compile(r"<answer>|</answer>|<\|begin_of_solution\|>", re.I)
OPTION_LINE = re.compile(
    r"(?m)^\s*[\(（\[]?([A-Ha-h])[\)）\]]?[\.．、:：]\s*"
)
OPTION_INLINE = re.compile(r"(?<![A-Za-z])\(([A-Ha-h])\)")
EXPLICIT_MULTI = re.compile(
    r"多项选择|多选题|可多选|多答案|不定项|选择所有|select\s+all|"
    r"all\s+that\s+apply|multiple\s+answers?", re.I,
)
ANSWER_SEQUENCES = [
    re.compile(
        r"(?:正确答案|参考答案|答案|正确选项|应选|故选)\s*"
        r"(?:是|为|选|:|：)?\s*\**[\(（\[]?\s*"
        r"([A-H](?:(?:\s*[,，、/和及]\s*|\s*)[A-H])*)", re.I,
    ),
    re.compile(
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\**[\(\[]?\s*"
        r"([A-H](?:(?:\s*[,，、/&]\s*|\s*)[A-H])*)", re.I,
    ),
    re.compile(r"\\boxed\{\s*\(?([A-H](?:\s*[,，、/]\s*[A-H])*)\)?\s*\}", re.I),
]
MEDICAL = re.compile(
    r"医|药|病|症|治疗|诊断|患者|临床|手术|护理|卫生|细胞|解剖|"
    r"激素|抗体|疫苗|感染|血液|心脏|肝|肾|肺|牙|妊娠|孕|婴儿|"
    r"营养素|维生素|health|medical|patient|disease|treatment", re.I,
)
MATH = re.compile(
    r"方程|函数|几何|三角形|概率|数列|整数|分数|矩阵|导数|积分|"
    r"计算|数学|多少|几只|米|千米|厘米|公斤|kg|cm|\d\s*[+\-*/=^]|math", re.I,
)
LAW = re.compile(r"法[律规]|刑法|民法|宪法|条例|行政部门|法院|检察|诉讼|合同|侵权|犯罪|执业|注册")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", type=Path, default=Path("/tmp/o45_full_eda_repro.json"))
    p.add_argument(
        "--candidate-audit", type=Path,
        default=Path("/tmp/o5_world_candidates_strict_repro.jsonl"),
    )
    return p.parse_args()


def text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := text(item)))
    if isinstance(value, dict):
        if "text" in value:
            return text(value["text"])
        if value.get("type") == "text" and "content" in value:
            return text(value["content"])
        return "\n".join(
            part for key, item in value.items()
            if key not in {"type", "role"} and (part := text(item))
        )
    return str(value)


def language(value: str) -> str:
    c = Counter()
    for char in value:
        code = ord(char)
        if 0x3400 <= code <= 0x9FFF or 0x20000 <= code <= 0x3134F:
            c["han"] += 1
        elif 0x3040 <= code <= 0x30FF:
            c["kana"] += 1
        elif 0xAC00 <= code <= 0xD7AF:
            c["hangul"] += 1
        elif char.isascii() and char.isalpha():
            c["latin"] += 1
        elif 0x0400 <= code <= 0x052F:
            c["cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF:
            c["arabic"] += 1
        elif char.isalpha() and not unicodedata.category(char).startswith("M"):
            c["other"] += 1
    letters = sum(c.values())
    if not letters:
        return "empty_or_symbolic"
    if c["kana"] / letters >= 0.03:
        return "ja"
    if c["hangul"] / letters >= 0.10:
        return "ko"
    if c["han"] + c["latin"]:
        ratio = c["han"] / (c["han"] + c["latin"])
        if c["han"] >= 8 and ratio >= 0.60:
            return "zh"
        if c["han"] >= 8 and 0.10 < ratio < 0.60:
            return "zh_en_mixed"
        if c["latin"] >= 12 and ratio <= 0.10:
            return "en"
    return max(("cyrillic", "arabic", "other"), key=lambda key: c[key])


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        char for char in value
        if not (char.isspace() or unicodedata.category(char)[0] in "PZ")
    )


def digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8", "surrogatepass"), digest_size=16).digest()


def option_markers(prompt: str, include_inline: bool = True) -> list[str]:
    markers = [m.upper() for m in OPTION_LINE.findall(prompt)]
    if include_inline and len(set(markers)) < 3:
        inline = [m.upper() for m in OPTION_INLINE.findall(prompt)]
        if len(set(inline)) >= 3:
            markers = inline
    return markers


def final_answer_sequence(response: str) -> tuple[str | None, str | None]:
    outside = THINK_SPAN.sub(" ", response)
    hits = []
    for pattern in ANSWER_SEQUENCES:
        for match in pattern.finditer(outside[-5000:]):
            letters = "".join(re.findall(r"[A-H]", match.group(1).upper()))
            if letters:
                hits.append((match.start(), letters, match.group(0)[:160]))
    if hits:
        _, letters, evidence = sorted(hits)[-1]
        return letters, evidence
    clean = re.sub(r"</?answer>", " ", outside, flags=re.I).strip()
    match = re.fullmatch(r"[\(（\[]?\s*([A-H])\s*[\)）\]]?[\.。]?", clean, re.I)
    return (match.group(1).upper(), clean) if match else (None, None)


def worker(job):
    asset_id, path = job
    per = defaultdict(Counter)
    roles = Counter()
    message_counts = Counter()
    candidates = []
    prompt_hash = bytearray()
    norm_hash = bytearray()
    message_hash = bytearray()
    uuid_hash = bytearray()
    source_id = bytearray()
    lengths = array("I")
    parquet = pq.ParquetFile(path)
    columns = [
        name for name in ("source", "uuid", "messages", "metadata", "text", "label")
        if name in parquet.schema_arrow.names
    ]
    for batch in parquet.iter_batches(columns=columns, batch_size=1024):
        for row in batch.to_pylist():
            source = str(row.get("source"))
            count = per[source]
            count["rows"] += 1
            if row.get("text") is None:
                count["text_null"] += 1
            elif row.get("text") == "":
                count["text_empty"] += 1
            if row.get("label") is not None:
                count["label_nonnull"] += 1
            metadata = row.get("metadata")
            if metadata is None or metadata == "null":
                count["metadata_null"] += 1
            elif isinstance(metadata, str):
                try:
                    json.loads(metadata)
                except Exception:
                    count["metadata_invalid"] += 1
            raw = row.get("messages")
            try:
                messages = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                count["json_invalid"] += 1
                continue
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                count["messages_shape_invalid"] += 1
                continue
            sequence = tuple(str(m.get("role", "<MISSING>")) for m in messages)
            roles[(source, "|".join(sequence))] += 1
            message_counts[(source, len(messages))] += 1
            by_role = defaultdict(list)
            for message in messages:
                by_role[str(message.get("role", "<MISSING>"))].append(text(message.get("content")))
            system = "\n".join(by_role.get("system", []))
            user = "\n".join(by_role.get("user", []))
            assistant = "\n".join(by_role.get("assistant", []))
            prompt = "\n".join(part for part in (system, user) if part)
            if not user:
                count["empty_user"] += 1
            if not assistant:
                count["empty_assistant"] += 1
            if len(by_role.get("user", [])) > 1:
                count["multi_user"] += 1
            count["language_" + language(user)] += 1
            opened = bool(THINK_OPEN.search(assistant))
            closed = bool(THINK_CLOSE.search(assistant))
            count["think_open"] += opened
            count["think_close"] += closed
            count["think_unclosed"] += opened and not closed
            count["think_close_only"] += closed and not opened
            count["answer_tag"] += bool(ANSWER_TAG.search(assistant))
            if closed and not re.sub(r"</?answer>", "", THINK_CLOSE.split(assistant)[-1], flags=re.I).strip():
                count["empty_after_think"] += 1
            broad_markers = option_markers(user, include_inline=True)
            broad_mc = len(set(broad_markers)) >= 3
            count["mc_broad"] += broad_mc
            lang = language(user)
            count["zh_mc_broad"] += broad_mc and lang in {"zh", "zh_en_mixed"}
            # Conservative world-compatible funnel: exactly four line options,
            # a single final letter, Chinese/mixed, and no explicit multi-select cue.
            line_markers = option_markers(user, include_inline=False)
            exact_abcd = set(line_markers) == set("ABCD") and all(
                line_markers.count(letter) == 1 for letter in "ABCD"
            )
            count["line_exact_abcd"] += exact_abcd
            exact_zh = exact_abcd and lang in {"zh", "zh_en_mixed"}
            count["line_exact_abcd_zh"] += exact_zh
            answer, evidence = final_answer_sequence(assistant)
            count["line_exact_zh_has_final"] += exact_zh and answer is not None
            count["line_exact_zh_multi_final"] += exact_zh and answer is not None and len(answer) > 1
            compatible = (
                asset_id == "O5" and exact_zh and answer is not None
                and len(answer) == 1 and not EXPLICIT_MULTI.search(user)
            )
            if compatible:
                family = (
                    "medical" if source == "medical-o1-reasoning-SFT-think" or MEDICAL.search(user)
                    else "math" if MATH.search(user)
                    else "law" if LAW.search(user)
                    else "other"
                )
                count["world_compatible"] += 1
                count["world_family_" + family] += 1
                candidates.append({
                    "source": source, "uuid": row.get("uuid"), "family": family,
                    "answer": answer, "answer_evidence": evidence, "prompt": user,
                    "response_tail": assistant[-2500:], "metadata": metadata,
                })
            prompt_hash.extend(digest(prompt))
            norm_hash.extend(digest(normalize(prompt)))
            canonical_raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, sort_keys=True)
            message_hash.extend(digest(canonical_raw))
            uuid_hash.extend(digest(str(row.get("uuid"))))
            source_id.append(SRC_ID[source])
            lengths.extend((len(prompt), len(assistant), len(prompt) + len(assistant)))
    return {
        "asset": asset_id,
        "per": {source: dict(count) for source, count in per.items()},
        "roles": dict(roles), "message_counts": dict(message_counts),
        "prompt_hash": bytes(prompt_hash), "norm_hash": bytes(norm_hash),
        "message_hash": bytes(message_hash), "uuid_hash": bytes(uuid_hash),
        "source_id": bytes(source_id), "lengths": lengths.tobytes(),
        "candidates": candidates,
    }


def duplicate_summary(values):
    unique, counts = np.unique(values, return_counts=True)
    return {
        "rows": int(len(values)), "unique": int(len(unique)),
        "duplicate_rows": int(len(values) - len(unique)),
        "duplicate_groups": int((counts > 1).sum()), "max_group": int(counts.max()),
    }


def overlap(left, right):
    lu, lc = np.unique(left, return_counts=True)
    ru, rc = np.unique(right, return_counts=True)
    common, li, ri = np.intersect1d(lu, ru, assume_unique=True, return_indices=True)
    return {
        "common_unique": int(len(common)),
        "rows_O4_in_common": int(lc[li].sum()),
        "rows_O5_in_common": int(rc[ri].sum()),
    }


def main():
    config = args()
    official_root = (ROOT / "assets/official").resolve()
    for output in (config.output.resolve(), config.candidate_audit.resolve()):
        if output.is_relative_to(official_root):
            raise ValueError(f"refusing to write inside official assets: {output}")
    jobs = [
        (asset, path) for asset, root in ASSETS.items()
        for path in sorted(root.glob("*.parquet"))
    ]
    per = {asset: defaultdict(Counter) for asset in ASSETS}
    roles = {asset: Counter() for asset in ASSETS}
    message_counts = {asset: Counter() for asset in ASSETS}
    buffers = {
        asset: {key: [] for key in ("prompt_hash", "norm_hash", "message_hash", "uuid_hash", "source_id", "lengths")}
        for asset in ASSETS
    }
    candidates = []
    with cf.ProcessPoolExecutor(max_workers=config.workers) as executor:
        for record in executor.map(worker, jobs, chunksize=1):
            asset = record["asset"]
            for source, count in record["per"].items():
                per[asset][source].update(count)
            roles[asset].update(record["roles"])
            message_counts[asset].update(record["message_counts"])
            for key in buffers[asset]:
                buffers[asset][key].append(record[key])
            candidates.extend(record["candidates"])
    arrays = {}
    for asset in ASSETS:
        arrays[asset] = {
            key: np.frombuffer(b"".join(buffers[asset][key]), dtype="S16")
            for key in ("prompt_hash", "norm_hash", "message_hash", "uuid_hash")
        }
        arrays[asset]["source_id"] = np.frombuffer(b"".join(buffers[asset]["source_id"]), dtype=np.uint8)
        arrays[asset]["lengths"] = np.frombuffer(
            b"".join(buffers[asset]["lengths"]), dtype=np.uint32
        ).reshape(-1, 3)
    result = {
        "method": {
            "scope": "all rows in every registered top-level O4/O5 Parquet",
            "normalization": "NFKC + casefold + remove Unicode whitespace/separators/punctuation",
            "fingerprint": "BLAKE2b-128",
            "note": "all counts exact under stated deterministic heuristics; token lengths remain sampled in scripts/data/eda_general_official.py",
        },
        "assets": {}, "cross_asset": {}, "visible_world_first5_overlap": {},
    }
    for asset, root in ASSETS.items():
        lens = arrays[asset]["lengths"]
        result["assets"][asset] = {
            "file_count": len(list(root.glob("*.parquet"))),
            "rows": int(len(lens)), "bytes": sum(path.stat().st_size for path in root.glob("*.parquet")),
            "per_source": {source: dict(count) for source, count in sorted(per[asset].items())},
            "role_patterns": {
                source: {pattern: count for (src, pattern), count in roles[asset].items() if src == source}
                for source in per[asset]
            },
            "message_counts": {
                source: {str(n): count for (src, n), count in message_counts[asset].items() if src == source}
                for source in per[asset]
            },
            "char_quantiles": {
                name: {str(q): int(value) for q, value in zip(
                    (0, .01, .1, .5, .9, .95, .99, 1),
                    np.quantile(lens[:, index], (0, .01, .1, .5, .9, .95, .99, 1), method="nearest"),
                )}
                for index, name in enumerate(("prompt", "response", "total"))
            },
            "duplicates": {
                key: duplicate_summary(arrays[asset][key])
                for key in ("prompt_hash", "norm_hash", "message_hash", "uuid_hash")
            },
            "duplicates_by_source": {
                key: {
                    source: duplicate_summary(arrays[asset][key][arrays[asset]["source_id"] == source_id])
                    for source, source_id in SRC_ID.items()
                    if np.any(arrays[asset]["source_id"] == source_id)
                }
                for key in ("prompt_hash", "norm_hash", "message_hash")
            },
        }
    for key in ("prompt_hash", "norm_hash", "message_hash", "uuid_hash"):
        result["cross_asset"][key] = overlap(arrays["O4"][key], arrays["O5"][key])
    visible = []
    path = ROOT / "assets/evaluation/visible/懂世界.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines()[:5]:
        row = json.loads(line)
        row = row[0] if isinstance(row, list) else row
        full = row["prompt"]
        stem = re.sub(r"^请回答以下问题：\s*", "", full)
        stem = re.sub(r"\s*请按以下格式作答：.*$", "", stem, flags=re.S)
        visible.append((full, stem))
    for index, (full, stem) in enumerate(visible, 1):
        row = {}
        for name, value in (("full", full), ("stem", stem)):
            for mode, transform, field in (
                ("exact", lambda x: x, "prompt_hash"),
                ("normalized", normalize, "norm_hash"),
            ):
                target = np.array([digest(transform(value))], dtype="S16")
                for asset in ASSETS:
                    row[f"{name}_{mode}_{asset}"] = int(np.isin(arrays[asset][field], target).sum())
        result["visible_world_first5_overlap"][str(index)] = row
    result["strict_world_candidate_funnel"] = {
        "rules": [
            "Chinese/mixed user text by Unicode-script heuristic",
            "exactly one A/B/C/D line marker each and no E-H",
            "mechanically parsed final answer is exactly one A-D letter",
            "no explicit multi-select instruction",
            "subject labels are heuristic and not answer verification",
        ],
        "rows": len(candidates),
        "unique_prompt_exact": len({row["prompt"] for row in candidates}),
        "unique_prompt_normalized": len({normalize(row["prompt"]) for row in candidates}),
        "by_source_family": dict(Counter(
            f"{row['source']}|{row['family']}" for row in candidates
        )),
    }
    config.candidate_audit.parent.mkdir(parents=True, exist_ok=True)
    with config.candidate_audit.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["candidate_audit_path"] = str(config.candidate_audit)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
