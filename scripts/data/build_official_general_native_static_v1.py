#!/usr/bin/env python3
"""Build a small, domain-balanced native General SFT pool for static knowledge.

This is deliberately not an A-D reconstruction.  It follows the official
``convertv2.py`` contract: native non-empty ``<think>`` responses are retained
and routed with ``/think``; native direct answers are routed with
``/no_think`` and receive the official empty-think prefix.

The source assistant response is official SFT supervision, not independently
verified factual gold.  Evaluation prompts and current-parent prompts are used
only as exclusion indexes.  Official assets are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import build_official_general_world_clean as base  # noqa: E402


PERSONAL_ROOT = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51")
O2_DIR = ROOT / "assets/official/hf_raw/OneReason_General"
TOKENIZER_DIR = ROOT / "assets/official/base_model"
OUT = ROOT / "assets/derived/processed/data_official_general_native_static_v1.jsonl"
LINEAGE_OUT = ROOT / "assets/derived/official_general/official_general_native_static_v1_lineage.jsonl"
AUDIT_OUT = ROOT / "logs/data/official_general_native_static_v1_audit.json"

RULESET_VERSION = "official-general-native-static-20260718-v1"
O2_REVISION = "registry-snapshot-20260717"

QUESTION_CUE = re.compile(
    r"是什么|是指|指什么|何谓|为何|为什么|由谁|是谁|哪个|哪一|哪些|何时|"
    r"有何(?:作用|特点|区别|联系|影响)|有什么(?:作用|特点|区别)|"
    r"主要(?:作用|特点|成分|原因|类型)|基本原理|工作原理|定义|区别|"
    r"包括什么|包括哪些|如何(?:形成|产生|传播|演化|工作)|"
    r"请(?:介绍|解释|说明|简述|概述)"
)

# Reject tasks whose supervision is mostly writing, transformation, personal
# advice, computation, tool operation, or open-ended planning rather than
# stable world knowledge.  The positive ``如何形成/产生/...`` forms above are
# allowed before this gate; unsafe operational forms remain rejected here.
NON_KNOWLEDGE_TASK = re.compile(
    r"写|撰写|生成|创作|润色|改写|翻译|总结|提取|抽取|分类|判断情感|"
    r"制定|设计|推荐|建议|方案|计划|文案|文章|故事|报告|歌词|诗歌|邮件|"
    r"代码|编程|数组|JSON|XML|函数|矩阵|方程|证明|计算|求解|概率|"
    r"逻辑推理|是否蕴含|从.{0,8}(?:选项|类别)中选择|"
    r"如何(?:做|制作|使用|设置|配置|安装|实现|处理|解决|选择|优化|"
    r"提高|提升|避免|管理|操作|准备|学习|开发)|"
    r"有哪些(?:方法|步骤|流程|技巧|工具|软件|框架|资源|措施|建议|注意事项)|"
    r"要求|字数|不少于|至少.{0,8}(?:句|字|项)|模拟|假设|场景|案例"
)

CONTEXT_DEPENDENT = re.compile(
    r"根据(?:以下|上述|上文|材料|文本|描述|信息|对话|图片)|"
    r"请(?:仔细)?阅读|给定|如下|下述|这段|该段|文中|文本中|材料中|"
    r"附件|图中|表中|选项\s*[:：]|[A-D][)）.．、:]"
)

PERSONALIZED = re.compile(
    r"(?:^|[，。！？?\s])(?:我|我们|本人)(?:是|在|有|想|要|需要|正在|最近|今年|家里|公司)|"
    r"帮我|给我|为我|我的|您认为|你认为|你觉得|分享.{0,8}(?:经历|经验)|"
    r"如果你|假如你"
)

FORMAT_OR_POLICY = re.compile(
    r"内容政策|违反政策|系统提示|模型身份|开发公司|输出格式|只(?:能|需)输出|"
    r"不要(?:包含|输出)|必须(?:包含|输出)|角色扮演"
)

ANSWER_BAD = re.compile(
    r"无法(?:确定|回答|核实)|没有提供|未提供|信息不足|没有足够信息|"
    r"不知道|不清楚|不确定|可能是|也许是|抱歉"
)

DOMAINS: dict[str, re.Pattern[str]] = {
    "history_culture": re.compile(
        r"历史|朝代|古代|文明|文化|文学|艺术|哲学|宗教|建筑|节日|语言|"
        r"汉字|诗人|作家|作品|考古|神话|乐器|戏剧|书法|绘画|民俗|遗址|"
        r"王朝|帝国|战争|思想家|学派"
    ),
    "geography_environment": re.compile(
        r"地理|国家|城市|首都|河流|山脉|海洋|湖泊|气候|地貌|大陆|岛屿|"
        r"生态|环境|自然保护|森林|沙漠|冰川|地球|火山|地震|洋流|纬度|"
        r"土壤|水文"
    ),
    "natural_science": re.compile(
        r"物理|化学|生物|天文|地质|元素|分子|原子|细胞|基因|蛋白质|"
        r"光合作用|进化|恒星|行星|矿物|能量|力学|电磁|电磁波|植物|动物|"
        r"微生物|酶|叶绿素|根冠|遗传|生态系统"
    ),
    "computing_technology": re.compile(
        r"计算机|互联网|网络|算法|数据库|操作系统|人工智能|机器学习|芯片|"
        r"通信|加密|密码学|软件|硬件|协议|云计算|编译器|存储器|传感器|"
        r"半导体|机器人|信息安全"
    ),
    "everyday_static": re.compile(
        r"食物|食品|烹饪|材料|交通工具|汽车|飞机|火车|农业|发酵|酿造|"
        r"茶|咖啡|葡萄酒|工具|能源|电池|纺织|陶瓷|玻璃|金属|纸张|"
        r"营养成分|草木灰|混凝剂|燃料"
    ),
}

DOMAIN_PRIORITY = tuple(DOMAINS)
STANDARD_THINK = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o2-dir", type=Path, default=O2_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--lineage-out", type=Path, default=LINEAGE_OUT)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_OUT)
    parser.add_argument("--per-domain-cap", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if not PERSONAL_ROOT.is_mount():
        raise RuntimeError(f"personal volume is not mounted: {PERSONAL_ROOT}")
    if not os.access(PERSONAL_ROOT, os.W_OK):
        raise RuntimeError(f"personal volume is not writable: {PERSONAL_ROOT}")
    if args.o2_dir.resolve() != O2_DIR.resolve():
        raise RuntimeError(f"must use registered O2.General: {O2_DIR}")
    if args.tokenizer.resolve() != TOKENIZER_DIR.resolve():
        raise RuntimeError(f"must use registered O6 tokenizer: {TOKENIZER_DIR}")
    outputs = (args.out, args.lineage_out, args.audit_out)
    if len({path.resolve(strict=False) for path in outputs}) != len(outputs):
        raise RuntimeError("outputs must be distinct")
    for path in outputs:
        target = path.resolve(strict=False)
        if not target.is_relative_to(PERSONAL_ROOT):
            raise RuntimeError(f"output is outside personal volume: {path}")
        if target.is_relative_to((ROOT / "assets/official").resolve()):
            raise RuntimeError(f"refusing to write official asset: {path}")


def primary_domain(prompt: str) -> tuple[str | None, dict[str, int]]:
    hits = {name: len(pattern.findall(prompt)) for name, pattern in DOMAINS.items()}
    maximum = max(hits.values(), default=0)
    if maximum == 0:
        return None, hits
    for name in DOMAIN_PRIORITY:
        if hits[name] == maximum:
            return name, hits
    raise AssertionError("unreachable")


def static_prompt_reasons(prompt: str) -> tuple[list[str], str | None, dict[str, int]]:
    reasons: list[str] = []
    zh_ok, _language = base.is_strict_zh(prompt, min_han=8)
    if not zh_ok:
        reasons.append("prompt_non_zh")
    if not 8 <= len(prompt) <= 180:
        reasons.append("prompt_length")
    if not QUESTION_CUE.search(prompt):
        reasons.append("no_direct_question_cue")
    if NON_KNOWLEDGE_TASK.search(prompt):
        reasons.append("non_knowledge_task")
    if CONTEXT_DEPENDENT.search(prompt):
        reasons.append("context_dependent")
    if PERSONALIZED.search(prompt):
        reasons.append("personalized")
    if FORMAT_OR_POLICY.search(prompt):
        reasons.append("format_or_policy")
    if len(base.broad_option_labels(prompt)) >= 3:
        reasons.append("multiple_choice")
    domain, hits = primary_domain(prompt)
    if domain is None:
        reasons.append("no_static_domain")
    return sorted(set(reasons)), domain, hits


def route_native(prompt: str, assistant: str) -> tuple[str | None, str | None, str, list[str]]:
    reasons: list[str] = []
    if base.MODE_SUFFIX.search(prompt):
        reasons.append("preexisting_mode_suffix")
    final, think_status = base.split_reasoning(assistant)
    if final is None:
        reasons.append(think_status)
        return None, None, think_status, sorted(set(reasons))
    if not 16 <= len(final) <= 1600:
        reasons.append("final_length")
    final_zh, _language = base.is_strict_zh(final, min_han=8)
    if not final_zh:
        reasons.append("final_non_zh")
    if ANSWER_BAD.search(final):
        reasons.append("bad_answer")
    if len(assistant) > 6000:
        reasons.append("assistant_char_limit")
    standard = STANDARD_THINK.search(assistant)
    if standard is not None and standard.group(1).strip():
        mode = "think"
        routed_prompt = prompt.rstrip() + "/think"
        routed_response = assistant
    elif standard is not None:
        mode = "no_think"
        routed_prompt = prompt.rstrip() + "/no_think"
        routed_response = assistant
    elif think_status == "no_think":
        mode = "no_think"
        routed_prompt = prompt.rstrip() + "/no_think"
        routed_response = "<think>\n\n</think>\n" + assistant
    else:
        mode = think_status
        routed_prompt = routed_response = None
        reasons.append("unsupported_official_think_form")
    return routed_prompt, routed_response, mode, sorted(set(reasons))


def stable_rank(record_id: str) -> str:
    return hashlib.sha256((RULESET_VERSION + "\0" + record_id).encode()).hexdigest()


def leakage_match(
    index: base.LeakageIndex,
    prompt: str,
    semantic_gram_sets: list[set[str]] | None,
) -> tuple[bool, list[str]]:
    """Match QA prompts with the same thresholds as ``LeakageIndex.match``.

    Exact checks can run without gram sets during the full scan.  At the final
    release gate, cached gram sets make the length-ratio/Jaccard-0.90 and
    containment checks exhaustive over all length-compatible E prompts.  This
    is fail-closed relative to the shared matcher's top-32 inverted-index
    shortcut.  MC candidates are rejected before this function, so parsed-MC
    branches are intentionally inapplicable here.
    """
    modes: list[str] = []
    raw = base.normalize_raw(prompt)
    mode = base.mode_normalize(prompt)
    core = base.strip_world_wrapper(prompt)
    semantic = base.semantic_normalize(core)
    if base.hash_text(raw) in index.raw_hashes:
        modes.append("raw_exact")
    if base.hash_text(mode) in index.mode_hashes:
        modes.append("mode_exact")
    if semantic and base.hash_text(semantic) in index.semantic_hashes:
        modes.append("core_exact")
    if semantic and base.hash_text(semantic) in index.stem_hashes:
        modes.append("stem_text_exact")
    if not modes and len(semantic) >= 24 and semantic_gram_sets is not None:
        if len(semantic_gram_sets) != len(index.semantic_texts):
            raise AssertionError("cached E gram-set count drift")
        grams = base.char_ngrams(semantic)
        for other_index, other in enumerate(index.semantic_texts):
            length_ratio = min(len(semantic), len(other)) / max(len(semantic), len(other))
            if length_ratio < 0.80:
                continue
            other_grams = semantic_gram_sets[other_index]
            intersection = len(grams & other_grams)
            union = len(grams) + len(other_grams) - intersection
            if union and intersection / union >= 0.90:
                modes.append("near_duplicate")
                break
            if semantic in other or other in semantic:
                modes.append("containment")
                break
    return bool(modes), sorted(set(modes))


def main() -> None:
    args = parse_args()
    ensure_paths(args)
    if args.per_domain_cap <= 0 or args.max_tokens <= 0:
        raise ValueError("caps must be positive")
    files = sorted(args.o2_dir.glob("*.parquet"))
    if len(files) != 158:
        raise AssertionError(f"O2.General shard signature drifted: {len(files)} != 158")

    print("[blacklist] loading E and current-parent prompt exclusions", flush=True)
    eval_index = base.load_eval_index()
    train_index, train_counts = base.load_train_index()
    # Full-scan exclusions need exact hashes only.  Near-duplicate gram sets are
    # built once after scanning and applied only while filling final domain caps.

    stats: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    import pyarrow.parquet as pq

    required = {"uuid", "messages", "source"}
    for file_number, path in enumerate(files, start=1):
        columns = set(pq.ParquetFile(path).schema.names)
        if required - columns:
            raise AssertionError(f"missing columns in {path.name}: {sorted(required - columns)}")
        for row_group, row_index, row in base.parquet_rows(path, ["uuid", "messages", "source"]):
            stats["rows_scanned"] += 1
            parsed = base.parse_messages(row.get("messages"))
            if parsed is None:
                stats["drop:roles_or_text"] += 1
                continue
            _roles, prompt, assistant = parsed
            # O2 contains tens of thousands of very long reasoning responses
            # (individual serialized messages can exceed 300k characters).
            # Reject them before any whole-response risk regex or think parsing;
            # they cannot pass the native-response gate below in any case.
            if len(assistant) > 6000:
                stats["drop:assistant_char_limit"] += 1
                continue
            prompt_reasons, domain, domain_hits = static_prompt_reasons(prompt)
            if prompt_reasons:
                stats.update(f"drop:{reason}" for reason in prompt_reasons)
                continue
            risk = base.risk_reasons(prompt, assistant)
            if risk:
                stats.update(f"drop:{reason}" for reason in risk)
                continue
            eval_hit, eval_modes = leakage_match(eval_index, prompt, None)
            if eval_hit:
                stats["drop:eval_overlap"] += 1
                stats.update(f"drop:eval_{mode}" for mode in eval_modes)
                continue
            train_hit, _train_modes = leakage_match(train_index, prompt, None)
            if train_hit:
                stats["drop:parent_overlap"] += 1
                continue
            routed_prompt, routed_response, mode, response_reasons = route_native(prompt, assistant)
            if response_reasons:
                stats.update(f"drop:{reason}" for reason in response_reasons)
                continue
            assert routed_prompt is not None and routed_response is not None and domain is not None
            record_id, lineage = base.source_locator(
                "O2.General", O2_REVISION, args.o2_dir, path, row_group, row_index,
                str(row.get("source") or ""), row.get("uuid"), row.get("messages"),
            )
            final, _status = base.split_reasoning(assistant)
            assert final is not None
            candidates.append({
                "record_id": record_id,
                "domain": domain,
                "domain_hits": domain_hits,
                "mode": mode,
                "prompt": prompt,
                "assistant": assistant,
                "final_answer": final,
                "training": {
                    "instruction": "",
                    "input": routed_prompt,
                    "output": routed_response,
                    "history": [],
                },
                "lineage": lineage,
                "quality": {
                    "prompt_hash": base.hash_text(base.mode_normalize(prompt)),
                    "selection_rank": stable_rank(record_id),
                },
            })
            stats["pre_dedupe_candidates"] += 1
        if file_number % 20 == 0:
            print(f"[scan] {file_number}/158 candidates={len(candidates)}", flush=True)

    # One supervision per normalized prompt.  Prefer the compact native answer,
    # then use the stable hash only as a deterministic tie-breaker.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["quality"]["prompt_hash"]].append(row)
    deduped: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda row: (len(row["training"]["output"]), row["quality"]["selection_rank"]),
        )
        deduped.append(group[0])
        stats["drop:duplicate_prompt"] += len(group) - 1
    stats["post_dedupe_candidates"] = len(deduped)

    # Render individually but tokenize in batches.  The previous per-record
    # ``apply_chat_template(tokenize=True)`` path spent minutes repeatedly
    # entering the tokenizer; batching preserves the exact rendered text while
    # reducing rebuild time substantially.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    token_kept: list[dict[str, Any]] = []
    for start in range(0, len(deduped), 64):
        batch_rows = deduped[start : start + 64]
        rendered_texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["training"]["input"]},
                    {"role": "assistant", "content": row["training"]["output"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for row in batch_rows
        ]
        encoded = tokenizer(
            rendered_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        if len(encoded) != len(batch_rows):
            raise AssertionError("batched tokenizer row-count drift")
        for row, token_ids in zip(batch_rows, encoded):
            total_tokens = len(token_ids)
            row["quality"]["total_tokens"] = total_tokens
            if total_tokens > args.max_tokens:
                stats["drop:token_limit"] += 1
            else:
                token_kept.append(row)
    deduped = token_kept
    stats["post_token_candidates"] = len(deduped)

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        by_domain[row["domain"]].append(row)
    eval_gram_sets = [base.char_ngrams(text) for text in eval_index.semantic_texts]
    selected: list[dict[str, Any]] = []
    for domain in DOMAIN_PRIORITY:
        rows = sorted(
            by_domain[domain],
            key=lambda row: (row["quality"]["selection_rank"], row["record_id"]),
        )
        selected_domain: list[dict[str, Any]] = []
        near_rejected_domain = 0
        for row in rows:
            eval_hit, eval_modes = leakage_match(eval_index, row["prompt"], eval_gram_sets)
            if eval_hit:
                # Exact hits were already removed; these counters should be
                # near_duplicate/containment unless an invariant drifted.
                stats["drop:eval_overlap_final"] += 1
                stats.update(f"drop:eval_final_{mode}" for mode in eval_modes)
                near_rejected_domain += 1
                continue
            row["quality"]["eval_near_checked"] = True
            selected_domain.append(row)
            if len(selected_domain) == args.per_domain_cap:
                break
        selected.extend(selected_domain)
        stats[f"selected:{domain}"] = len(selected_domain)
        stats[f"drop:domain_cap:{domain}"] = max(
            0, len(rows) - len(selected_domain) - near_rejected_domain
        )
    selected.sort(key=lambda row: (DOMAIN_PRIORITY.index(row["domain"]), row["record_id"]))

    training_rows = [row["training"] for row in selected]
    lineage_rows = [
        {
            "record_id": row["record_id"],
            "domain": row["domain"],
            "mode": row["mode"],
            "lineage": row["lineage"],
            "quality": row["quality"],
            "source_supervision": "official_native_sft_not_independent_factual_gold",
        }
        for row in selected
    ]

    # Final fail-closed invariants.
    assert len(training_rows) == len(lineage_rows) == len(selected)
    assert len({row["quality"]["prompt_hash"] for row in selected}) == len(selected)
    for row in selected:
        training = row["training"]
        assert set(training) == {"instruction", "input", "output", "history"}
        assert training["history"] == [] and training["instruction"] == ""
        if row["mode"] == "think":
            assert training["input"].endswith("/think")
            match = STANDARD_THINK.search(training["output"])
            assert match is not None and match.group(1).strip()
        else:
            assert training["input"].endswith("/no_think")
            assert training["output"].startswith("<think>")
        train_hit, _ = leakage_match(train_index, row["prompt"], None)
        assert row["quality"].get("eval_near_checked") is True
        assert not train_hit
        assert row["quality"]["total_tokens"] <= args.max_tokens

    builder_sha = base.sha256_file(Path(__file__))
    fingerprint = base.hash_text(base.stable_json({
        "builder_sha256": builder_sha,
        "ruleset_version": RULESET_VERSION,
        "o2_revision": O2_REVISION,
        "per_domain_cap": args.per_domain_cap,
        "max_tokens": args.max_tokens,
    }))
    base.atomic_jsonl(args.out, training_rows)
    base.atomic_jsonl(args.lineage_out, lineage_rows)
    audit = {
        "asset_class": "D(O2.General)-native-static-SFT",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ruleset_version": RULESET_VERSION,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": builder_sha,
        "build_fingerprint": fingerprint,
        "official_implementation_contract": {
            "reference": "docs/reference/demo_baseline/convertv2.py",
            "native_nonempty_think": "append /think and retain assistant response byte-for-byte",
            "native_direct_answer": "append /no_think and inject official empty-think prefix",
            "alpaca_mapping": "instruction/input/output/history",
        },
        "upstream": {
            "asset_id": "O2.General",
            "revision": O2_REVISION,
            "path": str(args.o2_dir.resolve()),
            "parquet_files": len(files),
            "rows": stats["rows_scanned"],
        },
        "parameters": {
            "per_domain_cap": args.per_domain_cap,
            "max_tokens": args.max_tokens,
            "tokenizer": str(args.tokenizer.resolve()),
            "selection_seed": "sha256 ruleset+record_id; no RNG",
        },
        "blacklist": {
            "eval_prompt_instances": sum(eval_index.source_counts.values()),
            "eval_sources": dict(sorted(eval_index.source_counts.items())),
            "current_parent_prompt_instances": dict(sorted(train_counts.items())),
            "policy": "prompt text/signatures are exclusion-only; E labels and model outputs never select rows",
        },
        "filter_counts": dict(sorted(stats.items())),
        "selected": {
            "rows": len(selected),
            "domains": dict(sorted(Counter(row["domain"] for row in selected).items())),
            "modes": dict(sorted(Counter(row["mode"] for row in selected).items())),
            "token_stats": {
                "min": min((row["quality"]["total_tokens"] for row in selected), default=0),
                "max": max((row["quality"]["total_tokens"] for row in selected), default=0),
                "mean": (
                    sum(row["quality"]["total_tokens"] for row in selected) / len(selected)
                    if selected else 0
                ),
            },
        },
        "supervision_status": {
            "source_response_role": "official_native_sft_supervision",
            "independently_verified_factual_gold": False,
            "training_format_created": True,
            "formal_training_mix_approved": False,
        },
        "outputs": {},
    }
    for name, path, rows in (
        ("training", args.out, len(training_rows)),
        ("lineage", args.lineage_out, len(lineage_rows)),
    ):
        audit["outputs"][name] = {
            "path": str(path.resolve()),
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
    base.atomic_json(args.audit_out, audit)
    print(json.dumps(audit["selected"], ensure_ascii=False, sort_keys=True), flush=True)
    print(f"[OK] training: {args.out}", flush=True)
    print(f"[OK] lineage: {args.lineage_out}", flush=True)
    print(f"[OK] audit: {args.audit_out}", flush=True)


if __name__ == "__main__":
    main()
