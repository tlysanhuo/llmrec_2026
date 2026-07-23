#!/usr/bin/env python3
"""Build the E-clean O1 video material pool for I-35 beam precomputation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/official/seed_sft/懂物料part4.jsonl"
OUTPUT = ROOT / "logs/data/i35_video_material_beam128_pool_v1.jsonl"
DEV_OUTPUT = ROOT / "logs/data/i35_video_material_beam128_pool_v1_dev.jsonl"
AUDIT = ROOT / "logs/data/i35_video_material_beam128_pool_v1_audit.json"
SOURCE_ROWS = 1621
SOURCE_SHA256 = "0a35f02b229e6b8e0d7e884a65bf12003d899f99914de277f88b1978959deccc"
SEED = 19260835
OFFICIAL_SYSTEM = "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。"
OFFICIAL_USER_PREFIX = "请解析以下视频内容并输出对应的视频token：\n\n"
SID_RE = re.compile(
    r"^<\|video_begin\|><s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(row.get("instruction", row.get("system", "")) or ""),
        "input": str(row.get("input", row.get("prompt", "")) or ""),
        "output": str(row.get("output", row.get("response", "")) or ""),
        "history": row.get("history") or [],
    }


def prompt_digest(row: dict[str, Any]) -> str:
    value = normalized(row)
    return digest([value["instruction"], value["input"], value["history"]])


def mode_prompt_digest(row: dict[str, Any]) -> str:
    value = normalized(row)
    prompt = re.sub(r"/(?:no_)?think\s*$", "", value["input"].rstrip())
    return digest([value["instruction"], prompt, value["history"]])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            values = value if isinstance(value, list) else [value]
            if len(values) != 1 or not isinstance(values[0], dict):
                raise RuntimeError(f"expected one object at {path}:{line_number}")
            rows.append(values[0])
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_e_paths() -> tuple[Path, ...]:
    helper_path = ROOT / "scripts/data/build_i34_material_beam_pool_v1.py"
    spec = importlib.util.spec_from_file_location("llmrec_i35_e_manifest", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    paths = list(helper.E_PATHS)
    paths.append(ROOT / "assets/evaluation/holdout/data_i34_material_beam_dev_v1.jsonl")
    resolved: list[Path] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"registered E asset is missing: {path}")
        resolved.append(path)
    if len({path.resolve() for path in resolved}) != len(resolved):
        raise RuntimeError("duplicate E path in I-35 manifest")
    return tuple(resolved)


def convert_source_row(raw: dict[str, Any], source_line: int) -> tuple[dict[str, Any], str]:
    row = normalized(raw)
    if row["history"] != [] or not row["instruction"] or not row["input"]:
        raise RuntimeError(f"unexpected O1 video schema at source line {source_line}")
    mode_match = re.search(r"/(think|no_think)\s*$", row["input"])
    if mode_match is None:
        raise RuntimeError(f"missing mode suffix at source line {source_line}")
    source_mode = mode_match.group(1)
    source_prompt_hash = prompt_digest(row)
    source_mode_prompt_hash = mode_prompt_digest(row)
    source_prompt = row["input"][: mode_match.start()]
    if "：" not in source_prompt:
        raise RuntimeError(f"source prompt has no description delimiter at line {source_line}")
    source_prefix, description = source_prompt.split("：", 1)
    if not source_prefix or not description.strip():
        raise RuntimeError(f"source prompt has an empty template/description at line {source_line}")
    row["input"] = OFFICIAL_USER_PREFIX + description + "/no_think"
    if "</think>" not in row["output"]:
        raise RuntimeError(f"missing think delimiter at source line {source_line}")
    body = row["output"].rsplit("</think>", 1)[1].strip()
    match = SID_RE.fullmatch(body)
    if match is None:
        raise RuntimeError(f"invalid video SID at source line {source_line}: {body[:120]!r}")
    row["output"] = f"<think>\n\n</think>\n{body}"
    source_instruction = row["instruction"]
    row["instruction"] = OFFICIAL_SYSTEM
    core_hash = digest(row)
    return {
        **row,
        "route": "beam_train_pool",
        "task": "material_desc2sid",
        "source_asset_id": "O1.懂物料part4",
        "source_line": source_line,
        "source_mode": source_mode,
        "source_instruction": source_instruction,
        "source_user_prefix": source_prefix + "：",
        "source_row_sha256": digest(raw),
        "source_prompt_sha256": source_prompt_hash,
        "source_mode_prompt_sha256": source_mode_prompt_hash,
        "row_sha256": core_hash,
        "prompt_sha256": prompt_digest(row),
        "mode_prompt_sha256": mode_prompt_digest(row),
        "gold_sid": body,
        "gold_domain": "video",
        "gold_s_a": int(match.group("a")),
        "gold_s_b": int(match.group("b")),
        "gold_s_c": int(match.group("c")),
        "prefix_group": f"video:{match.group('a')}:{match.group('b')}",
    }, source_mode


def build() -> dict[str, Any]:
    for path in (OUTPUT, DEV_OUTPUT, AUDIT):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("O1 video source hash drifted")
    raw_rows = load_jsonl(SOURCE)
    if len(raw_rows) != SOURCE_ROWS:
        raise RuntimeError(f"O1 video row count drifted: {len(raw_rows)}")

    e_paths = load_e_paths()
    e_rows = [row for path in e_paths for row in load_jsonl(path)]
    e_prompt = {prompt_digest(row) for row in e_rows}
    e_mode = {mode_prompt_digest(row) for row in e_rows}

    source_modes: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_sids: set[str] = set()
    for source_line, raw in enumerate(raw_rows, 1):
        row, source_mode = convert_source_row(raw, source_line)
        source_modes[source_mode] += 1
        exact = prompt_digest(row)
        mode = mode_prompt_digest(row)
        if (
            exact in e_prompt
            or mode in e_mode
            or row["source_prompt_sha256"] in e_prompt
            or row["source_mode_prompt_sha256"] in e_mode
        ):
            excluded["registered_E_exact_or_mode"] += 1
            continue
        if exact in seen_prompts or row["gold_sid"] in seen_sids:
            raise RuntimeError(f"duplicate eligible prompt or SID at source line {source_line}")
        seen_prompts.add(exact)
        seen_sids.add(row["gold_sid"])
        rows.append(row)

    if source_modes != Counter({"no_think": 788, "think": 833}):
        raise RuntimeError(f"O1 video mode counts drifted: {dict(source_modes)}")
    if not rows:
        raise RuntimeError("I-35 E-clean material pool is empty")

    # The reused beam runner requires a second pool.  Keep one deterministic
    # row separate, then recombine both ledgers in the formal builder.
    dev_index = min(range(len(rows)), key=lambda index: digest([SEED, rows[index]["row_sha256"]]))
    dev = [dict(rows[dev_index], route="beam_gate_pool")]
    train = rows[:dev_index] + rows[dev_index + 1 :]
    atomic_jsonl(OUTPUT, train)
    atomic_jsonl(DEV_OUTPUT, dev)
    audit = {
        "schema_version": "i35-video-material-beam-pool-v1",
        "seed": SEED,
        "source": {
            "asset_id": "O1.懂物料part4",
            "path": str(SOURCE.relative_to(ROOT)),
            "rows": len(raw_rows),
            "sha256": SOURCE_SHA256,
            "modes": dict(sorted(source_modes.items())),
        },
        "conversion": {
            "think_to_empty_no_think": source_modes["think"],
            "native_no_think_retained": source_modes["no_think"],
            "fixed_official_system": OFFICIAL_SYSTEM,
            "fixed_official_user_prefix": OFFICIAL_USER_PREFIX,
            "description_and_gold_sid_unchanged": True,
        },
        "e_manifest": [
            {"path": str(path.relative_to(ROOT)), "rows": len(load_jsonl(path)), "sha256": sha256(path)}
            for path in e_paths
        ],
        "excluded": dict(sorted(excluded.items())),
        "eligible_rows": len(rows),
        "unique_prompts": len(seen_prompts),
        "unique_full_sids": len(seen_sids),
        "outputs": {
            "train_pool": {"path": str(OUTPUT.relative_to(ROOT)), "rows": len(train)},
            "dev_pool": {"path": str(DEV_OUTPUT.relative_to(ROOT)), "rows": len(dev)},
        },
    }
    audit["outputs"]["train_pool"]["sha256"] = sha256(OUTPUT)
    audit["outputs"]["dev_pool"]["sha256"] = sha256(DEV_OUTPUT)
    atomic_json(AUDIT, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return audit


def self_test() -> None:
    raw = {
        "system": "system",
        "prompt": "source template：description/think",
        "response": "<think>reason</think>\n<|video_begin|><s_a_1><s_b_2><s_c_3>",
    }
    row, mode = convert_source_row(raw, 1)
    assert mode == "think" and row["input"] == OFFICIAL_USER_PREFIX + "description/no_think"
    assert row["instruction"] == OFFICIAL_SYSTEM
    assert row["output"] == "<think>\n\n</think>\n<|video_begin|><s_a_1><s_b_2><s_c_3>"
    assert row["source_mode_prompt_sha256"] == mode_prompt_digest(raw)
    assert mode_prompt_digest(raw) != mode_prompt_digest(row)
    print("i35 video material pool self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        build()


if __name__ == "__main__":
    main()
