#!/usr/bin/env python3
"""Build data_quality_swap_v1.

Equal-size replacement over riders_fk_lora_ep1:
  - remove old Frinkleko user action/topic rows
  - add official-shape U1 action-select, U2 chain MC, U3 topic-chain rows
  - keep world_zh/P3/world_mc_clean exactly as riders_fk_lora_ep1
  - trim longest remaining FK CoT rows to preserve the 37,267-row budget
"""

import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026")
P = ROOT / "data/processed"
OUT = P / "data_quality_swap_v1.jsonl"
TARGET_N = 37267
SEED = 2026
KEEP_KEYS = ("instruction", "input", "output", "history")
BAD_TRIPLE_RE = re.compile(r"<s_a_\d+>(?!<s_b_\d+><s_c_\d+>)")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if isinstance(r, list):
                r = r[0]
            if "prompt" in r and "instruction" not in r:
                r = {
                    "instruction": r.get("system", ""),
                    "input": r["prompt"],
                    "output": r["response"],
                    "history": r.get("history", []),
                }
            rows.append(r)
    return rows


def sanitize(r: dict) -> dict:
    out = {k: r[k] for k in KEEP_KEYS if k in r}
    out.setdefault("instruction", "")
    out.setdefault("input", "")
    out.setdefault("output", "")
    out.setdefault("history", [])
    if out["history"] is None:
        out["history"] = []
    return {k: out[k] for k in KEEP_KEYS}


def row_chars(r: dict) -> int:
    return len(json.dumps(sanitize(r), ensure_ascii=False))


def is_old_action(r: dict) -> bool:
    inp = r.get("input", "") or ""
    return "JSON数组" in inp or "相关的历史交互" in inp


def is_old_topic(r: dict) -> bool:
    s = (r.get("input", "") or "") + "\n" + (r.get("output", "") or "")
    return "logic_chain" in s


def is_old_user_row(r: dict) -> bool:
    return is_old_action(r) or is_old_topic(r)


def domains(r: dict) -> str:
    s = "\n".join(str(r.get(k, "") or "") for k in ("instruction", "input", "output"))
    hits = [d for d in ("video", "prod", "ad", "living") if f"<|{d}_begin|>" in s]
    return "+".join(hits) if hits else "none"


def is_world_mc_like(r: dict) -> bool:
    s = "\n".join(str(r.get(k, "") or "") for k in ("instruction", "input", "output"))
    if domains(r) != "none":
        return False
    return (
        "以下哪" in s
        or "请选择" in s
        or "答案:" in s
        or "选项" in s
        or bool(re.search(r"\b[A-D][\.、]", s))
    )


def coarse_class(r: dict) -> str:
    if is_old_action(r):
        return "old_action"
    if is_old_topic(r):
        return "old_topic"
    if is_world_mc_like(r):
        return "world_mc_like"
    inst = r.get("instruction", "") or ""
    inp = r.get("input", "") or ""
    out = r.get("output", "") or ""
    if "商品描述" in inp or "视频描述" in inp or "广告描述" in inp or "token生成" in inst:
        return "sid_input"
    if "该用户最近喜欢" in out or "请阅读用户历史行为" in inst:
        return "mat_or_rec"
    return "itemic_answer"


def think_body(r: dict) -> str:
    m = THINK_RE.search(r.get("output", "") or "")
    return m.group(1).strip() if m else ""


def load_u1_dedup() -> tuple[list[dict], Counter]:
    files = [
        P / "r2_gold_local.jsonl",
        P / "r2_gold_g1.jsonl",
        P / "r2_gold_g2.jsonl",
        P / "r2_gold_v4.jsonl",
    ]
    by_src = {}
    raw_counts = Counter()
    duplicate_src = 0
    for fp in files:
        for r in load_jsonl(fp):
            raw_counts[fp.name] += 1
            src = r.get("_src_idx")
            key = src if src is not None else ("no_src", fp.name, raw_counts[fp.name])
            duplicate_src += int(key in by_src)
            by_src[key] = sanitize(r)
    raw_counts["raw_total"] = sum(raw_counts.values())
    raw_counts["kept"] = len(by_src)
    raw_counts["duplicate_src"] = duplicate_src
    return list(by_src.values()), raw_counts


def load_world_zh() -> list[dict]:
    rows = [
        sanitize(r)
        for r in load_jsonl(P / "data_rebal_world.jsonl")
        if "/think" not in ((r.get("instruction", "") or "") + (r.get("input", "") or ""))
        and "/no_think" not in ((r.get("instruction", "") or "") + (r.get("input", "") or ""))
    ]
    assert len(rows) == 2824, f"world_zh {len(rows)} != 2824"
    return rows


def select_fk_drops(fk_keep: list[dict], need: int) -> tuple[set[int], list[tuple[int, int, int, str, str]]]:
    candidates = []
    for idx, r in enumerate(fk_keep):
        tb = think_body(r)
        if not tb:
            continue
        if is_world_mc_like(r):
            continue
        candidates.append((row_chars(r), len(tb), idx, coarse_class(r), domains(r)))
    assert len(candidates) >= need, f"drop candidates {len(candidates)} < need {need}"
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    selected = candidates[:need]
    return {idx for _, _, idx, _, _ in selected}, selected


def qc(rows: list[dict]) -> None:
    assert len(rows) == TARGET_N, f"final rows {len(rows)} != {TARGET_N}"
    seen = {}
    dup_examples = []
    for i, r in enumerate(rows):
        assert set(r.keys()) == set(KEEP_KEYS), (i, r.keys())
        assert str(r["output"]).strip(), i
        blob = json.dumps(r, ensure_ascii=False, sort_keys=True)
        if BAD_TRIPLE_RE.search((r["input"] or "") + "\n" + (r["output"] or "")):
            raise AssertionError(f"bad triple token at final row {i}")
        if blob in seen:
            dup_examples.append((seen[blob], i))
        else:
            seen[blob] = i
    if dup_examples:
        raise AssertionError(f"exact duplicate rows: {dup_examples[:5]}")


def main() -> None:
    rng = random.Random(SEED)

    fk_raw = [sanitize(r) for r in load_jsonl(P / "frinkleko_alpaca_32705.jsonl")]
    assert len(fk_raw) == 32705, len(fk_raw)
    old_user = [r for r in fk_raw if is_old_user_row(r)]
    fk_keep = [r for r in fk_raw if not is_old_user_row(r)]

    u1, u1_counts = load_u1_dedup()
    u2 = [sanitize(r) for r in load_jsonl(P / "blocks_v1/block_u2_mc.jsonl")]
    u3 = [sanitize(r) for r in load_jsonl(P / "u3_topic_a.jsonl")] + [
        sanitize(r) for r in load_jsonl(P / "u3_topic_b.jsonl")
    ]
    world = load_world_zh()
    p3 = [sanitize(r) for r in rng.sample(load_jsonl(P / "p3_quote_stop.jsonl"), 1500)]
    wmc = [sanitize(r) for r in load_jsonl(P / "world_mc_clean.jsonl")]
    assert len(u2) == 353, len(u2)
    assert len(u3) == 2372, len(u3)
    assert len(wmc) == 238, len(wmc)

    before_trim_n = len(fk_keep) + len(u1) + len(u2) + len(u3) + len(world) + len(p3) + len(wmc)
    trim_need = before_trim_n - TARGET_N
    assert trim_need > 0, trim_need
    drop_idx, drop_rows = select_fk_drops(fk_keep, trim_need)
    fk_final = [r for i, r in enumerate(fk_keep) if i not in drop_idx]

    final_rows = fk_final + u1 + u2 + u3 + world + p3 + wmc
    rng.shuffle(final_rows)
    qc(final_rows)

    with OUT.open("w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    old_user_chars = sum(row_chars(r) for r in old_user)
    new_user_chars = sum(row_chars(r) for r in (u1 + u2 + u3))
    drop_chars = sum(x[0] for x in drop_rows)

    print("quality_swap_v1 ledger")
    print(f"  fk_raw: {len(fk_raw)}")
    print(
        "  remove old user rows: "
        f"{len(old_user)} = {Counter(coarse_class(r) for r in old_user)}; chars={old_user_chars}"
    )
    print(f"  fk_after_old_user_removal: {len(fk_keep)}")
    print(
        "  add official user rows: "
        f"U1={len(u1)} raw={u1_counts['raw_total']} duplicate_src={u1_counts['duplicate_src']} "
        f"+ U2={len(u2)} + U3={len(u3)} = {len(u1) + len(u2) + len(u3)}; chars={new_user_chars}"
    )
    print(f"  keep riders anchors: world_zh={len(world)} P3={len(p3)} world_mc_clean={len(wmc)}")
    print(f"  before trim: {before_trim_n}; target={TARGET_N}; trim={trim_need}")
    print(f"  trim classes: {Counter(x[3] for x in drop_rows)}")
    print(f"  trim domains: {Counter(x[4] for x in drop_rows)}")
    print(f"  trim chars: {drop_chars}")
    print(f"  final: {len(final_rows)} -> {OUT}")


if __name__ == "__main__":
    main()
