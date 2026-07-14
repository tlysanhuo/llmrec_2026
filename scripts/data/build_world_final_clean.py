#!/usr/bin/env python3
"""build_world_final_clean.py — 清洗队友『懂世界final.jsonl』(747条,2026-07-07 用户转交)。

队友管线:官方eval标准单选 → 0.8B过lm_eval筛答错题(硬例挖掘) → DeepSeek补CoT。
QC发现的病(逐项对治):
  ① 111/747 CoT被"质检任务"污染(is_valid/options_seen/分析原文…)——同wmc那批teacher元叙述,剥think
  ② 22 条 think结论与最终答案不一致——整行剔(gold可信度存疑)
  ③ 747/747 答案无括号 + 空行格式≠riders约定——统一为 `<think>\\n\\n</think>\\n\\n\\n正确答案是 (X)`
  ④ 107 条缺 /think 尾缀、48 条选项 `(A)` 风格≠评测模板 `A.` ——修复
  ⑤ 399/747 题已在 riders 底盘内(硬例挖掘的复挖)——剔(重复上采样=毒药铁律)
产出(双格式,照 Frinkleko 懂世界约定):
  nothink 版:全部存活题,riders 字节级格式(评测通路=nothink直出)
  think 版:CoT干净的存活题,尾缀 /think(可选混入,采样时再定量)
用法: python scripts/data/build_world_final_clean.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "assets" / "third_party" / "teammate" / "懂世界final.jsonl"
OUT_NOTHINK = ROOT / "assets" / "derived" / "processed" / "world_final_nothink.jsonl"
OUT_THINK = ROOT / "assets" / "derived" / "processed" / "world_final_think.jsonl"
RIDERS = ROOT / "assets" / "derived" / "processed" / "data_riders_fk.jsonl"

CONTAM = re.compile(r"is_valid|is_multi|options_seen|answer_letter|自包含|分析原文|是否合法|json|JSON")
SYS = "你是一个非常聪明的助手，请直接遵循指示作答。"


def qkey(s):
    return re.sub(r"\s+", "", s)[:120]


def norm_options(prompt):
    # `(A) xx` 风格 → `A.xx`(对齐评测模板)
    return re.sub(r"\(([A-D])\)\s*", r"\1.", prompt)


def main():
    rows = [json.loads(l)[0] for l in open(SRC)]
    assert len(rows) == 747

    riders_q = set()
    for l in open(RIDERS):
        r = json.loads(l)
        if "正确答案是" in r["output"]:
            riders_q.add(qkey(r.get("input") or ""))

    stats = {"contam_strip": 0, "incons_drop": 0, "dup_riders_drop": 0, "opt_fix": 0}
    nothink_out, think_out = [], []
    for r in rows:
        assert r["system"] == SYS
        m = re.match(r"^<think>(.*?)</think>(.*)$", r["response"], flags=re.S)
        think_body, tail = (m.group(1), m.group(2)) if m else ("", r["response"])
        ans = re.search(r"正确答案是\s*\(?([A-D])\)?", tail)
        if not ans:
            continue
        a = ans.group(1)
        # ② think结论 vs 答案一致性
        if think_body.strip():
            concl = re.findall(r"(?:答案|选)[^A-D]{0,8}([A-D])", think_body)
            if concl and concl[-1] != a:
                stats["incons_drop"] += 1
                continue
        prompt = re.sub(r"\s*/(no_)?think\s*$", "", r["prompt"].rstrip())
        if "\nA." not in prompt:
            prompt = norm_options(prompt)
            stats["opt_fix"] += 1
        if len(re.findall(r"\n[A-D]\.", prompt)) != 4:
            continue  # 修完仍不是4选项,弃
        # ⑤ 与 riders 已有题去重(注意:riders 内 prompt 在 input 列,含同款模板文本)
        if any(qkey(prompt + s) in riders_q for s in ("/no_think", "")) or qkey(prompt + "/no_think") in riders_q or qkey(prompt) in riders_q:
            stats["dup_riders_drop"] += 1
            continue
        gold = f"正确答案是 ({a})"
        nothink_out.append({
            "instruction": SYS,
            "input": prompt + "/no_think",
            "output": f"<think>\n\n</think>\n\n\n{gold}",
            "history": [],
        })
        # think 版:仅 CoT 干净的
        if think_body.strip():
            if CONTAM.search(think_body):
                stats["contam_strip"] += 1
            else:
                think_out.append({
                    "instruction": SYS,
                    "input": prompt + "/think",
                    "output": f"<think>{think_body}</think>\n{gold}",
                    "history": [],
                })

    for path, data in [(OUT_NOTHINK, nothink_out), (OUT_THINK, think_out)]:
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] nothink {len(nothink_out)} → {OUT_NOTHINK.name}; think {len(think_out)} → {OUT_THINK.name}")
    print(f"[QC] {stats}")


if __name__ == "__main__":
    main()
