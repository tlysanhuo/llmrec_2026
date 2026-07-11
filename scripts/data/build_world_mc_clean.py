#!/usr/bin/env python3
"""build_world_mc_clean.py — 清洗队友的 懂世界_from_mc.jsonl(2026-07-05)。

输入: llmrec_2026/懂世界_from_mc.jsonl(272条,队友从评测集样例+公开案例风格蒸馏生成)
缺陷: ①268/272 think含teacher元叙述(JSON转换指令泄漏) ②34条答案越界(H/BCD多选)
      ③prompt无/no_think尾缀但think为filled(破坏全库不变量,评测强制空think)
清洗: 剥掉脏think → 逐字节对齐 Frinkleko/评测已证格式:
      prompt 补 '/no_think' 尾缀;response = '<think>\n\n</think>\n\n\n正确答案是 (X)';
      仅保留单选 A-D;prompt 级去重;剔除与评测5真题(懂世界.jsonl前5条)重复的题。
输出: data/processed/world_mc_clean.jsonl
用法: python scripts/data/build_world_mc_clean.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "懂世界_from_mc.jsonl"
EVAL5 = ROOT / "懂世界.jsonl"
DST = ROOT / "data" / "processed" / "world_mc_clean.jsonl"

ANS = re.compile(r"正确答案是\s*\(?([A-Z]+)\)?\s*$")
SYS = "你是一个非常聪明的助手，请直接遵循指示作答。"

def unwrap(x):
    return x[0] if isinstance(x, list) else x

def main():
    eval5_prompts = set()
    for line in open(EVAL5, encoding="utf-8"):
        d = unwrap(json.loads(line))
        eval5_prompts.add(re.sub(r"\s+", "", d["prompt"]))

    seen, out = set(), []
    dropped = {"越界答案": 0, "无答案": 0, "重复": 0, "撞评测真题": 0}
    for line in open(SRC, encoding="utf-8"):
        d = unwrap(json.loads(line))
        m = ANS.search(d["response"].strip())
        if not m:
            dropped["无答案"] += 1
            continue
        letter = m.group(1)
        if letter not in ("A", "B", "C", "D"):
            dropped["越界答案"] += 1
            continue
        prompt = d["prompt"].rstrip()
        if not prompt.endswith(("/think", "/no_think")):
            prompt = prompt + "/no_think"
        key = re.sub(r"\s+", "", prompt)
        if key in seen:
            dropped["重复"] += 1
            continue
        if re.sub(r"\s+", "", d["prompt"]) in eval5_prompts or key in eval5_prompts:
            dropped["撞评测真题"] += 1
            continue
        seen.add(key)
        out.append({"system": d.get("system") or SYS, "prompt": prompt,
                    "response": f"<think>\n\n</think>\n\n\n正确答案是 ({letter})"})

    with open(DST, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps([r], ensure_ascii=False) + "\n")
    print(f"[OK] {DST}: {len(out)} 条(平台种子格式 list[1],nothink直出,评测逐字模板)")
    print(f"[OK] 丢弃: {dropped}")

if __name__ == "__main__":
    main()
