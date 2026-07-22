#!/usr/bin/env python3
"""临时只读探测脚本：确认 懂推荐1~4.jsonl 各自的目标域（response 侧 SID 前缀分布）。
用完即删，不属于正式代码改动。"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.loaders import SAMPLED_DIR
from common.sid_utils import parse_sid_tokens


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                rows.extend(obj)
            else:
                rows.append(obj)
    return rows


for i in range(1, 5):
    path = SAMPLED_DIR / f"懂推荐{i}.jsonl"
    rows = read_rows(path)
    print(f"=== 懂推荐{i}.jsonl, n={len(rows)} ===")

    c = Counter()
    prompt_prefix_c = Counter()
    for row in rows:
        toks = parse_sid_tokens(row.get("response", ""))
        for t in toks:
            c[t[0]] += 1
        prompt = row.get("prompt", "")
        ptoks = parse_sid_tokens(prompt)
        # 只看prompt里最后一个出现的前缀在文本里的相对位置分布（粗略）
    print("全体 response SID 前缀分布:", dict(c))

    r0 = rows[0]
    print("样本0 prompt 结尾300字符:")
    print(repr(r0.get("prompt", "")[-300:]))
    print("样本0 response 前80字符:")
    print(repr(r0.get("response", "")[:80]))
    print()