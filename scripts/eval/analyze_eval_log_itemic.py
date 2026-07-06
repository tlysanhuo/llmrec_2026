#!/usr/bin/env python3
"""analyze_eval_log_itemic.py — 平台评测日志 itemic 错误分析。

平台日志每任务打印前 5 个样本的完整 beam 候选(grounding/rec: 64 条)。
无 gold,但可跨版本(v1 vs v4)对比同题的:
  - beam 候选多样性: distinct s_a / (s_a,s_b) / 完整三元组 数
  - 首候选(top-1 beam)是否一致
多样性坍缩 = pass@64 的直接杀手(s_a 错则 64 条全错)。

用法: python analyze_eval_log_itemic.py <log1> <log2> ...
"""
import re
import sys

SABC = re.compile(r"<\|(prod|video|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
TASK = re.compile(r"Task \[(\d)/8\]: (challenge_\w+)")
SAMPLE = re.compile(r"Sample ID: (\d+)")
OUTPUT = re.compile(r"Output\[(\d+)\]:")


def parse_log(path):
    """→ {task: {sample_id: [ (dom,a,b,c) or None per beam candidate ]}}"""
    tasks = {}
    cur_task = cur_sample = None
    cur_out_idx = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            for line in raw.replace("\r", "\n").split("\n"):
                m = TASK.search(line)
                if m:
                    cur_task = m.group(2)
                    tasks.setdefault(cur_task, {})
                    cur_sample = None
                    continue
                m = SAMPLE.search(line)
                if m and cur_task:
                    cur_sample = int(m.group(1))
                    tasks[cur_task][cur_sample] = []
                    cur_out_idx = None
                    continue
                m = OUTPUT.search(line)
                if m and cur_sample is not None:
                    cur_out_idx = int(m.group(1))
                    tasks[cur_task][cur_sample].append(None)
                    continue
                if cur_out_idx is not None and cur_sample is not None:
                    m = SABC.search(line)
                    if m and tasks[cur_task][cur_sample] and tasks[cur_task][cur_sample][-1] is None:
                        tasks[cur_task][cur_sample][-1] = (m.group(1), m.group(2), m.group(3), m.group(4))
    return tasks


def beam_stats(cands):
    toks = [c for c in cands if c]
    if not toks:
        return None
    sa = {t[1] for t in toks}
    sab = {(t[1], t[2]) for t in toks}
    full = set(toks)
    return {"n": len(cands), "valid": len(toks), "distinct_sa": len(sa),
            "distinct_sab": len(sab), "distinct_full": len(full),
            "top1": toks[0], "sa_set": sa}


def main():
    logs = {p.split("/")[-1]: parse_log(p) for p in sys.argv[1:]}
    beam_tasks = ["challenge_itemic_pattern_grounding", "challenge_recommendation_ad",
                  "challenge_recommendation_live", "challenge_recommendation_product",
                  "challenge_recommendation_video"]
    for task in beam_tasks:
        print(f"\n===== {task} =====")
        for name, tasks in logs.items():
            samples = tasks.get(task, {})
            if not samples:
                print(f"  {name}: (无样本)")
                continue
            rows = []
            for sid in sorted(samples):
                st = beam_stats(samples[sid])
                if st:
                    rows.append(st)
            if not rows:
                print(f"  {name}: (解析不到候选)")
                continue
            avg = lambda k: sum(r[k] for r in rows) / len(rows)
            print(f"  {name}: samples={len(rows)} beam候选/题={avg('n'):.0f} 合法率={avg('valid')/max(avg('n'),1):.2f}"
                  f" | distinct s_a={avg('distinct_sa'):.1f} s_ab={avg('distinct_sab'):.1f} full={avg('distinct_full'):.1f}")
        # 同题 top1 对比 + s_a 集合交集
        names = list(logs)
        if len(names) >= 2:
            a, b = names[0], names[-1]
            sa_ov, top1_same, common = [], 0, 0
            for sid in sorted(set(logs[a].get(task, {})) & set(logs[b].get(task, {}))):
                s1, s2 = beam_stats(logs[a][task][sid]), beam_stats(logs[b][task][sid])
                if s1 and s2:
                    common += 1
                    inter = len(s1["sa_set"] & s2["sa_set"])
                    uni = len(s1["sa_set"] | s2["sa_set"])
                    sa_ov.append(inter / uni if uni else 0)
                    top1_same += s1["top1"] == s2["top1"]
            if common:
                print(f"  [{a} vs {b}] 同题{common}对: top1一致={top1_same}/{common}"
                      f" s_a集合Jaccard均值={sum(sa_ov)/len(sa_ov):.2f}")


if __name__ == "__main__":
    main()
