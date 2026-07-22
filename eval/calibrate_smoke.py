#!/usr/bin/env python3
"""聚合 A-E 冒烟评测结果并与 SPEC 在线真值做方向性校准。"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

ANCHORS = {
    "A": {"online_total": 0.6655, "material": 0.1533, "user_f1": 0.0, "user_chain": 0.0055, "recommend": None, "world": 0.1387},
    "B": {"online_total": 0.8694, "material": 0.1840, "user_f1": 0.0608, "user_chain": 0.0357, "recommend": 0.4362, "world": 0.1528},
    "C": {"online_total": 0.9221, "material": 0.1840, "user_f1": 0.1222, "user_chain": 0.0380, "recommend": 0.4231, "world": 0.1539},
    "D": {"online_total": 0.9463, "material": 0.2146, "user_f1": 0.1213, "user_chain": 0.0399, "recommend": 0.4296, "world": 0.1409},
    "E": {"online_total": 0.9867, "material": 0.2453, "user_f1": 0.1207, "user_chain": 0.0386, "recommend": 0.4427, "world": 0.1394},
}
DIMS = ("material", "user_f1", "user_chain", "recommend", "world")


def average_ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for index in ordered[i:j]:
            result[index] = rank
        i = j
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sum((x - left_mean) ** 2 for x in left)
    right_norm = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_norm * right_norm)
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def direction_rate(ids: list[str], offline: list[float], online: list[float]) -> dict:
    concordant = comparable = ties = 0
    pairs = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        offline_delta = offline[j] - offline[i]
        online_delta = online[j] - online[i]
        if offline_delta == 0 or online_delta == 0:
            ties += 1
            status = "tie"
        else:
            comparable += 1
            matched = (offline_delta > 0) == (online_delta > 0)
            concordant += int(matched)
            status = "match" if matched else "mismatch"
        pairs.append({"pair": f"{ids[i]}-{ids[j]}", "status": status})
    return {
        "concordant": concordant,
        "comparable": comparable,
        "ties": ties,
        "rate": concordant / comparable if comparable else None,
        "pairs": pairs,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    results = {
        anchor: json.loads((root / "output" / f"calibration_smoke_{anchor}.json").read_text())
        for anchor in ANCHORS
    }
    rows = []
    for anchor, truth in ANCHORS.items():
        offline = {dim: results[anchor]["summary"][dim]["mean"] for dim in DIMS}
        rows.append({"anchor": anchor, **truth, "offline": offline})

    calibration = {}
    for dim in DIMS:
        ids = [anchor for anchor in ANCHORS if ANCHORS[anchor][dim] is not None]
        offline = [results[anchor]["summary"][dim]["mean"] for anchor in ids]
        online = [ANCHORS[anchor][dim] for anchor in ids]
        calibration[dim] = {
            "n": len(ids),
            "spearman": spearman(offline, online),
            "direction": direction_rate(ids, offline, online),
            "offline_values": dict(zip(ids, offline)),
            "online_values": dict(zip(ids, online)),
        }

    ids = list(ANCHORS)
    online_total = [ANCHORS[anchor]["online_total"] for anchor in ids]
    # 各维分值尺度不同；先对每维在五锚点内取 percentile rank，再等权平均，作为离线综合排序指标。
    dimension_ranks = {
        dim: average_ranks([results[anchor]["summary"][dim]["mean"] for anchor in ids])
        for dim in DIMS
    }
    composite = [sum(dimension_ranks[dim][i] for dim in DIMS) / len(DIMS) for i in range(len(ids))]
    total = {
        "method": "五个离线维度各自在 A-E 内取平均秩，再等权平均",
        "offline_composite": dict(zip(ids, composite)),
        "online_total": dict(zip(ids, online_total)),
        "spearman": spearman(composite, online_total),
        "direction": direction_rate(ids, composite, online_total),
    }
    report = {
        "protocol": "competition-smoke-calibration-v1",
        "n_anchors": 5,
        "dataset": str((root / "data" / "competition_smoke.jsonl").resolve()),
        "rows": rows,
        "per_dimension": calibration,
        "total_alignment": total,
        "limitations": [
            "每维仅 1～7 条，material/recommend 全锚点均为 0，无法用于排序校准。",
            "base 的在线 recommend 真值缺失，因此该维仅用 B-E 四点计算。",
            "n=5 且样本与赛事隐藏集不同，结果只能证明流程跑通并提供方向性信号。",
        ],
    }
    output = root / "output" / "calibration_smoke_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "per_dimension": calibration, "total_alignment": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
