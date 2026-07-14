#!/usr/bin/env python3
"""calibrate_offline.py — 同协议离线台保真度报告。

读 logs/offline_eval/*.json(每锚取最新)× 内嵌线上真值表 → 逐维:
  Spearman ρ + 超噪声对判对率 → 判定(可判决/仅方向/盲区)
真值表:历史平台面板,rec 四域已按 07-06 官方列序重标(video/prod/ad/live),
       题数 = 分数/量子(mat .030693 video .009614 prod .003401 ad .0014 live .0009)。
rebal_world 同 ckpt 两面板取均值(方差标定对)。
输出:控制台 markdown + logs/offline_eval/calibration_report.md
"""
import glob
import json
import os
import re
import argparse
from collections import defaultdict

PROJ = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"

# tag(=offline_eval --tag/模型目录名) → 线上真值(题数/分数)
TRUTH = {
    #                         total   mat video prod  ad live action  topic  world
    "OneReason-0.8B-pretrain-competition":
        dict(total=0.6655, mat=5, video=9, prod=16, ad=98, live=100, action=0.0000, topic=0.0055, world=0.1387),
    "baseline_sft_v1":
        dict(total=0.8100, mat=6, video=7, prod=31, ad=94, live=121, action=0.0362, topic=0.0392, world=0.1375),
    "run_a_r2":
        dict(total=0.8092, mat=6, video=5, prod=31, ad=91, live=117, action=0.0667, topic=0.0430, world=0.1294),
    "run_c_material":
        dict(total=0.8198, mat=6, video=8, prod=30, ad=91, live=122, action=0.0446, topic=0.0407, world=0.1346),
    "recipe1_bs32_lr1e4_ep3":
        dict(total=0.8428, mat=5, video=10, prod=32, ad=89, live=121, action=0.0687, topic=0.0401, world=0.1424),
    "recipe2_w5_ep1_platform":
        dict(total=0.7692, mat=4, video=4, prod=41, ad=99, live=114, action=0.0703, topic=0.0268, world=0.1305),
    "exp_seed_ep3":
        dict(total=0.8931, mat=8, video=5, prod=40, ad=95, live=113, action=0.0554, topic=0.0421, world=0.1316),
    "seed_ep5":
        dict(total=0.9081, mat=8, video=7, prod=36, ad=101, live=111, action=0.0584, topic=0.0427, world=0.1309),
    "rebal_world_ep3":  # 两面板均值(0.9009/0.8776)
        dict(total=0.8893, mat=7, video=5.0, prod=37.0, ad=98.5, live=114.5, action=0.0717, topic=0.0423, world=0.1459),
    "rebal_mat_ep3":
        dict(total=0.8454, mat=6, video=5, prod=35, ad=92, live=116, action=0.0747, topic=0.0430, world=0.1435),
    "pstack_v2_ep3":
        dict(total=0.8265, mat=5, video=7, prod=30, ad=92, live=120, action=0.0808, topic=0.0429, world=0.1435),
    "tokengeo_v1_ep3":
        dict(total=0.8338, mat=6, video=4, prod=34, ad=82, live=115, action=0.0905, topic=0.0424, world=0.1446),
    "fk_lora_embed_ep1_merged":
        dict(total=0.8672, mat=6, video=7, prod=35, ad=95, live=115, action=0.0756, topic=0.0429, world=0.1420),
    "riders_fk_lora_ep1_merged":
        dict(total=0.9177, mat=7, video=8, prod=37, ad=99, live=122, action=0.0655, topic=0.0427, world=0.1439),
    "global_v1_lora_ep1_merged":
        dict(total=0.8246, mat=7, video=7, prod=29, ad=87, live=115, action=0.0438, topic=0.0357, world=0.1394),
    # 该模型只完成了 legacy-v3 world，其他离线维度会自动跳过。
    "seed_cotfix_v1_lora_ep1":
        dict(total=0.8674, mat=7, video=8, prod=36, ad=97, live=123, action=0.0683, topic=0.0452, world=0.0937),
}

# 维度: (离线读数取值函数, 线上键, 超噪声判据 = 线上|Δ|下限)
DIMS = {
    "mat_fresh": (lambda r: r.get("mat_fresh", {}).get("pass@64"), "mat", 1),
    "mat_train": (lambda r: r.get("mat_train", {}).get("pass@64"), "mat", 1),
    "rec_video": (lambda r: r.get("rec", {}).get("video", {}).get("pass@64"), "video", 3),
    "rec_prod": (lambda r: r.get("rec", {}).get("prod", {}).get("pass@64"), "prod", 3),
    "rec_ad": (lambda r: r.get("rec", {}).get("ad", {}).get("pass@64"), "ad", 3),
    "rec_live": (lambda r: r.get("rec", {}).get("live", {}).get("pass@64"), "live", 4),
    "action": (lambda r: r.get("action", {}).get("f1"), "action", 0.006),
    "topic": (lambda r: r.get("topic", {}).get("score"), "topic", 0.003),
    "world": (lambda r: r.get("world", {}).get("acc"), "world", 0.010),
}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rk[order[k]] = r
        i = j + 1
    return rk


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da * db else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="legacy-v3",
        help="legacy-v3 accepts only reports without protocol_version; otherwise require an exact version",
    )
    args = parser.parse_args()
    latest = {}
    for fp in sorted(glob.glob(f"{PROJ}/logs/offline_eval/*.json")):
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        report_protocol = r.get("protocol_version")
        if args.protocol == "legacy-v3":
            if report_protocol is not None:
                continue
        elif report_protocol != args.protocol:
            continue
        tag = r.get("tag", "")
        if tag.startswith("smoke"):
            continue
        latest[tag] = r  # sorted → 后者覆盖 = 最新

    lines = [f"# 离线台校准报告 ({args.protocol})", "",
             f"锚数(有离线读数且有线上面板): 见各维 n | 生成: {__import__('datetime').datetime.now().isoformat()[:19]}", "",
             "| 维 | n锚 | Spearman ρ | 超噪声对 | 判对率 | 判定 |", "|---|---|---|---|---|---|"]
    verdicts = {}
    for dim, (getter, key, band) in DIMS.items():
        pts = []
        for tag, truth in TRUTH.items():
            if tag not in latest:
                continue
            v = getter(latest[tag])
            if v is None or truth.get(key) is None:
                continue
            pts.append((tag, v, truth[key]))
        if len(pts) < 4:
            lines.append(f"| {dim} | {len(pts)} | — | — | — | 锚不足 |")
            continue
        off = [p[1] for p in pts]
        on = [p[2] for p in pts]
        rho = spearman(off, on)
        good = tot = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d_on = on[i] - on[j]
                if abs(d_on) < band:
                    continue
                tot += 1
                d_off = off[i] - off[j]
                good += (d_on > 0) == (d_off > 0) and d_off != 0
        agree = good / tot if tot else float("nan")
        if rho >= 0.8 and tot and agree >= 0.85:
            verdict = "★可判决"
        elif rho >= 0.5 or (tot and agree >= 0.70):
            verdict = "仅方向"
        else:
            verdict = "✗盲区"
        verdicts[dim] = verdict
        lines.append(f"| {dim} | {len(pts)} | {rho:.3f} | {tot} | {agree:.0%} | {verdict} |")

    lines += ["", "## 逐锚读数 vs 真值(排查用)", ""]
    for dim, (getter, key, _) in DIMS.items():
        rows = [(t, getter(latest[t]), TRUTH[t].get(key)) for t in TRUTH if t in latest and getter(latest[t]) is not None]
        rows.sort(key=lambda x: -(x[2] if x[2] is not None else -1))
        s = ", ".join(f"{t}:{v}({o})" for t, v, o in rows)
        lines.append(f"- **{dim}** 离线(线上): {s}")

    out = "\n".join(lines)
    print(out)
    suffix = "" if args.protocol == "legacy-v3" else "_" + re.sub(r"[^a-zA-Z0-9_-]+", "_", args.protocol)
    report_path = f"{PROJ}/logs/offline_eval/calibration_report{suffix}.md"
    with open(report_path, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {os.path.relpath(report_path, PROJ)}")


if __name__ == "__main__":
    main()
