#!/usr/bin/env python3
"""build_r0_material.py — 懂物料 desc→token 数据构造 (teacher-free)。

机理(见 memory material-data-feasibility):
  从 Pid2Caption + Pid2Sid 自建 (caption -> itemic token) 对, 天然自洽, 零 teacher 零幻觉。
  ★关键: caption 做规则增广(换指令模板/随机截取/轻改写), 逼模型学"语义→token"而非背原文
  (官方种子的 caption 与表原文 0/8 匹配 = 官方就是改写训练的)。

格式对齐官方 desc→token 空-think 版:
  system = 域池随机;  user = 指令前缀池随机 + (增广后)描述 + /no_think
  assistant = "<think>\n</think>\n<|TYPE_begin|><s_a><s_b><s_c>"

域: prod/video/ad = 成句 caption 直接用; living = caption 是标签串, 拼成一句描述。
采样: 按 s_a(粗类目)分层, 防热门类目扎堆(直接对治评测里的 beam 前缀坍缩)。

用法:
  python scripts/data/build_r0_material.py --n 20000 --out <path> --seed 2026 \
     --domains prod,video,ad,living --aug 1
"""
import argparse
import json
import os
import random
import re
import sys

import pandas as pd

IDX = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/derived/index"

# 官方各域 system 池(从种子采样 top,每域取代表若干)
SYSTEM_POOL = {
    "prod": [
        "你是一个商品语义标识分析助手，能够根据商品描述生成对应的商品token。",
        "作为AI商品标识助手，你可以根据商品描述生成匹配的商品token。",
        "你具备从商品特征描述中提取关键信息并输出商品token的能力。",
        "作为商品标识生成助手，你需要根据给定的商品描述输出匹配的商品token。",
    ],
    "video": [
        "作为 AI 短视频标识助手，你可以根据短视频描述生成匹配的短视频token。",
        "你是一名专业的短视频token生成助手，请根据短视频的画面、主体、动作、场景与风格描述生成匹配的短视频token。",
        "你擅长把短视频的内容描述映射成精确的短视频token。",
        "你是一名资深的媒体内容分析师，能够准确识别视频特征并输出对应的视频token。",
    ],
    "ad": [
        "你是一名广告token生成助手，需要根据广告内容描述生成最匹配的广告token。",
        "请根据输入的广告描述，输出能与其语义最匹配的广告token。",
        "你擅长根据广告内容、风格和主题描述，输出对应的广告token。",
    ],
    "living": [
        "你是一个主播语义标识分析助手，能够根据主播描述生成对应的主播token。",
        "作为 AI 主播标识助手，你可以根据主播描述生成匹配的主播token。",
        "你擅长把主播的外在形象、直播内容和风格描述映射成精确的主播token。",
    ],
}

# 指令前缀池(冒号前的引导句;实际 prompt = 前缀 + "：" + 描述)
INSTR_POOL = {
    "prod": [
        "请阅读下面的商品描述，并输出对应的商品token",
        "请从以下商品描述中推断并生成对应的商品token",
        "下面是一段商品描述，请返回匹配的商品token",
        "请依据这段商品特征描述生成商品token",
    ],
    "video": [
        "请根据给定的短视频内容描述生成短视频token",
        "请分析这段短视频内容，并生成对应的短视频token",
        "阅读这段短视频描述后，请输出其对应的短视频token",
        "请解析以下视频内容并输出对应的视频token",
    ],
    "ad": [
        "请根据以下广告内容描述，生成匹配的广告token",
        "请根据这段广告dense caption生成匹配的广告token",
        "根据以下广告描述还原其token，只输出目标广告token",
    ],
    "living": [
        "请根据给定的主播形象与直播风格描述生成主播token",
        "请阅读下面这段主播描述，并输出其对应的主播token",
        "阅读这段主播描述后，请输出其对应的主播token",
    ],
}


def caption_from_living(cap):
    """living caption 是标签串 "['国风','微醺氛围',...]" → 拼成一句主播描述。"""
    try:
        tags = json.loads(cap.replace("'", '"')) if isinstance(cap, str) and cap.strip().startswith("[") else None
    except Exception:
        tags = None
    if not tags:
        return None
    return "这是一位主播，其风格与内容特征包括：" + "、".join(str(t) for t in tags) + "。"


def augment(desc, rng):
    """规则增广: 逼模型学语义映射而非背原文。返回增广后的描述。"""
    d = desc
    # 1) 随机截取: 保留前 60%~100% (模拟评测描述详略不同)
    if len(d) > 60 and rng.random() < 0.5:
        keep = int(len(d) * rng.uniform(0.6, 1.0))
        d = d[:keep].rstrip("，,、 ")
    # 2) 轻改写: 去掉常见模板噪声短语(价格/发货等), 概率触发
    if rng.random() < 0.4:
        d = re.sub(r"[，,。]?\s*(价格[\d.]+元|免运费|承诺\d+小时发货|无品牌|自营正品)", "", d)
    # 3) 去首尾空白/重复标点
    d = re.sub(r"\s+", "", d).strip("，,。、")
    return d if len(d) >= 8 else desc  # 太短则回退原文


TOK_DOMAINS = {"prod", "video", "ad", "living"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="目标样本数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--domains", default="prod,video,ad,living")
    ap.add_argument("--aug", type=int, default=1, help="1=开启caption增广")
    ap.add_argument("--stratify", type=int, default=1, help="1=按s_a分层去偏(防热门扎堆)")
    ap.add_argument("--per_sa_cap", type=int, default=40, help="每个 s_a 最多取多少样本(分层上限)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    domains = [d for d in args.domains.split(",") if d in TOK_DOMAINS]

    print("[load] index...", file=sys.stderr)
    cap = pd.read_parquet(f"{IDX}/pid2caption.parquet")
    sid = pd.read_parquet(f"{IDX}/pid2sid.parquet")
    sid_map = {k: (int(a), int(b), int(c)) for k, a, b, c in zip(sid.key, sid.s_a, sid.s_b, sid.s_c)}
    del sid
    print(f"[load] cap {len(cap):,} sid {len(sid_map):,}", file=sys.stderr)

    # 按域过滤 + join sid
    cap["dom"] = cap.key.str.split("|").str[0]
    cap = cap[cap.dom.isin(domains)]

    per_dom_target = max(1, args.n // len(domains))
    out_recs = []
    stats = {}
    for dom in domains:
        sub = cap[cap.dom == dom]
        # shuffle by sampling
        idxs = list(range(len(sub)))
        rng.shuffle(idxs)
        sa_count = {}
        got = 0
        recs_dom = []
        keys = sub.key.to_numpy(); caps = sub.caption.to_numpy()
        for i in idxs:
            if got >= per_dom_target:
                break
            k = keys[i]
            if k not in sid_map:
                continue
            a, b, c = sid_map[k]
            # 分层去偏
            if args.stratify:
                if sa_count.get(a, 0) >= args.per_sa_cap:
                    continue
            rawcap = caps[i]
            if dom == "living":
                desc = caption_from_living(rawcap)
                if desc is None:
                    continue
            else:
                desc = str(rawcap)
                if desc.strip().startswith("["):  # 意外的标签串, 跳过
                    continue
            if len(desc) < 8:
                continue
            if args.aug:
                desc = augment(desc, rng)
            # 组装
            system = rng.choice(SYSTEM_POOL[dom])
            instr = rng.choice(INSTR_POOL[dom])
            token = f"<|{dom}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"
            record = {
                "instruction": system,
                "input": f"{instr}：{desc}/no_think",
                "output": f"<think>\n</think>\n{token}",
                "history": [],
            }
            # 强校验
            assert record["output"].split("</think>")[-1].strip() == token
            recs_dom.append(record)
            sa_count[a] = sa_count.get(a, 0) + 1
            got += 1
        out_recs.extend(recs_dom)
        stats[dom] = got
        print(f"[{dom}] {got} 条 (unique s_a: {len(sa_count)})", file=sys.stderr)

    rng.shuffle(out_recs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # manifest
    man = {"seed": args.seed, "n": len(out_recs), "domains": stats, "aug": args.aug,
           "stratify": args.stratify, "per_sa_cap": args.per_sa_cap, "task": "desc2token_empty_think"}
    open(args.out + ".manifest.json", "w").write(json.dumps(man, ensure_ascii=False, indent=2))
    print(f"[done] wrote {len(out_recs)} -> {args.out}", file=sys.stderr)
    print(f"[manifest] {man}", file=sys.stderr)


if __name__ == "__main__":
    main()
