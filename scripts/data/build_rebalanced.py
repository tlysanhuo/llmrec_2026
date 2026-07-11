#!/usr/bin/env python3
"""build_rebalanced.py — 种子数据域重平衡采样(选手实测+0.046的机制复刻)。

依据(2026-07-03 实测):
- 种子 rec 答案域分布严重失衡: video 16640(65.4%) / prod 3459 / ad 3249 / living 2083
- 但线上边际收益倒挂: living 每样本收益≈video 的 21 倍(v4 vs 锚点: living+0.054, video+0.019, prod−0.013)
- 多名选手实测"采样调比率"+0.046(饭昱昶/滿栀,官方种子内部,非17G)

策略(纯种子内部重采样,固定seed,复现审核合规):
- rec_video: 降采样至 VIDEO_KEEP(默认6000, 保留高质量曝光又不再碾压)
- rec_living/prod/ad: 各上采样(重复)至 REC_TARGET(默认4000)
- 物料双向/懂用户/其他: 原样全保留(它们不失衡)
输出 alpaca jsonl。可再与通识数据合并。

用法:
  python scripts/data/build_rebalanced.py --out /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/data_rebal.jsonl \
      [--video_keep 6000 --rec_target 4000 --world /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/world_zh.jsonl --world_n 2824]
"""
import argparse
import json
import random
import re

BEGIN = re.compile(r"<\|(prod|video|ad|living)_begin\|>")


def classify(r):
    # ★物料任务优先判定(其答案也是itemic token, 不能混进rec被采样!)
    #   desc→token: instruction 含"token生成助手"/"输出匹配的…token"/"输出对应的…token"
    ins = r.get("instruction", "")
    if "token" in ins and ("生成助手" in ins or "输出匹配" in ins or "输出对应" in ins):
        return "mat_desc2token"  # 物料desc→token, 默认原样;--mat_target>0 时上采样
    body = r["output"].split("</think>")[-1]
    if body.strip().startswith("["):
        return "user_select"
    m = BEGIN.search(body)
    if m:
        return f"rec_{m.group(1)}"
    return "non_rec"  # 物料token→desc/topic_gen/其他, 原样保留


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/data_final.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video_keep", type=int, default=6000)
    ap.add_argument("--rec_target", type=int, default=4000)
    ap.add_argument("--world", default="", help="可选: 通识数据jsonl路径")
    ap.add_argument("--world_n", type=int, default=2824)
    ap.add_argument("--mat_target", type=int, default=0,
                    help="可选: 物料desc2token上采样目标条数(0=不动)。依据rebal_world_ep3归因:物料分=video_ad子空间曝光量函数,直接补物料样本比经由video推荐样本高效")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = [json.loads(l) for l in open(args.src)]
    groups = {}
    for r in recs:
        groups.setdefault(classify(r), []).append(r)

    out = []
    stats = {}
    for k, v in groups.items():
        if k == "rec_video":
            rng.shuffle(v)
            kept = v[: args.video_keep]
        elif k in ("rec_living", "rec_prod", "rec_ad"):
            kept = list(v)
            while len(kept) < args.rec_target:  # 上采样=重复
                kept.extend(v[: args.rec_target - len(kept)])
        elif k == "mat_desc2token" and args.mat_target > len(v):
            kept = list(v)
            while len(kept) < args.mat_target:  # 物料上采样=重复
                kept.extend(v[: args.mat_target - len(kept)])
        else:  # user_select / non_rec 原样
            kept = v
        stats[k] = (len(v), len(kept))
        out.extend(kept)

    if args.world:
        world = [json.loads(l) for l in open(args.world)]
        rng.shuffle(world)
        out.extend(world[: args.world_n])
        stats["world_zh"] = (len(world), min(args.world_n, len(world)))

    rng.shuffle(out)
    with open(args.out, "w") as g:
        for r in out:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{'组':14} {'原量':>7} {'采后':>7}")
    for k, (a, b) in sorted(stats.items()):
        print(f"{k:14} {a:7d} {b:7d}")
    print(f"总计 → {len(out)} 条 → {args.out}")


if __name__ == "__main__":
    main()
