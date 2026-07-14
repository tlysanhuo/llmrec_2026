#!/usr/bin/env python3
"""build_rec_loo.py — 从 UserProfile 原始行为序列构造"预测型"推荐 SFT 数据(teacher-free)。

动机(2026-07-03 数据盘点):种子推荐数据 97.3% 的 gold 是"新item"但任务形式是
"总结用户在各场景的目标内容"(归纳措辞);评测考的是"预测用户下一个会点击的X"。
本脚本按论文 R3 的思路做 teacher-free 简化:时间序 leave-one-out——
历史(除最后一个item) → gold=时间上最后一个item。产出与评测同构的 next-item 预测样本。

构造(固定seed,复现审核合规。输入=官方 UserProfile + Pid2Sid,均为官方原始素材):
  1. 建 pid→(dom,sid) 索引(Pid2Sid 198shard,~40M映射,内存dict约4-5GB)
  2. 每用户每域:取行为序列(video=深看列表, ec=点击列表, live=观看列表, ad=outer点击),
     按 ts 排序,join sid,丢 join 不上的
  3. 历史窗=最后 HIST_N 个(不含最后一个),gold=最后一个;历史须≥MIN_HIST
  4. prompt 措辞对齐评测(见 logs/eval 平台日志):
     system: 你是一个智能推荐助理，能根据多域历史行为，推荐用户下一个感兴趣的{X}。
     user: 用户{域}历史行为：...itemic序列... \n请推荐用户下一个会点击的{X}。/no_think
     output: <think>\n</think>\n<|dom_begin|><s_a><s_b><s_c>   (unCoT 直出——论文§8.2: ad域unCoT更好;
             其他域也先做unCoT版,零teacher成本; CoT版需teacher后置)
  5. 按域配额采样(默认每域 N_PER_DOM=3000)

用法:
  python scripts/data/build_rec_loo.py --out /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/data/processed/rec_loo.jsonl \
      [--n_per_dom 3000 --hist_n 60 --min_hist 10 --shards 2]
"""
import argparse
import glob
import json
import random

DOM_MAP = {  # Pid2Sid domain → (begin token域名, 中文名, 行为措辞)
    "video/video": ("video", "视频", "深度观看了"),
    "video/ad": ("ad", "广告", "点击了"),
    "goods": ("prod", "商品", "点击了"),
    "live": ("living", "直播", "观看了"),
}
SYS = "你是一个智能推荐助理，能根据多域历史行为，推荐用户下一个感兴趣的{cn}。"
ASK = "请推荐用户下一个会点击的{cn}。/no_think"


def tok(dom, sid):
    a, b, c = int(sid[0]), int(sid[1]), int(sid[2])
    return f"<|{dom}_begin|><s_a_{a}><s_b_{b}><s_c_{c}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_UserProfile")
    ap.add_argument("--sid_dir", default="/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/official/hf_raw/OneReason_Pid2Sid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_dom", type=int, default=3000)
    ap.add_argument("--hist_n", type=int, default=60)
    ap.add_argument("--min_hist", type=int, default=10)
    ap.add_argument("--shards", type=int, default=2, help="用几个UserProfile shard(每个5万用户)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import pyarrow.parquet as pq

    # ---- 1. pid→(dom,sid) 索引 ----
    print("建 Pid2Sid 索引 ...", flush=True)
    pid2 = {}
    for f in sorted(glob.glob(f"{args.sid_dir}/*.parquet")):
        t = pq.read_table(f)
        for p, d, s in zip(t["pid"].to_pylist(), t["domain"].to_pylist(), t["sid_three"].to_pylist()):
            if d in DOM_MAP and s and len(s) == 3:
                pid2[p] = (DOM_MAP[d][0], s)
    print(f"  索引 {len(pid2):,} pids", flush=True)

    # ---- 2. 逐用户构造 LOO ----
    # 每域的 (pid列, ts列, 过滤label列) 取法
    # ★2026-07-04 修正(队友17G调研采纳):video 按 play_done=1 过滤(60%正率主label,
    #   否则gold可能是划走的视频);ad 域 gold 优先用 outer_loop_deep_target_pid
    #   (全数据集唯一严格无泄漏的next-item金标,8851条,历史侧仍用click序列)
    DOM_COLS = {
        "video": ("video_history_sampled_pid_list", "video_history_ts_list", "video_history_play_done_list"),
        "prod": ("ec_good_click_item_id_list_extend", None, None),  # 无ts列, 保持列表原序(官方序)
        "ad": ("outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts", None),
        "living": ("live_hist_author_id_list", "live_hist_timestamp_list", None),
    }
    AD_GOLD_COL = "outer_loop_deep_target_pid"  # ad 域金标(当前字段,严格晚于全部历史)
    buckets = {d: [] for d in DOM_COLS}
    up_files = sorted(glob.glob(f"{args.up_dir}/*.parquet"))[: args.shards]
    need_cols = sorted({c for cols in DOM_COLS.values() for c in cols if c} | {AD_GOLD_COL})
    for uf in up_files:
        print(f"读 {uf.split('/')[-1]} ...", flush=True)
        t = pq.read_table(uf, columns=need_cols)
        rows = t.to_pylist()
        for row in rows:
            for dom, (pc, tc, lc) in DOM_COLS.items():
                if all(len(b) >= args.n_per_dom * 3 for b in buckets.values()):
                    break
                pids = row.get(pc)
                if not pids or len(pids) < args.min_hist + 1:
                    continue
                labels = row.get(lc) if lc else None
                if labels and len(labels) == len(pids):  # play_done=1 过滤
                    keep = [i for i, v in enumerate(labels) if v == 1]
                    if len(keep) < args.min_hist + 1:
                        continue
                    pids = [pids[i] for i in keep]
                    ts_raw = row.get(tc) if tc else None
                    ts = [ts_raw[i] for i in keep] if ts_raw and len(ts_raw) >= max(keep) + 1 else None
                else:
                    ts = row.get(tc) if tc else None
                seq = list(zip(pids, ts)) if ts and len(ts) == len(pids) else [(p, i) for i, p in enumerate(pids)]
                seq.sort(key=lambda x: x[1])
                # join sid; gold 必须 join 上且域匹配
                joined = [(p, pid2.get(p)) for p, _ in seq]
                joined = [(p, ds) for p, ds in joined if ds and ds[0] == dom]
                if len(joined) < args.min_hist + 1:
                    continue
                # ad 域:若有 deep_target 金标则历史全保留、gold=deep_target(无泄漏)
                gold_from_deep = False
                if dom == "ad":
                    dt = row.get(AD_GOLD_COL)
                    if dt:
                        dt_pid = dt[0] if isinstance(dt, list) else dt
                        ds = pid2.get(dt_pid)
                        if ds and ds[0] == "ad":
                            hist = joined[-args.hist_n:]
                            gpid, gds = dt_pid, ds
                            gold_from_deep = True
                if not gold_from_deep:
                    *hist, (gpid, gds) = joined[-(args.hist_n + 1):]
                if len(hist) < args.min_hist:
                    continue
                gold_tok = tok(gds[0], gds[1])
                hist_toks = [tok(ds[0], ds[1]) for _, ds in hist]
                if gold_tok in hist_toks:
                    continue  # 保证预测型(gold不在历史)
                _, cn, verb = DOM_MAP[[k for k, v in DOM_MAP.items() if v[0] == dom][0]]
                user = (f"用户{cn}历史行为：{verb} " + ", ".join(hist_toks) + "。\n\n" + ASK.format(cn=cn))
                buckets[dom].append({
                    "instruction": SYS.format(cn=cn),
                    "input": user,
                    "output": f"<think>\n</think>\n{gold_tok}",
                    "history": [],
                    "meta_gold_from_deep_target": gold_from_deep,
                })

    # ---- 3. 采样配额 & 输出 ----
    out = []
    for d, v in buckets.items():
        rng.shuffle(v)
        out.extend(v[: args.n_per_dom])
        print(f"  {d}: 候选{len(v)} → 取{min(len(v), args.n_per_dom)}")
    rng.shuffle(out)
    with open(args.out, "w") as g:
        for r in out:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"共 {len(out)} 条 → {args.out}")


if __name__ == "__main__":
    main()
