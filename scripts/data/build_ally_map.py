#!/usr/bin/env python3
"""build_ally_map.py — Ally 0.99 方向复刻包:riders 底盘 + 映射表衍生弹药(2026-07-07)。

OBSOLETE(2026-07-08): this v1 builder used the old mistaken rec panel order
and over-allocated rec_loo rows to ad/living. Use build_ally_map_v2.py.

情报依据(docs/competitor_intel.md 07-07 晚):Ally = LoRA + "加了那几个映射"(Pid2Sid/
Pid2Caption/Pid2Tag/UserProfile/General 五张表)→ 0.99。实测其截图分片组合
(Caption p3 全 goods × Sid p0 几乎全 video/ad)caption join=0,故按"表选择"复刻、join 用全表做对。
我方对应物早已造好且从未线上测过:
  fresh_mat 6000  = Pid2Caption⋈Pid2Sid(desc→SID,caption哈希去重/种子SID排除,07-04审计干净)
  rec_loo_v2 20000 = UserProfile⋈Pid2Sid(LOO next-item nothink 直出,prompt 对齐评测措辞)
General 不加(world 已 0.1439≈饱和0.145,滿栀 500/4000 双翻车);Tag 辅助任务不加(新形态,单独评估)。

组成(总 45267 条):
  data_riders_fk 37267(0.9177 底盘,原样不动)
  + rec_loo_v2 抽 5000:ad 2000(139 deep_target 金标全收)/living 1500/prod 800/video 700
    —— 按格点量子加权(ad 0.0096 > live 0.0034 > prod 0.0014 > video 0.0009,video 已 122 题史高)
  + fresh_mat 抽 3000(750/域,补 /no_think 尾缀,同 build_fk_fuse 修复)
外加行 +8000(+21.5%),低于 pstack 死亡档(+17% 但那是 3ep 全参),riders 已证 1ep LoRA 耐 +14%。

QC(违例即 assert 死):SID 结构合法;/no_think⇔空think 不变量;包内 exact 去重;
曝光画像(vs data_final 全体 SID:triple已曝光% / s_a未曝光%,07-05 尺子,P2 毒型=27% s_a 未曝光)。

用法: python scripts/data/build_ally_map.py   (seed=2026 固定,可复现)
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
SEED = 2026
REC_QUOTA = {"ad": 2000, "living": 1500, "prod": 800, "video": 700}
N_FRESH_PER_DOM = 750

SID = re.compile(r"<\|(\w+)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
EMPTY_THINK = re.compile(r"^<think>\s*</think>")


def load(path):
    return [json.loads(l) for l in open(path)]


def gold_dom(r):
    m = SID.search(r["output"])
    return m.group(1) if m else None


def main():
    rng = random.Random(SEED)
    riders = load(P / "data_riders_fk.jsonl")
    fresh = load(P / "fresh_mat.jsonl")
    loo = load(P / "rec_loo_v2.jsonl")
    assert len(riders) == 37267 and len(fresh) == 6000 and len(loo) == 20000

    # ---- rec_loo_v2 采样:deep_target 金标全收,余额按域配额 ----
    picked_loo = []
    by_dom = {d: [] for d in REC_QUOTA}
    for r in loo:
        d = gold_dom(r)
        if r.get("meta_gold_from_deep_target"):
            picked_loo.append(r)
        else:
            by_dom[d].append(r)
    n_deep = len(picked_loo)
    for d, q in REC_QUOTA.items():
        q_left = q - sum(1 for r in picked_loo if gold_dom(r) == d)
        rng.shuffle(by_dom[d])
        picked_loo.extend(by_dom[d][:q_left])
    for r in picked_loo:
        r.pop("meta_gold_from_deep_target", None)  # 训练文件不留 meta 列

    # ---- fresh_mat 采样 + /no_think 尾缀修复(同 build_fk_fuse)----
    fresh_by_dom = {}
    for r in fresh:
        fresh_by_dom.setdefault(gold_dom(r), []).append(r)
    picked_fresh = []
    for d, v in sorted(fresh_by_dom.items()):
        rng.shuffle(v)
        picked_fresh.extend(v[:N_FRESH_PER_DOM])
    fixed = 0
    for r in picked_fresh:
        if not r["input"].rstrip().endswith(("/think", "/no_think")):
            r["input"] = r["input"] + "/no_think"
            fixed += 1

    add = picked_loo + picked_fresh

    # ---- QC ----
    for r in add:
        assert SID.search(r["output"]), r["output"][:80]
        assert r["input"].rstrip().endswith("/no_think"), r["input"][-60:]
        assert EMPTY_THINK.match(r["output"]), r["output"][:40]  # nothink 不变量
    keys = {json.dumps(r, ensure_ascii=False, sort_keys=True) for r in riders}
    dups = sum(1 for r in add if json.dumps(r, ensure_ascii=False, sort_keys=True) in keys)
    assert dups == 0, f"外加行与底盘重复 {dups} 条"

    # 曝光画像(07-05 尺子):vs data_final 全体出现过的 SID
    seed_triples, seed_sa = set(), set()
    for line in open(P / "data_final.jsonl"):
        for m in SID.finditer(line):
            seed_triples.add(m.group(2, 3, 4))
            seed_sa.add(m.group(2))
    g = [SID.search(r["output"]).group(2, 3, 4) for r in add]
    trip_hit = sum(1 for t in g if t in seed_triples) / len(g)
    sa_miss = sum(1 for t in g if t[0] not in seed_sa) / len(g)

    out = riders + add
    rng.shuffle(out)
    dst = P / "data_ally_map.jsonl"
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] {dst}: {len(out)} 条 = riders 37267 + rec_loo {len(picked_loo)}(deep_target {n_deep}) + fresh_mat {len(picked_fresh)}(尾缀修 {fixed})")
    print(f"[QC] rec_loo 域分布: {Counter(gold_dom(r) for r in picked_loo)}")
    print(f"[QC] fresh_mat 域分布: {Counter(gold_dom(r) for r in picked_fresh)}")
    print(f"[QC] 外加行曝光画像: triple已曝光 {trip_hit:.1%} / s_a未曝光 {sa_miss:.1%} (P2毒型=27%未曝光,fresh_mat历史值1.9%)")


if __name__ == "__main__":
    main()
