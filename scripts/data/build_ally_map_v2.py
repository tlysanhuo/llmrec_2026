#!/usr/bin/env python3
"""build_ally_map_v2.py — corrected Ally map package.

This is the corrected version of build_ally_map.py after the official panel
order was confirmed as video/prod/ad/live.  The v1 package spent too much of
the 5000 rec_loo budget on low-quantum ad/live columns.  V2 keeps the same
total size and same data sources, but reallocates rec_loo toward video/prod:

  data_riders_fk 37267
  + rec_loo_v2 5000: video 2000, prod 1500, ad 800, living 700
  + fresh_mat 3000: 750 per domain

Usage:
  python scripts/data/build_ally_map_v2.py
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
SEED = 2026
REC_QUOTA = {"video": 2000, "prod": 1500, "ad": 800, "living": 700}
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

    picked_loo = []
    by_dom = {d: [] for d in REC_QUOTA}
    for r in loo:
        d = gold_dom(r)
        assert d in by_dom, d
        if r.get("meta_gold_from_deep_target"):
            picked_loo.append(r)
        else:
            by_dom[d].append(r)

    n_deep = len(picked_loo)
    for d, q in REC_QUOTA.items():
        q_left = q - sum(1 for r in picked_loo if gold_dom(r) == d)
        assert q_left >= 0, f"deep_target rows exceed quota for {d}: {q_left}"
        rng.shuffle(by_dom[d])
        picked_loo.extend(by_dom[d][:q_left])
    assert len(picked_loo) == sum(REC_QUOTA.values())

    for r in picked_loo:
        r.pop("meta_gold_from_deep_target", None)

    fresh_by_dom = {}
    for r in fresh:
        d = gold_dom(r)
        assert d, r["output"][:80]
        fresh_by_dom.setdefault(d, []).append(r)

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

    for r in add:
        assert SID.search(r["output"]), r["output"][:80]
        assert r["input"].rstrip().endswith("/no_think"), r["input"][-60:]
        assert EMPTY_THINK.match(r["output"]), r["output"][:40]

    keys = {json.dumps(r, ensure_ascii=False, sort_keys=True) for r in riders}
    dups = sum(1 for r in add if json.dumps(r, ensure_ascii=False, sort_keys=True) in keys)
    assert dups == 0, f"extra rows duplicate riders base: {dups}"

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
    dst = P / "data_ally_map_v2.jsonl"
    with open(dst, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] {dst}: {len(out)} rows = riders 37267 + rec_loo {len(picked_loo)}(deep_target {n_deep}) + fresh_mat {len(picked_fresh)}(suffix fixed {fixed})")
    print(f"[QC] rec_loo domain distribution: {Counter(gold_dom(r) for r in picked_loo)}")
    print(f"[QC] fresh_mat domain distribution: {Counter(gold_dom(r) for r in picked_fresh)}")
    print(f"[QC] extra exposure profile: triple_seen {trip_hit:.1%} / s_a_unseen {sa_miss:.1%}")


if __name__ == "__main__":
    main()
