#!/usr/bin/env python3
"""build_cap_grounding.py — 官方 Caption → 物料 grounding 增料行 v1(2026-07-09)

原料:官方 SFT 对齐数据 assets/official/sft_aligned/baseline_caption_tag_lists.parquet
     (19,204 行=种子 rec 行 1:1;56.9 万唯一 SID,97.7% 带官方 caption)。
产出:data/processed/cap_grounding_v1.jsonl —— desc→SID 增料行,形态与种子物料行
     逐字段同构(instruction/user前缀 采样自种子物料行真实模板;/no_think;空 think;
     单三元组答案),纯加法块,不动任何旧行,底盘无关(训练时与底盘数据集并联)。
剂量 v1:video 2500 + ad 2500(物料评测所在 video_ad 码本)+ prod 500 + living 500 = 6000。
筛选:caption 长度 ∈[80,500];每 SID 取其最长 caption;seed 固定 2026 可复现。
QC:格式正则全验/答案域=SID域/无重复(SID,caption)/统计落盘打印。
"""
import json, re, random, hashlib
import pyarrow.parquet as pq

SEED_PATH = 'data/processed/data_final.jsonl'
PARQUET   = 'assets/official/sft_aligned/baseline_caption_tag_lists.parquet'
OUT       = 'data/processed/cap_grounding_v1.jsonl'
QUOTA     = {'video': 2500, 'ad': 2500, 'prod': 500, 'living': 500}
TRIP_FULL = re.compile(r'^<\|(\w+?)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>$')
ANS_RE    = re.compile(r'^<think>\s*</think>\s*(<\|(\w+?)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>)\s*$', re.S)

random.seed(2026)

# 1) 收集种子物料行的 (instruction, user前缀) 真实模板,按域分族
tmpl = {d: set() for d in QUOTA}
for line in open(SEED_PATH):
    d = json.loads(line)
    m = ANS_RE.match(d['output'].strip())
    if not m or '/no_think' not in d['input']:
        continue
    dom = m.group(2)
    if dom not in tmpl:
        continue
    inp = d['input']
    ci = min([i for i in (inp.find('：'), inp.find(':')) if i > 0] or [-1])
    if 8 <= ci <= 40:
        tmpl[dom].add((d['instruction'], inp[:ci]))
tmpl = {k: sorted(v) for k, v in tmpl.items()}
print('模板族规模:', {k: len(v) for k, v in tmpl.items()})
assert all(len(v) >= 1 for v in tmpl.values()), '模板缺域'

# 2) 从 parquet 建 sid→最长caption(长度过滤+散文体过滤:剔除列表字面量/低中文密度)
CJK_ALL = re.compile(r'[一-鿿]')
best = {}
t = pq.ParquetFile(PARQUET).read().to_pandas()
for sids, caps in zip(t.sid_token_list, t.caption_list):
    for s, c in zip(sids, caps):
        if c is None:
            continue
        L = len(c)
        if L < 80 or L > 500:
            continue
        cs = c.lstrip()
        if cs.startswith('[') or cs.startswith('{'):
            continue
        if len(CJK_ALL.findall(c)) / L < 0.5:
            continue
        if s not in best or L > len(best[s]):
            best[s] = c
print('可用 (SID,caption):', len(best))

# 3) 分域抽样并生成行(同域同 caption 只保留一个 SID,防矛盾监督;input 全局唯一)
by_dom = {d: [] for d in QUOTA}
for s in best:
    dom = TRIP_FULL.match(s)
    if dom and dom.group(1) in by_dom:
        by_dom[dom.group(1)].append(s)
rows = []
seen_cap = set()   # (域, caption)
seen_input = set()
for dom, quota in QUOTA.items():
    pool = sorted(by_dom[dom])
    random.shuffle(pool)
    taken = 0
    for s in pool:
        if taken >= quota:
            break
        cap = best[s]
        if (dom, cap) in seen_cap:
            continue
        ins, pre = random.choice(tmpl[dom])
        inp = f'{pre}：{cap}/no_think'
        if inp in seen_input:
            continue
        seen_cap.add((dom, cap)); seen_input.add(inp)
        rows.append({
            'instruction': ins,
            'input': inp,
            'output': f'<think>\n</think>\n{s}',
            'history': [],
        })
        taken += 1
random.shuffle(rows)

# 4) QC
seen = set()
for r in rows:
    m = ANS_RE.match(r['output'])
    assert m, '答案格式违例'
    assert r['input'].endswith('/no_think'), 'marker 缺失'
    key = (m.group(1), r['input'])
    assert key not in seen, '重复行'
    seen.add(key)
with open(OUT, 'w') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
md5 = hashlib.md5(open(OUT, 'rb').read()).hexdigest()
from collections import Counter
doms = Counter(ANS_RE.match(r['output']).group(2) for r in rows)
lens = sorted(len(r['input']) for r in rows)
print(f'[ok] {OUT}: {len(rows)} 行,域分布={dict(doms)}')
print(f'[ok] input 长度 中位={lens[len(lens)//2]} p10={lens[len(lens)//10]} p90={lens[9*len(lens)//10]}')
print(f'[ok] md5 {md5}')
