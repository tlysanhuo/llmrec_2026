#!/usr/bin/env python3
"""build_capcot.py — 懂推荐 CoT 语义重构 v1(2026-07-09,官方三件套的直接实现)

官方依据:①Tips 四步框架(归纳→溯因→压缩→演绎)+"CoT 优化空间大";②产线工艺=CoT 不看 gold
生成、按与 gold 语义相关性过滤;③07-09 发布的 SFT 对齐 Caption/Tag(每条 rec 行逐 SID 语义)。

设计决定(定稿):
  D1 gold caption/tag 不进 think;演绎步只写"由历史推出的方向";gold 顶级类目仅用于事后过滤:
     预测 top3 类目命中该 prompt 组≥50% 答案的 gold 类目 ⇒ 采用重构;否则整组保留原 think(回退不删行)。
  D2 重复组(同 prompt 多答案)think 组内保持一致——按 prompt 统一重构、组级过滤,结构零破坏。
  D3 不变量:instruction/input(含 /think //no_think 标记)/output 答案体/行数/行序 全部零字节改动,
     只替换非空 <think> 内容;空 think 行原样。
  D4 重构 think 长度贴原行长度(内容按 tag 统计与官方框架生成,逐句可溯源 parquet)。
产物:data/processed/data_capcot_v1.jsonl(32,480 行)+ 构建统计。
"""
import json, re, hashlib
from collections import Counter, defaultdict
import pyarrow.parquet as pq

SEED = 'data/processed/data_final.jsonl'
PARQ = 'assets/official/sft_aligned/baseline_caption_tag_lists.parquet'
OUT  = 'data/processed/data_capcot_v1.jsonl'
TRIP = re.compile(r'<\|\w+?_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>')
THINK = re.compile(r'<think>(.*?)</think>', re.S)
DOMCN = {'video': '短视频', 'prod': '电商', 'ad': '广告', 'living': '直播'}

# ---- 1) parquet:构建 (user正文, 答案) → 逐SID tag/caption 的对齐 ----
t = pq.ParquetFile(PARQ).read().to_pandas()
align = {}
for _, r in t.iterrows():
    msgs = json.loads(r.messages)
    user = msgs[1]['content'][0]['text']
    asst = msgs[2]['content'][0]['text']
    m = THINK.search(asst)
    ans = (asst[m.end():] if m else asst).strip()
    sid2tag = {}
    for s, g in zip(r.sid_token_list, r.tag_list):
        if g is not None and s not in sid2tag:
            sid2tag[s] = g
    align[(user, ans)] = sid2tag

# ---- 2) 种子 rec 行按 prompt 分组 ----
rows = [json.loads(l) for l in open(SEED)]
groups = defaultdict(list)
for i, d in enumerate(rows):
    inp = d['input']
    marker = '/think' if inp.endswith('/think') else ('/no_think' if inp.endswith('/no_think') else '')
    core = inp[: -len(marker)] if marker else inp
    m = THINK.search(d['output'])
    ans = (d['output'][m.end():] if m else d['output']).strip()
    is_rec = len(TRIP.findall(ans)) >= 2 or '目标内容' in (d['instruction'] + inp[:80])
    if is_rec:
        groups[core].append(i)

def build_think(core_user, sid2tag, orig_len):
    hist_sids = TRIP.findall(core_user)
    tags = [sid2tag[s] for s in hist_sids if s in sid2tag]
    if len(tags) < 5:
        return None, None
    lvl1 = Counter(g.split('-')[0] for g in tags)
    lvl2 = Counter('-'.join(g.split('-')[:2]) for g in tags)
    top1 = lvl1.most_common(4)
    top2 = lvl2.most_common(6)
    pred_cats = [k for k, _ in top1[:3]]
    p = []
    p.append('【兴趣归纳】按内容类目统计用户历史行为:'
             + '、'.join(f'{k}({v}次)' for k, v in top1) + '。')
    p.append('【溯因分析】展开到二级类目,活跃度最高的方向是:'
             + ';'.join(f'{k}({v}次)' for k, v in top2[:4])
             + '。压缩降噪后,主导兴趣集中在“' + '、'.join(pred_cats[:2]) + '”。')
    p.append('【兴趣权衡】次要但持续出现的方向('
             + '、'.join(k for k, _ in top1[2:4])
             + ')作为兴趣补充;近期高频类目权重更高。')
    p.append('【演绎预测】综合上述归纳,用户在各场景的目标内容应围绕“'
             + '、'.join(k for k, _ in top2[:3])
             + '”一带的内容与商品展开,输出对应目标 token。')
    txt = '\n'.join(p)
    # 长度贴原:过长截段落,过短补二级类目列举
    if orig_len and len(txt) > orig_len * 1.8 and len(p) > 3:
        txt = '\n'.join([p[0], p[1], p[3]])
    return txt, set(pred_cats)

rebuilt = kept = nofill = nomap = 0
out_rows = [dict(d) for d in rows]
for core, idxs in groups.items():
    filled = [i for i in idxs if THINK.search(rows[i]['output']) and THINK.search(rows[i]['output']).group(1).strip()]
    if not filled:
        nofill += len(idxs); continue
    # 该组任一行的 (user, ans) 命中 parquet 即得对齐表(同组同 user)
    sid2tag = None
    for i in idxs:
        m = THINK.search(rows[i]['output'])
        ans = (rows[i]['output'][m.end():] if m else rows[i]['output']).strip()
        sid2tag = align.get((core, ans))
        if sid2tag: break
    if not sid2tag:
        nomap += len(filled); continue
    ol = len(THINK.search(rows[filled[0]]['output']).group(1))
    new_think, pred = build_think(core, sid2tag, ol)
    if not new_think:
        kept += len(filled); continue
    # 组级过滤:≥50% 答案的 gold 顶级类目 ∈ 预测 top3
    hit = tot = 0
    for i in idxs:
        m = THINK.search(rows[i]['output'])
        ans = (rows[i]['output'][m.end():] if m else rows[i]['output']).strip()
        for s in TRIP.findall(ans):
            g = sid2tag.get(s)
            if g:
                tot += 1
                if g.split('-')[0] in pred: hit += 1
    if tot == 0 or hit / tot < 0.5:
        kept += len(filled); continue
    for i in filled:
        o = rows[i]['output']
        m = THINK.search(o)
        out_rows[i]['output'] = o[:m.start()] + '<think>\n' + new_think + '\n</think>' + o[m.end():]
        rebuilt += 1

# ---- 3) 不变量核验 + 落盘 ----
assert len(out_rows) == len(rows) == 32480
diff_think = 0
for a, b in zip(rows, out_rows):
    assert a['instruction'] == b['instruction'] and a['input'] == b['input'] and a['history'] == b['history']
    ma, mb = THINK.search(a['output']), THINK.search(b['output'])
    ansa = a['output'][ma.end():] if ma else a['output']
    ansb = b['output'][mb.end():] if mb else b['output']
    assert ansa == ansb, '答案体被改动!'
    if a['output'] != b['output']: diff_think += 1
with open(OUT, 'w') as f:
    for d in out_rows:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
md5 = hashlib.md5(open(OUT, 'rb').read()).hexdigest()
print(f'[ok] 重构 think 行={rebuilt}(=diff核验 {diff_think}) 过滤回退保留原think={kept} 空think组行={nofill} parquet未对齐={nomap}')
print(f'[ok] {OUT} 32,480 行,md5 {md5};prompt/答案/标记/行序 零改动已断言')
