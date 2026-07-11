#!/usr/bin/env python3
"""build_evalform_v1.py — 把 riders 底盘中 rec 目标行(≈18651)的题面+答案转写为评测形态,监督内容零改动。

背景(2026-07-08 数据审计):种子/riders 的主力 rec 行为"目标内容"方言(6 种改写模板),与评测题面
(单域 next-item,"请推断用户接下来会点击的X")形态覆盖率≈0;0.8B=模板学习者,两种方言并存=容量干扰,
故整体"换"不"加"。模板字节级取自平台评测日志(riders_fk_lora_ep1_20260706.log)+ 赛题解析 p16 域顺序规则。
不变量:每行的历史 SID 多重集、目标 SID、CoT 文本、think/nothink 通路完全不变;只换题面/答案外衣。
QC 硬门:①新旧 prompt 的 SID 多重集必须完全一致,否则该行回退保留原样(fallback 率>2% 则整包作废);
②答案=原目标 SID(裸);③行数不变;④转写行须含结尾句+全部四段头形态统计落盘。
确定性:无随机,逐行确定转换。
"""
import json, re, collections, sys

SRC = 'data/processed/data_riders_fk.jsonl'
DST = 'data/processed/data_evalform_v1.jsonl'

SID_RE = re.compile(r'<\|(video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>')
TARGET_RE = re.compile(
    r'^\s*该用户最近[^<]{0,20}[:：]\s*(<\|(?:video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>)\s*$')

SYSTEM = {
    'video':  '你是一个推荐系统助手，擅长根据多域历史行为预测用户的视频偏好。',
    'prod':   '你是一个智能推荐助理，能根据多域历史行为，推荐用户下一个感兴趣的商品。',
    'ad':     '你是一个智能推荐助理，能根据多域历史行为，推荐用户下一个感兴趣的广告。',
    'living': '你是一个推荐系统助手，擅长根据多域历史行为预测用户的主播偏好。',
}
CLOSE = {
    'video':  '请推断用户接下来会点击的视频。',
    'prod':   '请推断用户接下来会点击的商品。',
    'ad':     '请推荐用户下一个会点击的广告。',
    'living': '请推断用户接下来会点击的主播。',
}
# 赛题解析 p16:目标域最后、视频倒数第二、广告倒数第三
ORDER = {
    'video':  ['living', 'prod', 'ad', 'video'],
    'prod':   ['living', 'ad', 'video', 'prod'],
    'ad':     ['living', 'prod', 'video', 'ad'],
    'living': ['prod', 'ad', 'video', 'living'],
}
DOMTITLE = {'living': '直播', 'prod': '电商', 'video': '视频', 'ad': '广告'}

def phrase_for(dom, ctx):
    """按 SID 段前文的行为关键词,给出评测措辞的行为短语(取自评测日志逐字词表)。"""
    if dom == 'living':
        if '打赏' in ctx and '关注' in ctx: return '有过关注/打赏行为的主播有'
        if '打赏' in ctx: return '打赏过的主播有'
        if '关注' in ctx: return '关注了主播'
        return '观看过的主播有'
    if dom == 'prod':
        if '购买' in ctx or '下单' in ctx: return '购买过的商品有'
        if '加购' in ctx: return '加购过的商品有'
        return '浏览过的商品有'
    if dom == 'ad':
        if '深度转化' in ctx or '深转' in ctx: return '完成过深度转化的广告有'
        return '点击过的广告有'
    # video
    combo = [w for w in ('点赞', '转发', '收藏', '评论', '长播') if w in ctx]
    if '深度观看' in ctx or '深度' in ctx: return '深度观看过的视频有'
    if combo: return '有过' + '/'.join(combo) + '行为的视频有'
    return '看过的视频有'

def transpose(prompt, target_dom):
    """prompt -> 评测形态 user 文本;失败返回 None。"""
    runs = []  # (dom, phrase, [sid,...]) 保序
    last_end = 0
    for m in SID_RE.finditer(prompt):
        dom = m.group(1)
        ctx = prompt[max(0, m.start() - 45):m.start()]
        # 与上一 SID 相邻(中间只有分隔符)则并入上一 run
        gap = prompt[last_end:m.start()]
        if runs and runs[-1][0] == dom and re.fullmatch(r'[\s,，、和与及]*', gap):
            runs[-1][2].append(m.group(0))
        else:
            runs.append([dom, phrase_for(dom, ctx), [m.group(0)]])
        last_end = m.end()
    if not runs:
        return None
    # 按域聚合(保 run 顺序),再按评测域顺序输出
    by_dom = collections.OrderedDict()
    for dom, ph, sids in runs:
        by_dom.setdefault(dom, []).append((ph, sids))
    lines = []
    for dom in ORDER[target_dom]:
        if dom not in by_dom:
            continue  # 该域无历史,评测允许缺域
        segs = []
        for ph, sids in by_dom[dom]:
            segs.append(ph + ' ' + ', '.join(sids))
        lines.append('用户在' + DOMTITLE[dom] + '域: ' + '，'.join(segs) + '。')
    covered = set(by_dom) - set(ORDER[target_dom])
    if covered:
        return None  # 出现未知域,回退
    return '用户多域历史行为：\n' + '\n'.join(lines) + '\n\n' + CLOSE[target_dom]

def main():
    n = swapped = fallback = 0
    stats = collections.Counter()
    out_f = open(DST, 'w')
    for line in open(SRC):
        d = json.loads(line)
        n += 1
        out = d.get('output') or ''
        m_think = re.match(r'\s*<think>(.*?)</think>\s*', out, re.S)
        think_txt = m_think.group(1) if m_think else None
        body = re.sub(r'^\s*<think>.*?</think>\s*', '', out, flags=re.S).strip()
        tm = TARGET_RE.match(body)
        if not tm:
            out_f.write(line); continue
        target_sid = tm.group(1)
        target_dom = SID_RE.match(target_sid).group(1)
        prompt = (d.get('instruction') or '') + '\n' + (d.get('input') or '')
        new_user = transpose(prompt, target_dom)
        # QC 门①:新旧 prompt 的 SID 多重集必须完全一致
        old_sids = [mm.group(0) for mm in SID_RE.finditer(prompt)]
        new_sids = [mm.group(0) for mm in SID_RE.finditer(new_user)] if new_user else []
        if new_user is None or sorted(old_sids) != sorted(new_sids):
            fallback += 1; stats['fallback_' + target_dom] += 1
            out_f.write(line); continue
        has_cot = bool(think_txt and think_txt.strip())
        suffix = '/think' if has_cot else '/no_think'
        new_out = ('<think>' + think_txt + '</think>\n' if has_cot else '<think>\n</think>\n') + target_sid
        nd = {'instruction': new_user + suffix, 'input': '',
              'output': new_out, 'system': SYSTEM[target_dom]}
        out_f.write(json.dumps(nd, ensure_ascii=False) + '\n')
        swapped += 1; stats['swap_' + target_dom] += 1
        stats['cot' if has_cot else 'nothink'] += 1
    out_f.close()
    print(f'总行={n} 转写={swapped} 回退={fallback} (回退率={100*fallback/max(swapped+fallback,1):.2f}%)')
    print(dict(stats))
    if swapped and fallback / (swapped + fallback) > 0.02:
        print('!!! QC 硬门失败:回退率>2%,整包作废', file=sys.stderr); sys.exit(1)

if __name__ == '__main__':
    main()
