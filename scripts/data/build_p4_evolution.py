#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_p4_evolution.py — P4「抽取式演化链」数据集(对治 topic_gen C2/C3)。

════════════════════════════════════════════════════════════════════════
预登记(训练前写死,不许事后改):
  topic_gen 当前 0.0429 → 预测 +0.005 ~ +0.015(与 P3 叠加共治 C1 引用变异);
  若无效 ⇒ 证明 NLI 评分吃"推断腔"(teacher 式心理叙事)而非事实句 —— 反向信号,
  下一步应改走"模仿种子 teacher 语体"路线而非"事实最小化"路线。

设计依据(docs/eval_log_mining_20260705.md §C / §2-P4):
  C2 选事件与主题无关;C3 语义标注错(logic 与行为不符)。
  该任务是**抽取式**:链名/主题在输入里给定,模型只需
    ①从真实 timeline 挑与主题相关的事件 ②按时间升序排链
    ③action 逐字引用 timeline 条目 ④每步一句 logic。
  评测端 EvolutionTopicGenEvaluator 用 NLI(nli-deberta-v3-base, CrossEncoder)
  评"行为准确性 + 推理逻辑合理性"(logs/eval/seed_ep3_20260703.log:42786-42790)。

评测模板(逐字,从 seed_ep3_20260703.log 3892-3961 五个样本交叉验证还原;
  五样本指令块完全一致,仅主题不同;详见脚本内 TPL):
  - timeline 与 角色任务 之间是单个 \\n(无空行;与种子 topic_gen 一致,
    区别于 action_select 的 \\n\\n);
  - 三种演化关系 = 场景需求补全 / 兴趣因果递进 / 需求深度细化(评测原文);
  - "按时间顺序提取 5 步以内的行为链" ⇒ 本数据 gold 链 3-5 步(不做 6 步);
  - action 字段"须严格对应 Timeline 中交互条目,不允许省略;
    若同一节点合并了多条交互,用"；"分隔";
  - 输出 JSON: {"logic_chain": {"name": <主题原文>, "events": [
        {"date","action","logic"}]}}, name 与主题逐字一致;
  - logic 形态 = "逻辑关键词：逻辑说明"(官方案例:初始触发/认知深化/需求闭环);
  - prompt 尾 = 案例 JSON 的 "}" 后接 "\\n/no_think";输出空 think(全库不变量)。

⚠️ 数据源现实(与任务书原设想的差异,已核实):
  - UserProfile 无任何搜索 query 列(全 66 列核查过);
  - 种子 data/懂用户.jsonl 的 timeline 也**零** [搜索] 事件(grep 命中的"搜索"
    全部来自指令文本"搜索-搜索");[搜索] 事件仅存在于评测端 timeline。
  ⇒ "搜索词=主题锚"不可实现。改用**程序可判定**的替代锚(相关性全部可机检):
  S1 商品决策链: 同 s_a 前缀点击 → 同 s_a 另一商品点击 → 购买其中同一 pid 商品
     (锚=SID s_a 前缀相等 + 末步与上一步 SID 逐字相同;从 prompt 文本即可复核)
  S2 视频兴趣深化链: 同 tag_lv3 类目(Pid2Tag)、互动等级严格递增
     (浏览<长播<点赞/收藏<评论/转发<关注;等级由行为标注文本决定,NLI 可见)
  S3 广告转化链: 同 tag_lv3 广告 点击→(再点击/前置同类视频)→深度转化
  S4 跨域场景延展: 同 tag_lv2 视频观看 → 直播关注(场景需求补全)
  Pid2Tag schema: pid/domain(video\\/video|video\\/ad|live)/tag_lv3("一级-二级-三级");
  goods 无 tag ⇒ S1 主题用轮换的通用决策链措辞(≥8 种)。

logic 语句立场(C3 对治核心):只陈述 action 序列里字面可见的事实
  (行为类型、互动等级变化、同 SID/同前缀、时间先后),不编心理活动、不过度推断;
  措辞每角色 ≥8 种轮换。逻辑关键词用评测原文三关系 + 初始触发/需求闭环(官方案例词)。

timeline 渲染逐字复用 build_p3_quote_stop.py:
  【YYYY-MM-DD】/两空格+时刻/[域-行为]/商品直播 --:--/视频广告 HH:MM/
  ec_*_lag=相对 ec_time_ms 的天数偏移/live 用 author_id join。

用法:
  python scripts/data/build_p4_evolution.py build           # 训练 1500 + 验证 100
  python scripts/data/build_p4_evolution.py verify <file> [--audit <audit>]
seed=2026;train=UserProfile 分片 0-8,val=分片 9(用户零重叠),val seed=2027。
产物: data/processed/p4_evolution.jsonl / p4_evolution_val.jsonl
      (+ 同名 _audit.jsonl 供相关性程序复核)
════════════════════════════════════════════════════════════════════════
"""
import argparse
import datetime
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
HF = f"{ROOT}/data/hf_full/data"
OUT_TRAIN = f"{ROOT}/data/processed/p4_evolution.jsonl"
OUT_VAL = f"{ROOT}/data/processed/p4_evolution_val.jsonl"
SEED_USER_FILE = f"{ROOT}/data/懂用户.jsonl"

TZ = datetime.timezone(datetime.timedelta(hours=8))

DOMAIN_MAP = {"video/video": "video", "video/ad": "ad", "goods": "prod", "live": "living"}
VIDEO_ORDER = ["关注", "转发", "评论", "收藏", "点赞"]

SID_RE = re.compile(r"<\|(\w+)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
EVENT_LINE_RE = re.compile(r"^  (--:--|\d\d:\d\d) \[([^\]]+)\] (<\|\w+_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>)$")
DATE_LINE_RE = re.compile(r"^【(\d{4}-\d{2}-\d{2})】$")

# ──────────────────────── 评测模板(逐字还原,{T}=主题) ────────────────────────
# 还原方法:log 3892-3961 为 rich 控制台 80 列折行;按段落结构重接,
# 与 5 个评测样本做去换行字符级比对——全部一致(唯一差异是 rich 在
# 『购买"专业登山鞋"』折行处吞掉的 1 个空格,由其余 8 处 " → " 两侧带空格佐证)。
TPL = '''角色任务：你是一名严格遵循标准的用户行为数据挖掘与数据格式化专家。请根据以上用户交互历史，针对给定主题，提取具有高阶逻辑演进与深度意图关联的交互行为链路。

核心提取逻辑：
按时间顺序提取 5 步以内的行为链，要求后项交互 B 必须与前项交互 A 构成场景需求补全、兴趣因果递进或需求深度细化的深度演化关系。严禁提取缺乏实质性逻辑关联的浅层并列行为，如品类平级罗列、相似内容重复、无关联交互等。
1. 场景需求补全：交互 B 与交互 A 必须在同一生活场景、主题类目下具有极强的需求补全关系。有效案例：购买"专业登山鞋" → 搜索"高原紫外线防护" → 点击"户外硬壳冲锋衣"。原因：户外极限场景的装备补齐。无效案例：点击"男鞋" → 购买"棉拖" → 购买"羽绒服"。原因：仅为冬季杂物凑单，应予剔除。
2. 兴趣因果递进：交互 B 是因交互 A 产生的副作用或新需求，或 A 是 B 的诱因。有效案例：搜索"全屋定制" → 观看"甲醛危害科普" → 搜索"工业级空气净化器"。原因：装修行为触发健康焦虑。
3. 需求深度细化：在同一主题类目下，交互 B 的需求比交互 A 更具体详细。有效案例：泛化搜索"新手露营" → 观看"黑胶帐篷测评" → 点击"三峰出征服者帐篷"。

要求与约束：
1. 节点精炼：每个 event 节点对应一个核心交互步骤，严禁重复交互内容；同一日期内属于同一演进步骤的同类型交互（如连续多条相关搜索）可合并为一个节点，用"；"分隔，但不同演进步骤的交互不得合并；
2. 逻辑溯源：logic 字段必须严格以 action 交互内容为依据，避免缺乏依据的过度推断；

请针对以下主题提取行为逻辑链：
主题：{T}

输出格式：请以 JSON 对象形式返回。其中 action 字段须严格对应 Timeline 中交互条目，不允许省略；若同一节点合并了多条交互，用"；"分隔。结构如下：
{{
  "logic_chain": {{
    "name": "{T}",
    "events": [
      {{
        "date": "YYYY-MM-DD",
        "action": "[交互行为类型] 交互内容",
        "logic": "逻辑关键词：逻辑说明"
      }}
    ]
  }}
}}
有效逻辑链案例（注意：以下案例来自其他用户，仅供参考输出格式和逻辑标准，与上述用户交互历史无关）：
{{
  "logic_chain": {{
    "name": "熬夜健康焦虑驱动需求演化链",
    "events": [
      {{
        "date": "2026-02-10",
        "action": "[搜索] 长期熬夜心慌怎么办；[搜索] 熬夜后心慌胸闷",
        "logic": "初始触发：基于生理不适的泛化求助。"
      }},
      {{
        "date": "2026-02-11",
        "action": "[视频-长播] <|video_begin|><s_a_6705><s_b_713><s_c_5747>",
        "logic": "认知深化：意识到生理症状可能与心理/神经因素有关，开始进行症状鉴别。"
      }},
      {{
        "date": "2026-02-12",
        "action": "[广告-点击] <|ad_begin|><s_a_6638><s_b_1822><s_c_7257>",
        "logic": "需求闭环：由认知深化转向针对性营养补剂购买。"
      }}
    ]
  }}
}}
/no_think'''


# ──────────────────── Pass 1: 用户事件收集(逐字复用 P3) ────────────────────

NEEDED_COLS = [
    "ec_good_click_item_id_list_extend", "ec_trunc_clk_lag",
    "ec_good_order_item_id_list_extend", "ec_trunc_buy_lag", "ec_time_ms",
    "video_history_sampled_pid_list", "video_history_ts_list",
    "video_history_like_list", "video_history_comment_list",
    "video_history_forward_list", "video_history_collect_list",
    "video_history_play_done_list", "video_history_watch_time_list",
    "live_hist_author_id_list", "live_hist_timestamp_list",
    "outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts",
    "outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts",
]


def _L(v):
    try:
        return len(v) if v is not None and hasattr(v, "__len__") and not isinstance(v, str) else 0
    except Exception:
        return 0


def _ok_ts(ms):
    return 1640966400000 <= ms <= 1798732800000  # 2022-01-01 .. 2026-12-31


def video_label(like, fwd, cm, col, long_play):
    parts = []
    for name, flag in zip(VIDEO_ORDER, [False, fwd, cm, col, like]):
        if flag:
            parts.append(name)
    parts.append("长播" if long_play else "浏览")
    return "视频-" + "/".join(parts)


def collect_raw_events(row):
    """→ [(dom, pid, ts_ms, action)];与 build_p3_quote_stop.collect_raw_events 逐字一致。"""
    ev = []

    def add(dom, pid, ts, action):
        if pid is not None and ts and _ok_ts(int(ts)):
            ev.append((dom, int(pid), int(ts), action))

    anchor = row.get("ec_time_ms")
    try:
        anchor = int(anchor) if anchor is not None and anchor == anchor else 0
    except Exception:
        anchor = 0
    if anchor:
        for pid_col, lag_col, act in [
            ("ec_good_click_item_id_list_extend", "ec_trunc_clk_lag", "商品-点击"),
            ("ec_good_order_item_id_list_extend", "ec_trunc_buy_lag", "商品-购买"),
        ]:
            pids, lags = row.get(pid_col), row.get(lag_col)
            n = min(_L(pids), _L(lags))
            for i in range(max(0, n - 250), n):
                add("prod", pids[i], anchor - int(lags[i]) * 86400000, act)
    vp = row.get("video_history_sampled_pid_list")
    if _L(vp):
        ts = row.get("video_history_ts_list")
        lk, cm = row.get("video_history_like_list"), row.get("video_history_comment_list")
        fw, cl = row.get("video_history_forward_list"), row.get("video_history_collect_list")
        pdn, wt = row.get("video_history_play_done_list"), row.get("video_history_watch_time_list")

        def g(a, i):
            return bool(a[i]) if _L(a) > i else False

        n = _L(vp)
        for i in range(max(0, n - 250), n):
            if _L(ts) <= i or not ts[i]:
                continue
            long_play = g(pdn, i) or (_L(wt) > i and wt[i] is not None and wt[i] >= 30)
            add("video", vp[i], ts[i], video_label(g(lk, i), g(fw, i), g(cm, i), g(cl, i), long_play))
    la, lts = row.get("live_hist_author_id_list"), row.get("live_hist_timestamp_list")
    if _L(la):
        n = _L(la)
        for i in range(max(0, n - 120), n):
            if _L(lts) <= i or not lts[i]:
                continue
            s = str(lts[i])[:10].replace("-", "")[:8]
            try:
                t = int(datetime.datetime.strptime(s, "%Y%m%d").replace(tzinfo=TZ).timestamp() * 1000)
            except Exception:
                continue
            add("living", la[i], t, "直播-关注")
    for pid_col, ts_col, act in [
        ("outer_loop_history_action_pid_list_click", "outer_loop_history_action_pid_list_click_ts", "广告-点击"),
        ("outer_loop_deep_target_pid", "outer_loop_deep_target_pid_ts", "广告-深度转化"),
    ]:
        pids, ts = row.get(pid_col), row.get(ts_col)
        n = min(_L(pids), _L(ts))
        for i in range(n):
            add("ad", pids[i], ts[i], act)
    ev.sort(key=lambda e: e[2])
    return ev


def scan_users(shards, max_users, min_raw=25):
    users = []
    for sh in shards:
        df = pd.read_parquet(sh, columns=NEEDED_COLS)
        cols = {c: df[c].to_numpy() for c in df.columns}
        nrows = len(df)
        del df
        for i in range(nrows):
            row = {c: cols[c][i] for c in cols}
            ev = collect_raw_events(row)
            if len(ev) >= min_raw:
                users.append(ev)
            if len(users) >= max_users:
                return users
        print(f"  [pass1] {os.path.basename(sh)} done, users={len(users)}", file=sys.stderr)
    return users


# ──────────────────── Pass 2: pid→sid / pid→tag join ────────────────────

def build_sid_map(needed_keys):
    needed_pids = {p for _, p in needed_keys}
    sid_map = {}
    shards = sorted(glob.glob(f"{HF}/OneReason_Pid2Sid/*.parquet"))
    for j, sh in enumerate(shards):
        df = pd.read_parquet(sh, columns=["pid", "domain", "sid_three"])
        df = df[df["pid"].isin(needed_pids) & df["sid_three"].notna()]
        for p, d, s in zip(df["pid"], df["domain"], df["sid_three"]):
            k = (DOMAIN_MAP.get(d, d), int(p))
            if k in needed_keys and k not in sid_map:
                sid_map[k] = (int(s[0]), int(s[1]), int(s[2]))
        if (j + 1) % 40 == 0:
            print(f"  [pass2] sid {j+1}/{len(shards)}, mapped={len(sid_map)}", file=sys.stderr)
    return sid_map


def build_tag_map(needed_keys):
    """needed_keys: set[(dom,pid)], dom∈{video,ad,living} → dict[(dom,pid)]=tag_lv3。"""
    needed_pids = {p for _, p in needed_keys}
    tag_map = {}
    shards = sorted(glob.glob(f"{HF}/OneReason_Pid2Tag/*.parquet"))
    for j, sh in enumerate(shards):
        df = pd.read_parquet(sh)
        df = df[df["pid"].isin(needed_pids) & df["tag_lv3"].notna()]
        for p, d, t in zip(df["pid"], df["domain"], df["tag_lv3"]):
            k = (DOMAIN_MAP.get(d, d), int(p))
            if k in needed_keys and k not in tag_map and isinstance(t, str) and t.count("-") >= 1:
                tag_map[k] = t
        if (j + 1) % 10 == 0:
            print(f"  [pass2] tag {j+1}/{len(shards)}, mapped={len(tag_map)}", file=sys.stderr)
    return tag_map


# ──────────────────── 渲染(逐字复用 P3) ────────────────────

def render_events(raw_events, sid_map, tag_map):
    out = []
    for dom, pid, ts, action in raw_events:
        sid = sid_map.get((dom, pid))
        if sid is None:
            continue
        dt = datetime.datetime.fromtimestamp(ts / 1000, TZ)
        hhmm = dt.strftime("%H:%M") if dom in ("video", "ad") else "--:--"
        out.append({
            "date": dt.strftime("%Y-%m-%d"), "hhmm": hhmm, "action": action, "dom": dom,
            "token": f"<|{dom}_begin|><s_a_{sid[0]}><s_b_{sid[1]}><s_c_{sid[2]}>", "ts": ts,
            "pid": pid, "tag": tag_map.get((dom, pid)),
        })
    return out


def render_timeline(events):
    lines = ["【用户交互历史】："]
    cur = None
    for e in events:
        if e["date"] != cur:
            cur = e["date"]
            lines.append(f"【{cur}】")
        lines.append(f"  {e['hhmm']} [{e['action']}] {e['token']}")
    return "\n".join(lines)


def s_a_of(token):
    return int(SID_RE.match(token).group(2))


# ──────────────────── 主题措辞库(每层 ≥8 种) ────────────────────

TOPICS_S1 = [
    "从同类商品的浏览比较到特定单品购买的决策收敛",
    "同类商品下从多次点击比较到最终下单的决策演化",
    "从泛化的同类商品点击到锁定单品完成购买的需求细化",
    "同一类目商品从浏览筛选走向购买落地的决策链",
    "从同类商品比较式浏览到目标单品下单的兴趣聚焦",
    "同类商品的点击探索逐步收敛为特定商品的购买决策",
    "从同类候选商品的反复查看到单一商品成交的选择收敛",
    "同一商品类目内从初步点击到复看下单的购买决策深化",
]
TOPICS_S2 = [
    "围绕「{K}」内容的兴趣从浏览到深度互动的演化",
    "对「{K}」内容的关注与互动逐步深化",
    "「{K}」主题下从观看到互动升级的行为递进",
    "对「{K}」类内容从初次接触到深度参与的兴趣养成",
    "围绕「{K}」的内容消费投入逐层加深",
    "「{K}」兴趣从泛化浏览走向深度互动的演化链",
    "从偶然观看到主动互动：对「{K}」内容的兴趣深化",
    "「{K}」相关视频从观看行为到互动行为的需求递进",
]
TOPICS_S3 = [
    "从「{K}」相关广告点击到深度转化的需求闭环",
    "「{K}」主题广告从初次点击到深度转化的兴趣递进",
    "对「{K}」类广告的兴趣从点击试探走向深度转化",
    "围绕「{K}」的广告交互从浏览点击到转化落地",
    "「{K}」广告兴趣沿点击到深度转化的路径收敛",
    "从接触「{K}」广告到完成深度转化的决策演化",
    "「{K}」相关推广内容从点击到深度转化的行为闭环",
    "对「{K}」广告由浅层点击递进至深度转化的需求演化",
]
TOPICS_S4 = [
    "从「{K}」视频观看到同主题直播关注的场景延展",
    "「{K}」兴趣由短视频消费补全至直播场景",
    "围绕「{K}」从视频内容到直播互动的场景需求补全",
    "对「{K}」主题的关注从视频观看扩展到直播关注",
    "「{K}」内容兴趣自视频域延伸至直播域的场景演化",
    "从观看「{K}」相关视频到关注同主题主播的行为延展",
    "「{K}」主题下视频消费向直播关注的跨场景递进",
    "由「{K}」视频兴趣引出的直播关注场景补全",
]


# ──────────────────── logic 措辞库(只陈述可见事实) ────────────────────
# 结构 = "关键词：{act}，{rel}。"  act/rel 均为事实句;关键词取评测原文三关系
# + 官方案例词(初始触发/需求闭环)。每个角色 ≥8 种组合。

FIRST_KW = ["初始触发", "初始触发", "兴趣起点", "初始触发"]
FIRST_REL = [
    "构成该主题下的起点交互",
    "是这条演化链的第一步",
    "出现与主题相关的首次交互",
    "主题相关行为由此开始",
    "形成后续行为的直接起点",
    "该主题下的行为序列自此展开",
    "属于链条的起始节点",
    "为后续演进提供了起点",
]
DEEPEN_PROD_ACT = [
    "又点击了一款同类商品",
    "点击了同一类目下的另一款商品",
    "继续点击查看同类的另一件商品",
    "再度点击浏览了一款同类商品",
]
DEEPEN_PROD_REL = [
    "两次点击的商品 SID 共享相同的 <s_a_{A}> 前缀，浏览范围仍在同一类目内并进一步聚焦",
    "该商品与上一步点击的商品同属 <s_a_{A}> 类目前缀，比较与筛选的对象更加具体",
    "SID 前缀 <s_a_{A}> 与前一步一致，需求仍锁定在同一类目且候选进一步收窄",
    "与上一步商品同为 <s_a_{A}> 前缀的同类商品，点击目标由泛化转向具体",
    "同类目（SID 前缀 <s_a_{A}>）内的再次点击，说明筛选在该类目下持续细化",
    "点击对象与前一步同属 <s_a_{A}> 前缀类目，浏览从初探推进到比较",
    "仍是 <s_a_{A}> 前缀类目下的商品，关注范围较上一步进一步收敛",
    "商品 SID 前缀与上一步相同（<s_a_{A}>），类目内的挑选更趋具体",
]
CLOSE_BUY_ACT = [
    "下单购买了此前点击过的那款商品",
    "购买了上一步点击的同一款商品",
    "对此前点击的商品完成了购买",
    "将上一步点击的商品下单买入",
]
CLOSE_BUY_REL = [
    "购买条目的 SID 与上一步点击条目完全一致，点击兴趣直接落地为购买行为",
    "所购商品 SID 与前一步点击逐字相同，浏览比较在此收敛为成交",
    "该购买与上一步点击指向同一 SID，由浏览引出的需求在此闭环",
    "SID 与上一步点击完全相同，说明购买由该次点击直接递进而来",
    "购买对象即上一步点击对象（SID 一致），决策链在此完成闭环",
    "与前一步点击的 SID 逐字一致，点击行为的意图最终转化为下单",
    "同一 SID 由点击推进到购买，构成从筛选到成交的因果递进",
    "购买的正是上一步点击的商品（SID 完全一致），需求就此落地",
]
VIDEO_LEVEL_DESC = {0: "浏览", 1: "长播观看", 2: "点赞/收藏级互动", 3: "评论/转发级互动", 4: "关注发布者"}
DEEPEN_VIDEO_REL = [
    "行为标注由上一步的“{p}”变为“{c}”，对该主题内容的投入程度提高",
    "相比上一步的“{p}”，本次出现“{c}”，互动层级上升",
    "从“{p}”推进到“{c}”，同主题下的参与深度增加",
    "标注显示互动由“{p}”升级为“{c}”，兴趣表达更进一步",
    "上一步为“{p}”，本次为“{c}”，行为强度在同主题内递增",
    "互动等级从“{p}”提高到“{c}”，主题兴趣持续深化",
    "本次交互的“{c}”较上一步的“{p}”投入更深，构成递进",
    "由“{p}”到“{c}”的变化表明该主题下的行为持续升级",
]
AD_CLICK2_ACT = [
    "点击了同一主题下的另一条广告",
    "再次点击了同主题的不同广告",
    "又点击了一条同主题广告",
    "继续点击同一主题类目下的另一条广告",
]
AD_CLICK2_REL = [
    "两条广告属于同一主题类目且素材不同，兴趣在同一主题内延续并加深",
    "与上一步不同的广告条目、相同的主题类目，说明关注点持续停留在该主题",
    "同主题下的第二次点击，兴趣由偶发接触转向重复关注",
    "点击对象换为同主题另一条目，主题兴趣得到再次确认",
    "同一主题类目内的再次点击，行为频次表明兴趣递进",
    "不同条目、同一主题的连续点击，构成主题内的需求细化",
    "再次点击同主题广告，表明上一次点击并非孤立行为",
    "同主题的重复点击行为，主题相关意图更加明确",
]
CLOSE_CONV_ACT = [
    "在同主题广告上完成了深度转化",
    "对同主题广告产生了深度转化行为",
    "完成了该主题广告的深度转化",
    "在该主题的广告条目上发生深度转化",
]
CLOSE_CONV_REL = [
    "行为类型由点击升级为深度转化，该主题下的需求在此闭环",
    "从广告点击递进到深度转化，兴趣转化为更深层的行为投入",
    "深度转化较此前的点击是更高强度的交互，主题需求就此落地",
    "由点击到深度转化的类型变化，表明该主题兴趣完成了递进闭环",
    "此前的点击行为在此升级为深度转化，链条到达终点",
    "交互强度从点击提高到深度转化，需求演化完成收口",
    "深度转化承接前序点击，构成该主题下的因果递进终点",
    "该主题的交互至此由浏览点击转化为深度行为，形成闭环",
]
S4_LIVE_ACT = [
    "关注了同主题的直播主播",
    "对同一主题类目的主播进行了直播关注",
    "在直播域关注了该主题下的主播",
    "关注了与该主题同类目的直播间主播",
]
S4_LIVE_REL = [
    "交互从视频域扩展到直播域、主题类目不变，补全了同一主题场景下的另一类交互",
    "与前序视频交互同属一个主题类目，行为由观看视频延伸为关注主播",
    "同一主题场景由视频消费补进直播关注，场景需求得到补全",
    "直播关注与此前的视频观看同属一个主题，交互形态跨域延展",
    "主题不变、域从视频切换到直播，构成场景内的需求补全",
    "在同主题下新增直播域交互，与前序视频行为形成场景互补",
    "该直播关注延续前序视频的主题类目，补齐了直播场景的交互",
    "视频域兴趣在直播域得到承接，同主题场景就此补全",
]


def video_act_desc(label):
    body = label[len("视频-"):]
    parts = body.split("/")
    tail = "较长时间观看了一条该主题视频" if parts[-1] == "长播" else "浏览了一条该主题视频"
    inter = [p for p in parts[:-1]]
    if not inter:
        return tail
    if "关注" in inter:
        rest = [p for p in inter if p != "关注"]
        s = "观看该主题视频后关注了发布者"
        if rest:
            s += "，并进行了" + "、".join(rest)
        return s
    return tail.replace("观看了", "观看并" + "、".join(inter) + "了")


def video_rank(label):
    parts = label[len("视频-"):].split("/")
    inter = set(parts[:-1])
    if "关注" in inter:
        return 4
    if inter & {"评论", "转发"}:
        return 3
    if inter & {"点赞", "收藏"}:
        return 2
    return 1 if parts[-1] == "长播" else 0


def first_act_desc(e):
    if e["action"] == "商品-点击":
        return "点击浏览了一款商品"
    if e["action"] == "广告-点击":
        return "点击了一条该主题下的广告"
    if e["dom"] == "video":
        return video_act_desc(e["action"])
    if e["action"] == "直播-关注":
        return "关注了该主题下的直播主播"
    return "发生了一次该主题相关交互"


# ──────────────────── 链构造(相关性全部程序可判定) ────────────────────

def find_s1(events, rng):
    """同 s_a 点击 → 同 s_a 另一商品点击(可并) → 购买同一 pid。dates 严格递增。"""
    clicks = [e for e in events if e["action"] == "商品-点击"]
    buys = [e for e in events if e["action"] == "商品-购买"]
    by_pid = defaultdict(list)
    for e in clicks:
        by_pid[e["pid"]].append(e)
    rng.shuffle(buys)
    for b in buys:
        # 同 pid ⇒ 同 SID token(sid_map 按 (dom,pid) 查表);购买与点击 token 逐字一致
        pcs = [c for c in by_pid.get(b["pid"], []) if c["date"] < b["date"]]
        if not pcs:
            continue
        pc = max(pcs, key=lambda c: c["date"])
        A = s_a_of(b["token"])
        p0s = [c for c in clicks
               if c["pid"] != b["pid"] and s_a_of(c["token"]) == A and c["date"] < pc["date"]
               and c["token"] != pc["token"] and c["token"] != b["token"]]
        if not p0s:
            continue
        p0 = min(p0s, key=lambda c: c["date"])
        nodes = [["first", [p0]], ["deepen_prod", [pc]], ["close_buy", [b]]]
        # 可选:p0 同日同 s_a 另一 pid 的点击 → 合并节点(教"；"合并格式)
        mates = [c for c in clicks
                 if c["date"] == p0["date"] and c["pid"] not in (p0["pid"], pc["pid"], b["pid"])
                 and s_a_of(c["token"]) == A and c["token"] != p0["token"]]
        if mates and rng.random() < 0.5:
            nodes[0][1].append(mates[0])
        # 可选:中间再插一步同 s_a 点击 → 4 步链
        mids = [c for c in clicks
                if p0["date"] < c["date"] < pc["date"] and s_a_of(c["token"]) == A
                and c["pid"] not in (p0["pid"], pc["pid"], b["pid"])
                and all(c["token"] != x["token"] for _, xs in nodes for x in xs)]
        if mids and rng.random() < 0.45:
            nodes.insert(1, ["deepen_prod", [min(mids, key=lambda c: c["date"])]])
        return "S1", ("s_a", A), nodes
    return None


def find_s2(events, rng):
    """同 tag_lv3 视频、互动等级严格递增、日期严格递增,3-5 步。"""
    groups = defaultdict(list)
    for e in events:
        if e["dom"] == "video" and e["tag"]:
            groups[e["tag"]].append(e)
    cands = list(groups.items())
    rng.shuffle(cands)
    for tag, evs in cands:
        evs = sorted(evs, key=lambda e: e["ts"])
        chain = []
        for e in evs:
            r = video_rank(e["action"])
            if not chain:
                if r <= 1:
                    chain = [e]
                continue
            if r > video_rank(chain[-1]["action"]) and e["date"] > chain[-1]["date"] \
                    and all(e["token"] != x["token"] for x in chain):
                chain.append(e)
        if len(chain) >= 3:
            nodes = [["first", [chain[0]]]] + [["deepen_video", [e]] for e in chain[1:5]]
            return "S2", ("tag_lv3", tag), nodes
    return None


def find_s3(events, rng):
    """同 tag_lv3 广告:点击→(再点击|前置同 tag 视频)→深度转化。"""
    ad_groups = defaultdict(list)
    for e in events:
        if e["dom"] == "ad" and e["tag"]:
            ad_groups[e["tag"]].append(e)
    cands = list(ad_groups.items())
    rng.shuffle(cands)
    for tag, evs in cands:
        convs = [e for e in evs if e["action"] == "广告-深度转化"]
        clicks = sorted([e for e in evs if e["action"] == "广告-点击"], key=lambda e: e["ts"])
        for cv in convs:
            pre = [c for c in clicks if c["date"] < cv["date"] and c["token"] != cv["token"]]
            dates = sorted({c["date"] for c in pre})
            if len(dates) >= 2:
                c1 = min(pre, key=lambda c: c["date"])
                c2s = [c for c in pre if c["date"] > c1["date"] and c["pid"] != c1["pid"]
                       and c["token"] != c1["token"]]
                if not c2s:
                    continue
                c2 = min(c2s, key=lambda c: c["date"])
                return "S3", ("tag_lv3", tag), [["first", [c1]], ["ad_click2", [c2]], ["close_conv", [cv]]]
            if len(dates) == 1:
                c1 = pre[0]
                vids = [e for e in events if e["dom"] == "video" and e["tag"] == tag
                        and e["date"] < c1["date"]]
                if vids:
                    v = min(vids, key=lambda e: e["ts"])
                    return "S3", ("tag_lv3", tag), [["first", [v]], ["ad_click2v", [c1]], ["close_conv", [cv]]]
    return None


def find_s4(events, rng):
    """同 tag_lv2 视频×2(等级递增)→ 直播关注。"""
    lives = [e for e in events if e["dom"] == "living" and e["tag"]]
    rng.shuffle(lives)
    for lv in lives:
        lv2 = "-".join(lv["tag"].split("-")[:2])
        vids = sorted([e for e in events if e["dom"] == "video" and e["tag"]
                       and "-".join(e["tag"].split("-")[:2]) == lv2 and e["date"] < lv["date"]],
                      key=lambda e: e["ts"])
        if len(vids) < 2:
            continue
        v1 = vids[0]
        v2s = [v for v in vids[1:] if v["date"] > v1["date"] and video_rank(v["action"]) > video_rank(v1["action"])
               and v["token"] != v1["token"]]
        if not v2s:
            continue
        v2 = v2s[0]
        if lv["date"] <= v2["date"]:
            continue
        return "S4", ("tag_lv2", lv2), [["first", [v1]], ["deepen_video", [v2]], ["s4_live", [lv]]]
    return None


FINDERS = {"S1": find_s1, "S2": find_s2, "S3": find_s3, "S4": find_s4}
QUOTAS = {"S1": 550, "S2": 550, "S3": 250, "S4": 150}


# ──────────────────── 组装 ────────────────────

def make_topic(stratum, anchor, rng):
    if stratum == "S1":
        return rng.choice(TOPICS_S1)
    K = anchor[1].split("-")[-1]
    bank = {"S2": TOPICS_S2, "S3": TOPICS_S3, "S4": TOPICS_S4}[stratum]
    return rng.choice(bank).format(K=K)


def make_logic(stratum, role, node_evs, prev_evs, anchor, rng):
    e = node_evs[0]
    if role == "first":
        kw = rng.choice(FIRST_KW)
        act = first_act_desc(e)
        if len(node_evs) > 1 and e["action"] == "商品-点击":
            act = "同日连续点击浏览了两款同类商品"
        return f"{kw}：{act}，{rng.choice(FIRST_REL)}。"
    if role == "deepen_prod":
        A = anchor[1]
        return f"需求深度细化：{rng.choice(DEEPEN_PROD_ACT)}，{rng.choice(DEEPEN_PROD_REL).format(A=A)}。"
    if role == "close_buy":
        return f"需求闭环：{rng.choice(CLOSE_BUY_ACT)}，{rng.choice(CLOSE_BUY_REL)}。"
    if role == "deepen_video":
        p = VIDEO_LEVEL_DESC[video_rank(prev_evs[0]["action"])]
        c = VIDEO_LEVEL_DESC[video_rank(e["action"])]
        return f"需求深度细化：{video_act_desc(e['action'])}，{rng.choice(DEEPEN_VIDEO_REL).format(p=p, c=c)}。"
    if role == "ad_click2":
        return f"兴趣因果递进：{rng.choice(AD_CLICK2_ACT)}，{rng.choice(AD_CLICK2_REL)}。"
    if role == "ad_click2v":
        return f"兴趣因果递进：点击了一条该主题下的广告，在此前同主题视频观看之后出现，兴趣由内容浏览转向广告条目。"
    if role == "close_conv":
        return f"需求闭环：{rng.choice(CLOSE_CONV_ACT)}，{rng.choice(CLOSE_CONV_REL)}。"
    if role == "s4_live":
        return f"场景需求补全：{rng.choice(S4_LIVE_ACT)}，{rng.choice(S4_LIVE_REL)}。"
    raise ValueError(role)


def build_window(events, chain_evs, rng, max_w=140, min_w=40):
    """先全局稀释非链 video(P3 规则),再取含链连续窗口;链跨度过大时对窗内
    非链事件均匀抽稀——保证链事件全保留、窗口 ≤ max_w、干扰事件 ≥10。"""
    chain_ids = {id(e) for e in chain_evs}
    nv = [e for e in events if e["dom"] != "video" or id(e) in chain_ids]
    vv = [e for e in events if e["dom"] == "video" and id(e) not in chain_ids]
    cap = max(10, 2 * len([e for e in nv if id(e) not in chain_ids]))
    if len(vv) > cap:
        step = len(vv) / cap
        vv = [vv[int(i * step)] for i in range(cap)]
    evs = sorted(nv + vv, key=lambda e: e["ts"])
    idxs = [i for i, e in enumerate(evs) if id(e) in chain_ids]
    lo, hi = min(idxs), max(idxs)
    if hi - lo + 1 > max_w:
        span = evs[lo:hi + 1]
        keep = [e for e in span if id(e) in chain_ids]
        rest = [e for e in span if id(e) not in chain_ids]
        budget = max_w - len(keep)
        if budget < 10:
            return None
        step = len(rest) / budget
        rest = [rest[int(i * step)] for i in range(min(budget, len(rest)))]
        win = sorted(keep + rest, key=lambda e: e["ts"])
    else:
        w = min(max_w, max(min_w, hi - lo + 1 + rng.randint(10, 40)))
        extra = w - (hi - lo + 1)
        a = max(0, lo - rng.randint(0, extra))
        b = min(len(evs), a + w)
        a = max(0, b - w)
        win = evs[a:b]
    if len(win) < 25 or len(win) - len(chain_evs) < 10:
        return None
    return win


def assemble(win, topic, nodes, stratum, anchor, rng):
    gold_events = []
    prev = None
    for role, evs in nodes:
        action = "；".join(f"[{e['action']}] {e['token']}" for e in evs)
        gold_events.append({
            "date": evs[0]["date"],
            "action": action,
            "logic": make_logic(stratum, role, evs, prev, anchor, rng),
        })
        prev = evs
    gold = {"logic_chain": {"name": topic, "events": gold_events}}
    rec = {
        "instruction": "",
        "input": render_timeline(win) + "\n" + TPL.format(T=topic),
        "output": "<think>\n</think>\n" + json.dumps(gold, ensure_ascii=False),
        "history": [],
    }
    audit = {
        "stratum": stratum,
        "anchor": {"type": anchor[0], "value": anchor[1]},
        "nodes": [{"date": evs[0]["date"], "role": role,
                   "tokens": [e["token"] for e in evs],
                   "actions": [e["action"] for e in evs],
                   "pids": [e["pid"] for e in evs],
                   "tags": [e["tag"] for e in evs]} for role, evs in nodes],
    }
    return rec, audit


def build_split(user_shards, n_target, seed, out_path):
    rng = random.Random(seed)
    print(f"[pass1] scanning users for {out_path} ...", file=sys.stderr)
    users = scan_users(user_shards, max_users=int(n_target * 12))
    print(f"[pass1] {len(users)} candidate users", file=sys.stderr)

    needed = {(dom, pid) for ev in users for dom, pid, _, _ in ev}
    print(f"[pass2] joining {len(needed):,} keys against Pid2Sid ...", file=sys.stderr)
    sid_map = build_sid_map(needed)
    print(f"[pass2] sid mapped {len(sid_map):,}/{len(needed):,}", file=sys.stderr)
    tag_needed = {k for k in needed if k[0] in ("video", "ad", "living")}
    print(f"[pass2] joining {len(tag_needed):,} keys against Pid2Tag ...", file=sys.stderr)
    tag_map = build_tag_map(tag_needed)
    print(f"[pass2] tag mapped {len(tag_map):,}/{len(tag_needed):,}", file=sys.stderr)

    scale = n_target / sum(QUOTAS.values())
    quotas = {k: max(1, round(v * scale)) for k, v in QUOTAS.items()}
    recs, audits, stats = [], [], Counter()
    seen_inputs = set()
    for ev_raw in users:
        if len(recs) >= n_target:
            break
        events = render_events(ev_raw, sid_map, tag_map)
        if len(events) < 25:
            stats["skip_short_after_join"] += 1
            continue
        made = None
        order = sorted(quotas, key=lambda k: -quotas[k])
        for st in order:
            if quotas[st] <= 0:
                continue
            found = FINDERS[st](events, rng)
            if not found:
                continue
            stratum, anchor, nodes = found
            chain_evs = [e for _, evs in nodes for e in evs]
            win = build_window(events, chain_evs, rng)
            if win is None:
                stats[f"skip_window:{st}"] += 1
                continue
            topic = make_topic(stratum, anchor, rng)
            rec, audit = assemble(win, topic, nodes, stratum, anchor, rng)
            if rec["input"] in seen_inputs:
                stats["skip_dup_input"] += 1
                continue
            made = (rec, audit, stratum)
            break
        if not made:
            # 兜底:配额外任取一个可构造层(保证总量,分布如实上报)
            for st in ("S1", "S2", "S3", "S4"):
                found = FINDERS[st](events, rng)
                if not found:
                    continue
                stratum, anchor, nodes = found
                chain_evs = [e for _, evs in nodes for e in evs]
                win = build_window(events, chain_evs, rng)
                if win is None:
                    continue
                topic = make_topic(stratum, anchor, rng)
                rec, audit = assemble(win, topic, nodes, stratum, anchor, rng)
                if rec["input"] in seen_inputs:
                    continue
                made = (rec, audit, stratum)
                stats[f"fallback:{st}"] += 1
                break
        if not made:
            stats["skip_no_chain"] += 1
            continue
        rec, audit, stratum = made
        if quotas.get(stratum, 0) > 0:
            quotas[stratum] -= 1
        stats[f"stratum:{stratum}"] += 1
        seen_inputs.add(rec["input"])
        recs.append(rec)
        audits.append(audit)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ap = out_path.replace(".jsonl", "_audit.jsonl")
    with open(ap, "w", encoding="utf-8") as f:
        for a in audits:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[done] {len(recs)} -> {out_path} (+audit)", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}", file=sys.stderr)
    return recs


# ──────────────────── verify:全量程序复核 ────────────────────

def parse_prompt(inp):
    """→ (timeline_entries, topic, tpl_ok)。timeline_entries=[(date,'[label] token')]"""
    lines = inp.split("\n")
    entries, cur = [], None
    i = 0
    assert lines[0] == "【用户交互历史】："
    for i in range(1, len(lines)):
        l = lines[i]
        m = DATE_LINE_RE.match(l)
        if m:
            cur = m.group(1)
            continue
        m = EVENT_LINE_RE.match(l)
        if m:
            entries.append((cur, f"[{m.group(2)}] {m.group(3)}"))
            continue
        break  # 第一行非时间线行 = 指令块开始
    instr = "\n".join(lines[i:])
    mt = re.search(r"^主题：(.+)$", instr, re.M)
    topic = mt.group(1)
    tpl_ok = instr == TPL.format(T=topic)
    return entries, topic, tpl_ok


def verify(path, audit_path=None, n_rel_sample=200):
    seed_prompts, seed_resps = set(), set()
    if os.path.exists(SEED_USER_FILE):
        for line in open(SEED_USER_FILE, encoding="utf-8"):
            for r in json.loads(line):
                seed_prompts.add(r["prompt"])
                seed_resps.add(r["response"])
    audits = None
    if audit_path and os.path.exists(audit_path):
        audits = [json.loads(l) for l in open(audit_path, encoding="utf-8")]

    n = 0
    ok = Counter()
    fails = []
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng = random.Random(42)
    rel_idx = set(rng.sample(range(len(recs)), min(n_rel_sample, len(recs))))
    for i, r in enumerate(recs):
        n += 1
        inp, out = r["input"], r["output"]
        f = []
        # 1 不变量:/no_think 结尾 ⇔ 空 think 起头
        if not (inp.endswith("/no_think") and out.startswith("<think>\n</think>\n")):
            f.append("invariant")
        # 2 模板逐字 + JSON 结构
        entries, topic, tpl_ok = parse_prompt(inp)
        if not tpl_ok:
            f.append("template")
        try:
            gold = json.loads(out.split("</think>\n", 1)[1])
            lc = gold["logic_chain"]
            assert set(gold.keys()) == {"logic_chain"} and set(lc.keys()) == {"name", "events"}
            evs = lc["events"]
            assert 3 <= len(evs) <= 5
            assert all(set(e.keys()) == {"date", "action", "logic"} for e in evs)
            assert lc["name"] == topic
            assert all(re.match(r"^\d{4}-\d{2}-\d{2}$", e["date"]) for e in evs)
            assert all(re.match(r"^[^：]{2,6}：.+。$", e["logic"]) for e in evs)
        except Exception:
            f.append("structure")
            evs = []
        # 3 action 逐字节 ∈ timeline 且 date 绑定正确
        eset = set(entries)
        for e in evs:
            for part in e["action"].split("；"):
                if (e["date"], part) not in eset:
                    f.append("quote")
                    break
        # 4 日期严格升序
        ds = [e["date"] for e in evs]
        if ds != sorted(ds) or len(set(ds)) != len(ds):
            f.append("dates")
        # 5 相关性程序复核(S1 全量可从 prompt 文本复核;tag 类比对 audit)
        if i in rel_idx and evs:
            au = audits[i] if audits else None
            toks = [SID_RE.search(p).group(0) for e in evs for p in e["action"].split("；")]
            if au:
                au_toks = [t for nd in au["nodes"] for t in nd["tokens"]]
                if toks != au_toks:
                    f.append("rel_audit_mismatch")
                st, anc = au["stratum"], au["anchor"]
                if st == "S1":
                    sas = {s_a_of(t) for t in toks}
                    if sas != {anc["value"]} or toks[-1].split(">", 1)[1] != toks[-2].split(">", 1)[1] \
                            or "商品-购买" not in evs[-1]["action"] or toks[-1] != toks[-2]:
                        f.append("rel_s1")
                elif st in ("S2", "S3"):
                    tags = [t for nd in au["nodes"] for t in nd["tags"]]
                    if any(t != anc["value"] for t in tags):
                        f.append("rel_tag")
                elif st == "S4":
                    tags = [t for nd in au["nodes"] for t in nd["tags"]]
                    if any("-".join(t.split("-")[:2]) != anc["value"] for t in tags):
                        f.append("rel_tag")
                # 主题-锚一致:tag 类主题必须内含类目叶子词
                if st in ("S2", "S3", "S4"):
                    K = anc["value"].split("-")[-1]
                    if st == "S4":
                        K = anc["value"].split("-")[-1]
                    if f"「{K}」" not in topic:
                        f.append("rel_topic")
        # 6 种子零碰撞
        if inp in seed_prompts or out in seed_resps:
            f.append("seed_collision")
        for k in ("invariant", "template", "structure", "quote", "dates", "seed_collision",
                  "rel_audit_mismatch", "rel_s1", "rel_tag", "rel_topic"):
            if k not in f:
                ok[k] += 1
        if f and len(fails) < 8:
            fails.append((i, f))
    print(f"[verify] {path}  n={n}")
    denom_rel = len(rel_idx)
    for k in ("invariant", "template", "structure", "quote", "dates", "seed_collision"):
        print(f"  {k}_ok = {ok[k]}/{n}")
    for k in ("rel_audit_mismatch", "rel_s1", "rel_tag", "rel_topic"):
        print(f"  {k}_ok = {ok[k]}/{n} (相关性抽检覆盖 {denom_rel} 条)")
    if fails:
        print("  FAILS:", fails)
    hard = all(ok[k] == n for k in ("invariant", "template", "structure", "quote", "dates", "seed_collision"))
    return hard and not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("path", nargs="?", default=OUT_TRAIN)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_val", type=int, default=100)
    args = ap.parse_args()
    if args.cmd == "verify":
        audit = args.audit or args.path.replace(".jsonl", "_audit.jsonl")
        sys.exit(0 if verify(args.path, audit) else 1)
    shards = sorted(glob.glob(f"{HF}/OneReason_UserProfile/*.parquet"))
    build_split(shards[:9], args.n_train, 2026, OUT_TRAIN)   # 训练:分片 0-8
    build_split(shards[9:], args.n_val, 2027, OUT_VAL)       # 验证:分片 9(用户零重叠)


if __name__ == "__main__":
    main()
