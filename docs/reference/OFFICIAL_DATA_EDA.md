# 官方数据最终 EDA（O1–O6，封板版）

> 封板日期：2026-07-12 UTC
> 资产身份、路径、来源和分类只以 [`ASSETS.md`](ASSETS.md) 为准；本文是分析结论，不是第二份资产清单。
> 本轮只读审计 O1–O6，并仅用 E 类题面做泄漏排除；未创建训练数据、配置或 checkpoint，未改动官方文件。

## 1. 结论先行

1. **当前最高价值不在追加大数据，而在修正 O1 的有效监督配比。**O1 按行看推荐占 59.13%，按 O6 chat template 的 target token 却占 **88.52%**；物料和懂用户只占 5.05%/6.43%，action 更只有 0.95%。任何只登记行数的“配比”都不可信。
2. **O1 推荐的重复题面不是普通重复。**19,204 行只有 6,460 个题面组；3,542 个重复组的 CoT 组内逐字相同，而 3,440 组的答案全部不同。应保留多 target，压缩 12,744 次冗余 CoT 曝光，而不是按 prompt 去重到一个答案。
3. **O3 是 metadata 分析表，不是可直接训练的 SFT。**它把 O1 推荐 user 末尾的 `/think` 全部删除；target caption/tag 又是标签侧 oracle。直接读取 `messages` 或把 target metadata 放进 prompt 都会改协议或泄漏标签。
4. **O2 规模大，但不等于适合 next-item SFT。**Caption→SID 是明显多值关系；更关键的是旧 builder 把 prod/live 历史项伪造成未来 gold，并用 8,000 条重构样本替换 O1 主干。逐用户全量时序复核反而证明 video current/history 只有 7/475,355 行逆序，旧报告的“普遍 10 分钟泄漏”结论作废。
5. **O4/O5 不能整块混入。**两者 5,210,887 行中存在严重源内重复，且有 **477,179 行 raw messages 在 O4/O5 间逐字重复**。O5 严格机械筛完只剩 101 条中文 A–D 单选上限，仍有明显医学/法律偏置和答案质量风险。
6. **O6 有四个实现陷阱。**模型真实上下文上限 40,960，不是 tokenizer 元数据的 131,072；`tokenizer.vocab_size` 只报基础词表，必须用 `len(tokenizer)=176,253`；模型卡提到的 `<|sid_begin|>` 实际不存在；embedding 与 LM head 配置上不共享但初始值逐元素相同，解冻两者会一次打开 360,966,144 参数。
7. **没有发现可合法利用的可见题精确泄漏。**43 个跨日志可见 prompt 与 O1/O3 规范化精确重合均为 0；8 个 material 描述与 O1/O3 exact、包含和固定片段命中均为 0；当前 5 道有效 visible world 与 O4/O5 exact/规范化重合也均为 0。E 类仍只能用于排除和门禁，禁止回灌。

因此，最终优先级是：**O1 同题多 target + 冗余 CoT 压缩 > O1 target-token/题面组配平 > 受控研究 O3 > 极小且人工核验的 General 候选**。O2 大规模推荐重构、O3 点映射、O4/O5 原块注入均不进入当前训练队列。

## 2. 证据口径与复现

本文使用四种证据标签：

- **全量精确**：扫描所有注册行或 Parquet footer/指定列；哈希重复统计使用 SHA256 或 BLAKE2b-128 实用精确指纹。
- **确定性启发式**：语言、学科、CoT 截断、选择题等规则；可复算，但不等于人工语义金标。
- **固定样本估计**：只用于 O4/O5 的 O6 tokenizer 长度；按 source 分层、用精确 source 行数加权。
- **实验事实**：来自 [`experiment_log.md`](../experiment_log.md) 和 [`EXPERIMENT_INDEX.md`](../EXPERIMENT_INDEX.md) 的本地门禁或线上得分。

稳定复现入口：

```bash
# O1–O6 核心结构、行数、O1重复组、O3覆盖、O6词表；默认不做昂贵分词
python scripts/data/verify_official_eda.py \
  --output /tmp/official_eda_verify.json

# 需要复算 O1 全量 chat-template 长度和 target-token 占比时显式开启
python scripts/data/verify_official_eda.py \
  --tokenize-o1 \
  --output /tmp/official_eda_verify_tokens.json

# O4/O5 固定 source 分层 token 统计
python scripts/data/eda_general_official.py \
  --assets O4 O5 --sample-files 32 --samples-per-source 400 \
  --seed 20260711 --output /tmp/o45_token_sample.json

# O4/O5 5,210,887 行全量结构、重复、跨集重合、严格 MC 漏斗
python scripts/data/eda_general_official_full.py \
  --output /tmp/o45_full.json \
  --candidate-audit /tmp/o5_world_candidates_audit.jsonl
```

O2 的完整字段级历史报告和 notebook 分别在 [`hf_raw_data_analysis.md`](hf_raw_data_analysis.md) 与 [`ideas/eda_notebooks/`](../../ideas/eda_notebooks/)；本文采用本轮全量复核后的纠错数字，旧报告中的训练建议不自动继承为当前方案。

O2 本轮复核口径：Pid2Sid/Caption/Tag 用 PyArrow 全量读取注册分片，并对排序后的 `(domain,pid)` 做集合检查；UserProfile 50 万行全量检查 video/ad 时间先后和序列顺序；General 152,005 行全量解析 JSON、角色、长度和重复。只有 UserProfile 的跨域映射与行内重复率使用固定抽样：每个注册 shard 前 1,000 行，共 10,000 行（2% 用户）。下文会逐项区分全量与抽样，不能把抽样率写成全库精确值。

限制：本轮没有做 embedding 语义近邻式泄漏搜索；“没有泄漏”只指文中明确列出的 exact、规范化、包含和固定片段检查。O5 的 101 条只是机械候选上限，没有经过逐题事实核验。

## 3. O1：官方种子 SFT

### 3.1 完整性、任务和模式

12 个 JSONL 共 32,480 行，全部是长度为 1 的 JSON 数组，内部字段统一为 `system/prompt/response`。JSON、think 标签、itemic 语法、0–8191 范围和模式对应错误均为 0。`dataset.tar.gz` 中的 12 个成员与展开文件逐字节一致，**不能重复计数**。

| 任务 | 行数 | 模式 |
|---|---:|---|
| 推荐 video / prod / ad / living | 14,868 / 1,489 / 1,576 / 1,271 | 19,204 行全部 `/think`、非空 CoT |
| 物料 desc→SID / SID→desc | 5,597 / 4,787 | think/no-think=`2,792/2,805`、`2,390/2,397` |
| 用户 action / topic | 1,588 / 1,304 | action 全 `/no_think`；topic 602 think + 702 no-think |

全库 `/think`+非空 CoT 为 24,988 行，`/no_think`+空 CoT 为 7,492 行，控制词和答案结构 100% 一致。

### 3.2 行数配比掩盖了真实 loss 配比

以下使用当前训练的 O6 `qwen3_nothink` 格式（空 system 不生成 system turn）；“计算 token”是完整训练序列，“target token”是 assistant 监督段及对话边界差值。

| 任务族 | 行数占比 | 完整计算 token 占比 | target token 占比 |
|---|---:|---:|---:|
| 推荐 | 59.13% | 69.65% | **88.52%** |
| 物料 | 31.97% | 3.57% | **5.05%** |
| 懂用户 | 8.90% | **26.78%** | **6.43%** |

action 有 1,588 行，占全库 4.89%，但只占 **0.95% target token**。懂用户的长历史吃掉 26.78% 计算量，却只产生 6.43% 监督；推荐长 CoT 则几乎垄断答案侧梯度。

全序列长度 p50/p90/p95/p99/max 为 `1,582/2,936/4,448/6,611/10,553`。`4096` 会截 1,898 行，`8192` 会截 41 行，全部风险集中在懂用户；`16384` 已覆盖 O1 全库，当前 `32768` 更保守。数据实验不应同时更改 cutoff，以免失去单变量归因。

### 3.3 推荐：同题多 target 才是核心结构

- 19,204 行只有 **6,460** 个去模式后缀题面组。
- 3,542 个重复组覆盖 16,286 行，最大组 23。
- 3,542/3,542 组的 CoT 组内逐字相同。
- 3,440/3,542 组的最终答案全部不同。
- 全库另有 113 个真正的逐字重复组，共 145 个额外重复行，最大重复 4 次。

这说明两种“重复”必须分开处理：

1. 真正逐字重复的 145 个额外行可以去重或 inverse-frequency；
2. 同题不同 target 是官方多正例分布，不能只留一条；应每组保留一份原始 CoT，把其余 **12,744** 行的 prompt 后缀切到 `/no_think`、assistant 改为官方空 `<think>` 前缀，同时保留最终 target，从而降低冗余 target token 并补齐推荐 no-think 通路。

推荐还存在 copy 型标签：target 已在本行历史中的 video/prod/ad/living 分别为 `109/218/198/155` 行。prod、ad、living 的比例达到 12%–15%，会教出复制先验。当前只应把 `target_in_history` 作为分层字段，分别报告表现；在没有单变量证据前不整块删除。

推荐 think 共引用 133,516 个 SID，全部逐字来自 prompt，没有新造 SID。132 行的 target 出现在 think 中，但这些 target 同时已经在历史中，不属于 novel-target 泄漏。

确定性截断启发式命中 1,495 行、425 个唯一题面组，最终答案都完整。`seed_cotfix_v1_lora_ep1=0.8674` 已证明只补这批 CoT 尾句不能提分，该方向关闭。

### 3.4 物料：双向任务并不成对

desc→SID 与 SID→desc 的唯一 SID 约为 5,585/4,784，但两方向仅 video 46、prod 160、ad 281 个 SID 相交；绝大多数不是同物料的双向配对。living 只有 784 条 desc→SID，完全没有 SID→desc。

desc→SID 描述字符长度存在强域差异：video p50/max=`118/203`，ad=`125/235`，living=`212/265`，prod=`228/285`。可见 material 描述的 8 个版本样本更长，规范化长度 154–424；与 O1 物料描述 exact/包含均为 0。

O1 物料 SID 在 O3 中覆盖很低：desc→SID 只有 278 个、SID→desc 只有 207 个；O1 物料描述与对应 O3 caption exact/规范化匹配均为 0。O3 不是现有物料题的“同题 metadata 补丁”。

### 3.5 懂用户：结构干净，但覆盖有洞

action 1,588/1,588 JSON 合法，答案全部为合法 itemic token、全部来自历史、无答案重复；1,539 行是历史的严格有序子序列。答案条数 p25/p50/p75/p90/max 为 `5/11/19/28/56`。

topic 1,304/1,304 JSON/schema 合法，事件数 p50=3、max=6；3 行超过官方五步约束，11 行 event action 无法在历史中逐字找到，132 行没有保持完整单调顺序。

真实覆盖洞是：O1 action/topic 的搜索行为为 0、action 文本答案为 0，而平台可见 action 题出现搜索和文本 action。这个洞已经被识别，但 `riders_act_v1=0.8835` 说明大剂量合成搜索/action 数据反而伤分，不能据此继续加量。

## 4. O2：Explorer 五表

### 4.1 表关系与 metadata 覆盖

| 表 | 行数 | 决定性事实 |
|---|---:|---|
| UserProfile | 500,000 | 10 shards、匿名用户行、63 列且无 uid；不能与 O1/O3 按用户联接 |
| Pid2Sid | 35,914,095 | `(domain,pid)` 唯一、无 null；SID→PID 多对一，域前缀不可省 |
| Pid2Caption | 21,061,327 | `(domain,pid)` 唯一、无 null；是 Pid2Sid 的严格子集 |
| Pid2Tag | 5,417,279 | `(domain,pid)` 唯一、无 null；是 Pid2Sid 的严格子集，goods 为 0 |
| General | 152,005 | 通用 messages 数据，不是推荐用户序列 |

Pid2Sid 的 `(domain,pid)` 组合键零 null、域内零重复；全局 pid 只有 35,914,092 个，恰有 3 个 pid 被跨域复用且 SID 不同。因此任何只以 pid 为键的 join 都是错误实现。goods/live/ad/video 行数分别为 `16,087,726/107,289/2,056,889/17,662,191`；唯一 `(a,b,c)` 数为 `10,295,563/62,070/1,069,596/10,345,920`，落在域内碰撞三元组上的行占 `48.93%/51.42%/61.09%/53.63%`。

相同数值 SID 三元组的跨域重合远高于旧报告：goods↔live 0、goods↔ad 9、goods↔video 88、live↔ad 28、live↔video 6,209、ad↔video **285,023**。旧报告的 ad↔video `2,907` 作废；SID 三元组不是全局语义键，必须始终保留 domain begin token。

Caption 覆盖 goods/live/ad/video 分别为 `9,769,747/106,953/1,423,643/9,760,984`，即 60.73%/99.69%/69.21%/55.26%；Tag 覆盖为 0%/56.14%/36.08%/26.13%。Caption/Tag 的 `(domain,pid)` 均唯一、无 null 且严格属于 Pid2Sid，不存在 orphan。

旧 2% Caption 抽样低估了重复。goods 9,769,747 行只有 **8,018,359** 个唯一 caption；2,524,532 行落在重复 caption 上，其中 1,287,793 行属于“同 caption 对多个 SID”（310,363 个冲突 caption，单个最多 1,602 SID）。ad 的重复 caption 涉及 180,545 行、多 SID 涉及 55,576 行；video 分别为 82,696/36,646 行。ad/video 精确空字符串为 2,386/11,380，跨 ad/video 还有 32,312 个完全相同 caption，其他域对为 0。Caption→SID 不是确定函数：必须过滤空串、显式加域、按 exact caption 去重/逆频率采样，并把歧义组做 multi-positive 或降权，不能强行单标签。

live caption 中 106,939/106,953 可被 `literal_eval` 解析为 list，14 条存在漏引号/漏方括号等坏格式；解析后的 tag 数 p50/p90/p99=`8/10/13`，正式使用必须配容错 parser。Pid2Tag 每条都恰为三级路径，三域一级 tag 均为 53 类；完整路径 unique live/ad/video=`1,854/2,132/3,539`，跨域 full-tag 重合 live-ad/live-video/ad-video=`473/636/2,127`。Tag 只能是粗辅助信号，不能拿来 hard-mask 少量 SID。

### 4.2 UserProfile 的有效信号与字段陷阱

决定性序列统计：

- `video_history_sampled_pid_list`：4.09 亿事件，100% 用户非空，长度 p50/p90/max=`649/1558/38041`；必须截断或分段。
- `video_sampled_pid_list`：1.66M 当前事件，95.07% 用户非空，长度 p50=2。
- `ec_colossus_rs_item_id_list`：1.56 亿事件，70.21% 用户非空且长度封顶 500。
- `ec_item_id_list`：只有 1,049 个事件、约 950 用户非空，不能作为 goods 主 target。
- `outer_loop_deep_target_pid`：8,851 个事件、1.5% 用户非空，是稀疏但时间上最干净的广告 target。
- live 序列 p50=8、p99=1,652、max=83,679，极端长尾。

标签漏洞：video 当前/历史 `neg_feedback` 全零；`watch_time` 被截到 64 秒；`play_done` 才是最密集的有效视频偏好信号。多个 live flag 全零或低于 0.01%，只有 `is_detect_game_live` 有稳定信号。

旧报告把全局时间窗边界重叠约 10 分钟解释为普遍泄漏，这是错误归因。全量逐用户检查中，475,355 行同时有 video current/history，475,348 行（**99.9985%**）current 最早时间严格晚于 history 最晚时间，仅 7 行负 gap；median gap=`49,303,711ms≈13.70h`。但 `video_sampled` 当前列表有 253,861 行未按时间升序，history 只有 42 行未升序，builder 必须按 timestamp 重排后再切分。ad deep target 有历史的 7,516 行全部严格晚于历史，min gap=47.379 秒、median≈22.46 小时。

固定 10,000 用户抽样还暴露了两个字段语义陷阱：video current 的 32,902 个事件中 383（1.164%）实际映射到 ad，video history 的 8,172,191 个事件中 79,812（0.977%）映射到 ad，其余才是 video；必须通过 Pid2Sid lookup 重路由，不能相信字段名。`live_hist_live_id_list` 的 714,851 个事件 0% 可映射，只有 `live_hist_author_id_list` 能转 SID。

同一抽样的行内重复 excess 很高：ec_colossus 37.1%、ec_click 30.1%、ec_order 16.6%、live_author 89.3%、ad_click 27.1%，而 video history 只有 0.032%。序列压缩时要保留 `count/recency/action` 强度，不能直接 `set()` 去重。

### 4.3 O2 的训练边界

`official_rec_v3_lora_ep1` 曾用 O2 UserProfile+Pid2Sid 重构 8,000 条 next-item 并替换原推荐主干，线上总分 **0.7948**，比 riders 0.9177 低 0.1229。失败不能解释成“O2 没价值”，而是 label 机制错配：video 取 current 字段；ad deep target 不足时又混入历史；prod 从 order/click/colossus 历史挑强交互；living 从 live 历史挑 follow/观看。后两类本质是 leave-one-out 历史项，不是未来 gold，还破坏了 O1 评测方言。

可保留的研究用途只有：

- 以 O1/O3 的 domain+SID 分布为白名单做 metadata 聚合；
- caption 分组后保留多 SID 集合，而不是制造一对一点映射；
- 用逐用户排序后的 action/recency/count 做辅助，不能丢弃重复强度；
- 若再次研究预测标签，只允许严格未来的 video current 与 ad deep target，以小剂量 additive 方式加入并保留 O1 主干；
- prod/live 禁止用历史末项或强交互项伪造 next-item gold。

### 4.4 General：不是 world 题直配库

General 152,005 行 uuid 全唯一，source 只有 `stepfun_general`；metadata/messages JSON 均可解析。149,900 行（98.62%）标记 reasoning 且含 `</think>`，2,105 行无 reasoning。两轮对话 129,563 行（85.24%），其余大量多轮，最大 239 turns。

prompt/assistant/think 字符长度 p50/p90/p99 分别为 `281/3,518/63,723`、`8,922/59,783/171,770`、`4,905/49,189/164,582`，长尾极重。prompt 含 CJK 31,192 行（20.52%），约 79.5% 不含中文；A–D 式选择题仅 750 行（0.49%）。prompt 只有 125,927 个 unique，46,277 行落在重复 prompt 中；完整 messages 141,832 个 unique，20,094 行落在重复对话中。

它与 E 前 5 道有效 visible world 的 whole/core/containment 规范化重合为 0，与 O1 prompt 规范化 exact 也为 0。因此 O2 General 既无已发现的可见题泄漏，也不是世界 MC 直配库；若未来研究，只能做“中文+短+单轮+去重+答案格式重写+极小比例”，英文、长 CoT、多轮默认排除。

## 5. O3：推荐对齐 Caption/Tag

### 5.1 物理对齐完整，但模式后缀被删除

O3 有 19,204 行，record_id 0–19,203 连续唯一，全部为 system/user/assistant。三个 list 逐行等长，3,539,794 个 SID 位置与 messages 抽取顺序完全一致。

与 O1 对比：system 和 assistant 19,204/19,204 精确相同；user 逐字相同为 0，但去掉 O1 末尾 `/think` 后 19,204/19,204 相同。**O3 raw messages 不能直接注册训练**；正确做法是用 record_id 把 metadata join 回 O1，并保留 O1 原 prompt、模式后缀和答案。

### 5.2 覆盖、歧义与模板偏置

| 指标 | 全位置 | final target |
|---|---:|---:|
| Caption 命中 | 3,478,100 / 3,539,794 = 98.26% | 17,846 / 19,204 = **92.93%** |
| Tag 命中 | 1,356,390 / 3,539,794 = 38.32% | 9,058 / 19,204 = 47.17% |

target caption 覆盖 video 只有 91.11%，prod/living/ad 为 99.93%/99.76%/97.97%；prod tag 全空。所有行至少有一个 caption，但只有 4,675 行所有位置 caption 完整，tag 全完整仅 14 行。

唯一 SID 568,944；caption/tag 覆盖 555,955/210,186。Caption 有 627,193 个唯一文本，长度 p50/p90/p99/max=`228/339/406/1739`。约 177.2 万位置使用“这是一个关于……/兴趣点在于……”固定模板，风格偏置强。

42,277 个 SID 对应多个不同 caption，最多 193 个；6,633 个 exact caption 映射多个 SID，最多 17 个；8,302 个 SID 有多个 tag。多视角可用于共识聚合，但不能随机选一条当唯一事实。

### 5.3 Target metadata 是标签侧 oracle

final target 的 caption/tag 是通过 gold SID 查到的答案信息。合规边界固定为：

- student prompt 必须保持 O1 原文，不能出现 target caption/tag；
- history metadata 只作为 teacher-side 证据；
- target metadata 最多供 teacher/Judge 做一致性判断；
- teacher rationale 不得含 target SID、目标专属实体或可逆提示；
- caption 缺失时回退 O1，不得编造；多 caption 先聚合共识。

两条实验反证已经存在：`seed_capg_v1` 的 5,441 条一对一点映射把 Pass@64 扇宽从 17 压到 11，本地门禁否决；`capcot_v1` 用单一四步模板重写 3,450 行 CoT，导致选择题格式存活仅 25%。因此 O3 只保留“低剂量、history-only、多个措辞模板、prompt/answer 不动”的研究资格，不是当前 ready 数据。

## 6. O4/O5：General Pretrain 与 General SFT

### 6.1 全量结构和语言

| 指标 | O4 | O5 |
|---|---:|---:|
| 行数 / shards | 2,655,181 / 310 | 2,555,706 / 301 |
| 中文 / 中英混合 prompt | 30,113 / 1,559 | 231,909 / 10,591 |
| 英文 prompt | 2,600,653 | 2,295,651 |
| 其他/无法按脚本归类 prompt | 22,856 | 17,555 |
| `text` null | 331,270（12.48%） | 1,530,490（59.89%） |
| think open | 2,293,912（86.39%） | 2,108,933（82.52%） |
| think 未闭合 | 51 | **4,997** |

统一正文必须读 `messages`，不能读 `text`。O5 另有 32,911 个 Infinity 多轮样本，最大 648 条 message；empty user 5 行、empty assistant 139 行、think 后无最终答案 5 行。不同 source 的 `<think>/<answer>` 协议不一致。

全量总字符 p50/p90/p99/max：O4=`7,874/24,615/53,578/197,025`，O5=`7,786/38,577/61,856/884,227`。固定 source 加权 O6 tokenizer 样本显示 O4 total-token p50=1,590、19.90% 超 4,096；O5 p50=2,408、38.44% 超 4,096、19.85% 超 8,192。正式候选必须逐行做真实 token gate，行配比必须同时登记 token 配比。

### 6.2 源内和跨集重复

| 重复指标 | O4 | O5 |
|---|---:|---:|
| exact prompt unique | 2,194,129 | 1,770,615 |
| 落在重复中的额外行 | 461,052（17.36%） | 785,091（30.72%） |
| raw messages 额外重复 | 923 | 31,251 |
| UUID 额外重复 | 0 | 4 |

极端 source 中，O5 OpenCoderReasoning 437,768 行只有 15,128 个 exact prompt，O4 OpenCode 109,292 行只有 14,416 个；原始行随机混会让少数模板占据训练预算。

O4/O5 raw messages 和 UUID 的精确交集都是 **477,179**，等于 O4 的 OpenMathReasoning 整块已出现在 O5。跨集 exact prompt 交集 330,733 个 unique，涉及 O4/O5 各 643,871/649,271 行。两套 General 若未经跨资产去重直接拼接，会完整双计至少 47.7 万行。

合法用法只能是：normalized prompt 分组、跨 O4/O5 去重、同题多 trajectory 先做最终答案一致性、每组 cap 1–2 条，再按 source 和 token 双重配平。

### 6.3 O5 中文 world 候选不足以成规模

全量有两套诊断：宽松中文 MC 启发式命中 4,457 行；独立的严格结构漏斗从“全语言中恰好各一个 A–D 行且无 E–H”43,097 行开始，筛到中文/混合 285 → 可机械解析最终答案 116 → 排除多答案 15 → **101 条机械上限**。

101 条均为唯一 prompt，学科启发式构成为医学 45、法律 18、数学 9、其他 29。人工查看仍有主观题、过时法律/政治题、错误或破损题、追加开放问题，答案真伪未验证；非医学且非法律只剩 38 条。结论是：O5 不能自动筛出有规模、可靠的通识保持集。I-03 降级，101 条只作 builder QC 清单，不是训练资产。

O4 仍以英文长推理/数学为主，当前训练框架是竞赛 SFT；O4 不进入直接 SFT 队列。

## 7. O6：模型和 tokenizer

| 项 | 全量精确结果 | 实现含义 |
|---|---|---|
| 参数量 | 801,433,600 BF16 | 0.8B pretrain-only checkpoint |
| 架构 | 28 层、hidden 1024、16 Q heads / 8 KV heads | 单卡 LoRA 可行 |
| 模型上下文 | `max_position_embeddings=40960` | 这是硬上限，不使用 tokenizer 的 131072 |
| 词表 | config/`len(tokenizer)`=176,253 | `tokenizer.vocab_size=151,643` 只含基础 BPE，不能据此建 mask |
| itemic | a/b/c 各 8,192，ID 151669–176244，全部原子化 | 每个 item 为 domain begin + 3 个 code token |
| domain token | video/prod/living/ad begin+end 共 8 个 | 数值三元组跨域碰撞，begin 不能删 |
| generic SID | `<|sid_begin|>` 不存在 | 模型卡示例与实际 tokenizer 不一致，禁止生成 ghost token |
| embedding/head | 配置 untied，各 176253×1024；初始值逐元素相同 | 两者合计 360,966,144 参数，占全模型 45.04% |

`<think>`、itemic 和 domain token 都是 added token，但不是 `all_special_ids` 中会被 `skip_special_tokens=True` 删除的协议 token。`/think`、`/no_think` 本身分别会切成 2/3 个普通 BPE token；chat template 的 `enable_thinking=False` 与题面文字后缀是两套控制面，训练和推理必须同时对齐，不能假定写了 `/no_think` 就自动改变 template 参数。

历史实验 `fk_lora_embed_ep1=0.8672` 已证明全秩解冻 embedding/lm_head 会记忆训练 item 并破坏未见 item 与已有推荐先验。当前默认应冻结两者，不新增 53 个 tag、19 个动作等 special token；自然语言 side-info 的收益应先在不改词表的条件下证明。

## 8. 跨资产泄漏审计

- 封板扫描的 21 份平台日志对应 20 个唯一 evalTaskId 加一个重复副本；新增 r64 E1/E2 后当前为 23 份、22 个唯一 evalTaskId，仍是同一组 43 个跨版本唯一可见 prompt：material 8 个，其余七任务各 5 个。
- 43 个 prompt 与 O1 prompt、O3 user 的 `NFKC + 去模式后缀 + 去空白 + casefold` exact overlap 均为 0。
- 8 个 material 描述与 O1 物料描述 exact/双向包含均为 0；与 O3 的 3,478,100 个非空 caption exact、双向包含、每题 4 个均匀 48 字片段命中也均为 0。
- 当前有效 visible world 前 5 题与 O4/O5 的完整 prompt 和去模板 stem，在 exact 与 `NFKC+casefold+去空白/标点` 四种口径下全部为 0。
- 既有 O2 shadow-gold 审计中，可见推荐题的 154 个唯一 SID 在 50 万 UserProfile 中没有恢复出同用户；单 shard 最大命中 13/154，不能据此恢复平台 gold。

以上只能说明未发现所检查口径的重合，不能反向把 E 题用于 teacher prompt、拒绝采样、近邻筛选或训练配比。任何未来 builder 都必须把全部 E prompt 作为**排除集合**。

## 9. 最终训练 trick 排序

| 优先级 | Trick | 证据与做法 | 状态 |
|---|---|---|---|
| P0 | O1 同题多 target、冗余 CoT 压缩 | 每组保留 1 个原始 CoT；其余 12,744 行切 `/no_think` + 官方空 `<think>`，最终 target 不变；普通 CE，不叠加 task-weighted loss | **唯一 ready 的官方数据单变量候选，待用户批准** |
| P0 | target-token + prompt-group 双口径配比 | 配置/台账同时记录行数、完整 token、target token、题面组和四域；不能只报行数 | 训练准入门禁 |
| P0 | 真正逐字重复去重 | 只处理 O1 的 145 个额外 exact 行；同题不同 target 绝不能删 | 低风险 |
| P0 | O6 冻结与实现修正 | 冻结 embed/head；使用 `len(tokenizer)`；上下文不超过 40960；保留 domain begin | 已由文件和负实验验证 |
| P1 | 推荐 copy 分层 | 按 `target_in_history` 分层报告和采样；先测，不一次性删除 prod/ad/living 的 12%–15% | 待单变量验证 |
| P1 | O3 history-only 多模板 teacher | join 回 O1；target metadata 不入 prompt；先聚合多 caption 共识；低剂量、多模板 | 两个旧实现失败，方向仅保留研究资格 |
| P2 | O2 metadata/时序特征 | 只在 O1/O3 domain+SID 白名单内聚合；caption 按多 SID；序列保留 action/count/recency | 只作小剂量辅助，不得替换 O1 推荐 |
| P3 | O5 严格 world 候选 | 最多 101 条机械上限，逐题核验后还会减少；不得重复上采样 | 降级，不足以单独立项 |

P0 首轮候选应复用已登记的 O1 派生构建逻辑，但保留全部 32,480 行（包括 145 个 exact extra），只改变**12,744 行的模式后缀与冗余 CoT**，并使用普通 CE。exact 去重、copy 分层和任何 task 权重都必须留到独立后续实验，不能同锅。历史 `seed_taskbal` 同时叠加 action 3×、终止 2×、topic 0.5×、物料答案 4×，action 仍 0/4 JSON，因此不能拿它否定纯数据压缩，也不能再复用其加权 loss。

## 10. 永久关闭或禁止的路线

- O2 UserProfile next-item 大规模重构或替换 O1 推荐；线上已得 0.7948。尤其禁止用 prod/live 历史项伪造未来 gold。
- 把 O3 raw `messages` 直接训练；它缺少 19,204 个 `/think` 后缀。
- 把 O3 target caption/tag 放入 student prompt；这是显式标签语义泄漏。
- O3 一对一 Caption→SID 点映射；会压窄 Pass@64 候选分布。
- 用单一固定模板批量重写 O3/O1 推荐 CoT；已造成模板回声和 world 格式崩坏。
- 按 prompt 把 O1 推荐去重到一个 target；会删除官方多正例分布。
- O4/O5 原块注入、只按行数混合、未做跨集去重的 O4+O5 拼接。
- 自动把 O5 的 101 条机械候选当 gold；必须逐题核验。
- 新增动作/tag special token 或解冻 embedding/head；先验破坏风险已被线上实验验证。
- 用 tag 对 SID 做 hard mask；同一 tag 可对应大量 SID，信息不足。
- 只修补截断 CoT 尾句、暴力 action token 加权、focal loss、评测形态转写；均已有负证据。
- 任何 E 类题面、答案、近邻或平台日志内容进入训练、teacher、数据筛选或 rejection sampling。

## 11. 封板规则

广义 O1–O6 EDA 到此关闭。后续允许的是**实验专属 builder QC**，不是重新扫数据：

1. 为已批准的单变量实验验证行数、hash、target-token mix、prompt-group 分布和 E 零重合；
2. 官方发布新 revision、注册路径缺失或 checksum 变化时，按 [`ASSETS.md`](ASSETS.md) 触发定向复核；
3. 新派生训练数据必须登记上游 asset ID、builder、行数、内容 hash、行/target-token mix ratio，正式训练前写入配置或 ledger；
4. 本文结论若被线上单变量实验推翻，只更新实验结论和对应条目，不再启动无目标的全盘 EDA。
