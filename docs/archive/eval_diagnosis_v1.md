# baseline_sft_v1 评测诊断报告

> 覆盖四维八任务（懂世界 / 懂用户 / 懂物料 / 懂推荐），基于评测日志逐样本证据。当前全量种子 SFT 后总分 **0.810**；官方 pretrain 参考分见各任务。本报告用于指导 R2 (action-select SFT) / R3 (短CoT+RFT) / General replay 的下一步投入。

---

## 1. 总览表

| 任务 | 维度 | 输入格式 | 评测协议 | 输出健康度 | 主要问题 |
|---|---|---|---|---|---|
| challenge_common_sense | 懂世界 | 纯文本四选一单选（807 test） | Accuracy，抽取选项字母；no_think 预填空 think + /no_think；max_tokens=60000 | ⚠️ 中：常识题干净，数学题坏 | no_think 失效退回 CoT；占位符复读未填字母；悬空 `</think>`；数学/多步计算错 |
| challenge_evolution_action_select | 懂用户 | 纯 SID 历史（含明文搜索 query），抽取式 JSON 数组（1739 test） | F1；n=1 采样，max_tokens=4096，no_think | ❌ 崩溃 | 灾难性复读（同一 token 600+ 次）→ JSON 未闭合、F1≈0；新造非历史 item |
| challenge_evolution_topic_gen | 懂用户 | 超长 timeline + 主题 → JSON logic_chain（≤5步）（905 test） | F1（events.action vs gold 条目集合匹配）；no_think | ⚠️ 格式健康、内容全错 | 整链幻觉：编造 SID/日期/搜索词；未检索真实条目；重复交互；logic 过度推断 |
| itemic_pattern_grounding | 懂物料 | 纯文本 caption → video token（574 test） | Pass@64；beam_width=64，max_tokens=3，强制前缀 `<|video_begin|>` | ✅ 格式满分（320/320） | beam 前缀坍缩（s_a/s_b 押 2-3 个高置信）；剧情类 caption 塌同一坨热门前缀；细粒度对齐弱 |
| recommendation_ad | 懂推荐 | 四域纯 SID 历史 → next-ad（1000 test） | Pass@64；two-stage(think 4096)+beam32 与 no_think 单阶段 | ⚠️ 格式合规、内容差 | /no_think 被违背；think 32 beam 逐字相同套话；think 幻觉不存在的 SID/搜索域；24/25 复读历史 |
| recommendation_live | 懂推荐 | 三域纯 SID 历史 → next-live（1000 test） | Pass@64；same two-stage + no_think | ⚠️ 格式合规、内容差 | think 幻觉地域/语义 + 编造 token；32 beam think 完全复读；候选抄直播历史 anchor |
| recommendation_product | 懂推荐 | 四域纯 SID 历史 → next-prod（1000 test） | Pass@64；same two-stage + no_think | ⚠️ 格式合规、内容差 | think 复读且幻觉（编造 ad token/品类/时间）；预测退化为高频历史复读；beam 尾部邻域抖动 |
| recommendation_video | 懂推荐 | 三域纯 SID 历史 → next-video（1000 test） | Pass@64；same two-stage + no_think | ⚠️ 格式合规、内容差 | /no_think 被无视；十进制假 item-id + HTML 脏标签污染；多样性坍塌；约 15/28 历史复读 |

**总览结论**：格式合规问题集中在**懂世界(common_sense)**与**懂用户(action_select)**；懂物料格式满分、问题在命中；懂推荐四子任务格式全合规但内容普遍是"复读历史 + think 幻觉套话"。**懂用户是最脆弱一维**（action_select 直接崩溃，topic_gen 内容全错），也是从 0.810 上探的最高杠杆。

---

## 2. ★ 关键问题清单（按严重度，会直接掉分的）

### ★★★ 极严重（直接归零/大面积掉分）

1. **[action_select] 灾难性复读 → F1≈0**：Sample 0 输出从第 8 项起无限重复 `<|prod_begin|><s_a_1156><s_b_1107><s_c_3798>` 约 600+ 次，打满 max_tokens=4096 被截断，JSON 数组未闭合（末项 `<|prod_begin|><s_a_1156><s_b_1107>` 半截 token）。解析失败 + 海量重复 FP，与官方 pretrain 参考分 0.0000 一致。**且复读 token 疑非历史真实项**（历史中 s_b_1107 系列为 s_a_6969/4653/5497，无 s_a_1156），说明退化时在无根据新造 item，未学到 copy 约束。

2. **[topic_gen] 整链幻觉，答案不可溯源 → F1 归零**：
   - Sample 0：输出 `[商品-点击] <|prod_begin|><s_a_700><s_b_1107><s_c_738>` 挂在 2026-01-19，但历史真实为 `<s_a_6969><s_b_1107><s_c_738>`（a 位 6969→700 被编造），且当天只有 video 事件。
   - Sample 1（主题=东北芥菜丝做法）：输出 `[搜索] 西北草酸清洁剂`（历史不存在，真实是 2025-12-24 `东北芥菜丝咸菜做法`）+ 编造 SID + 编造日期 `2026-01-26`，三个 event 全部不可溯源。
   - Sample 2：两个 event 用完全相同的 `<s_a_6693><s_b_4433><s_c_751>`（查无此项），既重复又幻觉。
   - 三条样本的主题在历史里都有明确锚点，模型却全部照 few-shot 例子"编一个像样的链"，没做检索对齐。

3. **[common_sense] 占位符复读致本可答对判错**：SID2（`b^4 - 2b^2 = 0 且 b≠0`）推理正确（应选 D），却原样吐出模板占位符 `正确答案是 (在此处填写选项字母)` 未替换字母，且末尾多一个悬空 `</think>`。**推理对但格式错直接失分**，是性价比最高的可回收失分。

### ★★ 严重（大量样本共性，拖累整维）

4. **[懂推荐全体 + common_sense] /no_think 失效**：prompt 明确带 /no_think 且模板预填空 `<think>\n\n</think>`，但 think-on 路径仍生成 4096-token 长 CoT（ad/live/product/video 均是），common_sense 的 SID2/SID3 也退回思考模式。soft_switch 软开关在 SFT 后失灵，模型对 /no_think 遵循不稳定。

5. **[懂推荐全体] think 幻觉物料/语义，污染推理信号**：
   - ad：think 引用 `<|living_begin|><s_a_4372><s_b_4879><s_c_2699>`（输入出现 0 次），并虚构"搜索域/早教启蒙标签"（输入无搜索域、无文本）。
   - live：编造"山西朔州/晋城/中年男性"语义，引用 `<|prod_begin|><s_a_1293>...`（输入 0 次、think 32 次）。
   - product：编造 `<|ad_begin|><s_a_901><s_b_100><s_c_1834>`（不存在）并把 ad 误称"加购行为"；Sample1 编造"2018-2024/猴头核桃/金刚菩提/相亲交友"。
   - video：混入十进制假 id `8359402637648`、脏 HTML 标签 `<br /></br></div>`、编造"河北唐山"。
   - **纯 SID 输入无 caption，模型把裸 SID 硬编成语义故事**，是 pretrain 遗留污染。

6. **[懂推荐全体] 预测退化为历史高频复读，无跨域泛化**：ad 24/25 复读历史 item、live 候选抄直播 anchor、product 集中在历史高频 prefix、video 约 15/28 是历史已见 item。若 ground-truth 是新 item（next-click 通常是新的），整批 miss。这正是四子任务 pretrain 分最低（ad 0.0864 / live 0.0544 / video 0.0900）的病灶。

### ★ 中等（拖累上限但非归零）

7. **[common_sense] 数学/多步推理弱**：SID3（约瑟夫/递归）推理链自相矛盾凑答案、SID4（植树 162/3+1=55 选成 56）、SID0（常识题也选错 A→D）。计算题是懂世界失分集中点，纯知识题（SID1）反而稳。

8. **[grounding] beam 前缀坍缩、命中靠覆盖率而非理解**：Sample0 64 beam 中 s_b 有 56/64 同一 s_b_6234，s_a 前三占 47/64；Sample1 53/64 同一 s_a_5254。若真值 s_a/s_b 不在这 2-3 个高置信前缀内，64 beam 全废。剧情类 caption（Sample0 悬疑 vs Sample2 家族争斗）塌到同一坨热门前缀，细粒度语义-码本对齐弱。

9. **[懂推荐全体] think 零多样性 + beam 邻域抖动**：two-stage Stage1 n=1 只采 1 条 thinking，Stage2 对同一 thinking 做 32-beam，导致 32 候选 think byte-for-byte 相同；候选差异仅在 s_c 后缀微调（如 product 526/6220 前缀仅变 s_c），未探索不同兴趣簇，Pass@64 有效覆盖被浪费。

---

## 3. 按四维度的具体短板与提分抓手

### 懂世界（common_sense，Accuracy，pretrain 0.1387）
**短板**：(1) no_think 未固化为"空 think 块=直接给答案"；(2) 格式不统一（时而单字母、时而散文、无稳定"正确答案是 X"定位串，判分正则脆弱）；(3) 数学/多步计算能力弱。
**抓手**：
- 把 no_think 单选目标**统一固化为『空 think 块 + 单行 正确答案是 X（纯字母）』**，禁止任何 CoT。
- 格式 RFT 奖励：以正则 `正确答案是\s*[（(]?([ABCD])` 命中为格式奖励项，对占位符复读、纯散文无定位串、悬空 `</think>` 给 0/负奖励。
- **优先回收 SID2 这类"推理对但格式错"**（格式 SFT 立即见效），再靠难题数据 + RFT 慢补 SID3/SID4 计算能力（补植树/约瑟夫递归/因式分解题型）。
- max_tokens=60000 过大：no_think 正常应为个位数 token，惩罚超长输出可抑制意外思考。

### 懂用户（action_select F1 + topic_gen F1，两者均 ≈0）
**短板**：任务本质是**抽取式（从历史 copy 子集）**，模型却在做生成式——action_select 停不下来（复读崩溃），topic_gen 整链编造。都没把输出锚定到历史 item 集合。
**抓手**：
- **走"可验证摘录"范式**：训练样本 events.action / 数组项必须逐字等于 timeline 真实条目（日期+行为类型+完整 SID/搜索词），构造时严格字符串校验，剔除任何编造 SID/日期的样本，教会"只能 copy 不能生成"。
- **注入停止先验**：相关项抽完即停（数组远小于历史），加"相关项为空则输出 []"样本，合法闭合 JSON + `<|im_end|>` 收尾，治复读。
- **RFT 直接用 F1 做 reward**：对每 prompt 采多条，reward = F1(去重集合 vs gold)；叠加格式 reward（能否 json.loads、item 四段式合法、是否全部 ∈ 历史集合）+ 复读/超长/非法 JSON 惩罚。
- **利用明文搜索意图**：历史含 `[搜索] 塞纳改装大屏教程 / 塞纳中排座椅怎么升高`，与主题高对齐——构造"搜索意图→相关 item"对齐关系显式化，提升召回。
- topic_gen **抑制 logic 过度推断**：模型看不到 caption，logic 引入具体商品名（"全顺专用/防滑脚垫/镀膜剂"）即幻觉；R2 让 logic 只依据"行为类型+搜索词明文+时间演进"，或让 logic 不参与匹配、只约束 action。
- 长上下文检索强化：SFT/RFT 样本同样长、锚点埋中后段，避免只看开头/few-shot。

### 懂物料（grounding，Pass@64，pretrain 0.1533）
**短板**：格式满分，但 beam 前缀坍缩、依赖热门先验、细粒度语义-码本对齐弱、对分布外 caption 泛化存疑。
**抓手**：
- **提升首级码本（s_a/s_b）区分度**：造长尾 + hard-negative 数据（同类目不同 item 的 caption 配对），让剧情类能区分到具体 item 而非都塌到 s_a_3373/7879。
- **码本层级式加权监督**：s_a→s_b→s_c 自回归、错在 s_a/s_b 一步定生死；对 s_a、s_b 命中单独加权（首级权重最大）。
- **RFT 用 Pass@64 同构信号**：以解码回 item id 是否命中做 reward，训练目标与评测口径一致；奖励中加 s_a/s_b top-k 覆盖率，逼模型在高不确定时把候选摊到多个前缀而非只改 s_c。
- **四类目均衡配比**（prod/video/ad/living），防 video 强而 ad/living 长尾拖后腿。

### 懂推荐（ad/live/product/video 四子任务，Pass@64，pretrain 0.0544~0.0900）
**短板**：格式全合规但（1）think 复读+幻觉+套话；（2）预测退化为历史复读、无跨域泛化；（3）think 多样性为零、beam 仅邻域抖动；（4）/no_think 失效。
**抓手**：
- **造"非复读"的 next-item 目标**：leave-one-out 且 ground-truth 不得出现在 history，强制"预测新 item"；对高频 anchor（ad 的 s_a_5687/s_a_1584、video 的 s_a_116 簇）去偏采样；把历史高频 item 作 hard negative 教"已消费 vs 将消费"。
- **RFT 以 Pass@64 命中为 reward**：beam32 已能产多样候选；对"复读历史但没命中"给低/负 reward，对"幻觉 SID（解码不出合法 item）"给负奖励，把"挑历史众数"推向"跨域泛化"。
- **think 事实性治理**：任何 CoT 中 `<|..|><s_..>` token 必须 ⊆ 输入（脚本校验），杜绝幻觉物料；清洗十进制假 id 与 HTML 脏标签（`<br/></br></div>`）。
- **打散 thinking 多样性**：Stage1 采多条（n>1）高温 thinking 或 group-based RFT（GRPO/best-of-n over 多 thinking），让 64 候选覆盖多个兴趣簇；对候选集合 coverage/去重加 reward，缓解前缀簇坍塌。
- **跨域协同建模**：造"视频/电商兴趣 → 推断下一 item"链式样本，think 引用真实 history token，用 s_a 前缀做同类目跨域召回；行为权重 grounding（打赏/首次打赏等强信号 anchor 家族优先）。
- **推理成本收敛**：living/prod 这类纯 item 完成任务，think 4096 token 性价比极低且引入幻觉，RFT 后切短 think / no_think 单阶段，把 token 预算留给 evolution/common_sense。

---

## 4. Quick Wins（低成本快速见效）

1. **common_sense 目标格式对齐**：所有 no_think 单选样本 assistant 固化为『空 think 块 + 单行 正确答案是 X（纯字母）』——立即回收 SID2 这类"推理对但输占位符"的失分，是全报告性价比最高项。
2. **判分正则加固 + 格式 RFT 奖励项**：`正确答案是\s*[（(]?([ABCD])`，对占位符/散文/悬空 `</think>` 给 0/负奖励，快速消除格式违规。
3. **action_select 推理侧临时止血**：加 repetition_penalty / presence_penalty / frequency_penalty + 连续重复 item 后处理去重截断，先把"复读打满 4096 → JSON 非法"的整批 miss 止住（根因仍需数据侧解决，但这是零训练成本的即时收益）。
4. **/no_think 遵循修复**：用 qwen3_soft_switch 软开关对齐数据，no_think 样本 assistant 段严格为『空 `<think>\n\n</think>` + 直接 item/答案』，参考 recommendation 各任务 no_think 单阶段候选（已干净、无幻觉、无 token 浪费）作为目标分布。
5. **think token 幻觉过滤（数据清洗）**：脚本校验 SFT/RFT 目标里所有 `<|TYPE_begin|><s_..>` token 均 ∈ 输入，剔除幻觉样本；去掉十进制假 id 与 `<br/></br></div>` HTML 残留——纯数据清洗，无需重训范式。
6. **max_tokens 收敛**：common_sense no_think 场景大幅下调 max_tokens（个位数 token 即够），抑制意外思考、省算力、降跑偏概率。
7. **json.loads 兜底容错**：评测/后处理侧对 action_select、topic_gen 的截断 JSON 做尽力修复解析，避免因半截 token 导致的整条 0 分（临时缓解，非根治）。

---

## 5. 与既定策略（R2 action-select SFT / R3 短CoT+RFT / General replay）的衔接

### R2 — action-select SFT（抽取式 + 停止先验）
直接对应**懂用户维度**（当前最脆弱、最高杠杆）：
- 大量构造 **action_select 抽取式样本**：目标 = 去重、来自历史、合法闭合、能主动 EOS 的短 JSON 数组，含空集 `[]` 样本建立停止先验，治疗复读崩溃。
- **topic_gen 同批纳入 R2**：走"可验证摘录"，events.action 逐字等于 timeline 真实条目，构造中间监督（主题 → 检索真实锚点 → 组链）。
- 强化 **copy 约束**（数据侧强调只能原样复制历史 token；可选 constrained decoding），针对 action_select 新造 s_a_1156、topic_gen 编造 s_a_700 这类幻觉。
- 顺带把 **common_sense no_think 单选目标固化**（格式 SFT）纳入 R2 批次，回收格式失分。

### R3 — 短CoT + RFT（Pass@64/F1 直接对齐评测口径）
对应**懂推荐 + 懂物料 + 懂用户 RFT 阶段**：
- **短 CoT**：把当前 4096-token 幻觉套话替换为"指向真实历史 token 的克制短推理"（think 内 token 必须 ∈ 输入），或对 living/prod 直接切 no_think 单阶段。
- **RFT reward 设计**：懂推荐/物料用 Pass@64 命中；懂用户用 F1；统一叠加格式 reward（json.loads / item 四段式合法 / ∈ 历史集合）+ 复读/超长/幻觉惩罚 + 历史复读降权（鼓励预测新 item）+ 候选多样性/覆盖率奖励。
- **多假设 thinking**：Stage1 n>1 采多条 thinking 打散 32-beam 共用一段的退化，让 Pass@64 真正覆盖多个兴趣簇。
- **码本层级加权**（grounding）与 **beam 前缀多样性**在 R3 RFT 中一并对齐。

### General replay（防遗忘 + 抑制 pretrain 污染）
- **回放 no_think 直答范式**，巩固 soft_switch 双模式可控，防止 R2/R3 训练后 /no_think 再次失灵。
- **回放清洗后的常识/知识题**（SID1 这类模型本就能干净给单字母的），保住懂世界纯知识题的稳定盘，同时用难题（数学/多步）补 SID3/SID4 计算短板。
- **通过 replay 冲刷 pretrain 遗留污染**：十进制 item-id、HTML 脏标签、think 编造搜索域/文本标签——这些是 pretrain 数据噪声，需在 replay 语料中持续以干净样本稀释。

**优先级建议**：R2 优先攻懂用户（action_select 复读崩溃 → 整维归零，杠杆最大）+ quick win 回收 common_sense 格式分；R3 攻懂推荐四子任务的历史复读/幻觉（占分权重大、pretrain 分最低）与懂物料命中率；General replay 全程护航防遗忘与去污染。