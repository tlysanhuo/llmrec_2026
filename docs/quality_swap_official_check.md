# quality_swap_v1 官方资料对照 memo

> 2026-07-07。目的:确认 `quality_swap_v1_lora_ep1` 是否有正期望,只依据官方赛题/FAQ/PPT/数据集资料和本地量化,不按选手分享下注。

## 官方依据

- 官方赛题目标是四维模型:懂物料、懂用户、懂推荐、懂世界。懂用户定义为“根据行为历史洞察用户需求,捕捉用户动态偏好与需求演化逻辑”。本地抽取版见 `tmp/official/contest_official.txt:125`。
- 懂用户 action_select 的官方评测是生成 1 个 JSON 数组,和 ground truth 计算 F1。官方构造方法是先由 API 从纯文本历史抽主题,再给定历史+主题抽相关交互。见 `tmp/official/contest_official.txt:182`。
- 懂用户 topic_gen 的官方评测不是自由主题生成,而是给定主题抽取有序 `logic_chain`: action 最优有序匹配,再在匹配 events 上算 Token-F1 和 ROUGE-L-F1,最终综合 Action Alignment 和 Logic Alignment。见 `tmp/official/contest_official.txt:215`、`:301`。
- 官方约束:逻辑链 5 步以内;同一日期同类型同演进步骤可合并;logic 必须基于 action,避免过度推断;三类有效关系为场景需求补全、兴趣因果递进、需求深度细化。见 `tmp/official/contest_official.txt:238`、`:262`、`:266`。
- 官方 SFT-CoT 样例里有“候选兴趣演化链选择题”:候选 A/B/C -> 短 think -> `答案: [A, B]`。见 `tmp/official/contest_official.txt:622`。

## 本次等量置换与官方要求的匹配

`quality_swap_v1` 相对 `riders_fk_lora_ep1` 的核心变化:

- 删除 FK 旧 user 行:old action 1254 + old topic 1253 = 2507。
- 加入官方形态 user 行:U1 action JSON 354(367 raw 按 `_src_idx` 去重)+U2 候选链 MC 353+U3 topic logic_chain 2372 = 3079。
- 为保持 37267 等量,从剩余 FK 删除最长 CoT 行 572 条,不动 world_zh/P3/world_mc_clean 锚。

本地 QC:

| 块 | 行数 | 官方对应 | 结构核验 |
|---|---:|---|---|
| U1 action | 354 | action_select JSON/F1 | JSON 可解析 354/354;输出 token 100% 是历史子集;空 think 100%;答案长度中位 15 |
| U2 chain MC | 353 | PPT SFT-CoT 候选链样例 | 候选 A/B/C -> 短 think -> `答案: [...]`;不是平台直接评测形态,但和官方 SFT-CoT 样例同形态 |
| U3 topic | 2372 | topic_gen logic_chain | JSON 可解析 2372/2372;输出 action token 100% 是历史子集;空 think 100%;事件数分布 3/4/5,中位 4 |

对比旧 FK:

- 旧 action 也是合法 JSON、历史子集,但不是按官方 PPT 新披露的 API 两步主题抽取流程重做;U1 的主题名和选中集来自新的 teacher 产线,更贴近“从 X 到 Y”的演化主题。
- 旧 topic 可解析,但有 46.3% 非空/长 think,而官方懂用户推理参数是 no_think;U3 全部空 think,并把 logic 压到更短、更像官方 rubric 的 action-grounded 说明。
- U2 是新增覆盖:旧 FK 没有官方 PPT 样例披露的候选链选择题形态。

## 结论

这不是“保证涨分”,但按官方资料是正期望实验:

1. **方向命中官方新增信息**:官方明确懂用户二子项是 action JSON 和有序 logic_chain,而我们替换的正是旧 FK 的 user/action/topic 区块。
2. **数据形态更贴官方**:U3 的 3-5 步、空 think、历史子集、JSON schema 与官方 topic_gen 约束一致;U2 与官方 SFT-CoT 候选链样例一致。
3. **风险受控**:保持总行数 37267,不做外加数据,不碰已线上有效的 world_zh/P3/world_mc_clean 锚;额外删除的是最长 FK CoT,降低 token 预算而不是增大稀释。
4. **主要收益应在懂用户 topic/action**:U3 数量接近旧 topic 的 1.9 倍且官方形态更准;U1 行数少但质量更高;U2 补官方样例形态。预期不是大跃迁,而是对 `riders_fk_lora_ep1` 的低风险小正增益。

红队风险:

- U1 答案长度中位 15,高于旧 action 的 11;如果评测 gold 更稀疏,可能伤 precision。
- U3 logic 文本比旧 topic 更短,若平台更看重丰富措辞,ROUGE 可能不一定更高。但官方计分先 action 对齐,且 logic 只在匹配 events 上算文本相似,短且贴 rubric 通常更稳。
- 删除 572 条最长 FK CoT 可能轻微伤推荐/物料泛化,但这些行集中在长 CoT 的 `mat_or_rec`/`itemic_answer`,且总 token 预算下降,比外加 3k 行更安全。
