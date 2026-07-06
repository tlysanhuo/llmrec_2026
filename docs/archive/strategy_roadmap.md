# 初赛后训练策略路线图 (研究团队分析, 2026-07-01)

> 来源:研究团队对我方现状 + OneReason 官方路线的战略分析。已被我方数据/文献/基线判断印证。
> 一句话:**初赛最优解不是重训 perception,而是用 17GB 原始序列构造 R2/R3 后训练闭环——先高质量 SFT 把格式/任务/选择能力训出来,再用 RFT/轻量 GRPO 对 Pass@64 做分布塑形。** 官方只放 pretrain checkpoint(SFT/RL 仍 coming soon),后训练必须我们自己做扎实。

参考锚点:pretrain 0.6655 → **baseline_sft_v1 = 0.810 (+0.145)**。评分见 [[scoring-structure]],方法见 [[onereason-methodology]]。

---

## 1. 赛题本质 = 可复现的数据工程 + 后训练策略

总分 = 8 子项直接相加 ⇒ 不按「四维度平均」思考,按 **子项数量 × 当前短板 × 可训练性** 排 ROI:

| 优先级 | 模块 | 原因 | 初赛主打法 |
|---|---|---|---|
| **P0** | 懂用户 / R2 | 现≈0,且有 50 万用户序列,监督数据最直接 | Action-selection SFT,快速拉 F1 |
| **P1** | 懂推荐 / R3 | 4 子项决定上限;但 SFT-only thinking 可能反伤 | non-thinking + 短 CoT SFT,随后 RFT/GRPO |
| **P2** | 懂世界 | 防遗忘,别让推荐 SFT 打坏通识 | General replay + MC verifiable tuning |
| **P3** | 懂物料 / R0 | 基座已强,收益低但能补 | description→SID no_think,小比例混入 |

## 2. 对原计划的两个修正

1. **R3 CoT 不要一上来大规模铺开。** OneReason 官方结果:SFT 阶段 thinking 在跨域推荐反而弱于 non-thinking,RFT 后 thinking 才稳定超过。⇒ 收益来源不是「CoT SFT 本身」,而是**高质量、无泄漏、可被 RFT/RL 奖励筛选的 CoT 起点**。
2. **初赛第一周先做 RFT,不必等完整 GRPO。** GRPO 工程成本高、奖励稀疏、rollout/显存/稳定性拖节奏;RFT(rejection sampling FT)更贴现状:SFT 模型采 16–64 条,留命中/合法/多样的轨迹回灌训练。

## 3. 第一阶段:先打穿「懂用户」R2 Action-Selection

**目标**:历史 + 主题 → 选相关行为,学会精确筛信号(不是生成漂亮解释)。

**数据构造**:每条序列取时间点 t,history=seq[:t],未来 item y=seq[t] 的 caption/tag/category 生成 topic。历史里找与 topic 相关行为为 positives,混 hard negatives,输出编号集合。
- 正样本确定性打分(便于复现):`score = 0.40·tag_jaccard + 0.20·caption_bm25 + 0.15·same_domain + 0.15·same_s_a_or_s_b + 0.10·recency`;positive = score≥阈值 的 top;无正样本时保留 top1 但标记 weak positive、低权重训练。
- **hard negative(必须,不能只随机)**:①同 domain/同 s_a 但 tag 不同 ②最近行为但与 topic 无关 ③跨域高频 ④同用户长期兴趣但非当前主题。
- 候选长度 8–20,正样本比例 10%–40%。**必须显式含「少量正样本 / 多个正样本 / 无相关行为」三种情况**(F1 最怕全选或少选)。
- 输出格式**极简**(只输出编号,如 `[1, 5, 9]`),以官方 seed SFT 同类任务为准,别自造复杂 JSON。thinking 版 `<think>` 只写证据、**不写最终编号列表**(防提前输出答案)。

## 4. 第二阶段:R3 推荐 = 短 CoT + 多 non-thinking

0.8B 小模型 + 平台固定推理 ⇒ CoT **不能长、不能散、不能剧透**。

**R3 SFT 数据配比建议**:
| 类型 | 占比 | 用途 |
|---|---|---|
| R3 no_think next-item | 45%–55% | 稳住 Pass@64 主体概率 |
| R2 action-selection | 25%–35% | 快速补懂用户 |
| R3 short-CoT | 10%–15% | 给 RFT/GRPO 提供 thinking 起点 |
| R0 description→SID | 3%–8% | 保持物料生成格式 |
| General MC/QA | 5%–10% | 防懂世界遗忘 |

**R3 CoT 模板压到 120–250 中文字**:画像(近期兴趣A+长期倾向B)→ 候选兴趣(1-2 个假设+证据)→ 转移判断(沿某假设,目标域一致)→ `</think>` 后才出单个 SID。

**★最重要规则:`<think>` 里禁止出现任何目标 SID / 目标 raw id / 目标 caption 原文 / 目标标题核心 n-gram。** 脚本化 hard filter:
```
reject if target_sid in think
reject if any target subtoken in think
reject if target_raw_id in think
reject if target_caption 4-gram overlap 过高
reject if think contains <s_a_ / <s_b_ / <s_c_
reject if final answer 不是恰好一个 valid SID pattern
```
**thinking 数据宁可少不可脏**:1 万条干净短 CoT > 20 万条带泄漏/幻觉的 CoT。

## 5. RFT:初赛最可能带来排名跃迁的步骤

流程:SFT-v1 模型对每个 R3 prompt 采 K=16/32/64(模拟平台 vLLM sampling)→ 解析最终 SID → 打 reward → 留高 reward 轨迹(exact hit/valid/非重复/多样)→ 用成功轨迹继续 SFT(学习率比首轮低 3–10 倍)。

**reward**:
```
r = 1.00·exact_hit + 0.15·valid_catalog_sid + 0.08·correct_domain
  + 0.05·same_s_a + 0.05·same_s_a_s_b + 0.03·unique_in_group
  - 0.20·duplicate_final_sid - 0.50·invalid_format - 1.00·leakage_in_think
```
Pass@64 = 64 条任一命中即得分 ⇒ 目标是**提高采样分布里目标 SID 的覆盖概率**,不是 greedy accuracy。故 reward 要含 valid/domain/prefix/unique,不能只用 exact_hit。

## 6. 先专后合,但只交单模型

`base → unified SFT → domain RFT experts → 收集成功轨迹 → 单一统一模型继续 SFT/RFT`。专家模型只用来**生成训练数据/高质量轨迹**,不做模型融合、不交 ensemble。最终交付:从 OneReason-0.8B-pretrain 继续训出的**单模型**(合规:复赛要交单模型)。

## 7. 训练顺序

- **Run A(R2-heavy,最快验证懂用户)**:R2 50% / R3 no_think 25% / R3 short-CoT 5% / R0 10% / General 10%。目标:懂用户从 0 拉起,不明显伤推荐/世界。**第一优先。**
- **Run B(R3-heavy,验证推荐)**:R3 no_think 50% / R3 short-CoT 15% / R2 20% / R0 5% / General 10%。目标:提四个推荐子项,看 think+nothink 合并 Pass@64。
- **Run C(RFT-v1,冲 Pass@64)**:起点 Run B 或 A/B 最优 ckpt;数据 R3 rollout 成功轨迹 + 少量 R2/General replay;低 lr 短 epoch。
- **Run D(Final balance)**:RFT R3 45% / R2 高质量 30% / General 10% / R0 5% / 官方 seed replay 10%。防 RFT 后懂用户/世界掉分。

## 8. 本地评估:三个代理指标(平台每日仅 3 次 + vLLM 波动)

1. **format validity**:valid_sid_rate / catalog_hit_rate / domain_begin_correct_rate / extra_text_after_final_rate / **think_leakage_rate(必须≈0)**。
2. **offline R2 F1**:从 50 万用户切**按用户去重**的固定 dev set(train/dev 不能同一用户,否则虚高)。
3. **offline R3 Pass@K proxy**:Pass@1/4/16/64 + unique_sid@64 / valid_sid@64 / domain_correct@64 / prefix_match@64。

**上传前门槛**:R2 F1 不降 ∧ R3 Pass@16/64 升 ∧ valid_sid@64 不降 ∧ world dev acc 不明显降。

## 9. 提交策略(每日 3 次 + ±1pt 波动)

- Day1:官方 SFT 复现(✅ 0.810) / Run A / Run B
- Day2–3:A/B 最优配比复训 / RFT-v1 / RFT-v1+replay
- Day4–7:专家轨迹蒸馏统一模型 / 更严 leakage-filter / final balanced
- **只有本地 proxy 与平台都连续改善,才把某 ckpt 作为新主干**;小于 ±1pt 不过度解读。

## 10. 五大坑

1. **CoT 泄漏目标 token**(最危险):学会在 think 里复述答案 → 平台 sampling 生成无效/重复/被解析误伤 → Pass@64 崩。
2. **R2 标签太松**:粗糙 tag 匹配出的 positives → 模型「看到同类就全选」→ precision 崩。必须加 hard negative + 近期但不相关反例。
3. **输入分布与评测 prompt 不一致**:训练大量给 caption/tag 而评测只给 SID 历史 → 模型依赖文本外挂。R3 建议 **80% itemic-only + 20% itemic+caption**;R2 可多用 caption/tag。
4. **过度 SFT 通识**:懂世界只 1 子项,General replay 防遗忘即可,别挤占 R2/R3 主容量。
5. **LoRA 未合并 / config 被改**:官方校验 config。所有产物必须:diff config.json vs base、vocab/special_tokens/tokenizer 不变、官方 inference stack 能加载、seed prompts 跑通。
