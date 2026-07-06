# 快手 LLM-Rec 2026 — 项目简报 (Project Brief)

> 面向研究人员/队友的单页概览。最后更新:2026-07-01。
> 详细文档索引见文末。分数更新见 `docs/experiment_log.md`。

---

## 〇、赛题背景

- **主办/平台**:快手探索者 LLM-Rec 挑战赛 2026,官方技术平台「万擎」(streamlake.com/product/wanqing)。目标:把大模型与生成式推荐融合,让一个模型同时具备懂物料/懂用户/懂推荐/懂世界四种能力。
- **核心范式**:物料(短视频/电商/直播/广告)不用自然语言,而用**语义 ID token** 表示——每个 item = 1 个域标记 + 3 层码本子 token,如 `<|prod_begin|><s_a_582><s_b_6878><s_c_7689>`(码本 3×8192)。模型直接「生成」这些 token 做推荐。
- **四维度与指标**:
  - 懂物料:描述 ↔ itemic token(Pass@64)
  - 懂用户:给历史 + 主题选相关行为(F1)
  - 懂推荐:全域历史 → next item(Pass@64;thinking 32 条 + non-thinking 32 条合并成 64)
  - 懂世界:通识不定项选择(Accuracy)
- **硬规则**:只能基于官方 `OneReason-0.8B-pretrain` 迭代(评测严格校验 config 一致,**不可改模型结构**);允许自造 SFT 数据/配比、自建 RL(baseline 不提供 RL 代码);允许蒸馏、不鼓励模型融合(须交单模型);复现阶段要交数据构造 + 训练脚本官方续训验证 ⇒ **一切须脚本化、可复现**。
- **评测机制**:自训模型上传万擎,平台跑 OneRec Benchmark 打分,**每日限 3 次**,推理参数平台固定(选手改不了)⇒ **提分靠训练数据与训练本身,不是推理调参**。
- **时间线**:初赛 7/1–7/31(进行中)→ 复赛 8/1–8/31 → 代码复现审核 9/1–9/15 → 决赛。奖金池 100 万。

---

## 一、官方 pre-train 参考分(锚点):总分 0.6655

| 维度 | 子项 | 小计 |
|---|---|---|
| 懂物料 (×1) | 0.1533 | 0.1533 |
| 懂用户 (×2) | 0.0000, 0.0055 | 0.0055 |
| 懂推荐 (×4) | 0.0864, 0.0544, 0.1372, 0.0900 | 0.3680 |
| 懂世界 (×1) | 0.1387 | 0.1387 |
| **总分** | | **0.6655** |

- **关键发现:总分 = 8 个子项直接相加(非加权平均),已用参考分精确验证。** 子项数:懂推荐×4、懂用户×2、懂物料×1、懂世界×1 ⇒ **懂推荐权重最大(小计 0.368),懂用户几乎为 0(最大提升空间)。**
- vLLM 推理有 **±1 个百分点** 不可避免波动(官方说明),不要过度解读亚 1pt 的变化。

---

## 二、已就绪(在手资产)

- **基座**:OneReason-0.8B-pretrain(Qwen3-0.6B 架构 + 3×8192 itemic 码本词表,vocab 176253),来源论文 = OneReason 技术报告(arXiv 2606.06260),已 108 页全文精读。
- **数据**:官方种子 SFT 3.2 万条(已加工)+ **完整 17GB 原始数据**(50 万用户行为序列 UserProfile + Pid2Sid/Caption/Tag 映射表 + 通识 General),已全部下载校验。
- **环境 + baseline**:LLaMA-Factory 全套环境搭好,已复现官方 SFT(train_loss 1.573,已上传评测/待出分);全流程脚本化可复现。
- **文献**:10 个赛题相关代码库(MiniOneRec/LC-Rec/ReRe/Rank-GRPO 等)+ 12 篇必读论文已拉取。

---

## 三、方法论主线(照 OneReason 论文)

有效推理 = **perception(itemic token 接地语义)+ cognition(连贯 CoT)**,顺序不可逆(perception 没打牢,thinking 打不过 non-thinking)。官方基座已内建强 perception ⇒ 算力应投在 **高质量 SFT CoT 数据 + RFT**,而非重训感知。

路线:R0–R3 三级认知 CoT SFT → specialize-then-unify RL(GRPO reward = 命中集合 × 码本多样性,对齐 Pass@64)。

---

## 四、下一步(按 ROI 排序)

1. **懂用户**(权重 2、现 ≈0,ROI 最高):从原始序列构造 R2 Action-Selection 数据(给历史 + 主题选相关行为,F1)。
2. **懂推荐**(权重 4、总分主力):复刻 R3 三阶段 CoT(Persona → Interest → Transition)+ 后续 RFT。
3. **懂世界**:混入通识数据防遗忘(能力注入偏 RL 而非纯 SFT)。
4. 全程守住:目标 token 不得进 `<think>`(防泄漏、否则 Pass@64 崩);R0/懂物料优先 non-thinking。

**关键风险**:懂推荐 SFT 若不混通识,可能拉低懂世界(遗忘)导致总分不升反降 ⇒ 每版必看**四维度分项 delta**,不只看总分。

---

## 文档索引 (docs/)

| 文件 | 内容 |
|---|---|
| `project_brief.md` | 本文件 — 单页概览,对外转发用 |
| `strategy_roadmap.md` | ★ 初赛后训练策略路线图(ROI/数据构造/RFT/提交计划) |
| `experiment_log.md` | 提交/分数记录(v0 锚点 0.6655 → v1 SFT **0.810**) |
| `onereason_report_digest.md` | ★ OneReason 技术报告全解读(基线来源,复刻蓝本) |
| `platform_and_baseline.md` | 万擎平台、赛事规则、官方 baseline 配方 |
| `baseline_repro_notes.md` | baseline 复现记录 + 上传步骤 |
| `data_report.md` | 官方种子 SFT 数据(3 赛道)剖析 |
| `hf_dataset_readme.md` | 17GB 原始数据 schema |
| `litreview_report.md` | 顶会论文调研报告 |
| `litreview_clone_list.json` | 已拉取代码库清单 |
| `papers/` | 12 篇必读论文 PDF |
| `demo_baseline/` | 官方 baseline 源码(convertv2/demo.yaml/scripts) |

**代码库**:`external_repos/`(MiniOneRec, SIDReasoner, ReRe, Rank-GRPO, RRec, LC-Rec, RPG, ReLLa, LETTER, RecZero)
**数据**:`ai_runtime/llmrec_2026/data/hf_full/`(17GB 原始)、`data/extracted/`(种子 SFT)
**模型/产物**:`models/OneReason-0.8B-pretrain-competition/`、`checkpoints/baseline_sft_v1/`、`submissions/baseline_sft_v1_platform/`
