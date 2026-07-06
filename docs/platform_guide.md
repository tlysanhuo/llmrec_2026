# 万擎平台完全指南(合并版)— 快手 LLM-Rec 2026

> 2026-07-04 合并自三处:`platform_and_baseline.md`(07-01 全量整理)+ `platform_intro_v2.md`(07-02 官方更新)+ 用户转交的平台训练服务默认参数表与官方任务定义(07-03/07-04)。**平台侧信息以本文件为唯一权威;上述两个旧文件保留作历史存档。**
> 官方指南 URL:https://www.streamlake.com/document/WANQING/mh1g8b8aunh8esspfm(开发机使用指南)

## 一、官方任务定义与评测机制(07-04 用户提供官方原文)

| # | 任务 | 官方定义 | 评测 |
|---|---|---|---|
| 1 | **物品理解(懂物料)** | 生成物料描述对应的 itemic pattern | **Pass@64**(beam64×3tok,日志实测) |
| 1+ | (**复赛开启**)pattern→物料描述 | 生成 itemic pattern 对应的描述 | **LLM-as-a-Judge**(准确、完整覆盖关键信息) |
| 2 | **用户兴趣演化(懂用户)** | ①判断哪些历史行为与给定主题相关(action_select);②**生成该兴趣的演化链**(topic_gen) | ①**F1**;②**综合评估演化链中行为的准确性与推理逻辑的合理性** |
| 3 | **推荐物料(懂推荐)** | 预测下一个物品(ad/live/prod/video 四域) | **Pass@64 = thinking 模式 32 条 + non-thinking 模式 32 条合并**(官方确认;与我方日志逆向的"两阶段 beam32 + 直通 beam32"一致) |
| 4 | **常识问答(懂世界)** | **单项选择题** | **Accuracy** |

⚠️ 本表修正两条旧认知:①懂世界是**单选**(旧记录误为"多选全对才得分");②topic_gen 不是自由生成,是**演化链**且评推理逻辑——懂用户第二子项的数据构造应对齐"演化链"形态。

**分项分数已加权(Q13)**:各子任务难度不同、平台设权重让量纲一致——我们看到的 8 个分项是加权后的值,总分=8 项直接相加。

**评测配额**:每日 3 次,**北京时间 15:00 刷新**,**账号级与队友共享**;失败不消耗。评测耗时 ~1h10-1h35。**评测方差警告(07-04 选手情报):同一模型可打出 0.86-0.94**——±0.04 内的比较不可靠,分项同涨同跌比总分可信。

## 二、平台训练服务(模型定制-精调)

**默认参数全表(07-03 用户从平台 UI 抄录)**:

| 参数 | 默认值 | 参数 | 默认值 |
|---|---|---|---|
| 迭代轮次 | 3(建议1-5) | LoRA Rank | 32 |
| **学习率** | **1e-6**(LoRA 默认;⚠️极低) | LoRA Alpha | 32 |
| 单卡批大小 | 1 | LoRA Dropout | 0.1 |
| **序列长度** | **32768** | 调度 | cosine |
| 预热比例 | 0.03 | 正则化(wd) | 0.01 |
| 梯度累积 | 8 | Checkpoint间隔 | 64 |
| Packing | true | 精度 | bf16 |
| thinking 模式 | 可开关(enable_thinking,训推需一致) | | |

**万擎指南页(选手指路)的参考配置**:LoRA lr=2e-4 / **全参 lr=2e-5**,cosine+warmup0.03+batch1×4+seq32768+packing+neat_packing+bf16+**seed 19260817**——与官方 demo.yaml 完全一致,即**我们的本地底座配置=官方推荐值**;心定的 LoRA 0.85+(r16/lr2e-4)就是抄的这份。

**★平台 loss 与本地不同(07-04 选手对账情报)**:平台训练服务 = **focal loss + token 加权**;本地 LLaMA-Factory = 普通 CE。平台文档自身的表述(v2):"自定义 loss 函数(如商品 token 加权、难样本聚焦)"。该选手本地 CE 训 0.94 > 平台训 0.88(方向:平台不一定更好,取决于数据与 focal 的咬合);另有转述"纯平台训练到 0.99"(**未核实**,与榜首 0.96 矛盾,可能是本地 eval/子项/误传)。→ 我方对策:本地实现 focal(scripts/train/train_focal.py,γ=2,rebal_focal_ep3 在训)做单变量验证。

**训练资源(Q14)**:初赛平台不保证训练资源,自备算力 + 每周限量(3B+);复赛才给每队资源;不提供免费 API 做数据构建。

**队友 baseline-epoch3 实例**:平台默认参数(LoRA lr1e-6)×3ep = 0.7807——lr 过低几乎不动模型,action 归零(pretrain 基线≈0 训不起来),但 ad 0.0768/prod 0.1498 反而好(保住 pretrain 原生推荐能力)。**用平台训练服务必须手动把 lr 提到有效量级**。

## 三、硬规则(FAQ 汇总)

- **Q2** 只能基于 OneReason-0.8B 迭代;评测**严格校验 config 与 baseline 一致**(vocab 176253/tokenizer 不可动——约束解码/加 special token 路线不可行)。
- **Q3** baseline 不提供 RL 代码,自行实现。
- **Q6** 可自定义 SFT 配比、自建 RL;不可改模型结构/预定义参数/评估设置。
- **Q7** 数据高度灵活(可引外部通识、自造推荐数据);**复现阶段交数据构造+训练脚本,官方续训复现,误差内通过**→ 一切构造脚本固定 seed。
- **Q8** 线下训练建议 Transformers v5.3.0;全参传 `model.safetensors`(分片加 index);LoRA 传 adapter 两件套。
- **Q10** 允许蒸馏;**不鼓励模型融合**;复赛须交单模型。
- **Q12** pretrain 锚点 **0.6655**(物料0.1533/用户0.0000,0.0055/推荐0.0864,0.0544,0.1372,0.0900/世界0.1387);vLLM 波动~1%(⚠️07-04 选手实测方差远大于此,见上)。

## 四、官方 baseline 配方(demo/,我们的本地底座)

框架 LLaMA-Factory 0.9.6.dev0,全参 SFT:`qwen3_nothink` 模板 / cutoff 32768 / packing+neat / lr 2e-5 / cosine / warmup 0.03 / wd 0 / 1ep / batch 1×4 / bf16 / seed 19260817 / liger+fa2。环境 pin:torch 2.7.1+cu126、flash-attn 2.7.4.post1、liger 0.8.0(`scripts/baseline/00_install.sh`,现装于 lustre `ai_runtime/llmrec_2026/LLaMA-Factory/.venv`)。

数据处理(convertv2.py)关键逻辑:`--max_token_types 3`(response 内 s_X 字母超 a/b/c 三种即丢);filter_sid 规范化(`<|live_begin|>`→`<|living_begin|>` 等);**think 注入标准**:assistant 无 think → user 加 `/no_think` 且 assistant 前缀 `<think>\n\n</think>\n`;有非空 think → user 加 `/think`。

## 五、评测生成机制(日志逆向,与官方定义互证)

- 物料 grounding:**beam64 × 3 token**(合法率100%;s_a 错全错;实测 distinct s_a 仅 11-15,beam 大量耗在 s_c 枚举)。
- 推荐四域:**两通路各 beam32**(thinking 采样 n=1/max4096 → 接 beam32;nothink 直通 beam32),合并 64 候选——与官方"thinking/non-thinking 各 32"表述一致。
- action_select:采样 n=1 / max_tokens 4096(复读会顶满截断)。
- 常识:采样,max_tokens 60000,thinking disabled;**单选,Accuracy**。
- 评分:SID+PID 双模式,sid2pid 映射 video_ad 共用 1.66M / prod 131k / live 35k(**ad 与 video 共享 codebook**)。

## 六、数据资产(官方,HF OpenOneRec/Explorer_LLM_Rec_Competition)

种子 SFT 32480 条(=data_final);原始素材:UserProfile 50万用户 / Pid2Sid 3591万 / Pid2Caption 2106万 / Pid2Tag 541万 / General 通识。详见 `docs/DATA_INVENTORY.md` 与队友全量调研 `docs/hf_raw_data_analysis.md`。

## ★★2026-07-06 官方FAQ全文要点(用户转发,当日最高优先级情报)

### 面板列序官方定义(引发全账本rec域重标)
> 评测结果栏从左往右:总分、懂物料、懂用户(选择)、懂用户(主题生成)、**懂推荐(短视频)、懂推荐(电商)、懂推荐(广告)、懂推荐(直播)**、世界知识。
- 官方baseline参考分(0.6655)与我方v0行逐位相同 ⇒ **col1=video(量子0.0096)、col2=prod(0.0034)、col3=ad(0.0014)、col4=live(0.0009)**。
- **我方历史台账把四列按 ad/live/prod/video 误读(按日志任务字典序推的),全部rec域结论需重标**:跷跷板=物料↔**video**;"保9争10"是video;5ep"+6题"是**ad**;tokengeo"−13暴跌"是**ad**;LoRA保住的先验=video+ad。总分/物料/action/topic/world 不受影响。
- 各rec域评测题量相同(~1000/域,日志实测),量子差=难度权重差(官方Q:权重为量纲一致而设,不公开)。

### 码本墙官方解释(Q5,定案)
> HF数据为原始表采样;SFT数据是**另一批采样**构造的,SFT中的sid在HF pid2sid可能找不到。
⇒ 评测物料在HF外是设计使然,HF物料数据永远覆盖不到评测题 ⇒ "界外物料=毒/无效"有了因果解释;队友48k物料判死进一步坐实。

### 规则要点
- 蒸馏**明确允许**;外部数据允许但**需提交所用数据以复现**(DS蒸馏合规,脚本+产物已留档);模型融合全程**不鼓励**,复赛需单模型方案过审。
- 上传只收 **bf16**(fp32过不了一致性校验);config须与baseline一致(锁0.8B);平台不支持CPT;eval阶段不可加程序。
- **评估任务需当日12:00前结束**才进当日15:00榜 ⇒ 上传实操截止≈10:30(评测1h+)。
- 平台精调服务训的模型无需上传ckpt;本地训的每次都要传。
- 官方承认vllm评测波动≈1个百分位,后续会多次取均值。

### 官方Tips(数据/RL方向)
- **懂推荐CoT Pattern**:baseline是"兴趣归纳→行为模式→预测总结"三段式,官方明说**优化空间大**;点名 **CoT/UnCoT配比** 值得探(=Frinkleko重组方向的官方背书)。
- 官方CoT样例披露:action_select出现**候选演化链选择题格式**(候选A/B/C→答案[A,B],与种子提取式并存);懂物料复赛含pattern→desc;还有"潜在需求共鸣"等新颖关系数据形态。
- RL tips:rollout要group diversity(勿盲目加size)、避免all-correct/all-wrong零优势样本;reward=做对推荐为主+format/validity约束有效;CoT与answer token分开clipping/loss weight;无效负样本降权。
