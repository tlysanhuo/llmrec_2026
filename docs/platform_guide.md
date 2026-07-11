# 官方全案(唯一官方信息文档)— 快手 LLM-Rec 2026

> **★★★每次会话必读(用户 2026-07-08 明令)**:所有官方材料(平台参数/任务定义/评测机制/FAQ/赛题解析PPT/官方Tips)只放这一个文档;**动任何训练配置之前,先过 §〇 官方参数对账表**;偏离官方默认值必须在 config 头注写明理由。
> 2026-07-04 合并自三处:`platform_and_baseline.md`(07-01 全量整理)+ `platform_intro_v2.md`(07-02 官方更新)+ 用户转交的平台训练服务默认参数表与官方任务定义(07-03/07-04)。2026-07-08 大修:并入赛题解析 PPT 40 页全量重读要点 + 官方参数对账表。旧文件保留作历史存档。
> 官方指南 URL:https://www.streamlake.com/document/WANQING/mh1g8b8aunh8esspfm(开发机使用指南);赛题解析 PPT 原件:`docs/reference/赛题解析_官方.pdf`(40页)。

## 〇、官方参数总对账表(2026-07-08 建;新配置出生前必查)

**训练参数(四个官方来源 × 我方现行)**:

| 参数 | 平台训练UI默认 | 万擎指南页参考 | 官方demo.yaml(全参) | 我方 riders(现行最高分LoRA) | 判读 |
|---|---|---|---|---|---|
| 制度 | LoRA | LoRA 或 全参 | 全参 | LoRA | ✓(群内四方佐证LoRA) |
| **迭代轮次** | **3(建议1-5)** | — | 1 | **1** | **✗ 官方默认3轮,我方lr2e-4档从未测过>1ep——最大的未对齐旋钮** |
| 学习率 | 1e-6(⚠️已证废:几乎不动模型) | **LoRA 2e-4 / 全参 2e-5** | 2e-5 | 2e-4 | ✓(按指南页;UI默认1e-6不可抄) |
| LoRA rank / α | 32 / 32 | — | — | 32 / 32 | ✓ |
| **LoRA dropout** | **0.1** | — | — | **0.05** | ✗ 无理由偏离 |
| 单卡批大小 / **梯度累积** | 1 / **8** | 1 / 4 | 1 / 4 | 1 / 4 | UI=8,指南页=4;我方=4 |
| 序列长度 | 32768 | 32768 | 32768 | 32768 | ✓ |
| 调度 / 预热 | cosine / 0.03 | cosine / 0.03 | cosine / 0.03 | cosine / 0.03 | ✓ |
| **正则化 wd** | **0.01** | 0.001(心定笔记) | 0 | **0.001** | ✗ 三个来源三个值;UI权威=0.01 |
| Packing | true | packing+neat_packing | packing+neat | packing+neat | ✓ |
| 精度 | bf16 | bf16 | bf16 | bf16 | ✓ |
| thinking 开关 | 可开关,**训推需一致** | 关thinking(心定0.85+) | qwen3_nothink 模板 | qwen3_nothink | ✓(评测侧think开关见下表) |
| seed | — | 19260817 | 19260817 | 19260817 | ✓ |
| **loss** | **focal + token加权(平台独有)** | — | 普通CE | 普通CE | 已知差异;本地focal γ=2已证伪,**平台版focal未测** |
| Checkpoint间隔 | 64 | — | — | epoch | 无影响 |

**评测侧推理参数(赛题解析PPT p22 逐字;训练数据 think/nothink 配比的官方依据)**:

| 任务 | think模式 | Temperature | Top-k | Top-p |
|---|---|---|---|---|
| 懂物料 | **\no_think** | 0.7 | 20 | 0.8 |
| 懂用户(两子项) | **\no_think** | 0.6 | 20 | 0.95 |
| 懂推荐 | **\no_think + \think 双路各32** | 0.6 | **50** | 0.95 |
| 懂世界 | **\no_think** | 0.7 | 20 | 0.8 |

> **★推论(07-08)**:评测时**只有懂推荐的一半通路开 think**,其余全部 nothink——而种子懂用户 100% 带 CoT,存在训/推 think 形态错位;Frinkleko 的 nothink 重组、官方 Tips 的"CoT/UnCoT 配比"、心定"关thinking"全部指向同一件事。**数据侧任何新版本必须按这张表配 think/nothink。**

**衍生纪律(07-08 用户两次纠正后的正确用法)**:这张表**不是抄数值用的**。①它的价值=理解测量仪器与官方判断:评测侧 think 开关/解码参数定义了训练数据该长什么样;"UI默认3轮(建议1-5)"是官方认为 LoRA 制度需要多遍的**机制信号**,不是"把3抄进config"的指令;②**已被线上验证的自家值(如 riders 0.9177 的 dropout0.05/wd0.001/accum4)优先于官方默认**,不因不同而改;③lr 1e-6 是唯一已线上证废的官方默认值;④改任何超参必须有机制假设,"官方是这个值"或"高分选手是这个值"都不构成机制假设。


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

### ★★2026-07-09 晚官方新发布:SFT 对齐 Caption/Tag 数据(HF `SFT/` 目录)

官方原话:"为便于参赛选手更好地理解预制的懂推荐数据,并进一步构造推荐任务样本。我们在 Hugging Face 发布了与预制 SFT 数据中'懂推荐'部分对齐的 Caption/Tag 数据。该数据提供物料 token 对应的内容描述与类目信息,选手可据此分析用户历史行为和目标内容语义,并**探索不同 CoT 构造方式对推荐任务表现的影响**。"

- **文件**:`SFT/baseline_caption_tag_lists.parquet`(730MB;本地已下 `data/hf_sft_aligned/`)。官方 README 字段:`record_id`(0-19203)/`messages`(原始 SFT messages JSON)/`sid_token_list`/`caption_list`/`tag_list`(三 list 按位置一一对齐,未找到为 null)。
- **官方口径规模**:19,204 行(=种子懂推荐行 1:1)、SID token 位置 3,539,794。
- **我方实测(07-09)**:caption 位置覆盖 98.3%、tag 38.3%;**唯一 SID 568,944、其中 97.7% 有 caption**(分域 video 98%/ad 97%/prod 100%/living 99%);caption 长度中位 234、p90 338(叙述式,与评测物料 desc 同风格带 259-424);41,110 个 SID 有多视角 caption;⚠️living 域 caption 多为标签列表体非散文。**与 SFT 对齐 ⇒ 不受 Q5 码本墙约束**(区别于旧 Pid2Sid 的"另一批采样")。
- **首个衍生品**:`data/processed/cap_grounding_v1.jsonl`(5,441 行 desc→SID 增料,构建器 `scripts/data/build_cap_grounding.py`,md5 `092f7ba5`;模板逐字采样自官方种子物料行,散文过滤+同域同 caption 去矛盾+input 全局唯一,QC 全绿)。

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

## ★★官方《赛题解析》PPT 全要点(07-06 入库;**07-08 逐页全量重读补全**,原件 docs/reference/赛题解析_官方.pdf,40页)

### 各赛道评测机制(比此前情报更细)
- **懂物料**:beam64 生成 itemic pattern → **经 pattern→item_id 映射表转成 item id**(一 pattern 多 id 时取最新Pid;无效pattern记0)→ 与 gold item_id 比对,任一命中即过。**评测 desc 是 API 从原生多模态信息生成的**(≠Pid2Caption,连 desc 风格都不同源——caption 类物料数据连"题面分布"都对不上)。评测 system prompt 逐字:"你是一位视频数据分析专家,负责将视频文本映射为精确的视频token。"
- **懂用户 action_select**:gold=**API 两步标注**(①API 从纯文本历史抽主题;②给定历史+主题,API 抽相关交互)→ F1。**不是规则标注!**我方 R2 规则 gold 与评测 gold 存在系统性分布差。
- **懂用户 topic_gen(logic chain)计分**:①按 action 做**最优有序匹配**(顺序敏感);②action 粒度惩罚漏生成+过度生成;③logic 文本在匹配上的 event 算 **Token-F1 与 ROUGE-L-F1 均值**(公式(30)-(38)逐字在原件 p14-15);④总分=action对齐与logic对齐的均值。规则:≤5步;同日同类交互合并用"；";必须是场景需求补全/兴趣因果递进/需求深度细化三类演进;禁浅层并列。**logic 文本按 ROUGE 计分 ⇒ 模仿官方 logic 行文风格直接得分**。**⚠️07-08 日志补充:平台 EvolutionTopicGenEvaluator 实际还加载 NLI CrossEncoder(nli-deberta-v3-base)——匹配含语义分量,与 PPT 的纯 Token-F1/ROUGE 描述不完全一致,topic 数据不能只堆 n-gram 风格。**
- **懂推荐**:prompt 域顺序规则=**目标域最后、视频倒数第二、广告倒数第三**;answer 可含**多个** gold item ids,64池任一命中任一 gold 即过。各域历史构造字段级配方(行为判定+保留条数,原件p17逐字):video 按组合event_type各留50(rsft_score>0.75→长播);live 同主播留更强行为、各50;电商 点击/加购/购买各50≈150(ts=ec_time_ms−lag天);广告 深转+点击各70≈140。**⇒ 评测同分布 prompt 的完整复刻配方到手(rec_loo v3/RFT 用)**。
- **懂世界(07-08 补全)**:system 逐字"你是一个非常聪明的助手,请直接遵循指示作答";题面末尾格式指令逐字:"请按以下格式作答:'正确答案是 (在此处填写选项字母)'"。**答案抽取=按优先级逐条正则扫全文**(样例:`(?:正确)?答案(?:应该)?(?:是|为|应为|应当是)\s*[:：]?\s*[\(（]?[A-Z][\)）]?`、`最佳答案(?:是|为)…`),第一个匹配到合法字母的提取、去重、升序返回;**少选/错选/多选/解析失败都算错**。⇒ 抽取对格式有一定容错,但训练里"占位符复读"型输出(rebal_pstack事故)会被正则抓成错误答案。
- **官方推理参数(p22 逐字,含 think 开关)**:已提炼为 §〇 第二张表——**除懂推荐 think 通路外,评测全部 \no_think**。

### 官方数据产线(两张图,可用 DS 复刻)
- **懂推荐 CoT=拒绝采样产线**:LLM **不见 gold** 生成 K 条 CoT → LLM-as-judge 按"CoT 与 gold 物品语义相关性"独立打分 → 阈值τ过滤留高分 CoT 入库(避免事后解释捷径)。(与 Tips"CoT Pattern 优化空间大"呼应)
- **懂用户 R2 产线**(官方就叫R2):全域行为时间线 → LLM 提取兴趣演化(触发/细化/修正/闭合) → 候选链筛选 → **LLM 裁决质量过滤**(时间顺序/证据支持/认知增量/因果合理/避免主观臆测) → 三种任务形态:①演化行为选择(=action_select) ②演化主题生成 ③演化链直接生成。
- **源数据总览(07-08 补全,p30)**:行为序列与 SFT 数据**来自同一份原始表、仅采样不同用户序列、且与预训练数据隔离**。Pid2Caption 全表覆盖率 58.64%(video 55.26%/live 99.69%/ad 69.21%/goods 60.73%),Pid2Tag 15.08%。

### Pretrain 背景(07-08 补全为精确三阶段,p24-25;定超参直觉用)
| 阶段 | 训练方式 | Token预算 | seq | 学习率 |
|---|---|---|---|---|
| Stage1 词表预热 | **冻结 Transformer 主干,只训扩展词表+LM Head** | 110B | 4K | **2e-4 退火至 1e-4** |
| Stage2 全参共训 | 全参;四粒度推荐数据×通用域混合 | 449B | 4K | **1e-4 退火至 1e-5** |
| Stage3 长序列 | 全参;扩至32K,建模长程行为依赖 | 19B | 32K | **1e-5 退火至 1e-6** |

- RQ-Kmeans 三级 8192 码本(8192³≈5498亿组合);Token格式 `<domain_begin><a_X><b_Y><c_Z>`;多域物料统一表示。
- 推荐对齐数据 416B(Token-Item-Relational-User 四粒度)+通用 162B。**官方原话(p9):"只有将推荐数据与通识数据混合训练,才有可能使得推荐模型获得 Reasoning 能力"**——通识混入的官方背书。
- 四粒度任务形式(p26):Token粒度=子Token语义组合/前缀语义预测/Token预测;Item=描述对齐/粗化降噪/多视图多来源;Relational=物品关联/兴趣转移路径/多步兴趣流;User=分域行为序列/时间交错序列。
- **Pretrain 数据样例形态(07-08 补全,p27-28)**:①Token粒度问答体("短视频域中,<a_3664><b_3076>表示什么?");②**User粒度含选择题体**("以下哪些商品是该用户浏览过的?A/B/C"→"A;C")和**多轮续推体**("请继续给出其他点击广告")——**MC 锚/续推形态在 pretrain 阶段官方就喂过**,解释了 pretrain 对选择题格式有底子、CEval 锚为何能精确治愈占位符复读。

### SFT-CoT 官方样例(07-08 补全,p34)
- 懂物料含 **item QA 新形态**:"潜在需求共鸣"关系判断题(两表面不同的视频满足同一深层心理需求,四选一)。
- **懂推荐同一 prompt 官方给出 think/nothink 两个版本**(一份带完整 think 正文,一份空 think 直出)——Frinkleko 重组机制的官方原型,坐实"同 prompt 双形态"是官方设计而非选手发明。
- 懂用户样例带完整 think(候选链逐一论证后输出 [A,B])。

### 评测 FAQ(p36)
- 分项已按难度加权(权重不公开);vLLM 评测波动难免,**后续会对波动大的 benchmark 多次评估取均值**。

### 官方 Tips(p38-39)
- **Tips-SFT**:baseline 懂推荐 CoT 是"兴趣归纳→行为模式→预测总结"三段式,**官方明说优化空间大**;推荐 CoT 设计四步框架:**Induction(归纳用户抽象)→ Abduction(溯因,兴趣发散)→ 溯因主导+演绎(兴趣权衡推导)→ Deduction(演绎,物料ID生成)**;点名"**溯因的解释空间大,需要压缩:降噪、找到最重要的部分**";鼓励自探 CoT Pattern 和 **CoT/UnCoT 配比**。
- **Tips-RL**:rollout 要 group diversity(勿盲目加 size)、避免 all-correct/all-wrong 零优势样本;reward=做对推荐为主+format/validity 约束有效;监控 entropy/response length/KL/reward variance;**CoT 与 answer token 分开 clipping/loss weight**;无效负样本降权。
