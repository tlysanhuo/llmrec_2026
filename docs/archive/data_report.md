# Kuaishou LLM-Rec 2026 数据集与基座模型综合分析报告

> 综合 4 份 profiling 报告（懂物料 / 懂用户 / 懂推荐 三个赛道 + OneReason-0.8B 基座模型/分词器）。数据总量、模型 config、token id、README recipe 已对照真实文件校验通过。
> 关键结论前置：**(1) 懂世界 / R1（Item2Item 派生）赛道未随训练数据下发；(2) 懂推荐全部为 /think、无 /no_think，而 Pass@64 需要 32 think + 32 non-think，需自行构造；(3) 下发的是 pretrain-only checkpoint，SFT 需自己做。**

---

## 1. 数据总览表

| 赛道 (R-stage) | 文件 | 条数 | 输入 (prompt) | 输出 (response) | 评测指标 |
|---|---|---:|---|---|---|
| **懂物料** (R0 感知/grounding) | 7 个 `懂物料part1-7.jsonl` | **10,384** | system(48 模板之一, 含方向+域) + Chinese 指令(内嵌描述 或 内嵌一个三元组) + `/think`\|`/no_think` | `<think>`(no_think 为空 / think 填充) + 换行 + 答案(desc→token 出 1 个三元组; token→desc 出中文描述) | **Pass@64**(desc→token 精确三元组召回); token→desc 语义相似/辅助 |
| **懂用户** (R2 演化/选择) | 1 个 `懂用户.jsonl` (76MB) | **2,892** | system **恒为空**; 纯文本带日期历史时间线 + 角色任务 + **主题(100% 存在)** + 输出格式 + `/think`\|`/no_think` | **`<think>` 恒为空** + JSON: FORMAT A `json_list`(扁平 token 数组) 或 FORMAT B `logic_chain` 对象 | **F1**(相关行为选择集的 P/R) |
| **懂推荐** (R3 推荐) | 4 个 `懂推荐1-4.jsonl` | **19,204** (5426+5442+5372+2964) | system(6 释义) + 多域历史行为块 + 收尾指令 + `/think` | 填充式 `<think>`(固定【兴趣归纳】+【行为模式】) + 换行 + **单行「该用户最近&lt;动作&gt;: &lt;单个 item token&gt;」** | **Pass@64**(32 think + 32 non-think 合并; 命中即真值出现在 64 个候选中) |
| **懂世界** (R1 派生/I2I) | **无** | **0（未下发）** | — | — | README 有 R1 I2I 列，训练数据缺失 |
| base_model | `OneReason-0.8B-pretrain-competition/` | — | Qwen3 safetensors + tokenizer | — | 需自行 SFT（README: SFT coming soon） |

**三赛道训练总量 = 32,480 条。** 三个数据文件共用同一物理 schema：**每行是长度为 1 的 JSON 数组，包裹一个 `{system, prompt, response}` 对象——`json.loads` 后取 `[0]`。** 单一 loader 即可覆盖三赛道。

---

## 2. 分赛道深度剖析

### 2.1 懂物料 — R0 感知 / itemic grounding（10,384 条）

**任务**：在 4 个内容域（prod=商品 / video=短视频 / living=直播主播 / ad=广告）上做**双向** 描述↔itemic-token 映射。

- **两个子任务近似均衡交织**：desc→token 5,597 条 (53.9%)，token→desc 4,787 条 (46.1%)。
- **思考模式近似 50/50**：`/think` 5,182 (49.9%) vs `/no_think` 5,202 (50.1%)；**每条 response 都含 `<think>` 块**——`/no_think` 下是字面空块 `<think>\n</think>`，`/think` 下填充。
- **`<think>` 内容随方向截然不同**：desc→token 是**逗号分隔关键词表**（中位 32 字符，约 5 段，几乎不成句）；token→desc 是**1-3 句抽象类目描述**（中位 66 字符，总以「。」结尾）。0/10384 的 think 块里出现 item token——思考永远是自然语言。
- **域分布**：prod 3,200 / ad 3,200 / video 3,200 各 30.8%；**living 仅 784 (7.6%) 且只出现在 desc→token 方向**（无 token→desc）。part2 整个文件都是 living。
- **token 语法 100% 干净**：10384/10384 三元组格式良好，严格 a→b→c 层序，`<|TYPE_begin|>` 与请求域匹配；层值范围 a∈[16,8189] / b∈[5,8191] / c∈[0,8183]（码本 0..8191）。**每条答案侧恰好 1 个 token，永不为 0 或多个**（prompt/resp item 计数完美互补 {0:5597,1:4787}↔{0:4787,1:5597}）。

**代表样例**：
- *desc→token /think (prod)*：PROMPT 内嵌「赶海专用不锈钢夹子…价格61.04元…/think」→ RESP `<think>赶海夹子，不锈钢材质，加长防滑，滩涂浅水，垂钓爱好者</think>\n<|prod_begin|><s_a_6091><s_b_2919><s_c_2941>`
- *token→desc /think (prod)*：PROMPT「给定商品token<|prod_begin|><s_a_2043><s_b_5277><s_c_1251>…/think」→ RESP `<think>面向大众及中老年女性的平价舒适女鞋与舞蹈用品…</think>\n这是一款专为民族舞蹈设计的女式高跟舞鞋…`（think 是**类目先验**，与最终具体描述不必精确一致）
- *desc→token /no_think (prod)*：RESP `<think>\n</think>\n<|prod_begin|><s_a_5051><s_b_5171><s_c_7920>`（空 think）

**关键陷阱**：token→desc 的 think 是「码本簇先验」而非最终描述规格；living 只有 desc→token；不存在 sid 域；商品描述含大量模板噪声（价格X元/免运费/172800小时发货），模型应关注语义属性而非模板短语。

---

### 2.2 懂用户 — R2 兴趣演化 / 子集选择（2,892 条）

**任务**：给定用户**完整带日期的交互时间线**（`【YYYY-MM-DD】` 后接 `  --:-- [域-行为] <token>` 行）与一个**兴趣演化主题**，**从用户自己的历史中抽取相关行为子集**。这不是未来预测、不是自由推荐——**输出 token 100% 是历史 token 的子集（2886 条非空 response 全部 frac=1.0 验证通过）**。

- **system 恒为空字符串**；**主题 100% 存在**（在「主题：」之后）。
- **思考控制 `/no_think` 2290 (79.2%) / `/think` 602 (20.8%)，但 `<think>` 块在 100% response 中都为空**（真正的推理被外化到 FORMAT B 的 `logic` 字段）。
- **两种互斥输出格式，由「输出格式」指令文本决定，而非 `/think`**：
  - **FORMAT A `json_list`**（1,588 条，全 /no_think）：扁平 token 字符串数组；选中数 min 1 / 中位 **11** / max 56——高召回「全部相关行为」。
  - **FORMAT B `logic_chain`**（1,304 条 = 702 /no_think + 602 /think）：`{"logic_chain":{"name":<主题>,"events":[{date,action,logic}...]}}`；events min 2 / 中位 **3** / max 6——紧凑因果链，高精度。
- **规模**：历史事件 min 5 / 中位 191 / max 438；prompt min 551 / 中位 **12,862** / max **28,699** 字符（长上下文）；97 条 response 含重复 token（gold 未严格去重）。
- **域分布（历史事件占比）**：prod 188,071 / video 186,351 / ad 152,012 / living 49,964；response 侧 prod 10,832 / video 5,954 / ad 3,232 / living 1,772。行为词表：点击/长播/购买/深度转化/关注/首次打赏/打赏/收藏·点赞·评论·转发/长播/浏览/加购 等。

**代表样例**：
- *FORMAT A*：指令「请仅以包含 SID 的 JSON 数组形式返回…」→ `<think>\n</think>\n["<|living_begin|>…","<|prod_begin|>…", …（17 个 token，全部来自上文历史）]`
- *FORMAT B*：`<think>\n</think>\n{"logic_chain":{"name":"从泛化皮肤问题到特定功效产品的精准定位","events":[{"date":"2026-02-04","action":"[视频-评论/长播] <|video_begin|>…","logic":"问题触发与初步认知…"},{…"[商品-点击] <|prod_begin|>…","logic":"行为转化与决策收敛…完成认知闭环。"}]}}`

**关键陷阱**：格式由「输出格式」文本分支（不是 /think）；把全部 response 当 list 解析会在 1304/2892 条上失败；指令字面写「SID」但实际吐出的是 prod/video/living/ad token（**无任何 `<|sid_begin|>`**）；FORMAT B 的 `name` 须与主题逐字一致，且遵守「场景驱动协同 / 兴趣因果递进 / 需求深度细化」并**禁止品类平级/重复/凑单**行为。

---

### 2.3 懂推荐 — R3 推荐 / 下一个 item（19,204 条）

**任务**：给定用户**多域行为历史**（视频深度观看/点赞长播、电商浏览/加购/购买、直播关注/打赏/首次打赏、广告点击/深度转化，皆为 itemic token），模型先在 `<think>` 内按**固定 rubric** 推理（【兴趣归纳】3-4 类兴趣并内联引用证据 token、【行为模式】+ 跨域协同），再输出**恰好一个** 目标 item：`该用户最近<动作>: <单个 item token>`。

- **100% `/think`，0 条 `/no_think`**；所有 19,204 条 response 均含非空 `<think>`。
- **最终答案恰好 1 个 item token**（5965/5965 抽样最终行验证 item-count==1）；**response 中出现的 ~8 个 token 是 `<think>` 里的证据引用，不是答案。**
- **每条只针对一个场景**（虽然 system 写「各场景/各推荐场景」，数据每行只给一个目标）。
- **输出目标分布（全量 19,204）**：视频 14,868 (77.4%) / 广告 1,576 (8.2%) / 商品 1,489 (7.8%) / 直播首次打赏 1,271 (6.6%)。
- **长度**：prompt 中位 ~7,831 字符 / ~167 个 item token（max 440 / 20,603 字符）；response 中位 ~1,474 字符，因 rubric 固定而**长度高度稳定**（多落在 1.2K-1.9K）。
- **输入域频次（抽样）**：video 494,804 / prod 442,893 / ad 357,578 / living 91,708（living 最稀疏）。

**代表样例**：
- *OUTPUT=video*：PROMPT「…深度观看了 <|video_begin|>…（200+ 视频 token）…/think」→ `<think>好的，请看我的分析：\n\n【兴趣归纳】1. 社会百态… 2. 乡村生活… 3. 传统文化… 4. 实用技能…【行为模式】…</think>\n该用户最近喜欢的视频有: <|video_begin|><s_a_532><s_b_4571><s_c_2820>`
- *OUTPUT=living*：即使 prompt 只有 3 个历史 token，仍产出 ~1.5K 字符 rubric 完整的 think——**薄证据也强推理，有幻觉兴趣类目风险。**

**关键陷阱**：目标是单 token 不是列表；无 /no_think 数据但指标需要 32 non-think；think 里引用了 prompt 中没有的元数据/「站内搜索」信号（教师有 item 侧 metadata，学生推理时没有，不可依赖）；behavior block 用中文标点拼接，header 正则须非贪婪；prod 域 header 是「用户购物行为/浏览了商品」但别处又叫「电商」，命名跨域不一致。

---

### 2.4 基座模型 / 分词器 — OneReason-0.8B（pretrain-only）

`Qwen3ForCausalLM`，路径 `models/OneReason-0.8B-pretrain-competition`（已对照 config/tokenizer 校验）。

- **架构**：hidden 1024，28 层，16 头 / 8 KV，head_dim 128，rope_theta 1e6，max_position 40960，QK-norm，`tie_word_embeddings=false`。
- **参数**：**801,433,600** BF16（embed 180.48M + **untied** lm_head 180.48M + backbone 440.47M）；`model.safetensors` 1.603GB（=params×2+header；index 的 `total_size` 1,871,839,232 高估）。
- **词表**：`vocab_size=176253` = 151,643 base BPE + 24,610 added（**24,576 码本 token = 3×8192** + 34 specials，含 8 个域 token）。id 布局：`<s_a_0>`=151669 … `<s_c_8191>`=176244；域 token video 176245/176246、prod 176247/176248、living 176249/176250、ad 176251/176252。
- **特殊 token**：`<think>`=151667 / `</think>`=151668；`<|im_start|>`=151644 / `<|im_end|>`=eos=151645 / `<|endoftext|>`=pad=151643。`generation_config` eos=[151645,151643]，do_sample 默认 temp 0.6 / top_k 20 / top_p 0.95；`add_bos_token=false`。
- **思考门控（chat template）**：由 `enable_thinking` kwarg 控制（非 slash 文本）；`enable_thinking=false` 会**自动插入空 think 块**；训练需在 prompt 上 mask loss。
- **训练配方（README）**：Pretrain 578B token（S1 扩词表+LM head 110B / S2 全参 449B / S3 长上下文 19B）→ SFT（coarse-to-fine CoT）→ RL（specialize-then-unify）。推理阶段 **R0 感知 / R1 派生(I2I) / R2 演化 / R3 推荐**。
- **现状**：**仅发布 pretrain checkpoint，SFT/RL「coming soon」——我们必须自己做 SFT。**

**关键陷阱**：第 5 个 sid 域**无专用 token，会被 BPE 切分**；embed 与 lm_head 已 untied，需确保 24,576 个 itemic 行真正参与训练；eos 在 config(151645) 与 gen(151645,151643) 不同。

---

## 3. 跨赛道观察

1. **统一物理 schema**：三赛道每行都是长度-1 JSON 数组包 `{system,prompt,response}`，取 `[0]`。单 loader 通用。
2. **统一 token 语法**：`<|TYPE_begin|><s_a_N><s_b_N><s_c_N>`，N∈0..8191，3 层 RQ 码本，TYPE∈{prod,video,living,ad}。每个「item」实为 ~4 个 vocab token。懂物料侧 100% 良构。
3. **「sid」第 5 域是幽灵**：spec 与懂用户指令文字写「SID」，但**任何赛道都无 `<|sid_begin|>`**——「SID」在此泛指「item 语义 ID token」。基座 tokenizer 也无 sid 专用 token。切勿生成 sid。
4. **`<think>` 块永远物理存在，但语义三态**：懂物料（no_think 空 / think 填关键词或类目句）；懂用户（**100% 为空**，推理外化到 logic 字段）；懂推荐（**100% 填充**固定 rubric）。
5. **思考控制面不一致**：数据里是 prompt 文本 `/think`·`/no_think`；模型 chat template 里是 `enable_thinking` kwarg。二者需在训练时对齐。
6. **思考比例逐赛道不同**：懂物料 ~50/50；懂用户 ~79/21（且**格式由输出指令决定，不由 think 决定**）；懂推荐 100% think / 0% no_think。
7. **赛道↔官方 R-stage 映射（README 佐证）**：懂物料=R0（item 理解/grounding/QA）；懂用户=R2（演化 action 选择 / topic 生成 / direct 演化生成——恰好对应 json_list 选择 vs logic_chain 生成）；懂推荐=R3（单域+跨域推荐）。**R1（Item2Item 派生）无训练文件——即缺失的「懂世界」赛道。**
8. **指标对齐（README benchmark）**：推荐类是 Pass@64 / Recall@64，按 C-Video / C-Product / C-Ad / C-Live **分域**报告；R0 grounding 用 Pass@64。README 同时报「SFT non-thinking」与「SFT thinking」两行——**证明 Pass@64 混合 think+non-think，所有赛道两种模式都要覆盖。**
9. **域不平衡系统性一致**：living/直播在每个赛道都最稀疏（懂物料 7.6% 且仅 desc→token；懂推荐目标 6.6%、输入 token 最少）。video/prod 主导。跨域协同（从密集域向稀疏域迁移）是懂推荐显式计分的推理要素。
10. **懂推荐 think 有 item 侧元数据泄漏**：引用了 prompt 中不存在的内容标签（「情感金曲演绎」「亲子育儿」）与「站内搜索」信号——教师有、学生推理时没有，架构上不可依赖。
11. **长度均为字符近似**：数据 agent 无 transformers/tokenizer，所有长度是字符/`char//4` 近似；真实 token 长度须用真分词器重测后再定 max_seq_len。

---

## 4. 建模启示与开放问题

**建模启示**
- **统一多任务 SFT**：一模型联合训三赛道两方向两模式。懂物料双向互相强化码本语义；把「永远存在的 `<think>` 脚手架」当结构学（no_think 下须可靠吐 `<think>\n</think>`），格式合规是白送的分。
- **Pass@64 需要 non-think 数据，而懂推荐 100% think**——必须自造 /no_think：取同一 (prompt, 最终单 item)，清空 think、改 `/no_think` 后缀补入，否则 32 个 non-think 候选处于分布外。
- **可评方向靠采样多样性**：desc→token 与推荐下一 item 都是 Pass@64「64 中命中」，候选跨码本的多样性 > 单点 greedy 精度；从 gen 默认(temp 0.6/top_k 20/top_p 0.95)起调温度/核采样，并约束解码为严格 a→b→c 层序 + 正确 `<|TYPE_begin|>`。
- **懂用户按输出指令分支**（非按 think）：json_list 高召回、logic_chain 高精度；强制子集不变式（输出 token 逐字来自历史）、logic_chain 的因果/去重/禁凑单约束、name 与主题逐字一致。F1 下过选伤精度、欠选伤召回，须按格式校准长度。
- **域再平衡**：若 eval 按域等权（README 分域报告暗示如此），则训练 77% 视频偏斜会低配 prod/ad/living——上采样 living/ad，并为懂物料 living 合成 token→desc 方向。
- **长上下文与拷贝保真**：懂用户 prompt 达 28.7K 字符、懂推荐达 20.6K；截断从最旧历史前端截，保住指令与 `/think` 后缀；itemic token 任何损坏都会破坏 F1/精确匹配。
- **基座**：确保 24,576 码本行 + 8 域行真正参与训练（embed/lm_head untied）；训练前用真分词器确认每个 itemic token 映射到单一 id 而不被 BPE 切分。

**开放问题**（详见结构化字段）
- R1/懂世界（Item2Item 派生）训练数据在哪？是仅评测、需自造、还是后续下发？——最大覆盖缺口。
- Pass@64 是严格 32 think + 32 non-think 合并，还是两模式取优？决定是否必须造 non-think 推荐数据。
- token→desc、logic 字段、懂用户两格式各自的真实计分函数与 eval 权重？是否提供 dev/test 划分？
- eval 场景/域配比是否等于训练配比（懂推荐 77/8/8/7）？README 分域等权暗示不等——影响上采样策略。

---

## 5. 推荐的下一步

1. **用真分词器重测**每赛道 token 长度分布（现全为字符近似），据 p90/p99 设 max_seq_len（懂用户、懂推荐最长）；确认 itemic token（id 151669-176252）不被 BPE 切分。
2. **搭统一 SFT loader**：解析行→取[0]→按赛道路由；显式建模 `<think>` 脚手架，prompt 上 mask loss；确认 24,576 码本行 + 8 域行有梯度。
3. **自造 /no_think 推荐数据**：懂推荐每条 (prompt, 单 item) 清空 think、换 `/no_think`，补入以覆盖 Pass@64 的 32 non-think；审计懂用户（其 /think 行 think 本已空，可两模式复用）。
4. **升级 R1/懂世界 缺口给组织方/人类**确认：I2I 训练数据是否会来、是否仅评测、还是需自造（若自造，可从懂用户历史与懂推荐 prompt 的共现关系合成 I2I 对）。
5. **实现 Pass@64 多样性采样**（desc→token 与推荐），从 gen 默认起调温度/核采样，约束严格 a→b→c 与正确域标记。
6. **按 eval 权重再平衡域**（先确认 eval 是否分域等权）：上采样懂推荐 living/ad、上采样并为懂物料 living 合成 token→desc。
7. **扩充指令释义**（超出 48/6/数十套固定模板），使方向/格式/场景识别泛化，勿过拟合下发模板。
8. **懂用户按「输出格式」文本分支** json_list vs logic_chain，强制子集不变式与 logic_chain 选择约束。
9. **小样本 SFT 冒烟测试**（每赛道 1-2K 条）先验证管线（schema 解析、token id 完整性、think 脚手架、loss mask、懂推荐单最终 item），再全量训；在留出 prompt 上校验格式合规（格式漂移=纯 Pass@64/F1 损失）。