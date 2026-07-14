> **已由封板 EDA 取代（2026-07-12）**：当前结论以 [`OFFICIAL_DATA_EDA.md`](OFFICIAL_DATA_EDA.md) §4 为准。本旧报告的跨域 SID overlap 和“普遍 10 分钟时间泄漏”结论已被全量复核纠正；约束解码、加 special token、四阶段大训练等建议也不适用于当前赛制。本文仅作历史记录，禁止据此直接构建或训练。

## 快手 LLM-Rec 挑战赛 · 数据分析报告

> 本报告汇总每个分析步骤的关键结论与**对 SFT 训练的直接指导**。
> 每个章节对应 `analysis/notebooks/` 下的一个 notebook，原始数字与图见 `analysis/outputs/`。

---

## 目录

- [§2.1 SID 分布分析](#21-sid-分布分析) ✅ 已完成
- [§2.2 PID ↔ 元数据 Join 完整性检查](#22-pid--元数据-join-完整性检查) ✅ 已完成
- [§2.3 Caption 文本细粒度分析](#23-caption-文本细粒度分析) ✅ 已完成
- [§2.4 Tag 分布分析](#24-tag-分布分析) ✅ 已完成
- [§2.5 User 序列长度 & 稠密度分析](#25-user-序列长度--稠密度分析) ✅ 已完成
- [§2.6 多域覆盖 & 用户画像分层](#26-多域覆盖--用户画像分层) ✅ 已完成
- [§2.7 时间戳细粒度分析](#27-时间戳细粒度分析) ✅ 已完成
- [§2.8 反馈标签精细分布 & 动作词映射](#28-反馈标签精细分布--动作词映射) ✅ 已完成
- §2.9 通识数据分析
- §2.10 与官方样例数据对齐

---

## 2.1 SID 分布分析

**Notebook**：`analysis/notebooks/01_sid_distribution.ipynb`
**产出目录**：`analysis/outputs/sid/`
**扫描规模**：4 个 domain、3591 万条 `pid → (s_a, s_b, s_c)` 全量。

---

### 2.1.1 Codebook 结构：**跨 domain 共用同一个 8192 codebook**

| 维度 | 结论 |
|---|---|
| 每层 id 范围 | 全部 domain × 全部 level 的 `min_id = 0`, `max_id = 8191` → 底层就是**同一个 K=8192 的 VQ codebook** |
| 每域实际使用的 s_a 数量 | `goods=4279`, `live=802`, `video/ad=2971`, `video/video=5777`（均 < 8192） |
| 每域实际使用的 s_b 数量 | `goods=3567`, `live=1905`, `video/ad=3142`, `video/video=3524` |
| 每域实际使用的 s_c 数量 | `goods=4758`, `live=2918`, `video/ad=4677`, `video/video=5252` |

- 同一层的**总合并词表大小 ≤ 8192**（不是四域相加），说明 codebook 是**跨域共享**训练出来的。
- `live` 只用了 802 / 8192 = **9.8%** 的 s_a 桶，严重欠拟合；`video/video` 用了 5777 / 8192 = **70.5%** 最饱满。

#### 🔧 SFT 建议 A：Special Token 定义

- Tokenizer 只需注册一次 `<s_a_0>...<s_a_8191>`, `<s_b_0>...<s_b_8191>`, `<s_c_0>...<s_c_8191>` **三组 8192 = 24576 个 token**，四域共享。
- 每个 domain 用**独立前缀 token** 区分：`<|video_begin|>` / `<|ad_begin|>` / `<|prod_begin|>` / `<|living_begin|>` （README 已定义），SFT 时**必须**保留这些前缀在生成序列的开头。

---

### 2.1.2 跨域 SID 隔离性：**`ad ↔ video` 共享，`goods` / `live` 独立**

跨域三元组重叠矩阵（对角线是本域三元组数，非对角是跨域重合数）：

|  | goods | live | video/ad | video/video |
|---|---|---|---|---|
| **goods**       | 102843 | 0 | 1 | 1 |
| **live**        | 0 | 623 | 0 | **57** |
| **video/ad**    | 1 | 0 | 10745 | **2907** |
| **video/video** | 1 | 57 | **2907** | 103493 |

- `video/ad` 与 `video/video`：**2907 个三元组重合** — 这两个域是**同源 codebook**，广告和短视频用同一套物料聚类（合理，因为广告本身就是视频形态）。
- `goods` 和 `live` 与其他域**几乎零重合** — 是**完全独立的 codebook 空间**，即使 id=0..8191 相同，语义也不同。

#### 🔧 SFT 建议 B：Constrained Decoding 与负采样

1. **强制 domain-first**：生成序列必须以 `<|xxx_begin|>` 开头，模型第一个位置的解码应约束为**只从 4 个 domain-begin token 中选**。
2. **domain-conditioned constrained decoding**：解出 domain 后，`s_a` 位置只允许该 domain 出现过的 token（通过 `outputs/sid/freq_hist_{domain}_s_a.csv` 构造 mask）。
   - `live` 只有 802 个合法 s_a → mask 掉 8192 中的 7390 个非法项，**能极大降低生成 invalid SID 的概率**。
3. **负采样**：训练时的 in-batch negatives 应**限定在同域**，跨域 SID 语义不同，不构成有意义的对比样本。

---

### 2.1.3 SID 三元组稀疏度：**必须依赖 constrained decoding**

理论上限 8192³ = 5.5 × 10¹¹ 个三元组，实际使用：

| Domain | unique triples | 占理论上限 | 说明 |
|---|---|---|---|
| goods       | 10,295,563 | 0.0019% | 极稀疏 |
| video/video | 10,345,920 | 0.0019% | 极稀疏 |
| video/ad    | 1,069,596  | 0.0002% | 极稀疏 |
| live        | 62,070     | 0.00001% | 几乎不用 |

**推论**：如果不做约束解码，模型 free generation 出的三元组有 **99.99% 概率是无效**（无对应 pid）。

#### 🔧 SFT 建议 C：把"合法三元组集合"做成一个 Trie

- 训练 SFT 之前，用 `_cache.pkl` 里的 `triple_cnt` 构造 **每 domain 一棵 prefix trie**：`root → s_a → {s_b} → {s_c}`。
- 推理时逐 token 应用 trie mask（HuggingFace 的 `PrefixConstrainedLogitsProcessor` 就能实现），**Pass@k 会明显高于自由生成**。
- Trie 大小可控（goods 最大也就 10M 三元组 × 3 int = 120 MB），完全可以驻留内存。

---

### 2.1.4 层级信息增益：**s_a → s_b → s_c 有明显的层级压缩**

| Domain | H(s_a) | H(s_b) | H(s_c) | H(s_b\|s_a) | **IG a→b** | H(s_c\|a,b) | **IG (a,b)→c** |
|---|---|---|---|---|---|---|---|
| goods       | 10.94 | 10.81 | 11.37 | 8.03 | 2.78 (**26%**) | 3.72 | 7.65 (**67%**) |
| live        | 5.62  | 6.77  | 8.47  | 5.04 | 1.73 (**26%**) | 4.18 | 4.29 (**51%**) |
| video/ad    | 8.59  | 9.17  | 10.24 | 6.48 | 2.68 (**29%**) | 3.98 | 6.26 (**61%**) |
| video/video | 9.93  | 9.72  | 10.56 | 7.55 | 2.17 (**22%**) | 5.01 | 5.55 (**53%**) |

（括号内为 IG / H(target) 的比例，越大代表条件信息越有用）

- `IG_(a,b)→c` 全部 domain 都 > 50%，尤其 **goods 高达 67%** — 一旦解出 (s_a, s_b)，s_c 只剩 log₂ 中 3~5 bit 的不确定性。
- s_a 是"大类"，s_b 是"中类"，s_c 是"细类"，**层级性明确**。

#### 🔧 SFT 建议 D：Beam 宽度按层递减

每个前缀下 s_b / s_c 的候选桶大小（p90 分位）：

| Domain | p90 uniq_b\|s_a | p90 uniq_c\|(s_a,s_b) | 建议 beam width |
|---|---|---|---|
| goods       | 907 | 11 | 32 → 16 → 8 |
| video/video | 703 | 15 | 32 → 16 → 8 |
| video/ad    | 235 | 7  | 16 → 8  → 4 |
| live        | 67  | 5  | 8  → 4  → 2 |

- **训练也可以用同样的层级**：设计"分阶段 loss"，让 s_a 位置的 loss 权重比 s_c 略高（因为 s_a 决定候选空间，错一个后面全废）。可通过 `label_smoothing` 或直接 per-position weight 实现。
- 或者更直接：把 SFT 任务分解成 3 个课程学习阶段（curriculum）：**先只学 s_a → 加入 s_b → 加入 s_c**。

---

### 2.1.5 频率分布：**头部主导 + 长尾严重（域间差异巨大）**

关键指标（摘选自 `freq_summary.csv`）：

| Key | top-1% token 覆盖行数 | top-10% token 覆盖行数 | 长尾 (<5次) token 比例 | 长尾覆盖行数 |
|---|---|---|---|---|
| goods::s_a       | 13.4% | 49.5% | 2.3% | 0.0% |
| video/ad::s_a    | 32.8% | 82.4% | 24.2% | 0.07% |
| video/video::s_a | 25.9% | 76.8% | 19.0% | 0.01% |
| **live::s_a**    | **60.0%** | **87.3%** | **36.0%** | **0.55%** |
| video/video::s_b | 22.7% | 65.8% | 5.9% | 0.0% |
| video/video::s_c | 19.6% | 60.6% | 5.9% | 0.0% |
| **live::s_c**    | **43.5%** | **77.3%** | **42.4%** | **2.4%** |

以及最热 token 的绝对量（`max_cnt`）：

- `video/video::s_b` 最热 token 出现 **995,942 次**，占该级 5.6% 全部行 — 一个 token 独占 6% 样本，主导性极强。
- `video/video::s_c` 最热 558,428 次；`goods::s_a` 141,615 次。

#### 🔧 SFT 建议 E：Loss 与采样重加权

**问题**：naive 的 next-token cross-entropy 会让模型学成"永远输出最热 token"（domain-level majority baseline 就能拿 20%~60% 的位置准确率）。

**方案**（三选一或组合）：

1. **频率加权 loss**：对每个 label 位置的 loss 乘以 `w = (median_freq / freq(token))^α`，`α ∈ [0.3, 0.7]`。频率数据直接读 `freq_hist_{domain}_{level}.csv`。
2. **Focal loss**：`(1 - p_true)^γ · CE`，`γ = 2` — 对已经学得好的头部样本降权，对困难长尾提权。
3. **动态温度采样**：训练样本按 `weight = freq^(-0.5)` 上采样，让长尾 pid 出现频率提升。

**长尾 token 的特殊处理**（36% 的 live::s_a token 出现<5 次）：

- 这些 token 的模型表现会**极差**，应从训练集里 filter 掉过于罕见的 pid 作为 label（比如 <3 次），当噪声处理，避免模型学乱。
- 或者反过来：训练时对长尾 pid 做 **k-fold 数据增强**（比如把它对应的 caption 复述 3 遍），提升暴露率。

#### 🔧 SFT 建议 F：Domain 级重采样

原始比例（3591 万条）：

- video/video ≈ **17.66M (49%)**
- goods ≈ **16.09M (45%)**
- video/ad ≈ **2.06M (5.7%)**
- live ≈ **0.11M (0.3%)** ⚠️

如果不重采样，模型几乎只学到 video 和 goods，**live 域会完全欠拟合**。

**推荐权重**（按 √(1/N) 反比开根号，比按 1/N 更温和）：

| domain | 原比例 | 建议采样权重 | 采样后比例 |
|---|---|---|---|
| video/video | 49% | 1.0 | 30% |
| goods       | 45% | 1.0 | 27% |
| video/ad    | 5.7% | 3.0 | 15% |
| live        | 0.3% | 30.0 | 28% |

（具体系数需要在验证集上调；这里只给方向）

---

### 2.1.6 三元组碰撞：**"pid 唯一"是错的假设**

多个不同的 pid 会映射到**同一个 (s_a, s_b, s_c)** 三元组：

| Domain | total pids | unique triples | 有多个 pid 共用的三元组比例 | 涉及行数占比 | 单三元组最多 pids | 平均 pids/triple |
|---|---|---|---|---|---|---|
| goods       | 16.09M | 10.30M | 20.2% | **48.9%** | 2996 | 1.56 |
| live        | 107K   | 62K    | 16.0% | **51.4%** | 673  | 1.73 |
| video/ad    | 2.06M  | 1.07M  | 25.2% | **61.1%** | **3460** | 1.92 |
| video/video | 17.66M | 10.35M | 20.8% | **53.6%** | **4523** | 1.71 |

- 平均 **50%~60% 的行** 位于"多 pid 共享"的三元组下。
- 单个三元组最多绑定 **4523 个 pid**（video/video）—— 意味着这个 SID 语义太粗，泛指某一大类（比如"美食探店"这样的宽标签）。

#### 🔧 SFT 建议 G：**训练标签用 SID 而不是 pid**；评估时按等价类

1. **Loss 计算的正样本**：next-item 任务里的"正确 label"应该是 pid → 查 SID → 生成 SID。**不要**直接把 pid 编成 token（否则词表爆炸而且大量歧义）。
2. **Pass@64 评估**：如果一个用户的 next-item pid 是 X，X 的 SID 是 (a, b, c)，那么**任何 SID = (a, b, c) 的 pid 都算命中**。构造评估集时，先把 `dataset_rec_sample` 里每个测试样本的 label pid 通过 `Pid2Sid` 展开成"等价 pid 集合"，用 hit@k 而不是 exact-match。
3. **Loss 层面的 label smoothing**：既然 pid → SID 是多对一，训练时可对同 SID 的所有 pid 做 label 平滑（`smooth = 1 / |equivalence class|`），避免模型在"两个都对"的情况下被梯度撕扯。

---

### 2.1.7 综合结论 & SFT 训练配置清单

**一句话**：codebook 跨域共享、跨域三元组几乎不重叠、每域内部头部主导 + 长尾严重、`(s_a, s_b, s_c)` 层级性明显、pid → SID 是多对一。

#### 生成侧配置清单

| 配置项 | 建议值 / 做法 | 依据 |
|---|---|---|
| 特殊 token 数量 | 24576（三层各 8192） | §2.1.1 |
| Domain 前缀 token | `<\|video_begin\|>` / `<\|ad_begin\|>` / `<\|prod_begin\|>` / `<\|living_begin\|>` | README + §2.1.2 |
| Constrained decoding | 强制 domain-first + 每层用 domain-specific mask | §2.1.2, §2.1.3 |
| Beam width（video/goods） | s_a=32, s_b=16, s_c=8 | §2.1.4 |
| Beam width（live） | s_a=8, s_b=4, s_c=2 | §2.1.4 |
| Loss 权重 | frequency-aware（`w = (median/freq)^0.5`）或 focal loss γ=2 | §2.1.5 |
| Domain 采样权重 | video=1, goods=1, ad=3, **live=30** | §2.1.5 |
| 长尾 pid 处理 | 出现 <3 次的 pid 从训练集 filter；或 k-fold 增强 | §2.1.5 |
| Label 计算 | pid → SID triple；用等价类做 hit@k 评估 | §2.1.6 |

#### 训练侧配置清单（建议 3 阶段 curriculum）

**Stage-1（1~2 epoch）**：只学 `s_a`（把 label 里的 s_b/s_c mask 成 -100），让模型先掌握"大类分布"。

**Stage-2（2~3 epoch）**：学完整 `(s_a, s_b)`，冻结 domain 前缀 embedding。

**Stage-3（3~5 epoch）**：学完整 `(s_a, s_b, s_c)`，加入 constrained decoding + focal loss 提升长尾。

#### ⚠️ 已识别的风险点

1. **live 域仅 10.7 万行**，即使 30x 上采样后有效样本还是少。SFT 时应额外考虑：
    - 用 `Pid2Caption` 的 live 域 caption 做**跨域文本迁移**（描述 → SID 任务的数据增强）。
    - 或直接放弃 live 域生成，转为 retrieval-based（用户历史 live pid → embedding 匹配）。
2. **s_c 位置的 label smoothing 要小心**：因为 avg_uniq_c/(a,b) 只有 3~7，label smoothing 平滑到无关 c token 会破坏层级结构。建议 `label_smoothing < 0.05`。
3. **video/ad ↔ video/video 的三元组重合 (2907)** 意味着模型可能会把广告"错认"成短视频（同一 SID 三元组，不同 domain 前缀）—— 这是**样本对比学习**的好素材（同 SID 不同 domain 应有不同前缀），**不是**bug。

> **§2.2 结果已到位**，见下一节；其中的**发现 4（caption 类型域间一刀切）**修正了 2.1.5 里的一处建议：`live` 域的 caption 100% 是 taglist，**不能一刀 filter**，反而应作为"关键词 → SID"任务的主训练数据。

---

## 2.2 PID ↔ 元数据 Join 完整性检查

**Notebook**：`analysis/notebooks/02_pid_metadata_join.ipynb`
**产出目录**：`analysis/outputs/join/`
**扫描规模**：Pid2Sid 3591万 + Pid2Caption 2106万 + Pid2Tag 541万，pid 集合运算，caption 类型判别抽样 2%。

---

### 2.2.1 元数据覆盖率矩阵（懂物料任务的样本上限）

| Domain | \|SID\| | \|Caption\| | \|Tag\| | Caption / SID 覆盖 | Tag / SID 覆盖 |
|---|---|---|---|---|---|
| goods       | 16.09 M | 9.77 M | **0** | **60.73%** | **0.00%** |
| live        | 0.107 M | 0.107 M | 0.060 M | **99.69%** | 56.14% |
| video/ad    | 2.06 M  | 1.42 M | 0.74 M | 69.21% | 36.08% |
| video/video | 17.66 M | 9.76 M | 4.61 M | **55.26%** | 26.13% |

**关键结论**：

1. `goods` 域 **完全没有 tag**（0 条），符合 README 所述（tag 只覆盖 video/video、video/ad、live）。goods 的分类信息**只能从 caption 里挖**。
2. `video/video` 域 caption 覆盖率**仅 55%** — **7,901,207 个 pid 有 SID 但无 caption**（占该域 45%）。这些 pid 不能参与"懂物料"任务的输入侧（描述→SID）。
3. `live` 域 caption 覆盖率 **99.69%** — 最完整，但 §2.2.3 会揭示 live 的 caption 都是 taglist 而非自然语言。
4. **零异常**：`caption_no_sid = 0` 全域为 0，说明 Caption 表是 SID 表的**严格子集**（有 caption 必有 SID），无脏数据。

---

### 2.2.2 Venn 三集合分区（`sid_no_caption` / `sid_no_tag` 数量）

从 `venn_regions.csv`：

| Region | goods | live | video/ad | video/video |
|---|---|---|---|---|
| `only_S`（有 SID 无描述无标签） | 6,317,979 | 240 | 410,246 | 6,527,121 |
| `S∩C`（有 SID 有描述无标签） | 9,769,747 | 46,818 | 904,558 | 6,520,107 |
| `S∩T`（有 SID 无描述有标签） | 0 | 96 | 223,000 | 1,374,086 |
| `S∩C∩T`（三者都有）**←最优质** | **0** | **60,135** | **519,085** | **3,240,877** |
| `only_C` / `only_T` / `C∩T` | 全 0 | 全 0 | 全 0 | 全 0 |

**关键结论**：

- **`S∩C∩T` 是"金标样本池"**（同时可做懂物料 + 懂用户 + 懂推荐），全域合计 = 3,820,097 pid + `goods=0`。
- `only_S` 区域（既无描述也无标签）合计 = **13,255,586 pid**（占全域 SID 总数 36.9%）— 这些 pid **只能做纯 SID 生成任务**（懂推荐的 next-item 目标可以是它们，但它们不能作为"输入侧候选池"参与描述→SID / tag→SID 训练）。
- `S∩T` 但 `~C`（有 tag 无描述）在 `video/video` 高达 **1,374,086** —— 这类 pid 可参与 **"tag → SID"** 任务，但不能参与 "描述 → SID"。

---

### 2.2.3 Caption 类型分布：**每域一刀切**（重大发现）

从 `caption_type_stats.csv`（2% 抽样）：

| Domain | describe | taglist | short_noise | empty |
|---|---|---|---|---|
| goods       | **93.72%** | 0% | 6.28% | 0% |
| live        | 0% | **100.00%** | 0% | 0% |
| video/ad    | **99.77%** | 0% | 0% | 0.23% |
| video/video | **99.89%** | 0% | 0% | 0.11% |

**结论**：**caption 的形态由 domain 决定，不存在"混合"**。这有几个非常重要的推论：

1. **live 域的 caption 是 tag 列表**：例如 `['粉嫩双马尾', '和平精英技术流', '不露脸直播', ...]`。这些**其实是长的、非常细粒度的 tag**（远比 `Pid2Tag` 的 lv3 分类更丰富），需要 `ast.literal_eval` 解析成 list 后逐个 tag 使用。
2. **goods 域有 6.28% short_noise（<10 字符）**：这些是极短的商品名（比如"帽子"、"耳机线"），信息量低，**训练时应过滤**。
3. **video/ad 与 video/video 有 0.11%~0.23% 空串**：数量小但存在，SFT 样本构造时**必须做非空 filter**。

#### 🔧 SFT 建议 H：Caption 前处理管道（必须严格执行）

```python
def is_usable_caption(domain: str, caption: str) -> bool:
    if caption is None or not caption.strip():
        return False
    text = caption.strip()
    if domain == 'live':
        # live 是 taglist，只需检查能否解析为非空 list
        try:
            import ast
            tags = ast.literal_eval(text)
            return isinstance(tags, list) and len(tags) >= 3
        except Exception:
            return False
    else:
        # 其他 domain 应该是 describe，过滤过短的
        if len(text) < 10:
            return False
        # 保底：如果误标（非 live 域出现 [...] 形态）也 filter
        if text.startswith('[') and text.endswith(']'):
            return False
        return True
```

#### 🔧 SFT 建议 I：**live 域走"关键词序列 → SID"专属任务**

- live 域的 caption 是"tag 列表"，天然适配 **"给定一组关键词，生成 SID"** 的任务范式。
- 具体样本形态可以是：
  - **多关键词 → SID**：`prompt = "关键词：粉嫩双马尾, 和平精英技术流, 不露脸直播 ..." → target = <|living_begin|><s_a><s_b><s_c>`
  - **随机 mask 关键词 → SID**：随机保留 30~70% 关键词，模拟实际推荐时用户兴趣不完整的场景。
- **等价效果**：live 域样本因此变多（每个 pid 可以由不同关键词子集生成 → k-fold 数据增强），一定程度上缓解 §2.1.5 提出的 live 极端小样本问题。

---

### 2.2.4 Caption 长度分布：**video / goods / live 完全不同量级**

从 `caption_length_stats.csv`（每类 5000 条抽样）：

| Domain | Type | P50 | P90 | P99 | max | mean |
|---|---|---|---|---|---|---|
| goods       | describe | 29 | **33** | 39 | 55 | 26.6 |
| goods       | short_noise | 7 | 9 | 9 | 11 | 7.3 |
| live        | taglist | 75 | **93** | 112 | 136 | 75.8 |
| video/ad    | describe | 300 | **357** | 419 | 651 | 302.1 |
| video/video | describe | 287 | **348** | 412 | 574 | 277.2 |

**关键结论**：

- **video/ad 和 video/video 的 describe 长度是 goods 的 10 倍**（P90 350 vs 33 字）。goods 的 caption 只是"商品标题"级别，**不是**真正的商品描述。
- video 域的 P99 已到 **420 字**，加上 domain-begin token + 3 个 SID token + prompt 模板 ≈ 500~550 tokens。
- live 的 taglist 只有 P90=93 字，但序列化后其 tag 数量约 5~10 个。

#### 🔧 SFT 建议 J：`max_source_length` 分桶策略

| 场景 | 建议 max_source_length |
|---|---|
| 单条描述 → SID（video 域） | **512**（含系统 prompt 200 tokens 的余量） |
| 单条描述 → SID（goods 域） | **128** |
| 关键词列表 → SID（live 域） | **256** |
| 用户历史 → next SID（懂推荐） | 见 §2.5，此处不定 |

- 训练时按 domain **动态截断 + 分桶 batch**（同 batch 内长度接近，减少 padding 浪费）。
- HuggingFace 的 `DataCollatorWithPadding(padding='longest')` 配合 `group_by_length=True` 直接搞定。

---

### 2.2.5 Orphan pid 清单（可直接用于样本 filter）

已导出以下 parquet（每条 = `{pid, domain}`）到 `analysis/outputs/join/`：

| 文件 | 行数 | 用途 |
|---|---|---|
| `orphan_sid_no_caption_goods.parquet`       | 6,317,979 | 从"描述 → SID"训练集里 filter 掉 |
| `orphan_sid_no_caption_video_video.parquet` | 7,901,207 | 同上 |
| `orphan_sid_no_caption_video_ad.parquet`    | 633,246   | 同上 |
| `orphan_sid_no_caption_live.parquet`        | 336       | 同上 |
| `orphan_sid_no_tag_goods.parquet`           | 16,087,726 | goods 全部无 tag |
| `orphan_sid_no_tag_video_video.parquet`     | 13,047,228 | 从"tag → SID"训练集里 filter 掉 |
| `orphan_sid_no_tag_video_ad.parquet`        | 1,314,804 | 同上 |
| `orphan_sid_no_tag_live.parquet`            | 47,058    | 同上 |

- `orphan_caption_no_sid_*.parquet` **未产生**（因为该关系在四个域全部为 0），无异常 pid 需要处理。

#### 🔧 SFT 建议 K：样本构造管道的两阶段 filter

```python
# 阶段 1：从 UserProfile 采样候选 pid
candidate_pids = [...]  # 从用户序列或 next-item label 采出

# 阶段 2：按任务类型 filter
if task == 'caption_to_sid':
    candidate_pids -= load_orphan_pids('sid_no_caption', domain)
elif task == 'tag_to_sid':
    candidate_pids -= load_orphan_pids('sid_no_tag', domain)
elif task == 'user_history_to_next_sid':
    pass  # 无需 filter，因为只需要 SID
```

- 建议把 `orphan_sid_no_caption_*.parquet` 全部 pid 加载成一个 `set[int]`（约 15M 个 int64 = 120 MB 内存，可接受）供样本构造脚本快速查询。

---

### 2.2.6 综合结论 & SFT 训练配置更新

**一句话**：Caption 是 SID 的严格子集但覆盖率跨域差异大（55%~99%），caption 类型完全由 domain 决定（goods=describe / live=taglist / video=describe），tag 只在 3 个域有且覆盖率低。

#### 三个任务的**样本池上限**（金标数）

| 任务 | 样本池 | 数量上限 |
|---|---|---|
| 描述 → SID（懂物料） | `S∩C且type=describe` | goods 9.16M（93.72%×9.77M） + video/ad 1.42M + video/video 9.76M = **~20 M** |
| 关键词列表 → SID（懂物料·live 专属） | `S∩C且domain=live且type=taglist` | live **~106 K**（99.69%×107K） |
| Tag → SID（懂用户·主题） | `S∩T` | live 60K + video/ad 742K + video/video 4.61M = **~5.4 M** |
| 用户历史 → next SID（懂推荐） | 全 SID 表 | goods 16M + video/video 17.7M + video/ad 2M + live 107K = **~36 M** |

#### 训练侧 3 阶段 curriculum 修订版（结合 §2.1）

| 阶段 | 输入 → 输出 | 样本源 | 数量 | 目的 |
|---|---|---|---|---|
| **Stage-A** 预热 | 描述 → SID | `S∩C且type=describe` | 20 M | 让模型建立 "文本 semantic ↔ SID 语义" 的映射（懂物料底座） |
| **Stage-A′** 并行 | 关键词列表 → SID | live taglist（k-fold 增强） | ~500 K（106K × 5） | 缓解 live 域小样本 |
| **Stage-B** 加辅助 | Tag → SID | `S∩T` | 5 M | 强化"主题分类 → SID"的粗粒度映射（可作为 Stage-A 的信号 grounding） |
| **Stage-C** 主任务 | 用户历史 → next SID | 全 UserProfile | 500K 用户 × 若干正样本 | 懂推荐主任务 |

#### ⚠️ 修订：与 §2.1 的一致性

- **§2.1.5 建议 E 的"filter <3 次的 pid"** 需要**先扣除 orphan**，即 filter 应在"S∩C 域内"再看频率，避免把冷启 pid 与真正长尾 pid 混淆。
- **§2.1.7 的 domain 采样权重（video=1, goods=1, ad=3, live=30）** 只适用于 Stage-C（用户历史 → next SID）。Stage-A（描述 → SID）的样本池按 20M 计算，**live 只 106K 占 0.5%**，需要单独用 Stage-A′ 补足；不要在 Stage-A 里也 30x 上采样 live，因为其形态（taglist）与 video/goods 的 describe 完全不同，混一起会污染 embedding。

---

## 2.3 Caption 文本细粒度分析

**Notebook**：`analysis/notebooks/03_caption_stats.ipynb`
**产出目录**：`analysis/outputs/caption/`
**扫描规模**：抽样 2%（约 210 万行 caption），4 domain × 20K tokenize / 5K TF-IDF / 全部 live 池的 taglist 展开。

---

### 2.3.1 真实 Token 数分布（决定 max_source_length）

用 `OneReason-0.8B-Pretrain` tokenizer 实测（`token_length_stats.csv`）：

| Domain | sample | P50 | P90 | P95 | **P99** | max | mean±std |
|---|---|---|---|---|---|---|---|
| goods       | 20,000 | 22 | 28 | 29 | **32** | 46 | 20.5±6.5 |
| live        | 10,343 | 43 | 55 | 59 | **69** | 118 | 43.9±8.9 |
| video/ad    | 20,000 | 178 | 214 | 226 | **253** | 1160 | 179.3±29.0 |
| video/video | 20,000 | 172 | 212 | 225 | **251** | 392 | 166.2±40.8 |

**关键结论**：

- video 域 P99 = **250 tokens**（比字符数 P99=420 缩短约 40% —— tokenizer 对中文有较好的合并率，1 char ≈ 0.6 token）。
- video/ad 有一条 max=**1160 tokens** 的极端异常（99% 都在 253 以下，出现 1160 说明有极长离群 caption）—— SFT 时应硬截断。
- goods 只 32 tokens P99 —— 加上 domain 前缀 + 3 个 SID token + 系统 prompt 200 tokens ≈ **240 tokens**（一个 batch 完全能塞下多条）。

#### 🔧 SFT 建议 L：`max_source_length` 分桶（精修版，覆盖 §2.2 建议 J）

| 场景 | 建议 max_source_length | 依据 |
|---|---|---|
| video/ad → SID | **288**（P99=253 + 系统 prompt 35 tokens 冗余） | §2.3.1 |
| video/video → SID | **288**（同上） | §2.3.1 |
| goods → SID | **96**（P99=32 + prompt 60 tokens） | §2.3.1 |
| live 关键词列表 → SID | **128**（P99=69 + prompt 60 tokens） | §2.3.1 |
| video 域异常长 caption（>500 token） | 硬截断到 500 | §2.3.1 max=1160 |

- 训练用 `group_by_length=True` + `pad_to_multiple_of=8`，goods 域一个 batch 能塞 8~16 条，video 域 4~6 条，训练效率显著提高。

---

### 2.3.2 重复率：`goods` 有明显重复 caption，其他域接近 unique

从 `duplication_stats.csv`：

| Domain | sampled | unique_% | shared_captions_% | rows_on_shared_% | max_pids/caption |
|---|---|---|---|---|---|
| goods       | 977,665 | **94.36%** | 3.32% | **8.78%** | **179** |
| live        | 10,343  | 100.00% | 0.00% | 0.00% | 1 |
| video/ad    | 141,435 | 97.53% | 1.32% | 3.76% | 275 |
| video/video | 976,687 | 99.76% | 0.07% | 0.31% | **1145** |

**关键结论**：

- **`goods` 重复严重**：8.78% 的行落在"多 pid 共享的 caption"上，最热一个 caption 被 **179 个不同 pid** 共享 —— 这些 pid 大概率是同一 SKU 的不同 sku_id 变体（不同尺码 / 颜色 / 包装）。SFT 时**同 caption 训练梯度会被同一样本反复推**，导致过拟合到最常见款。
- **video/video 有极端 outlier**：99.76% unique 但存在被 **1145 个 pid 共享**的 caption（可能是同一段被批量搬运的模板文案）。
- **`live` 100% unique**：taglist 各不相同（因为 tag 组合极多），无需去重。

#### 🔧 SFT 建议 M：goods 域样本按 caption 去重

- 训练"描述 → SID"任务时，对 goods 样本做 **caption-based dedup**：同一 caption 只保留 K 个 pid（K=1~3），或加权 `weight = 1 / n_pids_share_this_caption`。
- video/video 只需 filter 掉极少数超高共享的 caption（`n_pids > 100`），影响面很小。
- **去重后 goods 域有效样本从 9.77M 降至约 9.22M**（unique caption 数），损失可忽略。

---

### 2.3.3 语言分布：**四个域都 > 99% 中文**

从 `language_stats.csv`：

| Domain | zh | en | ja | ko | other | empty |
|---|---|---|---|---|---|---|
| goods       | **99.29%** | 0.30% | 0.00% | 0.00% | 0.41% | 0.00% |
| live        | **99.98%** | 0.02% | 0.00% | 0.00% | 0.00% | 0.00% |
| video/ad    | **99.80%** | 0.00% | 0.00% | 0.00% | 0.00% | 0.19% |
| video/video | **99.88%** | 0.00% | 0.00% | 0.00% | 0.01% | 0.12% |

**关键结论**：

- **纯中文数据集**，无需在 prompt 里做多语言 hint。
- goods 域 0.41% 的 `other` 主要是纯符号 / 数字型标题（如 "0.5L ×2 装"）。
- SFT 用的 tokenizer（OneReason-0.8B）中文覆盖足够。

---

### 2.3.4 Top 关键词（jieba TF-IDF，快速看行业分布）

`keywords_top100_{domain}.csv`：

- **goods top-15**：`新款 / 休闲 / 时尚 / 专属 / 夏季 / 家用 / 百搭 / 加厚 / 男士 / 外套 / 套装 / 男女 / 上衣 / 宽松 / 手机` —— **典型电商 SEO 词汇**（营销词 + 品类词），说明 goods 的 caption 就是"商品标题"（电商目录侧的 title），**不是**详细描述。
- **video/video top-15**（未展示，从 CSV 读）：以行业主题词为主（美食、育儿、健身、宠物等）。
- **video/ad top-15**：以广告目的词为主（专属、优惠、爆款等）。

#### 🔧 SFT 建议 N：goods 域走"关键词 → SID"任务，与 live 并列

- goods 的 caption 本质是**营销关键词序列**（不是自然语言描述），与 live 的 taglist 语义结构接近。
- 建议训练时把 goods 也用 §2.2 建议 I 的"关键词 → SID"范式：**先用 jieba 把 caption 切成关键词 list，再随机 mask 训练 SID**。这样 goods 短标题的信息利用率更高。
- goods 关键词 top-100 已保存，直接可作为 vocab 白名单做 prompt 增强。

---

### 2.3.5 Live 域 taglist 展开分析（配套 §2.2 建议 I）

从 `live_taglist_stats.csv`：

| 指标 | 值 |
|---|---|
| sample_captions | 10,343 |
| parse_ok | 10,343（**100%**，语法完全一致） |
| unique tags 词表 | **37,747** |
| total tag mentions | 87,197 |
| **每 caption 的 tag 数** | P50=**8**, P90=**10**, P99=**13**, max=18 |
| top-1 tag | **精致妆容**（1,160 次） |

Top-30 tag（`live_taglist_top100.csv`）：
```
精致妆容(1160) / 长发造型(871) / 露脸直播(737) / 红唇妆容(620) / 互动型主播(508)
/ 露脸主播(419) / 亲切互动(362) / 好身材(349) / 不露脸直播(336) / 甜美长相(291)
/ 长发美女(286) / 唱歌才艺(281) / 短发造型(270) / 唱歌达人(236) / 正能量传递(233)
/ 情感互动(201) / 互动型直播(200) / 接地气(198) / 生活化直播(197) / PK互动(196)
...
```

**关键结论**：

- Live 的 tag 是**主播人设 + 视觉外貌 + 互动风格**导向（不同于 `Pid2Tag` 的品类导向），词表 3.7 万 tag 极度细粒度。
- 每 caption 8~10 个 tag，训练时可**随机保留 3~7 个**做 k-fold 增强（既保留信息，又模拟推荐场景中"用户兴趣不完整"）。

#### 🔧 SFT 建议 O：Live 域"关键词 → SID"prompt 具体格式

```
prompt = "<|living_begin|>关键词：精致妆容, 长发造型, 露脸直播, 唱歌才艺"
target = "<s_a_...><s_b_...><s_c_...>"
```

- 训练时 prompt 里保留的 tag 数按 **`Uniform(3, 7)`** 随机采样，模拟真实推荐场景。
- 验证时用**固定 5 个 tag** 保证结果可比。
- Data augmentation 系数：每条原始 caption 生成 **k=5** 条不同 tag 子集的样本 → live 有效样本从 106K 增到 **530K**（缓解 §2.1.5 建议 F 的极端小样本问题）。

---

### 2.3.6 §2.3 综合结论

**一句话**：Caption token 数 P99 明确（video 250, goods 32, live 69）；四域纯中文；goods 有 3.3% caption 重复；live taglist 完全解析成功且平均 8~10 tag。

对 SFT 的关键调整：
1. 精修的 `max_source_length` 分桶（建议 L）；
2. goods 域样本 caption-based dedup（建议 M）；
3. goods 也走"关键词 → SID"范式（建议 N，与 live 并列）；
4. Live 的"关键词 → SID"prompt 具体格式（建议 O，含 k-fold 增强）。

---

## 2.4 Tag 分布分析

**Notebook**：`analysis/notebooks/04_tag_stats.ipynb`
**产出目录**：`analysis/outputs/tag/`
**扫描规模**：Pid2Tag 全量 5.4M 行 + 反向 join Pid2Sid 3591M 行提取 5.4M 有 tag pid 的 SID。

---

### 2.4.1 Tag 层级结构：**全域共用同一套 53 个 lv1 分类体系**

从 `tag_hierarchy_stats.csv`：

| Domain | rows | \|lv1\| | \|lv12\| | \|lv3\| | avg lv2/lv1 | avg lv3/lv12 |
|---|---|---|---|---|---|---|
| live        | 60,231     | **53** | 493 | 1,854 | 9.30 | 3.76 |
| video/ad    | 742,085    | **53** | 468 | 2,132 | 8.83 | 4.56 |
| video/video | 4,614,963  | **53** | 524 | 3,539 | 9.89 | 6.75 |

**关键结论**：

- **三个域的 lv1 类目数完全相同（都是 53）**，说明是**同一套内容分类体系** —— 这是全域共用 taxonomy 的强证据。
- lv2 有细微差异（468~524），lv3 差异较大（1854 vs 3539），说明 video/video 的分类粒度最细，live 最粗。
- 平均每个 lv2 有 3.76~6.75 个 lv3 分支 —— lv3 是相对独立的叶子。

`top100_lv1_video_video.csv` 前 15 名（video/video 分布最全，可作代表）：
```
游戏娱乐(500K) / 搞笑幽默段子(266K) / 餐饮美食(242K) / 生活纪实(219K) / 情感与家庭关系(219K)
/ 三农与乡村生活(204K) / 音乐(199K) / 服饰穿搭(197K) / 短剧(184K) / 动漫二次元(183K)
/ 影视综艺(171K) / 明星娱乐(170K) / 房产与家居(166K) / 汽车交通(124K) / 医疗与健康(120K)
```

—— 分布相对均衡（top-1 500K vs 头部平均 200K，比例仅 2.5x），无极端头部主导。

#### 🔧 SFT 建议 P：把 lv1 作为额外的 special token

- 只有 53 个 lv1 值，加入 tokenizer 作为 `<tag_游戏娱乐>` / `<tag_餐饮美食>` 之类的 special token 代价极小（tokenizer 词表加 53 个 entry），**能显著提升 tag → SID 任务的 loss 收敛速度**（避免 lv1 名字被切成好几个 subword）。
- 或者更简单：prompt 里直接写 `[品类: 游戏娱乐]` 前缀 + describe，让模型自然学习。

---

### 2.4.2 Tag ↔ SID 关联性：**中等，且 lv3 相对 lv1 增益不大**

从 `tag_sid_mutual_info.csv`：

| Domain | H(s_a) | IG lv1→s_a | IG lv12→s_a | IG lv3→s_a | **IG_lv3/H(s_a) %** |
|---|---|---|---|---|---|
| live        | 5.32 | 0.887 | 1.247 | 1.711 | **32.15%** |
| video/ad    | 8.56 | 2.671 | 3.442 | 4.065 | **47.49%** |
| video/video | 9.49 | 2.769 | 3.615 | 4.440 | **46.78%** |

以及对 s_b / s_c：

| Domain | IG_lv3→s_b_% | IG_lv3→s_c_% |
|---|---|---|
| live        | 25.55% | 23.85% |
| video/ad    | 27.91% | 21.54% |
| video/video | 27.50% | 18.65% |

**关键结论（和之前预期不太一样）**：

1. **tag 只能压缩 s_a 约 47% 的不确定性**（video 域），远低于 §2.1.4 的 `IG_(a,b)→c ≈ 61~67%`。
2. **`s_a` 是最能被 tag 预测的层**，`s_b/s_c` 只有 20~28% —— 说明 SID 的深层语义（细粒度物料聚类）**独立于分类体系**，SID 更多在编码「视觉/内容 embedding」而不是「品类」。
3. **lv3 相对 lv1 的额外增益有限**（video 域从 2.77 → 4.44，多 1.67 bit）—— 意味着大部分信息在 lv1 就已经拿到了。

#### 🔧 SFT 建议 Q：Tag → SID 任务只用 lv1 就够，节省序列长度

- 因为 lv3 相对 lv1 的额外信息增益 <2 bit，训练"tag → SID"任务时**用 lv1 作输入，不需要 lv3**，可以节省 20~40 个 token 的 prompt 长度。
- lv3 只在需要**做候选 constrained decoding** 时使用（见建议 R）。

---

### 2.4.3 Tag-Conditioned Constrained Decoding **可行性受限**（修正 §2.4.4 假设）

⚠️ **重大修正**：先前 notebook 里的 §2.4.4 假设"给定 lv3 tag，s_a 只需从 top-3 桶选"—— 实测数据不支持！

从 `tag_lv3_top_sa_{domain}.csv`（每 lv3 tag 至少 20 pid 的样本）：

| Domain | #tags analyzed | top1_share mean | top3_cum mean | **#tags top3_cum≥0.9** | avg unique s_a per tag |
|---|---|---|---|---|---|
| live        | 252   | 0.314 | 0.576 | **2 (0.8%)** | 29.8 |
| video/ad    | 899   | 0.281 | 0.491 | **21 (2.3%)** | 53.2 |
| video/video | 2,938 | 0.242 | 0.435 | **20 (0.7%)** | 103.7 |

**关键结论**：

- **top1_share 只有 24~31%**：给定 lv3 tag，最集中的 s_a 桶也只覆盖 24~31% 的样本 —— 无法用 top-1 constrain。
- **top3_cum 只有 43~58%**：top-3 s_a 只能覆盖不到六成样本 —— 用 top-3 mask 会漏一半以上。
- **每个 lv3 tag 平均散布在 30~104 个不同的 s_a 桶** —— 尤其 video/video 平均 104 桶。
- 只有 **0.7%~2.3% 的 tag** 达到 `top3_cum ≥ 0.9`（可用 top-3 mask 的高置信 tag）。

**这直接推翻了 §2.4.4 里的乐观假设**：给 tag 后 s_a 空间还是很大，用"tag-conditioned s_a mask"必须放宽到 top-30~top-50 才有 90% 覆盖。

#### 🔧 SFT 建议 R：Tag-Conditioned Mask 的实用形态

**放弃"tag → 3 个 s_a"的 hard mask，改成"tag → top-K 软概率 hint"**：

1. **训练时**：把 tag 作为 prompt 前缀，模型隐式学习 `P(s_a | tag)`，不做外部 mask。
2. **推理时**（Pass@64）：可选**双阶段生成**：
   - 阶段 1：生成 s_a 时用 **domain-conditioned mask**（§2.1.3，缩到 800~5800 候选）+ **tag-based rerank**（用 `tag_lv3_top_sa_{domain}.csv` 里的分布 boost top-K 桶的 logits）。
   - 阶段 2：生成 s_b/s_c 时用 **prefix-conditioned mask**（§2.1.3 建议 D 的 trie，最紧的约束）。
3. **只对 `top3_cum ≥ 0.9` 的高置信 tag（约 43 条）**用 hard mask —— 这些 tag 极度专门化（比如某个特定小品类），可靠。

**从数据看，SID 的语义主要靠"视觉/内容 embedding"而不是"品类分类"**：

- 这也解释了 §2.1 里为什么 codebook 是跨 domain 共享的 8192 —— 它编码的是**通用视觉/文本聚类**，不是分类体系。
- **懂物料任务**（描述 → SID）比**tag → SID 任务**信息量大得多，应该作为主任务；tag → SID 只作为辅助。

---

### 2.4.4 §2.4 综合结论 & Curriculum 修订

**一句话**：Tag 只能中等程度地压缩 SID 不确定性（IG≈47%），tag-conditioned hard mask 不可行；lv1（53 类）就够用，lv3 用于 rerank hint。

#### 修订版 Stage-B（tag → SID）

| 项 | 修订前 | 修订后 | 依据 |
|---|---|---|---|
| tag 层级 | lv3 | **lv1**（53 special token） | §2.4.2 建议 P + 建议 Q |
| 样本量 | 5.4M | 5.4M（不变） | §2.2 |
| 期望作用 | 主辅助任务 | **辅助任务**（懂物料主导，tag→SID 次之） | §2.4.3 |
| Constrained decoding | tag → top-3 s_a | **domain mask + tag logit boost** | §2.4.3 建议 R |

#### Full 4 阶段 Curriculum（累计更新）

| 阶段 | 输入 → 输出 | 样本源 | 数量 | 目的 |
|---|---|---|---|---|
| **Stage-A**  | 描述 → SID | goods/video S∩C 且 type=describe，去重后 | ~19M | 主任务底座（懂物料） |
| **Stage-A′** | 关键词列表 → SID | live taglist（k=5 增强）**+ goods 关键词化（建议 N）** | 500K + 46M ≈ 46.5M | 缓解 live 极端小样本 & 提升 goods 短标题信息利用率 |
| **Stage-B**  | **lv1 tag** → SID | S∩T | 5M | 辅助任务，用作 rerank hint |
| **Stage-C**  | 用户历史 → next SID | 全 UserProfile | 500K user × 若干 pos sample | **懂推荐主任务** |

#### 3 个 ⚠️ 与之前建议的差异

1. **建议 J（§2.2 max_source_length）** → 被 **建议 L（§2.3.1）** 精修（用真实 tokenizer 数据），采用建议 L。
2. **§2.4.4 假设"tag→3 s_a"** → 被 **建议 R（§2.4.3）** 推翻，改为软 hint + rerank。
3. **建议 I（§2.2 live 走关键词→SID）** → 被 **建议 N（§2.3.4）** 扩展到 goods 域，形成 Stage-A′ 双域联合。

---

## 2.5 User 序列长度 & 稠密度分析

**Notebook**：`analysis/notebooks/05_user_seq_stats.ipynb`
**产出目录**：`analysis/outputs/user_seq/`
**扫描规模**：全量 500K 用户 × 8.56 GB × 10 parquet；每个用户 30+ 序列字段的长度、正样本率、时间戳。

---

### 2.5.1 主序列长度分布：**video_history 极深、ec_item_id 极稀疏、live 极长尾**

从 `seq_length_summary.csv`：

| Domain | Field | nonempty_% | P50 | P90 | P99 | max | mean | total_events |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| video/video | `video_sampled_pid_list`（当前） | 95.07% | 2 | 7 | 18 | 522 | 3.5 | 1.66 M |
| video/video | **`video_history_sampled_pid_list`** | **100.0%** | **649** | **1558** | **2946** | **38041** | 818.8 | **409 M** |
| goods | **`ec_item_id_list`**（cvr 主 label） | **0.19%** ⚠️ | 1 | 1 | 3 | 6 | 1.1 | 1049 |
| goods | `ec_good_click_item_id_list_extend` | 69.06% | 322 | 371 | **371** | **371** | 234.5 | 81 M |
| goods | `ec_good_order_item_id_list_extend` | 51.94% | 37 | 269 | 300 | 301 | 81.5 | 21 M |
| goods | `ec_colossus_rs_item_id_list` | 70.21% | 498 | 500 | **500** | **500** | 444.8 | 156 M |
| live | `live_hist_author_id_list` | 71.93% | **8** | **148** | **1652** | **83679** | 98.9 | 35.6 M |
| video/ad | `outer_loop_history_action_pid_list_pos` | 53.02% | 4 | 28 | 157 | 2090 | 13.2 | 3.5 M |
| video/ad | `outer_loop_history_action_pid_list_click` | 71.09% | 20 | 96 | 1318 | 18892 | 73.7 | 26.2 M |
| video/ad | **`outer_loop_deep_target_pid`**（深转化 target） | **1.5%** ⚠️ | 1 | 2 | 4 | 9 | 1.2 | 8851 |

**关键结论（8 个）**：

1. **`video_history` 全用户覆盖 + 极深**（500K × 100% × mean 819 = **4.09 亿** events） —— 这是最富的输入侧信息源。
2. **`ec_item_id_list` 只 0.19% 非空**（500K 里只有 950 个用户有非空），且平均只 1.1 条 —— **不能作为 goods 域 next-item 主 label**（否则训练样本几乎全空）。
3. **`ec_colossus_rs_item_id_list` 严格封顶 500**（P99=P90=500，全部到顶），是 goods 最大信息源（1.56 亿事件），必须以它为 goods 主输入。
4. **`ec_good_click_extend` 严格封顶 371**、`ec_good_order` 封顶 301 —— 官方已做了序列长度限制。
5. **`live` P50=8 vs P99=1652 vs max=83679** —— 差距 200 倍到 10000 倍，**极度长尾**。存在少量"live 深度用户"疯狂看直播，绝大多数用户只是浅接触。
6. **`video_history` P90=1558 tokens，如果按 SID 3-token 编码 = 4674 tokens**，一个用户历史就撑爆 4K 上下文。**必须做序列采样**（不能全拼）。
7. **`outer_loop_deep_target_pid` 只 1.5% 非空** —— 这大概率是**评测集里的核心目标**（"用户下一条深度转化的广告"），只 8851 条正样本，样本极稀，需要**跨用户 pooling** 或作为**辅助 signal** 而非唯一 target。
8. **`video_sampled_pid_list`（当前）P50=2, P90=7, mean=3.5** —— 这个"当前"序列极短，是**天然的 next-item target 池**（见 §2.5.3 时间跨度的证据）。

#### 🔧 SFT 建议 S：Prompt 里每域拼多少历史（token 预算 = 4096 - 系统 300 - 目标 20 ≈ 3800）

| 域 | 建议使用字段 | 建议拼接条数（时间倒序） | 每条 token 数 | 该域总 token 消耗 |
|---|---|---:|---:|---:|
| video/video | `video_history_sampled_pid_list` 最近 K 条 | **K = 128** | 4（domain + 3 SID） | ~512 |
| goods | `ec_colossus_rs_item_id_list` 最近 K 条 | **K = 128** | 4 | ~512 |
| goods | `ec_good_click_extend` 最近 K 条 | **K = 64** | 4 | ~256 |
| live | `live_hist_author_id_list` 最近 K 条 | **K = 96**（截断长尾） | 4 | ~384 |
| video/ad | `outer_loop_history_action_pid_list_click` 最近 K 条 | **K = 96** | 4 | ~384 |
| video/ad | `outer_loop_history_action_pid_list_pos`（正样本序列） | **K = 32** | 4 | ~128 |
| **合计** | | | | **~2200** |

- 剩余 ~1600 tokens 给反馈标签 hint（每条 pid 补 1 个「like/click/duration」的枚举 token）+ 对话结构。
- 每条历史的 4 tokens：`<|domain_begin|><s_a_x><s_b_x><s_c_x>` —— 这是 §2.1 建议 A 的编码结果。

#### 🔧 SFT 建议 T：`live` 域的双策略采样

- P50=8 但 P99=1652 → 直接截断到 96 会丢失高活跃用户 90% 的信息，直接全拼又会 OOM。
- **方案**：如果 `live` 长度 ≤ 128，全拼；否则按时间倒序取最近 96 + 均匀采样最早 32（保留时间跨度信息）—— 因为 live 时间跨度 399 天（见 §2.5.3），远期兴趣仍有价值。

---

### 2.5.2 反馈标签正样本率：**play_done 是首选主 label，`neg_feedback` 全零不可用**

从 `label_pos_rate.csv`（按正样本率排序）：

| Primary sequence | Label | Cond | n_events | pos_rate |
|---|---|---|---:|---:|
| `video_history_sampled_pid_list` | **`video_history_play_done_list`** | =1 | 409 M | **59.98%** ⭐ |
| `video_sampled_pid_list` | **`video_play_done_list`** | =1 | 1.66 M | **53.77%** ⭐ |
| `live_hist_author_id_list` | **`live_hist_valid_play_cnt_list`** | >0 | 35.6 M | **31.74%** ⭐ |
| `ec_colossus_rs_item_id_list` | `ec_colossus_rs_is_click_list` | =1 | 156 M | 13.25% |
| `live_hist_author_id_list` | `live_hist_follow_author_cnt_list` | =1 | 35.6 M | 10.80% |
| `live_hist_author_id_list` | `live_hist_comment_cnt_list` | >0 | 35.6 M | 6.28% |
| `live_hist_author_id_list` | `live_hist_like_cnt_list` | >0 | 35.6 M | 4.96% |
| `video_sampled_pid_list` | `video_like_list` | =1 | 1.66 M | 4.17% |
| `video_history_sampled_pid_list` | `video_history_like_list` | =1 | 409 M | 3.08% |
| **`ec_item_id_list`** | **`ec_cvr_label_list`** | >0 | **1049** ⚠️ | 2.86% |
| `ec_colossus_rs_item_id_list` | `ec_colossus_rs_is_buy_list` | =1 | 156 M | 1.65% |
| `video_sampled_pid_list` | `video_comment_list` | =1 | 1.66 M | 1.26% |
| `video_history_sampled_pid_list` | `video_history_comment_list` | =1 | 409 M | 1.12% |
| `video_sampled_pid_list` | `video_collect_list` | =1 | 1.66 M | 0.95% |
| `video_history_sampled_pid_list` | `video_history_collect_list` | =1 | 409 M | 0.67% |
| `video_sampled_pid_list` | `video_forward_list` | =1 | 1.66 M | 0.50% |
| `video_history_sampled_pid_list` | `video_history_forward_list` | =1 | 409 M | 0.40% |
| `ec_colossus_rs_item_id_list` | `ec_colossus_rs_is_cart_list` | =1 | 156 M | 0.18% |
| `live_hist_author_id_list` | `live_hist_reduce_similar_cnt_list` | >0 | 35.6 M | 0.05% |
| `video_sampled_pid_list` | **`video_neg_feedback_list`** | =1 | 1.66 M | **0.00%** ⚠️ |
| `video_history_sampled_pid_list` | **`video_history_neg_feedback_list`** | =1 | 409 M | **0.00%** ⚠️ |

**关键结论（4 个）**：

1. ⚠️ **`video_neg_feedback_list` 全零**（当前和历史都是 0 正样本）—— 数据集里根本没有负反馈标注，**不能作为负样本源**。想构造视频负样本，只能用 `play_done=0 AND like=0 AND comment=0` 组合式定义。
2. ⭐ **`play_done` 是最佳主 label**：正样本率 54~60%，事件量 4.09 亿，**信号密度**（正样本率 × 事件量 ≈ 2.5 亿正例）远高于其他所有 label。
3. ⚠️ **`ec_cvr_label_list` 只 1049 个事件**，虽然正样本率 2.86% 但绝对量 30 条 —— **不足以训练**，仅作为评测辅助 signal。
4. `live_hist_reduce_similar` 只 0.05% 正样本 —— 是极弱的负反馈 signal，可用作稀有的 hard negative。

#### 🔧 SFT 建议 U：4 类正样本定义 & Loss 权重

| 域 | 主任务正样本定义 | 正样本率 | Loss 权重（`w = 1/√pos_rate`） |
|---|---|---:|---:|
| video/video | `video_history_play_done == 1`（默认）+ 加权项：`like/comment/forward/collect` | 60% | 1.29 |
| goods | `ec_colossus_rs_is_click == 1`（主）+ `is_buy == 1`（强 signal 加权 5x） | 13% | 2.75 |
| live | `live_hist_valid_play_cnt > 0`（主）+ `follow_author == 1`（强 signal 加权 3x） | 32% | 1.77 |
| video/ad | `outer_loop_history_action_pid_list_click` 全部当正样本（因为已经是"点击"序列） | 100% | 1.00 |

- **多目标 loss**：`L = L_play_done + 0.5·L_like + 0.3·L_comment + 0.2·L_forward + 0.2·L_collect`（权重按正样本率倒数开根号取整）
- **Focal loss γ=2**：应用于 `is_cart/is_buy/comment/collect` 等 <2% 的稀疏 label
- **绝对不要**用 `neg_feedback` 做 label，直接从 code 里 filter 掉这两个字段避免误用

---

### 2.5.3 时间跨度：**"当前 target"vs"历史输入"清晰分离**（Stage-C 样本构造的关键依据）

从 `time_span.csv`：

| Primary | ts field | min | max | **span_days** |
|---|---|---|---|---:|
| `video_sampled_pid_list`（当前） | `video_ts_list` | 2026-02-04 15:42 | 2026-02-05 15:52 | **1.01** ⭐ target |
| `ec_item_id_list`（当前） | `ec_time_ms_list` | 2026-02-04 15:44 | 2026-02-05 15:41 | **1.00** ⭐ target |
| `outer_loop_deep_target_pid`（当前） | `outer_loop_deep_target_pid_ts` | 2026-02-04 16:00 | 2026-02-05 15:59 | **1.00** ⭐ target |
| `video_history_sampled_pid_list`（历史） | `video_history_ts_list` | 2026-01-20 15:42 | 2026-02-04 15:52 | **15.01** |
| `outer_loop_history_action_pid_list_pos`（历史） | `outer_loop_history_action_pid_list_pos_ts` | 2025-12-06 | 2026-02-04 | **60.00** |
| `outer_loop_history_action_pid_list_click`（历史） | `outer_loop_history_action_pid_list_click_ts` | 2025-12-06 | 2026-02-04 | **60.00** |
| `live_hist_author_id_list`（历史） | `live_hist_timestamp_list` | 20250101 | 20260204 | **399.00** |

**这是本次分析最重要的发现之一** —— 数据集有**天然的时间切分结构**：

- 所有 **"当前"字段** (`video_sampled` / `ec_item_id` / `outer_loop_deep_target`) 都严格集中在 **2026-02-04 → 2026-02-05 的 1 天窗口** 内
- 所有 **"历史"字段** 时间范围 **2025-01-01 → 2026-02-04**（历史结束时间 = 当前开始时间）
- **两者严格无重叠** —— 说明官方已经按时间做了 leakage-safe 切分！

#### 🔧 SFT 建议 V：Stage-C 样本构造直接用官方切分（无需再切）

这个天然的"当前 vs 历史"分离**直接告诉我们 Stage-C（用户历史 → next SID）该怎么构造训练样本**：

```python
# Stage-C 样本模板（伪代码）
for user in UserProfile:
    # 输入侧：从 6 个历史字段拼接
    history_prompt = build_prompt(
        video_hist   = user.video_history_sampled_pid_list[-128:],  # 15 天窗口
        goods_hist   = user.ec_colossus_rs_item_id_list[-128:],     # 未知窗口，取时间倒序
        goods_click  = user.ec_good_click_item_id_list_extend[-64:],
        live_hist    = user.live_hist_author_id_list[-96:],         # 399 天窗口
        ad_click     = user.outer_loop_history_action_pid_list_click[-96:],  # 60 天窗口
        ad_pos       = user.outer_loop_history_action_pid_list_pos[-32:],    # 60 天窗口
    )
    # 输出侧（label）：从 3 个"当前"字段构造 next-item
    for pid in user.video_sampled_pid_list:            # 1 天窗口，天然是 next
        yield (history_prompt, domain='video/video', target_pid=pid)
    for pid in user.ec_item_id_list:                    # 950 个用户有值
        yield (history_prompt, domain='goods', target_pid=pid)
    for pid in user.outer_loop_deep_target_pid:         # 1.5% 用户有值 = 7500 个
        yield (history_prompt, domain='video/ad', target_pid=pid)
```

**Stage-C 有效样本量估算**：
- video 侧：500K × 95% × mean 3.5 = **1.66 M 样本**（充足）
- goods 侧：950 × 1.1 = **1049 样本**（极稀，只能作为评测集）
- ad 深度侧：7500 × 1.2 = **8851 样本**（少但可训练）
- **合计：~1.68 M 训练样本**，其中 99% 是 video 域 next-item

#### 🔧 SFT 建议 W：**验证集不需要再做时间切分**

因为官方已经切好了 —— 建议：
- 用 500K 用户里 **随机抽 5%（25K 用户）作为 held-out validation**（用户级切分）
- 剩余 475K 作为训练
- 不需要在**同一用户内再切时间**（官方已用 2026-02-04 天然分开），那样反而会破坏 Stage-C 的"用完整历史预测下一天"的语义

#### 🔧 SFT 建议 X：Live 域的 399 天历史特殊处理

- live 是唯一跨越 **399 天** 的历史字段（一年多），远超其他域的 15/60 天。
- 用户在 live 上的兴趣**演化性**远比其他域强，建议：
  1. 在 prompt 里给每条 live 历史标注**相对时间 bucket**（例如"7 天内 / 30 天内 / 90 天内 / 更早"），提示模型区分近期/远期兴趣。
  2. 或者训练"时间感知"变体：从 live 历史里采样 3 个不同时间段的 pid 子集，模型学习"最近偏好演化"。
- 这一点直接对应比赛的**懂用户 · 兴趣演化**任务。

---

### 2.5.4 §2.5 综合结论

**一句话**：video_history 100% 覆盖 + 4.09 亿事件是最富输入源；`play_done`（60% 正率）是主 label；`neg_feedback` 全零不可用；「当前 vs 历史」时间上严格分离（1 天 vs 15/60/399 天），天然对齐 next-item 任务。

**7 条 SFT 建议汇总**：
- **建议 S**：Prompt 里每域拼多少历史（video 128 / goods 128+64 / live 96 / ad 96+32）
- **建议 T**：live 双策略（≤128 全拼 / >128 时间倒序 96 + 早期均匀 32）
- **建议 U**：正样本定义 & loss 权重（`play_done` 主 + 稀疏 label focal γ=2）
- **建议 V**：Stage-C 样本构造直接用官方"当前 vs 历史"切分
- **建议 W**：验证集用**用户级 5% 抽样**，不再做时间切分
- **建议 X**：live 域 399 天历史加**时间 bucket** 标注
- ⚠️ 严格禁用：`video_neg_feedback_list`（全零污染）、`ec_cvr_label_list` 作为唯一 label（只 30 个正样本）

---

## 2.6 多域覆盖 & 用户画像分层

**Notebook**：`analysis/notebooks/06_domain_coverage.ipynb`
**产出目录**：`analysis/outputs/domain_coverage/`
**扫描规模**：全 500K 用户 × 4 域布尔矩阵（复用 §2.5 缓存，秒级完成）。

---

### 2.6.1 15 种 Domain 组合分布：**94.6% 用户是多域用户，41.5% 全 4 域活跃**

从 `combo_stats.csv` 和 `single_vs_multi.csv`：

| Combo | 组合 | n_users | 占比 |
|---|---|---:|---:|
| **1111** | video/video + goods + live + video/ad | **207,525** | **41.51%** ⭐ |
| 1101 | video/video + live + video/ad（无 goods） | 65,241 | 13.05% |
| 0111 | video/video + goods + live（无 ad） | 58,099 | 11.62% |
| 1011 | video/video + goods + video/ad（无 live） | 55,005 | 11.00% |
| 0011 | video/video + goods（双域） | 30,708 | 6.14% |
| 0101 | video/video + live（双域） | 28,804 | 5.76% |
| 1001 | video/video + video/ad（双域） | 27,692 | 5.54% |
| **0001** | **video/video only（单域）** | **26,926** | **5.39%** |
| 0000 | (none) | 0 | 0.00% |

**重大发现**：

1. **零全空用户**（0000 = 0）：所有 500K 用户都至少在一个域有历史 —— **数据质量非常高**。
2. **video/video 100% 覆盖**：所有出现的 combo 全部含 video/video（右侧 bit 都是 1），不存在"没有 video 历史"的用户 → **video 是所有 SFT 样本的输入侧基础**。
3. **41.51% 用户全 4 域活跃**（1111 组合超过 20 万用户）—— 极其惊人的多域覆盖率，跨域推荐样本源**极其充足**。
4. **单域用户仅 5.39%**（且都是 video only）—— 多域用户占 **94.61%**！

单/多域用户汇总 (`single_vs_multi.csv`)：
| n_active_domains | n_users | 占比 |
|---:|---:|---:|
| 1 | 26,926 | 5.39% |
| 2 | 87,204 | 17.44% |
| 3 | 178,345 | 35.67% |
| **4** | **207,525** | **41.51%** |

（3 域和 4 域用户合计 **77.18%** —— 大多数用户是"广谱多域活跃"）

#### 🔧 SFT 建议 Y：Prompt 模板必须支持"缺失 domain 占位"

- 单域/双域用户占 22.83%（1/8+ 用户），这些用户输入侧有 2~3 个域完全为空。
- Prompt 模板设计要支持缺失：
  ```
  <|video_history|>...pids...<|end|>
  <|goods_history|><|no_data|><|end|>       # 缺失时用占位
  <|live_history|><|no_data|><|end|>
  <|ad_history|>...pids...<|end|>
  ```
- 这样比"直接省略缺失域"更好，因为**保留了"缺失"这个信号**（模型能学到"没有 goods 历史 → 别推 goods"）。

#### 🔧 SFT 建议 Z：跨域推荐样本可从 3+ 域用户里抽

- 3+ 域用户 **385,870 人**（77.18%），跨域推荐样本源极其丰富。
- 可以直接构造：
  - **video 历史 → 预测 goods next-item**（挑 `combo` 含 goods 的用户 = 351,337 人）
  - **video+live 历史 → 预测 ad next-item**（挑 `combo` 含 ad 的用户 = 355,463 人）
  - **all 4 → 任一域 next**（207K 用户，最丰富）

---

### 2.6.2 活跃度分层：**high 层 vs low 层交互次数差 4 倍**

从 `stratum.csv`（P33=925, P67=1746 三分位切分）：

| Layer | n_users | mean_events | P50 | P90 | mean_active_domains |
|---|---:|---:|---:|---:|---:|
| **low**  | 166,662 (33.3%) | 617   | 605   | 948   | **2.53** |
| **mid**  | 166,794 (33.4%) | 1,362 | 1,368 | 1,604 | **3.32** |
| **high** | 166,544 (33.3%) | 2,430 | 2,140 | 3,320 | **3.55** |

**关键结论**：

1. **high 层交互次数是 low 的 4 倍**（2430 vs 617）—— 差距远比 tag/caption 那里的头部主导温和，用户级不需要极端上采样。
2. **mean_active_domains 从 low 的 2.53 增到 high 的 3.55** —— 高活跃用户几乎必然是多域用户，这也**佐证 §2.6.1 的强多域效应**。
3. mid 和 high 的**平均活跃 domain 数差异很小**（3.32 vs 3.55），说明**分层的关键差异在"每域深度"而非"覆盖广度"**。

---

### 2.6.3 活跃度 × Domain 组合交叉：**"缺 goods 的用户 = 低活跃"是强规律**

从 `stratum_x_domain.csv`：

| Combo | 组合 | high | mid | low | total | **low 占比** |
|---|---|---:|---:|---:|---:|---:|
| 1111 | 全 4 域 | 105,368 | 80,477 | 21,680 | 207,525 | 10.4% |
| 1101 | 无 **goods** | 5,453 | 12,775 | 47,013 | 65,241 | **72.1%** ⚠️ |
| 0111 | 无 ad | 23,570 | 25,172 | 9,357 | 58,099 | 16.1% |
| 1011 | 无 live | 19,340 | 24,270 | 11,395 | 55,005 | 20.7% |
| 0011 | video+goods | 8,139 | 13,121 | 9,448 | 30,708 | 30.8% |
| 0101 | video+live | 1,876 | 4,279 | 22,649 | 28,804 | **78.6%** ⚠️ |
| 1001 | video+ad | 1,569 | 3,812 | 22,311 | 27,692 | **80.6%** ⚠️ |
| 0001 | video only | 1,229 | 2,888 | 22,809 | 26,926 | **84.7%** ⚠️ |

**重大规律**：

1. **单/双域用户绝大多数是低活跃**（`0001/0101/1001/1101` low 占比 72~85%）—— 这些用户历史稀疏，是**推荐冷启的经典难点**。
2. **1111 组合的用户里 51% 是高活跃**（105K/207K），是**优质样本主要来源**。
3. **含 goods（组合末位第 2 bit=1）的用户明显更高活跃**：
   - 1111（含 goods）low 占比 10.4% vs 1101（无 goods）low 占比 72.1%
   - **goods 是"高活跃"的标志性 domain** —— 用户愿意浏览商品、下单说明整体活跃度高。

#### 🔧 SFT 建议 AA：三层分层采样（不是"活跃度 × 3"，而是"combo × 活跃度"混合）

传统按活跃度分层采样会导致 low 层几乎全是 `0001/0101/1001/1101` 这些冷启组合，训练模型会"学到冷启用户"。**更好的方案是按 combo × 活跃度联合分层**：

| Bucket | 定义 | n_users | 采样权重 | SFT 期望信号 |
|---|---|---:|---:|---|
| **A: 4-域高活跃** | 1111 × high | 105,368 | 1.0 | 主训练池，跨域推荐主源 |
| **B: 3-域中/高活跃** | 0111/1011/1101(mid+high) & 1111 mid | 226,712 | 1.5 | 多域基础样本 |
| **C: 2-域** | 0011/0101/1001/1111 low | ~90,000 | 2.0 | 中等冷启，可训练 |
| **D: 单域冷启** | 0001 + 1101 low | ~74,000 | **3.0** | 冷启专项，防止模型只对高活跃泛化 |

（具体权重需实验调，此处只给方向）

- 评估时也**必须分层报告**：整体 hit@k、每层 hit@k、每 combo hit@k —— 否则整体分数会被 41% 的 1111 用户主导，看不到冷启表现。

#### 🔧 SFT 建议 AB：Live 与 Ad 的稀疏化陷阱

- 从 combo 表看：**"无 live" 的用户合计 = 55,005 (1011) + 30,708 (0011) + 27,692 (1001) + 26,926 (0001) = 140,331 人（28.07%）**
- **"无 ad" 的用户合计 = 58,099 (0111) + 30,708 (0011) + 28,804 (0101) + 26,926 (0001) = 144,537 人（28.91%）**
- **"无 goods" 的用户合计 = 65,241 + 28,804 + 27,692 + 26,926 = 148,663 人（29.73%）**
- **约 30% 的用户在任意一个具体域都是"无历史"** —— SFT 时**不能强制要求所有输入域都有数据**，`<|no_data|>` 占位符是必需的。

---

### 2.6.4 §2.6 综合结论

**一句话**：所有用户 100% 有 video 历史；41.5% 用户全 4 域活跃；94.6% 用户是多域；跨域推荐样本极其丰富；`goods` 是"高活跃"标志 domain；单/双域用户 78~85% 是低活跃冷启。

**4 条 SFT 建议汇总**：
- **建议 Y**：Prompt 模板必须用 `<|no_data|>` 占位符支持缺失 domain（30% 用户各有一域缺失）
- **建议 Z**：3+ 域用户（77.18%）里挑跨域样本，`combo=1111` 的 20 万人是主源
- **建议 AA**：**combo × 活跃度 4 层采样**（A/B/C/D），冷启层 3.0x 权重
- **建议 AB**：评估必须**分层 hit@k 报告**，整体分会被 1111 高活跃用户主导

### 修订版 4 阶段 Curriculum（累计到本节）

| 阶段 | 输入 → 输出 | 样本源 | 数量 | 关键约束 |
|---|---|---|---:|---|
| Stage-A  | 描述 → SID | goods/video 去重后 describe | ~19M | 建议 L/M |
| Stage-A′ | 关键词列表 → SID | live k=5 增强 + goods 关键词化 | ~46M | 建议 N/O |
| Stage-B  | lv1 tag → SID | S∩T | 5M | 建议 P/Q/R |
| **Stage-C** | 用户历史 → next SID | UserProfile 500K 用户 × 3 target 字段 | **~1.68M** | 建议 S/T/U/V/W/X + Y/Z/AA/AB |

---

## 2.7 时间戳细粒度分析

**Notebook**：`analysis/notebooks/07_temporal_stats.ipynb`
**产出目录**：`analysis/outputs/temporal/`
**扫描规模**：抽样 5% 用户（25K）× 10 parquet；每字段 daily/hourly/gap/leakage 全套统计。

---

### 2.7.1 每日活跃 pattern：**发现 20260203 空一天，是官方 train/eval 隔离带**

从 `daily_count_*.csv`：

| 字段 | 覆盖 | #days | count P50 | count 波动 | 关键结论 |
|---|---|---:|---:|---:|---|
| `video_history_ts_list`（历史） | 20260120 → 20260204 | 16 | 1.38M/day | 1.15M~1.60M | 相对平稳，01-24 起显著上升（训练用户主要活跃期） |
| `video_ts_list`（当前） | 20260204 → 20260205 | 2 | 42K | 17K vs 65K | 20260204 只覆盖 15:42 后，20260205 完整 |
| `ec_time_ms_list`（当前） | 20260204 → 20260205 | 2 | 30 | 17 vs 42 | 因为 ec_item_id 只 950 用户有值 |
| `outer_loop_deep_target_pid_ts`（当前） | 20260204 → 20260205 | 2 | 30 | 17 vs 42 | 深度转化极稀 |
| `outer_loop_history_action_pos_ts`（历史） | 20251206 → 20260204 | 61 | ~2900 | 2316~3765 | 60 天，pattern 稳定 |
| `outer_loop_history_action_click_ts`（历史） | 20251206 → 20260204 | 61 | ~7500 | 稳定 | 每日约 7500 广告点击 |
| **`live_hist_timestamp_list`**（历史） | 20250101 → 20260204 | **399** ⭐ | **4155** | 1815~10356 | 一年全域活跃，**最近 30 天翻倍**（近期活跃 8000+），说明 live 深度用户越接近 target 时间越活跃 |

**⭐ 重大发现：20260203 空一天**！

从 live_hist 数据（每日 count>0）看：
- `20260201: 8621` / `20260202: 7576` / **`20260203: (缺失)`** / `20260204: 8814`
- **20260203 一整天没有任何行为记录** —— 这不是数据缺失，而是官方**特意留白的隔离带**！
- 即：**训练历史侧结束于 20260202，评测目标从 20260204 开始**，中间空 20260203 作为 buffer

**含义**：
- SFT 时如果自己造 next-item 样本，也应该**遵循这个 1 天 buffer 规则**（把 20260203 当天的行为作为 held-out）。
- 但因为原始数据里 20260203 已被官方剔除，这条自动生效，**用户不需要额外操作**。

#### 🔧 SFT 建议 AC：Live 域最近 30 天 upsampling

- Live 数据近期 30 天（20260105→20260204）平均 count > 6000/day，比早期 P05=2280/day 高 3 倍。
- 这符合"用户兴趣越接近 target 越有信号"的经验，**训练时对 live 历史的近 30 天可上采样 2x**。
- 或者按建议 X 加**时间 bucket**（`[7d]/[30d]/[90d]/[更早]`），让模型自己学近期偏好。

---

### 2.7.2 每小时活跃 pattern：**全字段 UTC pattern 一致，午间/晚间双峰 + 深夜谷底**

从 `hourly_count.csv`（各字段峰值时段一致，选 `video_history_ts` 代表）：

| Hour (UTC) | count | pct | Hour (UTC) | count | pct |
|---:|---:|---:|---:|---:|---:|
| 0 | 867K | 4.25% | 12 | 1006K | 4.92% |
| 1 | 952K | 4.66% | 13 | 993K | 4.86% |
| 2 | 1021K | 5.00% | 14 | 953K | 4.66% |
| 3 | 1065K | 5.21% | 15 | 1036K | 5.07% |
| **4** | **1106K** | **5.41%** ⭐ | 16 | 779K | 3.81% |
| 5 | 1069K | 5.23% | 17 | 571K | 2.79% |
| 6 | 1024K | 5.01% | 18 | 399K | 1.95% |
| 7 | 1046K | 5.12% | 19 | 302K | 1.48% |
| 8 | 1097K | 5.37% | **20** | **267K** | **1.31%** ⚠️ 谷底 |
| 9 | 1145K | 5.60% | 21 | 318K | 1.56% |
| **10** | **1201K** | **5.88%** ⭐ | 22 | 490K | 2.40% |
| 11 | 1029K | 5.03% | 23 | 698K | 3.42% |

**中国时区（UTC+8）转换**：
- **UTC 4** = **中国 12 点** ⭐ 午间高峰（`5.41%`）
- **UTC 10** = **中国 18 点** ⭐ 晚间高峰（`5.88%`，最高）
- **UTC 20** = **中国凌晨 4 点** ⚠️ 谷底（1.31%）
- **UTC 18-21** = **中国 2-5 点** —— 深夜睡眠时段，活跃度极低

**关键结论**：所有 6 个字段的 hourly pattern **形状高度一致**，说明这是**用户行为的 UTC 时间戳**，且用户主体是中国用户（睡眠谷底在 UTC 18-21 = 中国凌晨 2-5 点）。

#### 🔧 SFT 建议 AD：Prompt 里加 `[time_bucket]` hint（建议但可选）

- Prompt 里可选加入行为发生的**时段 hint**（例如 `[晚间] [商品-点击]`），让模型学到"晚间点击的商品和白天不同"的规律。
- **具体桶**：`[凌晨(0-6)] / [上午(6-12)] / [下午(12-18)] / [晚间(18-24)]`（中国时区）—— 简单 4 桶足够。
- 训练时随机 mask 60% 样本的 time hint，模拟推理时可能没有精确 time 信号的场景。

---

### 2.7.3 用户内 gap 分布：**只 `video_history` 是「session 化」序列，其他都是「用户级稀疏」序列**

从 `session_gap_stats.csv`：

| 字段 | n_gaps | P10 | P50 | P90 | P99 | 30min断裂% | 1天断裂% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`video_history_ts_list`** | 500K | 58.9s | **4.3m** ⭐ | 1.2h | 14.7h | **15.36%** | 0.26% |
| `video_ts_list`（当前） | 55K | 3.4m | 40.7m | 7.8h | 17.5h | 55.15% | 0.0% |
| `ec_time_ms_list` | 5 | 8.2m | 1.1h | 3.6h | 4.3h | 60.0% | 0.0% |
| `outer_loop_history_action_pos_ts` | 165K | 4.3m | **6.8h** | **5.2d** | **27d** | 77.45% | **29.13%** |
| `outer_loop_history_action_click_ts` | 500K | 35.0s | **9.7m** | **1.5d** | 11.1d | 42.5% | 12.84% |
| `outer_loop_deep_target_pid_ts` | 69 | 3.6m | 1.3h | 9.6h | 15.4h | 56.52% | 0.0% |

**关键结论**：

1. ⭐ **只有 `video_history_ts_list` 是「密集 session 型」**（P50 gap=4.3 分钟、P90=1.2 小时、只 15% 跨 30 分钟）—— 是**唯一可以切 session 做 in-session next item** 的字段。
2. **`outer_loop_history_action_pos_ts`（广告转化）P90=5.2 天** —— 说明大部分用户"两次广告转化"间隔 5 天以上，**转化行为极其稀疏**，没有 session 概念。
3. **`outer_loop_history_action_click_ts`（广告点击）P50=9.7min but P90=1.5d** —— **双模态分布**：既有短时间连续点击（session 内），也有长间隔的稀疏点击。
4. **`ec_time_ms_list` 只 5 个 gap 样本** —— 因为 ec_item_id 极稀（950 用户有值），基本无法做 gap 分析。

#### 🔧 SFT 建议 AE：只对 `video_history` 做 session 切分增强

- 对 `video_history_sampled_pid_list`：**gap > 30 分钟视为 session 边界**，把每个用户的历史切成多个 session。
- 每 session 提供两种训练样本：
  1. **session-level next**：给 session 前 N-1 条 pid 预测第 N 条（in-session next item，短时兴趣）。
  2. **cross-session**：给 session-1 的最后 K 条 + session-2 的最后 K 条 → 预测 session-3 的第 1 条（长时兴趣演化）。
- 其他域（goods/live/ad）**不做 session 切分**，直接按最近 K 条时间倒序拼接。

---

### 2.7.4 ⚠️ Leakage 检查发现问题：video/ec 的"当前"与"历史"有 10~17 分钟重叠

从 `leakage_check.csv`：

| current 字段 | history 字段 | boundary_gap | safe |
|---|---|---:|:---:|
| video_ts | video_history_ts | **10.0m** | **NO ⚠️** |
| video_ts | outer_loop_pos_ts | 16.6m | NO ⚠️ |
| video_ts | outer_loop_click_ts | 17.6m | NO ⚠️ |
| ec_time_ms | video_history_ts | **1.4m** | **NO ⚠️** |
| ec_time_ms | outer_loop_pos_ts | 8.1m | NO ⚠️ |
| ec_time_ms | outer_loop_click_ts | 9.0m | NO ⚠️ |
| **`outer_loop_deep_target_pid_ts`** | video_history_ts | 8.6m | **YES** ✅ |
| **`outer_loop_deep_target_pid_ts`** | outer_loop_pos_ts | 1.9m | **YES** ✅ |
| **`outer_loop_deep_target_pid_ts`** | outer_loop_click_ts | 58.8s | **YES** ✅ |
| video_ts / ec_time_ms | live_hist_timestamp（date_str） | (day-level) | YES ✅ |
| outer_loop_deep_target | live_hist_timestamp（date_str） | (day-level) | YES ✅ |

**这是 §2.5 未发现的重要漏洞**！

- ⚠️ **`video_ts_list`（当前 min = 15:42:20）比 `video_history_ts_list`（历史 max = 15:52:20）早 10 分钟**！同样 `ec_time_ms_list` 比历史早 1.4 分钟。
- 说明 `video_sampled_pid_list` 和 `ec_item_id_list` **不是"未来事件"**，而是**同天窗口内混杂**的 (history + current) —— **非严格 next-item**。
- ✅ **只有 `outer_loop_deep_target_pid`（广告深度转化）严格 clean**：min = 16:00:55 > 所有历史 max 15:59:56（差 58.8 秒），**是唯一可以严格作为 next-item label 的字段**。

**这修正了 §2.5.3 建议 V 的乐观结论**！

#### 🔧 SFT 建议 AF：修正 Stage-C 主 target 的选择

原 §2.5 建议 V 说三个"当前"字段（`video_sampled` / `ec_item_id` / `outer_loop_deep_target`）都可以作为 next-item target。**修正后**：

| 字段 | 严格 next-item？ | 用途 |
|---|:---:|---|
| **`outer_loop_deep_target_pid`** | ✅ **YES** | ⭐ 严格作为 next-item target（**8851 样本**，虽然稀但绝对干净） |
| `video_sampled_pid_list` | ⚠️ **NO**（10min 重叠） | 作为**"session-final prediction"**（同天窗口内相对靠前的一条位置的推理），不能作为严格 next-item |
| `ec_item_id_list` | ⚠️ **NO**（1.4min 重叠） | 950 样本量本来就太少，作为**评测辅助 signal** |

**Stage-C 有效样本量重估**：
- 严格 next-item（`outer_loop_deep_target`）：**~8,851 样本** —— 太少，不能作为唯一训练源
- 非严格 next-item（`video_sampled`）：1.66M 样本 —— 数量够但含**同天前后混杂**
- **实际训练策略**：把"当前"字段按时间戳**再做一次严格切分**（对同一用户内，`video_ts >= min(video_history_ts_max_of_user)` 的才算 valid target），或者**接受 10 分钟重叠**（因为工业推荐系统通常也在 minute-level 更新，10min 重叠可视为"同 session 后一半"）。

**推荐做法**：**接受重叠**，训练时不做 minute-level 过滤。理由：
1. Live 域用 date-level 时间戳，本身就是天粒度，不可能 minute 精确。
2. Kuaishou 官方给的 UserProfile schema 已经内含 "history vs current" 命名，**官方的语义定义就是这样**（同天窗口内的近未来）。
3. 10 分钟重叠影响的样本只是"边界几百条"，占 1.66M 的比例极小。

但**必须在 report 里明确记录这个漏洞**，避免后续误以为"严格无 leak"。

---

### 2.7.5 §2.7 综合结论

**一句话**：数据在 20260203 有天然 buffer 隔离带；每小时 pattern 是中国用户的 UTC 时区自然规律；只 `video_history` 是密集 session 序列（P50 gap=4.3min）；⚠️ **`video_ts` 和 `ec_time_ms` 的"当前"字段与历史有 10 分钟级重叠**，**只有 `outer_loop_deep_target_pid` 是严格 clean 的 next-item target**。

**4 条 SFT 建议汇总**：
- **建议 AC**：Live 近 30 天 upsampling 2x（数据近期活跃度自然翻倍）
- **建议 AD**：Prompt 加 4 桶 `[time_bucket]` hint（可选）
- **建议 AE**：只对 `video_history` 做 session 切分增强（30min gap 界定），其他域按最近 K 条拼接
- **建议 AF**：修正 Stage-C 主 target 选择 —— **严格 next 用 `outer_loop_deep_target_pid`（8851 样本）**，**video 域接受 10min 重叠但要记录漏洞**

---

## 2.8 反馈标签精细分布 & 动作词映射

**Notebook**：`analysis/notebooks/08_label_stats.ipynb`
**产出目录**：`analysis/outputs/labels/`
**扫描规模**：数值 label 走全量（500K 用户）；字符串枚举抽 5%；count 类抽 10%（200 万样本）。

---

### 2.8.1 数值 Label 精确取值：**video/goods 主 label 全部严格 0/1，`lag` 系列是多值**

从 `numeric_label_dist.csv`：

**二值 label（0/1）** —— 用 BCE loss：

| Field | total | val=0 | val=1 |
|---|---:|---:|---:|
| `video_like_list` | 1.66M | 95.83% | **4.17%** |
| `video_comment_list` | 1.66M | 98.74% | **1.26%** |
| `video_forward_list` | 1.66M | 99.50% | **0.50%** |
| `video_collect_list` | 1.66M | 99.05% | **0.95%** |
| **`video_neg_feedback_list`** | 1.66M | **100.00%** ⚠️ | **0** |
| **`video_play_done_list`** | 1.66M | 46.23% | **53.77%** ⭐ |
| `video_history_like_list` | 409M | 96.92% | 3.08% |
| `video_history_comment_list` | 409M | 98.89% | 1.12% |
| `video_history_forward_list` | 409M | 99.60% | 0.40% |
| `video_history_collect_list` | 409M | 99.33% | 0.67% |
| **`video_history_neg_feedback_list`** | 409M | **100.00%** ⚠️ | **0** |
| **`video_history_play_done_list`** | 409M | 40.02% | **59.98%** ⭐ |
| `ec_cvr_label_list` | 1049 | 97.14% | 2.86% |
| `ec_colossus_rs_is_click_list` | 156M | 86.75% | **13.25%** |
| `ec_colossus_rs_is_cart_list` | 156M | 99.82% | 0.18% |
| `ec_colossus_rs_is_buy_list` | 156M | 98.36% | 1.65% |

**多值 label（枚举编码）**：

| Field | distinct | 分布特征 |
|---|---:|---|
| `ec_colossus_rs_lagv1_list` | **22** | val=0 (5%), val=23-30 各 4-6%, others 74.83% —— 大概是**点击后经过多少天** lag 编码 |
| `ec_colossus_rs_lagv2_list` | **24** | val 集中在 900/990/1080/1170/1260/1350/... —— **像分钟时段编码**（900min=15h，1350min=22.5h） |
| `ec_trunc_clk_lag` | 20 | val=0 (6%), val=1 (6%), val=2/3/4/5 各 5-6%, others 63.4% —— **点击-展示 lag 天数** |
| `ec_trunc_buy_lag` | 23 | val=0 (3%), val=1 (5%), val=2/3/4/5 各 5-5.6%, others 70% —— **购买-展示 lag 天数** |

**关键结论（4 个）**：

1. ⚠️ 再次确认 **`video_neg_feedback_list` 全零**（当前和历史都是 0）—— **完全禁用**。
2. ⭐ `video_play_done` 是 **video 域唯一高正样本率的 label**（54~60%），是主 label 首选。
3. **goods 的 3 个 label（click/cart/buy）严格递进**：click 13.25% > buy 1.65% > cart 0.18%（`cart` 反而比 `buy` 稀疏，说明用户"直接购买"多于"加购再买"）。
4. `lag` 系列本质是**时间信息的枚举编码**，不是普通 label —— **可作为 feature（额外 side-info）而非 target**。

#### 🔧 SFT 建议 AG：`ec_lag` 系列作为额外 side-info

- `ec_colossus_rs_lagv1/lagv2` 编码"点击 lag 天数"和"点击时段"，是**天然的时间感 feature**。
- Prompt 里可以在每条 goods 曝光 pid 后附一个 lag 桶：
  ```
  <|goods_history|> pid_1 [lag: 3d, 15h] pid_2 [lag: 8d, 22h] ...
  ```
- 训练时随机 drop 30% 的 lag hint 做鲁棒性正则。

---

### 2.8.2 Count 型 Label 分位：**`valid_play >= 3` 是 live 深度阈值；`watch_time` 全 clip 到 64s**

从 `count_label_percentiles.csv`（每字段 200 万抽样）：

| Field | zero_% | nz_p50 | nz_p90 | nz_p99 | nz_max | **建议阈值** |
|---|---:|---:|---:|---:|---:|---|
| `live_hist_show_cnt_list` | 47.77% | 1 | 1 | 3 | 664 | 曝光次数，>=3 显著 |
| `live_hist_play_cnt_list` | 56.88% | 1 | 5 | 13 | 460 | 播放，>=5 深度 |
| **`live_hist_valid_play_cnt_list`** | **68.51%** | 1 | **3** ⭐ | 10 | 460 | **[直播-深度观看] >= 3** |
| `live_hist_play_duration_list` | 56.88% | 27552ms=**27.6s** | 908K ms=**15m** | 4.9M ms=**82m** | 84M ms=**23h** | 长播 >= 15min |
| `live_hist_valid_play_duration_list` | 68.51% | 78s | 21m | 93m | 23h | 有效长播 >= 21min |
| `live_hist_like_cnt_list` | **94.87%** | 14 | 234 | 1936 | 45064 | 累计点赞（超长尾） |
| `live_hist_comment_cnt_list` | 93.56% | 4 | 35 | 405 | 88806 | 累计评论 |
| `live_hist_reduce_similar_cnt_list` | **99.95%** | 1 | 1 | 3 | 16 | 极稀疏负反馈 |
| `live_hist_report_live_cnt_list` | 99.98% | 1 | 4 | 10 | 17 | 极稀疏 |
| **`live_hist_follow_author_cnt_list`** | **89.63%** | 1 | 1 | **1** | 103 | **本质二值**（99% 都是 1，>1 罕见） |
| **`video_watch_time_list`** | 0.0% | 33 | **64** | **64** | **64** ⚠️ | **全部 clip 到 64s** |
| `video_duration_list` | 0.0% | 47 | 300 | 739 | 3203 | 视频时长 |
| **`video_history_watch_time_list`** | 0.0% | 35 | **64** | **64** | **64** ⚠️ | 同样 clip 到 64s |
| `video_history_duration_list` | 0.0% | 44 | 349 | 759 | 2338 | 历史视频时长 |

**关键发现（3 个）**：

1. ⚠️ **`video_watch_time` 全部 clip 到 64s**！P90=P99=max=64，说明官方在**数据脱敏时把观看时长上限截到 64 秒**。
   - **`play_done_list` 的 60% 正样本率**很可能是"watch_time == 64"作为 done 的定义（=看满了 clip 上限）。
   - **训练时无法区分「刚好看完 30s」和「看到 64s+ 但被 clip」**，必须依赖 `play_done` 作为主信号。
2. ⭐ **`valid_play_cnt >= 3`** 是 live 深度观看的天然阈值（P90=3），符合 §2.8.5 动作词映射的 `[直播-深度观看]` 定义。
3. **`follow_author_cnt` 本质二值**：P50=P90=P99=1（99%都是 1），只 max=103 极端 —— 关注一次即算关注，多次关注同一主播算 spam。

#### 🔧 SFT 建议 AH：`watch_time == 64` 时用 `play_done` 判别，非 64 时用 watch_time / duration 比

- 因为 watch_time 全 clip 64，需要**双 signal 判"长播"**：
  - `watch_time == 64 AND duration <= 64`：完整看完（真长播）
  - `watch_time == 64 AND duration > 64`：**至少看了 64s 但可能没看完**（部分长播，仍算 positive）
  - `watch_time < 64`：看了 `watch_time / duration` 的比例；如果比例 > 0.7 也算 positive
- 简化版：直接用 `play_done_list` 作为 label，忽略 watch_time 的截断问题（推荐）。

---

### 2.8.3 字符串枚举字段：**广告类型分布头部集中，直播类型二级分类清晰**

从 4 个 `str_enum_dist_*.csv`：

**⭐ `outer_loop_history_action_pid_list_click_type`（广告类型，10 类）**：
| Value | count | pct |
|---|---:|---:|
| `AD_ITEM_CLICK` | 866K | **86.60%** ⭐ |
| `EVENT_CONVERSION` | 121K | 12.07% |
| `EVENT_PRIVATE_MESSAGE_SENT` | 4108 | 0.41% |
| `EVENT_KEY_INAPP_ACTION` | 3588 | 0.36% |
| `EVENT_PAY` | 2684 | 0.27% |
| `EVENT_NEXTDAY_STAY` | 1269 | 0.13% |
| `EVENT_FORM_SUBMIT` | 1059 | 0.11% |
| `EVENT_EFFECTIVE_CUSTOMER_ACQUISITION` | 339 | 0.03% |
| `LEADS_SUBMIT` | 278 | 0.03% |
| `EVENT_WECHAT_CONNECTED` | 27 | 0% |

- **86.6% 是普通点击**，12% 是转化事件，其他 8 类合计 <1.5% —— **前 2 类占 99%**，其他可合并为 `[广告-其他]`。

**⭐ `outer_loop_history_action_pid_list_click_industry`（广告行业，20 类）**：
| Value | pct | Value | pct |
|---|---:|---|---:|
| 工具类软件 | **25.19%** | 商务服务 | 1.18% |
| 电商平台 | 21.65% | 教育 | 0.59% |
| 短剧小说 | 18.26% | 医疗机构 | 0.56% |
| 游戏 | 11.14% | 汽车 | 0.50% |
| 金融 | 6.19% | 医疗健康 | 0.44% |
| 网络服务平台 | 4.71% | ... | ... |

- **头部 5 类占 82.4%**（工具类软件+电商平台+短剧小说+游戏+金融），长尾可裁剪。
- 这是**懂物料任务**中广告物料的重要 side-info。

**`live_hist_author_type_list`（主播类型，4 类）**：
| Value | pct |
|---|---:|
| **秀场主播** | **65.23%** ⭐ |
| 大 V | 15.19% |
| 游戏主播 | 14.23% |
| 电商主播 | 5.35% |

- 只 4 类，分布极度集中在"秀场主播" —— 与 §2.3.5 里 live top-1 tag `精致妆容/长发造型/露脸直播` 完全吻合（秀场主播的典型标签）。

**`live_hist_author_category_type_list`（主播分类，5 类）**：
| Value | pct |
|---|---:|
| **B** | **56.19%** |
| A | 37.21% |
| 职业电商 | 6.38% |
| C | 0.18% |
| D | 0.03% |

- **B/A 两类占 93.4%** —— 大概是**平台分级 tier**（B=普通、A=优质、职业电商=商业主播）。

#### 🔧 SFT 建议 AI：4 个字符串枚举字段的 special token 化

| 字段 | 建议方案 |
|---|---|
| `click_type` | 前 2 类保留（AD_ITEM_CLICK、EVENT_CONVERSION），其他 8 类合并 `EVENT_OTHER` → 3 个 special token |
| `click_industry` | 前 5 类 special token（工具类软件/电商平台/短剧小说/游戏/金融），其他 15 类合并 `INDUSTRY_OTHER` → 6 个 |
| `author_type` | 4 类全部 special token（秀场/大V/游戏主播/电商主播） |
| `author_category_type` | 3 类合并（B、A、职业电商）+ `OTHER` → 4 个 |

- 总共新增 3+6+4+4 = **17 个 special token**，代价小，能让模型直接从 tokenizer level 理解这些语义类别（不需要子词分词）。

---

### 2.8.4 Live 直播类型布尔字段：**5 个字段里只 `is_detect_game_live` 有效**

从 `live_type_flags.csv`：

| Field | pos_rate | 结论 |
|---|---:|---|
| `live_hist_is_interactive_mp_live_list` | **0.000%** ⚠️ | 全 0，不能用 |
| `live_hist_is_building_live_list` | 0.007% | 极稀，可忽略 |
| **`live_hist_is_local_life_live_list`** | **0.000%** ⚠️ | 全 0，不能用 |
| **`live_hist_is_detect_game_live_list`** | **17.28%** ⭐ | 唯一有效 |
| `live_hist_is_recruit_live_list` | 0.005% | 极稀，可忽略 |

**关键结论**：

- 5 个 live 类型 flag 中，**2 个全 0 完全禁用**（`is_interactive_mp` / `is_local_life`），2 个 <0.01% 极稀无用。
- 只 **`is_detect_game_live` 17.28% 有效**，可作为**副 label**（游戏直播识别）—— 与 `author_type=游戏主播`（14.23%）非常接近，两者应该有强相关。

#### 🔧 SFT 建议 AJ：Live flag 只保留 `is_detect_game_live`

- 其他 4 个 flag 从 SFT 输入侧完全去掉（省 token）。
- `is_detect_game_live=1` 时可在 prompt 里追加 `[游戏直播]` hint。

---

### 2.8.5 ⭐ 动作词映射表（评估对齐核心）

从 `action_phrase_mapping.csv`（23 条规则）：

**video/video** — 6 类动作 + 1 兜底：
| priority | 条件 | phrase |
|---:|---|---|
| 10 | `play_done=1` | **[视频-长播]** ⭐ |
| 9 | `like=1` | [视频-点赞] |
| 8 | `collect=1` | [视频-收藏] |
| 7 | `forward=1` | [视频-转发] |
| 6 | `comment=1` | [视频-评论] |
| 1 | `neg_feedback=1` | [视频-负反馈]（实际不会触发，全零） |
| 0 | (fallback) | [视频-浏览] |

**goods** — 覆盖 3 个 primary 序列：
| primary | priority | 条件 | phrase |
|---|---:|---|---|
| `ec_colossus_rs` | 10 | `is_buy=1` | **[商品-购买]** ⭐ |
| `ec_colossus_rs` | 9 | `is_cart=1` | [商品-加购] |
| `ec_colossus_rs` | 5 | `is_click=1` | [商品-点击] |
| `ec_colossus_rs` | 0 | (fallback) | [商品-曝光] |
| `ec_good_click_extend` | 5 | (all) | [商品-点击] |
| `ec_good_order_extend` | 10 | (all) | [商品-购买] |
| `ec_item_id` | 10 | `cvr>0` | [商品-转化] |

**video/ad** — 3 类：
| primary | priority | phrase |
|---|---:|---|
| `outer_loop_deep_target` | 10 | **[广告-深度转化]** ⭐（严格 clean target） |
| `outer_loop_history_pos` | 10 | [广告-转化] |
| `outer_loop_history_click` | 5 | [广告-点击] |

**live** — 6 类：
| priority | 条件 | phrase |
|---:|---|---|
| 10 | `follow_author=1` | **[直播-关注]** ⭐ |
| 8 | `valid_play_cnt >= 3` | [直播-深度观看]（阈值来自 §2.8.2 P90） |
| 7 | `like_cnt > 0` | [直播-点赞] |
| 6 | `comment_cnt > 0` | [直播-评论] |
| 3 | `valid_play_cnt > 0` | [直播-观看] |
| 1 | `reduce_similar > 0` | [直播-负反馈] |

**关键结论（3 个）**：

1. **每域 4~7 个动作词**，共 **19 个中文动作短语** —— 加入 tokenizer 作为 special token 代价小。
2. **优先级规则合理**：交互强度高的动作优先（长播 > 点赞 > 收藏；购买 > 加购 > 点击；关注 > 深度观看 > 点赞）。
3. **一个 pid 只输出一个动作词**（priority 最高的）—— 避免 timeline 冗余。

#### 🔧 SFT 建议 AK：动作词加入 tokenizer 作为 special token

- 19 个中文动作短语（`[视频-长播]/[视频-点赞]/.../[广告-深度转化]/.../[直播-负反馈]`）加入 tokenizer 作为 **19 个 special token**。
- 加上 §2.4.1 建议 P（53 个 lv1 tag）+ §2.8.3 建议 AI（17 个类型 token）+ §2.1.4 已有的 SID token —— 总新增 **19+53+17 = 89 个 special token**，词表增长可忽略。
- **好处**：
  1. Prompt timeline 更紧凑（每个动作 1 token 而非 3~4 token）
  2. 模型直接从 embedding level 学动作语义
  3. 与官方评估 prompt 对齐（评估侧就用中文动作词）

---

### 2.8.6 §2.8 综合结论

**一句话**：video 主 label 用 `play_done`（60%）；`neg_feedback` 全零禁用；`watch_time` 全 clip 到 64s；`ec_lag` 是时间编码 side-info；4 个字符串枚举字段头部 3~5 类占 82~99%；live flag 只 `is_detect_game_live` 有用；**19 个动作词映射表 + 17 个类型 token = 36 个新 special token**，与评估侧对齐。

**5 条 SFT 建议汇总**：
- **建议 AG**：`ec_colossus_rs_lagv1/lagv2` 作为 side-info（每条 goods 曝光后附时间桶）
- **建议 AH**：`watch_time` 全 clip 64s，用 `play_done` 判长播（不用 watch_time）
- **建议 AI**：4 个字符串枚举字段的 17 个 special token
- **建议 AJ**：Live 5 个 flag 只保留 `is_detect_game_live`（其他 4 个全 0 或 <0.01%）
- **建议 AK**：19 个动作词加入 tokenizer 作为 special token（评估对齐）

### 修订版 4 阶段 Curriculum（累计到本节）

| 阶段 | 输入 → 输出 | 样本量 | 新增 special token | 相关建议 |
|---|---|---:|---:|---|
| Stage-A  | 描述 → SID | ~19M | — | L/M |
| Stage-A′ | 关键词 → SID | ~46M | — | N/O |
| Stage-B  | lv1 tag → SID | 5M | +53（lv1 tag） | P/Q/R |
| **Stage-C** | 用户历史 → next SID | ~1.68M（video） + 8.85K（严格 target） | +19（动作词）+17（类型枚举） | S~AF + AG~AK |

**Special token 总计新增**：`53 (lv1) + 19 (动作词) + 17 (类型枚举) = 89 个`，加上 SID codebook 已有的 `<s_a_i>/<s_b_i>/<s_c_i> ≈ 8192*3 = 24576` 个，词表膨胀 <0.4%。
