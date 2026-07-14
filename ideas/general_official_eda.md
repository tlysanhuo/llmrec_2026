# O4/O5 官方直发数据首轮 EDA

> **已由全量封板 EDA 取代（2026-07-12）**：最终精确重复、格式和 O5 严格候选漏斗见 [`OFFICIAL_DATA_EDA.md`](../docs/reference/OFFICIAL_DATA_EDA.md) §6。本页保留首轮固定样本方法，不再作为训练决策依据。
>
> 日期：2026-07-11 UTC
> 范围：`O4 OpenOneRec-General-Pretrain`、`O5 OpenOneRec-General-SFT`
> 资产身份和路径仍以 [`docs/reference/ASSETS.md`](../docs/reference/ASSETS.md) 为唯一权威；本文不是第二份资产清单。

## 结论

1. **O4 不适合直接当竞赛 SFT 使用。**它约 98.1% 为英文 prompt，按 source 标签映射后 62.75% 是通用推理、30.18% 是数学推理；约 99.95% 的抽样回答呈现 CoT 特征。总长度中位数 1,590 tokens，但约 19.90% 超过 4,096 tokens。
2. **O5 也不能整块或仅按 source 名小剂量混入。**它的中文 prompt 约 9.60%，中文/英文混合约 0.65%；启发式中文选择题只约 0.22%，而且进一步复核显示约 92% 的候选来自医学题。
3. **O5 的行配比不等于训练剂量。**抽样总长度中位数 2,408 tokens，约 38.44% 超过 4,096、19.85% 超过 8,192。后续若使用 O5，配比必须同时按行数和 token 数登记。
4. **统一读取入口必须是 `messages`。**O4/O5 的 `messages` 和 `source` 都是 100% 非空；O4 的 `text` 有 12.48% 空值，O5 的 `text` 至少 59.77% 空值，不能把 `text` 当统一正文列。
5. **本轮没有生成训练数据。**没有创建 D 类数据、训练配置或 checkpoint，也没有改写 O4/O5 官方直发文件。

## 方法

可复现脚本：[`scripts/data/eda_general_official.py`](../scripts/data/eda_general_official.py)。正式首轮命令：

```bash
python scripts/data/eda_general_official.py \
  --assets O4 O5 \
  --sample-files 32 \
  --samples-per-source 400 \
  --seed 20260711 \
  --output /tmp/llmrec_o4_o5_eda_20260711.json
```

- 精确统计：读取全部 Parquet footer，并完整扫描轻量 `source` 列。
- 内容统计：按 `SHA256(seed, asset_id, filename)` 固定选择 32 个分片；在每个 source 内做 reservoir sampling；总体比例再按精确 source 行数加权。
- 长度：使用 O6 本地 `Qwen2Tokenizer`，不是字符数近似。
- 语言：对 user 文本做 Unicode script 启发式分类，不是语言识别模型。
- 选择题：user 文本至少出现 3 个不同的 A-H 选项标记。
- likely CoT：回答含显式 thought 标签，或回答不少于 400 字且出现推理提示词。
- 正式 JSON SHA256：`b0b48d85ee0aa543fdc508afd5bbccff8f70821662772ee96f5639d796e88901`。JSON 放在 `/tmp`，不是持久资产。

正式内容样本覆盖：

| 资产 | 固定分片 | 扫描行数 | reservoir 样本 | 分片清单哈希 |
|---|---:|---:|---:|---|
| O4 | 32 / 310 | 277,007 | 2,934 | `b7d56995...b7da5d` |
| O5 | 32 / 301 | 276,022 | 3,600 | `4acc7ae8...dd47d` |

O4 的最小来源 `Bespoke-Stratos-17k` 在选中分片中只有 134 行，因此 O4 reservoir 总数少于 `8 × 400`。

## 物理格式与完整性

| 项 | O4 | O5 |
|---|---:|---:|
| Parquet | 310 | 301 |
| bytes | 27,139,522,149 | 24,685,081,929 |
| footer 精确行数 | 2,655,181 | 2,555,706 |
| row groups | 310 | 301 |
| `messages` null | 0 | 0 |
| `source` null | 0 | 0 |
| `text` null | 331,270（12.48%） | 1,527,638（59.77%）+ 2,852 行 footer 未知 |
| 逻辑 schema 变体 | 1 | 3 |

两套数据共有 12 个顶层字段：`uuid/metadata/images/videos/source/messages/segments/image/video/text/label/line_id`。`messages` 是序列化 JSON，内部通常为 `system/user/assistant` 角色和字符串或 text block 内容。

O5 的 3 个逻辑 schema 变体来自极少数分片把全空 `segments` 或 `text` 推断成 Arrow `null` 类型。按列名逐文件流式读取没有问题；若以后用 dataset API 一次性拼接，需显式 schema promotion/cast。

## Source 与任务族

下表的行数精确；“任务族”是根据 source 标签作的 EDA 映射，不是官方新增标签。

| 任务族 | O4 行数 / 占比 | O5 行数 / 占比 |
|---|---:|---:|
| 通用推理 | 1,666,229 / 62.75% | 502,818 / 19.67% |
| 数学推理 | 801,449 / 30.18% | 603,049 / 23.60% |
| 代码推理 | 150,503 / 5.67% | 601,676 / 23.54% |
| 通用指令 | 0 | 446,773 / 17.48% |
| 中文通用推理 | 30,000 / 1.13% | 179,037 / 7.01% |
| 多学科推理 | 0 | 172,108 / 6.73% |
| 医学推理 | 7,000 / 0.26% | 50,245 / 1.97% |

主要 source：

- O4：`reasoning_v1_20m` 1,666,229（62.75%）、`OpenMathReasoning` 477,179（17.97%）、`NuminaMath-QwQ-CoT-5M` 324,270（12.21%）。
- O5：`OpenMathReasoning` 510,163（19.96%）、`R1-Distill-SFT` 502,818（19.67%）、`Infinity_Instruct` 446,773（17.48%）、`OpenCoderReasoning` 437,768（17.13%）。

## 内容统计

以下都是固定样本、按精确 source 占比加权后的估计，不应写成全量精确计数。

| 指标 | O4 | O5 |
|---|---:|---:|
| JSON 可解析 | 100% | 100% |
| 英文 prompt | 98.06% | 89.04% |
| 中文 prompt | 1.12% | 9.60% |
| 中英混合 prompt | 0.06% | 0.65% |
| 显式 thought 标签 | 86.39% | 82.52% |
| likely CoT | 99.95% | 86.76% |
| thought 标签未闭合 | 0% | 0.128% |
| 选择题 | 1.14% | 4.87% |
| 中文选择题 | 0.0165% | 0.221% |
| 选择题回答尾部可识别字母 | 41.67% | 28.22% |

O4 中 `NuminaMath-QwQ-CoT-5M` 和中文推理数据通常没有 `<think>`，但回答仍是长推理过程，因此“显式 thought 标签”不能代替 CoT 判断。

O5 的少量未闭合 thought 标签按总体比例外推约为数千行量级；任何正式筛选都必须逐行验证标签，而不能只按 source 名抽取。

## Token 长度

| 总 tokens | O4 | O5 |
|---|---:|---:|
| mean | 3,171 | 4,341 |
| p50 | 1,590 | 2,408 |
| p90 | 8,745 | 11,523 |
| p95 | 12,669 | 14,316 |
| p99 | 16,405 | 16,648 |
| sample max | 27,616 | 20,719 |
| `>4096` | 19.90% | 38.44% |
| `>8192` | 10.85% | 19.85% |
| `>16384` | 1.06% | 1.45% |
| `>32768` | 0% | 0% |

“样本中没有超过 32,768”不等于全量不存在超长行。正式构造前仍应对最终候选做全量 token gate。

## O5 中文选择题复核

为了验证 400 条/source reservoir 对低频中文选择题的估计，又对同一批 32 个 O5 固定分片中的四个候选来源逐行解析，共复核 94,925 行。复核 JSON SHA256：`3bf1ee49afc0287633dd33b8487349afe5e16a751dff7736c39edac40fa46111`。

| source | 复核行数 | 中文选择题 | 来源内比例 | 按该 source 全量外推 |
|---|---:|---:|---:|---:|
| `Chinese-Reasoning-Distil-Data-think` | 21,429 | 19 | 0.0887% | 约 159 |
| `Infinity_Instruct` | 48,708 | 25 | 0.0513% | 约 229 |
| `Reasoning_Multi_subject_RLVR` | 19,477 | 0 | 0% | 0 |
| `medical-o1-reasoning-SFT-think` | 5,311 | 495 | 9.3203% | 约 4,683 |

四个来源合计外推约 5,071 条，其中约 92% 是医学题；这与首轮全 source reservoir 的约 5,653 条量级一致。两者都只是启发式外推，不是候选数据集行数。

人工看样还发现：

- `Chinese-Reasoning-Distil-Data-think` 中有排序题和多选/多答案题，不能仅凭 A-D 标记当作单选。
- `Infinity_Instruct` 有小学数学、法律等较贴近通识的中文单选，但数量很少。
- 医学来源量最大但学科偏置强，且很多回答没有以单个选项字母收尾。
- 因此真正满足“中文、通识、单选、唯一答案、格式兼容、长度受控”的可用池必然小于上述约 5.1k–5.7k。

## 对活跃 Idea 的影响

### I-03：O5 小剂量保持懂世界/通用推理

结论是**保留研究资格，但不准直接混训**。下一步若继续，只先做全量只读候选计数与 QC：

1. 中文或中英混合 prompt；
2. 明确的单选题，优先恰好四个 A-D 选项；
3. 可验证的唯一答案，排除多选、排序和开放问答；
4. thought 标签成对、最终答案格式可统一；
5. 设 token 上限并报告过滤前后 token 总量；
6. 按学科统计，限制医学占比；
7. 与 O1 主体的混合比例同时按行数和 token 数预登记。

只有上述 QC 证明存在足够规模、足够多学科的候选池，才值得创建 `D(O5)` 数据；届时必须登记上游 O5、构建脚本、行数、哈希和 mix ratio。

### I-04：O4 表征补强

首轮 EDA 后降级。O4 是英文长推理/数学占主导的 pretrain 语料，不应转写成竞赛 SFT。后续若研究，应明确是表征预训练问题，并与当前单卡、1 epoch SFT 主线分离；当前不生成配置。

### 当前优先级

I-01 官方种子内部配比仍是第一主线。O5 精筛是研究准备，不构成已批准训练实验；O4 暂不进入训练队列。
