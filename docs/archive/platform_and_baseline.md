> **⚠️已合并(2026-07-04)**:平台侧信息的唯一权威文档是 `docs/platform_guide.md`,本文件保留作历史存档,不再更新。

# 万擎平台、赛事规则与官方 Baseline 配方（初赛）

> 来源：万擎平台介绍文档（更新 2026-07-01 10:36）+ HF 官方数据/baseline 仓库 `OpenOneRec/Explorer_LLM_Rec_Competition`。
> 本文档回答了数据剖析报告 §4 的绝大多数开放问题，并给出官方钦定的训练框架/模板/超参。

---

## 0. 我们的工作方式（用户决策）

**自备机器（本容器 8×H100）做训练与迭代，不依赖万擎平台训练；最终把模型 CKPT 上传平台评测。** 平台评测每日限 3 次，评测参数由平台固定、不可改。

含义：
- 我们走「路径二 / 自主训练」——本地/开发机训练 → 上传 CKPT → 提交竞赛评测。
- 评测是黑盒（OneRec Benchmark），我们**无法控制推理/采样参数**（temperature、n=64 等由平台设定）。因此「调采样多样性提 Pass@64」这类推理侧手段大多失效——**分数几乎全靠训练数据与训练本身**。
- 上传要求（FAQ Q8）：LoRA 传 `adapter_model.safetensors`+`adapter_config.json`；全量传 `model.safetensors`（分片再加 `model.safetensors.index.json`）。线下训练**建议 Transformers v5.3.0**（注意：与官方 baseline 用的 4.x 不同，见风险）。

---

## 1. 硬规则（FAQ 提炼）

- **只能基于 `OneReason-0.8B-pretrain-competition` 迭代**（Q2）。评测时**严格校验模型 config 与 baseline 一致**——不可改模型结构、预定义参数、评测参数（Q6）。改 config = 评测不通过。
- **不能改模型结构/超参定义/评测设置**；可自定义 SFT 数据配比、可自建 RL pipeline（Q6）。Baseline **不提供 RL 代码**，RL 需自己实现（Q3）。
- **数据高度自由**（Q7）：可引入外部通识数据、构造定制化推荐数据。**但复现阶段须提交构造好的数据集 + 训练脚本**，官方续训复现，效果在误差内即可 → 一切数据构造与训练必须**可复现、脚本化、留种子**。
- **允许蒸馏**；**不鼓励模型融合**，复赛结束须提交**单模型**方案过代码审核（Q10）。
- 平台每周提供限量训练资源（3B+），也可自备算力（Q5、Q9）。

---

## 2. 评测：四维度 + 每日 3 次

平台竞赛评测一键调用 **OneRec Benchmark**，产出**懂物料 / 懂用户 / 懂推荐 / 懂世界**四维度得分 + 总分。
- **每日限 3 次**；评测失败不消耗次数，可复制重试。
- 有官方排行榜。
- ⚠️ **懂世界维度会被评测**，即使训练数据里没有直接的懂世界赛道文件 → 通识能力要么靠基座保留、要么用 `OneReason_General` 数据训（见下）。

---

## 3. 官方数据仓库（HF: OpenOneRec/Explorer_LLM_Rec_Competition）

两层数据：**(A) 已加工好的 SFT 数据**（= 我们本地 `data/dataset.tar.gz` 的 3 赛道 jsonl，平台【数据管理】也可下载）；**(B) 原始素材**，用于自行构造更多/更好数据。

### (B) 原始素材（`data/` 目录，parquet）
| 目录 | 内容 | 规模 |
|---|---|---|
| `OneReason_UserProfile/` | 每行一个匿名用户的**多域原始行为序列**（电商/短视频/直播/广告） | ~50 万行 |
| `OneReason_Pid2Sid/` | `pid`→三段语义 ID `sid_three=[s_a,s_b,s_c]`（按 `(domain,pid)` join） | 映射表 |
| `OneReason_Pid2Caption/` | `pid`→文本描述 caption | 映射表 |
| `OneReason_Pid2Tag/` | `pid`→三级类目 `tag_lv3`（video/ad/live） | 映射表 |
| `OneReason_General/` | **通识知识数据**（懂世界的原料），parquet 分片 ~40+ 片 | 大 |

**域取值**：`video/video`→`<|video_begin|>`；`video/ad`→`<|ad_begin|>`；`goods`→`<|prod_begin|>`；`live`→`<|living_begin|>`。所有 pid/id 为 hash 后的 int64，用户表无 uid（匿名）。
**序列对齐规则**：每个域有「主序列（pid 序列）」+ 等长「对齐特征序列」（点赞/评论/转发/收藏/负反馈/观看时长/完播/时间戳/转化标签…）。主序列与其对齐特征全 null = 该域无行为。字段全表见 HF README（已存 `docs/hf_userprofile_schema.md` 摘要）。

→ **这解决了剖析报告的最大开放问题**：R1/懂世界不是缺数据，而是给了**原始素材让我们自造**。我们可用 Pid2Sid/Caption/Tag + UserProfile 共现关系，自行构造 Item2Item（R1/派生）、更多懂推荐/懂用户样本、以及用 `OneReason_General` 覆盖懂世界。

---

## 4. 官方 Baseline 训练配方（`demo/`）——我们要对齐的 ground truth

**框架：LLaMA-Factory `0.9.6.dev0`，全量 SFT，bf16，packing+neat_packing，FlashAttention-2，Liger Kernel。** 单 H800 起步。

### 关键流程
1. `00_install.sh`：clone LLaMA-Factory；**Python 3.11 venv**；pin **torch 2.7.1+cu126**、**flash-attn 2.7.4.post1**、**liger-kernel 0.8.0**、sympy 1.13.3；打一个 transformers flash_attention `s_aux None` guard 补丁。
2. `01_convert_data.sh` → `convertv2.py`：parquet→Alpaca JSONL。
3. `02_register_dataset.py`：把 `data_final` 注册进 `dataset_info.json`，`formatting: alpaca`，列映射 `prompt→instruction, query→input, response→output, history→history`。
4. `03_train.sh`：`llamafactory-cli train demo/config/demo.yaml`。

### `demo.yaml` 核心超参（官方默认）
```
finetuning_type: full          # 全量
template: qwen3_nothink         # ★ 关键：qwen3 模板的 nothink 变体
cutoff_len: 32768               # 长上下文
packing: true / neat_packing: true
enable_liger_kernel: true / flash_attn: fa2
per_device_train_batch_size: 1 / gradient_accumulation_steps: 4
learning_rate: 2.0e-5 / num_train_epochs: 1
lr_scheduler_type: cosine / warmup_ratio: 0.03 / weight_decay: 0.0
bf16: true / pure_bf16: true / seed: 19260817
```

### `convertv2.py` 揭示的官方数据处理逻辑（务必对齐）
- **输入格式二选一**：ChatML `messages`（parquet，多轮 role/content）→ 取 system=instruction、第一条 user=input、最后一条 assistant=output、中间成对进 history。或者已加工的 `[{system,prompt,response}]` jsonl（用 `convert_jsonl.py`）。
- **`--max_token_types 3`**：统计 response 里 `<s_X_>` 的字母种类，**超过 3 种（即不止 a/b/c）就丢弃**该样本。保证每个 item 恰好 a/b/c 三段。
- **filter_sid**：删除 `<|sid_end|>`/`<|*_end|>` 等尾 token；把 `<|live_begin|>`→`<|living_begin|>`、`<prod_s_`→`<s_` 等**规范化**。→ 印证「sid 是幽灵、_end token 不进训练」。
- **think 注入逻辑（`add_think_pattern`，默认开）**——这是 think 控制的**官方标准**：
  - assistant 里**无 `<think>`** → 给对应 user 追加 **`/no_think`**，并把 assistant 改写为 `"<think>\n\n</think>\n" + 原文`。
  - assistant `<think>` **非空** → 给 user 追加 **`/think`**（保留）。
  - assistant `<think>` **空** → 给 user 追加 **`/no_think`**（空 tag 保留）。
  → 即 **`/think`·`/no_think` 后缀是训练期自动生成的**，与 `qwen3_nothink` 模板配合。

### `qwen3_nothink` 模板（LLaMA-Factory 内置，已确认存在）
- LLaMA-Factory `template.py` 有 `name="qwen3"` 与 `name="qwen3_nothink"`。`_nothink` 变体在 `enable_thinking is False` 时**移除 CoT 且不对 think 段算 loss**。
- 与 OneReason tokenizer 的 `enable_thinking` 门控一致（chat template 里 think=151667/close=151668）。

---

## 5. 对建模策略的修正（相对数据剖析报告）

1. **推理侧调参基本无效**（评测黑盒、参数固定）→ 火力集中在**训练数据构造 + SFT/RL 训练**。Pass@64 的多样性得靠训练出的分布，而非解码温度。
2. **懂世界不是没法做**：用 `OneReason_General` 通识数据混训，且勿因过度 SFT 推荐任务而灾难性遗忘通识（保留基座通用能力是 R3/懂世界的显式目标）。
3. **think 控制照官方来**：无 think→`/no_think`+空 think 块；有 think→`/think`。我们自造 non-think 推荐数据时，直接套 `convertv2.py` 的 inject 规则即可，无需另发明。
4. **数据必须脚本化可复现**（复现审核硬要求）：所有构造脚本进 `scripts/`，固定 seed。
5. **框架路线待定**：官方 baseline = LLaMA-Factory + torch 2.7/py3.11。我们本地已装 torch 2.3/py3.10 + transformers 4.53 的 venv。二选一，见 [[python-env]] 的风险说明——是照搬官方 LLaMA-Factory 栈，还是自写 HF Trainer 循环。

---

## 6. 仍待确认的开放问题（缩小后）
- **Pass@64 精确协议**：平台是否严格 32 think + 32 non-think 合并？（baseline 数据两模式都有，稳妥起见两模式都覆盖。）
- **四维度在总分里的权重**？懂世界维度的具体题型与计分（不定项选择 Accuracy）用哪批数据评？
- **懂物料 token→desc、懂用户 logic 字段**是否计分、如何计分。
- 是否有官方 dev/test 划分，还是自行留出。
- 上传时 **Transformers 版本**：平台建议 v5.3.0，但 baseline demo 用 4.x；评测环境到底用哪个（影响 chat template 行为与兼容性）。
