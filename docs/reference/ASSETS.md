# 资产注册表（唯一权威入口）

> 状态：ACTIVE / AUTHORITATIVE
> 基线日期：2026-07-11 UTC
> 维护人规则：任何新增、删除、移动或来源定性变化，必须先更新本表。其他文档不得另建资产总表。

本文只回答三个问题：资产是谁提供的、唯一位置在哪里、能否直接用于训练。以后处理数据或模型时先查本表，**默认禁止为了确认资产而递归扫描全盘**。

## 1. 强制分类

| 类别 | 定义 | 训练规则 |
|---|---|---|
| `O` 官方直发 | 竞赛平台或官方 HF 仓库直接提供，内容未经我方改写 | 只读；允许作为基线或构造原料 |
| `D` 官方源派生 | 我方从 `O` 清洗、转换、重采样、补 CoT 或混合得到 | 必须写明上游、脚本、行数；不得简称“官方数据” |
| `T` 第三方 | 非赛事官方直接提供的数据或选手数据 | 默认禁用；用户明确批准后才能训练 |
| `E` 评测衍生 | 平台可见题、评测日志、离线题和 holdout | 只用于诊断；不得默认回灌训练 |
| `M` 模型与提交产物 | checkpoint、adapter、merged model、提交包 | 由 `EXPERIMENT_INDEX.md` 管理，不属于数据资产 |

**用词铁律：**只有 `O` 可以称“官方直发/官方原始”。`D` 必须称“官方源派生”。原料全部来自官方，不等于成品仍是官方直发。

## 2. 官方直发资产 `O`（隔离区）

固定访问入口：`assets/official/`。该目录是指向运行卷原件的稳定链接视图，不复制大文件，不改变现有训练路径。

### O1. 平台种子 SFT

- 固定入口：`assets/official/seed_sft/`
- 原件目录：`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/official/seed_sft/`
- 原始压缩包：`dataset.tar.gz`，73,005,815 bytes
- SHA256：`13a7220febc949587ed15ade52562b0d9a04056351b645615ba749a3a360827b`
- 原始格式：每行一个长度为 1 的 messages list

| 任务 | 官方文件 | 行数 |
|---|---|---:|
| 懂物料 | `懂物料part1.jsonl` ... `懂物料part7.jsonl` | 10,384 |
| 懂推荐 | `懂推荐1.jsonl` ... `懂推荐4.jsonl` | 19,204 |
| 懂用户 | `懂用户.jsonl` | 2,892 |
| 合计 | 12 个 JSONL | **32,480** |

官方种子中没有独立“懂世界”文件。

### O2. 官方 HF 原始五表

- 固定入口：`assets/official/hf_raw/`
- 原件目录：`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/official/hf_raw/`
- 来源：HF `OpenOneRec/Explorer_LLM_Rec_Competition`
- 总体积：约 17G

| 子集 | 行数 | shards | 内容 |
|---|---:|---:|---|
| `OneReason_UserProfile` | 500,000 | 10 | 四域匿名用户行为序列 |
| `OneReason_Pid2Caption` | 21,061,327 | 136 | `(domain,pid)` 到 caption |
| `OneReason_Pid2Sid` | 35,914,095 | 198 | `(domain,pid)` 到三段 SID |
| `OneReason_Pid2Tag` | 5,417,279 | 31 | `(domain,pid)` 到三级标签 |
| `OneReason_General` | 152,005 | 158 | 通用 QA/CoT 原料 |

这五表是构造原料，不是可直接等价替换平台种子的 SFT。此前 `official_rec_v3_lora_ep1` 已证明直接按 next-item 方式重构会严重负迁移。

### O3. 官方 SFT 对齐 Caption/Tag

- 固定入口：`assets/official/sft_aligned/`
- 原件目录：`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/official/sft_aligned/`
- 文件：`baseline_caption_tag_lists.parquet`
- 规模：19,204 行，729,700,717 bytes
- SHA256：`c307fe6d723ebdebc2d343de3481bdc878f6193d65456e3b933ae7f6b78b8d9d`
- 字段：`record_id/messages/sid_token_list/caption_list/tag_list`
- 来源：2026-07-09 官方新增发布，与种子全部推荐样本一一对齐

### O4. 官方 General Pretrain

- 固定入口：`assets/official/general_pretrain/`
- 原件目录：`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/official/general_pretrain/`
- 来源：HF `OpenOneRec/OpenOneRec-General-Pretrain`
- 官方 revision：`ed57951e14595112eb18d47b850776e9407b8ff9`
- 规模：310 个 Parquet，27,139,522,149 bytes（约 25.28GiB）
- 用途：OneRec 通用预训练语料；不能与竞赛 17GB 五表或 `OneReason_General` 混为一套数据

### O5. 官方 General SFT

- 固定入口：`assets/official/general_sft/`
- 原件目录：`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/data/official/general_sft/`
- 来源：HF `OpenOneRec/OpenOneRec-General-SFT`
- 官方 revision：`4b8e43913aeb8e6c66b9253df4ab64ecc77dfd6c`
- 规模：301 个 Parquet，24,685,081,929 bytes（约 22.99GiB），约 2,555,706 条
- 内容：OpenOneRec 官方清洗并统一格式的通用 SFT 数据。虽然其上游来自 9 个开源集合，但该发布包本身属于 OpenOneRec 官方资产

### O6. 指定基座模型

- 固定入口：`assets/official/base_model/`
- 原件目录：`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/models/OneReason-0.8B-pretrain-competition/`
- 身份：竞赛指定 `OneReason-0.8B` pretrain checkpoint
- `config.json` SHA256：`5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4`

### 官方下载来源

```bash
# O2 + O3: Explorer 原始五表与 SFT/Caption-Tag
hf download OpenOneRec/Explorer_LLM_Rec_Competition --repo-type dataset

# O4: General Pretrain
hf download OpenOneRec/OpenOneRec-General-Pretrain --repo-type dataset

# O5: General SFT
hf download OpenOneRec/OpenOneRec-General-SFT --repo-type dataset
```

O3 官方页面：`https://huggingface.co/datasets/OpenOneRec/Explorer_LLM_Rec_Competition/tree/main/SFT`。

## 3. 官方源派生资产 `D`

固定访问入口：`assets/derived/processed/`，原件为运行卷 `data/derived/processed/`。该目录内文件来源不同，**不能把整个目录统称为官方数据**。

当前关键资产：

| 文件 | 来源定性 | 用途/状态 |
|---|---|---|
| `data_final.jsonl` | `D(O1)` | O1 的格式转换版，32,480 行；纯种子基线训练入口 |
| `r2_base_v3.jsonl` | `D(O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)` | 4,200 条 action-select 语义增强构造原料；脚本 `scripts/data/build_r2_actionselect.py`；保留仅供 teacher 使用的 Caption/Tag 注释；SHA256 `2ba91b019dd9c0a445af96165fc36c1c1b7939c98b730a6890ca34e9bf59aeb1` |
| `action_distill_v5.jsonl` | `D(O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)` | 164 条唯一teacher标签；`gpt-5.6-sol-max` 按稳定事件号生成，`gpt-5.6-terra-max` 独立质检且只接收score=5/5、8项检查全真的候选，脚本回映历史原始SID；这里的5/5是单个独立judge满分，不是5个teacher投票，也不是官方gold；排除354个E类action评测源索引；构建器`scripts/data/distill_action_v5.py`；因Yunwu与DeepSeek均余额不足而在164条封板，正式累计用量11,432,127 token（含被拒绝/配额错误请求）；训练混合每条重复8次（1,312有效行），不得称为1,312条新数据；SHA256 `cb020185132c9ad06ee847798bc8d78e546cd8f5a77cae323f5d802f2dd3f189` |
| `data_i01_action_distill_v1.jsonl` | `D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)` | I-06 正式训练集，33,792 行；O1 I-01 32,480 行（96.1174%）+ 164个唯一、独立judge满分action teacher标签各重复8次形成1,312有效行（3.8826%）；action总行数2,900，target token 211,816/6,364,037（3.3283%）；I-01转换12,744条冗余推荐CoT；builder `scripts/data/build_i01_action_distill_v1.py`；审计`logs/data/i01_action_distill_v1_audit.json`；SHA256 `bbefa5f24d4c9a8e0c7573873fdc2947b35880955cb60b5debc0f619d6ce99d3` |
| `data_seed_scoremax_v1.jsonl` | `D(O1)` | I-07 O1-only 性能重采样，35,558 行；完整保留 O1 32,480 行/全部 target（91.3437%），另从 1,539 条严格有序 action 金标各构造 2 个保序硬负例历史视图，共 3,078 行（8.6563%）；推荐 6,460 个题面组各保留 1 条原 CoT，其余 12,744 条切 `/no_think`；602 条 topic 切 `/no_think`；action target token 408,950/6,115,749（6.6868%）；builder `scripts/data/build_seed_scoremax_v1.py`；审计 `logs/data/seed_scoremax_v1_audit.json`；SHA256 `7df558a8c08517667f2eab4fc283f2eddfaf7efde16874099a61d63574861cb3` |
| `data_seed_o2_action_v1.jsonl` | `D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)` | I-09低剂量O2 action性能混合，33,644行；O1全量32,480行（96.5402%）+ O2双模型流程独立judge满分的唯一teacher标签164行各一次（0.4875%）+ O2规则标签1,000行（2.9723%），无重复上采样；规则行保留完整80–260事件历史，筛选4–15个唯一target/2%–12%正例密度，并将旧分数序target全部重排为历史时序；不输出Caption/Tag注释；排除354个E源索引且交集0；action target token 190,657/5,897,456（3.2329%）；builder`scripts/data/build_seed_o2_action_v1.py`；审计`logs/data/seed_o2_action_v1_audit.json`；SHA256 `ffb865e6a29d746ea609d041ee0906bda7fb2236712bd09bdee8cbe271f294d8` |
| `data_seed_teacher_v1.jsonl` | `D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)` | I-10干净性能混合，32,644行；O1全量32,480行（99.4976%）+ O2双模型流程独立judge满分的唯一action teacher标签164行各一次（0.5024%），O2规则标签0；推荐保留全部target并仅压缩12,744条重复CoT，602条topic对齐no-think；排除354个E源索引且交集0；builder`scripts/data/build_seed_o2_action_v1.py --rule-rows 0`；审计`logs/data/seed_teacher_v1_audit.json`；SHA256 `13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f` |
| `data_seed_clean_v1.jsonl` | `D(O1)` | I-14纯O1单体LoRA训练集，32,480行；完整保留O1全部行和target，O1占比100%，O2/T/E行均为0；仅将12,744条推荐冗余CoT转no-think并将602条topic对齐no-think。builder`scripts/data/build_seed_clean_v1.py`（SHA256 `2d01951d0e6d3e0f406d3a59a74e35ab8f70b8cb51e3295f8f1792714d7dc214`）；审计`logs/data/seed_clean_v1_audit.json`（SHA256 `15767552e1feac3c21e500207cb76a4805b118dda061f6b2cc3c6116255c3b11`）；target token 5,845,479，action 138,680（2.372432%）；SHA256 `e526caea4a1afd8befbd5d266fb80d0378a5bf7eff90fdacd14934332d64d309` |
| `data_o1_reward_preference_v1_train.jsonl` | `D(O1)` | I-16推荐奖励对齐偏好训练集，15,382对；上游为`data_seed_clean_v1`，仅含O1派生标签，T/E/teacher/model rollout均为0。chosen逐字节保留O1派生金标；rejected仅将最终target替换为同题面、同域历史项，并排除题面组内全部已知正例。action共1,539个候选对只作审计，其中训练桶1,392对因E3父模型已高偏好金标而全部阻断，不进入正式训练；按题面SHA256分组切分，训练/holdout题面交集0；builder`scripts/data/build_o1_reward_preference_v1.py`（SHA256 `49c0cddd3b211a491b73eb692422de5bf2a2d9a5b0956584f4c32f1d365af475`）；审计`logs/data/o1_reward_preference_v1_audit.json`；SHA256 `579171020e764b9c360b94493257848e6408487c7ca6b0dd8ba1efa76c34b52e` |
| `data_user_residual_retention_v1.jsonl` | `D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General)` | I-12用户残差与父保持混合，6,106行；用户监督3,053（action1,752、合法2–5步topic1,301，164条teacher各一次）+父保持3,053（material两向各281、video/prod/ad/live各565、O2.General world231），严格1:1。排除3条6步topic；完整历史与原target不改写；保持行只用于E3 KL，不做gold CE；O2规则/T/E行0。builder`scripts/data/build_user_residual_retention_v1.py`；审计`logs/data/user_residual_retention_v1_audit.json`；SHA256 `bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0` |
| `data_seed_cotfix_v1.jsonl` | `D(O1)` | O1 种子 CoT 补全版，32,480 行；改动 1,495 行/425 个唯一后缀 |
| `data_riders_fk_clean.jsonl` | `D/MIXED(O1,O2.General,T)` | `data_riders_fk` 逐字保序删除 5 条已登记 E 类评测泄漏后得到，37,262 行；脚本 `scripts/data/build_riders_fk_clean.py`；SHA256 `ee12db531db8b1ab4ff6486f5e483d638b130f45ff3e3100c09bb5d33e9520ab`；仅用于用户本轮明确要求的 r64×3ep 本地对比，T 成分不得外推复用 |
| `world_zh*.jsonl` | `D(O2.General)` | 从官方 General 清洗的懂世界数据 |
| `cap_grounding_v1.jsonl`、`capcot*` | `D(O3)` | 从官方对齐 Caption/Tag 构造 |
| `rec_loo*`、`official_rec*` | `D(O2)` | 从 UserProfile/Pid2Sid 构造；历史实验，不等价于官方 SFT |
| `data_riders*`、`data_stage2*`、`data_rebal*`、`data_seed_world*` | `D/MIXED` | 我方改写或混合训练集；逐个查构建脚本和实验台账 |
| `assets/fewshot_seed.json`、`assets/fewshot_v2.json` | `D(O1)` | 从种子样本整理的 teacher 标注锚；仅供构造脚本使用 |
| `assets/derived/official_general/sft_world_knowledge.jsonl` | `D(O2.General)` | 从官方 General 清洗的 231 条世界知识题 |

### D 类只读索引

- 固定入口：`assets/derived/index/`
- 上游：`D(O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag)`
- 构建器：`scripts/data/build_item_index.py`
- 用途：只读 PID/SID/Caption/Tag 联查与评测诊断；不是训练集

| 文件 | 行数 | 字节 | 校验 |
|---|---:|---:|---|
| `pid2sid.parquet` | 35,914,095 | 795,612,955 | SHA256 `16be16b2a95c8743cb4c4c970bf053eaa970c5c05c07a3e94442eb1fd52a39df` |
| `pid2caption.parquet` | 21,061,327 | 6,126,887,737 | 上游行数与 O2.Pid2Caption 一致；大文件按需校验 |
| `pid2tag.parquet` | 5,417,279 | 103,530,674 | SHA256 `dc75ba448b94a45a2e6492b407dbc06b4c178893de53baad59a4797da1475a7d` |

该索引在 2026-07-06 已由 O2 构建，2026-07-12 补录注册表。任何由它生成的新 holdout 仍须单独登记为 E 类，不能把索引或可见题用于训练。

派生数据进入训练前必须在配置或实验台账记录：`上游资产 ID + 构建脚本 + 行数 + 内容哈希 + 混合比例`。缺任何一项，不得启动正式训练。

## 4. 第三方资产 `T`（与官方物理分区）

固定访问入口：`assets/third_party/`。

| 固定入口 | 原件 | 定性 | 默认状态 |
|---|---|---|---|
| `frinkleko_sft_091/` | 运行卷 `data/third_party/frinkleko_sft_091/`，约 232M | 选手/第三方 SFT | **禁止混入** |
| `teammate/懂物料.jsonl` | 队友从 Pid2Caption 构造，8,947 行 | 队友派生物料数据 | **禁止混入** |
| `teammate/dongwuliao_api_8h.jsonl` | 队友 API 构造，47,595 行 | 队友派生物料数据 | **禁止混入** |
| `teammate/懂世界_from_mc.jsonl` | 队友蒸馏，272 行 | 含评测样例风格的第三方数据 | **禁止直接混入** |
| `teammate/懂世界final.jsonl` | 队友硬例挖掘，747 行 | 与既有 MC 池高度重叠 | **禁止直接混入** |

除非用户明确点名批准，训练配置不得引用 `assets/third_party/` 或其原件路径。

## 5. 评测衍生 `E` 与模型产物 `M`

- `E`：`assets/evaluation/visible/懂世界.jsonl`、运行卷 `data/evaluation/offline_eval/`、`logs/eval/`、`logs/offline_eval/`、`logs/precheck/`、`logs/probe/`、平台可见题、人工 holdout。默认只做门禁与诊断，不回灌训练。
- `assets/evaluation/holdout/data_o1_reward_preference_v1_holdout.jsonl`：`E(D(O1))`，1,784对（推荐1,637、action147）；与I-16偏好训练集按完整题面分组切分且题面交集0，只用于父模型偏好审计、训练期诊断与结构门禁，不回灌训练；SHA256 `1c7292cb96d45e9d20c0b3add78d3e5a30ec7a559844217584408921f996696e`。
- `M`：`checkpoints/`、`submissions/`、merged model、adapter。唯一台账是 `docs/EXPERIMENT_INDEX.md`。
- W&B 凭据与运行日志不是数据资产；凭据不得写进本表。

### E 类边界说明

- `assets/evaluation/visible/懂世界.jsonl` 实际 7 行。仅前 5 行是当前平台日志固定 common-sense 可见题，人工 gold 依次为 `A,D,A,B,A`；第 6 行无 `/no_think`，第 7 行是与当前“单项选择”规则不兼容的历史多选题。任何门禁必须锁定前 5 行的 prompt hash，禁止把 7 行混算准确率。
- `assets/evaluation/offline_eval/` 当前冻结卷：mat_fresh 542、mat_train 300、rec 四域各 1,000、action 325、topic 110、world 500。旧卷缺稳定 user/PID/题源实体键，只能复现历史 v3 校准，不能据此做实体级 bootstrap。
- `logs/eval/` 当前包含 30 个唯一线上评测日志及一个 `seed_ep3` 原始哈希名重复副本；去重必须按evalTaskId，不能按文件数。最新五份为 `seed_teacher_e3_cont_r64_lr2e5_ep1_20260713.log`、`e3_userres_r80_retkl_v3_ep1_20260713.log`、`e3_userres_r80_retkl_v3_s875_20260714.log`、`seed_clean_r80_lr1e4_ep3_rerun1_20260714.log` 与 `seed_teacher_e3_dpo_rec_o1hard_lowdose_v2_step100_20260714.log`。
- 2026-07-13下午平台修复评测不稳定。仓内日志可证实切点位于I-10 E3（11:45）与I-11（16:40）之间：前25个唯一日志（含I-10 E1/E2/E3）使用action4096+itemic单跑的修复前协议；I-11/I-12/I-13/I-14/I-17使用action1024+itemic 7次race-average的固定协议。两边都打印`version: v3.1`，因此去重后还必须按协议指纹分层，禁止跨层校准或作差。
- `logs/offline_eval/` 和 `logs/probe/` 是评测运行产物，不是训练资产。旧 parser/protocol 结果必须按版本隔离，不能混入新校准。

## 6. 不再重复扫描协议

1. 数据、训练、评测任务开始时只读本表和目标实验配置，不执行全盘 `find/du/rg` 重新发现资产。
2. 只有以下情况允许重扫目标目录：登记路径不存在、哈希不符、官方发布新资产、用户明确要求重新审计。
3. 重扫只能限定在发生变化的资产目录，禁止从卷根递归扫描。
4. 发现新资产先分类为 `O/D/T/E/M` 并更新本表，再使用。
5. 旧文档出现冲突时，以本表为准；`DATA_INVENTORY.md` 已废止，不再维护第二套口径。

项目根 `data/` 是分类兼容入口：`official/derived/third_party/evaluation` 分别指向 `assets/` 对应分区，`processed` 仅为旧脚本兼容链接。项目根目录禁止直接存放 JSONL/Parquet 等数据文件。

## 7. 维护记录

- 2026-07-14：登记I-17 step100平台评测日志`logs/eval/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2_step100_20260714.log`，2,657,186 bytes，SHA256 `5e7a0dff1a9b9048862f00eed0f7a67094bb01acfc15b62f60d776c03dca3fc7`，evalTaskId `eval-task-eeve0r-1784036284`；固定协议指纹action1024+itemic 7次race-average，8/8任务完成、`Failed tasks 0`。该日志为E类，只作线上结果诊断与审计，不回灌训练。
- 2026-07-14：补登记I-14平台评测日志`logs/eval/seed_clean_r80_lr1e4_ep3_rerun1_20260714.log`，2,905,022 bytes，SHA256 `046a2e53b009206b1b88306c99682cc1a9444cc711a7820212e55efedf324153`，evalTaskId `eval-task-lfrrhq-1784013605`；固定协议指纹action1024+itemic 7次race-average，8/8任务完成、`Failed tasks 0`。该日志为E类，只作线上结果诊断与审计，不回灌训练。
- 2026-07-14：登记I-16 O1奖励对齐偏好资产。builder从`data_seed_clean_v1`构造17,019个推荐候选对和1,539个action候选对；推荐负例严格同题面同域并排除组内全部已知正例，action负例只增加一个保序非金标历史事件。E3父模型分层审计显示推荐chosen原始胜率17.19%–43.75%，action为93.75%，故正式D类训练集只保留推荐15,382对，阻断action训练桶1,392对；E类holdout保留推荐1,637+action147用于漂移门禁。训练/holdout题面交集0，训练侧O1派生占比100%，T/E/teacher/model rollout均为0；SHA256分别为`57917102...b52e`/`1c7292cb...696e`，完整构造审计见`logs/data/o1_reward_preference_v1_audit.json`。
- 2026-07-14：登记I-14训练集`data_seed_clean_v1.jsonl`。上游仅O1的D格式入口`data_final.jsonl`，32,480行/100%；builder完整保留全部target，只压缩12,744条推荐冗余CoT并将602条topic对齐no-think；O2/T/E行0。builder、审计和输出SHA256分别为`2d01951d...dc214`、`15767552...c3b11`、`e526caea...d309`；正式训练配置与混合比例登记在`docs/EXPERIMENT_INDEX.md`。
- 2026-07-14：登记I-13平台评测日志`logs/eval/e3_userres_r80_retkl_v3_s875_20260714.log`，2,777,778 bytes，SHA256 `9291f8bf87871bb93846dda4cfcf60d43812354fb87a18e6ef6a5a349bdb3315`，evalTaskId `eval-task-9ie86v-1783961075`；固定协议指纹action1024+itemic 7次race-average，8/8任务完成、`Failed tasks 0`。该日志为E类，只作线上结果诊断与审计，不回灌训练。
- 2026-07-14：登记I-13本地E类诊断产物。多尺度先导`logs/probe/i13_userres_scale_pareto_20260714.json`（17,073 bytes，SHA256 `a2e59102e6589351efc4e53ea3d12bde3e6dd3ce11e9fea1a13684f9331f1e14`）；严格六任务576条留出审计`logs/probe/i13_userres_scale_pareto_full_20260714.json`（20,574 bytes，SHA256 `c937b9be62728a87cd91b901c06af5c62b30604f6a821e8f7ca41e8de05f82fc`）；门禁`logs/precheck/e3_userres_r80_retkl_v3_s875_precheck.log`（1,612 bytes，SHA256 `cbd32b15b2401acd9b233b38419c7acc91df4d4df6ca1a1333bb11066cf66bda`）。留出源为已登记D(O1,O2)`data_seed_teacher_v1`，逐字节排除I-12训练数据；三者仅作模型选择与结构否决，不进入训练。I-13模型与组合审计属M类，登记在`docs/EXPERIMENT_INDEX.md`。
- 2026-07-13：登记平台“评测分数不稳定已修复上线”通知，并以原始日志指纹建立协议边界：`platform-pre-fix-v3.1`截至I-10 E3，`platform-stable-v3.1-20260713`从I-11起可证实。标签为仓内审计名，不是官方版本号；E3固定协议sentinel完成前，I-11/I-12不得与I-10旧分直接比较。
- 2026-07-13：登记I-12平台评测日志`logs/eval/e3_userres_r80_retkl_v3_ep1_20260713.log`，2,642,720 bytes，SHA256 `151bddf09f301794885e66a9df7387d3141475daa8f0e9a249cc8b96381cf450`，evalTaskId `eval-task-jnbjjq-1783944993`；8/8任务完成、`Failed tasks 0`。该日志为E类，只作线上结果诊断与审计，不回灌训练。
- 2026-07-13：登记I-12 v3本地诊断产物：训练日志`logs/train/e3_userres_r16_retkl_v3_ep1.log`（189,934 bytes，SHA256 `a2a2330a...7b450`）、训练内配对机制审计`logs/probe/e3_userres_r16_retkl_v3_ep1_paired_audit.json`（3,428 bytes，SHA256 `03caba63...45d45`）和硬结构门禁`logs/precheck/e3_userres_r80_retkl_v3_ep1_20260713.log`（1,611 bytes，SHA256 `afb989c8...5922`）。后两者均为E类本地诊断，只验证机制/灾难安全，不进入训练、不估线上分数。
- 2026-07-13：登记I-12训练集`data_user_residual_retention_v1.jsonl`。上游为I-10 `D(O1,O2)`混合与已登记`D(O2.General)`世界保持集；6,106行，用户CE/父KL保持各3,053行，O2规则/T/E为0；builder、分层配比、完整历史/target审计和3条超5步topic排除见`logs/data/user_residual_retention_v1_audit.json`；SHA256 `bd947aad...b08f0`。
- 2026-07-13：登记 I-11 平台评测日志`logs/eval/seed_teacher_e3_cont_r64_lr2e5_ep1_20260713.log`，2,716,035 bytes，SHA256 `95130e363ba16d873a74303405ca29fdf869628ed9a9558fa5a95bb3fa0e614b`，evalTaskId `eval-task-kxwokc-1783932031`；8/8任务完成、`Failed tasks 0`。该日志为E类，只作线上结果诊断与审计，不回灌训练。
- 2026-07-13：补录`logs/precheck/`为E类诊断产物；I-11门禁日志`seed_teacher_e3_cont_r64_lr2e5_ep1_20260713.log`及结构化摘要`logs/probe/seed_teacher_e3_cont_r64_lr2e5_ep1_gate_summary.json`只用于硬结构保险丝，不进入训练或本地估分。
- 2026-07-13：登记 I-10 E1/E2/E3 三份平台评测日志，均完成8/8任务且`Failed tasks 0`。规范文件、字节数、SHA256、evalTaskId依次为：`seed_teacher_r64_lr1e4_e1_20260713.log` / 2,660,286 / `99e691a9c95eca6232a2b1896bddd4ff1e4eab29404e9869b10de50f39612d36` / `eval-task-00fvcu-1783914281`；`seed_teacher_r64_lr1e4_e2_20260713.log` / 2,575,516 / `9e2de68438d6063a3ee9bc3795bd9fd165351953dbd18389f16ffd47c1b5d35e` / `eval-task-6usmb7-1783908972`；`seed_teacher_r64_lr1e4_e3_20260713.log` / 2,521,157 / `c6868c3e24b213553c4d3e8fb9f89fc2c1d61cf77cb02b7b95fde414e7e103b6` / `eval-task-3k8v5e-1783914292`。三份均为E类，只作线上结果诊断与审计，不回灌训练。
- 2026-07-13：纠正`action_distill_v5`质量口径：5/5表示独立judge单次满分且8项检查全真，不是5个teacher投票，也不是官方gold；规则标签F1仅表示其相对teacher参考严重不一致，不能反向证明teacher为官方真值。
- 2026-07-12：登记 I-10 `data_seed_teacher_v1.jsonl`。以同源164条独立judge满分teacher标签为参考的审计显示旧规则标签全量平均F1 0.0429、匹配I-09过滤条件子集平均F1 0.0813，因此正式混合删除全部1,000条规则标签，只保留O1全量与164条唯一teacher标签；32,644行，SHA256 `13c40526...eee4f`。
- 2026-07-12：登记 I-09 `data_seed_o2_action_v1.jsonl`：完整保留O1 32,480行，以一次采样加入164条O2双模型流程独立judge满分teacher标签和1,000条去泄漏、完整历史、target时序纠正的O2规则标签；O2总占比3.4598%，无T/E、无重复O2标签；33,644行，SHA256 `ffb865e6...f294d8`。
- 2026-07-12：登记 I-07 `data_seed_scoremax_v1.jsonl`：仅使用 O1，保留全部 32,480 条原始 target；新增 3,078 条 label-preserving action 历史视图，将 action target-token 占比提升至 6.6868%；推荐冗余 CoT 与 topic think 按评测模式压缩；35,558 行，SHA256 `7df558a8...61cb3`。
- 2026-07-12：登记并封板 `action_distill_v5.jsonl`：teacher 只选事件号、程序回映 SID，独立 judge 仅接收 5/5，显式排除 354 个 E 源索引。Yunwu/DeepSeek 余额耗尽后停止在 164 条唯一标签，未将 43 个配额错误计作质量拒绝；训练 mix 每条重复 8 次并分别登记 unique/effective rows；SHA256 `cb020185...3f189`，精确审计见 `logs/data/action_distill_v5_summary.json`。
- 2026-07-12：登记 I-06 正式训练集 `data_i01_action_distill_v1.jsonl`；上游 D(O1,O2)，无 T/E，33,792 行；I-01 转换 12,744 行，164 个唯一 action 标签按 8 倍显式采样为 1,312 有效行；SHA256 `bbefa5f2...ce99d3`。
- 2026-07-12：登记 r64 轨迹 E1/E2 两份平台评测日志并从原始下载哈希名规范化为 `riders_fk_clean_r64_e1_20260712.log`（SHA256 `4416ed18...`）和 `riders_fk_clean_r64_e2_20260712.log`（SHA256 `f14851bb...`）；日志均属 E 类，只作诊断与台账回填。
- 2026-07-12：补清 E 类边界：visible world 7 行中仅前 5 行属于当前固定题；登记 offline/probe 日志为 E 产物，并记录旧 offline 卷规模、实体键缺口和 evalTaskId 去重规则。
- 2026-07-12：补录既有 `assets/derived/index/` 只读索引；确认构建器、上游 O2、三表行数/字节，并校验 `pid2sid` 与 `pid2tag` SHA256。用途限定为联查和评测诊断。
- 2026-07-11：新增 `data_riders_fk_clean.jsonl`。上游为 `data_riders_fk`（SHA256 `e4f91c52...`），只删除文档已登记的 5 条 E 类评测真题，保留行字节与相对顺序不变；审计 `logs/data/riders_fk_clean_audit.json`。
- 2026-07-11：纠正重大来源误判：`OpenOneRec-General-SFT` 是 OpenOneRec 官方发布，不是第三方；补录官方 `OpenOneRec-General-Pretrain`；官方直发资产扩展为 O1-O6。
- 2026-07-11：O4/O5 按官方 revision、Parquet 数量和总字节完成本地验收；下载缓存已删除，只保留官方文件本体。
- 2026-07-11：运行卷数据物理拆分为 `official/derived/third_party/evaluation/archive`，建立 `assets/` 固定入口。
