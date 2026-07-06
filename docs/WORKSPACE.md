# 快手探索者 LLM-Rec 挑战赛 2026 工作区

官方页面: https://ks-llmrec.streamlake.com/#metrics

本目录用于沉淀比赛资料、代码、配置、提交文件和轻量说明。数据、模型、日志、缓存等大文件应写入本目录中的符号链接目录，它们实际指向个人卷运行区:

`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026`

## 赛事要点

比赛聚焦推荐模型与大语言模型融合，目标是让模型在真实推荐场景中同时具备物料理解、用户理解、推荐预测和通用知识能力。

四类能力方向:

- 懂物料: 理解多模态物料内容，对齐推荐 itemic pattern 与通识语义。
- 懂用户: 根据行为历史洞察用户需求，捕捉动态偏好与需求演化。
- 懂推荐: 根据用户全域历史行为，预测用户可能有需求的物料。
- 懂世界: 保持大模型的百科知识、社会常识与通用问答能力。

## 时间线

- 2026-06-13 至 2026-06-29: 报名注册
- 2026-07-01 至 2026-07-31: 初赛
- 2026-08-01 至 2026-08-31: 复赛
- 2026-09-01 至 2026-09-15: 代码复现审核
- 2026-09 底: 决赛与颁奖

## 评估指标

- 物品理解: 生成物料描述对应的 itemic pattern，采用 Pass@64；复赛还包括 itemic pattern 到物料描述，由 LLM-as-a-Judge 评估覆盖准确性与完整性。
- 用户兴趣演化: 给定用户历史和兴趣主题，判断相关历史行为，采用 F1；同时评估兴趣演化链中行为准确性与推理合理性。
- 推荐物料: 采用 Pass@64，thinking 与 non-thinking 模式各生成 32 条结果，合并为 64 条候选，检查目标物品是否命中。
- 常识问答: 不定项选择题，全部选项完全匹配才得分，采用 Accuracy。

## 奖项与资格

奖金池总额 100 万元。冠军 40 万元，亚军 20 万元，季军 10 万元，4-10 名每队 3 万元，技术创新专项奖 4 队每队 2 万元。入围复赛可获快手周边和赛事证书；前三名可获得 K-Star 级别 offer，决赛前 20 名免笔试直达快手算法终面。

参赛对象为全球范围内各类院校全日制在校学生，含初中、高中、本科、硕士、博士、博士后。每队 1 至 3 人，每人只能加入一支队伍。队长负责报名、沟通和最终成果提交；因平台实名认证要求，队长需持有中国内地有效身份证件和中国内地手机号码。快手在职实习生及与快手存在劳务或外包合作关系的在校生不可参赛。

## 目录约定

- `docs/`: 赛题说明、规则摘录、方案文档。**入口:`docs/project_brief.md`(单页概览+文档索引)。**
- `src/`: 可复用代码。
- `scripts/`: 数据处理、训练、评估、提交脚本。
- `configs/`: 训练与推理配置。
- `notebooks/`: 轻量分析 notebook。
- `submissions/`: 提交包和提交说明。
- `data/`: 数据目录，符号链接到个人卷运行区。
- `models/`: 模型和权重目录，符号链接到个人卷运行区。
- `checkpoints/`: 训练 checkpoint，符号链接到个人卷运行区。
- `logs/`: 日志目录，符号链接到个人卷运行区。
- `cache/`: 工具缓存，符号链接到个人卷运行区。
- `tmp/`: 临时文件，符号链接到个人卷运行区。
- `wandb/`: W&B 本地目录，符号链接到个人卷运行区。

## 环境默认值

运行可能写入较多文件的命令前，先设置:

```bash
export PERSONAL_VOLUME_ROOT=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
mountpoint -q "$PERSONAL_VOLUME_ROOT" || { echo "ERROR: personal volume is not mounted"; exit 1; }
export PROJECT_ROOT="$PERSONAL_VOLUME_ROOT/llmrec_2026"
export PROJECT_RUNTIME_ROOT="$PERSONAL_VOLUME_ROOT/ai_runtime/llmrec_2026"
export TMPDIR="$PROJECT_RUNTIME_ROOT/tmp"
export HF_HOME="$PROJECT_RUNTIME_ROOT/cache/hf"
export TRANSFORMERS_CACHE="$PROJECT_RUNTIME_ROOT/cache/hf"
export PIP_CACHE_DIR="$PROJECT_RUNTIME_ROOT/cache/pip"
export WANDB_DIR="$PROJECT_RUNTIME_ROOT/wandb"
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR"
```

注意: 当前个人卷已接近满载，下载数据集、模型或解压归档前必须先做容量检查。

## Change Log

- 2026-06-23 03:47 UTC - 统一竞赛工作区命名为 `llmrec_2026`。What changed: 将 README 移入 `llmrec_2026`，并把项目根目录与运行区路径从 `ks_llmrec_2026` 修正为 `llmrec_2026`。Why: 用户指定 `llmrec_2026` 为竞赛文件夹名，避免重复目录和路径混乱。
- 2026-06-23 03:10 UTC - 创建 LLM-Rec 竞赛工作区说明。What changed: 记录赛事目标、时间线、评估指标、奖项资格和本地目录约定。Why: 后续代码、数据、实验和提交工作需要一个统一入口，并避免大文件写入本地临时盘。
