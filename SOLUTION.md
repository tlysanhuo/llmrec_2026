# Solution — LLM-Rec 2026 (Team CornerCase)

**最终榜分:rank 55,总分 1.0567**(提交时间 2026-07-30 16:50 UTC)。

## 一句话方案

把两个 LoRA adapter —— **i35**(懂推荐 / video-boundary)与 **i50**(多教师懂物料)—— 通过**全参正交残差融合(full-weight orthogonal residual fusion)**在 λ=0.10 下合并进 OneReason-0.8B 基座,导出为 BF16 全参模型。

## 为什么有效

i35 是最强单模型(线上 1.0344),但 i50 掌握互补的物料知识(1.0302)。朴素平均(soup)会把强度洗向均值,task-vector 直接相加会有方向冲突。正交残差融合**只注入 i50 更新中与 i35 更新正交的分量**,因此保住 i35 的强项,同时叠加 i50 不冲突的物料方向。最终 1.0567 > i35 的 1.0344,+0.0223。

## 公式

对每个目标权重矩阵(q/k/v/o/gate/up/down proj):

```
ΔA = W_i35 − W_base          # i35 的有效增量
ΔB = W_i50 − W_base          # i50 的有效增量
R_B = ΔB − (⟨ΔB,ΔA⟩ / ‖ΔA‖²) · ΔA     # i50 增量中正交于 i35 的部分
W_fused = W_i35 + λ · R_B     # λ = 0.10
```

非目标权重:保留 i35。结果以 BF16 全参存储(`model.safetensors`,约 1.6 GB)。

> 实现见 `scripts/train/full_weight_orthogonal_fuse.py`。配套的 adapter 级变体 `orthogonal_residual_fuse.py` 在 LoRA 参数空间做同样的正交残差融合、输出一个更高秩的融合 adapter(用于 i73/i74 探索)。

## 复现

> **权重不发布。** 自行训练两个组件,再融合。

1. **基座**:从官方获取 [OneReason-0.8B](https://github.com/...) 预训练 checkpoint(本仓库不分发),放到 `models/OneReason-0.8B-pretrain-competition/`,或修改融合脚本里的 `BASE`。
2. **训练 i35(model A)**:`scripts/train/train_i35_video_boundary_retkl.py` + `configs/` 下对应配置。产出 r112 LoRA adapter(线上 1.0344)。
3. **训练 i50(model B)**:多教师懂物料 adapter,r128(线上 1.0302)。⚠️ **i50 训练脚本不在本发布内**(见下方"缺口")。
4. **融合**:
   ```bash
   python scripts/train/full_weight_orthogonal_fuse.py \
     --model-a <i35_adapter> --model-b <i50_adapter> \
     --lambda 0.10 --output <fused_model_dir>
   ```
5. **评测**:`eval/run_eval.py`(离线协议 `offline-eval-v4` 见 `docs/offline_eval.md`)。

## 分数轨迹

| 阶段 | 总分 | 日期 |
|---|---|---|
| v0 官方预训练锚点 | 0.6655 | 2026-07-01 |
| baseline SFT | 0.810 | 2026-07-01 |
| I-13 s875(固定协议主线) | 0.9978 | 2026-07-14 |
| I-23 seed_teacher_cotfix_v3(最强单 adapter) | 0.9915 | 2026-07-16 |
| I19-world-residual r96 | 1.0253 | 2026-07-19 |
| **I-35 step548 r112**(最强单模型,默认交付) | **1.0344** | 2026-07-22 |
| **I-35 ⊕ I-50 正交融合 λ=0.10(最终)** | **1.0567** | 2026-07-30 |

完整逐实验历史(I-01 → I-74 + 最终合并)见 `docs/experiment_log.md`、`docs/EXPERIMENT_INDEX.md`、`docs/EXPERIMENT_RECORDS_I41_I74.md`。

## 包含 / 不包含

**包含**:夺冠融合脚本 + adapter 级变体、i35 训练脚本与配置、评测代码、LLaMA-Factory 改动(数据注册表 / parser 补丁 / 测试,`.env.local` 含 W&B key 已排除)、完整实验记录。

**不包含**:
- **模型权重 / adapter**(基座、i35、i50、融合模型)——不发布,按代码自训。
- **竞赛原始数据**——不分发(许可)。数据卡与小份派生样本见配套仓库 `llmrec-post-training-notes`。
- **原始训练 / 评测日志、W&B runs**——不发布。
- **密钥**——无。`.env.local`、`.netrc`、API key 均已清除。

## 缺口 / 诚实声明

1. **i50 训练脚本不在本发布内。** i50 训于 2026-07-25,早于可恢复的会话记录,脚本未落盘保留。i50 的 adapter config 可获取,训练配方在 `docs/EXPERIMENT_RECORDS_I41_I74.md` 高层描述。若有原始脚本欢迎补回。
2. **I-40 之后的代码是从历史 Claude Code 会话 transcript 里逐字恢复的,不是磁盘原文件。** 原工作代码目录(卷根 `llmrec_2026/`)已丢失;此处的脚本是产出最终提交的同一份代码,从 transcript 提取。脚本内路径假设原始目录结构(见上文)。
3. **I-41→I-74 多数实验没有成文详细记录**,只有 artifact 与自动 model card。`docs/EXPERIMENT_RECORDS_I41_I74.md` 中这些条目是从 artifact / 日志重建的,不确定处已标注;**不编造没有的 SHA256 / evalTaskId 字段**。

## 许可

MIT(见 `LICENSE`)。基座 OneReason-0.8B 与上游 LLaMA-Factory 保留各自许可。
