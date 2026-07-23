# I-35 Step411 Decision Record

> 更新：2026-07-22 UTC。本文只记录 I-35 的 step411 与已完成评测的 step548 对照，不把离线指标当作线上分数证明。

## 结论

- 目标包：`submissions/i35_r96_video_boundary_retkl_r112_step411_platform/`。
- 建议：如果共享账号仍有一次线上额度，提交 step411 一次，作为 step548 的低剂量对照；不要替换当前默认交付包 step548。
- step548 当前线上最高分为 `1.0344285849069457`。step411 尚无线上结果，提交的主要价值是确认剂量曲线，不是已有本地证据支持的确定性提分。
- `i36_i35_user_expand_retkl_r128_step2063_platform/` 不属于本次对照，不建议提交；I-36 分支已因 step4125 的 `0.9865` 线上结果关闭。

## 已验收工件

| 项 | step411 | step548 |
|---|---|---|
| 组合结构 | I19-world r96 parent + I-35 fresh r16 residual | 相同 |
| rank / alpha | 112 / 112 | 112 / 112 |
| tensors / 两文件总大小 | 392 / 282,645,380 bytes | 392 / 282,645,380 bytes |
| adapter SHA256 | `e26eb9befd0ad2a1b60e7f088d6788e8101f32b7e1d43d8d9a0114f75da35d58` | `52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00` |
| config SHA256 | `4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996` | 相同 |

step411 与 step548 的 residual 权重 392 个张量、10,092,544 个元素的参数余弦相似度为 `0.9999952352`，差分范数相对 step411 范数约 `0.31%`。这说明它们是同一训练轨迹的相邻低/高剂量点，而不是两个独立方向。

## 成对离线对照

两包在同一远端 H100 环境使用 `offline_eval.py` v4、同一 base、GPU0、seed=42、`mat,rec`、`n_rec=32`、每域 material 16 条运行。原始结果已保存为：

- `logs/offline_eval/i35_step411_quick_20260722.json`，SHA256 `76e438e4c15c2c7689f29fc0c15a296cfeea4a58e4de9af300702a58c549015f`
- `logs/offline_eval/i35_step548_quick_20260722.json`，SHA256 `46e510cd0d19ffb28ea0b7db63d3b5b211d38f2cdf966fc365f9da8457c00b5b`

| 指标 | step411 | step548 | step548 - step411 |
|---|---:|---:|---:|
| material fresh pass@64 | 0.0938 | 0.0938 | 0 |
| material train pass@64 | 0.1562 | 0.1719 | +0.0157 |
| video copy 诊断 | 0.5977 | 0.5986 | +0.0009 |
| product copy 诊断 | 0.3662 | 0.3691 | +0.0029 |
| ad copy 诊断 | 0.4639 | 0.4697 | +0.0058 |
| live pass@64 | 0.0312 (1/32) | 0 (0/32) | -0.0312 |

`material train` 是圈内记忆对照，不能作为泛化收益证据；live 的差异只有 32 题中的 1 题，也不能作为排序依据。离线台的校准结论见 [`docs/offline_eval.md`](offline_eval.md)：除 world 方向外，离线数字不能预测平台总分，只能做回归和结构检查。

## 分数预期

这是提交规划用的启发式范围，不是平台预测器：

- 中心估计：约 `1.033--1.034`。
- 实用风险区间：约 `1.030--1.036`。
- 不应把“超过 step548 的 `1.0344286`”作为预期结果；本地证据更支持“同一分数带、小幅上下波动”。

只有线上评测能决定 step411 是否替换 step548。提交后仍以两者较高者作为交付模型。

## 复现命令

完整对照应在远端 GPU 环境运行，Mac 只适合做哈希和 safetensors 结构检查：

```bash
V=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
PY=$V/miniconda3/envs/verl_v071/bin/python
BASE=$V/llmrec_2026/models/OneReason-0.8B-pretrain-competition

for s in 411 548; do
  $PY $V/llmrec_2026/scripts/eval/offline_eval.py \
    --model "$BASE" \
    --adapter "$V/llmrec_2026/submissions/i35_r96_video_boundary_retkl_r112_step${s}_platform" \
    --gpu 0 --dims mat,rec --n_rec 32 --mat-per-domain 16 \
    --generation-seed 42 --tag "i35_step${s}_quick" \
    --out "/tmp/i35_step${s}_quick.json"
done
```

## 新会话快速入口

将下面整段作为新对话的第一条消息：

```text
你接手的是 /lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026 的 LLM-Rec 2026 竞赛工程。先读 AGENTS.md、README.md、ideas/README.md、docs/TODO.md、docs/EXPERIMENT_INDEX.md、docs/I35_STEP411_DECISION.md 和 docs/offline_eval.md，再开始任何操作。

当前事实：I-35 step548 的 r112 包已线上成功，当前最高分 1.0344285849069457；八项为 0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591。I-35 step411 是同一父模型和同一 fresh-r16 轨迹的较低剂量 r112 包，严格两文件、392 tensors、282,645,380 bytes，尚无线上结果。step411 与 step548 的 residual 参数余弦相似度 0.999995，成对离线小样本结果基本持平；离线台不能预测线上总分。

当前只讨论是否占用一次共享线上额度提交 i35_r96_video_boundary_retkl_r112_step411_platform。不要上传 step137/274/685，不要上传 I-36 step2063，也不要从 I-36 失败点续训。请先给出：1) 是否提交及理由；2) 预计分数只能给启发式区间并明确不确定性；3) 精确目录和平台填写信息；4) 提交后如何以 step411/step548 较高者作为默认模型。不要虚构线上结果，不要把离线数字写成提分证据，也不要在我确认前启动新训练或新增候选。
```
