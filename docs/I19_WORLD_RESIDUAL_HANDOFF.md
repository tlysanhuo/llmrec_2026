# I19-world-residual 当前最高方案交接

> 更新日期：2026-07-20 UTC
> 本文中的 `I19-world-residual` 专指懂世界残差方案，不是仓内已有的 I-19 DPO 实验 `i13_s875_dpo_rec_balhard_lowdose_v1`。

## 当前结论

- 当前最高单次线上观测是 `residual-scale=0.875` 的 r96 adapter，总分 `1.025259456`。
- 该点只评测一次；相邻 `0.75/0.8/0.9` 分别为 `0.996570849/0.990238620/0.977796092`，没有形成平滑或单调趋势。因此 `1.0253` 先作为当前最高观测，不声明稳定复现。
- 训练实际 parent 是独立复现的 `i13_repro_combined_r80_s875`，其线上分为 `0.9867038438530921`；它与仓内原 I-13 s875 配方相同但权重不 bitwise 一致，不能混用 `0.9978` 或 s800 的 `1.0048` 作父子净差。
- 当前服务器已接收并验收最高点 r96 严格两文件提交包；adapter SHA256 与报告逐字节命中。报告所列的 r80 parent、r16 residual、发布数据目录和三份复现脚本仍未到卷，因此只证明线上提交包身份，不证明完整训练链已在本卷闭合。

## 方法与训练

1. 冻结独立复现的 I-13-like r80 parent。
2. 用 1,573 条 Frinkleko `_clean` 懂世界数据训练 fresh LoRA r16；该数据由 1,578 条源桶剔除 5 条 competition smoke 重合行得到，属于用户明确授权的第三方源派生数据。
3. 从 `data_seed_teacher_v1` 的 action/topic/material 双向/video/prod/ad/living 八桶分层抽取 1,573 条保持样本；懂世界与保持严格 `1:1`，总计 3,146 行。
4. world 路由使用全响应 CE（报告描述为 gold/weighted CE）加 `0.05 * parent KL`；retention 路由不做 CE，只使用 `2.0 * parent KL`。
5. 单卡训练 fresh r16：`r=16, alpha=16, dropout=0.05, all-linear, lr=5e-5, cosine, warmup_ratio=0.03, weight_decay=0.001, batch=1, gradient_accumulation=4, cutoff=4096, bf16, seed=19260821, epoch=1`；共 787 steps，约 16m17s，train loss `0.5217204`。
6. 将 r16 residual 按 scale 与 r80 parent 做参数空间精确秩拼接，得到单个 r96 adapter。

训练混合报告 SHA256：`a8af6884cd8c5064686981ddf2b0ff9ad96bdcf46cb3f4680c5aca5458fedb86`。
实际 r80 parent 报告 SHA256：`a63a45c3ba5242f60979d4fbed66ad2a92e2cf3903b1bf70c7ae9b3fe3515ed0`。
r16 residual 报告 SHA256：`144ee8efc24ce5a43e428fc9db2c21b5b3c176970df44ea01fb6e253281d4d6d`。

## 线上结果

八项顺序为 material/action/topic/video/prod/ad/living/world；以下均为单次观测。

| scale | 总分 | 八项 | evalTaskId | 平台模型 ID |
|---:|---:|---|---|---|
| parent | 0.986703844 | `0.2453/0.1207/0.0386/0.0864/0.1190/0.1302/0.1071/0.1394` | `eval-task-kb6nyy-1784126461` | `md-r3nl9u-1784126402331424701` |
| 0.75 | 0.996570849 | `0.2453/0.1209/0.0397/0.0576/0.1326/0.1344/0.1044/0.1617` | `eval-task-1i7jj3-1784437859` | `md-2m1lyk-1784437811686478021` |
| 0.8 | 0.990238620 | `0.2453/0.1220/0.0392/0.0576/0.1258/0.1358/0.1062/0.1584` | `eval-task-wr6v4y-1784437512` | `md-fseu1w-1784437485490630282` |
| **0.875** | **1.025259456** | `0.2453/0.1225/0.0399/0.0768/0.1326/0.1400/0.1080/0.1602` | `eval-task-g4y7us-1784436397` | `md-cm6gw1-1784436350784564154` |
| 0.9 | 0.977796092 | `0.2453/0.1200/0.0394/0.0576/0.1224/0.1274/0.1062/0.1595` | `eval-task-4lmbg1-1784446875` | `md-tve9r7-1784446503752485428` |

r96 adapter 报告 SHA256：

| scale | SHA256 | 状态 |
|---:|---|---|
| 0.75 | `449bb105ba0cfdcd6317b9623a701a3cdb77d21f0ebcf8d72d6b3493f3b7ca4d` | 已线上评测，当前服务器缺失 |
| 0.8 | `d6f1ab14be328b693a49e2e0e1bb8d82600180561de981d6bf9e02b969773d25` | 已线上评测，当前服务器缺失 |
| **0.875** | `4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e` | 当前最高单点；严格两文件包已到卷并验收 |
| 0.9 | `bd23b752256f6cd072bc262e0462744c501e2ce4adbd5be502f5994d26d03f79` | 已线上评测，当前服务器缺失 |
| 1.0 | `61e83bee6867fdd06fc6a91400c95ad35691672389d4aa62403d80210a71d5b3` | 报告称已拼接，未线上评测，当前服务器缺失 |

## 待接收与验收

- [x] 最高点 r96 的严格两文件提交包：`submissions/i19_world_external_r96_s875_platform/`；adapter/config 为242,273,688/1,074 bytes，SHA256 `4fba17eb...078e`/`78b62143...1b64f`，r96/alpha96、392个LoRA tensors、仅两文件。
- 实际训练 parent r80（期望`a63a45c3...15ed0`）、r16 residual，以及每个候选的 combine audit/config。
- `assets/derived/releases/i19_userres_retention_v1/`、`assets/derived/releases/i19_frinkleko_world_1578/` 和相应 manifest/污染审计。
- 报告所列 builder、trainer、delta audit 与训练配置；当前本卷路径均不存在。
- 对应 W&B run ID 和完整训练日志。报告没有提供 run ID，未满足本仓库“新本地训练必须启用 W&B”的完整证据要求。

当前状态为 `HIGHEST_OBSERVED_ONLINE_PACKAGE_VERIFIED_REPRO_CHAIN_INCOMPLETE`。最高点提交包已可直接交付，但在 parent、residual、数据与训练证据到达并通过哈希、行数、路由和父权重身份检查前，不得写成当前服务器已完整复现训练链。
