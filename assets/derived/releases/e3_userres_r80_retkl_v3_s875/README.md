# I-13 最高分实现发布：`e3_userres_r80_retkl_v3_s875`

这是仓库当前最高单次显示分、也是固定协议最高分实现：**0.9978**。八项依次为：

```text
material/action/topic/video/prod/ad/live/world
0.2453/0.1183/0.0390/0.0960/0.1224/0.1316/0.1062/0.1390
```

它不是一次普通的单阶段 SFT：先从 O6 训练 I-10 r64 父 adapter，再训练 I-12 r16 用户残差，最后按下式精确拼为一个 r80 adapter：

```text
delta_I13 = delta_parent_r64 + 0.875 * delta_user_residual_r16
```

发布件包含这条链路实际使用的两份完整训练数据、可移植配置、训练/拼接脚本和原始小型审计。两份数据都是官方源派生 `D`，不是官方直发；训练行中没有第三方 `T` 或评测回灌 `E`。

## 先验证与还原数据

在仓库根目录运行：

```bash
scripts/reproduce/i13_highscore.sh verify-data
scripts/reproduce/i13_highscore.sh restore-data
```

`verify-data` 不写文件，会对两个 gzip 及其解压 payload 做双层 SHA256、字节数和行数校验。`restore-data` 原子还原：

```text
assets/derived/releases/e3_userres_r80_retkl_v3_s875/data_seed_teacher_v1.jsonl
assets/derived/releases/e3_userres_r80_retkl_v3_s875/data_user_residual_retention_v1.jsonl
```

原始 JSONL 由 `.gitignore` 排除，避免重复提交；LLaMA-Factory 注册文件已经指向上述仓库相对路径。

## 三阶段复现

先按 `docs/reference/ASSETS.md` 准备 O6 基座，使其可从 `models/OneReason-0.8B-pretrain-competition` 访问，并准备 LLaMA-Factory、单张 GPU 和 W&B 登录。先执行机制自测：

```bash
scripts/reproduce/i13_highscore.sh self-test
```

然后可分阶段执行：

```bash
WANDB_ENTITY=3120252125- WANDB_PROJECT=llmrec-2026 \
  scripts/reproduce/i13_highscore.sh train-parent 0

WANDB_ENTITY=3120252125- WANDB_PROJECT=llmrec-2026 \
  scripts/reproduce/i13_highscore.sh train-residual 0

scripts/reproduce/i13_highscore.sh combine
```

也可以在全新输出目录一次顺序运行：

```bash
WANDB_ENTITY=3120252125- WANDB_PROJECT=llmrec-2026 \
  scripts/reproduce/i13_highscore.sh all 0
```

若不在登记的集群环境，设置 `LLAMAFACTORY_PYTHON` 和 `LLAMAFACTORY_CLI` 指向自己的 LLaMA-Factory 虚拟环境。两个训练配置位于：

- `configs/active/i13_repro_parent_r64_ep3.yaml`
- `configs/active/i13_repro_residual_r16_retkl_ep1.yaml`

它们只把历史绝对路径、输出名、run name 和覆盖保护改为适合 clone 的版本；模型、数据、loss、优化器、epoch、学习率和随机种子保持历史实现一致。

## 参考产物哈希

| 产物 | SHA256 |
|---|---|
| I-10 E3 r64 父 adapter | `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2` |
| I-12 r16 残差 adapter | `e8caf0a39fee133b2172f2e74a3ef64c53b3d46f2d8dc82acbc821a00b524f98` |
| I-13 r80、scale=0.875 最终 adapter | `71bc3c2c86beb1c1aaafd41f98915ba94a7f964b6e8450079a883aebc32ffd5b` |
| I-13 adapter config | `e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0` |

数据可以逐字节复原；给定完全相同的父/残差 adapter，拼接也是数学与字节确定的。重新训练能否逐字节得到相同权重还受 CUDA、PyTorch、LLaMA-Factory 和硬件版本影响，因此参考权重哈希用于验收，不虚假承诺跨环境 bitwise determinism。

## 合规边界

平台提交物是相对 O6 的单个 r80 LoRA adapter，但它由两个 adapter 的低秩参数拼接而成。赛事 FAQ 对模型融合的口径是“不鼓励”，复赛训练方案审核是否接受这一路线仍是灰区。这里准确发布最高分实现，不把 `0.9978` 说成已经获得组委会书面合规确认。

I-18 `seed_teacher_cotfix_v2` 是另一条尚未线上评测的单 adapter 候选，不是本仓库最高分实现。
