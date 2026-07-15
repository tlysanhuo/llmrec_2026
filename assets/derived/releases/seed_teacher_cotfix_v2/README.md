# I-18 未评测候选数据发布：`seed_teacher_cotfix_v2`

这是 I-18 未评测候选的完整训练数据发布件，**不是仓库最高分实现**；当前最高固定协议分是 I-13 的 `0.9978`。本数据是 **官方源派生资产** `D(O1,O2,O3)`，不是官方直发或官方原始数据；不含第三方数据、评测回灌、失败 checkpoint 输出或目标物料元数据。

发布件把实际训练输入纳入普通 Git，队友无需重新调用生成器或 judge 即可获得逐字节一致的 32,644 行 JSONL。唯一权威的资产分类与位置仍是 [`docs/reference/ASSETS.md`](../../../../docs/reference/ASSETS.md)。

## 文件

| 文件 | 作用 |
|---|---|
| `data_seed_teacher_cotfix_v2.jsonl.gz` | 完整训练数据的确定性 gzip；52,199,218 bytes；SHA256 `193cd78f...07f9` |
| `manifest.json` | 上游、行数、内容哈希、混合比例、不变量和训练入口 |
| `audits/` | 原始 prepare/generation 摘要与最终 build audit，内容哈希与运行卷原件一致 |
| `scripts/data/restore_seed_teacher_cotfix_v2.py` | 同时校验压缩包和解压内容，并原子还原 JSONL |
| `scripts/data/build_cotfix_v2.py` | 从已登记上游重新执行 prepare/generate/build 的完整构建器 |
| `configs/datasets/seed_teacher_cotfix_v2/dataset_info.json` | LLaMA-Factory 数据注册，使用仓库内相对路径 |
| `configs/active/seed_teacher_cotfix_v2_r64_lr1e4_ep3.yaml` | I-18 的 r64、`1e-4`、3 epoch 单卡 W&B 配方 |

## clone 后验证与还原

在仓库根目录执行：

```bash
python3 scripts/data/restore_seed_teacher_cotfix_v2.py --verify-only
python3 scripts/data/restore_seed_teacher_cotfix_v2.py
```

第一条命令不写文件，会校验压缩包 SHA256，并流式解压校验原始 JSONL 的字节数、32,644 行和 SHA256。第二条命令把数据原子还原为：

```text
assets/derived/releases/seed_teacher_cotfix_v2/data_seed_teacher_cotfix_v2.jsonl
```

该 `.jsonl` 受仓库的通用数据忽略规则保护，不会被误重复提交；训练配置的数据注册已直接指向这个位置。如果目标文件已经存在且哈希正确，恢复脚本会返回 `already_present`；内容不一致时默认拒绝覆盖，只有明确使用 `--force` 才替换。

## 复现实验

先按 `docs/reference/ASSETS.md` 准备只读的 O6 `OneReason-0.8B` 基座，使其可从 `models/OneReason-0.8B-pretrain-competition` 访问；再准备仓库约定的 LLaMA-Factory 环境和 W&B 登录。正式配置已经冻结为历史成功配置，不要覆盖已有 checkpoint：

```bash
WANDB_ENTITY=3120252125- WANDB_PROJECT=llmrec-2026 \
  scripts/train/launch_wandb_online.sh 0 \
  configs/active/seed_teacher_cotfix_v2_r64_lr1e4_ep3.yaml
```

训练配置在成功后只把三处运行路径改为仓库相对路径；超参未改。原始启动配置 SHA256 `2bd234bd...42da` 和当前可移植配置 SHA256 都记录在 `manifest.json` 与实验台账中。若复跑，请使用新的 `output_dir`/`run_name`，不要覆盖 I-18 已保留产物。

## 从上游重建与直接还原的区别

- 日常协作或复现实验：使用已提交的完整 gzip 和恢复脚本，不需要 API 凭据。
- 审计构造逻辑：运行 `python3 scripts/data/build_cotfix_v2.py --help` 查看 `prepare`、`generate`、`build` 三阶段；这条路径需要台账内的 O1/O2/O3 上游、外部模型接口及原生成审计，不能把重新采样结果假定为与本发布逐字节相同。
- 压缩包之外的大型中间 request/judge 逐行日志不重复塞入 Git；三份小型原始审计摘要已随发布提交，其余日志的内容哈希、模型身份和验收计数写入 `manifest.json`，最终训练输入本身完整提交。

线上是否涨分仍需 I-18 E3 的固定协议评测确认；本发布保证的是实现、数据与配方可核验，不把本地结构门禁描述成线上成绩。
