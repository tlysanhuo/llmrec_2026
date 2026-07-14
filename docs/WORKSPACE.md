# Workspace Contract

项目根只保存代码、配置、文档和稳定链接；数据、环境、模型和提交包实体位于运行卷：

`/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026`

## 项目根

| 路径 | 职责 |
|---|---|
| `README.md` | 一分钟了解赛题、当前状态和下一步 |
| `ideas/` | 活跃 idea、选手/队友分享、EDA、论文仓库入口 |
| `docs/` | 平台规则、实验台账、评测协议、资产台账 |
| `configs/` | 训练、合并和数据注册配置 |
| `scripts/` | 数据构造、训练、评测、打包工具 |
| `assets/` | 数据资产权威分类链接 |
| `data/` | 兼容链接，不保存独立数据 |
| `models/checkpoints/submissions/logs/wandb/` | 运行卷链接 |

项目根禁止直接放置 JSONL、Parquet、压缩数据、模型权重和临时日志。

## 运行卷

```text
data/
  official/      官方 O1-O5 数据
  derived/       我方构造数据与索引
  third_party/   选手和队友数据
  evaluation/    离线评测与可见题
  archive/       历史探查文件
artifacts/
  submissions/   平台提交包
checkpoints/      只保留当前需要的最终 adapter
models/           官方基座
LLaMA-Factory/    训练框架与训练 venv
logs/             train/data/probe/precheck/eval 日志
wandb/            W&B 本地记录
references/       外部论文代码仓库
```

## 写入规则

- 官方数据只读，不在 `official/` 内写缓存或派生结果。
- 派生数据只写 `data/derived/`，并在 `ASSETS.md` 登记来源。
- checkpoint 只保存最终 adapter；失败实验完成门禁后删除。
- 提交包写 `artifacts/submissions/<run>_platform/`。
- 凭据只能放忽略版本控制的 secrets 文件，权限 0600。
- 任何目录结构变化同步更新 `README.md`、本文件和 `ASSETS.md`。
