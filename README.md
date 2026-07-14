# LLM-Rec 2026 竞赛工程

基座为 `OneReason-0.8B`。目标是通过 SFT/LoRA 提升 8 个加权子项：懂物料 1 项、懂用户 2 项、懂推荐 4 项、懂世界 1 项。

## 当前状态

- 2026-07-13下午平台修复评测不稳定问题。仓内日志证明协议切换发生在I-10 E3（11:45，旧）与I-11（16:40，新）之间：旧日志action上限4096且itemic只跑1次beam64；新日志action上限1024且itemic执行7次`Race averaged evaluation`。两边虽然都打印`version: v3.1`，仍必须隔离为`platform-pre-fix-v3.1`与`platform-stable-v3.1-20260713`，禁止直接做分数差。
- 旧协议最高单次显示分是`seed_teacher_r64_lr1e4_e3 = 0.9849`；I-10同轨迹E1/E2/E3=`0.9100/0.9680/0.9849`只保留为旧协议内部剂量曲线。E3的固定协议桥仍未建立，不能用旧0.9849直接压过新协议候选；用户已裁定不消耗本轮最后一次配额做重测。
- 固定协议结果为I-11/I-12/I-13/I-14 E3=`0.9618 → 0.9768 → 0.9978 / 0.9518`。I-13相对I-12总分`+0.0210`，action/topic=`-0.0023/-0.0003`，video/prod/ad/live=`+0.0288/-0.0068/0/+0.0009`，world=`+0.0007`；当前固定协议主模型为I-13。I-14按评测时间归入修复后协议，但原始日志指纹尚未复核。
- I-13只把I-12的r16用户残差缩到`0.875`，不重训。线上结果验证了本地Pareto选择的总分方向，但收益来自推荐四项合计`+0.0229`抵消用户两项合计`-0.0026`，不能解释成用户能力提升；E3固定协议桥仍缺失，继续禁止与旧协议0.9849作差。
- I-14从O6干净训练纯`D(O1)`单体r80，E3线上0.9518；它不含teacher、第三方、评测回灌或参数拼接。相对I-13的差值只回答榜分替换问题，不能拿融合灰区模型作纯O1路线的科学基线；更接近的I-11仍有teacher、续训和rank混杂。
- 2026-07-14复读官方赛题解析、HF数据说明和OneReason技术报告后，暂停“相近SFT adapter之间的朴素蒸馏”。I-14推荐CoT/UnCoT=`6,460/12,744`，与官方SFT报告的`29.56万/58.80万`几乎同配比；下一步先审计R1、itemic instruction、通用保持、官方三阶段CoT与RFT/MOPD的结构缺口，不照搬论文比例，也不启动未准入训练。
- `seed_scoremax_r32_ep1` 已完成单卡 1 epoch 训练和结构门禁：action 可见题 0/5 闭合、5/5 触顶，material 单题签名 41/14 未进入历史 8 题档；后验中点约 0.92，本地不建议占用提交次数。
- 90%涨跌判决仍为 `NOT_CERTIFIED`。E1 的冻结输出是 `ABSTAIN`，没有声称错误方向；但本地门禁选 E1、拒 E2，而线上排序相反，证明现有门禁不能可靠选择 checkpoint。协议与台账见 `docs/offline_eval.md` §9。
- O1–O6 官方数据 EDA 已封板；I-07 已验证“仅提高 action 样本/target 占比”仍不能解决长数组终止。Caption/Tag 与 General 均保持研究项，不据此自动启动下一轮训练。

活跃假设、选手分享和失败方案统一见 [`ideas/`](ideas/README.md)。

## 官方资产

唯一台账：[`docs/reference/ASSETS.md`](docs/reference/ASSETS.md)。

| ID | 官方资产 | 固定入口 |
|---|---|---|
| O1 | 平台预制种子 SFT，12 文件、32,480 条 | `assets/official/seed_sft/` |
| O2 | Explorer 17GB 原始五表 | `assets/official/hf_raw/` |
| O3 | 与预制“懂推荐”对齐的 Caption/Tag | `assets/official/sft_aligned/` |
| O4 | `OpenOneRec-General-Pretrain` | `assets/official/general_pretrain/` |
| O5 | `OpenOneRec-General-SFT` | `assets/official/general_sft/` |
| O6 | 竞赛指定 OneReason-0.8B 基座 | `assets/official/base_model/` |

注意：O4/O5 是 OpenOneRec 官方发布。即使 O5 内部汇集多个开源数据源，也仍属于官方资产，不能归为第三方。

## 训练铁律

1. 单卡训练；epoch 数与学习率日程必须由训练轨迹决定，不设统一的 1 epoch 上限。
2. 单点实验只保留最终 adapter；连续多 epoch 轨迹可按 epoch 保存 adapter-only checkpoint，用于选择训练时点。
3. 默认只使用官方资产；`assets/third_party/` 未经明确批准禁止引用。
4. 每次实验只改变一个主要变量，先在 `ideas/README.md` 写预期子项和失败条件。
5. 正式训练必须记录 W&B，训练结束立即登记数据、配置、adapter 哈希和门禁结论。
6. 门禁失败的模型不续训、不 warm start、不占线上配额。

## 快速入口

训练环境：

```bash
source /lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/activate
nvidia-smi
```

获批配置的单卡启动形式：

```bash
WANDB_ENTITY=3120252125- WANDB_PROJECT=llmrec-2026 \
  scripts/train/launch_wandb_online.sh 0 configs/active/<approved_run>.yaml
```

启动器会拒绝任何非 online 模式，并在训练前验证 W&B 登录。正式训练只有在 W&B 服务端显示 `running` 且收到首个指标点后，才登记为“已启动”。训练配置必须使用单卡并设置 `report_to: wandb`；多轮轨迹如按 epoch 保存，必须使用 adapter-only checkpoint 并限制保留数量。

开始工作前的结构门禁：

```bash
scripts/audit_workspace.sh
```

## 目录

```text
assets/       官方、派生、第三方、评测资产的固定入口
data/         兼容入口，不存独立副本
ideas/        活跃 idea、选手/队友分享、EDA 与历史方案
configs/      训练与合并配置
scripts/      数据构造、训练、评测和打包脚本
docs/         平台规则、实验台账、评测规范和工作区说明
models/       官方基座链接
checkpoints/  仅保留当前需要的最终 checkpoint
submissions/  运行卷提交包链接
logs/         训练、门禁和线上评测日志链接
wandb/        W&B 本地运行记录链接
```

## 必读文档

- [`ideas/README.md`](ideas/README.md)：下一步做什么以及为什么。
- [`docs/platform_guide.md`](docs/platform_guide.md)：官方规则与评测机制。
- [`docs/experiment_log.md`](docs/experiment_log.md)：线上分数和实验归因。
- [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md)：当前模型、配置和提交包。
- [`docs/reference/ASSETS.md`](docs/reference/ASSETS.md)：官方数据边界与物理位置。
- [`docs/reference/OFFICIAL_DATA_EDA.md`](docs/reference/OFFICIAL_DATA_EDA.md)：O1–O6 全量 EDA、数据漏洞、可用 trick、禁止路线与复现口径。
- [`docs/offline_eval.md`](docs/offline_eval.md)：离线门禁、历史校准和90%选择性判决协议。

## 当前禁止事项

- 不保存未登记、无训练时点选择用途或包含 optimizer state 的中间 checkpoint。
- 不从失败 checkpoint 继续训练。
- 不把平台可见题或离线评测题回灌训练。
- 不因为数据已下载就自动混入 General 或第三方数据。
- 不使用未经登记的数据文件启动正式训练。
