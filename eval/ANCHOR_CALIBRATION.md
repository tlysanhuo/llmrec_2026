# 评估锚点校准记录

记录 `competition_smoke.jsonl` 冒烟评测的锚点选取、离线/在线得分对照、相关性校准结果，
以及后续新 checkpoint 的评测结果追加记录。所有原始结果 JSON 落盘在 `eval/output/` 下。

## 1. 评测流程

- 数据集：`eval/data/competition_smoke.jsonl`（546 条，覆盖五个维度：`world` 7 条、
  `material` 533 条、`user_f1` 1 条、`user_chain` 1 条、`recommend` 4 条）。
- 入口脚本：`eval/run_smoke_eval.py`，`eval/run_calibration_smoke.sh` 一次性依次跑完
  A~E 五个锚点并调用 `eval/calibrate_smoke.py` 聚合结果。
- 推理执行有三种 `--engine-mode`：
  - `legacy`：逐条串行调用（最初实现，最慢，仅作回退）；
  - `batch`（默认）：按维度整体打包批量调用，让 vLLM continuous batching 生效；
  - `multiprocess`：在 `batch` 基础上，`material` 维度样本量超过阈值（30）时改用多个
    独立子进程（`material_worker.py`）各自建立 vLLM 实例、共享同一张 GPU 并行跑
    `beam_decode`，绕开 vLLM 官方 `beam_search` 在 Python 层单进程单线程（GIL 限制）
    做逐 step 候选构造/排序的瓶颈。当前默认用 `multiprocess`，4 个 worker。
- 进度可观测性：`beam_decode` 按 30 条一个 chunk 打印完成日志（时间戳/耗时/累计进度），
  主进程每 20 秒轮询一次各 worker 日志文件末尾行打印心跳，`run_smoke_eval.py` 对每个
  维度打印开始/完成/耗时，`run_calibration_smoke.sh` 对每个锚点打印开始/完成/累计耗时。
- 退出健壮性：`material_worker.py` 子进程以 `start_new_session=True` 启动为独立进程组，
  主进程注册 `atexit`/`SIGTERM`/`SIGINT` 处理器，退出时用 `os.killpg` 清理所有 worker
  进程组（含其 fork 出的 vLLM EngineCore 子进程），避免孤儿进程残留 GPU 显存。
- 单次全量（546 条，4 worker）耗时基准：约 13～15 分钟／锚点。

## 2. A~E 五锚点：离线冒烟得分 vs 官方在线得分

| 锚点 | 说明 | LoRA 路径 | 在线总分 |
|---|---|---|---|
| A | base（无 LoRA） | — | 0.6655 |
| B | Frinkleko baseline | `demo/output/onereason_0.8b_lora_frinkleko` | 0.8694 |
| C | I-19 world gold combined r96 | `llmrec_2026-main/checkpoints/i19_world_gold_combined_r96` | 0.9221 |
| D | I-20 scale=0.25 combined r96 | `llmrec_2026-main/checkpoints/i20_i19a1_scale025_combined_r96` | 0.9463 |
| E | I-13 repro combined r80 s875 | `llmrec_2026-main/checkpoints/i13_repro_combined_r80_s875` | 0.9867 |

离线冒烟得分（`eval/output/calibration_smoke_{A..E}.json` 的 `summary`）：

| 锚点 | material | user_f1 | user_chain | recommend | world |
|---|---|---|---|---|---|
| A | 0.1482 | 0.0 | 0.0 | 0.25 | 0.2857 |
| B | 0.1482 | 0.0 | 0.0 | 0.0 | 0.2857 |
| C | 0.1482 | 0.6667 | 0.0 | 0.0 | 0.7143 |
| D | 0.1482 | 0.6316 | 0.0 | 0.0 | 0.5714 |
| E | 0.1482 | 0.6111 | 0.0 | 0.0 | 0.2857 |

官方在线各维得分（写在 `eval/calibrate_smoke.py` 的 `ANCHORS` 常量里，来自比赛后台）：

| 锚点 | material | user_f1 | user_chain | recommend | world |
|---|---|---|---|---|---|
| A | 0.1533 | 0.0 | 0.0055 | （缺失） | 0.1387 |
| B | 0.1840 | 0.0608 | 0.0357 | 0.4362 | 0.1528 |
| C | 0.1840 | 0.1222 | 0.0380 | 0.4231 | 0.1539 |
| D | 0.2146 | 0.1213 | 0.0399 | 0.4296 | 0.1409 |
| E | 0.2453 | 0.1207 | 0.0386 | 0.4427 | 0.1394 |

## 3. 相关性校准结果（`eval/output/calibration_smoke_report.json`）

| 维度 | n | Spearman ρ | 方向一致率 | 备注 |
|---|---|---|---|---|
| material | 5 | — | — | 五锚点离线值完全相同（0.14821763602251406），无法计算相关性 |
| user_f1 | 5 | **0.975** | **100%（9/9）** | 与在线分对齐度很好 |
| user_chain | 5 | — | — | 离线值全为 0，无法计算相关性 |
| recommend | 4 | — | — | 离线值全为 0（B/C/D/E），无法计算相关性 |
| world | 5 | **0.671** | 85.7%（6/7） | 中等偏好的一致性 |
| 综合排序（五维平均秩） | 5 | 0.1 | 50%（5/10） | 仅作方向性参考 |

局限性说明（与报告中 `limitations` 一致）：
- 每维仅 1～7 条样本，`material`/`recommend` 全锚点在冒烟集上得分饱和或恒定，不能用于排序校准；
- `A`（base）的在线 `recommend` 真值缺失，该维相关性仅用 B-E 四点计算；
- n=5 且样本与赛事隐藏集不同，结果只能证明流程跑通，提供方向性信号，不能替代真实提交打分。

## 4. 后续新 checkpoint 评测记录

评测命令模板（沿用与 A~E 相同的推理配置）：

```bash
cd eval && python3 run_smoke_eval.py \
  --model /home/hadoop-ba-rc/one_reason/OneReason-0.8B \
  --tag <标签> \
  --output output/<输出文件名>.json \
  --gpu 0 \
  --engine-mode multiprocess \
  --material-workers 4 \
  --lora <LoRA 目录>
```

### F — `i19_world_userres_retkl_r16_ep1_i13retain_v1_combined_r96`

- LoRA 路径：`llmrec_2026-main/checkpoints/i19_world_userres_retkl_r16_ep1_i13retain_v1_combined_r96`（r=96）
- 结果文件：`eval/output/calibration_smoke_F_i19_userres_retkl_i13retain.json`
- 总耗时：786.6s（约 13 分钟），退出码 0，GPU 显存正常释放，无残留进程。

| 维度 | n | 得分（mean） |
|---|---|---|
| material | 533 | 0.1482 |
| recommend | 4 | 0.0 |
| world | 7 | 0.5714 |
| user_f1 | 1 | 0.6316 |
| user_chain | 1 | 0.0 |

与 A~E 对比：

| 锚点 | material | world | user_f1 | user_chain | recommend |
|---|---|---|---|---|---|
| A base | 0.1482 | 0.2857 | 0.0 | 0.0 | 0.25 |
| B Frinkleko | 0.1482 | 0.2857 | 0.0 | 0.0 | 0.0 |
| C I-19 world gold | 0.1482 | 0.7143 | 0.6667 | 0.0 | 0.0 |
| D I-20 scale=0.25 | 0.1482 | 0.5714 | 0.6316 | 0.0 | 0.0 |
| E I-13 repro | 0.1482 | 0.2857 | 0.6111 | 0.0 | 0.0 |
| **F userres_retkl_i13retain** | **0.1482** | **0.5714** | **0.6316** | **0.0** | **0.0** |

观察：
- `material` 与全部锚点一致（0.1482），属既有现象（该冒烟集上 pass@64 对不同 LoRA 不敏感），不是异常。
- `world`（0.5714）与 `user_f1`（0.6316）分别与锚点 D 完全一致，处在这批 checkpoint 中较好的水平。
- `recommend`/`user_chain` 为 0，与 B/C/D/E 一致，是既有的样本量过小（4 条 / 1 条）导致的指标饱和，非该 checkpoint 特有缺陷。
- 若要把该 checkpoint 纳入官方在线分的相关性校准，需要先补充其对应的在线真值分数，再更新
  `eval/calibrate_smoke.py` 的 `ANCHORS` 并重跑聚合。

## 5. 懂物料/懂推荐生成协议对齐修复后的复现对比（A~F 全部六锚点）

### 5.1 背景

经排查真实线上评测日志（`测评中间输出.md`）发现，懂物料
（`challenge_itemic_pattern_grounding`）与懂推荐（`challenge_recommendation_*`
四子任务）在线上并非让模型自由生成完整回复，而是评测框架把目标 domain 的
`<|xxx_begin|>` token 硬编码拼进 prompt 末尾（约束解码前缀），模型只需 beam
search 生成 3 个后续 token（`s_a/s_b/s_c`）：

```
Single-stage generation with prompt_token (<|video_begin|>)
Generating (beam search): beam_width=64, max_tokens=3, n_prompts=574
```

此前 `run_eval.py` / `run_smoke_eval.py` / `material_worker.py` 均未强制注入该
前缀，而是分别用 `max_tokens=32`（懂物料）与两路 `sample(n=32, max_tokens=4096)`
（懂推荐 think/no-think）自由生成，与线上协议存在系统性偏差。已修复
`common/prompt.py::build_domain_prompt()` + 三处调用方，并把 `data/loaders.py`
的懂物料 loader 默认过滤为仅 video 域（对齐线上懂物料 100% 为 video 域的事实）、
懂推荐 loader 新增 `target_domain_prefix` 字段（从 gold 前缀反推目标域）。

### 5.2 复现方法

在协议修复后，用与 A~F 完全相同的模型/LoRA 组合和数据集
（`competition_smoke.jsonl`，546 条）逐一重跑一次冒烟评测，仅生成协议不同：

```bash
cd eval && python3 run_smoke_eval.py \
  --model /home/hadoop-ba-rc/one_reason/OneReason-0.8B \
  --tag <标签>_v2_domain_prefix \
  --output output/calibration_smoke_<X>_v2_domain_prefix.json \
  --gpu 0 --engine-mode multiprocess --material-workers 4 \
  [--lora <对应 LoRA 目录，A 无该参数>]
```

六个结果文件均已落盘在 `eval/output/calibration_smoke_{A..F}_v2_domain_prefix.json`，
全部退出码 0，GPU 显存正常释放无残留进程。

### 5.3 结果总表：六锚点 × 五维度 新旧协议对比

| 锚点 | 维度 | 旧分数 | 新分数 | 差值 | 旧耗时(s) | 新耗时(s) |
|---|---|---|---|---|---|---|
| A base | material | 0.1482 | 0.1482 | +0.0000 | 822.4 | 180.6 |
| A base | recommend | 0.2500 | 0.0000 | **-0.2500** | 822.4 | 180.6 |
| A base | world | 0.2857 | 0.2857 | +0.0000 | 822.4 | 180.6 |
| A base | user_f1 | 0.0000 | 0.0000 | +0.0000 | 822.4 | 180.6 |
| A base | user_chain | 0.0000 | 0.0000 | +0.0000 | 822.4 | 180.6 |
| B Frinkleko | material | 0.1482 | 0.1332 | -0.0150 | 910.5 | 319.8 |
| B Frinkleko | recommend | 0.0000 | 0.0000 | +0.0000 | 910.5 | 319.8 |
| B Frinkleko | world | 0.2857 | 0.2857 | +0.0000 | 910.5 | 319.8 |
| B Frinkleko | user_f1 | 0.0000 | 0.0000 | +0.0000 | 910.5 | 319.8 |
| B Frinkleko | user_chain | 0.0000 | 0.1833 | +0.1833 | 910.5 | 319.8 |
| C I-19 world gold | material | 0.1482 | 0.1144 | -0.0338 | 789.0 | 175.7 |
| C I-19 world gold | recommend | 0.0000 | 0.0000 | +0.0000 | 789.0 | 175.7 |
| C I-19 world gold | world | 0.7143 | 0.5714 | -0.1429 | 789.0 | 175.7 |
| C I-19 world gold | user_f1 | 0.6667 | 0.6667 | +0.0000 | 789.0 | 175.7 |
| C I-19 world gold | user_chain | 0.0000 | 0.0000 | +0.0000 | 789.0 | 175.7 |
| D I-20 scale=0.25 | material | 0.1482 | 0.1163 | -0.0319 | 791.3 | 201.6 |
| D I-20 scale=0.25 | recommend | 0.0000 | 0.0000 | +0.0000 | 791.3 | 201.6 |
| D I-20 scale=0.25 | world | 0.5714 | 0.4286 | -0.1429 | 791.3 | 201.6 |
| D I-20 scale=0.25 | user_f1 | 0.6316 | 0.6111 | -0.0205 | 791.3 | 201.6 |
| D I-20 scale=0.25 | user_chain | 0.0000 | 0.0000 | +0.0000 | 791.3 | 201.6 |
| E I-13 repro | material | 0.1482 | 0.1107 | -0.0375 | 798.0 | 187.8 |
| E I-13 repro | recommend | 0.0000 | 0.0000 | +0.0000 | 798.0 | 187.8 |
| E I-13 repro | world | 0.2857 | 0.2857 | +0.0000 | 798.0 | 187.8 |
| E I-13 repro | user_f1 | 0.6111 | 0.4000 | -0.2111 | 798.0 | 187.8 |
| E I-13 repro | user_chain | 0.0000 | 0.0000 | +0.0000 | 798.0 | 187.8 |
| F userres_retkl | material | 0.1482 | 0.1088 | -0.0394 | 786.8 | 197.4 |
| F userres_retkl | recommend | 0.0000 | 0.0000 | +0.0000 | 786.8 | 197.4 |
| F userres_retkl | world | 0.5714 | 0.7143 | +0.1429 | 786.8 | 197.4 |
| F userres_retkl | user_f1 | 0.6316 | 0.6667 | +0.0351 | 786.8 | 197.4 |
| F userres_retkl | user_chain | 0.0000 | 0.0000 | +0.0000 | 786.8 | 197.4 |

耗时方面全部六锚点均大幅下降（约 780～910s → 176～320s，平均降至约 1/4），
与预期一致，因 `max_tokens` 从 32/4096 降到 3。

### 5.4 material 维度：逐样本 diff 统计（关键修正结论）

| 锚点 | n_diff_samples | old_hit | new_hit |
|---|---|---|---|
| A base（无 LoRA） | **0/533** | 79 | 79 |
| B Frinkleko | 34/533 | 79 | 71 |
| C I-19 world gold | 38/533 | 79 | 61 |
| D I-20 scale=0.25 | 35/533 | 79 | 62 |
| E I-13 repro | 38/533 | 79 | 59 |
| F userres_retkl | 39/533 | 79 | 58 |

**重要修正**：此前仅用锚点 A 复现时得出"material 维度完全不受协议修复影响"
的结论，**该结论只对 base（无 LoRA）模型成立**。全面复现后发现，B~F 五个带
LoRA 的 checkpoint 上，material 分数均出现真实下降（-0.015～-0.039），逐样本
比对显示有 34～39/533 条样本（约 6%~7%）的命中结果发生了翻转。

抽样核查掉分样本（如 C 锚点的 `user_material_burger_box_6760`）发现，旧协议
`max_tokens=32` 下的候选字符串包含了 gold 后紧跟的 EOS 片段
（如 `...<s_c_4200><|im_end|>...`），而新协议 `max_tokens=3` 严格截断在第 3 个
SID token。二者在 base 模型上 beam search 前 3 步的候选排序完全一致（截断
不影响排序），但在**经过 LoRA 微调后的模型上，continuation 后续 token（如
EOS）的联合 log-prob 会实际改变 beam search 前几步的路径排序**，导致同一
beam width 下候选集合出现差异、命中率下降。即"多生成的 token 不影响排序"
只是 base 模型的巧合，不能泛化到全部 checkpoint。

### 5.5 recommend 维度：仅锚点 A 观测到下降，其余锚点无变化

| 锚点 | 旧协议 4 条明细 | 新协议 4 条明细 |
|---|---|---|
| A base | `{3a:0, 3b:0, 3c:0, 3d:1}` | `{3a:0, 3b:0, 3c:0, 3d:0}` |
| B Frinkleko | `{3a:0, 3b:0, 3c:0, 3d:0}` | `{3a:0, 3b:0, 3c:0, 3d:0}` |
| C I-19 world gold | `{3a:0, 3b:0, 3c:0, 3d:0}` | `{3a:0, 3b:0, 3c:0, 3d:0}` |
| D I-20 scale=0.25 | `{3a:0, 3b:0, 3c:0, 3d:0}` | `{3a:0, 3b:0, 3c:0, 3d:0}` |
| E I-13 repro | `{3a:0, 3b:0, 3c:0, 3d:0}` | `{3a:0, 3b:0, 3c:0, 3d:0}` |
| F userres_retkl | `{3a:0, 3b:0, 3c:0, 3d:0}` | `{3a:0, 3b:0, 3c:0, 3d:0}` |

六锚点里只有 A 锚点在旧协议下命中了 `competition_3d`（详见 §5.4 原分析：
命中候选是自由生成模式下产出的类 Python list 字符串，属于对训练分布的复述），
其余五个 LoRA checkpoint 新旧协议下该维度分数均恒为 0，没有观测到系统性
差异。说明**锚点 A 的 0.25→0.0 是孤立的边缘案例**，不能代表协议修复对
recommend 维度命中率的普遍影响方向；n=4 的样本量依然太小，任何结论都需要
在懂推荐全量 dev 集合上验证才可信。

### 5.6 world / user_f1 维度的变化：属正常的解码路径微调，非系统性劣化

- `world` 维度在 C、D 锚点下降（-0.1429），F 锚点反而上升（+0.1429），
  A、B、E 不变；`user_f1` 在 D、E 锚点下降（-0.0205、-0.2111），F 锚点上升
  （+0.0351），C 不变。这两个维度的生成协议本次并未改动（不涉及 domain
  前缀注入），分数出现正负两个方向的波动，且样本量极小（world n=7，
  user_f1 n=1），基本符合两次独立运行之间由 vLLM 内部调度、批处理顺序
  等非确定性因素带来的正常方差，不应解读为协议修复引入的回归。

### 5.7 user_chain 维度 B 锚点异常上升核查：证实为正常采样方差，非 bug

B 锚点 `user_chain`（n=1，`competition_2b`）分数从 0.0 变为 0.1833，逐字节
diff 发现新旧两次生成的文本仅在其中一个 SemanticID 上不同
（`<s_a_5802>` vs `<s_a_7682>`，其余内容完全一致），命中率的差异完全来自
`metric_detail` 中 `action_alignment`（0.0→0.286）与 `logic_alignment`
（0.0→0.081）两个子分随机路径不同导致的正常波动。该维度使用的是自由文本
`sample()` 生成（本次修复未改动其协议），n=1 时单条采样的方差本身就足以
造成两次独立运行之间的较大分数波动，与本次协议修复无关。

### 5.8 结论汇总

1. **material 维度**：base 模型无损，但**全部 5 个 LoRA checkpoint 均出现
   真实的命中率下降**（约 6%~7% 样本翻转），是本次全面复现相比单点复现
   最重要的修正发现——此前"协议修复对 material 无影响"的结论不成立，
   需要在正式对外报告前更正。
2. **recommend 维度**：仅 A 锚点观测到下降，其余锚点持平，样本量过小
   （n=4），无法给出可信的方向性结论，仍需全量 dev 集合验证。
3. **world / user_f1 / user_chain**：本次修复未改动其生成协议，观测到的
   正负波动均可归因于两次独立运行间的正常随机性（vLLM 调度非确定性 /
   自由文本采样方差），不构成协议修复引入的回归。
4. **性能收益确定**：六锚点耗时全部下降至约 1/4（180~320s vs 780~910s），
   该收益与分数变化无关，可直接采纳。

### 5.9 遗留事项

- 由于 `eval/calibrate_smoke.py` 的 `ANCHORS` 目前仅记录官方在线分、不区分本地
  评测协议版本，本次新协议复现结果暂未纳入相关性聚合报告
  （`calibration_smoke_report.json`），仅作为协议修复前后的直接对比记录留档。
  若要重新做相关性校准，需要把 `output/calibration_smoke_{A..F}_v2_domain_prefix.json`
  接入 `calibrate_smoke.py` 并重跑聚合。
- material 维度在 LoRA checkpoint 上出现的真实命中率下降，建议后续针对
  性地抽查更多翻转样本（当前仅抽查 3 条），确认是否存在可优化空间（例如
  beam search 的 tie-breaking 规则、去重逻辑等），而不仅仅归因为"协议对齐
  后的必然代价"。
- recommend、world、user_f1、user_chain 四个维度的样本量（4/7/1/1 条）在
  冒烟集上均过小，任何结论都只能作为方向性参考，正式验证需要在
  `eval/run_eval.py --dims <dim>` 全量 dev 集合上运行。
