# Run A 笔记 — 懂用户 R2 增量 SFT (run_a_r2)

> 生成 2026-07-01。Run A = 从 17GB 原始 UserProfile 自造懂用户 R2 action-select 数据,增量混入种子 SFT。
> 目标:把懂用户(v1 = 0.036/0.039,垫底,权重2)拉起来。相关:`docs/eval_diagnosis_v1.md`, `docs/strategy_roadmap.md`。

## 做了什么

**数据构造管线(全脚本化,`scripts/`):**
1. `data/build_item_index.py` — 扫官方 17GB 的 Pid2Sid/Caption/Tag 全部分片,建 pid→sid/caption/tag 全局索引(3591万/2106万/542万 keys),存 `ai_runtime/.../assets/derived/index/`。域映射 video/video→video, video/ad→ad, goods→prod, live→living。
2. `data/build_r2_actionselect.py` — 从 UserProfile 序列造抽取式 action-select 样本:多域行为按时间戳合并成 `【日期】[域-行为] <token>` 历史块(格式与评测逐字对齐,9/9 锚点通过);选一个 tag 焦点生成主题(`从泛化X到聚焦Y`);确定性打分选正样本(`0.60·tag_jaccard + 0.30·同s_a/b + 0.10·同域 + recency`,阈值0.45,top-K 3..40);extractive 强校验(答案 100% ∈ 历史);~12% 空 `[]` 样本。分布对齐种子(历史中位179,答案中位3)。产 20000 训练 + 1000 dev(dev 用 `--shard_offset 8` 避免与 train 用户重叠)。
3. `eval/proxy_r2_f1.py` — 本地 F1 proxy(HF transformers 批量生成),算 F1/复读率/JSON可解析率/子集率/空样本答对率。

**训练:** `configs/history/run_a_r2.yaml`。种子 32480 + 自造 R2 13920(30%)= 46400 混合,单卡(GPU1),配方同 v1(全参 qwen3_nothink cutoff32768 lr2e-5 1ep seed19260817),bs1×accum4。715步,51分钟,**train_loss 1.487**(v1 是 1.573)。

## Proxy 门禁结果(160 dev 样本,与 v1 同 dev 同参数)

| 指标 | v1 | Run A | 判读 |
|---|---|---|---|
| **F1_mean** | 0.0947 | **0.1845** | ✓ 翻倍(+95%) |
| 复读率 | 0.694 | 0.738 | ✗ 未治好(略升) |
| JSON可解析率 | 0.30 | 0.25 | ✗ 略降 |
| 历史子集率 | 0.953 | 0.967 | ✓ 略升(更少幻觉) |
| 空样本答对 | 0/35 | 0/35 | = 都是0,完全不会"停止" |

**proxy 校准基准**(v1 F1 0.095 ↔ 线上 action_select 0.0362,方向一致,已确认 proxy 可信)。

## 诚实结论

- **F1 翻倍是真实进步**:加 R2 数据确实让模型更会"选相关行为",子集率也升。按 proxy 方向,懂用户线上分大概率涨。
- **但复读/空样本没解决**:说明 v2 数据设计有缺陷——extractive 约束能减少幻觉,但**没教会"何时停止"**(答案列几个后仍退化成复读),12% 空样本比例太低。这是 Run A2 要修的。

## 产物 & 上传

- CKPT: `checkpoints/run_a_r2/`(train_loss 1.487)。
- 上传包: `submissions/run_a_r2_platform/`(8 文件 1.62GB,纯 .safetensors/.json,训练权重 + base config/tokenizer verbatim,核心架构13项与base一致校验通过,已从包目录加载+生成自检通过)。含 SHA256SUMS.txt(Mac核对用,勿传平台)。
- **上传**:scp 到 Mac → 万擎「模型仓库→上传模型(全参)→去评测」。

## Run A2 改进方向(拿到 v2 线上分后定)
- 治复读:答案后强制 `<|im_end|>`,训练数据显式教"选完即停";降答案上限;可加 repetition 相关的数据侧信号。
- 空样本:提高空 `[]` 比例(12%→25%+),或单独构造"主题与历史完全无关"的强负例。
- 校准:v2 线上分回来后,建立 proxy F1 ↔ 线上 action_select 的定量映射,后续 run 更准。
