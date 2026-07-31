# 实验记录 I-41 → I-74 + 最终合并(2026-07-24 ~ 07-30)

> 本篇补齐 `main` 上 I-40(2026-07-24)之后到最终提交(2026-07-30)之间的实验记录,即"后半段怎么上分"的完整故事。
>
> **诚实声明**:这一段大多数实验**没有成文详细记录**,本地只有 checkpoint artifact、自动 model card、wandb run 与离线 eval JSON。下表中凡是标 `无线上评测` / `无记录` 的,就是真的没有;**不编造 SHA256 / evalTaskId**。可恢复的字段(方法、parent、rank、wandb run、离线指标)据实填写;不确定处标注 `约`。
>
> 状态码:`LOCAL_REJECTED_NO_SUBMISSION`(本地否决未投)、`LOCAL_EVAL_ONLY_NO_ONLINE`(只离线评测,无线上分)、`STAGED_NO_ONLINE_GAIN`(已 staging 到平台但无线上增益)、`SUBMITTED_ONLINE`(有线上分)。

## 概述

I-35 step548 r112(线上 1.0344,2026-07-22)是单模型时代的天花板。I-37(1.0276)、I-40(0.9891)相继回退关闭后,后半段沿着四条线探索,大多被逐一证伪关闭:**(1) adapter 合并/插值/SVD/tangent/task-vector**(i42–i46、i58、i73、i74);**(2) 强化学习 GRPO/DPO**(i69–i72);**(3) 单维度强化据点**(i41、i55、i57、i63、i66、i67);**(4) 多教师/全参/SGP T 等杂项 SFT**(i47、i48、i50–i54、i56、i61、i62)。最终在 2026-07-30 的**正交残差融合冲刺**里,i35⊕i50 全参正交融合(λ=0.10)拿到 **1.0567 / rank 55/1200**,成为全队最高分。详见末节。

## 1. 合并 / 插值探索组(最终孕育出冠军方案)

这一组系统性地否定"朴素合并 adapter"这条路,但正交残差融合(i73/i74 → 最终 fullparam 版)从中长出。

| 实验 | 方法 | parent | 产物 | 离线 / 线上 | 状态 |
|---|---|---|---|---|---|
| i42 | i35⊕i23 精确插值(interp) r128/r176,4 变体 | i35+i23 | 纯 artifact | 无记录 | LOCAL_REJECTED |
| i43 | i35⊕i37 lowdose r120,4 scale | i35+i37 | 纯 artifact | 无记录 | LOCAL_REJECTED |
| i44 | i35⊕i37 SVD r112 | i35+i37 | 纯 artifact | 无记录 | LOCAL_REJECTED |
| i45 | i35⊕i37 tangent r112 | i35+i37 | 纯 artifact | 无记录 | LOCAL_REJECTED |
| i46 | i35⊕i37 tangent 逐 proj(q/k/v/o/gate/up/down) | i35+i37 | 7 个 artifact | 无记录 | LOCAL_REJECTED |
| i58 | i35⊕i23 minus-i10 task-vector, prefix112 r128 fp32 | i35+i23 | 提交包(7-28) | 无线上分 | LOCAL_EVAL_ONLY |
| i73 | **i35⊕i50 正交残差融合**(LoRA r128,SVD 压缩,严格两文件 ~323MB) | i35(1.0344)+i50(1.0302) | fp32 + merged_bf16 + i73_eval | 离线对照 i35:action F1 Δ≈−0.0003、json_ok net=0(保住 i35 又叠 i50 物料);无线上分 | LOCAL_EVAL_ONLY_NO_ONLINE |
| i74 | i35⊕i23 正交残差融合 r128 | i35+i23(0.9882) | merged_bf16 全参 | 离线对照 i35:action F1 Δ≈−0.284、json_ok net=−279(action 崩) | LOCAL_REJECTED |

**关键转折**:i73 证明"i35 为主干 + 注入 i50 的正交残差"能保住 i35 又叠加 i50 物料;但 LoRA 空间的 r128 SVD 压缩版没投线上。7-30 把同一正交残差思路改为**全参烘焙**(full-weight),并对 i35⊕i50 扫 λ,最终 λ=0.10 拿到 1.0567(见末节)。

## 2. 强化学习组(GRPO / DPO)

| 实验 | 方法 | 产物 | 离线 | 状态 |
|---|---|---|---|---|
| i69 | rec_grpo,GRPO RL r16 | checkpoint 100/200/final | 无 README | LOCAL_EVAL_ONLY |
| i70 | video_grpo,GRPO RL r16 | checkpoint 50–600 + i70_eval | 有离线 eval 目录 | LOCAL_EVAL_ONLY_NO_ONLINE |
| i71 | copy_dpo,DPO r16 | checkpoint 500–3500 + i71_eval | 有离线 eval 目录 | LOCAL_EVAL_ONLY_NO_ONLINE |
| i72 | top_route_composition,LoRA r128 + video_grpo_full | checkpoint | 仅 model card | LOCAL_EVAL_ONLY |

RL 方向无论成败均无线上提交;离线 eval 未换算出可超过 I-35 的线上分。

## 3. 单维度强化据点组

每条都试图在 I-35 上单攻一个子项,均未突破 1.0344。

| 实验 | 方向 | rank | wandb | 离线 | 状态 |
|---|---|---|---|---|---|
| i41 | adonly_future_retkl | r8 | zid8re1y | 无 | LOCAL_REJECTED_NO_SUBMISSION |
| i55 | mathplus_worldformat | r16 | — | 无 | LOCAL_EVAL_ONLY |
| i57 | actionformat_retkl | r16 | — | 无 | LOCAL_EVAL_ONLY |
| i63 | adonly_stronghold(懂推荐-ad) | r8 | 7zyrl1m9 | i63_eval | LOCAL_EVAL_ONLY_NO_ONLINE |
| i66 | world_stronghold(懂世界) | r8 | 0fuhf132 | i66_eval | LOCAL_EVAL_ONLY_NO_ONLINE |
| i67 | seed_resample | r16 | 4tt47h77 | i67_eval | LOCAL_EVAL_ONLY_NO_ONLINE |

## 4. 多教师 / 全参 / SGD / 杂项 SFT

| 实验 | 方法 | rank | 状态 |
|---|---|---|---|
| i47 | i35+i37 selective_teacher | r8 | LOCAL_EVAL_ONLY |
| i48 | i35+i37 down_selective | r8 | LOCAL_EVAL_ONLY |
| i49 | i35_i23_minus_i10 taskvec | — | LOCAL_EVAL_ONLY |
| i50 | **i23 multiteacher_materialnull**(多教师懂物料) | r128 | **SUBMITTED_ONLINE=1.0302**(冠军方案的 model B) |
| i51 | material_adlive_multiteacher | r128 | LOCAL_EVAL_ONLY |
| i52 | fullparam SFT(全参) | — | LOCAL_EVAL_ONLY |
| i53 | fable_multigold | r128 | LOCAL_EVAL_ONLY |
| i54 | joint_action_video_math | r128 | LOCAL_EVAL_ONLY |
| i56 | step548_selfbeam | r16 | LOCAL_EVAL_ONLY |
| i61 | material_gradnull(SGD) | r16 | LOCAL_EVAL_ONLY |
| i62 | user_math_shuffle | r16 | LOCAL_EVAL_ONLY |

> i50 是冠军方案的 model B(线上 1.0302,多教师懂物料 r128)。其 **训练脚本不在本发布内**(见 `SOLUTION.md` 缺口),adapter config 可获取。

## 5. i68 — 最后一个单训模型(math no-think CoT)

- **方法**:在 I35 parent(r112 已合并进 base)上训 r8 残差,用"math no-think CoT"蒸馏攻 material/topic,route-all-retention-KL 保 I35 的 video-boundary retention。两阶段:(1) 冻结 I35 parent 用 vLLM+LoRA 生成数学教师候选(先 1024 后 1664 条);(2) 在 1792 行数据上 LoRA r8/alpha8,lr 5e-6 cosine,448 steps。
- **训练结果**:train_runtime 891.7s,train_loss 0.284。
- **失败首启**:`failed/i68_route_all_retention_wgijpp2f` 在 step~75 被外部抢占(无 traceback/OOM,仅留 checkpoint-64),后以 `i68_i35_math_nothink_cot_r8_v1` 重跑成功。
- **提交包**:`submissions/i68_i35_math_nothink_cot_r120_v1_platform/`(r120/alpha120,302MB,2026-07-29 04:37)。
- **离线 eval**:`logs/offline_eval/i68_vs_i35_bounded_pair_comparison_v1.json`,相对 I35 **中性偏负**(action F1 −0.0053、rec video −0.1094 / ad −0.0703 两大域退步,live/prod 小升)。
- **状态**:`STAGED_NO_ONLINE_GAIN`(staged 但无线上分),关闭为非胜出实验。

## 6. 最终合并冲刺(2026-07-30)— 冠军 1.0567 / rank 55/1200

7-30 全天围绕**全参正交残差融合**做 i35 ⊕ {i50 / i23 / i37 / i51 / i57} 的 λ 扫描,产出 ~12 个 fullparam 提交包(`artifacts/submissions/`)。其中 **i35⊕i50 在 λ=0.10 取得线上最高 1.0567**。

### 冠军:i35_i50_orthfuse_l010_fullparam_bf16

| 项 | 记录 |
|---|---|
| 实验 | i35 ⊕ i50 全参正交残差融合,λ=0.10 |
| 公式 | `W = W_i35 + 0.10·(ΔB − (⟨ΔB,ΔA⟩/‖ΔA‖²)·ΔA)`,Δ = W_merged − W_base,7 个 proj 目标;非目标权重保留 i35 |
| model A | i35_r96_video_boundary_retkl_r112_step548(线上 1.0344) |
| model B | i50_i23_multiteacher_materialnull_r128_v1(线上 1.0302) |
| 产物 | BF16 全参模型,`model.safetensors` = 1,602,902,832 bytes(~1.6GB),含 tokenizer |
| 构建时间 | 2026-07-30 08:03;提交 2026-07-30 16:50 UTC |
| 线上结果 | **总分 1.0567,rank 55/1200(队伍 CornerCase)** |
| 相对 i35 | +0.0223(单模型天花板 1.0344 → 1.0567) |
| 代码 | `scripts/train/full_weight_orthogonal_fuse.py` |

### 同冲刺其它(均为 i35 正交融合,扫不同 model-B / λ)

`i35_i50_orthfuse_l005/l007/l015/l020`、`i35_i23_orthfuse_l025`、`i35_i37_orthfuse_l050`、`i35_i51_orthfuse_l010`、`i35_i57_orthfuse_l050`、`multi_i50_i37_i57_l010`。除冠军 l010 外,其余未取得高于 1.0567 的线上分(具体逐包线上分未全部入账,不编造)。

> 离线协议 `offline-eval-v4` 与线上榜分不是同一量纲,离线 sa_pass@64(0.73–0.78 区间)不能直接换算线上总分;λ=0.10 的线上胜出是平台实测,非离线外推。

---

**结论**:后半段 I-41→I-74 是一条"单维度强化与朴素合并接连证伪 → 正交残差融合在 i35⊕i50 上突破"的路径。最终 1.0567 / rank 55/1200 来自 **i35(懂推荐)⊕ i50(懂物料)的全参正交残差融合 λ=0.10**,超越单模型天花板 I-35 的 1.0344。
