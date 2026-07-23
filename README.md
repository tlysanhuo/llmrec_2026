# LLM-Rec 2026 竞赛工程

基座为 `OneReason-0.8B`。目标是通过 SFT/LoRA 提升 8 个加权子项：懂物料 1 项、懂用户 2 项、懂推荐 4 项、懂世界 1 项。

> 变更记录（2026-07-23 UTC）：I-37 strict-future r120正式评测`SUCCEEDED`，evalTaskId `eval-task-0yco4c-1784766273`、modelId `md-z9m20x-1784766072216356022`，总分`1.02762520217381`，八项=`0.2452961672/0.1204379107/0.0394833175/0.0768/0.1292/0.1484/0.1089/0.1591078067`。相对I-35 step548总分`-0.0068033827`：material/world不变，用户合计`+0.0013966173`，推荐合计`-0.0082`；ad/live增益未抵消video/product回退。I-37分支关闭，不追加checkpoint或scale；I-35 step548继续作为当前最高与默认交付模型。

> 变更记录（2026-07-22 UTC）：I-36训练与双点打包完成。有效W&B run `mmenbci2`正常完成4,125/4,125步，runtime `7921.5533s`、train loss `1.0296107737`、路由retention/action/topic=`11000/4000/1500`；step2063与step4125均已精确拼成平台合法r128，两包严格两文件、392 tensors、323,015,596 bytes。第一次run `onqds9a5`在step18被外部SIGKILL且无checkpoint，禁止resume；step4125已线上否决，step2063仅保留审计，不再等待I-36上传。
> 变更记录（2026-07-22 UTC）：I-36 step4125线上失败，任务`i36_i35_user_expand_retkl_r128_step4125_V1_eval_20260722183137`总分`0.9865`，八项=`0.2453/0.1070/0.0331/0.0672/0.1292/0.1414/0.1035/0.1599`。相对I-35父模型约`-0.0479`，主要回退在用户和推荐；I-36关闭，step2063仅保留审计，不盲投。

> 变更记录（2026-07-22 UTC）：I-35的step548 r112正式评测成功，evalTaskId `eval-task-9nepj1-1784698215`，总分`1.0344285849069457`，八项=`0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591`，仍为当前84条StreamLake评测中的最高分。相对直接父模型原始`1.0252594563571054`总分`+0.009169129`：material不变，用户两项合计`-0.003815630`，推荐四项合计`+0.0141`，world `-0.001115242`。实际首投为step548；step411未评测且已停止追加I-35扫点。

> 变更记录（2026-07-22 UTC）：I-35 step411/step548已完成同seed离线成对对照，详见[`docs/I35_STEP411_DECISION.md`](docs/I35_STEP411_DECISION.md)。两包参数余弦相似度`0.9999952352`、相对差异约`0.31%`；material fresh pass@64均为`0.0938`，推荐copy诊断仅小幅变化。step411可作为一次受控低剂量线上对照，但不能预期确定性超过step548；规划用中心估计约`1.033--1.034`，实用风险区间约`1.030--1.036`，step548继续作为默认交付包。

> 变更记录（2026-07-21 UTC）：I-34在当前最高观测r96上完成结果前冻结的beam-aware material准入。train 1,024行中r96/I-23 full-gold命中=`147/148`，I-23-only仅7；独立gate 256行两者均命中37，I-23-only仅1，四域覆盖也远低于门槛。按预注册停止：没有正式训练数据、W&B训练、checkpoint、r112包或线上提交。该结果说明I-23的线上material高档没有在这套原生no-think desc2sid beam64分布上形成可供残差学习的广泛teacher-only支持，不应靠放宽阈值强训。

> 变更记录（2026-07-20 UTC）：I-30 material teacher fresh-r8已完成单GPU W&B 512步训练与四点冻结门禁。run `ir2r0nd4`正常结束，microbatch路由精确为material/retention=`512/1536`，train loss=`2.1403`。step128/256/384/512均未同时通过两向material与七任务保持门；最早通过点为空，正式报告SHA256=`3e12f088...92bf`。按结果前规则本地关闭，不跑itemic、不打包、不上传、不宣称线上提分。

> 变更记录（2026-07-20 UTC）：I-31零训练精确插值开发探针完成。冻结`lambda=0.05/0.10/0.20/0.40`，公式`delta=(1-lambda)·r96+lambda·I23`。`lambda=0.10`两向material开发门均过，但world精确题由49/64降至48/64且保持权衡未优于I-30；`lambda=0.05`守住world但一向material为负。按结果前advance rule关闭，不建验收集、不打包、不上传。

> 变更记录（2026-07-20 UTC）：登记用户提供的当前最高方案 `I19-world-residual`：在独立复现的 I-13-like r80 parent 上，用1,573条授权Frinkleko clean懂世界数据和1,573条八任务KL保持数据训练fresh r16，再按`scale=0.875`精确拼成r96；线上单次`1.025259456`。该点高于相邻scale但未形成单调趋势。严格两文件r96包现已到卷并以报告SHA256逐字节命中；parent/residual、发布数据、复现源码与W&B run仍待接收，故状态为“最高线上观测、提交包已验收、完整复现链未闭合”；完整边界见[`docs/I19_WORLD_RESIDUAL_HANDOFF.md`](docs/I19_WORLD_RESIDUAL_HANDOFF.md)。仓内原I-19 DPO编号保留不变。

> 变更记录（2026-07-18 07:52 UTC）：回填`s53125=0.9757`并关闭I-23跨父残差scale轴；完成I-29四格GPU校准，canonical主指标未复现已知线上video方向，按预注册停止128组扩展和video训练。原因：两个结果都已触发结果前停止条件，后续必须回到当前主模型s800重新选择可校准的单变量路线。

> 变更记录（2026-07-17 17:08 UTC）：按用户要求完成唯一条件二分点`s53125`的CPU精确拼接、审计和严格两文件打包，当前等待手工提交。原因：落实16:39 UTC已经锁定的最后一次scale动作，不改变停止条件或增加新点。

> 变更记录（2026-07-17 16:39 UTC）：回填用户提供的`s5625`平台结果`0.9925`，确认其material已掉档到0.2453而video升至0.0864；按结果前规则只保留一次`s53125`条件二分，且未构建、未提交。原因：本点触发了“video充分恢复但material掉档”的冻结分支，必须关闭上半区并防止后验新增scale。

> 变更记录（2026-07-17 14:16 UTC；14:27 UTC状态更新）：`i23_userres_r80_s5625`已完成精确拼接和严格两文件打包，现等待用户手工提交；I-29 canonical renderer四格校准已完成预注册、无gold生成器和CPU scorer预检。目标环境的他人pip构建已结束并复核为vLLM 0.12.0/flash-attn 2.8.3，但空卡窗口随即被既有任务重新占用，八卡当前均无足够安全显存，故尚未执行GPU生成。原因：清楚区分可交付包、已准备诊断与真实运行结果。

> 变更记录（2026-07-17 13:30 UTC）：回填用户提供的平台面板结果：同一`s800`提交包复测为`1.0048`，`i23_userres_r80_s500`为`0.9882`；据此撤销“s500尚未提交”和“s750为下一优先发”的旧状态。同步纠正规则口径：主办方表述是“不鼓励模型融合”，并非禁止参数拼接；后续按允许使用、完整披露构造与复现链处理。原因：缺失面板记录会导致重复提交和错误排期，旧“合规灰区”措辞也与用户已提供的官方表述不符。

## 当前状态

- I-34已在训练前关闭：固定O6+r96/I-23、原生`/no_think`空think、固定domain前缀、无约束beam64x3。train/gate的teacher-only gap仅`7/1`，对照门槛`128/32`；train四域=`2/0/3/2`，gate四域=`1/0/0/0`，均不满足覆盖。r96/I-23在train净命中差只有`+1/1024`，gate为`0/256`，所以这条“从I-23 beam差集训练first-divergence r16”的数据基础不存在。两份ledger与审计已登记；正式训练集、sidecar、W&B、checkpoint、包和提交均为0。
- I-30已完成并本地否决：verified r96作为冻结parent，I-23只作material构造评分/KL teacher，fresh r8；正式混合512 material+1,536七任务保持，严格1:3。四点中material gold mean-logp最多仅`+0.00207`，两向改善率与teacher-KL联合门均未通过；多项保持任务Top-1也低于冻结`0.99`。最早全过点为空，零提交。
- I-31开发探针已完成并停止：直接插值证明`lambda=0.10`能把两向material分布向I-23移动，但同时使world精确题少1题，并未改善video/ad等总分风险。该点只作为下一版“物料任务向量起点+按任务恢复r96”的机制证据，不是提交候选。
- I-32冻结门结论仍为本地否决；用户基于每日5次额度授权step128作一次门外探索。原FP32 r168包先因423,941,100 bytes超过400MB被拒，BF16 r168包又因平台要求rank为1~128被拒，均未进入评测。现已将同一step128逐模块截断SVD为合法r128/alpha128并用BF16存储：严格两文件共161,535,020 bytes；656行上material两向gold均值仍为正、world保持11/16，itemic `0/60`。唯一待手工上传目录为`submissions/i32_task_restore_retkl_r128_step128_svd_bf16_platform/`；不追认原门通过，也不声明线上提分。
- I-37 strict-future r120线上`1.027625202`，相对I-35 step548为`-0.006803383`。它提高action/topic/ad/live，但video/product分别回退`0.0096/0.0102`，推荐合计净回退`0.0082`；分支已关闭，full包只保留复现审计，不再提交step256、scale或同路线变体。
- 当前最高单次线上观测是I-35 step548 r112：`1.0344285849069457`，八项=`0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591`。它从已验收的I19-world `scale=0.875` r96父模型训练fresh r16物料边界残差并精确拼为r112；正式评测状态`SUCCEEDED`、`retryCount=0`。提升来自推荐合计而非material跳档，不能写成“物料0.2453→0.2760”。
- 当前服务器可直接交付且本地训练/打包链闭合的最高模型也已更新为I-35 step548：`submissions/i35_r96_video_boundary_retkl_r112_step548_platform/`。父模型I19-world原始线上点为`1.025259456`，同模型后续复测为`1.025362611`；I-35相对两者分别高`0.009169129/0.009065974`。I-13 scale0.80仍保留为更早的完整复现基线，不再是最高可交付模型。
- I-23 `seed_teacher_cotfix_v3_r64_lr1e4_ep3`固定协议E3线上`0.9915`，八项=`0.2760/0.1099/0.0383/0.0576/0.1258/0.1400/0.1053/0.1387`。它从O6直接训练单个r64 adapter，不做参数拼接，现为固定协议最高无参数拼接单adapter；仍含164条官方来源派生teacher行，不能写成纯O1。相对s800首测低0.0122、相对同包最好显示分低0.0133，但material高0.0307；已获准作为新action-answer-token CE + 冻结I-23 KL保持实验的父模型。
- I-28已完成128/128步单卡训练并在冻结主门本地否决：W&B `t3xega98`正常结束，512个microbatch严格路由为proposal/retention=`128/384`，step64/128 adapter均完整。prompt-disjoint的128组/539个主gold上，step64的set-logsumexp均值变化`+0.00220`但只改善61/128组；step128为`+0.01413`、改善69/128组，低于冻结的`>=0.55`（至少71组）门槛。两点均停止于第一层，不再跑保持门或N4×K8，不打包、不上传、不作RFT/GRPO父模型，也不消耗用户已确认的账号级、队友共享每日5次线上额度。
- score-first四点均已线上：s500/s53125/s5625/s625=`0.9882/0.9757/0.9925/0.9866`，material/video依次=`0.2760/0.0576`、`0.2453/0.0864`、`0.2453/0.0864`、`0.2453/0.0768`。末次预登记二分s53125未保持material，且总分低s5625 `0.0168`、低s800复测`0.0291`；因此I-23跨父残差scale轴已关闭，不追加s515625/s546875或任何更细scale。`s750`仅保留为接近s800抖动中心的零训练备包，不是明确提分候选。
- I-29已完成`I23/s800 × legacy/canonical`四格校准。canonical下主指标candidate prefix mass为s800/I23=`88/97`，与已知线上video关系`s800>I23`相反；secondary group-any-ab虽为`2/1`，但不能在看结果后替换主指标，exact两者均为0。按预注册结论为`COMPLETE_PROXY_CALIBRATION_FAIL_NO_TRAINING`：不做128组teacher-forced扩展，不据此训练I-23 video residual。正式报告`logs/probe/i29_i23_s800_renderer_calibration_n16.json` SHA256=`4fc9ca83...ce25`。
- 当前P0以I-35 step548为保留父模型：I-36与I-37均已线上否决并关闭，不从失败点续训，也不回到I-35 checkpoint扫点。下一候选必须更换可验证机制轴，不能把I-37改scale或补step256伪装成新实验。
- I-25 `i23_actionres_r16_ansretkl_ep1`已完成1,527/1,527；step250缺失不影响实战检查。按原冻结协议六个scale1点均失败，因此字面结论仍是`ORIGINAL_GATE_FAIL`，且不能外推线上方向。当前1024复评中step500的scale `0.5/0.625/0.75`分别为JSON `26/28/29`、截断`6/4/3`；material与推荐保护诊断没有发现硬灾难，但也不能证明上分。s500/s625已完成线上且均未替换主模型后，该action残差分支继续封存：不打包、不上传，I-26重训仍未获授权。
- I-18只修复538条上游截断推荐CoT，固定协议E3线上0.9697，八项=`0.2453/0.1083/0.0382/0.0768/0.1190/0.1316/0.1089/0.1416`，低I-13 0.0281，未替换主模型。数据、manifest与恢复说明仍位于[`assets/derived/releases/seed_teacher_cotfix_v2/`](assets/derived/releases/seed_teacher_cotfix_v2/)；I-10 E3缺同协议桥，因此不把该结果解释为CoT修复的净因果效果。
- 2026-07-13下午平台修复评测不稳定问题。仓内日志证明协议切换发生在I-10 E3（11:45，旧）与I-11（16:40，新）之间：旧日志action上限4096且itemic只跑1次beam64；新日志action上限1024且itemic执行7次`Race averaged evaluation`。两边虽然都打印`version: v3.1`，仍必须隔离为`platform-pre-fix-v3.1`与`platform-stable-v3.1-20260713`，禁止直接做分数差。
- 旧协议最高单次显示分是`seed_teacher_r64_lr1e4_e3 = 0.9849`；I-10同轨迹E1/E2/E3=`0.9100/0.9680/0.9849`只保留为旧协议内部剂量曲线。E3的固定协议桥仍未建立，不能用旧0.9849直接压过新协议候选；此前仅剩一次配额时未重测，I-17出分后它重新成为判断DPO净方向的必要直接对照。
- 仓内既有固定协议结果为I-11/I-12/I-13-s875/I-13-s800/I-14/I-17/I-18/I-19-DPO/I-23=`0.9618/0.9768/0.9978/(1.0037,同包复测1.0048)/0.9518/0.9727/0.9697/0.9763/0.9915`；I-13 s800是I19-world/I-35之前的完整本地复现基线。I19-world四点为`0.9966/0.9902/1.0253/0.9778`，其0.875 r96已作为I-35父包验收；当前最高由I-35 step548刷新为`1.034428585`。
- I-13只把I-12的r16用户残差缩到`0.875`，不重训。线上结果验证了本地Pareto选择的总分方向，但收益来自推荐四项合计`+0.0229`抵消用户两项合计`-0.0026`，不能解释成用户能力提升；E3固定协议桥仍缺失，继续禁止与旧协议0.9849作差。
- I-14从O6干净训练纯`D(O1)`单体r80，E3线上0.9518；它不含teacher、第三方、评测回灌或参数拼接。相对I-13的差值只回答榜分替换问题；I-13是不同构造血统的参数拼接模型，虽可提交但不能作纯O1路线的科学基线，更接近的I-11仍有teacher、续训和rank混杂。
- “最高干净模型”必须分口径：旧协议最高无参数拼接单adapter是I-10 E3（0.9849，但含164条官方来源派生teacher行）；固定协议最高无参数拼接单adapter现为I-23 E3（0.9915，同样含164条teacher行）；最严格纯O1、无teacher/融合的是I-14 E3（固定协议0.9518）。三者口径不得混写。
- I-16/I-17完成推荐金标排序DPO剂量搜索。I-16的step200/400/600均因gold-logp保护线失败而本地否决；I-17从I-10 E3重新启动并按预注册选最早过门槛的step100。本地推荐聚合raw win从32.03%升到43.36%，但固定协议线上仅0.9727，八项=`0.2453/0.1077/0.0380/0.0960/0.1156/0.1274/0.1044/0.1383`，未替换I-13且低0.0251。相对I-12，推荐四项合计虽高0.0101，但用户两项合计低0.0142，总分低0.0041；I-10 E3固定协议父分仍缺失，因此不能把0.9727归因为DPO净负收益，也不盲投step150/200。
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
