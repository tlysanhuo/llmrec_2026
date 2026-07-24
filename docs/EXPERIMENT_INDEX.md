# Experiment And Artifact Index

> 当前状态基线：2026-07-23 UTC。
> 旧版完整历史表已归档到 `docs/archive/EXPERIMENT_INDEX_pre_cleanup_20260711.md`。
> 变更记录（2026-07-23 UTC）：I-40已从原始I-35 r112全新恢复为真正脱离终端的GPU1后台run [`34k0sdcj`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/34k0sdcj)，detached wrapper PID/SID=`2665551`、PPID=`1`，data/W&B身份、policy/reference、optimizer-only与step0 logits差0门均再次通过。前一run `9dp9wnbo`本地运行到step169后无错误栈/OOM/NaN地被外部终止，W&B状态`crashed`，未到step515故无checkpoint；日志SHA256=`41c55bb...356`，完整归档且不resume。
> 变更记录（2026-07-23 UTC）：I-40有效正式run已在GPU1单卡启动，W&B [`9dp9wnbo`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/9dp9wnbo)。真实加载门通过：policy r112为392 tensors/70,647,808参数，冻结reference逐tensor精确快照后差0且未进入optimizer，step0 logits最大差0；25步loss=`0.2934384`、无OOM/NaN/路由错误，当前继续运行。第一次尝试在optimizer step0前因两条adapter加载精度路径最大差`2.43149698e-4`安全停止，无W&B run/checkpoint且禁止resume；失败日志已保留。
> 变更记录（2026-07-23 UTC）：用户授权I-40按冻结方案启动：不是在I-35上再挂或拼一个低秩LoRA，而是直接加载当前最高I-35 step548 r112并继续更新这同一个r112。正式数据8,240行由I-36全部5,500条审计用户样本与I-35原2,740条组成；后者只作当前I-35自KL，不复用旧boundary/preserve/margin/CE目标。batch1×acc4、2,060步、单GPU、W&B online；数据/sidecar/audit静态预检已通过，当前进入真实模型加载门禁与启动阶段。
> 变更记录（2026-07-23 UTC）：按用户要求补跑I-39队友full v4，GPU3单卡、seed42、mat/rec/action/topic/world全量完成，报告`logs/offline_eval/i39_i35_userab_firstdiv_retkl_r120_teammate_v4.json` 111,194 bytes/SHA256 `252736fd...cd12`，runtime `1967.8s`。I-39/I-35/I-37的圈外mat fresh为`60/542`、`55/542`、`57/542`；I-39增量集中在ad/video（相对I-35各`+2/+3`题），rec四域合计`36/4000`对I-35的`35/4000`，action/topic/world=`0.3018/0.0325/0.424`对`0.3050/0.0275/0.422`，未见数量级崩坏但动作格式长尾略差。该v4经I-37/I-38线上对照为`NOT_CERTIFIED`，不能换算线上分；它推翻“仅凭teacher-forced冻结门直接否决上传”的证据充分性，但不证明提分。I-39现建议只占一次正式额度作探索验证，不替换I-35默认交付、不续训或扫scale。
> 变更记录（2026-07-23 UTC）：I-39唯一一发已完成GPU3单卡训练、精确组合和训练前冻结门。W&B [`51yko99h`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/51yko99h)服务端`finished`，640/640步、runtime `1435.1195s`、train loss `1.4501802083`，三路与四类物料目标精确为`1920/512/128`和`64/192/128/128`；fresh r8 adapter SHA256=`5fdf62fc...1f9bd`，与固定I-35 step548 r112按1.0/1.0精确拼成唯一r120包`submissions/i39_i35_userab_firstdiv_retkl_r120_step640_platform/`（392 tensors、302,830,492 bytes、adapter SHA256=`746b9c89...f8be7`）。313行冻结门报告`logs/probe/i39_i35_ab_firstdiv_material_gate_v1.json`（SHA256=`25b69458...72ca`）显示A/B/C首分歧margin均改善，但full-anchor三项保护和全体KL失败，`teacher_forced_pass=false`；I-39不上传、不续训或扫scale，包仅留审计。
> 变更记录（2026-07-23 UTC）：用户授权I-39唯一一发。以当前线上最高I-35 step548 r112为父模型，从O3中构造与I-36用户历史实际相交且排除O1/E/I-35训练题面的3,072行视频池，用固定单父Beam64标出A/B/C首错与完整锚；正式混合已冻结为物料首分歧/关联用户微剂量/I-12父保持=`512/128/1,920`（20%/5%/75%），另冻结313行、256 AB门。正式数据/sidecar/gate SHA256=`0a5cb2e5...a3e2`/`d9d74eb5...bd14`/`293fc361...d40`；独立审计和trainer全量预检通过。唯一训练配方为fresh r8、单GPU/W&B、640步、无中间候选；当前尚未启动，不声称提分。
> 变更记录（2026-07-23 UTC）：用户授权启动唯一新候选I-38M。策略从material已达`0.275958188153`的I-23 r64开始，material 1,370行只对exact I-23起点做KL锚，非material 1,370行只对当前最高I-35 step548做KL蒸馏；fresh r16、单GPU/W&B、685步、只验收full r80一个候选。正式数据/gate SHA256=`5d8ca1a6...28d58`/`311b298f...ed41f`，T/E/model-generated训练行0；训练尚未启动，不声称涨分。
> 变更记录（2026-07-23 UTC）：I-38M已完成唯一单卡训练、组合和冻结门。W&B [`f92senkn`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/f92senkn)为`finished`，685/685步、runtime `1282.5365s`、train loss `0.6860710371`、material/retention路由`1370/1370`；fresh r16 adapter SHA256=`1690bdde...e45b52`，与I-23 r64精确拼成r80包`submissions/i38_i23_material_i35_teacher_retkl_r80_step685_platform/`（392 tensors、201,904,514 bytes、adapter SHA256=`74a86b03...3d5d94`）。400行冻结teacher-forced gate报告`logs/probe/i38_i23_material_i35_teacher_gate_v1.json`（SHA256=`2e2e9f3a...f69b13`）失败：material desc2sid KL/Top1=`0.0085404/0.9594`、sid2desc=`0.0020747/0.9722`，video/ad Top1保护也失败；retention aggregate KL ratio=`0.19885`通过。队友v4小样本行为诊断`logs/offline_eval/i38_i23_material_i35_teacher_retkl_r80_step685_matrec32.json`（SHA256=`da2a5b96...c6671`）不作线上分数估计；I-38不上传，包仅留审计，I-35 step548继续默认交付。
> 变更记录（2026-07-23 UTC）：I-37 full r120正式评测`SUCCEEDED`，evalTaskId `eval-task-0yco4c-1784766273`、modelId `md-z9m20x-1784766072216356022`、retryCount=0、耗时4,036秒，总分`1.02762520217381`，八项=`0.2452961672/0.1204379107/0.0394833175/0.0768/0.1292/0.1484/0.1089/0.1591078067`。相对I-35 step548总分`-0.0068033827`，用户合计`+0.0013966173`、推荐合计`-0.0082`、material/world不变；I-37关闭，不追加checkpoint/scale，I-35 step548继续作为最高与默认交付工件。
> 变更记录（2026-07-22 UTC）：I-37已完成唯一队友v4离线全量回归，报告`logs/offline_eval/i37_i35_strict_future_rec_r120_teammate_v4.json`（SHA256=`4698d1aea0dc3e714823839df1ed69c365adfbe806c6c7eeb28edf3dc3048ca3`）。mat fresh/train=`0.1052/0.1567`，rec video/prod/ad/live=`0.006/0.003/0.002/0.028`，action F1=`0.3030`、topic=`0.0244`、world=`0.432`；相对I-35 step548 v4为video/live/world正向、action/topic轻微回退。v4仅作离线行为回归，不能换算线上总分；I37仍待手工上传官方评测。
> 变更记录（2026-07-22 UTC）：I-37已完成唯一单卡训练与精确打包。W&B `c2crod0w`完成512/512步、runtime `1178s`、train loss `1.2019`；fresh r8 checkpoint SHA256=`8c25566c...fcea69`，与固定I-35 step548 r112精确拼成唯一r120包`submissions/i37_i35_strict_future_rec_r120_v1_platform/`，392 tensors、302,829,416 bytes、adapter SHA256=`e91c773c...675252`，逐tensor切片恒等式通过。尚未上传/评测，不声称已涨分；首轮step256保存校验器误用r16规则，已隔离且禁止resume。
> 变更记录（2026-07-22 UTC）：按用户要求停止I-35追加离线/线上扫点，启动唯一新候选I-37。它以I-35 step548 r112为父模型，只用O2 UserProfile/Pid2Sid严格未来video/ad各512条做低权答案CE，并用1,024条已登记I-12样本做强父KL；fresh r8、单GPU、512步，最终只打包full r120，不把step256扩成第二个评测点。数据与tokenizer预检已通过，正式训练尚未启动。
> 变更记录（2026-07-22 UTC）：I-36 step4125已完成线上评测但失败，任务名`i36_i35_user_expand_retkl_r128_step4125_V1_eval_20260722183137`，耗时`1h5m22s`，总分`0.9865`，八项=`0.2453/0.1070/0.0331/0.0672/0.1292/0.1414/0.1035/0.1599`。相对I-35父模型总分约`-0.0479`，用户合计约`-0.01842`、推荐合计`-0.0302`；I-36分支关闭，step2063包保留但不建议占用共享额度。
> 变更记录（2026-07-22 UTC）：I-36正式训练与双点打包完成。有效run W&B `mmenbci2`服务端`finished`，4,125/4,125步、runtime `7921.5533s`、train loss `1.0296107737`，最终路由retention/action/topic=`11000/4000/1500`；step2063/full fresh-r16已分别与固定I-35 r112精确拼成合法r128，两包均严格两文件、392 tensors、323,015,596 bytes并通过逐tensor加和。第一次run `onqds9a5`在step18被外部SIGKILL且无checkpoint，禁止resume或作候选。
> 变更记录（2026-07-22 UTC）：用户授权启动I-36，在当前线上最高I-35 step548 r112上训练fresh r16懂用户残差。原始15,023条生成数据完成结构审计后，只保留4,000 action与1,500 topic严格样本，并混入11,000条I-35父模型KL保持，形成16,500行1:2正式数据；最终只允许比较半程step2063与完整一轮step4125，分别精确拼成合法r128，不增加rank、scale或后验checkpoint。
> 变更记录（2026-07-22 UTC）：I-35 step548 r112正式评测`SUCCEEDED`，evalTaskId `eval-task-9nepj1-1784698215`，总分`1.0344285849069457`，八项=`0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591`，成为当前最高线上观测。相对直接父模型原始`1.0252594563571054`为`+0.009169129`；material不变，推荐合计`+0.0141`抵消用户合计`-0.003815630`和world `-0.001115242`。step411尚未评测，是两点限制下唯一剩余I-35候选。
> 变更记录（2026-07-22 UTC）：I-35 step411/step548已完成同seed离线成对对照，详见`docs/I35_STEP411_DECISION.md`。参数余弦相似度`0.9999952352`、相对差异约`0.31%`；material fresh pass@64均为`0.0938`，推荐copy诊断仅小幅变化。step411建议只占用一次线上额度做低剂量对照；规划用分数中心约`1.033--1.034`、风险区间约`1.030--1.036`，不能承诺超过step548。
> 变更记录（2026-07-22 UTC）：I-35已完成训练与五个r112打包，但按用户决策只授权step411和step548两次线上测试；实际已先评测step548，step411是唯一剩余点。step411约覆盖40--41/66边界样本且仍在有效学习率区间；step548约覆盖50/66，作为更强剂量。step137/274/685包仅保留审计，禁止上传。
> 变更记录（2026-07-21 UTC）：I-34固定beam64预计算完成并触发结果前停止条件。1,024行train上r96/I-23 full-gold命中=`147/148`，I-23-only gap仅7（四域`2/0/3/2`）；256行独立gate两者均命中37，I-23-only gap仅1（四域`1/0/0/0`），远低于预注册128/32及每域16/4。正式训练数据、sidecar、W&B run、checkpoint、r112包和线上提交均未生成；I-34 v1关闭。最终gate SHA256 `f5f589af17380082fae841ee3c14f0635d05ed65c39c7595e3d0377977b7d84a`，原结果前预注册SHA仍为下一行的`f3090a73...b49`。
> 变更记录（2026-07-21 UTC）：结果前预注册I-34 beam-aware material分支。只允许在E-clean且与I-30/I-33正式训练隔离的O1 desc2sid池上，用固定平台镜像beam64筛`I23 full-gold hit && r96 full-gold miss`；训练准入至少128条train gap、32条独立gate gap且四域有覆盖。通过后唯一配方为r96+fresh r16、128 hard/384 r96-KL保持、first-divergence margin、128步；真实beam门不过即停止，不扫LR/rank/scale。预注册文件SHA256 `f3090a7354bd6da25c34474f15046ac0cf49dbedf28682af8719478b65142b49`。
> 变更记录（2026-07-20 UTC）：I-33 `i33_r96_material_desc2sid_retkl_r8_v1` 单GPU/W&B正式训练与736行冻结门完成。W&B `io58fx1s`正常完成512/512步，路由retention/material精确为1,536/512；四个r104候选的desc2sid gold均值均为负、改善率最高仅step512的53.125%，sid2desc保护及多项保持Top-1亦未过线。`earliest_teacher_forced_pass=null`，按预注册不跑itemic、不打包、不上传、不续训或扫scale。
> 变更记录（2026-07-20 UTC）：I-32 step128原FP32 r168包因423,941,100 bytes超过400MB被拒；BF16 r168缩至211,997,868 bytes后又因平台要求rank为1~128被拒，均无evalTaskId。现对同一权重逐模块截断SVD至r128/alpha128并BF16存储，严格两文件共161,535,020 bytes；656行material两向gold均值仍为正、world 11/16不退，itemic 0/60。唯一允许手工上传的是`submissions/i32_task_restore_retkl_r128_step128_svd_bf16_platform/`，所有r168包禁止重传。
> 变更记录（2026-07-20 UTC）：用户在I-32冻结门失败结论完成后，基于每日5次额度明确授权step128作一次门外线上探索。该点提交级itemic 0/60并已严格两文件打包，当前待手工上传；本动作不追认冻结门通过，不开放续训、scale或其余checkpoint并行提交。
> 变更记录（2026-07-20 UTC）：I-32单GPU/W&B正式训练与656行冻结门完成。四个r168科学候选均保持world exact 11/16，但material双向改善率没有同时过55%，多项七任务保持Top-1也低于0.99；按预注册本地否决，不运行itemic、不打包、不上传。8个r8保存点和4个r168组合点仅保留审计，禁止作父模型或继续训练起点。
> 变更记录（2026-07-20 UTC）：报告版I19-world r96工件仍未到卷；按同一数据机制完成本地候选 `i19_local_world_residual_retkl_r16_ep1_s800`，以仓内可验收s800 r80为parent训练fresh r16。第二次W&B run `xdzb35cp` 完成787/787步、3146条路由精确为world/retention=1573/1573；r96 scale0.875本地包已生成并通过结构门，但尚无线上分数，不替代报告版最高观测。第一次路由前缀误判已收档，不得resume。
> 变更记录（2026-07-20 UTC）：登记 `I19-world-residual` 为当前最高单次线上观测：独立复现I-13-like r80 parent + fresh懂世界/八任务保持r16，按`scale=0.875`精确拼成r96后为`1.025259456`。严格两文件r96包现已到卷并以报告SHA256验收；实际parent/residual、发布数据、复现脚本和W&B证据仍不在本卷，故可登记为已保留提交包但不能声称完整训练链已复现；仓内I-19 DPO继续使用原编号。
> 变更记录（2026-07-18 19:48 UTC）：补登记训练前冻结的四点checkpoint门及256条八任务保持集；平台入口同步锁定gate资产，SHA256更新为`30531cfd...d47`。原因：训练只能产生日志与候选残差，是否允许r88拼接必须由结果前规则决定。
> 变更记录（2026-07-18 19:40 UTC）：登记用户批准且已完成静态预检的`s800_native_general_replay_r8_v1`：513行正式D、exact target-token路由、fresh r8残差trainer、冻结配置及GLM Training Task容器入口已就绪；开发机未训练，当前SSH也没有平台任务所需的`WANDB_API_KEY`环境。原因：从当前1.0048主模型启动覆盖广度更高的官方原生General最小回放，同时把“已准备”与“已在平台训练”严格分开。
> 变更记录（2026-07-18 07:52 UTC）：将`s53125`登记为线上完成0.9757并关闭I-23跨父残差轴；登记I-29三格rollout、正式报告及`PROXY_CALIBRATION_FAIL_NO_TRAINING`状态。原因：真实线上与离线校准结果均已触发冻结停止条件，索引不得继续显示待提交、待GPU或video训练准入。
> 变更记录（2026-07-17 17:08 UTC）：登记`s53125` checkpoint、组合审计和严格两文件待手工提交包。原因：用户确认交付该冻结末次scale，产物已真实生成并完成逐字节验收。
> 变更记录（2026-07-17 16:39 UTC）：把`s5625` checkpoint/包更新为线上完成0.9925，并登记唯一后续为尚不存在的`s53125`条件二分；不在保留模型表中伪造未构建产物。原因：线上结果已经触发冻结停止树，模型索引必须同步真实存在性和允许角色。
> 变更记录（2026-07-17 14:16 UTC）：登记`s5625`的r80 checkpoint、组合审计和严格两文件手工提交包；登记I-29 renderer四格校准预注册与两份已通过CPU预检的实现。GPU生成仍未发生。原因：产物索引必须区分“已打包待手工提交”“仅CPU准备完成”和“真实生成/训练已完成”。
> 变更记录（2026-07-17 13:30 UTC）：回填`s800`同包复测`1.0048`和`i23_userres_r80_s500=0.9882`的平台面板证据，把s500 checkpoint/包改为已线上完成，并将s750降级为备包；同步纠正参数拼接为“不鼓励而非禁止”。原因：产物索引必须反映真实线上状态，避免重复上传和错误优先级。

本文件只登记当前仍存在、仍可使用的模型产物。历史分数和实验归因见 `experiment_log.md`。

## 已冻结并获授权启动：i40_i35_direct_user_continue_r112_v1（I-40）

| 项 | 记录 |
|---|---|
| 目标/起点 | 从当前线上最高I-35 step548严格两文件r112 `submissions/i35_r96_video_boundary_retkl_r112_step548_platform/`直接续训；adapter/config SHA256=`52d945cc80c8933684921b98792fca84a7528929dbd74e23cd4856a93d9b2c00`/`4f90d28f006bd93f57da2f3f8e23708aeac9ae69247bd89be67d09c40e9b5996`。不加载fresh adapter、不做rank拼接，训练前后都必须保持392个LoRA tensor、r112/alpha112；O6基座、embedding与lm_head冻结 |
| 正式数据 | `assets/derived/processed/data_i40_i35_direct_user_continue_v1.jsonl`，8,240行/102,365,905 bytes/SHA256 `483a4bb2f98d41497600d078032634d4f36fe2970a53d98b4a7fccc488910c18`。I-36审计用户行5,500（action/topic=`4,000/1,500`）+ I-35原正式行2,740，比例=`5,500:2,740`；两路exact/mode交集0、T/E训练行0、最长8,864<16,384。I-35上游继承25个重复world行且各暴露2次，全部2,740条按用户要求逐行保留，不把其误写成2,740个唯一内容 |
| 路由sidecar | `assets/derived/processed/data_i40_i35_direct_user_continue_v1_sidecar.jsonl`，8,240行/7,605,609 bytes/SHA256 `e9bc129cd834bff161247985cc5430cf46872006cbae4a86fd37c3666b60acb2`；按prompt+response token hash多重集锁定8,240次暴露，user_ce/retention_kl=`5,500/2,740`，规范化唯一行8,215 |
| 损失语义 | 5,500条用户行：`0.05 * weighted answer CE + 16.0 * KL(I35_step548 || policy)`；2,740条I-35回放：`16.0 * KL(I35_step548 || policy)`，每行最多128个回答token参与KL。原I-35的boundary/preserve、margin、gold CE与旧r96 teacher均不复用；reference从真实加载后的step0 policy逐tensor精确快照后永久冻结，二者必须完全一致且step0 KL指纹为0 |
| 训练物理 | 单GPU、W&B online；直接更新既有r112，batch1×gradient accumulation4，1个确定性shuffle pass，2,060 optimizer steps；lr=`5e-7` cosine、warmup_ratio=`0.03`、weight_decay=`0.001`、max_grad_norm=`0.5`、BF16、seed=`19260840`。adapter-only保存点=`515/1030/1545/2060`，只保存policy r112，不保存冻结reference |
| 实现/预检 | builder/trainer/config/dataset-registry/online-launcher/detached-launcher SHA256=`24a548c168400a8e139dcc9802d26317d410f2c11e5e9881cba4b5b23748c5fe`/`cd5eb4095a692015bfb26831b10c02da1c0c2836a56581096ecf90e02f915473`/`992825270a4f16ed9a4ec26f2e0d974603a9c2c1f9ae9c4e79fcb50da2cdfbc3`/`6bd0f06fb18ea4d2a864e6515c1d0c85ce6f5f4bb2c0f0f22f047a61ba5d06e0`/`48a403eb7080a98b74e7371b8e1eebd855f3a5f56cc248dc9aa9e0c590288f1f`/`98c6008469b5800b1605c58e209d083e8002e3fac45fb53aa2599a6589ebc521`；audit `logs/data/i40_i35_direct_user_continue_v1_audit.json` SHA256 `c5c2323b2c9aa1dddd4e49936bad09a5cb342a1f32d592f3574cc442b6985b7c`。builder fail-closed、trainer self-test/py_compile及全量8,240行token路由预检已通过。第一次真实加载门在optimizer step0前发现policy加载路径发生BF16 round-trip、后加载reference保留源FP32，最大差`2.43149698e-4`并安全停止；修复为从真实step0 policy精确快照reference后再冻结，不从失败状态resume。失败日志`logs/train/i40_i35_direct_user_continue_r112_v1_failed_preoptimizer_attempt1.log` SHA256 `610fe028...91b`，无W&B run/checkpoint |
| 当前状态/停止边界 | `RUNNING_DETACHED_SINGLE_GPU_WANDB_ONLINE`：GPU1，W&B [`34k0sdcj`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/34k0sdcj)，detached PID/SID=`2665551`、PPID=`1`；从原始I-35 r112全新启动，真实policy/reference/optimizer/step0 logits门全部通过。旧run `9dp9wnbo`仅本地到step169后外部终止，W&B `crashed`、无checkpoint，归档日志`logs/train/i40_i35_direct_user_continue_r112_v1_crashed_terminal_attempt2_step169.log` SHA256 `41c55bbf...356`，禁止resume。任何reference参数进入optimizer、rank/tensor数漂移、路由计数不符、NaN/Inf或非单GPU均立即停止；四个保存点不是自动授权的四个线上提交 |

## 已完成唯一训练、双诊断证据冲突且建议一次官方验证：i39_i35_userab_firstdiv_retkl_r8_v1（I-39）

| 项 | 记录 |
|---|---|
| 目标/父模型 | 从当前线上最高I-35 step548 r112 `submissions/i35_r96_video_boundary_retkl_r112_step548_platform/`启动，直接扩大懂物料AB覆盖并只在父Beam首分歧位推动gold，同时以5%关联用户微剂量测试“用户历史中出现的物料”是否产生协同；父adapter/config SHA256=`52d945cc...2c00`/`4f90d28f...5996` |
| AB/首分歧构造 | O3中先保留完整video SID精确出现在I-36 `user_ce`历史的SID-caption对，再排除O1懂物料、全部登记E exact/mode与描述、I-35正式prompt；供给51,065唯一SID/25,244 AB，确定性取3,072唯一SID/2,560 AB，其中512个AB有第二个不同C视图。I-35 step548单父BF16 Beam64x3得到A/B/C首错/完整命中=`1076/1060/591/345`，模型候选只作负例、gold只来自O3 |
| 正式数据 | `assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1.jsonl`，2,560行/20,927,751 bytes/SHA256 `0a5cb2e55fff2c21deb1452216e08eae104bb5ce7d7e68a599ac52908261a3e2`；物料首分歧512（A/B/C/完整锚=`128/128/192/64`，480唯一AB、32个双C首错组）、关联用户128（action/topic=`96/32`，每行历史输入均命中所选物料SID）、I-12父保持1,920，比例`20%/5%/75%`；T/E训练行0 |
| 路由sidecar | `assets/derived/processed/data_i39_i35_userab_firstdiv_retkl_v1_sidecar.jsonl`，2,560行/2,200,966 bytes/SHA256 `d9d74eb573523eb70d0593076542c49d047fdcd4eb616a88d777476e5532bd14`；与训练行按qwen3_nothink prompt token hash严格双射，锁定route/task/response/父hash及物料gold ABC、focus、hard negatives；单父helper兼容字段`teacher_score`已删除且不得解释为第二教师 |
| 训练前冻结门 | `assets/evaluation/holdout/data_i39_userab_firstdiv_gate_v1.jsonl`，313行/418,832 bytes/SHA256 `293fc361295db56196acc035bd639d63e426168b93ea36f2d21c3890c2a34d40`；256唯一AB，A/B/C首错/完整锚=`85/120/65/43`，57个双C组、4个双C均首错组。与正式训练在exact/mode prompt、物料AB、三路输入输出完整SID/AB上均零交集。结果前配置`configs/evaluation/i39_i35_ab_firstdiv_material_checkpoint_gate_v1.json` SHA256 `89737747...d504`：A/B/C各层要求focus gold logp与hard-negative margin均值不退且margin改善行率>=0.55；43条完整锚要求parent Top1一致率>=0.99、KL<=0.005、gold logp delta>=-0.01；全体只要求KL<=0.01，不在首错行强保父错误Top1。不进入训练、不估线上分 |
| 训练目标/物理 | A/B/C首错行在对应位置做`0.50*softplus(margin=0.10)+0.02*gold CE+8.0*KL(parent||policy)`；完整锚做`16.0*parent KL`；用户微剂量做`0.05*CE+16.0*parent KL`；I-12保持做`16.0*parent KL`。r112 merge后fresh r8/alpha8/dropout0.05/all-linear；batch1xacc4、lr5e-6 cosine、warmup0.03、wd0.001、BF16、640步、无中间checkpoint |
| 实现/预检 | builder/trainer/config/registry/launcher/evaluator SHA256=`a12617ca...c810`/`a979a843...b343`/`ebe0227d...605d`/`6cdb3525...3221`/`fc8f1990...239b`/`c3b409e5...8ee`；formal audit `logs/data/i39_i35_userab_firstdiv_retkl_v1_audit.json` SHA256 `c52921ca...a718`。独立复算确认2,560双射、token prompt全唯一、teacher_score=0、跨路由/gate泄漏0；trainer全量重渲染预检最长9,431<16,384、路由与目标计数一致、精确640步；launcher额外锁定结果前gate-config，shell/负参数/dry-run/W&B online身份和`WANDB_DISABLED`拒绝检查通过。evaluator另绑定唯一I-39 canonical包及exact-combine audit，实测拒绝I-37 r120 |
| 唯一训练结果 | GPU3单卡、world size 1；W&B [`51yko99h`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/51yko99h)服务端`finished`，640/640步、epoch1、runtime `1435.1195s`、train loss `1.4501802083104849`。最终microbatch路由retention/material/user=`1920/512/128`，物料full/C/B/A=`64/192/128/128`，与冻结契约精确一致；训练日志`logs/train/i39_i35_userab_firstdiv_retkl_r8_v1.log`为114,935 bytes/SHA256 `d1534b07...62d6d`，无OOM/NaN/traceback。fresh r8 adapter/config SHA256=`5fdf62fcd80f7b71806fc58e6ce8b31caa36fb519ba97252460d6b6d9bd1f9bd`/`b869ae92e27f98736a1b02578c253a75724172772b84f7dc455e1301c324092b`，392 tensors全有限且只含5,046,272个LoRA参数。末行`[i36] training PASS`是I-39复用I-36已审计单卡runtime的固定打印标签；I-39数据/loss/常量/输出与最终合约均由wrapper覆盖并独立验收，不是第二次或错脚本训练 |
| 唯一组合产物 | fresh r8只与固定I-35 step548 r112按1.0/1.0组合一次；`submissions/i39_i35_userab_firstdiv_retkl_r120_step640_platform/`严格两文件，r120/alpha120、392 tensors，adapter/config=`302,829,416/1,076` bytes、SHA256=`746b9c8986a4c9f0e5cf87d2d0f6e93bf3e1fa81e68e9464474ec928418f8be7`/`7838994a5e6a608b1d0250826a9e06a0a76f1b241c8ac9b2911b45077a6fcf1c`。独立逐tensor复核196个A与196个B的parent112维/residual8维切片逐值相等；package audit `logs/package/i39_i35_userab_firstdiv_retkl_r120_step640_audit.json` SHA256 `7bdc36d9...5d950` |
| 冻结门结果 | 313/313完成、`error_count=0`，报告`logs/probe/i39_i35_ab_firstdiv_material_gate_v1.json` 9,476 bytes/SHA256 `25b694584a6f696c6deb814696743ccea65352bd6206f19224450f11d62772ca`。A/B/C focus margin delta=`+0.181618/+0.124826/+0.189423`、改善率=`0.741176/0.650000/0.830769`、focus gold logp delta均为正，首分歧方向全部通过；但full-anchor Top1 agreement/KL/gold-logp delta=`0.976744/0.0101559/-0.0479034`，分别未达`0.99/<=0.005/>=-0.01`，全体KL `0.0105687`也高于`0.01`，故`teacher_forced_pass=false`。全体gold Top1 token仅`+2`，不能据此估线上material档位或总分 |
| 队友full v4行为结果 | GPU3单卡、seed42、完整mat/rec/action/topic/world，报告/控制台SHA256=`252736fd...cd12`/`29f92f15...f16f`，无ERROR/OOM/traceback。I-39相对同协议I-35：mat fresh `60/542`对`55/542`（ad/living/prod/video=`18/1/21/20`对`16/1/21/17`），I-39训练物料与fresh集完整gold SID及描述子串重合均为0；mat train `50/300`对`48/300`但有4条训练prompt重合故只作辅助。rec四域总命中`36/4000`对`35/4000`；action F1/JSON-ok/trunc=`0.3018/0.932/0.068`对`0.3050/0.942/0.058`；topic/world=`0.0325/0.424`对`0.0275/0.422`。86条官方记录确认material `0.2452961672473868→0.27595818815331014`精确等价有效命中`8→9/574`，但唯一同协议完整父子校准I-35→I-37是离线`55→57/542`而官方仍`8→8`；故I-39只支持“物料方向比I-37更强且未观察到大崩坏”，不支持预计0.276或总分 |
| 唯一候选边界 | 只允许从固定I-35 step548干净启动一次fresh r8训练；只验收完整640步残差，并只允许与固定r112按1.0/1.0精确拼成一个r120候选。不生成/比较中间checkpoint，不扫rank、LR、epoch、scale或第二个seed；任何启动前契约失败先修复且不计训练，优化器一旦开始不得重跑 |
| 当前状态 | **UNIQUE_RUN_COMPLETE_DIAGNOSTICS_CONFLICT_ONE_OFFICIAL_PROBE_RECOMMENDED**；teacher-forced保持门失败，但独立full v4圈外material相对I-35多5/542且全维无结构性崩坏，因此撤销“仅凭冻结门不上传”的最终性结论，建议只占一次正式额度验证。该建议不是预计提分；正式结果前I-35 step548仍是当前最高与默认交付，不续训、不扫checkpoint/rank/LR/seed/scale |

## 已完成训练、冻结门否决且不上传：i38_i23_material_i35_teacher_retkl_r16_v1（I-38M）

| 项 | 记录 |
|---|---|
| 目标/可证伪点 | 反转I-30/I-33从r96追I-23 material的失败方向：从I-23开始保住其线上material一题阶跃，同时把I-35 step548的非material行为蒸馏进fresh residual。I-35总分加I-23相对I-35的material离散增量给出条件算术上限`1.0650906058125589`，只作上限，不是预计分 |
| 起点与教师 | policy/material anchor=`submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform`，adapter/config SHA256=`0e5fa9bb...c6b8`/`b3f2a1b5...e7e7`；nonmaterial teacher=`submissions/i35_r96_video_boundary_retkl_r112_step548_platform`，SHA256=`52d945cc...2c00`/`4f90d28f...5996` |
| 正式数据 | `assets/derived/processed/data_i38_i23_material_i35_teacher_retkl_v1.jsonl`，2,740行/16,532,122 bytes/SHA256 `5d8ca1a6fa9190841187543559ead1d497d48a50b082382c9fa8501add928d58`；逐行复用I-35登记D且不改题面/标签，material anchor/I35-retention=`1370/1370`，T/E/model-generated训练行0；builder/audit SHA256=`0581b1f3...926a`/`f2f0cb5d...a99c` |
| 训练目标/物理 | material只做`8.0*KL(policy || exact merged-I23 start)`；action/topic/video/prod/ad/living/world只做`8.0*KL(policy || frozen I35 step548)`，最多128个均匀答案位置；两路gold CE均0。I23 merge后fresh r16/alpha16/dropout0.05/all-linear；单GPU、W&B online、batch1xacc4、lr5e-6 cosine、warmup0.03、wd0.001、BF16、685步、无中间checkpoint |
| 预检/实现 | 全量2,740/2,740 qwen3_nothink renderer、route与cutoff通过，material/retention=`1370/1370`，最长8,864<16,384。trainer/config/registry SHA256=`1111bd6d...1c7`/`45a35e6f...893`/`de6260b8...7c0`；step0必须通过fresh residual与merged I23 logits `max_abs<=1e-4` |
| 冻结门 | `configs/evaluation/i38_i23_material_i35_teacher_checkpoint_gate.json`。400行gate SHA256 `311b298f...ed41f`是I33/I32已冻结holdout确定性子集，与训练prompt交集0；material双向要求对I23 KL<=0.005、Top1>=0.99、gold delta>=-0.01；七个非material任务均须比I23更接近I35且Top1 agreement不退，聚合I35-KL至少下降10%。通过后再做itemic 0/60与精确拼接审计 |
| 唯一候选/提交边界 | 只允许完整685步fresh r16与I23 r64以1.0/1.0精确拼成r80/alpha80；不生成或比较中间checkpoint，不扫rank、dtype、插值、residual scale或LR。冻结门是机制否决器，不是线上估分器 |
| 正式训练 | 单GPU GPU0/W&B [`f92senkn`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/f92senkn)服务端`finished`；685/685步、runtime `1282.5365s`（21:22.53）、train loss `0.6860710371149717`，最终路由material/retention=`1370/1370`，无NaN/OOM。残差位于`checkpoints/i38_i23_material_i35_teacher_retkl_r16_v1/`；adapter/config SHA256=`1690bdde80a19b0910484c842c73f8855f8f7e1da5e68b8c5d8c67ff59e45b52`/`a4e7e6eea49063a1de99d3126d0fab6a63b6c6c0224b748ad3c0a35565b07978` |
| 精确组合包 | `submissions/i38_i23_material_i35_teacher_retkl_r80_step685_platform/`，仅`adapter_model.safetensors`与`adapter_config.json`；r80/alpha80、392 tensors、201,904,514 bytes；adapter/config SHA256=`74a86b037d48aa4ec88e88873f348ab66f0d8dc2423b27d6b4deddf7de3d5d94`/`4768770a600b8ab4c60eb04ad81a026a5ef8b6f2f8be79a8c4ab3192fa664d06`。审计`logs/package/i38_i23_material_i35_teacher_retkl_r80_step685_audit.json` SHA256=`3460fcc26f082d2287b550f72adfd13bb60c204f67eb3959e58e72ad5a21691c`，逐tensor恒等式`delta_combined = delta_parent + delta_residual`通过 |
| 冻结门结果 | 报告`logs/probe/i38_i23_material_i35_teacher_gate_v1.json` SHA256=`2e2e9f3a39164da09fa52bd3b25003bc47752f420aea4eb1be6b539fc3f69b13`；`teacher_forced_pass=false`。material desc2sid candidate→I23 KL/Top1=`0.0085404261/0.959375`，sid2desc=`0.0020746744/0.9722088`，均未达`0.005/0.99`；rec_video与rec_ad Top1保护失败。retention aggregate candidate→I35 KL ratio=`0.1988513`通过，但不足以覆盖硬失败；按预注册不跑itemic、不上传、不续训、不扫scale |
| 离线行为诊断 | 队友`offline_eval.py` v4报告`logs/offline_eval/i38_i23_material_i35_teacher_retkl_r80_step685_matrec32.json` SHA256=`da2a5b96859f208770464d7b7694eacbc87d973db383fc411f252ef4d34c6671`；mat fresh/train pass@64=`0.0938/0.1719`，rec video/prod/ad/live pass@64=`0/0/0/0.0312`。该报告为小样本生成行为回归，不能换算线上总分或推翻冻结门 |
| 当前状态 | **LOCAL_GATE_FAIL_NO_UPLOAD_I35_STEP548_RETAINED**；I-38 full包仅作复现审计，当前最高且默认交付仍为I-35 step548（线上`1.0344285849069457`）；不提交I-38、不从该失败分支继续训练 |

## 已完成正式训练与官方评测、线上回退后关闭：i37_strict_future_rec_r8_v1（I-37）

| 项 | 记录 |
|---|---|
| 目的/父模型 | 从当前线上最高I-35 step548 r112 `submissions/i35_r96_video_boundary_retkl_r112_step548_platform/`启动，只补其已观测增益来源推荐域；父adapter/config SHA256=`52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00`/`4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996` |
| 正式数据 | `assets/derived/processed/data_i37_strict_future_rec_v1.jsonl`，2,048行/21,427,796 bytes/SHA256 `2f663a7e4f477126d765a9c8e8aaa676caf1a014b0be206978f3c93f19e948b4`。O2严格未来video/ad各512；I-12全任务KL-only保持1,024，future:retention=`1:1`；T/E/model-rollout=0 |
| 标签约束 | video只取`play_done=1`且时间晚于最后video历史超过10分钟的current；ad只取`outer_loop_deep_target_pid`且严格晚于最后广告历史；完整target泄漏0，video/ad唯一target=`512/504`。这是官方源派生D，不称官方直发 |
| 训练目标/物理 | future答案体`0.10*CE + 16.0*I35-parent-KL`；retention只做`16.0*I35-parent-KL`。r112 merge后fresh r8/alpha8/dropout0.05/all-linear；单GPU、W&B online、batch1xacc4、lr5e-6 cosine、warmup0.03、wd0.001、BF16、512步 |
| 配置/实现 | config `configs/active/i37_strict_future_rec_r8_v1.yaml` SHA256 `371a1df3be3694b6fc4d79b4b1056393ed010b2e40690d5332dce46e5d17fdd1`；trainer `scripts/train/train_i37_strict_future_rec.py` SHA256 `568fb843c33d3c719d73915f58697c72a60e62b7c077c44f8b20a78d1f48ef61`；dataset registry SHA256 `0d7b8a0b038f4c51a7acbc2307a1afea70b0e000f1b28772a84c183b7d9e2bd5`；builder/audit SHA256=`b390446e...150f`/`c30f0940...6ad0` |
| 预检/产物限制 | trainer self-test通过；qwen3_nothink 2,048/2,048路由一致、最长8,228<16,384、截断0、精确512 optimizer steps。首轮step256保存校验器误把r8按r16检查，目录已移入`checkpoints/i37_strict_future_rec_r8_v1_failed_save_guard_step256/`，禁止resume、评测或打包；正式run只从父模型step0开始 |
| 正式训练 | 单GPU/W&B [`c2crod0w`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/c2crod0w)完成512/512步、1 epoch；runtime `1178s`（19:37.98）、train loss `1.2019`，最终路由`future_ce/retention_kl=1024/1024`，无NaN/OOM；full residual位于`checkpoints/i37_strict_future_rec_r8_v1/checkpoint-512/`，adapter SHA256=`8c25566cebeebfac8d48cd66ae771e100d12c6d2d7548f4f8afd19223f8cea69`，config为r8/alpha8 |
| 唯一提交包 | `submissions/i37_i35_strict_future_rec_r120_v1_platform/`，只含`adapter_model.safetensors`与`adapter_config.json`；r120/alpha120、392 tensors、302,829,416 bytes；adapter/config SHA256=`e91c773cad2324a74dbb6cf58ff13ff9e44fec85dfbbdc2104279cd33f767252`/`7838994a5e6a608b1d0250826a9e06a0a76f1b241c8ac9b2911b45077a6fcf1c`。package audit=`logs/package/i37_i35_strict_future_rec_r120_v1_audit.json`，逐tensor拼接核对通过，恒等式为`delta_combined = delta_parent + delta_residual` |
| 离线队友v4回归 | 报告`logs/offline_eval/i37_i35_strict_future_rec_r120_teammate_v4.json`，SHA256=`4698d1aea0dc3e714823839df1ed69c365adfbe806c6c7eeb28edf3dc3048ca3`，runtime `1722.1s`；mat fresh/train=`0.1052/0.1567`，rec video/prod/ad/live=`0.006/0.003/0.002/0.028`，action=`0.3030`（JSON=`0.945`、截断=`0.055`），topic=`0.0244`，world=`0.432`。相对I-35 step548 v4：video/live/world改善，action/topic回退；v4不作线上分数估计器 |
| 正式线上评测 | StreamLake任务`i37_i35_strict_future_rec_r120_V1_eval_20260723082428`，evalTaskId `eval-task-0yco4c-1784766273`、modelId `md-z9m20x-1784766072216356022`、`SUCCEEDED`、retryCount=0、duration=4,036秒。总分`1.02762520217381`；material/action/topic/video/prod/ad/live/world=`0.2452961672/0.1204379107/0.0394833175/0.0768/0.1292/0.1484/0.1089/0.1591078067`。相对I-35 step548逐项=`0/+0.0006705816/+0.0007260356/-0.0096/-0.0102/+0.0098/+0.0018/0`，总分`-0.0068033827` |
| 当前状态 | `ONLINE_EVAL_FAILED_BRANCH_CLOSED_I35_STEP548_RETAINED`；full包仅保留复现审计，不上传step256、不扫scale、不把I-37作为后续父模型 |

## 已完成正式训练与双点打包：i36_i35_user_expand_retkl_r16_v1（I-36）

| 项 | 记录 |
|---|---|
| 目的/父模型 | 从当前线上最高I-35 step548 r112 `submissions/i35_r96_video_boundary_retkl_r112_step548_platform/`启动，直接补其action/topic短板，同时保持I-35已取得的material、推荐和world能力；父adapter/config SHA256=`52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00`/`4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996` |
| 原始数据审计 | 用户提供15,023行生成数据，其中10,001 action、5,022 topic。审计发现action重复target、非历史顺序以及事件label/domain错配；topic还存在同事件重复SID和prompt示例污染。原始文件已移入D区，SHA256=`62f13962d4cfc0d4c2b591f2b9fd598d820e37ffcf5ac51433a0d5b9b8dd5ffa`，不直接训练 |
| 正式数据 | `assets/derived/processed/data_i36_i35_user_expand_retkl_v1.jsonl`，16,500行/159,306,138 bytes/SHA256 `2720746a2e8aa7804d519698ce9f2b127e9be2db1d4488e642e800a5337b692d`。用户CE=4,000 action+1,500 topic，I-35保持=material 2,500、video 2,000、prod/ad各1,750、living/world各1,500，严格user:retention=`1:2`；O1/E历史交集均0，T/E训练行0 |
| 清洗约束 | action去掉错域事件、target去重并按历史首次出现重排，保留1--56个历史内SID；topic严格3--5个时间递增事件、每事件单SID且日期/action与清洗后timeline逐字匹配。全部正式SID在登记O2 Pid2Sid映射中可验证；用户history在action/topic内部唯一且两者互斥 |
| 训练目标/物理 | user答案体CE + 0.10冻结I-35 KL；11,000保持行只做4.0冻结I-35 KL，最多128个均匀答案位置。r112 merge后fresh r16/alpha16/dropout0.05/all-linear；单GPU、W&B online、batch1xacc4、lr5e-6 cosine、warmup0.03、wd0.001、BF16、完整一轮4,125步 |
| 配置/实现 | config `configs/active/i36_i35_user_expand_retkl_r16_v1.yaml` SHA256 `2a2194ecef159786368c37c334166922dbfedcf3f366bec7353c073c79f43db3`；trainer `scripts/train/train_i36_i35_user_expand_retkl.py` SHA256 `e760cba91fe02553e1545d1fff8f3da303bfa4304974a0106f3c00a1db9ff9e3`；dataset registry SHA256 `dee54b0c94a12bf04edc6c99b45fe20cce9950033d0aa4f48e7d132b19f4ffce`；builder SHA256 `9b1b31e5341bb443e1ab25555da4ee8c763b3cdda34ce206a57b176ce5a41574`；formal audit SHA256 `eb426018525f9e3e1d682e1c89e5ca3dc8963b0a57c104911e1197324e464240`；launcher/package SHA256=`9eeee580...0a67`/`1475c444...c1a` |
| 失败边界 | 第一次W&B `onqds9a5`在step18被外部SIGKILL；cgroup `oom_kill=0`，日志无CUDA OOM/NaN且没有生成checkpoint。失败日志SHA256 `8d653797...affe`；该run禁止resume、打包或作训练证据 |
| 正式训练 | GPU0单张H100，W&B [`mmenbci2`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/mmenbci2)服务端`finished`；4,125/4,125步、完整1 epoch，runtime `7921.5533s`、train loss `1.0296107737223308`、samples/s `2.083`、steps/s `0.521`。最终microbatch路由retention/action/topic=`11000/4000/1500`，无NaN/OOM/CUDA错误；正式日志SHA256 `67e349b3...5b66` |
| r16 checkpoint | step2063 adapter/config SHA256=`00860000...8b1`/`1ad9650e...d41`；最终root adapter/config SHA256=`954db7b1...7da`/`1ad9650e...d41`。两者均为fresh r16残差，不可单独作为平台最终模型上传 |
| r128提交包 | `submissions/i36_i35_user_expand_retkl_r128_step2063_platform/`与`submissions/i36_i35_user_expand_retkl_r128_step4125_platform/`已完成。adapter SHA256=`f6a219c9...f590`/`9e936b46...127d`，config SHA256均为`daa3106d...7f3`；每包r128/alpha128、392 tensors、严格两文件、323,015,596 bytes，逐tensor精确等于固定r112 parent + 对应r16 residual。package audit SHA256=`a3e07724...4be3`/`ad95b6f8...a427`；只允许这两个点，不扫scale、LR或额外checkpoint |
| 线上结果/停止 | step4125任务`i36_i35_user_expand_retkl_r128_step4125_V1_eval_20260722183137`完成，耗时`1h5m22s`，总分`0.9865`；八项=`0.2453/0.1070/0.0331/0.0672/0.1292/0.1414/0.1035/0.1599`。相对I-35父模型，用户和推荐显著回退；step2063残差范数仅比step4125小约9.2%、方向余弦约0.9869，不作为无条件线上补测点 |
| 当前状态 | **STEP4125_ONLINE_FAILED_I36_CLOSED_STEP2063_HELD** |

## 线上新高、剩余一个剂量对照：i35_r96_video_boundary_retkl_r16_v1（I-35）

| 项 | 记录 |
|---|---|
| 目的/父模型 | 从当前最高单次线上观测1.0253的验收r96 `submissions/i19_world_external_r96_s875_platform/`启动，只推动懂物料desc2sid中父模型已经接近解码阈值的gold，同时用同一父模型保持其余能力；父adapter/config SHA256=`4fba17eb...078e`/`78b62143...1b64f` |
| 与I-34的区别 | I-34检验“I-23 Beam64命中而r96未命中”的teacher-only gap并已失败关闭；I-35不使用I-23正例或其gap，改为平台真实system/user renderer、O1 think/no-think同标签统一后E-clean全池，以及r96自身gold在Beam128的rank65--128边界。不是降低I-34准入线或续训失败checkpoint |
| Beam128 | pool主/运行分片=`1369/1`，ledger SHA256=`3c5845f6...3595`/`84385a13...a4b`，audit `b2fde171...18c7`；r96 Top128完整gold命中230、invalid 0。边界66条覆盖41个rank，共583个负例；first-divergence A/B/C=`254/188/141`，来源rank56--63/shared-prefix/fallback=`349/193/41` |
| 正式数据 | `data_i35_video_boundary_retkl_v1.jsonl` 2,740行/16,223,872 bytes/SHA256 `9c044e47...7100`；material/retention=`1370/1370`。material boundary/preserve=`66/1304`；retention action/topic/video/prod/ad/living/world=`207/206/206/207/206/207/131`。E/T/其它teacher正例均0，exact+mode交集均0 |
| 训练目标 | boundary只在评测实际解码的A/B/C三位置做first-divergence margin0.1 + 0.05 gold CE + 0.10 r96 KL；preserve material只做4.0 r96 KL；七任务retention最多96个均匀位置做4.0 r96 KL。EOS/domain不做CE，非gold模型候选永不作正例 |
| 训练物理 | r96 merge后fresh r16/alpha16/dropout0.05/all-linear；single GPU、W&B online、batch1xacc4、lr1e-5 cosine、warmup0.03、wd0.001、BF16、685步、seed19260835；adapter-only五等分checkpoint=`137/274/411/548/685` |
| 配置/实现 | config `configs/active/i35_r96_video_boundary_retkl_r16_v1.yaml` SHA256 `0a1491af...1ebd`；trainer `scripts/train/train_i35_video_boundary_retkl.py` SHA256 `927ebe80...4e4`；dataset registry SHA256 `87278e05...3ff`；builder SHA256 `c588721d...023e`；formal audit SHA256 `7f72ebee...ad1` |
| 完整预检 | 2,740/2,740行renderer/tokenizer/sidecar/route全通过，最长8,864<16,384；legacy retention精确为unmatched-think 2、internal-EOS 1，均与I-33上游逐字一致且只作parent KL，material仍严格解析 |
| 正式训练 | GPU0单卡，W&B [`0b4p3siy`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/0b4p3siy)服务端`finished`；685/685步，runtime `1594.5949s`，train loss `0.16564965731036055`，无NaN/OOM/CUDA错误。step-0 parent fingerprint max_abs=`0`，optimizer仅含392个fresh-r16 tensors；最终路由material/retention=`1370/1370`、boundary/preserve=`66/1304`。训练日志SHA256 `b60d03dc...f95c` |
| r16 checkpoint | step137/274/411/548/685 adapter SHA256依次为`76d13e18...21ca`/`10af32f0...45a`/`a8554bed...db6`/`3295b75b...1811`/`32650ad0...144f5`；统一config SHA256 `68a0504b...feb7`。root adapter与step685逐字节一致；五点仅作待合并残差，禁止单独上传 |
| r112提交包 | 五包均已构建；`submissions/i35_r96_video_boundary_retkl_r112_step548_platform/`已上传并评测，`submissions/i35_r96_video_boundary_retkl_r112_step411_platform/`是唯一剩余授权点。五包adapter SHA256依次为`0a3904de...9866`/`06c40f2d...557c`/`e26eb9be...5d58`/`52d945cc...2c00`/`4b25e5d3...6c00`，config SHA256均为`4f90d28f...9966`。每包r112/alpha112、392 tensors、严格两文件、282,645,380 bytes；A/B分块已逐张量精确验证为`delta_parent + delta_residual`。step137/274/685禁止上传。 |
| step548线上结果 | FORMAL评测`SUCCEEDED`，evalTaskId `eval-task-9nepj1-1784698215`，modelId `md-kqvjn7-1784697851382918150`，总分`1.0344285849069457`；八项material/action/topic/video/prod/ad/live/world=`0.2452961672/0.1197673291/0.0387572819/0.0864/0.1394/0.1386/0.1071/0.1591078067`。相对父模型原始`1.0252594563571054`逐项=`0/-0.002721708/-0.001093922/+0.0096/+0.0068/-0.0014/-0.0009/-0.001115242`，总分`+0.009169129`；当前81条同步评测中排名第一。 |
| 当前状态 | **STEP548_ONLINE_SUCCEEDED_STEP411_READY_FOR_CONTROLLED_UPLOAD** |

## 已完成预计算并按门停止：i34_r96_material_beam_margin_r16_v1（I-34）

| 项 | 记录 |
|---|---|
| 目的/父模型 | 从当前最高单次观测r96 `4fba17eb...078e`启动，只测试能否把I-23已观测的material完整SID beam命中迁到r96；I-23 `0e5fa9bb...c6b8`只作冻结构造评分器，不初始化policy、不提供非gold正例 |
| 结果前数据协议 | 原始预注册文件SHA256 `f3090a7354bd6da25c34474f15046ac0cf49dbedf28682af8719478b65142b49`；D(O1) desc2sid按full SID去重、`domain+s_a+s_b`分组，固定seed19260834先隔离256行gate再取1,024行train pool，排除全部登记prompt型E/holdout及I-30/I-33正式训练。结果回填后的同文件SHA256为`f5f589af17380082fae841ee3c14f0635d05ed65c39c7595e3d0377977b7d84a` |
| beam准入 | 同一O6 vLLM进程、BF16、平台no-think renderer、固定domain前缀、无约束beam64×3；hard gap严格为I-23含完整gold且r96不含。train/gate至少128/32，且train/gate四域至少16/4；不足即停止，不训练 |
| 唯一训练物理 | 准入后固定128条hard material+384条七任务r96 KL-only保持；fresh r16/alpha16/dropout0.05/all-linear，first-divergence margin0.1 + 0.1 gold CE + 0.02 parent KL，保持4.0 parent KL；单GPU/W&B online、batch1×acc4、lr1e-5 cosine、128步，只看step64/128 |
| 晋级/包装 | 先在256行冻结beam gate要求gap恢复至少50%、parent已有命中保持至少95%、总命中至少+8且四域不退，再过反向material/七任务/结构门；只允许scale1精确拼成r112/alpha112。任一门失败则不打包、不上传、不续训或扫参数 |
| 完整beam结果 | runner SHA256 `0af4943f...0cc`，vLLM0.12.0/BF16/seed42/no-think/fixed-domain/beam64x3。train账本1,024行/SHA256 `364cd069...eb38`：r96/I-23命中147/148，共有141、r96-only 6、I-23-only 7，gap四域=`2/0/3/2`。gate账本256行/SHA256 `2242b179...a2f`：两者均命中37，共有36、各自only 1，gap四域=`1/0/0/0`。审计SHA256 `bd3afac4...8c6` |
| 当前状态 | **PRECOMPUTE_GATE_FAILED_NO_TRAINING**；train/gate gap=`7/1`远低于`128/32`，且两边每域最低均为0。没有生成正式D训练数据/sidecar，没有W&B run、checkpoint、r112或提交包；staged trainer/config/launcher不得执行，I-34 v1禁止后验降门槛或扫LR/rank/margin/scale |

## 已完成并本地否决：i33_r96_material_desc2sid_retkl_r8_v1（I-33）

| 项 | 预注册记录 |
|---|---|
| 目的/父模型 | 从当前最高单次观测r96 `4fba17eb...078e`启动，只测试把material监督从双向256+256改为desc2sid 512能否提高物料项并保持其余七项；I-23 `0e5fa9bb...c6b8`只作冻结material KL teacher |
| 成对数据 | treatment `data_i33_r96_material_desc2sid_retkl_v1` 2,048行/SHA256 `7d6a1e4a...70fd`；未训练control SHA256 `812a6f71...7d40`。两臂仅原256个sid2desc槽不同，共享1,536条逐位置相同保持；8个登记prompt型E/holdout exact+mode交集0，T/E训练行0 |
| 训练物理 | O6+r96 merge后fresh r8/alpha8/dropout0.05/all-linear；material=`CE+0.5 I23 KL+0.1 r96 KL`，保持=`4.0 r96 KL`且最多96位置；单GPU、W&B online、effective batch4、lr2e-5 cosine、warmup0.03、wd0.001、bf16、512步、seed19260831 |
| 配置/实现 | config `configs/active/i33_r96_material_desc2sid_retkl_r8_v1.yaml` SHA256 `c47c7f5d...53bb`；trainer `scripts/train/train_i33_r96_material_desc2sid_retkl.py` SHA256 `a437e351...d383`；builder `scripts/data/build_i33_r96_material_desc2sid_v1.py` SHA256 `9c27a666...ed1b`；dataset registry SHA256 `4fb0ef52...e8bd` |
| 冻结门 | `configs/evaluation/i33_r96_material_desc2sid_checkpoint_gate.json` SHA256 `b6279bda6167d311cd0d139261f02d15f94aa8453cb24565d4ecef6e4b183b26`。新门720行/SHA256 `76acc6a3...f99`，另只取冻结I-32门的16行world；按64→128→256→512选最早全通过，sid2desc只作保护，不要求改善 |
| 包装 | 只允许scale1.0的FP32精确r96+r8拼接，得到r104/alpha104；不扫scale、dtype、SVD或未注册checkpoint，严格两文件且总大小必须低于400MB |
| 正式训练 | GPU0单卡，W&B [`io58fx1s`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/io58fx1s)服务端`finished`；512/512步、runtime927.6212s、train loss2.278109，最终路由retention/material=`1536/512`。根目录与step512 adapter逐字节一致，adapter/config SHA256=`90411c77...296c`/`a561f54b...cbce`；无NaN/OOM/CUDA训练错误 |
| 冻结门结果 | step64/128/256/512的desc2sid gold均值=`-0.002863/-0.003636/-0.002636/-0.002306`，改善率=`46.484%/49.609%/47.266%/53.125%`，向I-23 teacher的KL变化也全部为正，四点均失败；sid2desc Top-1仅`0.9768–0.9783`，多项保持任务亦低于0.99。world均为11/16，与parent持平。报告`logs/probe/i33_r96_material_desc2sid_gate_v1.json` SHA256 `91d4ab80...ef3e` |
| 当前状态 | **COMPLETE_LOCAL_GATE_FAIL_NO_PACKAGE_NO_UPLOAD**；`earliest_teacher_forced_pass=null`，故不运行itemic、不生成提交包。正式四个r8/r104点只作审计，禁止上传、resume、warm start或作父模型；未注册192/320/384/448按预注册在决策后删除；control从未训练 |

## 当前结论

- 当前最高单次线上观测是I-35 step548 r112：`1.0344285849069457`，八项=`0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591`。其直接父模型是已验收的`I19-world-residual` scale0.875 r96，原始线上`1.025259456`、后续同模型复测`1.025362611`；I-35相对两者分别高`0.009169129/0.009065974`。本次material仍为0.2453，净增益来自推荐四项合计；step411尚未评测。
- `I19-world-residual` scale0.875 r96现为I-35的父模型与上一最高点。严格两文件包已在`submissions/i19_world_external_r96_s875_platform/`按报告SHA256验收；实际训练parent是独立复现的`i13_repro_combined_r80_s875`（线上`0.986703844`），不是仓内原I-13 s875或s800的bitwise同一权重。其parent/residual及训练链仍待接收；交接和哈希见[`I19_WORLD_RESIDUAL_HANDOFF.md`](I19_WORLD_RESIDUAL_HANDOFF.md)。
- 本地优化候选已完成：`i19_local_world_residual_retkl_r16_ep1_s800` 使用仓内已验收的I-13 s800 r80 parent和`data_i19_local_world_residual_retention_v1`（3,146行，world/retention 1:1）训练fresh r16；第二次单卡/W&B run `xdzb35cp` 完成787步，最终路由world/retention=1573/1573，train_loss=0.5331。以residual scale 0.875合并为r96，包路径`submissions/i19_local_world_residual_retkl_r16_ep1_s800_combined_r96_s875_platform/`，adapter/config SHA256分别为`e31087a7...d41ac79`/`110b7457...a86d929`。它是待线上评测的本地候选，不能与报告版r96或1.0253混称。
- I-14首次运行因启动器绑定临时PTY，在step 1,886/1,971被会话生命周期中断；`rerun1`随后按相同数据与超参从O6和全新输出目录干净完成。E3于2026-07-14 15:20平台评测为0.9518，八项=`0.2453/0.1045/0.0387/0.0480/0.1292/0.1414/0.1080/0.1368`。它没有替换I-13的当前榜分；I-13是E3 r64+用户残差r16参数拼接路线，主办方口径为不鼓励而非禁止，但其构造血统不同，只能作业务榜分对照，不能作I-14纯O1单体r80的科学基线。仓内没有同协议、同血统的直接对照，E1/E2亦未线上评测。
- 2026-07-13下午平台修复评测不稳定问题；仓内可证实的协议切点位于I-10 E3（11:45）与I-11（16:40）之间。旧协议指纹为action `max_tokens=4096`、itemic单次beam64；固定协议指纹为action `max_tokens=1024`、itemic 7次`Race averaged evaluation`。日志两边都打印`version: v3.1`，故必须靠指纹分为`platform-pre-fix-v3.1`与`platform-stable-v3.1-20260713`，禁止跨协议作差。
- I-10完整线上轨迹已完成：使用 `data_seed_teacher_v1` 32,644行（O1全量99.4976% + 164条独立judge满分teacher标签0.5024%，规则标签0）从O6训练r64连续3-epoch cosine；E1/E2/E3=`0.9100/0.9680/0.9849`。该曲线只在旧协议内部有效；E3是固定协议待重评的桥接父模型。
- I-11是最早可证实的固定协议日志，单次线上0.9618；它不能与E3旧协议0.9849直接比较。继续同数据续训仍因缺少固定协议父分而不启动，但旧版“相对E3 -0.0231”结论撤销。
- I-12固定协议单次线上0.9768，八项为`0.2453/0.1206/0.0393/0.0672/0.1292/0.1316/0.1053/0.1383`。同协议相对I-11总分+0.0150、用户合计+0.0097、推荐合计+0.0038、world+0.0015；ad单项-0.0098。I-12现为I-13的同协议直接对照。
- I-13 s875保持E3 r64不变，仅将I-12 r16用户残差缩放到0.875；固定协议线上0.9978，八项为`0.2453/0.1183/0.0390/0.0960/0.1224/0.1316/0.1062/0.1390`。同协议相对I-12总分+0.0210、用户合计-0.0026、推荐合计+0.0229、world+0.0007；现为上一主模型。
- I-13 s800只将同一r16用户残差系数继续降到0.80，不重训；同一提交包固定协议两次显示分为`1.0037/1.0048`，首测八项=`0.2453/0.1163/0.0401/0.0960/0.1224/0.1372/0.1089/0.1375`，复测八项=`0.2453/0.1166/0.0401/0.0960/0.1224/0.1358/0.1089/0.1398`。两次差0.0011按合理抖动处理，均值1.00425、最好显示分1.0048；当前本地保留、可直接交付的固定协议主模型。
- I-14 E3固定协议单次线上0.9518。相对I-13首测逐项为`0/-0.0138/-0.0003/-0.0480/+0.0068/+0.0098/+0.0018/-0.0022`、面板总分差-0.0460，但这只回答“是否替换不同构造主模型”的榜分问题，不支持纯O1单体路线的因果否决。更接近的无参数拼接固定协议参考I-11为0.9618，I-14名义低0.0100；I-11仍使用164条teacher、从I-10 E3续训且rank不同，也不是干净基线。原始日志已复核为action1024+itemic 7次race-average、8/8完成、失败0，evalTaskId `eval-task-lfrrhq-1784013605`。
- I-16于2026-07-14 12:19 UTC通过持久启动器在单卡启动，W&B [`packufor`](https://wandb.ai/3120252125-/llmrec-2026/runs/packufor)，600/600正常完成。policy从保留的I-10 E3继续同一个r64 adapter，reference为O6显式加载并合并同一E3 adapter；不新建adapter、不拼接参数、不做同容量蒸馏。step200/400/600使四推荐域聚合raw chosen胜率从32.03%升到52.73%/57.03%/58.59%，action始终保持93.75%，但推荐gold平均token logp分别下降0.01185/0.02030/0.02149，全部超过0.01保护线。I-16按原门槛本地否决，不上传；这些读数只作机制和剂量证据，不估线上分数。
- I-17已在I-16 step400结果后、正式启动前注册，并于2026-07-14 12:57 UTC在I-16正常退出后顺序持久启动；W&B [`pfjlvm70`](https://wandb.ai/3120252125-/llmrec-2026/runs/pfjlvm70)服务端`finished`。它从原始I-10 E3重新开始，数据、beta和冻结E3 reference不变，只将峰值lr降至7e-7并用30步warmup后的constant日程把step200累计LR面积降为I-16 step200的74.8434%。step100/150/200全部满足量化保护线；按预注册“最早全通过”规则选step100，其推荐聚合raw chosen胜率32.03%→43.36%、gold平均token logp仅下降0.00327、action保持93.75%，itemic断裂0/60。step100固定协议线上0.9727，八项=`0.2453/0.1077/0.0380/0.0960/0.1156/0.1274/0.1044/0.1383`，低I-13 0.0251；相对I-12推荐合计+0.0101但用户合计-0.0142，总分-0.0041。直接父模型I-10 E3固定协议分仍缺失，因此不作DPO因果结论，桥接前不提交step150/200。
- I-18截断CoT修复E3固定协议线上0.9697，八项=`0.2453/0.1083/0.0382/0.0768/0.1190/0.1316/0.1089/0.1416`，低I-13 0.0281、低I-17 0.0030，未替换主模型。日志8/8完成、失败0；I-10 E3缺同协议桥，不能将该差值作CoT修复的净因果结论。
- I-19从固定协议最高分I-13/0.875原地更新同一个r80 adapter，不新增adapter层；D(O1)同题同域hard-negative子集2,688对按ad/prod/living/video=`768/768/768/384`重平衡。单卡W&B [`0bm73wt9`](https://wandb.ai/3120252125-/llmrec-2026/runs/0bm73wt9)完成75/75、退出码0；按冻结门槛选出的step25本地ad/prod/living/video raw mean margin均改善且itemic断裂0/60，但固定协议线上仅0.9763，八项=`0.2453/0.1181/0.0402/0.0864/0.1156/0.1246/0.1071/0.1390`。相对I-13总分-0.0215，其中用户合计+0.0010而推荐合计-0.0225；该hard-negative偏好门与线上推荐指标方向失配，I-19分支封板，step50/75不再提交。I-13 0.9978继续为主模型。
- I-20从保留的I-13/0.875原地更新同一r80，未加载I-19失败点。D(O1,O2.General) 12,260行由prod/ad正例与冻结I-13保持严格1:1组成；200/200步正常完成，W&B `1i153nai`。十档统一圈外诊断中step100是唯一做到prod/ad三SID宽候选`+3/128`且video/live `0/128`的点，但gold mean-logp仍为负漂移，故只视作线上实验候选。双通路行为与父模型近乎重合、itemic断裂0/60；严格两文件包已生成，I-13 0.9978在出分前仍是主模型。
- I-21在I-13同一r80内做topic answer-token低剂量CE，其余行用冻结I-13 KL保持；单卡W&B [`wjjymcj9`](https://wandb.ai/3120252125-/llmrec-2026/runs/wjjymcj9)完成150/150。六点统一诊断选step150：topic/action gold sum-logp相对父模型`+0.09127/+0.03478`，prod/ad Top-64覆盖`+3/128`，video/live为`-2/128`；这些只用于选点，不是线上分数预测。结构门itemic断裂0/60、action复读1/30；严格双文件包已生成，等待一次线上实验。
- I-22在I-13同一r80内完成world答案token低剂量CE，单卡W&B [`cohd8617`](https://wandb.ai/3120252125-/llmrec-2026/runs/cohd8617)完成150/150。46条未训练D(O2.General)选择集上，step25虽gold logp`+0.06628`且KL`0.00870`，但top-1掉3/46；step125的top-1不降且gold logp`+0.24041`，但KL`0.03381`超过预注册0.02。六点无一满足全部主门，按原规则本地否决，不跑后续保持门、不打包、不上传。
- I-23在正式训练前冻结允许角色I-10 E3，以batch=1对538组全部1,836个去重gold只评分最终答案token，最终选中83组（video/prod/ad/live=`32/35/11/5`）；随后从O6按I-10/I-18同一r64三轮物理干净训练。E3固定协议线上0.9915，八项=`0.2760/0.1099/0.0383/0.0576/0.1258/0.1400/0.1053/0.1387`。相对I-13总分-0.0063，其中material +0.0307、用户合计-0.0091、推荐合计-0.0275、world-0.0003；没有替换I-13，但成为固定协议最高无参数拼接单adapter。成功E3可作已登记action-retKL分支父模型；I-27 N4×K8 strict exact-hit yield已早停否决，条件RFT训练资格未生效。任何新RFT设计必须重新预注册；E1/E2及I-24/I-25失败点均禁止作父模型。
- I-23跨父残差四点线上为s500/s53125/s5625/s625=`0.9882/0.9757/0.9925/0.9866`，material/video依次=`0.2760/0.0576`、`0.2453/0.0864`、`0.2453/0.0864`、`0.2453/0.0768`。末次预登记二分s53125相对s5625总分`-0.0168`，且material仍为0.2453；未满足M=0.2760、video≥0.0768、总分>1.0048的冻结成功条件。整个I-23跨父残差scale轴已关闭，禁止任何更细scale；s750只作接近s800抖动中心的低价值备包。四点原始日志/evalTaskId未入仓前只作用户面板结果，不登记伪造ID。
- I-24已从成功I-23 E3原地完成200步action-only低剂量训练，W&B `f3ayytob`服务端finished，8个adapter-only剂量点齐全；但8/8均未通过启动前冻结的action硬门。step50是唯一action sum-logp均值为正的点（`+0.05031`），仍同时失败改善率`0.46875<0.55`、top-1 delta `-0.00080`和topic delta `-0.01223<-0.01`；所有点四域最大KL约`0.013–0.014>0.005`。整条分支按原规则本地关闭，不打包、不上传、不作父模型；条件算术点0.9999已被否决证据取代。
- I-25已在任何正式训练前冻结：只从成功I-23有效模型新建隔离r16 action residual，不使用I-12残差或I-24失败点；复用已登记6106行数据，action1752行答案体CE+弱parent KL，其余4354行只做parent KL，完整一轮1527步。最终gate SHA256 `53b5b375...f212`先单轴冻结最早action通过checkpoint，再在该点按固定scale升序取最小全保持解；不得二维后验回选。配置/trainer/启动器静态验收通过，尚未因准备文件而宣称训练或涨分。
- I-27按结果前冻结的512组video多正例、N4 reasoning×K8 item beam做s800正例-only RFT-lite yield smoke。用户明确授权后在共享低利用率GPU3单卡执行，vLLM显存上限25%，未修改或终止原进程；176组内5,632个候选结构有效率100%，但完整gold-set命中候选仅6、全QC接受组2（1.136%）。已完成与剩余组gold-count分布一致；要过128组门，剩余需126/336=37.5%，而当前接受率99.9%单侧上界6.215%，对应保守尾概率约`2.31e-63`。故节省资源早停，本N4×K8配方关闭；partial rollout只作D(O1;M-s800)诊断，不生成positive/final mix、不训练、不提交。
- I-28已完成正式单卡训练并在第一层冻结门本地否决。W&B `t3xega98`正常完成128/128步，512个microbatch严格路由proposal/retention=`128/384`，step64/128均为policy adapter且root与step128逐字节一致。prompt-disjoint E gate的128组/539个主gold上，step64 set-logsumexp均值变化`+0.00219566`但只改善61/128组；step128为`+0.01413218`、改善69/128组，二者均未达到`improved_rate>=0.55`（至少71/128）。按预注册不运行后续保持门或N4×K8，不打包、不上传、不作RFT/GRPO父模型；本实验零线上提交。
- I-29离线代理校准已完成，不是训练实验：固定首16个video多正例组，交叉`I23/s800 × legacy-system/canonical-user`；三格GPU生成各16行，25%显存上限下峰值21,537 MiB，生成阶段不接触gold，之后CPU scorer才评分。canonical下s800/I23 candidate prefix mass=`88/97`，主方向与已知线上video关系相反；group-any-ab=`2/1`虽同向但只是预登记secondary，exact均0。正式报告`logs/probe/i29_i23_s800_renderer_calibration_n16.json` SHA256=`4fc9ca83...ce25`，结论`COMPLETE_PROXY_CALIBRATION_FAIL_NO_TRAINING`：不做128组扩展、不训练I-23 video residual，也不将失败代理迁移到其他域。
- `s800_native_general_replay_r8_v1`已完成训练前冻结但尚未训练：以s800为唯一父/reference，129条task-fit-reviewed官方原生General各一次做full-response `CE+0.05 KL`，384条八任务均衡保持只做`4.0 KL`。fresh r8设计最终拼为r88；数据/trainer/config/平台入口/checkpoint门SHA256=`87097135...fddd2`/`cee8c258...8071`/`c95c6116...de2f`/`30531cfd...d47`/`970d169d...c416`。I19-world r16/r96结果已覆盖其当前优先级，现暂停启动；保留预注册资产，不把该设计称为最高方案复现。当前无checkpoint、无W&B run、无提交包。
- 撤回旧 I-09 规则数据资格：规则标签相对同源独立judge满分teacher参考的平均F1仅0.0429；匹配实际过滤条件的42条平均F1 0.0813且32条零交集。该teacher参考不是官方gold；`seed_o2_action_r64_lr1e4_ep3`因此在step16中止，W&B `sh96a1sq`，`checkpoints/seed_o2_action_r64_lr1e4_ep3/`无adapter且禁止resume。
- 当前最高单次线上观测与本卷可交付最高均为I-35 step548 r112 `1.0344285849069457`；其父模型I19-world r96原始/复测为`1.025259456/1.025362611`。固定协议最高无参数拼接单adapter仍为I-23 `0.9915`，最严格纯O1单体仍为I-14 `0.9518`。I-10 E3旧协议0.9849仍只能作旧轨迹父模型记录；禁止跨协议计算净增益。
- r64 同一训练轨迹 E1/E2 已线上评测：E1=0.8839，E2=0.9187；E3未评测且不再建议上传。本地门禁原先只选 E1、拒绝 E2，线上排序相反，门禁不再承担正向 checkpoint 排名。
- `riders_fk_clean_r64_ep3` 训练事实不变：GPU1 单卡，r64/α64、3 epoch、353 steps/epoch、总 1,059 steps；W&B online run [`6gyi8mzc`](https://wandb.ai/3120252125-/llmrec-2026/runs/6gyi8mzc)。E2 action 0.0981 创本账号新高，但 material E1/E2 均为6题。
- `i01_action_distill_r64_ep3` 已完成：3 epoch/1,047 steps，action 截断相对预登记比较对象 riders 减半但 F1 未涨，world v4 大幅回退；状态为本地否决、不上传。蒸馏正式累计 11,432,127 API token。
- `seed_scoremax_r32_ep1` 已完成：只用 `D(O1)`，35,558 行，单卡 1 epoch/740 steps。硬结构保险丝通过，但可见 action 0/5 闭合、5/5 触顶；material 签名 41/14 未进历史 8 题档。后验中点约 0.92，状态为本地否决、不上传。
- `seed_o2_action_r64_lr15e5_ep1` 已完成：`D(O1,O2)` 33,644 行，O2 唯一 action 行 1,164（3.4598%），r64/alpha64、lr1.5e-4、单卡 1 epoch/710 steps。itemic 结构通过，但 action 0/5 闭合、material 39/13，状态为本地否决、不上传。
- E2 的本地 checkpoint 存在，但 `submissions/riders_fk_clean_r64_e2_platform/` 不存在；平台日志只记录临时 `/tmp/eval_model/merged`。在缺上传 manifest 时，不能声称平台工件哈希已由本地 adapter 哈希证明。
- 旧实验的中间 checkpoint、optimizer、失败 checkpoint 和 merged 工作副本已于 2026-07-11 删除；本轮用户批准的 r64 E1/E2/E3 例外已单独列入下表。

## 已完成训练并打包待线上评测：i19_local_world_residual_retkl_r16_ep1_s800

| 项目 | 记录 |
|---|---|
| 父模型/数据 | 仓内 I-13 s800 r80；`data_i19_local_world_residual_retention_v1` 3,146 行，world/retention=1,573/1,573，数据 SHA256 `ef64cb72...b7484b` |
| 训练 | fresh r16/alpha16；world full-response CE+0.05 parent KL，retention 0 CE+2.0 parent KL；单卡 GPU0、W&B `xdzb35cp`、787/787 steps、最终 train_loss `0.5331`；最终路由契约 PASS，step-0 parent fingerprint PASS |
| 工件 | residual root及同权重epoch点`checkpoints/i19_local_world_residual_retkl_r16_ep1_s800_v2/`、`checkpoints/i19_local_world_residual_retkl_r16_ep1_s800_v2/checkpoint-787/`，adapter/config SHA256均`5c9f452b...137538`/`1bd5e4c5...b2ea0`；r96 合并 `checkpoints/i19_local_world_residual_retkl_r16_ep1_s800_combined_r96_s875/`；严格两文件包 `submissions/i19_local_world_residual_retkl_r16_ep1_s800_combined_r96_s875_platform/` |
| 首次失败输出 | `checkpoints/i19_local_world_residual_retkl_r16_ep1_s800/`仅残留`trainer_log.jsonl`，无adapter/checkpoint；W&B `s3e4i3tz`因world路由前缀误判在约50%终止。禁止resume、禁止作父模型，不把该日志目录计为成功checkpoint |
| 哈希/状态 | residual adapter/config `5c9f452b...137538` / `1bd5e4c5...b2ea0`；r96 adapter/config `e31087a7...d41ac79` / `110b7457...a86d929`；392 tensors，scale=0.875；**COMPLETE_LOCAL_PACKAGED_AWAITING_ONLINE_SCORE**，不声称复现报告版 1.0253 |

## 已冻结但暂停启动：s800_native_general_replay_r8_v1

| 项目 | 记录 |
|---|---|
| 目的/父模型 | 只测试129条官方原生静态General能否在不损伤其余七项的情况下补少量world；唯一父/reference为当前本地保留模型`s800` adapter/config SHA256=`bb86eb8...f63c6`/`e3c3ace0...c4ac0`，线上同包显示`1.0037/1.0048`、world `0.1375/0.1398` |
| 正式数据 | `data_s800_native_general_replay_v1` 513行/SHA256 `87097135...fddd2`：General 129各一次，八个保持任务各48；旧world_zh、world_zh_ext、68数学MC、T/E/model rollout均0。builder/audit/route manifest SHA256=`ca9e1ee5...7130`/`96b6c636...5e04`/`630ec7d0...04af` |
| 损失/实现 | LLaMA-Factory先合并s800 r80，再建fresh r8；`disable_adapter()`为exact merged-s800 reference。General full target `CE+0.05 KL`，保持full target `4.0 KL-only`。trainer SHA256 `cee8c258...8071`按exact qwen3 target hash fail-closed路由；step0 residual/parent logits、最终129/384路由及32/64/96/129保存序列均有运行时硬闸 |
| 训练物理 | 单GPU，batch1×acc4，lr1e-5，warmup8后constant，129 optimizer steps；r8/alpha8/dropout0.05、target all。config/registry/GLM容器入口 SHA256=`c95c6116...de2f`/`571ea552...7c8`/`30531cfd...d47`；W&B必须online并由Training Task内secret实时verify |
| 冻结选点门 | `configs/evaluation/s800_native_general_replay_checkpoint_gate_v1.json` SHA256 `970d169d...c416`。32→64→96→129取最早全通过：23条scoring E的A-D/all-vocab正确数不低于5/3、gold logp均值不退且至少12/23改善；256条八任务保持的overall/task KL、row p95及各任务parent argmax agreement均过线；60条itemic断裂0。四点全失败即关闭，禁止scale、续训、阈值放宽或回混其他world数据 |
| 输出与拼接 | 预期仅生成fresh residual checkpoint 32/64/96/129；残差不能独立提交。最早通过General目标门与八任务保持门的点，才允许用现有exact additive脚本拼成`s800 r80 + residual r8 = 单个r88`；主办方口径是不鼓励模型融合而非禁止，必须完整披露血统 |
| 当前状态 | **FROZEN_BUT_SUSPENDED_SUPERSEDED_BY_I19_WORLD_HANDOFF**；开发机未训练、无checkpoint、无W&B run。I19-world工件接收、验收和稳定性判断完成前禁止创建平台训练任务；若未来恢复，仍必须使用原冻结secret/挂载/entry与选点门，不得后验改设计 |

## 已完成并本地主门否决：i28_i23_rec_multigold_proposal_retkl_v1（I-28）

| 项目 | 记录 |
|---|---|
| 目的/父模型 | 只改善成功I-23 E3的video proposal分布；policy与冻结reference都从`submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform`开始，adapter/config SHA256=`0e5fa9bb...c6b8`/`b3f2a1b5...e7e7`。I-23 E1/E2、I-24/I-25失败点和I-27 rollout均禁止作父或训练数据 |
| 正式数据 | `assets/derived/processed/data_i28_video_multigold_proposal_retkl_v1.jsonl`，`D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General)`，512行/SHA256 `f74a22b96bf7651a97cdb5b578f346eea0109045329efafb80092f91756fbb6a`；128 proposal（64组×2）+384 retention，严格1:3；builder/audit SHA256=`98b41293...bf11`/`b0036b97...3be2`；T/E/O3目标元数据/model rollout/负例训练行均0 |
| 损失/实现 | proposal只在`video-domain+a+b+c+EOS`做CE并加0.2×冻结I-23 forward KL；retention只做4.0×冻结I-23 KL、最多128位置。修复后trainer `scripts/train/train_i28_multigold_proposal_retkl.py` SHA256 `72fa991433698cd7f705a700d7d72c467356e5530e1f625547e84a6ecaef7253`：reference从post-upcast policy逐tensor复制并继续要求bit-identical/首logit一致；另强制active/enabled/unmerged、optimizer只含policy、policy-only保存无reference子目录、embedding/head冻结和最终128/384路由签名 |
| 配置/训练物理 | `configs/active/i28_i23_rec_multigold_proposal_retkl_v1.yaml` SHA256 `f6086e468b91a9df93804d7ab6421549bb8767a07522a1340aac9939772f077a`；I-23同一r64原地更新，单GPU、batch1×acc4、lr1e-7、warmup16后constant、128步/完整一轮、save64/128。哈希锁launcher SHA256 `74842f9be793f318eef4b2894d9d9f773dab37f04478ad53b14d1fdc5eb5146e`；W&B [`t3xega98`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/t3xega98) online并正常finished |
| 训练完成 | 128/128步、runtime1153.78s、train loss5.94656、退出码0；512 microbatch最终路由proposal_video/retention=`128/384`，step0 policy/reference logits max_abs=0。训练日志`logs/train/i28_i23_rec_multigold_proposal_retkl_v1.log` 93,410 bytes/SHA256 `8bdff61bf40c8e143215a3b34c234cc5aba95bc3d0f95c7992013446669ea01d`；无NaN/Inf/OOM/Traceback/route/fingerprint/save错误 |
| 产物/允许角色 | step64 adapter/config SHA256=`0683593b7d97ee4979753862ddd5f47c5f9a8b290a8b627ff30fb0c491c97376`/`33b749b6290e11715cf6a8612a0a52cb1fac59baa131c83b3ba1ed49dd6933b5`；step128=`06f43f197a91cb7fd4057ba270381708d1735939451f2fed082ac508b0dc7eec`/同config，根目录与step128两文件逐字节一致。无optimizer/scheduler/scaler/RNG/reference权重；门禁失败后两点只作审计，不得打包、提交、warm start或作RFT/GRPO父模型 |
| 评测实现机械重冻结 | 首次调用在模型加载前因PEFT对相同`target_modules`集合的JSON顺序不同而退出；修复为集合语义并拒绝重复。第二次在任何forward前发现原循环只做到batch内parent-first，主动终止于加载parent adapter，并改为完整parent→step64→step128。科学指标/阈值均未改；最终gate/evaluator SHA256=`fd777bd3594139dd02008e4587ad79dd69496a3885869f0ef0405ea407bc7e82`/`d8e5da2d4fdee326d00b4f1f246ea9f926a0f5a8ee5eeb38b392e3ac65095c55` |
| 冻结主门 | 报告`logs/probe/i28_multigold_set_path_v1.json` 188,781 bytes/SHA256 `96bd457768a0f406b7215b05f9e8a0333ac5bcb12d159438831f67b501e51dee`，status=`COMPLETE_NOT_AN_ONLINE_SCORE_ESTIMATE`。step64 set-LSE delta mean/improved=`+0.00219566`/`61/128=0.4765625`；step128=`+0.01413218`/`69/128=0.5390625`；冻结要求均值`>=0`且改善率`>=0.55`，两点均FAIL。best-gold只作诊断：step64/128均值变化=`-0.01204291/+0.01320961` |
| 状态 | **COMPLETE_LOCAL_REJECT_STAGE1_NO_PACKAGE_NO_UPLOAD**；两个候选均停止于set-path第一层，因此保持门、结构门和N4×K8均未运行。I-28分支关闭，不放宽阈值、不续训、不在线评测、不消耗全队约3发/日配额。旧失败run `u9d24puh`/`bbh3x6qr`继续禁止resume |

## 已完成并本地否决：i23_action_ansretkl_v1（I-24）

| 项 | 预注册记录 |
|---|---|
| 目的/父模型 | 只补I-23相对I-13的action缺口，同时守住material第9题及I-23的prod/ad优势；policy/reference均从成功I-23 E3单adapter SHA256 `0e5fa9bb...c6b8`开始，E1/E2及任何失败checkpoint均不用 |
| 数据 | 复用已登记`data_user_residual_retention_v1`，D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General) 6,106行、SHA256 `bd947aad...b08f0`；action 1,752（28.6931%），topic+非用户4,354（71.3069%）；T/E/model rollout训练行均0 |
| 损失/保持 | action答案体CE+5x frozen-I23 KL；其余全为50x KL-only。每行KL位置=`96个均匀全trace ∪ 48个均匀答案体 ∪ 尾16 content`，排序去重后最多160；闭合符/EOS为2x CE，LoRA dropout使用matched CPU/CUDA RNG |
| 配置/实现 | `configs/active/i23_action_ansretkl_v1.yaml` SHA256 `0903f235...938de`；trainer `scripts/train/train_i23_action_retkl.py` SHA256 `bfeea1dc...270a7`；online/detached launcher SHA256 `bab895d4...3de7` / `96a26d5e...e824`。self-test、AST、JSON/YAML、`bash -n`均PASS |
| 训练物理 | I-23同一r64/alpha64原地更新，不新增adapter/rank；单GPU、W&B online、lr7.5e-8、warmup25后constant、effective batch4、最多200步；25步一档且只保存adapter |
| checkpoint门 | `configs/evaluation/i23_action_ansretkl_checkpoint_gate.json` SHA256 `e898ded7...0271`在启动前冻结。按25→200取**最早全通过**：action gold-path与生成F1/JSON不退、topic轻微漂移受限、material各向KL≤0.005且top1≥99%、video/prod/ad Top-64不退、live最多-1/64、itemic断裂必须0；E只作选择/否决，不训练、不估线上分 |
| 线上成败 | 只有material仍为0.2760且总分超过I-13 0.9978才算成功；material回落0.2453即关闭分支。静态中心：只完整补回action 0.0084且其余不变时`0.9999`，不是校准预测 |
| 持久启动 | 2026-07-16 04:47 UTC在物理GPU0 `GPU-d3c522d6-ed0f-2579-01cd-2d97da749980`启动；detached PID `3709796`由PID1接管、SID同PID且无TTY，日志/PID/退出码为`logs/train/i23_action_ansretkl_v1.{log,pid,exit_code}` |
| W&B/训练 | W&B [`f3ayytob`](https://wandb.ai/3120252125-/llmrec-2026/runs/f3ayytob)服务端`finished`；6,106 examples、200/200 steps、40,370,176 trainable params。step0 policy/reference指纹`max_abs=0.00000000`；runtime1,312.0071s、train loss0.6484887、退出码0。800个microbatch action/retention=`226/574`，末批retention KL=`0.00138263`；无NaN/Inf/OOM/Traceback |
| 预注册输出 | 根目录`checkpoints/i23_action_ansretkl_v1/`；adapter-only剂量点`checkpoints/i23_action_ansretkl_v1/checkpoint-25/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-50/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-75/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-100/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-125/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-150/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-175/`、`checkpoints/i23_action_ansretkl_v1/checkpoint-200/`均完整保留为失败剂量轨迹，任何一点均不得作父模型或上传 |
| 首个剂量点 | step25已正常保存，adapter/config SHA256 `1fd38fa6...7fc3` / `4442cee9...95c2`；r64/alpha64，目录内无optimizer/scheduler/RNG。前100个microbatch action/retention=`32/68`，第100批retention KL=`0.00061204`；只证明训练机制正常，不代表已过离线门 |
| 完整轨迹 | step25/50/75/100/125/150/175/200 adapter SHA256依次=`1fd38fa6/53725fbf/c2308a42/df937aa3/9afa5b45/9e77d3fd/3fae0ed1/0476ff59`；config均`4442cee9...95c2`，全目录无optimizer/scheduler/RNG；根目录与step200逐字节一致。训练日志SHA256 `96e76b58...ef04` |
| 冻结门结果 | action/topic报告`logs/probe/i24_action_topic_path_20260716.json` SHA256 `90b9e946...3d24`，rec报告`logs/probe/i24_rec_gold_path_20260716.json` SHA256 `a4b26c23...c3f`。8点action改善率最高仅15/32，门为18/32；step50虽均值`+0.05031`，仍为top-1 `-0.00080`、topic `-0.01223`。四域最大KL各点约`0.013–0.014`，均超过0.005 |
| 当前状态 | **COMPLETE_LOCAL_REJECT_NO_PACKAGE_NO_UPLOAD**；8/8主门失败，不运行后验生成选点、不放宽门槛、不作后续父模型 |

## 训练前已冻结：i23_actionres_r16_ansretkl_ep1（I-25）

| 项 | 预注册记录 |
|---|---|
| 目的/父模型 | 只修I-23的action缺口；成功I-23 r64先合并进O6并完全冻结，再从零新建r16 residual。禁止使用I-12残差、任何I-24 checkpoint或其他失败工件 |
| 数据/路由 | 复用`data_user_residual_retention_v1`，D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O2.General) 6,106行、SHA256 `bd947aad...b08f0`；action 1,752（28.693089%）只做答案体CE，topic 1,301+其余3,053=4,354（71.306911%）只做parent KL；T/E/model rollout训练行0 |
| 损失/训练物理 | action答案体CE权重1，真实闭合符/EOS 2x，并在完整response trace加0.05 I-23 KL；非action仅2.0 I-23 KL。r16/alpha16/dropout0.05、lr5e-5 cosine、warmup46、weight decay0.001、effective batch4、完整1ep/1,527步；单GPU、W&B online |
| 配置/实现 | config `configs/active/i23_actionres_r16_ansretkl_ep1.yaml` SHA256 `da46f0b1...5dbd`；trainer `scripts/train/train_i23_actionres_retkl.py` SHA256 `0071a088...7c02`；online/detached launcher SHA256 `b76392a8...fcbd` / `f898bd44...26da`。self-test、`py_compile`、YAML、`bash -n`、原始6106行路由计数和workspace audit均PASS |
| 冻结门 | `configs/evaluation/i23_actionres_r16_checkpoint_gate.json`最终SHA256 `53b5b375...f212`。Stage1按250→500→750→1000→1250→1527仅用action机制门冻结最早全通过训练点；Stage2只在该点按0.25→0.375→0.5→0.625→0.75取最小material/topic/四推荐域/生成/结构全通过scale。任一阶段失败即关闭，禁止回选checkpoint、新增scale或放宽阈值 |
| 预注册输出 | 根目录`checkpoints/i23_actionres_r16_ansretkl_ep1/`；训练保存`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-250/`、`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-500/`、`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-750/`、`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-1000/`、`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-1250/`、`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-1500/`；成功完成后launcher从根adapter/config原子生成并逐字节复核`checkpoints/i23_actionres_r16_ansretkl_ep1/checkpoint-1527/`。step1500只作已登记轨迹，不进入Stage1候选序列 |
| 线上边界 | 只完整补回action 0.0084且其他七项完全不动的条件算术上限为0.9999；恢复75%仅追平I-13 0.9978。任何离线门都不保证material第9题。最终I-23 r64+r16 residual拼成r80属于参数拼接；规则是不鼓励而非禁止，但不得误称为直接单adapter训练 |
| 正式启动 | 2026-07-16 06:00 UTC在物理GPU0 `GPU-d3c522d6-ed0f-2579-01cd-2d97da749980` detached启动；父PID `3763410`由PID1接管且无TTY，日志/PID/退出码=`logs/train/i23_actionres_r16_ansretkl_ep1.{log,pid,exit_code}`；W&B [`x8fl7r86`](https://wandb.ai/3120252125-/llmrec-2026/runs/x8fl7r86)。运行时确认6,106 examples、1,527 steps、10,092,544 trainable params，step0 residual-disabled I-23 fingerprint `max_abs=0` |
| 正式训练结果 | Trainer完成1,527/1,527，runtime `2794.8687s`、train loss `0.4148307616`、最终route action/retention=`1752/4354`；W&B服务端`finished`，根与step1527 adapter SHA256均`676d80fd...b67b`，日志SHA256 `3cbefe68...693f`。无NaN/OOM/Traceback，GPU0已回基线 |
| 保留冲突/恢复锁 | Trainer终点额外保存step1527后，`save_total_limit=6`自动轮转删除已保存的step250，外层后置检查因缺step250返回1；这是工件postcondition失败，原退出码不得改写。恢复plan `configs/evaluation/i25_step250_deterministic_recovery_plan.json` SHA256 `94da5c04...a1ec`锁定同GPU从O6+I-23干净重放、原1527步scheduler horizon、callback在step250后停训；1000 microbatch route必须=`278/722`，adapter/config必须逐字节命中删除前记录的`4af72967...28e9`/`6c127a34...e976a`，否则不安装并关闭I-25 |
| 恢复结果 | 同GPU干净重放保持scheduler horizon1527并在step250精确停止，1000 microbatch route=`278/722`；W&B `k3s0oig9` finished。但adapter SHA256 `9198538f...68d9`未逐字节命中删除前记录的`4af72967...28e9`，身份硬闸exit1且未安装；receipt/正式step250均不存在，GPU0回基线。结果`configs/evaluation/i25_step250_deterministic_recovery_result.json` SHA256 `fad1adf5...29bb`同时纠正plan内部未来时间字段：plan mtime 06:57:47，GPU进程07:03:45启动 |
| 实战选点纠正 | 用户明确缺step250不影响竞赛选点；停止把完整checkpoint轴当作评估前提，直接横评现存step500/750/1000/1250/1500/1527。恢复失败仅保留为历史证据，不再作为I-25否决理由，也不据此自动发起重训 |
| scale1横评 | `logs/probe/i25_practical_action_topic_scale1_20260716.json` SHA256 `485b64c1...41ef`。六点action gold sum-logp均负：step500/750/1000/1250/1500/1527=`-6.4662/-8.6366/-9.7929/-8.6128/-8.4558/-8.6734`；step500最低伤也仅3/32改善，故不把任何scale1点送入保护域评估 |
| step500低scale | 训练同口径128条action报告`logs/probe/i25_step500_action_scale_curve_20260716.json` SHA256 `2b9114d7...2fb`：scale0.125/0.25/0.375/0.5的CE均值变化=`-0.00115/-0.00361/-0.00717/-0.01112`，说明残差确实学到答案体机制；但完整路径报告`logs/probe/i25_step500_low_scale_action_topic_20260716.json` SHA256 `56e8e213...4f1`中对应action均值=`-0.7413/-1.6898/-1.9104/-3.2683`，方向冲突 |
| 额外生成诊断 | step500 scale0.125不在原Stage2轴内，只作实战诊断。I-23→候选action F1=`0.357248→0.339625`（`-0.017624`），同时JSON合法率=`0.84375→0.90625`、截断率=`0.15625→0.09375`、平均重复项=`104.97→63.78`；证据冲突。对比报告`logs/probe/i25_step500_s0125_action_generation_compare_20260716.json` SHA256 `b8dd137b...3fff`，但仅32条/单seed且使用旧v4 action 4096上限，不能当当前平台方向门 |
| 门禁校准纠正 | 原冻结门字面结果仍为`ORIGINAL_GATE_FAIL`，不得事后篡改；但action离线历史校准仅Spearman `0.039`、方向50%。同一32题gold-path甚至把I-23（`-129.84/0.6028`）排在I-13（`-136.98/0.5818`）之前，线上action却为`0.1099<0.1183`；r64 E1/E2也出现本地JSON/截断排序与线上action、总分完全反向。因此原门失败不等于线上DOWN |
| 当前状态 | **ORIGINAL_GATE_FAIL_SCORE_DIRECTION_ABSTAIN_NO_HARD_SAFETY_FAILURE**；当前1024下scale0.75是0.5/0.625/0.75中结构最接近I-23的一点（JSON29/32、截断3/32），material双向KL约0.002且推荐保护无硬红线，但这些都不能证明线上上涨。I-25优先级低于完整保留I-23的`s625`首投，当前不打包、不上传、不作父模型。I-26重训未获用户授权，不得自动启动 |

## 已完成并线上0.9763：i13_s875_dpo_rec_balhard_lowdose_v1

| 项 | 完成记录 |
|---|---|
| 目的/父模型 | 直接优化固定协议最高分I-13/0.875；policy与冻结reference均为adapter SHA256 `71bc3c2c...ffd5b`。在原r80内更新，不叠加新adapter；I-13已有参数拼接来源不因此消失，规则仍是不鼓励而非禁止 |
| 数据 | `data_i13_rec_balanced_preference_v1_train`，D(O1) 2,688对，SHA256 `baf8a825...d2a8f`；从已登记O1 hard-negative训练集确定性抽取，ad/prod/living/video=`768/768/768/384`，标签改写0，O2/T/E/teacher/model-rollout均0 |
| 配置 | 启动时`configs/active/i13_s875_dpo_rec_balhard_lowdose_v1.yaml` SHA256 `467965b6...5800`；完成后只加禁重启头，当前SHA256 `e4633b6a...e195`。r80/alpha80、beta0.1、effective batch8、lr4e-7、warmup10、constant、75步；adapter-only每25步保存 |
| 训练 | GPU1单卡；75/75，runtime361.5s，train loss0.6637，退出码0，无NaN/Inf/OOM/Traceback；W&B [`0bm73wt9`](https://wandb.ai/3120252125-/llmrec-2026/runs/0bm73wt9)，日志SHA256 `4841bc3d...6a2f` |
| 冻结门槛 | 预注册`configs/evaluation/i19_i13_balhard_lowdose_checkpoint_gate.json` SHA256 `718f2b54...1b3`；结果`i19_i13_balhard_lowdose_checkpoint_gate_result.json` SHA256 `27bd03f8...c35`。同一E留出只作机制/漂移门，不估线上分数 |
| 轨迹 | step25/50/75推荐raw chosen胜率=`34.765625%/38.671875%/40.234375%`（父32.03125%）；action raw/normalized三点均保持`93.75%/85.9375%`；推荐gold mean-logp下降=`0.00107086/0.00043452/0.00449446` |
| 选点 | 按启动前“最早全通过”规则选step25，adapter/config SHA256 `e6ebe4dc...fb2e0` / `a7114c7d...caa4`；结构门itemic断裂0/60=`PASS`。step50/75虽数值过线，仍只作剂量轨迹，不上传、不作父模型 |
| 线上 | `i13_s875_dpo_rec_balhard_lowdose_v1_step25_V_eval_20260715153057`；2026-07-15 15:31:01平台记录，1h6m57s；总分0.9763；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1181/0.0402/0.0864/0.1156/0.1246/0.1071/0.1390` |
| 对I-13 | 总分-0.0215；逐项=`0/-0.0002/+0.0012/-0.0096/-0.0068/-0.0070/+0.0009/0`；用户合计+0.0010、推荐合计-0.0225。离线四域hard-negative margin全部改善却线上三域下降，证明该门不能再作正向选点依据 |
| 提交包 | `submissions/i13_s875_dpo_rec_balhard_lowdose_v1_step25_platform/`严格两文件且与step25逐字节一致；已完成本次固定协议评测 |
| 线上日志 | `logs/eval/i13_s875_dpo_rec_balhard_lowdose_v1_step25_20260715.log`，2,764,972 bytes，SHA256 `6c57f8fbf98cc965f28d6508154d8452508f5ceab6d1b012c98c99d34be87099`；action1024、itemic 7次race-average、8/8完成、失败0；evalTaskId `eval-task-40k2ig-1784100661` |
| 状态 | **COMPLETE_ONLINE_0.9763_REJECT_BRANCH**；不提交step50/75，不作后续训练父模型；本次评测后s875 0.9978继续为主模型，当前主模型已为s800（同包1.0037/1.0048） |

## 已完成训练并打包step100：i13_s875_posrec_pa_ansretkl_v1

| 项 | 完成记录 |
|---|---|
| 目的/父模型 | 定向恢复I-13的prod/ad，同时保持video及六个非目标子项；policy与冻结reference均从I-13 adapter SHA256 `71bc3c2c...ffd5b`开始，不使用I-19 checkpoint，不新增adapter/rank |
| 数据 | `data_i20_prod_ad_positive_retkl_v1`，D(O1,O2.General) 12,260行、SHA256 `0c08b8f5...62fa`；prod/ad positive 6,130与parent-KL retention 6,130严格1:1；负例、推荐伪标签、teacher/model rollout、T/E均0 |
| 损失 | prod/ad仅最终domain+3 SID token做gold CE并加0.20 frozen-I13 KL；material/action/topic/video/live/world只做4.0 frozen-I13 KL，最多均匀128个answer位置；matched RNG消除LoRA dropout造成的伪KL |
| 配置/训练 | `configs/active/i13_s875_posrec_pa_ansretkl_v1.yaml` SHA256 `e5bd8220...70d1`；GPU7单卡、r80原地更新、lr2e-7、warmup20后constant、effective batch4。step0 policy/reference最大差0；200/200步、runtime1082.13s、train loss5.75488、退出码0、W&B [`1i153nai`](https://wandb.ai/3120252125-/llmrec-2026/runs/1i153nai) |
| 轨迹 | `checkpoint-{20,40,60,80,100,120,140,160,180,200}`全部adapter-only、无optimizer/scheduler/RNG；adapter SHA256依次为`90895b7d`/`dd98d0eb`/`ecfdc855`/`8dc3af97`/`43199b5f`/`dcc5dfa1`/`e0559f6e`/`7456b17c`/`06ccedc1`/`bc084b69`；根目录与step200逐字节一致 |
| 保留路径 | 根目录`checkpoints/i13_s875_posrec_pa_ansretkl_v1/`；剂量点`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-20/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-40/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-60/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-80/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-100/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-120/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-140/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-160/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-180/`、`checkpoints/i13_s875_posrec_pa_ansretkl_v1/checkpoint-200/`；均只允许本轨迹诊断，正式候选仅step100 |
| 统一金标路径诊断 | `logs/probe/i20_gold_path_20260715.json` SHA256 `8662664c...771`；固定E样本每域64。step100相对父模型prod/ad token-rank≤64合计`+3/128`、video/live `0/128`，但prod/ad gold sum-logp均值变化`-0.01204`、video/live `-0.04055`；只支持选它进入生成安全门，不是涨分证明 |
| 双通路行为门 | 父/step100均按v4参数跑相同64×4域；旧dev在I-13上video/prod/ad命中均0，确认不能作正向分数门。step100相对父的copy-direct变化video/prod/ad/live=`0/+0.0049/+0.0015/+0.0010`，distinct-s_a变化=`-0.06/+0.10/-0.01/+0.18`，未见生成坍塌；日志SHA256 `9de45e6b...fca0` / `74d981f1...e13` |
| 结构门/选点 | step100临时merge后itemic断裂0/60=`PASS`；action复读3/30、选择题格式7/8仅作diagnostic。日志`logs/precheck/i20_step100_20260715.log` SHA256 `7b5ed2ed...bb69`。其余step只保留为剂量轨迹，不提交 |
| 提交包 | `submissions/i13_s875_posrec_pa_ansretkl_v1_step100_platform/`严格两个文件且与step100逐字节一致；adapter/config SHA256 `43199b5f...6ccc` / `fc2594ff...64b` |
| 当前状态 | **COMPLETE_LOCAL_PROVISIONAL_STEP100_PACKAGED_AWAITING_ONLINE**；离线证据混合，不宣称稳涨；未经s800新主模型重新排序，当前不占用优先提交位 |

## 已完成训练并打包step150：i13_s875_topic_ansretkl_v1

| 项 | 完成记录 |
|---|---|
| 目的/父模型 | 在固定协议最高分I-13同一r80内只优化topic JSON答案；不新增adapter/rank，不加载I-19/I-20。policy与冻结reference均从I-13 adapter SHA256 `71bc3c2c...ffd5b`开始 |
| 数据/损失 | 复用已登记`data_user_residual_retention_v1` 6,106行/SHA256 `bd947aad...b08f0`；topic CE+KL 1,301行（21.3069%），action及全部非用户KL-only 4,805行（78.6931%）；T/E/model rollout为0 |
| 配置/训练 | 启动配置`configs/active/i13_s875_topic_ansretkl_v1.yaml` SHA256 `9a9f703f...98dc`；trainer SHA256 `e8890cca...26cf`。GPU7单卡、r80原地更新、lr1e-7、warmup25后constant、effective batch4；150/150步、runtime327.65s、train loss1.18119、退出码0；W&B [`wjjymcj9`](https://wandb.ai/3120252125-/llmrec-2026/runs/wjjymcj9) |
| 保留轨迹 | 根目录`checkpoints/i13_s875_topic_ansretkl_v1/`与step150逐字节一致；`checkpoints/i13_s875_topic_ansretkl_v1/checkpoint-25/`、`checkpoints/i13_s875_topic_ansretkl_v1/checkpoint-50/`、`checkpoints/i13_s875_topic_ansretkl_v1/checkpoint-75/`、`checkpoints/i13_s875_topic_ansretkl_v1/checkpoint-100/`、`checkpoints/i13_s875_topic_ansretkl_v1/checkpoint-125/`、`checkpoints/i13_s875_topic_ansretkl_v1/checkpoint-150/`均为adapter-only，无optimizer/scheduler/RNG，仅step150进入提交池 |
| 统一诊断 | `logs/probe/i21_topic_path_20260715.json` SHA256 `832868c6...c09b`与`i21_rec_gold_path_20260715.json` SHA256 `112627b5...f2a`；step150 topic/action gold sum-logp变化`+0.09127/+0.03478`，prod/ad Top-64 `+3/128`、video/live `-2/128`。诊断源不回灌训练且不映射为线上分数 |
| 结构门 | step150临时merge后itemic断裂0/60=`PASS`；action复读1/30、选择题格式6/8、占位符0/8、简单题4/8只作diagnostic。日志`logs/precheck/i13_s875_topic_ansretkl_v1_step150_20260715.log` SHA256 `6978d655...647`；临时merge已删、GPU7归零 |
| 提交包 | `submissions/i13_s875_topic_ansretkl_v1_step150_platform/`严格两文件且与step150逐字节一致；adapter/config SHA256 `c8be262f...5e5d0` / `a7114c7d...caa4` |
| 状态 | **COMPLETE_LOCAL_PROVISIONAL_STEP150_PACKAGED_AWAITING_ONLINE**；只作为topic正交实验，当前不替换也不优先于I-13 s800（同包1.0037/1.0048）主模型 |

## 已完成并本地否决：i13_s875_world_ansretkl_v1

| 项 | 登记记录 |
|---|---|
| 目的/父模型 | 在I-13同一r80内只优化world最终选项token；不新增adapter/rank，不加载任何失败checkpoint；policy/reference均从I-13 SHA256 `71bc3c2c...ffd5b`开始 |
| 数据 | `data_i22_world_retkl_v1` 7,142行/SHA256 `81c7da5f...8350`：world CE 1,267（181 unique×7，17.740129%）+冻结I-13 KL-only 5,875（82.259871%）；上游与builder已登记ASSETS，T/E/model rollout为0。选择集`data_i22_world_retkl_v1_holdout` 46行/SHA256 `8aa4306f...402a`，train overlap0且完全不反传 |
| 配置/实现 | `configs/active/i13_s875_world_ansretkl_v1.yaml`启动SHA256 `ee443f82...a898e`；trainer/launcher SHA256 `1cb94586...a044` / `6ac84ce4...30f`。r80原地更新、lr1e-7、warmup25后constant、effective batch4、150步、每25步adapter-only保存 |
| 持久启动 | 2026-07-15 13:03 UTC；GPU7单卡；wrapper PID `3341716`由PID1接管、SID `3341716`、无TTY；step-0 policy/reference最大差0；W&B [`cohd8617`](https://wandb.ai/3120252125-/llmrec-2026/runs/cohd8617)服务端`finished` |
| 轨迹路径 | 根目录`checkpoints/i13_s875_world_ansretkl_v1/`；`checkpoints/i13_s875_world_ansretkl_v1/checkpoint-25/`、`checkpoints/i13_s875_world_ansretkl_v1/checkpoint-50/`、`checkpoints/i13_s875_world_ansretkl_v1/checkpoint-75/`、`checkpoints/i13_s875_world_ansretkl_v1/checkpoint-100/`、`checkpoints/i13_s875_world_ansretkl_v1/checkpoint-125/`、`checkpoints/i13_s875_world_ansretkl_v1/checkpoint-150/`。adapter SHA256依次=`c894fce8/9a61f127/da183bc7/007bde9f/a7f2970d/6bbbfe68`，config均`d7b4b961...5cbd`；根目录与step150逐字节一致，全部只作剂量证据 |
| 训练完成 | 150/150步，runtime320.71s、train loss2.30854、退出码0；600 microbatch中world/retention=`122/478`。六点及根目录均无optimizer/scheduler/RNG，根目录adapter/config与step150逐字节一致；训练日志SHA256 `549514cb...a409`，结束时本任务释放GPU7 |
| 冻结选点门 | `configs/evaluation/i22_world_checkpoint_gate.json` SHA256 `5f04e6ec...0914`在审计前登记；要求world logp正增、improved rate≥0.5、top-1不降、KL≤0.02，再进入topic/action/rec保持门。不得看结果后放宽 |
| world轨迹 | `logs/probe/i22_world_path_20260715.json` SHA256 `a5d14f20...bd78`；step25/50/75/100/125/150的gold logp变化=`+0.06628/+0.18402/+0.24064/+0.28053/+0.24041/+0.25042`，top-1变化=`-3/-1/-1/-1/0/-1`题，KL=`0.00870/0.02020/0.03474/0.03997/0.03381/0.03821`。46行含44题面组、两组同题同答案重复；与训练prompt交集0 |
| 决策 | 六点无一满足全部主门：step25 top-1失败；step50同时轻微越过KL且top-1失败；step75/100/150两门失败；step125仅KL失败。按预注册顺序停止，不跑后续E保持门和结构门，不生成提交包 |
| 当前状态 | **COMPLETE_LOCAL_REJECT_NO_PACKAGE**；所有I-22 checkpoint只保留作剂量证据，禁止上传和作为后续父模型 |

## 已完成并线上0.9915：seed_teacher_cotfix_v3_r64_lr1e4_ep3

| 项 | 冻结记录 |
|---|---|
| 目的 | 针对I-18“非video改善、video回退”的形状，只训练对已知多正例答案概率有确定正贡献的CoT补全；不融合、不蒸馏、不改最终标签 |
| 基座/隔离 | 从O6干净启动；I-10 E3只作为离线构造筛选器，不加载、resume、merge或warm-start其adapter；I-18及其他失败/低分checkpoint输入0 |
| 正式数据 | `assets/derived/processed/data_seed_teacher_cotfix_v3.jsonl`，32,644行，SHA256 `19dadb7dc7f1348ef18d31423177edfbc79af79f6e954f85cb8196f946e8ec42`；O1父行32,480（99.497611%）+ O2唯一teacher 164（0.502389%），T/E行0 |
| 构造门 | `scripts/data/build_cotfix_v3.py` SHA256 `839334d0...2c1f`；正式batch=1台账SHA256 `8ee28bbd...ff9a`，538组/1,836 gold中选83组。首跑batched结果因跨batch数值漂移废弃；两次8组batch=1 smoke与正式对应行逐字段全等 |
| 不变量 | 只改83条保留CoT（video/prod/ad/live=`32/35/11/5`），其余32,561条逐字不变；instruction/input/history、最终答案、行序、任务数、164条O2 teacher均0变化；target token 5,856,383，cutoff16384超限0 |
| 配置/训练物理 | `configs/active/seed_teacher_cotfix_v3_r64_lr1e4_ep3.yaml`启动SHA256 `514e63749ea63259dd79e7b489c7f75610304bc17563e1c6a8ad1c1d385f980c`；单卡W&B online，从O6训练r64/alpha64/dropout0.05、lr1e-4 cosine、warmup0.03、effective batch4、cutoff16384、3 epoch；每epoch保存adapter-only，最多3份，不保存optimizer/scheduler/RNG |
| 选择与停止 | 为保持与I-10/I-18同剂量比较，E3为唯一主候选，E1/E2只保留轨迹；本地结构门只否决灾难，不估总分。训练若出现NaN/OOM/结构损坏则停止，不自动消耗线上提交次数 |
| 持久启动 | 2026-07-15 15:45 UTC，物理GPU0 `GPU-d3c522d6-ed0f-2579-01cd-2d97da749980`；detached wrapper PID `3467172`由PID1接管、SID同PID、无TTY，日志/PID/退出码持久化到`logs/train/seed_teacher_cotfix_v3_r64_lr1e4_ep3.*`。启动器SHA256 `f0238644...f45` / `c37af997...1a4`；W&B [`t630t8ih`](https://wandb.ai/3120252125-/llmrec-2026/runs/t630t8ih) online |
| 首批验证 | 2,657 packed examples、665 steps/epoch、1,995 total，与I-10/I-18完全一致；step5/10/15/20/25 loss=`2.851/2.725/2.504/2.344/2.139`，grad norm `6.271→0.8614`；启动后GPU总占用约32.9GB、余48.1GB，Traceback/OOM/RuntimeError/NaN/Killed/Exception均0 |
| 保留输出 | 根目录`checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/`；E1/E2/E3依次为`checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/checkpoint-665/`、`checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/checkpoint-1330/`、`checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/checkpoint-1995/`。E1/E2仅作剂量轨迹，E3为唯一候选；全目录无optimizer/scheduler/RNG |
| 训练完成 | 1,995/1,995 steps，runtime4,772.7486s（1h19m32.75s），train loss1.3586777527，1.670 samples/s、0.418 steps/s，退出码0；日志SHA256 `93b744383f610c8b6ac58dc8701f31547bcb61d6c65387344bbb027f83a351e9`，未发现Traceback/OOM/RuntimeError/NaN/Killed/Exception |
| E1/E2/E3审计 | E1/E2/E3 adapter SHA256=`3071ab863c03746d434cae3bc474146d2c85f3d296905e4039b2491a63413d10` / `eaebdb3503e5e30cdc89e696c20684fa4aad5451dfaa6da26a81e64884e87d40` / `0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8`；config均`b3f2a1b5c5f77986a385c928450505aa0a8e01266016cfbd0c9ecc9c1feed7e7`。根目录与E3逐字节一致 |
| E3结构硬门 | 临时CPU merge后itemic断裂0/60=`PASS`；action复读3/30、选择题格式7/8、占位符0/8、简单题5/8仅作diagnostic。日志`logs/precheck/seed_teacher_cotfix_v3_r64_lr1e4_ep3_e3_precheck.log` SHA256 `5aeec99d8a3d9f39b622fbcb4cc468c427badd17ff40ab4195a1a30d36cccc87`；门检不预测线上分数 |
| E3平台包 | `submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform/`严格只有`adapter_config.json`（1,073 bytes，SHA256 `b3f2a1b5...e7e7`）与`adapter_model.safetensors`（161,533,160 bytes，SHA256 `0e5fa9bb...c6b8`），均与E3逐字节一致 |
| 线上 | UI `seed_teacher_cotfix_v3_r64_lr1e4_ep3_V1_eval_20260716104410`；2026-07-16 10:44:15平台记录，1h8m40s；总分0.9915；八项按material/action/topic/video/prod/ad/live/world为`0.2760/0.1099/0.0383/0.0576/0.1258/0.1400/0.1053/0.1387` |
| 线上日志 | 原始opaque路径`logs/eval/cTcxR1eV47RWIQxg-RK2TPl_Ogc9CHEFe6KS-zDweNUhj42UWiOmOKU12arRKtTTXrH1x7JrKIVHAhlE1zCsLXwKN6d42WnvrA6j-Z02MpufbYg4vQpJPrd8igyHdY7h.log`，2,779,725 bytes，SHA256 `dc354e4b5396b115af69be4ba297c100b584d7fc92ea6cd375f04804be5b4237`；action1024、itemic 7次race-average、8/8完成、失败0；evalTaskId `eval-task-t51r8o-1784169855` |
| 对照 | 相对I-13总分-0.0063，逐项=`+0.0307/-0.0084/-0.0007/-0.0384/+0.0034/+0.0084/-0.0009/-0.0003`；相对I-18总分+0.0218、material +0.0307、用户+0.0017、推荐-0.0076、world-0.0029。单次平台观测不作稳定方差或83条CoT筛选的净因果估计 |
| 当前状态 | **COMPLETE_ONLINE_0.9915_HIGHEST_NON_SPLICE_ADAPTER**；低I-13 s800首测0.0122、低同包最好显示分0.0133；成功E3仍允许作已登记action-answer-token分支父模型。I-27 N4×K8 strict exact-hit yield门已早停否决，未获得RFT训练资格；任何扩beam/改采样/改reward的新RFT设计必须重新预注册，E1/E2及I-24/I-25失败点继续禁止作父模型 |

## 已完成并线上0.9697：seed_teacher_cotfix_v2_r64_lr1e4_ep3

| 项 | 冻结与完成记录 |
|---|---|
| 目的 | 单变量检验“忠实补全上游截断推荐CoT”能否改善推荐四域；不采用参数融合，不把选手约1.03转述当收益证明 |
| 基座/隔离 | O6 `OneReason-0.8B`干净启动；不加载、resume、merge或warm-start任何adapter/checkpoint，旧`cotfix_v1`失败产物输入为0 |
| 正式数据 | `assets/derived/processed/data_seed_teacher_cotfix_v2.jsonl`，32,644行，`D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O3)`，SHA256 `634c4805367308b35dd729c17f59a1a8b4bb473b84a80d21cc71931a2c29c0e4` |
| 上游与混合 | 父I-10数据SHA256 `13c40526...eee4f`；O1父行32,480（99.497611%）+ O2唯一teacher 164（0.502389%），O2规则/T/E为0；O3 SHA256 `c307fe6d...b8d9d`只作历史侧证据，目标答案/目标元数据0 |
| 构造与质检 | `scripts/data/build_cotfix_v2.py` SHA256 `81aba7b962a4ad1c27d450d882a6774ab1771daae0fe7957e1f790a81227c9d0`；538/538为`TRUNCATED`且程序门0错误，独立judge全部score=5九项全真；新增非历史SID/重复前缀SID均0 |
| 单变量不变量 | 恰改538条保留推荐CoT；其余32,106行逐字不变。instruction/input/history、最终答案、行序、任务数和164条O2 teacher均0变化；target token 5,867,041，仅推荐四域增加；raw最大约9,744 token、16,384 cutoff超限0 |
| 审计 | `logs/data/seed_teacher_cotfix_v2_audit.json` SHA256 `bdea6db13a398dbcde973117dbf72d9ed81d5635bcbcbb7f1f1ba41fda79057c`；generation/judge审计SHA256 `05f5399b...5edf` / `994c8702...c861` |
| 配置 | `configs/active/seed_teacher_cotfix_v2_r64_lr1e4_ep3.yaml`启动时SHA256 `2bd234bd3d58a553c3f9b304e718e73531bbdb2a192d195866ba09350d3042da`；与I-10关键训练字段完全相同，仅dataset/dataset_dir及非训练的output/run name变化。完成后添加防覆盖头，并于GitHub数据发布时只将model/dataset/output三处路径改为仓库相对路径，训练字段不变；当前SHA256 `6722ffbe149c44031e3886fd3ecef79c1bc9005aba1e823a82365a66e630d622` |
| 训练物理 | 单卡、r64/alpha64/dropout0.05、lr1e-4 cosine、warmup0.03、effective batch4、cutoff16384、3 epoch、W&B online；按epoch保存adapter-only E1/E2/E3且最多3份，不保存optimizer/scheduler/RNG |
| 选择边界 | I-10旧协议轨迹只支持保留三轮剂量，不预测固定协议分数。主要预期在推荐四项；material/action/topic/world标签未改，结构灾难即否决。E1/E2/E3先保留，结构门只作硬失败保险丝，不本地估分排序；不自动消耗线上提交配额 |
| 持久启动 | 2026-07-14 17:26 UTC；单卡`GPU-66333310-c796-4db6-6772-684087c24bc9`（物理GPU1）。detached wrapper PID `2813778`由PID1接管、SID `2813778`且无TTY；PID/日志/退出码分别持久化为同名`.pid`、`.log`、`.exit_code`，不是临时PTY会话 |
| 训练规模 | 2,657 packed examples；665 update steps/epoch、1,995 total；40,370,176 trainable params（4.7957%）；effective batch4。与I-10精确相同的步数，新增CoT未改变packed样本总数或总步数 |
| 保留输出 | 根目录`checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/`；E1/E2/E3依次为`checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-665/`、`checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1330/`、`checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1995/`。E1/E2只作剂量轨迹；E3为本实验唯一主候选，尚未授权为后续训练父模型 |
| E1审计 | trainer state为global step665/epoch1；adapter SHA256 `02b404bdf3430ccd9adfe1ce943493b34cd2e2d8dd5ad7e98cd9aebe74147a85`，adapter config SHA256 `65732b4da48a0b2f93ea6d1bb3861e7e13bf0406635ae5e587de34a709ff26fa`；无optimizer/scheduler/RNG，只作轨迹，不打包或上传 |
| E2审计 | trainer state为global step1330/epoch2；adapter SHA256 `14071ab832288c59b114e882b8821e39931b0ed64217e7e79acf96a4532bdc38`，adapter config SHA256 `65732b4da48a0b2f93ea6d1bb3861e7e13bf0406635ae5e587de34a709ff26fa`；无optimizer/scheduler/RNG，只作轨迹，不打包或上传 |
| E3与最终产物 | trainer state为global step1995/epoch3；E3 adapter SHA256 `07cd662852ee1ef3654096adfce36891ce260129bdc68c7924c2b75554c2a9e3`，adapter config SHA256 `65732b4da48a0b2f93ea6d1bb3861e7e13bf0406635ae5e587de34a709ff26fa`。根目录adapter/config与E3逐字节一致；全目录无optimizer/scheduler/RNG |
| 训练完成 | 1,995/1,995 steps；runtime4,890.3287s（1h21m30.33s）；train loss1.3617670884；1.630 samples/s、0.408 steps/s；退出码0，训练结束时GPU1为0 MiB/0%。日志497,195 bytes、SHA256 `4b30ae5436c7b3b5eb45f354a99e35f29eecad891174059f9314df655a6f5edd`，未发现Traceback、OOM、RuntimeError、NaN、Killed或Exception |
| W&B | [`32av8e8z`](https://wandb.ai/3120252125-/llmrec-2026/runs/32av8e8z)服务端`finished`；global step1995、epoch3、train loss1.3617670884、runtime4,890.3287s；首批step5/10/15/20/25 loss=`2.8461/2.7248/2.5192/2.3375/2.1453` |
| 训练结果文件 | `train_results.json` SHA256 `a2a2f9d46b51387d9d11a0113845babb1d461f89306f4548f6edeba60aea6d62`；根目录`trainer_state.json` SHA256 `8709cfcfd0d0174bbfc1cedff0418d6829042decc4f7677c30243b3369ffffaa` |
| E3结构硬门禁 | 临时merge后itemic断裂0/60=`PASS`；action复读5/30、选择题格式3/8、占位符0/8、简单题2/8均按冻结规则只作diagnostic。日志`logs/precheck/seed_teacher_cotfix_v2_r64_lr1e4_ep3_e3_precheck.log`，1,683 bytes，SHA256 `2e6629cb4b3d7cd34443109967f27c7c2342c932ebe0c5338b48e4d5efa977c4`；临时merge与配置已删。门禁只排除结构灾难，不证明涨分 |
| E3平台包 | `submissions/seed_teacher_cotfix_v2_r64_lr1e4_ep3_platform/`严格只有`adapter_config.json`（1,138 bytes，SHA256 `65732b4d...26fa`）与`adapter_model.safetensors`（161,533,160 bytes，SHA256 `07cd6628...2a9e3`）；两文件均与E3逐字节一致，已完成固定协议评测 |
| 线上 | `seed_teacher_cotfix_v2_r64_lr1e4_ep3_V1_eval_20260715102021`；2026-07-15 10:21:16，1h11m48s；总分0.9697；八项=`0.2453/0.1083/0.0382/0.0768/0.1190/0.1316/0.1089/0.1416`；相对I-13总分-0.0281 |
| 线上日志 | `logs/eval/Ya0t3O9IfRz5MQq_6FBxvX0N-7_gjpxoUeScdLaFlMEkWqMSp04VgCQOyWH4wkVLolLzNlaB5qUoHeeAVwTGn3NfVnwJB5TwiPKBg1J-Q5Lz2QOJqbqc2fSR3LbbJtM_.log`，3,198,898 bytes，SHA256 `ab348f8b...d0bc`；8/8完成、失败0；evalTaskId `eval-task-h8gwve-1784082076` |
| GitHub数据发布 | `assets/derived/releases/seed_teacher_cotfix_v2/`包含完整确定性gzip、manifest、原始小型审计摘要和使用说明；gzip 52,199,218 bytes/SHA256 `193cd78f...07f9`，解压后32,644行/249,454,095 bytes/SHA256 `634c4805...c0e4`；`scripts/data/restore_seed_teacher_cotfix_v2.py`执行压缩与内容双校验及原子还原，无需API即可复现实训输入 |
| 当前状态 | **COMPLETE_ONLINE_0.9697_NOT_MAIN**；E1/E2不提交，E3未替换I-13，不作为后续默认父模型 |

## 保留模型

| 角色 | 实验 | 路径 | 哈希 | 状态 |
|---|---|---|---|---|
| 官方基座 | OneReason-0.8B | `models/OneReason-0.8B-pretrain-competition/` | `config.json` SHA256 `5fe26642...` | 只读 |
| I-18 E1剂量轨迹 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 | `checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-665/` | adapter/config SHA256 `02b404bd...47a85` / `65732b4d...26fa` | adapter-only；只作1 epoch轨迹，不打包、不上传、不作父模型 |
| I-18 E2剂量轨迹 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 | `checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1330/` | adapter/config SHA256 `14071ab8...bdc38` / `65732b4d...26fa` | adapter-only；只作2 epoch轨迹，不打包、不上传、不作父模型 |
| I-18已评测E3 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 E3 | `checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1995/`；根目录`checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/`逐字节同adapter | adapter/config SHA256 `07cd6628...2a9e3` / `65732b4d...26fa` | 固定协议线上0.9697；未替换I-13，不作默认父模型 |
| I-18已评测E3 LoRA包 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 | `submissions/seed_teacher_cotfix_v2_r64_lr1e4_ep3_platform/` | adapter/config SHA256 `07cd6628...2a9e3` / `65732b4d...26fa` | 严格两文件且与E3逐字节一致；固定协议线上0.9697 |
| I-23已评测E3/action-retKL父与I-30 material teacher | seed_teacher_cotfix_v3_r64_lr1e4_ep3 E3 | `checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/checkpoint-1995/`；根目录`checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/`逐字节同adapter | adapter/config SHA256 `0e5fa9bb...c6b8` / `b3f2a1b5...e7e7` | 固定协议线上0.9915；除已登记action-retKL角色外，用户现批准其只作I-30 material构造评分器和冻结KL teacher；不作为I-30 policy初始化。I-24/I-25失败点禁止复用 |
| I-23已评测E3 LoRA包 | seed_teacher_cotfix_v3_r64_lr1e4_ep3 | `submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform/` | adapter/config SHA256 `0e5fa9bb...c6b8` / `b3f2a1b5...e7e7` | 严格两文件且与E3逐字节一致；固定协议线上0.9915 |
| I-28本地主门否决剂量点 | i28_i23_rec_multigold_proposal_retkl_v1 step64 | `checkpoints/i28_i23_rec_multigold_proposal_retkl_v1/checkpoint-64/` | adapter/config SHA256 `0683593b...7376` / `33b749b6...6933b5` | set-path改善61/128，低冻结门71/128；仅作审计，禁止打包、上传、warm start或作父模型 |
| I-28本地主门否决末点 | i28_i23_rec_multigold_proposal_retkl_v1 step128 | `checkpoints/i28_i23_rec_multigold_proposal_retkl_v1/checkpoint-128/`；根目录`checkpoints/i28_i23_rec_multigold_proposal_retkl_v1/`两文件逐字节一致 | adapter/config SHA256 `06f43f19...eec` / `33b749b6...6933b5` | set-path改善69/128，低冻结门71/128；分支关闭，仅作审计，禁止打包、上传、warm start或作父模型 |
| I-33本地否决r8正式候选 | i33_r96_material_desc2sid_retkl_r8_v1 step64/128/256/512 | `checkpoints/i33_r96_material_desc2sid_retkl_r8_v1/checkpoint-64/`、`checkpoints/i33_r96_material_desc2sid_retkl_r8_v1/checkpoint-128/`、`checkpoints/i33_r96_material_desc2sid_retkl_r8_v1/checkpoint-256/`、`checkpoints/i33_r96_material_desc2sid_retkl_r8_v1/checkpoint-512/`；根目录与step512同adapter | adapter SHA256依次`01773f00...5af1`/`c4f40ceb...2fcc`/`91c2b1c6...81b3`/`90411c77...296c`；config统一`a561f54b...cbce` | r8/alpha8正式轨迹；四点冻结门均失败，仅作训练审计，禁止打包、上传、resume、warm start或作父模型 |
| I-33本地否决r104组合候选 | i33_r96_material_desc2sid_retkl_r8_v1 step64/128/256/512 scale1 | `checkpoints/i33_r96_material_desc2sid_retkl_r104_step64/`、`checkpoints/i33_r96_material_desc2sid_retkl_r104_step128/`、`checkpoints/i33_r96_material_desc2sid_retkl_r104_step256/`、`checkpoints/i33_r96_material_desc2sid_retkl_r104_step512/` | adapter SHA256依次`4e4d2185...c2d1`/`8135de55...d2a`/`1570765b...21a`/`a213deb4...df64`；config统一`46bd7146...a205`；各262,460,268 bytes | 精确FP32 r96+r8得到r104/alpha104；736行冻结门全部失败且`earliest_teacher_forced_pass=null`。无提交包，禁止上传、继续训练或作父模型 |
| I-32本地否决r8训练轨迹 | i32_task_restore_retkl_r8_v1 step64/128/192/256/320/384/448/512 | `checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-64/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-128/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-192/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-256/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-320/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-384/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-448/`、`checkpoints/i32_task_restore_retkl_r8_v1/checkpoint-512/`；根目录与step512同adapter | adapter SHA256依次`ff72efd8...c44`/`5e06010a...e33`/`16750417...641`/`888eb67e...01b`/`23854f67...a70`/`77ed9305...c31`/`aacb34bd...fe7`/`9e71d6cb...107`；config统一`fbf08a81...d67` | r8/alpha8训练轨迹；192/320/384/448从未获科学选择资格，其余四点门禁失败。全部仅作审计，禁止打包、上传、resume、warm start或作父模型 |
| I-32本地否决r168组合候选 | i32_task_restore_retkl_r8_v1 step64/128/256/512 scale1 | `checkpoints/i32_task_restore_retkl_r8_v1_combined_r168_step64/`、`checkpoints/i32_task_restore_retkl_r8_v1_combined_r168_step128/`、`checkpoints/i32_task_restore_retkl_r8_v1_combined_r168_step256/`、`checkpoints/i32_task_restore_retkl_r8_v1_combined_r168_step512/` | adapter SHA256依次`81e60e91...58b`/`6c0756e7...523`/`91190212...9d2`/`2c1c0f84...690`；config统一`b31258a0...031` | 656行冻结门全部失败，`earliest_teacher_forced_pass=null`。step128仅因用户事后授权每日额度探索而允许一次手工上传；其余点及全部点的继续训练/父模型角色仍禁止 |
| I-32门外探索FP32拒绝包 | i32_task_restore_retkl_r168_step128_exploratory | `submissions/i32_task_restore_retkl_r168_step128_exploratory_platform/` | adapter/config SHA256 `6c0756e7...523` / `b31258a0...031`，423,940,024 / 1,076 bytes；package audit SHA256 `826debff...d4e4` | r168/alpha168、392 FP32 tensors；平台因两文件总计423,941,100 bytes超过400MB在任务创建前拒绝，无evalTaskId。只作源身份参考，禁止重传 |
| I-32门外探索BF16 r168拒绝包 | i32_task_restore_retkl_r168_step128_exploratory_bf16 | `submissions/i32_task_restore_retkl_r168_step128_exploratory_bf16_platform/` | adapter/config SHA256 `a9cc127f...4da` / `b31258a0...031`，211,996,792 / 1,076 bytes；package audit `logs/model/i32_task_restore_r168_step128_exploratory_bf16_package.json` SHA256 `7f29d24f...1090` | r168/alpha168、392 BF16 tensors；大小合格但平台因rank超过128在任务创建前拒绝，无evalTaskId。只作身份参考，禁止重传 |
| I-32用户授权门外探索合法r128包 | i32_task_restore_retkl_r128_step128_svd_bf16 | `submissions/i32_task_restore_retkl_r128_step128_svd_bf16_platform/` | adapter/config SHA256 `edadde1f...d46` / `daa3106d...7f3`，161,533,944 / 1,076 bytes；package audit `logs/model/i32_task_restore_r128_step128_svd_bf16_package.json` SHA256 `10190999...0705` | r128/alpha128、392 BF16 tensors；总计161,535,020 bytes。656行material两向gold均值`+0.001988/+0.000350`、world 11/16，itemic 0/60。`READY_FOR_MANUAL_UPLOAD_NOT_ONLINE_EVALUATED`，唯一允许手工上传的I-32包 |
| I-31用户授权线上探索合法r128包 | i31_r96_i23_exact_interp_r128_l010_svd_bf16 | `checkpoints/i31_r96_i23_exact_interp_r128_l010_svd_bf16/`；`submissions/i31_r96_i23_exact_interp_r128_l010_svd_bf16_platform/` | adapter/config SHA256 `93d247a2...803f` / `daa3106d...07f3`，161,533,944 / 1,076 bytes；压缩审计`logs/model/i31_r96_i23_exact_interp_r128_l010_svd_bf16_compression.json` SHA256 `56e31406...3856` | `delta=0.9×I19最高r96+0.1×I23 material强模型`的r160精确组合压为r128/alpha128、392 BF16 tensors；全局保留谱能量`0.9999999329`，总计161,535,020 bytes。严格两文件，`READY_FOR_ONLINE_SCORE_FIRST_PROBE`，尚未线上评测 |
| I-32 r128压缩对照（禁止上传） | step128 global-SVD FP32 / prefix96+tail32-SVD FP32 | `submissions/i32_task_restore_retkl_r128_step128_svd_fp32_platform/`、`submissions/i32_task_restore_retkl_r128_step128_tail32_svd_fp32_platform/` | adapter SHA256 `3ef8710a...b92` / `f3990dc8...cf8`；门控报告SHA256 `ada2f34d...a6b9` / `17191448...69b0` | 全局FP32与选中BF16的656行指标逐项完全相同但更大；结构化96+32使material双向gold均值变负。两者只作压缩审计，禁止上传或作父模型 |
| I-23 + 0.625用户残差已评线上探针 | i23_userres_r80_s625 | `checkpoints/i23_userres_r80_s625/` | adapter/config SHA256 `4a46fd29...0c70` / `4768770a...4d06`；组合审计`logs/model/i23_userres_r80_s625_combine.json` SHA256 `e3c630b9...3d32` | 固定协议线上0.9866，八项=`0.2453/0.1170/0.0399/0.0768/0.1326/0.1316/0.1044/0.1390`；相对同日I-23复测非material `+0.0289`、material `-0.0307`。原始日志/evalTaskId待入仓 |
| I-23 + 0.625用户残差已评包 | i23_userres_r80_s625 | `submissions/i23_userres_r80_s625_platform/` | adapter/config SHA256 `4a46fd29...0c70` / `4768770a...4d06`，201,903,440 / 1,074 bytes | 严格两文件包与checkpoint逐字节一致；已线上0.9866，未替换I-13。参数拼接按规则是不鼓励而非禁止，构造链已披露 |
| I-23 + 0.5用户残差已评线上探针 | i23_userres_r80_s500 | `checkpoints/i23_userres_r80_s500/` | adapter/config SHA256 `d2b77c74...99bf` / `4768770a...4d06`；组合审计SHA256 `7b04c918...3be5` | CPU精确拼接`delta=I23+0.5×residual`；固定协议线上0.9882，八项=`0.2760/0.1156/0.0399/0.0576/0.1258/0.1316/0.1035/0.1383`；状态`ONLINE_COMPLETE`，原始日志/evalTaskId待入仓 |
| I-23 + 0.5用户残差已评包 | i23_userres_r80_s500 | `submissions/i23_userres_r80_s500_platform/` | adapter/config SHA256 `d2b77c74...99bf` / `4768770a...4d06`，201,903,440 / 1,074 bytes | 严格两文件且与checkpoint逐字节一致；UI `i23_userres_r80_s500_V1_eval_20260716190207`，已线上0.9882，不得重复提交 |
| I-23 + 0.5625用户残差已评探针 | i23_userres_r80_s5625 | `checkpoints/i23_userres_r80_s5625/` | adapter/config SHA256 `1f17f41...a1ba` / `4768770a...4d06`；组合审计`logs/model/i23_userres_r80_s5625_combine.json` SHA256 `1e0e0b7e...fff3` | CPU精确拼接`delta=I23+0.5625×residual`，单个r80/alpha80、392 tensors；线上0.9925，material/video=`0.2453/0.0864`；状态`ONLINE_COMPLETE_NOT_MAIN`，触发一次s53125向下二分 |
| I-23 + 0.5625用户残差已评包 | i23_userres_r80_s5625 | `submissions/i23_userres_r80_s5625_platform/` | adapter/config SHA256 `1f17f41...a1ba` / `4768770a...4d06`，201,903,440 / 1,074 bytes | 严格仅`adapter_model.safetensors`与`adapter_config.json`，与checkpoint逐字节一致；UI `i23_userres_r80_s5625_V1_eval_20260717220142`，已线上0.9925，不得重复提交 |
| I-23 + 0.53125用户残差已评末次探针 | i23_userres_r80_s53125 | `checkpoints/i23_userres_r80_s53125/` | adapter/config SHA256 `fec25351...65e0` / `4768770a...4d06`；组合审计`logs/model/i23_userres_r80_s53125_combine.json` SHA256 `d39e2fc8...a030` | CPU精确拼接`delta=I23+0.53125×residual`，单个r80/alpha80、392 tensors；线上0.9757，八项=`0.2453/0.1159/0.0400/0.0864/0.1190/0.1260/0.1044/0.1387`；状态`ONLINE_COMPLETE_AXIS_CLOSED` |
| I-23 + 0.53125用户残差已评包 | i23_userres_r80_s53125 | `submissions/i23_userres_r80_s53125_platform/` | adapter/config SHA256 `fec25351...65e0` / `4768770a...4d06`，201,903,440 / 1,074 bytes | 严格仅两文件且与checkpoint逐字节一致；UI `i23_userres_r80_s53125_V1_eval_20260718012851`，已线上0.9757，不得重复提交；冻结失败条件触发，整轴关闭 |
| I-23 + 用户残差加码关闭备件 | i23_userres_r80_s750 | `checkpoints/i23_userres_r80_s750/` | adapter/config SHA256 `363a3b59...f365` / `4768770a...4d06`；组合审计SHA256 `5d890803...12a6` | CPU精确拼接`delta=I23+0.75×residual`；原准入条件“s625 material保持”已被线上0.2453否决，故不评测、不打包、不上传，仅保留历史工件 |
| I-23 + 用户残差加码关闭备件 | i23_userres_r80_s875 | `checkpoints/i23_userres_r80_s875/` | adapter/config SHA256 `b24c17cc...f31f` / `4768770a...4d06`；组合审计SHA256 `ae4782ab...a8ad` | CPU精确拼接`delta=I23+0.875×residual`；上游准入条件已失败，故不评测、不打包、不上传，仅保留历史工件 |
| I-25 step250确定性恢复失败证据 | i25_step250_deterministic_replay | 根`checkpoints/i25_step250_deterministic_replay/`；保存点`checkpoints/i25_step250_deterministic_replay/checkpoint-250/` | step250/root adapter SHA256 `9198538f...68d9`，config `70a62d4e...29a771`；结果SHA256 `fad1adf5...29bb` | scheduler/route复现但未逐字节命中删除前身份，硬闸拒绝；不得安装、Stage1、打包、上传或作父模型 |
| I-10 已评测 E1 | seed_teacher_r64_lr1e4_ep3 | `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-665/` | adapter SHA256 `c1bfb4dada8260560327a5ce3a9a15cbb29c0249421616bb1a9d95d9dc11add8` | 1 epoch；线上0.9100 |
| I-10 E1 LoRA上传包 | seed_teacher_r64_lr1e4_ep3 | `submissions/seed_teacher_r64_lr1e4_e1_platform/` | adapter SHA256 `c1bfb4dada8260560327a5ce3a9a15cbb29c0249421616bb1a9d95d9dc11add8`；config SHA256 `f27c697e8bb611802822ea44b156b672c63f6d2ec16a380d868395a9d0eb213f` | 已上传并评测为0.9100 |
| I-10 已评测 E2 | seed_teacher_r64_lr1e4_ep3 | `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1330/` | adapter SHA256 `c4902871c31f1a29b895b3990b2af573808cfecef2a5e483720ce1e60b1ac267` | 2 epoch；线上0.9680 |
| I-10 E2 LoRA上传包 | seed_teacher_r64_lr1e4_ep3 | `submissions/seed_teacher_r64_lr1e4_e2_platform/` | adapter SHA256 `c4902871c31f1a29b895b3990b2af573808cfecef2a5e483720ce1e60b1ac267`；config SHA256 `f27c697e8bb611802822ea44b156b672c63f6d2ec16a380d868395a9d0eb213f` | 已上传并评测为0.9680 |
| 固定协议桥接/I-16与I-17获准父模型 | seed_teacher_r64_lr1e4_ep3 | `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1995/` | adapter SHA256 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2` | 3 epoch；旧协议线上0.9849，固定协议待重评；允许作为I-16/I-17同adapter policy初始化与冻结reference，仍禁止把旧分当固定协议基线 |
| I-16轨迹候选/当前未过完整门禁 | seed_teacher_e3_dpo_rec_o1hard_v1 step200 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_v1/checkpoint-200/` | adapter SHA256 `96817d4c2633fb6c9aeb26d73eeb54214d94c6069642930e31ed2be6a89fac04` | adapter-only；推荐排序改善但gold-logp保护线略失败；仅作I-16评估候选和轨迹证据，禁止作为新训练父模型 |
| I-16轨迹候选/当前未过完整门禁 | seed_teacher_e3_dpo_rec_o1hard_v1 step400 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_v1/checkpoint-400/` | adapter SHA256 `291c0b3e7a1ef05d247c79b8a3f842df2cdea5ac75e6ddf07a7ef7f23046eb94` | adapter-only；推荐排序进一步改善但gold-logp下降扩大；仅作I-16评估候选和轨迹证据，禁止作为新训练父模型 |
| I-16轨迹候选/未过完整门禁 | seed_teacher_e3_dpo_rec_o1hard_v1 step600 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_v1/checkpoint-600/` | adapter SHA256 `85cb7f002f1bed48240b2f377a84c5fd3bf9848e1edb1dda8faa3a5c3c773f00` | adapter-only；推荐排序改善但gold-logp保护线失败；仅作I-16评估候选和轨迹证据，禁止作为新训练父模型 |
| I-17最终训练输出/非选中末点 | seed_teacher_e3_dpo_rec_o1hard_lowdose_v2 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2/` | adapter/config SHA256 `b8667658...a492d` / `65b21290...90f3` | 与step200逐字节一致；W&B `pfjlvm70` finished；只保留为完整训练轨迹，不上传、不优先于最早通过的step100 |
| I-17剂量轨迹审计点/非候选 | seed_teacher_e3_dpo_rec_o1hard_lowdose_v2 step50 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2/checkpoint-50/` | adapter SHA256 `4058a3fbfa7778e3a04eed27a258ed0d21f0e963d4de62908947e3452f0f91db` | adapter-only保存审计点；不在预注册候选集合，未运行偏好门禁、不参与选择 |
| I-17固定协议已评测候选 | seed_teacher_e3_dpo_rec_o1hard_lowdose_v2 step100 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2/checkpoint-100/` | adapter/config SHA256 `a3b2fc9c...2d1dc` / `65b21290...90f3` | 固定协议线上0.9727；未替换I-13；直接父模型固定协议分缺失，暂不作DPO因果结论 |
| I-17通过但未选轨迹点 | seed_teacher_e3_dpo_rec_o1hard_lowdose_v2 step150 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2/checkpoint-150/` | adapter SHA256 `37a85b3af4f50abd612c11f8086b7a6ef58923a03154aed805dc57010d896661` | 量化门槛通过，但因晚于step100不选；I-10 E3固定协议桥接前不上传、不作为父模型 |
| I-17通过但未选轨迹点 | seed_teacher_e3_dpo_rec_o1hard_lowdose_v2 step200 | `checkpoints/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2/checkpoint-200/` | adapter SHA256 `b86676581d72fa68297d911b9be83476282c599238d41a488953d2d46f8a492d` | 量化门槛通过，但因晚于step100不选；与根目录最终adapter一致，仅作剂量轨迹 |
| I-17已评测LoRA包 | seed_teacher_e3_dpo_rec_o1hard_lowdose_v2 step100 | `submissions/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2_step100_platform/` | adapter/config SHA256 `a3b2fc9c...2d1dc` / `65b21290...90f3` | 严格两文件且与step100逐字节一致；固定协议线上0.9727；日志evalTaskId `eval-task-eeve0r-1784036284` |
| E3 LoRA提交包 | seed_teacher_r64_lr1e4_ep3 | `submissions/seed_teacher_r64_lr1e4_e3_platform/` | adapter SHA256 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2`；config SHA256 `f27c697e8bb611802822ea44b156b672c63f6d2ec16a380d868395a9d0eb213f` | 可直接用于固定协议桥接重评 |
| I-11 固定协议参考 | seed_teacher_e3_cont_r64_lr2e5_ep1 | `checkpoints/seed_teacher_e3_cont_r64_lr2e5_ep1/` | adapter SHA256 `6b2e4fbd7ee8e04b4704d31fb50e95dc60cf5a04f7537ee746e976d897b68626` | 固定协议线上0.9618；不与E3旧分作差 |
| I-11 LoRA上传包 | seed_teacher_e3_cont_r64_lr2e5_ep1 | `submissions/seed_teacher_e3_cont_r64_lr2e5_ep1_platform/` | adapter/config SHA256 `6b2e4fbd...68626` / `0d5282cd...2f7b` | 严格两文件且与训练输出逐字节一致；已上传并评测为0.9618 |
| I-12 v3残差 | e3_userres_r16_retkl_v3_ep1 | `checkpoints/e3_userres_r16_retkl_v3_ep1_residual/` | adapter SHA256 `e8caf0a39fee133b2172f2e74a3ef64c53b3d46f2d8dc82acbc821a00b524f98`；config `0ba92b34...63d7` | 相对已合并E3的r16；仅作训练审计，不得单独上传 |
| I-12 v3组合adapter | e3_userres_r80_retkl_v3_ep1 | `checkpoints/e3_userres_r80_retkl_v3_ep1/` | adapter/config SHA256 `3fe85158...87cc6` / `e3c3ace0...c4ac0` | E3 r64+残差r16精确拼接；相对O6的单个r80 |
| I-12 v3固定协议对照 | e3_userres_r80_retkl_v3_ep1 | `submissions/e3_userres_r80_retkl_v3_ep1_platform/` | 与组合adapter逐字节一致；严格两文件，SHA256同上 | 固定协议线上0.9768；已被I-13同协议高0.0210 |
| I-13 上一固定协议主模型 | e3_userres_r80_retkl_v3_s875 | `submissions/e3_userres_r80_retkl_v3_s875_platform/` | adapter/config SHA256 `71bc3c2c86beb1c1aaafd41f98915ba94a7f964b6e8450079a883aebc32ffd5b` / `e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0` | 固定协议线上0.9978；已被s800高至少0.0059；参数拼接不鼓励但不禁止，复现链已披露 |
| I-13 scale0.90参数探针 | e3_userres_r80_retkl_v3_s900 | `submissions/e3_userres_r80_retkl_v3_s900_platform/` | adapter/config SHA256 `7c966fb2...a60a` / `e3c3ace0...c4ac0` | 精确参数拼接、严格两文件、结构门0/60 PASS；本地保持KL和action CE未支配0.875，因此未作为I-19父模型；仅作零训练备选，不据此声称涨分 |
| I-35当前最高已评测包 | i35_r96_video_boundary_retkl_r112 step548 | `submissions/i35_r96_video_boundary_retkl_r112_step548_platform/` | adapter/config SHA256 `52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00` / `4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996` | 严格两文件，r112/alpha112、392 tensors；StreamLake description对应step548，正式线上`1.0344285849069457`，evalTaskId `eval-task-9nepj1-1784698215`；当前默认交付包 |
| I-35唯一剩余授权点 | i35_r96_video_boundary_retkl_r112 step411 | `submissions/i35_r96_video_boundary_retkl_r112_step411_platform/` | adapter/config SHA256 `e26eb9befd0ad2a1b60e7f088d6788e8101f32b7e1d43d8d9a0114f75da35d58` / `4f90d28f538e17cf70bc6876851fadd1d26a03a0e4574b7602fcb360b56e5996` | 严格两文件，r112/alpha112、392 tensors；离线成对对照已归档，建议只作一次低剂量线上对照；尚无同步评测记录；step137/274/685禁投 |
| I19-world上一最高/I-35父模型 | i19_world_userres_retkl_r16_ep1_i13retain_v1 scale0.875 | `submissions/i19_world_external_r96_s875_platform/` | adapter/config SHA256 `4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e` / `78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f`，242,273,688 / 1,074 bytes | 严格两文件，r96/alpha96、392 tensors；固定协议原始/复测`1.025259456/1.025362611`；I-35直接父模型。原parent/residual与完整复现链仍未到卷 |
| I-13 scale0.80上一完整本地复现主模型 | e3_userres_r80_retkl_v3_s800 | `submissions/e3_userres_r80_retkl_v3_s800_platform/` | adapter/config SHA256 `bb86eb8af0efd3560b7b7c8440f3830627e9255f4fcc2265b9274a27668f63c6` / `e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0` | 同包两次固定协议显示分1.0037/1.0048；差0.0011按合理抖动；训练复现链完整，线上分低于I19-world父模型与I-35当前最高包 |
| I-13 scale0.75备包 | e3_userres_r80_retkl_v3_s750 | `submissions/e3_userres_r80_retkl_v3_s750_platform/` | adapter/config SHA256 `5aa80992c517f9dbc4074f0f48dbf49f864e0ff9bfd19196f2262a858ae76623` / `e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0` | 精确参数拼接；itemic 0/60 PASS、action复读4/30；旧曲率中心约1.0048已落入s800复测范围，降级为备包，不优先提交 |
| I-19线上否决候选 | i13_s875_dpo_rec_balhard_lowdose_v1 step25 | `checkpoints/i13_s875_dpo_rec_balhard_lowdose_v1/checkpoint-25/` | adapter/config SHA256 `e6ebe4dc...fb2e0` / `a7114c7d...caa4` | 固定协议线上0.9763，低I-13 0.0215；仅保留作偏好目标失配证据，不上传、不作父模型 |
| I-19未选剂量轨迹 | i13_s875_dpo_rec_balhard_lowdose_v1 step50 | `checkpoints/i13_s875_dpo_rec_balhard_lowdose_v1/checkpoint-50/` | adapter/config SHA256 `9dddc6a2...729c` / `a7114c7d...caa4` | step25线上已否决偏好目标；只作轨迹，禁止上传和作为父模型 |
| I-19未选末点/最终输出 | i13_s875_dpo_rec_balhard_lowdose_v1 step75 | `checkpoints/i13_s875_dpo_rec_balhard_lowdose_v1/checkpoint-75/`；根目录逐字节同adapter | adapter/config SHA256 `42aa6577...bf1` / `a7114c7d...caa4` | step25线上已否决偏好目标；只作完整训练轨迹，禁止上传和作为父模型 |
| I-19已评测LoRA包 | i13_s875_dpo_rec_balhard_lowdose_v1 step25 | `submissions/i13_s875_dpo_rec_balhard_lowdose_v1_step25_platform/` | adapter/config SHA256 `e6ebe4dc...fb2e0` / `a7114c7d...caa4` | 严格两文件且与step25逐字节一致；固定协议线上0.9763，已否决 |
| I-12 v2启动失败 | e3_userres_r16_retkl_v2_ep1 | 无checkpoint；失败输出目录已删除 | W&B `fi4mneew`；step275/1527 | ChatML尾换行使终止权重错位；禁止resume，不作为模型实验结果 |
| I-12 v1启动失败 | e3_userres_r16_retkl_ep1 | 无checkpoint；失败输出目录已删除 | W&B `hkt762u2`；step8/1527 | 非用户保持路由格式错误；禁止resume，不作为模型实验结果 |
| 本地否决 checkpoint | seed_o2_action_r64_lr15e5_ep1 | `checkpoints/seed_o2_action_r64_lr15e5_ep1/` | adapter SHA256 `8b6fc2f9fbc2170298e31b83ea8c581880d7d76657c9a35d2afee305bef950d1` | I-09 单次性能组合；action/material 门禁未显示预期优势，不上传、不作 warm start |
| 本地否决 checkpoint | seed_scoremax_r32_ep1 | `checkpoints/seed_scoremax_r32_ep1/` | adapter SHA256 `74bb4fed78a72215caae354df4a4a4075d3d36fbde1f5efe2ea93a9cec4d8576` | I-07 单次实验；门禁未显示预期 action/material 优势，不上传；保留审计 |
| 历史单次参考 checkpoint | riders_fk_lora_ep1 | `checkpoints/riders_fk_lora_ep1/` | adapter MD5 `0c294240`；SHA256 `af5d8503...` | 单次线上 0.9177；仅作历史比较对象 |
| 历史单次参考提交包 | riders_fk_lora_ep1 | `submissions/riders_fk_lora_ep1_platform/` | model MD5 `c2046b60` | 单次线上 0.9177 |
| 历史比较 checkpoint | riders_fk_clean_r64 E2 | `checkpoints/riders_fk_clean_r64_ep3/checkpoint-706/` | adapter SHA256 `c206c86e1c43fadb8f0ff55ae2dea02d3722c93686cb029b24b64cfb4e545ef5` | 单次线上0.9187；已被I-10 E3高0.0662，暂保留作历史比较 |
| 已评测 checkpoint | riders_fk_clean_r64 E1 | `checkpoints/riders_fk_clean_r64_ep3/checkpoint-353/` | adapter SHA256 `6db7727faa0fd7900a4aca15fdbf96aaa6fc104beb1267543034e766874dac89` | 线上 0.8839；本地门禁假阳性 |
| 已评测 LoRA 包 | riders_fk_clean_r64 E1 | `submissions/riders_fk_clean_r64_e1_platform/` | adapter SHA256 `6db7727faa0fd7900a4aca15fdbf96aaa6fc104beb1267543034e766874dac89`；config SHA256 `76a9a1e59a1f69fde20901fafcee6f0d53265b6408f7be60a906792c17524f7a` | 对应 E1 线上 0.8839；暂存待统一清理 |
| 本地否决 checkpoint | riders_fk_clean_r64 E3 | `checkpoints/riders_fk_clean_r64_ep3/checkpoint-1059/` | adapter SHA256 `98230128a898d09cb06f203bdb1118d71a15b28c5c102c0dd147ac02ce880e3e` | material/action 进一步退化；禁止上传 |
| 本地否决 checkpoint | i01_action_distill_r64_ep3 | `checkpoints/i01_action_distill_r64_ep3/` | adapter SHA256 `67273f14373b4f7ee14c6077cba3ebf0b6f75336abf0491137901d1241c8a875`；MD5 `6dc62479` | action 停止改善但语义未涨，world 方向性大退；禁止上传 |
| 最新失败实验包 | seed_cotfix_v1_lora_ep1 | `submissions/seed_cotfix_v1_lora_ep1_platform/` | adapter MD5 `3bfc8803`；SHA256 `0b3bfea5...` | 线上 0.8674，已证伪；仅暂存交付 |

`submissions/` 同时保留已登记历史包、本地候选包和当前I19-world最高点严格两文件包；不再维护易失真的手工总数。历史riders r64 E2的本地标准提交包仍缺失；各包的允许角色、哈希、线上状态与复现边界以本表逐项记录为准。

I-10 根目录最终产物为 `checkpoints/seed_teacher_r64_lr1e4_ep3/adapter_model.safetensors`，SHA256与E3同为 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2`。三个checkpoint和根目录均未保存optimizer、scheduler或RNG状态。

## 已完成并本地否决：seed_teacher_e3_dpo_rec_o1hard_v1

| 项 | 启动前记录 |
|---|---|
| 目的 | 在当前最高保留的非融合单adapter I-10 E3上只修正推荐金标与同域历史假负例的相对排序；输出仍是一个从O6可复现的r64 adapter，不采用I-13参数拼接路线 |
| 父模型/允许角色 | I-10 E3 `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1995/`，adapter SHA256 `37678b2516...8fc2`；成功保留checkpoint，获准同时作为可训练policy初始化和冻结DPO reference；不是失败checkpoint |
| reference机制 | LLaMA-Factory默认LoRA DPO会对policy调用`disable_adapter()`并错误锚到O6。本配置显式设置`ref_model=O6`与`ref_model_adapters=I-10 E3`，由加载器合并出冻结E3 reference；policy通过`create_new_adapter: false`继续同一E3 adapter |
| 正式训练数据 | `assets/derived/processed/data_o1_reward_preference_v1_train.jsonl`，`D(O1)` 15,382对，SHA256 `579171020e764b9c360b94493257848e6408487c7ca6b0dd8ba1efa76c34b52e`；video/ad/prod/living=`11,845/1,420/1,349/768`；O1派生100%，O2/T/E/teacher/model-rollout 0 |
| 构造 | `scripts/data/build_o1_reward_preference_v1.py` SHA256 `49c0cddd3b211a491b73eb692422de5bf2a2d9a5b0956584f4c32f1d365af475`；上游`data_seed_clean_v1` 32,480行/SHA256 `e526caea...d309`；构造审计`logs/data/o1_reward_preference_v1_audit.json` SHA256 `d95edce9...07f` |
| pair语义 | chosen逐字节保留O1派生目标；rejected只替换答案最终SID，候选必须来自同一输入历史、同域，并排除完整题面组的全部已知正例；2,185条无合格同域负例的推荐行直接跳过，不跨域凑数 |
| E类holdout | `assets/evaluation/holdout/data_o1_reward_preference_v1_holdout.jsonl`，1,784对，SHA256 `1c7292cb...696e`；按完整题面哈希与训练集切分，交集0；只作父偏好诊断和训练后漂移门禁，配置不引用、训练梯度为0 |
| 父偏好审计 | `logs/probe/i16_e3_parent_preference_audit.json` SHA256 `d1790cff...eda6`；每任务确定性64对、共320对。chosen原始胜率action/ad/live/prod/video=`93.75%/43.75%/32.8125%/17.1875%/34.375%`；因此只训练四推荐域，阻断1,392个action训练候选 |
| 截断/格式 | 全候选18,558对source/chosen/rejected截断均0，截断后负例SID缺失0；正式推荐prompt最大2,924 token、response最大1,085 token。LLaMA-Factory完整预处理15,382/15,382对，无drop；配置解析为DPO sigmoid beta0.1/max_steps600/lr1e-6/W&B |
| 配置 | `configs/active/seed_teacher_e3_dpo_rec_o1hard_v1.yaml`，启动前SHA256 `517d68645fd2af607cba89b487f5e9b5ef060fd748d2d4af5e9cf90b5e0c7f0a`；完成后加历史禁启动头，当前SHA256 `15260f8b6c69705882c9e31b22f337aeb71d0685407c40c53740ab59e816ddef`；单卡、r64/alpha64、sigmoid DPO beta0.1、lr1e-6 cosine、warmup30步、effective batch8、600步 |
| checkpoint策略 | step200/400/600各保存adapter-only，数据累计暴露约10.40%/20.80%/31.21%；最多三份，不保存optimizer/scheduler/RNG；训练后使用action与非推荐结构门禁否决漂移，不以本地指标估线上分数 |
| 持久启动 | 2026-07-14 12:19:46 UTC；单卡`GPU-717b...98c2b`；detached session由PID1接管且无TTY，退出码单独落盘；W&B [`packufor`](https://wandb.ai/3120252125-/llmrec-2026/runs/packufor)。加载日志确认policy为可训练E3 adapter、reference为O6加载同一E3 adapter后冻结合并；15,382/15,382 pairs预处理成功 |
| step200 | adapter SHA256 `96817d4c...fac04`；训练batch偏好accuracy 0.95/margin 0.2700。锁定320对审计：推荐聚合raw win 32.03%→52.73%，四域mean margin均提升，action raw/normalized win保持93.75%/82.8125%；但推荐gold mean-logp下降0.01185293（限0.01），ad/prod下降0.02745775/0.02071816（各限0.02），因此未过完整门禁 |
| step400 | adapter SHA256 `291c0b3e...eb94`；推荐聚合raw win升至57.03125%，四域mean margin仍全部高于父模型，action raw/normalized仍为93.75%/82.8125%；推荐gold mean-logp下降扩大到0.02029787，ad/prod为0.04661955/0.02984328，因此未过完整门禁 |
| 输出边界 | 不从任何中途状态resume。step200/400/600只作本实验候选与轨迹证据，禁止打包、上传或成为后续训练父模型；I-17已重新从I-10 E3启动 |
| step600 | adapter SHA256 `85cb7f00...73f00`；推荐聚合raw win 58.59375%，四域mean margin均高于父模型，action raw/normalized仍为93.75%/82.8125%；推荐gold mean-logp下降0.02149041，ad/prod为0.04899744/0.03203897，因此未过完整门禁 |
| 训练完成 | 600/600 steps；runtime 2,190.54s（36m30.54s）；train loss 0.58184869；退出码0；W&B服务端`finished`。三个checkpoint均无optimizer/scheduler/RNG；没有任何候选进入itemic provisional gate，因此按预注册规则直接拒绝，不打包、不上传 |
| 冻结门禁 | 预注册镜像`configs/evaluation/i16_o1hard_checkpoint_gate_preregister.json` SHA256 `6c097f9c...f223`，与checkpoint200前运行时文件逐字节一致；结果`configs/evaluation/i16_o1hard_checkpoint_gate_result.json` SHA256 `69b38426...e0c7` |
| 状态 | **COMPLETE_LOCAL_REJECT_ALL_CANDIDATES_FAILED_GOLD_LOGP_GATE** |

## 已完成并线上0.9727：seed_teacher_e3_dpo_rec_o1hard_lowdose_v2

| 项 | 启动前记录 |
|---|---|
| 目的 | I-16已经验证推荐排序方向，但step200的gold-logp保护线窄幅失败；I-17只降低累计更新量，寻找同时满足排序收益与父能力保护的剂量窗口 |
| 父模型/隔离 | 重新从I-10 E3 `checkpoint-1995`启动，SHA256 `37678b2516...8fc2`；显式冻结同一E3作reference。禁止加载、resume、合并或热启任何I-16 checkpoint |
| 数据 | 与I-16逐字节相同的`D(O1)`推荐pair 15,382对，SHA256 `57917102...52e`；O2/T/E/teacher/model-rollout均0，不新建数据资产 |
| 唯一机制变量 | 峰值lr `1e-6→7e-7`；`cosine(600总步)`改为短程`constant_with_warmup(200总步)`，warmup仍30步。step200离散累计LR面积`1.2985e-4`，为I-16 step200 `1.73495529e-4`的74.8434%；beta0.1、effective batch8、reference、数据与adapter结构不变 |
| 候选与门禁 | step100/150/200，对应800/1,200/1,600 pair暴露；完整门槛逐字沿用I-16，见`configs/evaluation/i17_o1hard_lowdose_checkpoint_gate.json`，SHA256 `01115844...9f1`。按最早全通过点选择；看见结果后不放宽阈值 |
| 配置 | `configs/active/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2.yaml`，启动前SHA256 `04975b5a79191fd638d5d24d68c8c85607a6a9a414766dd04ee1839da99d419e`；完成后加历史禁启动头，当前SHA256 `648c59e330fca3d082e30b1e7f073512f562e56621bfa4a86f7e7552035168b5`；单卡W&B；adapter-only每50步保存，最多4份，不保存optimizer/scheduler/RNG |
| 统计边界 | I-17由I-16同一holdout轨迹驱动，复用门禁属于自适应选择；通过只说明机制与灾难安全，不是无偏验证、线上估分或涨分保证 |
| 持久启动 | 2026-07-14 12:57 UTC；I-16退出码0后在同一单卡`GPU-717b...98c2b`顺序启动；detached PID 2685166由PID1接管、无TTY；W&B [`pfjlvm70`](https://wandb.ai/3120252125-/llmrec-2026/runs/pfjlvm70)。加载日志确认policy可训练E3、reference合并冻结E3；15,382 examples、200 steps、40,370,176 trainable params |
| 训练完成 | 200/200 steps；runtime 799.399s（13m19.40s）；train loss 0.63655710；退出码0；W&B服务端`finished`。step50/100/150/200与根目录均无optimizer/scheduler/RNG；根目录adapter与step200逐字节一致 |
| 量化轨迹 | step100/150/200推荐聚合raw win=`43.359375%/48.046875%/50.0%`，gold平均token logp下降=`0.00326676/0.00560856/0.00798044`，最大单域下降=`0.00840225/0.01199127/0.01673081`；三点均过原门槛，action raw始终93.75%，normalized=`82.8125%/82.8125%/84.375%` |
| 选点 | 按“最早全通过”规则选step100，adapter/config SHA256 `a3b2fc9c...2d1dc` / `65b21290...90f3`。四推荐域mean margin均高于父模型；不因step150/200排序更强而牺牲更大gold漂移 |
| 硬门禁 | step100临时merge后itemic断裂0/60=`PASS`；action复读6/30、选择题格式6/8、简单题4/8只作diagnostic。日志SHA256 `b538bcef...aab9`，临时merge已删 |
| 门禁结果 | `configs/evaluation/i17_o1hard_lowdose_checkpoint_gate_result.json` SHA256 `036280c2...af80`；明确记录同holdout自适应复用，不是线上估分 |
| 候选包 | `submissions/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2_step100_platform/`严格两个文件，与step100逐字节一致；adapter/config SHA256 `a3b2fc9c...2d1dc` / `65b21290...90f3`；已上传并完成固定协议评测 |
| 线上结果 | 2026-07-14 21:38:04启动、1h10m34s；总分0.9727；八项=`0.2453/0.1077/0.0380/0.0960/0.1156/0.1274/0.1044/0.1383`。相对I-13总分-0.0251；相对I-12推荐合计+0.0101、用户合计-0.0142、总分-0.0041。I-10 E3固定协议父分缺失，禁止解释为DPO净收益或净伤害 |
| 线上日志 | `logs/eval/seed_teacher_e3_dpo_rec_o1hard_lowdose_v2_step100_20260714.log`，2,657,186 bytes，SHA256 `5e7a0dff1a9b9048862f00eed0f7a67094bb01acfc15b62f60d776c03dca3fc7`；action1024、itemic 7次race-average、8/8完成、失败0；evalTaskId `eval-task-eeve0r-1784036284` |
| 状态 | **COMPLETE_ONLINE_0.9727_NO_LEADERBOARD_GAIN_PARENT_BRIDGE_PENDING** |

## 失败收档：seed_clean_r80_lr1e4_ep3

| 项 | 收档记录 |
|---|---|
| 配置 | 启动时`configs/active/seed_clean_r80_lr1e4_ep3.yaml`，SHA256 `7fa138d0cca7381af8ec0430ac697cf9d68c868e26a88aca6739fe2522ab2ae6`；现已添加`ABORTED...DO_NOT_RESUME`禁用头，禁止再次启动 |
| 启动 | 2026-07-14 03:33 UTC，单卡；2,628 packed examples，657 steps/epoch、1,971 total；50,462,720 trainable params |
| 中断 | 日志终止于step 1,886/1,971（epoch约2.87），未产生E3；无Traceback、OOM、NaN或训练器主动失败，cgroup `oom_kill=0`。根因判定为前台训练绑定临时PTY/执行会话，会话到期后进程被回收，不是数据或训练超参错误 |
| W&B | [`s7fskx9u`](https://wandb.ai/3120252125-/llmrec-2026/runs/s7fskx9u)，2026-07-14 05:22 UTC复核服务端为`crashed` |
| E1证据 | `checkpoints/seed_clean_r80_lr1e4_ep3/checkpoint-657/`，adapter SHA256 `a5c7c1b8d347140727d1f89d777fac4218d8204e6fef8b0f816fbd8b80d56aef`，adapter config SHA256 `f56a1cc8d2b52caf4caebcb2c9a9a3d179514e37817e2787e389f8ac5d04f048` |
| E2证据 | `checkpoints/seed_clean_r80_lr1e4_ep3/checkpoint-1314/`，adapter SHA256 `753d131870499f88b22a211c62637a4be7ab5f269d2ce5fa4ce18ffcfa527f51`，adapter config SHA256 `f56a1cc8d2b52caf4caebcb2c9a9a3d179514e37817e2787e389f8ac5d04f048` |
| 故障日志 | `logs/train/seed_clean_r80_lr1e4_ep3.log`，SHA256 `568eedb2adccc04594d8d20f7084157dfe3983fe7f30019a5e8b1b37ba811330` |
| 允许角色 | `checkpoints/seed_clean_r80_lr1e4_ep3/`及其中E1/E2仅作中断诊断证据；**禁止resume、warm start、merge、评测、打包或上传**；不存在可用E3 |
| 状态 | **CRASHED_INFRASTRUCTURE_SESSION_ARCHIVED** |

## 最新已评测纯O1单体：seed_clean_r80_lr1e4_ep3_rerun1

| 项 | 完成记录 |
|---|---|
| 目的 | 回到无融合、无teacher的单模型路线，检验纯O1单体r80能否保持material/video先验并提供额外容量；不声称这是I-10的单变量消融 |
| 基座 | O6 `OneReason-0.8B`，从基座干净启动；不加载失败运行或任何其他adapter/checkpoint |
| 数据 | `assets/derived/processed/data_seed_clean_v1.jsonl`，`D(O1)` 32,480行，SHA256 `e526caea4a1afd8befbd5d266fb80d0378a5bf7eff90fdacd14934332d64d309`；O1 100%，O2/T/E 0 |
| 构造 | `scripts/data/build_seed_clean_v1.py`，SHA256 `2d01951d0e6d3e0f406d3a59a74e35ab8f70b8cb51e3295f8f1792714d7dc214`；全部32,480个target保留；12,744条推荐冗余CoT转no-think，602条topic对齐no-think |
| 数据审计 | `logs/data/seed_clean_v1_audit.json`，SHA256 `15767552e1feac3c21e500207cb76a4805b118dda061f6b2cc3c6116255c3b11`；target token 5,845,479，action 138,680（2.372432%） |
| 配置 | 启动时`configs/active/seed_clean_r80_lr1e4_ep3_rerun1.yaml` SHA256 `69a8b56b1e33a960fbbb6da4bd517c0d810e2d51299c7b40674db8a994486d76`；除新output/run name及故障隔离注释外，训练字段与首次运行一致：单卡r80/alpha80/dropout0.05、lr1e-4 cosine、warmup0.03、effective batch4、cutoff16384、3 epoch、W&B online。完成后添加`HISTORICAL_ONLY_AFTER_SUCCESS`防覆盖头，当前SHA256 `f94bb328ec3cadc1992a9ee107d7dd75dd651b29bddf53339fddabc0c7595fda` |
| 持久启动 | 启动时`scripts/train/launch_wandb_detached.sh` SHA256 `6e065edfcaf96769616b1cc1dd33d5489f521d1cbbb931c19649907e9e732d71`，使用`nohup + setsid --fork --wait`，stdin关闭、日志/PID/退出码落盘；跨独立exec确认实际session leader PID 2507845由PID1接管、SID 2507845且无TTY。启动后将脚本去掉瞬时`--fork`父进程以使后续PID文件直接准确，当前SHA256 `f0238644850754aaf40cc9e80fb6d88a6e5f01a38e6828648edea03849684f45`；本次PID记录已校正为2507845，训练未重启且正常退出 |
| checkpoint策略 | 每轮保存adapter-only，最多E1/E2/E3三份；禁止optimizer/scheduler/RNG状态；训练轨迹用于观察剂量，不用旧本地proxy估线上分 |
| 预期与失败边界 | 高收益分支是material 8→9 hit；同时观察action/topic容量与推荐先验。结构断裂或格式灾难为本地硬失败；线上若material/video显著低于当前固定协议主模型则否决。单次线上差异不自动称稳定提升 |
| 输出 | `checkpoints/seed_clean_r80_lr1e4_ep3_rerun1/`；启动前不存在，从O6干净创建 |
| 启动 | 2026-07-14 05:25 UTC，单卡`GPU-717bca2c-8756-e333-16e5-c3a3eda98c2b`；2,628 packed examples，657 steps/epoch、1,971 total；50,462,720 trainable params |
| 日志 | `logs/train/seed_clean_r80_lr1e4_ep3_rerun1.log` SHA256 `5702af2d716bc97f12c87fa65ddc82565723f33b5ab12ce658d87e71eb61eb98`；PID 2507845已退出、同名`.exit_code`为0；未发现Traceback、OOM、RuntimeError、Exception或Killed |
| W&B | [`3grnqgsh`](https://wandb.ai/3120252125-/llmrec-2026/runs/3grnqgsh)，服务端`finished`；global step1,971、epoch3、train loss 1.3422074429、runtime 4,862.2999s、1.621 samples/s、0.405 steps/s |
| E1 | `checkpoints/seed_clean_r80_lr1e4_ep3_rerun1/checkpoint-657/`，trainer state精确为step657/epoch1；adapter SHA256 `f441b83fbeb9ef4cb83f49e474589621badd48e6a6a5161e1ff684f6c54f187d`，adapter config SHA256 `25381f212cccced12f3544f9a7ced3d588f550fa02616b007544daca3966a6ad`；无optimizer/scheduler/RNG文件 |
| E2 | `checkpoints/seed_clean_r80_lr1e4_ep3_rerun1/checkpoint-1314/`，trainer state精确为step1314/epoch2；adapter SHA256 `182ba79b337dc957ff47d48f3c5d224205197a668de78f68785c52aff7ec79a1`，adapter config SHA256 `25381f212cccced12f3544f9a7ced3d588f550fa02616b007544daca3966a6ad`；无optimizer/scheduler/RNG文件 |
| E3 | `checkpoints/seed_clean_r80_lr1e4_ep3_rerun1/checkpoint-1971/`，trainer state精确为step1971/epoch3；adapter SHA256 `477d2acd1934bf12cb70b6f88a691328116778a265874b009ee5cea88760837b`，adapter config SHA256 `25381f212cccced12f3544f9a7ced3d588f550fa02616b007544daca3966a6ad`；无optimizer/scheduler/RNG文件 |
| 最终产物 | `checkpoints/seed_clean_r80_lr1e4_ep3_rerun1/adapter_model.safetensors` SHA256 `477d2acd1934bf12cb70b6f88a691328116778a265874b009ee5cea88760837b`，与E3逐字节一致；根目录adapter config SHA256 `25381f212cccced12f3544f9a7ced3d588f550fa02616b007544daca3966a6ad`；全目录无optimizer/scheduler/RNG，训练结束后目标GPU为0 MiB/0% |
| 平台包 | `submissions/seed_clean_r80_lr1e4_ep3_rerun1_platform/`严格两个文件；adapter 201,903,440 bytes，SHA256 `477d2acd1934bf12cb70b6f88a691328116778a265874b009ee5cea88760837b`；config 1,138 bytes，SHA256 `25381f212cccced12f3544f9a7ced3d588f550fa02616b007544daca3966a6ad`；与最终产物逐字节一致，已评测为0.9518 |
| 训练结果 | `train_results.json` SHA256 `1b092787656556084bbb885e626b05a46016fdfdd271e5b1a55a9fc49d678c6e`；train loss 1.3422074429，runtime 1:21:02.29；训练loss没有预测线上结果，实际E3为0.9518 |
| 线上 | `seed_clean_r80_lr1e4_ep3_rerun1_V1_eval_20260714152001`；2026-07-14 15:20:05；1h10m32s；总分0.9518；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1045/0.0387/0.0480/0.1292/0.1414/0.1080/0.1368`；账号`SL1ACE8AD6710`。原始日志确认属于`platform-stable-v3.1-20260713` |
| 线上日志 | `logs/eval/seed_clean_r80_lr1e4_ep3_rerun1_20260714.log`，2,905,022 bytes，SHA256 `046a2e53b009206b1b88306c99682cc1a9444cc711a7820212e55efedf324153`；action1024、itemic 7次race-average、8/8完成、失败0；evalTaskId `eval-task-lfrrhq-1784013605` |
| 榜分对账 | 相对I-13总分-0.0460，逐项=`0/-0.0138/-0.0003/-0.0480/+0.0068/+0.0098/+0.0018/-0.0022`。I-13由两个adapter参数拼接而成，这个差值只说明I-14不能替换当前最高榜分，不用于判定直接训练路线优劣 |
| 非融合参考 | 相对I-11总分名义-0.0100，逐项=`0/-0.0061/-0.0009/-0.0192/+0.0136/0/+0.0027/0`。I-11虽不是参数拼接模型，但含164条teacher、从I-10 E3续训且为r64，仍不能隔离O1-only或rank80效应；单次差值也不称稳定差异 |
| 状态 | **EVALUATED_0.9518_CLEAN_SINGLE_MODEL_NOT_LEADERBOARD_REPLACEMENT**；纯O1直训r80同协议基线缺失，E1/E2未线上评测，不作路线因果否决 |

## I-35父模型与上一最高、r96包已验收：I19-world-residual r96

| 项 | 记录 |
|---|---|
| 名称边界 | 本节稳定名为`I19-world-residual`；仓内既有I-19仍指`i13_s875_dpo_rec_balhard_lowdose_v1`，两者不重编号、不混写 |
| 构造 | 独立复现I-13-like r80 parent冻结；1,573条授权Frinkleko clean world与1,573条八任务KL保持组成3,146行1:1混合，训练fresh r16；再以`scale=0.875`参数空间精确拼接为单个r96 adapter |
| Parent边界 | 实际parent `i13_repro_combined_r80_s875`线上`0.986703844`，与仓内原I-13 s875配方相同但权重非bitwise一致；禁止用原s875 `0.9978`或s800 `1.0048`替代该父分计算净增益 |
| 训练 | r16/alpha16/dropout0.05/all-linear；lr5e-5 cosine、warmup0.03、wd0.001、batch1×acc4、cutoff4096、bf16、seed19260821、1 epoch/787 steps；world为CE+0.05 parent KL，保持为0 CE+2.0 parent KL |
| 上一最高点/I-35父分 | `scale=0.875`原始总分`1.025259456`，后续同模型复测`1.025362611`；原始八项=`0.2453/0.1225/0.0399/0.0768/0.1326/0.1400/0.1080/0.1602`；原始evalTaskId `eval-task-g4y7us-1784436397`，复测evalTaskId `eval-task-ol0sje-1784615810`，平台模型ID均为`md-cm6gw1-1784436350784564154` |
| 相邻点 | `scale=0.75/0.8/0.9`总分=`0.996570849/0.990238620/0.977796092`；三点video均`0.0576`，只有0.875为`0.0768`，所以最高点可能是孤立非单调效应或评测噪声，四点都只有一次观测 |
| 报告哈希 | 实际r80 parent `a63a45c3...15ed0`；r16 residual `144ee8ef...d4d6d`；0.875 r96 `4fba17eb...078e`；3,146行混合 `a8af6884...edb86` |
| 当前存在性 | r96严格两文件包已在`submissions/i19_world_external_r96_s875_platform/`验收：adapter/config SHA256=`4fba17eb...078e`/`78b62143...1b64f`，242,273,688/1,074 bytes，r96/alpha96、392 tensors。报告所列parent/residual、发布数据目录、builder/trainer/audit/config或W&B run仍缺 |
| 下一动作 | 接收实际r80 parent/r16 residual、数据发布件、源码/配置/combine audit、训练日志与W&B run；逐项通过哈希、行数、路由、父身份和逐tensor拼接恒等式验收，闭合完整复现链 |
| 详情 | [`I19_WORLD_RESIDUAL_HANDOFF.md`](I19_WORLD_RESIDUAL_HANDOFF.md) |
| 状态 | **I35_PARENT_PREVIOUS_HIGHEST_PACKAGE_VERIFIED_REPRO_CHAIN_INCOMPLETE** |

## 已完成并本地否决：i30_r96_material_teacher_retkl_r8_v1

| 项 | 预注册记录 |
|---|---|
| 目的 | 在当前最高观测`1.025259456` r96上只迁移I-23的material优势；不改world/video等第二个训练目标，不把离线logp解释成线上估分 |
| 父模型/允许角色 | `submissions/i19_world_external_r96_s875_platform/`，adapter/config SHA256=`4fba17eb...078e`/`78b62143...1b64f`；允许作为本实验冻结parent和fresh r8初始化基点。该许可不补齐其仍缺的原r80/r16复现链 |
| teacher/允许角色 | I-23 E3严格包，adapter/config SHA256=`0e5fa9bb...c6b8`/`b3f2a1b5...e7e7`；只允许作material构造评分器和material响应KL teacher，不作为policy初始化，不复用I-24/I-25/I-28失败点 |
| 构造选择 | `data_seed_teacher_v1`的O1 material按seed19260831每向先留128条门，再取1,024条构造池；以相同O6上I-23减r96的答案体gold mean-logp排序，每向只选前256条严格正优势行。固定算法在任何模型forward前登记 |
| 训练混合 | `assets/derived/processed/data_i30_r96_material_teacher_retkl_v1.jsonl`，2,048行/17,701,923 bytes/SHA256 `0df9a192...c4a4`；512条material teacher+1,536条七任务r96 KL-only保持，严格1:3。T/E训练行0；builder/ledger/audit SHA256=`a57c17e2...ab8a`/`b303a501...5b3d`/`6b46775b...9226` |
| 损失/训练物理 | material答案体CE+0.5 I-23 KL+0.1 r96 KL；保持4.0 r96 KL。fresh r8/alpha8/dropout0.05/all-linear，lr2e-5 cosine、warmup0.03、wd0.001、batch1×acc4、cutoff16384、单GPU、W&B online、1轮512步 |
| 冻结候选 | step128→256→384→512取最早全过点；residual scale固定1.0，禁止新增scale网格。两向material门、七任务保持、world MC和itemic零断裂任一失败则本地关闭 |
| 独立门 | `assets/evaluation/holdout/data_i30_r96_material_teacher_gate_v1.jsonl`，704行/SHA256 `dd744ee2...7a0`；material双向各128、其余七任务各64，与全部训练prompt交集0，训练梯度为0 |
| 配置/实现 | config/trainer/launcher SHA256=`e351827c...9db4`/`60fa31d5...9f39`/`97d55880...256b`；结果前门SHA256 `895a9a62...eadc`。全量O6 tokenizer预检2,048/2,048，路由512/1536，最长8,864<16,384 |
| 正式训练 | `checkpoints/i30_r96_material_teacher_retkl_r8_v1/`；单GPU0、W&B [`ir2r0nd4`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/ir2r0nd4)，512/512步、928.9s、train loss `2.1403`、路由material/retention=`512/1536`，正常退出。保留adapter-only点：`checkpoints/i30_r96_material_teacher_retkl_r8_v1/checkpoint-128/`、`checkpoints/i30_r96_material_teacher_retkl_r8_v1/checkpoint-256/`、`checkpoints/i30_r96_material_teacher_retkl_r8_v1/checkpoint-384/`、`checkpoints/i30_r96_material_teacher_retkl_r8_v1/checkpoint-512/`；adapter SHA256=`3c97c61c...ca2`/`ec78d0bd...45d7`/`6fc60bbb...b6ae`/`a52496c6...daf9`，均r8/alpha8且无optimizer/scheduler/RNG |
| 精确组合 | 与冻结r96按scale1拼为r104：`checkpoints/i30_r96_material_teacher_retkl_r8_v1_combined_r104_step128/`、`checkpoints/i30_r96_material_teacher_retkl_r8_v1_combined_r104_step256/`、`checkpoints/i30_r96_material_teacher_retkl_r8_v1_combined_r104_step384/`、`checkpoints/i30_r96_material_teacher_retkl_r8_v1_combined_r104_step512/`；adapter SHA256=`bf9ea69c...d42`/`506fc314...e3b`/`fd772199...5e4`/`192f6a93...21d`，config统一`46bd7146...205` |
| 冻结门结果 | 704/704完成；正式报告`logs/probe/i30_r96_material_teacher_gate_v1.json` SHA256 `3e12f088...92bf`。四点两向material联合门全部失败；最佳gold mean-logp delta约`+0.00207`，但改善率不足且candidate-to-I23 KL delta均为正。保持KL均小于0.005且gold保护通过，但topic/video/prod/ad/living/world若干Top-1低于0.99；world精确正确数为49/50/49/49，对照49，均不下降 |
| 决策 | `earliest_teacher_forced_pass=null`；依预注册关闭I-30，不运行下一层itemic，不生成submission，不上传，不把离线结果解释成线上估分，不续训或后验搜索失败残差scale |
| 状态 | **COMPLETE_LOCAL_GATE_FAIL_NO_PACKAGE_NO_UPLOAD** |

## 已完成开发探针并获用户授权线上探索：i31_r96_i23_exact_interpolation_dev_probe

| 项 | 记录 |
|---|---|
| 目的/边界 | 零训练检查直接参数插值能否绕过I-30训练残差的反向teacher-KL漂移；复用已看过的I-30 holdout只作开发，不能估线上分。2026-07-21用户明确要求取消本地门禁对上传的否决权，以真实平台分数决定是否继续该轴 |
| 冻结设计 | `configs/evaluation/i31_r96_i23_exact_interpolation_dev_probe.json` SHA256 `409fd7ad...c957`；结果前固定`lambda=0.05/0.10/0.20/0.40`，精确恒等式`delta=(1-lambda)·delta_r96+lambda·delta_I23`，统一r160/alpha160，不追加后验点 |
| 候选 | `checkpoints/i31_r96_i23_exact_interp_r160_l005/`、`checkpoints/i31_r96_i23_exact_interp_r160_l010/`、`checkpoints/i31_r96_i23_exact_interp_r160_l020/`、`checkpoints/i31_r96_i23_exact_interp_r160_l040/`；adapter SHA256=`4c28c7ca...1bb`/`5ad2e789...e64c`/`09e71cd3...64d7`/`39bc5d49...4df9`，config统一`6595df1b...92e6`，各403,754,928 bytes/392 tensors |
| 开发结果 | 报告`logs/probe/i31_r96_i23_exact_interpolation_dev_probe.json` SHA256 `be917998...6db0`。`lambda=0.10`两向material gold delta=`+0.00533/+0.00048`、改善率=`58.59%/57.81%`、teacher-KL delta=`-0.00060/-0.00049`，是唯一两向material均过点；但world精确49→48且video/ad等保持未优于I-30。`lambda=0.05`world保持49但sid2desc gold delta为负；更高lambda的world为47/41 |
| 平台合法化 | 将唯一两向material均改善的`lambda=0.10`候选逐模块最优截断SVD由r160压至r128并转BF16；`checkpoints/i31_r96_i23_exact_interp_r128_l010_svd_bf16/` adapter/config SHA256=`93d247a2...803f`/`daa3106d...07f3`，392 tensors，总计161,535,020 bytes；全局保留谱能量`0.9999999329`、BF16后相对Frobenius误差`0.00236582` |
| 决策 | 不再用开发门禁阻止线上探索。只提交预先存在的`lambda=0.10`点，不后验新增lambda；真实平台总分决定该轴继续或关闭，尚未声称涨分 |
| 状态 | **READY_FOR_ONLINE_SCORE_FIRST_PROBE** |

## 已完成并本地否决：i32_task_restore_retkl_r8_v1

| 项 | 预注册记录 |
|---|---|
| 目的 | 从I31唯一两向material有效的`lambda=0.10`起点学习task-conditioned恢复：保留I23 material移动，同时把七任务拉回当前1.0253 r96；离线门只作验收，不估线上分 |
| 起点/参考 | 起点`checkpoints/i31_r96_i23_exact_interp_r160_l010/`，adapter/config SHA256=`5ad2e789...e64c`/`6595df1b...92e6`，恒等式`0.9*r96+0.1*I23`；冻结r96/I23 adapter SHA256=`4fba17eb...078e`/`0e5fa9bb...c6b8`分别只作retention/material reference |
| 正式训练数据 | 复用已登记`data_i30_r96_material_teacher_retkl_v1.jsonl`，2,048行/SHA256 `0df9a192...c4a4`；material/retention=`512/1536`严格1:3，T/E训练行0。I32不创建或改写训练数据，只改变冻结损失路由，完整上游/构建器/混合比例沿用该D资产登记 |
| 损失/训练 | fresh r8/alpha8/dropout0.05/all-linear；material只做`4.0*I23-to-policy KL`，retention只做`8.0*r96-to-policy KL`，gold CE均0。lr1e-5 cosine、warmup0.03、wd0.001、batch1×acc4、cutoff16384、单GPU、W&B online、512步一轮；初始fresh residual logits必须与r160起点max_abs<=1e-4 |
| 新验收门 | `assets/evaluation/holdout/data_i32_task_restore_gate_v1.jsonl`，656行/4,998,481 bytes/SHA256 `f7510675...2f62`；material双向各128、action/topic/video/prod/ad/living各64、world16，与I30 train/dev prompt交集0。builder/audit SHA256=`758bda37...431c`/`035ca45e...ea8d`；world因严格排除后只有23个新prompt，结果前固定16，不复用旧门 |
| 候选/停止 | 训练保存64间隔共8点，但科学候选只允许step64→128→256→512，分别与起点精确拼成固定scale1 r168；最早全过material双向、七任务保持、world exact和itemic 0/60者才打包。不得看192/320/384/448选点，不扫scale；无通过点即本地关闭 |
| 实现哈希 | 训练前冻结trainer/config/evaluator/gate/launcher SHA256=`f6711e71...bce`/`58941c4b...7bc`/`d8682ae2...5d9c`/`41cb68a3...a1d`/`5ef67e1f...4830`；py_compile/self-test与全量2048行tokenizer预检PASS，路由512/1536、最长8,864<16,384。最终evaluator只机械修正两个旧常量为冻结`EXPECTED_COUNTS`引用，SHA256=`9bcde447...c161` |
| 正式训练 | `checkpoints/i32_task_restore_retkl_r8_v1/`；单GPU0、W&B [`2lu9hw9k`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/2lu9hw9k)，512/512步、runtime1,074.3476s、train loss`0.0819712`、路由material/retention=`512/1536`、step0 fresh residual max_abs=0，正常退出。trainer log SHA256=`4f512ae8...6a8a`；无optimizer/scheduler/RNG |
| 训练保存点 | `checkpoint-64/128/192/256/320/384/448/512` adapter SHA256=`ff72efd8...c44`/`5e06010a...e33`/`16750417...641`/`888eb67e...01b`/`23854f67...a70`/`77ed9305...c31`/`aacb34bd...fe7`/`9e71d6cb...107`，config统一`fbf08a81...d67`。192/320/384/448依预注册仅作审计、未评估 |
| 精确组合 | 科学候选step64/128/256/512与r160按scale1拼为r168；adapter SHA256=`81e60e91...58b`/`6c0756e7...523`/`91190212...9d2`/`2c1c0f84...690`，config统一`b31258a0...031`。逐点combine audit SHA256=`65749f01...bf5f`/`5aa841cc...c835`/`30358b2f...2ae`/`f35cbfb7...9cdc` |
| 冻结门结果 | 656/656完成；报告`logs/probe/i32_task_restore_gate_v1.json` SHA256=`c2912079...c4cd`。四点world exact均为parent/candidate=`11/11`，全部不退；material desc2sid/sid2desc改善率依次=`48.44/52.34%`、`53.91/51.56%`、`46.88/56.25%`、`48.44/54.69%`，无一点双向同时达到55%。四点KL均小于0.005且gold保护通过，但topic/video/prod/ad/living等多项Top-1低于0.99 |
| 决策 | 原冻结结论仍为`earliest_teacher_forced_pass=null`且不自动进入itemic/包/上传。用户在知晓失败后明确以每日5次额度授权step128作一次门外探索：该点因双向material gold均为正、desc2sid最强、world 11/16不退而优先；提交级itemic 0/60后已打包。此例外不允许回看中间点、扫scale、续训、复用失败残差或自动提交其余三点 |
| 探索包 | 原FP32 r168因423,941,100 bytes超过400MB被拒，BF16 r168又因rank168超过平台上限128被拒，均无evalTaskId。唯一待传为`submissions/i32_task_restore_retkl_r128_step128_svd_bf16_platform/`：逐模块截断SVD、r128/alpha128、392 BF16 tensors，adapter/config SHA256=`edadde1f...d46`/`daa3106d...7f3`，总计161,535,020 bytes。656行material两向gold均值`+0.001988/+0.000350`、world 11/16；precheck itemic 0/60、选择题8/8、action复读诊断2/30。package audit SHA256=`10190999...0705` |
| 状态 | **COMPLETE_LOCAL_GATE_FAIL_USER_AUTHORIZED_STEP128_R128_BF16_PACKAGE_READY_NOT_ONLINE_EVALUATED** |

## 上一完整本地复现主模型：e3_userres_r80_retkl_v3_s800

| 项 | 记录 |
|---|---|
| 目的/构造 | 不重训；保持I-10 E3 r64父adapter不变，只将I-12 r16用户残差系数从s875的`0.875`降为`0.80`，精确拼为相对O6的单个r80 adapter |
| 提交包 | `submissions/e3_userres_r80_retkl_v3_s800_platform/`严格只有两文件；adapter 201,903,440 bytes，SHA256 `bb86eb8af0efd3560b7b7c8440f3830627e9255f4fcc2265b9274a27668f63c6`；config 1,139 bytes，SHA256 `e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0` |
| 线上 | 首测UI `e3_userres_r80_retkl_v3_s800_V1_eval_20260717000053`；平台记录时间2026-07-17 00:01:02；1h10m7s；总分`1.0037`；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1163/0.0401/0.0960/0.1224/0.1372/0.1089/0.1375`。同包复测UI `e3_userres_r80_retkl_v3_s800_V1_eval_20260717000053_copy`；平台时间2026-07-17 15:07:19；1h9m3s；总分`1.0048`；八项=`0.2453/0.1166/0.0401/0.0960/0.1224/0.1358/0.1089/0.1398`；账号均`SL1ACE8AD6710` |
| 线上日志 | `logs/eval/e3_userres_r80_retkl_v3_s800_20260717.log`，2,737,350 bytes，SHA256 `85f0357bd151b2f6fe1053915081aa9cfefad6c01221228738983bd020a52eb6`；action1024、itemic 7次race-average、8/8生成与计分完成、失败0；evalTaskId `eval-task-olteal-1784217662` |
| 协议 | `platform-stable-v3.1-20260713`；与I-10 E3旧协议结果不可作差 |
| 相对s875 | 总分`+0.0059`；material/action/topic/video/prod/ad/live/world=`0/-0.0020/+0.0011/0/0/+0.0056/+0.0027/-0.0015`。用户两项合计`-0.0009`，推荐四项合计`+0.0083`，world `-0.0015` |
| 复测判读 | 同一包两次显示分差`+0.0011`，逐项显示差=`0/+0.0003/0/0/0/-0.0014/0/+0.0023`；按合理评测抖动处理，不当作第二个模型。两次均值`1.00425`、该包最好显示分`1.0048`；复测原始日志/evalTaskId尚未入仓 |
| 判读/下一发 | s750三点曲率中心约`1.0048`已经落入s800复测范围，缺少明确上行，故继续为备包。I-23残差轴和I-29 video分支均已关闭。I19-world r96随后成为I-35父模型，I-35 step548又刷新至1.034428585；暂停从s800派生的新world训练 |
| 状态 | **COMPLETE_FIXED_PROTOCOL_REPEAT_1.0037_1.0048_PREVIOUS_LOCAL_BASELINE** |

## 上一固定协议主模型：e3_userres_r80_retkl_v3_s875

| 项 | 记录 |
|---|---|
| 目的 | 在不重训的前提下收回I-12在ad等非用户任务上的部分漂移，同时尽量保留用户残差收益；用户已裁定本轮最后一次配额用于该提分候选而非E3协议桥 |
| 构造 | I-10 E3 r64与I-12 r16残差按低秩维拼接；唯一变化为残差系数`1.0 -> 0.875`，恒等式`delta_combined = delta_E3 + 0.875 * delta_residual`；组合实现`scripts/train/combine_lora_adapters.py` |
| 回归验证 | `--residual-scale 1.0`生成物与I-12逐字节一致：adapter/config SHA256 `3fe85158...87cc6` / `e3c3ace0...c4ac0`；缩放拼接CPU精确恒等式自测PASS |
| 用户审计 | 固定32 action+32 topic。0.875相对父模型CE变化为action `-0.0369122`、topic `-0.0123755`；分别保留full residual收益约93.2%和83.3% |
| 严格保持审计 | O1 `data_seed_teacher_v1`中按任务稳定哈希留出，逐字节排除I-12训练集；material desc2sid/sid2desc、video/prod/ad/live各96，共576条。0.875六任务平均父KL `0.00197349`，full residual为`0.00207764`，约下降5.0%；aggregate top-1一致率0.98024 vs 0.97973。该审计不保证任一线上子项上涨 |
| 审计证据 | `logs/probe/i13_userres_scale_pareto_full_20260714.json`，20,574 bytes，SHA256 `c937b9be...82fc`；小样本先导`logs/probe/i13_userres_scale_pareto_20260714.json`，17,073 bytes，SHA256 `a2e59102...f1e14` |
| 硬门禁 | itemic断裂0/60=`PASS`；action复读2/30、选择题格式6/8、占位符0/8、简单题4/8，全部与I-12一致。日志`logs/precheck/e3_userres_r80_retkl_v3_s875_precheck.log`，SHA256 `cbd32b15...66bda`；临时merge已删、GPU1归零 |
| 提交包 | `submissions/e3_userres_r80_retkl_v3_s875_platform/`严格两文件；adapter 201,903,440 bytes，SHA256 `71bc3c2c...ffd5b`；config 1,139 bytes，SHA256 `e3c3ace0...c4ac0`；组合审计`logs/model/e3_userres_r80_retkl_v3_s875_combine.json` |
| GitHub最高分实现发布 | `assets/derived/releases/e3_userres_r80_retkl_v3_s875/`完整提交I-10父训练数据与I-12残差训练数据的确定性gzip、manifest和小型原始审计；`configs/active/i13_repro_parent_r64_ep3.yaml`与`i13_repro_residual_r16_retkl_ep1.yaml`保留历史训练字段并改为portable路径；`scripts/reproduce/i13_highscore.sh`覆盖双数据校验/还原、两阶段单卡W&B训练和0.875精确拼接。两份数据已从发布件完整解压并与历史输入逐字节一致；用历史r64/r16源adapter重新拼接也与线上r80包逐字节一致 |
| 规则口径 | FAQ写明初赛基于OneReason-0.8B、允许蒸馏、全程不鼓励融合，并要求复赛结束提供单模型训练方案审核复现。“不鼓励”并非禁止；该包运行时是单个r80 adapter，参数由两个同基座LoRA拼接，当前按允许冲分处理，同时完整披露来源、拼接恒等式与单模型复现链 |
| 线上 | `e3_userres_r80_retkl_v3_s875_V1_eval_20260714004418`；平台记录时间2026-07-14 00:44:35；1h7m21s；总分0.9978；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1183/0.0390/0.0960/0.1224/0.1316/0.1062/0.1390`；账号`SL1ACE8AD6710` |
| 线上日志 | `logs/eval/e3_userres_r80_retkl_v3_s875_20260714.log`，2,777,778 bytes，SHA256 `9291f8bf87871bb93846dda4cfcf60d43812354fb87a18e6ef6a5a349bdb3315`；8/8任务、Failed tasks 0；evalTaskId `eval-task-9ie86v-1783961075` |
| 协议 | `platform-stable-v3.1-20260713`；action上限1024，itemic 7次race-average。与E3旧协议结果不可作差 |
| 同协议相对I-12 | 总分+0.0210；material 0、action -0.0023、topic -0.0003、video +0.0288、prod -0.0068、ad 0、live +0.0009、world +0.0007。用户两项合计-0.0026，推荐四项合计+0.0229 |
| 判读 | 缩放残差的总分方向得到一次线上支持，主要收益来自video而非用户两项。s875现为上一主模型；E3桥接仍缺失，不能声称相对父E3的净增益 |
| 状态 | **COMPLETE_FIXED_PROTOCOL_0.9978_PREVIOUS_MAIN** |

## 上一固定协议实验：e3_userres_r80_retkl_v3_ep1

| 项 | 预登记记录 |
|---|---|
| 父模型 | I-10 E3，线上单次0.9849；adapter SHA256 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2` |
| 数据 | `assets/derived/processed/data_user_residual_retention_v1.jsonl`，6,106行，SHA256 `bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0`；用户CE3,053/父KL保持3,053；无规则/T/E |
| 用户分支 | action1,752、合法2–5步topic1,301；完整history/target不改；164 teacher各一次；标准逐样本CE，仅闭合符/EOS 2x，父KL0.05 |
| 保持分支 | material desc2sid/sid2desc各281，video/prod/ad/live各565，D(O2.General) world231；只做E3 KL，权重2.0，不做gold CE |
| 配置 | `configs/history/e3_userres_r16_retkl_v3_ep1.yaml`，SHA256 `1b17a06551efdf6e90a9d7a797d774e87f9e5f658123f35cb0d2fd399b9d0556`；新r16/alpha16/dropout0.05、lr5e-5 cosine、effective batch4、cutoff16384、packing关闭、单卡1 epoch、`save_strategy: no`、W&B online |
| 实现验证 | CPU chunked CE/KL与adapter拼接自测PASS；真实E3模型2步烟测PASS。v1在step8发现101条world保持原生无think包装并修复路由；v2在step275发现ChatML EOS后换行使终止权重错位。两者无adapter且输出已删。v3真实action/topic模板回归确认闭合符/EOS 2x、尾换行1x；GPU1归零 |
| 训练 | GPU1；1,527/1,527 steps；45m43.40s；train loss1.1514281；W&B [`1xbo7k2e`](https://wandb.ai/3120252125-/llmrec-2026/runs/1xbo7k2e)服务端`finished`；无中间checkpoint/optimizer |
| 产物 | r16 adapter 40,422,168 bytes，SHA256 `e8caf0a3...4f98`；与E3 r64按低秩维精确拼接为r80/alpha80，201,903,440 bytes，SHA256 `3fe85158...87cc6`，组合审计 `logs/model/e3_userres_r80_retkl_v3_ep1_combine.json` |
| 配对机制审计 | 固定训练内32 action/32 topic/64 retention：action CE 0.3636767→0.3240751（-10.9%），topic 0.9066398→0.8917812（-1.6%）；保持KL均值0.0021131、top-1一致98.653%。日志`logs/probe/e3_userres_r16_retkl_v3_ep1_paired_audit.json`；不是线上估分 |
| 硬门禁 | r80临时merge后itemic断裂0/60=`PASS`；action复读2/30、选择题格式6/8、简单题4/8只作diagnostic。日志`logs/precheck/e3_userres_r80_retkl_v3_ep1_20260713.log`，临时merge已删，GPU1归零 |
| 上传包 | `submissions/e3_userres_r80_retkl_v3_ep1_platform/`严格两文件，与r80源逐字节一致；adapter/config SHA256 `3fe85158...87cc6` / `e3c3ace0...c4ac0` |
| 决策目标 | 用户两项+0.008～0.012，同时其余六项损失不超过0.002～0.003；这是机制验收目标，不是线上分数预测。门禁只做结构与父保持否决，不估总分 |
| 线上 | `e3_userres_r80_retkl_v3_ep1_V1_eval_20260713201614`；平台记录时间2026-07-13 20:16:34；1h7m28s；总分0.9768；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1206/0.0393/0.0672/0.1292/0.1316/0.1053/0.1383`；账号`SL1ACE8AD6710` |
| 线上日志 | `logs/eval/e3_userres_r80_retkl_v3_ep1_20260713.log`，2,642,720 bytes，SHA256 `151bddf09f301794885e66a9df7387d3141475daa8f0e9a249cc8b96381cf450`；8/8任务、Failed tasks 0；evalTaskId `eval-task-jnbjjq-1783944993` |
| 协议 | `platform-stable-v3.1-20260713`；action上限1024，itemic 7次race-average。与E3旧协议结果不可作差 |
| 同协议相对I-11 | 总分+0.0150；material 0、action +0.0100、topic -0.0003、video 0、prod +0.0136、ad -0.0098、live 0、world +0.0015。用户两项合计+0.0097，推荐四项合计+0.0038 |
| 判读 | I-12同协议优于I-11，但被I-13高0.0210；父E3仍缺固定协议分数，继续禁止跨协议比较和用户残差相对E3的净升级结论 |
| 状态 | **COMPLETE_FIXED_PROTOCOL_0.9768_SUPERSEDED_BY_I13** |

## 最新实验：seed_teacher_e3_cont_r64_lr2e5_ep1

| 项 | 记录 |
|---|---|
| 父模型 | I-10 E3 `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1995/`；线上0.9849；adapter SHA256 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2` |
| 数据 | 与I-10逐字节相同的`assets/derived/processed/data_seed_teacher_v1.jsonl`，32,644行，SHA256 `13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f`；O1 32,480 + O2独立judge满分teacher唯一164各一次；无规则/T/E行 |
| 配置 | `configs/active/seed_teacher_e3_cont_r64_lr2e5_ep1.yaml`，SHA256 `77b215f6d203cc50c0f7e1e0f46276ae696e5af15568f97c140f718b6ec11a39`；从E3加载可训练adapter，r64/alpha64/dropout0.05、lr2e-5 cosine、warmup0.03、effective batch4、cutoff16384、单卡1 epoch、`save_strategy: no`、seed19260820 |
| 训练 | GPU1；2,657 packed examples；665/665 steps；26m13.72s；train loss1.2266275764；1.688 samples/s、0.423 steps/s；正常退出 |
| W&B | [`3f8tas1s`](https://wandb.ai/3120252125-/llmrec-2026/runs/3f8tas1s)；服务端直接查询状态`finished`、global step665、train loss1.2266275764，与本地一致 |
| 产物 | `checkpoints/seed_teacher_e3_cont_r64_lr2e5_ep1/adapter_model.safetensors`，161,533,160 bytes，SHA256 `6b2e4fbd7ee8e04b4704d31fb50e95dc60cf5a04f7537ee746e976d897b68626`；与父E3哈希不同；无optimizer/scheduler/RNG或中间checkpoint |
| 日志 | `logs/train/seed_teacher_e3_cont_r64_lr2e5_ep1.log`，SHA256 `73f67ae4d998e428ace20a85b86e4cbc987c419ff218c8109c0f5bb70043f778`；W&B summary完整。W&B后台在EOF后的异步清理告警不影响服务端`finished`与完整summary |
| 硬门禁 | `logs/precheck/seed_teacher_e3_cont_r64_lr2e5_ep1_20260713.log`，SHA256 `87f714fe...aed8`；itemic断裂0/60=`PASS`；action复读3/30、选择题格式7/8、占位符0/8、简单题6/8均只作diagnostic |
| 门禁摘要 | `logs/probe/seed_teacher_e3_cont_r64_lr2e5_ep1_gate_summary.json`；明确不使用门禁估分或排序checkpoint |
| 上传包 | `submissions/seed_teacher_e3_cont_r64_lr2e5_ep1_platform/`，严格两文件；adapter/config SHA256分别为`6b2e4fbd...68626`/`0d5282cd...2f7b`，与训练输出逐字节一致 |
| 平台表单 | 模型来源=`本地上传`；上传文件=`文件夹`；训练方法=`LoRA`；模型类型=`文本生成`；保存方式=`新建模型`；模型名称=`seed_teacher_e3_cont_r64_lr2e5_ep1`；版本=`V1` |
| 线上 | `seed_teacher_e3_cont_r64_lr2e5_ep1_V1_eval_20260713164018`；平台记录时间2026-07-13 16:40:32；1h10m50s；总分0.9618；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1106/0.0396/0.0672/0.1156/0.1414/0.1053/0.1368`；账号`SL1ACE8AD6710` |
| 线上日志 | `logs/eval/seed_teacher_e3_cont_r64_lr2e5_ep1_20260713.log`，2,716,035 bytes，SHA256 `95130e363ba16d873a74303405ca29fdf869628ed9a9558fa5a95bb3fa0e614b`；8/8任务、Failed tasks 0；evalTaskId `eval-task-kxwokc-1783932031` |
| 协议 | 最早可证实的`platform-stable-v3.1-20260713`日志；action上限1024，itemic 7次race-average。不能与E3旧协议0.9849作差 |
| 预测复盘 | 训练前0.990估计建立在旧协议I-10曲线上；平台协议随后切换，因此不能用0.9618检验该数值预测。跨协议外推作废，后续必须先做sentinel桥接 |
| 判读 | I-11只作为固定协议参考点；它比同协议I-12低0.0150。继续同数据续训没有当前依据，但不能事后声称其相对E3稳定回退 |
| 状态 | **COMPLETE_FIXED_PROTOCOL_0.9618_REFERENCE** |

## 已完成实验：seed_teacher_r64_lr1e4_ep3

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_seed_teacher_v1.jsonl`，32,644行，`D(O1,O2)`；O1全量32,480 + O2双模型流程独立judge满分teacher标签164各一次；规则标签0，无T/E |
| 数据 SHA256 | `13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f` |
| 关键构造 | 保留O1全部target；每个推荐题面组只留一条原CoT，其余12,744条转no-think；602条topic转no-think；164条teacher每轮仅见一次，action target-token占比2.5249% |
| 标签依据 | 旧规则标签相对同源164条独立judge满分teacher参考全量平均F1 0.0429；匹配旧过滤条件的42条平均F1 0.0813且32条零交集，因此1,000条规则行全部删除。teacher不是官方gold；依据是标签直接对照，不是模型probe |
| 配置 | `configs/active/seed_teacher_r64_lr1e4_ep3.yaml`；O6起训，LoRA r64/alpha64/dropout0.05、lr `1e-4`、effective batch4、cutoff16384、3-epoch连续cosine |
| 训练 | GPU1；2,657 packed examples；665 steps/epoch、1,995 total；1h18m31.68s；train loss 1.3583；exit 0 |
| W&B | [`ev401ys9`](https://wandb.ai/3120252125-/llmrec-2026/runs/ev401ys9)，final sync `complete=true`、`exit_code=0` |
| 保存 | E1/E2/E3分别为step665/1330/1995，adapter均161,533,160 bytes且哈希不同；根目录最终adapter与E3一致；无optimizer/scheduler/RNG状态 |
| E2上传包 | `submissions/seed_teacher_r64_lr1e4_e2_platform/`；严格两文件，adapter/config SHA256分别为 `c4902871...267` / `f27c697e...13f`；与step1330源文件逐字节一致 |
| E1/E3上传包 | `submissions/seed_teacher_r64_lr1e4_e1_platform/`、`submissions/seed_teacher_r64_lr1e4_e3_platform/`；均严格两文件并与对应checkpoint逐字节一致；adapter SHA256分别为 `c1bfb4da...add8` / `37678b25...fc2` |
| E2平台表单 | 模型来源=`本地上传`；上传文件=`文件夹`；训练方法=`LoRA`；模型类型=`文本生成`；保存方式=`新建模型`；模型名称=`seed_teacher_r64_lr1e4_e2`；版本=`V1`；描述记录见上传模板 |
| E1/E3平台名 | 均选择`新建模型`、版本`V1`；模型名称分别为`seed_teacher_r64_lr1e4_e1`、`seed_teacher_r64_lr1e4_e3`；描述只将轨迹点改为E1(step665)/E3(step1995) |
| E1线上 | `seed_teacher_r64_lr1e4_e1_V1_eval_20260713114434`；2026-07-13 11:44:41；1h8m8s；总分0.9100；八项=`0.2146/0.0834/0.0327/0.0672/0.1224/0.1456/0.1080/0.1361`；账号`SL1ACE8AD6710` |
| E2线上 | `seed_teacher_r64_lr1e4_e2_V1_eval_20260713101607`；2026-07-13 10:16:13；1h7m28s；总分0.9680；八项=`0.2453/0.1031/0.0367/0.0864/0.1156/0.1372/0.1062/0.1375`；账号`SL1ACE8AD6710` |
| E3线上 | `seed_teacher_r64_lr1e4_e3_V1_eval_20260713114448`；2026-07-13 11:44:53；1h0m55s；总分0.9849；八项=`0.2453/0.1083/0.0391/0.0768/0.1258/0.1414/0.1080/0.1401`；账号`SL1ACE8AD6710` |
| 剂量曲线 | 总分=`0.9100→0.9680→0.9849`；用户两项合计=`0.1161→0.1398→0.1474`；推荐四项合计=`0.4432→0.4454→0.4520`；material=`0.2146→0.2453→0.2453`；world=`0.1361→0.1375→0.1401` |
| 日志 | E1/E2/E3均8/8任务、Failed tasks 0；规范日志分别为`logs/eval/seed_teacher_r64_lr1e4_e1_20260713.log`、`...e2...`、`...e3...`；SHA256=`99e691a9...12d36`/`9e2de684...5d35e`/`c6868c3e...103b6`；evalTaskId=`eval-task-00fvcu-1783914281`/`eval-task-6usmb7-1783908972`/`eval-task-3k8v5e-1783914292` |
| 行为趋势 | action生成耗时=`1363.60s→1248.42s→986.39s`，同时action分=`0.0834→0.1031→0.1083`；本轨迹第三轮同时改善停止效率与语义得分，但该关系不能外推为通用排名器 |
| 门禁结论 | 仓内没有登记过可为I-10 E1/E2/E3排序的独立probe产物，不能事后声称门禁选中E3。本次checkpoint选择依据是完整线上曲线；现有可见题/probe仅保留为格式、循环、截断和结构断裂保险丝，不用于估总分或选epoch |
| 结果判读 | E1明显欠训；E2获得material阶跃和主要用户增益；E3在material不退的前提下继续提高用户、推荐聚合与world。E3是旧协议轨迹主模型与固定协议待桥接父模型；组合收益不能单独归因给164条teacher标签 |
| 状态 | **COMPLETE_E3_PRIMARY_ONLINE_0.9849** |

## r64 同轨迹线上结果与门禁验尸

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_riders_fk_clean.jsonl`，37,262 行，`D/MIXED(O1,O2.General,T)`；只从 0.9177 数据删除 5 条登记的 E 泄漏 |
| 配置 | `configs/active/riders_fk_clean_r64_ep3.yaml`；r64/α64、lr `2e-4`、effective batch 4、3-epoch cosine；E1 不是独立 1-epoch cosine 的干净因果对照 |
| 训练 | 1h29m19s；最终 train loss 1.5962、eval loss 1.5171、eval accuracy 0.6317；曲线约在 step 680 后进入平台期 |
| W&B | [`6gyi8mzc`](https://wandb.ai/3120252125-/llmrec-2026/runs/6gyi8mzc)，状态 finished |
| material sample3 | r32 比较对象 `35/13`；E1 `51/21`；E2 `44/19`；E3 `43/20`（锁定/扇宽）；仅 E1 达预注册 `>=50/17` |
| visible action，同 seed 42 | r32 比较对象 `0/5` JSON、`5/5` 触顶、20,480 tokens；E1 `5/5`、`0/5`、268 tokens；E2 `4/5`、`1/5`；E3 `2/5`、`3/5` |
| 结构保险丝 | E1 itemic 断裂 `0/60`；world 格式 `8/8`，占位符 `0/8`；训练种子 action 复读 `9/30` 仅作 diagnostic |
| 冻结判决 | gate summary 只建议 E1，拒绝 E2/E3；形式化 score-direction 对 E1 输出 `ABSTAIN` |
| E1线上 | 0.8839；material/action/topic/video/prod/ad/live/world=`0.1840/0.0935/0.0421/0.0480/0.1326/0.1414/0.1062/0.1361`；相对 riders −0.0338 |
| E2线上 | 0.9187；八项=`0.1840/0.0981/0.0451/0.0768/0.1258/0.1372/0.1089/0.1428`；相对 riders 名义 +0.0010、相对 E1 +0.0348 |
| action生成时长 | riders `2083.36s`；E1 `864.85s`；E2 `1382.93s`。r64 都显著缩短，但 E2 比 E1 多 `518.08s`，同时 action 分反而高0.0046，说明停止效率与语义得分不能互相替代 |
| 归因 | E2 action 史高并恢复 E1 丢失的 video，但仍损失1道 material；本地 material/action/simple-world 门禁没有预测 E1/E2 顺序，只保留安全诊断用途 |
| 统计结论 | E2 仅是该 riders 轨迹内最高显示分，不是已证实升级；E1/E2同一轨迹只算一个实验族，当前90%方向协议仍 `NOT_CERTIFIED` |
| 平台日志 | 两者均8/8 tasks、Failed tasks 0。E1 `logs/eval/riders_fk_clean_r64_e1_20260712.log`（2,728,715 bytes，SHA256 `4416ed184f94b3b3493406ec3f62b4a7ab2e5ee6290e6a95d9cfa6fc4483d913`，evalTaskId `eval-task-ej7m61-1783833965`）；E2 `logs/eval/riders_fk_clean_r64_e2_20260712.log`（3,002,479 bytes，SHA256 `f14851bb6438acd822c610d45b803e95ba967198b0bd71269c0e1d2c654a1ac5`，`eval-task-kr8jrm-1783834695`） |
| 完整门禁 | `logs/probe/riders_fk_clean_r64_ep3_gate_summary.json` |

## 最新实验：seed_o2_action_r64_lr15e5_ep1

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_seed_o2_action_v1.jsonl`，33,644 行，`D(O1,O2)`；O1 全量 32,480 + O2 teacher 唯一 164 各一次 + O2 规则唯一 1,000；无 T/E、无重复 O2 标签 |
| 数据 SHA256 | `ffb865e6a29d746ea609d041ee0906bda7fb2236712bd09bdee8cbe271f294d8` |
| 配置 | `configs/active/seed_o2_action_r64_lr15e5_ep1.yaml`；LoRA r64/alpha64、dropout0.1、lr `1.5e-4`、effective batch4、cutoff16384、1 epoch、`save_strategy: no` |
| 训练 | GPU1；710 steps/2,840 packed examples；28m01.28s；train loss 1.4779；无中间 checkpoint |
| W&B | [`6qrsdits`](https://wandb.ai/3120252125-/llmrec-2026/runs/6qrsdits)，finished |
| adapter | `checkpoints/seed_o2_action_r64_lr15e5_ep1/adapter_model.safetensors`，161,533,160 bytes；SHA256 `8b6fc2f9fbc2170298e31b83ea8c581880d7d76657c9a35d2afee305bef950d1` |
| 结构门禁 | itemic断裂0/60；world格式6/8、占位符0/8、简单题5/8；action训练样本复读10/30，后三项仅诊断 |
| visible action | seed42 固定5题：JSON `0/5`、4096触顶 `5/5`、20,480 tokens/729.408s；低剂量 O2 完整历史、target 时序纠正和更强 r64 更新的组合未修复停止/重复。该结果不能外推线上 F1=0 |
| material 单题签名 | 锁定/扇宽=`39/13`；略低于 I-07 的41/14，未达到历史8题签名 `>=50/17`；只作分支指标，不是离线得分 |
| 分数后验 | 训练前分析中点0.949；门禁后约0.92、实用区间0.89–0.96；接近0.99为低概率尾部。不是置信区间，也不依赖所谓稳定父锚 |
| 判决 | **LOCAL_REJECT_DO_NOT_UPLOAD**；提交次数稀缺，两项预期优势均未出现。本地门禁曾误排checkpoint，因此该判决只用于提交筛选，不声称已证明线上回退 |
| 完整门禁 | `logs/probe/seed_o2_action_r64_lr15e5_ep1_gate_summary.json` |

## 前一实验：seed_scoremax_r32_ep1

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_seed_scoremax_v1.jsonl`，35,558 行，`D(O1)`；保留 O1 全部 32,480 行及 target，新增 3,078 条 action 保序硬负例历史视图；无 T/E |
| 数据 SHA256 | `7df558a8c08517667f2eab4fc283f2eddfaf7efde16874099a61d63574861cb3` |
| 配置 | `configs/active/seed_scoremax_r32_ep1.yaml`；LoRA r32/alpha32、lr `1e-4`、effective batch4、cutoff16384、1 epoch、`save_strategy: no` |
| 训练 | GPU1；740 steps/2,960 packed examples；29m03.95s；train loss 1.5039；无中间 checkpoint |
| W&B | [`q5uaa2fh`](https://wandb.ai/3120252125-/llmrec-2026/runs/q5uaa2fh)，finished |
| adapter | `checkpoints/seed_scoremax_r32_ep1/adapter_model.safetensors`，80,792,456 bytes；SHA256 `74bb4fed78a72215caae354df4a4a4075d3d36fbde1f5efe2ea93a9cec4d8576` |
| 结构门禁 | itemic断裂0/60；world格式8/8、占位符0/8、简单题4/8；action训练样本复读12/30，后三项仅诊断 |
| visible action | seed42 固定5题：JSON `0/5`、4096触顶 `5/5`、20,480 tokens/737.88s；说明 action 视图未修复停止，但不能外推为线上 F1=0。验尸显示target长度不是主因，裁短history后保持同target造成的选择密度偏移是更直接的风险 |
| material 单题签名 | 锁定/扇宽=`41/14`；高于 riders 历史比较对象35/13，未达到历史8题签名 `>=50/17`；只作分支指标，不是离线得分 |
| 分数后验 | 训练前中点0.976；门禁后中点约0.92、实用区间0.88–0.96；接近0.99为低概率尾部。这是分析预测，不是置信区间 |
| 判决 | **LOCAL_REJECT_DO_NOT_UPLOAD**；提交次数稀缺，预期的 action 修复和 material 8题信号都未出现；adapter保留作一次可审计实验，临时 merged 删除 |
| 完整门禁 | `logs/probe/seed_scoremax_r32_ep1_gate_summary.json` |

## 已完成实验：i01_action_distill_r64_ep3

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_i01_action_distill_v1.jsonl`，33,792 行；O1 32,480 行 + 164个唯一、独立judge满分action teacher标签各重复8次形成1,312有效行；I-01转换12,744条冗余推荐CoT |
| 数据 SHA256 | `bbefa5f24d4c9a8e0c7573873fdc2947b35880955cb60b5debc0f619d6ce99d3` |
| 蒸馏用量 | 11,432,127 API token（prompt 8,353,854 + completion/hidden reasoning 3,078,273）；Yunwu/DeepSeek余额耗尽后以164条封板 |
| 配置 | `configs/active/i01_action_distill_r64_ep3.yaml`；LoRA r64/alpha64、lr `5e-5`、effective batch4、3 epoch、`save_strategy: no` |
| 训练 | GPU1；1,047 steps；1h28m19s；train loss 1.4914；最终 eval loss 1.4088、eval accuracy 0.6490 |
| W&B | [`thbcz5k3`](https://wandb.ai/3120252125-/llmrec-2026/runs/thbcz5k3)，finished |
| adapter | `checkpoints/i01_action_distill_r64_ep3/adapter_model.safetensors`，161,533,160 bytes；SHA256 `67273f14373b4f7ee14c6077cba3ebf0b6f75336abf0491137901d1241c8a875` |
| 结构保险丝 | itemic断裂0/60，硬判PASS；action复读6/30；world格式4/8、占位符4/8，后两项仅诊断 |
| action同口径 | v4、n=325：候选/riders比较对象 F1=`0.0160/0.0171`，JSON=`0.6%/0.3%`，截断=`22.2%/44.3%`；停止效率改善但语义未涨、重复严重度未降 |
| world同口径 | v4、n=500：候选/riders比较对象 Acc=`0.206/0.380`，格式存活=`39.6%/100%`；方向性大幅回退 |
| 判决 | **LOCAL_REJECT_DO_NOT_UPLOAD**；不续训、不上传；adapter和日志保留，临时 merged 评测副本删除 |
| 完整门禁 | `logs/offline_eval/i01_action_distill_r64_ep3_gate_summary.json` |

## 最新线上失败实验：seed_cotfix_v1_lora_ep1

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_seed_cotfix_v1.jsonl`，32,480 行 |
| 数据 SHA256 | `6f6fe198c875cab6a71ece2d9524923fbb97ab23adf622e33ea2ed169a33f667` |
| 改动 | 官方种子中 425 个唯一 CoT 后缀、1,495 行补全；行数、顺序、prompt 和最终答案不变 |
| 配置 | `configs/history/seed_cotfix_v1_lora_ep1.yaml` |
| 训练 | 单卡 LoRA，lr `1e-4`，effective batch 2，1 epoch，29m01s，train loss 1.5313 |
| W&B | `https://wandb.ai/3120252125-/llmrec-2026/runs/5v9lpyqb` |
| 门禁 | itemic 断裂 0/60；选择题格式 0/8；占位符复读 8/8；world acc 0.128 |
| 线上 | 总分 0.8674；物料 0.2146；用户 0.0683/0.0452；推荐按官方序 video/prod/ad/live 为 0.0768/0.1224/0.1358/0.1107；world 0.0937 |
| 平台记录 | `seed_cotfix_v1_lora_ep1_V1_eval_20260711211350`；2026-07-11 21:13:55；1h23m43s；账号 ID `SL1ACE8AD6710`；内部 `evalTaskId=eval-task-d9xyqv-1783775634` |
| 线上日志 | `logs/eval/seed_cotfix_v1_lora_ep1_20260711.log`，3,287,872 bytes，SHA256 `38155b5b930632f37429ea0ebcc254cd00ab9d78805a039394c6911da406b70a`；原始下载名 `G651fvb5...SJTyW.log` |
| 日志诊断 | 8/8 任务完成、Failed tasks 0；可见 world 3/5 原样复读占位符；action 4/5 输出约 30–33KB、682 个 SID 且 JSON 未闭合，action 生成 2,236.50s（37m16s） |
| 对最好分项 | 相对 riders：物料/视频不变，action +0.0028、topic +0.0025、prod −0.0034、ad −0.0028、live +0.0009、world −0.0502；前 7 项显示值合计均为 0.7738。riders 的数据和 lr 不同，此对账不是 CoT 修补的干净因果对照 |
| 判决 | 已线上证伪；不复测、不续训、不作为 warm start；checkpoint 已删除，提交 adapter 暂存 |

## 历史单次参考：riders_fk_lora_ep1

| 项 | 记录 |
|---|---|
| 数据 | `assets/derived/processed/data_riders_fk.jsonl`，37,267 行 |
| 数据 SHA256 | `e4f91c5246e4c7e8cb9fe88fe19add7af2c9b0678d6688871a2c7a6be56f8d7e` |
| 配置 | `configs/retained/riders_fk_lora_ep1.yaml` |
| 训练 | LoRA r32/alpha32，lr `2e-4`，1 epoch |
| 线上 | 单次总分 0.9177；未做同 checkpoint 重复评测，不能据此证明稳定；详细分项见 `experiment_log.md` |

## 保留策略

1. 新训练的epoch数与保存策略由数据规模和训练轨迹决定；单点实验默认`save_strategy: "no"`，连续多轮剂量比较可按epoch保存adapter-only。
2. 中间checkpoint只有在承担训练时点选择时才保留，并须逐个登记；门禁失败的最终adapter仅可为审计保留，必须明确禁止上传和warm start。
3. merged model 只在门禁时临时生成，结束即删；提交平台若支持 adapter，优先 adapter。
4. 提交包、评测日志和 adapter 哈希必须在本表登记，禁止依赖目录修改时间猜“最新模型”。
5. 配置引用已删除 checkpoint 时只能作为历史记录，不得直接启动。

## 清理记录

- 2026-07-11：删除 49 个多余顶层 checkpoint、51 个中间 checkpoint、35 份 optimizer、全部 merged 工作副本；从约 176GB 清至只保留 riders 最终 adapter。
- 2026-07-11：删除项目根重复 `adapters/riders_fk_lora_ep1`，其内容与保留 checkpoint SHA256 相同。
- 2026-07-11：将 31GB `submissions/` 实体移入运行卷 `artifacts/submissions/`，项目根改为链接。
- 2026-07-11：删除其余 24 个低分、失败或重复提交包，只保留当前线上最好和最新用户交付包；提交区从约 31GB 降至约 1.6GB。
