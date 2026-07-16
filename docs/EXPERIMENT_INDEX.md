# Experiment And Artifact Index

> 当前状态基线：2026-07-16 UTC。
> 旧版完整历史表已归档到 `docs/archive/EXPERIMENT_INDEX_pre_cleanup_20260711.md`。

本文件只登记当前仍存在、仍可使用的模型产物。历史分数和实验归因见 `experiment_log.md`。

## 当前结论

- I-14首次运行因启动器绑定临时PTY，在step 1,886/1,971被会话生命周期中断；`rerun1`随后按相同数据与超参从O6和全新输出目录干净完成。E3于2026-07-14 15:20平台评测为0.9518，八项=`0.2453/0.1045/0.0387/0.0480/0.1292/0.1414/0.1080/0.1368`。它没有替换I-13的当前榜分，但I-13属于E3 r64+用户残差r16参数拼接的融合灰区路线，只能作业务榜分对照，不能作I-14纯O1单体r80的科学基线；仓内没有同协议、同血统的直接对照，E1/E2亦未线上评测。
- 2026-07-13下午平台修复评测不稳定问题；仓内可证实的协议切点位于I-10 E3（11:45）与I-11（16:40）之间。旧协议指纹为action `max_tokens=4096`、itemic单次beam64；固定协议指纹为action `max_tokens=1024`、itemic 7次`Race averaged evaluation`。日志两边都打印`version: v3.1`，故必须靠指纹分为`platform-pre-fix-v3.1`与`platform-stable-v3.1-20260713`，禁止跨协议作差。
- I-10完整线上轨迹已完成：使用 `data_seed_teacher_v1` 32,644行（O1全量99.4976% + 164条独立judge满分teacher标签0.5024%，规则标签0）从O6训练r64连续3-epoch cosine；E1/E2/E3=`0.9100/0.9680/0.9849`。该曲线只在旧协议内部有效；E3是固定协议待重评的桥接父模型。
- I-11是最早可证实的固定协议日志，单次线上0.9618；它不能与E3旧协议0.9849直接比较。继续同数据续训仍因缺少固定协议父分而不启动，但旧版“相对E3 -0.0231”结论撤销。
- I-12固定协议单次线上0.9768，八项为`0.2453/0.1206/0.0393/0.0672/0.1292/0.1316/0.1053/0.1383`。同协议相对I-11总分+0.0150、用户合计+0.0097、推荐合计+0.0038、world+0.0015；ad单项-0.0098。I-12现为I-13的同协议直接对照。
- I-13保持E3 r64不变，仅将I-12 r16用户残差缩放到0.875；固定协议线上0.9978，八项为`0.2453/0.1183/0.0390/0.0960/0.1224/0.1316/0.1062/0.1390`。同协议相对I-12总分+0.0210、用户合计-0.0026、推荐合计+0.0229、world+0.0007；当前固定协议主模型。
- I-14 E3固定协议单次线上0.9518。相对I-13逐项为`0/-0.0138/-0.0003/-0.0480/+0.0068/+0.0098/+0.0018/-0.0022`、面板总分差-0.0460，但这只回答“是否替换融合灰区主模型”的榜分问题，不支持纯O1单体路线的因果否决。更接近的非融合固定协议参考I-11为0.9618，I-14名义低0.0100；I-11仍使用164条teacher、从I-10 E3续训且rank不同，也不是干净基线。原始日志已复核为action1024+itemic 7次race-average、8/8完成、失败0，evalTaskId `eval-task-lfrrhq-1784013605`。
- I-16于2026-07-14 12:19 UTC通过持久启动器在单卡启动，W&B [`packufor`](https://wandb.ai/3120252125-/llmrec-2026/runs/packufor)，600/600正常完成。policy从保留的I-10 E3继续同一个r64 adapter，reference为O6显式加载并合并同一E3 adapter；不新建adapter、不拼接参数、不做同容量蒸馏。step200/400/600使四推荐域聚合raw chosen胜率从32.03%升到52.73%/57.03%/58.59%，action始终保持93.75%，但推荐gold平均token logp分别下降0.01185/0.02030/0.02149，全部超过0.01保护线。I-16按原门槛本地否决，不上传；这些读数只作机制和剂量证据，不估线上分数。
- I-17已在I-16 step400结果后、正式启动前注册，并于2026-07-14 12:57 UTC在I-16正常退出后顺序持久启动；W&B [`pfjlvm70`](https://wandb.ai/3120252125-/llmrec-2026/runs/pfjlvm70)服务端`finished`。它从原始I-10 E3重新开始，数据、beta和冻结E3 reference不变，只将峰值lr降至7e-7并用30步warmup后的constant日程把step200累计LR面积降为I-16 step200的74.8434%。step100/150/200全部满足量化保护线；按预注册“最早全通过”规则选step100，其推荐聚合raw chosen胜率32.03%→43.36%、gold平均token logp仅下降0.00327、action保持93.75%，itemic断裂0/60。step100固定协议线上0.9727，八项=`0.2453/0.1077/0.0380/0.0960/0.1156/0.1274/0.1044/0.1383`，低I-13 0.0251；相对I-12推荐合计+0.0101但用户合计-0.0142，总分-0.0041。直接父模型I-10 E3固定协议分仍缺失，因此不作DPO因果结论，桥接前不提交step150/200。
- I-18截断CoT修复E3固定协议线上0.9697，八项=`0.2453/0.1083/0.0382/0.0768/0.1190/0.1316/0.1089/0.1416`，低I-13 0.0281、低I-17 0.0030，未替换主模型。日志8/8完成、失败0；I-10 E3缺同协议桥，不能将该差值作CoT修复的净因果结论。
- I-19从固定协议最高分I-13/0.875原地更新同一个r80 adapter，不新增adapter层；D(O1)同题同域hard-negative子集2,688对按ad/prod/living/video=`768/768/768/384`重平衡。单卡W&B [`0bm73wt9`](https://wandb.ai/3120252125-/llmrec-2026/runs/0bm73wt9)完成75/75、退出码0；按冻结门槛选出的step25本地ad/prod/living/video raw mean margin均改善且itemic断裂0/60，但固定协议线上仅0.9763，八项=`0.2453/0.1181/0.0402/0.0864/0.1156/0.1246/0.1071/0.1390`。相对I-13总分-0.0215，其中用户合计+0.0010而推荐合计-0.0225；该hard-negative偏好门与线上推荐指标方向失配，I-19分支封板，step50/75不再提交。I-13 0.9978继续为主模型。
- I-20从保留的I-13/0.875原地更新同一r80，未加载I-19失败点。D(O1,O2.General) 12,260行由prod/ad正例与冻结I-13保持严格1:1组成；200/200步正常完成，W&B `1i153nai`。十档统一圈外诊断中step100是唯一做到prod/ad三SID宽候选`+3/128`且video/live `0/128`的点，但gold mean-logp仍为负漂移，故只视作线上实验候选。双通路行为与父模型近乎重合、itemic断裂0/60；严格两文件包已生成，I-13 0.9978在出分前仍是主模型。
- I-21在I-13同一r80内做topic answer-token低剂量CE，其余行用冻结I-13 KL保持；单卡W&B [`wjjymcj9`](https://wandb.ai/3120252125-/llmrec-2026/runs/wjjymcj9)完成150/150。六点统一诊断选step150：topic/action gold sum-logp相对父模型`+0.09127/+0.03478`，prod/ad Top-64覆盖`+3/128`，video/live为`-2/128`；这些只用于选点，不是线上分数预测。结构门itemic断裂0/60、action复读1/30；严格双文件包已生成，等待一次线上实验。
- I-22在I-13同一r80内完成world答案token低剂量CE，单卡W&B [`cohd8617`](https://wandb.ai/3120252125-/llmrec-2026/runs/cohd8617)完成150/150。46条未训练D(O2.General)选择集上，step25虽gold logp`+0.06628`且KL`0.00870`，但top-1掉3/46；step125的top-1不降且gold logp`+0.24041`，但KL`0.03381`超过预注册0.02。六点无一满足全部主门，按原规则本地否决，不跑后续保持门、不打包、不上传。
- I-23在正式训练前冻结允许角色I-10 E3，以batch=1对538组全部1,836个去重gold只评分最终答案token，最终选中83组（video/prod/ad/live=`32/35/11/5`）；随后从O6按I-10/I-18同一r64三轮物理干净训练。E3固定协议线上0.9915，八项=`0.2760/0.1099/0.0383/0.0576/0.1258/0.1400/0.1053/0.1387`。相对I-13总分-0.0063，其中material +0.0307、用户合计-0.0091、推荐合计-0.0275、world-0.0003；没有替换I-13，但成为固定协议最高无参数拼接单adapter。用户已批准只将成功E3作为新action-answer-token CE + 冻结I-23 KL保持实验父模型；E1/E2仍禁止作父模型。
- `i23_userres_r80_s625`线上为0.9866，未超过I-13；但相对同日I-23复测0.9884，material掉档`-0.0307`而其余七项合计`+0.0289`，说明用户残差确实补回用户并改善video/prod，只是0.625越过material临界点。下一发按冻结分支使用`s500`；若material保持，基于本次七项响应的条件中心约1.0115，仍掉档则约0.9808。原始日志未入仓前只作用户面板结果，不登记伪造evalTaskId。
- I-24已从成功I-23 E3原地完成200步action-only低剂量训练，W&B `f3ayytob`服务端finished，8个adapter-only剂量点齐全；但8/8均未通过启动前冻结的action硬门。step50是唯一action sum-logp均值为正的点（`+0.05031`），仍同时失败改善率`0.46875<0.55`、top-1 delta `-0.00080`和topic delta `-0.01223<-0.01`；所有点四域最大KL约`0.013–0.014>0.005`。整条分支按原规则本地关闭，不打包、不上传、不作父模型；条件算术点0.9999已被否决证据取代。
- I-25已在任何正式训练前冻结：只从成功I-23有效模型新建隔离r16 action residual，不使用I-12残差或I-24失败点；复用已登记6106行数据，action1752行答案体CE+弱parent KL，其余4354行只做parent KL，完整一轮1527步。最终gate SHA256 `53b5b375...f212`先单轴冻结最早action通过checkpoint，再在该点按固定scale升序取最小全保持解；不得二维后验回选。配置/trainer/启动器静态验收通过，尚未因准备文件而宣称训练或涨分。
- 撤回旧 I-09 规则数据资格：规则标签相对同源独立judge满分teacher参考的平均F1仅0.0429；匹配实际过滤条件的42条平均F1 0.0813且32条零交集。该teacher参考不是官方gold；`seed_o2_action_r64_lr1e4_ep3`因此在step16中止，W&B `sh96a1sq`，`checkpoints/seed_o2_action_r64_lr1e4_ep3/`无adapter且禁止resume。
- 当前最高单次显示分和固定协议最高仍为I-13 `0.9978`；固定协议最高无参数拼接单adapter为I-23 `0.9915`，最严格纯O1单体仍为I-14 `0.9518`。I-10 E3旧协议0.9849仍只能作旧轨迹父模型记录；E3固定协议桥缺失不妨碍I-13在现有固定协议候选中确定为主模型，但仍禁止计算I-13相对E3的净增益。
- r64 同一训练轨迹 E1/E2 已线上评测：E1=0.8839，E2=0.9187；E3未评测且不再建议上传。本地门禁原先只选 E1、拒绝 E2，线上排序相反，门禁不再承担正向 checkpoint 排名。
- `riders_fk_clean_r64_ep3` 训练事实不变：GPU1 单卡，r64/α64、3 epoch、353 steps/epoch、总 1,059 steps；W&B online run [`6gyi8mzc`](https://wandb.ai/3120252125-/llmrec-2026/runs/6gyi8mzc)。E2 action 0.0981 创本账号新高，但 material E1/E2 均为6题。
- `i01_action_distill_r64_ep3` 已完成：3 epoch/1,047 steps，action 截断相对预登记比较对象 riders 减半但 F1 未涨，world v4 大幅回退；状态为本地否决、不上传。蒸馏正式累计 11,432,127 API token。
- `seed_scoremax_r32_ep1` 已完成：只用 `D(O1)`，35,558 行，单卡 1 epoch/740 steps。硬结构保险丝通过，但可见 action 0/5 闭合、5/5 触顶；material 签名 41/14 未进历史 8 题档。后验中点约 0.92，状态为本地否决、不上传。
- `seed_o2_action_r64_lr15e5_ep1` 已完成：`D(O1,O2)` 33,644 行，O2 唯一 action 行 1,164（3.4598%），r64/alpha64、lr1.5e-4、单卡 1 epoch/710 steps。itemic 结构通过，但 action 0/5 闭合、material 39/13，状态为本地否决、不上传。
- E2 的本地 checkpoint 存在，但 `submissions/riders_fk_clean_r64_e2_platform/` 不存在；平台日志只记录临时 `/tmp/eval_model/merged`。在缺上传 manifest 时，不能声称平台工件哈希已由本地 adapter 哈希证明。
- 旧实验的中间 checkpoint、optimizer、失败 checkpoint 和 merged 工作副本已于 2026-07-11 删除；本轮用户批准的 r64 E1/E2/E3 例外已单独列入下表。

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
| 线上边界 | 只完整补回action 0.0084且其他七项完全不动的条件算术上限为0.9999；恢复75%仅追平I-13 0.9978。任何离线门都不保证material第9题。最终I-23 r64+r16 residual拼成r80仍属参数拼接/融合审核灰区，不得称直接单adapter训练 |
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
| 目的/父模型 | 直接优化固定协议最高分I-13/0.875；policy与冻结reference均为adapter SHA256 `71bc3c2c...ffd5b`。在原r80内更新，不叠加新adapter；I-13融合灰区属性不因此消失 |
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
| 状态 | **COMPLETE_ONLINE_0.9763_REJECT_BRANCH**；不提交step50/75，不作后续训练父模型，I-13 0.9978继续为主模型 |

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
| 当前状态 | **COMPLETE_LOCAL_PROVISIONAL_STEP100_PACKAGED_AWAITING_ONLINE**；离线证据混合，只值得占一次实验配额，不宣称稳涨；I-13 0.9978在出分前仍为主模型 |

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
| 状态 | **COMPLETE_LOCAL_PROVISIONAL_STEP150_PACKAGED_AWAITING_ONLINE**；只作为topic正交实验，不替换I-13 0.9978主模型 |

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
| 当前状态 | **COMPLETE_ONLINE_0.9915_HIGHEST_NON_SPLICE_ADAPTER**；未替换I-13 0.9978；只允许成功E3作为新action-answer-token CE + 冻结I-23 KL保持实验父模型，E1/E2继续只作剂量轨迹 |

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
| I-23已评测E3/action-retKL获准父模型 | seed_teacher_cotfix_v3_r64_lr1e4_ep3 E3 | `checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/checkpoint-1995/`；根目录`checkpoints/seed_teacher_cotfix_v3_r64_lr1e4_ep3/`逐字节同adapter | adapter/config SHA256 `0e5fa9bb...c6b8` / `b3f2a1b5...e7e7` | 固定协议线上0.9915；最高无参数拼接单adapter；只允许作新action-answer-token CE + 冻结I-23 KL保持父模型 |
| I-23已评测E3 LoRA包 | seed_teacher_cotfix_v3_r64_lr1e4_ep3 | `submissions/seed_teacher_cotfix_v3_r64_lr1e4_ep3_platform/` | adapter/config SHA256 `0e5fa9bb...c6b8` / `b3f2a1b5...e7e7` | 严格两文件且与E3逐字节一致；固定协议线上0.9915 |
| I-23 + 0.625用户残差已评线上探针 | i23_userres_r80_s625 | `checkpoints/i23_userres_r80_s625/` | adapter/config SHA256 `4a46fd29...0c70` / `4768770a...4d06`；组合审计`logs/model/i23_userres_r80_s625_combine.json` SHA256 `e3c630b9...3d32` | 固定协议线上0.9866，八项=`0.2453/0.1170/0.0399/0.0768/0.1326/0.1316/0.1044/0.1390`；相对同日I-23复测非material `+0.0289`、material `-0.0307`。原始日志/evalTaskId待入仓 |
| I-23 + 0.625用户残差已评包 | i23_userres_r80_s625 | `submissions/i23_userres_r80_s625_platform/` | adapter/config SHA256 `4a46fd29...0c70` / `4768770a...4d06`，201,903,440 / 1,074 bytes | 严格两文件包与checkpoint逐字节一致；已线上0.9866，未替换I-13。初赛参数拼接审核灰区不变 |
| I-23 + 0.5用户残差下一探针 | i23_userres_r80_s500 | `checkpoints/i23_userres_r80_s500/` | adapter/config SHA256 `d2b77c74...99bf` / `4768770a...4d06`；组合审计SHA256 `7b04c918...3be5` | CPU精确拼接`delta=I23+0.5×residual`；s625已满足“material掉档但七项显著正向”的预设触发条件，状态`READY_FOR_MANUAL_PLATFORM_SUBMISSION` |
| I-23 + 0.5用户残差下一探针包 | i23_userres_r80_s500 | `submissions/i23_userres_r80_s500_platform/` | adapter/config SHA256 `d2b77c74...99bf` / `4768770a...4d06`，201,903,440 / 1,074 bytes | 严格两文件且与checkpoint逐字节一致，尚未上传；material保持/掉档条件中心约1.0115/0.9808，均非校准预测 |
| I-23 + 用户残差加码备用 | i23_userres_r80_s750 | `checkpoints/i23_userres_r80_s750/` | adapter/config SHA256 `363a3b59...f365` / `4768770a...4d06`；组合审计SHA256 `5d890803...12a6` | CPU精确拼接`delta=I23+0.75×residual`；只有s625线上material保持且用户项确有正向时才允许进入评估/打包，当前未评测、未打包、未上传 |
| I-23 + 用户残差加码备用 | i23_userres_r80_s875 | `checkpoints/i23_userres_r80_s875/` | adapter/config SHA256 `b24c17cc...f31f` / `4768770a...4d06`；组合审计SHA256 `ae4782ab...a8ad` | CPU精确拼接`delta=I23+0.875×residual`；只有前一在线剂量证明material保持且仍需补用户缺口时才允许进入评估/打包，当前未评测、未打包、未上传 |
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
| I-13 固定协议主模型 | e3_userres_r80_retkl_v3_s875 | `submissions/e3_userres_r80_retkl_v3_s875_platform/` | adapter/config SHA256 `71bc3c2c86beb1c1aaafd41f98915ba94a7f964b6e8450079a883aebc32ffd5b` / `e3c3ace0c049f84726b257e3bff66e1954e316c249f9f2f7d931a80944dc4ac0` | 固定协议线上0.9978；当前主模型；复赛融合口径待官方确认 |
| I-13 scale0.90参数探针 | e3_userres_r80_retkl_v3_s900 | `submissions/e3_userres_r80_retkl_v3_s900_platform/` | adapter/config SHA256 `7c966fb2...a60a` / `e3c3ace0...c4ac0` | 精确参数拼接、严格两文件、结构门0/60 PASS；本地保持KL和action CE未支配0.875，因此未作为I-19父模型；仅作零训练备选，不据此声称涨分 |
| I-13 scale0.80线上探针 | e3_userres_r80_retkl_v3_s800 | `submissions/e3_userres_r80_retkl_v3_s800_platform/` | adapter/config SHA256 `bb86eb8a...63c6` / `e3c3ace0...c4ac0` | 仅改变I-12残差系数，严格两文件；I-13与I-12两点之间的高优先级零训练线上探针，不宣称稳涨 |
| I-13 scale0.75线上探针 | e3_userres_r80_retkl_v3_s750 | `submissions/e3_userres_r80_retkl_v3_s750_platform/` | adapter/config SHA256 `5aa80992...6233` / `e3c3ace0...c4ac0` | 精确参数拼接；itemic 0/60 PASS但action复读4/30，高风险高收益第二探针 |
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

`submissions/` 当前保留十一份：原十份已登记历史包，以及已上传并评测为0.9727的I-17 step100严格两文件包。历史riders r64 E2的本地标准提交包仍缺失；历史提交分数、配置和日志保存在实验台账与归档总账中。

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

## 当前固定协议主模型：e3_userres_r80_retkl_v3_s875

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
| 规则口径 | FAQ写明初赛基于OneReason-0.8B、允许蒸馏、全程不鼓励融合，并要求复赛结束提供单模型训练方案审核复现。该包运行时是单个r80 adapter，但参数由两个同基座LoRA拼接，存在融合认定灰区；没有官方书面确认，不把“初赛通常不审核”写成合规证明 |
| 线上 | `e3_userres_r80_retkl_v3_s875_V1_eval_20260714004418`；平台记录时间2026-07-14 00:44:35；1h7m21s；总分0.9978；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1183/0.0390/0.0960/0.1224/0.1316/0.1062/0.1390`；账号`SL1ACE8AD6710` |
| 线上日志 | `logs/eval/e3_userres_r80_retkl_v3_s875_20260714.log`，2,777,778 bytes，SHA256 `9291f8bf87871bb93846dda4cfcf60d43812354fb87a18e6ef6a5a349bdb3315`；8/8任务、Failed tasks 0；evalTaskId `eval-task-9ie86v-1783961075` |
| 协议 | `platform-stable-v3.1-20260713`；action上限1024，itemic 7次race-average。与E3旧协议结果不可作差 |
| 同协议相对I-12 | 总分+0.0210；material 0、action -0.0023、topic -0.0003、video +0.0288、prod -0.0068、ad 0、live +0.0009、world +0.0007。用户两项合计-0.0026，推荐四项合计+0.0229 |
| 判读 | 缩放残差的总分方向得到一次线上支持，主要收益来自video而非用户两项。I-13是当前固定协议主模型；E3桥接仍缺失，不能声称相对父E3的净增益 |
| 状态 | **COMPLETE_FIXED_PROTOCOL_0.9978_CURRENT_MAIN** |

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
