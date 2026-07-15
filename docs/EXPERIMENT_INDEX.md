# Experiment And Artifact Index

> 当前状态基线：2026-07-14 UTC。
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
- I-18已完成训练与本地结构硬门禁：官方源派生`data_seed_teacher_cotfix_v2`从I-10父数据逐行构建，只修复538条保留推荐CoT；538/538通过程序门与独立judge score=5九项全真。32,644行/SHA256 `634c4805...c0e4`，prompt/答案/行序/任务数/O2 teacher均0变化，target元数据/T/E为0，16k超限0。2026-07-14 17:26 UTC从O6在GPU1持久启动，1,995/1,995正常完成，train loss1.3617671，W&B [`32av8e8z`](https://wandb.ai/3120252125-/llmrec-2026/runs/32av8e8z)服务端`finished`；E3/root adapter SHA256 `07cd6628...2a9e3`且逐字节一致。E3临时merge后itemic断裂0/60=`PASS`，其余诊断不用于估分；严格两文件包已验收，尚未上传、未消耗线上配额。
- 撤回旧 I-09 规则数据资格：规则标签相对同源独立judge满分teacher参考的平均F1仅0.0429；匹配实际过滤条件的42条平均F1 0.0813且32条零交集。该teacher参考不是官方gold；`seed_o2_action_r64_lr1e4_ep3`因此在step16中止，W&B `sh96a1sq`，`checkpoints/seed_o2_action_r64_lr1e4_ep3/`无adapter且禁止resume。
- 当前最高单次显示分和固定协议最高均为I-13 `0.9978`。I-10 E3旧协议0.9849仍只能作旧轨迹父模型记录；E3固定协议桥缺失不妨碍I-13在现有固定协议候选中确定为主模型，但仍禁止计算I-13相对E3的净增益。
- r64 同一训练轨迹 E1/E2 已线上评测：E1=0.8839，E2=0.9187；E3未评测且不再建议上传。本地门禁原先只选 E1、拒绝 E2，线上排序相反，门禁不再承担正向 checkpoint 排名。
- `riders_fk_clean_r64_ep3` 训练事实不变：GPU1 单卡，r64/α64、3 epoch、353 steps/epoch、总 1,059 steps；W&B online run [`6gyi8mzc`](https://wandb.ai/3120252125-/llmrec-2026/runs/6gyi8mzc)。E2 action 0.0981 创本账号新高，但 material E1/E2 均为6题。
- `i01_action_distill_r64_ep3` 已完成：3 epoch/1,047 steps，action 截断相对预登记比较对象 riders 减半但 F1 未涨，world v4 大幅回退；状态为本地否决、不上传。蒸馏正式累计 11,432,127 API token。
- `seed_scoremax_r32_ep1` 已完成：只用 `D(O1)`，35,558 行，单卡 1 epoch/740 steps。硬结构保险丝通过，但可见 action 0/5 闭合、5/5 触顶；material 签名 41/14 未进历史 8 题档。后验中点约 0.92，状态为本地否决、不上传。
- `seed_o2_action_r64_lr15e5_ep1` 已完成：`D(O1,O2)` 33,644 行，O2 唯一 action 行 1,164（3.4598%），r64/alpha64、lr1.5e-4、单卡 1 epoch/710 steps。itemic 结构通过，但 action 0/5 闭合、material 39/13，状态为本地否决、不上传。
- E2 的本地 checkpoint 存在，但 `submissions/riders_fk_clean_r64_e2_platform/` 不存在；平台日志只记录临时 `/tmp/eval_model/merged`。在缺上传 manifest 时，不能声称平台工件哈希已由本地 adapter 哈希证明。
- 旧实验的中间 checkpoint、optimizer、失败 checkpoint 和 merged 工作副本已于 2026-07-11 删除；本轮用户批准的 r64 E1/E2/E3 例外已单独列入下表。

## 已完成训练并通过结构门：seed_teacher_cotfix_v2_r64_lr1e4_ep3

| 项 | 冻结与完成记录 |
|---|---|
| 目的 | 单变量检验“忠实补全上游截断推荐CoT”能否改善推荐四域；不采用参数融合，不把选手约1.03转述当收益证明 |
| 基座/隔离 | O6 `OneReason-0.8B`干净启动；不加载、resume、merge或warm-start任何adapter/checkpoint，旧`cotfix_v1`失败产物输入为0 |
| 正式数据 | `assets/derived/processed/data_seed_teacher_cotfix_v2.jsonl`，32,644行，`D(O1,O2.UserProfile,O2.Pid2Sid,O2.Pid2Caption,O2.Pid2Tag,O3)`，SHA256 `634c4805367308b35dd729c17f59a1a8b4bb473b84a80d21cc71931a2c29c0e4` |
| 上游与混合 | 父I-10数据SHA256 `13c40526...eee4f`；O1父行32,480（99.497611%）+ O2唯一teacher 164（0.502389%），O2规则/T/E为0；O3 SHA256 `c307fe6d...b8d9d`只作历史侧证据，目标答案/目标元数据0 |
| 构造与质检 | `scripts/data/build_cotfix_v2.py` SHA256 `81aba7b962a4ad1c27d450d882a6774ab1771daae0fe7957e1f790a81227c9d0`；538/538为`TRUNCATED`且程序门0错误，独立judge全部score=5九项全真；新增非历史SID/重复前缀SID均0 |
| 单变量不变量 | 恰改538条保留推荐CoT；其余32,106行逐字不变。instruction/input/history、最终答案、行序、任务数和164条O2 teacher均0变化；target token 5,867,041，仅推荐四域增加；raw最大约9,744 token、16,384 cutoff超限0 |
| 审计 | `logs/data/seed_teacher_cotfix_v2_audit.json` SHA256 `bdea6db13a398dbcde973117dbf72d9ed81d5635bcbcbb7f1f1ba41fda79057c`；generation/judge审计SHA256 `05f5399b...5edf` / `994c8702...c861` |
| 配置 | `configs/active/seed_teacher_cotfix_v2_r64_lr1e4_ep3.yaml`启动时SHA256 `2bd234bd3d58a553c3f9b304e718e73531bbdb2a192d195866ba09350d3042da`；与I-10关键训练字段完全相同，仅dataset/dataset_dir及非训练的output/run name变化。完成后添加`HISTORICAL_ONLY_AFTER_SUCCESS`防覆盖头，当前SHA256 `85b62d80e9654c7e8e6b1e910f6140ee85c63f4aeb55f9494053e3fd444791da` |
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
| E3平台包 | `submissions/seed_teacher_cotfix_v2_r64_lr1e4_ep3_platform/`严格只有`adapter_config.json`（1,138 bytes，SHA256 `65732b4da48a0b2f93ea6d1bb3861e7e13bf0406635ae5e587de34a709ff26fa`）与`adapter_model.safetensors`（161,533,160 bytes，SHA256 `07cd662852ee1ef3654096adfce36891ce260129bdc68c7924c2b75554c2a9e3`）；两文件均与E3逐字节一致，尚未上传 |
| 当前状态 | **PACKAGE_READY_LOCAL_GATE_PASS_E3_PENDING_UPLOAD**；E1/E2不选，只提交E3；尚未线上评测，本轮没有消耗提交次数 |

## 保留模型

| 角色 | 实验 | 路径 | 哈希 | 状态 |
|---|---|---|---|---|
| 官方基座 | OneReason-0.8B | `models/OneReason-0.8B-pretrain-competition/` | `config.json` SHA256 `5fe26642...` | 只读 |
| I-18 E1剂量轨迹 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 | `checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-665/` | adapter/config SHA256 `02b404bd...47a85` / `65732b4d...26fa` | adapter-only；只作1 epoch轨迹，不打包、不上传、不作父模型 |
| I-18 E2剂量轨迹 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 | `checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1330/` | adapter/config SHA256 `14071ab8...bdc38` / `65732b4d...26fa` | adapter-only；只作2 epoch轨迹，不打包、不上传、不作父模型 |
| I-18本地主候选 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 E3 | `checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1995/`；根目录`checkpoints/seed_teacher_cotfix_v2_r64_lr1e4_ep3/`逐字节同adapter | adapter/config SHA256 `07cd6628...2a9e3` / `65732b4d...26fa` | 3 epoch；W&B finished、结构门0/60 PASS；已制严格两文件包，未上传/线上评测，用户决定前不作新父模型 |
| I-18待上传E3 LoRA包 | seed_teacher_cotfix_v2_r64_lr1e4_ep3 | `submissions/seed_teacher_cotfix_v2_r64_lr1e4_ep3_platform/` | adapter/config SHA256 `07cd6628...2a9e3` / `65732b4d...26fa` | 严格两文件且与E3逐字节一致；保存方式新建模型、版本V1；待用户本地上传 |
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
