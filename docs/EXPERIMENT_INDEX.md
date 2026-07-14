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
- I-14 E3按平台修复后的时间切点归入固定协议，单次线上0.9518。相对I-13逐项为`0/-0.0138/-0.0003/-0.0480/+0.0068/+0.0098/+0.0018/-0.0022`、面板总分差-0.0460，但这只回答“是否替换融合灰区主模型”的榜分问题，不支持纯O1单体路线的因果否决。更接近的非融合固定协议参考I-11为0.9618，I-14名义低0.0100；I-11仍使用164条teacher、从I-10 E3续训且rank不同，也不是干净基线。未提供I-14原始评测日志，action/itemic协议指纹尚未独立复核。
- 撤回旧 I-09 规则数据资格：规则标签相对同源独立judge满分teacher参考的平均F1仅0.0429；匹配实际过滤条件的42条平均F1 0.0813且32条零交集。该teacher参考不是官方gold；`seed_o2_action_r64_lr1e4_ep3`因此在step16中止，W&B `sh96a1sq`，`checkpoints/seed_o2_action_r64_lr1e4_ep3/`无adapter且禁止resume。
- 当前最高单次显示分和固定协议最高均为I-13 `0.9978`。I-10 E3旧协议0.9849仍只能作旧轨迹父模型记录；E3固定协议桥缺失不妨碍I-13在现有固定协议候选中确定为主模型，但仍禁止计算I-13相对E3的净增益。
- r64 同一训练轨迹 E1/E2 已线上评测：E1=0.8839，E2=0.9187；E3未评测且不再建议上传。本地门禁原先只选 E1、拒绝 E2，线上排序相反，门禁不再承担正向 checkpoint 排名。
- `riders_fk_clean_r64_ep3` 训练事实不变：GPU1 单卡，r64/α64、3 epoch、353 steps/epoch、总 1,059 steps；W&B online run [`6gyi8mzc`](https://wandb.ai/3120252125-/llmrec-2026/runs/6gyi8mzc)。E2 action 0.0981 创本账号新高，但 material E1/E2 均为6题。
- `i01_action_distill_r64_ep3` 已完成：3 epoch/1,047 steps，action 截断相对预登记比较对象 riders 减半但 F1 未涨，world v4 大幅回退；状态为本地否决、不上传。蒸馏正式累计 11,432,127 API token。
- `seed_scoremax_r32_ep1` 已完成：只用 `D(O1)`，35,558 行，单卡 1 epoch/740 steps。硬结构保险丝通过，但可见 action 0/5 闭合、5/5 触顶；material 签名 41/14 未进历史 8 题档。后验中点约 0.92，状态为本地否决、不上传。
- `seed_o2_action_r64_lr15e5_ep1` 已完成：`D(O1,O2)` 33,644 行，O2 唯一 action 行 1,164（3.4598%），r64/alpha64、lr1.5e-4、单卡 1 epoch/710 steps。itemic 结构通过，但 action 0/5 闭合、material 39/13，状态为本地否决、不上传。
- E2 的本地 checkpoint 存在，但 `submissions/riders_fk_clean_r64_e2_platform/` 不存在；平台日志只记录临时 `/tmp/eval_model/merged`。在缺上传 manifest 时，不能声称平台工件哈希已由本地 adapter 哈希证明。
- 旧实验的中间 checkpoint、optimizer、失败 checkpoint 和 merged 工作副本已于 2026-07-11 删除；本轮用户批准的 r64 E1/E2/E3 例外已单独列入下表。

## 保留模型

| 角色 | 实验 | 路径 | 哈希 | 状态 |
|---|---|---|---|---|
| 官方基座 | OneReason-0.8B | `models/OneReason-0.8B-pretrain-competition/` | `config.json` SHA256 `5fe26642...` | 只读 |
| I-10 已评测 E1 | seed_teacher_r64_lr1e4_ep3 | `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-665/` | adapter SHA256 `c1bfb4dada8260560327a5ce3a9a15cbb29c0249421616bb1a9d95d9dc11add8` | 1 epoch；线上0.9100 |
| I-10 E1 LoRA上传包 | seed_teacher_r64_lr1e4_ep3 | `submissions/seed_teacher_r64_lr1e4_e1_platform/` | adapter SHA256 `c1bfb4dada8260560327a5ce3a9a15cbb29c0249421616bb1a9d95d9dc11add8`；config SHA256 `f27c697e8bb611802822ea44b156b672c63f6d2ec16a380d868395a9d0eb213f` | 已上传并评测为0.9100 |
| I-10 已评测 E2 | seed_teacher_r64_lr1e4_ep3 | `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1330/` | adapter SHA256 `c4902871c31f1a29b895b3990b2af573808cfecef2a5e483720ce1e60b1ac267` | 2 epoch；线上0.9680 |
| I-10 E2 LoRA上传包 | seed_teacher_r64_lr1e4_ep3 | `submissions/seed_teacher_r64_lr1e4_e2_platform/` | adapter SHA256 `c4902871c31f1a29b895b3990b2af573808cfecef2a5e483720ce1e60b1ac267`；config SHA256 `f27c697e8bb611802822ea44b156b672c63f6d2ec16a380d868395a9d0eb213f` | 已上传并评测为0.9680 |
| 固定协议桥接父模型 | seed_teacher_r64_lr1e4_ep3 | `checkpoints/seed_teacher_r64_lr1e4_ep3/checkpoint-1995/` | adapter SHA256 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2` | 3 epoch；旧协议线上0.9849，固定协议待重评 |
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

`submissions/` 当前保留十份：riders历史单次参考、seed_cotfix失败交付、已评测r64 E1、I-10 E1/E2/E3、已评测I-11/I-12/I-13，以及已评测的纯O1单体I-14 E3两文件包。历史riders r64 E2的本地标准提交包仍缺失；历史提交分数、配置和日志保存在实验台账与归档总账中。

I-10 根目录最终产物为 `checkpoints/seed_teacher_r64_lr1e4_ep3/adapter_model.safetensors`，SHA256与E3同为 `37678b2516011d52494e1c34b66ee072f768911d68884218da56779c8f1c8fc2`。三个checkpoint和根目录均未保存optimizer、scheduler或RNG状态。

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
| 线上 | `seed_clean_r80_lr1e4_ep3_rerun1_V1_eval_20260714152001`；2026-07-14 15:20:05；1h10m32s；总分0.9518；八项按material/action/topic/video/prod/ad/live/world为`0.2453/0.1045/0.0387/0.0480/0.1292/0.1414/0.1080/0.1368`；账号`SL1ACE8AD6710`。按时间切点归入`platform-stable-v3.1-20260713`，但未取得原始日志复核协议指纹 |
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
