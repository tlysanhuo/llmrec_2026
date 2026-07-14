# 实验总账 (EXPERIMENT_INDEX) — 每次实验的全部产物对账表

> **变更记录(2026-07-11 09:36 UTC)**:登记 `seed_taskbal_lora_ep1` 的官方种子数据构造、task-balanced loss、最终1ep adapter/merged模型及双门禁;action 0/4 JSON且全触顶4096,本地否决不传。按用户要求删除全部中间checkpoint,只留最终1ep产物。
> **变更记录(2026-07-11 07:38 UTC)**:归位 `stage2_gold_v1_lora_ep1` 的构建脚本、数据审计、config、adapter、merged模型、W&B和门禁日志,状态标为“本地否决不传”。原因是实验全链已完成但此前未进总账,容易被误列为待上传或重复训练。
> **变更记录(2026-07-10 14:58 UTC)**:移除未发生训练的DPO/RL路线登记;本表只维护真实实验及其产物链。
> **变更记录(2026-07-10 12:06 UTC)**:回填 `riders_act_v1` 线上结果、正式评测记录名、evalTaskId、时长及归位日志。原因是该包已消耗评测并确认 action 增料为负收益,不能继续显示为“待传”。
> **变更记录(2026-07-10 09:48 UTC)**:补录 `riders_act_v1` 的 config/checkpoint/merged/package 链路及实测哈希,并记录实际 381 steps。原因是该实验已完成打包但此前未进入总账,且 experiment_log 的 433 steps 为误记。
> **变更记录(2026-07-10 09:35 UTC)**:`seed_raw_lora_ep1` 按用户裁决改为封存不传。训练、合并和提交包保留用于审计,不再安排训练或平台评测;原因是相对既有全参 raw 基线的预期增益不足。
> 建于 2026-07-03(用户要求彻底理清文件管理后逐一哈希核验)。**每次新实验必须在此表加一行;每次上传/出分必须回填。**
> 与 `experiment_log.md` 分工:那边记**分数与归因**,这边记**文件在哪、对不对得上**。
> 哈希 = model.safetensors 的 md5 前 8 位,用于核对"提交包里的模型到底是哪个 ckpt"。

## ★ 资产速览(2026-07-11 盘点)

- **赛题/规则入口**:`README.md` 负责当前最高分/读分纪律/毒物清单;`docs/platform_guide.md` 是平台规则与评测机制唯一权威;`docs/experiment_log.md` 记录分数与归因;本文件记录 config/ckpt/包/日志对账;`docs/TODO.md` 记录下一步动作。
- **运行区入口**:`data/`、`models/`、`checkpoints/`、`logs/`、`cache/`、`tmp/`、`wandb/` 均为指向 `/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/` 的符号链接;大文件默认落运行区。
- **基座模型**:`models/OneReason-0.8B-pretrain-competition/`;上传包只收 bf16 safetensors,不得改 tokenizer/vocab/config 结构。
- **训练数据成品**:当前权威目录为 `data/processed/`,已登记/使用的关键集包括 `data_final`(32480 seed)、`data_riders_fk`(37267,当前最高分底盘)、`data_stage2_gold_v1`(8030,已训且门禁否决)、`stage2_gold_v1_holdout`(758,独立留出)、`data_official_rec_v3`(37267,已训已证伪)、`data_quality_swap_v1`(37267,已训已证伪)、`data_global_v1`(33637,已训已证伪)、`data_ally_map_v2`(45267,待重审)等;构建脚本统一在`scripts/data/`,审计落`logs/data/`。
- **当前可用模型资产**:最高已出分是 `riders_fk_lora_ep1`(0.9177);`stage2_gold_v1_lora_ep1` 已训练/merge但双门禁否决,仅留复现审计,不得上传或 warm start;已线上证伪的四个 LoRA 谱系:`official_rec_v3_lora_ep1` 0.7948、`quality_swap_v1_lora_ep1` 0.8235、`global_v1_lora_ep1` 0.8246、`evalform_v1_lora_ep1` 0.7571;`riders_fk_lora_ep2` 已训完/merge/precheck/打包但不是当前获批上传项。
- **清理红线**:`recipe2_w5_ep1_platform/` 是线上 v5 唯一模型副本,不可删;`baseline_sft_v1_upload/`、旧 ckpt 子目录、哈希名重复日志、误传 notebook 等清理项仍需用户批准。

## ★ 命名规矩(2026-07-03 用户指出命名混乱后定,最高优先级)

**问题根源**:同一个实验存在最多 5 个名字(内部代号 v6 / config名 exp_seed_ep3 / 提交包名 ep3_seed_platform / 平台上传名 seed_ep3 / 平台评测记录 seed_ep3_V1_eval_20260703150200),用户在平台上看到的名字和我们文档里的代号对不上。

**新规矩:一个实验只有一个名字 = 平台上传名。**
1. 起实验前先定这个名字(短、可读、含关键变量,如 `rebal_lr2e5_ep3`),然后 config/`checkpoints/<名>/`/`submissions/<名>_platform/`/wandb run/评测日志 `logs/eval/<名>_<日期>.log` **全部用它**,禁止再造别名。
2. 平台侧自动衍生的评测记录名(`<名>_V1_eval_<时间戳>`)与评测时间/时长,出分后**当天回填本表**。⚠️ 平台记录里的 `SL1ACE8AD6710` 在全部 7 条评测上都相同——**是账号级 ID,不是单次评测 ID**(07-03 用户贴全量记录后确认;此前两次误当评测 ID 记录,均已订正)。
3. 旧的 vN 编号只作为 experiment_log 的历史行号保留;**对话和新文档里提实验一律用平台名**(如"seed_ep3(0.8931)",不说"v6")。
4. **评测配额是账号级、与队友共享**(07-02 的 3 次里 baseline-epoch3 是队友传的)——上传前查平台当日已用次数并与队友协调。

## 主对账表(2026-07-03 全部经 md5 实测核验;平台名/时间由用户 07-03 提供的平台全量记录回填)

| 版本 | 训练名 | 平台上传名 | 平台评测记录名 | 评测时间 / 时长 | config | ckpt | 模型md5 | 提交包 | 评测日志 | evalTaskId | 总分 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | (官方pretrain) | — | 官方参考 | — | — | `models/OneReason-0.8B-pretrain-competition/` | — | — | — | — | 0.6655 |
| v1 | baseline_sft_v1 | baseline_sft | baseline_sft | 07-01 13:25:22 / 1h37m19s | `configs/baseline/baseline_sft_v1.yaml` | `checkpoints/baseline_sft_v1/` | `60690c0f` | `submissions/baseline_sft_v1_platform/` ✓ | `logs/eval/baseline_sft_v1_20260701.log` | eval-task-mvtb79 | **0.810** |
| v2 | run_a_r2 | v2 | v2_V1_eval_20260701195109 | 07-01 19:51:15 / 1h5m4s | `configs/history/run_a_r2.yaml` | `checkpoints/run_a_r2/` | `f8d56eba` | `submissions/run_a_r2_platform/` ✓ | `logs/eval/run_a_r2_20260701.log` | eval-task-38vfcj | 0.8092 |
| v3 | run_c_material | baseline_v3 | onereason-material-v3 | 07-02 11:38:42 / 1h35m0s | `configs/history/run_c_material.yaml` | `checkpoints/run_c_material/` | `f79a4e5a` | `submissions/run_c_material_platform/` ✓ | `logs/eval/run_c_material_20260702.log` | eval-task-g7h8gm | 0.8198 |
| **(队友)** | **baseline-epoch3(非本机训练)** | baseline-epoch3 | baseline-epoch3_V1_eval_20260702153958 | 07-02 15:40:12 / 1h9m27s | **不在本机,待向队友要** | 不在本机 | — | — | 无(日志未下载) | — | **0.7807** |
| (D) | run_d_r2material | (未上传) | — | — | `configs/history/run_d_r2material.yaml` | `checkpoints/run_d_r2material/` | `49273660` | `submissions/run_d_r2material_platform/` ✓ | — | — | — |
| v4 | recipe1_bs32_lr1e4_ep3 | OneReason-0.8B-recipe1-seed-bs32-ep3 | OneReason-0.8B-recipe1-seed-bs32-ep3 | 07-02 19:05:43 / 1h10m19s | `configs/history/recipe1_bs32_lr1e4_ep3.yaml` | `checkpoints/recipe1_bs32_lr1e4_ep3/` | `c9852e9b` | `submissions/recipe1_ep3_platform/` ✓ | `logs/eval/recipe1_20260702.log` | eval-task-b7l9ri | **0.8428** |
| v5 | recipe2_w5_ep1(=recipe2_itemic_w5 的 ckpt-49/1ep) | recipe2_w5_ep1 | recipe2_w5_ep1_V1_eval_20260703001348 | 07-03 00:13:56 / 1h31m0s | `configs/history/recipe2_itemic_w5.yaml` | ⚠️ ckpt-49 已删,仓库存的 ckpt-98(2ep,`45485fa0`)从未上传 | `8d250372` | `submissions/recipe2_w5_ep1_platform/` ⚠️**模型唯一副本,不可删** | `logs/eval/recipe2_w5_ep1_20260703.log` | eval-task-qmlx3b | 0.7692 |
| v6 | **seed_ep3**(本地ckpt目录 exp_seed_ep3,历史遗留) | seed_ep3 | seed_ep3_V1_eval_20260703150200 | 07-03 15:02:07 / 1h8m28s | `configs/history/exp_seed_ep3.yaml` | `checkpoints/exp_seed_ep3/` | `bb22501e` | `submissions/ep3_seed_platform/` ✓ | `logs/eval/seed_ep3_20260703.log` | eval-task-llfijj | **0.8931 ★** |
| **★** | **rebal_world_ep3** | rebal_world_ep3 | rebal_world_ep3_V1_eval_20260703223107 | 07-03 22:31:19 / 1h22m53s | `configs/history/rebal_world_ep3.yaml` | `checkpoints/rebal_world_ep3/`(含ckpt-320=1ep/640=2ep) | `3029136b` | `submissions/rebal_world_ep3_platform/` ✓ | `logs/eval/rebal_world_ep3_20260703.log` | eval-task-4twavz | **0.9009 ★新高** |
| ✗ | rebal_mat_ep3 | rebal_mat_ep3 | rebal_mat_ep3_V1_eval_20260704084058 | 07-04 08:40:58 / 1h15m10s | `configs/history/rebal_mat_ep3.yaml` | `checkpoints/rebal_mat_ep3/`(含ckpt-324/648/972) | `add0609f` | `submissions/rebal_mat_ep3_platform/` ✓ | `logs/eval/rebal_mat_ep3_20260704.log` | eval-task-zcb312 | **0.8454**(重复上采样=净毒药,假设证伪) |
| r3 | recipe3_nothink_1ep | (不传,已证伪:懂世界归零) | — | — | `configs/history/recipe3_nothink_1ep.yaml` | `checkpoints/recipe3_nothink_1ep/` | `f0e372bd` | — | — | — | — |
| r4 | recipe4_kexi_repro | (未上传;若传,平台名建议 kexi_repro) | — | — | `configs/history/recipe4_kexi_repro.yaml` | `checkpoints/recipe4_kexi_repro/` | `6ae212ec` | — | — | — | — |
| r5 | recipe5_seed_world8 | (不传,欠训) | — | — | `configs/history/recipe5_seed_world8.yaml` | `checkpoints/recipe5_seed_world8/` | `0d0011a5` | — | — | — | — |
| r6 | recipe6_rebal_world | (不传,复读70%) | — | — | `configs/history/recipe6_rebal_world.yaml` | `checkpoints/recipe6_rebal_world/` | `aefee748` | — | — | — | — |
| **★★** | **riders_fk_lora_ep1** | riders_fk_lora_ep1 | riders_fk_lora_ep1(V1) | 07-06 19:16:21 / 1h16m59s | `configs/retained/riders_fk_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/riders_fk_lora_ep1/`(adapter)+ `riders_fk_lora_ep1_merged/` | `c2046b60`(merged) | `submissions/riders_fk_lora_ep1_platform/` ✓ | `logs/eval/riders_fk_lora_ep1_20260706.log` | (未记录) | **0.9177 ★★新高** |
| ✗ | global_v1_lora_ep1 | global_v1_lora_ep1 | global_v1_lora_ep1_V1_eval_20260707203748 | 07-07 20:38:02 / 1h29m6s | `configs/history/global_v1_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/global_v1_lora_ep1/`(adapter)+ `global_v1_lora_ep1_merged/` | `a1243040`(merged) | `submissions/global_v1_lora_ep1_platform/` ✓ | `logs/eval/global_v1_lora_ep1_20260707.log` | eval-task-m1phs0-1783427882 | **0.8246**(07-08 用户从UI回填;证伪留档) |
| ✗ | quality_swap_v1_lora_ep1 | quality_swap_v1_lora_ep1 | quality_swap_v1_lora_ep1_V1_eval_20260708122552 | 07-08 12:49:29 / 1h46m11s | `configs/history/quality_swap_v1_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/quality_swap_v1_lora_ep1/`(adapter) | `830abb8b`(adapter) | LoRA adapter 直传,本地无 `submissions/quality_swap*` | `logs/eval/quality_swap_v1_lora_ep1_20260708.log` | eval-task-jysa9i-1783486168 | **0.8235**(证伪) |
| ✗ | official_rec_v3_lora_ep1 | official_rec_v3_lora_ep1 | official_rec_v3_lora_ep1_V1_eval_20260708171051 | 07-08 17:11:44 / 1h25m9s | `configs/history/official_rec_v3_lora_ep1.yaml` | `checkpoints/official_rec_v3_lora_ep1/`(adapter) | `746b7bc0`(adapter) | `submissions/official_rec_v3_lora_ep1_adapter/` + `.zip` ✓ | 待下载 | — | **0.7948**(证伪) |
| 待训 | ally_map_v2_lora_ep1 | ally_map_v2_lora_ep1 | — | — | `configs/history/ally_map_v2_lora_ep1.yaml` | — | — | LoRA adapter 路径,训后只需 `adapter_model.safetensors`+`adapter_config.json` | — | — | 数据+config就绪,训练需用户明示 |
| ✗ | evalform_v1_lora_ep1 | evalform_v1_lora_ep1 | evalform_v1_lora_ep1_V1_eval_20260709102947 | 07-09 10:30:10 / 1h26m17s | `configs/history/evalform_v1_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/evalform_v1_lora_ep1/`(adapter)+ `evalform_v1_lora_ep1_merged/` | `d5996527`(adapter)/`c61824ee`(merged) | `submissions/evalform_v1_lora_ep1_platform/` ✓(md5 与 ckpt adapter 一致) | `logs/eval/evalform_v1_lora_ep1_20260709.log` | eval-task-6tn1xs-1783564210 | **0.7571**(证伪:rec 全域大跌+mat 陪跌) |
| ⏳已打包未传 | riders_fk_lora_ep2 | riders_fk_lora_ep2(拟) | — | — | `configs/history/riders_fk_lora_ep2.yaml`(+`_merge.yaml`) | `checkpoints/riders_fk_lora_ep2/`(adapter,含ckpt-352/704)+ `riders_fk_lora_ep2_merged/` | `efaee6fb`(adapter)/`2292e24d`(merged) | `submissions/riders_fk_lora_ep2_platform/` ✓(md5 与 ckpt adapter 一致) | — | — | 待用户裁决是否上传 |
| ✗ | riders_fk_plat_ep1 | riders_fk_plat_ep1-epoch1 | riders_fk_plat_ep1-epoch1_V1_eval_20260709151911 | 07-09 15:19:16 / 1h23m28s | 平台 UI 精调(表单参数=riders 值,lr2e-4/r32/α32/dropout0.05/wd0.001/累积4/1ep/thinking关);数据 `data/processed/riders_fk_platform.jsonl`(md5 `fb81dd68`,构建脚本 `scripts/data/build_riders_fk_platform.py`) | 平台侧,无本地 ckpt(平台训练无需上传) | — | — | `logs/eval/riders_fk_plat_ep1_20260709.log` | eval-task-bwvp08-1783581556 | **0.8229**(判决修正:非平台loss证伪,是lr2e-4过冲) |
| ⏸封存不传 | seed_raw_lora_ep1 | —(不上传) | — | — | `configs/history/seed_raw_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/seed_raw_lora_ep1/`(adapter)+ `seed_raw_lora_ep1_merged/` | `55eb6a02`(adapter) | `submissions/seed_raw_lora_ep1_platform/` ✓(md5 与 ckpt 一致,仅留档) | — | — | 07-10 用户裁决:不重训、不上传、不评测;预期相对全参 raw 基线增益不足 |
| ✗ | riders_act_v1 | riders_act_v1 | act_v1_V1_eval_20260710180459 | 07-10 18:05:09 / 1h26m33s | `configs/history/riders_act_v1.yaml`(+`_merge.yaml`) | `checkpoints/riders_act_v1/`(adapter,381 steps)+ `riders_act_v1_merged/` | `547497c0`(adapter) | `submissions/riders_act_v1_platform/` ✓(SHA256/ckpt md5 均一致) | `logs/eval/riders_act_v1_20260710.log` | eval-task-cax6gm-1783677909 | **0.8835**(action 0.0573,低于 riders 0.0655;P3-v2 长尾/搜索增料证伪) |
| ✗本地门禁 | stage2_gold_v1_lora_ep1 | —(不上传) | — | 07-11 本地训练7m40s/门禁完成 | `configs/history/stage2_gold_v1_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/stage2_gold_v1_lora_ep1/`(adapter)+`stage2_gold_v1_lora_ep1_merged/` | `5a94d375`(adapter)/`b46c62fe`(merged) | 无(门禁前否决) | `logs/offline_eval/stage2_gold_v1_vs_seed_ep3_20260711_0543.json`+`logs/probe/visible_action_stage2_gold_v1_platform5_20260711.json` | — | **未传**:action可见题0/5 JSON,物料签名14/7;不续训/不warm start |
| ✗本地门禁 | seed_taskbal_lora_ep1 | —(不上传) | — | 07-11 单卡训练49m48s/门禁完成 | `configs/history/seed_taskbal_lora_ep1.yaml`(+`_merge.yaml`),启动器`scripts/train/train_task_balanced.py` | `checkpoints/seed_taskbal_lora_ep1/`(仅最终1ep adapter)+`seed_taskbal_lora_ep1_merged/` | `e323616a`(adapter)/`2f35222e`(merged) | 无(门禁前否决) | `logs/probe/visible_action_seed_taskbal_lora_ep1_platform5_20260711.json` | — | **未传**:material 35/13保7题锚;action 0/4 JSON且4/4触顶4096;不续训 |

**历史别名对照(旧账,以本表为准):** 平台 seed_ep3 = 本地 exp_seed_ep3 = 提交包 ep3_seed_platform = 代号 v6;平台 OneReason-0.8B-recipe1-seed-bs32-ep3 = recipe1_bs32_lr1e4_ep3 = recipe1_ep3_platform = v4;平台 baseline_v3/记录 onereason-material-v3 = run_c_material = v3;平台 v2 = run_a_r2;平台 baseline_sft = baseline_sft_v1 = v1;recipe2_w5_ep1 全链一致 = v5。

## ⚠️ 核验发现的问题(2026-07-03 起)

1. **v5 线上模型(0.7692)的唯一副本在 `submissions/recipe2_w5_ep1_platform/`**——它是 recipe2 训练中途的 checkpoint-49(1ep),后被"只留最终ckpt"的清理删掉了。`checkpoints/recipe2_itemic_w5/` 里的是 ckpt-98(2ep),`45485fa0`,**从未上过线**。此提交包是线上真值锚点,**绝对不可删**。
2. **全部 11 个 ckpt 目录存在"根目录 + checkpoint-N 子目录"双份模型**,md5 逐一核验完全相同,纯冗余 ≈17.6G。且 recipe4/5/6 的 checkpoint-N 里还有 optimizer.pt(3.0G×3)——7-03 上午的清理只清了老 ckpt,新的 3 个漏了。清理方案见 TODO(待用户批准后执行)。
3. **LOO 推荐数据其实已建完**:`rec_loo.jsonl` 12000 条(video/prod/ad/living 各 3000),构建日志显示正常跑完(07-03 15:07)。此前速览里"被打断需重跑"的记录已过时。文件已备份到 `data/processed/`。
4. **recipe4(克西复刻)训完后没有进 experiment_log 主表**(已补)。precheck 体检结果此前只留在会话里没落盘——今后体检输出必须存 `logs/precheck/<训练名>.txt`。
5. `notebooks/taskmanager 2.10.1-arm64/`(1.4G)是误传的 macOS 应用安装包(QQ/Obsidian 等),与项目无关;`src/` 为空目录但 README 还在引用。处置待用户批准。
6. **global_v1 配置头注释与实际数据不一致(07-07 盘点发现)**:`configs/history/global_v1_lora_ep1.yaml` 头注释写 `build_riders_fk.py`/37267 条,但实际 `dataset: data_global_v1`,本地行数 33637,构建脚本为 `scripts/data/build_global_v1_assembly.py`。不影响已上传模型复现链,但后续若复用该 config 必须先修注释/补构建日志。
7. **global_v1 平台日志已归位但分数缺口仍在**:原平台哈希日志已改名为 `logs/eval/global_v1_lora_ep1_20260707.log`;日志内只有 evalTaskId 与评测过程,没有 UI 总分/分项,需用户从平台回填。
8. **ally_map_v1 已被 v2 取代**:`configs/history/ally_map_lora_ep1.yaml`、`scripts/data/build_ally_map.py`、`data/processed/data_ally_map.jsonl`(45267 条)仍留作历史资产,但其 rec 配比沿用旧列序(ad/living过重),不得直接训练;当前待训版为 `ally_map_v2_lora_ep1`。
9. **quality_swap_v1 已线上证伪**:`checkpoints/quality_swap_v1_lora_ep1/` 的 LoRA adapter 线上 0.8235,不作 warm start;保留 ckpt/log 仅用于审计与复现。

## 产物归位约定(今后铁律)

| 产物 | 位置 | 命名 |
|---|---|---|
| 训练配置 | `configs/<训练名>.yaml` | 文件头注释写明:目的/单变量/对照版本 |
| 训练产出 | `checkpoints/<训练名>/`(只留根目录一份模型) | 训练日志放 ckpt 内 `train.log` |
| 训练数据 | 构建脚本 `scripts/data/`;成品落 `data/processed/` 并注册到 `ai_runtime/llmrec_2026/LLaMA-Factory/data/dataset_info.json` | 数据文件名与 dataset_info.json 注册名一致 |
| 体检 | 上传前跑 `scripts/eval/precheck.py`,输出存 `logs/precheck/<训练名>.txt` | 对照 recipe1 复读率 33% 基线 |
| 提交包 | `submissions/<训练名>_platform/` + SHA256SUMS.txt | 打包后 md5 核对 ckpt,把哈希填进本表 |
| 评测日志 | `logs/eval/<训练名>_<日期>.log`(平台哈希名下载后立即改名,删原名) | 出分后回填本表 + experiment_log |
| TODO | `docs/TODO.md`(唯一 TODO 文档) | 每会话开头读,变更即写 |
