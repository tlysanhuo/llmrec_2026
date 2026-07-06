# 实验总账 (EXPERIMENT_INDEX) — 每次实验的全部产物对账表

> 建于 2026-07-03(用户要求彻底理清文件管理后逐一哈希核验)。**每次新实验必须在此表加一行;每次上传/出分必须回填。**
> 与 `experiment_log.md` 分工:那边记**分数与归因**,这边记**文件在哪、对不对得上**。
> 哈希 = model.safetensors 的 md5 前 8 位,用于核对"提交包里的模型到底是哪个 ckpt"。

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
| v2 | run_a_r2 | v2 | v2_V1_eval_20260701195109 | 07-01 19:51:15 / 1h5m4s | `configs/run_a_r2.yaml` | `checkpoints/run_a_r2/` | `f8d56eba` | `submissions/run_a_r2_platform/` ✓ | `logs/eval/run_a_r2_20260701.log` | eval-task-38vfcj | 0.8092 |
| v3 | run_c_material | baseline_v3 | onereason-material-v3 | 07-02 11:38:42 / 1h35m0s | `configs/run_c_material.yaml` | `checkpoints/run_c_material/` | `f79a4e5a` | `submissions/run_c_material_platform/` ✓ | `logs/eval/run_c_material_20260702.log` | eval-task-g7h8gm | 0.8198 |
| **(队友)** | **baseline-epoch3(非本机训练)** | baseline-epoch3 | baseline-epoch3_V1_eval_20260702153958 | 07-02 15:40:12 / 1h9m27s | **不在本机,待向队友要** | 不在本机 | — | — | 无(日志未下载) | — | **0.7807** |
| (D) | run_d_r2material | (未上传) | — | — | `configs/run_d_r2material.yaml` | `checkpoints/run_d_r2material/` | `49273660` | `submissions/run_d_r2material_platform/` ✓ | — | — | — |
| v4 | recipe1_bs32_lr1e4_ep3 | OneReason-0.8B-recipe1-seed-bs32-ep3 | OneReason-0.8B-recipe1-seed-bs32-ep3 | 07-02 19:05:43 / 1h10m19s | `configs/recipe1_bs32_lr1e4_ep3.yaml` | `checkpoints/recipe1_bs32_lr1e4_ep3/` | `c9852e9b` | `submissions/recipe1_ep3_platform/` ✓ | `logs/eval/recipe1_20260702.log` | eval-task-b7l9ri | **0.8428** |
| v5 | recipe2_w5_ep1(=recipe2_itemic_w5 的 ckpt-49/1ep) | recipe2_w5_ep1 | recipe2_w5_ep1_V1_eval_20260703001348 | 07-03 00:13:56 / 1h31m0s | `configs/recipe2_itemic_w5.yaml` | ⚠️ ckpt-49 已删,仓库存的 ckpt-98(2ep,`45485fa0`)从未上传 | `8d250372` | `submissions/recipe2_w5_ep1_platform/` ⚠️**模型唯一副本,不可删** | `logs/eval/recipe2_w5_ep1_20260703.log` | eval-task-qmlx3b | 0.7692 |
| v6 | **seed_ep3**(本地ckpt目录 exp_seed_ep3,历史遗留) | seed_ep3 | seed_ep3_V1_eval_20260703150200 | 07-03 15:02:07 / 1h8m28s | `configs/exp_seed_ep3.yaml` | `checkpoints/exp_seed_ep3/` | `bb22501e` | `submissions/ep3_seed_platform/` ✓ | `logs/eval/seed_ep3_20260703.log` | eval-task-llfijj | **0.8931 ★** |
| **★** | **rebal_world_ep3** | rebal_world_ep3 | rebal_world_ep3_V1_eval_20260703223107 | 07-03 22:31:19 / 1h22m53s | `configs/rebal_world_ep3.yaml` | `checkpoints/rebal_world_ep3/`(含ckpt-320=1ep/640=2ep) | `3029136b` | `submissions/rebal_world_ep3_platform/` ✓ | `logs/eval/rebal_world_ep3_20260703.log` | eval-task-4twavz | **0.9009 ★新高** |
| ✗ | rebal_mat_ep3 | rebal_mat_ep3 | rebal_mat_ep3_V1_eval_20260704084058 | 07-04 08:40:58 / 1h15m10s | `configs/rebal_mat_ep3.yaml` | `checkpoints/rebal_mat_ep3/`(含ckpt-324/648/972) | `add0609f` | `submissions/rebal_mat_ep3_platform/` ✓ | `logs/eval/rebal_mat_ep3_20260704.log` | eval-task-zcb312 | **0.8454**(重复上采样=净毒药,假设证伪) |
| r3 | recipe3_nothink_1ep | (不传,已证伪:懂世界归零) | — | — | `configs/recipe3_nothink_1ep.yaml` | `checkpoints/recipe3_nothink_1ep/` | `f0e372bd` | — | — | — | — |
| r4 | recipe4_kexi_repro | (未上传;若传,平台名建议 kexi_repro) | — | — | `configs/recipe4_kexi_repro.yaml` | `checkpoints/recipe4_kexi_repro/` | `6ae212ec` | — | — | — | — |
| r5 | recipe5_seed_world8 | (不传,欠训) | — | — | `configs/recipe5_seed_world8.yaml` | `checkpoints/recipe5_seed_world8/` | `0d0011a5` | — | — | — | — |
| r6 | recipe6_rebal_world | (不传,复读70%) | — | — | `configs/recipe6_rebal_world.yaml` | `checkpoints/recipe6_rebal_world/` | `aefee748` | — | — | — | — |
| **★★** | **riders_fk_lora_ep1** | riders_fk_lora_ep1 | riders_fk_lora_ep1(V1) | 07-06 19:16:21 / 1h16m59s | `configs/riders_fk_lora_ep1.yaml`(+`_merge.yaml`) | `checkpoints/riders_fk_lora_ep1/`(adapter)+ `riders_fk_lora_ep1_merged/` | `c2046b60`(merged) | `submissions/riders_fk_lora_ep1_platform/` ✓ | `logs/eval/riders_fk_lora_ep1_20260706.log` | (未记录) | **0.9177 ★★新高** |

**历史别名对照(旧账,以本表为准):** 平台 seed_ep3 = 本地 exp_seed_ep3 = 提交包 ep3_seed_platform = 代号 v6;平台 OneReason-0.8B-recipe1-seed-bs32-ep3 = recipe1_bs32_lr1e4_ep3 = recipe1_ep3_platform = v4;平台 baseline_v3/记录 onereason-material-v3 = run_c_material = v3;平台 v2 = run_a_r2;平台 baseline_sft = baseline_sft_v1 = v1;recipe2_w5_ep1 全链一致 = v5。

## ⚠️ 核验发现的问题(2026-07-03)

1. **v5 线上模型(0.7692)的唯一副本在 `submissions/recipe2_w5_ep1_platform/`**——它是 recipe2 训练中途的 checkpoint-49(1ep),后被"只留最终ckpt"的清理删掉了。`checkpoints/recipe2_itemic_w5/` 里的是 ckpt-98(2ep),`45485fa0`,**从未上过线**。此提交包是线上真值锚点,**绝对不可删**。
2. **全部 11 个 ckpt 目录存在"根目录 + checkpoint-N 子目录"双份模型**,md5 逐一核验完全相同,纯冗余 ≈17.6G。且 recipe4/5/6 的 checkpoint-N 里还有 optimizer.pt(3.0G×3)——7-03 上午的清理只清了老 ckpt,新的 3 个漏了。清理方案见 TODO(待用户批准后执行)。
3. **LOO 推荐数据其实已建完**:`rec_loo.jsonl` 12000 条(video/prod/ad/living 各 3000),构建日志显示正常跑完(07-03 15:07)。此前速览里"被打断需重跑"的记录已过时。文件已备份到 `data/processed/`。
4. **recipe4(克西复刻)训完后没有进 experiment_log 主表**(已补)。precheck 体检结果此前只留在会话里没落盘——今后体检输出必须存 `logs/precheck/<训练名>.txt`。
5. `notebooks/taskmanager 2.10.1-arm64/`(1.4G)是误传的 macOS 应用安装包(QQ/Obsidian 等),与项目无关;`src/` 为空目录但 README 还在引用。处置待用户批准。

## 产物归位约定(今后铁律)

| 产物 | 位置 | 命名 |
|---|---|---|
| 训练配置 | `configs/<训练名>.yaml` | 文件头注释写明:目的/单变量/对照版本 |
| 训练产出 | `checkpoints/<训练名>/`(只留根目录一份模型) | 训练日志放 ckpt 内 `train.log` |
| 训练数据 | 构建脚本 `scripts/data/`;成品落 `/root/baseline_repro/data/`(训练读这里)**并同步备份到 `data/processed/`**(lustre 持久,overlay 会随容器重建丢失) | 数据文件名与 dataset_info.json 注册名一致 |
| 体检 | 上传前跑 `scripts/eval/precheck.py`,输出存 `logs/precheck/<训练名>.txt` | 对照 recipe1 复读率 33% 基线 |
| 提交包 | `submissions/<训练名>_platform/` + SHA256SUMS.txt | 打包后 md5 核对 ckpt,把哈希填进本表 |
| 评测日志 | `logs/eval/<训练名>_<日期>.log`(平台哈希名下载后立即改名,删原名) | 出分后回填本表 + experiment_log |
| TODO | `docs/TODO.md`(唯一 TODO 文档) | 每会话开头读,变更即写 |
