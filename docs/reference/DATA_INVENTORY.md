# 数据目录说明(DATA_INVENTORY)

> **变更记录(2026-07-11 07:38 UTC)**:登记 `data_stage2_gold_v1`、独立holdout、数据审计、checkpoint和平台可见题日志现状,并修正训练权威路径。原因是07-11二阶段实验已完成全链,旧清单停在07-03会造成数据与模型重复使用。
> 生成于 2026-07-03,盘点全部数据文件的位置/来源/用途,并记录清理记录。之后新增数据文件请更新本文档。

## 1. 种子 SFT 数据(万擎平台【数据管理】下载,训练主数据源)

位置:`llmrec_2026/data/懂*.jsonl`(原始12个文件,按官方任务命名)+ `llmrec_2026/data/dataset.tar.gz`(同一份数据的打包备份)。

| 文件 | 条数 | 任务 |
|---|---|---|
| 懂物料part1~7.jsonl | 1611/784/1581/1621/1589/1619/1579 | 物料 desc↔token(video/prod/ad/living混合) |
| 懂推荐1~4.jsonl | 5426/5442/5372/2964 | 推荐 CoT(ad/live/prod/video) |
| 懂用户.jsonl | 2892 | action_select / topic_gen / direct_gen |
| **合计** | **32480** | 三赛道全量种子 |

- 原始格式:`[{"system":..., "prompt":..., "response":...}]`(每行一个长度为1的list)。
- 转换脚本:`docs/demo_baseline/convert_jsonl.py` → Alpaca 格式(`instruction/input/output/history`)。
- **训练实际读取路径**:转换后的 `data_final.jsonl` 位于 `data/processed/data_final.jsonl`(32480条),LLaMA-Factory 通过卷上 `LLaMA-Factory/data/dataset_info.json` 的 `data_final` 条目引用。
- `data_nothink.jsonl`(同目录,32480条)= recipe3 实验用的 CoT 剥离版,产出脚本 `scripts/data/build_nothink.py`。recipe3 已被证伪(懂世界acc归零),**此文件仅作历史记录,不建议再用于训练**。

## 2. 官方 17GB 原始素材(HF `OpenOneRec/Explorer_LLM_Rec_Competition`)

位置:`llmrec_2026/data/hf_full/data/`,全量下载,完整性已核对(shard数与HF API tree一致)。

| 子目录 | 大小 | shards | 内容 |
|---|---|---|---|
| OneReason_UserProfile | 8.0G | 10 | 63列原始用户行为序列(未加工,pid为hash) |
| OneReason_Pid2Caption | 5.7G | 136 | item→中文文本描述 |
| OneReason_Pid2Sid | 501M | 198 | item pid→(s_a,s_b,s_c) 语义ID映射表 |
| OneReason_Pid2Tag | 61M | 31 | item→三级类目标签 |
| OneReason_General | 1.9G | 158 | 通用QA(含CoT),懂世界候选素材 |

用途:自行构造训练数据的原材料(种子数据是从这批原始素材加工出来的一个切片,官方文档称"仅作格式参考,可自由重新清洗/配比")。**目前尚未基于此构造过任何训练数据**(v2/v3 用的是另一个数据源,见下)。

`data/index/pid2*.parquet`(合并索引,原6.6G)**已于本次清理删除**——是从上表派生的单文件索引,可随时用 `pandas.concat` 从 hf_full 重新生成,不是原始数据。

## 3. 通用SFT补充数据(懂世界候选池,尚未使用)

位置:`llmrec_2026/data/hf_general_sft/`,301个parquet,255万条,9个HF数据源(OpenMathReasoning/R1-Distill-SFT/Infinity_Instruct等),apache-2.0。

- 与上面 `hf_full/data/OneReason_General`(1.9G/158shard)是**两个不同的数据集**,不重复:一个是官方17GB包里的原版(158shard/158k条),这份是后续另外拉取的更大规模补充池(255万条)。
- 中文占比高、可直接用于懂世界任务的只有3个源:Chinese-Reasoning-Distil-Data-think(17.9万,中文99%,带CoT)、Infinity_Instruct(44.7万,中文约10%)、medical-o1-reasoning-SFT-think(5万,中文约40%)。其余6个源(数学/代码/英文推理)中文占比<1%,已论证不适合用于懂世界混入。
- **尚未在任何训练里实际使用**。用户已确认保留,留作后续懂世界数据混入实验(momo实测同类混入 +0.024)。

## 4. 评测日志(平台评测生成,已规范化到 `logs/eval/`)

命名规范:`<版本名>_<日期>.log`。截至2026-07-11共有20份平台日志;完整版本、总分和evalTaskId以`docs/EXPERIMENT_INDEX.md`为准。日志均展示8任务各5道可见样例,稳定性与门禁用途见`docs/offline_eval.md` §8。

| 文件 | 版本 | 评测分 |
|---|---|---|
| baseline_sft_v1_20260701.log | v1 | 0.810 |
| run_a_r2_20260701.log | v2 | 0.8092 |
| run_c_material_20260702.log | v3 | 0.8198 |
| recipe1_20260702.log | v4 | 0.8428 |
| recipe2_w5_ep1_20260703.log | v5 | 0.7692 |
| seed_ep3_20260703.log | v6 | **0.8931 ★** |

> ⚠️ v6 日志还有一个平台哈希文件名副本(`Kne1N...y7r.log`,cmp 逐字节相同),待批准删除(见 docs/TODO.md P2)。

> 之前散落在 `data/` 目录下的4份 + `logs/eval/` 下1份**哈希文件名重复副本**(平台传输时的原始文件名)已核实 evalTaskId 与上表逐一匹配,本次清理已删除。

## 5. 训练/评测派生数据(★2026-07-03 已全部迁离 overlay)

> **用户指令 + 卷根 AGENTS.md 守则:一切持久内容(数据+环境)放个人卷。** 2026-07-03 执行:数据 md5 核验迁移后,`/root/baseline_repro/` **已整体删除**(含环境);LLaMA-Factory+venv 重建于 `ai_runtime/llmrec_2026/LLaMA-Factory/`(项目根符号链接),LF `dataset_info.json` 指向本目录(死条目 run_a/c/d_mix 清除,如需复现用 `scripts/data/` 重建)。
> **训练数据权威路径 = `llmrec_2026/data/processed/`;训练环境 = `ai_runtime/llmrec_2026/LLaMA-Factory/.venv`(激活它起训练)。**

| 文件(`data/processed/`) | 条数 | 用途 |
|---|---|---|
| `data_final.jsonl` | 32480 | 训练主数据(种子转换后,LF注册名 data_final) |
| `data_rebal_world.jsonl` | 29019 | recipe6/7:rec域重平衡+8%通识 |
| `data_seed_world8.jsonl` | 35304 | recipe5:种子+8%通识 |
| `world_zh.jsonl` | 16237 | 中文通识池(官方General洗出,未注册LF) |
| `rec_loo.jsonl` | 12000 | LOO推荐预测数据(四域各3000,RFT/RL用,未注册LF) |
| `data_nothink.jsonl` | 32480 | recipe3 CoT剥离版(已证伪,存档) |
| `data_stage2_gold_v1.jsonl` | 8030 | 07-11二阶段原生非物料no-think集:action1430/topic300/rec四域各1200/world1500;material0;md5 `76fc1685`;已训且门禁否决 |
| `stage2_gold_v1_holdout.jsonl` | 758 | 与上项按题面分组隔离的独立holdout:action158/topic100/rec四域各50/world300;md5 `5ef50e4c` |
| `sample_mini.jsonl` / `smoke_200.jsonl` | 100/200 | 调试用(LF注册名 sample_mini/smoke_200) |
| `dataset_info.json` | — | LF 注册表的 lustre 备份(权威副本在 overlay LF 里,已指向本目录) |
| `build_rec_loo.log` / `build_world_zh.log` | — | 数据构建日志 |

另:`data/overlay_archive_20260703/`(5.1M)= overlay 上其余小件归档(OneReason_report.txt、eval_log_chunks、log3way、hf_samples、seed_samples、onereason_chunks、official_data_final.jsonl)。早期 proxy 文件(mat_data/r2_data/proxy_*.txt)在 proxy 套件废弃时已删,不在归档内。

## 6. Checkpoints(`llmrec_2026/checkpoints/`)

> **★2026-07-03 二次盘点(md5 全量核验)**:11 个训练目录**全部**存在「根目录 + `checkpoint-N/` 子目录」双份模型(md5 逐一相同,纯冗余 ≈17.6G);且 recipe4/5/6 的子目录里还各有 optimizer.pt(3.0G×3)——上午的 optimizer 清理漏掉了这三个新目录。删除方案已列 `docs/TODO.md` P2,**等用户批准后执行**。训练日志已统一为各 ckpt 目录内 `train.log`(原散落的 4 个 `*_launch.log` 已归位)。
> 全部版本的 config/ckpt/提交包/评测日志对账表(含 md5)见 **`docs/EXPERIMENT_INDEX.md`**。

| 版本 | 目录 | 模型md5(前8) | 当前大小 | 备注 |
|---|---|---|---|---|
| v1 | baseline_sft_v1 | 60690c0f | 3.1G(含dup) | 线上0.810 |
| v6 | exp_seed_ep3 | bb22501e | 3.1G(含dup) | **线上0.8931 ★最高** |
| v4 | recipe1_bs32_lr1e4_ep3 | c9852e9b | 3.1G(含dup) | 线上0.8428 |
| — | recipe2_itemic_w5 | 45485fa0 | 3.1G(含dup) | ⚠️这是ckpt-98(2ep),**从未上线**;线上v5(0.7692)=ckpt-49,唯一副本在 `submissions/recipe2_w5_ep1_platform/`(md5 8d250372),**该提交包不可删** |
| — | recipe3_nothink_1ep | f0e372bd | 3.1G(含dup) | 已证伪,留档 |
| — | recipe4_kexi_repro | 6ae212ec | 6.1G(含dup+optim) | 克西复刻,训完未传 |
| — | recipe5_seed_world8 | 0d0011a5 | 6.1G(含dup+optim) | 欠训不传 |
| — | recipe6_rebal_world | aefee748 | 6.1G(含dup+optim) | 体检复读70%不传 |
| v2 | run_a_r2 | f8d56eba | 3.1G(含dup) | 线上0.8092 |
| v3 | run_c_material | f79a4e5a | 3.1G(含dup) | 线上0.8198 |
| (D) | run_d_r2material | 49273660 | 3.1G(含dup) | 未上传 |
| — | stage2_gold_v1_lora_ep1 | 5a94d375(adapter)/b46c62fe(merged) | adapter目录340M(含末轮checkpoint)/merged1.5G | 07-11本地双门禁否决,未上传,不续训/不warm start |

各版本 proxy/线上评测分数记录见 `docs/experiment_log.md`,不受本次清理影响(分数已固化在文档里,不依赖 checkpoint 文件本身)。

## 清理记录(2026-07-03)

| 项 | 释放空间 | 说明 |
|---|---|---|
| 所有checkpoint的optimizer/scheduler/rng_state | ~14G | 推理不需要,续训需要则重新指向对应ckpt目录(已不可能,已删) |
| 中间step checkpoint(仅留最终ckpt) | ~12G | 本地proxy已记录数字,未上传评测,不影响可复现性 |
| `data/extracted/`(与顶层jsonl逐字节重复) | 190M | diff确认完全一致 |
| `data/index/`(Pid2*合并索引) | 6.6G | 派生数据,可从hf_full重新生成 |
| 5个评测日志哈希文件名重复副本 | ~15M | evalTaskId核对与规范命名版一致 |
| **合计释放** | **约33G** | |

**未清理、留待后续决定**:`/root/baseline_repro/` 下的 `r2_smoke*.jsonl`(88M,早期调试文件,似已无用)、`hf_samples/`(6.3M,HF数据的采样探查文件)。体量小,不紧急。
