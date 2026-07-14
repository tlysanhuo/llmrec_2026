# Current Work

> 只记录尚未完成的动作。旧 TODO 已归档到 `docs/archive/TODO_pre_cleanup_20260711.md`。

## P0

- [ ] 用户醒后决定是否把I-18 E3制作成严格两文件包并占用一次线上提交：本地主候选为`seed_teacher_cotfix_v2_r64_lr1e4_ep3/checkpoint-1995`，结构门0/60 PASS；E1/E2只作轨迹，不提交。尚未打包或上传，本轮训练与门禁未消耗线上次数；提交前重新核对当日剩余额度。
- [x] 完成I-18推荐截断CoT语义修复的本地训练与硬门禁：538/538程序门与独立judge满分质检、32,644行正式数据登记及逐行不变量审计均通过；2026-07-14 17:26 UTC在GPU1用detached PID1持久会话启动，1,995/1,995正常完成，train loss1.3617671、退出码0，W&B [`32av8e8z`](https://wandb.ai/3120252125-/llmrec-2026/runs/32av8e8z)服务端`finished`。E1/E2/E3 adapter SHA256依次为`02b404bd...47a85`/`14071ab8...bdc38`/`07cd6628...2a9e3`，E3与根目录逐字节一致，全目录无optimizer/scheduler/RNG；E3 itemic断裂0/60 PASS，临时merge已删。
- [x] 完成I-16推荐偏好续训及门禁：W&B `packufor` 600/600正常退出，36m30.54s、train loss0.5818；step200/400/600均显著提高推荐排序且action不退，但gold mean-logp分别下降0.01185/0.02030/0.02149，全部超过0.01保护线。按原门槛本地否决，不打包、不上传、不作为后续父模型。
- [x] 完成I-17低剂量窗口及线上评测：W&B `pfjlvm70` 200/200正常完成；从原始I-10 E3重新开始，未使用I-16 checkpoint。按最早全通过选step100；固定协议线上0.9727，八项=`0.2453/0.1077/0.0380/0.0960/0.1156/0.1274/0.1044/0.1383`。低I-13 0.0251；相对I-12推荐合计+0.0101但用户合计-0.0142，总分-0.0041。日志8/8完成、失败0并已归档；直接父模型固定协议分缺失，不作DPO因果结论。
- [ ] 在用户确认且当日配额允许时重评I-10 E3固定协议桥：它是I-17的逐字节父adapter，也是判断step100 DPO净方向的唯一直接线上对照；桥接前不提交I-17 step150/200。若step100不高于父模型则关闭该DPO剂量线；若推荐合计明确提高且非推荐保持，再决定是否值得消耗一次step150配额。
- [ ] 完成I-15官方SFT/RL结构缺口审计：逐项映射O1-O6到R0/R1/R2/R3、itemic instruction与General，区分比赛可直接复用、需builder验证和当前不可复现三类；基于单一主要变量提出一个最小实验。朴素同容量adapter蒸馏已暂停，未形成获批配置，不启动训练。
- [x] 收档I-14 E3线上结果与比较边界：`seed_clean_r80_lr1e4_ep3_rerun1=0.9518`，八项=`0.2453/0.1045/0.0387/0.0480/0.1292/0.1414/0.1080/0.1368`。它未替换I-13的0.9978榜分，但I-13是参数拼接的融合灰区模型，只能作业务榜分对照；更接近的非融合I-11为0.9618，仍因teacher、续训与rank差异不能作干净基线。I-14纯O1单体路线不作因果否决，E1/E2未线上评测。
- [x] 完成I-14干净重跑`seed_clean_r80_lr1e4_ep3_rerun1`：2026-07-14 05:25 UTC从O6及全新输出目录在单卡`GPU-717b...98c2b`持久启动，1,971/1,971 steps、train loss1.3422074、runtime4,862.30s；W&B [`3grnqgsh`](https://wandb.ai/3120252125-/llmrec-2026/runs/3grnqgsh)服务端`finished`。E1/E2/E3 adapter SHA256依次为`f441b83f...f187d`/`182ba79b...79a1`/`477d2acd...0837b`，E3与根目录逐字节一致；全目录无optimizer/scheduler/RNG，退出码0，目标GPU归零，配置已加成功防覆盖头。
- [x] 收档I-14首次运行`seed_clean_r80_lr1e4_ep3`：前台训练绑定临时PTY，在step1,886/1,971被会话生命周期中断；W&B `s7fskx9u`服务端`crashed`，无Traceback/OOM/NaN、无E3。E1/E2只作故障证据，禁止resume、热启动、评测、打包或上传；原配置已加禁用头。
- [x] 完成I-13固定协议评测：`e3_userres_r80_retkl_v3_s875=0.9978`，八项为`0.2453/0.1183/0.0390/0.0960/0.1224/0.1316/0.1062/0.1390`。日志action上限1024、itemic 7次race-average、8/8任务、`Failed tasks 0`；同协议相对I-12总分+0.0210，用户-0.0026、推荐+0.0229、world+0.0007。日志已规范命名并登记，E3固定协议桥仍缺失。
- [x] 用户裁定本轮不消耗最后一次配额原样重评I-10 E3，改投可提分候选I-13。科学上旧/新协议sentinel桥仍缺失，继续禁止比较E3旧0.9849与I-11/I-12/I-13固定协议分数。
- [x] 完成I-13 residual-scale Pareto审计与打包：固定I-10 E3 r64，只将I-12 r16残差缩放为0.875；严格排除I-12训练集的O1六任务留出共576条。相对full residual保留action/topic CE收益93.2%/83.3%，平均保持KL下降约5.0%；itemic断裂0/60=PASS，门禁其余读数与I-12完全一致；严格两文件包SHA256 `71bc3c2c...ffd5b` / `e3c3ace0...c4ac0`，临时merge已删、GPU1归零。
- [x] 完成评测协议切点审计：I-10 E3（11:45）仍是action4096+itemic单跑；I-11（16:40）和I-12（20:16）是action1024+itemic 7次race-average。两边虽同报`v3.1`，已按本地协议标签隔离，旧校准全部失效。
- [x] 完成I-12固定协议评测：`e3_userres_r80_retkl_v3_ep1=0.9768`。同协议相对I-11总分+0.0150、用户+0.0097、推荐+0.0038、world+0.0015，ad单项-0.0098；当时为固定协议较优候选，现作为I-13直接对照；相对E3涨跌仍待桥接。
- [x] 完成I-12 v3训练、拼接与门禁：GPU1单卡1 epoch/1,527 steps/45m43.40s、train loss1.1514281；W&B `1xbo7k2e`服务端finished；残差r16 SHA256 `e8caf0a3...4f98`，与E3精确拼接r80 SHA256 `3fe85158...87cc6`。固定训练内配对审计action/topic CE均下降、保持KL均值0.00211；itemic断裂0/60=PASS；严格两文件包验收，所有GPU任务后GPU1为0 MiB。
- [x] 收档I-12 v2启动失败：W&B `fi4mneew`在step275主动停止；ChatML在EOS后监督换行，旧终止权重实际落到换行而非闭合符/EOS。无adapter/checkpoint，输出目录已删，禁止resume。v3反向跳过格式尾缀，真实action/topic样本验证闭合符与EOS为2x、尾换行为1x。
- [x] 收档I-12 v1启动失败：W&B `hkt762u2`在step8因101条world保持回答无`</think>`触发严格路由中止；无adapter/checkpoint，输出目录已删，禁止resume。v2仅修复非用户保持路由；3,053条用户行仍要求完整think闭合。
- [x] 构造并审计I-12数据：`data_user_residual_retention_v1` 6,106行，用户CE/父KL各3,053；action1,752、合法topic1,301、164 teacher一次，四域rec各565、material562、world231；3条6步topic排除，规则/T/E为0；SHA256 `bd947aad...b08f0`。CPU损失/拼接自测、2步真实模型烟测及真实模板终止权重回归均通过；每次任务后GPU1均为0 MiB。
- [x] 完成I-11线上评测：`seed_teacher_e3_cont_r64_lr2e5_ep1=0.9618`，八项为`0.2453/0.1106/0.0396/0.0672/0.1156/0.1414/0.1053/0.1368`。该日志是最早可证实的固定协议参考；与E3旧协议差值已撤销，日志8/8完成、Failed tasks 0。
- [x] 完成I-11训练与硬门禁：GPU1单卡1 epoch/665 steps、26m13.72s、train loss1.2266276；W&B `3f8tas1s`服务端`finished`；itemic断裂0/60，硬门禁PASS；adapter SHA256 `6b2e4fbd...68626`，无中间checkpoint/optimizer。action复读3/30和选择题7/8只作diagnostic，不用于估分。
- [x] 完成 I-10 同轨迹线上评测：E1/E2/E3=`0.9100/0.9680/0.9849`；三份原始日志均8/8任务、Failed tasks 0并已规范命名登记。E3在material持平E2的同时提高用户合计、推荐合计和world，确定为主模型。
- [x] 完成 I-10：从 O6 在 `data_seed_teacher_v1`（O1 32,480 + 独立judge满分teacher标签164；规则标签0）上训练连续3-epoch cosine；r64/alpha64/dropout0.05、lr1e-4，1,995 steps/1h18m31.68s、train loss 1.3583，W&B `ev401ys9` finished；最终线上主模型E3=0.9849。
- [x] 完成 I-09 标签验尸：规则标签相对同源164条独立judge满分teacher参考全量平均F1 0.0429；匹配I-09过滤条件的42条平均F1 0.0813、32条零交集。teacher不是官方gold；删除正式混合中的全部1,000条规则标签。
- [x] 完成 I-09：`data_seed_o2_action_v1` 33,644 行、O2 3.4598%；r64/alpha64、lr1.5e-4、单卡 1 epoch/710 steps/28m01s，W&B `6qrsdits`。itemic 0/60，但 action 固定题 0/5 闭合、5/5 触顶，material 39/13 未进历史 8 题签名；后验约 0.92（分析区间 0.89–0.96），**本地否决、不上传**。
- [x] 完成 I-07 O1-only score-max：35,558 行，r32/alpha32、lr1e-4、cutoff16384、单卡 1 epoch/740 steps/29m03.95s，W&B `q5uaa2fh` finished。itemic断裂0/60、world格式8/8；但action可见题0/5闭合且5/5触顶，material签名41/14未进历史8题档。后验中点约0.92，**本地否决、不上传**；adapter保留、临时merged删除。
- [x] 完成 I-06 数据：`action_distill_v5` 因两个API余额耗尽以164条唯一、独立judge满分teacher标签封板（事件号回映SID、排除354个E源索引），每条重复8次后与O1 I-01合并；最终33,792行、SHA256 `bbefa5f2...ce99d3`。
- [x] 完成 I-06 训练与同口径门禁：GPU1、r64/alpha64、lr5e-5、3 epoch、1,047 steps、1h28m19s，W&B `thbcz5k3`；结构断裂 0/60。相对 riders 历史比较对象，action 截断率 44.3%→22.2%，但 F1 0.0171→0.0160；world v4 Acc 0.380→0.206、格式存活 100%→39.6%。**本地否决，不上传**；adapter 保留，临时 merged 删除。

## P1

- [ ] 为`score-direction-v1`积累按时间前推、按实验族独立的candidate-parent线上标签；冻结阈值前目标40–60个开发标签/至少8族，最终UP和DOWN各自按一侧95%置信下界≥90%验收并报告覆盖率。

## Done

- [x] 回填 r64 轨迹线上结果与方向台账：E1 0.8839（相对 riders −0.0338），E2 0.9187（名义 +0.0010、未超过噪声）；E1 原判决保持 `ABSTAIN`，E2 同轨迹不计独立校准样本。
- [x] 完成 r64 门禁验尸：本地选 E1、拒 E2，线上却 E2−E1=+0.0348；material `51/21` 与 visible action 闭合率均不能继续作正向 checkpoint 排名器。E3 不再上传。
- [x] 完成并封板 O1–O6 官方数据 EDA：全量核对结构/重复/时序/metadata/O6 tokenizer，完成跨资产泄漏排除，纠正旧 O2 overlap 与“普遍时间泄漏”结论；最终文档为 `reference/OFFICIAL_DATA_EDA.md`，未生成训练数据。
- [x] 完成 O4/O5 5,210,887 行全量扫描：O5 严格中文 A–D 单选仅 101 条机械上限且偏医学/法律，I-03 降级；O4 不进入竞赛 SFT 队列。
- [x] 完成 shadow-gold 可行性首轮只读核查：O3 对5道可见material描述零命中；O2全50万UserProfile对一个154-SID可见rec prompt最高仅重合13，未找到同用户，不能据此恢复平台gold。
- [x] 建立90%选择性涨跌判决器、前瞻台账和精确置信下界审计；修正离线镜像解码参数并隔离v3/v4协议。当前状态`NOT_CERTIFIED`，E1为`ABSTAIN`。
- [x] 完成 `riders_fk_clean_r64_ep3` 在线训练及 E1/E2/E3 同口径门禁：出分前 E1 material `51/21`、action `5/5`、itemic `0/60`、world 格式 `8/8`；E2/E3 随 epoch 增加出现 material 锁定下降与 action 触顶，当时均未提议上传（E2 后由用户实际评测，结果见上）。
- [x] 官方、派生、第三方、评测、归档数据完成物理分区。
- [x] `OpenOneRec-General-SFT` 从第三方纠正为官方资产，并按 301 Parquet/24,685,081,929 bytes 验收。
- [x] 官方 `OpenOneRec-General-Pretrain` 下载完成，并按 310 Parquet/27,139,522,149 bytes 验收。
- [x] 2026-07-11 完成旧实验 checkpoint 清理，当时只保留 `riders_fk_lora_ep1`；之后新增的 r64 E1/E2/E3 为用户批准的临时 checkpoint-search 例外。
- [x] 建立 `ideas/`，归位选手分享、队友分享、EDA 和历史方案。
- [x] 重写根 README 与当前 artifact index。
- [x] DeepSeek 凭据移入 `configs/secrets/` 并改为 0600，脚本引用已同步。
- [x] 3 个缺失 dataset key 的历史配置已增加禁止启动标记；新配置只允许进入 `configs/active/`。
- [x] 提交包从 26 份清理到 2 份：当前最好和最新交付包。
- [x] 删除可再生的 uv/conda 安装缓存约 8.4GB；训练 venv 和推理 conda env 保留。
