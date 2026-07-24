# Current Work

> 只记录尚未完成的动作。旧 TODO 已归档到 `docs/archive/TODO_pre_cleanup_20260711.md`。
>
> 变更记录（2026-07-24 UTC）：I-40 step1030正式评测`SUCCEEDED`但仅`0.9890615139753605`，相对I-35 step548回退`0.0453670709`，主要来自video/product/ad推荐合计下降`0.0452`；I-40关闭，不再上传其它保存点、续训或扫参数。完整训练数据release继续保留，官方结果已回填。
>
> 变更记录（2026-07-23 UTC）：I-40已从原始I-35 r112全新恢复为GPU1独立后台run `34k0sdcj`，detached PID/SID `2665551`、PPID1，不依赖交互终端；全量data/W&B身份、reference/policy/optimizer与step0 logits差0门已再次通过。旧run `9dp9wnbo`本地到step169后外部终止，W&B `crashed`，无错误栈/OOM/NaN且未生成checkpoint，已归档并禁止resume。
>
> 变更记录（2026-07-23 UTC）：I-40有效run已在GPU1单卡启动，W&B `9dp9wnbo`。真实门确认r112/392 tensors/70,647,808参数、reference逐tensor精确快照后差0、reference不进optimizer、step0 logits差0；25步loss `0.2934384`，无OOM/NaN/路由异常，继续运行。第一次精度路径不一致尝试在optimizer step0前安全停止，无W&B/checkpoint且不resume。
>
> 变更记录（2026-07-23 UTC）：用户授权启动I-40直接r112续训。正式8,240行混合已冻结为I-36审计用户5,500 + I-35原正式2,740；后者统一改作当前I-35 step548 KL-only回放，不复用旧boundary/preserve目标。数据、sidecar、audit和trainer全量静态预检已通过；唯一当前动作是完成真实模型门禁后在一张空闲GPU上启动W&B online训练。
>
> 变更记录（2026-07-23 UTC）：I-39队友full v4已在GPU3单卡完整结束，报告SHA256 `252736fd...cd12`。相对同协议I-35，mat fresh=`60/542`对`55/542`，rec总命中=`36/4000`对`35/4000`，action/topic/world=`0.3018/0.0325/0.424`对`0.3050/0.0275/0.422`；物料正向、用户基本持平、动作轻微负向且无结构性崩坏。官方material从`0.2452961672473868`到`0.27595818815331014`只差1道有效命中（`8→9/574`），但I-37离线多2/542时官方仍是8/574，因此不能声称I-39预计达到0.276。仅凭teacher-forced门否决上传已不充分，当前新增待办是只提交I-39一次作正式探索验证，结果前I-35仍为默认交付。
>
> 变更记录（2026-07-23 UTC）：I-39唯一一发已完成GPU3单卡/W&B训练、fresh r8验收、唯一r120精确组合和313行冻结门。run `51yko99h`服务端finished，640/640步、runtime `1435.1195s`、train loss `1.4501802083`，路由/目标计数全部精确；r120包392 tensors、302,830,492 bytes并通过逐tensor切片恒等式。门上A/B/C首分歧方向全部改善，但full-anchor三项和aggregate KL保护失败，`teacher_forced_pass=false`；本分支关闭，不上传、不续训或扫scale。
>
> 变更记录（2026-07-23 UTC）：I-39 AB/首分歧构造池、I-35 step548单父Beam64账本、2,560行正式混合、全量sidecar和313行冻结门已构建并登记。独立审计与trainer全量预检通过：正式物料/关联用户/I-12保持=`512/128/1,920`，480个训练AB含32个双C首错组，训练与门的prompt/SID/AB泄漏均0，最长9,431 token、640步。当前唯一待办是GPU3单卡/W&B启动一次fresh r8完整训练；尚未启动，不声称提分。
>
> 变更记录（2026-07-23 UTC）：I-38M唯一full候选已完成GPU0单卡/W&B 685步、r80精确组合、冻结门和队友离线行为诊断。W&B `f92senkn` finished，runtime `1282.5365s`、train loss `0.6860710371`；冻结teacher-forced门失败（material desc2sid/sid2desc Top1=`0.9594/0.9722`，rec video/ad保护失败），因此不跑itemic、不上传、不续训或扫scale，I-35 step548继续默认交付。
>
> 变更记录（2026-07-23 UTC）：I-37 full r120官方线上评测`SUCCEEDED`，evalTaskId `eval-task-0yco4c-1784766273`，总分`1.02762520217381`，相对I-35 step548为`-0.0068033827`。用户合计`+0.0013966173`，但推荐合计`-0.0082`；I-37已关闭，不再等待上传、不追加checkpoint/scale，I-35 step548继续作为当前最高模型。
>
> 变更记录（2026-07-22 UTC）：I-37已用队友 `offline_eval.py` v4 完成唯一全量离线回归（GPU0，runtime `1722.1s`），报告SHA256=`4698d1ae...48ca`。结果：mat fresh/train=`0.1052/0.1567`、rec=`0.006/0.003/0.002/0.028`、action F1=`0.3030`、topic=`0.0244`、world=`0.432`；只作行为回归，不能换算线上总分，下一步仍是手工上传并回填官方结果。
> 变更记录（2026-07-22 UTC）：I-37已完成唯一单卡/W&B训练与full r120精确打包。run `c2crod0w`完成512/512步，最终包`submissions/i37_i35_strict_future_rec_r120_v1_platform/`为302,829,416 bytes、r120/alpha120、392 tensors；尚未上传或评测，不声称涨分。首轮step256保存校验器错误已隔离，禁止resume。
> 变更记录（2026-07-22 UTC）：停止I-35追加扫点，转入唯一新候选I-37。2,048行严格未来video/ad与I-35父KL混合已登记并通过tokenizer预检；配方锁定I-35 step548 r112 + fresh r8、单GPU/W&B、512步，只交付full r120一个包。
>
> 变更记录（2026-07-22 UTC）：I-36 step4125线上评测失败，总分`0.9865`，用户合计约`-0.01842`、推荐合计`-0.0302`；step2063不再默认上传，I-36分支关闭。后续若重启，先修复CE/KL梯度标定并降低残差容量。
>
> 变更记录（2026-07-22 UTC）：I-36有效W&B run `mmenbci2`已完成4,125/4,125步，runtime `7921.5533s`、train loss `1.0296107737`、路由retention/action/topic=`11000/4000/1500`；step2063/full已各自精确拼成合法r128并通过两文件、rank、大小和逐tensor加和验收。第一次run `onqds9a5`在step18被外部SIGKILL且无checkpoint，禁止resume。step4125已线上否决，step2063仅保留审计，I-36不再等待上传。
>
> 变更记录（2026-07-22 UTC）：I-35 step548已正式评测成功并以`1.0344285849069457`成为新高；evalTaskId `eval-task-9nepj1-1784698215`，八项=`0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591`。实际首投为548，step411尚未评测；两点限制下只剩step411，step137/274/685继续禁投。
>
> 变更记录（2026-07-22 UTC）：I-35 step411/step548离线成对对照已完成并写入[`docs/I35_STEP411_DECISION.md`](I35_STEP411_DECISION.md)。两包参数余弦相似度`0.9999952352`；step411不具备确定性上分证据，规划分数约`1.033--1.034`，只建议作为一次受控线上对照。
> 变更记录（2026-07-22 UTC）：I-35五个r112均已构建并验收，但用户将线上测试限制为两个；实际已先上传并评测step548，当前只剩step411。step137/274/685仅作备份，不占用额度。
>
> 变更记录（2026-07-21 UTC）：最高优先级I-34已完成固定beam64准入并在训练前停止。train/gate的I-23-only full-gold gap仅`7/1`，对照门槛`128/32`；四域最低均0。未生成正式训练数据/sidecar、未启动W&B、未产出checkpoint/r112、未消耗线上提交。原因：冻结机制缺少足够且跨域的teacher优势样本，继续训练只能靠结果后放宽规则。
>
> 变更记录（2026-07-20 UTC）：因报告版 I19-world r96 工件仍缺失，完成本地可复现候选 `i19_local_world_residual_retkl_r16_ep1_s800`：以仓内 s800 r80 为 parent，使用已登记的3,146行1:1 world/retention 数据训练 fresh r16。第二次单卡W&B run `xdzb35cp` 已通过787步/3146路由精确契约并生成r96 scale0.875严格两文件包；首次路由前缀误判已收档，不能resume。该候选不冒充报告中的1.0253模型，线上分数仍待平台评测。
>
> 变更记录（2026-07-20 UTC）：登记 `I19-world-residual` 的 r96 `scale=0.875` 为当前最高单次线上观测`1.025259456`，并把P0切换为接收/验收其严格包和完整复现链。已准备但未训练的`s800_native_general_replay_r8_v1`与旧`s800_world_ceval_r8_v1`暂停启动。原因：新结果已覆盖“在旧s800上再试world residual”的优先级，但当前本卷仍缺模型、数据/源码和W&B证据，不能把报告结果误写成本地已复现。
>
> 变更记录（2026-07-18 19:48 UTC）：在任何正式训练前冻结checkpoint门`970d169d...c416`并构造256条prompt-task-disjoint八任务保持集`3206e91a...c30f`。候选严格按32→64→96→129取最早同时通过23条General E目标门、八任务KL/argmax保持门和60条itemic零断裂的点；四点全失败即关闭，不允许scale、续训或放宽门。平台入口同步哈希锁这些gate资产，当前SHA256 `30531cfd...d47`。原因：防止训练后看曲线再造选点规则或把任何残差checkpoint直接拼接上传。
>
> 变更记录（2026-07-18 19:40 UTC；19:48 UTC复核）：用户批准继续后，完成`s800_native_general_replay_r8_v1`正式混合、精确token路由、r8残差trainer、配置、dataset registry和GLM Training Task容器入口。正式D为129条reviewed General各一次+8个父任务各48条KL-only保持，共513行；qwen3_nothink逐行复核129/384路由一致、最长8871 token、截断0，加入保持门后62项测试通过。开发机未启动训练；当前SSH没有`WANDB_API_KEY`，这不代表平台secret失效，但Training Task必须显式配置并由入口在线verify后才会训练。原因：把官方原生SFT转成最小单变量实验，同时用exact target-token manifest阻断文本启发式误路由。
>
> 变更记录（2026-07-18 17:36 UTC）：转向官方原生SFT实现，不再要求General必须是A-D。按官方`convertv2.py`保留原生CoT并补`/think`，全量O2.General机械筛得270条；任务适配复核再拒绝141条企业战略、虚构、岗位、保健和操作任务，形成129条reviewed静态知识监督，SHA256 `867f1093...16237`。E/当前父训练exact+near门、2048-token门及47项测试均通过；source assistant明确只是官方SFT监督，不声称独立事实gold。尚未创建正式混合、训练配置或Training Task。原因：修复旧`world_zh`只按中文比例洗数、无mode suffix且混入大量非知识任务的问题，同时避免大剂量General挤占当前1.0048主模型的其他七项。
>
> 变更记录（2026-07-18 16:22 UTC）：修复官方General评测边界并完成O5 Infinity非数学格式恢复。双侧反查发现25条永久E中2条已存在当前父训练，二者继续永久E但退出评测/checkpoint选择；68条D仍为历史E/父训练零命中且四字段投影哈希不变。s800干净基线更正为5/23=`0.21739130`。116条O5 Infinity纯格式近拒绝严格过滤后仅9条prompt-only候选，低于48条最小门，关闭该来源且不做答案/干扰项合成。原因：原7/25包含父训练重复，不能作为训练决策基线；9条也不能解决世界知识广度。
>
> 变更记录（2026-07-18 15:09 UTC）：完成官方General英文A-D扩展、答案隔离翻译、双路中文盲解与最终反查。97条reviewed中永久隔离2条CMMLU-test E及2条当前父训练重复，修正过滤后cohort分层并保留已暴露的24条E，最终形成68条unique-only D+25条永久E。历史E/父训练exact+MC-near最终命中均0；s800基线7/25，math仅2/17。训练侧为49 math+1 other+18 legacy-unclassified，因此它只能视为数学/逻辑补丁，不能声称已解决广义世界知识。尚未创建混合、训练配置或Training Task；下一步先从既有O5 Infinity近拒绝账本定向恢复非数学原生四选一，不扫O4、不造干扰项。原因：增加数据量不能代替新信息、学科覆盖与可证伪的保持门。
>
> 变更记录（2026-07-18 10:48 UTC）：完成O2.General+O5严格世界题扫描与人工复核。270.8万行只形成5条严格MC、26条开放QA；再复核118条答案格式型近拒绝后，最终仅29条MC通过，开放QA首轮training eligible=0。29<48最小切分门，未创建训练投影、配置或任务；当前不启动world训练，P0仍为从当前主模型验证下一条全局单变量路线。原因：拒绝用重复、长QA、法律/医疗/时效题或错误标签凑量，避免重演181 unique×7的过拟合与保持门失败。
>
> 变更记录（2026-07-18 07:52 UTC）：回填`s53125=0.9757`并关闭整个I-23跨父残差scale轴；完成I-29三格GPU生成与正式评分，主校准方向失败，关闭128组扩展和video训练。原因：两个冻结分支均已到停止叶，当前P0改为从s800全局重排下一单变量实验，未经用户确认不启动新训练。
>
> 变更记录（2026-07-17 17:08 UTC）：完成`s53125` checkpoint、组合审计和严格两文件包，任务从“构建”切换为“用户手工提交并回填结果”。原因：用户已明确索要SCP交付，且该点是冻结停止树允许的唯一末次scale。
>
> 变更记录（2026-07-17 16:39 UTC）：完成`s5625=0.9925`线上结果回填，关闭`[0.5625,0.625]`并将唯一剩余scale动作锁为一次`s53125`条件二分；当前只建议、未构建、未提交。原因：严格执行结果前“material掉档且video充分恢复→向下二分一次”的停止树。
>
> 变更记录（2026-07-17 14:16 UTC；14:27 UTC状态更新）：完成`s5625`精确拼接与严格两文件打包，线上上传改为用户手工待办；同时完成I-29 renderer四格校准的预注册、无gold生成器、CPU scorer及旧格复现。目标环境现已稳定并复核版本，但短暂空闲的0–3号卡被既有任务重新占用，八卡均无足够安全显存，故GPU尚未启动。原因：把已完成的交付与仍需GPU/线上动作拆开，避免重复构建或误报实验已经运行。
>
> 变更记录（2026-07-17 13:30 UTC）：回填`s800`同包复测`1.0048`与`i23_userres_r80_s500=0.9882`，完成s500任务并撤销s750优先位；新增`s5625`条件二分和canonical renderer/video定向训练的下一步。同步把参数拼接口径改为“不鼓励而非禁止”。原因：以完整线上证据重新排序提交额度和研发路线。

## P0

- [x] I-40数据契约与静态预检：I-36中全部5,500条审计用户行加I-35原2,740条形成8,240次暴露；user:retention=`5,500:2,740`，两路exact/mode交集0，T/E=0。保留上游25个重复world行并按多重集路由，data/sidecar/audit SHA256=`483a4bb2...0c18`/`e9bc129c...cb2`/`c5c2323b...5b7c`。
- [x] I-40唯一单卡训练完成：W&B [`34k0sdcj`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/34k0sdcj)完成2,060/2,060步、runtime `5135.3893s`、train loss `0.3024507062`；四个r112保存点均为392 tensors且无OOM/NaN/traceback，旧run `9dp9wnbo`仍禁止resume。
- [x] I-40完整数据与实验设置发布：formal data/sidecar均为完整8,240行确定性gzip，manifest、数据审计、v4摘要、恢复器、builder、trainer、单卡W&B配置与launcher已推GitHub；release为`assets/derived/releases/i40_i35_direct_user_continue_r112_v1/`。
- [x] I-40 step1030官方结果与停止：evalTaskId `eval-task-bwvd45-1784866180`，总分`0.9890615139753605`，相对I-35 `-0.0453670709`；用户合计`+0.0002046763`、推荐合计`-0.0452`。I-40关闭，515/1545/2060不上传，不续训或追加rank/LR/seed搜索；I-35继续默认交付。
- [x] I-39 AB/首分歧数据与冻结门：O3/I-36相交且O1/E/I-35训练排除后的3,072行池完成单父Beam64；正式2,560行与全量sidecar已登记，物料/关联用户/I-12保持=`512/128/1,920`，独立审计和trainer预检通过。
- [x] I-39唯一训练：固定I-35 step548 r112起点，GPU3单卡/W&B online，fresh r8、batch1xacc4、lr5e-6、640/640步、无中间checkpoint；run [`51yko99h`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/51yko99h) finished，runtime `1435.1195s`、train loss `1.4501802083`，三路与四目标计数均精确，残差392 tensors全有限。
- [x] I-39唯一产物与停止：full fresh r8与固定r112按1.0/1.0精确拼成唯一r120包`submissions/i39_i35_userab_firstdiv_retkl_r120_step640_platform/`，严格两文件/392 tensors/302,830,492 bytes并通过逐tensor恒等式。313行冻结门报告SHA256 `25b69458...72ca`：A/B/C首分歧均改善，但full-anchor Top1/KL/gold与全体KL失败；`teacher_forced_pass=false`，不上传、不续训、不扫scale，本门不换算线上分。
- [x] I-39队友full v4行为评测：GPU3单卡、seed42、mat/rec/action/topic/world完整结束；报告/控制台SHA256=`252736fd...cd12`/`29f92f15...f16f`。圈外mat fresh相对I-35多`5/542`，rec合计多`1/4000`，action轻微负向，topic/world无崩坏；该协议`NOT_CERTIFIED`，不换算线上分。
- [ ] I-39只占一次官方探索额度：提交`submissions/i39_i35_userab_firstdiv_retkl_r120_step640_platform/`，描述明确为I-35 r112 + fresh r8 AB/首分歧/关联用户微剂量残差。只验收这一个r120；正式结果前不替换I-35，不追加训练、checkpoint、seed或scale。
- [x] I-38M单GPU/W&B正式训练、唯一full门与r80打包：W&B [`f92senkn`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/f92senkn)完成685/685步（runtime `1282.5365s`、train loss `0.6860710371`、路由`1370/1370`），r80精确组合包已通过两文件/392 tensors/逐tensor恒等式审计；冻结teacher-forced门失败，按预注册不跑itemic、不上传、不续训或扫scale，包仅留审计。
- [x] I-38M队友离线行为诊断：`logs/offline_eval/i38_i23_material_i35_teacher_retkl_r80_step685_matrec32.json`（SHA256 `da2a5b96...c6671`）完成；mat fresh/train pass@64=`0.0938/0.1719`，rec video/prod/ad/live=`0/0/0/0.0312`。这是v4小样本回归，不是线上分数估计，不改变冻结门结论。
- [x] I-37数据与训练前门：O2严格未来video/ad各512、I-12 KL-only保持1,024，SHA256 `2f663a7e...48b4`；future/retention=`1024/1024`，最长8,228 token、截断0，trainer self-test与正式data preflight通过。
- [x] I-37单GPU/W&B正式训练与唯一打包：W&B [`c2crod0w`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/c2crod0w)完成512/512步、runtime `1178s`、train loss `1.2019`；full fresh-r8与固定I-35 step548 r112精确拼成唯一r120两文件包`submissions/i37_i35_strict_future_rec_r120_v1_platform/`（302,829,416 bytes、392 tensors、adapter SHA256 `e91c773c...675252`）。step256不评测/上传，首轮保存guard失败目录禁止resume。
- [x] I-37队友v4离线回归：唯一full r120在GPU0完成`mat,rec,action,topic,world`全量评测，报告`logs/offline_eval/i37_i35_strict_future_rec_r120_teammate_v4.json`；mat=`0.1052/0.1567`、rec video/prod/ad/live=`0.006/0.003/0.002/0.028`、action F1=`0.3030`、topic=`0.0244`、world=`0.432`。仅作为v4行为诊断，不把绝对值换算成线上总分。
- [x] I-37官方线上结果与停止：总分`1.02762520217381`，八项=`0.2452961672/0.1204379107/0.0394833175/0.0768/0.1292/0.1484/0.1089/0.1591078067`；相对I-35 step548，action/topic/ad/live改善而video/product回退，净`-0.0068033827`。分支关闭，full包仅留审计，不追加同路线评测。
- [x] I-36正式数据构造与登记：15,023条原始懂用户生成数据完成错域、重复SID、时序和prompt示例清洗，产出4,000 action+1,500 topic，并加入11,000条I-35父模型KL保持；正式16,500行、SHA256 `2720746a...692d`，user:retention严格1:2，T/E=0。
- [x] I-36单GPU/W&B正式训练与双点打包：有效run [`mmenbci2`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/mmenbci2)完成4,125步/1 epoch，runtime `7921.5533s`、train loss `1.0296107737`、路由retention/action/topic=`11000/4000/1500`。step2063/full两个r128包均为392 tensors、严格两文件、323,015,596 bytes且逐tensor精确相加；不新增第三个候选或scale。
- [x] I-36 step4125线上回填：任务`i36_i35_user_expand_retkl_r128_step4125_V1_eval_20260722183137`完成，耗时`1h5m22s`，总分`0.9865`，八项=`0.2453/0.1070/0.0331/0.0672/0.1292/0.1414/0.1035/0.1599`；相对I-35父模型显著回退，分支关闭。
- [x] I-36 step2063停止线上提交：该残差与失败step4125方向余弦约0.9869、范数仅小约9.2%，不能把它当作独立安全候选；保留文件与审计，不占用共享额度。
- [x] I-35第一优先实验的数据、训练与打包完成：W&B [`0b4p3siy`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/0b4p3siy) finished，685/685步；五个r112/alpha112包均为392 tensors、严格两文件、282,645,380 bytes，且逐张量精确等于验收r96+对应r16。
- [x] I-35 step548线上回填：`1.0344285849069457`，八项=`0.2453/0.1198/0.0388/0.0864/0.1394/0.1386/0.1071/0.1591`，evalTaskId `eval-task-9nepj1-1784698215`，当前新高；确认material未跳档、主要增益来自video/product。
- [x] I-35追加点暂停：用户本轮明确要求停止逐点评测并产出新提点adapter；step411保留文件但不占当前离线/线上资源，step137/274/685继续禁投。
- [x] I-34 beam-aware material准入完成并否决：在结果前冻结的1,024行train/256行gate上，用同一O6 vLLM0.12.0进程对r96与I-23跑BF16、seed42、原生no-think、固定domain、无约束beam64x3。train r96/I-23命中=`147/148`，teacher-only gap 7、四域=`2/0/3/2`；gate命中=`37/37`，teacher-only gap 1、四域=`1/0/0/0`。账本SHA256=`364cd069...eb38`/`2242b179...a2f`，审计`bd3afac4...8c6`。按预注册不建正式混合、不训练、不打包、不上传，也不后验降门槛或扫参数。
- [x] I-33正式训练与冻结门完成并本地否决：单GPU W&B [`io58fx1s`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/io58fx1s)正常完成512/512步，runtime927.62s、train loss2.278109、路由retention/material精确`1536/512`。step64/128/256/512精确拼成r104后跑736行冻结门；desc2sid gold均值全部为负、最高改善率step512仅53.125%，sid2desc和多项保持Top-1亦未过0.99，world均11/16不退。报告`logs/probe/i33_r96_material_desc2sid_gate_v1.json` SHA256 `91d4ab80...ef3e`；`earliest_teacher_forced_pass=null`，不跑itemic、不打包、不上传、不续训或扫scale。
- [x] I-30 v1完成并本地否决：单GPU W&B `ir2r0nd4`完成512/512步，train loss `2.1403`，路由material/retention精确`512/1536`；step128/256/384/512均完成r104精确拼接与704行冻结评测。四点material两向联合门均失败，且多项保持Top-1低于0.99；`earliest_teacher_forced_pass=null`。报告`logs/probe/i30_r96_material_teacher_gate_v1.json` SHA256 `3e12f088...92bf`；不跑itemic、不打包、不上传、不续训失败点。
- [x] I-31零训练精确插值开发探针完成：结果前冻结`lambda=0.05/0.10/0.20/0.40`。`lambda=0.10`两向material门均过，但world精确题49→48且保持权衡不优于I-30；其它点未同时通过两向material。报告`logs/probe/i31_r96_i23_exact_interpolation_dev_probe.json` SHA256 `be917998...6db0`；按advance rule不建新验收集、不打包、不上传、不后验补点。
- [x] I-32正式训练与冻结门已完成：单GPU W&B [`2lu9hw9k`](https://wandb.ai/thaongocnguyendo0-/llmrec-2026/runs/2lu9hw9k)正常完成512/512步，runtime1,074.35s、train loss`0.0819712`、路由material/retention=`512/1536`。step64/128/256/512的r168候选完成656行新门；world均11/16不退，但material双向无同时通过点，且多任务保持Top-1低于0.99。报告`logs/probe/i32_task_restore_gate_v1.json` SHA256 `c2912079...c4cd`；`earliest_teacher_forced_pass=null`，不跑itemic、不打包、不上传、不续训或扫scale。
- [ ] 用户已于门禁完成后明确授权I-32 step128占用一次每日5次额度作探索提交：FP32 r168因423,941,100 bytes超过400MB被拒，BF16 r168虽降到211,997,868 bytes又因`r=168`超过平台`1~128`上限被拒，均无evalTaskId。唯一待传包现为`submissions/i32_task_restore_retkl_r128_step128_svd_bf16_platform/`：逐模块截断SVD、r128/alpha128、392个BF16 tensors，adapter/config SHA256=`edadde1f...d46`/`daa3106d...7f3`，总计161,535,020 bytes。656行material两向gold均值`+0.001988/+0.000350`、world 11/16，itemic `0/60`、选择题格式`8/8`、action复读诊断`2/30`。下一步仅为手工上传该r128目录并回填结果；禁止重传任何r168目录或在出分前追加step64/256/512。
- [x] 完成 `I19-world-residual` 结果登记和命名消歧：实际parent为独立复现I-13-like r80（线上`0.986703844`），fresh world/retention r16按`scale=0.875`拼为r96后单次`1.025259456`；四个scale及StreamLake evalTask/model ID已写入[`I19_WORLD_RESIDUAL_HANDOFF.md`](I19_WORLD_RESIDUAL_HANDOFF.md)。仓内原I-19继续专指DPO实验。
- [ ] I-35父模型/上一最高I19-world的复现链接收：0.875 r96严格两文件包已到卷，adapter SHA256 `4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e`精确命中，config SHA256 `78b62143...1b64f`，r96/alpha96、392 tensors、包内仅adapter/config；仍需实际r80 parent（期望`a63a45c3...15ed0`）、r16 residual（期望`144ee8ef...d4d6d`）、combine audit与各自config，才能验证父身份和逐tensor拼接恒等式并关闭此项。
- [ ] 接收完整复现源：两个发布数据目录、3,146行训练混合（期望SHA256 `a8af6884...edb86`）、builder/trainer/delta-audit、训练配置和日志；补齐W&B run ID并核验787/787 steps、路由计数和超参。任一身份不一致时保持`AWAITING_LOCAL_ARTIFACTS`，不得从失败/未知checkpoint继续。
- [x] 完成本地 I19-style 候选训练与结构打包：`configs/active/i19_local_world_residual_retkl_r16_ep1_s800.yaml`，单卡GPU0、W&B online run `xdzb35cp`，s800 r80 parent、fresh r16、3,146行/787步；第二次训练前通过数据SHA256 `ef64cb72...b7484b`、trainer self-test，运行中step-0 parent fingerprint PASS，最终world/retention=1573/1573、train_loss=0.5331。r96 scale0.875合并与严格两文件包结构门通过；本地候选待线上评测，不自动声称涨分。
- [x] 完成官方O2.General原生静态知识清洗与task-fit复核：机械池270条只作候选；正式reviewed投影129条，历史/地理/科学/计算/日常=`7/33/34/6/49`，全部按官方原生`/think`路由保留CoT，训练数据SHA256 `867f1093...16237`、lineage `adf174f3...690e4`、audit `f3348b0b...ebf6`。它是官方SFT监督而非独立事实gold；未加入任何混合、配置或训练任务。
- [x] 用户已批准并冻结**小剂量原生General replay**：唯一父/reference为`s800=1.0048`；正式数据`data_s800_native_general_replay_v1`为129 General CE+384分层KL-only保持，SHA256 `87097135...fddd2`。fresh r8/alpha8、129步，trainer/config/入口/checkpoint门均已冻结；但I19-world新结果已覆盖其当前优先级，现保留资产和预注册证据、暂停启动，不把r8/r88设计冒充r16/r96最高方案的复现。
- [x] 暂停在GLM UI创建`s800_native_general_replay_r8_v1` Training Task；I19-world已成为I-35父模型且I-35 step548刷新新高，该旧s800分支不再按原P0启动。若未来重新授权，仍必须沿用已冻结入口、W&B在线verify及32→64→96→129最早全通过门，不允许后验改规则。
- [x] 完成官方General世界题首轮可训练发布门及评测修复：97 reviewed先硬隔离2条CMMLU-test E与2条父训练重复，93条进入永久split pool；最终68 train/25永久E。新增holdout-vs-parent反查后发现2条已暴露E也是当前父训练重复，永久留E但退出计分，故有效评分E=23、干净D或scoring总数91。68条D历史E/父训练exact+MC-near均0，投影SHA256 `f8cccd1f...2e79f`；E SHA256 `fb67b76d...e13df`，split audit `94cb6bd7...663cd`。s800干净基线5/23，math 2/17；训练仍为49 math/1 other/18 legacy-unclassified，不能代表广义世界知识。尚未创建正式混合、配置或Training Task。
- [x] 完成既有`world_clean_near_rejections.jsonl`中的O5 `Infinity_Instruct`非数学原生四选一恢复：116条纯格式拒绝经严格四选一、非数学、风险、E/父训练/既有复核过滤后仅9条prompt-only候选，answer/gold/assistant字段物理隔离，training eligible=0。因低于48条最小复核/切分门，关闭该来源，不做双路答案盲解、不合成选项凑数。
- [x] 完成I-28 `i28_i23_rec_multigold_proposal_retkl_v1`单卡正式训练与冻结主门：W&B `t3xega98`正常完成128/128步，512个microbatch严格路由proposal/retention=`128/384`，step64/128 adapter完整且root与step128逐字节一致。128组/539 gold上step64 set-logsumexp均值`+0.00220`、改善61/128，step128均值`+0.01413`、改善69/128；均低于冻结的改善率`>=0.55`（至少71/128）。按预注册停止，不运行后续保持门或N4×K8，不打包、不上传、不作RFT/GRPO父模型、不消耗线上配额；报告`logs/probe/i28_multigold_set_path_v1.json` SHA256 `96bd4577...1dee`。
- [x] 完成I-13 scale0.80固定协议线上探针及同包复测：首测`1.0037`，八项=`0.2453/0.1163/0.0401/0.0960/0.1224/0.1372/0.1089/0.1375`；同一包复测`1.0048`，八项=`0.2453/0.1166/0.0401/0.0960/0.1224/0.1358/0.1089/0.1398`。两次差`+0.0011`按合理评测抖动处理，当前最好显示分为`1.0048`。首测日志action1024、itemic 7次race-average、8/8完成、失败0，evalTaskId `eval-task-olteal-1784217662`；复测UI为`e3_userres_r80_retkl_v3_s800_V1_eval_20260717000053_copy`，原始日志/evalTaskId尚未入仓。
- [x] 将I-13 scale0.75 `e3_userres_r80_retkl_v3_s750`降级为已就绪备包，不再占下一优先提交位：其三点曲率中心约`1.0048`已经落入s800同包复测范围，缺少明确上行。严格两文件包及SHA256=`5aa80992...6233`/`e3c3ace0...c4ac0`保持不变，除非出现新证据不消耗配额。
- [x] 完成I-23 E3固定协议评测：线上0.9915，八项=`0.2760/0.1099/0.0383/0.0576/0.1258/0.1400/0.1053/0.1387`；action1024、itemic 7次race-average、8/8完成、失败0。它未替换当时s875 0.9978，当前低s800首测0.0122、低同包最好显示分0.0133，但仍是固定协议最高无参数拼接单adapter；用户已批准只将成功的I-23 E3作为新action-answer-token CE + 冻结I-23 KL保持实验的父模型，E1/E2仍只作剂量轨迹。
- [x] 完成score-first首个线上探针`I-23 + 0.625×I-12用户残差`：固定协议线上`0.9866`，八项=`0.2453/0.1170/0.0399/0.0768/0.1326/0.1316/0.1044/0.1390`。相对同日I-23复测`0.9884`，material `-0.0307`，其余七项合计`+0.0289`，净`-0.0018`；未超过I-13。原始平台日志尚未入仓，当前只登记用户提供面板结果，不补造evalTaskId。
- [x] 完成`I-23 + 0.5×I-12用户残差`线上评测：`i23_userres_r80_s500=0.9882`，八项=`0.2760/0.1156/0.0399/0.0576/0.1258/0.1316/0.1035/0.1383`；UI `i23_userres_r80_s500_V1_eval_20260716190207`，平台时间`2026-07-16 19:02:20`，耗时1h7m20s。相对同日I-23复测面板总分`-0.0002`，material保持、video不变；与s625一起将material/video两个离散阈值夹在`[0.5,0.625]`。当前只有用户面板，原始日志/evalTaskId尚未入仓，不重复提交。
- [x] 完成`i23_userres_r80_s5625`构建、严格两文件提交和线上评测：总分`0.9925`，八项=`0.2453/0.1160/0.0392/0.0864/0.1224/0.1372/0.1062/0.1398`；UI `i23_userres_r80_s5625_V1_eval_20260717220142`，平台时间`2026-07-17 22:01:46`，耗时1h5m41s。相对s500显示总分`+0.0043`但material `-0.0307`、video `+0.0288`；相对s800复测仍`-0.0123`，不替换主模型。原始日志/evalTaskId尚未入仓。
- [x] 完成末次预登记点`i23_userres_r80_s53125`线上评测：UI `i23_userres_r80_s53125_V1_eval_20260718012851`，总分`0.9757`，八项=`0.2453/0.1159/0.0400/0.0864/0.1190/0.1260/0.1044/0.1387`，平台时间`2026-07-18 01:28:57`、耗时1h6m41s。相对s5625总分`-0.0168`，主要为ad `-0.0112`，且material仍为0.2453；未满足M=0.2760、video≥0.0768、总分>1.0048的冻结成功条件。I-23跨父残差scale轴关闭，不再追加任何后验scale。
- [x] 完成I-29 canonical renderer校准的CPU阶段：冻结首16、相同seed/decode的`I23/s800 × legacy/canonical`四格；旧`s800×legacy`复现512有效、LCP=`414/87/11/0`、any=`8/2/0`、stop-close=`62/64`。配置/无gold生成器/scorer SHA256=`2f8a3994...61f8`/`09607134...2090`/`66ec9829...a701`，均通过py_compile/self-test/preflight；scorer还硬验三格meta/config并在打开gold前拒绝缺失或decode/renderer漂移，未授权训练。
- [x] 完成I-29缺失三格GPU生成与CPU正式评分：首次用完整GPU UUID在vLLM 0.12子进程的`int(CUDA_VISIBLE_DEVICES)`处安全失败，未加载模型、未发布工件；同一物理GPU改用index0后成功，25%显存上限、观测峰值21,537 MiB，三格各16行原子发布且GPU回到基线。canonical主指标candidate prefix mass为s800/I23=`88/97`，方向与已知线上video关系相反；group-any-ab=`2/1`仅是secondary，exact均0。按预注册判为`COMPLETE_PROXY_CALIBRATION_FAIL_NO_TRAINING`，不做128组扩展、不训练video residual。报告`logs/probe/i29_i23_s800_renderer_calibration_n16.json` SHA256=`4fc9ca83...ce25`；成功运行日志SHA256=`ede04354...f0f3`。
- [x] O5 Infinity补广度仅9条、68条数学/逻辑补丁路线均不作为本轮首发；用户已选择覆盖更广的129条官方原生静态General最小回放。旧68条提案参数继续视为未批准且不得与本轮混入；若本轮保持门或线上失败，再决定是寻找独立可验证非数学来源还是关闭General增量轴。
- [x] 完成I-24 `i23_action_ansretkl_v1`预注册选点并关闭分支：GPU0单卡200/200正常完成，W&B [`f3ayytob`](https://wandb.ai/3120252125-/llmrec-2026/runs/f3ayytob) finished；8个剂量点无一满足action硬门。最接近step50虽action均值`+0.05031`，但改善率`15/32<18/32`、top-1 `-0.00080`、topic `-0.01223`；四域最大KL约`0.013–0.014>0.005`。按冻结规则不选点、不打包、不上传、不放宽门槛。
- [x] 完成I-25实战安全复评：原冻结协议仍为`ORIGINAL_GATE_FAIL`，线上方向仍为`ABSTAIN`。当前1024协议下step500 scale0.5/0.625/0.75的JSON为`26/28/29`、截断为`6/4/3`，scale0.75最接近I-23的`30/2`；material双向KL约0.002且推荐域无硬红线，但离线方向不能证明涨分。I-25不作为首投、不打包、不上传；I-26重训未获授权，不得自动启动。
- [x] 完成I-18 E3固定协议评测：线上0.9697，八项=`0.2453/0.1083/0.0382/0.0768/0.1190/0.1316/0.1089/0.1416`；8/8完成、失败0。低I-13 0.0281，未替换主模型；I-10 E3缺同协议桥，不能作CoT修复净因果归因。
- [x] 发布I-13源链（发布时最高分s875 `0.9978`）队友复现包：`assets/derived/releases/e3_userres_r80_retkl_v3_s875/`包含I-10父训练集与I-12残差训练集的完整确定性gzip、manifest及原始小型审计；portable configs和`scripts/reproduce/i13_highscore.sh`覆盖父训练、残差训练与0.875拼接。当前榜分主模型已替换为同源scale0.80（同包两次`1.0037/1.0048`）。
- [x] 发布I-18未评测候选数据包（非最高分实现）：完整32,644行训练输入以确定性gzip提交到`assets/derived/releases/seed_teacher_cotfix_v2/`，manifest登记上游/哈希/混合比例/不变量，恢复脚本完成压缩包与解压内容双校验；数据注册及历史配方改为仓库相对路径，训练字段不变。
- [x] 完成I-18推荐截断CoT语义修复的本地训练与硬门禁：538/538程序门与独立judge满分质检、32,644行正式数据登记及逐行不变量审计均通过；2026-07-14 17:26 UTC在GPU1用detached PID1持久会话启动，1,995/1,995正常完成，train loss1.3617671、退出码0，W&B [`32av8e8z`](https://wandb.ai/3120252125-/llmrec-2026/runs/32av8e8z)服务端`finished`。E1/E2/E3 adapter SHA256依次为`02b404bd...47a85`/`14071ab8...bdc38`/`07cd6628...2a9e3`，E3与根目录逐字节一致，全目录无optimizer/scheduler/RNG；E3 itemic断裂0/60 PASS，临时merge已删。
- [x] 完成I-16推荐偏好续训及门禁：W&B `packufor` 600/600正常退出，36m30.54s、train loss0.5818；step200/400/600均显著提高推荐排序且action不退，但gold mean-logp分别下降0.01185/0.02030/0.02149，全部超过0.01保护线。按原门槛本地否决，不打包、不上传、不作为后续父模型。
- [x] 完成I-17低剂量窗口及线上评测：W&B `pfjlvm70` 200/200正常完成；从原始I-10 E3重新开始，未使用I-16 checkpoint。按最早全通过选step100；固定协议线上0.9727，八项=`0.2453/0.1077/0.0380/0.0960/0.1156/0.1274/0.1044/0.1383`。低I-13 0.0251；相对I-12推荐合计+0.0101但用户合计-0.0142，总分-0.0041。日志8/8完成、失败0并已归档；直接父模型固定协议分缺失，不作DPO因果结论。
- [ ] 在用户确认且当日配额允许时重评I-10 E3固定协议桥：它是I-17的逐字节父adapter，也是判断step100 DPO净方向的唯一直接线上对照；桥接前不提交I-17 step150/200。若step100不高于父模型则关闭该DPO剂量线；若推荐合计明确提高且非推荐保持，再决定是否值得消耗一次step150配额。
- [ ] 完成I-15官方SFT/RL结构缺口审计：逐项映射O1-O6到R0/R1/R2/R3、itemic instruction与General，区分比赛可直接复用、需builder验证和当前不可复现三类；基于单一主要变量提出一个最小实验。朴素同容量adapter蒸馏已暂停，未形成获批配置，不启动训练。
- [x] 收档I-14 E3线上结果与比较边界：`seed_clean_r80_lr1e4_ep3_rerun1=0.9518`，八项=`0.2453/0.1045/0.0387/0.0480/0.1292/0.1414/0.1080/0.1368`。它未替换I-13榜分；I-13参数拼接按主办方口径是不鼓励而非禁止，但构造血统不同，只能作业务榜分对照。更接近的无拼接I-11为0.9618，仍因teacher、续训与rank差异不能作干净基线。I-14纯O1单体路线不作因果否决，E1/E2未线上评测。
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
