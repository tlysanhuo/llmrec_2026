# TODO — 快手 LLM-Rec 2026(唯一 TODO 文档)

> 建于 2026-07-03,大修 2026-07-04 晚(会话交接)。**规则:每个会话开头必读;状态变了立刻改这里。**
> 标记:`[ ]` 待办 / `[~]` 进行中 / `[x]` 完成 / `[!]` 等用户拍板。
> **变更记录(2026-07-11 07:38 UTC)**:`stage2_gold_v1_lora_ep1` 已完成并被 action+物料双门禁否决,当前主线改为固化平台可见题提交前门禁并重新分析低扰动SFT数据方案。原因是“无物料增量集即可保持物料”已被当前剂量实验否定,不能继续按旧二阶段假设点火。
> **变更记录(2026-07-10 14:58 UTC)**:当前唯一工作改为SFT实验归因与下一版数据设计;RL/DPO封存,不训练、不上传。原因是最高分仍为0.9177,必须先从现有全部资料中找出可解释的SFT增益。
> **变更记录(2026-07-10 09:40 UTC)**:新增 LoRA/全参配对实验规则。原因是 `baseline_sft_v1` 与 `seed_raw_lora_ep1` 除训练方式外还混入 lr、weight decay、验证集切分和并行方式差异,不能用于归因 LoRA vs full。
> **变更记录(2026-07-10 09:35 UTC)**:用户裁决 `seed_raw_lora_ep1` 相对既有全参 raw 基线预期增益不足,停止该路线。已训练产物仅留档,不重训、不上传、不评测,也不再作为后续实验前置;原因是避免重复实验和浪费评测配额。
> **大局(07-08 榜单快照后)**:Top1=数据科学社 **1.0535**(单日+0.034 仍在加速);破1.0已六家;**top10门槛 0.9950**(+0.0096/天集体爬升);07-07 又两队百名级跳升(筷快块↑119→1.0048/试一下↑154→0.9976)=第三波可复制配方扩散。我们最好 `riders_fk_lora_ep1` 0.9177(±0.03噪声),距top10 0.077≈物料2题+ad1题;**ally_map_v2 预期中位 0.94-0.95 已够不到 top10**。`quality_swap_v1_lora_ep1` 线上 0.8235 已证伪“等量换官方 user 数据”;`official_rec_v3_lora_ep1` 线上 0.7948 已证伪“官方 UserProfile 推荐重构替换 riders rec 行”;`evalform_v1_lora_ep1` 线上 0.7571(07-09)已证伪“rec 行评测形态转写”。评测方差已标定:物料零噪声/rec小样本列有噪声/总分±0.03。

## P0 — 进行中/今日

- [~] **★★当前唯一主线:提交前门禁固化+低扰动SFT重新设计**:先把20份平台日志中的8任务×5可见题固化成只读回归集,再基于`stage2_gold_v1`验尸重做数据/剂量方案。可见题永不入训,只拦格式、循环、截断和物料beam签名退化;离线好转不作收益证据。
- [x] **stage2_gold_v1_lora_ep1 闭环(07-11)**:数据8030/holdout758,构建审计、训练103步、merge、独立holdout和真实action/物料门禁均完成。action 0/5 JSON、物料14/7,**不上传、不续训、不warm start**;全链见`experiment_log.md`最新行和`EXPERIMENT_INDEX.md`。
- [!] **训练/上传状态**:当前没有获批的新训练或上传。`stage2_gold_v1_lora_ep1`、`seed_raw_lora_ep1`、`riders_act_v1`、`ep5_rider_v1`和DPO路线均已封存;`riders_fk_lora_ep2`虽有历史包,但不是当前获批提交项。

- [!] **★★★情报最高优先(07-09 17:2x 定位)**:①拿到 **臭机炸炸炸《[笔记]LLM-Rec挑战赛》全文**——OneThree 评"他这个懂用户是真高"=靶二(action,+0.05)方法源头,流通配方文档实锤;②关关关 17:22 问"大家咋调的参"之后的全部回复(尤其 OneThree 的 lr);③数据科学社/筷快块/试一下 发言(旧任务保留)。
- [!] **★★★官方 SFT 对齐 Caption/Tag 数据发布(07-09 晚,全场同时拿到;已下载解剖)**:`assets/official/sft_aligned/baseline_caption_tag_lists.parquet`(730MB),19,204 行=种子 rec 行 1:1;**56.9 万唯一 SID 97.7% 有官方 caption**(中位 234 字,评测 desc 同风格带),4.1 万 SID 多视角;官方明示"探索 CoT 构造方式"。**=15 倍于种子 think 的 grounding 母矿,与 SFT 对齐无码本墙**。新头号数据牌=官方 caption grounding 行(评测风格 desc→SID,`/no_think` 空 think 形态与物料行 2,805 条空 think 同构,video_ad 码本优先,几千行级,纯加法不动旧行)。后续路线不再等待 seed_raw,必须直接证明相对已有线上最优底盘的独立增量。
- [x] **★五线 EDA 第二轮完成(07-09 晚,`ideas/archive/data_eda_round2_20260709.md`,含勘误页)**:raw 频次、action 幻觉项、历史搜索行缺口和本地探针结论均留档。`seed_raw` 的 44/17 只作为本地诊断结果,不再外推线上收益;07-10 用户裁决不上传评测。
- [!] **★靶一候选(更新):#1=官方 caption grounding 的多视角/分布式载体;#2=平台剂量倒U(等笔记/lr);#3=rank。raw 频次假说停止单独验证,多遍轴维持搁置。**
- [x] **★★riders_fk_plat_ep1 已出分(07-09 15:19,0.8229,−0.0948 vs riders)**:精度任务塌向轻触档(mat 7→**5题**/video −2/prod −8),先验型反升(ad **107**/live **124**);action 持平、时长 1h23m ⇒ focal 不治复读。日志核查 mat 段 beam 多样性与本地逐字一致 ⇒ 坏的是映射指向。**07-09 晚判决修正(OneThree 情报):不是"平台loss证伪",是 lr2e-4 在加权 loss 下过冲——平台臂改为"找内部最优剂量"重开。**日志 `logs/eval/riders_fk_plat_ep1_20260709.log`。
- [!] **riders_fk_lora_ep2 已训完打包,待用户裁决是否上传(07-08 训完;15:00 新窗口 3 发可用)**:704步/78min,eval_acc 0.6233;precheck 保险丝过;包 `submissions/riders_fk_lora_ep2_platform/`(adapter md5 `efaee6fb`)。预登记分支:A(~25%)0.94-0.97 / B(~48%)0.915-0.935 / C(~27%)0.88-0.91,物料 0.2453/0.2146/0.1840 一眼定分支。0.995 面板(mat8 为离散跳变、保先验制度内可得)+两 0.99 人证共同要素=LoRA×充分更新量 ⇒ 分支 A 概率上修。
- [x] **纯官方 raw LoRA 加码轴停止(07-10 用户裁决)**:`seed_raw_lora_ep1` 已封存,原提案 `seed_lora_ep3` 同属该轴,不再训练。理由:相对已有全参 raw 基线预期提升不足,且多 epoch 还增加过拟合风险。
- [!] **r64 轴(riders_fk_r64_ep1)保留为独立候选**:容量假说尚未验证;不再等待 `seed_lora_ep3`,但启动仍需独立收益论证和用户明示批准。风险预登记=video 掉档(rank↑=先验侵蚀)。
- [!] **★★战略转向(07-10 更新)**:纯官方 raw LoRA 的数据归零测试已完成本地训练并封存,不再训练或评测;后续只讨论相对 `riders_fk_lora_ep1` 等已有线上底盘的真正新变量。当前候选仅保留 riders 数据上的 epochs/rank/lr 和自建数据增量,每项启动前必须证明预期增益超过评测噪声与配额门槛并获用户批准。
- [x] **★evalform_v1_lora_ep1 已线上证伪(07-09,0.7571,−0.1606 vs riders)**:预登记双判据全反向——mat 没保住 7 题(0.1840,6题)、video 主判据不升反跌(8→5题);动的 rec 四域全大跌(video −3/prod −14/ad −25/live −23 题),没动的 action 0.0631/topic 0.0432/world 0.1480 全持平 ⇒ 毒性精确定位=转写本身。**"训练形态贴评测形态"假设在 0.8B×LoRA1ep 制度证伪,"换不加"路线关闭,不得沿评测形态转写加码;riders 原方言 rec 行外衣不许再动。**平台记录 `evalform_v1_lora_ep1_V1_eval_20260709102947`/evalTaskId `eval-task-6tn1xs-1783564210`;日志已归位 `logs/eval/evalform_v1_lora_ep1_20260709.log`。机制已按面板证据收档(experiment_log 版本详情),深挖按用户指示中止。
- [!] **riders_fk_lora_ep2 已训完打包,待用户裁决是否上传(07-08 训完;evalform 出局后=唯一在手弹药)**:704步/78min,eval_acc 0.6233(ep1 锚 0.6122,第二遍仍在学);precheck 保险丝过(B 0.0%/C 100%/A 复读 26.7% 家族最低档);包 `submissions/riders_fk_lora_ep2_platform/`(adapter md5 `efaee6fb`)。预登记分支:A(~25%)0.94-0.97 / B(~48%)0.915-0.935 / C(~27%)0.88-0.91,物料 0.2453/0.2146/0.1840 一眼定分支。⚠️配额:07-09 窗口已被 evalform 用 1 发,且上传实操截止≈10:30 已过——**最早 15:00 刷新后传**。

- [x] **global_v1_lora_ep1 已回填(07-08 用户贴UI)**:总分 **0.8246**(−0.0931 vs riders),mat 0.2146(7题)/action 0.0438/topic 0.0357/video 7题/prod 29题/ad 87题/live 115题/world 0.1394。**官方解析重构血统第三证伪;四个 LoRA-1ep 数据点物料全部=7题 ⇒ 数据轴撼不动物料墙,只剩制度轴(遍数/rank)——直接支撑 ep2。**
- [x] **★riders_fk_lora_ep2 已训完(07-08;第二次训斥后回退为严格单变量;上传裁决见 P0 顶部)**:超参=riders 0.9177 线上验证值一个不改,**唯一变量 epochs 1→2**。机制依据:物料阶梯=子空间多样曝光遍数(LoRA lr2e-4 档遍数轴只测过1ep=7题)+LoRA冻结基座保video/ad先验(两个LoRA谱系已证)+action欠训分支。官方侧佐证(理解用非依据):UI默认3轮=官方认为LoRA需多遍。配置 `configs/history/riders_fk_lora_ep2.yaml`(+merge),预登记判据在头注(物料8/7/≤6题三分支各定下一步)。**07-08核查:riders action类行(≥1474,含P3)已全nothink ⇒ action缺口主因=复读不停,不是think错位。⚠️07-08全日志扫尾注:平台日志每任务仅5可见样本,action失败率/mat收敛度两轴在n=14上相关性归零(0.01/−0.05),日志不能为ep2预期提供上调或下调依据,预期维持机制先验(A 30-35% 0.94-0.97 / B 35-40% ≈平 / C 25-30% 0.88-0.91)。**单卡~1h。
- [!] **平台UI训练臂(备选,不占本地GPU)**:海飞丝未否认"只在网站上训练";平台loss=focal+token加权,"纯平台训练0.99"传闻可信度上调。若 ep2 出分证伪本地遍数路线,候选=平台UI LoRA r32/lr手动提至2e-4/2-3ep×FK数据(平台训的模型无需传ckpt)。需你在平台操作。
- [!] **★数据撞车裁决(07-07;校准判定后修订建议)**:①`quality_swap_v1` 的 U1/U2 把 r2_gold 全部 367 条金标吃进训练 → dev_action 325/325 全烧。**校准已判 action 维本就盲区(离线读数不作判决),烧掉的代价大降——修订建议:包不动、不必新标 dev_action_v2**(除非未来拿到更好 gold 源再翻案);②`ally_map` 撞 dev_rec 1113/4000 → 已建 `dev_rec_*_v2_exally`(rec 也判盲,仅供行为仪表盘用);③长期制度不变:训练包 QC 必查与 dev_*.jsonl 零重叠(offline_eval.md §2)。
- [x] **★离线评测台 v3 建成+15 锚校准完成(07-07,存量命令闭环)**:判定 **8 维盲区 + world 仅方向**(Spearman/超噪声对判对率见 `docs/offline_eval.md` §6;敏感性分析无一维获救)——**平台分布不可克隆=测量定案,离线数字(除 world 方向)禁作收益证据**;台子降级重定位=回归保险丝(fmt/json/trunc)+离群检测(recipe2 action 0.24 十倍离群=当场抓到 R2 分布内记忆);包出门判据回归"平台真分+格点分解+预登记红队+precheck"。工具/协议常青:新面板出分回填 `calibrate_offline.py` TRUTH 重跑;拿到平台题面级 dev 源可翻案。副产品硬情报:八任务真题规模(574/1000×4/1739/905/807)、CEval 被 MC 锚烧光(圈外 MC=CMMLU)、八任务模板逐字(offline_eval.md §1)。
- [x] **quality_swap_v1_lora_ep1 线上证伪(07-08)**:平台总分 0.8235,相对 `riders_fk_lora_ep1` −0.0942;物料 7→6题、action/topic/world 同跌,rec 仅 video +1题但 prod/ad 同跌。日志 `logs/eval/quality_swap_v1_lora_ep1_20260708.log`;evalTaskId `eval-task-jysa9i-1783486168`;ckpt `checkpoints/quality_swap_v1_lora_ep1/` 仅留审计。**不做 ep2,不再沿官方形态 user 替换线加码。**
- [x] **official_rec_v3_lora_ep1 线上证伪(07-08)**:只用官方 `OneReason_UserProfile`+`OneReason_Pid2Sid` 重构 8000 条多域 next-item 推荐样本,替换 riders 中 8000 条旧 video-heavy rec 行,总量仍 37267。线上总分 **0.7948**(vs riders −0.1229);物料持平 0.2146,action/topic 0.0630/0.0442,rec 官方序 video/prod/ad/live = 0.0960/0.0476/0.0994/0.0891,world 0.1409。**判决:官方 UserProfile 重构样本与评测推荐分布/答案机制不等价,替换 riders rec 行会严重破坏 prod/ad/live;不得继续 v4/ep2/加量。**
- [x] **ally_map_lora_ep1(v1) 作废但留档(07-08)**:数据 `data_ally_map.jsonl` 45267 条和配置仍在,但 rec_loo 配比按旧面板读法把 ad/living 放太高,会浪费当前有限提交配额。**不要直接训练 v1。**
- [!] **ally_map_v2_lora_ep1 降级搁置(07-08 海飞丝对照后)**:数据/配置就绪不动(`data_ally_map_v2.jsonl` 45267 条,QC 已过),但两个 0.99 的公因子=LoRA 制度而非映射表数据,"映射表=魔法"假设被削弱;且需先排除 official_rec_v3 同类风险。**排在 riders_fk_lora_ep2 之后再议。**
- [x] **rebal_focal_ep3 已证伪并退出 P0**:γ=2 训练完成但 precheck 硬FAIL(itemic断裂+选择题崩),未花配额;不再作为今日主线。
- [ ] **★情报(07-08 升级,当前期望值最高的一项):群里挖 筷快块(↑119→1.0048)/试一下(↑154→0.9976) 07-06后发言 + 数据科学社 07-07 两连跳(1.0194→1.0535)前后发言**——第三波百名级跳升,5-10名挤在0.995-1.005的0.01带内=同一份配方在流通;任何被大量感谢的分享=方法源头。(旧目标 Southside旧↑44/冰激凌↑247 并入;冰激凌已跌出top10,说明配方还在迭代)
- [ ] 下一发必须先复盘 `official_rec_v3` 翻车:为什么 video +2 题但 prod/ad/live 大跌,尤其检查新样本答案机制、官方 UserProfile gold 与评测多 gold/候选池差异、以及替换而非增量对 riders 底盘的破坏。
- [ ] `ally_map_v2_lora_ep1` 若还要做:先重审与 `official_rec_v3` 是否共享同类风险,确认不是“替换/冲淡 riders 推荐底盘”后再查 GPU 单卡训练。

## P0-历史(07-03定,多数已完成)


- [x] **① ad 塌因已坐实(2026-07-03,完整分析 `docs/ad_collapse_analysis.md`)**:塌因=**nothink 直通路退化为历史复读机**(直通路整句抄史 89.4%、s_a 抄史 99.4%,新候选仅 16 个全垫底;抄史率与 ad 分严格反向单调 75.0/73.8/71.2% ↔ 0.048/0.067/0.096;种子数据 ad gold∈历史仅 12.6% 封死复读上限)。排除:thinking 记忆化(同 Sample 32 beam 共享 1 条 thinking 是机制)、全局多样性坍缩、video 泄漏、候选池独立致塌。**thinking 通路反而健康(62 新候选三者最多)**。机制订正:直通路 beam32 非 64。
  → 派生动作:a) rebal_world_ep3 出分日志首查直通路抄史率(≥85% 预期仍塌);b) **ad 样本按 gold∈历史分桶加权**(gold∉历史上采样——比无差别上采样对症,进 recipe8/两阶段设计);c) 两阶段第二期用 **nothink 格式 ad 样本**;d) 队友 config 最优先核对 lr。
- [ ] **② recipe7 = data_rebal_world × v6 配方(lr2e-5 / 3ep / batch4)**——头号训练候选,目标 0.91+。数据就绪(29019条,LF 已注册,读 lustre `data/processed/`)。⚠️ 三点:a) 起训前 `nvidia-smi` 查空闲卡;b) 用新 venv `ai_runtime/llmrec_2026/LLaMA-Factory/.venv`;c) **出分后首查物料是否从 0.2453 回落**(ad 上采样动的是 video_ad 共享子空间,recipe2 翻车同源风险)。注:相对 v6 同时改了重平衡+通识两个变量,experiment_log 里标注组合实验。
- [ ] ③ recipe7 训完 → precheck(对照 recipe1 复读 33%)→ 结果存 `logs/precheck/recipe7_<日期>.txt` → 过检才传。
- [x] ④.5 baseline-epoch3 底细已明(07-03 用户两次补充信息,**第二次修正第一次的错误推断**):它是**平台训练服务默认参数**训的。用户贴出平台默认参数全表:**LoRA rank32/alpha32/dropout0.1、lr 1e-6(!)、seq 32768(不是我猜的8192!)、batch1×accum8、cosine、warmup0.03、wd0.01、3ep、packing、bf16**。修正三点:①我此前"cutoff8192截断→action归零"的推断**作废**——默认seq就是32768,截断解释不成立;②action_select 0.0000 的真因更可能是 **lr1e-6 的 LoRA 太弱**:懂用户任务 pretrain 基线本来就≈0(v0=0.0000/0.0055),1e-6 LoRA 训不动它,而物料/推荐 pretrain 有底子(物料从 0.1533→0.2146 说明 LoRA 也在涨);③但由此产生**更有价值的信息**:lr1e-6-LoRA×3ep 这么弱的训练,ad=0.0768(>v1 全参 0.0672)、prod=0.1498(全账号最高)——**"几乎不动模型"反而保住了 pretrain 的 ad/prod 能力**,佐证 ad 塌因分析的结论:全参 SFT 教会模型抄历史,把 pretrain 本来会的"预测新 item"能力洗掉了(seed_ep3 抄史 89% vs pretrain 行为)。cutoff 变量重新存疑:克西 8192 是他自述,与平台默认无关。
- [x] **precheck 校准三连(2026-07-03,落盘 `logs/precheck/`)**:recipe4 复读 16.7% / recipe1(0.8428 锚)33.3% / seed_ep3(0.8931 锚)**50.0%**——复读率与线上分不负相关,15% 阈值作废;新规:仅当"结构断裂>10% 或选择题格式崩"伴随出现才拦。recipe4 提交包已就绪(`submissions/recipe4_kexi_repro_platform/`,md5/冒烟均过)但按 ④ 降级为备用弹。
- [ ] **④ 今日剩 2 发的用法(修订)**:第 1 发 = rebal_world_ep3(训完过检即传);第 2 发 = **等 rebal 出分后适应性决定**——若 ad 仍塌(直通路抄史率≥85%),首选传它自己的 checkpoint-640(2ep,从未测过的跷跷板拐点,免费拿到的);**recipe4 降级为明天 15:00 窗口关闭前的备用填充弹,不再是今天的优先项**(理由:recipe1 已给出无trick下限≈0.8428 的近似;且 cutoff8192 的 action 归零假象会污染其读数)。
- [ ] **④.5 向队友要 baseline-epoch3 的 config+训练日志+评测日志**(0.7807,07-02 15:40)——其 ad 0.0768 未塌 / prod 0.1498 全账号最高 / action 0.0000 全崩。若真是官方baseline×3ep(cutoff8192),则"低lr多ep→ad必塌"被推翻,cutoff/packing 差异才是 ad 塌因候选——**直接影响 P0① 塌因分析和 recipe7 归因**。
- [ ] ⑤ **两阶段训练**(跷跷板破解主候选,recipe7 出分后定稿):v6 权重 warm start → 第二阶段小步高 lr 只喂 ad/user 侧数据。**设计修正(复核结论)**:第二阶段 lr 用 **5e-5 级**而非 1e-4(1e-4 是物料塌方已定罪真凶,且物料/ad 共享子空间,高 lr 扰动会把 0.2453 打回去)、0.3-0.5ep、必须混种子 replay;ad 数据可叠加 unCoT 化(见 P1-数据侧)。v6 ckpt 无 optimizer.pt,fresh optimizer 即可(SFT 无妨)。

## P1 — 未测的高价值方向(复核后新增/修正)

**数据侧(不耗配额,可并行开工):**
- [x] **mat 7→8 的"FK 丢曝光"假说已审计否定(07-07)**:同一分类器下 FK 重组 vs 种子,video 域行数 9742→9525(−2.2%)/唯一gold 9705→9494,ad −34 行——**FK 没砍 video_ad 子空间曝光多样性,LoRA 档物料卡 7 题是制度问题(1ep遍数/LoRA容量),不是数据问题**;物料数据侧空间被"重复毒/墙外毒/稀释税+本审计"四面封死,第 8 题只能从制度找(ep2 单变量/全参对照/两阶段)。
- [ ] **★action 复读抑制专项数据(当前数据侧头号,07-07 离线台抬价)**:离线台实测复读顶格截断率 riders 54% vs recipe1 17%(LoRA 血统病更重);病灶=输出不会停,方案=教"JSON 数组正确闭合即 EOS"+条数对齐官方分布(评测逐字模板在手,见 offline_eval.md §1);与 U1/U2/U3 官方形态互补,可进 quality_swap 下一版。
- [ ] **Token 粒度数据构造**(全文档复核第一新发现):OneReason Table 2 Exp2——+Token 粒度让 Item Understanding_**ad** 16.4%→37.9%、Itemic Grounding_prod 2.4%→5.8%,**唯一同时打跷跷板两端(ad+物料)的数据类型**,占比仅 2.5% 挤占风险小。构造法(onereason_data_method.md §A.1):共享前缀 sub-token 对的 item 共同语义总结 + 反向 grounding,原料 Pid2Sid+Caption 本地有。这也是 competitor_intel L103 否定 itemic 加权后指名的"trick 候选①数据组织方式"的具体抓手。⚠️tokengeo_v1 已线上证明 T2 前缀截断子型有毒,重试必须去 T2/全三元组/降剂量。
- [ ] **ad 域 unCoT 化**(第二新发现):论文 §D.14——ad 是**唯一 CoT 混入反而降分的域**(prod 偏 CoT-heavy,video/live 均衡);我们训练数据 76.9% 含 CoT,与 v6 ad 塌方方向吻合。只对 ad 域样本剥 CoT(≠recipe3 的全域剥,那个害懂世界归零被证伪)从未测过——可能是"两头兼顾"的数据侧钥匙,可进两阶段第二期数据或 recipe8。
- [ ] **action_select EOS/复读抑制专项数据**(eval_analysis_v4 §5.2 留的尾巴):v4 可见样本 4/5 复读到 4096 截断、JSON 合法率 1/5;懂用户权重×2,此项**独立于 lr 之争**,预计有一截独立收益。至今没人做。
- [ ] LOO 数据修正(队友报告驱动):video 域 gold 按 `play_done=1` 过滤(否则 gold 可能是划走的视频);ad 域并入 `outer_loop_deep_target_pid` **8851 条全数据集唯一严格无泄漏金标**;动作词规则对照报告 §2.8.5 优先级表逐条校验;顺带落实 strategy_roadmap 坑3 的输入分布对齐(80% itemic-only)。
- [ ] run_d_r2material 重估(experiment_log L49 悬空尾巴):当时"等 ep3 验证"暂不传,v6 已回答 ep3 问题;其种子+R2+物料组合在 lr2e-5 配方下的价值需重估,可能给 recipe8 省一次训练。

**训练侧(耗配额,排 P0 之后):**
- [ ] RFT-first(**GRPO 押后**,roadmap §2 修正2:"初赛先 RFT 不必等 GRPO"):rec_loo 12000 条(修正后)拒绝采样 K=32/64 → reward 过滤 → 低 lr 回灌;ad 域(deep_target 金标)先行。roadmap §5 称之"初赛最可能带来排名跃迁的一步",至今一步未跑。
- [ ] lr2e-5 × **2ep** 纯种子:跷跷板拐点测绘(选手锚点 1533/1840/2146 阶梯的 2ep 位),我们从没干净测过。**预期总分 < v6,不配当下配额**,GPU 空闲时低成本训着备用。
- [ ] recipe8 = rebal + R2:**等 recipe7 归因后再定**。⚠️ v2 的 R2 曾致 ad −0.019(与救 ad 目标相抵),且未过论文 §B.8 的 11 项质检——先过滤再上。

**已否决/降级(有落盘证据,勿再提):**
- ✗ 答案段 itemic 加权:competitor_intel L126 **点名否定**("换答案段-only 仍是全域一刀切");L103 明令"不再投配额验证 itemic 加权变体"。唯一存活变体 = **按 codebook 子空间分治**(video_ad 组内单独平衡,live/prod 不动),如要做按这个设计。
- ✗ GRPO 直接上:改 RFT-first(见上)。
- ↓ 权重融合 v4⊕v6:**降级为纯诊断**——两模型在共享 itemic 子空间学到的是冲突解(v3→v4 同题 top1 全变),线性插值大概率两头平庸;且 proxy 已删无法离线选 α、平台不鼓励融合、复现审核交不了"无训练脚本的产物"。仅当主线配额富余时用 α=0.5 单点回答"跷跷板能否权重插值"这一科学问题,不作提交路径。

## P1.5 — 第二曲线原条目(并入上面 RFT-first,保留记录)

- [x] LOO 推荐数据构建(2026-07-03):`rec_loo.jsonl` 12000 条(四域各3000),已备份 `data/processed/`。待按上面"LOO 数据修正"升级。

## E — 复赛储备(不占初赛资源)

- [ ] 队友分工建议:优先补 §2.10(与官方样例对齐——自建数据不踩格式坑的关键)和 §2.9(通识分析);约束解码/加 token 路线明确告知走不通(黑盒评测,只传 safetensors,vocab 176253 是校验项)。
- [ ] pattern→desc(复赛新增 LLM-as-Judge 任务)数据预研。
- [ ] 若需 ad 域 SID 频率分布做塌因分析,向队友要 `analysis/outputs/sid/freq_hist_video_ad_*.csv`(本机没有)。

## P2 — 文件管理收尾(2026-07-03 盘点后遗留)

- [!] **`stage2_gold_v1_lora_ep1/checkpoint-103/`重复末轮目录**:根目录已保留最终adapter,子目录另含约162MB optimizer和同一末轮adapter,整个实验目录约340MB。该实验已门禁否决且不续训,理论可删子目录;按破坏性操作纪律,等用户明确批准后再删。
- [!] **删 11 个 ckpt 的 `checkpoint-N/` 子目录**(根目录模型 md5 已逐一核验相同,≈17.6G;其中 recipe4/5/6 的子目录还各有 3.0G optimizer.pt 漏删)。**破坏性操作,等用户批准**——批准后执行并在 DATA_INVENTORY 清理记录加行。
- [!] **删 `submissions/baseline_sft_v1_upload/`**(1.6G,与 `_platform` 版模型同 md5 60690c0f,纯重复)。等用户批准。
- [!] **删 `logs/eval/Kne1N...y7r.log`**(v6 日志哈希名副本,cmp 逐字节相同)。等用户批准。
- [!] **`notebooks/taskmanager 2.10.1-arm64/`(1.4G)**:内容是 macOS 应用安装包(QQ/Obsidian/wpsoffice…),与项目无关,疑似误传。**等用户确认后删**。
- [ ] `src/` 为空目录:要么删掉并从 README 移除引用,要么开始用。README 的目录说明同步更新。
- [x] overlay 盘数据清空(2026-07-03,用户指令):全部数据 md5 核验备份到 lustre 后从 `/root/baseline_repro/` 删除;`data/processed/` 成为训练数据权威路径。
- [x] **环境迁移 lustre(2026-07-03,用户指出 miniconda3 先例后执行)**:`/root/baseline_repro` 已整体删除。LF+venv 重建于 `ai_runtime/llmrec_2026/LLaMA-Factory/`(根目录符号链接 `LLaMA-Factory`),验证:torch/flash-attn/liger/wandb import 全过 + llamafactory-cli tokenize 冒烟 OK + transformers patch 已打。12个 configs 的 dataset_dir、scripts 缺省路径、00_install.sh 全部改指 lustre。**新训练启动:`source ai_runtime/llmrec_2026/LLaMA-Factory/.venv/bin/activate`。**教训:环境放 overlay 从一开始就是错的(卷上 miniconda3 9个env就是先例)。
- [x] 4 个散落的 `*_launch.log` 归位为各 ckpt 内 `train.log`(2026-07-03)。
- [x] 实验总账建立:`docs/EXPERIMENT_INDEX.md`(config/ckpt/提交包/评测日志/evalTaskId 全链路 md5 对账)。发现:**v5 线上模型唯一副本在提交包里(ckpt-49 已删),该包不可删**。

- [x] **offline_probe v1 建成并回测——pass@64 维度校验未过,不上岗(2026-07-03,按预登记规则执行)**。工具 `scripts/eval/offline_probe.py`(vLLM 两通路 beam32+beam32 / 物料 beam64,机制 1:1 复刻),五 ckpt 回测报告在 `logs/probe/` + wandb(run=probe_*)。失败证据:①物料维 probe 排序 recipe1≈seed_ep3(0.1133=0.1133),线上实际 0.1533 vs 0.2453——探针样本抽自训练数据,测记忆不测泛化;②rec 四域 pass@64 全≈0(LOO gold 太难+n=100 无功效),零分辨力;③行为指标在 LOO 分布上的排序与平台日志实测不一致(probe 说 recipe1 抄史最多 0.60,平台日志实测 seed_ep3 最多 89.4%)。**幸存**:行为仪表盘保留(抄史率的数据干预方向信号真实:rebal_mat 0.42<rebal_world 0.46<seed 0.50),只诊断不决策。**v2 修法排队**:物料改用训练集外新鲜 desc→SID 对(17G Pid2Caption+Pid2Sid 构造);rec 需 n≥1000/域+复合指标。

## 惯例(防再犯)

- **★官方优先(07-08,两次训斥后定稿)**:每会话必读 `docs/platform_guide.md`(官方全案);新官方材料一律并入该文档。§〇 对账表的用法=**理解测量仪器**(评测think开关/解码参数/计分器→数据形态该长什么样)和**读官方机制信号**(如默认3轮),**不是抄数值**——已线上验证的自家超参优先;改任何超参必须有机制假设。
- **LoRA vs 全参必须做配对 A/B**:固定同一基础模型、代表性数据及 train/val 切分、样本顺序、模板、cutoff、packing、global batch、epoch/有效 token 曝光、scheduler、seed 和评测流程,分别训练 full 与 LoRA。严格归因臂只改变训练方式及其必需字段;实用性能臂则给两者相同搜索预算,各自做小规模 lr 搜索后比较最优结果。历史上同时改变数据或多项超参的实验不得用于归因 LoRA/全参优劣。
- 新实验:先 experiment_log 加行(分数⏳)→ 训完填 loss/acc → EXPERIMENT_INDEX 加行 → 体检落盘 → 上传 → 出分回填两表。
- 评测日志下载后**立即**改名 `<训练名>_<日期>.log` 并删平台哈希名原件。
- 提交包命名 `<训练名>_platform/`,打包后 md5 对 ckpt。
- 每日会话结束前:更新本文件 + experiment_log 速览。
