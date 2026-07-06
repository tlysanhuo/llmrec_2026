# TODO — 快手 LLM-Rec 2026(唯一 TODO 文档)

> 建于 2026-07-03,大修 2026-07-04 晚(会话交接)。**规则:每个会话开头必读;状态变了立刻改这里。**
> 标记:`[ ]` 待办 / `[~]` 进行中 / `[x]` 完成 / `[!]` 等用户拍板。
> **大局(07-04)**:Top1=1.0118(分可>1.0);前8≥0.951;我们最好 0.9009(±0.03噪声);两队一夜跳升⇒存在扩散中的方法跃迁。评测方差已标定:物料零噪声/ad±0.02/总分±0.03。

## P0 — 进行中/今日

- [~] **rebal_focal_ep3 续训**(GPU2,~685/960):loss CE→focal γ=2 单变量。训完→precheck→打包→传(今日剩2发配额)。判读:物料(零噪声)定 focal 生死;分项形态>总分。若挂:resume_from_checkpoint 在 config 里,换卡重启。
- [ ] **情报:群里挖 Southside旧(↑44)/冰激凌(↑247) 07-03后发言**——两队一夜进前五,方法可复制且正在扩散。任何被大量感谢的分享=方法源头。
- [ ] focal 出分后:γ 变体(1 或 3)或转向,按分项形态定。

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
- [ ] **Token 粒度数据构造**(全文档复核第一新发现):OneReason Table 2 Exp2——+Token 粒度让 Item Understanding_**ad** 16.4%→37.9%、Itemic Grounding_prod 2.4%→5.8%,**唯一同时打跷跷板两端(ad+物料)的数据类型**,占比仅 2.5% 挤占风险小。构造法(onereason_data_method.md §A.1):共享前缀 sub-token 对的 item 共同语义总结 + 反向 grounding,原料 Pid2Sid+Caption 本地有。这也是 competitor_intel L103 否定 itemic 加权后指名的"trick 候选①数据组织方式"的具体抓手。
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

- 新实验:先 experiment_log 加行(分数⏳)→ 训完填 loss/acc → EXPERIMENT_INDEX 加行 → 体检落盘 → 上传 → 出分回填两表。
- 评测日志下载后**立即**改名 `<训练名>_<日期>.log` 并删平台哈希名原件。
- 提交包命名 `<训练名>_platform/`,打包后 md5 对 ckpt。
- 每日会话结束前:更新本文件 + experiment_log 速览。
