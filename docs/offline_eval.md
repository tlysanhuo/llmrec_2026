# 离线评测台(offline_eval)— 设计与校准记录

> **变更记录(2026-07-11 07:38 UTC)**:新增 §8“平台可见题提交前门禁”,记录20份日志的题面稳定性、训练重叠审计和 `stage2_gold_v1_lora_ep1` 首次双门禁案例。原因是平台日志给出了比自建dev更接近真实分布的固定样例,但无大多数gold,必须明确其用途只限回归否决而非收益预测。
> 建于 2026-07-07(用户存量命令:"搭建离线测评",本文档=该命令的唯一记录本)。
> **地位**:离线台是**出门门禁**——校准完成前,禁止提出任何训练/提交/新包方案;任何包出门必须带自己的预测表+离线台数字。唯一裁决仍=平台真分(每日3发,账号级共享)。
> **状态:2026-07-07 当日建成并完成 15 锚校准。判定:8 维盲区 + world 仅方向(§6)。离线数字不作收益证据;台子降级重定位为"回归保险丝+离群检测"(§7)。**

---

## §0 三代工具沿革(为什么前两代不可信)

| 代 | 工具 | 结局 | 死因 |
|---|---|---|---|
| v1 proxy 套件(07-02) | proxy_material_beam / proxy_r2_f1 等 | **07-03 全删** | 对 5 个线上真分验证:只有懂用户 F1 可靠;物料无分辨力;recipe2 全 proxy 上涨、线上崩到 0.7692(假阳性烧掉一发配额) |
| offline_probe.py(07-03) | 机制 1:1 beam 台(rec n=150/域 + mat 300) | **回测未过,降级为行为仪表盘** | ①mat 样本抽自训练集=测记忆不测泛化(fk_lora_embed 灯塔:probe +58% 线上掉一档);②rec LOO pass@64 全≈0(gold 难 + n=150 无功效);③行为指标排序与平台日志实测不符 |
| probe_v2.py(07-07) | 五组任务镜像(经 :8123 服务) | **从未校准,机制近似** | ①world/rec 仍从训练块采样(记忆化陷阱没解决);②mat 用采样 n=32 近似 beam64(机制失真);③action 留出集仅 28 条;④只跑过 global_v1 一次,零校准 |

**本代(offline_eval v3)修法**:机制逐项 1:1(beam64/两通路 beam32/官方采样参数)+ 全部维度用**圈外或留出**数据 + **先校准后上岗**(拿 15 个有线上分项面板的 ckpt 验排序保真,不保真的维度明标"离线盲区")。

---

## §1 平台评测机制镜像(硬情报,2026-07-06/07 从评测日志+官方FAQ逐字核实)

评测环境:simple-eval + vLLM v0.11.1,seed=42,bf16,enforce_eager,max_seq_len 40960,prefix_caching。
**rec 域列序(07-06 官方FAQ定案)**:平台面板从左到右 = video / prod / ad / live。读历史台账必过换算表。

| 任务 | 真题 n | 生成机制 | 计分 | 量子(1题价值) | 线上噪声 |
|---|---|---|---|---|---|
| itemic_pattern_grounding(物料) | **574** | nothink 空think直出,**beam64 × 3 token** | Pass@64(SID+PID 双模式) | **0.030693** | **零**(beam 确定性) |
| recommendation_video | **1000** | 两阶段:采样 thinking(n=1, ≤4096tok)→ **beam32**×3tok;+ nothink 直通 **beam32**×3tok;合并 64 候选 | Pass@64 | **0.009614** | ±2题 |
| recommendation_product | **1000** | 同上 | Pass@64 | 0.003401 | ±2题 |
| recommendation_ad | **1000** | 同上 | Pass@64 | 0.001400 | ≈0(±1题) |
| recommendation_live | **1000** | 同上 | Pass@64 | 0.000900 | ±3题 |
| evolution_action_select(懂用户) | **1739** | 采样 n=1, ≤4096tok, nothink | F1(官方,难度归一) | 连续 | ±0.003 |
| evolution_topic_gen | **905** | 采样 n=1, nothink | action 有序匹配+logic TokenF1/ROUGE-L | 连续 | ±0.001 |
| common_sense(懂世界) | **807** | 采样 n=1, nothink(max_tokens 60000) | 单选答案抽取匹配 | 连续 | ±0.005 |

**真题模板(2026-07-06 riders 日志逐字提取)**:
- 物料:system=「视频属性提取器/媒体内容分析师…」同义改写族;user=「请解析以下视频内容并输出对应的视频token:\n\n<desc>/no_think」;可见 5 样本**全为 video 域**(与"物料=video_ad 共享子空间"一致)。
- rec 四域:system 与种子训练数据同款(「你是一个智能推荐助理,能根据多域历史行为,推荐用户下一个感兴趣的广告。」等);多域历史 5-18k 字。
- action_select:**无 system**,「【用户交互历史】…以下案例来自其他用户,仅供参考输出格式…["<item>", …]/no_think」→ 输出 JSON 数组(含 itemic 串,亦可含文本项)。
- topic_gen:**无 system**,历史+few-shot logic_chain JSON,/no_think → 输出 `{"logic_chain":{"name":…,"events":[{date,action,logic}…]}}`。
- world:system=「你是一个非常聪明的助手,请直接遵循指示作答。」;user=「请回答以下问题:\n\n<题干>\nA.…\nB.…\nC.…\nD.…\n\n请按以下格式作答:"正确答案是 (在此处填写选项字母)"/no_think」。题源混合(地理/生活常识/数学),不纯是 CEval。
- **所有任务均为 nothink 空 think 前缀直出,除 rec 的 thinking 通路。**

换算:离线命中率 r × 官方 n × 量子 = 预测分。例:物料 8 题 = 8/574 = **1.39% 线上命中率**(我们所有离线 mat 读数都远高于此 ⇒ 平台题面显著更难,绝对值永远对不上,**只赌排序**)。

---

## §2 离线台 v3 设计

### dev 集(`data/offline_eval/`,由 `scripts/eval/build_offline_dev.py` 生成,建集脚本+md5 落盘)

| 文件 | n | 来源 | 圈外性 | 对应任务 |
|---|---|---|---|---|
| dev_mat_fresh.jsonl | 542 | mat_probe_fresh(队友建,Pid2Caption/Pid2Sid;prod150/video150/ad150/living92) | **圈外**(与已入训 fresh_mat 3000 按 gold SID 去重,剔 1 条) | grounding |
| dev_mat_train.jsonl | 300 | block_mat 抽样(训练集内) | 圈内——**仅作记忆化对照,不作判决** | grounding |
| dev_rec_{video,prod,ad,live}.jsonl | 各1000 | rec_loo_v2(LOO,原始 parquet 直建,play_done 修正版) | LOO gold 非训练目标;题面分布=种子同构 | rec 四域 |
| dev_action.jsonl | 325 | r2_gold_v4(133)+r2_gold_local(192)(teacher/本地标注,0 重叠) | 未入任何训练混合(风险注记:r2_base 家族入过 v2/RunD 训练,题面同源但 gold 行不同) | action_select |
| dev_topic.jsonl | 110 | u3_heldout(r2_base 2800-3000 行,teacher 标注) | 未入训 | topic_gen |
| dev_world.jsonl | 500 | **CMMLU test**(与训练 MC 锚仅重合 2 条,已剔) | 对全部锚 ckpt 圈外 | common_sense |

> **建集时的硬发现(07-07)**:CEval val 1346 题剔除训练 MC 锚后只剩 **23** 题——证实 Frinkleko 的 1573 条 MC ≈ CEval val 几乎全集,**CEval 这个源已被训练烧光**,world 圈外题只能用 CMMLU(11603 条候选,与锚仅撞 2 条)。

### ★ dev/训练分池制度(07-07 撞车事故后立规)

**事故**:离线台建成当天,并行构建的两个待训包与 dev 集撞源——①`data_quality_swap_v1` 的 U1/U2 把 r2_gold 全部 367 条金标吃进训练(**dev_action 325/325 全烧**,另撞 dev_topic 11 条);②`data_ally_map` 含 rec_loo_v2 5000 条(撞 dev_rec 1113/4000)。若不处理,离线台对这两个血统的模型恰好在其主打维度上读数虚高——门禁失效。
**处置**:dev 原版(md5 见上)在 15 锚校准期内**冻结不动**(全部锚均早于两包训练,校准有效);另建 `dev_rec_*_v2_exally.jsonl`(避开 ally 5000 后重抽)+ 校准后落地 `dev_topic_v2`(剔 11 条)/`dev_action_v2`(需新标注,见 TODO)。校准中途一度误改 dev 文件,已按原构建序字节级还原(md5 核验),污染窗口内在跑的 2 锚读数作废重跑。
**制度(以后不可违)**:
1. **训练包 QC 必查一条:与 `data/offline_eval/dev_*.jsonl` 全部题面零重叠**,不过检不注册。
2. dev 集与训练数据共享源池(r2_gold/u3/rec_loo_v2/fresh_mat…)时,**先划永久 holdout 再造训练包**,顺序不许反。
3. 校准批跑期间 dev 文件冻结;任何变更走版本化新文件(`_v2` 后缀),旧版留档供复算。
4. 判读某模型的离线读数前,先查其训练数据与所用 dev 版本的圈外性;圈内维度读数一律作废。

### 机制对齐(与 §1 逐行对应)

- 物料:beam64×3tok,空 think 前缀,域 begin token 由 gold 域给出;指标 pass@64 / sa_pass@64,分域拆。
- rec:通路1 = nothink 空think + 域 token + beam32;通路2 = 采样 thinking(T0.6/p0.95/k20,seed42,stop=</think>,≤1024tok)→ beam32;合并 64 候选查 gold。附行为仪表盘:直通抄史率 / distinct s_a / 新候选数(诊断不判决)。**已知歧义**:thinking 通路的软开关形态平台不可见(日志打印 nothink 形态 Input 但 Output[0..31] 带真实 thinking),默认保留 `/no_think` 裸采样(v1 探针同款),`--think_suffix switch` 备翻案。
- action:采样 n=1,T0.6/p0.95/k20,≤3072tok;指标 itemic 集合 F1 + JSON 合法率 + 顶格截断率(复读病显影)。
- topic:采样 ≤900tok;官方公式(action 有序 LCS 匹配罚漏罚多 + logic TokenF1/ROUGE-L 均值)。**解析宽容化(07-07 冒烟教训)**:严格 json.loads 在 0.8B 模型的中文引号/截断输出上 100% 全灭,计分改为"严格 JSON 失败→正则抽 action/logic 对";json_fail 保留为格式诊断。
- world:采样 ≤128tok;抽取「正确答案是 (X)」;指标 Acc + 格式存活率。
- 引擎:vLLM 离线(verl_v071 env),bf16,seed42,enforce_eager,prefix_caching——与平台评测器同款设置。单锚全维 ~40-70 min(H100)。

### 与平台的已知不可消差异(诚实声明)

1. **题面分布**:物料/rec 的平台真题不可下载;码本墙(评测物料 54-81% 在 HF 外)⇒ mat_fresh 绝对值必失真,赌的是"desc→SID 映射质量是全局性质→排序保真"。
2. **action/topic 的 gold**:自建 teacher 标注 vs 平台 gold,F1 绝对值不可比。
3. rec LOO gold 与平台 gold 生成方式不同(平台疑似同样是真实下一行为,同构性最好,但候选难度未知)。

---

## §3 校准协议(本代的核心,前两代全死在没做这步)

**锚集 = 15 次线上评测面板 × 本地全部在手的 ckpt**(物料阶梯 0.1226/0.1533/0.1840/0.2146/0.2453 五档全覆盖;rebal_world 同 ckpt 两面板=天然噪声标定对):

| 锚 | 本地路径 | 总分 | mat题 | video | prod | ad | live | action | topic | world |
|---|---|---|---|---|---|---|---|---|---|---|
| v0_pretrain | models/OneReason-0.8B-pretrain-competition | 0.6655 | 5 | 9 | 16 | 98 | 100 | 0.0000 | 0.0055 | 0.1387 |
| baseline_sft_v1 | checkpoints/baseline_sft_v1 | 0.8100 | 6 | 7 | 31 | 94 | 121 | 0.0362 | 0.0392 | 0.1375 |
| run_a_r2 | checkpoints/run_a_r2 | 0.8092 | 6 | 5 | 31 | 91 | 117 | 0.0667 | 0.0430 | 0.1294 |
| run_c_material | checkpoints/run_c_material | 0.8198 | 6 | 8 | 30 | 91 | 122 | 0.0446 | 0.0407 | 0.1346 |
| recipe1(v4) | checkpoints/recipe1_bs32_lr1e4_ep3 | 0.8428 | 5 | 10 | 32 | 89 | 121 | 0.0687 | 0.0401 | 0.1424 |
| recipe2_w5(v5) | submissions/recipe2_w5_ep1_platform | 0.7692 | 4 | 4 | 41 | 99 | 114 | 0.0703 | 0.0268 | 0.1305 |
| seed_ep3(v6) | checkpoints/exp_seed_ep3 | 0.8931 | 8 | 5 | 40 | 95 | 113 | 0.0554 | 0.0421 | 0.1316 |
| seed_ep5 | checkpoints/seed_ep5 | 0.9081 | 8 | 7 | 36 | 101 | 111 | 0.0584 | 0.0427 | 0.1309 |
| rebal_world_ep3 ×2面板 | checkpoints/rebal_world_ep3 | 0.9009/0.8776 | 7/7 | 6/4 | 38/36 | 99/98 | 113/116 | 0.0733/0.0701 | 0.0428/0.0418 | 0.1431/0.1487 |
| rebal_mat_ep3 | checkpoints/rebal_mat_ep3 | 0.8454 | 6 | 5 | 35 | 92 | 116 | 0.0747 | 0.0430 | 0.1435 |
| pstack_v2_ep3 | checkpoints/pstack_v2_ep3 | 0.8265 | 5 | 7 | 30 | 92 | 120 | 0.0808 | 0.0429 | 0.1435 |
| tokengeo_v1_ep3 | checkpoints/tokengeo_v1_ep3 | 0.8338 | 6 | 4 | 34 | 82 | 115 | 0.0905 | 0.0424 | 0.1446 |
| fk_lora_embed_ep1 | checkpoints/fk_lora_embed_ep1_merged | 0.8672 | 6 | 7 | 35 | 95 | 115 | 0.0756 | 0.0429 | 0.1420 |
| riders_fk_lora_ep1 | checkpoints/riders_fk_lora_ep1_merged | 0.9177 | 7 | 8 | 37 | 99 | 122 | 0.0655 | 0.0427 | 0.1439 |
| global_v1(仅总分锚) | checkpoints/global_v1_lora_ep1_merged | 0.8246 | ? | ? | ? | ? | ? | ? | ? | ? |

(rec 四域已按 07-06 官方列序重标;题数 = 分数/量子。)

**逐维保真判据(出结果前锁定)**:
1. **Spearman ρ**(离线读数 vs 线上题数/分数,n=15 面板)。
2. **超噪声对判对率**:只取线上差距 > 2×噪声带的 ckpt 对(mat 任何 ≥1 题差都算;video ≥3题;prod ≥3题;ad ≥3题;live ≥4题;action ≥0.006;world ≥0.010;topic ≥0.003),统计离线方向判对比例。
3. **判定三档**:
   - **可判决**:ρ≥0.8 且对判对率≥85% ⇒ 该维离线数字可作为出门门禁依据;
   - **仅方向**:ρ≥0.5 或对判对率≥70% ⇒ 只作方向参考,不作数字承诺;
   - **离线盲区**:其余 ⇒ 明标盲区,该维只能靠平台真分,任何"离线显示 X 涨了"的说法禁止出现。

## §4 预登记预测(校准跑完前锁定,红队纪律:先列出哪个已知机制能杀死每一维)

| 维 | 预测 | 能杀死它的已知机制 |
|---|---|---|
| mat_train(圈内对照) | **盲区**(高把握) | 已两次实锤:测记忆不测泛化(fk_embed +58% 假涨) |
| mat_fresh | 中等把握"仅方向"以上;能否"可判决"看码本墙 | 评测物料 54-81% 在 HF 外——若排序也被墙切断则死;beam64 n=543 对阶梯 1 题差(1.39‰)分辨力可能不足 |
| rec_ad/live | 保真概率中(线上命中率 8-12%,n=1000 功效够) | LOO gold 分布≠平台 gold;若离线命中率仍≈0(v1 探针旧况)则功效死 |
| rec_video/prod | 低功效风险(线上真率 0.4-4%) | n=1000 下 video 期望命中 4-10 个,±2 采样噪声=信号量级 |
| action | **高把握保真**(唯一历史验证过方向+幅度的维) | gold 换成 v4/local 新标注,若标注质量差可能稀释 |
| topic | 低分辨(线上全史带宽仅 0.027-0.043) | 官方难度归一细节未知 |
| world | 保真,但**必须用圈外题**(MC 锚 1816 在 pstack/riders/tokengeo 训练内,圈内题必然膨胀) | 平台题源混合、CEval 风格偏置 |

**总分**:离线台**永不合成预测总分**(v1 proxy 死因之一)。只做分维判决。

---

## §5 运行手册

```bash
# 0) 建 dev 集(一次性;世界维需外网下 CEval/CMMLU)
python3 scripts/eval/build_offline_dev.py

# 1) 单 ckpt 全维评测(~25-40min/ckpt,单卡;VERL env)
V=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
$V/miniconda3/envs/verl_v071/bin/python scripts/eval/offline_eval.py \
    --model checkpoints/xxx --gpu 3 [--dims mat,rec,action,topic,world] [--n_rec 1000] [--tag xxx]
# 输出 logs/offline_eval/<tag>_<ts>.json

# 2) 校准批跑(锚集全量,双卡轮转)
bash scripts/eval/run_calibration.sh   # GPU1+GPU3,逐锚跑,断点续跑(已有 JSON 跳过)

# 3) 保真报告(读全部 JSON × §3 真值表 → 逐维 ρ/对判对率/判定档)
python3 scripts/eval/calibrate_offline.py
```

precheck.py 保持原定位:结构保险丝(B 断裂/C 格式崩才拦),过检不作质量证据,与本台互不替代。

---

## §6 校准结果(2026-07-07 19:13 定案,15 锚全部同卷=冻结版 dev,污染窗口 4 锚已重跑/复算对账)

| 维 | Spearman ρ | 超噪声对判对率 | 判定 | 死因(对账预登记 §4) |
|---|---|---|---|---|
| mat_fresh | −0.339 | 34%(71对) | **✗盲区** | 码本墙兑现:pretrain 离线最高(0.127)线上最低(5题)——SFT 越贴评测分布离 HF 分布越远,方向整体反转 |
| mat_train(对照) | −0.475 | 27% | **✗盲区** | 记忆化,预登记"高把握盲区"兑现(第 4 次实锤) |
| rec_video | +0.465 | 67%(36对) | ✗盲区(全场最接近) | 命中 1-10/1000 泊松噪声,功效不足 |
| rec_prod | −0.406 | 23% | **✗盲区** | pretrain 离线碾压 SFT、线上最差:LOO gold 量"预测新item",平台 gold 疑似奖励"抄史+排序"(平台 ad 命中率≈10%≈gold∈史比例 12.6%) |
| rec_ad | +0.059 | 48% | ✗盲区 | 同上 |
| rec_live | −0.338 | 30% | ✗盲区 | 同上 |
| action | +0.108 | 53% | **✗盲区** | **预登记"高把握保真"被打脸**:①R2 血统分布内膨胀(recipe2 离线 0.24=十倍离群,老 proxy 0.7692 惨案同款失明,本次被当场抓住);②新 gold(v4/local 标注)F1 绝对值 0.01-0.02,标注质量稀释(预登记列过的 kill 机制) |
| topic | −0.045 | 36% | ✗盲区 | 线上带宽 0.027-0.043 本就无分辨,预登记兑现 |
| world | +0.292 | **70%**(30对) | **仅方向** | 唯一幸存维:CMMLU 圈外 Acc 可作方向参考,禁止数字承诺 |

**敏感性分析(预先声明)**:①剔 pretrain(SFT-only 13 锚)无一维获救(video 反降 60%,world 稳 70%);②action/topic 再剔 R2 血统(11 锚)更差(action ρ=−0.264)。
**探索性行为指标(预登记外,仅生成假设)**:全灭——mat~sa_pass **ρ=−0.705 强反相关**(码本墙的反向信号,可作"警惕指标":新模型 fresh sa_pass 大涨=向 HF 分布过拟合的嫌疑);video~copy/distinct_sa、action~trunc 判对率 30-31%;ad~distinct_sa 61% 最好也不及格。

**核心科学结论**:平台评测分布不可本地克隆,现在是**测量结果**而非猜测(9 维 15 锚 71 对超噪声比较)。07-03 删 proxy 的判断被系统性证实;"平台真分+格点分解"作为唯一判决的地位由此**抬升**,每一发配额的信息价值同步抬升。

## §7 校准后的判决规则(2026-07-07 判定落地版)

1. **任何离线维度数字(除 world 方向)不得作为收益证据出现在预测表/上传论证里。** 包出门判据回归三件套:平台真分历史 + 预登记机制预测(过红队:哪个已知机制能杀死它)+ precheck 结构保险丝。
2. **world 维(CMMLU 圈外 Acc)可作方向参考**:适用于世界知识数据增删的回归检查;只说方向,禁止数字承诺。
3. **离线台的存活价值(降级重定位)**:①**回归保险丝**——fmt_alive/json_ok/trunc_rate/断裂类结构指标,拦"变坏",不证"变好";②**离群检测**——如 recipe2 action 0.24 十倍离群=当场抓到分布内记忆,新模型任何维度出现离群读数先查数据圈内性;③**校准框架常青**——新平台面板出分回填 TRUTH 重跑,若未来拿到更好的 dev 源(如平台日志题面批量泄出),重校准即可翻案。
4. 校准锚集随新面板扩充;dev 集分池制度(§2)长期有效。

---

## §8 平台可见题提交前门禁(2026-07-11)

### 题面来源与稳定性

- 已扫描 `logs/eval/` 20份平台评测日志。每份日志在8个任务中各打印 `Sample ID=0..4`、完整 `Input` 和模型 `Output[0]`,共40道可见样例。
- common_sense、action_select、topic_gen、video/prod/ad/live 推荐的5个题面在20份日志中逐题固定;itemic_pattern_grounding 每个 sample ID 有2个历史版本,07-01后使用新版本。
- 将最新40道题面按去空白规范化后,与 `data_final`、`data_stage2_gold_v1`、`stage2_gold_v1_holdout` 和全部 `data/offline_eval/dev_*.jsonl` 核对,精确重叠均为0。
- 日志不打印 action/topic/material/rec 的 gold。common_sense 可人工标注;其余题只允许做结构与行为回归,不得计算或宣称平台分数。

### 冻结规则

1. 可见题永不进入训练、蒸馏、拒绝采样或数据筛选;只从原始平台日志读取,不另造“增强版”。
2. 候选必须与锚模型在同一推理后端、chat template、seed、temperature/top-p/top-k、max tokens和think开关下配对;后端不一致的结果作废。
3. 可见题只能提供**否决证据**:明显退化可以拦截上传,持平或改善不能证明会上分。5题样本量不能替代574/807/905/1000/1739道正式评测。
4. 首选锚为当前上传目标的直接父模型;跨制度比较同时保留 `seed_ep3`(物料8题)与 `riders_fk_lora_ep1`(总分0.9177)两锚。

### 分任务门禁

| 任务 | 可见题检查 | 硬否决示例 | 当前实现 |
|---|---|---|---|
| material | sample3 beam64 的目标 `s_a` 锁定数及其下 `s_b` 扇宽 | 相对父模型锁定/扇宽断崖式下降 | `scripts/eval/probe_v2.py` |
| action | JSON数组闭合、生成长度、重复、历史引用率、5题逐题输出 | 0/5 JSON、长循环或乱码 | `scripts/eval/replay_visible_action.py` |
| topic | logic_chain JSON、events 1-5步、date/action/logic字段、重复/截断 | 结构全灭或较父模型新增截断 | 待统一门禁入口 |
| rec四域 | SID三元组格式、候选有效性、重复、抄史率、异常长thinking | 域token/SID结构断裂或长循环 | 待统一门禁入口 |
| world | 5题人工gold准确率、官方答案格式、首个答案抽取 | 格式存活或准确数低于父模型 | 待人工gold+统一入口 |

### 首次案例:`stage2_gold_v1_lora_ep1`

- 训练数据8,030行、material0,从 `seed_ep3` 增量LoRA。独立holdout中world方向变好,但action F1下降、重复恶化,ad/live回落。
- 平台固定5道action题回放为 **0/5 JSON**,共生成18,342 token,原始输出为符号/域begin token循环。报告:`logs/probe/visible_action_stage2_gold_v1_platform5_20260711.json`。
- 物料sample3从 `seed_ep3` 标定签名锁定55/64、扇宽18降到锁定14/64、扇宽7。按既定门槛属于明确负向门禁,但仍不外推线上题数。
- 结论:该候选不上传、不续训、不作warm start。该结果是“无物料增量集仍可发生能力漂移”的本地反例,不是平台线上分数。

### 实现状态

- 已完成:题面稳定性核对、训练/dev精确重叠审计、action固定5题回放、material单题beam签名。
- 待完成:将8任务抽取、锚/候选配对、阈值判决和单份JSON报告收口到一个只读入口;完成前仍按现有两个脚本分别运行。
