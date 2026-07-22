# eval — 统一评测模块

`one_reason/eval/` 是独立于 `llmrec_2026-main/` 的评测项目，完整覆盖官方定义的五个评测维度：
懂物料、懂用户（F1 / 逻辑链）、懂推荐、懂世界。评测流程本身（跑 vLLM 推理、计算指标）不依赖
外部大模型 API；`common/llm_client.py` 封装的多渠道并行 API 客户端仅作为未来"复现官方两步
标注/教师裁决"式 gold 构造场景的预留能力，按需调用。

不修改、不依赖修改 `llmrec_2026-main/` 下的任何文件，只读复用其数据资产。

## 目录结构

```
eval/
  README.md
  common/
    llm_client.py     # 多渠道并行 LLM 客户端（TokenBucket 限速 + pick_channel 分配 + 线程池并发），预留扩展点
    prompt.py          # 统一 prompt 构造（system/user 模板、/no_think、/think）与采样参数
    engine.py           # vLLM 推理封装：beam_decode / sample
    text_match.py       # tok_f1、rouge_l（LCS）、Dice 系数、最优有序匹配（DP 保序）
    sid_utils.py        # SemanticID token 解析 + Pid2Sid 反查表构建/查询
    text_utils.py       # think 段剥离、JSON 数组/对象提取
  metrics/
    material.py         # 懂物料 Pass@64
    user_f1.py           # 懂用户-兴趣行为抽取 F1
    user_chain.py         # 懂用户-逻辑链：Action Alignment + Logic Alignment
    recommend.py          # 懂推荐：双路候选合并 Pass@64
    world.py               # 懂世界：Accuracy + 正则答案抽取
  data/
    loaders.py           # 只读加载各维度 dev 数据（官方 sampled 数据 + Pid2Sid 反查表）
  run_eval.py             # 统一 CLI 入口
  tests/                   # 各模块对应的单元测试
  output/                   # 评测结果落盘目录（含 Pid2Sid 索引缓存 output/.cache/）
```

## 五个评测维度

1. **懂物料**（`metrics/material.py`）：beam64 生成候选 SemanticID pattern，通过 Pid2Sid
   反查表映射为 item_id 集合，判定 Pass@64（候选集合与 gold item_id 是否有交集）。
2. **懂用户-F1**（`metrics/user_f1.py`）：模型抽取的兴趣相关行为 JSON 数组与 gold 比较，
   算标准 F1（`2PR/(P+R)`）。
3. **懂用户-逻辑链**（`metrics/user_chain.py`）：标准链与生成链的 events 按 action 做
   "最优有序匹配"（DP 保序，相似度矩阵元素为逐 event 的 Dice 系数），得到匹配对后：
   - Action Alignment：由匹配对的 Dice 系数走 P/R/F1 得出；
   - Logic Alignment：对匹配对的 logic 文本算 `0.5*TokenF1 + 0.5*ROUGE-L`，再走 P/R/F1；
   - 综合分 = 两者均值。
   实现已用官方文档给出的示例表格数值（0.667/1/1/0/0 五条连线 → Action Alignment ≈ 0.593）
   做单元测试验证。
4. **懂推荐**（`metrics/recommend.py`）：think 路与 no-think 路各生成 32 条候选，去重合并成
   64 候选池，任一命中 gold 即算 Pass@64。
5. **懂世界**（`metrics/world.py`）：Accuracy + 按优先级正则链抽取答案字母，同时支持单选与
   多选（多选要求预测字母集合与 gold 完全一致，少选/错选/多选/解析失败均判错）。

Logic Alignment 当前按官方公式实现（不含 NLI 项），`text_match.logic_similarity` 预留了
`use_nli` 开关，待后续安装 NLI 模型（如 `nli-deberta-v3-base`）后再决定是否接入。

## 数据来源

各维度数据均直接读取官方 sampled 数据，只读、不改动：

| 维度 | 数据源 |
|---|---|
| 懂世界 | `demo/baseline-data/baseline_data/sampled/懂世界_from_mc.jsonl` + `懂世界.jsonl` |
| 懂用户-F1 / 逻辑链 | `demo/baseline-data/baseline_data/sampled/懂用户.jsonl`（按是否含 `logic_chain` 拆分两个子集） |
| 懂物料 | `demo/baseline-data/baseline_data/sampled/懂物料part1~4.jsonl`（text→token 方向） |
| 懂推荐 | `demo/baseline-data/baseline_data/sampled/懂推荐1~4.jsonl` |

懂物料 / 懂推荐的 GT SemanticID 通过 `data/OneReason_Pid2Sid/part-*.parquet` 反查表映射为
真实 item_id（索引会缓存到 `eval/output/.cache/pid2sid_index.pkl`）。由于该反查表与
sampled 数据是从物料池中分别独立采样得到的两批数据，部分样本的 SID 查不到对应 item_id 属于
正常现象，`data/loaders.py` 会将这类样本过滤掉并计入 `n_gt_map_failed`，不影响其余样本的
评测有效性。

## 使用方法

```bash
# 跑全部维度（self-check 模式：不提供 --model 时，用 gold 本身模拟预测，用于验证流水线连通性）
python3 run_eval.py --dry-run

# 只跑指定维度，限制样本数
python3 run_eval.py --dims world,material --limit 10 --dry-run

# 提供被测模型跑真实推理
python3 run_eval.py --dims user_chain --model /path/to/checkpoint --gpu 0
```

评测结果默认写入 `eval/output/<tag>_<时间戳>.json`（统一 JSON schema，含每个维度的
`available`/`n`/对应指标均值）；加 `--dry-run` 则只打印到 stdout 不落盘。

## 测试

```bash
cd eval && python3 -m pytest tests/ -v
```

测试区分两类：A 类直接复现官方文档给出的数值算例（如懂用户-逻辑链 0.593、懂世界三角形
数对原题精确抽取出 B）；B 类是文档未给出具体数值、依据文档规则自行构造的边界场景测试。
