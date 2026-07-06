# Baseline 复现笔记 & 上传说明 (baseline_sft_v1)

> 生成于 2026-07-01。这是复现官方 LLaMA-Factory baseline、拿首个平台评测分数的完整记录。
> 相关文档：`docs/platform_and_baseline.md`（规则/平台）、`docs/data_report.md`（数据）、计划 `lovely-popping-starfish.md`。

## 训练结果

| 项 | 值 |
|---|---|
| 基座 | OneReason-0.8B-pretrain-competition (Qwen3-0.6B 架构 + itemic 词表, 0.801B 参数) |
| 框架 | LLaMA-Factory 0.9.6.dev0, 全量 SFT, bf16, packing+neat_packing, FA2, Liger |
| 数据 | 官方全量 3 赛道 SFT，32,480 条 → convert_jsonl → alpaca；packing 后 1,597 序列 |
| 硬件 | 2×H100 (GPU 3,6), DDP |
| 超参 | template=qwen3_nothink, cutoff_len 32768, lr 2e-5, 1 epoch, cosine, warmup 0.03, seed 19260817 |
| global batch | bs1 × grad_accum2 × 2gpu = **4**（对齐官方单卡 bs1×accum4） |
| 步数/耗时 | 400 steps, **13分26秒** |
| **train_loss** | **1.573**（从 2.9 收敛） |
| 产物 | `checkpoints/baseline_sft_v1/`（含 loss 曲线 training_loss.png、tensorboard runs/） |

### 环境偏差（记录备查）
- 官方 demo 用 transformers 4.x；LLaMA-Factory 0.9.6.dev0 的最新依赖装成 **transformers 5.6.0**。已验证 OneReason 模型 + qwen3_nothink 模板在 5.6 下加载/训练/推理均正常。
- torch 2.7.1+cu126 / flash-attn 2.7.4.post1 / liger 0.8.0 均按官方 00_install.sh pin。
- 打了官方的 flash_attention `s_aux None`-guard 补丁。

### 预处理验证结论（训练前已确认）
- prompt 段（含 `/think`·`/no_think` 后缀 + assistant header）loss 正确 mask 为 -100。
- `/think` 样本监督完整 `<think>...</think>`；`/no_think` 样本监督空 `<think>\n</think>`+答案。
- itemic token 全部单 id，未被 BPE 切碎。qwen3_nothink 的 ReasoningTemplate 与已含 think 的数据配合正确。

### 推理自检（训练后）
- 懂物料 desc→token：正确输出空 think + 单个 `<|prod_begin|><s_a><s_b><s_c>`。
- 懂物料 token→desc：输出结构化商品描述（含类目层级）。
- 懂推荐：`/no_think` 下输出 `该用户最近喜欢的视频有: <|video_begin|>...` 单 item。

## 上传包：`submissions/baseline_sft_v1_upload/`

**内容**（全参模型，1.6G）：`model.safetensors`（训练权重）+ `config.json`/`generation_config.json`（**用 base 原版 verbatim**，保证评测 config 校验通过——训练产物的 config 只是 tf5.6 键名重写，架构逐键与 base 完全一致）+ tokenizer 全套（base 原版，全参 SFT 不改分词器）+ chat_template.jinja。

**config 一致性**：核心架构 13 个关键项（architectures/model_type/hidden_size/层数/heads/head_dim/intermediate/vocab_size/max_pos/rms_norm_eps/tie_embeddings/hidden_act）与 base **完全相同**，已校验。单文件 safetensors，无需 index.json（已移除，避免 total_size 元数据不一致）。

**加载验证**：直接从上传包目录 `AutoModelForCausalLM.from_pretrained` 加载成功（0.801B，vocab 176253），生成正常。

## 上传与提交步骤（用户在万擎平台操作）

平台唯一站点：https://www.streamlake.com/product/wanqing （队长账号，已实名认证）

1. **上传模型**：左侧「模型仓库」→「上传模型」。
   - 训练方法选 **全参**（非 LoRA）。
   - 上传 `submissions/baseline_sft_v1_upload/` 内**全部文件**（至少 `model.safetensors` + `config.json`；建议连 tokenizer/generation_config 一起传）。
   - 发布方式：新模型（命名如 `baseline_sft_v1`）。
2. **提交评测**：仓库中选该模型 →「去评测」→ 新建竞赛评测任务。
   - ⚠️ 每日限 3 次；评测失败不消耗次数，可复制重试。
3. **查看得分**：「模型评测」→「竞赛评测」列表，看总分 + 懂物料/懂用户/懂推荐/懂世界四维度分。回填此处作为 baseline 锚点。

## 如何从零复现（复现审核用）

```bash
export REPRO=/root/baseline_repro          # overlay 盘工作区
bash scripts/baseline/00_install.sh        # 建 py3.11 venv + 官方依赖栈
bash scripts/baseline/01_convert_data.sh   # 12 jsonl -> data_final.jsonl (seed 2026)
REPRO=$REPRO python scripts/baseline/02_register_dataset.py
REPRO=$REPRO GPUS=3,6 bash scripts/baseline/03_train.sh   # 全量 SFT
```
配置：`configs/baseline/baseline_sft_v1.yaml`。依赖冻结：`configs/baseline/requirements.frozen.txt`。

## 待办 / 下一步
- 用户上传评测 → 得四维度分（baseline 锚点）。
- 之后迭代方向（另行规划）：数据构造（自造 non-think 推荐、懂世界用 OneReason_General、R1/I2I 自造）、防遗忘、SFT 配比、RL pipeline。
