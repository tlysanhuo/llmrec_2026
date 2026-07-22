# I-35 video boundary residual handoff

## 结论先说

I-35不是直接训练`rec_video`。训练侧的video来自O1 `懂物料part4`，任务是“视频描述 -> 三段video SID”。线上相对直接父模型的主要变化是：

| 子项 | 父模型 | I-35 step548 | 变化 |
|---|---:|---:|---:|
| material | 0.2453 | 0.2453 | 0 |
| rec_video | 0.0768 | 0.0864 | +0.0096 |
| rec_prod | 0.1326 | 0.1394 | +0.0068 |
| rec_ad | 0.1400 | 0.1386 | -0.0014 |
| rec_living | 0.1080 | 0.1071 | -0.0009 |
| world | 0.1602 | 0.1591 | -0.0011 |

因此`1.0344285849069457`是一次跨任务迁移结果，不能写成“直接优化rec_video”或“material跳档”。

## 原实验怎样使用video

1. 从O1 `懂物料part4`的1,621行开始，把think/no-think统一成平台真实的空think、system/user模板，排除与登记E资产同题面或同mode的251行，得到1,370行E-clean池。
2. 用直接父模型I19-world r96对1,370行运行Beam128，只解码video-domain之后的A/B/C三个SID token。父模型Top128完整gold命中230行，其中66行的gold位于1-based rank 65--128。
3. 这66行是boundary：只在A/B/C位置做first-divergence margin `0.1`、gold CE权重`0.05`和父KL权重`0.10`。负例只来自父模型Beam128，不把其他模型候选当正例。
4. 其余1,304行material只做权重`4.0`的父KL；另加入1,370行七任务retention，也只做权重`4.0`的父KL。正式混合严格为material/retention=`1:1`。
5. 在merge后的r96父模型上训练fresh r16；batch1、accum4、LR `1e-5` cosine、685步，保存137/274/411/548/685。step548 r16与r96按scale1精确拼成r112。

训练日志、W&B结果和完整数据审计位于：

- `assets/derived/releases/i35_r96_video_boundary_retkl_r16_v1/logs/`
- `assets/derived/releases/i35_r96_video_boundary_retkl_r16_v1/audits/`
- W&B run `0b4p3siy`

## 查看和恢复原实验

```bash
scripts/reproduce/i35_video_boundary_release.sh verify-data
scripts/reproduce/i35_video_boundary_release.sh restore-original-data
scripts/reproduce/i35_video_boundary_release.sh self-test
```

原正式数据和sidecar严格绑定父adapter SHA256：

```text
4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e
```

## 换成你自己的检查点

不要复用原始正式数据里的66条boundary标签。边界rank、负例和parent KL都依赖父模型，换父后必须重算。

1. 先恢复公共E-clean池和retention源：

```bash
scripts/reproduce/i35_video_boundary_release.sh restore-pool
scripts/reproduce/i35_video_boundary_release.sh restore-retention
```

2. 令Beam runner的parent和teacher都指向你的同一个adapter。I-35不使用I-23 teacher：

```bash
python scripts/eval/generate_i35_video_material_beam128_v1.py \
  --gpu 0 \
  --parent-adapter /path/to/your_parent_adapter \
  --teacher-adapter /path/to/your_parent_adapter \
  --train-input logs/data/i35_video_material_beam128_pool_v1.jsonl \
  --dev-input logs/data/i35_video_material_beam128_pool_v1_dev.jsonl \
  --train-output logs/data/your_parent_i35_train_ledger.jsonl \
  --dev-output logs/data/your_parent_i35_dev_ledger.jsonl \
  --audit-output logs/probe/your_parent_i35_beam128_audit.json \
  --gpu-memory-utilization 0.85
```

3. 为你的实验复制一份builder、trainer、config和dataset registry，至少替换：

- `PARENT_ADAPTER_SHA256`和`PARENT_CONFIG_SHA256`；
- Beam ledger/audit路径；
- `adapter_name_or_path`、`output_dir`和`run_name`；
- 正式数据、sidecar、audit输出名；
- launcher中的父路径和所有预期哈希。

原I-35文件故意锁死原父身份，避免把旧sidecar误用于新父模型。不要直接放宽或删除这些身份门。

4. 保持初始对照为fresh r16和相同loss权重。父rank为`r`时，最终拼接rank是`r+16`，必须满足平台上限`r+16 <= 128`。例如r96 -> r112，r80 -> r96。

5. 训练后使用精确参数拼接：

```bash
python scripts/train/combine_lora_adapters.py \
  /path/to/your_parent_adapter \
  /path/to/your_fresh_r16/checkpoint-548 \
  /path/to/output_combined_adapter \
  --audit /path/to/output_combined_adapter.audit.json
```

step548约为0.8个epoch，只是原父模型上的成功剂量。换父后不能假设它仍最优；至少保留step411和step548作预登记的中/强剂量对照，不应看线上结果后继续补点。

## 可归因边界

- 可确认：step548相对直接父模型总分`+0.009169129`，推荐四项合计`+0.0141`。
- 不可确认：这不是隔离的rec_video CE因果效应，也没有使material从`0.2453`升档。
- 复用条件：新父必须重新生成Beam128 ledger、boundary sidecar和父KL目标；只复用E-clean输入池、retention配方和训练机制。
