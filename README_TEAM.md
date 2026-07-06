# 快手 LLM-Rec 2026 — 团队工作仓(初赛)

> 本仓 = 文档 + 全部可复现脚本/配方。**不含**数据成品与权重(用 `scripts/data/` 重建)、评测日志、任何密钥。
> 当前最高分:**riders_fk_lora_ep1 = 0.9177**(2026-07-06)。

## ★ 先读:面板域序重标(2026-07-06 官方 FAQ 定案)

平台评测面板四个"懂推荐"列从左往右 = **短视频(video)、电商(prod)、广告(ad)、直播(live)**。
在此之前我们(可能也包括你)按 ad/live/prod/video 误读。**读任何历史记录先过换算表**:

| 面板列 | 真实域 | 每命中分值(量子) | 观测天花板 |
|---|---|---|---|
| rec 第1列 | video | 0.0096 | 10 命中(lr1e-4 制度) |
| rec 第2列 | prod | 0.0034 | 40(seed 3ep) |
| rec 第3列 | ad | 0.0014 | 107(lr1e-6 LoRA) |
| rec 第4列 | live | 0.0009 | 122(riders) |

物料量子 0.0307(1 题=3.2 个 video 命中)。各 rec 域题量相同(~1000/域),量子差=官方难度权重。
**核心矛盾改写:跷跷板 = 物料 ↔ video**(低 lr 多 epoch 全参把 video 打到 4-7;LoRA/高 lr 保 video)。ad 从来没塌过。

## 当前最优配方(0.9177,可全复现)

数据(37267 条,`scripts/data/build_riders_fk.py`):
- Frinkleko 重组种子 32705(同 prompt 同 think 组留 1 条、其余转 nothink 直出 + CEval 1573)——HF 公开数据集,0.9107 线上验证
- - world_zh 2824(通识,两次线上验证 +0.011/+0.013)
- - P3 quote-and-stop 1500(action 专项)+ world_mc 238(MC 格式锚,治复制训练把选择题带崩)

训练(`configs/riders_fk_lora_ep1.yaml`):LoRA r32/α32/dropout0.05,lr 2e-4,**1 epoch**,seq32768,packing,qwen3_nothink 模板,liger on。30 分钟/单卡。
面板(重标后):mat 7题 / action 0.0655 / topic 0.0427 / video 8 / prod 37 / ad 99 / live 122 / world 0.1439。

## 毒物清单(都有线上真分尸检,勿再踩)

1. **重复上采样**(纯复制样本 2.76×):物料 2146→1840,全线下跌(rebal_mat 0.8454)。上采样必须造新样本。
2. **HF 墙外物料数据**(desc→SID 的 item 不在种子分布内):把映射函数整体拽偏,物料直坠 1533(pstack_v2 0.8265)。官方 FAQ Q5 已解释:SFT/评测物料与 HF 是两批采样,**HF 物料永远覆盖不到评测题**。48k/8947 物料数据同判。
3. **全域剥 CoT**:懂世界归零(recipe3)。ad 域单独剥未测。
4. **focal loss γ=2**:itemic 结构断裂+选择题崩,零收益(rebal_focal)。
5. **解冻 embedding/lm_head**:接口层漂移打断全部冻结回路,物料掉档+video 跌回(fk_lora_embed 0.8672)。
6. **答案截断式数据**(答案止于 s_b 无 s_c):教会模型提前终止,ad −13 命中(tokengeo 0.8338)。
7. **低 lr(2e-5)多 epoch 全参**:物料最强(8 题)但 video 交死税(4-7 命中,−0.03)。
8. **复制类训练(P3 等)不配 MC 锚**:选择题吐占位符,world 崩(rebal_pstack 未传)。P3:MC 锚 ≈ 1500:1800 安全。

## 有效部件(线上验证,可叠加)

- Frinkleko nothink 重组(官方 Tips 点名 CoT/UnCoT 配比方向)/ LoRA r32 lr2e-4 1ep 制度(保 video+ad 先验)
- world_zh 2824(+0.011~0.013)/ CEval+world_mc 选择题锚 / P3 1500(action +0.02~0.035,制度相关)
- 多 epoch(3-5)全参:物料 8 题唯一持有者,但 video 交税——制度二选一
- 纯种子 5ep:ad +6 命中(95→101),物料墙 8 题(4/5ep 分毫不差,遍数手段报废)

## 读分纪律

- 每日 3 次评测,**账号级共享**,北京 15:00 刷新;**评估须 12:00 前结束**→上传实操截止≈10:30。只收 **bf16**。
- 方差(自测标定+官方承认):总分 ±0.03 内比较无效;video 列 ±2 命中不作数;物料/ad/live 接近零噪声;分项同涨才可信。
- 格点读分:分数 = 命中数 × 量子,先换算成整数命中数再比较。

## 目录

- `docs/experiment_log.md` 全部实验+归因(顶部有重标警告);`docs/EXPERIMENT_INDEX.md` 产物对账
- `docs/platform_guide.md` 平台/官方 FAQ/评测机制;`docs/TODO.md` 在制品
- `scripts/data/` 全部数据构建器(R2/P3/world_zh/rebal/riders/蒸馏);`scripts/eval/` precheck+offline_probe+日志分析
- `configs/` 全部训练配方 yaml(锚配置 config_diff 对账:`scripts/train/config_diff.py`)

## 环境速记

训练:LLaMA-Factory 0.9.6 + torch2.7 venv,单卡 H100 即可;推理探针:vllm 0.12 env。
数据原料:HF `OpenOneRec/Explorer_LLM_Rec_Competition`(17G)+ 官方种子 SFT;index 用 `scripts/data/build_item_index.py` 重建。
蒸馏 teacher:DeepSeek v4-flash(合规:官方允许蒸馏,需留数据供复现——脚本+产物都在)。
