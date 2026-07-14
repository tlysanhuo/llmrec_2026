# Config Policy

配置按生命周期分区，不能根据文件名或修改时间判断是否可训练。

## 新训练

新实验配置只能放在 `configs/active/`，并同时满足：

- idea 已登记到 `ideas/README.md`；
- 单卡；
- `num_train_epochs: 1`；
- `save_strategy: "no"`；
- `report_to: wandb`；
- 数据路径来自 `assets/` 注册资产；
- 输出目录是唯一 run name；
- 训练前通过 dataset key 和文件存在性检查。

- `configs/active/`：唯一允许启动的新实验配置；当前为空。
- `configs/retained/`：当前保留模型的原始配方，不代表允许重训。
- `configs/history/`：已完成、失败或搁置的历史配方。
- `configs/baseline/`：官方基线复现配置。
- `configs/datasets/`：需要独立 registry 的已登记数据。
- `configs/evaluation/`：离线评测协议、前瞻判决台账和候选证据；不是训练配置，禁止被训练启动器读取。

## 已知不可直接复现

- `history/run_a_r2.yaml`：`run_a_mix` registry key 已不存在。
- `history/run_c_material.yaml`：`run_c_mix` registry key 已不存在。
- `history/run_d_r2material.yaml`：`run_d_mix` registry key 已不存在。

这些文件只作历史记录，禁止直接启动。
