# docs/ 文档管理规则(2026-07-05 用户整顿后定)

**顶层只保留工作入口文档；idea 和分享进入项目根 `ideas/`，历史分析进入 `archive/`。**

| 文件 | 职责 | 更新时机 |
|---|---|---|
| `TODO.md` | 唯一 TODO | 状态一变就改 |
| `experiment_log.md` | 实验台账(预登记+出分回填) | 每次训练/上传/出分 |
| `EXPERIMENT_INDEX.md` | 产物对账(config/ckpt/包/日志 md5) | 每次产出新工件 |
| `platform_guide.md` | 平台侧唯一权威(规则/评测机制/默认参数) | 平台有新信息 |
| `experiment_report_for_team.md` | 对外分享版总结 | 大版本节点 |
| `offline_eval.md` | 离线评测与行为门禁协议 | 评测机制变化 |
| `WORKSPACE.md` | 目录职责和写入规则 | 目录结构变化 |

- `reference/`:稳定参考(论文解读、数据盘点、蓝图、文献)。其中 `ASSETS.md` 是唯一资产注册表，`OFFICIAL_DATA_EDA.md` 是 O1–O6 封板分析。基本只读。
- `archive/`:已被吸收进台账/记忆的历史分析(含旧平台文档 platform_intro_v2 / platform_and_baseline,内容已并入 platform_guide)。**分析类文档的生命周期:写完 → 结论进 experiment_log/记忆 → 本体进 archive。**
- `../ideas/`:活跃 idea、选手分享、队友分享、EDA 和历史方案。

## 资产唯一入口

- `reference/ASSETS.md` 是唯一权威资产注册表，严格区分官方直发、官方源派生、第三方、评测衍生和模型产物。
- 开始数据或训练工作时先读注册表，不再递归扫描全盘确认“有哪些数据”。
- `reference/DATA_INVENTORY.md` 已废止，仅保留跳转，禁止继续维护第二套资产口径。

铁律:不再新建 `xxx_analysis_日期.md` 式顶层文档;临时分析产物放 /tmp 或直接写进台账对应条目。
