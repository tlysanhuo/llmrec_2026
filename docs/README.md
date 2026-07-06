# docs/ 文档管理规则(2026-07-05 用户整顿后定)

**顶层只允许 7 个活文档,新建文档前先问:能不能写进这 7 个里?**

| 文件 | 职责 | 更新时机 |
|---|---|---|
| `TODO.md` | 唯一 TODO | 状态一变就改 |
| `experiment_log.md` | 实验台账(预登记+出分回填) | 每次训练/上传/出分 |
| `EXPERIMENT_INDEX.md` | 产物对账(config/ckpt/包/日志 md5) | 每次产出新工件 |
| `platform_guide.md` | 平台侧唯一权威(规则/评测机制/默认参数) | 平台有新信息 |
| `competitor_intel.md` | 选手群情报 | 有新情报 |
| `teammate_log.md` | 队友提交与共享 | 队友有动作 |
| `experiment_report_for_team.md` | 对外分享版总结 | 大版本节点 |

- `reference/`:稳定参考(论文解读、数据盘点、蓝图、文献)。基本只读。
- `archive/`:已被吸收进台账/记忆的历史分析(含旧平台文档 platform_intro_v2 / platform_and_baseline,内容已并入 platform_guide)。**分析类文档的生命周期:写完 → 结论进 experiment_log/记忆 → 本体进 archive。**

铁律:不再新建 `xxx_analysis_日期.md` 式顶层文档;临时分析产物放 /tmp 或直接写进台账对应条目。
