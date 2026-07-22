#!/usr/bin/env python3
"""tests/test_loaders.py — data/loaders.py 数据加载测试。

【2026-07-18 修订】此前版本假设官方 sampled 数据与 Pid2Sid 反查表"不可达"，因此
测试只验证"能优雅降级"。经实测确认这批官方文件在当前环境完全可读（详见
`eval/SPEC.md` 附录"忠实度对比评估"），`data/loaders.py` 已改为直接对接：
  - 懂世界：`懂世界_from_mc.jsonl`（272条）+ `懂世界.jsonl`（7条）
  - 懂用户-F1：`懂用户.jsonl` 中不含 logic_chain 的行（约1588条）
  - 懂用户-逻辑链：`懂用户.jsonl` 中含 logic_chain 的行（约1304条）
  - 懂物料：`懂物料part1~4.jsonl`（合计约5597条）+ Pid2Sid 反查表映射真实 item_id
  - 懂推荐：`懂推荐1~4.jsonl`（合计约19204条）+ Pid2Sid 反查表映射真实 item_id

本文件断言这些真实数据源均可用（`available=True`），且各字段结构符合预期；
同时保留"未知维度报错"等边界测试。

注：懂物料/懂推荐用例首次运行需要构建 Pid2Sid 反查表（约2分钟，会缓存到
`eval/output/.cache/pid2sid_index.pkl`，之后的测试运行会直接复用缓存，数秒内完成）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders import load_dev, load_material_dev, load_recommend_dev, load_user_chain_dev, load_user_f1_dev, load_world_dev


def test_world_dev_loads_real_sampled_data():
    """懂世界数据应从官方 sampled 数据（懂世界_from_mc.jsonl + 懂世界.jsonl）加载，
    每条记录含 prompt/gold 字段，gold 为升序字母串（单选单字符，多选多字符）。
    """
    result = load_world_dev(limit=5)
    assert result["available"] is True
    assert len(result["items"]) > 0
    item = result["items"][0]
    assert "gold" in item
    assert isinstance(item["gold"], str) and len(item["gold"]) >= 1
    assert set(item["gold"]) <= set("ABCD")


def test_user_f1_dev_loads_real_data():
    """懂用户-F1 数据应从官方 `懂用户.jsonl` 中筛出不含 logic_chain 的行，gold为list。"""
    result = load_user_f1_dev(limit=5)
    assert result["available"] is True
    assert len(result["items"]) > 0
    item = result["items"][0]
    assert isinstance(item["gold"], list)
    assert "logic_chain" not in "".join(item["gold"]) if item["gold"] else True


def test_user_chain_dev_loads_logic_chain_events():
    """懂用户-逻辑链数据应从 `懂用户.jsonl` 中筛出含 logic_chain 的行。"""
    result = load_user_chain_dev(limit=5)
    assert result["available"] is True
    assert len(result["items"]) > 0
    item = result["items"][0]
    assert isinstance(item["gold_events"], list)
    assert len(item["gold_events"]) > 0
    assert "action" in item["gold_events"][0]
    assert "logic" in item["gold_events"][0]


def test_material_dev_uses_real_item_ids_via_pid2sid():
    """懂物料数据应通过 Pid2Sid 反查表映射出真实 item_id 集合（不再是代理哈希值）。"""
    result = load_material_dev(limit=20)
    assert result["available"] is True
    assert len(result["items"]) > 0
    item = result["items"][0]
    assert "gold_item_ids" in item
    assert isinstance(item["gold_item_ids"], list)
    assert len(item["gold_item_ids"]) > 0
    assert all(isinstance(x, int) for x in item["gold_item_ids"])
    assert "pattern" in item
    assert item["pattern"].startswith("<|")  # 完整带 domain 前缀的 SID token
    assert "item_id_is_proxy" not in item  # 不再使用代理值标记


def test_recommend_dev_loads_real_data_via_pid2sid():
    """懂推荐当前应能从官方 sampled 数据 + Pid2Sid 反查表加载出真实可用样本。"""
    result = load_recommend_dev(limit=20)
    assert result["available"] is True
    assert len(result["items"]) > 0
    item = result["items"][0]
    assert "gold_item_ids" in item
    assert len(item["gold_item_ids"]) > 0
    assert "prompt_think" in item and item["prompt_think"].rstrip().endswith("/think")
    assert "prompt_nothink" in item and item["prompt_nothink"].rstrip().endswith("/no_think")


def test_load_dev_dispatches_by_dim_name():
    for dim in ("material", "user_f1", "user_chain", "recommend", "world"):
        result = load_dev(dim, limit=1)
        assert "available" in result
        assert "items" in result


def test_load_dev_rejects_unknown_dim():
    import pytest

    with pytest.raises(ValueError):
        load_dev("not_a_real_dim")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
