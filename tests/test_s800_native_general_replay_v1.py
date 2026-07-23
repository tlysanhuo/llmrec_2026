import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_SCRIPT_DIR = ROOT / "scripts/data"
if str(DATA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPT_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "s800_native_general_replay_builder",
    ROOT / "scripts/data/build_s800_native_general_replay_v1.py",
)
trainer = load_module(
    "s800_native_general_replay_trainer",
    ROOT / "scripts/train/train_s800_native_general_replay.py",
)

DATA = ROOT / "assets/derived/processed/data_s800_native_general_replay_v1.jsonl"
MANIFEST = (
    ROOT / "assets/derived/official_general/s800_native_general_replay_v1_routes.json"
)
RETENTION_GATE = (
    ROOT
    / "assets/evaluation/holdout/s800_native_general_replay_retention_gate_v1.jsonl"
)
CHECKPOINT_GATE = (
    ROOT / "configs/evaluation/s800_native_general_replay_checkpoint_gate_v1.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_replay_assets_have_exact_route_and_task_contract():
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8") if line.strip()]
    assert len(rows) == 513
    assert Counter(row["route"] for row in rows) == {
        "general_ce": 129,
        "retention_kl": 384,
    }
    assert Counter(
        row["task"] for row in rows if row["route"] == "retention_kl"
    ) == builder.RETENTION_QUOTAS
    assert len({row["record_id"] for row in rows}) == 513
    assert sha256(DATA) == trainer.EXPECTED_TRAINING_SHA256


def test_route_manifest_is_fail_closed_and_matches_trainer_lock():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    general = set(manifest["general_ce_target_sha256"])
    retention = set(manifest["retention_kl_target_sha256"])
    assert len(general) == 129
    assert len(retention) == 384
    assert not general & retention
    assert manifest["cross_route_target_sha256_collisions"] == 0
    assert sha256(MANIFEST) == trainer.EXPECTED_ROUTE_MANIFEST_SHA256
    loaded = trainer.load_route_manifest()
    assert loaded == {"general": general, "retention": retention}


def test_response_and_prompt_normalization_contracts():
    static = {
        "instruction": "",
        "input": "为什么会出现潮汐？/think",
        "output": "<think>分析引力。</think>月球和太阳引力会产生潮汐。",
        "history": [],
    }
    assert builder.valid_response_structure(static["output"])
    assert "/think" not in builder.normalized_prompt_key(static)
    assert not builder.valid_response_structure("直接回答")
    assert not builder.valid_response_structure("<think></think>")


def test_trainer_math_and_full_route_guard_self_test():
    trainer.run_self_test()


def test_retention_gate_is_balanced_and_prompt_task_disjoint():
    train = [json.loads(line) for line in DATA.open(encoding="utf-8") if line.strip()]
    gate = [
        json.loads(line) for line in RETENTION_GATE.open(encoding="utf-8") if line.strip()
    ]
    assert len(gate) == 256
    assert Counter(row["task"] for row in gate) == {
        task: 32 for task in builder.RETENTION_QUOTAS
    }
    train_keys = {
        (row["task"], builder.normalized_prompt_key(row))
        for row in train
        if row["route"] == builder.RETENTION_ROUTE
    }
    gate_keys = {(row["task"], builder.normalized_prompt_key(row)) for row in gate}
    assert len(train_keys) == 384
    assert len(gate_keys) == 256
    assert not train_keys & gate_keys


def test_checkpoint_gate_was_frozen_before_training():
    gate = json.loads(CHECKPOINT_GATE.read_text(encoding="utf-8"))
    assert gate["status"] == "PREREGISTERED_BEFORE_FORMAL_TRAINING"
    assert gate["formal_training"]["candidate_steps_in_order"] == [32, 64, 96, 129]
    assert gate["retention_gate"]["sha256"] == sha256(RETENTION_GATE)
    assert gate["implementation_lock"]["formal_training_started_at_registration"] is False
