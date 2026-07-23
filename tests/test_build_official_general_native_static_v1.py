import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/data/build_official_general_native_static_v1.py"
SPEC = importlib.util.spec_from_file_location("official_general_native_static", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)
base = builder.base


def test_native_think_route_matches_official_contract():
    prompt, response, mode, reasons = builder.route_native(
        "电磁波是什么？",
        "<think>先分析定义。</think>电磁波是振荡电磁场在空间中的传播。",
    )
    assert reasons == []
    assert mode == "think"
    assert prompt == "电磁波是什么？/think"
    assert response == "<think>先分析定义。</think>电磁波是振荡电磁场在空间中的传播。"


def test_native_direct_route_injects_official_empty_think_prefix():
    prompt, response, mode, reasons = builder.route_native(
        "根冠的主要作用是什么？",
        "根冠保护根尖分生组织，并参与重力感知。",
    )
    assert reasons == []
    assert mode == "no_think"
    assert prompt == "根冠的主要作用是什么？/no_think"
    assert response == "<think>\n\n</think>\n根冠保护根尖分生组织，并参与重力感知。"


def test_static_prompt_rejects_personal_advice_and_accepts_static_science():
    reasons, domain, _hits = builder.static_prompt_reasons("请解释电磁波是什么？")
    assert reasons == []
    assert domain == "natural_science"

    reasons, _domain, _hits = builder.static_prompt_reasons(
        "我想给家里的网络设置家长控制，有哪些软件可以推荐？"
    )
    assert "personalized" in reasons
    assert "non_knowledge_task" in reasons


def test_cached_leakage_match_preserves_exact_and_near_rules():
    index = base.LeakageIndex()
    index.add("请解释电磁波的基本定义和主要性质。", "fixture")
    gram_sets = [base.char_ngrams(text) for text in index.semantic_texts]

    hit, modes = builder.leakage_match(
        index, "请解释电磁波的基本定义和主要性质。/think", gram_sets
    )
    assert hit
    assert "mode_exact" in modes or "core_exact" in modes

    near = "请解释电磁波的基本定义及主要性质。"
    assert builder.leakage_match(index, near, gram_sets) == index.match(near)
