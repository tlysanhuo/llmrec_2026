#!/usr/bin/env python3
"""run_eval.py — 统一评测 CLI 入口。

对齐 SPEC.md 任务7验收标准：
  - `--dims` 参数可选择任意维度子集运行（可选值：material,user_f1,user_chain,recommend,world）。
  - 数据源改为直接读取 `demo/baseline-data/baseline_data/sampled/` 下的官方全量数据
    （懂世界/懂用户/懂物料/懂推荐），并用 `data/OneReason_Pid2Sid/` 反查表把 SemanticID
    映射为真实 item_id（详见 `data/loaders.py` 顶部注释与 `eval/SPEC.md` 附录"忠实度
    对比评估"）。不再使用哈希代理 item_id。
  - 输出统一 JSON schema 落盘（默认写到 eval/output/<tag>_<ts>.json，也可 --dry-run
    只打印到 stdout）。

用法示例：
    python3 run_eval.py --dims world,material --limit 10
    python3 run_eval.py --dims user_chain --model /path/to/checkpoint --gpu 0
    python3 run_eval.py --dims world --limit 5 --dry-run

注意：真正调用 vLLM 生成需要 `--model` 参数并安装好 vllm；若省略 `--model`，
本 CLI 会退化为"self-check"模式——直接用 dev 数据里的参考答案/gold本身作为
"预测"喂给对应 metrics 模块打分，用于验证整条流水线（数据加载 -> 打分 -> 落盘）
是否连通，不代表真实模型的评测结果（会在输出 JSON 中显式标注 "mode": "self_check"）。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.prompt import SAMPLING, build_domain_prompt, build_prompt
from common.sid_utils import parse_sid_tokens, sid_tokens_to_item_ids
from data.loaders import get_pid2sid_index, load_dev
from metrics.recommend import merge_candidate_pools
from metrics.user_chain import Event, score_chain
from metrics.user_f1 import f1_score
from metrics.world import accuracy

ALL_DIMS = ["material", "user_f1", "user_chain", "recommend", "world"]


def _get_engine(model: str | None, gpu: str, lora: str | None = None):
    if model is None:
        return None
    from common.engine import VLLMEngine

    return VLLMEngine(model=model, gpu=gpu, lora_path=lora)


def _patterns_to_item_ids(patterns: list[str], index: dict) -> set[int]:
    """把一批生成文本（beam candidate 或最终回复正文）解析出的 SemanticID 映射为
    item_id 集合的并集（供 Pass@64 判定使用）。单条 pattern 内解析出多个 SID token
    时全部纳入（beam candidate 通常只含 1 个 SID，但兼容多 SID 的情形）。
    """
    ids: set[int] = set()
    for p in patterns:
        tokens = parse_sid_tokens(p)
        ids |= sid_tokens_to_item_ids(tokens, index)
    return ids


# ---------------------------------------------------------------------------
# 各维度评测流程
# ---------------------------------------------------------------------------


def run_material(dev: dict, engine, limit: int | None) -> dict:
    """懂物料 text->token：对齐线上「Single-stage generation with prompt_token
    (<|video_begin|>)」协议——video-begin token 强制拼进 prompt 末尾，模型只需
    beam search 生成 3 个后续 token（s_a/s_b/s_c），而非自由生成完整 pattern。

    `dev["items"]` 经 `data/loaders.py::load_material_dev` 的 `domain_filter`
    过滤后应已仅剩 video 域样本（对齐线上懂物料 100% 为 video 域的事实），此处
    仍显式使用每条样本自身的 `domain_prefix` 字段而非硬编码 "video"，以便
    未来若 loader 的过滤策略调整也能正确生成。
    """
    items = dev["items"][:limit] if limit else dev["items"]
    if not items:
        return {"available": False, "reason": dev.get("reason", "no data"), "n": 0}

    index = get_pid2sid_index()
    if index is None:
        return {"available": False, "reason": "Pid2Sid 反查表不可用", "n": 0}

    results = []
    n_gt_map_failed = 0
    for it in items:
        gold_ids = set(it["gold_item_ids"])
        if not gold_ids:
            n_gt_map_failed += 1
            continue
        domain_prefix = it["domain_prefix"]
        if engine is not None:
            prompt = build_domain_prompt(
                it.get("system", ""), it["prompt"], domain_prefix=domain_prefix, mode="no_think"
            )
            # beam_decode 返回值已去除 prompt 前缀（裸 s_a/s_b/s_c，不含 domain-begin
            # token，因为该 token 已被硬编码进 prompt），评分前需拼回前缀才能被
            # parse_sid_tokens 正确解析。
            beam_suffixes = engine.beam_decode([prompt], beam_width=64, max_tokens=3)[0]
            begin_token = f"<|{domain_prefix}_begin|>"
            beam_texts = [begin_token + s for s in beam_suffixes]
        else:
            # self-check：用 gold pattern 本身重复 64 次模拟"beam64 全部命中"
            beam_texts = [it["pattern"]] * 64
        candidate_ids = _patterns_to_item_ids(beam_texts, index)
        passed = int(bool(candidate_ids & gold_ids))
        results.append(passed)

    return {
        "available": True,
        "n": len(results),
        "pass@64_mean": sum(results) / len(results) if results else 0.0,
        "n_gt_map_failed": n_gt_map_failed,
    }


def run_user_f1(dev: dict, engine, limit: int | None) -> dict:
    items = dev["items"][:limit] if limit else dev["items"]
    if not items:
        return {"available": False, "reason": dev.get("reason", "no data"), "n": 0}

    f1s = []
    for it in items:
        if engine is not None:
            prompt = build_prompt(it.get("system", ""), it["prompt"], mode="no_think")
            gen = engine.sample(
                [prompt],
                max_tokens=SAMPLING["user"].max_tokens,
                temperature=SAMPLING["user"].temperature,
                top_p=SAMPLING["user"].top_p,
                top_k=SAMPLING["user"].top_k,
            )[0]
            try:
                pred = json.loads(gen.text.split("</think>")[-1].strip())
            except json.JSONDecodeError:
                pred = []
        else:
            pred = it["gold"]  # self-check
        result = f1_score(pred, it["gold"])
        f1s.append(result["f1"])

    return {"available": True, "n": len(f1s), "f1_mean": sum(f1s) / len(f1s) if f1s else 0.0}


def run_user_chain(dev: dict, engine, limit: int | None) -> dict:
    items = dev["items"][:limit] if limit else dev["items"]
    if not items:
        return {"available": False, "reason": dev.get("reason", "no data"), "n": 0}

    action_scores, logic_scores, overall_scores = [], [], []
    for it in items:
        gold_events = [Event(action=e["action"], logic=e.get("logic", "")) for e in it["gold_events"]]
        if engine is not None:
            prompt = build_prompt(it.get("system", ""), it["prompt"], mode="no_think")
            gen = engine.sample(
                [prompt],
                max_tokens=SAMPLING["user"].max_tokens,
                temperature=SAMPLING["user"].temperature,
                top_p=SAMPLING["user"].top_p,
                top_k=SAMPLING["user"].top_k,
            )[0]
            try:
                payload = json.loads(gen.text.split("</think>")[-1].strip())
                pred_events = [
                    Event(action=e["action"], logic=e.get("logic", ""))
                    for e in payload.get("logic_chain", {}).get("events", [])
                ]
            except (json.JSONDecodeError, KeyError):
                pred_events = []
        else:
            pred_events = gold_events  # self-check
        result = score_chain(gold_events, pred_events)
        action_scores.append(result.action_alignment)
        logic_scores.append(result.logic_alignment)
        overall_scores.append(result.overall_score)

    n = len(overall_scores)
    return {
        "available": True,
        "n": n,
        "action_alignment_mean": sum(action_scores) / n if n else 0.0,
        "logic_alignment_mean": sum(logic_scores) / n if n else 0.0,
        "overall_mean": sum(overall_scores) / n if n else 0.0,
    }


def run_recommend(dev: dict, engine, limit: int | None) -> dict:
    """懂推荐：对齐线上四个 `challenge_recommendation_*` 子任务的双路约束解码协议：

    - no-think 路：目标域的 domain-begin token 直接硬编码拼进 prompt 末尾，
      beam search 直接生成 3 个后续 token（beam_width=32），不再自由 sample 完整
      回复。
    - think 路：先用 sample（n=1，max_tokens=4096）采样出 1 条 thinking 文本（Stage 1），
      再把 thinking 文本 + 目标域 domain-begin token 拼回 prompt 末尾，对其做
      beam search 生成 3 个后续 token（beam_width=32，Stage 2）。

    目标域取自每条样本自身的 `target_domain_prefix` 字段（由
    `data/loaders.py::load_recommend_dev` 从 gold SemanticID 前缀反推得到，
    19204 条样本全量实测 gold 均为单一域，反推完全可靠），与线上每条测试样本
    本身就已确定了预测目标域的事实对齐。
    """
    items = dev["items"][:limit] if limit else dev["items"]
    if not items:
        return {"available": False, "reason": dev.get("reason", "no data"), "n": 0}

    index = get_pid2sid_index()
    if index is None:
        return {"available": False, "reason": "Pid2Sid 反查表不可用", "n": 0}

    results = []
    for it in items:
        gold_ids = set(it["gold_item_ids"])
        if not gold_ids:
            continue
        domain_prefix = it["target_domain_prefix"]
        begin_token = f"<|{domain_prefix}_begin|>"
        if engine is not None:
            # --- no-think 路：直接 beam search 3 token ---
            prompt_nothink = build_domain_prompt(
                it.get("system", ""), it["prompt_nothink"], domain_prefix=domain_prefix, mode="no_think"
            )
            nothink_suffixes = engine.beam_decode([prompt_nothink], beam_width=32, max_tokens=3)[0]
            non_thinking_texts = [begin_token + s for s in nothink_suffixes]

            # --- think 路：Stage 1 采样 thinking，Stage 2 在其后 beam search 3 token ---
            prompt_think_stage1 = build_prompt(it.get("system", ""), it["prompt_think"], mode="think")
            gen_thinking = engine.sample(
                [prompt_think_stage1],
                max_tokens=SAMPLING["recommend"].max_tokens,
                temperature=SAMPLING["recommend"].temperature,
                top_p=SAMPLING["recommend"].top_p,
                top_k=SAMPLING["recommend"].top_k,
                n=1,
            )
            thinking_text = gen_thinking[0].text
            prompt_think_stage2 = build_domain_prompt(
                it.get("system", ""),
                it["prompt_think"],
                domain_prefix=domain_prefix,
                mode="think",
                thinking_text=thinking_text,
            )
            think_suffixes = engine.beam_decode([prompt_think_stage2], beam_width=32, max_tokens=3)[0]
            thinking_texts = [begin_token + s for s in think_suffixes]

            thinking_ids = list(_patterns_to_item_ids(thinking_texts, index))
            non_thinking_ids = list(_patterns_to_item_ids(non_thinking_texts, index))
        else:
            # self-check：用 gold item_ids 本身模拟"两路候选池均命中"
            thinking_ids = list(gold_ids)
            non_thinking_ids = []
        merged = merge_candidate_pools(thinking_ids, non_thinking_ids)
        passed = int(bool(set(merged) & gold_ids))
        results.append(passed)

    return {
        "available": True,
        "n": len(results),
        "pass@64_mean": sum(results) / len(results) if results else 0.0,
    }


def run_world(dev: dict, engine, limit: int | None) -> dict:
    items = dev["items"][:limit] if limit else dev["items"]
    if not items:
        return {"available": False, "reason": dev.get("reason", "no data"), "n": 0}

    predictions, golds = [], []
    for it in items:
        gold = it.get("gold")
        if not gold:
            continue
        if engine is not None:
            prompt = build_prompt(it.get("system", ""), it["prompt"], mode="no_think")
            gen = engine.sample(
                [prompt],
                max_tokens=SAMPLING["world"].max_tokens,
                temperature=SAMPLING["world"].temperature,
                top_p=SAMPLING["world"].top_p,
                top_k=SAMPLING["world"].top_k,
            )[0]
            pred_text = gen.text
        else:
            pred_text = f"正确答案是({gold})"  # self-check
        predictions.append(pred_text)
        golds.append(gold)

    result = accuracy(predictions, golds)
    return {"available": True, "n": result["n"], "accuracy": result["accuracy"]}


RUNNERS = {
    "material": run_material,
    "user_f1": run_user_f1,
    "user_chain": run_user_chain,
    "recommend": run_recommend,
    "world": run_world,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="统一评测 CLI（eval/run_eval.py）")
    parser.add_argument(
        "--dims", type=str, default=",".join(ALL_DIMS), help=f"逗号分隔的维度子集，可选: {ALL_DIMS}"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="被测模型路径（完整/merged 模型，或搭配 --lora 时的 base model 路径）；不提供则退化为 self-check 模式",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="可选的 LoRA adapter 目录（含 adapter_config.json/adapter_model.safetensors）；提供时 --model 作为 base model 加载",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--limit", type=int, default=None, help="每个维度最多评测多少条样本")
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--dry-run", action="store_true", help="只打印结果到 stdout，不落盘")
    args = parser.parse_args()

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    unknown = set(dims) - set(ALL_DIMS)
    if unknown:
        parser.error(f"未知维度: {unknown}，可选: {ALL_DIMS}")

    engine = _get_engine(args.model, args.gpu, lora=args.lora)

    manifest = {
        "protocol_version": "eval-v1-unified",
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "lora": args.lora,
        "mode": "model_inference" if engine is not None else "self_check",
        "dims": dims,
        "results": {},
    }

    for dim in dims:
        dev = load_dev(dim, limit=args.limit)
        runner = RUNNERS[dim]
        manifest["results"][dim] = runner(dev, engine, args.limit)

    output_json = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(output_json)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{args.tag}_{ts}.json"
        out_path.write_text(output_json, encoding="utf-8")
        print(f"评测结果已写入: {out_path}", file=sys.stderr)
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
