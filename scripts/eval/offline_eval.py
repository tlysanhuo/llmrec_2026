#!/usr/bin/env python3
"""offline_eval.py — 离线评测台 v4(平台参数镜像 + 圈外 dev 集)。

设计/校准记录:docs/offline_eval.md。dev 集:assets/evaluation/offline_eval/(build_offline_dev.py 生成)。
八子项全覆盖:
  mat    dev_mat_fresh(圈外,判决候选) + dev_mat_train(圈内,记忆化对照) → beam64×3tok Pass@64
  rec    dev_rec_{video,prod,ad,live} 各≤1000 → 直通 nothink beam32 + 采样thinking→beam32,合并64候选
  action dev_action → 平台参数采样 n=1 ≤4096tok,itemic 集合 F1
  topic  dev_topic → 平台参数采样 ≤4096tok,官方公式(action有序LCS匹配 + logic TokenF1/ROUGE-L)
  world  dev_world(CMMLU圈外) → 平台参数采样 ≤10240tok,「正确答案是 (X)」抽取 Acc

用法:
  $V/miniconda3/envs/verl_v071/bin/python scripts/eval/offline_eval.py \
      --model checkpoints/xxx --gpu 3 [--dims mat,rec,action,topic,world] \
      [--n_rec 1000] [--tag xxx] [--think_suffix keep|switch]
输出: logs/offline_eval/<tag>_v4_<ts>.json(含协议版本与解码参数)
绝对值永远不可信(题面分布≠平台),只用于**校准过的维度**上的版本排序。

注意:v3 历史结果使用过较短生成上限和 rec top_k=20。v4 修正参数后与
v3 不可混合校准；现有 v3 的“8维盲区+world仅方向”结论仍然有效。
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJ = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026"
DEV = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/assets/evaluation/offline_eval"
DOM_TOKEN = {"video": "<|video_begin|>", "ad": "<|ad_begin|>", "prod": "<|prod_begin|>", "living": "<|living_begin|>"}
ITEM = re.compile(r"<\|(?:video|prod|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")
ITEM_CAPTURE = re.compile(
    r"<\|(video|prod|ad|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>"
)
SIDPAT = re.compile(r"<s_([abc])_(\d+)>")
OFFICIAL_N = {"mat": 574, "video": 1000, "prod": 1000, "ad": 1000, "live": 1000}
PROTOCOL_VERSION = "offline-eval-v4-platform-params"
ACTION1024_DIAGNOSTIC_VERSION = "offline-eval-action1024-diagnostic-v1"
REC_SAMPLE = {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 50}
USER_SAMPLE = {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
ACTION1024_SAMPLE = {"max_tokens": 1024, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
WORLD_SAMPLE = {"max_tokens": 10240, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def load(name):
    return [json.loads(l) for l in open(f"{DEV}/{name}")]


def canonical_row_hash(row):
    packed = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def stable_rows(path, count):
    """Match audit_i21_topic_path.py's frozen SHA256 row selection exactly."""
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows.append((canonical_row_hash(row), row))
    if len(rows) < count:
        raise ValueError(f"{path} has {len(rows)} rows, need {count}")
    return [row for _, row in sorted(rows)[:count]]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage2_holdout(path):
    tasks = {"action": [], "topic": [], "world": []}
    rec_groups = {}
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        body = row["output"].split("</think>")[-1].strip()
        record = {"system": row.get("instruction", ""), "user": row["input"]}
        if body.startswith("["):
            record["gold"] = json.loads(body)
            tasks["action"].append(record)
        elif body.startswith("{") and "logic_chain" in body:
            record["gold"] = json.loads(body)["logic_chain"]["events"]
            tasks["topic"].append(record)
        elif "该用户最近" in body:
            match = ITEM_CAPTURE.search(body)
            if match is None:
                raise ValueError("stage2 recommendation holdout row has no itemic target")
            domain, a, b, c = match.groups()
            key = (record["system"], record["user"], domain)
            grouped = rec_groups.setdefault(
                key,
                {"system": record["system"], "user": record["user"], "golds": set()},
            )
            grouped["golds"].add((a, b, c))
        else:
            match = re.search(r"正确答案是\s*[\(（]?\s*([A-D])", body)
            if match is None:
                raise ValueError("stage2 world holdout row has no answer letter")
            record["gold"] = match.group(1)
            tasks["world"].append(record)

    for (_, _, domain), record in rec_groups.items():
        record["golds"] = sorted(record["golds"])
        tasks.setdefault(f"rec_{domain}", []).append(record)
    return tasks


def prompt_of(r, mode):
    """mode: nothink=空think前缀直出; think=裸assistant起始采样thinking"""
    u = r["user"]
    if mode == "nothink" and not u.rstrip().endswith("/no_think"):
        u = u + "/no_think"
    p = ""
    if r.get("system"):
        p += f"<|im_start|>system\n{r['system']}<|im_end|>\n"
    p += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    if mode == "nothink":
        p += "<think>\n\n</think>\n"
    return p


def lcs(a, b):
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            m[i + 1][j + 1] = m[i][j] + 1 if a[i] == b[j] else max(m[i][j + 1], m[i + 1][j])
    return m[-1][-1]


def tok_f1(a, b):
    ca, cb = Counter(a), Counter(b)
    ov = sum((ca & cb).values())
    if not ov:
        return 0.0
    p, r = ov / max(sum(cb.values()), 1), ov / max(sum(ca.values()), 1)
    return 2 * p * r / (p + r)


def rouge_l(a, b):
    l = lcs(a, b)
    if not l:
        return 0.0
    p, r = l / max(len(b), 1), l / max(len(a), 1)
    return 2 * p * r / (p + r)


def topic_metric(gold_events, pred_events):
    ga = [e["action"] for e in gold_events]
    pa = [e.get("action", "") for e in pred_events]
    act_align = 2 * lcs(ga, pa) / max(len(ga) + len(pa), 1)
    gi = {e["action"]: e["logic"] for e in gold_events}
    logics = [(tok_f1(list(gi[e.get("action", "")]), list(str(e.get("logic", "")))) +
               rouge_l(list(gi[e.get("action", "")]), list(str(e.get("logic", ""))))) / 2
              for e in pred_events if e.get("action", "") in gi]
    logic_align = sum(logics) / len(logics) if logics else 0.0
    return (act_align + logic_align) / 2, act_align, logic_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--adapter",
        default="",
        help="Optional PEFT LoRA directory loaded directly by vLLM; --model remains the base model.",
    )
    ap.add_argument("--gpu", default="3")
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of visible GPU memory reserved by vLLM.",
    )
    ap.add_argument("--dims", default="mat,rec,action,topic,world")
    ap.add_argument("--n_rec", type=int, default=1000)
    ap.add_argument("--n_mat", type=int, default=0, help="0=全量")
    ap.add_argument(
        "--mat-per-domain",
        type=int,
        default=0,
        help="Stable SHA256 sample count per material domain; overrides --n_mat.",
    )
    ap.add_argument("--n_action", type=int, default=0)
    ap.add_argument(
        "--action-protocol",
        choices=["legacy-v4", "action1024-diagnostic"],
        default="legacy-v4",
        help=(
            "legacy-v4 preserves the historical 4096-token action/topic probe; "
            "action1024-diagnostic is an action-only structural diagnostic using the "
            "post-fix platform token cap and is not a score estimator"
        ),
    )
    ap.add_argument(
        "--stable_action_rows",
        type=int,
        default=0,
        help="Select action rows by canonical-JSON SHA256, matching audit_i21_topic_path.py.",
    )
    ap.add_argument("--n_world", type=int, default=0)
    ap.add_argument("--generation-seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional deterministic report path; defaults to the historical timestamped path.",
    )
    ap.add_argument(
        "--stage2_holdout",
        default="",
        help="Use grouped stage-2 holdout rows for rec/action/topic/world regression checks.",
    )
    ap.add_argument("--think_suffix", choices=["keep", "switch"], default="keep",
                    help="thinking通路的软开关:keep=保留/no_think(v1探针同款);switch=改/think")
    args = ap.parse_args()
    if args.stable_action_rows and args.n_action:
        ap.error("--stable_action_rows and --n_action are mutually exclusive")
    if args.mat_per_domain and args.n_mat:
        ap.error("--mat-per-domain and --n_mat are mutually exclusive")
    if args.stable_action_rows and args.stage2_holdout:
        ap.error("--stable_action_rows cannot be combined with --stage2_holdout")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tag = args.tag or os.path.basename(args.model.rstrip("/"))
    dims = {value.strip() for value in args.dims.split(",") if value.strip()}
    if args.action_protocol == "action1024-diagnostic" and dims != {"action"}:
        ap.error("--action-protocol action1024-diagnostic requires --dims action")
    action_sample = (
        ACTION1024_SAMPLE
        if args.action_protocol == "action1024-diagnostic"
        else USER_SAMPLE
    )
    protocol_version = (
        ACTION1024_DIAGNOSTIC_VERSION
        if args.action_protocol == "action1024-diagnostic"
        else PROTOCOL_VERSION
    )

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams

    t0 = time.time()
    lora_request = None
    llm_kwargs = {}
    adapter_metadata = None
    if args.adapter:
        adapter_dir = Path(args.adapter).resolve()
        adapter_file = adapter_dir / "adapter_model.safetensors"
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_file.is_file() or not adapter_config.is_file():
            raise FileNotFoundError(f"invalid PEFT adapter directory: {adapter_dir}")
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        rank = int(config["r"])
        try:
            max_lora_rank = next(value for value in VLLM_LORA_RANKS if value >= rank)
        except StopIteration as exc:
            raise ValueError(f"adapter rank {rank} exceeds vLLM's supported maximum") from exc
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(adapter_dir.name, 1, lora_path=str(adapter_dir))
        llm_kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank)
        adapter_metadata = {
            "path": str(adapter_dir),
            "rank": rank,
            "vllm_max_lora_rank": max_lora_rank,
            "adapter_sha256": sha256(adapter_file),
            "config_sha256": sha256(adapter_config),
        }
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=40960,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
              seed=args.generation_seed,
              enable_prefix_caching=True, trust_remote_code=True, max_logprobs=130,
              **llm_kwargs)
    report = {
        "protocol_version": protocol_version,
        "model": args.model,
        "tag": tag,
        "date": datetime.now().isoformat()[:19],
        "n_rec": args.n_rec,
        "think_suffix": args.think_suffix,
        "generation_seed": args.generation_seed,
        "evaluated_dims": sorted(dims),
        "sampling": {"rec": REC_SAMPLE, "action_topic": action_sample, "world": WORLD_SAMPLE},
    }
    if args.action_protocol == "action1024-diagnostic":
        report["diagnostic_boundary"] = (
            "Action-only structural diagnostic aligned to the post-fix 1024-token cap. "
            "Offline action quality metrics are uncalibrated and must not be converted "
            "to an online score or used as a positive ranking gate."
        )
        report["online_score_estimate"] = None
        report["protocol_scope"] = "action-only local generation diagnostic"
        report["full_current_platform_mirror"] = False
    if adapter_metadata is not None:
        report["adapter"] = adapter_metadata
    stage2_holdout = load_stage2_holdout(args.stage2_holdout) if args.stage2_holdout else None
    if args.stage2_holdout:
        report["stage2_holdout"] = str(Path(args.stage2_holdout).resolve())

    def beam_decode(prompts, width, chunk=250):
        params = BeamSearchParams(beam_width=width, max_tokens=3)
        res = []
        for i in range(0, len(prompts), chunk):
            outs = llm.beam_search(
                [{"prompt": p} for p in prompts[i:i + chunk]],
                params,
                lora_request=lora_request,
            )
            for p, o in zip(prompts[i:i + chunk], outs):
                cands = []
                for seq in o.sequences:
                    gen = seq.text[len(p):] if seq.text.startswith(p) else seq.text
                    toks = SIDPAT.findall(gen)
                    if len(toks) == 3 and [t[0] for t in toks] == ["a", "b", "c"]:
                        cands.append(tuple(t[1] for t in toks))
                    else:
                        cands.append(None)
                res.append(cands)
        return res

    def sample(prompts, max_tokens, stop=None, temperature=0.6, top_p=0.95, top_k=20):
        sp = SamplingParams(n=1, max_tokens=max_tokens, temperature=temperature,
                            top_p=top_p, top_k=top_k, seed=args.generation_seed, stop=stop)
        return [o.outputs[0] for o in llm.generate(prompts, sp, lora_request=lora_request)]

    # ============ mat(fresh=判决候选 / train=记忆化对照) ============
    if "mat" in dims:
        for name, key in [("dev_mat_fresh.jsonl", "mat_fresh"), ("dev_mat_train.jsonl", "mat_train")]:
            rows = load(name)
            if args.mat_per_domain:
                by_domain = {}
                for row in rows:
                    by_domain.setdefault(row["gold"]["dom"], []).append(row)
                sampled = []
                for domain, domain_rows in sorted(by_domain.items()):
                    if len(domain_rows) < args.mat_per_domain:
                        raise ValueError(
                            f"{name}/{domain} has {len(domain_rows)} rows, "
                            f"need {args.mat_per_domain}"
                        )
                    sampled.extend(
                        sorted(domain_rows, key=canonical_row_hash)[: args.mat_per_domain]
                    )
                rows = sampled
            elif args.n_mat:
                rows = rows[: args.n_mat]
            prompts = [prompt_of(r, "nothink") + DOM_TOKEN[r["gold"]["dom"]] for r in rows]
            cands = beam_decode(prompts, 64)
            hit, sa_hit, by_dom = 0, 0, {}
            for r, cc in zip(rows, cands):
                gold = tuple(r["gold"]["abc"])
                cset = set(c for c in cc if c)
                h = gold in cset
                hit += h
                sa_hit += gold[0] in set(c[0] for c in cset)
                d = by_dom.setdefault(r["gold"]["dom"], [0, 0])
                d[0] += h
                d[1] += 1
            n = len(rows)
            report[key] = {"n": n, "pass@64": round(hit / n, 4), "sa_pass@64": round(sa_hit / n, 4),
                           "by_dom": {k: f"{a}/{b}" for k, (a, b) in sorted(by_dom.items())},
                           "pred_hits_of_574": round(hit / n * OFFICIAL_N["mat"], 1)}
            print(f"[{key}] {report[key]}", file=sys.stderr)

    # ============ rec 四域(两通路合并64候选) ============
    if "rec" in dims:
        rec_out = {}
        for dom_file, dom in [("video", "video"), ("prod", "prod"), ("ad", "ad"), ("live", "living")]:
            if stage2_holdout is None:
                rows = load(f"dev_rec_{dom_file}.jsonl")[: args.n_rec]
            else:
                rows = stage2_holdout[f"rec_{dom}"][: args.n_rec]
            direct = beam_decode([prompt_of(r, "nothink") + DOM_TOKEN[dom] for r in rows], 32)
            tp = []
            for r in rows:
                r2 = dict(r)
                if args.think_suffix == "switch":
                    r2["user"] = re.sub(r"/no_think\s*$", "/think", r2["user"])
                tp.append(prompt_of(r2, "think"))
            thinks = sample(tp, stop=["</think>"], **REC_SAMPLE)
            staged = beam_decode([p + t.text + "</think>\n" + DOM_TOKEN[dom] for p, t in zip(tp, thinks)], 32)
            hit64 = hit_d = hit_t = copy_d = tot_d = 0
            sa_direct = []
            for r, dc, tc in zip(rows, direct, staged):
                gold_values = r["golds"] if "golds" in r else [r["gold"]["abc"]]
                golds = set(tuple(gold) for gold in gold_values)
                dset = set(c for c in dc if c)
                tset = set(c for c in tc if c)
                hit_d += bool(golds & dset)
                hit_t += bool(golds & tset)
                hit64 += bool(golds & (dset | tset))
                hist = set(m[1:] for m in re.findall(
                    r"<\|(video|ad|prod|living)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>", r["user"]) if m[0] == dom)
                for c in dset:
                    tot_d += 1
                    copy_d += c in hist
                sa_direct.append(len(set(c[0] for c in dset)))
            n = len(rows)
            rec_out[dom_file] = {
                "n": n, "pass@64": round(hit64 / n, 4),
                "p32_direct": round(hit_d / n, 4), "p32_think": round(hit_t / n, 4),
                "copy_direct": round(copy_d / max(tot_d, 1), 4),
                "distinct_sa_direct": round(statistics.mean(sa_direct), 2) if sa_direct else 0,
                "gold_targets": sum(len(r.get("golds", [r.get("gold")])) for r in rows),
                "pred_hits_of_1000": round(hit64 / n * 1000, 1)}
            print(f"[rec/{dom_file}] {rec_out[dom_file]}", file=sys.stderr)
        report["rec"] = rec_out

    # ============ action(懂用户) ============
    if "action" in dims:
        action_source = Path(DEV) / "dev_action.jsonl"
        if args.stable_action_rows:
            rows = stable_rows(action_source, args.stable_action_rows)
        else:
            rows = load("dev_action.jsonl") if stage2_holdout is None else stage2_holdout["action"]
        if args.n_action:
            rows = rows[: args.n_action]
        selected_hashes = [canonical_row_hash(row) for row in rows]
        report["action_selection"] = {
            "method": "canonical_json_sha256_ascending" if args.stable_action_rows else "source_order",
            "requested": args.stable_action_rows or args.n_action or 0,
            "source": str(action_source.resolve()) if stage2_holdout is None else str(Path(args.stage2_holdout).resolve()),
            "source_sha256": sha256(action_source) if stage2_holdout is None else sha256(args.stage2_holdout),
            "selected_row_sha256": selected_hashes,
            "manifest_sha256": hashlib.sha256(
                "".join(value + "\n" for value in selected_hashes).encode()
            ).hexdigest(),
        }
        outs = sample([prompt_of(r, "nothink") for r in rows], **action_sample)
        f1s, cnts, jok, trunc = [], [], 0, 0
        generated_lengths, duplicate_counts, max_repeats = [], [], []
        row_metrics = []
        for row_hash, r, o in zip(selected_hashes, rows, outs):
            gold = set(r["gold"])
            occurrences = ITEM.findall(o.text)
            occurrence_counts = Counter(occurrences)
            pred = set(occurrences)
            json_ok = False
            try:
                json.loads(re.search(r"\[.*\]", o.text, re.S).group(0))
                jok += 1
                json_ok = True
            except Exception:
                pass
            is_truncated = len(o.token_ids) >= action_sample["max_tokens"]
            trunc += is_truncated
            generated_lengths.append(len(o.token_ids))
            duplicate_count = len(occurrences) - len(pred)
            max_repeat = max(occurrence_counts.values(), default=0)
            duplicate_counts.append(duplicate_count)
            max_repeats.append(max_repeat)
            ov = len(gold & pred)
            p_ = ov / max(len(pred), 1)
            rc = ov / max(len(gold), 1)
            f1 = 2 * p_ * rc / max(p_ + rc, 1e-9)
            f1s.append(f1)
            cnts.append(len(pred))
            row_metrics.append({
                "row_sha256": row_hash,
                "f1": f1,
                "json_ok": json_ok,
                "truncated": bool(is_truncated),
                "generated_tokens": len(o.token_ids),
                "n_pred": len(pred),
                "duplicate_items": duplicate_count,
                "max_repeat": max_repeat,
            })
        n = len(rows)
        report["action"] = {"n": n, "f1": round(sum(f1s) / n, 4),
                            "f1_unrounded": sum(f1s) / n,
                            "json_ok": round(jok / n, 3), "json_ok_count": jok,
                            "trunc_rate": round(trunc / n, 3), "trunc_count": trunc,
                            "n_pred_median": statistics.median(cnts) if cnts else 0,
                            "generated_tokens_p95": sorted(generated_lengths)[max(0, math.ceil(0.95 * n) - 1)] if n else 0,
                            "duplicate_items_mean": round(statistics.mean(duplicate_counts), 2) if duplicate_counts else 0,
                            "max_repeat_p95": sorted(max_repeats)[max(0, math.ceil(0.95 * n) - 1)] if n else 0,
                            "rows": row_metrics}
        print(f"[action] {report['action']}", file=sys.stderr)

    # ============ topic ============
    if "topic" in dims:
        rows = load("dev_topic.jsonl") if stage2_holdout is None else stage2_holdout["topic"]
        outs = sample([prompt_of(r, "nothink") for r in rows], **USER_SAMPLE)

        def parse_events(text):
            """先严格 JSON;失败退正则抽 action/logic 对(平台计分不至于因中文引号全灭)"""
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))["logic_chain"]["events"], True
                except Exception:
                    pass
            evs = [{"action": a, "logic": g} for a, g in
                   re.findall(r'"action"\s*[:：]\s*"(.*?)"\s*[,，]\s*"logic"\s*[:：]\s*"(.*?)"', text, re.S)]
            return evs, False

        scores, aas, las, jbad = [], [], [], 0
        for r, o in zip(rows, outs):
            pred, strict = parse_events(o.text)
            jbad += not strict
            if not pred:
                scores.append(0.0)
                continue
            s, aa, la = topic_metric(r["gold"], pred)
            scores.append(s)
            aas.append(aa)
            las.append(la)
        n = len(rows)
        report["topic"] = {"n": n, "score": round(sum(scores) / n, 4),
                           "act_align": round(sum(aas) / max(len(aas), 1), 4),
                           "logic_align": round(sum(las) / max(len(las), 1), 4),
                           "json_fail": round(jbad / n, 3)}
        print(f"[topic] {report['topic']}", file=sys.stderr)

    # ============ world ============
    if "world" in dims:
        rows = load("dev_world.jsonl") if stage2_holdout is None else stage2_holdout["world"]
        if args.n_world:
            rows = rows[: args.n_world]
        outs = sample([prompt_of(r, "nothink") for r in rows], **WORLD_SAMPLE)
        pat = re.compile(r"正确答案是\s*[\(（]?\s*([A-D])")
        fb = re.compile(r"[\(（]([A-D])[\)）]|\b([A-D])\b")
        ok = alive = 0
        for r, o in zip(rows, outs):
            m = pat.search(o.text)
            alive += bool(m)
            letter = m.group(1) if m else None
            if not letter:
                m2 = fb.search(o.text)
                letter = (m2.group(1) or m2.group(2)) if m2 else None
            ok += letter == r["gold"]
        n = len(rows)
        report["world"] = {"n": n, "acc": round(ok / n, 4), "fmt_alive": round(alive / n, 3)}
        print(f"[world] {report['world']}", file=sys.stderr)

    report["runtime_s"] = round(time.time() - t0, 1)
    if args.out is None:
        protocol_suffix = "action1024diag_v1" if args.action_protocol == "action1024-diagnostic" else "v4"
        out_path = Path(PROJ) / "logs" / "offline_eval" / f"{tag}_{protocol_suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    else:
        out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"saved -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
