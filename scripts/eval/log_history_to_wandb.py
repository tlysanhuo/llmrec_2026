#!/usr/bin/env python3
"""log_history_to_wandb.py — 把已完成的 v1/v2/v3 训练曲线 + 平台分数回填进 wandb。

已有训练只有本地 trainer_log.jsonl(禁用过 wandb), 平台分数在 experiment_log。
本脚本为每个历史版本建一个 wandb run, 记录 loss 曲线(train/loss) + 平台分项(summary)。
这样 wandb 里有完整历史, 后续新实验可横向对比。

project: llmrec-2026 ; run 名 = 版本名。
"""
import json, os, sys

os.environ.setdefault("WANDB_DIR", "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026/wandb")
import wandb

CKPT = "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/llmrec_2026/checkpoints"

# 版本 -> (ckpt目录, 平台分项, 配置摘要)
RUNS = {
    "v1_baseline_sft": {
        "dir": "baseline_sft_v1",
        "scores": {"total": 0.8100, "wu_material": 0.1840, "user_action_select": 0.0362, "user_topic_gen": 0.0392,
                   "rec_ad": 0.0672, "rec_live": 0.1054, "rec_prod": 0.1316, "rec_video": 0.1089, "world": 0.1375},
        "cfg": {"data": "seed 32480", "material": 0, "r2": 0, "epochs": 1, "lr": 2e-5, "gbatch": 4},
    },
    "v2_run_a_r2": {
        "dir": "run_a_r2",
        "scores": {"total": 0.8092, "wu_material": 0.1840, "user_action_select": 0.0667, "user_topic_gen": 0.0430,
                   "rec_ad": 0.0480, "rec_live": 0.1054, "rec_prod": 0.1274, "rec_video": 0.1053, "world": 0.1294},
        "cfg": {"data": "seed 32480 + R2 13920", "material": 0, "r2": 13920, "epochs": 1, "lr": 2e-5, "gbatch": 4},
    },
    "v3_run_c_material": {
        "dir": "run_c_material",
        "scores": {"total": 0.8198, "wu_material": 0.1840, "user_action_select": 0.0446, "user_topic_gen": 0.0407,
                   "rec_ad": 0.0768, "rec_live": 0.1020, "rec_prod": 0.1274, "rec_video": 0.1098, "world": 0.1346},
        "cfg": {"data": "seed 32480 + material 10000", "material": 10000, "r2": 0, "epochs": 1, "lr": 2e-5, "gbatch": 4},
    },
}


def main():
    for name, meta in RUNS.items():
        jl = f"{CKPT}/{meta['dir']}/trainer_log.jsonl"
        run = wandb.init(project="llmrec-2026", name=name, config=meta["cfg"],
                         reinit=True, resume="never")
        # 训练曲线
        if os.path.exists(jl):
            for line in open(jl):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "loss" in r:
                    step = int(r.get("current_steps", 0))
                    wandb.log({"train/loss": r["loss"], "train/lr": r.get("lr", 0),
                               "train/epoch": r.get("epoch", 0)}, step=step)
        # 平台分项 -> summary
        for k, v in meta["scores"].items():
            run.summary[f"online/{k}"] = v
        run.summary["online/rank_note"] = "day1 leaderboard: top1=0.8784"
        run.finish()
        print(f"[ok] logged {name}: total={meta['scores']['total']}", file=sys.stderr)


if __name__ == "__main__":
    main()
