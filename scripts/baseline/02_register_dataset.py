#!/usr/bin/env python3
"""02_register_dataset.py — 把 data_final 注册进 LLaMA-Factory dataset_info.json（幂等）。
改编自 docs/demo_baseline/scripts/02_register_dataset.py，用绝对路径。"""
import json
import os
import pathlib

REPRO = os.environ.get("REPRO", "/lustre/prod_glm_volumes/volume-20260201002229-o7c51/ai_runtime/llmrec_2026")
INFO_PATH = pathlib.Path(REPRO, "LLaMA-Factory/data/dataset_info.json").resolve()
DATA_PATH = str(pathlib.Path(REPRO, "data/data_final.jsonl").resolve())

ENTRY = {
    "file_name": DATA_PATH,
    "formatting": "alpaca",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "history": "history",
    },
}

info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
info["data_final"] = ENTRY
INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] upserted 'data_final' -> {DATA_PATH} in {INFO_PATH}")
