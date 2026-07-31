# LLaMA-Factory Customizations

This directory holds our local modifications to
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), which was the
training stack used to produce the `i35` and `i50` LoRA adapters that are fused
into the final submission.

## Contents

- `dataset_info.json` — our dataset registry (LLaMA-Factory
  `data/dataset_info.json`). File paths are relative to the repo root.
- `src/llamafactory/hparams/parser.py` — our patched parser (diff vs upstream).

## Not included

- `.env.local` — intentionally excluded. It contained a W&B API key and must
  never be committed.
- `tests_v1/` — this is upstream LLaMA-Factory's own test suite, not our
  customization; get it from the official repo.
- The full upstream LLaMA-Factory tree. Start from the official repo and apply
  these files on top of it.

## How to use

1. Clone upstream LLaMA-Factory at the pinned version.
2. Copy `dataset_info.json` → `data/dataset_info.json`.
3. Copy `src/llamafactory/hparams/parser.py` over the upstream file.
4. Install upstream deps, then run the per-experiment training scripts under
   `scripts/train/` (e.g. `train_i35_video_boundary_retkl.py`) with the
   corresponding config under `configs/`.
