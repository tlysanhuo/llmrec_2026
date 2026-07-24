# I-40 complete training-data release

This directory publishes the complete I-40 formal training input and its complete loss-routing sidecar. Both files are official-source derived class `D` assets, not official direct or official original data.

The release contains all 8,240 training exposures. It is not a sample. The 102,365,905-byte training JSONL and 7,605,609-byte sidecar are stored as deterministic gzip files so they fit normal Git without Git LFS.

## Contents

| File | Purpose |
|---|---|
| `data_i40_i35_direct_user_continue_v1.jsonl.gz` | Complete 8,240-row formal training JSONL |
| `data_i40_i35_direct_user_continue_v1_sidecar.jsonl.gz` | Complete 8,240-row token/loss routing sidecar |
| `manifest.json` | Data lineage, hashes, mix, training recipe, checkpoint identities and release boundaries |
| `audits/data_audit.json` | Frozen pre-training data audit copied byte-for-byte from the formal run |
| `audits/eval_summary.json` | Four-checkpoint behavior diagnostics and the explicit non-calibration boundary |
| `scripts/data/restore_i40_i35_direct_user_continue_v1.py` | Double-layer verification and atomic restore utility |

## Verify and restore

Run from the repository root:

```bash
python3 scripts/data/restore_i40_i35_direct_user_continue_v1.py --verify-only
python3 scripts/data/restore_i40_i35_direct_user_continue_v1.py
```

The first command checks each gzip SHA256 and then streams the decompressed payload to verify its byte count, row count and SHA256. The second command atomically restores both files to `assets/derived/processed/`, which is the path consumed by the exact I-40 trainer.

Existing correct files are left unchanged. An existing mismatched file is rejected unless `--force` is explicitly supplied.

## Data lineage and mix

The formal data has 5,500 audited user rows copied from the registered I-36 derived dataset and 2,740 rows copied from the registered I-35 formal dataset:

| Route | Rows | Training objective |
|---|---:|---|
| user CE | 5,500 | `0.05 * weighted answer CE + 16.0 * KL(I35-step548 || policy)` |
| retention KL | 2,740 | `16.0 * KL(I35-step548 || policy)`, maximum 128 answer positions |

The user rows contain 4,000 action and 1,500 topic examples. The retention rows retain the original I-35 task distribution but do not reuse I-35's old boundary, margin or CE objectives. There are zero third-party `T` rows and zero evaluation-derived `E` rows.

The I-35 source contains 25 inherited duplicate world exposures, each appearing twice. They are deliberately retained because the frozen contract requires every one of the 2,740 source rows once; the complete mixture therefore has 8,240 exposures and 8,215 normalized unique rows.

The imported user annotation generator implementation was not supplied. Every formal SID was independently checked against the registered official `O2.Pid2Sid` source, but this verification does not turn the imported annotations into official gold.

## Exact experiment setup

The Git-tracked experiment entrypoints are:

- `configs/active/i40_i35_direct_user_continue_r112_v1.yaml`
- `configs/datasets/i40_i35_direct_user_continue_v1/dataset_info.json`
- `scripts/data/build_i40_i35_direct_user_continue_v1.py`
- `scripts/train/train_i40_i35_direct_user_continue.py`
- `scripts/train/launch_i40_wandb_online.sh`
- `scripts/train/launch_i40_wandb_detached.sh`

I-40 directly loaded I-35 step548 r112 and continued updating the same 392 LoRA tensors. It did not create or concatenate a fresh adapter. The run used one GPU, W&B online run `34k0sdcj`, batch 1, gradient accumulation 4, effective batch 4, learning rate `5e-7`, cosine schedule, warmup ratio `0.03`, weight decay `0.001`, max grad norm `0.5`, BF16 and seed `19260840`. It completed 2,060 optimizer steps and saved steps 515, 1030, 1545 and 2060.

## Evaluation boundary

The local v4 suite is explicitly `NOT_CERTIFIED` for online score prediction. It is used only for behavior regression and structural diagnostics. Before the official result, its four-checkpoint local priority was step1030, step1545, step515, step2060 and the formal prediction was `ABSTAIN`.

Step1030 was subsequently evaluated once on the official platform. Evaluation `eval-task-bwvd45-1784866180` succeeded with score `0.9890615139753605`, versus `1.0344285849069457` for the I-35 step548 parent. The delta was `-0.04536707093158521`, dominated by recommendation video/product/ad. I-40 is therefore closed; the other checkpoints are not recommended for upload or continued training. Model weights are not included in this Git data release.
