# I-35 video boundary residual release

This release contains the exact rank-112 step-548 adapter, data, sidecar,
audits, and training logs for `i35_r96_video_boundary_retkl_r16_v1`. The
step-548 residual was concatenated with the rank-96 parent to produce the
adapter that scored `1.0344285849069457`.

The training-side word `video` means the O1 material task that maps a video
description to a three-token video SID. It does not mean direct CE training on
the `rec_video` next-item task. The online gain was transfer: material stayed
at `0.2453`, while `rec_video` increased by `0.0096` and `rec_prod` by `0.0068`
relative to the direct parent.

Verify and restore the original artifacts with:

```bash
scripts/reproduce/i35_video_boundary_release.sh verify-data
scripts/reproduce/i35_video_boundary_release.sh restore-original-data
scripts/reproduce/i35_video_boundary_release.sh self-test
```

## Restore the scored adapter

The exact `adapter_model.safetensors` is published under `adapter/` as three
byte-contiguous GitHub-compatible parts. From this release directory, run:

```bash
sha256sum -c SHA256SUMS
cat adapter/adapter_model.safetensors.part-aa \
    adapter/adapter_model.safetensors.part-ab \
    adapter/adapter_model.safetensors.part-ac \
    > adapter/adapter_model.safetensors
printf '%s  %s\n' \
  '52d945cc297248848c5d20619f79d68a35ec42b1f76dc674afdbb320dbf12c00' \
  'adapter/adapter_model.safetensors' | sha256sum -c -
```

Load the restored weights together with `adapter/adapter_config.json`. The
adapter is LoRA rank/alpha `112/112`, contains 392 tensors, and requires the
competition `OneReason-0.8B` base model. The base model is not included.

For a different parent adapter, restore only the E-clean Beam128 pool and the
retention source. Rerun Beam128 and rebuild the sidecar for that parent. Do not
reuse the published 66 boundary labels. See
`docs/I35_VIDEO_BOUNDARY_HANDOFF.md` for the porting checklist.

Adapter identity, split-part hashes, and the exact concatenation provenance are
recorded in `manifest.json` and the step-548 combination audit.
