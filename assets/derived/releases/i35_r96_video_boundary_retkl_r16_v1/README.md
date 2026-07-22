# I-35 video boundary residual release

This release contains the data, sidecar, audits, and training logs for
`i35_r96_video_boundary_retkl_r16_v1`. The step-548 residual was concatenated
with the rank-96 parent to produce the rank-112 adapter that scored
`1.0344285849069457`.

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

For a different parent adapter, restore only the E-clean Beam128 pool and the
retention source. Rerun Beam128 and rebuild the sidecar for that parent. Do not
reuse the published 66 boundary labels. See
`docs/I35_VIDEO_BOUNDARY_HANDOFF.md` for the porting checklist.

The combined rank-112 adapter is intentionally not included in this data/log
release. Adapter identity and hashes are recorded in `manifest.json` and the
step-548 combination audit.
