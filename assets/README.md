# Stable asset entry points

This directory contains stable links only; large files remain on the runtime volume.

- `official/`: competition-official direct assets; immutable.
- `derived/`: locally generated data; provenance must be tracked.
- `third_party/`: non-official data; disabled for training by default.
- `evaluation/`: platform-visible or offline evaluation assets; disabled for training by default.

Legacy `fewshot_seed.json` and `fewshot_v2.json` are official-seed-derived annotation anchors (`D(O1)`), not official direct assets. They remain at their existing paths because a checked-in builder references `fewshot_v2.json`.

Canonical registry: [`docs/reference/ASSETS.md`](../docs/reference/ASSETS.md).

Official includes the platform seed SFT, Explorer 17GB raw data, aligned Caption/Tag SFT data, OpenOneRec General-Pretrain, OpenOneRec General-SFT, and the competition base model.
