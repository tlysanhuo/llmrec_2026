# Data entry points

This directory is a compatibility catalog. It contains no standalone dataset copies.

- `official/` -> competition-official direct assets
- `derived/` -> locally generated datasets
- `third_party/` -> teammate and other non-official datasets; disabled by default
- `evaluation/` -> evaluation-only assets; never train by default
- `processed/` -> compatibility link to `derived/processed/`

Canonical provenance and policy: [`docs/reference/ASSETS.md`](../docs/reference/ASSETS.md).

Official includes Explorer competition data, aligned SFT Caption/Tag, OpenOneRec General-Pretrain, OpenOneRec General-SFT, and the base model entry registered through `assets/official/`.
