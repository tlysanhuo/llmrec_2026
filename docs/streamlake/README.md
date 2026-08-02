# StreamLake Experiment Snapshot

This directory is the public, sanitized snapshot of the StreamLake experiment registry.
It is generated from the mounted experiment volume with:

```bash
python scripts/maintenance/sync_streamlake_snapshot.py
```

Published files contain experiment IDs, names, timestamps, and approved aggregate metrics only:

- `catalog.md` is the human-readable combined experiment index.
- `evaluations.csv` and `finetunes.csv` are metadata indexes with raw paths removed.
- `metrics.csv` contains allowlisted aggregate scores and training-trajectory summaries.
- `manifest.json` records source export hashes, row counts, and the exclusion boundary.

Raw evaluation logs, raw experiment JSON/SQLite, prompts, generated responses, credentials,
tokenizers, and model weights remain on the experiment volume and are intentionally not synced.
