# Repository operating rules

## Asset management

- Before any data, training, evaluation, or packaging task, read `docs/reference/ASSETS.md` and use it as the sole asset registry.
- Do not recursively scan the workspace or volume to rediscover datasets. A targeted check is allowed only when a registered path is missing, a checksum fails, an official release changes, or the user explicitly requests an audit.
- Only registry class `O` may be called official direct data. Class `D` must be described as official-source derived, never as official original data.
- Official direct assets are immutable. Never edit, delete, rename, overwrite, or write generated files inside `assets/official/` or their source directories.
- Third-party class `T` data is disabled by default and requires explicit user approval before training use.
- Every new derived training dataset must record upstream asset IDs, builder script, row count, content hash, and mix ratio in its experiment config or ledger before formal training starts.
- Any asset addition, removal, move, or provenance correction must update `docs/reference/ASSETS.md` immediately. Do not create another inventory.
- Never place JSONL, Parquet, archives, or other dataset files directly in the repository root. Put them in the registered `assets/` partition and expose compatibility paths through `data/` only when required.

## Competition execution

- Read `README.md`, `ideas/README.md`, and `docs/TODO.md` before proposing or launching an experiment.
- New training configs belong in `configs/active/`; root-level YAML files are historical unless explicitly stated otherwise.
- All new local training is single-GPU with W&B enabled. Choose the epoch count,
  learning-rate schedule, and adapter-only checkpoint cadence from the training
  trajectory and experiment goal; do not impose a universal one-epoch limit.
- Do not start training from a failed checkpoint. Use `docs/EXPERIMENT_INDEX.md`
  for the retained-checkpoint list and each artifact's allowed role.
- Do not infer the latest artifact from modification time. Use `docs/EXPERIMENT_INDEX.md`.
