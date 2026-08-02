#!/usr/bin/env python3
"""Publish a sanitized, reproducible snapshot of StreamLake experiment metadata.

The StreamLake volume contains raw evaluation logs, raw experiment JSON/SQLite,
and model artifacts. Only the exported experiment index and approved aggregate
metrics are copied into the repository by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_COLUMNS = ("id", "name", "status", "created_at", "updated_at")
METRIC_COLUMNS = (
    "experiment_type",
    "experiment_id",
    "name",
    "category",
    "numeric_value",
    "display_value",
    "unit",
    "direction",
)

# These are aggregate scores and training-trajectory summaries. Names outside
# this allowlist are intentionally omitted so a future exporter cannot publish
# arbitrary payload fields by accident.
METRIC_NAME_PATTERNS = (
    re.compile(r"^score$"),
    re.compile(r"^detail\.score$"),
    re.compile(r"^detail\.metrics\.score$"),
    re.compile(r"^detail\.metrics\.metrics\.summary\.[A-Za-z0-9_.-]+$"),
    re.compile(r"^(train_loss|eval_loss|learning_rate|percentage)\.[A-Za-z0-9_.-]+$"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    def write(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    atomic_write(path, write)


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metric_is_public(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in METRIC_NAME_PATTERNS)


def parse_catalog_generated_at(catalog_path: Path) -> str | None:
    prefix = "Generated:"
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def display_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    except ValueError:
        return value


def render_catalog(evaluations: list[dict[str, str]], finetunes: list[dict[str, str]], generated_at: str | None) -> str:
    rows = [("evaluation", row) for row in evaluations] + [("finetune", row) for row in finetunes]

    def created_sort_key(item):
        value = item[1].get("created_at", "")
        try:
            return (0, float(value), item[1].get("id", ""))
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return (0, parsed.timestamp(), item[1].get("id", ""))
            except ValueError:
                return (1, value, item[1].get("id", ""))

    rows.sort(key=created_sort_key)
    lines = [
        "# StreamLake Experiment Catalog",
        "",
        "> Sanitized metadata snapshot. Raw logs, raw JSON/SQLite, prompts, generations, and model artifacts are excluded.",
        f"> Source catalog timestamp: `{generated_at or 'unknown'}`.",
        "",
        "| Type | ID | Name | Status | Created | Updated |",
        "|---|---|---|---|---|---|",
    ]
    for experiment_type, row in rows:
        values = [
            experiment_type,
            row.get("id", ""),
            row.get("name", "").replace("|", "\\|"),
            row.get("status", ""),
            display_timestamp(row.get("created_at", "")),
            display_timestamp(row.get("updated_at", "")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            f"Total: **{len(rows)}** experiments ({len(evaluations)} evaluations, {len(finetunes)} fine-tuning runs).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/streamlake_experiments"),
    )
    parser.add_argument(
        "--evaluation-logs-dir",
        type=Path,
        default=Path("/lustre/prod_glm_volumes/volume-20260201002229-o7c51/streamlake_evaluation_logs"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/streamlake"))
    args = parser.parse_args()

    exports = args.experiments_dir / "exports"
    evaluation_csv = exports / "evaluation.csv"
    finetune_csv = exports / "finetune.csv"
    metrics_csv = exports / "metrics.csv"
    catalog_md = args.experiments_dir / "catalog.md"
    sync_state_path = args.experiments_dir / "sync_state.json"
    required = (evaluation_csv, finetune_csv, metrics_csv, catalog_md)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing required StreamLake export(s): " + ", ".join(missing))

    evaluations = read_csv(evaluation_csv)
    finetunes = read_csv(finetune_csv)
    source_metrics = read_csv(metrics_csv)
    metrics = [row for row in source_metrics if metric_is_public(row.get("name", ""))]
    skipped_metrics = len(source_metrics) - len(metrics)

    output_dir = args.output_dir
    write_csv(output_dir / "evaluations.csv", evaluations, EXPERIMENT_COLUMNS)
    write_csv(output_dir / "finetunes.csv", finetunes, EXPERIMENT_COLUMNS)
    write_csv(output_dir / "metrics.csv", metrics, METRIC_COLUMNS)
    atomic_write(output_dir / "catalog.md", lambda handle: handle.write(render_catalog(evaluations, finetunes, parse_catalog_generated_at(catalog_md))))

    source_catalog_generated_at = parse_catalog_generated_at(catalog_md)
    source_sync_state = None
    if sync_state_path.is_file():
        with sync_state_path.open("r", encoding="utf-8") as handle:
            candidate_state = json.load(handle)
        if isinstance(candidate_state, dict):
            source_sync_state = {
                key: candidate_state[key]
                for key in ("error_count", "experiment_count", "metric_count", "parameter_count", "source_counts", "updated_at")
                if key in candidate_state
            }
    source_files = [evaluation_csv, finetune_csv, metrics_csv, catalog_md]
    manifest = {
        "schema_version": 1,
        "source_catalog_generated_at": source_catalog_generated_at,
        "source_sync_state": source_sync_state,
        "snapshot": {
            "evaluation_count": len(evaluations),
            "finetune_count": len(finetunes),
            "metric_count": len(metrics),
            "metric_rows_skipped_by_allowlist": skipped_metrics,
        },
        "source_exports": [
            {
                "label": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_files
        ],
        "excluded": {
            "evaluation_logs": str(args.evaluation_logs_dir.name),
            "raw_experiment_json_and_sqlite": True,
            "model_weights_and_tokenizers": True,
            "prompts_and_generated_outputs": True,
            "credentials_and_environment_values": True,
        },
        "generated_at": source_catalog_generated_at,
    }
    atomic_write(output_dir / "manifest.json", lambda handle: json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True))

    print(
        f"wrote {len(evaluations)} evaluations, {len(finetunes)} fine-tunes, "
        f"{len(metrics)} public metrics to {output_dir} (skipped {skipped_metrics} metrics)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
