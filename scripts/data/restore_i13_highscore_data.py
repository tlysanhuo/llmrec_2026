#!/usr/bin/env python3
"""Verify and restore both training datasets used by the 0.9978 I-13 pipeline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DIR = REPO_ROOT / "assets/derived/releases/e3_userres_r80_retkl_v3_s875"
BUFFER_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    file_name: str
    rows: int
    data_bytes: int
    data_sha256: str
    archive_bytes: int
    archive_sha256: str

    @property
    def archive_name(self) -> str:
        return f"{self.file_name}.gz"


DATASETS = (
    DatasetSpec(
        key="parent",
        file_name="data_seed_teacher_v1.jsonl",
        rows=32_644,
        data_bytes=249_379_075,
        data_sha256="13c40526b93c81d428e39e68404fdb9ebf6cb9b910bdad31ebf70c7d054eee4f",
        archive_bytes=52_167_271,
        archive_sha256="10991aca557359f873aa372c56cda8684d9cdb00a38b88712b398e7dc2b11d01",
    ),
    DatasetSpec(
        key="residual",
        file_name="data_user_residual_retention_v1.jsonl",
        rows=6_106,
        data_bytes=74_716_566,
        data_sha256="bd947aad4f2e3e0ef409b53dbffa914e6a73bec7e32be41ca0834a9d923b08f0",
        archive_bytes=14_327_379,
        archive_sha256="b61e5578b4332b7e27db29f418cf0ddea9d8f7847191f6486ac6f4e25cb9cda4",
    ),
)


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_archive(path: Path, spec: DatasetSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"release archive not found: {path}")
    sha256, size = sha256_and_size(path)
    if size != spec.archive_bytes:
        raise ValueError(
            f"{spec.key} archive size mismatch: expected {spec.archive_bytes}, got {size}"
        )
    if sha256 != spec.archive_sha256:
        raise ValueError(
            f"{spec.key} archive SHA256 mismatch: expected {spec.archive_sha256}, got {sha256}"
        )


def stream_dataset(
    source: BinaryIO, spec: DatasetSpec, destination: BinaryIO | None = None
) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    rows = 0
    for line in source:
        size += len(line)
        rows += 1
        if size > spec.data_bytes or rows > spec.rows:
            raise ValueError(f"{spec.key} payload exceeds registered size or row count")
        digest.update(line)
        if destination is not None:
            destination.write(line)

    sha256 = digest.hexdigest()
    if size != spec.data_bytes:
        raise ValueError(
            f"{spec.key} data size mismatch: expected {spec.data_bytes}, got {size}"
        )
    if rows != spec.rows:
        raise ValueError(f"{spec.key} row mismatch: expected {spec.rows}, got {rows}")
    if sha256 != spec.data_sha256:
        raise ValueError(
            f"{spec.key} data SHA256 mismatch: expected {spec.data_sha256}, got {sha256}"
        )
    return {"bytes": size, "rows": rows, "sha256": sha256}


def verify_raw(path: Path, spec: DatasetSpec) -> dict[str, int | str]:
    with path.open("rb") as source:
        return stream_dataset(source, spec)


def verify_payload(archive: Path, spec: DatasetSpec) -> dict[str, int | str]:
    with gzip.open(archive, "rb") as source:
        return stream_dataset(source, spec)


def restore(
    archive: Path, output: Path, spec: DatasetSpec, force: bool
) -> tuple[str, dict[str, int | str]]:
    if output.exists():
        try:
            summary = verify_raw(output, spec)
        except (OSError, ValueError):
            if not force:
                raise FileExistsError(
                    f"output exists but does not match registered {spec.key} data: {output}; "
                    "use --force only if replacement is intentional"
                )
        else:
            return "already_present", summary

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as destination:
            temporary_path = Path(destination.name)
            with gzip.open(archive, "rb") as source:
                summary = stream_dataset(source, spec, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
        return "restored", summary
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or restore the two checked-in datasets for the I-13 high-score pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *(spec.key for spec in DATASETS)),
        default="all",
        help="select the parent dataset, residual dataset, or both",
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=DEFAULT_RELEASE_DIR, help="directory containing .gz files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RELEASE_DIR, help="directory for restored JSONL files"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify archives and decompressed payloads without writing JSONL files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output only when it fails registered checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_dir = args.archive_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected = [spec for spec in DATASETS if args.dataset in ("all", spec.key)]
    results = []

    try:
        for spec in selected:
            archive = archive_dir / spec.archive_name
            output = output_dir / spec.file_name
            verify_archive(archive, spec)
            if args.verify_only:
                status = "verified"
                dataset = verify_payload(archive, spec)
            else:
                status, dataset = restore(archive, output, spec, args.force)
            result = {
                "key": spec.key,
                "status": status,
                "archive": str(archive),
                "archive_bytes": spec.archive_bytes,
                "archive_sha256": spec.archive_sha256,
                "dataset": dataset,
            }
            if not args.verify_only:
                result["output"] = str(output)
            results.append(result)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps({"release": "I-13-highscore-0.9978", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
