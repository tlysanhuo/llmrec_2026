#!/usr/bin/env python3
"""Verify and restore the checked-in I-18 training dataset release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO_ROOT / "assets/derived/releases/seed_teacher_cotfix_v2"
DEFAULT_ARCHIVE = RELEASE_DIR / "data_seed_teacher_cotfix_v2.jsonl.gz"
DEFAULT_OUTPUT = RELEASE_DIR / "data_seed_teacher_cotfix_v2.jsonl"

ARCHIVE_BYTES = 52_199_218
ARCHIVE_SHA256 = "193cd78f1689a3ec2ffb2fc1d2167c8d55bfe364ddcb33ca61576939231807f9"
DATASET_BYTES = 249_454_095
DATASET_ROWS = 32_644
DATASET_SHA256 = "634c4805367308b35dd729c17f59a1a8b4bb473b84a80d21cc71931a2c29c0e4"
BUFFER_SIZE = 4 * 1024 * 1024


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_archive(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"release archive not found: {path}")

    sha256, size = sha256_and_size(path)
    if size != ARCHIVE_BYTES:
        raise ValueError(f"archive size mismatch: expected {ARCHIVE_BYTES}, got {size}")
    if sha256 != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {sha256}")


def stream_dataset(source: BinaryIO, destination: BinaryIO | None = None) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    rows = 0

    for line in source:
        size += len(line)
        rows += 1
        if size > DATASET_BYTES or rows > DATASET_ROWS:
            raise ValueError("decompressed dataset exceeds the registered size or row count")
        digest.update(line)
        if destination is not None:
            destination.write(line)

    sha256 = digest.hexdigest()
    if size != DATASET_BYTES:
        raise ValueError(f"dataset size mismatch: expected {DATASET_BYTES}, got {size}")
    if rows != DATASET_ROWS:
        raise ValueError(f"dataset row count mismatch: expected {DATASET_ROWS}, got {rows}")
    if sha256 != DATASET_SHA256:
        raise ValueError(f"dataset SHA256 mismatch: expected {DATASET_SHA256}, got {sha256}")

    return {"bytes": size, "rows": rows, "sha256": sha256}


def verify_raw(path: Path) -> dict[str, int | str]:
    with path.open("rb") as source:
        return stream_dataset(source)


def verify_compressed_payload(archive: Path) -> dict[str, int | str]:
    with gzip.open(archive, "rb") as source:
        return stream_dataset(source)


def restore(archive: Path, output: Path, force: bool) -> tuple[str, dict[str, int | str]]:
    if output.exists():
        try:
            summary = verify_raw(output)
        except (OSError, ValueError):
            if not force:
                raise FileExistsError(
                    f"output exists but does not match the registered dataset: {output}; "
                    "use --force only if replacing it is intentional"
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
                summary = stream_dataset(source, destination)
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
        description="Verify or atomically restore the checked-in seed_teacher_cotfix_v2 JSONL release."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="path to the .jsonl.gz release")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination for restored JSONL")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the archive and its decompressed payload without writing a JSONL file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output only when it fails registered hash/size/row checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()

    try:
        verify_archive(archive)
        if args.verify_only:
            status = "verified"
            dataset = verify_compressed_payload(archive)
        else:
            status, dataset = restore(archive, output, args.force)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    result = {
        "status": status,
        "archive": str(archive),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "dataset": dataset,
    }
    if not args.verify_only:
        result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
