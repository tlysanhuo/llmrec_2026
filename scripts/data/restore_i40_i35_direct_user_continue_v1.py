#!/usr/bin/env python3
"""Verify and restore the checked-in I-40 training data release."""

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
RELEASE_DIR = REPO_ROOT / "assets/derived/releases/i40_i35_direct_user_continue_r112_v1"
PROCESSED_DIR = REPO_ROOT / "assets/derived/processed"
BUFFER_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class Asset:
    name: str
    archive_bytes: int
    archive_sha256: str
    payload_bytes: int
    payload_rows: int
    payload_sha256: str

    @property
    def archive(self) -> Path:
        return RELEASE_DIR / f"{self.name}.gz"

    @property
    def output(self) -> Path:
        return PROCESSED_DIR / self.name


ASSETS = (
    Asset(
        name="data_i40_i35_direct_user_continue_v1.jsonl",
        archive_bytes=17_610_470,
        archive_sha256="a6e710a45e73658449f74a541c9c21fe7c5c055dd49bc41814c2cdd94433906f",
        payload_bytes=102_365_905,
        payload_rows=8_240,
        payload_sha256="483a4bb2f98d41497600d078032634d4f36fe2970a53d98b4a7fccc488910c18",
    ),
    Asset(
        name="data_i40_i35_direct_user_continue_v1_sidecar.jsonl",
        archive_bytes=1_786_904,
        archive_sha256="637a9b01735fdbe2d896513983d583b0211e610c6165c743b0defe39def0f16c",
        payload_bytes=7_605_609,
        payload_rows=8_240,
        payload_sha256="e9bc129cd834bff161247985cc5430cf46872006cbae4a86fd37c3666b60acb2",
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


def verify_archive(asset: Asset) -> None:
    if not asset.archive.is_file():
        raise FileNotFoundError(f"release archive not found: {asset.archive}")
    digest, size = sha256_and_size(asset.archive)
    if size != asset.archive_bytes:
        raise ValueError(
            f"archive size mismatch for {asset.name}: expected {asset.archive_bytes}, got {size}"
        )
    if digest != asset.archive_sha256:
        raise ValueError(
            f"archive SHA256 mismatch for {asset.name}: expected {asset.archive_sha256}, got {digest}"
        )


def stream_payload(
    asset: Asset, source: BinaryIO, destination: BinaryIO | None = None
) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    rows = 0
    for line in source:
        size += len(line)
        rows += 1
        if size > asset.payload_bytes or rows > asset.payload_rows:
            raise ValueError(f"payload exceeds registered limits for {asset.name}")
        digest.update(line)
        if destination is not None:
            destination.write(line)

    observed_sha256 = digest.hexdigest()
    if size != asset.payload_bytes:
        raise ValueError(
            f"payload size mismatch for {asset.name}: expected {asset.payload_bytes}, got {size}"
        )
    if rows != asset.payload_rows:
        raise ValueError(
            f"payload row mismatch for {asset.name}: expected {asset.payload_rows}, got {rows}"
        )
    if observed_sha256 != asset.payload_sha256:
        raise ValueError(
            f"payload SHA256 mismatch for {asset.name}: expected {asset.payload_sha256}, got {observed_sha256}"
        )
    return {"bytes": size, "rows": rows, "sha256": observed_sha256}


def verify_compressed_payload(asset: Asset) -> dict[str, int | str]:
    with gzip.open(asset.archive, "rb") as source:
        return stream_payload(asset, source)


def verify_raw(asset: Asset) -> dict[str, int | str]:
    with asset.output.open("rb") as source:
        return stream_payload(asset, source)


def restore(asset: Asset, force: bool) -> tuple[str, dict[str, int | str]]:
    if asset.output.exists():
        try:
            summary = verify_raw(asset)
        except (OSError, ValueError):
            if not force:
                raise FileExistsError(
                    f"output exists but is not the registered payload: {asset.output}; "
                    "use --force only when replacement is intentional"
                )
        else:
            return "already_present", summary

    asset.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{asset.output.name}.",
            suffix=".tmp",
            dir=asset.output.parent,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            with gzip.open(asset.archive, "rb") as source:
                summary = stream_payload(asset, source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, asset.output)
        temporary_path = None
        return "restored", summary
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or atomically restore the complete I-40 data and routing sidecar."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify both archives and decompressed payloads without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output only if it fails the registered checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, object] = {}
    try:
        for asset in ASSETS:
            verify_archive(asset)
            if args.verify_only:
                status = "verified"
                payload = verify_compressed_payload(asset)
            else:
                status, payload = restore(asset, args.force)
            results[asset.name] = {
                "status": status,
                "archive": str(asset.archive),
                "archive_bytes": asset.archive_bytes,
                "archive_sha256": asset.archive_sha256,
                "output": None if args.verify_only else str(asset.output),
                "payload": payload,
            }
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
