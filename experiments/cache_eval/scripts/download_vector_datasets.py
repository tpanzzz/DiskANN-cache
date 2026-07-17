#!/usr/bin/env python3
"""Download and verify the small vector datasets used by cache experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "experiments/cache_eval/datasets/catalog.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Dataset catalog JSON.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fashion_mnist", "glove_25", "lastfm_64"],
        help="Catalog dataset names to download.",
    )
    parser.add_argument(
        "--max-base-vectors",
        type=int,
        default=10_000_000,
        help="Refuse datasets above this base-vector count.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_local_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


def verify(path: Path, spec: dict[str, object]) -> None:
    expected_bytes = int(spec["expected_bytes"])
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path}: expected {expected_bytes} bytes, found {actual_bytes}"
        )

    expected_sha = spec.get("sha256")
    if expected_sha:
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{path}: expected sha256 {expected_sha}, found {actual_sha}"
            )
        print(f"verified {path} sha256={actual_sha}")
    else:
        print(f"verified {path} size={actual_bytes}; no catalog checksum")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"downloading {url} -> {destination}")
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    partial.replace(destination)


def main() -> int:
    args = parse_args()
    catalog = json.loads(args.catalog.read_text())

    for name in args.datasets:
        if name not in catalog:
            raise KeyError(f"unknown dataset {name!r}")
        spec = catalog[name]
        base_count = int(spec["base_count"])
        if base_count > args.max_base_vectors:
            raise RuntimeError(
                f"{name}: {base_count} base vectors exceeds limit "
                f"{args.max_base_vectors}"
            )

        destination = resolve_local_path(str(spec["local_hdf5"]))
        if not destination.exists():
            if args.verify_only:
                raise FileNotFoundError(destination)
            download(str(spec["url"]), destination)
        verify(destination, spec)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
