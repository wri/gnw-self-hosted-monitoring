# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
#
# Compute a lightweight integrity "fingerprint" of a zarr store (a folder / S3 prefix made
# of many files), so a user who copies the zarr locally (or to their own bucket) can verify
# they copied it correctly and completely.
#
# The fingerprint is a SHA-256 over, for every file in the store, its relative path
# and size, PLUS the full byte-contents of the small zarr metadata files (zarr.json /
# .zarray / .zattrs / .zgroup / .zmetadata). It catches missing, extra, truncated, or
# wrong-version files and any metadata change, but not a same-size bit-flip inside a
# chunk (transfer tools such as `aws s3 cp/sync` already verify that during copy).
#
# The same fingerprint is produced whether the store is on S3 or a local filesystem, so a
# value computed on the source can be compared against a local (or re-uploaded) copy.
#
# To install (same environment as post2020.py):
#  - Make sure python 3.12 or close to that is installed, and "pipenv" is installed
#  - Make a directory, copy hash_zarr.py into directory, cd to directory
#  - Run 'pipenv install xarray shapely pandas fiona zarr fsspec s3fs rioxarray "dask[array,dataframe,distributed,diagnostics]" coiled'
#  - For an S3 store, 'export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...'
#
# Usage:
#    pipenv run python ./hash_zarr.py <local path or s3://... path to the zarr folder>

import sys
import hashlib

import fsspec

# Files whose full contents are folded into the hash (small metadata; cheap to read).
# Every other file (chunk data) contributes only its relative path and size.
META_NAMES = ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")


def hash_zarr(store_uri: str) -> str:
    """Return a hex SHA-256 fingerprint of the zarr store at `store_uri` (local or S3)."""
    # requester_pays only matters for S3 (the published zarrs are requester-pays); a local
    # path ignores it.
    storage_options = {"requester_pays": True} if store_uri.startswith("s3://") else {}
    fs, root = fsspec.core.url_to_fs(store_uri, **storage_options)
    root = root.rstrip("/")

    # fs.find returns the leaf files (not directories) under the store, with their sizes.
    # We keep only (rel, size) per file; the full path is root + "/" + rel when we need it.
    found = fs.find(root, detail=True)
    entries = []
    for path, meta in found.items():
        rel = path[len(root):].lstrip("/")
        if rel:  # skip any entry for the store root itself
            entries.append((rel, meta["size"]))
    entries.sort(key=lambda e: e[0])

    if not entries:
        sys.exit(f"Error: no files found at {store_uri} (is it a valid zarr store?)")

    h = hashlib.sha256()
    total = 0
    for rel, size in entries:
        total += size
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        # Fold in the full contents of the small metadata files.
        if rel.rsplit("/", 1)[-1] in META_NAMES:
            h.update(fs.cat_file(f"{root}/{rel}"))
            h.update(b"\0")

    # A short summary to stderr; stdout carries only the hash so it is easy to script.
    print(f"{len(entries)} files, {total} bytes", file=sys.stderr)

    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python ./hash_zarr.py <local-or-s3-path-to-zarr>")
        sys.exit(1)
    print(hash_zarr(sys.argv[1]))


if __name__ == "__main__":
    main()
