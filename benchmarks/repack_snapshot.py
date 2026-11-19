"""Reorder a version-1 snapshot by IVF list with bounded component buffers."""

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from vortex import Index


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_info(path):
    # Reuse the native format/corruption validation, rather than maintain a
    # weaker parser. IDs/postings metadata are resident during this check;
    # component arrays are never materialized as a complete matrix.
    checked = Index.open(str(path), cache_bytes=0)
    del checked
    with Path(path).open("rb") as handle:
        raw = handle.read(128)
    values = struct.unpack_from("<8sIIIIQQ7Q", raw)
    return {
        "header": raw,
        "dimension": values[2],
        "metric": values[3],
        "dtype": values[4],
        "count": values[5],
        "nlist": values[6],
        "positions": values[7:],
    }


def repack(source, destination, chunk_rows=4096, check_resources=lambda: None):
    started = time.perf_counter()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    partial = destination.with_name(destination.name + ".partial")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if source == destination or destination.exists() or partial.exists():
        raise FileExistsError("output or partial output already exists")
    check_resources()
    info = snapshot_info(source)
    n, d, lists = info["count"], info["dimension"], info["nlist"]
    if not lists:
        raise ValueError("an IVF snapshot is required for list packing")
    positions = info["positions"]
    source_hash = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A byte mapping avoids an N*d component copy. Advanced indexing is confined
    # to chunk_rows rows; each destination section is written sequentially.
    mapped = np.memmap(source, mode="r", dtype=np.uint8)
    postings = np.ndarray((n,), dtype="<u8", buffer=mapped, offset=positions[5])
    ids = np.ndarray((n,), dtype="<u8", buffer=mapped, offset=positions[0])
    originals = np.ndarray(
        (n, d * 4), dtype=np.uint8, buffer=mapped, offset=positions[1]
    )
    stride = d * 2 if info["dtype"] == 1 else d + 4
    encoded = (
        np.ndarray((n, stride), dtype=np.uint8, buffer=mapped, offset=positions[2])
        if info["dtype"]
        else None
    )
    with partial.open("xb+") as target:
        target.truncate(positions[-1])
        for start in range(0, n, chunk_rows):
            check_resources()
            end = min(n, start + chunk_rows)
            order = postings[start:end].astype(np.intp)
            target.seek(positions[0] + start * 8)
            ids[order].tofile(target)
            target.seek(positions[1] + start * d * 4)
            originals[order].tofile(target)
            if encoded is not None:
                target.seek(positions[2] + start * stride)
                encoded[order].tofile(target)
            target.seek(positions[5] + start * 8)
            np.arange(start, end, dtype="<u8").tofile(target)
        # Preserve centroids, padding and offsets byte for byte.
        for start in range(
            positions[2] + (n * stride if encoded is not None else 0),
            positions[5],
            65536,
        ):
            check_resources()
            end = min(positions[5], start + 65536)
            target.seek(start)
            target.write(mapped[start:end].tobytes())
        target.seek(0)
        target.write(info["header"])
        target.flush()
        os.fsync(target.fileno())
    del encoded, originals, ids, postings, mapped
    check_resources()
    verified = Index.open(str(partial), cache_bytes=0)
    del verified
    # Atomic publication without replacing an existing destination, even if a
    # second writer created it after the initial check. Retain partial failures.
    os.link(partial, destination)
    partial.unlink()
    return {
        "source_sha256": source_hash,
        "snapshot_sha256": sha256(destination),
        "snapshot": str(destination),
        "snapshot_bytes": destination.stat().st_size,
        "count": n,
        "dimension": d,
        "nlist": lists,
        "dtype": info["dtype"],
        "chunk_rows": chunk_rows,
        "seconds_including_validation": time.perf_counter() - started,
        "time_boundary": "input validation/hash through output publication/validation/hash",
        "layout": "version-1, physical IVF-list order; stable public IDs",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(repack(args.source, args.destination, args.chunk_rows), indent=2))


if __name__ == "__main__":
    main()
