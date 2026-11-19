"""Add FP16/INT8 search payloads to a version-1 FP32 snapshot in bounded chunks."""

import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from repack_snapshot import sha256, snapshot_info
from vortex import Index


def encode_snapshot(
    source, destination, dtype, chunk_rows=4096, check_resources=lambda: None
):
    started = time.perf_counter()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    partial = destination.with_name(destination.name + ".partial")
    if dtype not in {"int8", "fp16"} or chunk_rows < 1:
        raise ValueError("require fp16/int8 and positive chunk_rows")
    if destination.exists() or partial.exists() or source == destination:
        raise FileExistsError("output or partial already exists")
    check_resources()
    info = snapshot_info(source)
    if info["dtype"]:
        raise ValueError("source must use FP32")
    n, d, lists = info["count"], info["dimension"], info["nlist"]
    old = info["positions"]
    stride = d * 2 if dtype == "fp16" else d + 4
    centroids = (old[2] + n * stride + 7) & ~7
    offsets = centroids + lists * d * 4
    postings = offsets + (lists + 1) * 8
    total = postings + (n * 8 if lists else 0)
    positions = (old[0], old[1], old[2], centroids, offsets, postings, total)
    raw = bytearray(info["header"])
    struct.pack_into("<I", raw, 20, 1 if dtype == "fp16" else 2)
    struct.pack_into("<7Q", raw, 40, *positions)
    checksum = 14695981039346656037
    for byte in raw[:96]:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    struct.pack_into("<Q", raw, 96, checksum)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, partial.open("xb+") as writer:
        writer.truncate(total)
        # Copy IDs/originals unchanged and relocate centroids/offsets/postings.
        for begin, end, target in [(128, old[2], 128), (old[3], old[6], centroids)]:
            reader.seek(begin)
            writer.seek(target)
            remaining = end - begin
            while remaining:
                check_resources()
                block = reader.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("truncated source snapshot")
                writer.write(block)
                remaining -= len(block)
        reader.seek(old[1])
        writer.seek(old[2])
        for start in range(0, n, chunk_rows):
            check_resources()
            rows = min(chunk_rows, n - start)
            matrix = np.fromfile(reader, dtype="<f4", count=rows * d).reshape(rows, d)
            if not np.isfinite(matrix).all():
                raise ValueError("source vectors must be finite")
            if dtype == "fp16":
                if np.max(np.abs(matrix), initial=0) > 65504:
                    raise ValueError("source exceeds finite FP16 range")
                matrix.astype("<f2").tofile(writer)
            else:
                maxima = np.max(np.abs(matrix), axis=1)
                scales = np.maximum(
                    maxima / np.float32(127),
                    np.finfo(np.float32).tiny,
                )
                scales[maxima == 0] = np.float32(1)
                divided = matrix / scales[:, None]
                codes = np.clip(
                    np.copysign(np.floor(np.abs(divided) + np.float32(0.5)), divided),
                    -127,
                    127,
                ).astype(np.int8)
                encoded = np.empty((rows, stride), dtype=np.uint8)
                encoded[:, :4] = scales.astype("<f4").view(np.uint8).reshape(rows, 4)
                encoded[:, 4:] = codes.view(np.uint8)
                encoded.tofile(writer)
        writer.seek(0)
        writer.write(raw)
        writer.flush()
        os.fsync(writer.fileno())
    check_resources()
    checked = Index.open(str(partial), cache_bytes=0)
    del checked
    os.link(partial, destination)
    partial.unlink()
    return {
        "source_sha256": sha256(source),
        "snapshot_sha256": sha256(destination),
        "snapshot_bytes": total,
        "dtype": dtype,
        "count": n,
        "dimension": d,
        "seconds": time.perf_counter() - started,
        "retains_fp32_originals": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dtype", required=True, choices=["fp16", "int8"])
    args = parser.parse_args()
    print(
        json.dumps(encode_snapshot(args.source, args.destination, args.dtype), indent=2)
    )


if __name__ == "__main__":
    main()
