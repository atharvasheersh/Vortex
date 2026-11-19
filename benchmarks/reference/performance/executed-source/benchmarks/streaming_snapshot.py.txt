"""Build one FP32 L2 snapshot with bounded buffers and native IVF centroids."""

import hashlib
import os
import struct
import time
from pathlib import Path

import numpy as np
from vortex import Index


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout(count, dimension, clusters):
    ids = 128
    original = ids + count * 8
    encoded = original + count * dimension * 4
    centroids = (encoded + 7) & ~7
    offsets = centroids + clusters * dimension * 4
    postings = offsets + (clusters + 1) * 8
    return (ids, original, encoded, centroids, offsets, postings, postings + count * 8)


def header(count, dimension, clusters):
    value = bytearray(128)
    struct.pack_into(
        "<8sIIIIQQ7Q",
        value,
        0,
        b"VORTEX01",
        1,
        dimension,
        0,
        0,
        count,
        clusters,
        *layout(count, dimension, clusters),
    )
    checksum = 14695981039346656037
    for byte in value[:96]:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    struct.pack_into("<Q", value, 96, checksum)
    return value


def blocks(path, count, dimension, chunk_rows):
    """Read ordinary bounded arrays, never a resident duplicate corpus matrix."""
    with Path(path).open("rb") as handle:
        handle.seek(layout(count, dimension, 1)[1])
        for start in range(0, count, chunk_rows):
            rows = min(chunk_rows, count - start)
            values = np.fromfile(handle, dtype="<f4", count=rows * dimension)
            if values.size != rows * dimension:
                raise ValueError("incomplete vector section")
            yield start, values.reshape(rows, dimension)


def vectors_at(path, count, dimension, identifiers):
    result = []
    with Path(path).open("rb") as handle:
        for identifier in identifiers:
            if not 0 <= int(identifier) < count:
                raise ValueError("vector row outside corpus")
            handle.seek(
                layout(count, dimension, 1)[1] + int(identifier) * dimension * 4
            )
            value = np.fromfile(handle, dtype="<f4", count=dimension)
            if value.size != dimension:
                raise ValueError("incomplete sampled vector")
            result.append(value)
    return np.array(result, dtype=np.float32)


def build_snapshot(
    directory,
    count,
    dimension,
    clusters,
    chunk_rows=4096,
    training_limit=10000,
    check_resources=lambda: None,
):
    """Generate full disk vectors, train a bounded native sample, and assign every row."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "collection.vtx"
    partial = directory / "collection.vtx.partial"
    labels_path = directory / "assignments.u32"
    training_path = directory / "training.vtx"
    if any(p.exists() for p in [path, partial, labels_path, training_path]):
        raise FileExistsError("snapshot build artifacts already exist")
    if count < clusters or dimension < 1 or chunk_rows < 1 or training_limit < clusters:
        raise ValueError("invalid streaming build configuration")
    positions = layout(count, dimension, clusters)
    timings, digest = {}, hashlib.sha256()
    started = time.perf_counter()
    rng = np.random.default_rng(42)
    with partial.open("xb+") as handle:
        handle.truncate(positions[-1])
        for start in range(0, count, chunk_rows):
            check_resources()
            rows = min(chunk_rows, count - start)
            values = rng.standard_normal((rows, dimension), dtype=np.float32)
            digest.update(memoryview(values).cast("B"))
            handle.seek(positions[0] + start * 8)
            np.arange(start, start + rows, dtype="<u8").tofile(handle)
            handle.seek(positions[1] + start * dimension * 4)
            values.astype("<f4", copy=False).tofile(handle)
            if start // 500000 != (start + rows) // 500000:
                print(f"generated {start + rows:,}/{count:,} vectors", flush=True)
        handle.flush()
        timings["generate_and_write_vectors"] = time.perf_counter() - started

        before = time.perf_counter()
        sample_ids = np.sort(
            np.random.default_rng(4201).choice(
                count,
                min(count, training_limit),
                replace=False,
            )
        )
        samples = vectors_at(partial, count, dimension, sample_ids)
        training = Index(dimension, metric="l2", threads=1)
        for identifier, vector in zip(sample_ids, samples):
            training.add(int(identifier), vector)
        training.build_ivf(
            clusters, iterations=8, seed=42, training_limit=training_limit
        )
        training.save(str(training_path))
        with training_path.open("rb") as source:
            native_header = source.read(128)
            source.seek(struct.unpack_from("<Q", native_header, 64)[0])
            centers = np.fromfile(source, dtype="<f4", count=clusters * dimension)
        centers = centers.reshape(clusters, dimension)
        del training, samples
        timings["sample_and_native_training"] = time.perf_counter() - before

        before = time.perf_counter()
        router = Index(dimension, metric="l2", threads=4)
        for identifier, vector in enumerate(centers):
            router.add(identifier, vector)
        counts = np.zeros(clusters, dtype=np.int64)
        with labels_path.open("xb") as labels_file:
            for start, values in blocks(partial, count, dimension, chunk_rows):
                check_resources()
                hits = router.search_batch(values, k=1)
                assigned = np.fromiter(
                    (row["hits"][0][0] for row in hits),
                    dtype="<u4",
                    count=len(values),
                )
                assigned.tofile(labels_file)
                counts += np.bincount(assigned, minlength=clusters)
                if start // 500000 != (start + len(values)) // 500000:
                    print(
                        f"assigned {start + len(values):,}/{count:,} vectors",
                        flush=True,
                    )
        del router
        if int(counts.sum()) != count:
            raise ValueError("assignment count differs from the full corpus")
        timings["assign_all_rows_native"] = time.perf_counter() - before

        before = time.perf_counter()
        offsets = np.concatenate(
            [np.array([0], dtype="<u8"), counts.cumsum().astype("<u8")]
        )
        cursors = offsets[:-1].copy()
        handle.seek(positions[3])
        centers.astype("<f4", copy=False).tofile(handle)
        handle.seek(positions[4])
        offsets.tofile(handle)
        with labels_path.open("rb") as labels_file:
            for start in range(0, count, chunk_rows):
                check_resources()
                assigned = np.fromfile(
                    labels_file, dtype="<u4", count=min(chunk_rows, count - start)
                )
                for cluster in np.unique(assigned):
                    rows = (start + np.flatnonzero(assigned == cluster)).astype("<u8")
                    handle.seek(positions[5] + int(cursors[cluster]) * 8)
                    rows.tofile(handle)
                    cursors[cluster] += len(rows)
        if not np.array_equal(cursors, offsets[1:]):
            raise ValueError("postings do not cover the complete assignment")
        # Publish a recognizable header only after every section has been written.
        handle.seek(0)
        handle.write(header(count, dimension, clusters))
        handle.flush()
        os.fsync(handle.fileno())
        timings["postings_and_finalize"] = time.perf_counter() - before
    partial.replace(path)
    return {
        "snapshot": str(path),
        "snapshot_bytes": path.stat().st_size,
        "corpus_sha256": digest.hexdigest(),
        "stages_seconds": timings,
        "training_sample_count": len(sample_ids),
        "training_sample_ids_sha256": hashlib.sha256(
            sample_ids.astype("<u8").tobytes()
        ).hexdigest(),
        "training_sample_seed": 4201,
        "list_sizes": counts.tolist(),
        "construction": "bounded Python snapshot writer; native sample IVF and native centroid assignment; one full FP32 L2 collection",
    }
