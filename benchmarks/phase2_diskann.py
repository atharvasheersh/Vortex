"""Build and query a DiskANN 0.7.0 disk index from a Vortex L2 snapshot."""

import argparse
import importlib.metadata
import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np
from run_performance import digest, header, paging_observation, write_json
from run_scaling import available_memory
from vortex.evaluation import process_memory

VERSION = "0.7.0"
MIB = 1024**2


def diskann():
    if importlib.metadata.version("diskannpy") != VERSION:
        raise RuntimeError(f"install diskannpy=={VERSION} in the benchmark environment")
    import diskannpy

    return diskannpy


def export_snapshot(snapshot, directory):
    """Copy bounded chunks in physical row order, including the public-ID map."""
    from vortex import Index

    info = header(snapshot)
    if not 10 <= info["n"] <= 2**31 - 1 or not 1 <= info["d"] <= 2**31 - 1:
        raise ValueError("snapshot exceeds the supported graph row or dimension range")
    checked = Index.open(str(snapshot), cache_bytes=0, threads=1)
    del checked
    n, d = info["n"], info["d"]
    with (
        snapshot.open("rb") as source,
        (directory / "vectors.bin").open("xb") as vectors,
        (directory / "ids.u64").open("xb") as ids,
    ):
        vectors.write(struct.pack("<ii", n, d))
        for target, offset, count in (
            (ids, info["offsets"][0], n * 8),
            (vectors, info["offsets"][1], n * d * 4),
        ):
            source.seek(offset)
            while count:
                block = source.read(min(count, 1024**2))
                if not block:
                    raise ValueError("incomplete snapshot payload")
                if (
                    target is vectors
                    and not np.isfinite(np.frombuffer(block, "<f4")).all()
                ):
                    raise ValueError("graph input contains nonfinite vectors")
                target.write(block)
                count -= len(block)
    return n, d


def build(snapshot, directory, degree, complexity, threads, build_gib, search_gib):
    """Publish a completion marker only after build and reopen succeed."""
    info = header(snapshot)
    export_bytes = info["n"] * (info["d"] * 4 + 8)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Conservative disk gate for staging, graph sectors and intermediate files.
    graph_bytes = info["n"] * (info["d"] * 4 + degree * 4 + 64)
    if (
        shutil.disk_usage(directory.parent).free
        < export_bytes + 3 * graph_bytes + 512 * MIB
    ):
        raise OSError("insufficient free disk for graph export and build intermediates")
    available = available_memory()
    if available is not None and available < (build_gib * 1024 + 384) * MIB:
        raise MemoryError(
            "insufficient available RAM for the declared graph build budget"
        )
    api = diskann()
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "status": "started",
        "version": VERSION,
        "metric": "squared_l2",
        "snapshot_sha256": digest(snapshot),
        "snapshot_bytes": snapshot.stat().st_size,
        "degree": degree,
        "build_complexity": complexity,
        "build_threads": threads,
        "build_memory_gib": build_gib,
        "search_memory_gib": search_gib,
        "pq_disk_bytes": 0,
        "available_before_bytes": available,
        "vectors": info["n"],
        "dimension": info["d"],
    }
    write_json(directory / "build.json", manifest)
    try:
        begin = time.perf_counter()
        export_snapshot(snapshot, directory)
        manifest["export_seconds"] = time.perf_counter() - begin
        manifest["vectors_sha256"] = digest(directory / "vectors.bin")
        manifest["id_map_sha256"] = digest(directory / "ids.u64")
        begin = time.perf_counter()
        api.build_disk_index(
            str(directory / "vectors.bin"),
            "l2",
            str(directory),
            complexity,
            degree,
            search_gib,
            build_gib,
            threads,
            pq_disk_bytes=0,
            vector_dtype=np.float32,
        )
        manifest["build_seconds"] = time.perf_counter() - begin
        write_json(directory / "build.json", manifest)
        index = api.StaticDiskIndex(
            str(directory), num_threads=2, num_nodes_to_cache=0, cache_mechanism=0
        )
        del index
        manifest["index_files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name.startswith("ann")
        }
        if not any(name.endswith("_disk.index") for name in manifest["index_files"]):
            raise ValueError("DiskANN did not publish the disk index")
        manifest["index_bytes"] = (
            sum(f["bytes"] for f in manifest["index_files"].values()) + info["n"] * 8
        )
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(
            elapsed_seconds=time.perf_counter() - started,
            memory=process_memory(),
            paging=paging_observation(),
        )
        write_json(directory / "build.json", manifest)
    return manifest


class Graph:
    """Read-only benchmark adapter exposing public IDs and squared L2 scores."""

    def __init__(self, directory, threads=2, cache_nodes=0):
        self.info = json.loads((directory / "build.json").read_text(encoding="utf-8"))
        if self.info["status"] != "completed" or self.info["version"] != VERSION:
            raise ValueError("graph build is incomplete or has an unsupported version")
        self.count = self.info["vectors"]
        self.dimension = self.info["dimension"]
        if (directory / "ids.u64").stat().st_size != self.count * 8:
            raise ValueError("graph public-ID map has an unexpected size")
        self.ids = np.memmap(
            directory / "ids.u64", mode="r", dtype="<u8", shape=(self.count,)
        )
        self.index = diskann().StaticDiskIndex(
            str(directory),
            num_threads=threads,
            num_nodes_to_cache=cache_nodes,
            cache_mechanism=1 if cache_nodes else 0,
        )

    def _translate(self, ids, scores):
        if np.any(ids >= self.count) or not np.isfinite(scores).all():
            raise RuntimeError("graph returned invalid rows or scores")
        return sorted(
            zip(self.ids[ids].tolist(), scores.tolist()),
            key=lambda hit: (hit[1], hit[0]),
        )

    def search(self, vector, k, complexity, beam):
        if (
            vector.dtype != np.float32
            or vector.shape != (self.dimension,)
            or not np.isfinite(vector).all()
        ):
            raise ValueError(
                "expected one finite float32 query with matching dimension"
            )
        rank = self.index.search(vector, k, complexity, beam)
        return self._translate(rank.identifiers, rank.distances)

    def search_batch(self, vectors, k, complexity, beam, threads):
        if (
            vectors.dtype != np.float32
            or vectors.ndim != 2
            or vectors.shape[1] != self.dimension
            or not np.isfinite(vectors).all()
        ):
            raise ValueError(
                "expected a finite float32 query batch with matching dimension"
            )
        response = self.index.batch_search(vectors, k, complexity, beam, threads)
        return [
            self._translate(ids, scores)
            for ids, scores in zip(response.identifiers, response.distances)
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--complexity", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--build-gib", type=float, default=1.5)
    parser.add_argument("--search-gib", type=float, default=0.25)
    args = parser.parse_args()
    build(
        args.snapshot,
        args.directory,
        args.degree,
        args.complexity,
        args.threads,
        args.build_gib,
        args.search_gib,
    )


if __name__ == "__main__":
    main()
