"""Collect isolated, reproducible optimization measurements without changing archives."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def paging_observation():
    # Optional observation only; the experiment does not require psutil.
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {
            "page_faults": getattr(info, "num_page_faults", None),
            "rss_bytes": info.rss,
        }
    except ImportError:
        return {"page_faults": None, "method": "psutil unavailable"}


def header(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(128)
    fields = struct.unpack_from("<8sIIIIQQ7Q", raw)
    if fields[0] != b"VORTEX01" or fields[1] != 1 or fields[3] != 0:
        raise ValueError("benchmark reference requires a version-1 L2 snapshot")
    return {"n": fields[5], "d": fields[2], "nlist": fields[6], "offsets": fields[7:]}


def collect_reference(snapshot, output, count=132, seed=20261119, query_file=None):
    import faiss

    output.mkdir(parents=True, exist_ok=False)
    info = header(snapshot)
    queries = np.random.default_rng(seed).standard_normal(
        (count, info["d"]), dtype=np.float32
    )
    if query_file is not None:
        supplied = np.load(query_file, allow_pickle=False)
        if (
            supplied.dtype != np.float32
            or supplied.ndim != 2
            or supplied.shape[1] != info["d"]
            or len(supplied) < count
            or not np.isfinite(supplied).all()
        ):
            raise ValueError(
                "supplied queries must be finite float32 with matching dimensions/count"
            )
        queries = np.ascontiguousarray(supplied[:count])
    np.save(output / "queries.npy", queries)
    best_scores = np.full((count, 10), np.inf, dtype=np.float32)
    best_ids = np.full((count, 10), np.iinfo(np.uint64).max, dtype=np.uint64)
    direct = [0, count // 2, count - 1]
    direct_scores = np.full((3, 10), np.inf)
    direct_ids = np.full((3, 10), np.iinfo(np.uint64).max, dtype=np.uint64)
    faiss.omp_set_num_threads(1)
    started = time.perf_counter()
    with snapshot.open("rb") as vectors, snapshot.open("rb") as identifiers:
        vectors.seek(info["offsets"][1])
        identifiers.seek(info["offsets"][0])
        for start in range(0, info["n"], 32768):
            rows = min(32768, info["n"] - start)
            matrix = np.fromfile(vectors, dtype="<f4", count=rows * info["d"]).reshape(
                rows, info["d"]
            )
            ids = np.fromfile(identifiers, dtype="<u8", count=rows)
            index = faiss.IndexFlatL2(info["d"])
            index.add(matrix)
            scores, positions = index.search(queries, min(10, rows))
            for q in range(count):
                all_scores = np.concatenate([best_scores[q], scores[q]])
                all_ids = np.concatenate([best_ids[q], ids[positions[q]]])
                order = np.lexsort((all_ids, all_scores))[:10]
                best_scores[q], best_ids[q] = all_scores[order], all_ids[order]
            for number, q in enumerate(direct):
                distances = np.sum(
                    (matrix.astype(np.float64) - queries[q]) ** 2, axis=1
                )
                scores64 = np.concatenate([direct_scores[number], distances])
                ids64 = np.concatenate([direct_ids[number], ids])
                order = np.lexsort((ids64, scores64))[:10]
                direct_scores[number], direct_ids[number] = (
                    scores64[order],
                    ids64[order],
                )
    for number, q in enumerate(direct):
        if set(best_ids[q]) != set(direct_ids[number]):
            raise AssertionError("FAISS/direct float64 reference boundary mismatch")
        np.testing.assert_allclose(
            best_scores[q], direct_scores[number], atol=1e-4, rtol=1e-5
        )
    np.save(output / "ids.npy", best_ids)
    np.save(output / "scores.npy", best_scores)
    write_json(
        output / "reference.json",
        {
            **info,
            "snapshot_sha256": digest(snapshot),
            "queries_sha256": digest(output / "queries.npy"),
            "ids_sha256": digest(output / "ids.npy"),
            "scores_sha256": digest(output / "scores.npy"),
            "query_seed": seed,
            "tuning_count": 32,
            "heldout_count": count - 32,
            "method": "blocked FAISS IndexFlatL2; three direct float64 full-corpus checks",
            "seconds": time.perf_counter() - started,
            "faiss_version": faiss.__version__,
            "query_origin": "supplied local embedding vectors"
            if query_file is not None
            else "synthetic independent seed",
            "supplied_queries_sha256": digest(query_file)
            if query_file is not None
            else None,
        },
    )


def measure(args):
    # Select the extension before importing any project modules. Baseline and
    # optimized modules have the same name and must run in separate processes.
    if args.module_dir:
        sys.path.insert(0, str(args.module_dir.resolve()))
    import _vortex
    from capacity_resources import working_set_limit
    from run_capacity import power_state
    from run_scaling import available_memory
    from vortex.evaluation import process_memory

    args.output.mkdir(parents=True, exist_ok=False)
    query_path = args.reference / "queries.npy"
    queries = np.load(query_path, allow_pickle=False)
    truth = np.load(args.reference / "ids.npy", allow_pickle=False)
    scores = np.load(args.reference / "scores.npy", allow_pickle=False)
    info = header(args.snapshot)
    reference_info = json.loads(
        (args.reference / "reference.json").read_text(encoding="utf-8")
    )
    if (info["n"], info["d"]) != (reference_info["n"], reference_info["d"]):
        raise ValueError("reference and snapshot shape mismatch")
    if (
        args.query_offset < 0
        or args.query_count < 1
        or args.query_offset + args.query_count > len(queries)
    ):
        raise ValueError("query range is outside the reference")
    if args.repeats < 1 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("positive repeats and time budget required")
    probes = [int(p) for p in args.probes.split(",")]
    if any(p < 0 or p > info["nlist"] for p in probes):
        raise ValueError("probe count outside snapshot lists")
    manifest = {
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_sha256": digest(args.snapshot),
        "snapshot_bytes": args.snapshot.stat().st_size,
        "snapshot_info": info,
        "module": str(Path(_vortex.__file__).resolve()),
        "module_sha256": digest(_vortex.__file__),
        "queries_sha256": digest(query_path),
        "reference_ids_sha256": digest(args.reference / "ids.npy"),
        "reference_scores_sha256": digest(args.reference / "scores.npy"),
        "query_offset": args.query_offset,
        "query_count": args.query_count,
        "repeats": args.repeats,
        "warmups": 3,
        "probes": probes,
        "threads": args.threads,
        "kernel": args.kernel,
        "cache_bytes": 8388608,
        "block_bytes": 4096,
        "working_set_mib": args.working_set_mib,
        "max_seconds": args.max_seconds,
        "numpy": np.__version__,
        "python": sys.version,
        "avx2_available": _vortex.avx2_available(),
        "cache_state": "warmed/OS-managed, not controlled cold",
        "time_boundary": "complete Index.search Python call",
        "started_unix": time.time(),
        "status": "started",
        "condition": args.condition,
        "round_id": args.round_id,
        "rerank_n": args.rerank_n,
        "active_kernel": "scalar"
        if args.kernel == "scalar"
        else (
            _vortex.active_kernel()
            if hasattr(_vortex, "active_kernel")
            else "not exposed by baseline"
        ),
        "paging_start": paging_observation(),
        "power_start": power_state(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): digest(p)
            for pattern in [
                "src/*.cpp",
                "include/vortex/*.hpp",
                "python/bindings.cpp",
                "CMakeLists.txt",
                "benchmarks/run_performance.py",
                "benchmarks/repack_snapshot.py",
                "benchmarks/encode_snapshot.py",
                "benchmarks/streaming_snapshot.py",
                "benchmarks/capacity_resources.py",
            ]
            for p in Path(__file__).resolve().parents[1].glob(pattern)
        },
    }
    write_json(args.output / "manifest.json", manifest)
    all_rows = []
    started = time.perf_counter()
    try:
        with working_set_limit(args.working_set_mib) as policy:
            manifest["memory_policy"] = policy
            opened = time.perf_counter()
            index = _vortex.Index.open(
                str(args.snapshot),
                cache_bytes=8388608,
                block_bytes=4096,
                threads=args.threads,
                kernel=args.kernel,
            )
            manifest["open_seconds"] = time.perf_counter() - opened
            manifest["memory_after_open"] = process_memory()
            with (args.output / "queries.jsonl").open("x", encoding="utf-8") as raw:
                for repeat in range(args.repeats):
                    order = np.random.default_rng(
                        742 + args.round_id + repeat
                    ).permutation(probes)
                    for probe in order.tolist():
                        for query in queries[
                            args.query_offset : args.query_offset
                            + min(3, args.query_count)
                        ]:
                            index.search(
                                query, k=10, nprobe=probe, rerank_n=args.rerank_n
                            )
                        sequence = np.random.default_rng(
                            1234 + args.round_id + repeat
                        ).permutation(
                            np.arange(
                                args.query_offset, args.query_offset + args.query_count
                            )
                        )
                        for qid in sequence.tolist():
                            available = available_memory()
                            if available is not None and available < 384 * 1024**2:
                                raise MemoryError("available RAM fell below 384 MiB")
                            if time.perf_counter() - started > args.max_seconds:
                                raise TimeoutError("declared run budget exhausted")
                            cpu = time.process_time()
                            before = time.perf_counter_ns()
                            found = index.search(
                                queries[qid], k=10, nprobe=probe, rerank_n=args.rerank_n
                            )
                            elapsed = (time.perf_counter_ns() - before) / 1e6
                            cpu_ms = (time.process_time() - cpu) * 1000
                            identifiers = [int(hit[0]) for hit in found["hits"]]
                            recall = (
                                len(set(identifiers) & set(truth[qid].tolist())) / 10
                            )
                            if probe == 0 and index.dtype == "fp32":
                                if recall != 1 or found["scanned"] != info["n"]:
                                    raise AssertionError(
                                        "exact retrieval disagrees with independent reference"
                                    )
                                np.testing.assert_allclose(
                                    [h[1] for h in found["hits"]],
                                    scores[qid],
                                    atol=1e-4,
                                    rtol=1e-5,
                                )
                            row = {
                                "query_id": qid,
                                "repeat": repeat,
                                "round_id": args.round_id,
                                "nprobe": probe,
                                "wall_ms": elapsed,
                                "native_ms": found["milliseconds"],
                                "cpu_ms": cpu_ms,
                                "recall_at_10": recall,
                                "scanned": found["scanned"],
                                "hits": found["hits"],
                                "unix": time.time(),
                                "timing_gap": elapsed > 60000
                                and elapsed > 10 * max(cpu_ms, 1),
                            }
                            raw.write(json.dumps(row, allow_nan=False) + "\n")
                            raw.flush()
                            all_rows.append(row)
            manifest["cache_stats"] = index.cache_stats()
            manifest["memory_final"] = process_memory()
            manifest["paging_end"] = paging_observation()
            manifest["power_end"] = power_state()
            if (
                policy["enabled"]
                and manifest["memory_final"]["peak_rss_bytes"]
                > policy["maximum_bytes"] + 65536
            ):
                raise MemoryError(
                    "process peak exceeded the resident limit plus 64 KiB accounting allowance"
                )
            manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["recorded_calls"] = len(all_rows)
        write_json(args.output / "manifest.json", manifest)
    summaries = []
    for probe in probes:
        rows = [r for r in all_rows if r["nprobe"] == probe]
        values = [r["wall_ms"] for r in rows]
        summaries.append(
            {
                "nprobe": probe,
                "calls": len(rows),
                "distinct_queries": len({r["query_id"] for r in rows}),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "mean_ms": float(np.mean(values)),
                "sequential_qps": 1000 / float(np.mean(values)),
                "recall_at_10": float(np.mean([r["recall_at_10"] for r in rows])),
                "mean_scanned": float(np.mean([r["scanned"] for r in rows])),
                "timing_gaps": sum(r["timing_gap"] for r in rows),
            }
        )
    write_json(args.output / "summary.json", summaries)
    print(json.dumps(summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reference = sub.add_parser("reference")
    reference.add_argument("--snapshot", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--count", type=int, default=132)
    reference.add_argument("--seed", type=int, default=20261119)
    reference.add_argument("--query-file", type=Path)
    run = sub.add_parser("measure")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--reference", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--module-dir", type=Path)
    run.add_argument("--query-offset", type=int, default=32)
    run.add_argument("--query-count", type=int, default=100)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--round-id", type=int, default=0)
    run.add_argument("--probes", default="0,24")
    run.add_argument("--kernel", choices=["auto", "scalar"], default="auto")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--rerank-n", type=int, default=0)
    run.add_argument("--working-set-mib", type=int, default=1536)
    run.add_argument("--max-seconds", type=float, default=1200)
    run.add_argument("--condition", default="unspecified")
    args = parser.parse_args()
    if args.command == "reference":
        if args.count <= 32:
            parser.error(
                "reference needs tuning queries and at least one held-out query"
            )
        collect_reference(
            args.snapshot, args.output, args.count, args.seed, args.query_file
        )
    else:
        measure(args)


if __name__ == "__main__":
    main()
