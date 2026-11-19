"""Time one Vortex INT8 or DiskANN condition against a shared exact reference."""

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import _vortex
import numpy as np
import psutil
from capacity_resources import working_set_limit
from phase2_diskann import Graph
from run_capacity import power_state
from run_performance import digest, header, paging_observation, write_json
from vortex import Index
from vortex.evaluation import process_memory

MIB = 1024**2
THREADS = 2


def observe_io(process):
    try:
        return process.io_counters()._asdict()
    except (AttributeError, psutil.Error):
        return None


def run(args):
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("time budget must be positive and finite")
    reference = json.loads(
        (args.reference / "reference.json").read_text(encoding="utf-8")
    )
    query_path = args.reference / "queries.npy"
    ids_path = args.reference / "ids.npy"
    query_hash, ids_hash = digest(query_path), digest(ids_path)
    if (query_hash, ids_hash) != (reference["queries_sha256"], reference["ids_sha256"]):
        raise ValueError("exact reference checksums disagree")
    queries = np.load(query_path, allow_pickle=False)
    truth = np.load(ids_path, allow_pickle=False)
    stop = args.offset + args.count
    if args.offset < 0 or args.count < 1 or stop > len(queries):
        raise ValueError("query range outside reference")
    params = (args.complexity, args.beam)
    snapshot_sha = digest(args.snapshot)
    if (
        snapshot_sha != reference["snapshot_sha256"]
        or header(args.snapshot)["n"] != reference["n"]
    ):
        raise ValueError("snapshot and independent reference disagree")
    if args.backend == "graph" and not args.graph:
        raise ValueError("--graph required for graph query")
    if args.scan_threads is not None and args.backend != "int8":
        raise ValueError("scan workers apply only to Vortex INT8")
    package = (
        importlib.metadata.distribution("diskannpy")
        if args.backend == "graph"
        else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "started",
        "backend": args.backend,
        "round": args.round,
        "query_offset": args.offset,
        "query_count": args.count,
        "k": 10,
        "corpus_count": reference["n"],
        "dimension": reference["d"],
        "snapshot_sha256": snapshot_sha,
        "queries_sha256": query_hash,
        "reference_ids_sha256": ids_hash,
        "memory_limit_mib": 1536,
        "threads": THREADS,
        "warmups": 3,
        "seconds_budget": args.seconds,
        "cache_state": "warmed, OS-managed",
        "distance_metric": "squared_l2",
        "runtime": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_sha256": digest(_vortex.__file__),
        "active_vortex_kernel": _vortex.active_kernel(),
        "source_sha256": digest(Path(__file__)),
        "adapter_sha256": digest(Path(__file__).with_name("phase2_diskann.py")),
        "backend_version": package.version if package else _vortex.__version__,
        "int8_scan_threads": args.scan_threads,
        "graph_parameters": {
            "complexity": args.complexity,
            "beam": args.beam,
            "cache_nodes": args.cache_nodes,
        }
        if package
        else None,
        "ivf_parameters": {"nprobe": args.probes, "rerank_n": 100}
        if not package
        else None,
        "power_start": power_state(),
    }
    if package:
        manifest["backend_native_sha256"] = {
            str(path): digest(package.locate_file(path))
            for path in package.files
            if path.suffix == ".pyd"
        }
    write_json(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    samples = []
    process = psutil.Process()
    try:
        with working_set_limit(1536) as policy:
            manifest["memory_policy"] = policy
            before = time.perf_counter()
            if package:
                index = Graph(args.graph, threads=THREADS, cache_nodes=args.cache_nodes)
                if index.info["snapshot_sha256"] != snapshot_sha:
                    raise ValueError("graph build belongs to a different snapshot hash")

                def search(query):
                    return index.search(query, 10, *params), None

                def batch(queries):
                    return index.search_batch(queries, 10, *params, threads=THREADS)
            else:
                index = Index.open(str(args.snapshot), threads=THREADS)
                if index.dtype != "int8":
                    raise ValueError("control requires packed INT8")

                def search(query):
                    options = (
                        {}
                        if args.scan_threads is None
                        else {"scan_threads": args.scan_threads}
                    )
                    found = index.search(
                        query, k=10, nprobe=args.probes, rerank_n=100, **options
                    )
                    return found["hits"], found["scanned"]

                def batch(queries):
                    if args.scan_threads is not None:
                        # Compare the same serial request stream for all
                        # native ablations using the same per-query API.
                        return [search(query) for query in queries]
                    return index.search_batch(
                        queries, k=10, nprobe=args.probes, rerank_n=100
                    )

            manifest["open_seconds"] = time.perf_counter() - before
            io_before = observe_io(process)
            page_before = paging_observation()
            # Warm up on tuning rows, including for held-out rounds.
            for query in queries[:3]:
                search(query)
            order = np.random.default_rng(8429 + args.round).permutation(
                np.arange(args.offset, stop)
            )
            with (args.output / "queries.jsonl").open("x", encoding="utf-8") as raw:
                for qid in order.tolist():
                    if time.perf_counter() - started > args.seconds:
                        raise TimeoutError("condition exceeded its time budget")
                    if psutil.virtual_memory().available < 384 * MIB:
                        raise MemoryError("available system RAM below 384 MiB")
                    cpu = time.process_time()
                    before = time.perf_counter_ns()
                    hits, scanned = search(queries[qid])
                    wall_ms = (time.perf_counter_ns() - before) / 1e6
                    cpu_ms = (time.process_time() - cpu) * 1000
                    identifiers = {hit[0] for hit in hits}
                    if len(identifiers) != 10 or len(hits) != 10:
                        raise RuntimeError("missing or duplicate returned neighbor")
                    row = {
                        "query_id": qid,
                        "round_id": args.round,
                        "wall_ms": wall_ms,
                        "cpu_ms": cpu_ms,
                        "hits": hits,
                        "scanned": scanned,
                        "recall_at_10": len(identifiers & set(truth[qid].tolist()))
                        / 10,
                        "timing_gap": wall_ms > 60000 and wall_ms > 10 * max(cpu_ms, 1),
                    }
                    raw.write(json.dumps(row, allow_nan=False) + "\n")
                    raw.flush()
                    samples.append(row)
            io_after = observe_io(process)
            manifest["query_io_delta"] = (
                {k: io_after[k] - io_before[k] for k in io_before}
                if io_before and io_after
                else None
            )
            manifest["query_paging_start"] = page_before
            manifest["query_paging_end"] = paging_observation()
            if args.batch:
                sample = np.ascontiguousarray(queries[order[: min(32, len(order))]])
                start_batch = time.perf_counter()
                batch(sample)
                elapsed_batch = time.perf_counter() - start_batch
                manifest["batch"] = {
                    "queries": len(sample),
                    "seconds": elapsed_batch,
                    "qps": len(sample) / elapsed_batch,
                    "mode": "serial_request_stream"
                    if args.scan_threads is not None
                    else "native_batch",
                }
            memory = process_memory()
            if memory["peak_rss_bytes"] > policy["maximum_bytes"] + 65536:
                raise MemoryError(
                    "resident peak exceeded the cap plus 64 KiB accounting allowance"
                )
            manifest.update(status="completed", query_peak_memory=memory)
    except Exception as error:
        manifest.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(
            recorded_calls=len(samples),
            elapsed_seconds=time.perf_counter() - started,
            power_end=power_state(),
            memory=process_memory(),
        )
        write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "summary.json", summarize(samples))


def summarize(samples):
    times = [row["wall_ms"] for row in samples]
    scans = [row["scanned"] for row in samples if row["scanned"] is not None]
    return {
        "calls": len(samples),
        "distinct_queries": len({row["query_id"] for row in samples}),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "recall_at_10": float(np.mean([row["recall_at_10"] for row in samples])),
        "mean_scanned": float(np.mean(scans)) if scans else None,
        "timing_gaps": sum(row["timing_gap"] for row in samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["int8", "graph"])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--complexity", type=int, default=64)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--cache-nodes", type=int, default=0)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--scan-threads", type=int, choices=[0, 1, 2])
    parser.add_argument("--batch", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
