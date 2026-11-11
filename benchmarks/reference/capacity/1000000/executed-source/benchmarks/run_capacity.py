"""Measure one full mapped collection using bounded construction/reference buffers."""

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from capacity_resources import working_set_limit
from run_scaling import available_memory, memory_stage, stats
from streaming_snapshot import blocks, build_snapshot, file_sha256, layout, vectors_at
from vortex import Index
from vortex.evaluation import fingerprint, manifest, write_json
from vortex.real_rag import append_record, writer_lock

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-4


def power_state():
    if sys.platform != "win32":
        return {"available": False}
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [
            (k, ctypes.c_ubyte) for k in ["ac", "battery", "percent", "saver"]
        ] + [(k, ctypes.c_uint32) for k in ["lifetime", "full_lifetime"]]

    value = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        return {"available": False}
    return {
        "available": True,
        "ac_line_status": int(value.ac),
        "battery_saver": int(value.saver),
    }


def check_resources():
    free = available_memory()
    if free is not None and free < 384 * 1024**2:
        raise MemoryError(
            "available physical memory below 384 MiB; preserving partial evidence"
        )


def references(path, count, dimension, queries, k, chunk_rows):
    """Blocked exact FAISS for every query, plus direct float64 NumPy for three."""
    import faiss
    from threadpoolctl import threadpool_limits

    faiss.omp_set_num_threads(1)
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_scores = np.empty((len(queries), 0), dtype=np.float32)
    sample = sorted({0, len(queries) // 2, len(queries) - 1})
    numpy_best = {
        q: (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)) for q in sample
    }
    with threadpool_limits(limits=1):
        for start, values in blocks(path, count, dimension, chunk_rows):
            check_resources()
            distances, identifiers = faiss.knn(queries, values, min(k, len(values)))
            identifiers += start
            candidate_ids = np.concatenate([best_ids, identifiers], axis=1)
            candidate_scores = np.concatenate([best_scores, distances], axis=1)
            order = np.lexsort((candidate_ids, candidate_scores), axis=1)[:, :k]
            best_ids = np.take_along_axis(candidate_ids, order, axis=1)
            best_scores = np.take_along_axis(candidate_scores, order, axis=1)
            precise = values.astype(np.float64)
            for query_id in sample:
                difference = precise - queries[query_id].astype(np.float64)
                scores = np.einsum("ij,ij->i", difference, difference)
                selected = np.argpartition(scores, min(k, len(scores)) - 1)[:k]
                old_ids, old_scores = numpy_best[query_id]
                merged_ids = np.concatenate([old_ids, start + selected])
                merged_scores = np.concatenate([old_scores, scores[selected]])
                positions = np.lexsort((merged_ids, merged_scores))[:k]
                numpy_best[query_id] = merged_ids[positions], merged_scores[positions]
            if start // 1000000 != (start + len(values)) // 1000000:
                print(
                    f"independent references {start + len(values):,}/{count:,}",
                    flush=True,
                )
    checks = {}
    for query_id, (ids, scores) in numpy_best.items():
        if not np.allclose(best_scores[query_id], scores, atol=TOLERANCE, rtol=1e-5):
            raise AssertionError("FAISS and direct float64 distances disagree")
        checks[str(query_id)] = {"ids": ids.tolist(), "scores": scores.tolist()}
    return {
        "ids": best_ids.tolist(),
        "scores": best_scores.tolist(),
        "numpy_float64": checks,
        "faiss_version": faiss.__version__,
        "faiss_threads": 1,
        "absolute_tie_tolerance": TOLERANCE,
    }


def check_exact(path, count, dimension, query, found, expected_ids, expected_scores):
    actual_ids = [int(h[0]) for h in found["hits"]]
    actual_scores = [float(h[1]) for h in found["hits"]]
    if len(actual_ids) != len(expected_ids) or len(set(actual_ids)) != len(actual_ids):
        raise AssertionError("incorrect number of exact neighbors")
    if not np.allclose(actual_scores, expected_scores, atol=TOLERANCE, rtol=1e-5):
        raise AssertionError("native exact scores disagree with independent reference")
    if actual_ids != expected_ids:
        actual = vectors_at(path, count, dimension, actual_ids).astype(np.float64)
        expected = vectors_at(path, count, dimension, expected_ids).astype(np.float64)
        actual_dist = np.sum((actual - query.astype(np.float64)) ** 2, axis=1)
        threshold = np.max(np.sum((expected - query.astype(np.float64)) ** 2, axis=1))
        if np.any(actual_dist > threshold + TOLERANCE):
            raise AssertionError("native exact IDs disagree outside boundary ties")
    return actual_ids != expected_ids


def measure(index, queries, truth, probe, args, raw_path, snapshot, reference=None):
    for query in queries[:5]:
        index.search(query, k=args.k, nprobe=probe)
    rounds, actual, scans, all_times, tie_differences = [], {}, [], [], 0
    for repeat in range(args.repeats):
        times = []
        for query_id in np.random.default_rng(1700 + repeat).permutation(len(queries)):
            check_resources()
            power_before = power_state()
            cpu_before, before = time.process_time(), time.perf_counter_ns()
            row = index.search(queries[query_id], k=args.k, nprobe=probe)
            elapsed = (time.perf_counter_ns() - before) / 1e6
            cpu = time.process_time() - cpu_before
            ids = [int(h[0]) for h in row["hits"]]
            reference_tie = False
            if query_id in actual and actual[query_id] != ids:
                raise AssertionError("native output changed across repeats")
            actual[int(query_id)] = ids
            if probe == 0:
                if row["scanned"] != args.size:
                    raise AssertionError(
                        "exhaustive query did not scan the entire collection"
                    )
                reference_tie = check_exact(
                    snapshot,
                    args.size,
                    args.dimension,
                    queries[query_id],
                    row,
                    reference["ids"][query_id],
                    reference["scores"][query_id],
                )
                tie_differences += reference_tie
                precise = reference["numpy_float64"].get(str(query_id))
                if precise:
                    check_exact(
                        snapshot,
                        args.size,
                        args.dimension,
                        queries[query_id],
                        row,
                        precise["ids"],
                        precise["scores"],
                    )
            elif probe == args.nlist and ids != truth[int(query_id)]:
                raise AssertionError("full-probe IVF differs from exhaustive search")
            append_record(
                raw_path,
                {
                    "condition": f"probe{probe}:r{repeat}",
                    "query_id": int(query_id),
                    "mode": "mapped_exact" if probe == 0 else "mapped_ivf",
                    "nprobe": probe,
                    "repeat": repeat,
                    "wall_ms": elapsed,
                    "native_ms": row["milliseconds"],
                    "process_cpu_seconds": cpu,
                    "scanned": row["scanned"],
                    "ids": ids,
                    "scores": [float(h[1]) for h in row["hits"]],
                    "power_before": power_before,
                    "power_after": power_state(),
                    "reference_boundary_tie": reference_tie,
                },
            )
            times.append(elapsed)
            scans.append(row["scanned"])
            if len(times) % 5 == 0:
                print(
                    f"probe={probe} round={repeat + 1}/{args.repeats} queries={len(times)}/{len(queries)}",
                    flush=True,
                )
        rounds.append(stats(times))
        all_times.extend(times)
    ordered = [actual[q] for q in range(len(queries))]
    recall = (
        1.0
        if truth is None
        else float(
            np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ordered, truth)])
        )
    )
    return {
        **stats(all_times),
        "nprobe": probe,
        "rounds": rounds,
        "median_round_p95_ms": float(np.median([r["p95_ms"] for r in rounds])),
        "query_count": len(queries),
        "repeats": args.repeats,
        "recall_at_k": recall,
        "mean_scanned": float(np.mean(scans)),
        "reference_boundary_tie_differences": tie_differences,
    }, ordered


def run(args):
    with working_set_limit(getattr(args, "working_set_mib", 0)) as policy:
        return run_with_policy(args, policy)


def run_with_policy(args, policy):
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "use an empty output directory; previous measurements are preserved"
        )
    root.mkdir(parents=True, exist_ok=True)
    import _vortex

    sources = [
        "benchmarks/run_capacity.py",
        "benchmarks/streaming_snapshot.py",
        "benchmarks/run_scaling.py",
        "benchmarks/capacity_10m_spec.md",
        "benchmarks/validate_capacity.py",
        "benchmarks/validate_scaling.py",
        "benchmarks/capacity_resources.py",
        "benchmarks/capacity_10m_bounded_spec.md",
    ]
    state = {
        "schema_version": 1,
        "status": "running",
        "size": args.size,
        "dimension": args.dimension,
        "k": args.k,
        "nlist": args.nlist,
        "query_count": args.queries,
        "repeats": args.repeats,
        "chunk_rows": args.chunk_rows,
        "cache_budget": 8 * 1024**2,
        "memory_policy": policy,
        "power_at_start": power_state(),
        "quality_target": 0.95,
        "sources_sha256": {p: fingerprint(ROOT / p) for p in sources},
        "native_module_sha256": fingerprint(_vortex.__file__),
        "native_module_version": _vortex.__version__,
        "numpy_version": np.__version__,
        "boundary": "one full mapped FP32 L2 snapshot; streamed construction, not mutable-builder capacity",
        "timing_boundary": "native mapped query wall time including bindings; validation/checkpoint IO excluded; OS cache not flushed",
    }
    started = time.perf_counter()
    with writer_lock(root):
        write_json(root / "result.json", state)
        archived = root / "executed-source"
        for name in sources:
            target = archived / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        try:
            needed = (
                layout(args.size, args.dimension, args.nlist)[-1]
                + args.size * 4
                + 512 * 1024**2
            )
            if shutil.disk_usage(root).free < needed:
                raise MemoryError(
                    "insufficient disk headroom for the snapshot and auxiliary files"
                )
            needed_ram = max(
                1024**3,
                args.size * 128
                + args.chunk_rows * args.dimension * 4 * 6
                + min(args.size, 10000) * args.dimension * 4 * 3
                + args.queries * args.chunk_rows * 8
                + 512 * 1024**2,
            )
            state["estimated_resident_headroom_bytes"] = needed_ram
            free = available_memory()
            if free is not None and free < needed_ram:
                raise MemoryError(
                    "insufficient available physical RAM for bounded buffers and native metadata"
                )
            memory_stage(state, "start")
            construction = build_snapshot(
                root,
                args.size,
                args.dimension,
                args.nlist,
                args.chunk_rows,
                check_resources=check_resources,
            )
            state["construction"] = construction
            snapshot = Path(construction["snapshot"])
            state["snapshot_sha256"] = file_sha256(snapshot)
            memory_stage(state, "snapshot_built")
            write_json(root / "result.json", state)
            queries = np.random.default_rng(2026).standard_normal(
                (args.queries, args.dimension), dtype=np.float32
            )
            np.save(root / "queries.npy", queries, allow_pickle=False)
            state["queries_sha256"] = fingerprint(root / "queries.npy")
            before = time.perf_counter()
            reference = references(
                snapshot, args.size, args.dimension, queries, args.k, args.chunk_rows
            )
            write_json(root / "reference.json", reference)
            state["reference_sha256"] = fingerprint(root / "reference.json")
            state["reference_seconds"] = time.perf_counter() - before
            memory_stage(state, "references_complete")
            print(
                "Full-corpus independent references complete; opening the native collection",
                flush=True,
            )
            before = time.perf_counter()
            index = Index.open(
                str(snapshot), cache_bytes=state["cache_budget"], threads=4
            )
            state["open_seconds"] = time.perf_counter() - before
            state["opened_vector_count"] = len(index)
            if len(index) != args.size:
                raise AssertionError("native index did not open the entire corpus")
            memory_stage(state, "native_index_open")
            state["self_lookup_ids"] = []
            for identifier in sorted({0, args.size // 2, args.size - 1}):
                query = vectors_at(snapshot, args.size, args.dimension, [identifier])[0]
                row = index.search(query, k=1)
                if (
                    row["hits"][0][0] != identifier
                    or row["hits"][0][1] != 0
                    or row["scanned"] != args.size
                ):
                    raise AssertionError(
                        "full-corpus first/middle/last self lookup failed"
                    )
                state["self_lookup_ids"].append(identifier)
                print(
                    f"self lookup {identifier:,} passed; scanned {row['scanned']:,}",
                    flush=True,
                )
            state["exact"], truth = measure(
                index,
                queries,
                None,
                0,
                args,
                root / "measurements.jsonl",
                snapshot,
                reference,
            )
            write_json(root / "result.json", state)
            state["ivf"] = []
            for probe in sorted(
                {p for p in [1, 4, 8, 16, 24, 28, 32, args.nlist] if p <= args.nlist}
            ):
                print(f"measuring mapped IVF probe {probe}", flush=True)
                value, _ = measure(
                    index,
                    queries,
                    truth,
                    probe,
                    args,
                    root / "measurements.jsonl",
                    snapshot,
                )
                state["ivf"].append(value)
                memory_stage(state, f"probe_{probe}")
                write_json(root / "result.json", state)
            qualified = [
                v for v in state["ivf"] if v["recall_at_k"] >= state["quality_target"]
            ]
            state["selected_ivf"] = min(qualified, key=lambda v: v["nprobe"])
            state["selection_rule"] = (
                "smallest predeclared probe with recall >=0.95 on measured queries; not held-out tuning"
            )
            state["cache_stats"] = index.cache_stats()
            if state["cache_stats"]["resident_bytes"] > state["cache_budget"]:
                raise AssertionError("application cache exceeds payload budget")
            state["status"] = "passed"
        except Exception as error:  # noqa: BLE001 -- retain evidence for every failed stage
            state.update(
                status="resource_limited"
                if isinstance(error, MemoryError)
                else "failed",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            print(state["traceback"], flush=True)
        finally:
            state["elapsed_seconds"] = time.perf_counter() - started
            state["power_at_end"] = power_state()
            state["manifest"] = manifest(vars(args))
            memory_stage(state, "final")
            raw = root / "measurements.jsonl"
            state["measurements_sha256"] = fingerprint(raw) if raw.exists() else None
            write_json(root / "result.json", state)
    print(f"Capacity probe {args.size:,}: {state['status']}", flush=True)
    return 0 if state["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10000000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=32)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--working-set-mib",
        type=int,
        default=0,
        help="Windows-only hard resident working-set maximum for this process; 0 leaves OS management unchanged",
    )
    parser.add_argument("--output", default="results/capacity-10m-20261111")
    args = parser.parse_args()
    if (
        args.dimension < 1
        or args.dimension > 1048576
        or not 1 <= args.nlist <= 10000
        or args.queries < 3
        or args.repeats < 1
        or args.k < 1
        or args.chunk_rows < args.k
        or args.size < max(args.k, args.nlist)
        or (args.working_set_mib != 0 and args.working_set_mib < 256)
    ):
        parser.error(
            "require valid dimensions, 1..10000 lists, >=3 queries, positive repeats/k and size/chunk >=k"
        )
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    sys.exit(main())
