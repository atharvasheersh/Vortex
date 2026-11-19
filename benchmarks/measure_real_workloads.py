"""Measure API lookup, queue-inclusive concurrency and separate batch work."""

import argparse
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, BoundedSemaphore, Lock

import _vortex
import faiss
import numpy as np
import psutil
from capacity_resources import working_set_limit
from real_workload_data import guard, read_json
from run_capacity import power_state
from run_performance import digest, paging_observation, write_json
from vortex import Index
from vortex.evaluation import process_memory

MODES = ("cached", "direct1", "direct2", "fp32", "faiss")
SCAN_THREADS = {"cached": 0, "direct1": 1, "direct2": 2, "fp32": 0}
THREADS = 2
K = 10
RERANK = 100


class Retriever:
    def __init__(self, data, mode, probes):
        self.mode, self.probes = mode, probes
        if mode == "faiss":
            self.index = faiss.read_index(str(data / "faiss.ivfpq"))
            self.index.nprobe = probes
            self.corpus = np.load(
                data / "corpus.npy", mmap_mode="r", allow_pickle=False
            )
            self.ids = np.load(
                data / "public_ids.npy", mmap_mode="r", allow_pickle=False
            )
        else:
            name = "packed-fp32.vtx" if mode == "fp32" else "packed-int8.vtx"
            self.index = Index.open(str(data / name), threads=THREADS)

    def _rerank(self, query, rows, k):
        rows = rows[rows >= 0]
        ids = self.ids[rows]
        delta = self.corpus[rows] - query
        scores = np.einsum("ij,ij->i", delta, delta)
        order = np.lexsort((ids, scores))[:k]
        return list(zip(ids[order].tolist(), scores[order].tolist()))

    def lookup(self, query, k=K):
        if self.mode == "faiss":
            _, rows = self.index.search(query[None], max(k, RERANK))
            return self._rerank(query, rows[0], k), None
        result = self.index.search(
            query,
            k=k,
            nprobe=self.probes,
            rerank_n=0 if self.mode == "fp32" else max(k, RERANK),
            scan_threads=SCAN_THREADS[self.mode],
        )
        return result["hits"], result["scanned"]

    def batch(self, queries):
        if self.mode in {"cached", "fp32"}:
            return self.index.search_batch(
                queries,
                k=K,
                nprobe=self.probes,
                rerank_n=0 if self.mode == "fp32" else RERANK,
            )
        if self.mode == "faiss":
            _, rows = self.index.search(queries, RERANK)
            return [self._rerank(q, c, K) for q, c in zip(queries, rows)]
        raise ValueError("direct worker modes do not expose native batch search")


def document_scores(top_docs, qrels):
    gains = [qrels.get(doc, 0) for doc in top_docs]
    ideal = sorted(qrels.values(), reverse=True)[:10]

    def dcg(values):
        return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(values))

    best_dcg = dcg(ideal)
    return {
        "ndcg_at_10": dcg(gains) / best_dcg if best_dcg else 0.0,
        "document_recall_at_10": sum(v > 0 for v in gains)
        / sum(v > 0 for v in qrels.values()),
    }


def summarize(rows, duration, cpuseconds):
    if not rows or duration <= 0:
        raise ValueError("empty measurement")
    wall = [r["wall_ms"] for r in rows]
    scan = [r["scanned"] for r in rows if r["scanned"] is not None]
    return {
        "calls": len(rows),
        "p50_ms": float(np.percentile(wall, 50)),
        "p95_ms": float(np.percentile(wall, 95)),
        "mean_ms": float(np.mean(wall)),
        "p95_service_ms": float(np.percentile([r["service_ms"] for r in rows], 95)),
        "qps": len(rows) / duration,
        "cpu_ms_per_query": 1000 * cpuseconds / len(rows),
        "recall_at_10": float(np.mean([r["recall_at_10"] for r in rows])),
        "mean_scanned": float(np.mean(scan)) if scan else None,
        "timing_gaps": sum(r["timing_gap"] for r in rows),
    }


def io_counters():
    try:
        return psutil.Process().io_counters()._asdict()
    except (AttributeError, psutil.Error):
        return None


def collect(args, lookup, queries, truth, selected, raw, started):
    order = np.random.default_rng(6124 + args.round).permutation(
        np.arange(args.offset, args.offset + args.count)
    )
    slots = THREADS // (2 if args.mode == "direct2" else 1)
    admission = BoundedSemaphore(slots)
    log_lock = Lock()
    rows = []

    def query_loop(worker, ready):
        if args.mode == "faiss":
            faiss.omp_set_num_threads(1)
        if ready:
            ready.wait()
        for number, row in enumerate(order[worker :: args.clients]):
            if time.perf_counter() - started > args.seconds:
                raise TimeoutError("timed query condition exceeded its budget")
            if number % 32 == 0:
                guard()
            submitted = time.perf_counter_ns()
            before_cpu = time.process_time_ns() if args.clients == 1 else None
            with admission:
                service_started = time.perf_counter_ns()
                hits, scanned = lookup.lookup(queries[row])
                completed = time.perf_counter_ns()
            cpu_ms = (
                (time.process_time_ns() - before_cpu) / 1e6
                if before_cpu is not None
                else None
            )
            found = {hit[0] for hit in hits}
            if len(found) != K or len(hits) != K:
                raise ValueError("retriever failed to return ten unique neighbors")
            wall_ms = (completed - submitted) / 1e6
            item = {
                "position": int(row),
                "query_id": int(selected[row]),
                "started_ns": submitted,
                "completed_ns": completed,
                "wall_ms": wall_ms,
                "service_ms": (completed - service_started) / 1e6,
                "cpu_ms": cpu_ms,
                "scanned": scanned,
                "hits": hits,
                "recall_at_10": len(found & set(truth[row].tolist())) / K,
                "timing_gap": wall_ms > 60000
                and cpu_ms is not None
                and wall_ms > 10 * max(cpu_ms, 1),
            }
            with log_lock:
                raw.write(json.dumps(item, allow_nan=False) + "\n")
                raw.flush()
                rows.append(item)

    before_cpu = time.process_time()
    if args.clients == 1:
        query_loop(0, None)
    else:
        ready = Barrier(args.clients + 1)
        with ThreadPoolExecutor(max_workers=args.clients) as pool:
            futures = [
                pool.submit(query_loop, worker, ready) for worker in range(args.clients)
            ]
            ready.wait()
            for job in futures:
                job.result()
    elapsed = (
        max(row["completed_ns"] for row in rows)
        - min(row["started_ns"] for row in rows)
    ) / 1e9
    return rows, elapsed, time.process_time() - before_cpu, slots


def run(args):
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    build_path = args.dataset / "build.json"
    build = read_json(build_path)
    if build["status"] != "completed":
        raise ValueError("dataset preparation is incomplete")
    queries = np.load(args.dataset / "queries.npy", allow_pickle=False)
    truth = np.load(args.dataset / "reference_ids.npy", allow_pickle=False)
    selected = np.load(args.dataset / "query_ids.npy", allow_pickle=False)
    if args.offset < 0 or args.count < 1 or args.offset + args.count > len(queries):
        raise ValueError("invalid query range")
    if not 1 <= args.probes <= build["nlist"] or not 1 <= args.clients <= 8:
        raise ValueError("invalid probes or client count")
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = build["artifacts"]
    manifest = {
        "status": "started",
        "mode": args.mode,
        "probes": args.probes,
        "count": args.count,
        "offset": args.offset,
        "round": args.round,
        "clients": args.clients,
        "slots": THREADS,
        "timeout_seconds": args.seconds,
        "dataset": build["dataset"],
        "dataset_build_sha256": digest(build_path),
        "native_sha256": digest(_vortex.__file__),
        "active_kernel": _vortex.active_kernel(),
        "python": sys.version,
        "os": platform.platform(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "phase": args.phase,
        "source_sha256": digest(Path(__file__)),
        "power_start": power_state(),
    }
    index_files = (
        ["faiss.ivfpq", "public_ids.npy", "corpus.npy"]
        if args.mode == "faiss"
        else ["packed-fp32.vtx" if args.mode == "fp32" else "packed-int8.vtx"]
    )
    manifest["index_bytes"] = sum(artifacts[f]["bytes"] for f in index_files)
    write_json(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        with working_set_limit(args.working_set) as policy:
            manifest["memory_policy"] = policy
            faiss.omp_set_num_threads(1)
            guard()
            before = time.perf_counter_ns()
            lookup = Retriever(args.dataset, args.mode, args.probes)
            manifest["open_ms"] = (time.perf_counter_ns() - before) / 1e6
            before = time.perf_counter_ns()
            lookup.lookup(queries[0])
            manifest["first_lookup_ms"] = (time.perf_counter_ns() - before) / 1e6
            for query in queries[1:8]:
                lookup.lookup(query)
            manifest["io_start"] = io_counters()
            manifest["paging_start"] = paging_observation()
            with (args.output / "queries.jsonl").open("x", encoding="utf-8") as raw:
                rows, duration, cpuseconds, admitted = collect(
                    args, lookup, queries, truth, selected, raw, started
                )
            memory = process_memory()
            manifest.update(
                query_duration=duration,
                query_cpu_seconds=cpuseconds,
                admitted_requests=admitted,
                io_end=io_counters(),
                paging_end=paging_observation(),
                query_peak_memory=memory,
            )
            if (
                policy["enabled"]
                and memory["peak_rss_bytes"] > policy["maximum_bytes"] + 65536
            ):
                raise MemoryError(
                    "query peak exceeded the working-set cap and allowance"
                )
            if args.batch:
                if args.mode in {"direct1", "direct2"}:
                    manifest["batch"] = {"status": "unsupported"}
                else:
                    q_batch = np.ascontiguousarray(
                        queries[args.offset : args.offset + min(64, args.count)]
                    )
                    if args.mode == "faiss":
                        faiss.omp_set_num_threads(THREADS)
                    begin = time.perf_counter()
                    lookup.batch(q_batch)
                    elapsed = time.perf_counter() - begin
                    manifest["batch"] = {
                        "status": "completed",
                        "queries": len(q_batch),
                        "seconds": elapsed,
                        "qps": len(q_batch) / elapsed,
                    }
            if args.relevance:
                labels = read_json(args.dataset / "relevance.json")
                counts = {}
                for doc in labels["documents"].values():
                    counts[doc] = counts.get(doc, 0) + 1
                depth = min(build["count"], 9 * max(counts.values()) + 1)
                qids = labels["tuning_ids"] + labels["test_ids"]
                with (args.output / "relevance.jsonl").open(
                    "x", encoding="utf-8"
                ) as raw:
                    for row in range(args.offset, args.offset + args.count):
                        qid = qids[int(selected[row])]
                        hits, _ = lookup.lookup(queries[row], k=depth)
                        documents = list(
                            dict.fromkeys(
                                labels["documents"][str(hit[0])] for hit in hits
                            )
                        )[:K]
                        metrics = document_scores(
                            documents, labels["official_qrels"][qid]
                        )
                        raw.write(
                            json.dumps(
                                {"query_id": qid, "document_ids": documents, **metrics}
                            )
                            + "\n"
                        )
                manifest["relevance_chunk_depth"] = depth
            manifest["status"] = "completed"
            write_json(
                args.output / "summary.json", summarize(rows, duration, cpuseconds)
            )
    except Exception as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(
            seconds=time.perf_counter() - started,
            memory=process_memory(),
            power_end=power_state(),
        )
        write_json(args.output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--probes", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--working-set", type=int, default=1536)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument(
        "--phase",
        choices=["pilot", "tuning", "primary", "tier", "load", "os-managed"],
        required=True,
    )
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--relevance", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
