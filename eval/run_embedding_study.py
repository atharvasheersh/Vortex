"""Checkpointed local embedding study, development selection and held-out retrieval."""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vortex import Index
from vortex.embedding_models import StudyEncoder, formatted_inputs
from vortex.embedding_study import (
    bind_contract,
    check_exact,
    exact_units,
    load_batch,
    load_study,
    save_batch,
    search_units,
)
from vortex.evaluation import fingerprint, manifest, process_memory, write_json
from vortex.metrics import retrieval_metrics
from vortex.real_rag import append_record, digest_object, load_records, writer_lock

DOMAINS = ("squad", "scifact")
SPLITS = ("development", "test")
LOCAL = ("minilm", "qwen", "jina", "embeddinggemma")
EXACT = {"dtype": "fp32", "nlist": 0, "nprobe": 0, "rerank_multiplier": 0}


def registry():
    return json.loads((ROOT / "eval/embedding_models.json").read_text())


def bind(args):
    sources = [
        "eval/embedding_comparison_spec.md",
        "eval/embedding_models.json",
        "python/vortex/embedding_models.py",
        "python/vortex/embedding_study.py",
        "eval/run_embedding_study.py",
        "eval/prepare_embedding_study.py",
        "python/vortex/real_rag.py",
        "python/vortex/metrics.py",
    ]
    contract = {
        "sources": {p: fingerprint(ROOT / p) for p in sources},
        "data": {
            f"{d}-{s}": fingerprint(args.dataset / f"{d}-{s}.json")
            for d in DOMAINS
            for s in SPLITS
        },
        "api_budget_usd": 0,
    }
    bind_contract(args.run, contract)
    if not (args.run / "manifest.json").exists():
        packages = {}
        for p in [
            "torch",
            "transformers",
            "sentence-transformers",
            "peft",
            "numpy",
            "torchvision",
        ]:
            try:
                packages[p] = importlib.metadata.version(p)
            except importlib.metadata.PackageNotFoundError:
                packages[p] = None
        write_json(
            args.run / "manifest.json",
            {
                **manifest(contract),
                "packages": packages,
                "credential_available": {
                    k: bool(os.getenv(v["environment"]))
                    for k, v in registry().items()
                    if "environment" in v
                },
                "api_status": "pending_owner_deferred",
                "actual_api_charges_usd": 0,
                "thread_policy": {"encoder": 4, "retrieval": 1, "numpy_blas": 1},
                "background_load": "Shared Windows desktop; applications left running",
            },
        )


def embed(args):
    spec = registry()[args.model]
    if spec["backend"] != "local":
        raise ValueError(
            "hosted execution deferred by owner; this CLI permits local models only"
        )
    directory = args.run / args.model
    with writer_lock(args.run), writer_lock(directory):
        bind(args)
        started = time.perf_counter()
        encoder = StudyEncoder(args.model, spec)
        load_seconds = time.perf_counter() - started
        encoder.encode([{"query": "What causes rain?"}], "query", "squad")
        for domain in DOMAINS:
            for split in SPLITS:
                data = load_study(args.dataset / f"{domain}-{split}.json")
                output = directory / f"{domain}-{split}"
                output.mkdir(exist_ok=True)
                reusable = None
                if split == "test":
                    development = load_study(
                        args.dataset / f"{domain}-development.json"
                    )
                    if digest_object(development["chunks"]) == digest_object(
                        data["chunks"]
                    ):
                        reusable = directory / f"{domain}-development"
                corpus, seconds = [], []
                new_encoding_seconds, reused_batches = 0.0, 0
                for start in range(0, len(data["chunks"]), spec["batch_size"]):
                    batch = data["chunks"][start : start + spec["batch_size"]]
                    name = f"corpus-{start:06d}"
                    saved = load_batch(output, name, len(batch), spec["dimensions"])
                    if saved is None and reusable is not None:
                        origin = load_batch(
                            reusable, name, len(batch), spec["dimensions"]
                        )
                        if origin is not None:
                            metadata = {**origin[1], "reused_from_split": "development"}
                            save_batch(output, name, origin[0], metadata)
                            saved = origin[0], metadata
                    if saved is None:
                        before = time.perf_counter()
                        try:
                            values = encoder.encode(batch, "document", domain)
                        except Exception as error:
                            write_json(
                                output / f"{name}-failure.json",
                                {
                                    "batch_start": start,
                                    "status": "failed",
                                    "error_type": type(error).__name__,
                                },
                            )
                            raise
                        metadata = {
                            "seconds": time.perf_counter() - before,
                            "ids": [c["id"] for c in batch],
                        }
                        save_batch(output, name, values, metadata)
                        saved = values, metadata
                    if saved[1].get("reused_from_split"):
                        reused_batches += 1
                    else:
                        new_encoding_seconds += saved[1]["seconds"]
                    if saved[1]["ids"] != [c["id"] for c in batch]:
                        raise ValueError("batch stable-ID mapping differs")
                    corpus.append(saved[0])
                    seconds.append(saved[1]["seconds"])
                    if start % 400 == 0:
                        print(
                            f"{args.model} {domain}-{split} documents {start}/{len(data['chunks'])}",
                            flush=True,
                        )
                records = load_records(output / "query-timings.jsonl")
                completed = {(r["condition"], r["query_id"]): r for r in records}
                queries = [None] * len(data["cases"])
                latencies = []
                for repeat in range(3):
                    for position in np.random.default_rng(42 + repeat).permutation(
                        len(data["cases"])
                    ):
                        case = data["cases"][position]
                        name = f"query-{position:04d}-r{repeat}"
                        saved = load_batch(output, name, 1, spec["dimensions"])
                        if saved is None:
                            before = time.perf_counter()
                            values = encoder.encode([case], "query", domain)
                            metadata = {
                                "seconds": time.perf_counter() - before,
                                "query_id": case["query_id"],
                            }
                            save_batch(output, name, values, metadata)
                            saved = values, metadata
                        if saved[1]["query_id"] != case["query_id"]:
                            raise ValueError("query checkpoint order differs")
                        latencies.append(saved[1]["seconds"] * 1000)
                        key = (f"r{repeat}", case["query_id"])
                        if key not in completed:
                            append_record(
                                output / "query-timings.jsonl",
                                {
                                    "condition": key[0],
                                    "query_id": key[1],
                                    "milliseconds": saved[1]["seconds"] * 1000,
                                },
                            )
                        if repeat == 0:
                            queries[position] = saved[0][0]
                np.savez(
                    output / "embeddings.npz",
                    corpus=np.concatenate(corpus),
                    queries=np.array(queries),
                )
                write_json(
                    output / "embedding.json",
                    {
                        "status": "completed",
                        "corpus_items": len(data["chunks"]),
                        "queries": len(queries),
                        "query_timing_observations": len(latencies),
                        "corpus_seconds": sum(seconds),
                        "corpus_seconds_definition": "Sum of original batch encode times, including the origin time for reused batches",
                        "new_encoding_seconds": new_encoding_seconds,
                        "reused_document_batches": reused_batches,
                        "documents_per_second": len(data["chunks"]) / sum(seconds),
                        "query_p50_ms": float(np.median(latencies)),
                        "query_p95_ms": float(np.percentile(latencies, 95)),
                        "dimensions": spec["dimensions"],
                        "dtype": spec["dtype"],
                        "load_seconds": load_seconds,
                        "truncated_inputs": 0,
                        "embedding_sha256": fingerprint(output / "embeddings.npz"),
                        "process_memory": process_memory(),
                    },
                )
                print(f"embedded {args.model} {domain}-{split}", flush=True)
        write_json(
            directory / "encoder-complete.json",
            {
                "status": "completed",
                "process_memory": process_memory(),
                "load_seconds": load_seconds,
            },
        )


def snapshot(directory, data, vectors, config):
    path = directory / f"snapshot-{config['dtype']}-{config['nlist']}.vtx"
    if not path.exists():
        before = time.perf_counter()
        index = Index(vectors.shape[1], metric="cosine", threads=1)
        for chunk, vector in zip(data["chunks"], vectors):
            index.add(chunk["id"], vector)
        if config["nlist"]:
            index.build_ivf(nlist=config["nlist"], seed=42)
        index.set_dtype(config["dtype"])
        index.save(str(path))
        write_json(
            path.with_suffix(".json"),
            {
                "seconds": time.perf_counter() - before,
                "bytes": path.stat().st_size,
                "sha256": fingerprint(path),
                "process_memory": process_memory(),
            },
        )
    elif (
        fingerprint(path) != json.loads(path.with_suffix(".json").read_text())["sha256"]
    ):
        raise ValueError("snapshot hash mismatch")
    return path


def evaluate(args, model, domain, split, config, k, rounds):
    from threadpoolctl import threadpool_limits

    directory = args.run / model / f"{domain}-{split}"
    tag = digest_object({"config": config, "k": k, "rounds": rounds})[:16]
    result_path = directory / f"retrieval-{tag}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    data = load_study(args.dataset / f"{domain}-{split}.json")
    if (
        fingerprint(directory / "embeddings.npz")
        != json.loads((directory / "embedding.json").read_text())["embedding_sha256"]
    ):
        raise ValueError("embedding archive differs")
    with np.load(directory / "embeddings.npz") as arrays:
        corpus, queries = arrays["corpus"], arrays["queries"]
    path = snapshot(directory, data, corpus, config)
    index = Index.open(str(path), cache_bytes=8 * 1024 * 1024, threads=1)
    reference = corpus.astype(np.float64)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    checkpoint = directory / f"retrieval-{tag}.jsonl"
    records = load_records(checkpoint)
    completed = {(r["condition"], r["query_id"]): r for r in records}
    index.search(queries[0], k=min(k, len(corpus)), nprobe=config["nprobe"])
    with threadpool_limits(limits=1):
        for repeat in range(rounds):
            for position in np.random.default_rng(42 + repeat).permutation(
                len(queries)
            ):
                case, query = data["cases"][position], queries[position]
                key = (f"r{repeat}", case["query_id"])
                if key in completed:
                    continue
                expected, scores = exact_units(reference, query, data["chunks"], k)
                before = time.perf_counter()
                found = search_units(index, query, data["chunks"], k, config)
                milliseconds = (time.perf_counter() - before) * 1000
                if config == EXACT:
                    check_exact(found, expected, scores)
                row = {
                    "condition": key[0],
                    "query_id": case["query_id"],
                    "group_id": case["group_id"],
                    "answerable": case["answerable"],
                    "status": "ok",
                    **found,
                    "retrieval_ms": milliseconds,
                    "ann_neighbor_recall": len(set(found["ids"]) & set(expected))
                    / len(expected),
                    "metrics": retrieval_metrics(found["ids"], case["relevant_ids"], k)
                    if case["answerable"]
                    else None,
                }
                append_record(checkpoint, row)
                completed[key] = row
    rows = [completed[("r0", c["query_id"])] for c in data["cases"]]
    positive = [r for r in rows if r["answerable"]]
    times = [r["retrieval_ms"] for r in completed.values()]
    result = {
        "configuration": config,
        "k": k,
        "rounds": rounds,
        "rows": rows,
        "labeled_recall": float(np.mean([r["metrics"]["recall"] for r in positive])),
        "ndcg": float(np.mean([r["metrics"]["ndcg"] for r in positive])),
        "ann_neighbor_recall": float(np.mean([r["ann_neighbor_recall"] for r in rows])),
        "retrieval_p50_ms": float(np.median(times)),
        "retrieval_p95_ms": float(np.percentile(times, 95)),
        "timing_observations": len(times),
        "cache": index.cache_stats(),
        "snapshot": json.loads(path.with_suffix(".json").read_text()),
        "process_memory": process_memory(),
    }
    write_json(result_path, result)
    return result


def tune(args):
    with writer_lock(args.run):
        bind(args)
        selection_path = args.run / "selection.json"
        if selection_path.exists():
            raise FileExistsError("development selections already frozen")
        selections, candidates = {}, []
        for model in LOCAL:
            selections[model] = {}
            development_quality, query_p95 = [], []
            for domain in DOMAINS:
                exact = evaluate(args, model, domain, "development", EXACT, 10, 1)
                development_quality.append(exact["ndcg"])
                metadata = json.loads(
                    (
                        args.run / model / f"{domain}-development/embedding.json"
                    ).read_text()
                )
                query_p95.append(metadata["query_p95_ms"])
                grid = [
                    {
                        "dtype": dtype,
                        "nlist": lists,
                        "nprobe": probe,
                        "rerank_multiplier": rerank,
                    }
                    for lists in [16, 32]
                    for probe in [1, 4, 8, lists]
                    for dtype in ["fp32", "fp16", "int8"]
                    for rerank in ([0] if dtype == "fp32" else [0, 4])
                ]
                np.random.default_rng(42).shuffle(grid)
                results = [
                    evaluate(args, model, domain, "development", c, 10, 1) for c in grid
                ]
                qualified = [
                    r
                    for r in results
                    if r["ann_neighbor_recall"] >= 0.95
                    and r["labeled_recall"] >= exact["labeled_recall"] - 0.02
                ]
                chosen = (
                    min(qualified, key=lambda r: r["retrieval_p50_ms"])
                    if qualified
                    else exact
                )
                selections[model][domain] = {
                    "config": chosen["configuration"],
                    "qualified": len(qualified),
                    "evaluated": len(results),
                    "exact_development_ndcg10": exact["ndcg"],
                    "full_list_probing": chosen["configuration"]["nprobe"]
                    == chosen["configuration"]["nlist"]
                    != 0,
                }
                print(f"tuned {model} {domain}: {chosen['configuration']}", flush=True)
            peak = json.loads((args.run / model / "encoder-complete.json").read_text())[
                "process_memory"
            ]["peak_rss_bytes"]
            candidates.append(
                {
                    "model": model,
                    "development_ndcg10": float(np.mean(development_quality)),
                    "query_p95_ms": max(query_p95),
                    "encoder_peak_rss_bytes": peak,
                    "resource_eligible": peak <= 4 * 1024**3 and max(query_p95) <= 5000,
                }
            )
        challengers = sorted(
            [
                c
                for c in candidates
                if c["model"] != "minilm" and c["resource_eligible"]
            ],
            key=lambda c: (-c["development_ndcg10"], c["query_p95_ms"], c["model"]),
        )[:2]
        write_json(
            selection_path,
            {
                "selections": selections,
                "candidates": candidates,
                "rag_models": ["minilm"] + [c["model"] for c in challengers],
                "frozen_before_test": True,
            },
        )


def heldout(args):
    with writer_lock(args.run):
        bind(args)
        selection = json.loads((args.run / "selection.json").read_text())
        for model in LOCAL:
            for domain in DOMAINS:
                tasks = [(name, k) for name in ["exact", "selected"] for k in [5, 10]]
                np.random.default_rng(42).shuffle(tasks)
                for name, k in tasks:
                    config = (
                        EXACT
                        if name == "exact"
                        else selection["selections"][model][domain]["config"]
                    )
                    result = evaluate(args, model, domain, "test", config, k, 3)
                    write_json(
                        args.run / model / f"{domain}-test/{name}-k{k}.json", result
                    )
                print(f"heldout {model} {domain} complete", flush=True)


def forecast(args):
    costs = {}
    for key in ["voyage", "gemini"]:
        nbytes = 0
        for domain in DOMAINS:
            for split in SPLITS:
                data = load_study(args.dataset / f"{domain}-{split}.json")
                for role, items, repeats in [
                    ("document", data["chunks"], 1),
                    ("query", data["cases"], 3),
                ]:
                    nbytes += repeats * sum(
                        len(t.encode("utf-8")) + 128
                        for t in formatted_inputs(key, items, role, domain)
                    )
        price = 0.12 if key == "voyage" else 0.20
        costs[key] = {
            "rough_tokens_at_four_bytes": nbytes / 4,
            "conservative_token_reservation": nbytes,
            "list_price_usd_per_million": price,
            "rough_cost_usd": nbytes / 4 * price / 1e6,
            "two_attempt_reserved_cost_usd": 2 * nbytes * price / 1e6,
            "actual_charges_usd": 0,
            "status": "pending_owner_deferred",
            "provider_version": None,
        }
    write_json(args.run / "cost-forecast.json", costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["forecast", "embed", "tune", "heldout"])
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/embedding-study/prepared")
    )
    parser.add_argument(
        "--run", type=Path, default=Path("results/embedding-study-20261109-v3")
    )
    parser.add_argument("--model", choices=list(registry()), default="minilm")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    {"forecast": forecast, "embed": embed, "tune": tune, "heldout": heldout}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
