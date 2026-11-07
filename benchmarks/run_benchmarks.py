"""Controlled synthetic sweeps with exact neighbors as the ANN reference."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from vortex import Index, avx2_available
from vortex.evaluation import manifest, write_json


def neighbors(rows):
    return [[hit[0] for hit in row["hits"]] for row in rows]


def measure(index, queries, truth, k, nprobe=0, rerank_n=0, reset_each=False):
    if not reset_each:
        for query in queries[:5]: index.search(query, k=k, nprobe=nprobe, rerank_n=rerank_n)
    times, outputs, scans = [], [], []
    initial = index.cache_stats()
    cumulative = {key:0 for key in initial if key != "resident_bytes"}
    for query in queries:
        if reset_each: index.clear_cache()
        start = time.perf_counter()
        row = index.search(query, k=k, nprobe=nprobe, rerank_n=rerank_n)
        times.append((time.perf_counter()-start)*1000)
        outputs.append([hit[0] for hit in row["hits"]])
        scans.append(row["scanned"])
        if reset_each:
            current = index.cache_stats()
            for key in cumulative: cumulative[key] += current[key]
    elapsed = sum(times) / 1000
    recall = np.mean([len(set(actual) & set(expected))/len(expected) for actual,expected in zip(outputs,truth)])
    stats = index.cache_stats()
    if reset_each:
        stats.update(cumulative)
    else:
        for key in cumulative: stats[key] -= initial[key]
    return {"ann_recall": float(recall), "p50_ms": float(np.percentile(times,50)),
            "p95_ms": float(np.percentile(times,95)), "p99_ms": float(np.percentile(times,99)),
            "sequential_qps": len(queries)/elapsed, "mean_scanned": float(np.mean(scans)),
            **stats}


def run(args):
    if args.vectors < args.nlist or args.queries < 2 or args.dimension < 1 or not 1 <= args.k <= args.vectors:
        raise ValueError("invalid benchmark shape or index parameters")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.vectors,args.dimension)).astype(np.float32)
    queries = rng.normal(size=(args.queries,args.dimension)).astype(np.float32)
    digest = hashlib.sha256(vectors.tobytes()+queries.tobytes()).hexdigest()
    builder = Index(args.dimension, threads=1)
    for identifier, vector in enumerate(vectors): builder.add(identifier,vector)
    truth = neighbors(builder.search_batch(queries,k=args.k))
    started = time.perf_counter()
    builder.build_ivf(args.nlist,iterations=8,seed=args.seed)
    build_seconds = time.perf_counter()-started
    records = []
    def record(index, factor, name, nprobe=0, **kwargs):
        value = measure(index,queries,truth,args.k,nprobe,**kwargs)
        value.update(factor=factor,setting=name,nprobe=nprobe,vectors=args.vectors,dimension=args.dimension,k=args.k)
        records.append(value)
        print(f"{factor:15s} {str(name):16s} recall={value['ann_recall']:.4f} p95={value['p95_ms']:.3f}ms",flush=True)
    for kernel in ("scalar","auto"):
        index = Index(args.dimension,kernel=kernel)
        for identifier,vector in enumerate(vectors): index.add(identifier,vector)
        record(index,"kernel",kernel)
    for probe in sorted(set([1,2,4,8,args.nlist])):
        if probe <= args.nlist: record(builder,"nprobe",str(probe),probe)
    for dtype in ("fp32","fp16","int8"):
        builder.set_dtype(dtype)
        path = output/f"{dtype}.vtx"
        start = time.perf_counter()
        builder.save(str(path))
        saved = time.perf_counter()-start
        start = time.perf_counter()
        mapped = Index.open(str(path),cache_bytes=8192)
        loaded = time.perf_counter()-start
        record(mapped,"representation",dtype,args.nlist)
        records[-1].update(snapshot_bytes=path.stat().st_size,bytes_per_vector=path.stat().st_size/args.vectors,
                           save_seconds=saved,load_seconds=loaded,retains_fp32_originals=True)
        if dtype != "fp32":
            record(mapped,"rerank",dtype,args.nlist,rerank_n=min(args.vectors,max(args.k,100)))
        del mapped
    path = output/"fp32.vtx"
    for budget in (0,4096,65536,4*1024*1024):
        mapped = Index.open(str(path),cache_bytes=budget)
        record(mapped,"cache_budget",str(budget),min(4,args.nlist))
        records[-1]["cache_budget"] = budget
        del mapped
    mapped = Index.open(str(path),cache_bytes=4*1024*1024)
    record(mapped,"cache_state","application_cold",min(4,args.nlist),reset_each=True)
    record(mapped,"cache_state","warm",min(4,args.nlist))
    del mapped
    for threads in (1,2,4):
        index = Index(args.dimension,threads=threads)
        for identifier,vector in enumerate(vectors): index.add(identifier,vector)
        index.search_batch(queries[:5],k=args.k)
        start = time.perf_counter()
        rows = index.search_batch(queries,k=args.k)
        elapsed = time.perf_counter()-start
        assert neighbors(rows) == truth
        records.append({"factor":"threads","setting":str(threads),"threads":threads,
                        "batch_qps":len(queries)/elapsed,"batch_seconds":elapsed,"ann_recall":1.0})
    for size in sorted(set([max(args.nlist,args.vectors//4), max(args.nlist,args.vectors//2),args.vectors])):
        index = Index(args.dimension)
        for identifier,vector in enumerate(vectors[:size]): index.add(identifier,vector)
        start = time.perf_counter()
        index.build_ivf(args.nlist,iterations=8,seed=args.seed)
        training = time.perf_counter()-start
        local_truth = neighbors(index.search_batch(queries,k=args.k))
        value = measure(index,queries,local_truth,args.k)
        value.update(factor="corpus_size",setting=str(size),vectors=size,build_seconds=training)
        records.append(value)
    faiss_results = []
    if not args.skip_faiss:
        import faiss
        faiss.omp_set_num_threads(1)
        for family in ("flat","ivf"):
            start = time.perf_counter()
            if family == "flat": baseline = faiss.IndexFlatL2(args.dimension)
            else:
                baseline = faiss.IndexIVFFlat(faiss.IndexFlatL2(args.dimension),args.dimension,args.nlist)
                baseline.cp.seed = args.seed
                baseline.cp.niter = 8
                baseline.train(vectors)
            baseline.add(vectors)
            training = time.perf_counter()-start
            for probe in ([0] if family == "flat" else sorted(set([1,2,4,8,args.nlist]))):
                if probe > args.nlist: continue
                if family == "ivf": baseline.nprobe = probe
                for query in queries[:5]: baseline.search(query[None,:],args.k)
                times,recalls = [],[]
                for query,expected in zip(queries,truth):
                    start = time.perf_counter()
                    _, ids = baseline.search(query[None,:],args.k)
                    times.append((time.perf_counter()-start)*1000)
                    recalls.append(len(set(ids[0].tolist()) & set(expected))/len(expected))
                serialized = faiss.serialize_index(baseline)
                faiss_results.append({"family":family,"nprobe":probe,"ann_recall":float(np.mean(recalls)),
                    "p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),
                    "p99_ms":float(np.percentile(times,99)),"sequential_qps":len(queries)/(sum(times)/1000),
                    "build_seconds":training,"serialized_bytes":len(serialized),"threads":1})
    config = vars(args).copy()
    config.update(dataset_sha256=digest,distribution="numpy default_rng normal(0,1)",
                  build_seconds=build_seconds,avx2_available=avx2_available(),
                  cache_semantics="application cache only; OS page cache is uncontrolled",
                  latency_semantics="sequential Python wall time including binding overhead",
                  memory_semantics="resident_bytes counts cached payload only, excludes index metadata and OS pages",
                  original_matrix_bytes=int(vectors.nbytes),logical_cpu_count=os.cpu_count())
    config["build_description"] = os.environ.get("VORTEX_BUILD_DESCRIPTION", "not supplied; record compiler and CMake configuration externally")
    try:
        config["compiler"] = subprocess.check_output(["c++","--version"],text=True,timeout=10).splitlines()[0]
    except (OSError,subprocess.SubprocessError): config["compiler"] = None
    write_json(output/"manifest.json",manifest(config))
    write_json(output/"measurements.json",records)
    write_json(output/"faiss.json",faiss_results)
    columns = sorted({key for row in records for key in row})
    with (output/"measurements.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader(); writer.writerows(records)
    print(f"Saved {len(records)} engine measurements and {len(faiss_results)} baseline measurements.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors",type=int,default=10000)
    p.add_argument("--dimension",type=int,default=64)
    p.add_argument("--queries",type=int,default=100)
    p.add_argument("--nlist",type=int,default=32)
    p.add_argument("--k",type=int,default=10)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",default="results/benchmark")
    p.add_argument("--skip-faiss",action="store_true")
    run(p.parse_args())
