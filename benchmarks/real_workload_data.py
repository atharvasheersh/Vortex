"""Prepare bounded Vortex and FAISS indexes from real HDF5 vectors."""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import time
import zipfile
from itertools import pairwise
from pathlib import Path

import faiss
import h5py
import numpy as np
import psutil
from encode_snapshot import encode_snapshot
from run_performance import digest, write_json
from streaming_snapshot import header, layout
from vortex import Index
from vortex.evaluation import process_memory

DATASETS = {
    "sift": ("sift-128-euclidean", 1_000_000, 128, "euclidean", 128),
    "glove": ("glove-100-angular", 1_183_514, 100, "angular", 128),
    "deep": ("deep-image-96-angular", 9_990_000, 96, "angular", 256),
    "scifact": ("scifact-cached-minilm", 3_328, 384, "angular", 16),
}
TUNE = 192
HOLDOUT = 1000
SEED = 31827
BLOCK = 32768


def guard():
    if psutil.virtual_memory().available < 384 * 1024**2:
        raise MemoryError("available system RAM below 384 MiB")


def f32(values, angular):
    result = np.asarray(values, dtype="<f4", order="C")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite vector in corpus or queries")
    if angular:
        norms = np.sqrt(np.einsum("ij,ij->i", result, result, dtype=np.float64))
        if np.any(norms == 0):
            raise ValueError("angular distance requires nonzero vectors")
        result = (result / norms[:, None]).astype("<f4")
    return result


def query_split(queries, tune, holdout, fixed=False):
    order = (
        np.arange(len(queries))
        if fixed
        else np.random.default_rng(SEED).permutation(len(queries))
    )
    chosen, seen = [], set()
    for row in order:
        key = queries[row].tobytes()
        if key in seen:
            if fixed:
                raise ValueError("duplicate cached query bytes cross the fixed split")
            continue
        seen.add(key)
        chosen.append(row)
        if len(chosen) == tune + holdout:
            break
    if len(chosen) != tune + holdout:
        raise ValueError("insufficient distinct queries")
    return np.asarray(chosen, dtype=np.int64)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_scifact(cache, prepared, official_zip, output):
    """Verify vector source IDs, cached hashes and official test judgments."""
    out = output.parent
    out.mkdir(parents=True, exist_ok=True)
    test = read_json(prepared / "scifact-test.json")
    development = read_json(prepared / "scifact-development.json")
    documents = {int(c["id"]): str(c["source_id"]) for c in test["chunks"]}
    if documents != {int(c["id"]): str(c["source_id"]) for c in development["chunks"]}:
        raise ValueError("cached development and test use different corpora")
    with zipfile.ZipFile(official_zip) as source:
        rows = csv.DictReader(
            io.StringIO(source.read("scifact/qrels/test.tsv").decode()), delimiter="\t"
        )
        official = {}
        for row in rows:
            official.setdefault(row["query-id"], {})[row["corpus-id"]] = int(
                row["score"]
            )
    split_ids, queries, hashes = {}, [], {}
    for split, data in [("development", development), ("test", test)]:
        directory = cache / f"scifact-{split}"
        split_ids[split] = []
        for i, case in enumerate(data["cases"]):
            path = directory / f"query-{i:04d}-r0.npy"
            metadata = read_json(path.with_suffix(".json"))
            if (
                metadata["query_id"] != str(case["query_id"])
                or digest(path) != metadata["sha256"]
            ):
                raise ValueError("cached SciFact query provenance mismatch")
            values = np.load(path, allow_pickle=False)
            if values.shape != (1, 384):
                raise ValueError("unexpected cached query dimensions")
            queries.append(values[0])
            split_ids[split].append(metadata["query_id"])
            hashes[f"{directory.name}/{path.name}"] = metadata["sha256"]
    if set(split_ids["development"]) & set(split_ids["test"]):
        raise ValueError("SciFact query IDs overlap")
    vectors, ids = [], []
    for path in sorted((cache / "scifact-test").glob("corpus-*.npy")):
        metadata = read_json(path.with_suffix(".json"))
        if digest(path) != metadata["sha256"]:
            raise ValueError("cached SciFact corpus provenance mismatch")
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(metadata["ids"]), 384):
            raise ValueError("cached SciFact corpus dimensions mismatch")
        vectors.append(values)
        ids.extend(metadata["ids"])
        hashes[f"scifact-test/{path.name}"] = metadata["sha256"]
    if len(ids) != len(set(ids)) or set(ids) != set(documents):
        raise ValueError("cached embeddings do not cover the prepared corpus")
    test_qrels = {qid: official[qid] for qid in split_ids["test"]}
    for case in test["cases"]:
        qid = str(case["query_id"])
        positives = {doc for doc, relevance in test_qrels[qid].items() if relevance > 0}
        if positives != {
            str(doc) for doc in case["relevant_ids"]
        } or not positives <= set(documents.values()):
            raise ValueError(
                "official SciFact judgments do not match the cached corpus"
            )
    with h5py.File(output, "x") as file:
        file.create_dataset("train", data=np.concatenate(vectors))
        file.create_dataset("test", data=np.asarray(queries))
        file.create_dataset("public_ids", data=np.asarray(ids, dtype="<u8"))
    write_json(
        output.with_suffix(".json"),
        {
            "tuning_ids": split_ids["development"],
            "test_ids": split_ids["test"],
            "documents": {str(k): v for k, v in documents.items()},
            "official_qrels": test_qrels,
            "official_archive_sha256": digest(official_zip),
            "cached_files": hashes,
        },
    )


def training_sample(dataset, angular, count):
    ids = np.sort(
        np.random.default_rng(SEED + 1).choice(len(dataset), count, replace=False)
    )
    chunks = []
    for start in range(0, len(dataset), BLOCK):
        end = min(start + BLOCK, len(dataset))
        a, b = np.searchsorted(ids, [start, end])
        if a != b:
            chunks.append(f32(dataset[start:end][ids[a:b] - start], angular))
    return np.concatenate(chunks), hashlib.sha256(
        ids.astype("<i8").tobytes()
    ).hexdigest()


def exact_spot(corpus, ids, query):
    """Find twelve stable top scores via independently computed float64 L2."""
    top_ids, top_scores = np.empty(0, dtype="<u8"), np.empty(0, dtype=np.float64)
    for start in range(0, len(corpus), BLOCK):
        guard()
        slab = np.asarray(corpus[start : start + BLOCK], dtype=np.float64)
        diff = slab - query
        distances = np.einsum("ij,ij->i", diff, diff)
        merged_ids = np.concatenate([top_ids, ids[start : start + len(slab)]])
        merged_scores = np.concatenate([top_scores, distances])
        order = np.lexsort((merged_ids, merged_scores))[:12]
        top_ids, top_scores = merged_ids[order], merged_scores[order]
    return top_ids, top_scores


def exact_reference(file, query_ids, queries, corpus, public_ids, angular):
    checks = []
    if "neighbors" not in file:
        return np.array(
            [exact_spot(corpus, public_ids, q)[0][:10] for q in queries]
        ), checks
    # h5py fancy indexing requires increasing row order.
    order = np.argsort(query_ids)
    published = np.empty((len(queries), 10), dtype=np.int64)
    published[order] = file["neighbors"][query_ids[order], :10]
    if np.any(published < 0) or np.any(published >= len(corpus)):
        raise ValueError("published neighbors are outside the corpus")
    if any(len(np.unique(row)) != 10 for row in published):
        raise ValueError("duplicate published neighbors")
    for rank in np.linspace(0, len(queries) - 1, 8, dtype=int):
        query = queries[rank].astype(np.float64)
        ids, distances = exact_spot(corpus, public_ids, query)
        diff = corpus[published[rank]].astype(np.float64) - query
        values = np.einsum("ij,ij->i", diff, diff)
        archived = np.asarray(
            file["distances"][int(query_ids[rank]), :10], dtype=np.float64
        )
        expected = archived * 2 if angular else archived**2
        np.testing.assert_allclose(values, expected, rtol=1e-4, atol=3e-6)
        np.testing.assert_allclose(
            np.sort(values), distances[:10], rtol=1e-5, atol=2e-6
        )
        checks.append(
            {
                "query_id": int(query_ids[rank]),
                "same_id_set": set(ids[:10]) == set(public_ids[published[rank]]),
                "boundary_tie": bool(
                    np.isclose(distances[9], distances[10], rtol=1e-8, atol=1e-8)
                ),
            }
        )
    return public_ids[published], checks


def packed_snapshot(out, corpus, public_ids, labels, counts, centers):
    """Scatter contiguous batch runs into each final IVF list."""
    name = out / "packed-fp32.vtx"
    partial = out / "packed-fp32.vtx.partial"
    n, d = corpus.shape
    k = len(counts)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype("<u8")
    ids_offset, original, _, centroid, offset, postings, total = layout(n, d, k)
    cursors = offsets[:-1].copy()
    with partial.open("xb+") as file:
        file.truncate(total)
        file.seek(centroid)
        centers.astype("<f4", copy=False).tofile(file)
        file.seek(offset)
        offsets.tofile(file)
        for start in range(0, n, BLOCK):
            guard()
            end = min(n, start + BLOCK)
            slab_labels = labels[start:end]
            order = np.argsort(slab_labels, kind="stable")
            sorted_labels = slab_labels[order]
            bounds = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(order)]
            for first, last in pairwise(bounds):
                cluster = int(sorted_labels[first])
                cursor = int(cursors[cluster])
                indices = order[first:last]
                file.seek(ids_offset + cursor * 8)
                public_ids[start + indices].astype("<u8", copy=False).tofile(file)
                file.seek(original + cursor * d * 4)
                corpus[start + indices].tofile(file)
                cursors[cluster] += len(indices)
            file.seek(postings + start * 8)
            np.arange(start, end, dtype="<u8").tofile(file)
        if not np.array_equal(cursors, offsets[1:]):
            raise ValueError("packed IVF assignment did not cover the corpus")
        file.seek(0)
        file.write(header(n, d, k))
        file.flush()
        os.fsync(file.fileno())
    checked = Index.open(str(partial), cache_bytes=0)
    if len(checked) != n or checked.nlist != k:
        raise ValueError("packed snapshot failed native reopen")
    del checked
    os.link(partial, name)
    partial.unlink()
    return name


def prepare(name, source, out):
    label, n, d, metric, clusters = DATASETS[name]
    angular = metric == "angular"
    minimum_disk = n * (d * 14 + 128) + 1024**3
    if shutil.disk_usage(out.parent).free < minimum_disk + 8 * 1024**3:
        raise OSError(
            "insufficient disk for vectors, indexes, labels, and 8 GiB reserve"
        )
    out.mkdir(exist_ok=False)
    manifest = {
        "status": "started",
        "dataset": name,
        "label": label,
        "count": n,
        "dimension": d,
        "metric": metric,
        "nlist": clusters,
        "source_sha256": digest(source),
        "minimum_disk_bytes": minimum_disk,
        "build_threads": 4,
        "stages_seconds": {},
    }
    write_json(out / "build.json", manifest)
    started = time.perf_counter()
    stage = manifest["stages_seconds"]
    faiss.omp_set_num_threads(4)
    try:
        with h5py.File(source, "r") as file:
            if (
                file["train"].shape != (n, d)
                or file["test"].ndim != 2
                or file["test"].shape[1] != d
            ):
                raise ValueError("dataset shape mismatch")
            if name != "scifact" and (
                file["test"].shape != (10_000, d) or file["neighbors"].shape[1] < 10
            ):
                raise ValueError(
                    "unexpected published ANN-Benchmarks query/reference shape"
                )
            tune, holdout = (50, 150) if name == "scifact" else (TUNE, HOLDOUT)
            queries = f32(file["test"][:], angular)
            selected = query_split(queries, tune, holdout, name == "scifact")
            queries = queries[selected]
            ids = (
                file["public_ids"][:]
                if "public_ids" in file
                else np.arange(n, dtype="<u8")
            )
            if "public_ids" in file and (ids.shape != (n,) or len(np.unique(ids)) != n):
                raise ValueError("invalid public IDs")
            np.save(out / "public_ids.npy", ids)
            np.save(out / "queries.npy", queries)
            np.save(out / "query_ids.npy", selected)
            manifest.update(tuning_queries=tune, holdout_queries=holdout)

            before = time.perf_counter()
            sample, sample_hash = training_sample(file["train"], angular, min(n, 32768))
            model = faiss.Kmeans(d, clusters, niter=12, seed=SEED, gpu=False)
            model.train(sample)
            centers = np.ascontiguousarray(model.centroids)
            stage["shared_coarse_training"] = time.perf_counter() - before
            manifest.update(
                training_sample_count=len(sample),
                training_sample_ids_sha256=sample_hash,
            )
            before = time.perf_counter()
            quantizer = faiss.IndexFlatL2(d)
            quantizer.add(centers)
            code_bits = 4 if name == "scifact" else 8
            external = faiss.IndexIVFPQ(quantizer, d, clusters, d // 4, code_bits)
            external.pq.cp.niter = 10
            external.pq.cp.seed = SEED
            external.train(sample)
            np.testing.assert_array_equal(quantizer.reconstruct_n(0, clusters), centers)
            del sample, model
            stage["faiss_pq_training"] = time.perf_counter() - before
            manifest.update(faiss_pq_subquantizers=d // 4, faiss_pq_bits=code_bits)
            original = np.lib.format.open_memmap(
                out / "corpus.npy", mode="w+", dtype="<f4", shape=(n, d)
            )
            labels = np.lib.format.open_memmap(
                out / "labels.npy", mode="w+", dtype="<i4", shape=(n,)
            )
            counts = np.zeros(clusters, dtype=np.int64)
            assign_seconds, add_seconds = 0.0, 0.0
            before = time.perf_counter()
            for start in range(0, n, BLOCK):
                guard()
                data = f32(file["train"][start : start + BLOCK], angular)
                original[start : start + len(data)] = data
                assign_start = time.perf_counter()
                _, belonging = quantizer.search(data, 1)
                belonging = belonging[:, 0]
                labels[start : start + len(data)] = belonging
                counts += np.bincount(belonging, minlength=clusters)
                assign_seconds += time.perf_counter() - assign_start
                add_start = time.perf_counter()
                external.add(data)
                add_seconds += time.perf_counter() - add_start
            original.flush()
            labels.flush()
            stage["ingest_and_build_shared_pass"] = time.perf_counter() - before
            stage["coarse_assignment"] = assign_seconds
            stage["faiss_pq_add"] = add_seconds
            if external.ntotal != n or counts.sum() != n:
                raise ValueError("index construction lost vectors")
            before = time.perf_counter()
            faiss.write_index(external, str(out / "faiss.ivfpq"))
            stage["faiss_persist"] = time.perf_counter() - before
            del external, quantizer
            before = time.perf_counter()
            packed = packed_snapshot(out, original, ids, labels, counts, centers)
            stage["vortex_pack"] = time.perf_counter() - before
            before = time.perf_counter()
            encoded = encode_snapshot(
                packed, out / "packed-int8.vtx", "int8", check_resources=guard
            )
            stage["vortex_int8_encode"] = time.perf_counter() - before
            before = time.perf_counter()
            truth, checks = exact_reference(
                file, selected, queries, original, ids, angular
            )
            np.save(out / "reference_ids.npy", truth)
            stage["independent_reference_audit"] = time.perf_counter() - before
            del original, labels
            if name == "scifact":
                shutil.copyfile(source.with_suffix(".json"), out / "relevance.json")
            artifacts = [
                "packed-fp32.vtx",
                "packed-int8.vtx",
                "faiss.ivfpq",
                "corpus.npy",
                "public_ids.npy",
                "queries.npy",
                "query_ids.npy",
                "reference_ids.npy",
            ]
            if name == "scifact":
                artifacts.append("relevance.json")
            manifest.update(
                status="completed",
                list_sizes=counts.tolist(),
                reference_kind="independent float64 exact"
                if name == "scifact"
                else "published ANN-Benchmarks top ten",
                independent_checks=checks,
                artifacts={
                    f: {"bytes": (out / f).stat().st_size, "sha256": digest(out / f)}
                    for f in artifacts
                },
                encode_snapshot_sha256=encoded["snapshot_sha256"],
            )
    except Exception as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(seconds=time.perf_counter() - started, memory=process_memory())
        write_json(out / "build.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("dataset", choices=DATASETS)
    make.add_argument("source", type=Path)
    make.add_argument("output", type=Path)
    cache = sub.add_parser("scifact")
    for name in ["cache", "prepared", "official_zip", "output"]:
        cache.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "scifact":
        materialize_scifact(args.cache, args.prepared, args.official_zip, args.output)
    else:
        prepare(args.dataset, args.source, args.output)


if __name__ == "__main__":
    main()
