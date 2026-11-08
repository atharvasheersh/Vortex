"""Execute frozen real-model retrieval and generation experiments in stages."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vortex import Index
from vortex.evaluation import fingerprint, manifest, process_memory, write_json
from vortex.metrics import answer_metrics, retrieval_metrics
from vortex.rag import SentenceEmbedder
from vortex.real_rag import (
    CITATION,
    PROMPT,
    answer_scores,
    append_record,
    digest_object,
    evidence_context,
    load_real_dataset,
    load_records,
    writer_lock,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 96,
    "num_thread": 4,
}


def http_json(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def bind_run(args):
    contract = {
        "spec_sha256": fingerprint(ROOT / "eval/real_rag_spec.md"),
        "dataset_hashes": {
            s: fingerprint(args.dataset / f"{s}.json") for s in ["development", "test"]
        },
        "embedding_model": MODEL,
        "embedding_revision": REVISION,
        "generator": args.model,
        "options": OPTIONS,
        "prompt": PROMPT,
        "context_bytes": 3000,
        "runner_sha256": fingerprint(__file__),
        "helpers_sha256": fingerprint(ROOT / "python/vortex/real_rag.py"),
    }
    path = args.output / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["contract"] != contract:
            raise ValueError("resume contract differs; choose a new run directory")
    else:
        write_json(
            path,
            {
                **manifest(contract),
                "contract": contract,
                "starting_commit": "84e95f6a4efc59f6a90c3698e43aa7478fd3e95e",
            },
        )


def embed(args):
    import torch

    torch.set_num_threads(4)
    encoder = SentenceEmbedder(MODEL, REVISION)
    for split in ["development", "test"]:
        output = args.output / split
        output.mkdir(exist_ok=True)
        if (output / "embeddings.npz").exists():
            continue
        dataset = load_real_dataset(args.dataset / f"{split}.json")
        start = time.perf_counter()
        vectors = encoder.encode([c["text"] for c in dataset["chunks"]])
        corpus_seconds = time.perf_counter() - start
        queries, query_ms = [], []
        for case in dataset["cases"]:
            start = time.perf_counter()
            queries.append(encoder.encode([case["query"]])[0])
            query_ms.append((time.perf_counter() - start) * 1000)
        np.savez(
            output / "embeddings.npz",
            corpus=vectors,
            queries=np.array(queries),
            query_ms=query_ms,
        )
        write_json(
            output / "embedding.json",
            {
                "corpus_seconds": corpus_seconds,
                "sha256": fingerprint(output / "embeddings.npz"),
                "dimension": vectors.shape[1],
                "model_max_tokens": encoder.model.max_seq_length,
                "process_memory": process_memory(),
            },
        )
        print(f"embedded {split}: {vectors.shape}", flush=True)


def configuration_name(config):
    return f"{config['dtype']}-l{config['nlist']}-p{config['nprobe']}-k{config['k']}-r{config['rerank_n']}"


def snapshot(directory, data, vectors, dtype, nlist):
    path = directory / f"{dtype}-{nlist}.vtx"
    if path.exists():
        return path
    start = time.perf_counter()
    index = Index(vectors.shape[1], metric="cosine", threads=1)
    for chunk, vector in zip(data["chunks"], vectors):
        index.add(chunk["id"], vector)
    index.build_ivf(nlist=nlist, seed=42)
    index.set_dtype(dtype)
    index.save(str(path))
    write_json(
        path.with_suffix(".json"),
        {
            "build_seconds": time.perf_counter() - start,
            "bytes": path.stat().st_size,
            "sha256": fingerprint(path),
            "process_memory": process_memory(),
        },
    )
    return path


def search_configuration(args, split, config):
    directory = args.output / split
    name = configuration_name(config)
    destination = directory / f"retrieval-{name}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    data = load_real_dataset(args.dataset / f"{split}.json")
    arrays = np.load(directory / "embeddings.npz")
    path = snapshot(directory, data, arrays["corpus"], config["dtype"], config["nlist"])
    index = Index.open(str(path), cache_bytes=8 * 1024 * 1024, threads=1)
    vectors = arrays["corpus"].astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.array([c["id"] for c in data["chunks"]])
    positions = {int(identifier): i for i, identifier in enumerate(ids)}
    rows = []
    for case, query in zip(data["cases"], arrays["queries"]):
        q = query.astype(np.float64)
        q /= np.linalg.norm(q)
        scores = vectors @ q
        order = np.lexsort((ids, -scores))[: config["k"]]
        expected = ids[order].tolist()
        start = time.perf_counter()
        found = index.search(
            query, k=config["k"], nprobe=config["nprobe"], rerank_n=config["rerank_n"]
        )
        milliseconds = (time.perf_counter() - start) * 1000
        hits = [int(h[0]) for h in found["hits"]]
        if config["dtype"] == "fp32" and config["nprobe"] == 0:
            if set(expected) != set(hits):
                unexpected = set(hits) - set(expected)
                if any(
                    scores[positions[i]] < scores[order[-1]] - 1e-5 for i in unexpected
                ):
                    raise AssertionError(
                        "exact retrieval disagrees with independent cosine scores"
                    )
            if any(
                abs(score - scores[positions[int(identifier)]]) > 1e-5
                for identifier, score in found["hits"]
            ):
                raise AssertionError("exact returned score mismatch")
        rows.append(
            {
                "query_id": case["query_id"],
                "ids": hits,
                "scores": found["hits"],
                "retrieval_ms": milliseconds,
                "ann_neighbor_recall": len(set(hits) & set(expected)) / config["k"],
                "labeled_metrics": retrieval_metrics(
                    hits, case["relevant_chunk_ids"], config["k"]
                )
                if case["answerable"]
                else None,
            }
        )
    labeled = [
        r["labeled_metrics"]["recall"] for r in rows if r["labeled_metrics"] is not None
    ]
    result = {
        "configuration": config,
        "rows": rows,
        "labeled_recall": float(np.mean(labeled)),
        "ann_neighbor_recall": float(np.mean([r["ann_neighbor_recall"] for r in rows])),
        "median_retrieval_ms": float(np.median([r["retrieval_ms"] for r in rows])),
        "cache": index.cache_stats(),
    }
    write_json(destination, result)
    return result


def tune(args):
    selection_path = args.output / "selection.json"
    if selection_path.exists():
        raise FileExistsError("selection already frozen")
    exacts = [
        search_configuration(
            args,
            "development",
            {"dtype": "fp32", "nlist": 16, "nprobe": 0, "k": k, "rerank_n": 0},
        )
        for k in [3, 5]
    ]
    exact = max(exacts, key=lambda r: (r["labeled_recall"], -r["configuration"]["k"]))
    k = exact["configuration"]["k"]
    candidates = []
    for dtype in ["fp32", "fp16", "int8"]:
        for nlist in [16, 32]:
            for probe in [1, 4, 8, nlist]:
                for rerank in [0] if dtype == "fp32" else [0, 20]:
                    candidates.append(
                        search_configuration(
                            args,
                            "development",
                            {
                                "dtype": dtype,
                                "nlist": nlist,
                                "nprobe": probe,
                                "k": k,
                                "rerank_n": rerank,
                            },
                        )
                    )
    chosen = {"exact": exact["configuration"]}
    gates = {}
    for name, pool in [
        ("ivf", [r for r in candidates if r["configuration"]["dtype"] == "fp32"]),
        (
            "encoded",
            [
                r
                for r in candidates
                if r["configuration"]["dtype"] != "fp32"
                and r["configuration"]["rerank_n"]
            ],
        ),
    ]:
        qualified = [
            r
            for r in pool
            if r["ann_neighbor_recall"] >= 0.95
            and r["labeled_recall"] >= exact["labeled_recall"] - 0.02
        ]
        winner = (
            min(qualified, key=lambda r: r["median_retrieval_ms"])
            if qualified
            else max(
                pool, key=lambda r: (r["labeled_recall"], r["ann_neighbor_recall"])
            )
        )
        chosen[name] = winner["configuration"]
        gates[name] = {
            "passed": bool(qualified),
            "development_labeled_recall": winner["labeled_recall"],
            "development_ann_neighbor_recall": winner["ann_neighbor_recall"],
        }
    write_json(
        selection_path,
        {
            "configurations": chosen,
            "selection_gates": gates,
            "development_dataset_sha256": fingerprint(
                args.dataset / "development.json"
            ),
        },
    )
    print(json.dumps(chosen), flush=True)


def generate(args):
    selected = json.loads((args.output / "selection.json").read_text())[
        "configurations"
    ]
    data = load_real_dataset(args.dataset / f"{args.split}.json")
    retrieval = {
        name: {
            r["query_id"]: r
            for r in search_configuration(args, args.split, config)["rows"]
        }
        for name, config in selected.items()
    }
    directory = args.output / args.split
    records_path = directory / ("smoke.jsonl" if args.smoke else "generations.jsonl")
    cases = data["cases"]
    if args.smoke:
        cases = [
            next(c for c in cases if c["answerable"]),
            next(c for c in cases if not c["answerable"]),
        ]
    version = http_json(args.endpoint + "/api/version")
    tags = http_json(args.endpoint + "/api/tags")
    model = next(m for m in tags["models"] if m["name"] == args.model)
    metadata_path = args.output / "generator.json"
    metadata = {
        "version": version,
        "model": model,
        "show": http_json(args.endpoint + "/api/show", {"model": args.model}),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (
            previous["model"]["digest"] != model["digest"]
            or previous["version"] != version
        ):
            raise ValueError("generator changed during resume")
    else:
        write_json(metadata_path, metadata)
    existing = load_records(records_path)
    done = {(r["condition"], r["query_id"]) for r in existing}
    lookup = {c["id"]: c for c in data["chunks"]}
    arrays = np.load(directory / "embeddings.npz")
    embedding_times = dict(
        zip([c["query_id"] for c in data["cases"]], arrays["query_ms"].tolist())
    )
    warmup = http_json(
        args.endpoint + "/api/generate",
        {
            "model": args.model,
            "prompt": "Reply OK.",
            "stream": False,
            "options": OPTIONS,
            "keep_alive": "30m",
        },
    )
    append_record(
        directory / "warmups.jsonl",
        {"condition": "warmup", "query_id": str(time.time_ns()), "response": warmup},
    )
    for case in cases:
        for condition in ["no_context", "oracle", "exact", "ivf", "encoded"]:
            if (condition, case["query_id"]) in done:
                continue
            start = time.perf_counter()
            retrieval_row = (
                retrieval[condition][case["query_id"]]
                if condition in retrieval
                else None
            )
            ids = (
                retrieval_row["ids"]
                if retrieval_row
                else []
                if condition == "no_context"
                else case["relevant_chunk_ids"]
                if case["answerable"]
                else case["source_chunk_ids"]
            )
            context = evidence_context(ids, lookup)
            context_ms = (time.perf_counter() - start) * 1000
            prompt = PROMPT.format(context=context["context"], query=case["query"])
            attempts, payload = [], None
            request_start = time.perf_counter()
            for attempt in range(2):
                began = time.perf_counter()
                try:
                    payload = http_json(
                        args.endpoint + "/api/generate",
                        {
                            "model": args.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": OPTIONS,
                            "keep_alive": "30m",
                        },
                    )
                    if not isinstance(payload.get("response"), str) or not payload.get(
                        "done"
                    ):
                        raise ValueError("incomplete generation response")
                    if (
                        payload.get("prompt_eval_count", 0)
                        > OPTIONS["num_ctx"] - OPTIONS["num_predict"]
                    ):
                        raise ValueError("prompt exceeds context allocation")
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "status": "ok",
                            "seconds": time.perf_counter() - began,
                        }
                    )
                    break
                except (
                    HTTPError,
                    URLError,
                    TimeoutError,
                    OSError,
                    ValueError,
                ) as error:
                    payload = None
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "status": "failed",
                            "error": str(error),
                            "seconds": time.perf_counter() - began,
                        }
                    )
            generation_ms = (time.perf_counter() - request_start) * 1000
            answer = payload["response"] if payload else ""
            scores = answer_scores(answer, case["reference_answers"])
            if payload is None or scores["empty_output"]:
                scores.update(squad_exact_match=0.0, squad_token_f1=0.0)
            citations = [int(x) for x in CITATION.findall(answer)]
            row = {
                "condition": condition,
                "query_id": case["query_id"],
                "article": case["article"],
                "answerable": case["answerable"],
                "status": "ok" if payload else "failed",
                "query": case["query"],
                "reference_answers": case["reference_answers"],
                **context,
                "requested_ids": ids,
                "answer": answer,
                "scores": scores,
                "diagnostics": answer_metrics(
                    scores["scored_answer"], None, context["context"]
                ),
                "citations": citations,
                "valid_citations": sum(c in context["included_ids"] for c in citations),
                "prompt_sha256": digest_object(prompt),
                "generation_response": payload,
                "attempts": attempts,
                "timings_ms": {
                    "embedding": embedding_times[case["query_id"]]
                    if retrieval_row
                    else 0,
                    "retrieval": retrieval_row["retrieval_ms"] if retrieval_row else 0,
                    "context": context_ms,
                    "generation": generation_ms,
                },
                "process_memory": process_memory(),
            }
            row["timings_ms"]["total"] = sum(row["timings_ms"].values())
            append_record(records_path, row)
            done.add((condition, case["query_id"]))
            print(
                f"{args.split} {len(done)}/{len(cases) * 5} {condition} {case['query_id']} {row['status']}",
                flush=True,
            )
    if any(r["status"] != "ok" for r in load_records(records_path)):
        raise RuntimeError("generation finished with recorded failures")
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["embed", "tune", "generate"])
    parser.add_argument("--dataset", type=Path, default=Path("data/real-rag/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "test"], default="test")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with writer_lock(args.output):
        bind_run(args)
        {"embed": embed, "tune": tune, "generate": generate}[args.stage](args)


if __name__ == "__main__":
    main()
