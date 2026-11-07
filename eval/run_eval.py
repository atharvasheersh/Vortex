"""Run one fixed dataset across retrieval settings; offline mode is explicit."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from vortex import Index
from vortex.evaluation import load_dataset, manifest, write_json
from vortex.metrics import answer_metrics, retrieval_metrics
from vortex.rag import FixtureEmbedder, SentenceEmbedder, answer_case, build_collection


def run(args):
    dataset = load_dataset(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = (FixtureEmbedder() if args.embedding == "fixture"
                else SentenceEmbedder(args.embedding_model, args.embedding_revision))
    index_path = output / "collection.vtx"
    builder, embedding_seconds = build_collection(dataset["chunks"], embedder, index_path,
                                                  args.nlist, args.dtype, args.threads)
    del builder
    index = Index.open(str(index_path), cache_bytes=args.cache_bytes, threads=args.threads)
    summary, records = [], []
    for nprobe in args.nprobe:
        rows = []
        for case in dataset["cases"]:
            row = answer_case(index, case, dataset["chunks"], embedder, args.k, nprobe,
                              args.rerank_n, args.generator, args.endpoint, args.generator_model)
            row["nprobe"] = nprobe
            row["retrieval_metrics"] = retrieval_metrics([h[0] for h in row["retrieved"]["hits"]],
                case["relevant_chunk_ids"], args.k, case.get("grades"))
            row["answer_metrics"] = answer_metrics(row["answer"], case.get("reference_answer"), row["context"])
            records.append(row)
            rows.append(row)
        result = {"nprobe": nprobe, "cases": len(rows), "generator": args.generator}
        for field in ("retrieval_metrics", "answer_metrics"):
            result[field] = {}
            for key in rows[0][field]:
                values = [row[field][key] for row in rows if row[field][key] is not None]
                result[field][key] = float(np.mean(values)) if values else None
        result["p95_retrieval_ms"] = float(np.percentile([row["timings_ms"]["retrieval"] for row in rows], 95))
        summary.append(result)
    configuration = vars(args).copy()
    configuration.pop("endpoint", None)
    configuration["embedding_model"] = embedder.name
    configuration["embedding_build_seconds"] = embedding_seconds
    configuration["generation_settings"] = {"temperature": 0, "seed": 42}
    run_manifest = manifest(configuration, args.dataset)
    run_manifest["limitations"] = [
        "Small project-authored fixture; labels are not an independent benchmark.",
        "Lexical support is a token-overlap proxy, not semantic faithfulness.",
        "Fixture generation is extractive plumbing verification, not LLM evaluation."
        if args.generator == "fixture" else "HTTP generation requires independent factual review.",
    ]
    write_json(output / "manifest.json", run_manifest)
    write_json(output / "summary.json", summary)
    with (output / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="eval/datasets/support_v1.json")
    p.add_argument("--output", default="results/eval")
    p.add_argument("--embedding", choices=["fixture", "sentence-transformers"], default="fixture")
    p.add_argument("--embedding-model")
    p.add_argument("--embedding-revision")
    p.add_argument("--generator", choices=["fixture", "http"], default="fixture")
    p.add_argument("--endpoint", default=os.environ.get("VORTEX_GENERATION_URL"))
    p.add_argument("--generator-model", default=os.environ.get("VORTEX_GENERATION_MODEL"))
    p.add_argument("--nlist", type=int, default=4)
    p.add_argument("--nprobe", type=int, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rerank-n", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "fp16", "int8"], default="fp32")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache-bytes", type=int, default=8192)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
