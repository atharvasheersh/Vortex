"""Check public comparison checksums, coverage and aggregate consistency."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vortex.evaluation import fingerprint


def validate(directory):
    checksums = json.loads((directory / "checksums.json").read_text())
    if set(checksums) != {
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.name != "checksums.json"
    }:
        raise ValueError("unexpected or missing public evidence files")
    for name, expected in checksums.items():
        if Path(name).name != name or fingerprint(directory / name) != expected:
            raise ValueError("public evidence checksum mismatch")
    summary = json.loads((directory / "summary.json").read_text())
    for path, expected in summary.get("analysis_sources_sha256", {}).items():
        if fingerprint(ROOT / path) != expected:
            raise ValueError("analysis implementation differs from published evidence")
    models = summary["models"]
    if set(models) != {"minilm", "qwen", "jina", "embeddinggemma", "voyage", "gemini"}:
        raise ValueError("missing selected model status")
    if summary["human_review_status"] != "pending":
        raise ValueError("human review status requires actual owner annotations")
    audit = summary["supplemental_timing_audit"]
    if audit["status"] != "completed":
        raise ValueError("final timing interpretation requires the supplemental audit")
    for model, row in audit["models"].items():
        if (
            row["calls"] != 84
            or row["output_dtypes"] != ["float32"]
            or not row["matches_requested_fp32"]
            or row["inference_threads"] != 4
            or set(row["parameter_devices"]) != {"cpu"}
        ):
            raise ValueError(
                "actual encoder precision, device or thread settings differ from the CPU study"
            )
    if summary["rag"]["status"] != "completed" or summary["rag"][
        "total_requests"
    ] != 70 * (2 + len(summary["selection"]["rag_models"])):
        raise ValueError(
            "final reference requires complete main and repeated RAG records"
        )
    for name in ["voyage", "gemini"]:
        if (
            models[name]["status"] != "pending_owner_deferred"
            or models[name]["requests"]
            or models[name]["actual_cost_usd"]
        ):
            raise ValueError(
                "deferred APIs must have zero executed requests and charges"
            )
    with (directory / "retrieval.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(r["model"], r["domain"], r["mode"], int(r["k"])) for r in rows}
    expected = {
        (m, d, s, k)
        for m in ["minilm", "qwen", "jina", "embeddinggemma"]
        for d in ["squad", "scifact"]
        for s in ["exact", "selected"]
        for k in [5, 10]
    }
    if keys != expected or len(rows) != len(expected):
        raise ValueError("aggregate retrieval coverage is incomplete or duplicated")
    for row in rows:
        model = models[row["model"]]
        if model["status"] != "completed":
            raise ValueError("CSV contains incomplete model")
        metric = model["datasets"][row["domain"]]["retrieval"][
            f"{row['mode']}@{row['k']}"
        ]
        if (
            metric["positive_cases"] != 150
            or metric["failures"]
            or metric["timing_observations"] != metric["cases"] * 3
        ):
            raise ValueError("denominator or timing coverage mismatch")
        for field in ["recall", "hit", "mrr", "ndcg"]:
            value = metric["labeled"][field]
            if (
                not 0 <= value["mean"] <= 1
                or not 0 <= value["ci95"][0] <= value["ci95"][1] <= 1
            ):
                raise ValueError("invalid ranking metric or interval")
            if abs(float(row[field]) - value["mean"]) > 1e-12:
                raise ValueError("CSV disagrees with JSON")
    print(
        f"Validated {len(rows)} aggregate retrieval configurations; all six model states recorded."
    )
    return summary


def validate_private(summary, dataset, run=None):
    import numpy as np
    from vortex.embedding_study import load_study

    for name, expected in summary["data"]["dataset_hashes"].items():
        if fingerprint(dataset / name) != expected:
            raise ValueError("prepared data differs from published experiment")
    for domain in ["squad", "scifact"]:
        development, test = [
            load_study(dataset / f"{domain}-{split}.json")
            for split in ["development", "test"]
        ]
        for field in ["query_id", "group_id"]:
            if {c[field] for c in development["cases"]} & {
                c[field] for c in test["cases"]
            }:
                raise ValueError("query or source-group split leakage")
        if domain == "squad":
            if {c["text"] for c in development["chunks"]} & {
                c["text"] for c in test["chunks"]
            }:
                raise ValueError("duplicate QA chunks across splits")
            if set(summary["data"]["excluded_previous_query_ids"]) & {
                c["query_id"] for c in test["cases"]
            }:
                raise ValueError("previously inspected query entered held-out data")
        else:
            if {i for c in development["cases"] for i in c["relevant_ids"]} & {
                i for c in test["cases"] for i in c["relevant_ids"]
            }:
                raise ValueError("scientific positive-document split leakage")
        if run is not None:
            for model in ["minilm", "qwen", "jina", "embeddinggemma"]:
                dimension = summary["models"][model]["identity"]["dimensions"]
                for split, data in [("development", development), ("test", test)]:
                    directory = run / model / f"{domain}-{split}"
                    metadata = json.loads((directory / "embedding.json").read_text())
                    if (
                        fingerprint(directory / "embeddings.npz")
                        != metadata["embedding_sha256"]
                    ):
                        raise ValueError("private vector archive checksum mismatch")
                    with np.load(
                        directory / "embeddings.npz", allow_pickle=False
                    ) as arrays:
                        for key, count in [
                            ("corpus", len(data["chunks"])),
                            ("queries", len(data["cases"])),
                        ]:
                            matrix = arrays[key]
                            if (
                                matrix.shape != (count, dimension)
                                or matrix.dtype != np.float32
                                or not np.isfinite(matrix).all()
                                or not np.allclose(
                                    np.linalg.norm(matrix, axis=1), 1, atol=1e-5
                                )
                            ):
                                raise ValueError(
                                    "private vector dtype, dimensions or normalization mismatch"
                                )
    print("Private dataset separation, freshness and vector archives verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path("eval/reference/embedding-comparison")
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    summary = validate(args.reference)
    if args.run is not None and args.dataset is None:
        parser.error("--run requires --dataset")
    if args.dataset is not None:
        if args.run is None:
            validate_private(summary, args.dataset)
        else:
            from vortex.real_rag import writer_lock

            with writer_lock(args.run):
                validate_private(summary, args.dataset, args.run)


if __name__ == "__main__":
    main()
