"""Validate compact optimization evidence, paired results and recomputed statistics."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from report_performance import paired_comparison, sha256, summarize


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(directory):
    checksums = load(directory / "checksums.json")
    actual_files = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    if set(checksums) != actual_files:
        raise ValueError("archive inventory mismatch")
    for name, expected in checksums.items():
        path = directory / name
        if (
            path.resolve().parent != directory.resolve()
            and directory.resolve() not in path.resolve().parents
        ):
            raise ValueError("archive checksum path escapes directory")
        if sha256(path) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    rows = [
        json.loads(line)
        for line in gzip.decompress((directory / "measurements.jsonl.gz").read_bytes())
        .decode()
        .splitlines()
    ]
    if summarize(rows) != load(directory / "summary.json"):
        raise ValueError("query records and summary disagree")
    manifests = load(directory / "manifests.json")
    plan = load(directory / "plan.json")
    builds = load(directory / "native_builds.json")
    cases = {case["name"]: case for case in plan}
    execution = load(directory / "execution.json")
    if execution["status"] != "completed" or set(execution["completed"]) != set(
        manifests
    ):
        raise ValueError("experiment completion mismatch")
    by_condition = defaultdict(list)
    expected_sources = next(iter(manifests.values()))["source_sha256"]
    for name, expected in expected_sources.items():
        if sha256(directory / "executed-source" / (name + ".txt")) != expected:
            raise ValueError("measured source fingerprint mismatch")
    references = {}
    for size in ["100k", "1m", "10m", "real-minilm"]:
        reference = directory / "references" / size
        info = load(reference / "reference.json")
        for filename, field in [
            ("queries.npy", "queries_sha256"),
            ("ids.npy", "ids_sha256"),
            ("scores.npy", "scores_sha256"),
        ]:
            if sha256(reference / filename) != info[field]:
                raise ValueError("independent reference hash mismatch")
        queries = np.load(reference / "queries.npy", allow_pickle=False)
        if len(np.unique(queries, axis=0)) != len(queries):
            raise ValueError("duplicate query vectors counted as independent")
        references[size] = np.load(reference / "ids.npy", allow_pickle=False)
    seen = set()
    for row in rows:
        condition, round_id, qid, probe = (
            row["condition"],
            row["round_id"],
            row["query_id"],
            row["nprobe"],
        )
        key = condition, round_id, qid, probe
        if key in seen:
            raise ValueError("duplicate query measurement")
        seen.add(key)
        case = cases[condition]
        if (
            round_id not in range(case["rounds"])
            or qid not in range(case["offset"], case["offset"] + case["count"])
            or probe not in [int(p) for p in case["probes"].split(",")]
        ):
            raise ValueError("query is outside declared split/configuration")
        if row["family"] != case["family"]:
            raise ValueError("condition family mismatch")
        size = (
            "real-minilm"
            if condition.startswith("real-minilm")
            else condition.split("-")[0]
        )
        ids = [h[0] for h in row["hits"]]
        if (
            len(ids) != 10
            or len(set(ids)) != 10
            or any(
                type(identifier) is not int or not 0 <= identifier < 2**64
                for identifier in ids
            )
        ):
            raise ValueError("invalid result IDs")
        manifest = manifests[f"{condition}-r{round_id}"]
        count = manifest["snapshot_info"]["n"]
        if type(row["scanned"]) is not int or not 0 <= row["scanned"] <= count:
            raise ValueError("invalid candidate count")
        if probe == 0 and (
            row["scanned"] != count
            or (row["family"] == "matched-fp32" and row["recall_at_10"] != 1)
        ):
            raise ValueError("exhaustive retrieval coverage/reference mismatch")
        ordered = [(hit[1], hit[0]) for hit in row["hits"]]
        if ordered != sorted(ordered):
            raise ValueError("result order or stable-ID tie handling changed")
        recall = len(set(ids) & set(references[size][qid].tolist())) / 10
        if abs(recall - row["recall_at_10"]) > 1e-12:
            raise ValueError("recall disagrees with independent reference")
        numeric = [
            row["wall_ms"],
            row["native_ms"],
            row["cpu_ms"],
            *[h[1] for h in row["hits"]],
        ]
        if not np.isfinite(numeric).all() or row["wall_ms"] <= 0:
            raise ValueError("invalid timing or score")
        if row["timing_gap"] != (
            row["wall_ms"] > 60000 and row["wall_ms"] > 10 * max(row["cpu_ms"], 1)
        ):
            raise ValueError("timing diagnostic mismatch")
        by_condition[condition, round_id].append(row)
    for name, case in cases.items():
        for round_id in range(case["rounds"]):
            records = by_condition[name, round_id]
            expected_count = case["count"] * len(case["probes"].split(","))
            manifest = manifests[f"{name}-r{round_id}"]
            if (
                len(records) != expected_count
                or manifest["recorded_calls"] != expected_count
                or manifest["status"] != "completed"
            ):
                raise ValueError("condition coverage mismatch")
            if (
                manifest["module_sha256"] != case["module_sha256"]
                or manifest["snapshot_sha256"] != case["snapshot_sha256"]
            ):
                raise ValueError("condition data/build identity mismatch")
            if (
                manifest["native_revision"]
                != builds[manifest["module_sha256"]]["revision"]
            ):
                raise ValueError("native revision mismatch")
            if manifest["source_sha256"] != expected_sources:
                raise ValueError("collector/native source changed between conditions")
            if (
                manifest["working_set_mib"] != 1536
                or manifest["threads"] != 1
                or manifest["cache_bytes"] != 8388608
                or manifest["block_bytes"] != 4096
            ):
                raise ValueError("memory/thread/cache comparison policy changed")
            policy = manifest["memory_policy"]
            if (
                not policy["enabled"]
                or policy["maximum_bytes"] != 1536 * 1024**2
                or manifest["memory_final"]["peak_rss_bytes"]
                > policy["maximum_bytes"] + 65536
            ):
                raise ValueError("resident-limit verification failed")
            if manifest["rerank_n"] != case["rerank"]:
                raise ValueError("reranking configuration mismatch")
    for comparison in load(directory / "comparisons.json"):
        recalculated = paired_comparison(
            rows, comparison["baseline"], comparison["optimized"], comparison["nprobe"]
        )
        if recalculated != comparison:
            raise ValueError("paired comparison or bootstrap interval mismatch")
    tuning_rows = [
        json.loads(line)
        for line in gzip.decompress((directory / "tuning.jsonl.gz").read_bytes())
        .decode()
        .splitlines()
    ]
    tuning_summary = load(directory / "tuning.json")
    selected = load(directory / "selection.json")["selected"]
    eligible = [row for row in tuning_summary if row["recall_at_10"] >= 0.95]
    if (
        min(eligible, key=lambda row: row["p95_ms"])["nlist"] != selected["nlist"]
        or min(eligible, key=lambda row: row["p95_ms"])["nprobe"] != selected["nprobe"]
    ):
        raise ValueError("selection does not follow the predeclared tuning rule")
    for lists in {r["nlist"] for r in tuning_rows}:
        grouped = [
            {"condition": "tuning", **r} for r in tuning_rows if r["nlist"] == lists
        ]
        for summary in summarize(grouped):
            stored = next(
                r
                for r in tuning_summary
                if r["nlist"] == lists and r["nprobe"] == summary["nprobe"]
            )
            for field in [
                "p50_ms",
                "p95_ms",
                "mean_ms",
                "recall_at_10",
                "mean_scanned",
                "calls",
            ]:
                if abs(stored[field] - summary[field]) > 1e-10:
                    raise ValueError("tuning summary mismatch")
        if any(r["query_id"] >= 32 for r in grouped):
            raise ValueError("held-out data leaked into tuning")
    optional_records = 0
    optional = directory / "optional-10m-int8"
    if optional.exists() and load(optional / "plan.json")["status"] == "completed":
        extra = [
            json.loads(line)
            for line in gzip.decompress(
                (optional / "measurements.jsonl.gz").read_bytes()
            )
            .decode()
            .splitlines()
        ]
        extra_manifests = load(optional / "manifests.json")
        if len(extra) != 64 or summarize(extra) != load(optional / "summary.json"):
            raise ValueError("optional compression coverage/summary mismatch")
        if paired_comparison(
            extra, "10m-int8-baseline", "10m-int8-optimized", 24
        ) != load(optional / "comparison.json"):
            raise ValueError("optional compression paired comparison mismatch")
        for label, manifest in extra_manifests.items():
            data = [r for r in extra if r["condition"] == f"10m-int8-{label}"]
            if {r["query_id"] for r in data} != set(range(32, 64)) or len(data) != 32:
                raise ValueError("optional compression query split mismatch")
            if (
                manifest["source_sha256"] != expected_sources
                or manifest["rerank_n"] != 100
                or manifest["working_set_mib"] != 1536
            ):
                raise ValueError("optional compression source/policy mismatch")
            if (
                manifest["status"] != "completed"
                or manifest["memory_final"]["peak_rss_bytes"] > 1536 * 1024**2 + 65536
            ):
                raise ValueError("optional compression resident bound mismatch")
            for row in data:
                ids = [hit[0] for hit in row["hits"]]
                recall = (
                    len(set(ids) & set(references["10m"][row["query_id"]].tolist()))
                    / 10
                )
                if row["nprobe"] != 24 or abs(row["recall_at_10"] - recall) > 1e-12:
                    raise ValueError("optional compression reference recall mismatch")
        optional_records = len(extra)
    return {
        "status": "passed",
        "records": len(rows),
        "tuning_records": len(tuning_rows),
        "conditions": len(plan),
        "archive_files": len(checksums),
        "optional_records": optional_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("benchmarks/reference/performance")
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))


if __name__ == "__main__":
    main()
