"""Validate archived scale measurements against hashes and per-query evidence."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SIZES = (1000, 10000, 100000, 1000000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, fraction):
    """Linear interpolation equivalent to the recorded NumPy percentile method."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_archive(directory):
    root = Path(directory)
    inventory = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in inventory["files"].items():
        path = root / relative
        require(
            path.resolve().is_relative_to(root.resolve()),
            "evidence path escapes archive",
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f"checksum mismatch: {relative}",
        )
    runner_hash = hashlib.sha256(
        (root / "runner_snapshot.txt").read_bytes()
    ).hexdigest()
    summaries = {}
    if (root / "summary.json").exists():
        summaries = {
            row["vectors"]: row
            for row in json.loads((root / "summary.json").read_text(encoding="utf-8"))
        }
        require(set(summaries) == set(SIZES), "summary size coverage mismatch")
    for size in SIZES:
        result = json.loads(
            (root / str(size) / "result.json").read_text(encoding="utf-8")
        )
        require(
            result["status"] == "passed" and result["size"] == size,
            f"run did not pass: {size}",
        )
        require(
            result["manifest"]["configuration"]["runner_sha256"] == runner_hash,
            "runner identity mismatch",
        )
        require(
            result["manifest"]["git_commit"] == inventory["engine_revision"],
            "source revision mismatch",
        )
        require(
            result["manifest"]["source_sha256"] == inventory["source_sha256"],
            "source fingerprint mismatch",
        )
        records = [
            json.loads(line)
            for line in (root / str(size) / "queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        expected_count = result["repeats"] * (
            result["query_count"] * (1 + len(result["ivf"]))
            + result["mapped_ivf"]["query_count"]
        )
        require(len(records) == expected_count, "per-query evidence count mismatch")
        grouped = defaultdict(list)
        indexed = {}
        for row in records:
            key = (row["mode"], row["nprobe"], row["repeat"], row["query_id"])
            require(key not in indexed, "duplicate timed query")
            require(
                len(row["ids"]) <= result["k"]
                and len(set(row["ids"])) == len(row["ids"]),
                "invalid returned IDs",
            )
            require(
                all(
                    type(identifier) is int and 0 <= identifier < size
                    for identifier in row["ids"]
                ),
                "out-of-range ID",
            )
            require(
                math.isfinite(row["wall_ms"]) and row["wall_ms"] >= 0, "invalid timing"
            )
            indexed[key] = row
            grouped[(row["mode"], row["nprobe"], row["repeat"])].append(row["wall_ms"])
        configurations = [("exact", result["exact"])] + [
            ("ivf", item) for item in result["ivf"]
        ]
        configurations.append(("mapped_ivf", result["mapped_ivf"]))
        for mode, configuration in configurations:
            probes = configuration["nprobe"]
            p95s = []
            for repeat in range(result["repeats"]):
                times = grouped[(mode, probes, repeat)]
                require(
                    len(times) == configuration["query_count"],
                    "query coverage mismatch",
                )
                actual_p95 = percentile(times, 0.95)
                require(
                    math.isclose(
                        actual_p95,
                        configuration["rounds"][repeat]["p95_ms"],
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    ),
                    "round percentile mismatch",
                )
                p95s.append(actual_p95)
                for query_id in range(configuration["query_count"]):
                    actual = indexed[(mode, probes, repeat, query_id)]["ids"]
                    first = indexed[(mode, probes, 0, query_id)]["ids"]
                    require(actual == first, "repeat output mismatch")
                    truth = indexed[("exact", 0, 0, query_id)]["ids"]
                    if mode == "ivf" and probes == result["nlist"]:
                        require(actual == truth, "full-probe exact agreement failed")
                    if mode == "mapped_ivf":
                        require(
                            actual == indexed[("ivf", probes, 0, query_id)]["ids"],
                            "save/reopen retrieval mismatch",
                        )
            require(
                math.isclose(
                    statistics.median(p95s),
                    configuration["median_round_p95_ms"],
                    rel_tol=1e-10,
                ),
                "median percentile mismatch",
            )
            recalls = []
            for query_id in range(configuration["query_count"]):
                truth = indexed[("exact", 0, 0, query_id)]["ids"]
                actual = indexed[(mode, probes, 0, query_id)]["ids"]
                recalls.append(len(set(truth) & set(actual)) / len(truth))
            require(
                math.isclose(
                    statistics.mean(recalls),
                    configuration["recall_at_k"],
                    rel_tol=1e-10,
                ),
                "recall mismatch",
            )
        qualified = [
            row
            for row in result["ivf"]
            if row["recall_at_k"] >= result["quality_target"]
        ]
        selected = min(qualified, key=lambda row: row["nprobe"])
        require(
            selected == result["selected_ivf"],
            "quality operating-point selection mismatch",
        )
        require(
            result["faiss_flat"]["exact_neighbor_agreement"] == 1,
            "FAISS disagreement recorded",
        )
        require(
            len(result["numpy_checked_query_ids"]) == 3, "missing independent checks"
        )
        require(
            result["mapped_ivf"]["cache_stats"]["resident_bytes"]
            <= result["cache_budget"],
            "cache payload exceeded budget",
        )
        if size in summaries:
            summary = summaries[size]
            pairs = [
                (summary["exact_p95_ms"], result["exact"]["median_round_p95_ms"]),
                (summary["ivf_p95_ms"], selected["median_round_p95_ms"]),
                (summary["ann_recall_at_10"], selected["recall_at_k"]),
                (
                    summary["mapped_ivf_p95_ms"],
                    result["mapped_ivf"]["median_round_p95_ms"],
                ),
                (
                    summary["total_build_seconds"],
                    result["stages_seconds"]["insert"]
                    + result["stages_seconds"]["ivf_build"],
                ),
                (
                    summary["process_peak_rss_mib"],
                    result["manifest"]["process_memory"]["peak_rss_bytes"] / 1048576,
                ),
            ]
            require(
                all(math.isclose(a, b, rel_tol=1e-10) for a, b in pairs),
                "summary does not match evidence",
            )
    return len(SIZES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="benchmarks/reference/scaling")
    args = parser.parse_args()
    try:
        count = validate_archive(args.directory)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"Evidence validation failed: {error}\n")
    print(
        f"{count} sizes verified against checksums, query logs, recall and latency records."
    )
