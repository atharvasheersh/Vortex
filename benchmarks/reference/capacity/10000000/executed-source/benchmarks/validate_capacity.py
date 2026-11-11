"""Independently reconcile a completed capacity run with its saved query evidence."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from validate_scaling import percentile, require


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), message)


def validate_run(directory, check_snapshot=False):
    root = Path(directory)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    require(result["status"] == "passed", "run did not pass")
    count, queries, repeats, k = (
        result[key] for key in ["size", "query_count", "repeats", "k"]
    )
    require(result["opened_vector_count"] == count, "partial collection opened")
    require(
        result["self_lookup_ids"] == sorted({0, count // 2, count - 1}),
        "self lookups missing",
    )
    for name, expected in result["sources_sha256"].items():
        target = root / "executed-source" / name
        require(
            target.resolve().is_relative_to((root / "executed-source").resolve()),
            "unsafe source path",
        )
        require(sha256(target) == expected, f"executed source changed: {name}")
    for name, key in [
        ("queries.npy", "queries_sha256"),
        ("reference.json", "reference_sha256"),
        ("measurements.jsonl", "measurements_sha256"),
    ]:
        require(sha256(root / name) == result[key], f"checksum mismatch: {name}")
    import numpy as np

    vectors = np.load(root / "queries.npy", allow_pickle=False)
    require(
        vectors.dtype == np.float32 and vectors.shape == (queries, result["dimension"]),
        "query shape/type mismatch",
    )
    require(
        np.array_equal(
            vectors,
            np.random.default_rng(2026).standard_normal(
                vectors.shape, dtype=np.float32
            ),
        ),
        "query seed mismatch",
    )
    if check_snapshot:
        snapshot = root / "collection.vtx"
        require(
            snapshot.stat().st_size == result["construction"]["snapshot_bytes"],
            "snapshot size mismatch",
        )
        require(
            sha256(snapshot) == result["snapshot_sha256"], "snapshot checksum mismatch"
        )
    require(
        sum(result["construction"]["list_sizes"]) == count, "postings coverage mismatch"
    )
    require(
        result["cache_stats"]["resident_bytes"] <= result["cache_budget"],
        "cache budget exceeded",
    )
    policy = result.get("memory_policy", {"enabled": False})
    if policy["enabled"]:
        require(
            policy["flags"] & 0x4 and not policy["flags"] & 0x8,
            "resident bound was not confirmed",
        )
        require(
            all(
                stage["peak_rss_bytes"] <= policy["maximum_bytes"]
                for stage in result["memory_stages"].values()
            ),
            "observed peak exceeds resident bound",
        )
    reference = json.loads((root / "reference.json").read_text(encoding="utf-8"))
    require(
        len(reference["ids"]) == len(reference["scores"]) == queries,
        "reference query count mismatch",
    )
    require(
        set(reference["numpy_float64"])
        == {str(q) for q in {0, queries // 2, queries - 1}},
        "independent float64 checks missing",
    )
    for q, precise in reference["numpy_float64"].items():
        require(
            len(precise["ids"]) == len(precise["scores"]) == k,
            "incomplete float64 reference",
        )
        require(
            all(
                math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4)
                for a, b in zip(precise["scores"], reference["scores"][int(q)])
            ),
            "float64 distances disagree",
        )
    records = [
        json.loads(line)
        for line in (root / "measurements.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    probes = sorted(
        {p for p in [1, 4, 8, 16, 24, 28, 32, result["nlist"]] if p <= result["nlist"]}
    )
    require(
        [row["nprobe"] for row in result["ivf"]] == probes, "probe coverage mismatch"
    )
    require(
        len(records) == (1 + len(probes)) * queries * repeats,
        "measurement count mismatch",
    )
    grouped, indexed = defaultdict(list), {}
    for row in records:
        key = row["nprobe"], row["repeat"], row["query_id"]
        require(key not in indexed, "duplicate measurement")
        require(
            key[0] in [0, *probes] and 0 <= key[1] < repeats and 0 <= key[2] < queries,
            "invalid query key",
        )
        require(
            row["mode"] == ("mapped_exact" if key[0] == 0 else "mapped_ivf"),
            "mode mismatch",
        )
        require(row["condition"] == f"probe{key[0]}:r{key[1]}", "condition mismatch")
        require(
            len(row["ids"]) == len(row["scores"]) == k and len(set(row["ids"])) == k,
            "invalid neighbor count",
        )
        require(
            all(isinstance(i, int) and 0 <= i < count for i in row["ids"]),
            "neighbor ID out of range",
        )
        require(
            all(math.isfinite(s) and s >= 0 for s in row["scores"]), "invalid distances"
        )
        require(row["scores"] == sorted(row["scores"]), "distances not ordered")
        require(
            all(
                math.isfinite(row[key]) and row[key] >= 0
                for key in ["wall_ms", "native_ms", "process_cpu_seconds"]
            ),
            "invalid clocks",
        )
        require(
            row["wall_ms"] > 0 and k <= row["scanned"] <= count,
            "invalid timing or scan count",
        )
        indexed[key] = row
        grouped[key[0]].append(row)
    for probe in [0, *probes]:
        rows = grouped[probe]
        for repeat in range(repeats):
            for q in range(queries):
                row, truth = indexed[probe, repeat, q], indexed[0, 0, q]
                require(row["ids"] == indexed[probe, 0, q]["ids"], "repeat instability")
                if probe == 0:
                    require(row["scanned"] == count, "exact search did not scan N")
                    require(
                        all(
                            math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4)
                            for a, b in zip(row["scores"], reference["scores"][q])
                        ),
                        "exact reference score mismatch",
                    )
                    require(
                        row["reference_boundary_tie"]
                        == (row["ids"] != reference["ids"][q]),
                        "boundary tie flag mismatch",
                    )
                if probe == result["nlist"]:
                    require(
                        row["ids"] == truth["ids"]
                        and row["scores"] == truth["scores"]
                        and row["scanned"] == count,
                        "full probe differs from exact",
                    )
        summary = (
            result["exact"]
            if probe == 0
            else next(v for v in result["ivf"] if v["nprobe"] == probe)
        )
        times = [r["wall_ms"] for r in rows]
        for key, fraction in [("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99)]:
            close(summary[key], percentile(times, fraction), f"{key} mismatch")
        close(
            summary["qps_excluding_loop_overhead"],
            len(times) * 1000 / sum(times),
            "QPS mismatch",
        )
        round_p95 = []
        for repeat in range(repeats):
            times = [r["wall_ms"] for r in rows if r["repeat"] == repeat]
            p95 = percentile(times, 0.95)
            close(summary["rounds"][repeat]["p95_ms"], p95, "round p95 mismatch")
            round_p95.append(p95)
        close(
            summary["median_round_p95_ms"],
            statistics.median(round_p95),
            "median p95 mismatch",
        )
        recall = statistics.mean(
            len(set(r["ids"]) & set(indexed[0, 0, r["query_id"]]["ids"])) / k
            for r in rows
        )
        close(summary["recall_at_k"], recall, "recall mismatch")
        close(
            summary["mean_scanned"],
            statistics.mean(r["scanned"] for r in rows),
            "scan count mismatch",
        )
        require(
            summary["reference_boundary_tie_differences"]
            == sum(r["reference_boundary_tie"] for r in rows),
            "tie count mismatch",
        )
    qualified = [
        r for r in result["ivf"] if r["recall_at_k"] >= result["quality_target"]
    ]
    require(
        result["selected_ivf"] == min(qualified, key=lambda r: r["nprobe"]),
        "operating point mismatch",
    )
    return {
        "status": "passed",
        "size": count,
        "records": len(records),
        "snapshot_hash_checked": check_snapshot,
    }


def validate_archive(directory):
    root = Path(directory)
    inventory = json.loads((root / "checksums.json").read_text(encoding="utf-8"))[
        "files"
    ]
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "checksums.json"
    }
    require(actual == set(inventory), "archive inventory coverage mismatch")
    for relative, expected in inventory.items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), "unsafe archive path")
        require(sha256(path) == expected, f"archive checksum mismatch: {relative}")
    runs = [validate_run(p.parent) for p in sorted(root.glob("*/result.json"))]
    require(bool(runs), "no capacity runs archived")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    require(
        {r["size"] for r in runs} == {r["vectors"] for r in summary},
        "summary size coverage mismatch",
    )
    for row in summary:
        result = json.loads(
            (root / str(row["vectors"]) / "result.json").read_text(encoding="utf-8")
        )
        close(
            row["exact_median_round_p95_ms"],
            result["exact"]["median_round_p95_ms"],
            "summary exact latency mismatch",
        )
        close(
            row["ivf_median_round_p95_ms"],
            result["selected_ivf"]["median_round_p95_ms"],
            "summary IVF latency mismatch",
        )
        close(
            row["ivf_recall_at_k"],
            result["selected_ivf"]["recall_at_k"],
            "summary recall mismatch",
        )
    return {"status": "passed", "runs": runs, "archive_files": len(inventory)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--check-snapshot", action="store_true")
    args = parser.parse_args()
    if (Path(args.directory) / "result.json").exists():
        outcome = validate_run(args.directory, args.check_snapshot)
    else:
        if args.check_snapshot:
            parser.error("snapshot checking requires a complete local run directory")
        outcome = validate_archive(args.directory)
    print(json.dumps(outcome, indent=2))
