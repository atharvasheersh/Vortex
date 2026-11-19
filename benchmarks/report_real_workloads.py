"""Validate and summarize real-workload raw traces. Plots are optional."""

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from measure_real_workloads import MODES, K, document_scores, summarize
from real_workload_data import read_json
from run_performance import digest, write_json
from study_real_workloads import CONTINUATION_PHASES, TIERS, TUNED, freeze

BOOTSTRAPS = 1000
LABELS = {
    "cached": "Vortex INT8 cached",
    "direct1": "Vortex INT8 direct ×1",
    "direct2": "Vortex INT8 direct ×2",
    "fp32": "Vortex FP32 IVF",
    "faiss": "FAISS IVF-PQ + FP32",
}
MARKERS = ("o", "s", "D", "^", "P")
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def same_number(actual, expected):
    if expected is None or actual is None:
        return actual is expected
    return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def document_summary(measured, labels, expected_ids):
    require(
        sorted(row["query_id"] for row in measured) == sorted(expected_ids),
        "unexpected document query IDs",
    )
    for row in measured:
        ids = row["document_ids"]
        require(
            len(set(ids)) == len(ids) <= K and set(ids) <= labels["corpus_docs"],
            "invalid document hits",
        )
        scores = document_scores(ids, labels["official_qrels"][row["query_id"]])
        require(
            all(same_number(row[k], v) for k, v in scores.items()),
            "document metric mismatch",
        )
    return {
        key: float(np.mean([row[key] for row in measured]))
        for key in ("ndcg_at_10", "document_recall_at_10")
    }


def validate_case(execution, root, name, case, ids, truth, labels):
    folder = root / name
    manifest = read_json(folder / "manifest.json")
    summary = read_json(folder / "summary.json")
    hits_path = folder / "queries.jsonl"
    logged = rows(hits_path)
    trace = sorted(logged, key=lambda row: row["position"])
    for field in ("dataset", "build_sha256", "worker_sha256"):
        key = {
            "build_sha256": "dataset_build_sha256",
            "worker_sha256": "source_sha256",
        }.get(field, field)
        require(manifest[key] == execution[field], f"{name}: mismatched {key}")
    for field in ("mode", "probes", "count", "offset", "round", "clients", "phase"):
        require(manifest[field] == case[field], f"{name}: mismatched {field}")
    require(
        manifest["status"] == "completed",
        f"{name}: incomplete data for a completed case",
    )
    capped = case["phase"] != "os-managed"
    policy = manifest["memory_policy"]
    require(policy["enabled"] == capped, f"{name}: different working-set policy")
    if capped:
        require(
            policy["maximum_bytes"] == 1536 * 1024**2
            and policy["flags"] & 4
            and manifest["query_peak_memory"]["peak_rss_bytes"]
            <= policy["maximum_bytes"] + 65536,
            f"{name}: working-set bound failed",
        )
    expected_positions = list(range(case["offset"], case["offset"] + case["count"]))
    require(
        [row["position"] for row in trace] == expected_positions,
        f"{name}: invalid query coverage",
    )
    for position, row in zip(expected_positions, trace):
        found = {identifier for identifier, _ in row["hits"]}
        require(
            row["query_id"] == ids[position] and len(found) == len(row["hits"]) == K,
            f"{name}: bad IDs",
        )
        require(
            all(math.isfinite(score) for _, score in row["hits"]),
            f"{name}: non-finite result",
        )
        require(
            0 < row["service_ms"] <= row["wall_ms"], f"{name}: invalid service time"
        )
        require(
            same_number(
                row["wall_ms"], (row["completed_ns"] - row["started_ns"]) / 1e6
            ),
            f"{name}: elapsed sample differs from timestamps",
        )
        expected = len(found & set(truth[position])) / K
        require(
            same_number(row["recall_at_10"], expected),
            f"{name}: recall differs from reference",
        )
    duration = (
        max(row["completed_ns"] for row in trace)
        - min(row["started_ns"] for row in trace)
    ) / 1e9
    require(
        same_number(manifest["query_duration"], duration), f"{name}: duration mismatch"
    )
    calculated = summarize(logged, duration, manifest["query_cpu_seconds"])
    require(
        all(same_number(summary[k], val) for k, val in calculated.items()),
        f"{name}: summary mismatch",
    )
    if manifest.get("batch", {}).get("status") == "completed":
        batch = manifest["batch"]
        require(
            same_number(batch["qps"], batch["queries"] / batch["seconds"]),
            f"{name}: batch mismatch",
        )
    document = None
    ordered = sorted(trace, key=lambda row: row["started_ns"])
    half = len(ordered) // 2
    restart_diagnostics = {
        key: float(np.percentile([row["wall_ms"] for row in part], 95))
        for key, part in (
            ("first_half_p95_ms", ordered[:half]),
            ("last_half_p95_ms", ordered[half:]),
        )
        if part
    }
    evidence = {
        "query_trace_sha256": digest(hits_path),
        "manifest_sha256": digest(folder / "manifest.json"),
    }
    relevance = folder / "relevance.jsonl"
    if labels is not None and case["phase"] == "primary" and case["round"] == 0:
        qids = labels["tuning_ids"] + labels["test_ids"]
        document = document_summary(
            rows(relevance), labels, [qids[ids[p]] for p in expected_positions]
        )
        evidence["relevance_trace_sha256"] = digest(relevance)
    return {
        **case,
        **calculated,
        **restart_diagnostics,
        "build_id": manifest["dataset_build_sha256"],
        "native_sha256": manifest["native_sha256"],
        "active_kernel": manifest["active_kernel"],
        "index_bytes": manifest["index_bytes"],
        "memory": manifest["query_peak_memory"],
        "open_ms": manifest["open_ms"],
        "first_lookup_ms": manifest["first_lookup_ms"],
        "io_start": manifest["io_start"],
        "io_end": manifest["io_end"],
        "paging_start": manifest["paging_start"],
        "paging_end": manifest["paging_end"],
        "batch": manifest.get("batch"),
        "document_relevance": document,
        "power_start": manifest["power_start"],
        "power_end": manifest["power_end"],
        **evidence,
    }, trace


def bootstrap_p95(latencies, sample):
    return np.percentile(latencies[:, sample], 95, axis=(0, 2))


def interval(values):
    return np.percentile(values, (2.5, 97.5)).tolist()


def aggregate(primary, traces):
    results = {}
    for mode, rounds in primary.items():
        actual = [traces[name] for name, _ in sorted(rounds)]
        first = actual[0]
        require(
            all(
                [r["query_id"] for r in round_rows] == [r["query_id"] for r in first]
                for round_rows in actual
            ),
            f"{mode}: repeated rounds used different queries",
        )
        latencies = np.array([[r["wall_ms"] for r in repeat] for repeat in actual])
        sample = np.random.default_rng(88681).integers(
            0, len(first), size=(BOOTSTRAPS, len(first))
        )
        rounds_summary = [case for _, case in sorted(rounds)]
        results[mode] = {
            "distinct_queries": len(first),
            "rounds": len(rounds),
            "probes": rounds_summary[0]["probes"],
            "recall_at_10": float(
                np.mean([case["recall_at_10"] for case in rounds_summary])
            ),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p95_ci_ms": interval(bootstrap_p95(latencies, sample)),
            "round_p95_ms": [case["p95_ms"] for case in rounds_summary],
            "qps": float(np.mean([case["qps"] for case in rounds_summary])),
            "cpu_ms_per_query": float(
                np.mean([case["cpu_ms_per_query"] for case in rounds_summary])
            ),
            "peak_rss_bytes": max(
                case["memory"]["peak_rss_bytes"] for case in rounds_summary
            ),
            "median_final_rss_bytes": int(
                np.median(
                    [case["memory"]["current_rss_bytes"] for case in rounds_summary]
                )
            ),
            "index_bytes": rounds_summary[0]["index_bytes"],
            "median_open_ms": float(
                np.median([case["open_ms"] for case in rounds_summary])
            ),
            "median_first_lookup_ms": float(
                np.median([case["first_lookup_ms"] for case in rounds_summary])
            ),
            "batch": rounds_summary[0]["batch"],
            "document_relevance": rounds_summary[0]["document_relevance"],
        }
    return results


def pair_speedups(primary, traces):
    paired = {}
    baseline = {case["round"]: (name, case) for name, case in primary.get("cached", [])}
    for mode in ("direct1", "direct2"):
        comparison = {
            case["round"]: (name, case) for name, case in primary.get(mode, [])
        }
        shared = sorted(baseline.keys() & comparison.keys())
        if not shared:
            continue
        same_results, left, right = True, [], []
        for round_id in shared:
            anchor = traces[baseline[round_id][0]]
            candidate = traces[comparison[round_id][0]]
            require(
                [r["query_id"] for r in anchor] == [r["query_id"] for r in candidate],
                "direct and cached queries differ",
            )
            same_results &= all(
                a["hits"] == b["hits"] for a, b in zip(anchor, candidate)
            )
            left.append([a["wall_ms"] for a in anchor])
            right.append([b["wall_ms"] for b in candidate])
        sample = np.random.default_rng(88681).integers(
            0, len(left[0]), size=(BOOTSTRAPS, len(left[0]))
        )
        paired[mode] = {
            "identical_ordered_ids_and_scores": same_results,
            "paired_rounds": len(shared),
            "factor": float(np.percentile(left, 95) / np.percentile(right, 95)),
            "factor_ci": interval(
                bootstrap_p95(np.array(left), sample)
                / bootstrap_p95(np.array(right), sample)
            ),
        }
    return paired


def incomplete_case(root, name, case):
    result = dict(case)
    transcript = root / f"{name}.log"
    if transcript.exists():
        result["transcript_sha256"] = digest(transcript)
    partial = root / name / "queries.jsonl"
    if partial.exists():
        result["partial_trace_sha256"] = digest(partial)
        with partial.open("rb") as file:
            result["partial_rows"] = sum(line.endswith(b"\n") for line in file)
    manifest = root / name / "manifest.json"
    if manifest.exists():
        result["partial_manifest_sha256"] = digest(manifest)
        try:
            details = read_json(manifest)
            result["incomplete_worker_status"] = details["status"]
            result["worker_error"] = details.get("error")
            result["attempt_seconds_budget"] = details["timeout_seconds"]
        except (json.JSONDecodeError, UnicodeError):
            result["incomplete_worker_status"] = "truncated manifest"
    return result


def validate_study(data, measured, continuation=None):
    build = read_json(data / "build.json")
    execution = read_json(measured / "execution.json")
    require(
        execution["build_sha256"] == digest(data / "build.json"),
        "dataset manifest fingerprint changed",
    )
    for name, artifact in build["artifacts"].items():
        path = data / name
        require(
            path.stat().st_size == artifact["bytes"]
            and digest(path) == artifact["sha256"],
            f"invalid dataset artifact {name}",
        )
    ids = np.load(data / "query_ids.npy", allow_pickle=False)
    truth = np.load(data / "reference_ids.npy", allow_pickle=False)
    queries = np.load(data / "queries.npy", allow_pickle=False)
    tune, holdout = build["tuning_queries"], build["holdout_queries"]
    require(
        len(ids) == len(set(ids)) == tune + holdout
        and len({q.tobytes() for q in queries}) == len(ids),
        "query IDs or vectors overlap across splits",
    )
    require(
        truth.shape == (len(ids), K) and all(len(set(row)) == K for row in truth),
        "invalid top ten reference",
    )
    labels = None
    if build["dataset"] == "scifact":
        labels = read_json(data / "relevance.json")
        labels["corpus_docs"] = set(labels["documents"].values())
    followup, followup_info = {}, None
    if continuation:
        followup_path = continuation / "execution.json"
        record = read_json(followup_path)
        require(
            record["continues_sha256"] == digest(measured / "execution.json"),
            "wrong continuation parent",
        )
        for field in (
            "dataset",
            "build_sha256",
            "protocol_sha256",
            "worker_sha256",
            "condition_seconds",
        ):
            require(record[field] == execution[field], f"continuation changed {field}")
        followup = record["cases"]
        require(
            followup.keys() <= execution["cases"].keys(),
            "continuation added unplanned cases",
        )
        followup_info = {
            key: record[key]
            for key in ("seconds_elapsed", "seconds_budget", "driver_sha256")
        }
        followup_info["execution_sha256"] = digest(followup_path)
    cases, traces = {}, {}
    for name, original in execution["cases"].items():
        root, case = measured, original
        if name in followup:
            root, case = continuation, followup[name]
            require(
                original["status"] != "completed"
                and original["phase"] in CONTINUATION_PHASES,
                f"{name}: continuation repeated an ineligible condition",
            )
            for field in (
                "phase",
                "mode",
                "probes",
                "offset",
                "count",
                "round",
                "clients",
            ):
                require(
                    case[field] == original[field],
                    f"{name}: continuation changed {field}",
                )
        if case["status"] != "completed":
            cases[name] = incomplete_case(root, name, case)
        else:
            expected_offset = 0 if case["phase"] == "tuning" else tune
            expected_count = min(holdout, 256) if case["phase"] == "load" else holdout
            if case["phase"] == "tuning":
                expected_count = tune
            require(
                (case["offset"], case["count"]) == (expected_offset, expected_count),
                f"{name}: condition deviates from frozen query split",
            )
            cases[name], traces[name] = validate_case(
                execution, root, name, case, ids, truth, labels
            )
        if name in followup:
            cases[name]["original_attempt"] = incomplete_case(measured, name, original)
    tune_points = {mode: [] for mode in TUNED}
    primary = defaultdict(list)
    for name, case in cases.items():
        if case["status"] != "completed":
            continue
        if case["phase"] == "tuning":
            tune_points[case["mode"]].append(
                {key: case[key] for key in ("probes", "p95_ms", "recall_at_10")}
            )
        if case["phase"] == "primary":
            primary[case["mode"]].append((name, case))
    selection = read_json(measured / "selection.json")
    require(
        freeze(tune_points) == selection,
        "selection was not reproduced from tuning cases",
    )
    for case in cases.values():
        if case["phase"] == "tuning":
            continue
        mode = case["mode"]
        anchor = selection["cached" if mode.startswith("direct") else mode]
        chosen = (
            {anchor["primary"]["probes"]}
            if case["phase"] != "tier"
            else {point["probes"] for point in anchor["tiers"].values() if point}
        )
        require(case["probes"] in chosen, "unfrozen held-out operating point")
    require(
        len(
            {
                case["native_sha256"]
                for case in cases.values()
                if case["status"] == "completed"
            }
        )
        <= 1,
        "mixed native binaries",
    )
    aggregates = aggregate(primary, traces)
    paired = pair_speedups(primary, traces)
    alternatives = defaultdict(lambda: defaultdict(list))
    matched = {}
    for name, case in cases.items():
        if case["status"] == "completed" and case["phase"] == "tier":
            alternatives[case["probes"]][case["mode"]].append((name, case))
        if case["status"] != "completed" or case["mode"] not in ("direct1", "direct2"):
            continue
        # The runner uses the same query range for each peer at a given load and stage.
        peer = name.replace(f"-{case['mode']}-", "-cached-", 1)
        if peer in traces:
            left, right = traces[peer], traces[name]
            require(
                [row["query_id"] for row in left] == [row["query_id"] for row in right],
                f"{name}: peer ran a different query split",
            )
            matched[name] = all(a["hits"] == b["hits"] for a, b in zip(left, right))
    tiers = {}
    latency_points = []
    for mode in MODES:
        anchor = selection["cached" if mode.startswith("direct") else mode]
        points = {}
        for tier in TIERS:
            tuned = anchor["tiers"][str(tier)]
            eligible = [
                case
                for case in cases.values()
                if tuned
                and case["phase"] in {"primary", "tier"}
                and case["round"] == 0
                and case["mode"] == mode
                and case["probes"] == tuned["probes"]
                and case["status"] == "completed"
            ]
            if eligible:
                point = eligible[0]
                points[tier] = {
                    key: point[key] for key in ("probes", "recall_at_10", "p95_ms")
                }
                points[tier]["met_holdout"] = point["recall_at_10"] >= tier / 100
            else:
                points[tier] = None
        tiers[mode] = points
        latency_points.extend(
            {
                (mode, p["probes"], p["p95_ms"], p["recall_at_10"])
                for p in points.values()
                if p
            }
        )
    comparable = []
    for left, right in combinations(sorted(latency_points), 2):
        if (
            left[0] != right[0]
            and max(left[2], right[2]) / min(left[2], right[2]) <= 1.20
        ):
            comparable.append([left, right])
    return {
        "build": build,
        "execution_sha256": digest(measured / "execution.json"),
        "protocol_sha256": execution["protocol_sha256"],
        "git_head": execution["git_head"],
        "seconds_elapsed": execution["seconds_elapsed"],
        "planned_rounds": execution["planned_rounds"],
        "continuation": followup_info,
        "environment": {
            key: execution[key]
            for key in (
                "platform",
                "cpu",
                "physical_cores",
                "logical_cores",
                "physical_memory",
                "initial_available_memory",
                "initial_free_disk",
                "seconds_budget",
                "condition_seconds",
                "driver_sha256",
                "worker_sha256",
            )
        },
        "selection": selection,
        "cases": cases,
        "primary": aggregates,
        "paired": paired,
        "tier_pairs": {
            probes: match
            for probes, points in alternatives.items()
            if (match := pair_speedups(points, traces))
        },
        "ordered_result_match": matched,
        "recall_tiers": tiers,
        "at_comparable_latency": comparable,
    }


def plots(studies, output):
    import matplotlib.pyplot as plt

    for metric, filename in (
        ("recall", "real-workloads-recall.png"),
        ("load", "real-workloads-load.png"),
    ):
        fig, axes = plt.subplots(
            1, len(studies), figsize=(5.1 * len(studies), 4.7), squeeze=False
        )
        for ax, (dataset, study) in zip(axes[0], studies.items()):
            for (mode, label), marker, color in zip(LABELS.items(), MARKERS, COLORS):
                if metric == "recall":
                    points = {
                        (p["recall_at_10"] * 100, p["p95_ms"])
                        for p in study["recall_tiers"][mode].values()
                        if p
                    }
                    data = sorted(points)
                else:
                    selected = sorted(
                        (case["clients"], case["qps"], case["p95_ms"])
                        for case in study["cases"].values()
                        if case["status"] == "completed"
                        and case["phase"] == "load"
                        and case["mode"] == mode
                    )
                    data = [(qps, p95) for _, qps, p95 in selected]
                if data:
                    ax.plot(
                        *zip(*data),
                        marker=marker,
                        color=color,
                        markersize=5,
                        label=label,
                    )
                    if metric == "load":
                        for clients, qps, p95 in selected:
                            ax.annotate(
                                str(clients),
                                (qps, p95),
                                xytext=(3, 4),
                                textcoords="offset points",
                                fontsize=7,
                            )
            ax.set(
                title=f"{dataset.upper()} · {study['build']['count']:,} vectors",
                yscale="log",
                ylabel="p95 latency (ms)",
                xlabel="Held-out Recall@10 (%)"
                if metric == "recall"
                else "Achieved queries/s (points labeled by clients)",
            )
            ax.grid(alpha=0.25)
        handles, legend = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend,
            loc="lower center",
            ncol=min(len(handles), 5),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(output / filename, dpi=155)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--study",
        action="append",
        required=True,
        nargs=2,
        type=Path,
        metavar=("BUILD", "TRACES"),
    )
    parser.add_argument("--plots", type=Path)
    parser.add_argument(
        "--continuation",
        action="append",
        nargs=2,
        default=[],
        metavar=("DATASET", "TRACES"),
    )
    parser.add_argument(
        "--expect", type=Path, help="verify an existing published summary"
    )
    args = parser.parse_args()
    continuation = {name: Path(path) for name, path in args.continuation}
    require(len(continuation) == len(args.continuation), "duplicate continuations")
    studies = {}
    for build, results in args.study:
        name = read_json(build / "build.json")["dataset"]
        require(name not in studies, f"duplicate study {name}")
        verified = validate_study(build, results, continuation.pop(name, None))
        studies[name] = verified
        print(
            f"Validated {name}: {len(verified['cases'])} recorded conditions",
            flush=True,
        )
    require(not continuation, "continuation does not match any study")
    signatures = {
        (study["git_head"], study["protocol_sha256"], case["native_sha256"])
        for study in studies.values()
        for case in study["cases"].values()
        if case["status"] == "completed"
    }
    require(
        len(signatures) <= 1, "studies used different engine revisions or protocols"
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"bootstrap_repetitions": BOOTSTRAPS, "studies": studies}
    if args.expect:
        require(
            json.loads(json.dumps(report)) == read_json(args.expect),
            "published summary mismatch",
        )
    write_json(args.output, report)
    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)
        plots(studies, args.plots)


if __name__ == "__main__":
    main()
