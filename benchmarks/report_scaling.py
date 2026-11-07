"""Build a report for the standard four-size 64D FP32 scaling experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate(input_dir, output_dir):
    evidence_root = Path(input_dir)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    summary = []
    for size in [1000, 10000, 100000, 1000000]:
        result = json.loads((evidence_root / str(size) / "result.json").read_text())
        assert result["status"] == "passed", (
            size,
            result.get("error", result["status"]),
        )
        expected = {
            "dimension": 64,
            "nlist": 32,
            "k": 10,
            "query_count": 100,
            "repeats": 3,
            "iterations": 8,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "This report requires the standard 64D/32-list/top-10/100-query/3-round experiment; other configurations retain their raw results."
            )
        results.append(result)
        selected = result["selected_ivf"]
        summary.append(
            {
                "vectors": size,
                "dimensions": result["dimension"],
                "nlist": result["nlist"],
                "insert_seconds": result["stages_seconds"]["insert"],
                "ivf_build_seconds": result["stages_seconds"]["ivf_build"],
                "total_build_seconds": result["stages_seconds"]["insert"]
                + result["stages_seconds"]["ivf_build"],
                "exact_p95_ms": result["exact"]["median_round_p95_ms"],
                "selected_nprobe": selected["nprobe"],
                "ann_recall_at_10": selected["recall_at_k"],
                "ivf_p95_ms": selected["median_round_p95_ms"],
                "scanned_fraction": selected["mean_scanned"] / size,
                "mapped_ivf_p95_ms": result["mapped_ivf"]["median_round_p95_ms"],
                "mapped_recall_at_10": result["mapped_ivf"]["recall_at_k"],
                "batch_4_worker_qps": result["batch_4_workers"]["median_qps"],
                "faiss_flat_p95_ms": result["faiss_flat"]["median_round_p95_ms"],
                "faiss_exact_agreement": result["faiss_flat"][
                    "exact_neighbor_agreement"
                ],
                "snapshot_mib": result["snapshot_bytes"] / 1048576,
                "process_peak_rss_mib": result["manifest"]["process_memory"][
                    "peak_rss_bytes"
                ]
                / 1048576,
                "mapped_postquery_rss_mib": (
                    result["memory_stages"]["mapped_after_queries"]["current_rss_bytes"]
                    / 1048576
                    if result["memory_stages"]["mapped_after_queries"].get(
                        "current_rss_bytes"
                    )
                    is not None
                    else None
                ),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    counts = [row["vectors"] for row in summary]
    axes[0, 0].loglog(
        counts, [r["exact_p95_ms"] for r in summary], "o-", label="Vortex exact"
    )
    axes[0, 0].loglog(
        counts,
        [r["ivf_p95_ms"] for r in summary],
        "s-",
        label="Vortex IVF, recall >= 0.95",
    )
    axes[0, 0].loglog(
        counts, [r["faiss_flat_p95_ms"] for r in summary], "^--", label="FAISS exact"
    )
    axes[0, 0].set(
        xlabel="Vectors",
        ylabel="Median round p95 (ms)",
        title="Single-query wall time, in-memory",
    )
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].loglog(
        counts,
        [r["total_build_seconds"] for r in summary],
        "o-",
        label="Insert + train + assign",
    )
    axes[0, 1].loglog(
        counts,
        [r["ivf_build_seconds"] for r in summary],
        "s--",
        label="IVF train + assign only",
    )
    axes[0, 1].set(
        xlabel="Vectors",
        ylabel="Seconds",
        title="Build cost; sampled training, full assignment",
    )
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].semilogx(
        counts,
        [r["process_peak_rss_mib"] for r in summary],
        "o-",
        label="Full benchmark process peak",
    )
    axes[1, 0].semilogx(
        counts, [r["snapshot_mib"] for r in summary], "s--", label="Snapshot size"
    )
    axes[1, 0].set(
        xlabel="Vectors",
        ylabel="MiB",
        title="Peak includes source arrays and FAISS baseline",
    )
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].loglog(
        counts, [r["ivf_p95_ms"] for r in summary], "o-", label="In-memory, 100 queries"
    )
    axes[1, 1].loglog(
        counts,
        [r["mapped_ivf_p95_ms"] for r in summary],
        "s--",
        label="Mapped, 8MiB cache, 20 queries",
    )
    axes[1, 1].set(
        xlabel="Vectors",
        ylabel="Median round p95 (ms)",
        title="Mapped access cost; OS cache uncontrolled",
    )
    axes[1, 1].legend(fontsize=9)
    fig.suptitle(
        "Vortex local scalability: 64D FP32, 32 lists, k=10, three rounds", fontsize=15
    )
    fig.savefig(root / "scalability.png", dpi=180, metadata={"Software": "Vortex"})
    plt.close(fig)

    table = "| Vectors | Insert + IVF build (s) | Exact p95 (ms) | IVF p95 (ms) | Recall@10 | nprobe | Process peak MiB |\n| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for row in summary:
        table += f"| {row['vectors']:,} | {row['total_build_seconds']:.3f} | {row['exact_p95_ms']:.3f} | {row['ivf_p95_ms']:.3f} | {row['ann_recall_at_10']:.3f} | {row['selected_nprobe']} | {row['process_peak_rss_mib']:.1f} |\n"
    mapped = "| Vectors | Snapshot MiB | Open mapped (s) | Mapped IVF p95 (ms) | Mapped Recall@10 | Four-worker exact batch QPS |\n| ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for row, result in zip(summary, results):
        mapped += f"| {row['vectors']:,} | {row['snapshot_mib']:.2f} | {result['stages_seconds']['open_mapped']:.3f} | {row['mapped_ivf_p95_ms']:.3f} | {row['mapped_recall_at_10']:.3f} | {row['batch_4_worker_qps']:.1f} |\n"
    text = (
        """# Vortex local scalability results

    All four requested corpus sizes were attempted in isolated local processes, without a cloud service. The unchanged installed engine was exercised with synthetic 64-dimensional FP32 vectors. This is a fixed-configuration scale experiment, not a claim about all embedding dimensions or production workloads.

    ## Configuration and method

    - Machine: Intel Core i7-1255U, 16 GB RAM, Windows x86-64; approximately 2.7 GB free before testing.
    - Engine: installed Vortex 0.1.0; GNU 16.1.0 MinGW-w64 UCRT, CMake Release, runtime AVX2.
    - Sizes: 1,000; 10,000; 100,000; 1,000,000. Identical seeded corpus prefixes and the same independent query set.
    - Metric: squared L2; dimension 64; k=10; nlist fixed at 32; eight clustering iterations; at most 10,000 training samples followed by assignment of every corpus vector.
    - Single-query latency: 100 distinct queries, three randomized-order rounds, five warm-ups per configuration. Tables use the median of the three round p95 values. These are repetitions of one dataset, not independent corpora or independent machines.
    - IVF sweep: nprobe 1, 4, 8, 16, 24, 28, 32. The reported operating point is the smallest prespecified probe reaching mean Recall@10 >= 0.95 on the measured queries. No independent tuning split was used.
    - Validation: repeated output stability; all-list IVF/exhaustive agreement; independent float64 NumPy checks on three query IDs per corpus; FAISS exact-neighbor comparison on all measured queries; saved-index round-trip checks; cache payload bound.
    - FAISS Flat: same arrays, metric and k, one thread. Its Python adapter differs from Vortex, which copies validated input and constructs Python result records. Timings were not collected in a randomized interleaving of implementations.
    - Four-worker throughput uses full batch wall time and must not be confused with single-query latency or a network service's sustained QPS.

    ## In-memory results

    """
        + table
        + """
    Peak memory is the process-wide Windows working-set high-water mark, including the generated NumPy matrix, builder, temporary reference calculations, FAISS baseline and mapped pages. It is not engine-only RAM and does not include the operating system's entire file cache. Each size ran in a separate process to avoid carrying the preceding process's peak forward.

    ## Persistence and mapped access

    """
        + mapped
        + """
    Mapped IVF uses the same selected nprobe, but a 20-query subset and three repetitions. The original builder/source matrix are released before reopening. The explicit cache has an 8 MiB payload limit. The operating-system page cache is not flushed, and files were recently written, so these are not physical cold-SSD measurements. A collection larger than this explicit cache is not necessarily larger than physical RAM.

    ## Interpretation

    - Completing a one-million-vector test establishes that this particular 64D configuration works locally. It does not establish billion-vector capacity, multi-node scaling, or real-model RAG quality.
    - High-recall IVF can be slower than exact scans here. Independent random vectors have weak cluster structure, the fixed 32-list setup needs many probes, and list traversal has less favorable memory access. Lower-probe speed should always be shown with its lower recall.
    - No per-size search for the optimal cluster count was performed. Index tuning and cache/layout changes may improve performance, but improvements are not assumed in these measurements.
    - One million 768D FP32 vectors would require 3.072 GB for just the raw vector components. This 64D result does not validate that larger-memory configuration on the currently busy machine.
    - Cloud infrastructure was not required. More available RAM or a cloud instance would help test higher dimensions, collections beyond physical RAM, and sustained isolated service load.

    ![Measured local scalability](scalability.png)

    ## Reproduction and evidence

    Validate the published archive and regenerate its tables/chart without repeating the measurements:

    ```bash
    python benchmarks/validate_scaling.py
    python benchmarks/report_scaling.py --input-dir benchmarks/reference/scaling --output-dir results/scaling-report
    ```

    To collect a new dataset in isolated processes, run from the repository root using an environment containing the installed native package, NumPy, FAISS and Matplotlib:

    ```bash
    python benchmarks/run_scaling.py --output results/scaling-new
    python benchmarks/report_scaling.py --input-dir results/scaling-new --output-dir results/scaling-new/report
    ```

    The JSON result in each size directory records build stages, per-configuration percentiles, recall, memory stages, dataset/source fingerprints, package versions, source revision and runner hash. The corresponding queries.jsonl preserves all timed query results. Snapshot files are retained locally. Summary CSV/JSON and plots are derived from these records. The archived runner_snapshot.txt preserves the exact historical runner bytes; the maintained runner includes CLI validation and evidence-preservation improvements. Working-tree dirty flags in the original manifests include incidental untracked local files and do not represent native-engine changes.

    Large snapshots remain local under the ignored /results/ directory. The published JSON/query logs preserve the original measurements. The native engine was not modified for these tests; subsequent changes package the harness, evidence and reports.
    """
    )
    text = "\n".join(line.removeprefix("    ") for line in text.splitlines()) + "\n"
    hardware = results[0]["manifest"]["hardware"]
    configuration = results[0]["manifest"]["configuration"]
    text = text.replace(
        "- Machine: Intel Core i7-1255U, 16 GB RAM, Windows x86-64; approximately 2.7 GB free before testing.",
        f"- Recorded environment: {hardware['processor']}; {hardware['platform']}; {hardware['machine']}.",
    ).replace(
        "- Engine: installed Vortex 0.1.0; GNU 16.1.0 MinGW-w64 UCRT, CMake Release, runtime AVX2.",
        f"- Engine build: {configuration['build']}; runtime AVX2 available: {configuration['avx2_available']}.",
    )
    (root / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(table)
    print(mapped)
    print("Saved report.md, summary.csv, summary.json and scalability.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="benchmarks/reference/scaling")
    parser.add_argument("--output-dir", default="results/scaling-report")
    args = parser.parse_args()
    generate(args.input_dir, args.output_dir)
