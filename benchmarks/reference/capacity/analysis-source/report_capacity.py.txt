"""Export compact capacity evidence and tables without copying large snapshots."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from validate_capacity import sha256, validate_run


def report(directories, output, archive=None, resource_attempts=()):
    roots = sorted(
        map(Path, directories),
        key=lambda p: json.loads((p / "result.json").read_text())["size"],
    )
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("report output must be empty")
    if archive is not None and Path(archive).exists():
        raise FileExistsError("archive already exists")
    attempts = []
    for source in map(Path, resource_attempts):
        source = source / "result.json" if source.is_dir() else source
        attempt = json.loads(source.read_text(encoding="utf-8"))
        if attempt["status"] not in {"resource_limited", "failed"}:
            raise ValueError("resource-attempt evidence must record an incomplete run")
        attempts.append(source)
    results, validations = [], []
    for root in roots:
        validations.append(validate_run(root))
        results.append(json.loads((root / "result.json").read_text(encoding="utf-8")))
    if len({r["size"] for r in results}) != len(results):
        raise ValueError("duplicate corpus sizes")
    fields = [
        "dimension",
        "k",
        "nlist",
        "query_count",
        "repeats",
        "chunk_rows",
        "cache_budget",
        "sources_sha256",
        "native_module_sha256",
        "queries_sha256",
        "memory_policy",
    ]
    if any(any(r[f] != results[0][f] for f in fields) for r in results[1:]):
        raise ValueError("runs do not share a matched protocol and executable")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for result, validated in zip(results, validations):
        chosen = result["selected_ivf"]
        rows.append(
            {
                "vectors": result["size"],
                "construction_seconds": sum(
                    result["construction"]["stages_seconds"].values()
                ),
                "open_seconds": result["open_seconds"],
                "snapshot_bytes": result["construction"]["snapshot_bytes"],
                "exact_median_round_p95_ms": result["exact"]["median_round_p95_ms"],
                "ivf_median_round_p95_ms": chosen["median_round_p95_ms"],
                "ivf_recall_at_k": chosen["recall_at_k"],
                "nprobe": chosen["nprobe"],
                "peak_working_set_bytes": result["memory_stages"]["final"][
                    "peak_rss_bytes"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "timing_gap_count": len(validated["timing_gaps"]),
                "resident_peak_excess_bytes": validated["resident_peak_excess_bytes"],
            }
        )
    with (destination / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Single-collection mapped capacity",
        "",
        f"Synthetic {results[0]['dimension']}D FP32 squared L2, top {results[0]['k']}, {results[0]['nlist']} lists; {results[0]['query_count']} queries, {results[0]['repeats']} rounds, five warmups per configuration, {results[0]['cache_budget'] / 1024**2:g} MiB application cache. Latencies are median round p95. A bounded snapshot writer constructs each collection; the native mutable builder is unchanged. OS page cache was not flushed. This is sequential local capacity, not a concurrent service or real RAG-quality experiment.",
        "",
        f"| Vectors | Construction s | Exact p95 ms | IVF p95 ms | Recall@{results[0]['k']} | nprobe | Peak working set MiB | Snapshot GiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if results[0].get("memory_policy", {}).get("enabled"):
        lines.insert(
            4,
            f"Requested Windows hard process working-set maximum: {results[0]['memory_policy']['maximum_bytes'] / 1024**2:g} MiB, applied equally to both sizes. Paging/replacement costs are included in query latency; this does not limit commit or the OS file cache. Actual observed peaks are reported below.",
        )
        lines.insert(5, "")
    for row in rows:
        lines.append(
            f"| {row['vectors']:,} | {row['construction_seconds']:.3f} | {row['exact_median_round_p95_ms']:.3f} | {row['ivf_median_round_p95_ms']:.3f} | {row['ivf_recall_at_k']:.3f} | {row['nprobe']} | {row['peak_working_set_bytes'] / 1024**2:.1f} | {row['snapshot_bytes'] / 1024**3:.3f} |"
        )
    lines += [
        "",
        "The IVF operating point is the smallest tested probe with mean recall >=0.95 on the measured queries. Historical in-memory measurements use different training-row selection, query seed/count and storage paths; keep those curves separate. Construction includes disk generation, sampled native training, full-corpus native assignment and snapshot finalization. Reference generation, snapshot hashing and query measurement are outside construction time.",
        "",
        "## Probe sweep",
        "",
        f"| Vectors | nprobe | p95 ms | Recall@{results[0]['k']} | Mean vectors scanned |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for result, checked in zip(results, validations):
        for value in result["ivf"]:
            marker = (
                " †"
                if any(
                    gap["nprobe"] == value["nprobe"] for gap in checked["timing_gaps"]
                )
                else ""
            )
            lines.append(
                f"| {result['size']:,} | {value['nprobe']} | {value['median_round_p95_ms']:.3f}{marker} | {value['recall_at_k']:.3f} | {value['mean_scanned']:,.1f} |"
            )
    lines += [
        "",
        "Every timed exact query scans N vectors. Blocked FAISS covers all queries over the complete corpus; direct float64 NumPy checks three queries. Full-list IVF matches native exact IDs and scores. Repeated runs use one corpus and one machine; they do not provide independent dataset or hardware replication. Working-set peaks include mapped pages and metadata and are not private heap size or a limit imposed by the cache.",
        "",
    ]
    lines += [
        "## Analysis diagnostics",
        "",
        "Long wall/CPU gaps are flagged when wall time exceeds 60 seconds and ten times process CPU time. This is a post-collection diagnostic; no sample is removed or replaced. Conditions marked † contain such a gap and should not be treated as clean latency measurements.",
        "",
    ]
    for checked in validations:
        lines.append(
            f"- {checked['size']:,} vectors: {len(checked['timing_gaps'])} flagged gap(s); observed resident peak exceeds requested maximum by {checked['resident_peak_excess_bytes']:,} bytes."
        )
        for gap in checked["timing_gaps"]:
            lines.append(
                f"  - Probe {gap['nprobe']}, repeat {gap['repeat']}, query {gap['query_id']}: {gap['wall_ms'] / 1000:.3f} s wall versus {gap['process_cpu_seconds']:.3f} s CPU. Cause is unconfirmed; suspension or long descheduling is possible."
            )
    lines += [
        "",
        "Windows confirmed the hard-limit flags, but peak accounting exceeded the requested resident maximum by a few pages. The maintained validator was corrected after collection to allow at most 64 KiB of accounting excess and still reports the actual excess. Frozen collection sources and raw results are unchanged; maintained analysis sources are archived separately. The free-memory floor and all correctness checks remain unchanged.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (destination / "timing_diagnostics.json").write_text(
        json.dumps({"runs": validations}, indent=2) + "\n", encoding="utf-8"
    )
    if archive is not None:
        archive = Path(archive)
        archive.mkdir(parents=True)
        for root, result in zip(roots, results):
            target = archive / str(result["size"])
            target.mkdir()
            for name in [
                "result.json",
                "reference.json",
                "queries.npy",
                "measurements.jsonl",
            ]:
                shutil.copyfile(root / name, target / name)
            shutil.copytree(root / "executed-source", target / "executed-source")
            validate_run(target)
        for name in [
            "summary.json",
            "summary.csv",
            "report.md",
            "timing_diagnostics.json",
        ]:
            shutil.copyfile(destination / name, archive / name)
        for number, source in enumerate(attempts, 1):
            target = archive / "attempts" / str(number)
            target.mkdir(parents=True)
            shutil.copyfile(source, target / "result.json")
            for name in ["reference.json", "queries.npy", "measurements.jsonl"]:
                if (source.parent / name).exists():
                    shutil.copyfile(source.parent / name, target / name)
            if (source.parent / "executed-source").exists():
                shutil.copytree(
                    source.parent / "executed-source", target / "executed-source"
                )
        analysis = archive / "analysis-source"
        analysis.mkdir()
        for name in [
            "report_capacity.py",
            "validate_capacity.py",
            "validate_scaling.py",
        ]:
            shutil.copyfile(
                Path(__file__).resolve().parent / name, analysis / (name + ".txt")
            )
        files = {
            p.relative_to(archive).as_posix(): sha256(p)
            for p in sorted(archive.rglob("*"))
            if p.is_file()
        }
        (archive / "checksums.json").write_text(
            json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive-dir")
    parser.add_argument(
        "--resource-attempt",
        nargs="*",
        default=[],
        help="preserve incomplete result JSON files and compact adjacent evidence in the archive",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            report(args.input, args.output, args.archive_dir, args.resource_attempt),
            indent=2,
        )
    )
