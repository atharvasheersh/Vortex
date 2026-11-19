"""Audit phase 2 records and plot the measured recall/latency frontier."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from run_performance import digest, write_json
from run_retrieval_phase2 import summarize
from study_parallel_int8 import summarize_native
from study_retrieval_phase2 import eligible, paired, read


def audit_case(path, reference, truth):
    manifest = read(path / "manifest.json")
    rows_path = path / "queries.jsonl"
    rows = [
        json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
    ]
    offset = manifest["query_offset"]
    count = manifest["query_count"]
    if (
        manifest["status"] != "completed"
        or len(rows) != count
        or {r["query_id"] for r in rows} != set(range(offset, offset + count))
        or any(r["round_id"] != manifest["round"] for r in rows)
    ):
        raise ValueError(f"incomplete or duplicate query coverage in {path.name}")
    for key in ["snapshot_sha256", "queries_sha256"]:
        if manifest[key] != reference[key]:
            raise ValueError(f"reference provenance mismatch in {path.name}")
    if manifest["reference_ids_sha256"] != reference["ids_sha256"]:
        raise ValueError(f"reference ID provenance mismatch in {path.name}")
    if (
        manifest["memory_policy"]["maximum_bytes"] != 1536 * 1024**2
        or not manifest["memory_policy"]["enabled"]
        or manifest["threads"] != 2
        or manifest["query_peak_memory"]["peak_rss_bytes"] > 1536 * 1024**2 + 65536
    ):
        raise ValueError(f"query resource policy mismatch in {path.name}")
    for row in rows:
        hits = row["hits"]
        ids = {identifier for identifier, _ in hits}
        recall = len(ids & set(truth[row["query_id"]].tolist())) / 10
        if (
            len(ids) != 10
            or len(hits) != 10
            or row["recall_at_10"] != recall
            or any(
                not 0 <= identifier < 2**64 or not math.isfinite(score)
                for identifier, score in hits
            )
            or hits != sorted(hits, key=lambda hit: (hit[1], hit[0]))
            or not math.isfinite(row["wall_ms"])
            or row["wall_ms"] <= 0
            or row["timing_gap"]
            != (row["wall_ms"] > 60000 and row["wall_ms"] > 10 * max(row["cpu_ms"], 1))
        ):
            raise ValueError(f"query result or diagnostic mismatch in {path.name}")
    summary = summarize(rows)
    if summary != read(path / "summary.json"):
        raise ValueError(f"summary differs from measurements in {path.name}")
    return {
        "calls": len(rows),
        "sha256": digest(rows_path),
        "manifest": manifest,
        "summary": summary,
    }


def audit_stage(stage):
    execution = read(stage / "execution.json")
    reference_path = stage / "reference"
    reference = read(reference_path / "reference.json")
    for name, key in [
        ("queries.npy", "queries_sha256"),
        ("ids.npy", "ids_sha256"),
        ("scores.npy", "scores_sha256"),
    ]:
        if digest(reference_path / name) != reference[key]:
            raise ValueError(f"reference file checksum mismatch in {stage.name}")
    queries = np.load(reference_path / "queries.npy", allow_pickle=False)
    if len(np.unique(queries, axis=0)) != len(queries):
        raise ValueError(f"duplicate reference queries in {stage.name}")
    truth = np.load(reference_path / "ids.npy", allow_pickle=False)
    cases = {
        f.parent.name: audit_case(f.parent, reference, truth)
        for f in sorted(stage.glob("*/summary.json"))
    }
    tuning = read(stage / "tuning.json") if (stage / "tuning.json").exists() else None
    if tuning and tuning["int8"]:
        preset = tuning["int8"][0]
        for trial in tuning["int8"]:
            path = f"tune-int8-{trial['probes']}"
            if cases[path]["summary"] != {
                k: v for k, v in trial.items() if k != "probes"
            }:
                raise ValueError("control tuning summary differs from measurements")
        for trial in tuning["graph"]:
            path = f"tune-d{trial['degree']}-l{trial['complexity']}-b{trial['beam']}"
            if cases[path]["summary"] != {
                k: v
                for k, v in trial.items()
                if k not in {"degree", "complexity", "beam"}
            }:
                raise ValueError("graph tuning summary differs from measurements")
        eligible_graphs = [g for g in tuning["graph"] if eligible(preset, g)]
        expected = (
            min(eligible_graphs, key=lambda g: g["p95_ms"]) if eligible_graphs else None
        )
        if tuning["selected"] != expected:
            raise ValueError("tuning selection does not match the quality rule")
    outcome = None
    if (stage / "heldout.json").exists():
        rounds = execution["rounds"]
        tuning_count = execution["tuning_queries"]
        for name, case in cases.items():
            offset = case["manifest"]["query_offset"]
            if offset != (tuning_count if name.startswith("heldout-") else 0):
                raise ValueError("tuning and held-out split boundary changed")
        outcome = paired(
            [stage / f"heldout-int8-r{i}" for i in range(rounds)],
            [stage / f"heldout-graph-r{i}" for i in range(rounds)],
        )
        if outcome != read(stage / "heldout.json"):
            raise ValueError("paired metrics do not reproduce")
    build_records = {
        p.parent.name: read(p) for p in sorted(stage.glob("graph-*/build.json"))
    }
    # The shared reference generator's default split labels belong to phase 1.
    # Phase 2 receives its split from the recorded execution and per-case plan.
    reference = {
        k: v for k, v in reference.items() if k not in {"tuning_count", "heldout_count"}
    }
    reference["query_count"] = len(queries)
    return {
        "execution": execution,
        "reference": reference,
        "tuning": tuning,
        "heldout": outcome,
        "builds": build_records,
        "calls": sum(c["calls"] for c in cases.values()),
        "case_evidence": {
            name: {"calls": c["calls"], "sha256": c["sha256"]}
            for name, c in cases.items()
        },
    }


def plot(stages, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [
        (name, stage)
        for name, stage in stages.items()
        if stage["tuning"] and stage["tuning"]["graph"]
    ]
    if not data:
        raise ValueError("no completed tuning sweep to plot")
    fig, axes = plt.subplots(
        1,
        len(data),
        figsize=(6.6 * len(data), 4.5),
        layout="constrained",
        squeeze=False,
    )
    for ax, (name, stage) in zip(axes[0], data):
        tuning = stage["tuning"]
        ivf = sorted(tuning["int8"], key=lambda item: item["probes"])
        ax.plot(
            [r["p95_ms"] for r in ivf],
            [100 * r["recall_at_10"] for r in ivf],
            "o-",
            color="#34565c",
            label="Packed INT8 IVF",
        )
        for i, degree in enumerate(sorted({r["degree"] for r in tuning["graph"]})):
            graph = sorted(
                (r for r in tuning["graph"] if r["degree"] == degree),
                key=lambda item: item["complexity"],
            )
            ax.plot(
                [r["p95_ms"] for r in graph],
                [100 * r["recall_at_10"] for r in graph],
                marker=["s", "^"][i % 2],
                color=["#7c493f", "#566d25"][i % 2],
                label=f"DiskANN degree {degree}",
            )
        outcome = stage["heldout"]
        if outcome:
            for backend, color in [("control", "#183d42"), ("graph", "#903221")]:
                item = outcome[backend]
                ax.scatter(
                    item["p95_ms"],
                    100 * item["recall_at_10"],
                    color=color,
                    marker="x",
                    s=65,
                    label=f"Held-out {backend}",
                    zorder=4,
                )
        ax.axhline(97.6, color="#777777", linestyle=":", label="Minimum recall floor")
        ax.set(
            title=f"{name} vectors" if name != "real" else "Cached MiniLM vectors",
            xlabel="p95 lookup latency (ms)",
            ylabel="Recall@10 (%)",
            xscale="log",
            ylim=(90, 101.5) if name == "real" else (0, 103),
        )
        ax.grid(alpha=0.18)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def audit_native(stage, reference_path):
    execution = read(stage / "execution.json")
    reference = read(reference_path / "reference.json")
    for name, field in [
        ("queries.npy", "queries_sha256"),
        ("ids.npy", "ids_sha256"),
        ("scores.npy", "scores_sha256"),
    ]:
        if digest(reference_path / name) != reference[field]:
            raise ValueError("native reference checksum mismatch")
    truth = np.load(reference_path / "ids.npy", allow_pickle=False)
    cases = {}
    for f in sorted(stage.glob("*/summary.json")):
        case = audit_case(f.parent, reference, truth)
        manifest = case["manifest"]
        _, expected, round_label = f.parent.name.split("-")
        if (
            manifest["int8_scan_threads"] != int(expected)
            or manifest["round"] != int(round_label[1:])
            or manifest["query_offset"] != execution["query_offset"]
            or manifest["query_count"] != execution["query_count"]
            or manifest["batch"]["mode"] != "serial_request_stream"
        ):
            raise ValueError("native scan configuration or coverage mismatch")
        cases[f.parent.name] = case
    if execution["status"] != "completed":
        return {
            "execution": execution,
            "calls": sum(c["calls"] for c in cases.values()),
        }
    if (
        len(cases) != 3 * execution["rounds"]
        or len({c["manifest"]["native_sha256"] for c in cases.values()}) != 1
    ):
        raise ValueError("native study changed build or is missing arms")
    result = summarize_native(stage, execution["rounds"])
    recorded = read(stage / "heldout.json")
    for arm in recorded.values():
        # Early native runs also recorded the graph-specific recall gate.
        arm.pop("quality_passed", None)
    if result != recorded:
        raise ValueError("native ablation statistics do not reproduce")
    reference = {
        k: v for k, v in reference.items() if k not in {"tuning_count", "heldout_count"}
    }
    reference["query_count"] = len(truth)
    return {
        "execution": execution,
        "reference": reference,
        "comparison": result,
        "calls": sum(c["calls"] for c in cases.values()),
        "native_sha256": next(iter(cases.values()))["manifest"]["native_sha256"],
        "case_evidence": {
            name: {"calls": c["calls"], "sha256": c["sha256"]}
            for name, c in cases.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    stages = {
        n: audit_stage(args.study / n)
        for n in ["1m", "10m", "real"]
        if (args.study / n / "execution.json").exists()
    }
    native = {}
    for name, reference in [
        ("100k", "100k-reference"),
        ("1m", "1m/reference"),
        ("10m", "native-10m-reference"),
        ("real", "real/reference"),
    ]:
        stage = args.study / f"native-{name}"
        if (stage / "execution.json").exists():
            native[name] = audit_native(stage, args.study / reference)
    write_json(args.summary, {"graph": stages, "native": native})
    plot(stages, args.figure)
    print(
        json.dumps(
            {
                name: {"calls": data["calls"], "heldout": data["heldout"]}
                for name, data in stages.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
