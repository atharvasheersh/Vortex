"""Export compact, auditable optimization evidence and recomputed statistics."""

import argparse
import gzip
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(
        (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["condition"], row["nprobe"]].append(row)
    summaries = []
    for (condition, probe), rows in sorted(groups.items()):
        times = np.array([r["wall_ms"] for r in rows])
        rounds = sorted({r["round_id"] for r in rows})
        summaries.append(
            {
                "condition": condition,
                "nprobe": probe,
                "calls": len(rows),
                "distinct_queries": len({r["query_id"] for r in rows}),
                "rounds": len(rounds),
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "mean_ms": float(np.mean(times)),
                "sequential_qps": 1000 / float(np.mean(times)),
                "recall_at_10": float(np.mean([r["recall_at_10"] for r in rows])),
                "mean_scanned": float(np.mean([r["scanned"] for r in rows])),
                "round_p95_ms": {
                    str(n): float(
                        np.percentile(
                            [r["wall_ms"] for r in rows if r["round_id"] == n], 95
                        )
                    )
                    for n in rounds
                },
                "timing_gaps": sum(r["timing_gap"] for r in rows),
            }
        )
    return summaries


def paired_comparison(records, baseline, optimized, probe):
    left = {
        (r["round_id"], r["query_id"]): r
        for r in records
        if r["condition"] == baseline and r["nprobe"] == probe
    }
    right = {
        (r["round_id"], r["query_id"]): r
        for r in records
        if r["condition"] == optimized and r["nprobe"] == probe
    }
    if not left or left.keys() != right.keys():
        raise ValueError("paired condition coverage mismatch")
    for key, row in left.items():
        if row["hits"] != right[key]["hits"] or row["scanned"] != right[key]["scanned"]:
            raise ValueError("matched comparison changed results or scan count")
    qids = sorted({q for _, q in left})
    rounds = sorted({r for r, _ in left})
    a = np.array([[left[r, q]["wall_ms"] for r in rounds] for q in qids])
    b = np.array([[right[r, q]["wall_ms"] for r in rounds] for q in qids])
    ratio = float(np.percentile(a, 95) / np.percentile(b, 95))
    rng = np.random.default_rng(9183)
    bootstraps = []
    for _ in range(1000):
        selected = rng.integers(0, len(qids), len(qids))
        bootstraps.append(
            float(np.percentile(a[selected], 95) / np.percentile(b[selected], 95))
        )
    return {
        "baseline": baseline,
        "optimized": optimized,
        "nprobe": probe,
        "p95_speedup": ratio,
        "latency_reduction_percent": 100 * (1 - 1 / ratio),
        "paired_query_bootstrap_95_interval": np.percentile(
            bootstraps, [2.5, 97.5]
        ).tolist(),
        "bootstrap_unit": "query IDs, retaining all rounds together",
        "identical_hits_and_scanned": True,
        "distinct_queries": len(qids),
        "rounds": len(rounds),
    }


def export(study, output):
    final = study / "final"
    execution = json.loads((final / "execution.json").read_text(encoding="utf-8"))
    if execution["status"] != "completed":
        raise ValueError(
            "final experiment is incomplete; retain it rather than publish success"
        )
    output.mkdir(parents=True, exist_ok=False)
    records, manifests = [], {}
    plan = json.loads((final / "plan.json").read_text(encoding="utf-8"))
    family = {p["name"]: p["family"] for p in plan}
    builds = json.loads((study / "native_builds.json").read_text(encoding="utf-8"))
    for name in execution["completed"]:
        manifest = json.loads(
            (final / name / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["status"] != "completed":
            raise ValueError(f"incomplete condition: {name}")
        rows = [
            json.loads(line)
            for line in (final / name / "queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if len(rows) != manifest["recorded_calls"]:
            raise ValueError("record count mismatch")
        for row in rows:
            records.append(
                {
                    "condition": manifest["condition"],
                    "family": family[manifest["condition"]],
                    **row,
                }
            )
        manifest["snapshot"] = Path(manifest["snapshot"]).name
        manifest["module"] = Path(manifest["module"]).name
        manifest["native_revision"] = builds[manifest["module_sha256"]]["revision"]
        manifests[name] = manifest
    payload = "".join(
        json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in records
    ).encode()
    (output / "measurements.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    write_json(output / "manifests.json", manifests)
    public_plan = []
    for case in plan:
        first = manifests[f"{case['name']}-r0"]
        public_plan.append(
            {
                **{
                    key: case[key]
                    for key in [
                        "name",
                        "probes",
                        "count",
                        "rounds",
                        "offset",
                        "rerank",
                        "family",
                    ]
                },
                "module_sha256": first["module_sha256"],
                "snapshot_sha256": first["snapshot_sha256"],
            }
        )
    write_json(output / "plan.json", public_plan)
    summaries = summarize(records)
    write_json(output / "summary.json", summaries)
    comparisons = []
    for size in ["100k", "1m", "10m", "real-minilm"]:
        for probe in [0, 24]:
            for name in ["locality", "optimized"]:
                comparisons.append(
                    paired_comparison(
                        records, f"{size}-baseline", f"{size}-{name}", probe
                    )
                )
    for probe in [0, 24]:
        comparisons.append(
            paired_comparison(records, "1m-int8-baseline", "1m-int8-optimized", probe)
        )
    write_json(output / "comparisons.json", comparisons)
    write_json(
        output / "selection.json", json.loads((study / "selection.json").read_text())
    )
    write_json(output / "execution.json", execution)
    write_json(output / "native_builds.json", builds)
    write_json(
        output / "environment.json",
        json.loads((study / "environment.json").read_text(encoding="utf-8-sig")),
    )
    for size, reference in [
        ("100k", "100k/final-reference"),
        ("1m", "1m/reference"),
        ("10m", "10m/reference"),
        ("real-minilm", "real-minilm/reference"),
    ]:
        destination = output / "references" / size
        destination.mkdir(parents=True)
        for filename in ["queries.npy", "ids.npy", "scores.npy", "reference.json"]:
            shutil.copyfile(study / reference / filename, destination / filename)
    tuning = []
    tuning_records = []
    for directory in sorted(study.glob("tuning-*")):
        lists = int(directory.name.split("-")[1])
        tuning.extend(
            {"nlist": lists, **row}
            for row in json.loads((directory / "summary.json").read_text())
        )
        tuning_records.extend(
            {"nlist": lists, **json.loads(line)}
            for line in (directory / "queries.jsonl").read_text().splitlines()
        )
    write_json(output / "tuning.json", tuning)
    (output / "tuning.jsonl.gz").write_bytes(
        gzip.compress(
            (
                "".join(
                    json.dumps(r, separators=(",", ":")) + "\n" for r in tuning_records
                )
            ).encode(),
            mtime=0,
        )
    )
    construction = {}
    for path in study.glob("**/build.json"):
        if "final" not in path.parts:
            row = json.loads(path.read_text())
            row.pop("snapshot", None)
            construction[path.parent.relative_to(study).as_posix()] = row
    for path in study.glob("**/packing.json"):
        row = json.loads(path.read_text())
        row.pop("snapshot", None)
        construction[path.parent.relative_to(study).as_posix() + "-packing"] = row
    for path in study.glob("**/encoding-*.json"):
        construction[path.relative_to(study).as_posix()] = json.loads(path.read_text())
    write_json(output / "construction.json", construction)
    real = json.loads((study / "real-minilm/source.json").read_text())
    write_json(output / "real_embeddings.json", real)
    root = Path(__file__).resolve().parents[1]
    source_destination = output / "executed-source"
    # These collector/native bytes are checked against the recorded source
    # fingerprints. They are separate from the older baseline native revision.
    expected_sources = next(iter(manifests.values()))["source_sha256"]
    for name, expected in expected_sources.items():
        source = root / name
        if sha256(source) != expected:
            raise ValueError(f"source changed since final collection: {name}")
        destination = source_destination / (name + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(
        output / "checksums.json",
        {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return summaries, comparisons


def export_optional(study, output):
    local = study / "optional-10m-int8"
    destination = output / "optional-10m-int8"
    destination.mkdir(exist_ok=False)
    plan = json.loads((local / "plan.json").read_text())
    if "error" in plan:
        plan["error_type"] = plan.pop("error").split(":", 1)[0]
    write_json(destination / "plan.json", plan)
    if (local / "encoding.json").exists():
        write_json(
            destination / "encoding.json",
            json.loads((local / "encoding.json").read_text()),
        )
    if plan["status"] == "completed":
        builds = json.loads((output / "native_builds.json").read_text())
        manifests, records = {}, []
        for name in ["baseline", "optimized"]:
            manifest = json.loads((local / name / "manifest.json").read_text())
            manifest["snapshot"] = Path(manifest["snapshot"]).name
            manifest["module"] = Path(manifest["module"]).name
            manifest["native_revision"] = builds[manifest["module_sha256"]]["revision"]
            manifests[name] = manifest
            records.extend(
                {
                    "condition": manifest["condition"],
                    "family": "optional-compression",
                    **json.loads(line),
                }
                for line in (local / name / "queries.jsonl").read_text().splitlines()
            )
        write_json(destination / "manifests.json", manifests)
        (destination / "measurements.jsonl.gz").write_bytes(
            gzip.compress(
                (
                    "".join(
                        json.dumps(r, separators=(",", ":")) + "\n" for r in records
                    )
                ).encode(),
                mtime=0,
            )
        )
        write_json(destination / "summary.json", summarize(records))
        comparison = paired_comparison(
            records, "10m-int8-baseline", "10m-int8-optimized", 24
        )
        write_json(destination / "comparison.json", comparison)
    write_json(
        output / "checksums.json",
        {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "checksums.json"
        },
    )
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append-optional", action="store_true")
    args = parser.parse_args()
    if args.append_optional:
        print(json.dumps(export_optional(args.study, args.output), indent=2))
        return
    summary, comparisons = export(args.study, args.output)
    print(
        json.dumps({"conditions": len(summary), "comparisons": comparisons}, indent=2)
    )


if __name__ == "__main__":
    main()
