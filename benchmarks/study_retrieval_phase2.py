"""Run independent references, tuning and paired held-out disk search."""

import argparse
import json
import math
import os
import site
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from run_performance import write_json
from run_retrieval_phase2 import summarize

ROOT = Path(__file__).resolve().parents[1]
# Keep two candidates for physical graph construction separate from query tuning.
SETTINGS = {
    "pilot": (100000, 16, 16, 0, 0.003),
    "1m": (1000000, 48, 100, 3, 0.060),
    "10m": (10000000, 48, 64, 2, 0.600),
    "real": (3328, 48, 100, 3, 0.001),
}
SWEEP = [(32, 2), (64, 2), (128, 2), (256, 4), (512, 4), (1024, 8)]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_worker(arguments, path, timeout, guard=False, module_dir=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(module_dir or ROOT / "build-phase2/python"),
            str(ROOT / "python"),
            *site.getsitepackages(),
        ]
    )
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    command = [sys._base_executable, "-u", *map(str, arguments)]
    start = time.monotonic()
    with (
        path.open("x", encoding="utf-8") as log,
        subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        ) as worker,
    ):
        try:
            while worker.poll() is None:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"worker exceeded {timeout:.0f}s: {path.name}")
                if guard and psutil.virtual_memory().available < 384 * 1024**2:
                    raise MemoryError(f"system memory below 384 MiB: {path.name}")
                try:
                    worker.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    continue
            if worker.returncode:
                raise subprocess.CalledProcessError(worker.returncode, command)
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


def eligible(control, alternative):
    return (
        alternative["recall_at_10"]
        >= max(0.976, control["recall_at_10"] - 0.005) - 1e-12
    )


def paired(control_paths, graph_paths):
    def rows(paths):
        return [
            json.loads(line)
            for path in paths
            for line in (path / "queries.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    control, graph = rows(control_paths), rows(graph_paths)
    a, b = summarize(control), summarize(graph)
    keys = [(r["round_id"], r["query_id"]) for r in control]
    if len(keys) != len(set(keys)) or keys != [
        (r["round_id"], r["query_id"]) for r in graph
    ]:
        raise ValueError("backend query coverage or order differs")
    n = a["distinct_queries"]
    qids = sorted({qid for _, qid in keys})
    control_by_key = dict(zip(keys, control))
    graph_by_key = dict(zip(keys, graph))
    rounds = sorted({round_id for round_id, _ in keys})
    x = np.array([[control_by_key[r, q]["wall_ms"] for r in rounds] for q in qids])
    y = np.array([[graph_by_key[r, q]["wall_ms"] for r in rounds] for q in qids])
    samples = np.random.default_rng(751).integers(0, n, size=(1000, n))
    bootstrap = np.percentile(x[samples], 95, axis=(1, 2)) / np.percentile(
        y[samples], 95, axis=(1, 2)
    )
    for summary, data, paths in [(a, control, control_paths), (b, graph, graph_paths)]:
        summary["round_p95_ms"] = [
            float(
                np.percentile(
                    [row["wall_ms"] for row in data if row["round_id"] == r], 95
                )
            )
            for r in rounds
        ]
        manifests = [read(p / "manifest.json") for p in paths]
        summary["peak_resident_mib"] = (
            max(m["query_peak_memory"]["peak_rss_bytes"] for m in manifests) / 1024**2
        )
        summary["batch_qps"] = [m["batch"]["qps"] for m in manifests if "batch" in m]
    speed = a["p95_ms"] / b["p95_ms"]
    return {
        "control": a,
        "graph": b,
        "speedup": speed,
        "latency_reduction_percent": 100 * (1 - 1 / speed),
        "recall_delta_points": 100 * (b["recall_at_10"] - a["recall_at_10"]),
        "bootstrap_speedup_interval": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "same_ordered_ids_calls": sum(
            [h[0] for h in x["hits"]] == [h[0] for h in y["hits"]]
            for x, y in zip(control, graph)
        ),
        "quality_passed": eligible(a, b),
        "faster": speed > 1,
    }


def study(args):
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.seconds, args.build_seconds, args.build_gib)
    ):
        raise ValueError("stage and build budgets must be finite and positive")
    n, tune, heldout, rounds, search_memory = SETTINGS[args.stage]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = args.snapshot.resolve()
    ref = out / "reference"
    state = {
        "stage": args.stage,
        "status": "started",
        "count": n,
        "tuning_queries": tune,
        "heldout_queries": heldout,
        "rounds": rounds,
        "seconds_budget": args.seconds,
        "free_ram_bytes": psutil.virtual_memory().available,
        "free_disk_bytes": psutil.disk_usage(str(out)).free,
        "quality_floor": 0.976,
        "max_recall_drop_points": 0.5,
        "graph_builds": args.degrees,
        "search_memory_gib": search_memory,
        "sweep": SWEEP,
    }
    write_json(out / "execution.json", state)
    start = time.monotonic()

    def worker(command, name, maximum, guard=False):
        remaining = args.seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("study budget exhausted")
        print(f"START {args.stage} {name}", flush=True)
        run_worker(command, out / (name + ".log"), min(remaining, maximum), guard=guard)
        print(f"DONE {args.stage} {name}", flush=True)

    def measure(
        backend,
        label,
        offset,
        count,
        *,
        graph=None,
        complexity=64,
        beam=2,
        probes=24,
        round_id=0,
        batch=False,
    ):
        path = out / label
        cmd = [
            "benchmarks/run_retrieval_phase2.py",
            backend,
            snapshot,
            ref,
            path,
            "--offset",
            offset,
            "--count",
            count,
            "--round",
            round_id,
            "--complexity",
            complexity,
            "--beam",
            beam,
            "--probes",
            probes,
        ]
        if graph is not None:
            cmd.extend(["--graph", graph])
        if batch:
            cmd.append("--batch")
        worker(cmd, label, 240)
        return path

    try:
        ref_cmd = [
            "benchmarks/run_performance.py",
            "reference",
            "--snapshot",
            snapshot,
            "--output",
            ref,
            "--count",
            tune + heldout if tune >= 32 else 64,
            "--seed",
            20261124,
        ]
        if args.query_file:
            ref_cmd.extend(["--query-file", args.query_file.resolve()])
        worker(ref_cmd, "exact-reference", 300)
        if read(ref / "reference.json")["n"] != n:
            raise ValueError("input size does not match stage")
        controls = []
        for probes in [24, 8, 16, 28, 32]:
            case = measure("int8", f"tune-int8-{probes}", 0, tune, probes=probes)
            controls.append({"probes": probes, **read(case / "summary.json")})
        graphs = []
        failed = []
        for degree in args.degrees:
            target = out / f"graph-d{degree}"
            try:
                worker(
                    [
                        "benchmarks/phase2_diskann.py",
                        snapshot,
                        target,
                        "--degree",
                        degree,
                        "--complexity",
                        degree + 16,
                        "--threads",
                        args.build_threads,
                        "--build-gib",
                        args.build_gib,
                        "--search-gib",
                        search_memory,
                    ],
                    target.name,
                    args.build_seconds,
                    guard=True,
                )
            except (MemoryError, TimeoutError, subprocess.CalledProcessError) as error:
                failed.append(
                    {
                        "degree": degree,
                        "type": type(error).__name__,
                        "log": target.name + ".log",
                    }
                )
                continue
            for complexity, beam in SWEEP:
                case = measure(
                    "graph",
                    f"tune-d{degree}-l{complexity}-b{beam}",
                    0,
                    tune,
                    graph=target,
                    complexity=complexity,
                    beam=beam,
                )
                graphs.append(
                    {
                        "degree": degree,
                        "complexity": complexity,
                        "beam": beam,
                        **read(case / "summary.json"),
                    }
                )
        reference_control = controls[0]
        valid = [g for g in graphs if eligible(reference_control, g)]
        selection = min(valid, key=lambda row: row["p95_ms"]) if valid else None
        within_latency = [
            g for g in graphs if g["p95_ms"] <= reference_control["p95_ms"]
        ]
        arms = {
            "int8": controls,
            "graph": graphs,
            "selected": selection,
            "failed_builds": failed,
            "best_graph_recall_within_int8_p95": max(
                within_latency, key=lambda g: g["recall_at_10"]
            )
            if within_latency
            else None,
        }
        write_json(out / "tuning.json", arms)
        if rounds and selection:
            by_backend = {"int8": [], "graph": []}
            for round_id in range(rounds):
                order = ["int8", "graph"] if round_id % 2 == 0 else ["graph", "int8"]
                for backend in order:
                    path = measure(
                        backend,
                        f"heldout-{backend}-r{round_id}",
                        tune,
                        heldout,
                        round_id=round_id,
                        batch=True,
                        graph=out / f"graph-d{selection['degree']}"
                        if backend == "graph"
                        else None,
                        complexity=selection["complexity"],
                        beam=selection["beam"],
                    )
                    by_backend[backend].append(path)
            write_json(
                out / "heldout.json", paired(by_backend["int8"], by_backend["graph"])
            )
        state["status"] = "completed" if graphs else "blocked"
    except Exception as error:
        state.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - start
        write_json(out / "execution.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SETTINGS)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--query-file", type=Path)
    parser.add_argument("--degrees", type=int, nargs="+", default=[32])
    parser.add_argument("--build-gib", type=float, default=1.5)
    parser.add_argument("--build-threads", type=int, default=4)
    parser.add_argument("--build-seconds", type=float, default=900)
    parser.add_argument("--seconds", type=float, default=1500)
    args = parser.parse_args()
    study(args)


if __name__ == "__main__":
    main()
