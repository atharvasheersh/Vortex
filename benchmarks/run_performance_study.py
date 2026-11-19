"""Run the predeclared sequential optimization ablations in isolated processes."""

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


def make_plan(study, original_1m, original_10m, baseline, optimized):
    conditions = []
    for size, original, reference in [
        ("100k", study / "100k/collection.vtx", study / "100k/final-reference"),
        ("1m", original_1m, study / "1m/reference"),
        (
            "real-minilm",
            study / "real-minilm/collection.vtx",
            study / "real-minilm/reference",
        ),
    ]:
        for name, module, snapshot in [
            ("baseline", baseline, original),
            ("locality", baseline, study / size / "packed.vtx"),
            ("storage", optimized, original),
            ("optimized", optimized, study / size / "packed.vtx"),
        ]:
            conditions.append(
                {
                    "name": f"{size}-{name}",
                    "module": str(module),
                    "snapshot": str(snapshot),
                    "reference": str(reference),
                    "probes": "0,24",
                    "count": 100,
                    "rounds": 3,
                    "offset": 32,
                    "rerank": 0,
                    "family": "matched-fp32",
                }
            )
    selection = json.loads((study / "selection.json").read_text(encoding="utf-8"))[
        "selected"
    ]
    conditions.append(
        {
            "name": "1m-tuned",
            "module": str(optimized),
            "snapshot": str(study / f"1m/lists-{selection['nlist']}/packed.vtx"),
            "reference": str(study / "1m/reference"),
            "probes": str(selection["nprobe"]),
            "count": 100,
            "rounds": 3,
            "offset": 32,
            "rerank": 0,
            "family": "tuned-heldout",
        }
    )
    for name, module in [("baseline", baseline), ("optimized", optimized)]:
        conditions.append(
            {
                "name": f"1m-int8-{name}",
                "module": str(module),
                "snapshot": str(study / "1m/packed-int8.vtx"),
                "reference": str(study / "1m/reference"),
                "probes": "0,24",
                "count": 100,
                "rounds": 3,
                "offset": 32,
                "rerank": 100,
                "family": "compressed-heldout",
            }
        )
        conditions.append(
            {
                "name": f"100k-fp16-{name}",
                "module": str(module),
                "snapshot": str(study / "100k/packed-fp16.vtx"),
                "reference": str(study / "100k/final-reference"),
                "probes": "0,24",
                "count": 8,
                "rounds": 1,
                "offset": 0,
                "rerank": 100,
                "family": "compression-pilot",
            }
        )
    for name, module, snapshot in [
        ("baseline", baseline, original_10m),
        ("locality", baseline, study / "10m/packed.vtx"),
        ("optimized", optimized, study / "10m/packed.vtx"),
    ]:
        conditions.append(
            {
                "name": f"10m-{name}",
                "module": str(module),
                "snapshot": str(snapshot),
                "reference": str(study / "10m/reference"),
                "probes": "0,24",
                "count": 32,
                "rounds": 1,
                "offset": 32,
                "rerank": 0,
                "family": "matched-fp32",
            }
        )
    return conditions


def execute(plan, output, total_seconds=2700):
    if not math.isfinite(total_seconds) or total_seconds <= 0:
        raise ValueError("total time budget must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # Launch the interpreter directly, avoiding an extra Windows venv launcher
    # process so a timeout terminates the actual benchmark worker.
    env["PYTHONPATH"] = os.pathsep.join([str(root / "python"), *site.getsitepackages()])
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    started = time.perf_counter()
    completed = []
    try:
        # Interleave variants by round, with fixed shuffled order before any
        # held-out measurement. Run large capacity cases after smaller gates.
        for large in [False, True]:
            for round_id in range(3):
                cases = [
                    c
                    for c in plan
                    if c["name"].startswith("10m-") == large and round_id < c["rounds"]
                ]
                for position in np.random.default_rng(2026 + round_id).permutation(
                    len(cases)
                ):
                    case = cases[int(position)]
                    remaining = total_seconds - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise TimeoutError("total final-query budget exhausted")
                    destination = output / f"{case['name']}-r{round_id}"
                    command = [
                        sys._base_executable,
                        str(root / "benchmarks/run_performance.py"),
                        "measure",
                        "--snapshot",
                        case["snapshot"],
                        "--reference",
                        case["reference"],
                        "--module-dir",
                        case["module"],
                        "--output",
                        str(destination),
                        "--query-count",
                        str(case["count"]),
                        "--query-offset",
                        str(case["offset"]),
                        "--round-id",
                        str(round_id),
                        "--probes",
                        case["probes"],
                        "--rerank-n",
                        str(case["rerank"]),
                        "--condition",
                        case["name"],
                        "--max-seconds",
                        str(min(1200, remaining)),
                    ]
                    print(f"START {destination.name}", flush=True)
                    with (output / f"{destination.name}.log").open(
                        "x", encoding="utf-8"
                    ) as log:
                        subprocess.run(
                            command,
                            cwd=root,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=min(1230, remaining + 30),
                            check=True,
                        )
                    completed.append(destination.name)
                    print(f"DONE {destination.name}", flush=True)
    finally:
        (output / "execution.json").write_text(
            json.dumps(
                {
                    "completed": completed,
                    "elapsed_seconds": time.perf_counter() - started,
                    "budget_seconds": total_seconds,
                    "expected_conditions": sum(c["rounds"] for c in plan),
                    "status": "completed"
                    if len(completed) == sum(c["rounds"] for c in plan)
                    else "incomplete",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--original-1m", type=Path, required=True)
    parser.add_argument("--original-10m", type=Path, required=True)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--optimized-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=2700)
    args = parser.parse_args()
    plan = make_plan(
        args.study.resolve(),
        args.original_1m.resolve(),
        args.original_10m.resolve(),
        args.baseline_module.resolve(),
        args.optimized_module.resolve(),
    )
    execute(plan, args.output, args.budget_seconds)


if __name__ == "__main__":
    main()
