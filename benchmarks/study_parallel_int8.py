"""Compare cached, direct and two-worker scans with identical INT8 neighbors."""

import argparse
import json
import math
import time
from pathlib import Path

from run_performance import write_json
from study_retrieval_phase2 import paired, run_worker


def read_rows(path):
    return [
        json.loads(line)
        for line in (path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def summarize_native(output, rounds):
    groups = [
        [output / f"workers-{workers}-r{round_id}" for round_id in range(rounds)]
        for workers in (0, 1, 2)
    ]
    baseline = [r for path in groups[0] for r in read_rows(path)]
    comparisons = {}
    for workers in (1, 2):
        candidate = [r for path in groups[workers] for r in read_rows(path)]
        if len(baseline) != len(candidate) or any(
            (a["round_id"], a["query_id"], a["hits"], a["scanned"])
            != (b["round_id"], b["query_id"], b["hits"], b["scanned"])
            for a, b in zip(baseline, candidate)
        ):
            raise ValueError(
                "native ablation changed ordered IDs, scores or candidate counts"
            )
        result = paired(groups[0], groups[workers])
        result["direct"] = result.pop("graph")
        result.pop("quality_passed")
        result["identical_ordered_hits_and_scans"] = True
        comparisons[str(workers)] = result
    return comparisons


def run(args):
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.rounds < 1:
        raise ValueError("provide positive rounds and a finite time budget")
    root, snapshot = args.output.resolve(), args.snapshot.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "started",
        "rounds": args.rounds,
        "query_offset": args.offset,
        "query_count": args.count,
        "modes": [0, 1, 2],
        "seconds_budget": args.seconds,
    }
    write_json(root / "execution.json", state)
    started = time.monotonic()
    try:
        for round_id in range(args.rounds):
            # Reverse endpoints in the two-round 10m track. A Latin order
            # balances all three positions in the three-round 1m study.
            if args.rounds == 2:
                order = [0, 1, 2] if round_id == 0 else [2, 1, 0]
            else:
                order = [(round_id + n) % 3 for n in range(3)]
            for workers in order:
                remaining = args.seconds - (time.monotonic() - started)
                if remaining < 1:
                    raise TimeoutError("native study budget exhausted")
                label = f"workers-{workers}-r{round_id}"
                print(f"START {label}", flush=True)
                run_worker(
                    [
                        "benchmarks/run_retrieval_phase2.py",
                        "int8",
                        snapshot,
                        args.reference.resolve(),
                        root / label,
                        "--offset",
                        args.offset,
                        "--count",
                        args.count,
                        "--round",
                        round_id,
                        "--scan-threads",
                        workers,
                        "--seconds",
                        min(240, remaining),
                        "--batch",
                    ],
                    root / f"{label}.log",
                    min(270, remaining),
                    module_dir=args.module_dir.resolve(),
                )
                print(f"DONE {label}", flush=True)
        write_json(root / "heldout.json", summarize_native(root, args.rounds))
        state["status"] = "completed"
    except Exception as error:
        state.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "execution.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--module-dir", type=Path, default=Path("build-phase2-int8/python")
    )
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
