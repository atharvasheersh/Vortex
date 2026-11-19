"""Pilot, tune and measure each real-vector corpus in isolated processes."""

import argparse
import math
import os
import platform
import random
import site
import subprocess
import sys
import time
from pathlib import Path

import psutil
from measure_real_workloads import MODES
from real_workload_data import read_json
from run_performance import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "benchmarks" / "measure_real_workloads.py"
TUNED = ("cached", "fp32", "faiss")
TIERS = (95, 98, 99)
CONTINUATION_PHASES = ("tier", "load", "os-managed")


def pick(points, tier):
    eligible = [point for point in points if point["recall_at_10"] >= tier / 100]
    return min(eligible, key=lambda p: (p["p95_ms"], p["probes"])) if eligible else None


def freeze(tuning):
    result = {}
    for mode in TUNED:
        points = tuning[mode]
        tiers = {str(tier): pick(points, tier) for tier in TIERS}
        primary = tiers["98"] or tiers["99"] or tiers["95"]
        if primary is None and points:
            primary = min(
                points, key=lambda p: (-p["recall_at_10"], p["p95_ms"], p["probes"])
            )
        result[mode] = {"primary": primary, "tiers": tiers}
    return result


def probe_grid(nlist):
    return [2**i for i in range(nlist.bit_length()) if 2**i < nlist] + [nlist]


class Study:
    def __init__(self, args):
        self.args = args
        self.data = args.dataset.resolve()
        self.out = args.output.resolve()
        self.build = read_json(self.data / "build.json")
        self.out.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.env = os.environ.copy()
        import_paths = [str(ROOT / "python"), *site.getsitepackages()]
        if args.native_dir:
            native = args.native_dir.resolve(strict=True)
            if len(list(native.glob("_vortex.*"))) != 1:
                raise ValueError(
                    "native directory must contain exactly one Vortex extension"
                )
            import_paths.insert(0, str(native))
        self.env["PYTHONPATH"] = os.pathsep.join(import_paths)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.env[name] = "1"
        self.record = {
            "dataset": self.build["dataset"],
            "build_sha256": digest(self.data / "build.json"),
            "protocol_sha256": digest(
                ROOT / "benchmarks" / "real_workloads_protocol.md"
            ),
            "worker_sha256": digest(WORKER),
            "driver_sha256": digest(Path(__file__)),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "planned_rounds": args.rounds,
            "seconds_budget": args.seconds,
            "condition_seconds": args.condition_seconds,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "physical_memory": psutil.virtual_memory().total,
            "initial_available_memory": psutil.virtual_memory().available,
            "initial_free_disk": psutil.disk_usage(str(self.out)).free,
            "cases": {},
        }
        self.save()

    def save(self):
        self.record["seconds_elapsed"] = time.monotonic() - self.start
        write_json(self.out / "execution.json", self.record)

    def case(self, phase, mode, probes, offset, count, *, clients=1, round_id=0):
        name = f"{phase}-{mode}-p{probes}-c{clients}-r{round_id}"
        remaining = self.args.seconds - (time.monotonic() - self.start)
        entry = {
            "phase": phase,
            "mode": mode,
            "probes": probes,
            "offset": offset,
            "count": count,
            "round": round_id,
            "clients": clients,
        }
        self.record["cases"][name] = entry
        if remaining < 4:
            entry.update(status="skipped", reason="study time budget")
            self.save()
            return None
        timeout = min(remaining, self.args.condition_seconds)
        command = [
            sys._base_executable,
            str(WORKER),
            str(self.data),
            str(self.out / name),
            "--mode",
            mode,
            "--probes",
            str(probes),
            "--offset",
            str(offset),
            "--count",
            str(count),
            "--round",
            str(round_id),
            "--clients",
            str(clients),
            "--phase",
            phase,
            "--seconds",
            str(timeout),
            "--working-set",
            "0" if phase == "os-managed" else "1536",
        ]
        if phase == "primary" and round_id == 0:
            command.append("--batch")
            if self.build["dataset"] == "scifact":
                command.append("--relevance")
        print(f"{self.build['dataset']} {name} ({count} queries)", flush=True)
        transcript = self.out / f"{name}.log"
        try:
            with transcript.open("x", encoding="utf-8") as stdout:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=self.env,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    timeout=timeout + 5,
                    check=False,
                )
            entry.update(
                status="completed" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
            )
        except subprocess.TimeoutExpired:
            entry.update(status="timeout", reason="condition time budget")
        self.save()
        if entry["status"] == "completed":
            return read_json(self.out / name / "summary.json")
        print(f"  {entry['status']}; see {transcript.name}", flush=True)
        return None

    def run(self):
        tune_count = self.build["tuning_queries"]
        holdout = self.build["holdout_queries"]
        nlist = self.build["nlist"]
        if self.args.pilot_only:
            for mode in MODES:
                self.case("pilot", mode, max(nlist // 8, 1), 0, min(32, tune_count))
            return
        tuning = {mode: [] for mode in TUNED}
        for index, probes in enumerate(probe_grid(nlist)):
            modes = TUNED[index % len(TUNED) :] + TUNED[: index % len(TUNED)]
            for mode in modes:
                summary = self.case("tuning", mode, probes, 0, tune_count)
                if summary is not None:
                    tuning[mode].append(
                        {
                            "probes": probes,
                            "p95_ms": summary["p95_ms"],
                            "recall_at_10": summary["recall_at_10"],
                        }
                    )
        selection = freeze(tuning)
        write_json(self.out / "selection.json", selection)
        probes = {}
        for mode in MODES:
            primary = selection["cached" if mode.startswith("direct") else mode][
                "primary"
            ]
            if primary is not None:
                probes[mode] = primary["probes"]
        order = list(probes)
        random.Random(8871).shuffle(order)
        for round_id in range(self.args.rounds):
            shift = round_id % max(len(order), 1)
            for mode in order[shift:] + order[:shift]:
                self.case(
                    "primary",
                    mode,
                    probes[mode],
                    tune_count,
                    holdout,
                    round_id=round_id,
                )
        for mode in order:
            anchor = "cached" if mode.startswith("direct") else mode
            tiers = {
                point["probes"]
                for point in selection[anchor]["tiers"].values()
                if point
            }
            for alternative in sorted(tiers - {probes[mode]}):
                self.case("tier", mode, alternative, tune_count, holdout)
        for index, clients in enumerate((1, 2, 4, 8)):
            shift = index % max(len(order), 1)
            for mode in order[shift:] + order[:shift]:
                self.case(
                    "load",
                    mode,
                    probes[mode],
                    tune_count,
                    min(holdout, 256),
                    clients=clients,
                )
        if self.args.os_managed:
            for mode in order:
                self.case("os-managed", mode, probes[mode], tune_count, holdout)

    def continue_after(self, previous):
        provenance = previous / "execution.json"
        original = read_json(provenance)
        for field in (
            "dataset",
            "build_sha256",
            "protocol_sha256",
            "worker_sha256",
            "condition_seconds",
        ):
            if self.record[field] != original[field]:
                raise ValueError(f"continued study changed {field}")
        self.record["continues_sha256"] = digest(provenance)
        self.save()
        for case in original["cases"].values():
            if case["status"] != "completed" and case["phase"] in CONTINUATION_PHASES:
                self.case(
                    case["phase"],
                    case["mode"],
                    case["probes"],
                    case["offset"],
                    case["count"],
                    clients=case["clients"],
                    round_id=case["round"],
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--condition-seconds", type=float, default=150)
    parser.add_argument("--rounds", type=int, choices=range(1, 4), default=3)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--pilot-only", action="store_true")
    scope.add_argument(
        "--continue-from",
        type=Path,
        help="finish incomplete tier, load and OS-managed cases",
    )
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--os-managed", action="store_true")
    args = parser.parse_args()
    if not all(
        math.isfinite(x) and x > 0 for x in (args.seconds, args.condition_seconds)
    ):
        parser.error("time budgets must be positive and finite")
    study = Study(args)
    if args.continue_from:
        study.continue_after(args.continue_from)
    else:
        study.run()


if __name__ == "__main__":
    main()
