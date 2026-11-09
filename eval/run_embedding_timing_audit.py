"""A separate small CPU timing audit with elapsed/CPU clocks and power observations."""

import argparse
import ctypes
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from run_embedding_study import registry
from vortex.embedding_models import StudyEncoder
from vortex.embedding_study import bind_contract, load_study
from vortex.evaluation import fingerprint, manifest, process_memory, write_json
from vortex.real_rag import append_record, digest_object, load_records, writer_lock


def power_state():
    if sys.platform != "win32":
        return {"available": False}

    class Status(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte),
            ("battery", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte),
            ("saver", ctypes.c_ubyte),
            ("lifetime", ctypes.c_uint32),
            ("full_lifetime", ctypes.c_uint32),
        ]

    status = Status()
    if not ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(status)):
        return {"available": False}
    return {
        "available": True,
        "ac_line_status": int(status.ac),
        "battery_saver": int(status.saver),
    }


def sample(data, domain):
    return {
        "document": sorted(
            data["chunks"],
            key=lambda c: digest_object(f"timing-audit-{domain}-{c['id']}"),
        )[:16],
        "query": sorted(
            data["cases"],
            key=lambda c: digest_object(f"timing-audit-{domain}-{c['query_id']}"),
        )[:10],
    }


def run(args):
    spec = registry()[args.model]
    datasets = {
        d: load_study(args.dataset / f"{d}-development.json")
        for d in ["squad", "scifact"]
    }
    samples = {d: sample(data, d) for d, data in datasets.items()}
    contract = {
        "primary_contract_sha256": fingerprint(args.primary / "contract.json"),
        "spec_sha256": fingerprint(ROOT / "eval/embedding_timing_audit_spec.md"),
        "runner_sha256": fingerprint(__file__),
        "encoder_sha256": fingerprint(ROOT / "python/vortex/embedding_models.py"),
        "registry_sha256": fingerprint(ROOT / "eval/embedding_models.json"),
        "datasets": {
            d: fingerprint(args.dataset / f"{d}-development.json") for d in datasets
        },
        "sample_ids": {
            d: {
                r: [c["id"] if r == "document" else c["query_id"] for c in items]
                for r, items in rows.items()
            }
            for d, rows in samples.items()
        },
    }
    directory = args.run / args.model
    with writer_lock(args.primary), writer_lock(args.run), writer_lock(directory):
        bind_contract(args.run, contract)
        if not (args.run / "manifest.json").exists():
            write_json(args.run / "manifest.json", manifest(contract))
        started = time.perf_counter()
        encoder = StudyEncoder(args.model, spec)
        initialization_seconds = time.perf_counter() - started
        dtypes = Counter()
        devices = Counter()
        for parameter in encoder.model.parameters():
            dtypes[str(parameter.dtype)] += parameter.numel()
            devices[str(getattr(parameter, "device", "unavailable"))] += (
                parameter.numel()
            )
        records = load_records(directory / "timings.jsonl")
        done = {(r["condition"], r["query_id"]) for r in records}
        ordering_checks = {}
        for domain, rows in samples.items():
            batch = encoder.encode(rows["document"][:4], "document", domain)
            individual = np.concatenate(
                [
                    encoder.encode([item], "document", domain)
                    for item in rows["document"][:4]
                ]
            )
            difference = float(np.max(np.abs(batch - individual)))
            if not np.allclose(batch, individual, atol=1e-4, rtol=1e-4):
                raise ValueError(
                    "local document batch ordering or padding invariance failed"
                )
            ordering_checks[domain] = {
                "items": 4,
                "max_absolute_difference": difference,
                "absolute_tolerance": 1e-4,
                "passed": True,
            }
            encoder.encode(rows["query"][:1], "query", domain)
            for repeat in range(3):
                for role, items in rows.items():
                    order = np.random.default_rng(142 + repeat).permutation(len(items))
                    size = 4 if role == "document" else 1
                    for start in range(0, len(order), size):
                        batch = [items[i] for i in order[start : start + size]]
                        key = (f"{domain}:{role}:r{repeat}", str(start))
                        if key in done:
                            continue
                        state = power_state()
                        cpu_start = time.process_time()
                        wall_start = time.perf_counter()
                        try:
                            vectors = encoder.encode(batch, role, domain)
                        except Exception as error:
                            append_record(
                                directory / "failures.jsonl",
                                {
                                    "condition": str(time.time_ns()),
                                    "query_id": key[0] + key[1],
                                    "error_type": type(error).__name__,
                                    "status": "failed",
                                },
                            )
                            raise
                        wall = time.perf_counter() - wall_start
                        cpu = time.process_time() - cpu_start
                        row = {
                            "condition": key[0],
                            "query_id": key[1],
                            "domain": domain,
                            "role": role,
                            "items": len(batch),
                            "wall_seconds": wall,
                            "process_cpu_seconds": cpu,
                            "power_before": state,
                            "power_after": power_state(),
                            "status": "ok",
                            "shape": list(vectors.shape),
                            "dtype": str(vectors.dtype),
                            "input_ids": [
                                c["id"] if role == "document" else c["query_id"]
                                for c in batch
                            ],
                        }
                        append_record(directory / "timings.jsonl", row)
                        records.append(row)
                        done.add(key)
        if len(records) != 84:
            raise ValueError("timing audit call grid is incomplete or duplicated")
        domains = {}
        for domain in datasets:
            rows = [r for r in records if r["domain"] == domain]
            docs = [r for r in rows if r["role"] == "document"]
            queries = [r["wall_seconds"] * 1000 for r in rows if r["role"] == "query"]
            domains[domain] = {
                "document_sample": 16,
                "query_sample": 10,
                "rounds": 3,
                "documents_per_second": sum(r["items"] for r in docs)
                / sum(r["wall_seconds"] for r in docs),
                "document_process_cpu_seconds": sum(
                    r["process_cpu_seconds"] for r in docs
                ),
                "query_p50_ms": float(np.median(queries)),
                "query_p95_ms": float(np.percentile(queries, 95)),
                "power_changed_calls": sum(
                    r["power_before"] != r["power_after"] for r in rows
                ),
                "low_cpu_ratio_calls": sum(
                    r["wall_seconds"] > max(10, 10 * r["process_cpu_seconds"])
                    for r in rows
                ),
            }
        write_json(
            directory / "summary.json",
            {
                "status": "completed",
                "model": spec,
                "domains": domains,
                "parameter_dtypes": dict(dtypes),
                "parameter_count": sum(dtypes.values()),
                "parameter_devices": dict(devices),
                "inference_threads": getattr(
                    sys.modules.get("torch"), "get_num_threads", lambda: None
                )(),
                "output_dtypes": sorted({r["dtype"] for r in records}),
                "batch_order_checks": ordering_checks,
                "matches_requested_fp32": set(dtypes) == {"torch.float32"},
                "initialization_seconds": initialization_seconds,
                "process_memory": process_memory(),
                "calls": len(records),
                "power_states": [
                    json.loads(v)
                    for v in sorted(
                        {
                            json.dumps(r[k], sort_keys=True)
                            for r in records
                            for k in ["power_before", "power_after"]
                        }
                    )
                ],
                "failed_attempts": len(load_records(directory / "failures.jsonl")),
                "records_sha256": fingerprint(directory / "timings.jsonl"),
                "contract_sha256": fingerprint(args.run / "contract.json"),
            },
        )
        print(f"timing audit {args.model} completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["embeddinggemma", "jina", "qwen", "minilm"], required=True
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/embedding-study/prepared")
    )
    parser.add_argument(
        "--primary", type=Path, default=Path("results/embedding-study-20261109-v3")
    )
    parser.add_argument(
        "--run", type=Path, default=Path("results/embedding-timing-audit-20261109")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
