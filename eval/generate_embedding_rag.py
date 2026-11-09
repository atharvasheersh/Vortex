"""Run the frozen development-selected embedding conditions with one generator."""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from run_real_rag import OPTIONS, http_json
from vortex.embedding_study import bind_contract, load_study
from vortex.evaluation import fingerprint, process_memory, write_json
from vortex.metrics import answer_metrics
from vortex.real_rag import (
    CITATION,
    PROMPT,
    answer_scores,
    append_record,
    digest_object,
    evidence_context,
    load_records,
    writer_lock,
)

GENERATOR = "gemma3:1b"
DIGEST = "8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc"


def server_process_memory():
    """Sample only model-runtime processes; never publish unrelated application names."""
    try:
        import psutil
    except ImportError:
        return {"available": False, "processes": []}
    rows, targets = [], {}
    for process in psutil.process_iter(["pid", "name"]):
        if (process.info["name"] or "").lower() not in {
            "ollama.exe",
            "ollama",
            "ollama_llama_server.exe",
            "ollama-runner.exe",
        }:
            continue
        targets[process.pid] = process
        try:
            targets.update(
                {child.pid: child for child in process.children(recursive=True)}
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    for process in targets.values():
        try:
            memory = process.memory_info()
            rows.append(
                {
                    "pid": process.pid,
                    "name": process.name(),
                    "rss_bytes": memory.rss,
                    "peak_rss_bytes": getattr(memory, "peak_wset", None),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "available": True,
        "processes": rows,
        "method": "psutil per-process RSS and Windows lifetime peak working set, when available",
    }


def selected_cases(data):
    cases = []
    for positive, count in [(True, 50), (False, 10)]:
        cohort = [c for c in data["cases"] if c["answerable"] == positive]
        cases += sorted(
            cohort, key=lambda c: digest_object("rag-subset-" + c["query_id"])
        )[:count]
    return cases


def generate(args):
    selection = json.loads((args.run / "selection.json").read_text())
    data = load_study(args.dataset / "squad-test.json")
    cases = selected_cases(data)
    if args.smoke:
        cases = [cases[0], cases[-1]]
    conditions = ["no_context", "oracle"] + selection["rag_models"]
    retrieval = {
        key: {
            r["query_id"]: r
            for r in json.loads(
                (args.run / key / "squad-test/exact-k5.json").read_text()
            )["rows"]
        }
        for key in selection["rag_models"]
    }
    directory = args.run / ("rag-smoke" if args.smoke else "rag")
    with writer_lock(args.run), writer_lock(directory):
        timings = {
            key: {
                r["query_id"]: r["milliseconds"]
                for r in load_records(args.run / key / "squad-test/query-timings.jsonl")
                if r["condition"] == "r0"
            }
            for key in selection["rag_models"]
        }
        version = http_json(args.endpoint + "/api/version")
        tags = http_json(args.endpoint + "/api/tags")
        model = next(m for m in tags["models"] if m["name"] == GENERATOR)
        if model["digest"] != DIGEST:
            raise ValueError("generator digest differs from frozen specification")
        contract = {
            "study_contract": fingerprint(args.run / "contract.json"),
            "selection": fingerprint(args.run / "selection.json"),
            "generator_digest": DIGEST,
            "runtime": version,
            "options": OPTIONS,
            "prompt": PROMPT,
            "context_bytes": 3000,
            "runner": fingerprint(__file__),
            "case_ids": [c["query_id"] for c in cases],
        }
        bind_contract(directory, contract)
        write_json(
            directory / "generator.json",
            {
                "model": model,
                "version": version,
                "show": http_json(args.endpoint + "/api/show", {"model": GENERATOR}),
            },
        )
        existing = load_records(directory / "generations.jsonl")
        done = {(r["condition"], r["query_id"]) for r in existing}
        lookup = {c["id"]: c for c in data["chunks"]}
        repeat_ids = {
            c["query_id"]
            for c in sorted(
                cases, key=lambda c: digest_object("repeat-" + c["query_id"])
            )[:5]
        }
        tasks = []
        for repeat in range(3):
            for position in np.random.default_rng(42 + repeat).permutation(len(cases)):
                case = cases[position]
                if repeat and case["query_id"] not in repeat_ids:
                    continue
                order = list(conditions)
                np.random.default_rng(
                    int(digest_object(case["query_id"] + str(repeat))[:8], 16)
                ).shuffle(order)
                tasks.extend((case, condition, repeat) for condition in order)
        warmup = http_json(
            args.endpoint + "/api/generate",
            {
                "model": GENERATOR,
                "prompt": "Reply OK.",
                "stream": False,
                "options": OPTIONS,
                "keep_alive": "30m",
            },
        )
        append_record(
            directory / "warmups.jsonl",
            {
                "condition": "warmup",
                "query_id": str(time.time_ns()),
                "response": warmup,
            },
        )
        runtime = http_json(args.endpoint + "/api/ps")
        write_json(
            directory / "runtime.json",
            {
                "models": [
                    m
                    for m in runtime.get("models", [])
                    if m.get("name") == GENERATOR or m.get("model") == GENERATOR
                ],
                "process_memory": server_process_memory(),
            },
        )
        for case, base, repeat in tasks:
            condition = f"{base}:r{repeat}"
            if (condition, case["query_id"]) in done:
                continue
            start = time.perf_counter()
            row = retrieval[base][case["query_id"]] if base in retrieval else None
            ids = (
                row["ids"]
                if row
                else (
                    []
                    if base == "no_context"
                    else case["relevant_ids"]
                    if case["answerable"]
                    else case["source_chunk_ids"]
                )
            )
            context = evidence_context(ids, lookup, 3000)
            context_ms = (time.perf_counter() - start) * 1000
            prompt = PROMPT.format(context=context["context"], query=case["query"])
            payload, attempts = None, []
            before = time.perf_counter()
            for attempt in range(2):
                began = time.perf_counter()
                try:
                    payload = http_json(
                        args.endpoint + "/api/generate",
                        {
                            "model": GENERATOR,
                            "prompt": prompt,
                            "stream": False,
                            "options": OPTIONS,
                            "keep_alive": "30m",
                        },
                    )
                    if not isinstance(payload.get("response"), str) or not payload.get(
                        "done"
                    ):
                        raise ValueError("incomplete generation")
                    if (
                        payload.get("prompt_eval_count", 0)
                        > OPTIONS["num_ctx"] - OPTIONS["num_predict"]
                    ):
                        raise ValueError("prompt exceeds allocation")
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "status": "ok",
                            "seconds": time.perf_counter() - began,
                        }
                    )
                    break
                except (
                    HTTPError,
                    URLError,
                    TimeoutError,
                    OSError,
                    ValueError,
                ) as error:
                    payload = None
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "seconds": time.perf_counter() - began,
                        }
                    )
            generation_ms = (time.perf_counter() - before) * 1000
            answer = payload["response"] if payload else ""
            scores = answer_scores(answer, case["reference_answers"])
            if payload is None or scores["empty_output"]:
                scores.update(squad_exact_match=0.0, squad_token_f1=0.0)
            citations = [int(c) for c in CITATION.findall(answer)]
            result = {
                "condition": condition,
                "base_condition": base,
                "repeat": repeat,
                "query_id": case["query_id"],
                "group_id": case["group_id"],
                "answerable": case["answerable"],
                "query": case["query"],
                "reference_answers": case["reference_answers"],
                "status": "ok" if payload else "failed",
                "requested_ids": ids,
                **context,
                "prompt_sha256": digest_object(prompt),
                "answer": answer,
                "scores": scores,
                "citations": citations,
                "valid_citations": sum(c in context["included_ids"] for c in citations),
                "diagnostics": answer_metrics(
                    scores["scored_answer"], None, context["context"]
                ),
                "attempts": attempts,
                "generation_response": payload,
                "timings_ms": {
                    "embedding": timings[base][case["query_id"]] if row else 0,
                    "retrieval": row["retrieval_ms"] if row else 0,
                    "context": context_ms,
                    "generation": generation_ms,
                },
                "process_memory": process_memory(),
                "server_process_memory": server_process_memory(),
            }
            result["timings_ms"]["total"] = sum(result["timings_ms"].values())
            append_record(directory / "generations.jsonl", result)
            done.add((condition, case["query_id"]))
            print(
                f"RAG {len(done)}/{len(tasks)} {condition} {result['status']}",
                flush=True,
            )
        http_json(
            args.endpoint + "/api/generate", {"model": GENERATOR, "keep_alive": 0}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/embedding-study/prepared")
    )
    parser.add_argument(
        "--run", type=Path, default=Path("results/embedding-study-20261109-v3")
    )
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--smoke", action="store_true")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
