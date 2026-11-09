"""Validate complete local study records and export compact, private-text-free evidence."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from generate_embedding_rag import selected_cases
from run_embedding_study import DOMAINS, LOCAL, registry
from vortex.embedding_study import group_interval, load_study
from vortex.evaluation import fingerprint
from vortex.metrics import retrieval_metrics
from vortex.real_rag import (
    CITATION,
    PROMPT,
    answer_scores,
    digest_object,
    load_records,
    normalize,
    writer_lock,
)


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def paired(rows, reference, field):
    return group_interval(
        [
            {
                "group_id": r["group_id"],
                "difference": r[field] - reference[r["query_id"]][field],
            }
            for r in rows
        ],
        "difference",
    )


def validate_retrieval(data, result):
    cases = {c["query_id"]: c for c in data["cases"]}
    rows = result["rows"]
    if len(rows) != len(cases) or {r["query_id"] for r in rows} != set(cases):
        raise ValueError("retrieval grid incomplete or duplicated")
    for row in rows:
        case = cases[row["query_id"]]
        if (
            row["group_id"] != case["group_id"]
            or row["answerable"] != case["answerable"]
        ):
            raise ValueError("retrieval identity mismatch")
        if case["answerable"] and row["metrics"] != retrieval_metrics(
            row["ids"], case["relevant_ids"], result["k"]
        ):
            raise ValueError("retrieval scoring mismatch")


def encoding_timing_diagnostics(directory):
    records = [
        json.loads(p.read_text()) for p in sorted(directory.glob("corpus-*.json"))
    ]
    batches = [r for r in records if "ids" in r]
    per_item = [r["seconds"] / len(r["ids"]) for r in batches for _ in r["ids"]]
    return {
        "retained_all_elapsed_observations": True,
        "batches": len(batches),
        "median_batch_seconds": float(np.median([r["seconds"] for r in batches])),
        "median_batch_seconds_per_item": float(np.median(per_item)),
        "p95_batch_seconds_per_item": float(np.percentile(per_item, 95)),
        "batches_over_60_seconds": sum(r["seconds"] > 60 for r in batches),
        "largest_elapsed_batches": [
            {"seconds": r["seconds"], "ids": r["ids"], "sha256": r["sha256"]}
            for r in sorted(batches, key=lambda r: r["seconds"], reverse=True)[:3]
        ],
        "interpretation": "Post hoc timing diagnostic; no outliers removed and no quality selection changed",
    }


def annotated_span_coverage(records, data, raw_path, expected_sha256):
    """Trace original annotated character spans through the submitted byte budget."""
    if fingerprint(raw_path) != expected_sha256:
        raise ValueError("raw QA source differs from preparation provenance")
    cases = {c["query_id"]: c for c in data["cases"]}
    chunks = {c["id"]: c for c in data["chunks"]}
    exclusions = [
        r
        for r in data.get("exclusions", [])
        if r["reason"] == "no_chunk_contains_answer_span"
    ]
    wanted = set(cases) | {r["query_id"] for r in exclusions[:3]}
    originals = {}
    for article in json.loads(raw_path.read_text(encoding="utf-8"))["data"]:
        for paragraph in article["paragraphs"]:
            source = hashlib.sha256(
                (article["title"] + "\0" + paragraph["context"]).encode()
            ).hexdigest()
            for qa in paragraph["qas"]:
                if qa["id"] in wanted:
                    originals[qa["id"]] = {
                        "source_id": source,
                        "context": paragraph["context"],
                        "qa": qa,
                    }
    result = []
    for row in records:
        case = cases[row["query_id"]]
        source = originals[case["query_id"]]
        if source["source_id"] != case["source_id"]:
            raise ValueError("QA source paragraph mapping differs")
        cursor, survives = 0, False
        gold = set(case["relevant_ids"])
        for identifier in row["included_ids"]:
            prefix = f"[{identifier}] "
            if not row["context"].startswith(prefix, cursor):
                raise ValueError("submitted evidence ID framing differs")
            cursor += len(prefix)
            chunk = chunks[identifier]
            if identifier in row["truncated_ids"]:
                body = row["context"][cursor:-1]
                cursor = len(row["context"])
                if not chunk["text"].startswith(body):
                    raise ValueError("submitted partial evidence differs")
            else:
                body = chunk["text"]
                if row["context"][cursor : cursor + len(body) + 1] != body + "\n":
                    raise ValueError("submitted full evidence differs")
                cursor += len(body) + 1
            if identifier in gold:
                if (
                    chunk["source_id"] != case["source_id"]
                    or chunk["text"] != source["context"][chunk["start"] : chunk["end"]]
                ):
                    raise ValueError(
                        "annotated gold chunk differs from its original source offsets"
                    )
                survives |= any(
                    chunk["start"] <= a["answer_start"]
                    and a["answer_start"] + len(a["text"]) <= chunk["start"] + len(body)
                    for a in source["qa"]["answers"]
                )
        if cursor != len(row["context"]):
            raise ValueError("unaccounted submitted evidence text")
        result.append(
            dict(
                row,
                annotated_evidence_requested=bool(
                    gold.intersection(row["requested_ids"])
                ),
                annotated_answer_span_survives=bool(survives),
            )
        )
    examples = [
        {"query_id": r["query_id"], "reason": r["reason"], **originals[r["query_id"]]}
        for r in exclusions[:3]
        if r["query_id"] in originals
    ]
    return result, {
        "excluded_no_full_answer_span_cases": len(exclusions),
        "examples": examples,
    }


def rag_summary(args, selection):
    directory = args.run / "rag"
    if not (directory / "generations.jsonl").exists():
        return {"status": "pending_generation", "human_review_status": "pending"}
    data = load_study(args.dataset / "squad-test.json")
    cases = {c["query_id"]: c for c in selected_cases(data)}
    repeat_ids = {
        c["query_id"]
        for c in sorted(
            cases.values(), key=lambda c: digest_object("repeat-" + c["query_id"])
        )[:5]
    }
    conditions = ["no_context", "oracle"] + selection["rag_models"]
    expected = {
        (f"{base}:r{repeat}", qid)
        for base in conditions
        for repeat in range(3)
        for qid in cases
        if repeat == 0 or qid in repeat_ids
    }
    records = load_records(directory / "generations.jsonl")
    generation_contract = json.loads((directory / "contract.json").read_text())
    if (
        generation_contract["study_contract"] != fingerprint(args.run / "contract.json")
        or generation_contract["selection"] != fingerprint(args.run / "selection.json")
        or generation_contract["runner"]
        != fingerprint(ROOT / "eval/generate_embedding_rag.py")
    ):
        raise ValueError("generation contract changed")
    if (
        len(records) != len(expected)
        or {(r["condition"], r["query_id"]) for r in records} != expected
    ):
        raise ValueError(f"incomplete generation grid: {len(records)}/{len(expected)}")
    for row in records:
        case = cases[row["query_id"]]
        if row["prompt_sha256"] != digest_object(
            PROMPT.format(context=row["context"], query=case["query"])
        ):
            raise ValueError("generation prompt hash mismatch")
        scores = answer_scores(row["answer"], case["reference_answers"])
        if row["status"] != "ok" or scores["empty_output"]:
            scores.update(squad_exact_match=0.0, squad_token_f1=0.0)
        if scores != row["scores"]:
            raise ValueError("answer score mismatch")
        if row["citations"] != [int(c) for c in CITATION.findall(row["answer"])]:
            raise ValueError("citation extraction mismatch")
        if row["valid_citations"] != sum(
            c in row["included_ids"] for c in row["citations"]
        ):
            raise ValueError("citation denominator mismatch")
    records, chunking = annotated_span_coverage(
        records,
        data,
        args.raw_squad,
        json.loads((args.dataset / "provenance.json").read_text())["source_hashes"][
            "dev-v2.0.json"
        ],
    )
    main = [r for r in records if r["repeat"] == 0]
    official_path = ROOT / "data/real-rag/evaluate-v2.0.py"
    official_hash = "710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4"
    if fingerprint(official_path) != official_hash:
        raise ValueError("official SQuAD scorer hash differs")
    scorer = runpy.run_path(str(official_path))["get_raw_scores"]
    positive_records = [r for r in records if r["answerable"]]
    official_data = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "id": str(i),
                            "answers": [{"text": a} for a in r["reference_answers"]],
                        }
                    ]
                }
                for i, r in enumerate(positive_records)
            ]
        }
    ]
    em, f1 = scorer(
        official_data,
        {str(i): r["scores"]["scored_answer"] for i, r in enumerate(positive_records)},
    )
    if any(
        abs(em[str(i)] - r["scores"]["squad_exact_match"]) > 1e-12
        or abs(f1[str(i)] - r["scores"]["squad_token_f1"]) > 1e-12
        for i, r in enumerate(positive_records)
    ):
        raise ValueError("official answer scoring differs")
    summaries = {}
    for condition in conditions:
        rows = [r for r in main if r["base_condition"] == condition]
        positive = [
            dict(
                r, f1=r["scores"]["squad_token_f1"], em=r["scores"]["squad_exact_match"]
            )
            for r in rows
            if r["answerable"]
        ]
        negative = [
            dict(r, abstention=float(r["scores"]["abstained"]))
            for r in rows
            if not r["answerable"]
        ]
        substantive = [
            r
            for r in rows
            if not r["scores"]["empty_output"] and not r["scores"]["abstained"]
        ]
        summaries[condition] = {
            "records": len(rows),
            "positive_cases": len(positive),
            "paragraph_negative_cases": len(negative),
            "f1": group_interval(positive, "f1"),
            "exact_match": group_interval(positive, "em"),
            "negative_abstention": group_interval(negative, "abstention"),
            "request_failures": sum(r["status"] != "ok" for r in rows),
            "attempts": sum(len(r["attempts"]) for r in rows),
            "raw_empty_responses": sum(not r["answer"].strip() for r in rows),
            "citation_only_responses": sum(
                bool(r["answer"].strip()) and r["scores"]["empty_output"] for r in rows
            ),
            "empty_scored_answers": sum(
                not r["scores"]["scored_answer"].strip() for r in rows
            ),
            "explicit_abstentions": sum(r["scores"]["abstained"] for r in rows),
            "substantive_answers": len(substantive),
            "answers_with_citations": sum(bool(r["citations"]) for r in substantive),
            "citation_mentions": sum(len(r["citations"]) for r in rows),
            "valid_id_mentions": sum(r["valid_citations"] for r in rows),
            "truncated_contexts": sum(bool(r["truncated_ids"]) for r in rows),
            "positives_with_annotated_gold_requested": sum(
                r["annotated_evidence_requested"] for r in positive
            ),
            "positives_with_annotated_answer_span_submitted": sum(
                r["annotated_answer_span_survives"] for r in positive
            ),
            "positives_losing_annotated_span_to_context_budget": sum(
                r["annotated_evidence_requested"]
                and not r["annotated_answer_span_survives"]
                for r in positive
            ),
            "reference_text_in_context_positives": sum(
                any(
                    a.casefold() in r["context"].casefold()
                    for a in r["reference_answers"]
                )
                for r in positive
            ),
            "generation_p50_ms": float(
                np.median([r["timings_ms"]["generation"] for r in rows])
            ),
            "generation_p95_ms": float(
                np.percentile([r["timings_ms"]["generation"] for r in rows], 95)
            ),
            "lexical_diagnostics": {
                key: {
                    "mean": float(np.mean(values)) if values else None,
                    "n": len(values),
                }
                for key in ["lexical_support", "verbatim_sentence_support"]
                for values in [
                    [
                        r["diagnostics"][key]
                        for r in rows
                        if r["diagnostics"][key] is not None
                    ]
                ]
            },
        }
    reference = {
        r["query_id"]: dict(r, f1=r["scores"]["squad_token_f1"])
        for r in main
        if r["base_condition"] == "minilm"
    }
    pairs = {
        condition: paired(
            [
                dict(r, f1=r["scores"]["squad_token_f1"])
                for r in main
                if r["base_condition"] == condition and r["answerable"]
            ],
            reference,
            "f1",
        )
        for condition in conditions
    }
    matched = []
    for _, group in itertools.groupby(
        sorted(records, key=lambda r: (r["query_id"], r["prompt_sha256"])),
        key=lambda r: (r["query_id"], r["prompt_sha256"]),
    ):
        matched.extend(itertools.combinations(list(group), 2))
    sample = []
    for condition in conditions:
        for answerable, count in [(True, 5), (False, 1)]:
            eligible = [
                r
                for r in main
                if r["base_condition"] == condition and r["answerable"] == answerable
            ]
            sample.extend(
                sorted(
                    eligible,
                    key=lambda r: digest_object("blind-" + r["query_id"] + condition),
                )[:count]
            )
    sample.sort(
        key=lambda r: digest_object(
            "review-order-" + r["query_id"] + r["base_condition"]
        )
    )
    blind = [
        {
            "review_id": f"B{i + 1:03d}",
            "question": r["query"],
            "answer": r["answer"],
            "evidence": r["context"],
            "references": r["reference_answers"],
            "semantic_correctness": None,
            "claim_support": None,
            "citation_correctness": None,
            "evidence_coverage": None,
            "abstention_appropriateness": None,
            "claim_annotations": [],
        }
        for i, r in enumerate(sample)
    ]
    json_file(directory / "blinded-review.json", blind)
    json_file(
        directory / "review-key.json",
        [
            {
                "review_id": f"B{i + 1:03d}",
                "condition": r["base_condition"],
                "query_id": r["query_id"],
            }
            for i, r in enumerate(sample)
        ],
    )
    (directory / "review-rubric.md").write_text(
        "# Pending human review\n\nThe model labels are hidden. Each condition contributes five positive and one paragraph-negative case. The key is stored separately. No automated semantic judge, calibration or human annotations are available.\n\nUse scores 0 incorrect or unsupported, 1 partial, 2 complete, and null when inapplicable.\n\n- Semantic correctness measures the factual answer against the question and references, accepting valid paraphrases. Use null for pure abstentions.\n- Claim support measures whether substantive generated claims are supported by submitted evidence. Use null when no factual claims are made.\n- Citation correctness checks both ID validity and whether each cited chunk supports its associated claim. Use null when there are no citations; citation presence is measured separately.\n- Evidence coverage asks whether submitted evidence supplies the information needed to answer the question, independently of the generated answer.\n- Abstention appropriateness asks whether declining to answer is justified by submitted evidence. Source-negative labels apply to one original paragraph and do not prove corpus-wide absence.\n\nAdd individual claims and their source support in claim_annotations before estimating an unsupported-claim rate. Coarse row grades cannot determine that rate. Record reviewer identity, reasons and uncertainties separately.\n",
        encoding="utf-8",
    )
    # Private, explicitly unjudged examples help inspect mechanical failures.
    examples = {
        "citation_only": [
            r for r in main if r["scores"]["empty_output"] and r["answer"].strip()
        ][:3],
        "truncated": [r for r in main if r["truncated_ids"]][:3],
        "oracle_zero_lexical_score": [
            r
            for r in main
            if r["base_condition"] == "oracle"
            and r["answerable"]
            and r["scores"]["squad_token_f1"] == 0
        ][:3],
        "invalid_citation_ids": [
            r for r in main if r["valid_citations"] < len(r["citations"])
        ][:3],
        "annotated_span_lost_to_context_budget": [
            r
            for r in main
            if r["answerable"]
            and r["annotated_evidence_requested"]
            and not r["annotated_answer_span_survives"]
        ][:3],
        "chunking_exclusions": chunking,
    }
    retrieval_misses = {}
    lookup = {c["id"]: c for c in data["chunks"]}
    case_lookup = {c["query_id"]: c for c in data["cases"]}
    for model in selection["rag_models"]:
        ranked = json.loads(
            (args.run / model / "squad-test/exact-k5.json").read_text()
        )["rows"]
        misses = [r for r in ranked if r["answerable"] and not r["metrics"]["hit"]][:2]
        retrieval_misses[model] = [
            {
                "query": case_lookup[r["query_id"]]["query"],
                "query_id": r["query_id"],
                "references": case_lookup[r["query_id"]]["reference_answers"],
                "retrieved_chunks": [lookup[i] for i in r["ids"]],
                "annotated_gold_chunks": [
                    lookup[i] for i in case_lookup[r["query_id"]]["relevant_ids"]
                ],
            }
            for r in misses
        ]
    examples["retrieval_misses"] = retrieval_misses
    json_file(directory / "diagnostic-examples.json", examples)
    server_snapshots = [r.get("server_process_memory", {}) for r in records]
    return {
        "status": "completed",
        "conditions": summaries,
        "paired_f1_difference_from_minilm": pairs,
        "total_requests": len(records),
        "http_generation_attempts": sum(len(r["attempts"]) for r in records),
        "total_request_failures": sum(r["status"] != "ok" for r in records),
        "generator_runtime_memory": {
            "maximum_observed_combined_rss_bytes": max(
                (
                    sum(p["rss_bytes"] for p in snap.get("processes", []))
                    for snap in server_snapshots
                    if snap.get("available")
                ),
                default=None,
            ),
            "maximum_individual_lifetime_peak_rss_bytes": max(
                (
                    p["peak_rss_bytes"]
                    for snap in server_snapshots
                    for p in snap.get("processes", [])
                    if p.get("peak_rss_bytes") is not None
                ),
                default=None,
            ),
            "boundary": "Model-runtime process samples after requests; lifetime peaks are individual process peaks, not a sum of simultaneous peaks; Python client/encoders separate",
        },
        "generator_running_model": json.loads((directory / "runtime.json").read_text())[
            "models"
        ],
        "repeat_questions": len(repeat_ids),
        "identical_prompt_pairs": len(matched),
        "different_raw_answers": sum(a["answer"] != b["answer"] for a, b in matched),
        "different_normalized_answers": sum(
            normalize(a["scores"]["scored_answer"])
            != normalize(b["scores"]["scored_answer"])
            for a, b in matched
        ),
        "human_review_status": "pending",
        "human_review_samples": len(blind),
        "semantic_correctness": None,
        "citation_correctness": None,
        "unsupported_claim_rate": None,
        "automated_semantic_judge": None,
        "official_scoring_validation": {
            "positive_records": len(positive_records),
            "sha256": official_hash,
            "mismatches": 0,
        },
        "generator": {
            k: v
            for k, v in json.loads((directory / "generator.json").read_text()).items()
            if k in {"model", "version"}
        },
    }


def report(args):
    with writer_lock(args.run):
        other_locks = [
            p for p in args.run.rglob(".writer.lock") if p.parent != args.run
        ]
        if other_locks:
            raise RuntimeError("a study writer is active")
        contract = json.loads((args.run / "contract.json").read_text())
        for path, expected in contract["sources"].items():
            if fingerprint(ROOT / path) != expected:
                raise ValueError("study source contract changed")
        for name, expected in contract["data"].items():
            if fingerprint(args.dataset / f"{name}.json") != expected:
                raise ValueError("study input contract changed")
        selection = json.loads((args.run / "selection.json").read_text())
        summary = {
            "schema_version": 1,
            "models": {},
            "paired_exact_differences": {},
            "selection": selection,
            "data": json.loads((args.dataset / "provenance.json").read_text()),
            "manifest": json.loads((args.run / "manifest.json").read_text()),
            "contract": contract,
            "cost_forecast": json.loads((args.run / "cost-forecast.json").read_text()),
            "human_review_status": "pending",
        }
        summary["analysis_sources_sha256"] = {
            p: fingerprint(ROOT / p)
            for p in [
                "eval/report_embedding_study.py",
                "eval/validate_embedding_study.py",
                "eval/generate_embedding_rag.py",
                "eval/run_embedding_timing_audit.py",
                "eval/embedding_timing_audit_spec.md",
            ]
        }
        imported = args.run / "pre-score-cache-import.json"
        if imported.exists():
            summary["pre_score_cache_import"] = json.loads(imported.read_text())
        summary["cost_forecast_scope"] = (
            "Conservative original request-volume upper estimate counts shared scientific corpus twice; no API calls executed."
        )
        flat = []
        for model, spec in registry().items():
            if model not in LOCAL:
                summary["models"][model] = {
                    "identity": spec,
                    "status": "pending_owner_deferred",
                    "requests": 0,
                    "failures": 0,
                    "actual_cost_usd": 0,
                    "required_access": spec["environment"],
                    "provider_returned_version": None,
                }
                continue
            output = {"identity": spec, "status": "completed", "datasets": {}}
            for domain in DOMAINS:
                data = load_study(args.dataset / f"{domain}-test.json")
                metrics = {}
                for mode in ["exact", "selected"]:
                    for k in [5, 10]:
                        result = json.loads(
                            (
                                args.run / model / f"{domain}-test/{mode}-k{k}.json"
                            ).read_text()
                        )
                        validate_retrieval(data, result)
                        positive = [r for r in result["rows"] if r["answerable"]]
                        scored = [dict(r, **r["metrics"]) for r in positive]
                        aggregate = {
                            "configuration": result["configuration"],
                            "cases": len(result["rows"]),
                            "positive_cases": len(positive),
                            "failures": sum(
                                r["status"] != "ok" for r in result["rows"]
                            ),
                            "labeled": {
                                field: group_interval(scored, field)
                                for field in ["recall", "hit", "mrr", "ndcg"]
                            },
                            "ann_neighbor_recall": result["ann_neighbor_recall"],
                            "retrieval_p50_ms": result["retrieval_p50_ms"],
                            "retrieval_p95_ms": result["retrieval_p95_ms"],
                            "timing_observations": result["timing_observations"],
                            "snapshot": result["snapshot"],
                            "retrieval_client_memory": result["process_memory"],
                        }
                        aggregate["overfetch"] = {
                            "mean_effective_chunk_k": float(
                                np.mean(
                                    [r["effective_chunk_k"] for r in result["rows"]]
                                )
                            ),
                            "max_effective_chunk_k": max(
                                r["effective_chunk_k"] for r in result["rows"]
                            ),
                            "mean_search_calls": float(
                                np.mean([r["search_calls"] for r in result["rows"]])
                            ),
                            "mean_rerank_n": float(
                                np.mean([r["rerank_n"] for r in result["rows"]])
                            ),
                        }
                        metrics[f"{mode}@{k}"] = aggregate
                        flat.append(
                            {
                                "model": model,
                                "domain": domain,
                                "mode": mode,
                                "k": k,
                                **{
                                    field: aggregate["labeled"][field]["mean"]
                                    for field in ["recall", "hit", "mrr", "ndcg"]
                                },
                                "ann_neighbor_recall": aggregate["ann_neighbor_recall"],
                                "p50_ms": aggregate["retrieval_p50_ms"],
                                "p95_ms": aggregate["retrieval_p95_ms"],
                                "snapshot_bytes": aggregate["snapshot"]["bytes"],
                            }
                        )
                output["datasets"][domain] = {
                    "embedding": json.loads(
                        (args.run / model / f"{domain}-test/embedding.json").read_text()
                    ),
                    "retrieval": metrics,
                    "encoding_timing_diagnostics": encoding_timing_diagnostics(
                        args.run / model / f"{domain}-test"
                    ),
                }
            output["encoder"] = json.loads(
                (args.run / model / "encoder-complete.json").read_text()
            )
            summary["models"][model] = output
        for domain in DOMAINS:
            reference = {
                r["query_id"]: dict(r, **r["metrics"])
                for r in json.loads(
                    (args.run / "minilm" / f"{domain}-test/exact-k10.json").read_text()
                )["rows"]
                if r["answerable"]
            }
            summary["paired_exact_differences"][domain] = {
                model: {
                    field: paired(
                        [
                            dict(r, **r["metrics"])
                            for r in json.loads(
                                (
                                    args.run / model / f"{domain}-test/exact-k10.json"
                                ).read_text()
                            )["rows"]
                            if r["answerable"]
                        ],
                        reference,
                        field,
                    )
                    for field in ["recall", "ndcg"]
                }
                for model in LOCAL
                if model != "minilm"
            }
        summary["tradeoffs"] = {}
        for domain in DOMAINS:
            points = {}
            for model in LOCAL:
                m = summary["models"][model]
                d = m["datasets"][domain]
                points[model] = [
                    -d["retrieval"]["exact@10"]["labeled"]["ndcg"]["mean"],
                    d["embedding"]["query_p95_ms"],
                    m["encoder"]["process_memory"]["peak_rss_bytes"],
                    d["retrieval"]["exact@10"]["snapshot"]["bytes"],
                ]
            frontier = [
                a
                for a in LOCAL
                if not any(
                    all(x <= y for x, y in zip(points[b], points[a]))
                    and any(x < y for x, y in zip(points[b], points[a]))
                    for b in LOCAL
                    if b != a
                )
            ]
            summary["tradeoffs"][domain] = {
                "descriptive_highest_ndcg10": min(LOCAL, key=lambda m: points[m][0]),
                "observed_pareto_models": frontier,
                "objectives": "Maximize exact nDCG@10; minimize query p95, encoder peak RSS and FP32 snapshot bytes",
                "scope": "Point estimates only; sampling intervals and uncontrolled runtime load can change dominance",
            }
        summary["rag"] = rag_summary(args, selection)
        audit_summaries = {
            m: json.loads((args.timing_audit / m / "summary.json").read_text())
            for m in LOCAL
            if (args.timing_audit / m / "summary.json").exists()
        }
        summary["supplemental_timing_audit"] = {
            "status": "completed" if len(audit_summaries) == 4 else "pending",
            "models": audit_summaries,
            "purpose": "Separate small timing diagnostic following observed sleep/standby interruptions; primary metrics and selection retained",
        }
        if len(audit_summaries) == 4:
            audit_contract = json.loads(
                (args.timing_audit / "contract.json").read_text()
            )
            if (
                audit_contract["primary_contract_sha256"]
                != fingerprint(args.run / "contract.json")
                or audit_contract["spec_sha256"]
                != fingerprint(ROOT / "eval/embedding_timing_audit_spec.md")
                or audit_contract["runner_sha256"]
                != fingerprint(ROOT / "eval/run_embedding_timing_audit.py")
                or audit_contract["encoder_sha256"]
                != contract["sources"]["python/vortex/embedding_models.py"]
                or audit_contract["registry_sha256"]
                != contract["sources"]["eval/embedding_models.json"]
            ):
                raise ValueError(
                    "supplemental audit contract differs from primary model/data implementation"
                )
            if any(
                audit_contract["datasets"][d] != contract["data"][f"{d}-development"]
                for d in DOMAINS
            ):
                raise ValueError("supplemental audit uses different development inputs")
            for model, audit in audit_summaries.items():
                if audit["model"] != registry()[model] or audit["calls"] != 84:
                    raise ValueError(
                        "supplemental audit model identity or call grid mismatch"
                    )
                if (
                    fingerprint(args.timing_audit / model / "timings.jsonl")
                    != audit["records_sha256"]
                ):
                    raise ValueError("supplemental timing records differ")
            summary["supplemental_timing_audit"]["contract"] = audit_contract
        weight_archive = ROOT / "data/embedding-study/model-snapshots/manifest.json"
        if weight_archive.exists():
            summary["private_weight_archive_manifest_sha256"] = fingerprint(
                weight_archive
            )
        events = args.run / "power-events.json"
        if events.exists():
            power_records = json.loads(events.read_text(encoding="utf-8-sig"))
            summary["timing_interruption_evidence"] = {
                "private_event_log_sha256": fingerprint(events),
                "kernel_power_event_counts": {
                    str(i): sum(r["event_id"] == i for r in power_records)
                    for i in [42, 107, 506, 507, 105]
                },
                "interpretation": "Windows sleep/standby events occurred during collection; precise per-batch CPU/sleep split was not recorded. Raw elapsed observations remain in primary results.",
            }
        args.output.mkdir(parents=True, exist_ok=True)
        json_file(args.output / "summary.json", summary)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
        (args.output / "retrieval.csv").write_text(
            buffer.getvalue(), encoding="utf-8", newline="\n"
        )
        json_file(
            args.output / "checksums.json",
            {
                p.name: fingerprint(p)
                for p in args.output.iterdir()
                if p.name != "checksums.json" and p.is_file()
            },
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rag_status": summary["rag"]["status"],
                    "models": {k: v["status"] for k, v in summary["models"].items()},
                },
                indent=2,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/embedding-study/prepared")
    )
    parser.add_argument(
        "--run", type=Path, default=Path("results/embedding-study-20261109-v3")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/reference/embedding-comparison")
    )
    parser.add_argument(
        "--raw-squad", type=Path, default=Path("data/real-rag/dev-v2.0.json")
    )
    parser.add_argument(
        "--timing-audit",
        type=Path,
        default=Path("results/embedding-timing-audit-20261109"),
    )
    report(parser.parse_args())


if __name__ == "__main__":
    main()
