"""Summarize paired real-model records and prepare blinded evidence review."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vortex.evaluation import fingerprint, write_json
from vortex.real_rag import (
    digest_object,
    load_real_dataset,
    load_records,
    normalize,
    writer_lock,
)


def clustered_interval(rows, field, seed=42):
    """Resample articles, retaining all questions within each sampled article."""
    groups = sorted({r["article"] for r in rows})
    if not groups:
        return None
    totals = np.array(
        [sum(r[field] for r in rows if r["article"] == g) for g in groups]
    )
    counts = np.array([sum(r["article"] == g for r in rows) for g in groups])
    sampled = np.random.default_rng(seed).integers(
        len(groups), size=(2000, len(groups))
    )
    means = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "mean": sum(r[field] for r in rows) / len(rows),
        "ci95": np.percentile(means, [2.5, 97.5]).tolist(),
        "n": len(rows),
        "articles": len(groups),
    }


def report(args):
    dataset = load_real_dataset(args.dataset / "test.json")
    records_path = args.run / "test/generations.jsonl"
    records = load_records(records_path)
    selection = json.loads((args.run / "selection.json").read_text())
    expected_conditions = ["no_context", "oracle", "exact", "ivf", "encoded"]
    expected = {
        (condition, c["query_id"])
        for condition in expected_conditions
        for c in dataset["cases"]
    }
    observed = {(r["condition"], r["query_id"]) for r in records}
    if observed != expected:
        raise ValueError(
            f"incomplete or unexpected records: {len(observed)}/{len(expected)}"
        )
    conditions, paired = {}, {}
    for condition in expected_conditions:
        rows = [r for r in records if r["condition"] == condition]
        answerable = [
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
        citations = sum(len(r["citations"]) for r in rows)
        substantive = [
            r
            for r in rows
            if not r["scores"]["abstained"] and not r["scores"]["empty_output"]
        ]
        conditions[condition] = {
            "records": len(rows),
            "answerable": len(answerable),
            "source_unanswerable": len(negative),
            "request_failures": sum(r["status"] != "ok" for r in rows),
            "http_attempts": sum(len(r["attempts"]) for r in rows),
            "failed_http_attempts": sum(
                attempt["status"] == "failed" for r in rows for attempt in r["attempts"]
            ),
            "retry_attempts": sum(max(0, len(r["attempts"]) - 1) for r in rows),
            "empty_outputs": sum(r["scores"]["empty_output"] for r in rows),
            "generation_length_stops": sum(
                (r["generation_response"] or {}).get("done_reason") == "length"
                for r in rows
            ),
            "squad_token_f1_answerable": clustered_interval(answerable, "f1"),
            "squad_exact_match_answerable": clustered_interval(answerable, "em"),
            "source_unanswerable_abstention": clustered_interval(
                negative, "abstention"
            ),
            "answerable_abstentions": sum(r["scores"]["abstained"] for r in answerable),
            "citation_mentions": citations,
            "valid_citation_mentions": sum(r["valid_citations"] for r in rows),
            "citation_presence": {
                "answered_records": len(substantive),
                "answered_with_citations": sum(
                    bool(r["citations"]) for r in substantive
                ),
                "meaning": "Citation presence among nonempty responses without the abstention marker",
            },
            "citation_id_validity": sum(r["valid_citations"] for r in rows) / citations
            if citations
            else None,
            "truncated_context_records": sum(bool(r["truncated_ids"]) for r in rows),
            "answerable_reference_text_present_in_context": {
                "count": sum(
                    any(answer in r["context"] for answer in r["reference_answers"])
                    for r in answerable
                ),
                "denominator": len(answerable),
                "meaning": "Exact reference text occurrence in submitted context; lexical diagnostic",
            },
            "timings_ms": {
                name: {
                    "median": float(np.median([r["timings_ms"][name] for r in rows])),
                    "p95": float(
                        np.percentile([r["timings_ms"][name] for r in rows], 95)
                    ),
                }
                for name in ["embedding", "retrieval", "context", "generation", "total"]
            },
            "lexical_diagnostics": {
                name: float(
                    np.mean(
                        [
                            r["diagnostics"][name]
                            for r in rows
                            if r["diagnostics"][name] is not None
                        ]
                    )
                )
                if any(r["diagnostics"][name] is not None for r in rows)
                else None
                for name in ["lexical_support", "verbatim_sentence_support"]
            },
            "lexical_diagnostic_denominators": {
                name: sum(r["diagnostics"][name] is not None for r in rows)
                for name in ["lexical_support", "verbatim_sentence_support"]
            },
        }
        if condition in selection["configurations"]:
            config = selection["configurations"][condition]
            name = f"{config['dtype']}-l{config['nlist']}-p{config['nprobe']}-k{config['k']}-r{config['rerank_n']}"
            retrieval = json.loads(
                (args.run / "test" / f"retrieval-{name}.json").read_text()
            )
            conditions[condition]["retrieval"] = {
                "configuration": config,
                "ann_neighbor_recall": retrieval["ann_neighbor_recall"],
                "ann_cases": len(retrieval["rows"]),
                "labeled_cases": sum(
                    r["labeled_metrics"] is not None for r in retrieval["rows"]
                ),
                "p95_retrieval_ms": float(
                    np.percentile([r["retrieval_ms"] for r in retrieval["rows"]], 95)
                ),
                "snapshot": json.loads(
                    (
                        args.run / "test" / f"{config['dtype']}-{config['nlist']}.json"
                    ).read_text()
                ),
                "labeled_metrics": {
                    key: float(
                        np.mean(
                            [
                                r["labeled_metrics"][key]
                                for r in retrieval["rows"]
                                if r["labeled_metrics"] is not None
                            ]
                        )
                    )
                    for key in ["recall", "precision", "hit", "mrr", "ndcg"]
                },
            }
    exact = {
        r["query_id"]: r
        for r in records
        if r["condition"] == "exact" and r["answerable"]
    }
    exact_all = {r["query_id"]: r for r in records if r["condition"] == "exact"}
    repeatability = {}
    for condition in ["ivf", "encoded"]:
        matched = [
            (r, exact_all[r["query_id"]])
            for r in records
            if r["condition"] == condition
            and r["prompt_sha256"] == exact_all[r["query_id"]]["prompt_sha256"]
        ]
        repeatability[condition] = {
            "identical_prompt_pairs": len(matched),
            "different_raw_answers": sum(
                a["answer"] != b["answer"] for a, b in matched
            ),
            "different_normalized_answers": sum(
                normalize(a["scores"]["scored_answer"])
                != normalize(b["scores"]["scored_answer"])
                for a, b in matched
            ),
        }
    for condition in expected_conditions:
        differences = [
            {
                "article": r["article"],
                "difference": r["scores"]["squad_token_f1"]
                - exact[r["query_id"]]["scores"]["squad_token_f1"],
            }
            for r in records
            if r["condition"] == condition and r["answerable"]
        ]
        paired[condition] = clustered_interval(differences, "difference")
    summary = {
        "conditions": conditions,
        "paired_answerable_f1_difference_from_exact": paired,
        "identical_prompt_comparison_with_exact": repeatability,
        "selection": selection,
        "human_review_status": "pending",
        "semantic_correctness": None,
        "citation_correctness": None,
        "unsupported_claim_rate": None,
        "limitations": [
            "Public SQuAD development subset and known source corpus; pretraining exposure is possible.",
            "Unanswerable labels apply to original paragraphs; corpus-wide answer absence is unverified.",
            "Exact match and token F1 do not establish semantic correctness or entailment.",
            "One local machine and one generation per question and condition.",
            "Identical prompts can produce different outputs despite requested temperature zero and seed 42.",
            "Per-stage timings were collected in separate stages; total is their sum.",
            "CI and semantic review statuses must be reported separately.",
        ],
        "record_sha256": fingerprint(records_path),
        "manifest_sha256": fingerprint(args.run / "manifest.json"),
        "selection_sha256": fingerprint(args.run / "selection.json"),
        "generator_sha256": fingerprint(args.run / "generator.json"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", summary)
    with (args.output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "answerable_n",
                "token_f1",
                "exact_match",
                "source_negative_n",
                "abstention_rate",
                "failures",
                "total_p95_ms",
            ]
        )
        for condition, result in conditions.items():
            writer.writerow(
                [
                    condition,
                    result["answerable"],
                    result["squad_token_f1_answerable"]["mean"],
                    result["squad_exact_match_answerable"]["mean"],
                    result["source_unanswerable"],
                    result["source_unanswerable_abstention"]["mean"],
                    result["request_failures"],
                    result["timings_ms"]["total"]["p95"],
                ]
            )
    prepare_review(args, records)
    print(json.dumps(conditions, indent=2))


def prepare_review(args, records):
    """Freeze representative blinded rows; preserve any subsequent annotations."""
    path = args.output / "review.jsonl"
    if path.exists():
        return
    cases = {r["query_id"]: r for r in records}
    sample = []
    for answerable, count in [(True, 8), (False, 4)]:
        eligible = [qid for qid, r in cases.items() if r["answerable"] == answerable]
        sample.extend(sorted(eligible, key=digest_object)[:count])
    blinded = sorted(
        [r for r in records if r["query_id"] in sample],
        key=lambda r: digest_object([r["query_id"], r["condition"]]),
    )
    mapping = {}
    with path.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(blinded, 1):
            identifier = f"review-{number:03d}"
            mapping[identifier] = {
                "query_id": row["query_id"],
                "condition": row["condition"],
            }
            handle.write(
                json.dumps(
                    {
                        "review_id": identifier,
                        "question": row["query"],
                        "reference_answers": row["reference_answers"],
                        "context": row["context"],
                        "answer": row["answer"],
                        "reviewer": None,
                        "review_type": None,
                        "semantic_correct": None,
                        "claims_supported": None,
                        "citations_correct": None,
                        "evidence_coverage": None,
                        "abstention_appropriate": None,
                        "failure_cause": None,
                        "reasoning": None,
                    }
                )
                + "\n"
            )
    write_json(args.output / "review-mapping.json", mapping)
    (args.output / "review-rubric.md").write_text(
        "# Evidence review rubric\n\nRecord reviewer identity and human or automated review type. "
        "Read the question, references, supplied context and answer. Judge semantic correctness "
        "against the question and references, allowing valid paraphrases. Check each factual claim "
        "for support in the supplied evidence, including numbers, negation and entity attribution. "
        "Check whether cited chunks support the claims and whether essential supporting evidence "
        "was omitted. Use true, false or null when unknown. For abstentions, judge sufficiency of "
        "the supplied evidence. No-context substantive answers lack supplied evidence even if correct. "
        "Classify errors as corpus absence, chunking, retrieval miss, truncation, generation, "
        "unsupported addition or request failure, and explain the judgment. Review types and "
        "sample denominators must accompany any reported semantic rates.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/real-rag/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with writer_lock(args.run):
        report(args)
