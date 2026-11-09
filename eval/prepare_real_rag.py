"""Prepare a bounded SQuAD 2.0 experiment with source-span chunk labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from vortex.evaluation import fingerprint, write_json

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prepare_split(
    articles,
    tokenizer,
    answerable_count,
    unanswerable_count,
    *,
    chunk_tokens=160,
    overlap_tokens=32,
    excluded_query_ids=(),
):
    if chunk_tokens <= overlap_tokens or overlap_tokens < 0:
        raise ValueError("chunk window must exceed its nonnegative overlap")
    chunks, candidates, exclusions = [], [], []
    for article in articles:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"]
            source = stable(article["title"] + "\0" + text)
            offsets = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            source_chunks = []
            for start in range(0, len(offsets), chunk_tokens - overlap_tokens):
                window = offsets[start : start + chunk_tokens]
                if not window:
                    continue
                begin, end = window[0][0], window[-1][1]
                identifier = len(chunks) + 1
                chunks.append(
                    {
                        "id": identifier,
                        "text": text[begin:end],
                        "source_id": source,
                        "article": article["title"],
                        "start": begin,
                        "end": end,
                    }
                )
                source_chunks.append(chunks[-1])
                if start + chunk_tokens >= len(offsets):
                    break
            for qa in paragraph["qas"]:
                if qa["id"] in excluded_query_ids:
                    exclusions.append(
                        {"query_id": qa["id"], "reason": "previously_inspected_query"}
                    )
                    continue
                reason = None
                answers = qa["answers"]
                if len(tokenizer.encode(qa["question"], add_special_tokens=True)) > 256:
                    reason = "question_exceeds_encoder_limit"
                if any(
                    text[a["answer_start"] : a["answer_start"] + len(a["text"])]
                    != a["text"]
                    for a in answers
                ):
                    reason = "invalid_answer_offset"
                relevant = sorted(
                    {
                        c["id"]
                        for c in source_chunks
                        for a in answers
                        if c["start"] <= a["answer_start"]
                        and c["end"] >= a["answer_start"] + len(a["text"])
                    }
                )
                if answers and not relevant:
                    reason = "no_chunk_contains_answer_span"
                if reason:
                    exclusions.append({"query_id": qa["id"], "reason": reason})
                    continue
                candidates.append(
                    {
                        "query_id": qa["id"],
                        "query": qa["question"],
                        "source_id": source,
                        "article": article["title"],
                        "answerable": not qa["is_impossible"],
                        "reference_answers": sorted({a["text"] for a in answers}),
                        "relevant_chunk_ids": relevant,
                        "source_chunk_ids": [c["id"] for c in source_chunks],
                    }
                )
    selected = []
    for answerable, count in [(True, answerable_count), (False, unanswerable_count)]:
        seen = set()
        for case in sorted(candidates, key=lambda c: stable(c["query_id"])):
            if case["answerable"] != answerable or case["source_id"] in seen:
                continue
            selected.append(case)
            seen.add(case["source_id"])
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError(
                "insufficient distinct source paragraphs for requested cases"
            )
    return {
        "schema_version": 2,
        "chunks": chunks,
        "cases": selected,
        "exclusions": exclusions,
        "selection_eligible_cases": len(candidates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/real-rag"))
    parser.add_argument("--output", type=Path, default=Path("data/real-rag/prepared"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new dataset output directory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    train_path, test_path = [
        args.input_dir / f"{split}-v2.0.json" for split in ("train", "dev")
    ]
    train = json.loads(train_path.read_text(encoding="utf-8"))["data"]
    test = json.loads(test_path.read_text(encoding="utf-8"))["data"]
    test_titles = {a["title"] for a in test}
    eligible = [a for a in train if a["title"] not in test_titles]
    development = sorted(eligible, key=lambda a: stable(a["title"]))[:10]
    dev = prepare_split(development, tokenizer, 50, 10)
    heldout = prepare_split(test, tokenizer, 150, 30)
    if {c["source_id"] for c in dev["chunks"]} & {
        c["source_id"] for c in heldout["chunks"]
    }:
        raise ValueError("source leakage")
    if {stable(c["text"]) for c in dev["chunks"]} & {
        stable(c["text"]) for c in heldout["chunks"]
    }:
        raise ValueError("duplicate chunk text across splits")
    args.output.mkdir(parents=True)
    for name, data in [("development", dev), ("test", heldout)]:
        write_json(args.output / f"{name}.json", data)
    write_json(
        args.output / "provenance.json",
        {
            "dataset": "SQuAD 2.0",
            "license": "CC-BY-SA-4.0",
            "attribution": "Rajpurkar, Jia and Liang, Know What You Don't Know, ACL 2018",
            "source": "https://rajpurkar.github.io/SQuAD-explorer/",
            "files": [
                {
                    "url": f"https://rajpurkar.github.io/SQuAD-explorer/dataset/{p.name}",
                    "sha256": fingerprint(p),
                }
                for p in [train_path, test_path]
            ],
            "embedding_model": MODEL,
            "embedding_revision": REVISION,
            "window_wordpieces": 160,
            "overlap_wordpieces": 32,
            "development_articles": [a["title"] for a in development],
            "development_chunks": len(dev["chunks"]),
            "test_chunks": len(heldout["chunks"]),
            "case_counts": {
                "development": len(dev["cases"]),
                "test": len(heldout["cases"]),
            },
            "prepared_hashes": {
                name: fingerprint(args.output / f"{name}.json")
                for name in ["development", "test"]
            },
            "answerability_scope": "source paragraph; corpus-wide negative judgments unverified",
        },
    )
    print(
        json.dumps(
            {
                "development_chunks": len(dev["chunks"]),
                "test_chunks": len(heldout["chunks"]),
            }
        )
    )


if __name__ == "__main__":
    main()
