"""Prepare fresh QA cases and a declared restricted scientific retrieval corpus."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from prepare_real_rag import MODEL, REVISION, prepare_split, stable
from vortex.evaluation import fingerprint, write_json


def decorate_squad(data):
    data.update(
        schema_version=3,
        domain="squad",
        relevance_unit="chunk",
        generation_applicable=True,
    )
    for chunk in data["chunks"]:
        chunk.update(unit_id=chunk["id"], title=chunk["article"].replace("_", " "))
    for case in data["cases"]:
        case.update(relevant_ids=case["relevant_chunk_ids"], group_id=case["article"])
    return data


def qrels(archive, split):
    text = archive.read(f"scifact/qrels/{split}.tsv").decode()
    result = {}
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        if int(row["score"]) > 0:
            result.setdefault(row["query-id"], []).append(int(row["corpus-id"]))
    return result


def evidence_groups(relations):
    parents = {
        identifier: identifier for ids in relations.values() for identifier in ids
    }

    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    for ids in relations.values():
        roots = [find(identifier) for identifier in ids]
        for root in roots:
            parents[root] = min(roots)
    return {qid: str(find(ids[0])) for qid, ids in relations.items()}


def prepare_scifact(path, tokenizer):
    with ZipFile(path) as archive:
        corpus = {
            int(r["_id"]): r
            for r in map(json.loads, archive.read("scifact/corpus.jsonl").splitlines())
        }
        queries = {
            r["_id"]: r["text"]
            for r in map(json.loads, archive.read("scifact/queries.jsonl").splitlines())
        }
        train, test = qrels(archive, "train"), qrels(archive, "test")
    test_ids = sorted(test, key=lambda qid: stable("study-test-" + qid))[:150]
    test_docs = {doc for qid in test_ids for doc in test[qid]}
    eligible = [
        qid
        for qid in train
        if not test_docs.intersection(train[qid]) and qid not in test_ids
    ]
    dev_ids = sorted(eligible, key=lambda qid: stable("study-development-" + qid))[:50]
    if len(dev_ids) != 50 or len(test_ids) != 150:
        raise ValueError("insufficient source-separated scientific queries")
    chosen_relations = {
        qid: (train if qid in dev_ids else test)[qid] for qid in dev_ids + test_ids
    }
    gold = {doc for ids in chosen_relations.values() for doc in ids}
    distractors = sorted(
        set(corpus) - gold,
        key=lambda identifier: stable(f"study-document-{identifier}"),
    )
    documents = sorted(gold | set(distractors[: 1000 - len(gold)]))
    if len(documents) != 1000:
        raise ValueError("restricted corpus construction failed")
    chunks, by_document = [], {}
    for identifier in documents:
        document = corpus[identifier]
        text = document["text"]
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        for start in range(0, len(offsets), 104):
            window = offsets[start : start + 128]
            begin, end = window[0][0], window[-1][1]
            chunk = {
                "id": len(chunks) + 1,
                "unit_id": identifier,
                "source_id": str(identifier),
                "title": document["title"],
                "text": text[begin:end],
                "start": begin,
                "end": end,
            }
            chunks.append(chunk)
            by_document.setdefault(identifier, []).append(chunk["id"])
            if start + 128 >= len(offsets):
                break
    groups = evidence_groups(chosen_relations)
    outputs = {}
    for split, ids in [("development", dev_ids), ("test", test_ids)]:
        cases = []
        for qid in ids:
            relevant = chosen_relations[qid]
            cases.append(
                {
                    "query_id": qid,
                    "query": queries[qid],
                    "group_id": groups[qid],
                    "answerable": True,
                    "reference_answers": [],
                    "relevant_ids": relevant,
                    "source_chunk_ids": [
                        c for doc in relevant for c in by_document[doc]
                    ],
                }
            )
        outputs[split] = {
            "schema_version": 3,
            "domain": "scifact",
            "relevance_unit": "document",
            "generation_applicable": False,
            "chunks": chunks,
            "cases": cases,
            "exclusions": [],
            "original_documents": len(corpus),
            "restricted_documents": 1000,
            "corpus_selection": "All selected positive documents plus hash-selected distractors",
            "shared_corpus": True,
            "development_eligible_queries": len(eligible),
        }
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/embedding-study/prepared")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new prepared dataset directory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    old = Path("data/real-rag/prepared")
    excluded = {
        c["query_id"]
        for split in ["development", "test"]
        for c in json.loads((old / f"{split}.json").read_text())["cases"]
    }
    previous_articles = set(
        json.loads((old / "provenance.json").read_text())["development_articles"]
    )
    source = Path("data/real-rag")
    train = json.loads((source / "train-v2.0.json").read_text(encoding="utf-8"))["data"]
    test = json.loads((source / "dev-v2.0.json").read_text(encoding="utf-8"))["data"]
    heldout_titles = {a["title"] for a in test}
    eligible = [
        a for a in train if a["title"] not in heldout_titles | previous_articles
    ]
    development = sorted(
        eligible, key=lambda a: stable("embedding-study-" + a["title"])
    )[:10]
    squad = {
        "development": decorate_squad(
            prepare_split(
                development,
                tokenizer,
                50,
                10,
                chunk_tokens=128,
                overlap_tokens=24,
                excluded_query_ids=excluded,
            )
        ),
        "test": decorate_squad(
            prepare_split(
                test,
                tokenizer,
                150,
                30,
                chunk_tokens=128,
                overlap_tokens=24,
                excluded_query_ids=excluded,
            )
        ),
    }
    scientific_path = Path("data/embedding-study/scifact.zip")
    scifact = prepare_scifact(scientific_path, tokenizer)
    outputs = {"squad": squad, "scifact": scifact}
    hashes, counts = {}, {}
    for domain, splits in outputs.items():
        if {c["query_id"] for c in splits["development"]["cases"]} & {
            c["query_id"] for c in splits["test"]["cases"]
        }:
            raise ValueError("query split leakage")
        if {c["group_id"] for c in splits["development"]["cases"]} & {
            c["group_id"] for c in splits["test"]["cases"]
        }:
            raise ValueError("source-group split leakage")
        for split, data in splits.items():
            for chunk in data["chunks"]:
                text = f"{chunk['title']}\n{chunk['text']}"
                if len(tokenizer.encode(text)) > 256:
                    raise ValueError("common titled chunk exceeds baseline token limit")
            filename = f"{domain}-{split}.json"
            write_json(args.output / filename, data)
            hashes[filename] = fingerprint(args.output / filename)
            counts[filename] = {
                "chunks": len(data["chunks"]),
                "cases": len(data["cases"]),
                "positive_cases": sum(c["answerable"] for c in data["cases"]),
            }
    if excluded.intersection(c["query_id"] for c in squad["test"]["cases"]):
        raise ValueError("previously inspected question entered the new held-out set")
    write_json(
        args.output / "provenance.json",
        {
            "dataset_hashes": hashes,
            "counts": counts,
            "excluded_previous_query_ids": sorted(excluded),
            "squad_source": "https://rajpurkar.github.io/SQuAD-explorer/",
            "squad_license": "CC-BY-SA-4.0",
            "scifact_source": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
            "scifact_license": "CC-BY-SA-4.0",
            "scifact_attribution": "Wadden et al. 2020; BEIR by Thakur et al. 2021",
            "source_hashes": {
                p.name: fingerprint(p)
                for p in [
                    source / "train-v2.0.json",
                    source / "dev-v2.0.json",
                    scientific_path,
                ]
            },
            "chunking": {
                "tokenizer": MODEL,
                "revision": REVISION,
                "window": 128,
                "overlap": 24,
            },
            "squad_development_articles": [a["title"] for a in development],
            "scope": "Fresh QA subset and label-informed 1000-document scientific corpus with document-level judgments",
        },
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
