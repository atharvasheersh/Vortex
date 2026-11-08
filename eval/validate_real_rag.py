"""Verify public aggregate hashes and, optionally, private per-query evidence."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vortex.evaluation import fingerprint
from vortex.real_rag import CITATION, answer_scores, digest_object, load_real_dataset


def validate(archive, run=None, dataset=None):
    checksums = json.loads((archive / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = (archive / name).resolve()
        if not path.is_relative_to(archive.resolve()) or fingerprint(path) != expected:
            raise ValueError(f"archive checksum mismatch: {name}")
    summary = json.loads((archive / "summary.json").read_text())
    conditions = summary["conditions"]
    if set(conditions) != {"no_context", "oracle", "exact", "ivf", "encoded"}:
        raise ValueError("unexpected condition set")
    if len({r["records"] for r in conditions.values()}) != 1:
        raise ValueError("unpaired condition counts")
    if run is None:
        return
    for filename, field in [
        ("test/generations.jsonl", "record_sha256"),
        ("manifest.json", "manifest_sha256"),
        ("selection.json", "selection_sha256"),
        ("generator.json", "generator_sha256"),
    ]:
        if fingerprint(run / filename) != summary[field]:
            raise ValueError(f"run evidence checksum mismatch: {filename}")
    rows = [
        json.loads(line)
        for line in (run / "test/generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    keys = [(r["query_id"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result rows")
    contract = json.loads((run / "manifest.json").read_text())["contract"]
    if fingerprint(archive / "runner_snapshot.txt") != contract["runner_sha256"]:
        raise ValueError("historical collector mismatch")
    for row in rows:
        prompt = contract["prompt"].format(context=row["context"], query=row["query"])
        if digest_object(prompt) != row["prompt_sha256"]:
            raise ValueError("saved prompt and submitted context mismatch")
        scores = answer_scores(row["answer"], row["reference_answers"])
        if row["status"] != "ok" or scores["empty_output"]:
            scores.update(squad_exact_match=0.0, squad_token_f1=0.0)
        if scores != row["scores"]:
            raise ValueError("answer scoring mismatch")
        citations = [int(x) for x in CITATION.findall(row["answer"])]
        if citations != row["citations"] or row["valid_citations"] != sum(
            c in row["included_ids"] for c in citations
        ):
            raise ValueError("citation validity mismatch")
        if len(row["context"].encode("utf-8")) > 3000:
            raise ValueError("evidence budget exceeded")
    for condition, result in conditions.items():
        part = [r for r in rows if r["condition"] == condition]
        positive = [r for r in part if r["answerable"]]
        negative = [r for r in part if not r["answerable"]]
        if (
            len(part) != result["records"]
            or len(positive) != result["answerable"]
            or len(negative) != result["source_unanswerable"]
        ):
            raise ValueError("aggregate count mismatch")
        if sum(r["status"] != "ok" for r in part) != result["request_failures"]:
            raise ValueError("failure accounting mismatch")
        for field, key in [
            ("squad_token_f1", "squad_token_f1_answerable"),
            ("squad_exact_match", "squad_exact_match_answerable"),
        ]:
            if (
                abs(
                    sum(r["scores"][field] for r in positive) / len(positive)
                    - result[key]["mean"]
                )
                > 1e-12
            ):
                raise ValueError("aggregate answer score mismatch")
    if dataset is not None:
        for split in ["development", "test"]:
            if (
                fingerprint(dataset / f"{split}.json")
                != contract["dataset_hashes"][split]
            ):
                raise ValueError("prepared dataset mismatch")
        data = load_real_dataset(dataset / "test.json")
        expected = {
            (c["query_id"], condition)
            for c in data["cases"]
            for condition in conditions
        }
        if set(keys) != expected:
            raise ValueError("question grid mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("eval/reference/real-rag"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    validate(args.archive, args.run, args.dataset)
    print("Real RAG evidence checks passed")
