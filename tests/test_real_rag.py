"""Regression checks for real-data scoring, evidence and checkpoint integrity."""

import importlib.util
import json
from pathlib import Path

import pytest


def module(relative, name):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


helpers = module("python/vortex/real_rag.py", "real_rag_helpers")
report = module("eval/report_real_rag.py", "real_rag_report")


def test_squad_normalization_multiple_references_and_citations():
    scores = helpers.answer_scores("The Eiffel Tower! [32]", ["wrong", "Eiffel Tower"])
    assert scores["squad_exact_match"] == 1
    assert scores["squad_token_f1"] == 1
    assert helpers.normalize("don't") == "dont"
    assert (
        helpers.answer_scores("INSUFFICIENT_EVIDENCE", ["Paris"])["squad_token_f1"] == 0
    )
    assert helpers.answer_scores("INSUFFICIENT_EVIDENCE [2]", [])["abstained"]
    assert not helpers.answer_scores("Maybe INSUFFICIENT_EVIDENCE", [])["abstained"]
    assert helpers.answer_scores("[123]", ["Paris"])["empty_output"]
    assert helpers.answer_scores("[123]", ["Paris"])["squad_token_f1"] == 0


def test_context_enforces_utf8_budget_and_preserves_citation_ids():
    lookup = {1: {"text": "é" * 100}, 2: {"text": "other"}}
    context = helpers.evidence_context([1, 2], lookup, 20)
    assert len(context["context"].encode()) <= 20
    assert context["included_ids"] == [1]
    assert context["truncated_ids"] == [1, 2]
    assert context["context"].startswith("[1] ")


def test_checkpoint_preserves_failures_and_recovers_interrupted_append(tmp_path):
    path = tmp_path / "records.jsonl"
    row = {"condition": "exact", "query_id": "q1", "status": "failed"}
    helpers.append_record(path, row)
    with path.open("ab") as handle:
        handle.write(b'{"interrupted":')
    assert helpers.load_records(path) == [row]
    assert path.with_suffix(".jsonl.interrupted").read_bytes() == b'{"interrupted":'
    helpers.append_record(path, row)
    with pytest.raises(ValueError, match="duplicate"):
        helpers.load_records(path)


def test_exclusive_writer_rejects_concurrent_run(tmp_path):
    with (
        helpers.writer_lock(tmp_path),
        pytest.raises(FileExistsError),
        helpers.writer_lock(tmp_path),
    ):
        pass
    assert not (tmp_path / ".writer.lock").exists()


def test_separate_unanswerable_contract_preserves_legacy_rejection(tmp_path):
    from vortex.evaluation import load_dataset

    data = {
        "schema_version": 2,
        "chunks": [{"id": 1, "text": "evidence"}],
        "cases": [
            {
                "query_id": "q1",
                "query": "question",
                "answerable": False,
                "reference_answers": [],
                "relevant_chunk_ids": [],
                "source_chunk_ids": [1],
            }
        ],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    assert len(helpers.load_real_dataset(path)["cases"]) == 1
    data["schema_version"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid query"):
        load_dataset(path)


def test_cluster_bootstrap_preserves_article_weights_and_pairing():
    rows = [
        {"article": "a", "score": 1},
        {"article": "a", "score": 1},
        {"article": "b", "score": 0},
    ]
    result = report.clustered_interval(rows, "score")
    assert result["mean"] == pytest.approx(2 / 3)
    assert result["articles"] == 2
    assert result["ci95"] == [0, 1]


def test_report_rejects_missing_condition_records(tmp_path):
    from argparse import Namespace

    dataset = tmp_path / "data"
    dataset.mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "selection.json").write_text('{"configurations": {}}')
    (dataset / "test.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "chunks": [{"id": 1, "text": "x"}],
                "cases": [
                    {
                        "query_id": "q",
                        "query": "q",
                        "answerable": True,
                        "reference_answers": ["x"],
                        "relevant_chunk_ids": [1],
                        "source_chunk_ids": [1],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete"):
        report.report(Namespace(dataset=dataset, run=tmp_path, output=tmp_path / "out"))


def test_resume_rejects_changed_contract_without_replacing_manifest(tmp_path):
    from argparse import Namespace

    runner = module("eval/run_real_rag.py", "real_rag_runner")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ["development", "test"]:
        (dataset / f"{split}.json").write_text("{}")
    args = Namespace(dataset=dataset, output=tmp_path, model="model-a")
    runner.bind_run(args)
    manifest = (tmp_path / "manifest.json").read_bytes()
    args.model = "model-b"
    with pytest.raises(ValueError, match="resume contract differs"):
        runner.bind_run(args)
    assert (tmp_path / "manifest.json").read_bytes() == manifest


def test_report_cli_rejects_active_writer_without_repairing_records(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "test" / "generations.jsonl"
    path.parent.mkdir()
    path.write_bytes(b'{"currently_writing":')
    script = Path(__file__).resolve().parents[1] / "eval/report_real_rag.py"
    with helpers.writer_lock(tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--run",
                str(tmp_path),
                "--output",
                str(tmp_path / "report"),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    assert result.returncode != 0
    assert path.read_bytes() == b'{"currently_writing":'
    assert not path.with_suffix(".jsonl.interrupted").exists()


def test_independent_reference_supports_nonconsecutive_stable_ids(tmp_path):
    from argparse import Namespace

    import numpy as np

    runner = module("eval/run_real_rag.py", "real_rag_arbitrary_ids")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "test"
    output.mkdir()
    (dataset / "test.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "chunks": [{"id": 100, "text": "first"}, {"id": 900, "text": "second"}],
                "cases": [
                    {
                        "query_id": "q",
                        "query": "first",
                        "answerable": True,
                        "reference_answers": ["first"],
                        "relevant_chunk_ids": [100],
                        "source_chunk_ids": [100],
                    }
                ],
            }
        )
    )
    np.savez(
        output / "embeddings.npz",
        corpus=np.eye(2, dtype=np.float32),
        queries=np.array([[1, 0]], dtype=np.float32),
        query_ms=[1.0],
    )
    result = runner.search_configuration(
        Namespace(dataset=dataset, output=tmp_path),
        "test",
        {"dtype": "fp32", "nlist": 1, "nprobe": 0, "k": 1, "rerank_n": 0},
    )
    assert result["rows"][0]["ids"] == [100]
    assert result["ann_neighbor_recall"] == 1.0
