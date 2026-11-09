"""Comparison regressions independent of model downloads or hosted credentials."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from vortex import Index
from vortex.embedding_models import (
    StudyEncoder,
    checked_vectors,
    formatted_inputs,
    ordered_voyage_vectors,
)
from vortex.embedding_study import (
    bind_contract,
    check_exact,
    exact_units,
    group_interval,
    load_batch,
    load_study,
    save_batch,
    search_units,
)
from vortex.real_rag import append_record, load_records, writer_lock


def test_document_information_and_task_formatting():
    rows = [{"title": "Science", "text": "Evidence"}]
    assert formatted_inputs("minilm", rows, "document", "squad") == [
        "Science\nEvidence"
    ]
    assert formatted_inputs("embeddinggemma", rows, "document", "squad") == [
        "title: Science | text: Evidence"
    ]
    assert formatted_inputs(
        "embeddinggemma", [{"query": "Claim"}], "query", "scifact"
    ) == ["task: fact checking | query: Claim"]
    assert formatted_inputs("qwen", [{"query": "Why?"}], "query", "squad")[0].endswith(
        "\nQuery: Why?"
    )
    assert formatted_inputs("jina", rows, "document", "scifact") == [
        "Science\nEvidence"
    ]
    with pytest.raises(ValueError):
        formatted_inputs("minilm", rows, "other", "squad")


def test_hosted_response_order_and_invalid_indices():
    assert ordered_voyage_vectors(
        {"data": [{"index": 1, "embedding": [2]}, {"index": 0, "embedding": [1]}]}, 2
    ) == [[1], [2]]
    for rows in [[{"index": 0}, {"index": 0}], [{"index": 2}], []]:
        with pytest.raises(ValueError):
            ordered_voyage_vectors({"data": rows}, 2)


@pytest.mark.parametrize(
    "values", [[[0, 0]], [[float("nan"), 1]], [[1]], [[float("inf"), 1]]]
)
def test_invalid_vectors(values):
    with pytest.raises(ValueError):
        checked_vectors(values, 1, 2)


def test_cache_contract_and_corruption(tmp_path):
    bind_contract(tmp_path, {"revision": "a", "dataset": "b"})
    with pytest.raises(ValueError, match="contract differs"):
        bind_contract(tmp_path, {"revision": "c", "dataset": "b"})
    save_batch(tmp_path, "batch", np.array([[1, 0]], dtype=np.float32), {"ids": [97]})
    assert load_batch(tmp_path, "batch", 1, 2)[1]["ids"] == [97]
    (tmp_path / "batch.npy").write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_batch(tmp_path, "batch", 1, 2)


def test_orphaned_batch_recomputed(tmp_path):
    np.save(tmp_path / "batch.npy", np.array([[1, 0]]))
    assert load_batch(tmp_path, "batch", 1, 2) is None


def test_interrupted_metadata_is_not_a_completed_checkpoint(tmp_path, monkeypatch):
    import vortex.embedding_study as study

    original = study.os.fsync
    calls = []

    def interrupted(descriptor):
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError("simulated interrupted metadata flush")
        return original(descriptor)

    monkeypatch.setattr(study.os, "fsync", interrupted)
    with pytest.raises(OSError):
        save_batch(
            tmp_path, "batch", np.array([[1, 0]], dtype=np.float32), {"ids": [97]}
        )
    assert load_batch(tmp_path, "batch", 1, 2) is None
    assert (tmp_path / "batch.json.tmp").exists()
    monkeypatch.setattr(study.os, "fsync", original)
    save_batch(tmp_path, "batch", np.array([[1, 0]], dtype=np.float32), {"ids": [97]})
    assert load_batch(tmp_path, "batch", 1, 2)[1]["ids"] == [97]


def test_exact_reference_checks_ranking_order():
    with pytest.raises(AssertionError, match="ranking order"):
        check_exact(
            {"ids": [97, 42], "scores": [[97, 0.5], [42, 1.0]]},
            [42, 97],
            {42: 1.0, 97: 0.5},
        )


def test_arbitrary_ids_and_document_overfetch():
    vectors = np.array([[1, 0], [0.99, 0.01], [0.9, 0.1], [0, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = [
        {"id": i, "unit_id": u} for i, u in [(97, 80), (1003, 80), (9001, 80), (42, 22)]
    ]
    index = Index(2, metric="cosine")
    for c, v in zip(chunks, vectors):
        index.add(c["id"], v)
    config = {"nprobe": 0, "rerank_multiplier": 0}
    found = search_units(index, vectors[0], chunks, 2, config)
    assert found["ids"] == [80, 22]
    assert found["search_calls"] == 2
    expected, scores = exact_units(vectors.astype(np.float64), vectors[0], chunks, 2)
    check_exact(found, expected, scores)
    found["scores"][0][1] = 0.4
    with pytest.raises(AssertionError, match="scores disagree"):
        check_exact(found, expected, scores)


def test_interrupted_append_and_exclusive_writer(tmp_path):
    path = tmp_path / "records.jsonl"
    with writer_lock(tmp_path):
        append_record(path, {"condition": "a", "query_id": "q"})
        with path.open("ab") as handle:
            handle.write(b'{"unfinished":')
        assert len(load_records(path)) == 1
        assert path.with_suffix(".jsonl.interrupted").exists()
        with pytest.raises(FileExistsError), writer_lock(tmp_path):
            pass


def test_group_bootstrap_is_paired_and_deterministic():
    rows = [{"group_id": "a", "delta": 0.2}, {"group_id": "b", "delta": 0.2}]
    assert group_interval(rows, "delta")["ci95"] == [0.2, 0.2]
    assert group_interval(rows, "delta") == group_interval(rows, "delta")


def test_missing_relevance_rejected(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "chunks": [{"id": 97, "unit_id": 5}],
                "cases": [
                    {
                        "query_id": "q",
                        "query": "Q",
                        "answerable": True,
                        "relevant_ids": [6],
                        "source_chunk_ids": [97],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="relevant units"):
        load_study(path)


def test_normal_heldout_cli_and_resume(tmp_path):
    """Synthetic vectors exercise the actual CLI, snapshots and report inputs."""
    root = Path(__file__).resolve().parents[1]
    dataset, run = tmp_path / "data", tmp_path / "run"
    dataset.mkdir()
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"synthetic_cli_smoke": True}))
    vectors = np.eye(12, dtype=np.float32)
    data = {
        "schema_version": 3,
        "chunks": [
            {"id": 97 + i * 31, "unit_id": 5 + i, "text": f"Evidence {i}"}
            for i in range(12)
        ],
        "cases": [
            {
                "query_id": "q",
                "query": "Evidence?",
                "group_id": "g",
                "answerable": True,
                "relevant_ids": [5],
                "source_chunk_ids": [97],
            }
        ],
    }
    exact = {"dtype": "fp32", "nlist": 0, "nprobe": 0, "rerank_multiplier": 0}
    selections = {}
    import hashlib

    for domain in ["squad", "scifact"]:
        for split in ["development", "test"]:
            (dataset / f"{domain}-{split}.json").write_text(json.dumps(data))
    for model in ["minilm", "qwen", "jina", "embeddinggemma"]:
        selections[model] = {}
        for domain in ["squad", "scifact"]:
            selections[model][domain] = {"config": exact}
            directory = run / model / f"{domain}-test"
            directory.mkdir(parents=True)
            np.savez(directory / "embeddings.npz", corpus=vectors, queries=vectors[:1])
            (directory / "embedding.json").write_text(
                json.dumps(
                    {
                        "embedding_sha256": hashlib.sha256(
                            (directory / "embeddings.npz").read_bytes()
                        ).hexdigest()
                    }
                )
            )
    (run / "selection.json").write_text(json.dumps({"selections": selections}))
    command = [
        sys.executable,
        str(root / "eval/run_embedding_study.py"),
        "heldout",
        "--dataset",
        str(dataset),
        "--run",
        str(run),
    ]
    subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=60)
    result = json.loads((run / "qwen/scifact-test/exact-k10.json").read_text())
    assert result["rows"][0]["ids"][0] == 5
    assert result["timing_observations"] == 3
    before = (run / "contract.json").read_bytes()
    subprocess.run(command, cwd=root, check=True, capture_output=True, timeout=60)
    assert (run / "contract.json").read_bytes() == before


def test_hosted_credentials_and_zero_budget_never_request(monkeypatch):
    monkeypatch.delenv("STUDY_TEST_KEY", raising=False)
    spec = {
        "backend": "voyage",
        "id": "voyage-4-large",
        "dimensions": 2,
        "environment": "STUDY_TEST_KEY",
    }
    encoder = StudyEncoder("voyage", spec)
    with pytest.raises(ValueError, match="credentials"):
        encoder.encode([{"query": "q"}], "query", "squad")
    monkeypatch.setenv("STUDY_TEST_KEY", "test-value")
    with pytest.raises(ValueError, match="spending cap"):
        encoder.encode([{"query": "q"}], "query", "squad")
    assert encoder.calls == []


def test_shared_corpus_reuse_preserves_ids_and_timing_provenance(tmp_path, monkeypatch):
    import importlib.util
    from argparse import Namespace

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "study_runner_test", root / "eval/run_embedding_study.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    dataset = tmp_path / "data"
    dataset.mkdir()
    data = {
        "schema_version": 3,
        "chunks": [{"id": 97, "unit_id": 5, "text": "Evidence", "title": "Title"}],
        "cases": [
            {
                "query_id": "q",
                "query": "Question",
                "group_id": "g",
                "answerable": True,
                "relevant_ids": [5],
                "source_chunk_ids": [97],
            }
        ],
    }
    for domain in ["squad", "scifact"]:
        for split in ["development", "test"]:
            (dataset / f"{domain}-{split}.json").write_text(json.dumps(data))

    class FakeEncoder:
        document_items = 0

        def __init__(self, *args):
            pass

        def encode(self, items, role, domain):
            if role == "document":
                FakeEncoder.document_items += len(items)
            return np.tile(np.array([[1, 0]], dtype=np.float32), (len(items), 1))

    monkeypatch.setattr(runner, "StudyEncoder", FakeEncoder)
    monkeypatch.setattr(
        runner,
        "registry",
        lambda: {
            "minilm": {
                "backend": "local",
                "batch_size": 4,
                "dimensions": 2,
                "dtype": "float32",
            }
        },
    )
    monkeypatch.setattr(runner, "bind", lambda args: None)
    runner.embed(Namespace(dataset=dataset, run=tmp_path / "run", model="minilm"))
    assert FakeEncoder.document_items == 2
    metadata = json.loads(
        (tmp_path / "run/minilm/scifact-test/embedding.json").read_text()
    )
    assert metadata["reused_document_batches"] == 1
    assert metadata["new_encoding_seconds"] == 0
    assert metadata["corpus_seconds"] > 0


def test_timing_audit_grid_resume_and_primary_writer_guard(tmp_path, monkeypatch):
    import importlib.util
    from argparse import Namespace
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "eval"))
    spec = importlib.util.spec_from_file_location(
        "timing_audit_test", root / "eval/run_embedding_timing_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    dataset, primary = tmp_path / "data", tmp_path / "primary"
    dataset.mkdir()
    primary.mkdir()
    (primary / "contract.json").write_text("{}")
    data = {
        "schema_version": 3,
        "chunks": [
            {"id": 97 + i, "unit_id": i, "text": f"Evidence {i}"} for i in range(16)
        ],
        "cases": [
            {
                "query_id": str(i),
                "query": f"Question {i}",
                "group_id": str(i),
                "answerable": True,
                "relevant_ids": [i],
                "source_chunk_ids": [97 + i],
            }
            for i in range(10)
        ],
    }
    for domain in ["squad", "scifact"]:
        (dataset / f"{domain}-development.json").write_text(json.dumps(data))

    class FakeEncoder:
        def __init__(self, *args):
            self.model = SimpleNamespace(
                parameters=lambda: [
                    SimpleNamespace(dtype="torch.float32", numel=lambda: 2)
                ]
            )

        def encode(self, items, role, domain):
            return np.tile(np.array([[1, 0]], dtype=np.float32), (len(items), 1))

    monkeypatch.setattr(audit, "StudyEncoder", FakeEncoder)
    monkeypatch.setattr(
        audit, "registry", lambda: {"minilm": {"dtype": "float32", "backend": "local"}}
    )
    monkeypatch.setattr(
        audit,
        "power_state",
        lambda: {"available": True, "ac_line_status": 1, "battery_saver": 0},
    )
    args = Namespace(
        dataset=dataset, primary=primary, run=tmp_path / "audit", model="minilm"
    )
    with writer_lock(primary), pytest.raises(FileExistsError):
        audit.run(args)
    audit.run(args)
    rows = load_records(args.run / "minilm/timings.jsonl")
    assert len(rows) == 84
    assert all(r["wall_seconds"] >= 0 and r["process_cpu_seconds"] >= 0 for r in rows)
    before = (args.run / "minilm/timings.jsonl").read_bytes()
    audit.run(args)
    assert (args.run / "minilm/timings.jsonl").read_bytes() == before
    result = json.loads((args.run / "minilm/summary.json").read_text())
    assert result["matches_requested_fp32"] and result["parameter_count"] == 2


def test_source_span_survival_through_utf8_context_truncation(tmp_path, monkeypatch):
    import hashlib

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "eval"))
    from report_embedding_study import annotated_span_coverage
    from vortex.evaluation import fingerprint
    from vortex.real_rag import evidence_context

    context = "é" * 100 + "answer" + "x" * 100
    source = hashlib.sha256(("A\0" + context).encode()).hexdigest()
    raw = {
        "data": [
            {
                "title": "A",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": "q",
                                "question": "Q?",
                                "answers": [{"text": "answer", "answer_start": 100}],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw))
    chunks = [
        {
            "id": 97,
            "text": "noise" * 596,
            "source_id": "other",
            "start": 0,
            "end": 2980,
        },
        {
            "id": 42,
            "text": context,
            "source_id": source,
            "start": 0,
            "end": len(context),
        },
    ]
    data = {
        "chunks": chunks,
        "cases": [{"query_id": "q", "source_id": source, "relevant_ids": [42]}],
        "exclusions": [],
    }
    lookup = {c["id"]: c for c in chunks}
    rows = [
        {"query_id": "q", "requested_ids": ids, **evidence_context(ids, lookup, 3000)}
        for ids in [[97, 42], [42, 97], []]
    ]
    annotated, _ = annotated_span_coverage(rows, data, raw_path, fingerprint(raw_path))
    assert [r["annotated_evidence_requested"] for r in annotated] == [True, True, False]
    assert [r["annotated_answer_span_survives"] for r in annotated] == [
        False,
        True,
        False,
    ]
