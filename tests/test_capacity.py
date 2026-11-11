"""Exercise the bounded writer, real native retrieval and evidence rejection."""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from capacity_resources import working_set_limit
from report_capacity import report
from run_capacity import run
from streaming_snapshot import blocks, build_snapshot, layout
from validate_capacity import sha256, validate_archive, validate_run
from vortex import Index


@pytest.mark.parametrize("dimension,chunk", [(7, 11), (8, 32)])
def test_streamed_snapshot_covers_every_row(tmp_path, dimension, chunk):
    count, clusters = 137, 4
    result = build_snapshot(
        tmp_path, count, dimension, clusters, chunk, training_limit=47
    )
    path = Path(result["snapshot"])
    corpus = np.concatenate(
        [values for _, values in blocks(path, count, dimension, chunk)]
    )
    expected = np.random.default_rng(42).standard_normal(
        (count, dimension), dtype=np.float32
    )
    np.testing.assert_array_equal(corpus, expected)
    assert result["corpus_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    index = Index.open(str(path), cache_bytes=8192)
    assert len(index) == count
    positions = layout(count, dimension, clusters)
    with path.open("rb") as handle:
        header = handle.read(128)
        assert struct.unpack_from("<Q", header, 88)[0] == path.stat().st_size
        handle.seek(positions[3])
        centers = np.fromfile(handle, dtype="<f4", count=clusters * dimension).reshape(
            clusters, dimension
        )
        offsets = np.fromfile(handle, dtype="<u8", count=clusters + 1)
        postings = np.fromfile(handle, dtype="<u8", count=count)
    np.testing.assert_array_equal(np.sort(postings), np.arange(count))
    labels = np.argmin(np.sum((corpus[:, None] - centers) ** 2, axis=2), axis=1)
    for c in range(clusters):
        assert np.all(labels[postings[offsets[c] : offsets[c + 1]]] == c)
    for query in np.random.default_rng(99).standard_normal(
        (6, dimension), dtype=np.float32
    ):
        scores = np.sum((corpus.astype(np.float64) - query) ** 2, axis=1)
        expected_ids = np.lexsort((np.arange(count), scores))[:5].tolist()
        for probe in [0, clusters]:
            found = index.search(query, k=5, nprobe=probe)
            assert [h[0] for h in found["hits"]] == expected_ids
            assert found["scanned"] == count
    with pytest.raises(FileExistsError):
        build_snapshot(tmp_path, count, dimension, clusters)


def test_partial_build_is_preserved(tmp_path):
    def stop():
        raise MemoryError("test stop")

    with pytest.raises(MemoryError):
        build_snapshot(tmp_path, 100, 8, 4, check_resources=stop)
    assert not (tmp_path / "collection.vtx").exists()
    assert (tmp_path / "collection.vtx.partial").read_bytes()[:8] != b"VORTEX01"
    with pytest.raises(FileExistsError):
        build_snapshot(tmp_path, 100, 8, 4)


def test_capacity_evidence_rejects_changed_results(tmp_path):
    pytest.importorskip("faiss")
    args = SimpleNamespace(
        size=128,
        dimension=8,
        nlist=4,
        queries=6,
        repeats=2,
        k=5,
        chunk_rows=32,
        output=str(tmp_path),
    )
    assert run(args) == 0
    assert validate_run(tmp_path, check_snapshot=True)["records"] == 36
    result_path = tmp_path / "result.json"
    original = result_path.read_text()
    accounting = json.loads(original)
    peak = max(s["peak_rss_bytes"] for s in accounting["memory_stages"].values())
    accounting["memory_policy"] = {
        "enabled": True,
        "flags": 6,
        "maximum_bytes": peak - 16384,
    }
    result_path.write_text(json.dumps(accounting))
    assert validate_run(tmp_path)["resident_peak_excess_bytes"] == 16384
    accounting["memory_policy"]["maximum_bytes"] = peak - 65537
    result_path.write_text(json.dumps(accounting))
    with pytest.raises(ValueError, match="observed peak exceeds resident bound"):
        validate_run(tmp_path)
    result_path.write_text(original)
    rows = report([tmp_path], tmp_path / "report", tmp_path / "archive")
    assert rows[0]["vectors"] == 128
    assert validate_run(tmp_path / "archive" / "128")["status"] == "passed"
    assert validate_archive(tmp_path / "archive")["runs"][0]["records"] == 36
    assert not (tmp_path / "archive" / "128" / "collection.vtx").exists()
    with pytest.raises(FileExistsError):
        run(args)
    path = tmp_path / "result.json"
    result = json.loads(path.read_text())
    result["exact"]["p95_ms"] += 1
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="p95_ms mismatch"):
        validate_run(tmp_path)
    result["exact"]["p95_ms"] -= 1
    raw = tmp_path / "measurements.jsonl"
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    records[0]["scanned"] -= 1
    raw.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    result["measurements_sha256"] = sha256(raw)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="exact search did not scan N"):
        validate_run(tmp_path)


def test_capacity_memory_guard_preserves_stop(tmp_path, monkeypatch):
    monkeypatch.setattr("run_capacity.available_memory", lambda: 0)
    args = SimpleNamespace(
        size=128,
        dimension=8,
        nlist=4,
        queries=6,
        repeats=1,
        k=5,
        chunk_rows=32,
        output=str(tmp_path),
    )
    assert run(args) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "resource_limited"
    assert "RAM" in result["error"]
    assert not (tmp_path / "collection.vtx.partial").exists()


def test_requested_resident_bound_requires_supported_platform(monkeypatch):
    monkeypatch.setattr("capacity_resources.sys.platform", "linux")
    with pytest.raises(OSError, match="requires Windows"), working_set_limit(1536):
        pytest.fail("unsupported bound was silently accepted")


@pytest.mark.skipif(
    sys.platform != "win32", reason="Windows resident-bound integration"
)
def test_resident_bounded_capacity_cli(tmp_path):
    pytest.importorskip("faiss")
    runner = Path(__file__).resolve().parents[1] / "benchmarks" / "run_capacity.py"
    command = [
        sys.executable,
        str(runner),
        "--size",
        "128",
        "--dimension",
        "8",
        "--nlist",
        "4",
        "--queries",
        "6",
        "--repeats",
        "1",
        "--k",
        "5",
        "--chunk-rows",
        "32",
        "--working-set-mib",
        "256",
        "--output",
        str(tmp_path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["memory_policy"]["maximum_bytes"] == 256 * 1024**2
    assert result["memory_policy"]["flags"] & 0x4
    assert validate_run(tmp_path, check_snapshot=True)["status"] == "passed"
