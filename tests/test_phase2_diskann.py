"""Check the benchmark adapter where the optional Windows wheel is installed."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from vortex import Index

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from phase2_diskann import Graph, build


@pytest.mark.skipif(sys.platform != "win32", reason="Windows disk graph study")
def test_disk_graph_round_trip_with_uint64_ids(tmp_path):
    pytest.importorskip("diskannpy")
    rng = np.random.default_rng(739)
    matrix = rng.standard_normal((768, 64), dtype=np.float32)
    ids = rng.permutation(len(matrix)).astype(np.uint64) + np.uint64(2**63)
    index = Index(64)
    for identifier, vector in zip(ids, matrix):
        index.add(int(identifier), vector)
    snapshot = tmp_path / "original.vtx"
    index.save(str(snapshot))
    target = tmp_path / "graph"
    metadata = build(snapshot, target, 16, 32, 2, 0.25, 0.01)
    assert metadata["status"] == "completed"
    with pytest.raises(FileExistsError):
        build(snapshot, target, 16, 32, 2, 0.25, 0.01)
    reopened = Graph(target)
    for query in matrix[:3]:
        scores = np.sum((matrix.astype(np.float64) - query) ** 2, axis=1)
        neighbors = np.lexsort((ids, scores))[:10]
        hits = reopened.search(query, 10, 128, 2)
        assert [h[0] for h in hits] == ids[neighbors].tolist()
        np.testing.assert_allclose(
            [h[1] for h in hits], scores[neighbors], atol=1e-4, rtol=1e-5
        )
    assert reopened.search_batch(matrix[:3], 10, 128, 2, threads=2) == [
        reopened.search(query, 10, 128, 2) for query in matrix[:3]
    ]
    with pytest.raises(ValueError, match="finite float32"):
        reopened.search(np.full(64, np.nan, dtype=np.float32), 10, 128, 2)
    del reopened
    manifest = target / "build.json"
    manifest.write_text(
        json.dumps({**metadata, "status": "incomplete"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete"):
        Graph(target)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with (target / "ids.u64").open("r+b") as file:
        file.truncate(1)
    with pytest.raises(ValueError, match="unexpected size"):
        Graph(target)
