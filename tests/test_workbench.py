"""Exercise the workbench's real native retrieval and local HTTP contract."""

import importlib.util
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "workbench_server", ROOT / "tools/serve_workbench.py"
)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


@pytest.fixture(scope="module")
def app():
    return web.Workbench()


@pytest.fixture(scope="module")
def server(app):
    instance = web.make_server(app, 0)
    worker = threading.Thread(target=instance.serve_forever)
    worker.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}"
    finally:
        instance.shutdown()
        instance.server_close()
        worker.join(timeout=5)


def test_exact_native_neighbors_agree_with_independent_cosine(app):
    query = "persistent memory mapped snapshot cache"
    result = app.search({"query": query, "k": 5})
    corpus = app.vectors.astype(np.float64)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    vector = app.embedder.encode([query])[0].astype(np.float64)
    vector /= np.linalg.norm(vector)
    scores = corpus @ vector
    positions = {doc["id"]: row for row, doc in enumerate(app.documents)}
    actual = [scores[positions[hit["id"]]] for hit in result["hits"]]
    np.testing.assert_allclose(
        actual, [hit["score"] for hit in result["hits"]], atol=1e-6
    )
    np.testing.assert_allclose(actual, np.sort(scores)[-5:][::-1], atol=1e-6)
    assert result["scanned"] == len(app.index)
    assert result["recall_at_k"] == 1
    assert len({hit["id"] for hit in result["hits"]}) == 5
    assert all(isinstance(hit["id"], str) for hit in result["hits"])


def test_ivf_full_probe_agrees_and_low_probe_recall_is_measured(app):
    payload = {"query": "IVF centroid candidate search", "k": 5}
    exact = app.search(payload)
    full = app.search({**payload, "mode": "ivf", "nprobe": app.index.nlist})
    assert [r["id"] for r in full["hits"]] == [r["id"] for r in exact["hits"]]
    assert full["scanned"] == len(app.index)
    partial = app.search({**payload, "mode": "ivf", "nprobe": 1})
    expected = (
        len({r["id"] for r in partial["hits"]} & {r["id"] for r in exact["hits"]}) / 5
    )
    assert partial["recall_at_k"] == expected
    assert partial["scanned"] < len(app.index)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"query": ""},
        {"query": "a the"},
        {"query": "x" * 2001},
        {"query": "cache", "k": True},
        {"query": "cache", "k": 100},
        {"query": "cache", "mode": []},
        {"query": "cache", "nprobe": 0},
    ],
)
def test_invalid_search_requests_are_rejected(app, payload):
    with pytest.raises((TypeError, ValueError)):
        app.search(payload)


def test_chunk_sources_ids_and_projection_are_consistent(app):
    bootstrap = app.bootstrap()
    assert bootstrap["collection"]["vectors"] == len(bootstrap["documents"])
    assert {point["id"] for point in bootstrap["projection"]} == {
        doc["id"] for doc in app.documents
    }
    assert all(
        np.isfinite([point["x"], point["y"]]).all() for point in bootstrap["projection"]
    )
    repeated = web.read_documents(ROOT)
    assert [d["id"] for d in repeated] == [d["id"] for d in app.documents]
    for doc in app.documents:
        assert doc["source"] in web.SOURCES
        assert doc["line"] > 0 and doc["words"] > 0


def test_http_live_search_and_static_assets(server, app):
    with urlopen(server + "/api/bootstrap", timeout=10) as response:
        assert json.load(response)["collection"]["vectors"] == len(app.index)
        assert "nosniff" in response.headers["X-Content-Type-Options"]
    request = Request(
        server + "/api/search",
        data=json.dumps({"query": "vector cache storage", "k": 3}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)
    assert len(payload["hits"]) == 3 and payload["scanned"] == len(app.index)
    for route, content_type in [
        ("/", "text/html"),
        ("/styles.css", "text/css"),
        ("/app.js", "text/javascript"),
        ("/architecture.svg", "image/svg+xml"),
        ("/source/README.md", "text/plain"),
    ]:
        with urlopen(server + route, timeout=10) as response:
            assert content_type in response.headers["Content-Type"]
            assert response.read()


@pytest.mark.parametrize(
    "route,headers,body,status",
    [
        ("/source/../pyproject.toml", {}, None, 404),
        ("/source/%2e%2e/.env", {}, None, 404),
        ("/api/bootstrap", {"Host": "unrelated.example"}, None, 403),
        (
            "/api/search",
            {"Origin": "https://unrelated.example", "Content-Type": "application/json"},
            b"{}",
            403,
        ),
        ("/api/search", {"Content-Type": "text/plain"}, b"{}", 415),
        ("/api/search", {"Content-Type": "application/json"}, b"{broken", 400),
        ("/api/search", {"Content-Type": "application/json"}, b"[]", 400),
        ("/api/search", {"Content-Type": "application/json"}, b"x" * 17000, 413),
    ],
)
def test_http_invalid_inputs_and_out_of_scope_paths(
    server, route, headers, body, status
):
    for _ in range(5 if status == 413 else 1):
        request = Request(server + route, data=body, headers=headers)
        with pytest.raises(HTTPError) as caught:
            urlopen(request, timeout=10)
        assert caught.value.code == status
        assert isinstance(json.load(caught.value)["error"], str)


def test_recorded_benchmarks_preserve_scope_and_timing_gap(server):
    with urlopen(server + "/api/benchmarks", timeout=10) as response:
        evidence = json.load(response)
    assert evidence["kind"] == "recorded" and evidence["dimensions"] == 64
    assert [r["vectors"] for r in evidence["runs"]] == [1000000, 10000000]
    assert evidence["runs"][1]["timing_gap_count"] == 1
    assert evidence["diagnostics"][1]["timing_gaps"][0]["nprobe"] == 32
