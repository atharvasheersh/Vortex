import concurrent.futures
import json
import math
import struct

import numpy as np
import pytest

from vortex import Index
from vortex.evaluation import load_dataset, manifest
from vortex.metrics import answer_metrics, retrieval_metrics
from vortex.rag import FixtureEmbedder, answer_case, build_collection


@pytest.mark.parametrize("dimension", [1, 7, 8, 17, 129])
@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
@pytest.mark.parametrize("kernel", ["scalar", "auto"])
def test_independent_numpy_reference(dimension, metric, kernel, tmp_path):
    rng = np.random.default_rng(194)
    vectors = rng.normal(size=(93, dimension)).astype(np.float32)
    queries = rng.normal(size=(9, dimension)).astype(np.float32)
    ids = np.arange(93, dtype=np.uint64)[::-1] + 100
    index = Index(dimension, metric=metric, kernel=kernel, threads=3)
    for identifier, vector in zip(ids, vectors):
        index.add(int(identifier), vector)
    corpus = vectors.astype(np.float64)
    reference_queries = queries.astype(np.float64)
    if metric == "cosine":
        corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
        reference_queries /= np.linalg.norm(reference_queries, axis=1, keepdims=True)
    index.build_ivf(7, iterations=4)
    for query, ref_query in zip(queries, reference_queries):
        scores = np.sum((corpus - ref_query) ** 2, axis=1) if metric == "l2" else corpus @ ref_query
        # Float32 cosine in one dimension creates mathematical ties at +/-1.
        if metric == "cosine" and dimension == 1:
            scores = np.round(scores, 6)
        order = np.lexsort((ids, scores if metric == "l2" else -scores))[:10]
        out = index.search(query, k=10)
        assert [h[0] for h in out["hits"]] == ids[order].tolist()
        np.testing.assert_allclose([h[1] for h in out["hits"]], scores[order], rtol=3e-5, atol=3e-5)
        assert index.search(query, k=10, nprobe=7)["hits"] == out["hits"]
    path = tmp_path / "vectors.vtx"
    index.save(str(path))
    mapped = Index.open(str(path), cache_bytes=64, block_bytes=31, threads=3, kernel=kernel)
    assert [[h[0] for h in r["hits"]] for r in mapped.search_batch(queries)] == [
        [h[0] for h in r["hits"]] for r in index.search_batch(queries)]
    assert mapped.cache_stats()["resident_bytes"] <= 64


@pytest.mark.parametrize("dtype", ["fp16", "int8"])
def test_compression_and_reranking(dtype, tmp_path):
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(150, 31)).astype(np.float32)
    queries = rng.normal(size=(8, 31)).astype(np.float32)
    index = Index(31)
    for i, vector in enumerate(matrix):
        index.add(i, vector)
    expected = index.search_batch(queries)
    index.build_ivf(9)
    index.set_dtype(dtype)
    path = tmp_path / f"{dtype}.vtx"
    index.save(str(path))
    opened = Index.open(str(path), cache_bytes=128, block_bytes=64)
    assert opened.dtype == dtype
    actual = opened.search_batch(queries, nprobe=9, rerank_n=len(matrix))
    assert [r["hits"] for r in actual] == [r["hits"] for r in expected]
    approximated = opened.search_batch(queries)
    if dtype == "fp16":
        decoded = matrix.astype(np.float16).astype(np.float32)
    else:
        scales = np.maximum(np.abs(matrix).max(axis=1) / np.float32(127), np.finfo(np.float32).tiny)
        decoded = np.clip(np.copysign(np.floor(np.abs(matrix/scales[:,None])+0.5), matrix), -127,127) * scales[:,None]
    for query, row in zip(queries, approximated):
        distances = np.sum((decoded.astype(np.float64)-query) ** 2, axis=1)
        order = np.lexsort((np.arange(len(matrix)), distances))[:10]
        assert [hit[0] for hit in row["hits"]] == order.tolist()


def test_validation_and_mutation(tmp_path):
    index = Index(2)
    vector = np.ones(2, dtype=np.float32)
    index.add(3, vector)
    with pytest.raises(ValueError):
        index.add(3, vector)
    for bad in (np.ones(3, dtype=np.float32), np.array([np.nan, 1], dtype=np.float32)):
        with pytest.raises(ValueError):
            index.search(bad)
    with pytest.raises(TypeError):
        index.search(np.ones(2, dtype=np.float64))
    with pytest.raises(TypeError):
        index.search(np.ones(4, dtype=np.float32)[::2])
    index.build_ivf(1)
    index.add(5, vector)
    with pytest.raises(RuntimeError):
        index.search(vector, nprobe=1)
    with pytest.raises(RuntimeError):
        index.save(str(tmp_path / "stale.vtx"))
    index.build_ivf(1)
    assert [hit[0] for hit in index.search(vector)["hits"]] == [3, 5]


def rewrite_header_checksum(raw):
    value = 14695981039346656037
    for byte in raw[:96]:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    struct.pack_into("<Q", raw, 96, value)


@pytest.mark.parametrize("kind", ["magic", "version", "offset", "huge", "duplicate_id", "list_offset", "posting", "centroid", "reserved", "truncated", "trailing"])
def test_corrupted_snapshots(kind, tmp_path):
    index = Index(3)
    for i in range(12):
        index.add(i, np.array([i, i+1, i+2], dtype=np.float32))
    index.build_ivf(3)
    path = tmp_path / "bad.vtx"
    index.save(str(path))
    raw = bytearray(path.read_bytes())
    if kind == "magic": raw[0] = 0
    elif kind == "version": struct.pack_into("<I", raw, 8, 42)
    elif kind == "offset": struct.pack_into("<Q", raw, 48, 1)
    elif kind == "huge": struct.pack_into("<Q", raw, 24, (1 << 64)-1)
    elif kind == "duplicate_id": raw[136:144] = raw[128:136]
    elif kind == "list_offset": struct.pack_into("<Q", raw, struct.unpack_from("<Q",raw,72)[0], 1)
    elif kind == "posting": struct.pack_into("<Q", raw, struct.unpack_from("<Q",raw,80)[0], 999)
    elif kind == "centroid": struct.pack_into("<f", raw, struct.unpack_from("<Q",raw,64)[0], math.inf)
    elif kind == "reserved": raw[104] = 1
    elif kind == "truncated": raw = raw[:-1]
    elif kind == "trailing": raw += b"x"
    rewrite_header_checksum(raw)
    path.write_bytes(raw)
    with pytest.raises(RuntimeError):
        Index.open(str(path))


def test_concurrent_batches_and_failure_recovery():
    index = Index(3, threads=4)
    for i in range(50):
        index.add(i, np.array([i,1,2],dtype=np.float32))
    queries = np.ones((20,3),dtype=np.float32)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        outputs = list(pool.map(lambda _: index.search_batch(queries),range(16)))
    assert all(out[0]["hits"][0][0] == 1 for out in outputs)
    queries[3,1] = np.nan
    with pytest.raises(ValueError):
        index.search_batch(queries)
    queries[3,1] = 1
    assert len(index.search_batch(queries)) == 20


def test_known_ranking_metrics():
    result = retrieval_metrics([9,2,3], [2,3], 3)
    assert result["precision"] == pytest.approx(2/3)
    assert result["recall"] == 1
    assert result["mrr"] == .5
    expected = (1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3))
    assert result["ndcg"] == pytest.approx(expected)
    assert retrieval_metrics([2], [2,3], 3)["precision"] == pytest.approx(1/3)
    assert retrieval_metrics([], [2], 3)["mrr"] == 0
    assert retrieval_metrics([3,2], [2,3], 2, {2:3,3:1})["ndcg"] < 1
    with pytest.raises(ValueError): retrieval_metrics([2,2],[2],2)
    with pytest.raises(ValueError): retrieval_metrics([2],[],2)
    with pytest.raises(ValueError): retrieval_metrics([2],[2],0)
    with pytest.raises(ValueError): retrieval_metrics([2],[2],1,{2:float("nan")})
    with pytest.raises(ValueError): retrieval_metrics([2],[2],1,{3:1})


def test_answer_diagnostics():
    result = answer_metrics("Seven days.", "seven days", "Exports expire after seven days.")
    assert result == {"exact_match":1,"token_f1":1,"lexical_support":1,"verbatim_sentence_support":1}
    assert answer_metrics("invented claim",None,"seven days")["exact_match"] is None
    assert answer_metrics("", "", "")["lexical_support"] is None
    unsupported = answer_metrics("Exports last eight days.",None,"Exports last seven days. Backups last eight days.")
    assert unsupported["lexical_support"] == 1
    assert unsupported["verbatim_sentence_support"] == 0


def test_offline_rag_and_dataset_contract(tmp_path):
    dataset = load_dataset("eval/datasets/support_v1.json")
    embedder = FixtureEmbedder()
    index, _ = build_collection(dataset["chunks"],embedder,tmp_path/"rag.vtx")
    row = answer_case(index,dataset["cases"][0],dataset["chunks"],embedder)
    assert 101 in [h[0] for h in row["retrieved"]["hits"]]
    assert "30 days" in row["answer"]
    assert row["generator"] == "fixture"
    assert set(row["timings_ms"]) == {"embedding","retrieval","context_lookup","generation","total"}
    dataset["cases"][0]["relevant_chunk_ids"] = [999999]
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError): load_dataset(path)
    assert manifest({},"eval/datasets/support_v1.json")["dataset_sha256"]
