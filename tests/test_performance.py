"""Correctness and compatibility of bounded scans and physical list packing."""

import concurrent.futures
import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vortex import Index

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from encode_snapshot import encode_snapshot
from run_performance import collect_reference, measure

spec = importlib.util.spec_from_file_location(
    "repack_snapshot", ROOT / "benchmarks/repack_snapshot.py"
)
packing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing)


@pytest.mark.parametrize("dimension", [1, 7, 17, 256, 16385])
@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_block_scans_and_packed_stable_ids(dimension, metric, tmp_path):
    rng = np.random.default_rng(72)
    matrix = rng.normal(size=(41, dimension)).astype(np.float32)
    # Duplicate vectors exercise deterministic public-ID ties after reordering.
    matrix[5] = matrix[1]
    identifiers = (
        rng.permutation(len(matrix)).astype(np.uint64) + np.uint64(2**63)
    ).tolist()
    index = Index(dimension, metric=metric)
    for identifier, vector in zip(identifiers, matrix):
        index.add(identifier, vector)
    index.build_ivf(5, iterations=2)
    original, packed = tmp_path / "original.vtx", tmp_path / "packed.vtx"
    index.save(str(original))
    before = packing.sha256(original)
    result = packing.repack(original, packed, chunk_rows=3)
    native_packed = tmp_path / "native-packed.vtx"
    index.save(str(native_packed), clustered=True)
    assert native_packed.read_bytes() == packed.read_bytes()
    assert packing.sha256(original) == before
    assert result["snapshot_bytes"] == original.stat().st_size == packed.stat().st_size
    assert packed.read_bytes()[:128] == original.read_bytes()[:128]
    opened = Index.open(str(packed), cache_bytes=64, block_bytes=31, threads=2)
    queries = np.ascontiguousarray(matrix[[1, 3, 18]])
    for query in queries:
        for probes in [0, 1, 5]:
            assert (
                opened.search(query, k=10, nprobe=probes)["hits"]
                == index.search(query, k=10, nprobe=probes)["hits"]
            )
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        outputs = list(
            pool.map(lambda _: opened.search_batch(queries, nprobe=5), range(5))
        )
    assert all(out[0]["hits"] == index.search(queries[0])["hits"] for out in outputs)
    assert opened.cache_stats()["resident_bytes"] <= 64
    with pytest.raises(FileExistsError):
        packing.repack(original, packed)


@pytest.mark.parametrize("dtype", ["fp16", "int8"])
def test_packing_keeps_encoded_and_original_vectors_associated(dtype, tmp_path):
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(97, 19)).astype(np.float32)
    index = Index(19)
    for row, vector in enumerate(matrix):
        index.add(1000 - row, vector)
    index.build_ivf(7, iterations=3)
    index.set_dtype(dtype)
    original, packed = tmp_path / "original.vtx", tmp_path / "packed.vtx"
    index.save(str(original))
    packing.repack(original, packed, chunk_rows=11)
    left, right = (
        Index.open(str(original), cache_bytes=0),
        Index.open(str(packed), cache_bytes=0),
    )
    native_packed = tmp_path / "native-packed.vtx"
    left.save(str(native_packed), clustered=True)
    assert native_packed.read_bytes() == packed.read_bytes()
    for query in matrix[:4]:
        for probes, rerank in [(0, 0), (3, 0), (7, 97)]:
            assert (
                left.search(query, nprobe=probes, rerank_n=rerank)["hits"]
                == right.search(query, nprobe=probes, rerank_n=rerank)["hits"]
            )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_block_scan_rejects_nonfinite_payload_on_later_tile(value, tmp_path):
    index = Index(17)
    vector = np.ones(17, dtype=np.float32)
    for identifier in range(1200):
        index.add(identifier, vector)
    path = tmp_path / "corrupt.vtx"
    index.save(str(path))
    with path.open("r+b") as handle:
        raw = handle.read(128)
        start = struct.unpack_from("<Q", raw, 48)[0]
        handle.seek(start + 1199 * 17 * 4)
        handle.write(struct.pack("<f", value))
    opened = Index.open(str(path), cache_bytes=0)
    with pytest.raises(RuntimeError, match="non-finite"):
        opened.search(vector)


def test_repacking_preserves_partial_failure_and_rejects_flat(tmp_path):
    index = Index(3)
    for identifier in range(12):
        index.add(identifier, np.full(3, identifier, dtype=np.float32))
    source = tmp_path / "source.vtx"
    index.save(str(source))
    with pytest.raises(ValueError, match="IVF"):
        packing.repack(source, tmp_path / "flat.vtx")
    index.build_ivf(3, iterations=2)
    index.save(str(source))
    calls = 0

    def resource_stop():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise MemoryError("test resource guard")

    destination = tmp_path / "stop.vtx"
    with pytest.raises(MemoryError):
        packing.repack(source, destination, chunk_rows=2, check_resources=resource_stop)
    assert not destination.exists()
    assert destination.with_name("stop.vtx.partial").exists()


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
@pytest.mark.parametrize("kernel", ["scalar", "auto"])
def test_direct_int8_scores_match_decoded_reference(metric, kernel, tmp_path):
    rng = np.random.default_rng(381)
    matrix = rng.normal(size=(71, 19)).astype(np.float32)
    index = Index(19, metric=metric, kernel=kernel)
    if metric == "cosine":
        matrix = (
            matrix / np.linalg.norm(matrix.astype(np.float64), axis=1)[:, None]
        ).astype(np.float32)
    for row, vector in enumerate(matrix):
        index.add(row, vector)
    index.build_ivf(7, iterations=2)
    index.set_dtype("int8")
    path = tmp_path / "quantized.vtx"
    index.save(str(path), clustered=True)
    # Read exactly the quantized scale/code bytes, independent of their physical
    # order or the original vectors retained for reranking.
    raw = path.read_bytes()
    ids = np.frombuffer(raw, dtype="<u8", count=71, offset=128)
    encoded = struct.unpack_from("<Q", raw, 56)[0]
    decoded = np.empty((71, 19), dtype=np.float64)
    for row in range(71):
        scale = struct.unpack_from("<f", raw, encoded + row * 23)[0]
        codes = np.frombuffer(
            raw, dtype=np.int8, count=19, offset=encoded + row * 23 + 4
        )
        decoded[row] = (np.float32(scale) * codes).astype(np.float64)
    mapped = Index.open(str(path), cache_bytes=113, block_bytes=31, kernel=kernel)
    for vector in matrix[:3]:
        query = vector.astype(np.float64)
        if metric == "cosine":
            query = (
                (query / np.linalg.norm(query)).astype(np.float32).astype(np.float64)
            )
        scores = (
            np.sum((decoded - query) ** 2, axis=1)
            if metric == "l2"
            else decoded @ query
        )
        order = np.lexsort((ids, scores if metric == "l2" else -scores))[:10]
        found = mapped.search(vector)
        assert [h[0] for h in found["hits"]] == ids[order].tolist()
        np.testing.assert_allclose(
            [h[1] for h in found["hits"]], scores[order], atol=1e-5, rtol=1e-5
        )
        assert mapped.search(vector, nprobe=7)["hits"] == found["hits"]


@pytest.mark.parametrize(
    "scale", [0, -1, float("nan"), float("inf"), np.finfo(np.float32).max]
)
def test_direct_int8_rejects_corrupt_scale_and_overflow(scale, tmp_path):
    index = Index(9)
    index.add(7, np.ones(9, dtype=np.float32))
    index.set_dtype("int8")
    path = tmp_path / "bad-scale.vtx"
    index.save(str(path))
    with path.open("r+b") as handle:
        raw = handle.read(128)
        offset = struct.unpack_from("<Q", raw, 56)[0]
        handle.seek(offset)
        handle.write(struct.pack("<f", scale))
    opened = Index.open(str(path))
    with pytest.raises((RuntimeError, OverflowError)):
        opened.search(np.ones(9, dtype=np.float32))


@pytest.mark.parametrize("dtype", ["fp16", "int8"])
def test_bounded_encoder_matches_native_bytes_and_preserves_original(dtype, tmp_path):
    rng = np.random.default_rng(391)
    matrix = rng.normal(size=(107, 17)).astype(np.float32)
    matrix[0] = 0
    index = Index(17)
    for row, vector in enumerate(matrix):
        index.add(900 - row, vector)
    index.build_ivf(7, iterations=3)
    original, encoded, native = (
        tmp_path / name for name in ["source.vtx", "encoded.vtx", "native.vtx"]
    )
    index.save(str(original), clustered=True)
    before = packing.sha256(original)
    result = encode_snapshot(original, encoded, dtype, chunk_rows=3)
    index.set_dtype(dtype)
    index.save(str(native), clustered=True)
    assert encoded.read_bytes() == native.read_bytes()
    assert packing.sha256(original) == before
    assert result["retains_fp32_originals"]
    assert result["snapshot_bytes"] > original.stat().st_size
    opened = Index.open(str(encoded))
    full = Index.open(str(original))
    assert (
        opened.search(matrix[0], nprobe=7, rerank_n=107)["hits"]
        == full.search(matrix[0])["hits"]
    )
    with pytest.raises(FileExistsError):
        encode_snapshot(original, encoded, dtype)


def test_scalar_collector_uses_independent_reference_and_correct_diagnostics(tmp_path):
    import json

    rng = np.random.default_rng(782)
    index = Index(7)
    for row, vector in enumerate(rng.normal(size=(41, 7)).astype(np.float32)):
        index.add(row, vector)
    index.build_ivf(5, iterations=2)
    snapshot = tmp_path / "source.vtx"
    index.save(str(snapshot), clustered=True)
    reference = tmp_path / "reference"
    collect_reference(snapshot, reference, count=33)
    output = tmp_path / "measurement"
    measure(
        SimpleNamespace(
            module_dir=None,
            snapshot=snapshot,
            reference=reference,
            output=output,
            query_offset=32,
            query_count=1,
            repeats=1,
            probes="0,5",
            threads=1,
            kernel="scalar",
            max_seconds=30,
            working_set_mib=0,
            condition="scalar-fixture",
            round_id=0,
            rerank_n=0,
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["active_kernel"] == "scalar"
    summaries = json.loads((output / "summary.json").read_text())
    assert len(summaries) == 2 and all(r["recall_at_10"] == 1 for r in summaries)
