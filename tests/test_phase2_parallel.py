"""Exercise direct mapped scans, deterministic merges and worker failures."""

import concurrent.futures
import struct

import numpy as np
import pytest
from vortex import Index


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
@pytest.mark.parametrize("clustered", [False, True])
def test_direct_workers_match_cached_scores_and_public_ties(
    tmp_path, metric, clustered
):
    rng = np.random.default_rng(822)
    vectors = rng.normal(size=(257, 17)).astype(np.float32)
    vectors[[7, 41, 221]] = vectors[0]
    index = Index(17, metric=metric)
    for row, vector in enumerate(vectors):
        index.add((2**64 - 1) - row, vector)
    index.build_ivf(7, iterations=3)
    index.set_dtype("int8")
    path = tmp_path / "quantized.vtx"
    index.save(str(path), clustered=clustered)
    with pytest.raises(ValueError, match="mapped INT8"):
        index.search(vectors[0], scan_threads=1)
    current = Index.open(str(path), threads=3, cache_bytes=333, block_bytes=31)
    with pytest.raises(ValueError, match="configured threads"):
        current.search(vectors[0], scan_threads=4)
    for query in vectors[:4]:
        for probes in [0, 1, 4, 7]:
            for rerank in [0, 31, 257]:
                expected = current.search(query, nprobe=probes, rerank_n=rerank)
                for workers in [1, 2, 3]:
                    result = current.search(
                        query, nprobe=probes, rerank_n=rerank, scan_threads=workers
                    )
                    assert result["hits"] == expected["hits"]
                    assert result["scanned"] == expected["scanned"]
    expected = current.search(vectors[0], nprobe=4, rerank_n=31)["hits"]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(
                lambda _: current.search(
                    vectors[0], nprobe=4, rerank_n=31, scan_threads=2
                ),
                range(8),
            )
        )
    assert all(result["hits"] == expected for result in results)


def test_direct_worker_error_keeps_pool_usable(tmp_path):
    rng = np.random.default_rng(44)
    vectors = rng.normal(size=(101, 11)).astype(np.float32)
    index = Index(11)
    for row, vector in enumerate(vectors):
        index.add(row, vector)
    index.build_ivf(len(vectors), iterations=2)
    index.set_dtype("int8")
    clean = tmp_path / "clean.vtx"
    broken = tmp_path / "broken.vtx"
    index.save(str(clean), clustered=True)
    raw = bytearray(clean.read_bytes())
    valid = Index.open(str(clean))
    query = vectors[0]
    hit = valid.search(query, nprobe=1)["hits"][0][0]
    ids = np.frombuffer(raw, dtype="<u8", count=len(vectors), offset=128)
    # Corrupt a different row so the next query has a usable list.
    bad_row = int(np.flatnonzero(ids != hit)[-1])
    offset = struct.unpack_from("<Q", raw, 56)[0]
    struct.pack_into("<f", raw, offset + bad_row * 15, float("nan"))
    broken.write_bytes(raw)
    opened = Index.open(str(broken), threads=2)
    with pytest.raises(RuntimeError, match="invalid INT8 scale"):
        opened.search(query, scan_threads=2)
    assert (
        opened.search(query, nprobe=1, scan_threads=2)["hits"]
        == valid.search(query, nprobe=1)["hits"]
    )
