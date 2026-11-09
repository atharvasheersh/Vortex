"""Validated study data, immutable contracts and recoverable vector batches."""

import json
import os
from pathlib import Path

import numpy as np

from .embedding_models import checked_vectors
from .evaluation import fingerprint, write_json
from .real_rag import digest_object


def load_study(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 3 or not data["chunks"] or not data["cases"]:
        raise ValueError("expected nonempty study schema 3")
    ids = [c["id"] for c in data["chunks"]]
    units = {c["unit_id"] for c in data["chunks"]}
    if len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("invalid stable chunk IDs")
    seen = set()
    for case in data["cases"]:
        if case["query_id"] in seen or not case["query"].strip():
            raise ValueError("duplicate query or empty text")
        seen.add(case["query_id"])
        if not set(case["relevant_ids"]).issubset(units):
            raise ValueError("missing relevant units")
        if bool(case["relevant_ids"]) != case["answerable"]:
            raise ValueError("inconsistent answerability")
        if not set(case["source_chunk_ids"]).issubset(ids):
            raise ValueError("missing source chunks")
    return data


def bind_contract(directory, contract):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("resume contract differs; choose a new directory")
    else:
        write_json(path, contract)
    return digest_object(contract)


def save_batch(directory, name, vectors, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npy"
    temporary = directory / f"{name}.npy.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    record = {**metadata, "sha256": fingerprint(path)}
    temporary_metadata = directory / f"{name}.json.tmp"
    with temporary_metadata.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_metadata.replace(path.with_suffix(".json"))


def load_batch(directory, name, count, dimension):
    path = Path(directory) / f"{name}.npy"
    metadata = path.with_suffix(".json")
    if not path.exists() or not metadata.exists():
        return None
    record = json.loads(metadata.read_text())
    if fingerprint(path) != record["sha256"]:
        raise ValueError("checkpoint vector hash mismatch")
    values = np.load(path, allow_pickle=False)
    checked_vectors(values, count, dimension)
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
        raise ValueError("checkpoint vectors are not normalized")
    return values, record


def collapse_hits(hits, units, k):
    best = {}
    for identifier, score in hits:
        unit = units[int(identifier)]
        best[unit] = max(best.get(unit, -float("inf")), float(score))
    return sorted(best, key=lambda u: (-best[u], u))[:k], best


def search_units(index, query, chunks, k, config):
    units = {c["id"]: c["unit_id"] for c in chunks}
    requested, calls = min(k, len(chunks)), 0
    while True:
        rerank = min(len(chunks), requested * config["rerank_multiplier"])
        found = index.search(
            query, k=requested, nprobe=config["nprobe"], rerank_n=rerank
        )
        calls += 1
        ids, scores = collapse_hits(found["hits"], units, k)
        if len(ids) >= k or requested == len(chunks) or len(found["hits"]) < requested:
            return {
                "ids": ids,
                "scores": [[i, scores[i]] for i in ids],
                "chunk_ids": [int(h[0]) for h in found["hits"]],
                "search_calls": calls,
                "effective_chunk_k": requested,
                "rerank_n": rerank,
            }
        requested = min(len(chunks), requested * 2)


def exact_units(vectors, query, chunks, k):
    q = query.astype(np.float64)
    q /= np.linalg.norm(q)
    scores = vectors @ q
    ids, best = collapse_hits(
        zip([c["id"] for c in chunks], scores),
        {c["id"]: c["unit_id"] for c in chunks},
        k,
    )
    return ids, best


def check_exact(found, expected, scores, tolerance=1e-5):
    if len(found["ids"]) != len(expected) or len(set(found["ids"])) != len(expected):
        raise AssertionError(
            "exact retrieval returned the wrong number of unique units"
        )
    threshold = min(scores[i] for i in expected)
    if any(
        scores[a] < scores[b] - tolerance
        for a, b in zip(found["ids"], found["ids"][1:])
    ):
        raise AssertionError("exact ranking order disagrees with independent cosine")
    if any(scores[i] < threshold - tolerance for i in found["ids"]):
        raise AssertionError("exact IDs disagree with independent cosine")
    if any(abs(score - scores[i]) > tolerance for i, score in found["scores"]):
        raise AssertionError("exact scores disagree with independent cosine")


def group_interval(rows, field, seed=42):
    if not rows:
        return None
    groups = sorted({r["group_id"] for r in rows})
    totals = np.array(
        [sum(r[field] for r in rows if r["group_id"] == g) for g in groups]
    )
    counts = np.array([sum(r["group_id"] == g for r in rows) for g in groups])
    sampled = np.random.default_rng(seed).integers(
        len(groups), size=(2000, len(groups))
    )
    means = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "mean": float(np.mean([r[field] for r in rows])),
        "ci95": np.percentile(means, [2.5, 97.5]).tolist(),
        "n": len(rows),
        "groups": len(groups),
    }
