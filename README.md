# Vortex

A C++20 vector retrieval engine with custom exact search, IVF indexing, persistent mapped snapshots, and a reproducible evaluation harness. Python bindings connect the engine to a small RAG example.

Vortex implements storage, distance kernels, top-k selection, k-means, IVF lists, quantization, and caching itself. FAISS is used only by the benchmark scripts.

![Vortex system architecture](assets/system-architecture.png)

## Capabilities

| Area | Implemented behavior |
| --- | --- |
| Similarity | Squared L2, inner product, normalized cosine; stable ID tie-breaking |
| Search | Exhaustive search, seeded IVF, tunable nprobe, optional original-FP32 reranking |
| CPU | Scalar reference; runtime-dispatched AVX2 on x86 GCC/Clang; reusable batch worker pool |
| Persistence | Versioned little-endian single-file snapshot, validated metadata, read-only file mapping |
| Storage access | Windows file mapping and POSIX mmap; bounded LRU block cache |
| Representation | FP32, FP16, per-vector symmetric INT8; original FP32 vectors retained |
| Python | Strict contiguous float32 inputs; single and batch search; cache statistics |
| Evaluation | Labeled retrieval metrics, per-query evidence, lexical answer diagnostics, run manifests |
| Integration | Offline fixture; optional pinned sentence-transformer embeddings and external HTTP generation |

## Measured release results

One local Windows Release run: Intel Core i7-1255U, GNU 16.1.0, 10,000 vectors x 64 dimensions, 100 queries, k=10, 32 lists, one thread, seed 42. These in-memory query timings include Python binding overhead; mapped/cache experiments are reported separately. This small synthetic run does not establish general superiority or production capacity.

| Configuration | ANN Recall@10 | p95 ms |
| --- | ---: | ---: |
| Vortex exact FP32 | 1.000 | 0.103 |
| Vortex IVF nprobe=4 | 0.374 | 0.031 |
| Vortex IVF nprobe=32 | 1.000 | 0.128 |
| FAISS flat | 1.000 | 0.166 |
| FAISS ivf nprobe=4 | 0.376 | 0.027 |
| FAISS ivf nprobe=32 | 1.000 | 0.141 |

Original FP32 vectors are retained: total snapshots use approximately 273 bytes/vector for FP32, 401 for FP16 and 341 for INT8. In this run the additional encoding increases total storage, and decoding/cache overhead can increase latency.

The fictional 12-query fixture achieves relevant-chunk Recall@3 of 1.000 with exact search and 0.667 with one IVF probe; extractive answer exact match is 0.917 and 0.667 respectively. These are fixture integration results, not real LLM evaluations.

![Measured systems trade-offs](assets/systems.png)

![Illustrative retrieval and answer diagnostics](assets/fixture-quality.png)

Full measurements, run identity, limitations and reproduction commands are in [the benchmark report](docs/benchmark_results.md). Raw exported evidence is in [benchmarks/reference](benchmarks/reference/).

## Build and test

Requirements: Python 3.10+, a C++20 compiler, and CMake 3.20+. No vector database is required. Build dependencies for Python packaging are installed automatically.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install ".[test,benchmark]"
python -m pytest -q
python examples/quickstart.py
python eval/run_eval.py
```

For native development:

```bash
python -m pip install cmake ninja pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

The native library has no third-party runtime algorithm dependencies. The Python extension is optional for the CMake build. Enable it with `-DVORTEX_PYTHON=ON`; the standard package installation builds it automatically.

On Windows, MSVC is supported by the build configuration and CI. MinGW-w64 is also usable: put its compiler on PATH, use the Ninja generator, and select it with `-DCMAKE_CXX_COMPILER=g++`. For package builds, set the `CMAKE_ARGS` environment variable to that compiler option and `CMAKE_GENERATOR=Ninja`. MinGW runtime libraries are linked statically. The explicit AVX2 implementation is enabled for GCC/Clang; MSVC uses the scalar fallback.

Linux sanitizer build:

```bash
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug -DVORTEX_SANITIZERS=ON
cmake --build build-sanitize --parallel 4
ctest --test-dir build-sanitize --output-on-failure
```

## Python usage

```python
import numpy as np
from vortex import Index

index = Index(3, metric="cosine", threads=2)
index.add(100, np.array([1, 0, 0], dtype=np.float32))
index.add(200, np.array([0, 1, 0], dtype=np.float32))
index.build_ivf(nlist=2, seed=42)
index.save("data/example.vtx")

opened = Index.open("data/example.vtx", cache_bytes=8192, threads=2)
result = opened.search(np.array([1, 0, 0], dtype=np.float32), k=1, nprobe=2)
print(result)  # hits, scanned vectors, native elapsed milliseconds
```

`nprobe=0` selects exhaustive search. Positive nprobe searches that many IVF lists. `rerank_n=0` disables reranking; otherwise it must be at least k. Scores are ascending squared distances for L2 and descending similarities for inner product/cosine. All ties use ascending stable IDs.

`search_batch` parallelizes queries across a fixed worker pool. Single-query searches run on the calling thread. Inputs are copied while the Python GIL is held, then native work releases it.

Builders accept inserts. An insert invalidates existing IVF state; exhaustive search remains available, while IVF search and saving require rebuilding. Reset the representation to FP32 before further inserts. Reopened snapshots are immutable; use a new builder to create a changed collection.

## Architecture and storage

Offline ingestion produces chunk IDs and embeddings. The C++ engine stores those vectors and builds IVF lists. At query time it normalizes the query where needed, selects an exact scan or IVF lists, retains top candidates, optionally reranks with originals, and returns IDs. Python resolves IDs to text and supplies that evidence to the generator.

The snapshot contains IDs, original FP32 vectors, optional encoded vectors, centroids, offsets, and postings. Vectors remain mapped after opening; IDs, centroids, and postings are loaded into memory. The cache budget applies to cached block payload only. It is not a cap on total process memory or operating-system file caching.

Saving writes a complete sibling temporary file before replacing the destination. There is no WAL, transaction system, or power-loss durability guarantee. Windows may reject replacement while a destination is actively mapped; close its readers or publish under a new name. Failures preserve the previous snapshot.

See [file format](docs/file_format.md), [design decisions](docs/architecture.md), and [verification checklist](docs/verification.md).

## Benchmarks and evaluation

```bash
python benchmarks/run_benchmarks.py --vectors 10000 --dimension 64 --queries 100
python eval/run_eval.py --nprobe 0 1 2 4
python benchmarks/plot_results.py --evaluation results/eval/summary.json
```

Raw runtime output is written under ignored `results/`. Each run records configuration, dataset fingerprint, source fingerprint, Git revision and dirty state, package versions, and hardware. The manifest records whether compiler/build information was supplied. Set `VORTEX_BUILD_DESCRIPTION` to the actual compiler flags and build configuration when collecting results for publication.

ANN recall compares approximate neighbors with exhaustive neighbors. Labeled relevance recall measures whether document evidence is retrieved; these answer different questions. Latency percentiles are sequential query wall times. Batch throughput is measured separately. Application cache misses are not physical SSD reads.

The 16-chunk, 12-query support fixture is fictional and illustrative. The default embedding hashes tokens, and its generator extracts a sentence. Answer exact match/token F1, lexical support, and conservative verbatim sentence support validate this pipeline, not real LLM correctness or semantic faithfulness. A support score of 1 can coexist with a wrong answer.

For real model integration, install `.[embeddings]`, use `--embedding sentence-transformers`, and supply both `--embedding-model` and a pinned `--embedding-revision`. Select `--generator http` and set `VORTEX_GENERATION_URL` and `VORTEX_GENERATION_MODEL`. The endpoint accepts a JSON model/prompt request and returns an `answer` or `response` string. The adapter requests temperature 0 and seed 42; endpoint compliance must be checked separately. External model execution requires the relevant packages, model access, and endpoint.

See [evaluation methodology](docs/evaluation.md) and the measured release results above.

## Limitations

- Original FP32 vectors are retained for reranking. FP16/INT8 reduce the encoded search payload but increase total snapshot size in this release.
- Encoded vectors are decoded before scoring; FP16 conversion and cache locking can dominate latency.
- Compressed cosine uses approximate dot products of quantized normalized vectors; it does not renormalize each decoded vector.
- Index construction is in memory. Disk-backed querying does not imply out-of-core training.
- Metadata remains in RAM, and the OS controls mapped-page residency. Application-cold benchmarks do not flush the OS cache.
- Metadata headers are checksummed; full vector payload integrity is not checksummed. Non-finite fetched values are rejected.
- HNSW, PQ, GPU support, online index maintenance, filtering, distributed search, and crash recovery are future work.

## Development layout

`include/` and `src/` contain the engine; `python/` contains bindings and integration; `tests/` contains native and Python checks; `eval/` contains the labeled fixture and runner; `benchmarks/` contains controlled sweeps and plots; `assets/` contains the diagram and public plots.

Regenerate the editable SVG and PNG with `python tools/render_architecture.py` after installing the benchmark extra.
