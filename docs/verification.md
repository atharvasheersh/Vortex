# Requirement and verification checklist

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Exact L2/IP/cosine | distance.cpp and index.cpp | NumPy references across dimensions and kernels |
| Stable top-k | Bounded heap and ID tie-break | Ties, k above count, empty collection |
| Persistence | Single versioned snapshot | Save/reopen, malformed fields, truncation, duplicate IDs |
| SIMD | Runtime AVX2 with scalar fallback | Numerical comparison and microbenchmark |
| IVF | Seeded Lloyd training and inverted lists | All-list exact agreement, stale-index rejection |
| Mapped storage | Windows/POSIX ownership abstraction | Reopened queries and mapping lifetime checks |
| Bounded cache | LRU payload accounting | Small/zero budget, eviction, reset, concurrent batches |
| Compression | FP16 and per-vector INT8 | All finite half round-trips, NumPy quantization references, full rerank |
| Python API | pybind11 | Dtype, shape, contiguity, batch errors, installed-package checks |
| Retrieval metrics | Python deterministic calculations | Hand-calculated ranking fixtures |
| RAG plumbing | Separate chunk lookup and generation | Fictional support fixture and evidence log |
| External models | Optional pinned embedder and HTTP adapter | Local contract tests; real-model results require configured access |
| Baselines | Separate FAISS benchmark | Matched data/metric/thread count; saved sweeps |
| Reproducibility | JSON/CSV outputs and manifests | Configuration, source and dataset fingerprints |
| Portability | CMake and CI | Windows and Linux build jobs; Linux sanitizer job |

The repository's current CI run is the authority for remote platform verification. A configured CI job alone does not establish a successful run.
