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

Local Windows GCC Release and Debug CTest runs pass, with 63,666 native assertions. All 55 Python tests pass against the built extension and installed package. A separate fresh checkout passes native tests, installed-package tests, the quick start, and offline evaluation. Architecture, measured plots, and both private Word documents were rendered and visually reviewed.

The [initial hosted CI run](https://github.com/atharvasheersh/Vortex/actions/runs/42356271644) failed before runner assignment because of an account restriction. None of its Windows/MSVC, Linux, or Linux sanitizer steps executed. Local MinGW sanitizer linking was unavailable because the installed toolchain lacks libsanitizer.spec. Remote platform and sanitizer validation remain pending; configured jobs are not evidence of passing checks.
