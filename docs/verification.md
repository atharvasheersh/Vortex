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
| External models | Pinned MiniLM encoder and local Gemma generation | 10 development smoke records and 900 held-out records; official scoring agreement |
| Baselines | Separate FAISS benchmark | Matched data/metric/thread count; saved sweeps |
| Scalability | Isolated 1k/10k/100k/1m runs at 64D | Saved query logs, full-probe agreement, NumPy samples and FAISS exact checks |
| Benchmark tooling | Public scaling runner and archive validator | Small subprocess run, argument rejection, overwrite protection and tamper detection |
| Reproducibility | JSON/CSV outputs and manifests | Configuration, source and dataset fingerprints |
| Portability | CMake and CI | Windows and Linux build jobs; Linux sanitizer job |

Local Windows GCC Release and Debug CTest runs pass, with 63,666 native assertions. The expanded Python suite has 73 passing tests, including eight scaling-runner/evidence checks and ten real-data scoring, recovery, concurrent-report and stable-ID checks. The original 55-test release also passed separate fresh-checkout verification of native tests, the installed package, quick start and offline evaluation. Architecture, measured plots, and both private Word documents were rendered and visually reviewed.

The real-model experiment records 900 held-out generations across five conditions, with zero failed HTTP requests or retries and 16 citation-only answers scored zero. The independent official SQuAD scorer agrees on every stored answer score. Data auditing verifies source spans and split separation. Public hashes and local records are checked by eval/validate_real_rag.py. The archived collector preserves measured source bytes before the maintained runner's stable-ID reference-check correction. Human semantic and citation review is pending at the owner's request.

The four scale runs retain the unchanged native-engine revision and the historical runner hash in each manifest. Public validation recomputes recall and percentiles from all 9,840 query records, checks full-probe and mapped agreement, and verifies the export checksums. New full-size measurements are intentionally outside CI; CI exercises the maintained runner with a small fixture and includes its FAISS dependency. Large generated snapshots and private documents remain excluded from Git.

The [initial hosted CI run](https://github.com/atharvasheersh/Vortex/actions/runs/42356271644) failed before runner assignment because of an account restriction. None of its Windows/MSVC, Linux, or Linux sanitizer steps executed. Local MinGW sanitizer linking was unavailable because the installed toolchain lacks libsanitizer.spec. Remote platform and sanitizer validation remain pending; configured jobs are not evidence of passing checks.
