# Real-vector retrieval study

Baseline tree is Vortex b992313. This study benchmarks the existing engine.
Prior synthetic evidence remains separate. Use the public ANN-Benchmarks
SIFT, GloVe and DEEP vectors, and cached MiniLM SciFact chunk vectors with
official BEIR judgments. Use the full corpus for each case. DEEP is the
9,990,000-row ANN-Benchmarks release, with a conditional resource gate.

## Inputs and quality

Use squared L2 for SIFT and L2 on unit-normalized vectors for angular data.
The latter preserves cosine neighbor order. Use the same normalized float32
queries and vectors in Vortex and FAISS. Record published-file and effective
vector hashes. Verify published top ten distances and sets against independent
float64 blocked search for a deterministic sample. A tied boundary can
produce several valid sets. Report strict published-ID overlap and boundary
ties where detected. For SciFact use independent float64 exact search and
keep chunk and document ID spaces explicit.

ANN-Benchmarks query IDs are partitioned once with seed 31827. The tuning
split has 192 distinct queries and the primary holdout has 1,000. Check for
duplicate query bytes before finalizing the split. Use the cached SciFact
development/test split, with 50 development and 150 test queries where
verified. Treat its timing as a smaller workload and score document relevance
using official test qrels after deduplicating source IDs. Include all official
relevant IDs in denominators for the selected queries.

Build 128 IVF lists at million scale, 256 for DEEP, 16 for SciFact. Use a
bounded shared sample for FAISS IVF-PQ and native IVF centroid training.
Use packed FP32 and INT8 snapshots with stable IDs. INT8 and FAISS IVF-PQ
rescore a shortlist of 100 against the same FP32 original representation.
Record the additional external rerank payload in footprint. The primary
Vortex ablation uses the same packed INT8 snapshot for all three paths.

## Measurement sequence

1. A 32-query tuning-side pilot estimates query cost after the first complete
   million-scale build. Record practical time and disk limits before final
   collection and save incomplete runs if those limits are reached.
2. Sweep geometric probes from 1 up to nlist for cached INT8, FP32 IVF and
   FAISS IVF-PQ, using only tuning queries. Select the fastest p95 at each
   of 95%, 98%, 99% mean strict Recall@10. Use the point selected at 98% as
   default; use the highest qualifying tier or max-recall tested setting
   if 98% is unavailable, and identify it. Direct worker modes reuse the
   cached INT8 operating point, including the same reranking.
3. Freeze selections then run 1,000 distinct held-out queries in three
   rotated rounds when practical. Validate additional distinct tier
   selections in one round. Report single-call p50/p95 and round p95.
   Use paired-query bootstrap intervals, keeping repeats together.
4. On the same selected operating points run ordinary closed-loop
   concurrency at 1/2/4/8 clients. A request starts before a shared admission
   gate. There are two active CPU slots; a two-worker query consumes both.
   Other engines use one slot per synchronous request. Report measured
   queue-inclusive p95, achieved QPS, deadlines and shared-slot policy.
   Concurrency uses a declared 256-query holdout subset, one rotated pass;
   treat tail estimates there as descriptive. Measure true batch paths
   separately and label unsupported batch modes.

The main track applies the 1536 MiB Windows process working-set maximum
to each worker, with the prior 64 KiB accounting allowance. This excludes
the system file cache and process committed memory. Run a distinct OS-managed
track after the main track. Build memory and staging disk costs are separate.
Use pinning of dependency versions and fixed BLAS/OpenMP limits. Warm up by
running eight tuning queries for timed lookups. Separately record fresh-process
open and first lookup without claiming OS-cold caches. Rotate test cases,
keep every finished sample, and record resource stops and incomplete coverage.

The timing boundary is the Python call to retrieval through ordered public IDs
and scores. It includes FP32 reranking, ID translation and sorting as applicable.
Query embedding is precomputed. Report exact persisted bytes, effective build
cost, process CPU and working set, I/O observations, individual recall and
document relevance. Recompute summaries from raw evidence before publication.

## Execution budgets

Allow 1,500 seconds for each million-scale acquisition attempt, 1,500
seconds for each million-scale measurement suite, and 240 seconds for
SciFact. Allow 150 seconds per condition. The independent validation and
plot generation run after collection. Require at least 384 MiB free physical
memory throughout, with a separate eight GiB disk reserve. For the DEEP
admission decision, bound acquisition to 2,400 seconds using demonstrated
verified transfer rates, require staging space of
`N × (14 × dimension + 128) + 1 GiB` after acquisition and the disk
reserve, and use the observed million-scale build and query pilot to project
a separate 2,400-second construction and 3,600-second measurement limit.
Record any missed gate. Record frozen held-out points within 20% p95 of
each other for comparisons at comparable latency.
