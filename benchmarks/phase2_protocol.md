# Phase 2 retrieval protocol

The primary control is Vortex packed INT8 with 32 lists, 24 probes, k=10 and
100 FP32 reranking candidates. The historical 534 ms result motivated this
study. Compute speedups against a fresh control on the same inputs.

Use one Windows host and one serving thread per backend. Apply the same
1536 MiB Windows process working-set maximum. Log the actual peak and fail
eligibility if the peak exceeds the cap plus the 64 KiB accounting allowance
used by the previous study. System file cache and memory committed by the
process are outside this policy. Log graph preparation and build memory and
time separately from query measurements.

Feasibility amendment before timed graph runs: diskannpy 0.7.0 on Windows
stalled twice during a one-thread reopen of the 100k graph. A fresh two-thread
reopen and lookup succeeded. Configure both backends with two thread slots
and submit one query at a time. The graph worker uses a Windows build of
CPython 3.11, with the Vortex control rebuilt from the same source for that
runtime. Both share the same NumPy version and process memory policy. The
separate batch pass uses up to two threads for each backend.

The first build-validation reopen still uses DiskANN's default sample-based
cache setup. The adapter disables that setup for zero-node-cache timed graph
workers before the first 1m graph timing. Initialization remains outside
lookup time and is recorded separately.

Use seeded 64D synthetic L2 snapshot originals from the previous study. Preserve
their row order for the disk graph export and map graph row IDs back to the same
uint64 public IDs. Use squared L2 and top 10. Fresh independent references use
blocked FAISS plus three direct float64 checks per corpus. Use seed 20261124.
Reserve the first 48 queries for tuning and the remaining 100 for held-out
evaluation at 1m. At 10m reserve 48 tuning and 64 held-out queries. The 100k
pilot is only for feasibility, profiling and budget estimation.

Sweep a small set of graph build and search parameters on tuning queries.
Declare an eligible tuning setting to have recall at least 97.6% and within
0.5 percentage points of fresh INT8 recall on the same queries. Among eligible
settings choose the lowest p95. Freeze selection before evaluating held-out
queries. Apply the same quality rule on the held-out split and require lower
p95 to advance to 10m. Report actual recall deltas, including any differences
in ordered neighbors. An improvement of 2x is a stretch target.

Collect three held-out rounds at 1m and two at 10m, alternating backend
execution order and using identical query order per round. Use exclusive
worker processes and three warmup calls before timing. Cache state is warmed
and OS-managed. Single-query wall time includes Python input/output, public
ID mapping and any reranking. Measure a separate one-thread batch pass for
throughput. Keep every sample, including timing interruptions. Report pooled
p95 and round p95. Bootstrap paired query IDs with their rounds kept together.

Run the cached MiniLM corpus separately if available. These are retrieval
metrics over cached embeddings. Establish graph validity on small fixtures
including IDs greater than 2^63, independent scores, reopen and failure paths.

After the 100k pilot, record the time and disk budget before the larger runs.
Keep failed build and run records. Integrate the graph only if its measured
lookup gain and build, resource, and dependency costs justify the change.

The 100k pilot used degree 32, build complexity 48, a 32-byte PQ routing
payload and two serving slots. Build took 42.6 seconds with about 269 MiB
peak process residency. The sampled lookup at complexity 128 and beam 2
failed the recall gate. For 1m compare degrees 32 and 48, build complexity
degree + 16 and a search-memory allocation of 0.060 GiB for richer routing
codes. Use build budget 1.5 GiB, four build threads, a 960-second limit per
graph build and a 2400-second limit for the 1m stage. Sweep (complexity, beam)
pairs (32,2), (64,2), (128,2), (256,4), (512,4), (1024,8).
Require 384 MiB available system RAM throughout and use the build preflight
estimate for staging and intermediate disk files. Budget at most 30 GiB of
additional disk for the larger stages. A 10m stage requires the 1m holdout
gate and an estimated build plus query time of no more than 50 minutes.
Forecast from measured 1m build, reference and query costs before launching.

The first 1m graph peaked at 83.33% tuning recall, at 116.58 ms p95.
Prepare a Vortex-native fallback in a separate build while the denser graph
is evaluated. If the graph cannot meet both quality and latency, compare
the current packed INT8 path against opt-in direct scans using one and two
workers. Both use the same two-slot configuration and identical candidates,
scores and reranking. The one-worker direct path isolates the copy/cache
change. The two-worker path additionally uses one local heap per worker and
a deterministic merge. Use the same holdout and memory policy, and require
identical ordered results for this comparison. Advance the two-worker path
to 10m only if it lowers 1m p95.
