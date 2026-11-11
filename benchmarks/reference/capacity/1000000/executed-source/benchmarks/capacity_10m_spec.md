# Ten million vector capacity protocol

Defined before collecting new capacity measurements on 11 November 2026.
Starting revision: eebd9ebcd57563a01319dacb58833edae621a355.

## Objective and resource boundary

Demonstrate one Vortex snapshot containing 10,000,000 vectors, not ten separate
one-million-vector indexes. Retain the 64D FP32 independent standard-normal
baseline, squared L2, stable IDs 0 through N-1, top ten, 32 lists, eight native
clustering iterations and seed 42. This is synthetic vector capacity, not semantic
RAG quality at 768/1024 dimensions or simultaneous production traffic.

The original runner genuinely attempted 10m and stopped before allocation: its
duplicate-buffer estimate was 11,654,217,728 bytes with approximately 4.07 GB
available. Preserve that resource-limited attempt separately. Do not weaken its
guard or overwrite its evidence. Do not close user apps or provision cloud.

## Memory-bounded construction

Generate the full corpus in bounded 4096-row chunks directly into a new snapshot
file. Train Vortex's native IVF algorithm on at most 10,000 uniformly selected
rows (sample seed 4201). Assign every corpus row using native exact L2 search
against those centroids, in bounded native batch calls with four workers.
Write all IDs, originals, centroids, offsets and postings in version-1 format.
Publish the final snapshot name only when the sections/header are complete.
Native Index.open must accept it and report the full collection size.

The native mutable builder and query engine are unchanged. This construction
uses a different training-row selection scheme than the historical in-memory
runner; do not merge the old/new timing curves or claim a 10m mutable-builder pass.
Preserve original source bytes, native module hash, package/build identity,
corpus/query fingerprints, timings, stage memory and power observations.

Require disk headroom for the snapshot and bounded auxiliary files; retain at
least 384 MiB available physical memory during construction/query boundaries.
Refuse existing run artifacts. Record resource stops and preserve partial files.

## Validation and measurement

Use 20 independent queries from NumPy generator seed 2026 and three shuffled
timing rounds, top ten. The complete 10m corpus participates in every exact query;
the query count is reduced from the historical 100-query in-memory benchmark.
Use five warmups per configuration, one calling thread per query, four workers
for centroid assignment, and an 8 MiB native application cache.

Compute a blocked FAISS exact reference for all 20 queries, scanning the full
corpus without constructing a second full IndexFlat copy. Also check queries
0, middle and last with direct squared L2 in independent float64 NumPy blocks.
FAISS reference-collection time is not a comparable FAISS per-query baseline.
Permit numerical boundary ties within 1e-4 absolute distance; preserve any tie
differences explicitly. Check three first/middle/last corpus self-lookups.

Measure mapped exhaustive search and probes 1,4,8,16,24,28,32. Preserve every
timed query's IDs, scores, scanned count, wall/native times, repeat and query ID.
Require repeat stability and all-list agreement with the native exhaustive truth.
Select the smallest probe reaching mean neighbor recall >=0.95 on these measured
queries; this is descriptive operating-point selection, not held-out tuning.
Record cache bounds, stage RSS, system memory, AC/battery state and elapsed/CPU
clocks. OS page cache is not flushed; these are warm-process mapped measurements.

Run a fresh 1m control with the same streaming protocol before 10m. Keep that
control separate from the historical 1m archive. Accept capacity only when the
single native index opens, all exact rows scan N, independent checks pass, full
probing agrees, all query rounds complete and the output evidence validates.
