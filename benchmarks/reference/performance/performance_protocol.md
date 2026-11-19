# Retrieval optimization protocol

Baseline: repository `c2fdefc653e9a11d4dee2bd1a30d7b8a5929016b` and its
installed native module, copied before native changes. Historical capacity
archives are immutable. All new raw outputs use a unique ignored `results/`
directory. This protocol precedes optimization measurements.

## Questions and sequence

1. Measure resident/mapped exact and IVF, cache reads, distance/validation,
   compressed scoring and batch execution with scoped microbenchmarks.
2. Test bounded block scans on the same version-1 snapshots.
3. Repack vectors by IVF list without changing public IDs or format version;
   compare identical vectors, centroids, assignments and queries.
4. Tune 32/64/128/256 lists on independent tuning queries and representative
   training rows. Evaluate the selected configuration on held-out queries.
5. Evaluate compression and SIMD/parallel settings separately. Keep only
   justified changes; document deferred experiments explicitly.

## Controls

Synthetic corpus: seeded NumPy standard-normal FP32, 64 dimensions, seed 42.
Queries: seed 20261119, independent of training/corpus; first 32 are tuning,
next 100 are held out at 100k and 1m. At 10m use 32 held-out queries initially,
increasing only under the pilot budget. Three shuffled rounds, three warmups.
Repeated measurements do not increase the independent query count.
L2, k=10, one native query thread for the primary latency comparison. Use an
independent blocked FAISS exact reference and direct float64 checks on a subset.
Choose the fastest tuning configuration with mean Recall@10 >= .95; never tune
on final held-out measurements. Preserve failed configurations.

Time each complete Python `Index.search` call with perf_counter, including
binding preparation and output conversion; retain native and process CPU time
separately. Primary statistic is p95 over all recorded query calls, with round
p95 and paired-query bootstrap intervals also reported. Do not substitute batch
throughput for single-query latency. Randomize condition order by round.

Identical baseline/optimized comparisons use the same query/corpus bytes,
centroids, layout, cache budget (8 MiB), block size (4096), kernel and resident
limit. Locality comparisons change layout only. Tuning comparisons explicitly
change routing/training parameters. Mark those separately from engine speedup.
Use the same 1536 MiB Windows process working-set policy for the mapped
capacity track; the policy excludes OS file cache and committed memory.
Retain actual peaks and any accounting excess. Guard available system RAM at
384 MiB; never close apps, lift the limit, or replace archived outputs.

Cache state is warmed/OS-managed, not a controlled cold-cache experiment.
Clear only the application cache where declared. Alternate conditions and
retain timestamps, power state and page-fault/working-set observations where
available. Flag wall/CPU gaps (>60s and >10x CPU), but retain every sample.

## Pilot and resource gates

Start with small fixtures, then 100k, then 1m. After profiling, record a forecast
and a final-run time budget in a separate pre-run plan before final measurements.
Avoid a full Cartesian parameter sweep. A run may stop at its declared resource
or runtime limit; preserve records/status and report incomplete coverage.
Do not turn a resource stop into a speedup claim. Report whether 10m completed.
Large construction, reference collection, tuning and query time are separate.

### Pre-final pilot amendment

The initial 100k pilot used seed 20261119. Final references use a fresh seed
20261120; no pilot query from that first set enters the final set. The 10m pilot
uses only final tuning queries 0 and 1, never the held-out suffix. It measured
approximately 23.05s mean baseline IVF and 2.01s packed IVF, with exact scans
approximately 3.4s. Those two-query timings are a forecast, not final evidence.

Before final collection: retain three rounds and 100 held-out queries at 100k
and 1m. At 10m use 32 distinct held-out queries and **one** final round, rather
than three, to keep the approximately 17-minute baseline/final paired query
forecast practical. Include three warmups per configuration. Use a 45-minute
total final-query budget and 1200s per isolated condition process; keep partial
runs if either limit is reached. Optional 10m locality-only ablation is included
only if the remaining budget permits. The 10m tail estimate has no repeated-round
stability evidence and must be reported with that limitation.

Use the historical fixed 32-list/24-probe setting for the matched 10m comparison;
do not select a new 10m setting using held-out results. Use the 1m tuning split
for the separate 32/64/128/256 routing study. Compression experiments remain at
1m or below unless the remaining resource/time budget clearly permits expansion.

Reuse existing local real embedding matrices for a separate small experiment if
resources permit. Do not download encoders or mix real-data relevance with
synthetic nearest-neighbor recall.

## Evidence and acceptance

Save per-call timings, IDs, scores, recall and scanned rows, source/binary/data
hashes, environment, all configuration and raw failures. Acceptance requires
independent correctness, deterministic stable-ID tie handling, old snapshots
remaining readable, unchanged public archives, and passing regression checks.
Select optimizations by measured wall time at matched recall/memory, not by
smaller scan counts alone. No required numerical speedup is assumed.
