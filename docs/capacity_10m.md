# Ten-million-vector single-collection capacity

This extension tests one read-only Vortex collection containing 10,000,000
synthetic 64-dimensional FP32 vectors. It uses the installed native query engine
and its version-1 mapped snapshot format. The benchmark's new bounded snapshot
writer avoids materializing the entire input matrix and a second full native
builder in RAM. The native mutable insertion path has not been changed.

## Measured results

Both final sizes passed all 480 recorded queries, including every exact-reference
check, repeated-output check and all-list/exact agreement check. No cloud was used.
The complete snapshots were also verified by streaming SHA-256 after collection.
The matching native-module hash is recorded in both results. These are twenty
distinct queries repeated three times, not sixty independent queries.

| Vectors | Construction s | Open s | Exact p95 ms | Selected IVF p95 ms | Recall@10 | nprobe | Peak working set MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 2.734 | 0.124 | 233.142 | 1151.608 | 0.985 | 28 | 382.171875 |
| 10,000,000 | 36.145 | 0.942 | 4003.070 | 23053.437 | 0.980 | 24 | 1536.015625 |

The single 10m snapshot contains 2,720,008,584 bytes (2.533 GiB). The selected IVF
setting scans a mean 7,501,908.65 vectors per query. In this fixed 32-list,
resident-limited experiment, high-recall IVF is slower than exact. This is a
capacity demonstration and a measured performance limit, not a latency improvement.
Construction includes generation, native sample training, full assignment and
snapshot finalization; hashing, independent references and queries are additional.

The 1m experiment elapsed 298.927 seconds. The complete 10m experiment elapsed
10,755.191 seconds, including the long wall/CPU gap described below. The build
itself took 36.145 seconds. Do not interpret total experiment duration as build time.

Full sweeps, raw query records, sources, hashes and failure evidence are in the
[compact archive](../benchmarks/reference/capacity/report.md). Both large local
snapshots remain under ignored `results/`.

## Timing gap and analysis correction

The final all-list condition contains a sample with 3,832.125 seconds of wall
time and 35.797 seconds of process CPU time (repeat 2, query 19). Suspension or
long descheduling is possible; the cause is unconfirmed. The sample returned
the correct neighbors and remains in every unfiltered timing statistic. That
condition is flagged as interrupted and should not be presented as a clean
latency result. Neither exact nor the selected 24-probe condition contains a
flagged gap. Repeating the interrupted all-list condition would be necessary
before making a clean latency claim for that condition.

Post-collection diagnostics flag wall time over 60 seconds and over ten times
process CPU time. This diagnostic was added after observing the gap; it does not
remove any samples or change the predeclared recall gate.

Windows confirmed the requested 1536 MiB hard working-set maximum, but its
reported peak was 16,384 bytes above the requested value. The maintained analysis
validator corrects the original overly strict byte comparison with an explicit
64 KiB accounting allowance and reports the actual excess. Larger excess still
fails validation. This correction was made after collection. The frozen initial
validator, collector sources and raw results are preserved unchanged, with
maintained analysis sources archived separately. The 384 MiB free-memory floor
and all correctness requirements remain unchanged.

## Construction and resources

The original in-memory runner attempted 10 million and stopped at its memory
guard: estimated duplicate-buffer headroom was 11,654,217,728 bytes, versus
4,072,943,616 bytes available. That attempt remains recorded separately. The new
path does not weaken that guard.

The first unrestricted mapped run subsequently built and queried the complete
10m index, but stopped during the IVF sweep when available RAM fell below
384 MiB. Its approximately 3.14 GiB peak working set and partial measurements are
retained as a resource-limited run, not a completed capacity result. The
[resident-bounded amendment](../benchmarks/capacity_10m_bounded_spec.md) defines
the final paired protocol before collecting new bounded results.

`benchmarks/streaming_snapshot.py` generates the complete seeded corpus in
4096-row chunks, writes stable IDs and originals, and chooses at most 10,000
uniformly sampled rows for native IVF training. Sample selection uses seed 4201;
native clustering uses seed 42, eight iterations and 32 lists. Every corpus row
is then assigned by the native exact centroid-search path. Complete centroids,
offsets and postings are written before publishing the snapshot header and name.
The file is one collection, not a query fan-out across separate shards.

Ten million 64D vectors contain 2.56 GB of raw FP32 components. The snapshot also
stores IDs and postings, taking approximately 2.72 GB. Auxiliary assignments take
40 MB; the small training snapshot is also retained locally. Initial disk/RAM
checks and a 384 MiB available-memory floor at construction/query boundaries
preserve resource failures and partial files. Existing outputs are never reused
or overwritten. No cloud service, paid API, or app closure is required.

## Validation and timing

The [protocol](../benchmarks/capacity_10m_spec.md) was defined before the capacity
measurements. A fresh one-million-vector control uses the same final runner,
query vectors, native binary, chunk size and cache budget as the ten-million run.
Both use `--working-set-mib 1536`: a Windows hard maximum on the benchmark
process's resident working set, verified through the Windows API and restored on
exit. This affects only the benchmark process. Paging and page replacement are
included in query time. It does not bound committed virtual memory or the total
OS file cache. See [Microsoft's API documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsizeex).

- Twenty independent queries use seed 2026, top ten, three shuffled-order rounds
  and five warmups per configuration. Exact search and probes 1/4/8/16/24/28/32
  produce 480 recorded measurements per corpus size.
- Every timed exact query must scan the entire collection. Exact output is checked
  against blocked FAISS over all corpus rows for all twenty queries. Direct
  float64 NumPy checks queries 0, 10 and 19. First/middle/last corpus self-lookups
  verify IDs 0, N/2 and N-1. Full-list IVF must match native exact output.
- Raw records preserve IDs, distances, scan counts, wall/native/CPU clocks,
  repeat/query identities and power observations. Numerical reference-boundary
  ties use the declared 1e-4 absolute tolerance and are flagged explicitly.
- Tables report median round p95. The IVF operating point is the smallest tested
  probe reaching mean Recall@10 >=0.95 on the measured queries; it is not an
  independently tuned or held-out operating point.
- The 8 MiB LRU budget covers application-cache payload. Native IDs/postings,
  mapped-page residency and the OS file cache are additional memory. OS caches
  are not flushed, background desktop load is uncontrolled, and queries run
  sequentially. FAISS reference-collection time is not a FAISS latency baseline.

The new uniform training-row selection differs from the historical mutable
builder's shuffled selection. The historical in-memory experiment also used 100
queries and a different query seed. Keep its [one-million results](scalability.md)
separate from this mapped control; do not append these timings to that curve.

## Reproduction

Install the package's test and benchmark extras, then choose unused output paths:

```bash
python benchmarks/run_capacity.py --size 1000000 --working-set-mib 1536 --output results/capacity-control-new
python benchmarks/validate_capacity.py --directory results/capacity-control-new --check-snapshot
python benchmarks/run_capacity.py --size 10000000 --working-set-mib 1536 --output results/capacity-10m-new
python benchmarks/validate_capacity.py --directory results/capacity-10m-new --check-snapshot
python benchmarks/report_capacity.py --input results/capacity-control-new results/capacity-10m-new --output results/capacity-report-new
```

The final command refuses to combine runs with different recorded protocols,
native-module hashes or query fingerprints. The validator independently
reconciles saved measurements with the summaries, coverage, recall and selected
probe. `--check-snapshot` also streams the large snapshot checksum.

Raw snapshots stay in ignored `results/`. Compact exported evidence can retain
the executed sources, all query records, reference results, query vectors and
checksums without checking gigabytes of synthetic data into Git. Regeneration
uses the saved seeds and exact source/native identities; regenerating a corpus
is distinct from independently rerunning every timing measurement.

The published archive can be checked without rerunning the long experiment:

```bash
python benchmarks/validate_capacity.py --directory benchmarks/reference/capacity
```

`attempts/1` retains the original duplicate-buffer stop. `attempts/2` retains the
unrestricted mapped stop, its partial query log and executed sources. The main
size directories contain the final bounded pair. `analysis-source` retains the
maintained reporting/validation code as text; each run's `executed-source` keeps
the bytes captured before that collection began. Snapshot binaries are excluded.

## Scope

Passing demonstrates native retrieval capacity for this one 64D synthetic
collection and query set. It does not prove ten-million-vector mutable insertion,
concurrent-service throughput, multi-node scaling, cold-disk performance or
semantic RAG quality. At 768 dimensions, ten million FP32 vectors alone require
30.72 GB before metadata and temporary buffers; that is a different experiment.
