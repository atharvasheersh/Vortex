# Single-collection mapped capacity

Synthetic 64D FP32 squared L2, top 10, 32 lists; 20 queries, 3 rounds, five warmups per configuration, 8 MiB application cache. Latencies are median round p95. A bounded snapshot writer constructs each collection; the native mutable builder is unchanged. OS page cache was not flushed. This is sequential local capacity, not a concurrent service or real RAG-quality experiment.

Requested Windows hard process working-set maximum: 1536 MiB, applied equally to both sizes. Paging/replacement costs are included in query latency; this does not limit commit or the OS file cache. Actual observed peaks are reported below.

| Vectors | Construction s | Exact p95 ms | IVF p95 ms | Recall@10 | nprobe | Peak working set MiB | Snapshot GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 2.734 | 233.142 | 1151.608 | 0.985 | 28 | 382.2 | 0.253 |
| 10,000,000 | 36.145 | 4003.070 | 23053.437 | 0.980 | 24 | 1536.0 | 2.533 |

The IVF operating point is the smallest tested probe with mean recall >=0.95 on the measured queries. Historical in-memory measurements use different training-row selection, query seed/count and storage paths; keep those curves separate. Construction includes disk generation, sampled native training, full-corpus native assignment and snapshot finalization. Reference generation, snapshot hashing and query measurement are outside construction time.

## Probe sweep

| Vectors | nprobe | p95 ms | Recall@10 | Mean vectors scanned |
| ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 1 | 54.195 | 0.145 | 30,728.1 |
| 1,000,000 | 4 | 262.024 | 0.415 | 125,272.6 |
| 1,000,000 | 8 | 487.772 | 0.575 | 249,601.0 |
| 1,000,000 | 16 | 665.091 | 0.850 | 500,674.8 |
| 1,000,000 | 24 | 1197.052 | 0.945 | 751,050.4 |
| 1,000,000 | 28 | 1151.608 | 0.985 | 874,879.6 |
| 1,000,000 | 32 | 1215.556 | 1.000 | 1,000,000.0 |
| 10,000,000 | 1 | 975.361 | 0.145 | 318,808.5 |
| 10,000,000 | 4 | 3668.548 | 0.440 | 1,257,753.4 |
| 10,000,000 | 8 | 7705.178 | 0.635 | 2,507,750.8 |
| 10,000,000 | 16 | 15494.385 | 0.900 | 5,007,842.8 |
| 10,000,000 | 24 | 23053.437 | 0.980 | 7,501,908.7 |
| 10,000,000 | 28 | 27210.988 | 0.995 | 8,752,150.7 |
| 10,000,000 | 32 | 37055.646 † | 1.000 | 10,000,000.0 |

Every timed exact query scans N vectors. Blocked FAISS covers all queries over the complete corpus; direct float64 NumPy checks three queries. Full-list IVF matches native exact IDs and scores. Repeated runs use one corpus and one machine; they do not provide independent dataset or hardware replication. Working-set peaks include mapped pages and metadata and are not private heap size or a limit imposed by the cache.

## Analysis diagnostics

Long wall/CPU gaps are flagged when wall time exceeds 60 seconds and ten times process CPU time. This is a post-collection diagnostic; no sample is removed or replaced. Conditions marked † contain such a gap and should not be treated as clean latency measurements.

- 1,000,000 vectors: 0 flagged gap(s); observed resident peak exceeds requested maximum by 0 bytes.
- 10,000,000 vectors: 1 flagged gap(s); observed resident peak exceeds requested maximum by 16,384 bytes.
  - Probe 32, repeat 2, query 19: 3832.125 s wall versus 35.797 s CPU. Cause is unconfirmed; suspension or long descheduling is possible.

Windows confirmed the hard-limit flags, but peak accounting exceeded the requested resident maximum by a few pages. The maintained validator was corrected after collection to allow at most 64 KiB of accounting excess and still reports the actual excess. Frozen collection sources and raw results are unchanged; maintained analysis sources are archived separately. The free-memory floor and all correctness checks remain unchanged.
