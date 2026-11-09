# Supplemental timing audit

Defined on 2026-11-09 after observing anomalous encoding elapsed times and before
comparative retrieval/answer scores. The primary study remains unchanged.

Qwen's SQuAD corpus records include three extreme elapsed batches, approximately
3090, 18484 and 6955 seconds, against a median around six seconds. Windows logs
show sleep/Modern Standby transitions during the run. These events support an
interruption explanation; per-batch CPU time was not captured in the primary run,
so the exact cause/duration cannot be reconstructed for every batch. Retain all
primary elapsed observations and report their distributions, without trimming.

Run a separate, labeled supplemental CPU timing experiment after primary data
collection. Same immutable models, FP32 request, four inference threads and batch
four. Reverse the primary model order: embeddinggemma, jina, qwen, minilm. Load
each in a fresh process and release it before the next. No model or candidate
selection, corpus, question, quality gate or primary result is changed.

For each domain, hash-select 16 development document chunks and ten development
queries. Selection depends on IDs only, not labels or scores. Warm up with the
first document batch and one query. Also compare its four document vectors
against separate single-item calls, preserving order with absolute/relative
1e-4 tolerance for FP32 padding/kernel differences. These are untimed adapter
checks. Then run three independently shuffled rounds of
all sampled documents and queries. Documents use batches of four and queries
batch one. Checkpoints bind source/spec/model/data/sample identities. Record
elapsed and process CPU seconds, actual parameter dtype counts, dimensions,
encoder-process peak RSS and Windows AC/battery-saver status before/after calls.
No power settings or user applications are changed. On other platforms mark the
Windows power observation unavailable. Retain failed or interrupted attempts.

Report sampled document throughput, query p50/p95 and all raw per-call timing
records privately. Aggregate CPU time sums CPU consumed by all process threads;
it excludes sleep and can exceed elapsed time with parallel threads. It is not
single-core latency. Power-state changes and low CPU/elapsed ratios are diagnostic
flags, not grounds to discard cases. This small supplemental sample is not a
replacement for the primary full-corpus measurements and has no quality scores.

The primary model comparisons use native dimensions. The audit also verifies the
actual loaded parameter dtypes and effective parameter count for the same pinned
constructor used during primary execution. Report any mismatch from requested
precision explicitly rather than assuming a successful cast.

API spending remains zero. Source APIs are documented by
[Python](https://docs.python.org/3.12/library/time.html#time.process_time) and
[Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status).
