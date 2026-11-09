# Frozen CPU embedding comparison

Frozen on 2026-11-09 after neutral-text loader/resource pilots and before comparative
retrieval scores. This file is part of every run contract. Changes require a new run.
Starting revision: 01b2f35afc25e96d00f54e802fe00a84a80146f7.

Execution revision 2 was frozen before any comparative quality scores on the
same date. It reuses SciFact document batches across identical split corpora,
without changing vectors, model settings, cases or selection rules. Completed
pre-score batches from the initial execution are imported with their original
hashes and timings; the original contract and migration evidence are retained.
Execution revision 3 subsequently closes an atomic metadata-write gap found in
review and checks independent exact ranking order as well as scores/IDs. It also
precedes all comparative scores, with verified pre-score batches imported again.
The final run directory is results/embedding-study-20261109-v3. Prior contracts
and the second revision's executed source snapshot are preserved privately.

## Models and authorization

The six identities, immutable downloadable revisions, dimensions, licenses and
precision are in embedding_models.json. The four local models run sequentially,
CPU only, four inference threads, FP32 parameters, four documents per batch and
one query per batch. Outputs are normalized contiguous FP32. Jina uses its
retrieval adapter, official Query/Document prefixes and last-token pooling.
Qwen uses the published English retrieval instruction for queries. Gemma uses
the official text-only configuration and task/title prefixes. MiniLM is the control.
Native dimensions are the primary comparison; no dimension reduction is planned.

Selection is a hardware-oriented comparison of a small established baseline,
instruction-trained multilingual embeddings, an adapter-based small encoder,
and a text-only multimodal-family encoder. These are candidates, not a claim
of the six highest ranked models on any leaderboard. Jina weights have a
noncommercial license. Its pinned custom Python code was inspected and hashed.

Owner decision: run local models with available RAM; leave Voyage and Gemini
pending. Approved API spending is USD 0. No hosted calls, gated-license acceptance,
cloud resources or user-app termination are authorized. Hosted adapters are
disabled in the study CLI. Approximate download sizes are 0.09, 1.2, 0.5 and
1.5 GB. Pilots measured encoder-process peaks around 0.5, 2.7, 1.5 and 1.9 GiB.
Full encoding is expected to take tens of minutes, and generation another hour,
with substantial uncertainty on a shared 16 GiB Windows laptop.

## Frozen data and relevance

Use existing checksummed SQuAD 2.0 official train/dev files and the BEIR SciFact
archive at the source URL in preparation provenance. Both distributions are
CC-BY-SA 4.0. Store full data privately under data/. Preparation is deterministic.

SQuAD development: ten hash-selected training articles excluding previous
development articles and official held-out titles; 50 answerable and 10 negative
questions from distinct source paragraphs within each cohort. Held-out: 150
answerable and 30 negative questions from official dev, excluding all previously
inspected query IDs. Article groups cannot overlap across development and test.
The entire corresponding article corpus remains searchable. Positive chunk labels
mean a source chunk fully contains an annotated answer span. Other plausible
answers are incompletely judged. Negatives apply to the source paragraph only,
not the whole searchable corpus. Reference answers are scoring/oracle inputs only.

SciFact development: 50 claims from official train qrels; test: 150 claims from
official test qrels. Hash selection is fixed. Exclude development claims whose
positive documents overlap selected test positives. Freeze a common corpus of
1000 documents, comprising every selected positive document plus hash-selected
distractors. This is a label-informed restricted-corpus experiment, not a standard
full-BEIR score. The shared searchable corpus is transductive; positive-document
groups and queries are disjoint across the two splits. Connected positive-document
components are bootstrap groups. Relevance is binary at document level. No
unanswerability or answer-generation claim is made from SciFact qrels.

Chunk bodies into 128 MiniLM WordPieces, overlap 24; preserve character offsets,
stable IDs, titles and source mappings. All models see the same title and body.
Validate titled chunks against MiniLM's 256 token limit and each encoder's actual
formatted token limit. Reject overlong inputs, with no silent truncation. SciFact
document score is maximum chunk cosine, with unique documents in ranking.

## Execution and exact retrieval

Bind every checkpoint to specification, model registry, input hashes and source
code hashes. Exclusive writer locks protect writers and reports. Save complete
embedding batches atomically with hashes and recover only interrupted final JSONL
appends. Revalidate all completed batch files before resuming. Failure records
remain explicit; do not fabricate replacement vectors. API charges are zero.

Reuse document batches only when the entire ordered chunk records match across
splits. Report reused batches and newly spent encoding time separately; corpus
throughput retains the original measured batch encoding time.

Load one encoder per process and release it before the next. Warm up on neutral
text. Save corpus batch timings and three seeded shuffled query timing rounds;
first round vectors supply retrieval. Query latency includes adapter validation,
formatting and normalization, excludes model loading. Corpus throughput includes
encoding, excludes model initialization and checkpoint disk writes. Record load
time, process peak/current RSS and package versions. Shared background activity
is uncontrolled. Peak encoder RSS is not model-only tensor memory.

Build/reopen separate cosine snapshots, one retrieval thread, 8 MiB block cache.
Independent normalized float64 NumPy cosine is the exact reference. Ties within
1e-5 of the kth score may differ in IDs; returned scores must agree within 1e-5.
Use arbitrary chunk IDs through a lookup, never ID-as-array-offset. For SciFact
progressively double chunk overfetch until k distinct documents or all chunks
are returned. Record effective chunk k, calls and rerank budget. Search timings
include overfetch/deduplication and exclude independent-reference calculation.

Report labeled Recall/Hit/MRR/nDCG at k=5 and 10 for positives, and ANN neighbor
recall against that model's exact document/chunk neighbors. Preserve negative
retrieval records but exclude them from positive ranking metric denominators.

## Development tuning and held-out execution

Bounded grid: nlist in {16,32}, nprobe in {1,4,8,nlist}, storage dtype in
{fp32,fp16,int8}, reranking multiplier in {0,4}; omit redundant FP32 reranking.
FP32 originals are retained for reranking and affect snapshot size. Tune k=10.
Quality gates: mean ANN neighbor recall >=0.95 and labeled Recall@10 no more than
0.02 below that model's exact development recall. Choose the fastest median
development retrieval latency among qualifying configurations per model/domain.
Retain unsuccessful configurations. Fall back to exact if none qualify.
Full-list probing is explicitly reported. Freeze selections before test retrieval.

Run exact and selected configuration at both k values on every test case, three
shuffled timing rounds after warmups. Counterbalance configuration order with
seed 42. Cache timings are warm-process measurements, not controlled cold SSD
measurements. Sequential model order is minilm, qwen, jina, embeddinggemma to
reuse the baseline first; background drift remains a limitation. Report every
model's status, coverage, failures, build times, snapshot bytes and retrieval RSS.

Use 2000 seeded source-group bootstrap resamples for 95% intervals. Paired model
differences resample the same groups/cases. These exploratory intervals have no
multiple-comparison correction and do not establish general leaderboard rank.

## Candidate selection and end-to-end RAG

Choose two local challengers using mean exact development nDCG@10 across domains,
among encoders with measured peak RSS <=4 GiB and query p95 <=5 seconds. These
limits express a laptop deployment budget, not a universal production target.
Keep MiniLM as control. Break ties by query p95 then model key. Selection occurs
before held-out retrieval results are examined. If fewer qualify, report it.

Use 50 positive and 10 paragraph-negative held-out SQuAD questions, hash-selected
without reference-answer inspection. Conditions: exact top-5 retrieval for the
three selected models, source-paragraph oracle, and no context. No IVF RAG claim
is planned. Use the existing gemma3:1b generator and pinned digest
8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc,
same evidence-only PROMPT, 3000 context bytes, 96 output tokens, 4096 context,
four CPU threads, temperature 0 and seed 42. Verify digest before generation.
Counterbalance condition order per query. Save submitted context, hashes,
references, response metadata, attempts and failures. Retry twice at most.
Repeat five hash-selected questions twice more in every condition. Fixed
settings do not guarantee deterministic output on this runtime.

Score max-reference official SQuAD EM/token-F1 on positives and explicit
abstention on the separate negative cohort. Failed requests score zero. Report
raw-empty, citation-only, empty-scored and abstained answers separately. Citation
presence uses substantive-answer denominator; citation-ID validity uses all
mentions, without implying claim support. Lexical/verbatim support are diagnostics.
Compare identical prompt hashes across conditions and repeats for output variation.
Prepare blinded examples and semantic/citation rubric; human review stays pending.

## Publication and limits

Publish aggregate JSON/CSV, safe provenance, checksums, commands and methodology.
Keep data, weights, embeddings, snapshots, full responses/reviews private and
ignored. Update the ignored owner reports and render/inspect Word pages.
No completion claim for deferred APIs. Record forecast token/cost ranges from
common formatted inputs, clearly distinct from actual charges. Record hosted
version details as unavailable until an authorized call executes. Quality
recommendations apply to these English subsets and this shared CPU machine.
