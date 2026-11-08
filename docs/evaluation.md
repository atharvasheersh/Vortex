# Evaluation methodology

The [real-model experiment](real_rag.md) adds pretrained embeddings and local
generation over a source-separated SQuAD 2.0 subset. Its
[recorded aggregate results](../eval/reference/real-rag/) include article-cluster
bootstrap intervals, held-out per-condition counts and model provenance.
The fictional fixture described below retains its original scoring contract.

## Retrieval

ANN neighbor recall compares returned IDs against exhaustive top-k IDs on the same corpus and metric. Labeled retrieval metrics instead compare against the relevant_chunk_ids in a versioned dataset. They must not be combined under one ambiguous recall label.

Precision@k divides relevant results by requested k, even if fewer than k results are returned. Recall@k divides relevant results by the number of labeled relevant IDs. Hit@k is one if any relevant result appears. MRR@k uses the reciprocal rank of the first relevant result, or zero. nDCG@k uses gain 2^grade - 1 and a log2(rank + 1) discount; absent grades default to binary relevance.

Queries with no relevant labels are rejected by this evaluation contract. Duplicate retrieved IDs are rejected. Missing judgments are treated as non-relevant, so incomplete labels can understate quality. Graded relevance must use exactly the positively relevant IDs.

## Answer diagnostics

Normalized exact match compares case-folded word tokens. Token F1 uses multiset overlap with the reference answer. Lexical support measures the fraction of answer tokens present anywhere in supplied context. It does not check entailment, negation, attribution, ordering, numerical consistency, or factual truth. Empty answers have undefined lexical support.

The default pipeline uses hashed token features and extractive sentence selection. Its results establish deterministic integration behavior on fictional support facts. They do not establish pretrained embedding performance or LLM correctness/faithfulness. External embeddings and HTTP generation are executable optional paths and must be evaluated separately when configured.

Verbatim sentence support is a conservative grounding diagnostic: each generated sentence must occur as a contiguous normalized token sequence within a context sentence. This catches some unsupported recombinations that token overlap misses. It rejects valid paraphrases and is not a general semantic entailment evaluator; it also does not validate the truth of the source document. Report its full name rather than relabeling it as universal faithfulness.

The fixture keeps text in a separate ID lookup and passes only retrieved context to generation. Per-query records include IDs, scores, context, answer, and separated timings. Inspect these records to distinguish missing evidence from mishandled evidence.

## Systems measurements

Synthetic vectors and queries are independent standard-normal float32 arrays from a seeded NumPy generator. The corpus, queries, metric, k, and thread settings are held fixed across a parameter sweep. FAISS uses one thread, the same L2 inputs, the same list count, and the same number of clustering iterations. Different clustering implementations can produce different recall, so compare configurations at similar quality, not just matching nprobe.

Single-query percentiles include Python binding and input-copy overhead. Native result timing is also available. Worker scaling measures total batch wall time separately. Warm-ups are explicit. Application-cold trials reset the explicit cache before each query without flushing operating-system pages. Cache counters describe mapped data copies, not physical I/O.

The benchmark manifest includes process-wide peak working set/RSS using GetProcessMemoryInfo on Windows or getrusage on POSIX. This includes interpreter, arrays, index builders, mappings, and baselines and cannot be attributed to an individual configuration. Cached payload is measured separately. Snapshot size includes IDs, index metadata, originals, and compressed payload.

Published timings are measurements of one local run, not a statistically established speedup. Repeat runs, randomize configuration order, isolate the process, and use representative corpora before making deployment decisions.
