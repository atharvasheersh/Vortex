# Real model evaluation specification

Frozen before model evaluation on 8 November 2026, starting from commit 84e95f6.

## Data and splits

Use SQuAD 2.0 official train and development JSON files from the Stanford project,
distributed under CC BY SA 4.0. Development tuning uses 50 answerable and 10
unanswerable questions from ten deterministically selected training articles.
The held out evaluation uses 150 answerable and 30 unanswerable questions from
the official development split. This local test is a subset of public development
data. Model pretraining exposure is possible.

Use all paragraphs of the selected training articles as the development corpus
and every paragraph in the official development split as the test corpus.
Select cases by SHA256 of question ID, taking at most one question per paragraph
within each answerability group. Exclude overlapping article titles across splits,
invalid answer offsets and answer spans longer than a chunk. Log exclusions.
Use 160 WordPiece token windows with 32 token overlap. Labels identify chunks
fully containing an annotated answer span. These span labels can be incomplete
for equivalent evidence elsewhere. Unanswerability is defined against the source
paragraph and requires manual qualification for the broader retrieval corpus.

## Models and inputs

Use sentence-transformers/all-MiniLM-L6-v2 at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 on CPU, normalized 384D vectors,
without query or document prefixes. Its model card specifies 256 WordPiece input
tokens. Keep questions within that limit and record all truncation.
Use the already installed local gemma3:1b Q4_K_M model via Ollama, recording
the complete model digest and server metadata before generation.

Generation uses one fixed evidence-only prompt, temperature zero, seed 42,
4096 runtime context tokens, at most 96 generated tokens and four CPU threads.
Bound evidence to 3000 UTF8 bytes as a conservative tokenizer-independent token
upper bound. Preserve chunk IDs, record truncated text and server token counts,
and reject requests whose reported prompt tokens exceed the context allowance.
Use one warmup request excluded from summaries and sequential requests thereafter.

## Retrieval selection

Use cosine, one retrieval thread, seed 42, reopened snapshots and 8 MiB cache.
Evaluate k in 3 and 5, nlist in 16 and 32, probes in 1, 4, 8 and all lists,
FP32, FP16, INT8, and optional reranking of 20 candidates using FP32 originals.
Record exact FP32 as reference. Compare every exact result to float64 NumPy
cosine ranking, allowing score ties within 0.00001. Require no discrepancies
outside that tolerance before generation.

Select k by highest development labeled Recall at k, breaking ties toward
smaller k. Select IVF by development ANN neighbor recall at least 0.95 and
labeled recall within 0.02 of exact at the chosen k, then lowest measured
median latency. Select one encoded reranked configuration by the same rules.
If none qualify, report the failed criterion and use the highest quality candidate
as a clearly marked fallback. Freeze selections before test retrieval/generation.
These are experiment selection gates. There is no predefined universal answer
quality pass threshold.

## Conditions and measurements

Generate no-context, source-evidence oracle, exact FP32, selected IVF and selected
encoded reranked answers. For unanswerable oracle cases supply the source paragraph.
Keep prompts, ordering and evidence budget constant across configurations.
Use the official SQuAD normalization and maximum exact match/token F1 over
reference answers, after removing well-formed chunk citations. Abstention requires
the exact marker INSUFFICIENT_EVIDENCE, optionally followed by citations.
Request failures and invalid output remain in all-case denominators with zero
answer scores. Report failures, empty outputs and generation length stops.

Report labeled retrieval metrics on answerable cases, ANN neighbor recall,
citation ID validity, lexical diagnostics, abstention rates, staged timings,
snapshot size and process memory. Citation correctness and semantic faithfulness
require evidence review. Prepare a blinded review sample and rubric with human
review status pending until actual annotations arrive. Do not infer entailment
from word overlap.

Use 2000 seeded source-article cluster bootstrap resamples for 95 percent
intervals and paired differences. Preserve cluster pairing across conditions.
This is one machine and one deterministic generation run per condition.

## Resources and evidence

Use existing local generation weights and freely accessible embedding weights.
No paid API, cloud provisioning or new gated model access is needed.
Use unique run directories with manifest-bound resume, per-request checkpoints,
bounded retries, explicit failure records and exclusive writer locks.
Keep datasets, full records, model caches and snapshots ignored. Publish compact
aggregate results, hashes and reproduction instructions. Target 200 answerable
and 40 unanswerable cases in total, preserving failures. Inspect evidence and
review the implementation before meaningful commits and the authorized main push.

Sources are the [Stanford dataset page](https://rajpurkar.github.io/SQuAD-explorer/),
[official scoring implementation](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py),
[embedding model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
and [Ollama generation API](https://docs.ollama.com/api/generate).
