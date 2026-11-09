# CPU embedding comparison

This study asks which embedding candidates offer useful quality and resource
tradeoffs for Vortex on one shared CPU laptop. Four downloadable encoders run
locally. Voyage and Gemini remain pending at the owner's request. No API calls
or charges are included. The selection is a practical candidate set, with no
claim that these are the six highest ranked models on a general benchmark.

The [frozen specification](../eval/embedding_comparison_spec.md) defines the
selection rules before comparative retrieval scores. The
[model registry](../eval/embedding_models.json) pins weights, reviewed custom
Python code, licenses, dimensions, precision and retrieval formatting.
Native dimensions are retained, so memory and search costs differ across models.
Jina's weights have a noncommercial license.

## Models and sources

| Candidate | Native dimensions | Configuration | Source |
| --- | ---: | --- | --- |
| MiniLM L6 v2 | 384 | FP32 baseline | [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Qwen3 Embedding 0.6B | 1024 | FP32, query instruction | [Model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Jina embeddings v5 text nano | 768 | FP32 retrieval adapter | [Model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano) |
| EmbeddingGemma 2 | 768 | FP32 official text-only encoder | [Model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) |
| Voyage 4 large | 1024 | Pending hosted query/document input types | [Provider documentation](https://docs.voyageai.com/docs/embeddings) |
| Gemini embedding 2 | 3072 | Pending hosted text task/title formatting | [Provider documentation](https://ai.google.dev/gemini-api/docs/embeddings) |

Sources and pins were inspected on 2026-11-09. Jina's nested adapter loaders omit
revision propagation, so the implementation downloads an immutable snapshot,
checks the reviewed Python file hashes and loads from that local directory.
EmbeddingGemma uses `vision_config=None` and `audio_config=None`; its generic
processor still requires the image dependencies in the installed library release.
Neutral-text pilots selected FP32 for Qwen because BF16 was slower on this CPU.
Pilot measurements are separate from retrieval quality.

## Data and scoring

SQuAD uses 50 positive development queries and 150 positive held-out queries,
with separate 10/30 paragraph-negative cohorts. New held-out questions exclude
all 240 previously inspected cases. Training development articles and held-out
articles are disjoint. Chunk positives are source spans containing an annotated
answer. Equivalent evidence elsewhere can be unjudged. A paragraph-negative
label does not prove absence of an answer from the entire searchable corpus.
[SQuAD source and license](https://rajpurkar.github.io/SQuAD-explorer/)

SciFact uses 50 official training-qrel claims for development and 150 official
test-qrel claims for evaluation. Its 1,000-document corpus includes every selected
positive document plus hash-selected distractors. This label-informed restriction
makes the workload feasible locally and changes its difficulty. These results
must not be described as standard full-BEIR scores. The searchable scientific
corpus is shared across splits, while positive-document groups and queries are
disjoint. Full abstracts become 3,328 chunks. Each document receives its maximum
chunk cosine score, then ranking removes duplicate documents.
[SciFact distribution](https://huggingface.co/datasets/BeIR/scifact)

All models see identical chunk bodies and titles. Chunk bodies use 128 MiniLM
WordPieces with 24 overlap, preserving source offsets. Every actual formatted
input is checked against the relevant encoder limit. The run rejects overlong
inputs instead of silently truncating them. The prepared corpus contains 662
development and 2,188 test SQuAD chunks. SciFact's 3,328 chunks are searchable
for both splits. References and qrels are scoring inputs, with references also
permitted in the explicitly labeled source oracle only.

Exact cosine results from reopened FP32 Vortex snapshots are checked against
independent normalized float64 NumPy scores, with 1e-5 tie/score tolerance.
Positive-query metrics include Recall, Hit, MRR and binary nDCG at 5 and 10.
Scientific metrics rank unique documents. Negative cases retain retrieval and
timing records but do not enter positive ranking denominators.

Development tuning evaluates a bounded list/probe/precision/reranking grid.
Selections must retain at least 0.95 mean neighbor recall against the model's own
exact neighbors and lose at most 0.02 labeled Recall@10. Fastest qualifying median
latency wins. Full-list probing is identified explicitly. Compression snapshots
retain FP32 originals, so encoded-vector size is not total snapshot size.

## Timing and uncertainty

Encoders load sequentially in separate CPU processes with four inference threads.
Corpus batches contain four chunks. Query timing uses batch one and three shuffled
rounds after a neutral-text warmup. Encoding measurements include formatting,
token-length validation and normalization, while excluding model loading and
checkpoint disk writes. Loading and process-wide peak RSS are reported separately.

Retrieval runs after encoder processes exit, with one Vortex thread, one NumPy
BLAS thread and an 8 MiB mapped block cache. Held-out exact and selected
configurations use three shuffled timing rounds. Progressive chunk overfetch and
document deduplication are included in retrieval latency. Independent-reference
calculation is excluded. These are warm-process shared-desktop measurements;
background load, OS file cache and clock drift are uncontrolled. Retrieval-client
RSS excludes the separate generator server and encoder process.
Retrieval peak RSS is the high-water mark of a CLI process that handles multiple
configurations and models; it is not isolated per-index allocation. Current RSS
is recorded alongside it. Encoder peaks come from separate per-model processes.

The execution-only second revision reuses scientific document batches after an
ordered chunk-record identity check. It was frozen before comparative scores,
preserving the same model/data settings and the original contract. Imported
pre-score batches retain checksums and original times. Reported corpus throughput
uses original encoding time; newly spent encoding time and reused-batch counts
are separate fields. The conservative API forecast still counts both scientific
corpus passes. Actual hosted requests and charges remain zero.

Quality intervals and paired differences use 2,000 seeded source-group bootstrap
resamples. SQuAD groups are articles; SciFact groups connect claims sharing positive
documents. Intervals are exploratory and have no multiple-comparison correction.
They do not account for pretraining contamination or incomplete relevance labels.
The reported Pareto set uses observed exact nDCG@10, query p95, encoder-process
peak RSS and FP32 snapshot bytes. Dominance among point estimates does not account
for interval overlap or uncontrolled timing noise.

During collection, three Qwen SQuAD document batches recorded approximately
51 minutes, 5.1 hours and 1.9 hours of elapsed time, while the median batch took
about six seconds. Windows logs show sleep/Modern Standby events during this
period. The raw full-corpus elapsed throughput includes those interruptions and
is retained unchanged. Its value describes the interrupted job, not steady CPU
inference. The report includes batch-time distributions and the private OS-log
hash rather than removing slow observations.

A [separate timing audit](../eval/embedding_timing_audit_spec.md) was defined after
that anomaly and before comparative quality scores. It uses fixed small samples
from development, the same model constructors and reverse model order, with both
elapsed and process CPU clocks and recorded power state. It also checks actual
parameter dtypes, devices and inference threads. These supplemental measurements
do not change primary vectors, quality metrics, gates or selected candidates.
The sample is 16 documents and ten queries per domain, with three rounds; its
throughput and latency are sample diagnostics. Process CPU seconds sum work over
all process threads and can exceed elapsed seconds.
The completed supplemental runs recorded battery power for EmbeddingGemma, Jina
and Qwen, and AC power for MiniLM, with battery saver off. No sampled call changed
power state or triggered the declared low-CPU-ratio flag. Power nevertheless
differed between models; these are not controlled matched-power comparisons.
Actual audit parameter counts are 22.7M, 595.8M, 238.9M and 271.0M for MiniLM,
Qwen, Jina and the text-only Gemma constructor respectively, all FP32 on CPU.
Jina's count includes the loaded adapters; Gemma's downloaded checkpoint also
contains multimodal weights that the text-only constructor does not retain.

## RAG conditions

Two challengers are selected using mean development exact nDCG@10 across domains,
subject to encoder-process peak RSS at most 4 GiB and development query p95 at
most 5 seconds. MiniLM remains the control. Selection is frozen before held-out
retrieval scores. Fifty positive and ten paragraph-negative SQuAD cases form
the generation subset. SciFact is evaluated as document retrieval only.

All three retrieval conditions use exact top-5 with the same local `gemma3:1b`
digest, prompt, 3,000-byte evidence budget, 96-token output limit, temperature zero
and seed 42. Source oracle and no-context controls share the settings. Five
questions repeat twice more in every condition, and condition order is shuffled
per question. Prompts, truncation, server token counts, failures and attempts stay
in private checkpoints. Fixed settings can still produce varying answers.
Generation records sample the model-server processes separately from the Python
client and retain runtime-reported VRAM allocation. Recorded total stage times
sum measurements collected at separate offline stages. A simultaneous live
encoder/retrieval/generator service would need its combined memory and load tested.
The earlier real-RAG experiment used different cases, chunking and encoder inputs.
Its MiniLM answer score and this study's control score have different experimental
conditions; comparisons here use the frozen common cohort within this study.

Official SQuAD normalization/max-reference scores measure lexical answer agreement.
The run verifies its scores against the [official evaluation script](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py).
Raw-empty, citation-only, empty-scored, abstained and failed requests have separate
counts. Citation presence uses substantive answers as its denominator; valid-ID
rates use citation mentions. Neither establishes claim support. Lexical and
verbatim support are named diagnostics. A blinded sample and rubric are prepared
privately. Human semantic/citation review stays pending; no semantic judge is used.
These counters overlap. Empty-scored answers include abstentions converted to
empty predictions, whereas citation-only means nonempty raw output with no text
left after citation removal. Failed requests also have empty raw responses.
The report also traces original annotated character offsets through the exact
submitted UTF8 byte budget. It counts positive queries whose annotated source
span was requested and whose complete span survived submission. This is a literal
source-span diagnostic, with incomplete alternative-evidence labels; it does not
establish semantic coverage. Private diagnostics track retrieval misses, chunking
exclusions, annotated span loss, zero lexical oracle scores and invalid IDs. Empty
categories remain explicit rather than receiving invented examples.

## Measured findings

Jina has the highest mean exact SQuAD nDCG@10 at 88.17%, compared with MiniLM's
75.55%. EmbeddingGemma leads the restricted SciFact corpus at 87.73%, compared
with 80.85%. Jina's paired SQuAD difference is +12.62 points with an exploratory
95% interval of [+7.78, +17.73]; Gemma's scientific difference is +6.88 points
with [+1.69, +12.31]. Qwen's scientific interval includes zero. The intervals
compare each candidate with MiniLM, not every challenger with every other one.

MiniLM used 0.57 GiB encoder peak RSS, versus 2.80 for Qwen, 1.49 for Jina and
1.81 for Gemma. Its smaller 384-dimensional vectors also use less snapshot space.
Jina offers the strongest observed SQuAD quality/resource tradeoff among the
challengers, subject to its noncommercial license. Gemma is a useful scientific
retrieval candidate with Apache 2.0 weights. Qwen is dominated by Jina on the
reported point estimates, but these measurements do not establish universal
dominance or a controlled comparison of latency at matched power.

Only MiniLM on both domains and Qwen on SciFact retained both development gates
on the held-out k=10 results. Qwen SQuAD and Jina SQuAD missed both gates; Jina
SciFact missed neighbor recall; Gemma SQuAD missed labeled recall; Gemma SciFact
missed both. These five misses are preserved in the JSON/CSV. Development quality
matching did not reliably generalize. Test results did not trigger retuning.

| RAG condition | Positive answer F1 | Positive exact match | Paired F1 difference from MiniLM, points with 95% interval |
| --- | ---: | ---: | --- |
| No context | 3.08% | 2.00% | -38.21 [-50.72, -26.42] |
| Source oracle | 60.69% | 40.00% | +19.40 [+7.96, +31.30] |
| MiniLM | 41.30% | 28.00% | Reference |
| Jina | 43.84% | 26.00% | +2.54 [-7.79, +12.66] |
| EmbeddingGemma | 38.03% | 22.00% | -3.26 [-16.25, +9.35] |

All conditions use the same 50 positive and ten paragraph-negative questions.
All 350 main/repeated requests succeeded in one attempt; a separate 30-request
smoke also completed. Seven of 75 identical-prompt pairs have different normalized
answers. Both challengers' answer-F1 difference intervals include zero. Better
retrieval therefore did not establish better generated answers with this generator.

MiniLM, Gemma and Jina requested annotated gold chunks on 41, 46 and 49 of 50
positive cases respectively. All those annotated spans survived the actual context
budget. This distinguishes the observed misses from literal loss of annotated
spans during truncation, without judging alternative evidence or semantic coverage.
Citation presence among substantive responses was 2/57 for MiniLM, 1/60 for Gemma
and 0/60 for Jina. Paragraph-negative abstention was 1/10, 0/10 and 0/10 respectively.
Human semantic and citation review remains pending. The generator processes reached
3.71 GiB maximum sampled combined RSS, separately from encoder and client memory;
the runtime reported zero VRAM allocation. Concurrent live-service capacity was
not measured.

## Reproduction

Install a C++20 compiler and the package first. On the recorded Windows environment
the compiler is MinGW GCC 16.1 and Python is 3.12.14. The study dependency pins are
in [embedding_requirements.txt](../eval/embedding_requirements.txt). A fresh install
also needs normal project build prerequisites from the README. Keep downloaded
data, weights, full responses and snapshots under ignored paths.

```powershell
python -m pip install -r eval/embedding_requirements.txt
python -m pip install ".[test]"
$env:HF_HOME = Join-Path (Get-Location) 'data/embedding-study/huggingface'
New-Item -ItemType Directory -Force data/real-rag, data/embedding-study
Invoke-WebRequest https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -OutFile data/real-rag/train-v2.0.json
Invoke-WebRequest https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -OutFile data/real-rag/dev-v2.0.json
Invoke-WebRequest https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py -OutFile data/real-rag/evaluate-v2.0.py
Invoke-WebRequest https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -OutFile data/embedding-study/scifact.zip
python eval/prepare_real_rag.py
python eval/prepare_embedding_study.py
python eval/run_embedding_study.py forecast
foreach ($model in @('minilm','qwen','jina','embeddinggemma')) {
  python eval/run_embedding_study.py embed --model $model
  if ($LASTEXITCODE -ne 0) { throw "Encoding failed" }
}
python eval/run_embedding_study.py tune
python eval/run_embedding_study.py heldout
python eval/generate_embedding_rag.py --smoke
python eval/generate_embedding_rag.py
foreach ($model in @('embeddinggemma','jina','qwen','minilm')) {
  python eval/run_embedding_timing_audit.py --model $model
  if ($LASTEXITCODE -ne 0) { throw "Timing audit failed" }
}
python eval/report_embedding_study.py
python eval/validate_embedding_study.py
python eval/validate_embedding_study.py --dataset data/embedding-study/prepared --run results/embedding-study-20261109-v3
```

The prior-data preparation recreates the exclusion cohort; it refuses an existing
output directory. Use the recorded raw-source hashes and model registry to verify
inputs. Rerun an interrupted encoding/generation command unchanged to resume.
The run refuses source, model or input contract drift. Never remove an active
writer lock; after a terminated process, inspect its recorded PID before manually
removing its stale lock. Reports also acquire the run lock and reject nested writers.
For this existing environment managed by uv, use
`uv pip install --python .venv/Scripts/python.exe -r eval/embedding_requirements.txt`
and `uv pip install --python .venv/Scripts/python.exe ".[test]"` for installation.
The recorded execution used the existing private shared model cache; immutable
snapshot copies and checksums are retained under ignored data/embedding-study/model-snapshots.
The HF_HOME setting above keeps fresh reproduction downloads inside ignored data/.

Generation requires the existing local Ollama server and the exact digest in the
specification; it verifies the digest before requests. The hosted model adapters
are preparation scaffolding and have not received integration validation. The
study CLI rejects hosted execution. Enabling future hosted/resumed runs requires
owner-approved credentials and a spending cap, plus durable cross-process budget
accounting and provider response/version tests before paid execution.

Public aggregate evidence includes all six completion states, safe provenance,
CSV measurements and checksums under eval/reference/embedding-comparison. Full
blinded reviews and examples remain private. The expanded owner report is ignored
at docs/embedding_comparison_report.md; the two owner Word documents also stay ignored.
