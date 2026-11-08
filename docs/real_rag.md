# Real model RAG evaluation

The real-data experiment uses a pretrained encoder, mapped Vortex snapshots and
a local language model. The frozen specification defines the corpus, case
selection, development tuning and held-out evaluation before inspecting results.
Its results apply to this dataset subset and these model versions.

## Data preparation

Install the package with its test, benchmark and embeddings extras. Download the
official SQuAD 2.0 train and development JSON files into data/real-rag. The
preparation command writes a new directory and refuses to replace an existing
dataset. Keep the downloaded files and prepared data under ignored data/.

```powershell
python -m pip install ".[test,benchmark,embeddings]"
New-Item -ItemType Directory -Force data/real-rag
Invoke-WebRequest https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -OutFile data/real-rag/train-v2.0.json
Invoke-WebRequest https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -OutFile data/real-rag/dev-v2.0.json
python eval/prepare_real_rag.py
```

For the existing environment managed with uv, install the extras with
`uv pip install --python .venv/Scripts/python.exe ".[test,benchmark,embeddings]"`.

The dataset is attributed to Rajpurkar, Jia and Liang and distributed under
CC BY SA 4.0. Raw text and derived chunks remain local. Preparation records the
original download URLs, SHA256 hashes, selected articles, case IDs, chunk offsets,
answer-span mappings and excluded cases. Corpus restriction and sampled cases
are described in the [experiment specification](../eval/real_rag_spec.md).
The Stanford project publishes the data and scoring protocol. [Dataset source](https://rajpurkar.github.io/SQuAD-explorer/)

The experiment uses the full official development paragraph corpus for the
held-out local test. Its tuning corpus comprises ten selected training articles.
It retains official train/development membership and checks article/source and
exact chunk-text separation. This local held-out subset comes from public
development data and can have appeared in model training.

## Model execution

The encoder is all-MiniLM-L6-v2 at the revision in the specification. Its
384-dimensional embeddings are normalized. Chunks use 160 WordPiece tokens with
32-token overlap to stay within the encoder's 256-token input limit. Questions
exceeding that limit are excluded before sampling. [Encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The recorded generator is an existing gemma3:1b Q4_K_M installation. The runner
uses the local Ollama API, records its server version, weight digest, model
metadata, prompt and decoding settings, and verifies those identities on resume.
The generation service must already be running and have the selected model.
Users obtaining model weights must follow the applicable model terms.
The request and response fields follow the runtime documentation. [Generation API](https://docs.ollama.com/api/generate)

Use a new output directory for each experiment. Run stages in the following order.

```powershell
python eval/run_real_rag.py embed --output results/real-rag-new
python eval/run_real_rag.py tune --output results/real-rag-new
python eval/run_real_rag.py generate --output results/real-rag-new --split development --smoke
python eval/run_real_rag.py generate --output results/real-rag-new --split test
python eval/report_real_rag.py --run results/real-rag-new --output results/real-rag-new/report
```

Exact FP32 retrieval is checked against independent float64 NumPy cosine scores.
Development selection chooses k, an IVF configuration and an encoded reranked
configuration, then freezes them in selection.json before test retrieval.
The test generation conditions are no context, source evidence, exact FP32,
selected IVF, and selected encoded reranking.
The no-context condition retains the same evidence-only prompt and measures
behavior when supporting evidence is absent from the input.

Evidence receives a conservative 3000 UTF8 byte budget, with truncation recorded
per chunk. The response stores the runtime's actual prompt and output token
counts. The same prompt and options apply to all conditions. A warmup request is
excluded from reported timings. Query embedding, retrieval and generation run
in separate stages, so the reported total is their measured-stage sum.
Conditions execute in the fixed order shown above. The model server retains its
prompt cache, and later conditions can reuse matching prefixes. Generation and
total timings therefore reflect this execution order and cache state.
Independent NumPy scoring precedes each timed retrieval, so the corpus has
recently been accessed in memory. The experiment runs on a shared desktop with
background applications.

## Recovery and evidence

Re-running a generation command resumes completed and failed records using
condition and question ID. Existing failures remain in the final denominators.
Changing the specification, runner, helper, dataset or model configuration
requires a new run directory. A writer lock rejects concurrent writers. After a
process is forcibly terminated, inspect .writer.lock and confirm its recorded
PID is no longer active before removing the lock to resume.
The report command acquires the same lock before reading or repairing records,
so an active inference writer prevents report generation.

Each completed record is flushed to disk. An interrupted final JSONL fragment is
saved as an interrupted sidecar, and complete records are retained. Malformed
complete records or duplicate keys cause validation to stop. HTTP requests use
bounded retries, and all attempts and failures are retained.

The report requires the complete expected condition/question grid. It produces
JSON and CSV summaries, paired article-bootstrap confidence intervals, and a
blinded evidence-review sample with a separate configuration mapping. Keep full
records and review annotations under ignored results/. The public compact
evidence contains aggregate results and provenance hashes.

## Meaning of the scores

Answer exact match and token F1 use the official SQuAD normalization, maximum
score over supplied reference answers, and removal of well-formed chunk
citations. The exact abstention marker is INSUFFICIENT_EVIDENCE. Failed requests
and empty responses receive zero answer scores. Source-unanswerable abstention,
answerable abstentions and request failures have separate counts.
The official scorer supplies the reference rules. [Scoring source](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py)

Labeled retrieval metrics use chunks containing annotated answer spans. ANN
neighbor recall compares IDs with exact cosine neighbors. Span judgments can
miss equivalent evidence in other chunks. Unanswerable labels describe the
original paragraph, so corpus-wide negative judgments require further review.

Citation ID validity checks whether mentioned IDs occur in the actual supplied
context. Citation correctness, claim support, evidence coverage and semantic
answer correctness require the documented evidence-review rubric. The generated
review file starts with empty annotation fields and pending human review status.
Lexical support and verbatim sentence support retain their diagnostic names.

## Recorded findings

The [compact results and provenance](../eval/reference/real-rag/) record 900
held-out requests with zero HTTP failures or retries. Sixteen citation-only responses
remain in the denominator with zero answer scores. Exact retrieval measured
43.53% answer token F1, 26.67% answer exact match, 82.67% labeled Recall@5 and
84% labeled Hit@5. The oracle measured 59.94% answer F1.

The selected IVF and encoded reranking configurations probe every list and
match all exact neighbors. Each matches the exact condition's prompt on 180
questions, yet 21 normalized generated answers differ. Fixed request settings
therefore coexist with observed output variation. The source of that variation
was not isolated, and the small F1 difference includes runtime behavior.

The oracle abstained with the required marker on 1 of 30 source-unanswerable
questions. Exact retrieval supplied references verbatim in 129 of 150 positive
contexts. It produced citations in 6 of 175 substantive responses. All eight
mentions used supplied IDs. Evidence was truncated in 95 of 180 exact records.
These counts distinguish reference-text occurrence, citation presence and ID
validity from the semantic review that remains pending.

The historical collector is archived as runner_snapshot.txt. The maintained
runner uses a stable-ID lookup for independent reference scoring, including
nonconsecutive IDs. The measured prepared corpus uses consecutive IDs. Both
collector hashes are recorded in the public provenance.

Validate public exports or extend the check to private evidence with these commands.

```powershell
python eval/validate_real_rag.py
python eval/validate_real_rag.py --run results/real-rag-20261108 --dataset data/real-rag/prepared
```

The private expanded explanation is docs/real_rag_quality_report.md, excluded
from Git together with the two private Word documents. The blinded 60-record
review sample and rubric are under results/real-rag-20261108/report. Human review
remains pending at the owner's explicit request.
