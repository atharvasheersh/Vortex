# Recorded real model quality results

One local CPU experiment on a held-out subset of public SQuAD 2.0 development data. Each condition has 150 answerable and 30 source-unanswerable questions. Full per-query records remain under ignored results/real-rag-20261108.

| Condition | Answer F1 | Answer EM | Source negative abstention | Failures |
| --- | ---: | ---: | ---: | ---: |
| no_context | 2.85% | 0.67% | 73.33% | 0 |
| oracle | 59.94% | 39.33% | 3.33% | 0 |
| exact | 43.53% | 26.67% | 3.33% | 0 |
| ivf | 43.29% | 26.67% | 3.33% | 0 |
| encoded | 43.29% | 26.67% | 3.33% | 0 |

All rates retain their declared denominators. The full JSON includes article-cluster bootstrap intervals and paired F1 differences. Source-unanswerable labels describe the original paragraph and require qualification for the full retrieval corpus.

The exact answer F1 interval is 36.60% to 50.75%. Exact labeled Recall@5 is
82.67%, with Hit@5 of 84%. All 900 HTTP requests completed without retries.
Sixteen citation-only answers were retained with zero answer scores.

Abstention and citation compliance are weak in this experiment. The oracle uses
the required insufficient-evidence marker on 1 of 30 source-unanswerable cases.
Exact retrieval includes citations in 6 of 175 substantive responses. Its eight
citation mentions all use supplied IDs. Ninety-five exact contexts are truncated.

The selected full-list configurations share all 180 exact prompts, yet 21
normalized answers differ. The small F1 difference includes observed runtime
variation at fixed requested settings. The historical collector is retained in
runner_snapshot.txt. The maintained runner corrects reference-score lookup for
nonconsecutive stable IDs. The recorded prepared corpus uses consecutive IDs.

The generator is the existing gemma3:1b Q4_K_M installation and the encoder is commit-pinned all-MiniLM-L6-v2. Model identities, dataset hashes, source hashes and package versions are in provenance.json. The prompt and options are recorded in the frozen specification and private run manifest.

Human semantic review remains pending at the owner's request. Citation ID validity, reference-answer metrics and lexical diagnostics have their stated meanings. Semantic correctness, citation correctness and unsupported-claim rates are unreported until evidence review.

Generation and total timings reflect fixed execution order and model-server prompt caching. Full-list selection means every IVF list was probed. This experiment establishes the recorded behavior for this dataset subset and these model versions.

Validate the public exports with python eval/validate_real_rag.py. Add --run results/real-rag-20261108 --dataset data/real-rag/prepared to verify local records, source files and scoring against the published hashes.

The SQuAD dataset is attributed to Rajpurkar, Jia and Liang and distributed under CC BY SA 4.0. Original data and benchmark information are available at https://rajpurkar.github.io/SQuAD-explorer/. Public exports contain aggregates and provenance.
