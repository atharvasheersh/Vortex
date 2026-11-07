# Recorded local scaling evidence

These measurements were collected on engine revision `13450d720ad45ae65388f9d356674609f11771b3`, before the runner and reports were packaged for publication. They cover 1,000, 10,000, 100,000 and 1,000,000 independent synthetic 64D FP32 vectors.

Each size directory contains the original result metadata and all 2,460 timed query records. `runner_snapshot.txt` preserves the historical runner byte for byte, with SHA-256 `b2963e7892086668e4b160a2eed3cff4cd43ee5de693122c44991ec8bc0cd9c0`. That identity matches the runner hash captured inside every run manifest. The maintained entry point is now `benchmarks/run_scaling.py`; its CLI checks and evidence-preservation behavior were added afterward and did not generate these timings.

`checksums.json` inventories the historical runner and all eight result/query files. Text exports use LF line endings. `summary.json` and `summary.csv` are derived from the result records. Large `.vtx` snapshots, partial progress files and private owner documents are excluded.

The manifests record the original source fingerprint and working-tree dirty state. The dirty flags include incidental untracked local files; no tracked native-engine modifications were made for the measurement run. The later publication commits add benchmark tooling, evidence and documentation.

From the repository root:

```bash
python benchmarks/validate_scaling.py
python benchmarks/report_scaling.py --input-dir benchmarks/reference/scaling --output-dir results/scaling-report
```

Validation checks file hashes, query coverage, stable repeated results, all-list exact agreement, mapped/in-memory agreement, recall, round percentiles, operating-point selection, cache payload bounds and derived summary values. It does not recreate the unavailable historical operating-system cache state or turn repeated queries into independent statistical samples.

For interpretation, see [the scaling report](../../../docs/scalability.md). High-recall IVF is slower than exact search at the larger sizes in this fixed 32-list experiment, and the mapped path has additional cache costs. The results do not validate one million higher-dimensional embeddings or production service capacity.
