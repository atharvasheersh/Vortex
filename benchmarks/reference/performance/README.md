# Retrieval optimization evidence

This archive contains 8,924 final calls across 20 build/layout cases and 768
routing-tuning calls. Queries/reference matrices, per-condition manifests,
selection, paired comparisons and measured source bytes are retained.

Run `python benchmarks/validate_performance.py` from the repository root.
Large snapshots and native binaries remain outside Git; their hashes are in
the manifests. Native build revisions are distinguished from collector-tree
revisions. The baseline binary matches the earlier capacity module hash.

Read `docs/performance_optimization.md` and `benchmarks/performance_protocol.md`
for the experiment order, resource policy, failed held-out routing selection,
one-round 10m limitation and exact reproduction commands. Recall is nearest-
neighbor recall, not answer accuracy. The old archives remain separate.

The executed-source files retain the bytes used during final collection.
Subsequent maintained-helper hardening handles zero-only INT8 rows, explicit
scalar diagnostics and invalid time budgets; the recorded Gaussian inputs and
all-auto measurement conditions are unchanged by those fixes.
