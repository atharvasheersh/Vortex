# Scoped profiling pilots

Single-pass 100k/64D C++ microbenchmarks, twelve queries per search scope.
The original native path, first blocked prototype and fused-scoring prototype
were sampled at different times without fixed CPU frequency or a sampling
profiler. These expose costs and hypotheses; do not subtract their timings or
use their ratios as controlled speedup evidence. The paired held-out study
is the source of the reported p95 improvements.
