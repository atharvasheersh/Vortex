# Architecture and design decisions

Vortex separates mutable in-memory construction from immutable mapped query snapshots. This keeps concurrent query semantics explicit and avoids silently searching stale postings after insertion.

## Query execution

Queries must contain finite FP32 values and match the collection dimension. Cosine normalizes the corpus on insertion and each query before search; zero-norm vectors are rejected. L2 returns squared distances. Inner product and cosine return similarities. Internally a lower-is-better score supports one bounded heap implementation, with stable IDs breaking ties.

IVF uses seeded sampling and Euclidean Lloyd clustering on original vectors. Empty clusters are reseeded from training examples. Query routing ranks centroids using the configured metric. This is a heuristic routing policy; quality is measured rather than guaranteed for partial probing. Each corpus ordinal appears in exactly one posting list. Probing every list agrees with an exhaustive scan over the same representation.

Optional reranking first retains rerank_n candidates, reads their original FP32 vectors, recomputes scores, and returns k. It cannot recover vectors excluded during candidate generation.

## Storage and cache

A single snapshot provides consistent section metadata. Opening validates the header checksum, canonical layout, file length, IDs, finite centroids, monotonic list offsets, and posting coverage before queries are accepted. Vector payload checks are lazy to avoid eagerly decoding the corpus.

The block cache copies fixed-size blocks from the mapping, uses LRU eviction, and serializes cache accounting with a mutex. Blocks larger than the configured budget bypass caching. Resident payload stays within budget; allocator/container overhead, decoding scratch buffers, ID sets, centroids, postings, and mapped OS pages are additional memory.

FP16 uses finite binary16 values with round-to-nearest, ties-to-even. Values exceeding 65504 are rejected. INT8 stores a positive per-vector scale and signed values in [-127,127], using symmetric quantization with rounded values. Both representations retain originals for reranking. The first release studies compressed search behavior and does not claim total storage savings.

## Concurrency and mutation

A shared mutex permits concurrent searches and excludes mutation. A persistent worker pool distributes query batches, with no nested per-query parallelism. Batch failures drain all submitted work before unwinding, so tasks cannot outlive referenced query/result buffers. The Python layer copies validated inputs before releasing the GIL.

Insertions into builders are append-only, reject duplicate IDs, and invalidate an existing IVF index. Saving stale IVF state is rejected. Open snapshots reject insertion, retraining, and representation changes. File mappings are owned by the index and released on destruction.

## Publication and failures

A complete temporary snapshot is closed before replacing the target name. Readers see a coherent file on supported successful replacement paths. Windows rejects replacing some actively mapped destinations, which is reported as an error; publish another name or close readers first. An open POSIX mapping continues to reference the old inode after rename.

Concurrent writers to the same destination are unsupported. There is no fsync-based power-loss guarantee, multi-collection transaction, WAL, or corruption recovery. The header checksum is not a cryptographic integrity mechanism.

## External integration

Embeddings and generation stay outside C++. Chunk text is held separately and retrieved by returned IDs. The fixture and optional external-model paths share the same engine API. Per-query evidence and versioned run manifests support diagnosing retrieval separately from generation.
