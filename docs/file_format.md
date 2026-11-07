# Snapshot format version 1

All integers and IEEE-754 floats are little-endian. A host with another byte order is rejected. The file contains one collection, with no ABI-dependent struct dumps.

| Header byte offset | Width | Value |
| --- | --- | --- |
| 0 | 8 | ASCII VORTEX01 |
| 8 | 4 | Version 1 |
| 12 | 4 | Dimension |
| 16 | 4 | Metric: 0 L2, 1 inner product, 2 cosine |
| 20 | 4 | Representation: 0 FP32, 1 FP16, 2 INT8 |
| 24 | 8 | Vector count |
| 32 | 8 | IVF list count, zero for no index |
| 40 | 8 | IDs section offset, always 128 |
| 48 | 8 | Original vector section offset |
| 56 | 8 | Encoded vector section offset |
| 64 | 8 | Centroid section offset |
| 72 | 8 | IVF offsets section offset |
| 80 | 8 | Posting section offset |
| 88 | 8 | Total file length |
| 96 | 8 | FNV-1a 64-bit checksum of bytes 0 through 95 |
| 104 | 24 | Reserved zero bytes |

Sections follow in this exact order:

1. N stable uint64 IDs.
2. N x D original FP32 components. For cosine these are normalized originals.
3. Optional encoded rows: D uint16 FP16 values, or a float scale followed by D int8 values. FP32 stores no duplicate encoded section.
4. Padding up to an 8-byte boundary.
5. nlist x D FP32 centroids.
6. nlist + 1 uint64 posting offsets, beginning at zero. A flat-only snapshot still contains one zero offset.
7. N uint64 row ordinals if nlist is positive; otherwise no postings.

The loader recalculates canonical section positions with overflow-checked arithmetic before allocating metadata. File length must match exactly, with no trailing bytes. Each indexed ordinal must be present exactly once. Duplicate external IDs, non-finite centroids, malformed offsets, and unsupported configuration values are rejected.

Payload bytes are not checksummed in version 1. Vector decoding rejects non-finite accessed FP32/FP16 values and invalid INT8 scales, but finite payload corruption can remain undetected. External modification or truncation of files under active mappings is unsupported.
