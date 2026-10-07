# AO-PQ: Adaptive Online Product Quantization for Streaming Vector Embeddings

An asynchronous Multi-Version Concurrency Control (MVCC) vector storage engine designed for zero-downtime serving and continuous adaptation under streaming embedding drift.

## Key Features
- **Zero-Downtime Codebook Swaps**: Multi-version generation management decouples centroid updates from query serving paths.
- **$O(m \cdot K^2)$ Nearest-Centroid Discrete Morphing**: Eliminates $O(N \cdot d)$ vector decoding during background compaction.
- **SIMD Multi-Epoch ADC**: Search queries scan mixed generations in a single pass with sub-millisecond tail latency.
- **Bounded Generational Staleness**: Compaction debt is strictly controlled via an active sliding epoch window ($W$).
- **Dual-Mode Memory Architecture**: Transparent separation between pure standalone PQ index footprint (Mode A) and two-stage hybrid cache (Mode B).
