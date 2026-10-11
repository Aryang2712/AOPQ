# AO-PQ: Multi-Version Concurrency Control for Streaming Product Quantization

AO-PQ is an asynchronous vector quantization and storage engine designed for non-stationary, streaming embedding workloads. When embedding distributions experience concept drift, production vector engines (e.g., Faiss, Milvus) face an operational trade-off:
1. **Stop-the-World Retraining:** In-place centroid re-clustering freezes live reader queries for hundreds to thousands of milliseconds ($O(N \cdot d)$ pause).
2. **Blue-Green Rebuilds:** Building a secondary shadow index in the background avoids downtime, but requires a $2.0\times$ peak index memory spike.

AO-PQ introduces an **asynchronous Multi-Version Concurrency Control (MVCC)** protocol that decouples codebook updates from the live query path. Queries execute against concurrent epoch snapshots, while stale codes are translated out-of-line via $O(m \cdot K^2)$ discrete centroid morphing.

---

## Empirical Benchmark Summary

Evaluated on 128-dimensional vectors across three random seeds (macOS ARM64, Python 3.11):

| Corpus Scale ($N$) | FAISS In-Place STW Downtime | FAISS Blue-Green Peak RAM | FAISS P99 Read (`nprobe=10`) | FAISS Recall@10 | AO-PQ Max Cutover Jitter | AO-PQ P99 Read (Pure ADC) | AO-PQ Pure ADC Recall@10 | AO-PQ Two-Stage Recall@10 | AO-PQ Dropped Queries |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25,000** | $763.2\text{ ms}$ | $0.38\text{ MB}$ | $0.013\text{ ms}$ | $68.4\%$ | $0.80\text{ ms}$ | $0.34\text{ ms}$ | $65.2\%$ | $99.8\%$ | **0** |
| **50,000** | $1,445.4\text{ ms}$ | $0.76\text{ MB}$ | $0.012\text{ ms}$ | $67.1\%$ | $1.62\text{ ms}$ | $0.64\text{ ms}$ | $64.8\%$ | $99.7\%$ | **0** |
| **100,000** | $1,844.6\text{ ms}$ | $1.53\text{ MB}$ | $0.012\text{ ms}$ | $66.5\%$ | $3.43\text{ ms}$ | $1.11\text{ ms}$ | $64.1\%$ | $99.5\%$ | **0** |

*Full 1,000,000-vector streaming drift evaluations are available in `results/sift1m_benchmark_metrics.csv`.*

---

## Architecture & Systems Invariants

- **Snapshot Consistency:** Monotonic 64-bit epoch identifiers ensure query readers evaluate vector codes strictly against immutable, generation-matched centroid snapshots.
- **Lock-Free Multi-Epoch ADC:** Search queries scan contiguous chunk arenas in a single pass via a JIT-compiled Numba distance kernel (`nogil=True`).
- **$O(m \cdot K^2)$ Nearest-Centroid Morphing:** Eliminates intermediate $O(N \cdot d)$ floating-point decoding during background compaction by mapping codes through discrete subspace transition tables.
- **Bounded Generational Debt:** An active epoch window ($W=4$) caps live generational divergence, preventing unbounded memory fragmentation.

---

## Known Limitations & Future Work

1. **Read Latency Gap ($O(N)$ Flat Arena Scan):**
   - FAISS IVFPQ leverages inverted file space-partitioning (`nlist=100, nprobe=10`) to prune $90\%$ of vectors, maintaining $\sim 12\,\mu\text{s}$ read latency.
   - AO-PQ currently performs an exhaustive, unpruned scan across the flat chunk arena to verify MVCC correctness. Latency scales linearly with corpus size ($0.34\text{ ms}$ at 25k to $1.11\text{ ms}$ at 100k, reaching $\sim 8\text{ ms}$ at 1M scale).
   - **Future Work (IVF-AO-PQ):** Applying epoch versioning to inverted postings lists will prune the search space while preserving zero-downtime background codebook morphing.

2. **GIL Contention Under Write Concurrency:**
   - While distance kernels release the GIL (`nogil=True`), Python-level thread scheduling during high-frequency batch ingestion introduces observable reader jitter ($0.8\text{ ms} \to 3.4\text{ ms}$). A native C++/Rust storage layer is required for sub-millisecond tail guarantees under saturated write streams.

---

## Quickstart

```bash
# 1. Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the 18-Point Architectural Correctness Suite
python benchmark/test_correctness.py

# 3. Run the Rigorous Multi-Scale FAISS Benchmark
python benchmark/bench_faiss_ivfpq.py

# 4. Launch the Interactive Systems Telemetry Dashboard
streamlit run demo/app.py