# AO-PQ: Multi-Version Concurrency Control for Streaming Product Quantization

AO-PQ is an asynchronous vector quantization and storage engine designed for non-stationary, streaming embedding workloads. When embedding distributions experience concept drift, production vector engines (e.g., Faiss, Milvus) face operational trade-offs:
1. **In-Place Retraining:** Centroid re-clustering freezes live reader queries ($O(N \cdot d)$ stop-the-world pause).
2. **Blue-Green Rebuilds:** Building a secondary index in the background achieves zero downtime, but requires a $2.0\times$ peak index memory spike and repeated full-corpus clustering.

AO-PQ introduces an **asynchronous Multi-Version Concurrency Control (MVCC)** protocol that decouples codebook updates from the live query path. Queries execute against concurrent epoch snapshots, while stale codes are translated out-of-line via $O(m \cdot K^2)$ discrete centroid morphing without full retraining.

---

## Empirical Benchmark Summary

Evaluated on 128-dimensional clustered vector workloads across three random seeds:

<!-- BENCHMARK_TABLE_START -->
| Corpus Scale ($N$) | FAISS In-Place STW | FAISS BG Build Time | FAISS P99 Read | FAISS Recall@10 | AO-PQ Cutover Max Jitter | AO-PQ P99 Read | AO-PQ Pure ADC R@10 | AO-PQ Two-Stage R@10 | Dropped Queries |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25,000** | 747.9 ms | 750.6 ms | 0.027 ms | 60.2% | 0.75 ms | 0.39 ms | 55.7% | 96.4% | **0** |
| **50,000** | 1485.6 ms | 1481.8 ms | 0.034 ms | 60.9% | 1.53 ms | 0.69 ms | 55.1% | 98.9% | **0** |
| **100,000** | 1962.4 ms | 1963.1 ms | 0.042 ms | 57.5% | 2.69 ms | 1.29 ms | 53.7% | 99.8% | **0** |
<!-- BENCHMARK_TABLE_END -->

*Full 1,000,000-vector continuous drift evaluations on SIFT-1M are available in `results/sift1m_benchmark_metrics.csv`.*

---

## Architectural Comparison: AO-PQ vs. Blue-Green

| Dimension | Standard Blue-Green Rebuild | AO-PQ (This Work) |
| :--- | :--- | :--- |
| **Query Downtime** | $0.00\text{ ms}$ (Atomic pointer swap) | $0.00\text{ ms}$ (Lock-free multi-epoch reads) |
| **Peak Index RAM Overhead** | **$2.0\times$ (Full duplicate index)** | **Bounded by $W=4$ ($\le 1.05\times$)** |
| **Background Rebuild Cost** | $O(N \cdot d)$ (Re-clusters all records) | $O(m \cdot K^2)$ (Discrete centroid translation) |
| **Recall Drift Profile** | Sawtooth drops between rebuild cycles | Continuous retention via adaptive promotion |

---

## Known Limitations & Scope

1. **Read Latency Gap ($O(N)$ Flat Arena Scan):**
   - FAISS IVFPQ leverages inverted file space-partitioning (`nlist=100, nprobe=10`) to prune $90\%$ of vectors, maintaining $\sim 0.01\text{--}0.04\text{ ms}$ read latency.
   - AO-PQ currently performs an unpruned flat scan across contiguous chunks to isolate and verify the MVCC concurrency protocol. Latency scales linearly with corpus size ($0.3\text{ ms}$ at 25k to $1.1\text{ ms}$ at 100k, and $\sim 8\text{ ms}$ at 1M scale).
   - **Future Work (IVF-AO-PQ):** Integrating inverted list partitioning on top of multi-epoch chunk arenas will address the read latency gap while preserving zero-downtime morphing.

2. **GIL Contention Under Write Concurrency:**
   - While Numba kernels release the GIL (`nogil=True`), Python-level thread scheduling during high-frequency batch ingestion introduces observable reader jitter ($0.8\text{ ms} \to 3.4\text{ ms}$). A native C++/Rust implementation would provide sub-millisecond tail guarantees under saturated write streams.

---

## Quickstart

```bash
# 1. Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the Invariant Correctness Audit
python benchmark/test_correctness.py

# 3. Run the Comparative FAISS Benchmark
python benchmark/bench_faiss_ivfpq.py

# 4. Synchronize the README Table
python benchmark/update_readme_table.py

# 5. Launch the Dashboard
streamlit run demo/app.py