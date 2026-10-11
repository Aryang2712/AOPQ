"""
AO-PQ vs. FAISS Rigorous Comparative Benchmark (Clustered Workload)
===================================================================
Measures:
1. FAISS In-Place Stop-The-World pause vs. FAISS Blue-Green Build-Aside vs. AO-PQ.
2. Ground-truth Recall@10 on manifold-clustered vectors.
3. Concurrent live reader jitter with strictly counted query success/drop tracking.
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import AdaptiveOnlinePQ

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def load_evaluation_data(n_total: int, d: int = 128):
    """Loads clustered SIFT vectors if present; otherwise generates structured cluster blobs."""
    sift_path = "datasets/sift1m/sift_base.fvecs"
    if os.path.exists(sift_path):
        record_size = 4 + d * 4
        with open(sift_path, "rb") as f:
            raw = f.read(n_total * record_size)
            data = np.zeros((n_total, d), dtype=np.float32)
            for i in range(n_total):
                offset = i * record_size + 4
                data[i] = np.frombuffer(raw[offset : offset + d * 4], dtype=np.float32)
        return data
    else:
        # Structured manifold clusters (NOT isotropic white noise)
        n_clusters = 20
        centers = np.random.randn(n_clusters, d) * 15.0
        cluster_ids = np.random.choice(n_clusters, size=n_total)
        return (centers[cluster_ids] + np.random.randn(n_total, d)).astype(np.float32)


def compute_ground_truth_recall(queries, corpus, retrieved_ids, k=10):
    recalls = []
    for i, q in enumerate(queries):
        gt_dists = np.sum((corpus - q) ** 2, axis=1)
        gt_topk = set(np.argpartition(gt_dists, k)[:k])
        hits = len(gt_topk.intersection(set(retrieved_ids[i])))
        recalls.append(hits / float(k))
    return float(np.mean(recalls)) * 100.0


def benchmark_single_trial(n_vectors: int, seed: int, d: int = 128, m: int = 8, k: int = 256, nlist: int = 100):
    np.random.seed(seed)
    n_bootstrap = n_vectors // 2
    n_stream = n_vectors // 2

    data_all = load_evaluation_data(n_vectors, d=d)
    X_train = data_all[:n_bootstrap]
    # Add continuous manifold drift to the streaming partition
    X_stream = data_all[n_bootstrap:] + 1.5
    corpus_all = np.vstack([X_train, X_stream])

    # Sample on-manifold queries
    q_idx = np.random.choice(len(X_stream), size=min(100, len(X_stream)), replace=False)
    test_queries = X_stream[q_idx] + np.random.normal(0, 0.5, size=(len(q_idx), d)).astype(np.float32)

    # 1. FAISS IVFPQ Setup
    quantizer_a = faiss.IndexFlatL2(d)
    faiss_index_a = faiss.IndexIVFPQ(quantizer_a, d, nlist, m, 8)
    faiss_index_a.train(X_train)
    faiss_index_a.add(X_train)
    faiss_index_a.nprobe = 10

    # 2. FAISS In-Place Stop-The-World Rebuild
    t0_stw = time.perf_counter()
    quantizer_stw = faiss.IndexFlatL2(d)
    faiss_index_stw = faiss.IndexIVFPQ(quantizer_stw, d, nlist, m, 8)
    faiss_index_stw.train(corpus_all)
    faiss_index_stw.add(corpus_all)
    faiss_stw_downtime_ms = (time.perf_counter() - t0_stw) * 1000.0
    faiss_index_stw.nprobe = 10

    # 3. FAISS Blue-Green Build-Aside
    t0_bg = time.perf_counter()
    quantizer_bg = faiss.IndexFlatL2(d)
    faiss_index_bg = faiss.IndexIVFPQ(quantizer_bg, d, nlist, m, 8)
    faiss_index_bg.train(corpus_all)
    faiss_index_bg.add(corpus_all)
    faiss_bg_build_time_ms = (time.perf_counter() - t0_bg) * 1000.0
    faiss_index_bg.nprobe = 10

    t0_swap = time.perf_counter()
    active_faiss = faiss_index_bg
    faiss_bg_swap_jitter_ms = (time.perf_counter() - t0_swap) * 1000.0

    # 4. AO-PQ Engine Execution
    aopq = AdaptiveOnlinePQ(d=d, m=m, k=k, max_active_window=4, enable_raw_cache=True)
    aopq.fit_initial(X_train)
    aopq.ingest_stream_batch(X_train)

    concurrent_reader_latencies = []
    stop_signal = threading.Event()
    successful_reads = 0
    failed_reads = 0

    def reader_loop():
        nonlocal successful_reads, failed_reads
        while not stop_signal.is_set():
            try:
                probe_q = test_queries[np.random.choice(len(test_queries))]
                t0 = time.perf_counter()
                ids, dists = aopq.search_adc_only(probe_q, top_k=10)
                if len(ids) == 10:
                    concurrent_reader_latencies.append((time.perf_counter() - t0) * 1000.0)
                    successful_reads += 1
                else:
                    failed_reads += 1
            except Exception:
                failed_reads += 1
            time.sleep(0.0002)

    reader_th = threading.Thread(target=reader_loop, daemon=True)
    reader_th.start()

    aopq.ingest_stream_batch(X_stream, force_swap=True)
    aopq.migration_queue.join()

    stop_signal.set()
    reader_th.join()

    # 5. Latency & Recall Evaluation
    faiss_latencies, faiss_ids = [], []
    for q in test_queries:
        t0 = time.perf_counter()
        _, idxs = active_faiss.search(q.reshape(1, -1), 10)
        faiss_latencies.append((time.perf_counter() - t0) * 1000.0)
        faiss_ids.append(idxs[0])

    aopq_adc_latencies, aopq_adc_ids = [], []
    for q in test_queries:
        t0 = time.perf_counter()
        ids, _ = aopq.search_adc_only(q, top_k=10)
        aopq_adc_latencies.append((time.perf_counter() - t0) * 1000.0)
        aopq_adc_ids.append(ids)

    aopq_twostage_ids = []
    for q in test_queries:
        ids, _ = aopq.search(q, top_k=10)
        aopq_twostage_ids.append(ids)

    faiss_r10 = compute_ground_truth_recall(test_queries, corpus_all, faiss_ids, k=10)
    aopq_adc_r10 = compute_ground_truth_recall(test_queries, corpus_all, aopq_adc_ids, k=10)
    aopq_twostage_r10 = compute_ground_truth_recall(test_queries, corpus_all, aopq_twostage_ids, k=10)

    aopq.close()

    return {
        "corpus_size": n_vectors,
        "seed": seed,
        "faiss_stw_downtime_ms": faiss_stw_downtime_ms,
        "faiss_bg_build_time_ms": faiss_bg_build_time_ms,
        "faiss_bg_swap_jitter_ms": faiss_bg_swap_jitter_ms,
        "faiss_p99_read_ms": float(np.percentile(faiss_latencies, 99)),
        "faiss_recall10": faiss_r10,
        "aopq_cutover_max_jitter_ms": float(np.max(concurrent_reader_latencies)) if concurrent_reader_latencies else 0.0,
        "aopq_cutover_p99_jitter_ms": float(np.percentile(concurrent_reader_latencies, 99)) if concurrent_reader_latencies else 0.0,
        "aopq_steady_p99_ms": float(np.percentile(aopq_adc_latencies, 99)),
        "aopq_adc_recall10": aopq_adc_r10,
        "aopq_twostage_recall10": aopq_twostage_r10,
        "aopq_successful_queries": successful_reads,
        "aopq_failed_queries": failed_reads
    }


def run_benchmark():
    if not HAS_FAISS:
        print("[!] FAISS is required. Install with 'pip install faiss-cpu'.")
        return

    scales = [25000, 50000, 100000]
    seeds = [42, 101, 777]
    raw_results = []

    print("=" * 80)
    print("      AO-PQ vs. FAISS COMPARATIVE BENCHMARK (CLUSTERED WORKLOAD)")
    print("=" * 80)

    for n in scales:
        for s in seeds:
            res = benchmark_single_trial(n_vectors=n, seed=s)
            raw_results.append(res)
            print(f"[{n:,} vecs | Seed {s}] FAISS STW: {res['faiss_stw_downtime_ms']:.1f}ms | "
                  f"AO-PQ Jitter: {res['aopq_cutover_max_jitter_ms']:.2f}ms | "
                  f"FAISS R@10: {res['faiss_recall10']:.1f}% | "
                  f"AO-PQ TwoStage R@10: {res['aopq_twostage_recall10']:.1f}% | "
                  f"Reads: {res['aopq_successful_queries']} ok / {res['aopq_failed_queries']} dropped")

    df_trials = pd.DataFrame(raw_results)
    summary_rows = []
    for n in scales:
        sub = df_trials[df_trials["corpus_size"] == n]
        summary_rows.append({
            "Corpus_Size": n,
            "FAISS_STW_Downtime_ms": sub["faiss_stw_downtime_ms"].mean(),
            "FAISS_BG_Build_Time_ms": sub["faiss_bg_build_time_ms"].mean(),
            "FAISS_P99_Read_ms": sub["faiss_p99_read_ms"].mean(),
            "FAISS_Recall10": sub["faiss_recall10"].mean(),
            "AOPQ_Max_Jitter_ms": sub["aopq_cutover_max_jitter_ms"].mean(),
            "AOPQ_P99_Read_ms": sub["aopq_steady_p99_ms"].mean(),
            "AOPQ_ADC_Recall10": sub["aopq_adc_recall10"].mean(),
            "AOPQ_TwoStage_Recall10": sub["aopq_twostage_recall10"].mean(),
            "AOPQ_Failed_Queries": int(sub["aopq_failed_queries"].sum())
        })

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs("results", exist_ok=True)
    df_trials.to_csv("results/faiss_vs_aopq_trials.csv", index=False)
    df_summary.to_csv("results/faiss_vs_aopq_comparison.csv", index=False)
    print("\n[SUCCESS] Benchmark completed. Metrics written to results/faiss_vs_aopq_comparison.csv")


if __name__ == "__main__":
    run_benchmark()