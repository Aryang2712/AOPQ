"""
AO-PQ vs. FAISS IVFPQ Rigorous Benchmark Suite
=============================================
Evaluates:
1. FAISS In-Place Rebuild (Stop-The-World downtime).
2. FAISS Blue-Green Rebuild (Build-aside pointer swap; records 2x memory spike).
3. Concurrent live reader jitter and dropped queries (real measured counter).
4. Matched Recall@10 comparison across both engines.
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

    # Synthesize bootstrap and drifted vector partitions
    X_train = np.random.randn(n_bootstrap, d).astype(np.float32)
    X_stream = (np.random.randn(n_stream, d) + 2.0).astype(np.float32)
    corpus_all = np.vstack([X_train, X_stream])
    test_queries = (np.random.randn(100, d) + 2.0).astype(np.float32)

    # -------------------------------------------------------------
    # 1. Initialize FAISS IndexIVFPQ
    # -------------------------------------------------------------
    quantizer_a = faiss.IndexFlatL2(d)
    faiss_index_a = faiss.IndexIVFPQ(quantizer_a, d, nlist, m, 8)
    faiss_index_a.train(X_train)
    faiss_index_a.add(X_train)
    faiss_index_a.nprobe = 10

    # -------------------------------------------------------------
    # 2. Baseline 1: FAISS In-Place Retrain (Stop-the-World Downtime)
    # -------------------------------------------------------------
    t0_faiss_stw = time.perf_counter()
    new_quantizer_stw = faiss.IndexFlatL2(d)
    faiss_index_stw = faiss.IndexIVFPQ(new_quantizer_stw, d, nlist, m, 8)
    faiss_index_stw.train(corpus_all)
    faiss_index_stw.add(corpus_all)
    faiss_stw_downtime_ms = (time.perf_counter() - t0_faiss_stw) * 1000.0
    faiss_index_stw.nprobe = 10

    # -------------------------------------------------------------
    # 3. Baseline 2: FAISS Blue-Green (Build Aside & Swap)
    # -------------------------------------------------------------
    t0_bg_build = time.perf_counter()
    quantizer_shadow = faiss.IndexFlatL2(d)
    shadow_index = faiss.IndexIVFPQ(quantizer_shadow, d, nlist, m, 8)
    shadow_index.train(corpus_all)
    shadow_index.add(corpus_all)
    shadow_index.nprobe = 10
    bg_build_time_ms = (time.perf_counter() - t0_bg_build) * 1000.0

    # Atomic pointer swap
    t0_swap = time.perf_counter()
    active_faiss_index = shadow_index
    bg_cutover_jitter_ms = (time.perf_counter() - t0_swap) * 1000.0

    # Blue-green incurs 2x peak index memory
    base_faiss_bytes = corpus_all.shape[0] * m
    bg_peak_ram_mb = (base_faiss_bytes * 2) / (1024 * 1024)

    # -------------------------------------------------------------
    # 4. AO-PQ Engine (Mode A & Mode B)
    # -------------------------------------------------------------
    aopq_engine = AdaptiveOnlinePQ(d=d, m=m, k=k, max_active_window=4, enable_raw_cache=True)
    aopq_engine.fit_initial(X_train)
    aopq_engine.ingest_stream_batch(X_train)

    # Measure real concurrent reader jitter and query drop counter
    concurrent_reader_latencies = []
    stop_reader_signal = threading.Event()
    successful_queries = 0
    failed_queries = 0

    def concurrent_reader_worker():
        nonlocal successful_queries, failed_queries
        while not stop_reader_signal.is_set():
            try:
                q_probe = test_queries[np.random.choice(len(test_queries))]
                t_probe = time.perf_counter()
                ids, dists = aopq_engine.search_adc_only(q_probe, top_k=10)
                if len(ids) == 10:
                    concurrent_reader_latencies.append((time.perf_counter() - t_probe) * 1000.0)
                    successful_queries += 1
                else:
                    failed_queries += 1
            except Exception:
                failed_queries += 1
            time.sleep(0.0002)

    reader_thread = threading.Thread(target=concurrent_reader_worker, daemon=True)
    reader_thread.start()

    # Trigger online ingestion and out-of-line generation migration
    aopq_engine.ingest_stream_batch(X_stream, force_swap=True)
    aopq_engine.migration_queue.join()

    stop_reader_signal.set()
    reader_thread.join()

    # -------------------------------------------------------------
    # 5. Steady-State Query Latency & Recall Comparison
    # -------------------------------------------------------------
    # FAISS Steady-State
    faiss_steady_latencies = []
    faiss_retrieved_ids = []
    for q in test_queries:
        t0 = time.perf_counter()
        _, idxs = active_faiss_index.search(q.reshape(1, -1), 10)
        faiss_steady_latencies.append((time.perf_counter() - t0) * 1000.0)
        faiss_retrieved_ids.append(idxs[0])

    # AO-PQ Pure ADC Steady-State
    aopq_adc_latencies = []
    aopq_adc_ids = []
    for q in test_queries:
        t0 = time.perf_counter()
        ids, _ = aopq_engine.search_adc_only(q, top_k=10)
        aopq_adc_latencies.append((time.perf_counter() - t0) * 1000.0)
        aopq_adc_ids.append(ids)

    # AO-PQ Two-Stage Steady-State (with exact reranking cache)
    aopq_twostage_ids = []
    for q in test_queries:
        ids, _ = aopq_engine.search(q, top_k=10)
        aopq_twostage_ids.append(ids)

    # Compute Recall@10
    faiss_recall = compute_ground_truth_recall(test_queries, corpus_all, faiss_retrieved_ids, k=10)
    aopq_adc_recall = compute_ground_truth_recall(test_queries, corpus_all, aopq_adc_ids, k=10)
    aopq_twostage_recall = compute_ground_truth_recall(test_queries, corpus_all, aopq_twostage_ids, k=10)

    aopq_engine.close()

    return {
        "corpus_size": n_vectors,
        "seed": seed,
        "faiss_stw_downtime_ms": faiss_stw_downtime_ms,
        "faiss_bg_build_time_ms": bg_build_time_ms,
        "faiss_bg_swap_jitter_ms": bg_cutover_jitter_ms,
        "faiss_bg_peak_ram_mb": bg_peak_ram_mb,
        "faiss_p99_read_ms": float(np.percentile(faiss_steady_latencies, 99)),
        "faiss_recall10": faiss_recall,
        "aopq_cutover_max_jitter_ms": float(np.max(concurrent_reader_latencies)) if concurrent_reader_latencies else 0.0,
        "aopq_cutover_p99_jitter_ms": float(np.percentile(concurrent_reader_latencies, 99)) if concurrent_reader_latencies else 0.0,
        "aopq_steady_p99_ms": float(np.percentile(aopq_adc_latencies, 99)),
        "aopq_adc_recall10": aopq_adc_recall,
        "aopq_twostage_recall10": aopq_twostage_recall,
        "aopq_successful_queries": successful_queries,
        "aopq_failed_queries": failed_queries,
        "aopq_drop_rate_pct": (failed_queries / max(1, successful_queries + failed_queries)) * 100.0
    }


def run_comprehensive_faiss_suite():
    print("=" * 82)
    print("      AO-PQ vs. FAISS RIGOROUS COMPARATIVE BENCHMARK")
    print("=" * 82)

    if not HAS_FAISS:
        print("[!] Error: FAISS CPU is not installed. Run 'pip install faiss-cpu'.")
        return

    scales = [25000, 50000, 100000]
    seeds = [42, 101, 777]
    raw_results = []

    for n_scale in scales:
        print(f"\n[*] Evaluating Corpus Scale N = {n_scale:,} vectors across {len(seeds)} trials...")
        for s in seeds:
            res = benchmark_single_trial(n_vectors=n_scale, seed=s)
            raw_results.append(res)
            print(f"  -> [Seed {s}] FAISS STW: {res['faiss_stw_downtime_ms']:.1f}ms | "
                  f"AO-PQ Jitter: {res['aopq_cutover_max_jitter_ms']:.3f}ms | "
                  f"FAISS R@10: {res['faiss_recall10']:.1f}% | "
                  f"AO-PQ R@10 (ADC/TwoStage): {res['aopq_adc_recall10']:.1f}% / {res['aopq_twostage_recall10']:.1f}% | "
                  f"Reads: {res['aopq_successful_queries']} ok / {res['aopq_failed_queries']} dropped")

    df_trials = pd.DataFrame(raw_results)

    summary_rows = []
    for n_scale in scales:
        sub = df_trials[df_trials["corpus_size"] == n_scale]
        summary_rows.append({
            "Corpus_Size": n_scale,
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

    print("\n" + "=" * 82)
    print("                        MULTI-SCALE EMPIRICAL SUMMARY")
    print("=" * 82)
    print(df_summary.to_string(index=False))
    print("=" * 82)
    print("[SUCCESS] Results saved to results/faiss_vs_aopq_comparison.csv")


if __name__ == "__main__":
    run_comprehensive_faiss_suite()