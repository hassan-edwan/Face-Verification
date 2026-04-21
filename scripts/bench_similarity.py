import numpy as np
import time
import os
from src.similarity import cosine_similarity, euclidean_distance

def run_benchmark(N=5000, D=128):
    # 1. Generate synthetic vectors
    rng = np.random.default_rng(42)
    a = rng.random((N, D)).astype(np.float32)
    b = rng.random((N, D)).astype(np.float32)

    # --- NumPy Vectorized Timing ---
    start_np = time.perf_counter()
    cos_np = cosine_similarity(a, b)
    euc_np = euclidean_distance(a, b)
    end_np = time.perf_counter()
    time_np = end_np - start_np

    # --- Python Loop Timing ---
    cos_loop = np.zeros(N)
    euc_loop = np.zeros(N)
    
    start_loop = time.perf_counter()
    for i in range(N):
        # Naive Cosine
        dot = np.sum(a[i] * b[i])
        norm = np.sqrt(np.sum(a[i]**2)) * np.sqrt(np.sum(b[i]**2))
        cos_loop[i] = dot / (norm + 1e-9)
        # Naive Euclidean
        euc_loop[i] = np.sqrt(np.sum((a[i] - b[i])**2))
    end_loop = time.perf_counter()
    time_loop = end_loop - start_loop

    # 2. Correctness Checks
    max_diff_cos = np.max(np.abs(cos_np - cos_loop))
    max_diff_euc = np.max(np.abs(euc_np - euc_loop))
    
    # 3. Report Results
    results = f"""
=== BENCHMARK RESULTS (N={N}, D={D}) ===
Vectorized Time: {time_np:.6f}s
Python Loop Time: {time_loop:.6f}s
Speedup Factor: {time_loop / time_np:.2f}x

=== CORRECTNESS CHECKS ===
Cosine Max Diff: {max_diff_cos:.2e}
Euclidean Max Diff: {max_diff_euc:.2e}
Status: {'PASSED' if max_diff_cos < 1e-6 else 'FAILED'}
"""
    print(results)
    
    # Save to outputs/
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/benchmark_results.txt", "w") as f:
        f.write(results)

if __name__ == "__main__":
    run_benchmark()