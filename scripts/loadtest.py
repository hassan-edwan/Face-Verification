"""
Concurrency / load test for the face verifier.

Drives the verifier with N worker threads for a fixed wall-clock duration,
cycling through pairs from --pairs (CSV with left_path,right_path), and
reports throughput and latency percentiles.

Example:
    python scripts/loadtest.py --pairs configs/loadtest_pairs.csv \
        --concurrency 4 --duration 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.verifier import FaceVerifier  # noqa: E402


def read_pairs(path: str) -> list[tuple[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"left_path", "right_path"}
        if not required.issubset(reader.fieldnames or []):
            sys.exit(
                f"error: {path} must have columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        return [(row["left_path"], row["right_path"]) for row in reader]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                        help="CSV with left_path,right_path")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of worker threads")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration in seconds")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup calls before timing starts")
    parser.add_argument("--output", default=None,
                        help="Optional path to write a JSON summary")
    args = parser.parse_args()

    pairs = read_pairs(args.pairs)
    if not pairs:
        sys.exit("error: no pairs to run")

    print(f"Loading verifier...")
    verifier = FaceVerifier()

    print(f"Warmup ({args.warmup} calls)...")
    for i in range(args.warmup):
        verifier.verify(*pairs[i % len(pairs)])

    print(f"Running: concurrency={args.concurrency}, duration={args.duration}s, "
          f"pair pool={len(pairs)}")

    latencies_ms: list[float] = []
    failures = {"count": 0}
    stop_at = time.perf_counter() + args.duration
    counter = {"i": 0}

    def worker():
        local, local_fails = [], 0
        while time.perf_counter() < stop_at:
            idx = counter["i"] % len(pairs)
            counter["i"] += 1
            left, right = pairs[idx]
            t0 = time.perf_counter()
            try:
                verifier.verify(left, right)
            except Exception:
                local_fails += 1
                continue
            local.append((time.perf_counter() - t0) * 1000.0)
        return local, local_fails

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(worker) for _ in range(args.concurrency)]
        for fut in as_completed(futs):
            lat, fails = fut.result()
            latencies_ms.extend(lat)
            failures["count"] += fails
    elapsed = time.perf_counter() - t_start

    if not latencies_ms:
        sys.exit(f"error: no successful calls recorded ({failures['count']} failures)")

    arr = np.asarray(latencies_ms)
    summary = {
        "concurrency":    args.concurrency,
        "duration_s":     args.duration,
        "pair_pool":      len(pairs),
        "completed":      int(len(arr)),
        "failures":       int(failures["count"]),
        "elapsed_s":      round(elapsed, 3),
        "throughput_pps": round(len(arr) / elapsed, 3),
        "latency_ms": {
            "mean": round(float(arr.mean()), 2),
            "p50":  round(float(np.percentile(arr, 50)), 2),
            "p90":  round(float(np.percentile(arr, 90)), 2),
            "p95":  round(float(np.percentile(arr, 95)), 2),
            "p99":  round(float(np.percentile(arr, 99)), 2),
            "max":  round(float(arr.max()), 2),
        },
    }

    print()
    print(f"Completed pairs : {summary['completed']}")
    print(f"Failures        : {summary['failures']}")
    print(f"Elapsed         : {summary['elapsed_s']} s")
    print(f"Throughput      : {summary['throughput_pps']} pairs/sec")
    print(f"Latency p50     : {summary['latency_ms']['p50']} ms")
    print(f"Latency p90     : {summary['latency_ms']['p90']} ms")
    print(f"Latency p95     : {summary['latency_ms']['p95']} ms")
    print(f"Latency p99     : {summary['latency_ms']['p99']} ms")
    print(f"Latency mean    : {summary['latency_ms']['mean']} ms")
    print(f"Latency max     : {summary['latency_ms']['max']} ms")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written → {args.output}")


if __name__ == "__main__":
    main()
