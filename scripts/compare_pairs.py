"""
Pairs CSV Comparison Report
============================
Compares the four pair CSV versions across key quality metrics and
quantifies how much each data-centric change reduced pair overlap
between successive versions.

Metrics reported per version:
  - Total pairs, unique identities, same/diff pair counts
  - Per-split label balance
  - Intra-version duplicate rate (pairs that appear more than once)

Overlap analysis:
  - Fingerprint intersection between consecutive versions (v1→v2, v2→v3, v3→v4)
  - Expressed as count and percentage of the smaller version's total pairs

Output:
  outputs/pairs_comparison.json   — machine-readable summary
  stdout                          — human-readable report table

Usage:
    python pairs_comparison.py
"""

import os
import json
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
VERSIONS = {
    "v1_baseline":          "configs/pairs.csv",
    "v2_uniform_weighting": "configs/pairs_v2.csv",
    "v3_deduplication":     "configs/pairs_v3.csv",
    "v4_identity_cap":      "configs/pairs_v4.csv",
}
OUTPUT_PATH = "outputs/pairs_comparison.json"
# ─────────────────────────────────────────────────────────────────────────────


def make_fingerprint(row: pd.Series) -> str:
    """
    Produces a canonical string key for a pair so (A,B,i,j) and (B,A,j,i)
    hash to the same value. Used for duplicate detection and overlap analysis.
    """
    pair = sorted([(row["left_identity"],  int(row["left_index"])),
                   (row["right_identity"], int(row["right_index"]))])
    return f"{pair[0][0]}_{pair[0][1]}--{pair[1][0]}_{pair[1][1]}"


def analyze_csv(path: str) -> dict | None:
    """
    Loads a pairs CSV and computes quality metrics.
    Returns None if the file does not exist.

    Returns a dict with:
      total_pairs, unique_identities, same_pairs, diff_pairs,
      duplicate_pairs, split_balance, fingerprints (set, used for overlap)
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["fingerprint"] = df.apply(make_fingerprint, axis=1)

    # Per-split label balance: {split: {label: count}}
    split_balance = {}
    for split_name in df["split"].unique():
        counts = df[df["split"] == split_name]["label"].value_counts().to_dict()
        split_balance[split_name] = {
            "same": int(counts.get(1, 0)),
            "diff": int(counts.get(0, 0)),
        }

    return {
        "total_pairs":       len(df),
        "unique_identities": int(pd.concat(
                                 [df["left_identity"], df["right_identity"]]
                             ).nunique()),
        "same_pairs":        int((df["label"] == 1).sum()),
        "diff_pairs":        int((df["label"] == 0).sum()),
        "duplicate_pairs":   int(df["fingerprint"].duplicated().sum()),
        "split_balance":     split_balance,
        "fingerprints":      set(df["fingerprint"].tolist()),
    }


def compute_overlap(stats_a: dict, stats_b: dict) -> dict:
    """
    Computes the fingerprint intersection between two versions.
    Returns count and percentage relative to the smaller version.
    """
    fp_a = stats_a["fingerprints"]
    fp_b = stats_b["fingerprints"]
    count = len(fp_a & fp_b)
    base  = min(stats_a["total_pairs"], stats_b["total_pairs"])
    return {
        "shared_pairs": count,
        "pct_of_smaller": round(count / base * 100, 1) if base > 0 else 0.0,
    }


def print_report(results: dict, overlaps: dict):
    """Prints a formatted comparison table to stdout."""
    labels  = list(results.keys())
    metrics = ["total_pairs", "unique_identities", "same_pairs",
               "diff_pairs", "duplicate_pairs"]

    col_w = 22
    header = f"{'Metric':<25}" + "".join(f"{l:<{col_w}}" for l in labels)
    print("\n── DATA-CENTRIC EVOLUTION REPORT ──────────────────────────────────────")
    print(header)
    print("─" * (25 + col_w * len(labels)))

    for m in metrics:
        row = f"{m:<25}"
        for label in labels:
            val = results[label].get(m, "N/A")
            row += f"{str(val):<{col_w}}"
        print(row)

    print("\n── PER-SPLIT LABEL BALANCE ─────────────────────────────────────────────")
    for label, stats in results.items():
        print(f"\n  {label}")
        for split_name, counts in stats["split_balance"].items():
            same, diff = counts["same"], counts["diff"]
            total = same + diff
            ratio = same / total if total > 0 else 0
            print(f"    {split_name:6s}: {same} same / {diff} diff  ({ratio:.1%} positive)")

    print("\n── PAIR OVERLAP BETWEEN VERSIONS ───────────────────────────────────────")
    for transition, overlap in overlaps.items():
        print(f"  {transition:<30}  "
              f"{overlap['shared_pairs']} shared pairs  "
              f"({overlap['pct_of_smaller']}% of smaller version)")

    print()


def run_comparison(versions: dict = VERSIONS,
                   output_path: str = OUTPUT_PATH) -> dict:
    """
    Loads all pair CSVs, computes per-version metrics and cross-version
    overlap, prints a report table, and saves a JSON summary.
    """
    # Load and analyze
    results = {}
    for name, path in versions.items():
        stats = analyze_csv(path)
        if stats:
            results[name] = stats
            print(f"  Loaded {name}  ({stats['total_pairs']} pairs)")
        else:
            print(f"  Skipped {name}  — file not found: {path}")

    if not results:
        print("No CSV files found. Check your configs/ directory.")
        return {}

    # Compute consecutive-version overlap
    version_keys = list(results.keys())
    overlaps = {}
    for i in range(len(version_keys) - 1):
        a, b = version_keys[i], version_keys[i + 1]
        if a in results and b in results:
            key = f"{a} → {b}"
            overlaps[key] = compute_overlap(results[a], results[b])

    # Print report
    print_report(results, overlaps)

    # Build serializable output (drop fingerprint sets — not JSON-serializable)
    serializable = {
        name: {k: v for k, v in stats.items() if k != "fingerprints"}
        for name, stats in results.items()
    }
    output = {
        "versions":   serializable,
        "overlaps":   overlaps,
        "files":      {name: path for name, path in versions.items()
                       if name in results},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Summary saved → {output_path}")

    return output


if __name__ == "__main__":
    run_comparison()