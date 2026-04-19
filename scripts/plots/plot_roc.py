"""
Plot ROC + Threshold Trade-off — All Runs
==========================================
Reads every run JSON in outputs/runs/ and saves one analysis PNG per run.

Outputs:
  outputs/plots/run_001_analysis.png
  outputs/plots/run_002_analysis.png
  ...

Usage:
    python scripts/plots/plot_roc.py
"""

import os
import glob
import json
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Auto-discover every run_*.json artifact — keeps new runs plotting without
# maintaining a hardcoded list (prior list went stale on each new run).
RUNS_DIR  = "outputs/runs"
OUT_DIR   = "outputs/plots"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def plot_run(json_path: str, out_path: str):
    with open(json_path) as f:
        data = json.load(f)

    sweep  = data["val_sweep"]
    best_t = data["best_threshold_from_val"]
    run_id = data.get("run_id", os.path.basename(json_path))

    fpr        = [s["fpr"]       for s in sweep]
    tpr        = [s["tpr"]       for s in sweep]
    thresholds = [s["threshold"] for s in sweep]
    fps        = [s["fp"]        for s in sweep]
    fns        = [s["fn"]        for s in sweep]

    best_idx = min(range(len(thresholds)),
                   key=lambda i: abs(thresholds[i] - best_t))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(run_id, fontsize=13, fontweight="bold")

    # ROC curve
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label="Val ROC Curve")
    ax1.plot([0, 1], [0, 1], color="navy", linestyle="--", alpha=0.5)
    ax1.scatter(fpr[best_idx], tpr[best_idx], color="red", s=100, zorder=5,
                label=f"Chosen T={best_t:.2f}")
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title("ROC Curve (Validation Sweep)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Threshold trade-off
    ax2.plot(thresholds, fps, color="red",  lw=2, label="False Accepts (Security Risk)")
    ax2.plot(thresholds, fns, color="blue", lw=2, label="False Rejects (User Frustration)")
    ax2.axvline(x=best_t, color="black", linestyle="--", alpha=0.7,
                label=f"Optimal T ({best_t:.2f})")
    ax2.set_xlabel("Similarity Threshold")
    ax2.set_ylabel("Number of Pairs")
    ax2.set_title("Threshold Trade-off Analysis")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_all_runs():
    json_paths = sorted(glob.glob(os.path.join(RUNS_DIR, "run_*.json")))
    if not json_paths:
        print(f"  No run_*.json under {RUNS_DIR}/")
        return
    for json_path in json_paths:
        run_label = os.path.splitext(os.path.basename(json_path))[0]
        out_path  = os.path.join(OUT_DIR, f"{run_label}_analysis.png")
        plot_run(json_path, out_path)

    print(f"\nDone. All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    plot_all_runs()