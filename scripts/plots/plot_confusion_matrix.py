"""
Plot Confusion Matrix — All Runs
==================================
Reads every run JSON in outputs/runs/ and saves one confusion matrix PNG per run.

Outputs:
  outputs/plots/run_001_confusion_matrix.png
  outputs/plots/run_002_confusion_matrix.png
  ...

Usage:
    python scripts/plots/plot_confusion_matrix.py
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Auto-discover every run_*.json artifact (see plot_roc.py for rationale).
RUNS_DIR  = "outputs/runs"
OUT_DIR   = "outputs/plots"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def plot_confusion_matrix(json_path: str, out_path: str):
    with open(json_path) as f:
        data = json.load(f)

    metrics = data["test_metrics"]
    best_t  = data["best_threshold_from_val"]
    run_id  = data.get("run_id", os.path.basename(json_path))

    tp, fp = metrics["tp"], metrics["fp"]
    fn, tn = metrics["fn"], metrics["tn"]

    # Layout: rows = Actual, cols = Predicted
    # [[TN, FP], [FN, TP]]
    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Diff (0)", "Predicted Same (1)"],
                yticklabels=["Actual Diff (0)",    "Actual Same (1)"],
                ax=ax)

    ax.set_title(f"{run_id}\nFinal Test Confusion Matrix  "
                 f"(Threshold: {best_t:.2f})", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)

    stats_text = (f"Test F1: {metrics['f1']:.4f}  |  "
                  f"TPR: {metrics['tpr']:.4f}  |  "
                  f"FPR: {metrics['fpr']:.4f}")
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=11,
             bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    plt.tight_layout(rect=[0, 0.07, 1, 1])
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
        out_path  = os.path.join(OUT_DIR, f"{run_label}_confusion_matrix.png")
        plot_confusion_matrix(json_path, out_path)

    print(f"\nDone. All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    plot_all_runs()