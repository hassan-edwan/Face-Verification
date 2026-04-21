"""
Run 004 — Identity Cap
=======================
Pairs  : configs/pairs_v4.csv
Changes: Pair construction updated to cap each identity's contribution to
         at most MAX_PAIRS_PER_IDENTITY=3, ensuring the evaluation set draws
         from a maximally diverse population rather than being dominated by
         the ~20 heavily-photographed LFW identities.

Outputs:
  outputs/runs/run_004.json
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from keras_facenet import FaceNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.similarity import cosine_similarity

# ── CONFIG ───────────────────────────────────────────────────────────────────
PAIRS_CSV = "configs/pairs_v4.csv"
LFW_BASE  = os.path.join(os.getcwd(), "data", "lfw")
OUT_DIR   = "outputs/runs"
RUN_ID    = "run_004_identity_cap"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def get_path(identity: str, index: int):
    path = os.path.join(LFW_BASE, identity, f"{identity}_{str(index).zfill(4)}.jpg")
    return path if os.path.exists(path) else None


def load_pairs(split_df: pd.DataFrame):
    left_imgs, right_imgs, valid_indices = [], [], []
    for idx, row in split_df.iterrows():
        p1 = get_path(row["left_identity"],  int(row["left_index"]))
        p2 = get_path(row["right_identity"], int(row["right_index"]))
        if p1 and p2:
            img1, img2 = cv2.imread(p1), cv2.imread(p2)
            if img1 is not None and img2 is not None:
                left_imgs.append(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                right_imgs.append(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                valid_indices.append(idx)
    return left_imgs, right_imgs, split_df.loc[valid_indices].copy()


def get_scores(split_df: pd.DataFrame, embedder: FaceNet) -> pd.DataFrame:
    left_imgs, right_imgs, final_df = load_pairs(split_df)
    print(f"  Validated {len(final_df)} / {len(split_df)} pairs. Generating embeddings...")
    emb1 = embedder.embeddings(left_imgs)
    emb2 = embedder.embeddings(right_imgs)
    final_df["score"] = cosine_similarity(emb1, emb2)
    return final_df


def threshold_sweep(final_df: pd.DataFrame) -> list:
    sweep  = []
    y_true = final_df["label"].values
    scores = final_df["score"].values
    for t in np.linspace(0, 1, 101):
        y_pred = (scores >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1  = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        sweep.append({"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "tpr": float(tpr), "fpr": float(fpr), "f1": float(f1)})
    return sweep


# ── LOAD DATA ────────────────────────────────────────────────────────────────
print(f"Starting {RUN_ID}...")
if not os.path.exists(PAIRS_CSV):
    print(f"Error: {PAIRS_CSV} not found.")
    sys.exit(1)

df      = pd.read_csv(PAIRS_CSV)
val_df  = df[df["split"] == "val"].copy()
test_df = df[df["split"] == "test"].copy()
print(f"Val pairs: {len(val_df)}  |  Test pairs: {len(test_df)}")

# ── EMBEDDINGS & SCORES ──────────────────────────────────────────────────────
embedder = FaceNet()

print("Running threshold sweep on val split...")
val_final  = get_scores(val_df, embedder)
val_sweep  = threshold_sweep(val_final)
best       = max(val_sweep, key=lambda x: x["f1"])
best_threshold = best["threshold"]
print(f"Best threshold: {best_threshold:.2f}  (Val F1={best['f1']:.4f})")

print("Evaluating on test split with fixed threshold...")
test_final = get_scores(test_df, embedder)
y_true     = test_final["label"].values
y_pred     = (test_final["score"].values >= best_threshold).astype(int)
tp = int(np.sum((y_pred == 1) & (y_true == 1)))
fp = int(np.sum((y_pred == 1) & (y_true == 0)))
fn = int(np.sum((y_pred == 0) & (y_true == 1)))
tn = int(np.sum((y_pred == 0) & (y_true == 0)))
test_metrics = {
    "threshold": best_threshold,
    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
    "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    "f1":  float((2 * tp) / (2 * tp + fp + fn + 1e-9)),
}

# ── SAVE ARTIFACT ────────────────────────────────────────────────────────────
output = {
    "run_id":                  RUN_ID,
    "pairs_csv":               PAIRS_CSV,
    "preprocessing":           "none",
    "val_pairs":               len(val_final),
    "test_pairs":              len(test_final),
    "best_threshold_from_val": best_threshold,
    "val_sweep":               val_sweep,
    "test_metrics":            test_metrics,
}

json_path = os.path.join(OUT_DIR, "run_004.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Results saved → {json_path}")
print(f"Test F1: {test_metrics['f1']:.4f}  @  threshold {best_threshold:.2f}")