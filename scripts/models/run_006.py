"""
Run 006 — First-Contact Rematch Threshold
==========================================
Failure mode : #1 Re-acquisition (person leaves + returns within ~60s).
Hypothesis   : Introduce REMATCH_THRESHOLD=0.60 used only on first-contact
               match against the enrolled embedding bank. Leave
               MATCH_THRESHOLD=0.70 untouched for cross-track dedup
               (_matches_another_candidate) and recent-cache writes, so
               two-unknowns-standing-near-each-other do not merge.
               A returning face whose cosine lands in the former UNCERTAIN
               dead-zone (0.60–0.70) now re-locks to its prior identity.

Scope        : `src/gatekeeper.py` only. No changes to offline scoring
               code paths, so offline F1 is expected to be statistically
               indistinguishable from run_005 — this script re-runs the
               pair eval as the regression-floor check per protocol.

Pairs        : configs/pairs_v4.csv  (same as run_005; inherits 15 %
               center-crop preprocessing via load_and_preprocess below).

Outputs      : outputs/runs/run_006.json
               outputs/plots/run_006_analysis.png (via scripts/plots/plot_roc.py)
               outputs/plots/run_006_confusion_matrix.png (via plot_confusion_matrix.py)
"""

import os
import sys
import json
import subprocess
import datetime
import cv2
import numpy as np
import pandas as pd
from keras_facenet import FaceNet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.similarity import cosine_similarity
from src.gatekeeper import MATCH_THRESHOLD, REMATCH_THRESHOLD  # imported for the config_diff

# ── CONFIG ───────────────────────────────────────────────────────────────────
PAIRS_CSV   = "configs/pairs_v4.csv"
LFW_BASE    = os.path.join(os.getcwd(), "data", "lfw")
OUT_DIR     = "outputs/runs"
RUN_ID      = "run_006_rematch_threshold"
CROP_MARGIN = 0.15    # same as run_005 — pairs_v4 + center-crop is the current baseline
TARGET_SIZE = (160, 160)

HYPOTHESIS = (
    "Introduce REMATCH_THRESHOLD=0.60 on first-contact matches so a "
    "returning face in the old UNCERTAIN dead-zone (0.60–0.70) re-locks "
    "to its prior identity, while MATCH_THRESHOLD=0.70 stays in place "
    "for cross-track dedup."
)
# REMATCH_THRESHOLD is a NEW knob — `null` in the `before` slot signals that.
CONFIG_DIFF = {
    "REMATCH_THRESHOLD": [None, REMATCH_THRESHOLD],
}
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def get_path(identity: str, index: int):
    path = os.path.join(LFW_BASE, identity, f"{identity}_{str(index).zfill(4)}.jpg")
    return path if os.path.exists(path) else None


def load_and_preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    top, bottom = int(h * CROP_MARGIN), int(h * (1 - CROP_MARGIN))
    left, right = int(w * CROP_MARGIN), int(w * (1 - CROP_MARGIN))
    return cv2.resize(img[top:bottom, left:right], TARGET_SIZE,
                      interpolation=cv2.INTER_LINEAR)


def load_pairs(split_df: pd.DataFrame):
    left_imgs, right_imgs, valid_indices = [], [], []
    for idx, row in split_df.iterrows():
        p1 = get_path(row["left_identity"],  int(row["left_index"]))
        p2 = get_path(row["right_identity"], int(row["right_index"]))
        if p1 and p2:
            left_imgs.append(load_and_preprocess(p1))
            right_imgs.append(load_and_preprocess(p2))
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


def _git_short_sha() -> str:
    """Best-effort; returns 'uncommitted' if this script runs before the
    run_006 commit lands."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or "uncommitted"
    except Exception:
        return "uncommitted"


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
print(f"Preprocessing with center-crop (margin={CROP_MARGIN})...")
embedder = FaceNet()

print("Threshold sweep on val...")
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
val_metrics = {k: v for k, v in best.items()}

# ── SAVE ARTIFACT ────────────────────────────────────────────────────────────
# Schema is additive: top-level legacy fields (run_id, val_sweep, test_metrics,
# best_threshold_from_val) stay identical to run_005 so scripts/plots/*.py
# keep working. The new-convention fields (hypothesis, config_diff, eval,
# decision, notes, created_at, git_sha) are added alongside.
output = {
    # Legacy fields (unchanged shape — preserves plot-script compatibility)
    "run_id":                  RUN_ID,
    "pairs_csv":               PAIRS_CSV,
    "preprocessing":           f"center_crop_margin={CROP_MARGIN}",
    "val_pairs":               len(val_final),
    "test_pairs":              len(test_final),
    "best_threshold_from_val": best_threshold,
    "val_sweep":               val_sweep,
    "test_metrics":            test_metrics,

    # New-convention fields
    "created_at":  datetime.datetime.now(datetime.timezone.utc)
                      .replace(microsecond=0).isoformat(),
    "git_sha":     _git_short_sha(),
    "hypothesis":  HYPOTHESIS,
    "config_diff": CONFIG_DIFF,
    "eval": {
        "offline": {
            "pairs_csv":      PAIRS_CSV,
            "best_threshold": best_threshold,
            "val_metrics":    val_metrics,
            "test_metrics":   test_metrics,
        },
        # Live scenarios are filled in by the operator after running S1–S6
        # (see docs/prompts/optimize_recognition.md). Left empty here so
        # the artifact is valid out-of-the-box — decision is "inconclusive"
        # until live counts are attached.
        "live": [],
    },
    "decision": "inconclusive",
    "notes": (
        "Offline re-run is a regression-floor check; this change only "
        "affects the live first-contact match path (REMATCH_THRESHOLD) "
        "and does not enter the offline pair-verification code. Awaiting "
        "operator S1–S6 protocol for decision. Cross-track dedup + cache "
        "writes still guarded by MATCH_THRESHOLD=0.70."
    ),
}

json_path = os.path.join(OUT_DIR, "run_006.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Results saved -> {json_path}")
print(f"Test F1: {test_metrics['f1']:.4f}  @  threshold {best_threshold:.2f}")
