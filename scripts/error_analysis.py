"""
Error Analysis — Step 5.8
==========================
Produces two error slices from the val split and saves visual montages
and CSV exports for report inclusion.

Slice 1 — False Negatives:
    label=1, pred=0. Same-identity pairs the model said were different.
    Sorted ascending by score so the worst misses appear first.

Slice 2 — Boundary False Positives:
    label=0, pred=1, score within BOUNDARY_BAND of the threshold.
    Different-identity pairs the model was nearly certain about but got wrong.
    Sorted descending by score (closest to threshold first).

Outputs:
  outputs/error_analysis/slice1_fn.jpg
  outputs/error_analysis/slice2_fp.jpg
  outputs/error_analysis/fn_details.csv
  outputs/error_analysis/fp_details.csv

Usage:
    python scripts/error_analysis.py
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from keras_facenet import FaceNet

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.similarity import cosine_similarity

# ── CONFIG ───────────────────────────────────────────────────────────────────
PAIRS_CSV     = "configs/pairs_v4.csv"
RUN_JSON      = "outputs/runs/run_005.json"
LFW_BASE      = "data/lfw"
OUT_DIR       = "outputs/error_analysis"
THUMB_SIZE    = (120, 120)
GRID_COLS     = 4
BOUNDARY_BAND = 0.05
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def get_image_path(identity: str, index: int) -> str:
    """Constructs the expected LFW file path for a given identity and index."""
    filename = f"{identity}_{str(index).zfill(4)}.jpg"
    return os.path.join(LFW_BASE, identity, filename)


def safe_load(path: str) -> np.ndarray:
    """
    Loads an image as RGB. Returns a blank 160x160 image if the file is
    missing or unreadable, keeping the batch index aligned so downstream
    embedding calls don't receive a None and crash.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"  Warning: could not load {path} — substituting blank image")
        return np.zeros((160, 160, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_montage(pairs_df: pd.DataFrame, title: str, out_path: str,
                  threshold: float, max_pairs: int = 12):
    """
    Renders a visual grid of error pairs to a JPEG file.
    Each cell shows the left and right face thumbnails side by side,
    annotated with the similarity score and its delta from the threshold.
    Blue text = false negative (label 1), red text = false positive (label 0).
    """
    rows_data = pairs_df.head(max_pairs)
    n = len(rows_data)
    if n == 0:
        print(f"  Skipping montage (0 pairs): {title}")
        return

    cols   = min(GRID_COLS, n)
    n_rows = (n + cols - 1) // cols
    cell_w = THUMB_SIZE[0] * 2 + 10
    cell_h = THUMB_SIZE[1] + 40
    canvas = np.ones((n_rows * cell_h + 50, cols * cell_w + 20, 3),
                     dtype=np.uint8) * 255

    cv2.putText(canvas, title, (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

    for i, (_, row) in enumerate(rows_data.iterrows()):
        x0 = (i % cols) * cell_w + 10
        y0 = (i // cols) * cell_h + 50

        for j, side in enumerate(["left", "right"]):
            p   = get_image_path(row[f"{side}_identity"], int(row[f"{side}_index"]))
            img = cv2.imread(p)
            thumb = cv2.resize(img, THUMB_SIZE) if img is not None \
                    else np.zeros((*THUMB_SIZE, 3), dtype=np.uint8)
            x = x0 + j * (THUMB_SIZE[0] + 2)
            canvas[y0:y0 + THUMB_SIZE[1], x:x + THUMB_SIZE[0]] = thumb

        delta = row["score"] - threshold
        color = (0, 0, 200) if row["label"] == 1 else (200, 0, 0)
        cv2.putText(canvas, f"S:{row['score']:.3f} | D:{delta:+.3f}",
                    (x0, y0 + THUMB_SIZE[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path}")


def build_slices(val_df: pd.DataFrame,
                 threshold: float,
                 boundary_band: float = BOUNDARY_BAND):
    """
    Partitions val_df into the two error slices.

    Slice 1 — False Negatives: label=1, pred=0, sorted ascending by score.
    Slice 2 — Boundary FPs: label=0, pred=1, score <= threshold + band,
              sorted descending by score.
    """
    val_df = val_df.copy()
    val_df["pred"] = (val_df["score"] >= threshold).astype(int)

    slice1 = (val_df[(val_df["label"] == 1) & (val_df["pred"] == 0)]
              .sort_values("score", ascending=True)
              .reset_index(drop=True))

    slice2 = (val_df[(val_df["label"] == 0) & (val_df["pred"] == 1) &
                     (val_df["score"] <= threshold + boundary_band)]
              .sort_values("score", ascending=False)
              .reset_index(drop=True))

    return slice1, slice2


def run_error_analysis(pairs_csv: str = PAIRS_CSV,
                       run_json: str = RUN_JSON) -> tuple:
    """
    Full error analysis pipeline:
      1. Load pairs CSV and extract val split
      2. Load threshold from run JSON
      3. Generate embeddings and score all val pairs
      4. Build error slices
      5. Save montages and CSV exports

    Returns (slice1, slice2) DataFrames.
    """
    print("Starting Error Analysis...")

    if not os.path.exists(pairs_csv):
        print(f"Error: {pairs_csv} not found.")
        sys.exit(1)
    if not os.path.exists(run_json):
        print(f"Error: {run_json} not found.")
        sys.exit(1)

    # Load data
    df     = pd.read_csv(pairs_csv)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    with open(run_json) as f:
        run_data = json.load(f)
    threshold = run_data["best_threshold_from_val"]
    print(f"Threshold: {threshold:.3f}  |  Val pairs: {len(val_df)}")

    # Embed
    embedder    = FaceNet()
    left_paths  = [get_image_path(r.left_identity,  r.left_index)
                   for _, r in val_df.iterrows()]
    right_paths = [get_image_path(r.right_identity, r.right_index)
                   for _, r in val_df.iterrows()]

    print("Generating embeddings...")
    emb1 = embedder.embeddings([safe_load(p) for p in left_paths])
    emb2 = embedder.embeddings([safe_load(p) for p in right_paths])

    val_df["score"] = cosine_similarity(emb1, emb2)

    # Slice
    slice1, slice2 = build_slices(val_df, threshold)
    print(f"\nSlice 1 — False Negatives:          {len(slice1)}")
    print(f"Slice 2 — Boundary False Positives: {len(slice2)}")

    # Save montages
    build_montage(slice1,
                  "Slice 1: False Negatives (Same Person, Low Similarity)",
                  os.path.join(OUT_DIR, "slice1_fn.jpg"), threshold)
    build_montage(slice2,
                  "Slice 2: False Positives (Different People, High Similarity)",
                  os.path.join(OUT_DIR, "slice2_fp.jpg"), threshold)

    # Save CSVs
    slice1.to_csv(os.path.join(OUT_DIR, "fn_details.csv"), index=False)
    slice2.to_csv(os.path.join(OUT_DIR, "fp_details.csv"), index=False)

    print(f"\nDone. Outputs in {OUT_DIR}/")
    return slice1, slice2


if __name__ == "__main__":
    run_error_analysis()