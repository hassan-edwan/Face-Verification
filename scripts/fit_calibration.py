"""
Fit Platt-scaling calibration on the val split of pairs_v4.csv.

Reads outputs/runs/run_005.json for the frozen decision threshold, replays the
run_005 embedding pipeline over the val split, fits a sigmoid mapping from
cosine score to P(same identity), and writes configs/calibration.json.

Platt form: P(y=1 | s) = 1 / (1 + exp(A*s + B))
Parameters are fit by minimising binary cross-entropy with scipy.optimize.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.verifier import load_and_preprocess  # noqa: E402
from src.similarity import cosine_similarity  # noqa: E402

PAIRS_CSV = "configs/pairs_v4.csv"
LFW_BASE = os.path.join(os.getcwd(), "data", "lfw")
RUN_JSON = "outputs/runs/run_005.json"
OUT_PATH = "configs/calibration.json"


def get_path(identity: str, index: int):
    p = os.path.join(LFW_BASE, identity, f"{identity}_{str(index).zfill(4)}.jpg")
    return p if os.path.exists(p) else None


def collect_val_scores(embedder):
    df = pd.read_csv(PAIRS_CSV)
    val_df = df[df["split"] == "val"].copy()

    lefts, rights, labels = [], [], []
    for _, row in val_df.iterrows():
        p1 = get_path(row["left_identity"], int(row["left_index"]))
        p2 = get_path(row["right_identity"], int(row["right_index"]))
        if p1 and p2:
            lefts.append(load_and_preprocess(p1))
            rights.append(load_and_preprocess(p2))
            labels.append(int(row["label"]))

    print(f"  Validated {len(labels)} / {len(val_df)} val pairs.")
    emb_l = embedder.embeddings(lefts)
    emb_r = embedder.embeddings(rights)
    scores = cosine_similarity(emb_l, emb_r)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


PLATT_L2 = 1e-4  # small ridge penalty so Nelder-Mead stays finite on near-separable data


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit P(y=1|s) = 1 / (1 + exp(A*s + B)) by minimising binary cross-entropy
    with a tiny L2 penalty. The ridge keeps the optimum bounded when val data
    is (near-)separable and BFGS/Nelder-Mead would otherwise diverge.

    Loss derivation:
        P(y=1) = 1/(1+e^z)  →  -log P(y=1) = log(1+e^z) = logaddexp(0, z)
        P(y=0) = e^z/(1+e^z) → -log P(y=0) = logaddexp(0, z) - z
        combined: logaddexp(0, z) - (1 - y) * z
    """

    def nll(params):
        a, b = params
        z = a * scores + b
        loss = np.logaddexp(0.0, z) - (1 - labels) * z
        return float(np.mean(loss) + PLATT_L2 * (a * a + b * b))

    res = minimize(nll, x0=np.array([-5.0, 2.0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
    return float(res.x[0]), float(res.x[1])


def main():
    if not os.path.exists(RUN_JSON):
        sys.exit(f"Error: {RUN_JSON} not found. Run: python -m scripts.models.run_005")
    if not os.path.exists(PAIRS_CSV):
        sys.exit(f"Error: {PAIRS_CSV} not found.")

    with open(RUN_JSON) as f:
        threshold = float(json.load(f)["best_threshold_from_val"])
    print(f"Using decision threshold {threshold:.3f} from {RUN_JSON}")

    from keras_facenet import FaceNet
    print("Loading FaceNet...")
    embedder = FaceNet()

    print("Computing val scores...")
    scores, labels = collect_val_scores(embedder)

    print("Fitting Platt sigmoid...")
    a, b = fit_platt(scores, labels)
    print(f"  A = {a:.6f}   B = {b:.6f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "platt_a": a,
            "platt_b": b,
            "threshold": threshold,
            "source_run": RUN_JSON,
            "pairs_csv": PAIRS_CSV,
            "n_val_pairs": int(len(scores)),
        }, f, indent=2)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
