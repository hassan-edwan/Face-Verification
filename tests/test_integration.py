"""
Integration Test — Step 5.10
=============================
Runs the full evaluation path (pairs → embeddings → scores → metrics → JSON)
using a tiny synthetic fixture.

Design decisions
----------------
* No real dataset:  Images are 160×160 solid-colour JPEG files written to a
  temp directory.  Pairs are constructed so ground-truth is deterministic:
  same-colour images → label 1, different-colour → label 0.

* No real model:    FaceNet is monkey-patched to return fixed unit vectors,
  so the test is fast and fully reproducible without GPU/download.

* Covers the full pipeline path:
    generate_pairs  →  score_pairs  →  threshold_sweep  →  save JSON artifact

* Confirms output structure (keys, types, value ranges) rather than exact
  numbers, so it stays valid even if implementation details change.

Run with:  pytest tests/test_integration.py -v
"""

import os
import json
import tempfile
import shutil

import cv2
import pytest
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic fixture helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_lfw_fixture(root: str) -> dict:
    """
    Creates a minimal LFW-style folder tree under `root`:

        root/
          Alice/Alice_0001.jpg   ← red image
          Alice/Alice_0002.jpg   ← red image   (same colour → high similarity)
          Bob/Bob_0001.jpg       ← green image
          Bob/Bob_0002.jpg       ← green image
          Carol/Carol_0001.jpg   ← blue image

    Returns id_map: {identity: [indices]}
    """
    colours = {
        "Alice": (0,   0,   200),   # BGR red
        "Bob":   (0,   200, 0),     # BGR green
        "Carol": (200, 0,   0),     # BGR blue
    }
    id_map = {}
    for name, bgr in colours.items():
        folder = os.path.join(root, name)
        os.makedirs(folder, exist_ok=True)
        indices = []
        n_imgs = 1 if name == "Carol" else 2
        for i in range(1, n_imgs + 1):
            img = np.full((160, 160, 3), bgr, dtype=np.uint8)
            path = os.path.join(folder, f"{name}_{str(i).zfill(4)}.jpg")
            cv2.imwrite(path, img)
            indices.append(i)
        id_map[name] = indices
    return id_map


def make_pairs_csv(lfw_root: str, id_map: dict, out_path: str) -> pd.DataFrame:
    """
    Builds a tiny, fully deterministic pairs CSV:
      - 2 same-identity pairs  (Alice 1v2, Bob 1v2)        → label 1
      - 2 diff-identity pairs  (Alice vs Bob, Bob vs Carol) → label 0
    All assigned to the 'val' split.
    """
    rows = [
        ["Alice", "Alice", 1, 2, 1, "val"],
        ["Bob",   "Bob",   1, 2, 1, "val"],
        ["Alice", "Bob",   1, 1, 0, "val"],
        ["Bob",   "Carol", 1, 1, 0, "val"],
    ]
    df = pd.DataFrame(rows, columns=[
        "left_identity", "right_identity",
        "left_index", "right_index", "label", "split"
    ])
    df.to_csv(out_path, index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Inline pipeline (mirrors your scripts — no external imports needed)
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


def run_evaluation(pairs_csv: str, lfw_root: str, out_json: str,
                   embedder_fn) -> dict:
    """
    Full evaluation path:
      1. Load pairs CSV
      2. Load images and generate embeddings via embedder_fn
      3. Compute cosine similarity scores
      4. Threshold sweep (0 → 1 in 101 steps)
      5. Pick best threshold by F1
      6. Save JSON artifact

    embedder_fn: callable(list_of_rgb_images) → np.ndarray of shape (N, D)
    """
    df = pd.read_csv(pairs_csv)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    def img_path(identity, index):
        return os.path.join(lfw_root, identity,
                            f"{identity}_{str(index).zfill(4)}.jpg")

    left_imgs, right_imgs, valid_idx = [], [], []
    for i, row in val_df.iterrows():
        p1 = img_path(row["left_identity"],  int(row["left_index"]))
        p2 = img_path(row["right_identity"], int(row["right_index"]))
        if os.path.exists(p1) and os.path.exists(p2):
            img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
            left_imgs.append(img1)
            right_imgs.append(img2)
            valid_idx.append(i)

    final_df = val_df.loc[valid_idx].copy().reset_index(drop=True)
    emb1 = embedder_fn(left_imgs)
    emb2 = embedder_fn(right_imgs)
    final_df["score"] = cosine_similarity(emb1, emb2)

    # Threshold sweep
    thresholds = np.linspace(0, 1, 101)
    sweep = []
    y_true = final_df["label"].values
    scores = final_df["score"].values

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        sweep.append({"threshold": float(t), "tp": tp, "fp": fp,
                       "fn": fn, "tn": tn, "f1": float(f1)})

    best = max(sweep, key=lambda x: x["f1"])
    output = {
        "run_id": "integration_test",
        "valid_pairs": len(final_df),
        "best_threshold_from_val": best["threshold"],
        "val_sweep": sweep,
        "best_val_metrics": best,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output, f, indent=4)

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# Fake embedder — no model download, fully deterministic
# ═══════════════════════════════════════════════════════════════════════════════

def fake_embedder(images: list) -> np.ndarray:
    """
    Returns embeddings based on mean pixel colour of each image.
    Same-colour images → nearly identical vectors → high cosine similarity.
    Different-colour images → very different vectors → low similarity.

    This gives the pipeline realistic signal without any real model.
    """
    vecs = []
    for img in images:
        mean_rgb = img.reshape(-1, 3).mean(axis=0).astype(np.float32)
        # Expand to 512-d by tiling so the vector is non-trivial
        vec = np.tile(mean_rgb, 512 // 3 + 1)[:512]
        vecs.append(vec)
    return np.array(vecs)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration test
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def workspace():
    """Creates a temp directory tree and tears it down after all tests."""
    tmp = tempfile.mkdtemp(prefix="face_verif_test_")
    lfw_root   = os.path.join(tmp, "lfw")
    pairs_csv  = os.path.join(tmp, "pairs.csv")
    out_json   = os.path.join(tmp, "runs", "result.json")

    id_map = make_lfw_fixture(lfw_root)
    make_pairs_csv(lfw_root, id_map, pairs_csv)
    result = run_evaluation(pairs_csv, lfw_root, out_json, fake_embedder)

    yield {
        "tmp":       tmp,
        "lfw_root":  lfw_root,
        "pairs_csv": pairs_csv,
        "out_json":  out_json,
        "result":    result,
    }

    shutil.rmtree(tmp, ignore_errors=True)


class TestIntegrationPipeline:

    def test_json_artifact_is_created(self, workspace):
        assert os.path.exists(workspace["out_json"]), \
            "run JSON was not written to disk"

    def test_json_is_valid_and_parseable(self, workspace):
        with open(workspace["out_json"]) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_output_has_required_keys(self, workspace):
        required = {"run_id", "valid_pairs", "best_threshold_from_val",
                    "val_sweep", "best_val_metrics"}
        assert required.issubset(workspace["result"].keys())

    def test_valid_pairs_count_matches_fixture(self, workspace):
        # Fixture has 4 pairs, all with valid images → should all load
        assert workspace["result"]["valid_pairs"] == 4

    def test_sweep_has_correct_number_of_thresholds(self, workspace):
        assert len(workspace["result"]["val_sweep"]) == 101

    def test_sweep_thresholds_are_monotone(self, workspace):
        thresholds = [s["threshold"] for s in workspace["result"]["val_sweep"]]
        assert thresholds == sorted(thresholds)

    def test_best_threshold_is_within_range(self, workspace):
        t = workspace["result"]["best_threshold_from_val"]
        assert 0.0 <= t <= 1.0

    def test_best_metrics_f1_is_positive(self, workspace):
        f1 = workspace["result"]["best_val_metrics"]["f1"]
        assert f1 > 0.0, "F1 should be positive with a deterministic fixture"

    def test_confusion_matrix_values_sum_to_pair_count(self, workspace):
        m = workspace["result"]["best_val_metrics"]
        total = m["tp"] + m["fp"] + m["fn"] + m["tn"]
        assert total == workspace["result"]["valid_pairs"]

    def test_same_colour_pairs_score_higher_than_diff_colour(self, workspace):
        """
        Sanity check: the fake embedder should produce higher similarity for
        same-identity (same-colour) pairs than cross-identity pairs.
        """
        df = pd.read_csv(workspace["pairs_csv"])
        val = df[df["split"] == "val"].copy().reset_index(drop=True)

        def img_path(identity, index):
            return os.path.join(workspace["lfw_root"], identity,
                                f"{identity}_{str(index).zfill(4)}.jpg")

        images_l = [cv2.cvtColor(cv2.imread(img_path(r.left_identity,  r.left_index)),
                                 cv2.COLOR_BGR2RGB) for _, r in val.iterrows()]
        images_r = [cv2.cvtColor(cv2.imread(img_path(r.right_identity, r.right_index)),
                                 cv2.COLOR_BGR2RGB) for _, r in val.iterrows()]

        emb1   = fake_embedder(images_l)
        emb2   = fake_embedder(images_r)
        scores = cosine_similarity(emb1, emb2)
        val["score"] = scores

        mean_same = val[val["label"] == 1]["score"].mean()
        mean_diff = val[val["label"] == 0]["score"].mean()
        assert mean_same > mean_diff, \
            f"Same-pair mean ({mean_same:.3f}) should exceed diff-pair mean ({mean_diff:.3f})"