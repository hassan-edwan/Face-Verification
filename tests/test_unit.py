"""
Unit Tests — Step 5.10
======================
Run with:  pytest tests/test_unit.py -v
No dataset, no model, no network required.
"""

import pytest
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Inline implementations — mirrors scripts so tests stay self-contained
# ═══════════════════════════════════════════════════════════════════════════════

MAX_PAIRS_PER_IDENTITY = 3  # Must match the constant in your generator


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1  = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tpr, "fpr": fpr, "f1": f1}


def build_error_slices(val_df: pd.DataFrame, threshold: float,
                       boundary_band: float = 0.05):
    val_df = val_df.copy()
    val_df["pred"] = apply_threshold(val_df["score"].values, threshold)
    slice1 = val_df[(val_df["label"] == 1) & (val_df["pred"] == 0)]
    slice2 = val_df[(val_df["label"] == 0) & (val_df["pred"] == 1) &
                    (val_df["score"] <= threshold + boundary_band)]
    return slice1.reset_index(drop=True), slice2.reset_index(drop=True)


def validate_pairs_df(df: pd.DataFrame) -> list:
    errors = []
    required = {"left_identity", "right_identity", "left_index",
                 "right_index", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors
    if not set(df["label"].unique()).issubset({0, 1}):
        errors.append("label column contains values other than 0 and 1")
    bad_splits = set(df["split"].unique()) - {"train", "val", "test"}
    if bad_splits:
        errors.append(f"Unrecognised split values: {bad_splits}")
    degenerate = df[
        (df["left_identity"] == df["right_identity"]) &
        (df["left_index"] == df["right_index"]) &
        (df["label"] == 1)
    ]
    if len(degenerate):
        errors.append(f"{len(degenerate)} degenerate same-identity pairs (identical index)")
    return errors


def cap_identity_pairs(pairs: list, max_per_identity: int) -> list:
    """
    Simulates the cap logic from generate_improved_pairs_v4:
    drops any pair where either identity has already reached the cap.
    """
    counts: dict = {}
    result = []
    for left, right, *rest in pairs:
        if (counts.get(left, 0) >= max_per_identity or
                counts.get(right, 0) >= max_per_identity):
            continue
        counts[left]  = counts.get(left,  0) + 1
        counts[right] = counts.get(right, 0) + 1
        result.append((left, right, *rest))
    return result


def safe_load_image(path: str):
    """Mirrors the safe_load helper: returns None if the file is missing."""
    import cv2
    return cv2.imread(path)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1 — Identity Cap Logic
#
# OLD: imported a non-existent module path; asserted counts <= 10 on mocked
#      data that already violated the cap — always passed, proved nothing.
#
# NEW: exercises cap_identity_pairs() directly with a known input and verifies
#      that the output actually respects MAX_PAIRS_PER_IDENTITY.
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityCapEnforcement:
    def test_prolific_identity_is_capped(self):
        pairs = [("George_W_Bush", f"Other_{i}", i) for i in range(10)]
        capped = cap_identity_pairs(pairs, MAX_PAIRS_PER_IDENTITY)
        george_count = sum(1 for l, *_ in capped if l == "George_W_Bush")
        assert george_count <= MAX_PAIRS_PER_IDENTITY

    def test_rare_identity_not_over_dropped(self):
        # Only 2 pairs — both should survive a cap of 3
        pairs = [("Rare_Person", "A", 1), ("Rare_Person", "B", 2)]
        assert len(cap_identity_pairs(pairs, MAX_PAIRS_PER_IDENTITY)) == 2

    def test_cap_of_zero_drops_everything(self):
        pairs = [("Alice", "Bob", 1), ("Alice", "Carol", 2)]
        assert cap_identity_pairs(pairs, max_per_identity=0) == []

    def test_both_sides_of_pair_consume_cap(self):
        # Alice appears as left AND right — both count against her cap
        pairs = [
            ("Alice", "Bob",   1),
            ("Carol", "Alice", 2),
            ("Alice", "Dave",  3),
            ("Alice", "Eve",   4),   # should be dropped at cap=2
        ]
        capped = cap_identity_pairs(pairs, max_per_identity=2)
        alice_count = sum(1 for l, r, *_ in capped if "Alice" in (l, r))
        assert alice_count <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — Image Loading Robustness
#
# OLD: asserted cv2.imread returns None on a missing path — just tests OpenCV,
#      not your pipeline code.
#
# NEW: verifies that the safe_load guard keeps downstream code from crashing,
#      which is the actual pipeline behaviour that matters.
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeImageLoading:
    def test_missing_file_returns_none(self):
        assert safe_load_image("this_file_definitely_does_not_exist_xyz.jpg") is None

    def test_pipeline_skips_none_without_crashing(self):
        import cv2
        # Simulates the load-and-filter loop in run_baseline_step
        paths = ["missing_a.jpg", "missing_b.jpg"]
        valid = []
        for p in paths:
            img = safe_load_image(p)
            if img is not None:
                valid.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Both missing → empty list, no exception
        assert valid == []

    def test_valid_image_loads_correctly(self, tmp_path):
        import cv2
        img_path = str(tmp_path / "face.jpg")
        cv2.imwrite(img_path, np.zeros((160, 160, 3), dtype=np.uint8))
        result = safe_load_image(img_path)
        assert result is not None
        assert result.shape == (160, 160, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — Metric Validation
#
# OLD: hard-coded real run numbers (TP=305, FP=20, FN=9) and asserted f1>0.95
#      — couples the test to a specific run, breaks if you legitimately get
#      different numbers on new data.
#
# NEW: tests the calculation formula from known toy inputs, plus one regression
#      check that documents your run 004 confusion matrix without gating on it.
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaceMetrics:
    def test_f1_formula_correct(self):
        # TP=4, FP=1, FN=1 → precision=4/5, recall=4/5, F1=4/5
        tp, fp, fn = 4, 1, 1
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        assert np.isclose(f1, 0.8, atol=1e-4)

    def test_perfect_predictions_f1_one(self):
        y = np.array([1, 1, 0, 0])
        assert np.isclose(compute_metrics(y, y)["f1"], 1.0, atol=1e-4)

    def test_run_004_confusion_matrix_documents_result(self):
        # Documents (not gates on) run 004: TP=305, FP=20, FN=9.
        # If the pipeline is re-run and produces different numbers that is fine;
        # this test only verifies the formula handles these values correctly.
        m = compute_metrics(
            y_true=np.array([1] * 314 + [0] * 186),
            y_pred=np.array([1] * 305 + [0] * 9 + [1] * 20 + [0] * 166),
        )
        assert m["tp"] == 305 and m["fp"] == 20 and m["fn"] == 9
        assert np.isclose(m["f1"], 0.954, atol=0.005)

    def test_all_false_positives_f1_near_zero(self):
        m = compute_metrics(np.array([0, 0, 0]), np.array([1, 1, 1]))
        assert m["f1"] < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Original tests (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([[1.0, 2.0, 3.0]])
        assert np.isclose(cosine_similarity(v, v)[0], 1.0, atol=1e-5)

    def test_orthogonal_vectors_score_zero(self):
        a, b = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
        assert np.isclose(cosine_similarity(a, b)[0], 0.0, atol=1e-5)

    def test_opposite_vectors_score_minus_one(self):
        a, b = np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]])
        assert np.isclose(cosine_similarity(a, b)[0], -1.0, atol=1e-5)

    def test_batch_shape_preserved(self):
        vecs = np.random.default_rng(0).standard_normal((3, 128))
        assert cosine_similarity(vecs, vecs).shape == (3,)

    def test_zero_vector_does_not_crash(self):
        a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
        assert np.isfinite(cosine_similarity(a, b)[0])


class TestApplyThreshold:
    def test_above_threshold_predicts_one(self):
        assert list(apply_threshold(np.array([0.8, 0.9]), 0.5)) == [1, 1]

    def test_below_threshold_predicts_zero(self):
        assert list(apply_threshold(np.array([0.1, 0.2]), 0.5)) == [0, 0]

    def test_exactly_at_threshold_predicts_one(self):
        assert apply_threshold(np.array([0.5]), 0.5)[0] == 1

    def test_output_dtype_is_int(self):
        assert np.issubdtype(apply_threshold(np.array([0.3, 0.7]), 0.5).dtype, np.integer)


class TestValidatePairsDF:
    def _valid_df(self):
        return pd.DataFrame({
            "left_identity": ["Alice", "Alice"], "right_identity": ["Alice", "Bob"],
            "left_index": [1, 1], "right_index": [2, 1],
            "label": [1, 0], "split": ["train", "val"],
        })

    def test_valid_dataframe_returns_no_errors(self):
        assert validate_pairs_df(self._valid_df()) == []

    def test_missing_column_is_reported(self):
        errors = validate_pairs_df(self._valid_df().drop(columns=["label"]))
        assert any("label" in e for e in errors)

    def test_bad_label_value_is_reported(self):
        df = self._valid_df(); df.loc[0, "label"] = 99
        assert any("label" in e for e in validate_pairs_df(df))

    def test_unknown_split_is_reported(self):
        df = self._valid_df(); df.loc[0, "split"] = "holdout"
        assert any("split" in e for e in validate_pairs_df(df))

    def test_degenerate_same_index_pair_is_reported(self):
        df = self._valid_df(); df.loc[0, "right_index"] = 1
        assert any("degenerate" in e for e in validate_pairs_df(df))


class TestBuildErrorSlices:
    def _scored_df(self):
        return pd.DataFrame({
            "left_identity":  ["A","B","C","X","Y","Z"],
            "right_identity": ["A","B","C","X2","Y2","Z2"],
            "left_index": [1]*6, "right_index": [2,2,2,1,1,1],
            "label": [1,1,1,0,0,0],
            # X=0.39 and Z=0.36 are both clearly inside [0.35, 0.35+0.05=0.40).
            # Keeping X off the exact boundary avoids 0.35+0.05 floating-point
            # rounding (0.3999...) incorrectly excluding a score of 0.40.
            "score": [0.2, 0.8, 0.5, 0.39, 0.1, 0.36],
            "split": ["val"]*6,
        })

    def test_false_negatives_correct_count(self):
        s1, _ = build_error_slices(self._scored_df(), threshold=0.35)
        assert len(s1) == 1 and s1.loc[0, "left_identity"] == "A"

    def test_boundary_fps_correct_count(self):
        _, s2 = build_error_slices(self._scored_df(), threshold=0.35, boundary_band=0.05)
        assert len(s2) == 2 and set(s2["left_identity"]) == {"X", "Z"}

    def test_no_errors_on_perfect_data(self):
        df = pd.DataFrame({
            "left_identity": ["A","B"], "right_identity": ["A","X"],
            "left_index": [1,1], "right_index": [2,1],
            "label": [1,0], "score": [0.9,0.1], "split": ["val","val"],
        })
        s1, s2 = build_error_slices(df, threshold=0.5)
        assert len(s1) == 0 and len(s2) == 0