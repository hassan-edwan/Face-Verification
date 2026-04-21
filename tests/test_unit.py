"""
Unit tests for src/ primitives used by the live pipeline.

No dataset, no model, no network — synthetic fixtures only.
Live-pipeline state-machine tests live in tests/test_live_eval.py.
"""

import cv2
import numpy as np
import pytest

from src.similarity import cosine_similarity, euclidean_distance


# ── Cosine similarity ────────────────────────────────────────────────────────

class TestCosineSimilarity:
    """Cosine is the scoring primitive inside Gatekeeper._find_best_match.
    Broken cosine -> broken recognition, so these are regression tripwires."""

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


# ── Euclidean distance ───────────────────────────────────────────────────────

class TestEuclideanDistance:
    def test_identical_vectors_distance_zero(self):
        v = np.array([[1.0, 2.0, 3.0]])
        assert np.isclose(euclidean_distance(v, v)[0], 0.0, atol=1e-5)

    def test_unit_axis_separation(self):
        a, b = np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])
        assert np.isclose(euclidean_distance(a, b)[0], 5.0, atol=1e-5)


# ── Image loading (used by load_persistent_faces in the live server/app) ─────

class TestSafeImageLoading:
    """The live pipeline scans `data/enrollments/<Person N>/*.jpg` on start
    via cv2.imread. A missing / unreadable file must yield None rather
    than raise, so the enumeration loop can skip it gracefully."""

    def test_missing_file_returns_none(self):
        assert cv2.imread("this_file_definitely_does_not_exist_xyz.jpg") is None

    def test_pipeline_skips_none_without_crashing(self):
        paths = ["missing_a.jpg", "missing_b.jpg"]
        valid = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                valid.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        assert valid == []

    def test_valid_image_loads_correctly(self, tmp_path):
        img_path = str(tmp_path / "face.jpg")
        cv2.imwrite(img_path, np.zeros((160, 160, 3), dtype=np.uint8))
        result = cv2.imread(img_path)
        assert result is not None
        assert result.shape == (160, 160, 3)
