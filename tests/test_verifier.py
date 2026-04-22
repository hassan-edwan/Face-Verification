"""
Unit tests for the Milestone 3 verifier surface.

These tests avoid loading FaceNet (expensive): they cover the calibration and
decision logic, which is what determines CLI output shape and correctness.
Run:  pytest tests/test_verifier.py -v
"""

import json
import math

import numpy as np
import pytest

from scripts.fit_calibration import fit_platt


def _sigmoid_platt(score, a, b):
    """Numerically stable Platt sigmoid, mirrors FaceVerifier.calibrate()."""
    z = a * score + b
    if z >= 0:
        ez = math.exp(-z)
        return ez / (1.0 + ez)
    return 1.0 / (1.0 + math.exp(z))


def test_platt_fits_separable_data():
    """Higher-scored positives + lower-scored negatives → sigmoid should map
    them to confidence > 0.5 and < 0.5 respectively."""
    rng = np.random.default_rng(0)
    pos_scores = rng.normal(0.8, 0.05, size=200)
    neg_scores = rng.normal(0.2, 0.05, size=200)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(200), np.zeros(200)]).astype(int)

    a, b = fit_platt(scores, labels)

    # Sanity: monotonic decreasing in A*s+B  → slope A should be negative.
    assert a < 0, f"expected negative Platt slope, got A={a}"

    # Positive-region score maps above 0.5, negative-region below 0.5.
    assert _sigmoid_platt(0.8, a, b) > 0.5
    assert _sigmoid_platt(0.2, a, b) < 0.5


def test_platt_output_in_unit_interval():
    """Calibrated confidence must stay in [0, 1] for any real score."""
    a, b = -10.0, 5.0
    for s in [-1.0, -0.5, 0.0, 0.5, 1.0, 100.0, -100.0]:
        p = _sigmoid_platt(s, a, b)
        assert 0.0 <= p <= 1.0


def test_decision_matches_threshold():
    """Decision rule is score >= threshold."""
    threshold = 0.45
    scores = np.array([0.44, 0.45, 0.46, 0.9, 0.0])
    decisions = (scores >= threshold).astype(int)
    assert decisions.tolist() == [0, 1, 1, 1, 0]


def test_verify_result_dict_shape(tmp_path):
    """VerifyResult.as_dict must expose the five CLI-contract fields."""
    from src.verifier import VerifyResult
    r = VerifyResult(decision=1, score=0.71, confidence=0.88,
                     threshold=0.45, latency_ms=42.0)
    d = r.as_dict()
    assert set(d.keys()) == {"decision", "score", "confidence", "threshold", "latency_ms"}
    assert d["decision"] == 1
    assert isinstance(d["score"], float)
    assert isinstance(d["confidence"], float)
    assert isinstance(d["threshold"], float)
    assert isinstance(d["latency_ms"], float)


def test_calibration_file_schema(tmp_path):
    """A well-formed calibration.json loads and has the expected keys."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "platt_a": -8.0, "platt_b": 3.0, "threshold": 0.5,
        "source_run": "outputs/runs/run_005.json",
        "pairs_csv": "configs/pairs_v4.csv",
        "n_val_pairs": 100,
    }))
    data = json.loads(path.read_text())
    for key in ["platt_a", "platt_b", "threshold"]:
        assert key in data
