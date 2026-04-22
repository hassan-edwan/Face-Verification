"""
FaceVerifier — embedding-based face verification.

Wraps the run_005 pipeline (center-crop preprocessing + FaceNet embeddings +
cosine similarity) behind a small interface that the CLI and load-test share.

Decision threshold is loaded from outputs/runs/run_005.json (picked on the val
split). Confidence is produced by a Platt-scaling sigmoid fit on val scores
(see scripts/fit_calibration.py, which writes configs/calibration.json).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.similarity import cosine_similarity

CROP_MARGIN = 0.15
TARGET_SIZE = (160, 160)


@dataclass
class VerifyResult:
    decision: int              # 1 = same identity, 0 = different
    score: float               # cosine similarity in [-1, 1]
    confidence: float          # Platt-calibrated P(same | score) in [0, 1]
    threshold: float           # decision boundary used for this call
    latency_ms: float          # wall time for this verify() call

    def as_dict(self) -> dict:
        return {
            "decision": int(self.decision),
            "score": round(self.score, 6),
            "confidence": round(self.confidence, 6),
            "threshold": round(self.threshold, 6),
            "latency_ms": round(self.latency_ms, 3),
        }


def load_and_preprocess(path: str) -> np.ndarray:
    """Center-crop + resize — same preprocessing as run_005."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    top, bottom = int(h * CROP_MARGIN), int(h * (1 - CROP_MARGIN))
    left, right = int(w * CROP_MARGIN), int(w * (1 - CROP_MARGIN))
    return cv2.resize(
        img[top:bottom, left:right], TARGET_SIZE, interpolation=cv2.INTER_LINEAR
    )


class FaceVerifier:
    def __init__(
        self,
        threshold: Optional[float] = None,
        calibration_path: str = "configs/calibration.json",
        run_json_path: str = "outputs/runs/run_005.json",
    ):
        from keras_facenet import FaceNet  # lazy — heavy import

        self._embedder = FaceNet()

        calib = self._load_calibration(calibration_path)
        self.platt_a: float = calib["platt_a"]
        self.platt_b: float = calib["platt_b"]

        if threshold is not None:
            self.threshold = float(threshold)
        elif "threshold" in calib:
            self.threshold = float(calib["threshold"])
        else:
            self.threshold = self._load_threshold_from_run(run_json_path)

    @staticmethod
    def _load_calibration(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Calibration file not found at {path}. "
                f"Run: python -m scripts.fit_calibration"
            )
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _load_threshold_from_run(path: str) -> float:
        with open(path) as f:
            return float(json.load(f)["best_threshold_from_val"])

    def embed(self, img: np.ndarray) -> np.ndarray:
        return self._embedder.embeddings([img])[0]

    def embed_batch(self, imgs: list) -> np.ndarray:
        return self._embedder.embeddings(imgs)

    def score(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        a = emb_a.reshape(1, -1)
        b = emb_b.reshape(1, -1)
        return float(cosine_similarity(a, b)[0])

    def calibrate(self, score: float) -> float:
        """Platt sigmoid: P(same | score) = 1 / (1 + exp(A*score + B)).

        Computed in the numerically stable form that avoids math.exp overflow
        when A*score+B has large magnitude.
        """
        z = self.platt_a * score + self.platt_b
        if z >= 0:
            ez = math.exp(-z)
            return float(ez / (1.0 + ez))
        return float(1.0 / (1.0 + math.exp(z)))

    def verify(self, path_a: str, path_b: str) -> VerifyResult:
        t0 = time.perf_counter()
        img_a = load_and_preprocess(path_a)
        img_b = load_and_preprocess(path_b)
        embs = self.embed_batch([img_a, img_b])
        s = self.score(embs[0], embs[1])
        conf = self.calibrate(s)
        decision = int(s >= self.threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return VerifyResult(
            decision=decision, score=s, confidence=conf,
            threshold=self.threshold, latency_ms=latency_ms,
        )
