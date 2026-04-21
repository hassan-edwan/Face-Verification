"""
Face Image Quality Assessment (FIQA)
=====================================
Provides quality gates that must be passed before any frame is sent to the
embedding model. All thresholds are tunable constants at the top of the file.

Design principle: reject bad frames early (cheapest checks first) to avoid
wasting GPU/CPU on blurry or poorly-posed faces that will produce noisy
embeddings.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

# ── TUNABLE THRESHOLDS ───────────────────────────────────────────────────────
# Full FIQA (used by assess_quality — runs before refinement frames only)
SHARPNESS_MIN      = 50.0   # Laplacian variance; lowered for real webcam conditions
ILLUMINATION_MIN   = 30.0
ILLUMINATION_MAX   = 230.0
POSE_YAW_MAX_DEG   = 25.0
POSE_PITCH_MAX_DEG = 20.0
POSE_ROLL_MAX_DEG  = 20.0

# Quick FIQA (used by quick_quality_check — fast path, no pose, <5ms)
QUICK_SHARPNESS_MIN    = 20.0   # Very lenient — only reject extreme motion blur
QUICK_ILLUMINATION_MIN = 15.0
QUICK_ILLUMINATION_MAX = 245.0
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QualityResult:
    """Result of a quality assessment. Always check `passed` first."""
    passed:       bool
    score:        float        # Composite quality score in [0.0, 1.0]
    reason:       str          # "OK" on success, failure message otherwise
    sharpness:    float = 0.0
    illumination: float = 0.0
    yaw_deg:      float = 0.0
    pitch_deg:    float = 0.0
    roll_deg:     float = 0.0


def check_sharpness(face_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Computes the variance of the Laplacian (a measure of edge energy).
    A high variance means sharp edges → good image.
    A low variance means few edges → blurry / motion-blurred frame.

    Returns (passed, laplacian_variance).
    """
    gray     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return variance >= SHARPNESS_MIN, variance


def check_illumination(face_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Checks mean pixel brightness via the L channel of the LAB color space.
    LAB's L channel is perceptually uniform, which makes it more reliable
    than raw RGB mean for illumination assessment.

    Returns (passed, mean_L).
    """
    lab    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    mean_L = float(np.mean(lab[:, :, 0]))
    passed = ILLUMINATION_MIN <= mean_L <= ILLUMINATION_MAX
    return passed, mean_L


def check_pose_from_landmarks(landmarks: Dict) -> Tuple[bool, float, float, float]:
    """
    Estimates yaw, pitch, and roll from MTCNN's 5-point facial landmarks:
        left_eye, right_eye, nose, mouth_left, mouth_right.

    These are PROXY estimates — not true 3D head-pose (which requires a
    calibrated camera + PnP solve). They are sufficient for detecting
    profile views and extreme tilts.

    Returns (passed, yaw_deg, pitch_deg, roll_deg).
    """
    le = np.array(landmarks['left_eye'],   dtype=float)
    re = np.array(landmarks['right_eye'],  dtype=float)
    n  = np.array(landmarks['nose'],       dtype=float)
    ml = np.array(landmarks['mouth_left'], dtype=float)
    mr = np.array(landmarks['mouth_right'],dtype=float)

    # ── ROLL: angle of the eye axis from horizontal ──────────────────────────
    # arctan2 of (right_eye - left_eye) vector gives the rotation.
    dy       = re[1] - le[1]
    dx       = re[0] - le[0]
    roll_deg = abs(float(np.degrees(np.arctan2(dy, dx))))
    if roll_deg > 90:
        roll_deg = 180.0 - roll_deg   # Normalize to [0°, 90°]

    # ── YAW: horizontal asymmetry of eye→nose distances ─────────────────────
    # For a frontal face, the nose is horizontally equidistant from both eyes.
    # As the head yaws, one distance shrinks and the other grows.
    left_dist  = abs(n[0] - le[0])
    right_dist = abs(re[0] - n[0])
    eye_width  = abs(re[0] - le[0]) + 1e-9

    # asymmetry in [-1, 1]; positive = turned right; 0 = frontal
    asymmetry = (right_dist - left_dist) / eye_width
    yaw_deg   = float(np.degrees(np.arcsin(np.clip(asymmetry, -1.0, 1.0))))

    # ── PITCH: vertical ratio of eye-to-nose vs nose-to-mouth ───────────────
    # For a frontal face, the eye-center → nose span and nose → mouth-center
    # span have a roughly 1:1 ratio. Tilting changes this ratio.
    eye_center_y   = (le[1] + re[1]) / 2.0
    mouth_center_y = (ml[1] + mr[1]) / 2.0
    eye_to_nose    = n[1] - eye_center_y
    nose_to_mouth  = mouth_center_y - n[1]
    total_span     = eye_to_nose + nose_to_mouth + 1e-9

    # deviation from 0.5 (perfect frontal)
    ratio_dev = (eye_to_nose / total_span) - 0.5
    pitch_deg = float(np.degrees(np.arcsin(np.clip(ratio_dev * 2.0, -1.0, 1.0))))

    passed = (
        abs(yaw_deg)   <= POSE_YAW_MAX_DEG   and
        abs(pitch_deg) <= POSE_PITCH_MAX_DEG and
        roll_deg       <= POSE_ROLL_MAX_DEG
    )
    return passed, yaw_deg, pitch_deg, roll_deg


def assess_quality(
    face_bgr:  np.ndarray,
    landmarks: Optional[Dict] = None,
) -> QualityResult:
    """
    Master quality gate. Runs all checks in cheapest-first order and
    short-circuits on the first failure.

    Args:
        face_bgr:   Face crop in BGR format (any size). Should already be
                    cropped to the face bounding box before calling.
        landmarks:  Optional MTCNN keypoints dict. If provided, pose
                    estimation is included. If None, pose check is skipped.

    Returns:
        QualityResult — always check `.passed` before using `.score`.
    """
    # ── 1. Sharpness (cheapest) ──────────────────────────────────────────────
    sharp_ok, sharpness = check_sharpness(face_bgr)
    if not sharp_ok:
        return QualityResult(
            passed=False,
            score=sharpness / SHARPNESS_MIN,   # partial credit for debug
            reason=f"Blurry (Laplacian={sharpness:.1f} < {SHARPNESS_MIN})",
            sharpness=sharpness,
        )

    # ── 2. Illumination ──────────────────────────────────────────────────────
    illum_ok, illumination = check_illumination(face_bgr)
    if not illum_ok:
        reason = (
            f"Overexposed (L={illumination:.1f} > {ILLUMINATION_MAX})"
            if illumination > ILLUMINATION_MAX
            else f"Too dark (L={illumination:.1f} < {ILLUMINATION_MIN})"
        )
        return QualityResult(
            passed=False, score=0.0, reason=reason,
            sharpness=sharpness, illumination=illumination,
        )

    # ── 3. Pose (requires landmarks) ─────────────────────────────────────────
    yaw_deg = pitch_deg = roll_deg = 0.0
    if landmarks is not None:
        pose_ok, yaw_deg, pitch_deg, roll_deg = check_pose_from_landmarks(landmarks)
        if not pose_ok:
            return QualityResult(
                passed=False, score=0.0,
                reason=(
                    f"Pose out of range (yaw={yaw_deg:.1f}°, "
                    f"pitch={pitch_deg:.1f}°, roll={roll_deg:.1f}°)"
                ),
                sharpness=sharpness, illumination=illumination,
                yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg,
            )

    # ── All checks passed — compute composite score in [0, 1] ────────────────
    # Weighted blend: sharpness matters most for embedding quality.
    sharpness_score = min(sharpness / 300.0, 1.0)
    illum_score     = 1.0 - abs(illumination - 130.0) / 130.0   # peaks at 130

    if landmarks is not None:
        pose_score = 1.0 - max(
            abs(yaw_deg)   / max(POSE_YAW_MAX_DEG,   1e-9),
            abs(pitch_deg) / max(POSE_PITCH_MAX_DEG, 1e-9),
            roll_deg       / max(POSE_ROLL_MAX_DEG,  1e-9),
        )
        pose_score = float(np.clip(pose_score, 0.0, 1.0))
    else:
        pose_score = 1.0

    composite = (sharpness_score * 0.40 + illum_score * 0.30 + pose_score * 0.30)

    return QualityResult(
        passed=True,
        score=float(np.clip(composite, 0.0, 1.0)),
        reason="OK",
        sharpness=sharpness,
        illumination=illumination,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )


def quick_quality_check(face_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Lightweight quality check used ONLY for refinement frames (not fast enrollment).
    Runs in <5ms: grayscale Laplacian variance + mean brightness only.
    No pose estimation, no LAB conversion.

    Returns (passed, score_0_to_1).
    """
    gray       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    if sharpness < QUICK_SHARPNESS_MIN:
        return False, 0.0
    if not (QUICK_ILLUMINATION_MIN <= brightness <= QUICK_ILLUMINATION_MAX):
        return False, 0.0

    score = (
        min(sharpness / 200.0, 1.0) * 0.6
        + (1.0 - abs(brightness - 128.0) / 128.0) * 0.4
    )
    return True, float(np.clip(score, 0.0, 1.0))
