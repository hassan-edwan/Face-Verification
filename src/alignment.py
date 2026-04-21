"""
Face alignment — shared by the live pipeline and the real-data harness.

`align_face` affine-aligns a face crop to a canonical 160×160 layout
using MTCNN eye landmarks. When landmarks are missing, falls back to a
plain bbox crop + resize.

This is a 2-point similarity transform (left_eye → right_eye only). It
corrects roll + scale but not yaw or pitch — a known limitation. See
`docs/prompts/improve_pose_and_distance.md` §3 for the 5-point
extension candidate.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


TARGET_SIZE        = (160, 160)
_LEFT_EYE_TARGET   = np.float32([44.0,  60.0])
_RIGHT_EYE_TARGET  = np.float32([116.0, 60.0])

# ArcFace / InsightFace canonical 5-point template @ 112×112.
# Constants pulled from InsightFace's published reference template —
# keep as-is; eye/nose/mouth positions are calibrated against the
# w600k_r50 training distribution.
ARCFACE_TARGET_SIZE = (112, 112)
_ARCFACE_5PT = np.float32([
    [38.2946, 51.6963],   # left_eye
    [73.5318, 51.5014],   # right_eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # mouth_left
    [70.7299, 92.2041],   # mouth_right
])


def align_face(frame_bgr: np.ndarray,
               bbox:      Tuple[int, int, int, int],
               landmarks: Optional[Dict]) -> np.ndarray:
    """Affine-align a face crop to 160×160 using eye landmarks.

    Args:
        frame_bgr: full-frame BGR image (any size).
        bbox:      (x, y, w, h) bounding box of the detected face.
        landmarks: MTCNN 5-point dict with at least 'left_eye' and
                   'right_eye' keys; each value is (px, py). Pass None
                   to skip alignment and use the plain crop fallback.

    Returns:
        160×160 BGR crop, aligned when possible.
    """
    if landmarks is not None:
        src_pts = np.float32([landmarks['left_eye'], landmarks['right_eye']])
        dst_pts = np.float32([_LEFT_EYE_TARGET, _RIGHT_EYE_TARGET])
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is not None:
            return cv2.warpAffine(frame_bgr, M, TARGET_SIZE)

    # Fallback: plain crop + resize. Used when MTCNN didn't return
    # landmarks (rare) or estimateAffinePartial2D failed.
    x, y, w, h = bbox
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, TARGET_SIZE)


def align_face_5point(frame_bgr: np.ndarray,
                      bbox:      Tuple[int, int, int, int],
                      landmarks: Optional[Dict]) -> np.ndarray:
    """ArcFace-compatible 5-point alignment to 112×112.

    Uses all 5 MTCNN keypoints (eyes + nose + mouth corners) fitted
    via similarity transform against InsightFace's canonical template.
    Strictly better than `align_face`'s 2-point alignment for off-angle
    faces because nose + mouth anchor the fit when eyes alone are
    ambiguous. Required geometry for the ArcFace embedder.

    Falls back to a bbox crop + resize when landmarks are missing or
    the transform fails — same behavior pattern as `align_face`."""
    if landmarks is not None:
        keys = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
        if all(k in landmarks for k in keys):
            src_pts = np.float32([landmarks[k] for k in keys])
            M, _ = cv2.estimateAffinePartial2D(src_pts, _ARCFACE_5PT)
            if M is not None:
                return cv2.warpAffine(frame_bgr, M, ARCFACE_TARGET_SIZE)

    # Fallback: plain crop + resize to 112×112.
    x, y, w, h = bbox
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*ARCFACE_TARGET_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, ARCFACE_TARGET_SIZE)
