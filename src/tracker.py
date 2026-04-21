"""
Lightweight IoU-Based Face Tracker
====================================
Assigns a stable `track_id` to each detected face and maintains it
across consecutive frames using bounding-box overlap (IoU matching).

No external ML dependency — pure geometry with NumPy.

Design rationale:
    The tracker decouples "face location per frame" from "who is this person".
    The Gatekeeper accumulates embeddings keyed by track_id, not by frame, so
    it can require N _consecutive quality frames of the same track_ before
    making any enrollment decision. Without this, the same physical person
    could trigger multiple independent enrollment attempts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ── TRACKER SETTINGS (tune these) ────────────────────────────────────────────
IOU_MATCH_THRESHOLD = 0.25   # Minimum IoU for a detection to be assigned
                              # to an existing track. Lower = more lenient.
MAX_FRAMES_LOST     = 30     # A track is pruned after this many consecutive
                              # frames with no matching detection.
                              # Was 10 (~0.33s); 30 (~1s) tolerates brief occlusion.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Track:
    """Represents one active face track."""
    track_id:    int
    bbox:        Tuple[int, int, int, int]   # (x, y, w, h) in frame pixels
    landmarks:   Optional[Dict]              # MTCNN keypoints or None
    confidence:  float                       # Detector confidence [0, 1]
    frames_lost: int = 0      # Consecutive frames with no matching detection
    frames_seen: int = 1      # Total frames this track was successfully matched
    age:         int = 1      # Total frames since track was spawned


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _xywh_to_xyxy(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Convert (x, y, w, h) → (x1, y1, x2, y2) corner format."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Computes Intersection-over-Union of two bounding boxes.
    Input boxes must be in (x, y, w, h) format.
    Returns a scalar in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(box_b)

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter + 1e-9

    return float(inter / union)


# ── Tracker class ─────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Maintains a list of active face tracks and updates them each frame.

    Usage (call once per captured frame):
        active_tracks = tracker.update(detections)

    Where `detections` is a list of (bbox, landmarks, confidence) tuples
    as returned (after filtering) by the MTCNN detector.
    """

    def __init__(self):
        self._tracks:  List[Track] = []
        self._next_id: int = 0

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], Optional[Dict], float]],
    ) -> List[Track]:
        """
        Matches incoming detections to existing tracks and returns the
        currently visible (frames_lost == 0) set of tracks.

        Algorithm:
            1. Age every existing track (increment frames_lost).
            2. Greedily match each detection to the highest-IoU unmatched track.
            3. Unmatched detections → spawn new tracks.
            4. Prune tracks that have been lost for > MAX_FRAMES_LOST frames.
            5. Return only tracks that were matched this frame.

        Args:
            detections: List of (bbox, landmarks, confidence).
                        bbox = (x, y, w, h) in frame pixel coordinates.

        Returns:
            List of active Track objects sorted by track_id.
        """
        # ── Step 1: Age all tracks ───────────────────────────────────────────
        for t in self._tracks:
            t.frames_lost += 1
            t.age         += 1

        # ── Step 2: Greedy IoU matching ──────────────────────────────────────
        matched_track_ids: set = set()

        for det_bbox, det_lm, det_conf in detections:
            best_iou   = IOU_MATCH_THRESHOLD   # minimum bar to count as a match
            best_track: Optional[Track] = None

            for track in self._tracks:
                if track.track_id in matched_track_ids:
                    continue   # Already claimed this track this frame
                iou = compute_iou(det_bbox, track.bbox)
                if iou > best_iou:
                    best_iou   = iou
                    best_track = track

            if best_track is not None:
                # Update existing track
                best_track.bbox        = det_bbox
                best_track.landmarks   = det_lm
                best_track.confidence  = det_conf
                best_track.frames_lost = 0          # reset lost counter
                best_track.frames_seen += 1
                matched_track_ids.add(best_track.track_id)
            else:
                # Unmatched detection → new track
                new_track = Track(
                    track_id   = self._next_id,
                    bbox       = det_bbox,
                    landmarks  = det_lm,
                    confidence = det_conf,
                )
                self._tracks.append(new_track)
                self._next_id += 1

        # ── Step 3: Prune dead tracks ────────────────────────────────────────
        self._tracks = [
            t for t in self._tracks
            if t.frames_lost <= MAX_FRAMES_LOST
        ]

        # ── Step 4: Return only currently visible tracks ─────────────────────
        return [t for t in self._tracks if t.frames_lost == 0]

    def get_track(self, track_id: int) -> Optional[Track]:
        """Look up a track by its ID (including lost-but-not-pruned tracks)."""
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None

    def remove_track(self, track_id: int) -> None:
        """Explicitly remove a track. Used to clean up after enrollment."""
        self._tracks = [t for t in self._tracks if t.track_id != track_id]

    @property
    def active_count(self) -> int:
        """Number of currently visible tracks."""
        return sum(1 for t in self._tracks if t.frames_lost == 0)
