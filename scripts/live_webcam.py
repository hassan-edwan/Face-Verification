"""
Live Webcam Face Recognition - Gatekeeper Architecture
=======================================================
Refactored pipeline with modular stages:

  Stage 1 - Detection   : MTCNN (replaces Haar Cascade; gives landmarks)
  Stage 2 - Tracking    : IoU tracker (stable track_id per physical face)
  Stage 3 - Alignment   : Affine transform via eye landmarks → 160×160 crop
  Stage 4 - Quality Gate: FIQA (sharpness, illumination, pose proxy)
  Stage 5 - Embedding   : FaceNet (keras-facenet, unchanged from original)
  Stage 6 - Gatekeeper  : Consensus buffer + dual-threshold de-duplication
  Stage 7 - Persistence : Save best-quality frame only on confirmed enrollment

Key behavioural changes from the original:
  • New face NOT enrolled immediately - requires CONSENSUS_FRAMES quality frames.
  • "Awkward angle" zone is now a HARD DROP (uncertain zone) instead of saving
    extra images. This prevents database bloat.
  • A track_id that has already enrolled is session-blacklisted to prevent
    re-enrollment when the same person leaves and re-enters the frame.
  • Haar Cascade replaced with MTCNN for far fewer false positives and richer
    per-detection metadata (confidence score + 5-point landmarks).
"""

import os
import sys
import cv2
import glob
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keras_facenet import FaceNet
from mtcnn import MTCNN

from src.quality    import assess_quality, quick_quality_check
from src.tracker    import FaceTracker
from src.gatekeeper import Gatekeeper, GatekeeperDecision, CONSENSUS_FRAMES, MAX_EMBEDDINGS_PER_IDENTITY

# ── PIPELINE CONFIG ───────────────────────────────────────────────────────────
# All tunable parameters are here. Thresholds for matching/enrollment are in
# src/gatekeeper.py. Quality thresholds are in src/quality.py.

TARGET_SIZE       = (160, 160)  # FaceNet input resolution (do not change)
MIN_FACE_PX       = 60          # Ignore MTCNN detections smaller than this (px)
MTCNN_CONF_MIN    = 0.90        # Minimum MTCNN detection confidence to accept
DETECT_DOWNSAMPLE = 2           # Run MTCNN on 1/N resolution frame for speed
                                # (coordinates are scaled back up automatically)
INFERENCE_SKIP    = 3           # Send to worker at most once every N frames per
                                # track. Limits worker load without losing coverage.
WORKER_QUEUE_SIZE = 4           # Max outstanding tasks in the worker queue

LFW_BASE          = os.path.join(PROJECT_ROOT, "data", "lfw")
DEBUG_OVERLAY       = True        # Show track IDs and quality scores on screen
STATE_STALE_TIMEOUT = 4.0         # Seconds before a QUALITY_FAIL state is auto-cleared

# ── Affine alignment targets (canonical 160×160 FaceNet eye positions) ────────
# These reference points were calibrated for FaceNet input. Changing them
# would require retraining or at least re-evaluating threshold values.
_LEFT_EYE_TARGET  = np.float32([44.0,  60.0])
_RIGHT_EYE_TARGET = np.float32([116.0, 60.0])
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 - FACE ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def align_face(
    frame_bgr: np.ndarray,
    bbox:      Tuple[int, int, int, int],
    landmarks: Optional[Dict],
) -> np.ndarray:
    """
    Crops and affine-aligns a face to TARGET_SIZE using MTCNN landmarks.

    The affine transform normalises for roll (in-plane rotation) and scale
    by mapping the actual eye positions to a canonical target layout. This
    produces far more consistent embeddings than a plain rectangular crop.

    Fallback: if landmarks are unavailable or the transform fails, falls back
    to a simple bbox crop + resize (same as the original pipeline).

    Args:
        frame_bgr: Full camera frame in BGR.
        bbox:      MTCNN bounding box (x, y, w, h).
        landmarks: MTCNN keypoints dict, or None.

    Returns:
        Aligned BGR face image of shape (160, 160, 3).
    """
    if landmarks is not None:
        src_pts = np.float32([
            landmarks['left_eye'],
            landmarks['right_eye'],
        ])
        dst_pts = np.float32([_LEFT_EYE_TARGET, _RIGHT_EYE_TARGET])

        # estimateAffinePartial2D: similarity transform (rotation + isotropic scale)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is not None:
            return cv2.warpAffine(frame_bgr, M, TARGET_SIZE)

    # Fallback: plain crop + resize
    x, y, w, h = bbox
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, x);  y1 = max(0, y)
    x2 = min(w_img, x + w);  y2 = min(h_img, y + h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, TARGET_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
#  STORAGE - unchanged from original for full backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def load_persistent_faces(embedder: FaceNet) -> Tuple[Dict, int]:
    """
    Scans data/lfw/ for 'Person N' folders and loads their stored face
    images as embeddings. Unchanged from the original implementation.

    Returns:
        (known_faces dict, next_person_counter int)
    """
    known_faces    = {}
    person_counter = 1

    if not os.path.exists(LFW_BASE):
        os.makedirs(LFW_BASE, exist_ok=True)
        return known_faces, person_counter

    print("[Worker] Scanning data/lfw/ for previously saved faces...")
    dirs = [
        d for d in os.listdir(LFW_BASE)
        if os.path.isdir(os.path.join(LFW_BASE, d))
    ]

    for d in dirs:
        if not d.startswith("Person "):
            continue
        # Track the highest person number seen
        try:
            num = int(d.split(" ")[1])
            if num >= person_counter:
                person_counter = num + 1
        except (IndexError, ValueError):
            pass

        img_paths = glob.glob(os.path.join(LFW_BASE, d, "*.jpg"))
        if not img_paths:
            continue

        known_faces[d] = []
        for path in img_paths:
            img = cv2.imread(path)
            if img is not None:
                img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_sized = cv2.resize(img_rgb, TARGET_SIZE)
                emb       = embedder.embeddings(np.expand_dims(img_sized, 0))
                known_faces[d].append(emb)

    print(f"[Worker] Loaded {len(known_faces)} known identity/ies from disk.")
    return known_faces, person_counter


def save_person_image(person_name: str, face_bgr: np.ndarray) -> None:
    """
    Saves the best-quality aligned face crop (BGR) to data/lfw/<person_name>/.
    Only called on ENROLLED events - one save per new identity.
    """
    person_dir = os.path.join(LFW_BASE, person_name)
    os.makedirs(person_dir, exist_ok=True)

    existing = glob.glob(os.path.join(person_dir, "*.jpg"))
    next_idx = len(existing) + 1
    stem     = person_name.replace(" ", "_")
    filepath = os.path.join(person_dir, f"{stem}_{str(next_idx).zfill(4)}.jpg")

    cv2.imwrite(filepath, face_bgr)
    print(f"[Persistent Storage] Saved → {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE - main thread ↔ worker thread
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisplayState:
    """
    Per-track display information, written by the worker and read by main.
    One entry per active track_id in the shared _display_states dict.
    """
    identity:       Optional[str]   # Name shown in label
    score:          float           # Cosine similarity (0–1)
    frame_count:    int             # Consensus frames buffered so far
    decision:       str             # GatekeeperDecision.name for colour routing
    quality:        float           # Composite quality score (for debug overlay)
    last_updated:   float = 0.0    # time.time() when worker last wrote this state
    quality_reason: str   = ""     # Specific failure reason from quality module
    is_reentry:     bool  = False  # True if identity was seen under a different track before


_display_states: Dict[int, DisplayState] = {}   # track_id → DisplayState
_state_lock     = threading.Lock()
_task_queue     = queue.Queue(maxsize=WORKER_QUEUE_SIZE)
_stop_flag      = threading.Event()
_worker_ready   = threading.Event()              # Set when worker finishes loading
_known_count    = 0                              # Number of known faces loaded from disk


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD - Stages 4, 5, 6, 7
# ═══════════════════════════════════════════════════════════════════════════════

def recognition_worker() -> None:
    """
    Background thread: FaceNet model + Gatekeeper.
    Stages: Quality Gate -> Embedding -> Gatekeeper -> Persist -> UI update.
    """
    print("[Worker] Loading FaceNet model...")
    embedder = FaceNet()

    known_faces, person_counter = load_persistent_faces(embedder)
    gatekeeper = Gatekeeper(known_faces, person_counter)
    global _known_count
    _known_count = len(known_faces)
    _worker_ready.set()
    print("[Worker] Ready. Listening for frames...")

    while not _stop_flag.is_set():
        try:
            task = _task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            aligned_bgr, track_id, landmarks = task

            # ── Stage 4: Quality Gate (two-path) ──────────────────────────────
            # First contact  → full quality check; show failure reason if bad.
            # Already enrolled → quick_quality_check only (<5ms, no pose).
            is_enrolled = gatekeeper.is_already_enrolled(track_id)

            if not is_enrolled:
                qr = assess_quality(aligned_bgr, landmarks)
                quality_score = qr.score
                if not qr.passed:
                    with _state_lock:
                        _display_states[track_id] = DisplayState(
                            identity       = None,
                            score          = 0.0,
                            frame_count    = 0,
                            decision       = "QUALITY_FAIL",
                            quality        = quality_score,
                            last_updated   = time.time(),
                            quality_reason = qr.reason,
                        )
                    _task_queue.task_done()
                    continue
            else:
                ok, quality_score = quick_quality_check(aligned_bgr)
                if not ok:
                    # Identity already showing in UI - skip silently
                    _task_queue.task_done()
                    continue

            # ── Stage 5: Embedding Extraction ─────────────────────────────────
            rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            emb = embedder.embeddings(np.expand_dims(rgb, 0))   # shape (1, 512)

            # ── Stage 6: Gatekeeper Decision ──────────────────────────────────
            result = gatekeeper.process(emb, track_id, aligned_bgr, quality_score)

            # ── Stage 7: Persist on new enrollment events ──────────────────────
            if result.decision in (
                GatekeeperDecision.FAST_ENROLLED,
                GatekeeperDecision.REFINED,
            ):
                img_to_save = (
                    result.best_face_image
                    if result.best_face_image is not None
                    else aligned_bgr
                )
                save_person_image(result.identity, img_to_save)

            # ── Check re-entry (identity seen under a different track before) ──
            is_reentry = False
            if result.identity and result.decision == GatekeeperDecision.MATCHED:
                meta = gatekeeper._identity_metadata.get(result.identity, {})
                tracker_ids = meta.get("tracker_ids", set())
                if len(tracker_ids) > 1:
                    is_reentry = True

            # ── Update display state ───────────────────────────────────────────
            with _state_lock:
                _display_states[track_id] = DisplayState(
                    identity       = result.identity,
                    score          = result.score,
                    frame_count    = result.frame_count,
                    decision       = result.decision.name,
                    quality        = quality_score,
                    last_updated   = time.time(),
                    is_reentry     = is_reentry,
                )

        except Exception as exc:
            import traceback
            print(f"[Worker] Unhandled error on track {track_id}: {exc}")
            traceback.print_exc()
        finally:
            _task_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def should_exit() -> bool:
    """System never auto-exits on empty detections. Only 'q'/ESC terminates."""
    return False


def update_state(
    current_state:    Optional[str],
    detection_exists: bool,
    is_known:         bool,
    is_new:           bool,
) -> str:
    """Maps detection outcomes to display state strings."""
    if not detection_exists:
        return "IDLE"
    if is_known:
        return "RECOGNIZED"
    if is_new:
        return "ENROLLING"
    return "ANALYZING"


# ═══════════════════════════════════════════════════════════════════════════════
#  UI RENDERING - bounding box + label overlay
# ═══════════════════════════════════════════════════════════════════════════════


# Colour palette (BGR)
_COLOR_SCANNING   = (200, 200, 200)  # White   - no verdict yet (neutral)
_COLOR_IDENTIFYING= (0,  140, 255)   # Orange  - worker processing
_COLOR_RECOGNIZED = (40, 210,  40)   # Green   - MATCHED / REFINED / steady state
_COLOR_LEARNING   = (0,  210, 230)   # Cyan    - REFINING (building consensus)
_COLOR_ADJUST     = (0,  190, 230)   # Amber   - uncertain angle (actionable)
_COLOR_QUALITY    = (80,  80, 180)   # Muted red - quality failure
_COLOR_FAST       = (0,  200, 100)   # Teal-green - FAST_ENROLLED (fresh, not yet HD)
_COLOR_STATUSBAR  = (30,  30,  30)   # Dark gray - global status bar background


def _quality_reason_to_label(reason: str) -> str:
    """Map quality module failure reason string to an actionable user-facing label."""
    r = reason.lower()
    if "blurry" in r or "laplacian" in r:
        return "Hold still"
    if "overexposed" in r or reason.find("L=") != -1 and ">" in reason:
        return "Too bright"
    if "dark" in r or "underexposed" in r:
        return "More light needed"
    if "pose" in r or "yaw" in r or "pitch" in r or "roll" in r:
        return "Face the camera"
    return "Adjust position"


def _lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linear interpolation between two BGR colors. t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _draw_face_overlay(
    frame:       np.ndarray,
    track_id:    int,
    bbox:        Tuple[int, int, int, int],
    state:       Optional[DisplayState],
    debug_on:    bool = False,
) -> None:
    """
    Draws the bounding box, label, progress bar, and optional debug info
    for a single tracked face.

    Colour semantics:
        White      - Scanning (no worker verdict yet)
        Orange     - Identifying (worker processing, animated dots)
        Green      - Recognized / Verified
        Cyan       - Learning (building consensus, progress bar shown)
        Amber      - Adjust position (uncertain angle)
        Muted red  - Quality issue (specific actionable message)
        Teal       - New face (fast enrolled, refinement pending)
    """
    x, y, w, h = bbox
    now = time.time()

    # Auto-expire stale QUALITY_FAIL / UNCERTAIN labels so the track can retry
    # Fade out in the last second before expiry
    fade_alpha = 1.0
    if state is not None and state.decision in ("QUALITY_FAIL", "UNCERTAIN"):
        elapsed = now - state.last_updated
        if elapsed > STATE_STALE_TIMEOUT:
            state = None   # Revert to scanning - worker will re-assess
        elif elapsed > (STATE_STALE_TIMEOUT - 1.0):
            fade_alpha = 1.0 - (elapsed - (STATE_STALE_TIMEOUT - 1.0))

    # ── Determine colour, label, and sub-label based on state ─────────────
    sub_label = ""           # Smaller hint text below bbox
    show_progress = False    # Whether to draw refinement progress bar
    progress_frac = 0.0      # Progress bar fill fraction

    if state is None:
        # No worker verdict yet - scanning phase
        color = _COLOR_SCANNING
        label = "Scanning..."
        # Use thinner box for scanning state
        box_thickness = 1
    elif state.decision == "MATCHED":
        color = _COLOR_RECOGNIZED
        # Show name only; append "?" briefly for marginal matches, "(back)" for re-entry
        label = state.identity or "Unknown"
        if state.score < 0.85 and (now - state.last_updated) < 2.0:
            label += "?"
        if state.is_reentry and (now - state.last_updated) < 3.0:
            label += " (back)"
        box_thickness = 2
    elif state.decision == "FAST_ENROLLED":
        color = _COLOR_FAST
        label = f"New face - {state.identity}"
        box_thickness = 2
    elif state.decision == "REFINING":
        color = _COLOR_LEARNING
        progress_frac = state.frame_count / CONSENSUS_FRAMES
        if state.frame_count >= 5:
            label = f"Almost done... {state.identity}"
        else:
            label = f"Learning... {state.identity}"
        show_progress = True
        box_thickness = 2
    elif state.decision == "REFINED":
        color = _COLOR_RECOGNIZED
        if (now - state.last_updated) < 2.0:
            label = f"{state.identity} [Verified]"
        else:
            label = state.identity or "Unknown"
        box_thickness = 2
    elif state.decision == "UNCERTAIN":
        color = _COLOR_ADJUST
        label = "Turn slightly"
        box_thickness = 2
    elif state.decision == "QUALITY_FAIL":
        color = _COLOR_QUALITY
        label = _quality_reason_to_label(state.quality_reason)
        box_thickness = 2
    else:
        # Worker submitted but hasn't returned yet - identifying phase
        dots = "." * (int(now) % 3 + 1)
        elapsed_since = now - state.last_updated if state else 0
        if elapsed_since > 3.0:
            label = "Please wait - processing"
        else:
            label = f"Identifying{dots}"
        color = _COLOR_IDENTIFYING
        box_thickness = 2

    # Apply fade-out for expiring states
    if fade_alpha < 1.0:
        gray = (120, 120, 120)
        color = _lerp_color(gray, color, fade_alpha)

    # ── Bounding box ──────────────────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

    # ── Label with filled background ──────────────────────────────────────
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scl  = 0.50
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scl, thickness)
    label_y = max(y, th + 6)   # Prevent label from going above frame edge
    cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 6, label_y), color, -1)
    cv2.putText(frame, label, (x + 3, label_y - 3), font, font_scl, (0, 0, 0), thickness)

    # ── Progress bar (LEARNING / REFINING state only) ─────────────────────
    if show_progress:
        bar_y1 = y + h + 4
        bar_y2 = y + h + 10
        bar_x2 = x + w
        # Background
        cv2.rectangle(frame, (x, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
        # Fill
        fill_x = x + int(w * progress_frac)
        bar_color = _COLOR_LEARNING
        cv2.rectangle(frame, (x, bar_y1), (fill_x, bar_y2), bar_color, -1)

    # ── Debug overlay (toggled at runtime with 'd' key) ───────────────────
    if debug_on and state is not None:
        dbg_y = y + h + (22 if show_progress else 16)
        dbg_txt = f"T{track_id}  Q={state.quality:.2f}  S={state.score:.2f}"
        if state.quality_reason:
            dbg_txt += f"  [{state.quality_reason[:30]}]"
        cv2.putText(
            frame, dbg_txt,
            (x, dbg_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
        )


def _draw_status_bar(
    frame:        np.ndarray,
    num_faces:    int,
    queue_size:   int,
    status_msg:   Optional[str] = None,
    debug_on:     bool = False,
    fps:          float = 0.0,
) -> None:
    """
    Draws a 36px dark status bar across the top of the frame.
    Shows face count (left) and queue health (right).
    Optional status message overrides the face count (e.g. "Initializing...").
    """
    h, w = frame.shape[:2]
    bar_h = 36

    # Dark background strip
    cv2.rectangle(frame, (0, 0), (w, bar_h), _COLOR_STATUSBAR, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scl  = 0.50
    thickness = 1

    # Left side: face count or status message
    if status_msg:
        left_text = status_msg
    elif num_faces == 0:
        left_text = "No faces detected"
    elif num_faces == 1:
        left_text = "1 person"
    else:
        left_text = f"{num_faces} people"

    cv2.putText(frame, left_text, (10, 24), font, font_scl, (255, 255, 255), thickness)

    # Right side: queue health indicator
    if queue_size >= WORKER_QUEUE_SIZE:
        q_text  = "[dropping frames]"
        q_color = (80, 80, 220)   # Red
    elif queue_size >= WORKER_QUEUE_SIZE - 1:
        q_text  = "[busy]"
        q_color = (0, 200, 255)   # Yellow
    else:
        q_text  = ""
        q_color = (255, 255, 255)

    if q_text:
        (tw, _), _ = cv2.getTextSize(q_text, font, font_scl, thickness)
        cv2.putText(frame, q_text, (w - tw - 10, 24), font, font_scl, q_color, thickness)

    # Debug: FPS counter (only shown with debug overlay on)
    if debug_on:
        fps_text = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, font, 0.45, thickness)
        x_pos = w - tw - 10 - (len(q_text) * 8 + 20 if q_text else 0)
        cv2.putText(frame, fps_text, (x_pos, 24), font, 0.45, (180, 180, 180), thickness)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN THREAD - stages 1 & 2 (detection + tracking), rendering loop
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Start worker thread ──────────────────────────────────────────────────
    worker = threading.Thread(
        target=recognition_worker, daemon=True, name="RecognitionWorker"
    )
    worker.start()

    # ── Initialise MTCNN detector ─────────────────────────────────────────────
    # MTCNN is a cascade of three CNNs (P-Net, R-Net, O-Net). It is slower than
    # Haar Cascade on CPU but provides far fewer false positives and gives us
    # 5-point landmarks needed for alignment and pose estimation.
    print("[Main] Loading MTCNN detector...")
    detector = MTCNN()
    tracker  = FaceTracker()

    # Per-track frame counter for INFERENCE_SKIP throttling
    # Maps track_id → frame index of last worker submission
    last_submitted: Dict[int, int] = {}
    frame_idx = 0

    print("[Main] Starting webcam. Press 'q' or ESC to quit. Press 'd' to toggle debug overlay.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Could not open webcam.")
        sys.exit(1)

    fps_time  = time.time()
    fps_disp  = 0.0
    debug_on  = False          # Toggle with 'd' key at runtime
    status_msg: Optional[str] = "Initializing model..."  # Shown until worker is ready

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] Lost webcam feed - exiting.")
            break

        frame_idx += 1

        # ── Stage 1: Detection ────────────────────────────────────────────────
        # Downsample for faster MTCNN inference; scale detections back up.
        h_full, w_full = frame.shape[:2]
        scale          = DETECT_DOWNSAMPLE
        small_rgb      = cv2.cvtColor(
            cv2.resize(frame, (w_full // scale, h_full // scale)),
            cv2.COLOR_BGR2RGB,
        )
        try:
            raw_detections = detector.detect_faces(small_rgb)
        except ValueError:
            # MTCNN bug in TF > 2.10: crashes when P-Net finds boxes but R-Net/O-Net receive none.
            raw_detections = []

        # Filter and scale detections back to full resolution
        detections: List = []
        for r in raw_detections:
            if r['confidence'] < MTCNN_CONF_MIN:
                continue

            x, y, w, h = [v * scale for v in r['box']]

            # Clamp to frame boundaries (MTCNN can return slightly out-of-bound boxes)
            x = max(0, x);  y = max(0, y)
            w = min(w, w_full - x);  h = min(h, h_full - y)

            if w < MIN_FACE_PX or h < MIN_FACE_PX:
                continue

            # Scale keypoints back up
            kpts = {
                k: (int(v[0] * scale), int(v[1] * scale))
                for k, v in r.get('keypoints', {}).items()
            }

            detections.append(((x, y, w, h), kpts, r['confidence']))

        # ── Stage 2: Tracking ─────────────────────────────────────────────────
        active_tracks = tracker.update(detections)

        # Remove display state for tracks that are no longer active so stale
        # labels don't linger after a face leaves the frame.
        active_ids = {t.track_id for t in active_tracks}
        now = time.time()
        with _state_lock:
            # Clear states for tracks the IoU tracker has dropped
            stale_ids = [tid for tid in _display_states if tid not in active_ids]
            for tid in stale_ids:
                del _display_states[tid]
            # Also clear QUALITY_FAIL states older than STATE_STALE_TIMEOUT so
            # they get re-assessed on the next submission instead of staying red
            expired_qf = [
                tid for tid, s in _display_states.items()
                if s.decision == "QUALITY_FAIL"
                and (now - s.last_updated) > STATE_STALE_TIMEOUT
            ]
            for tid in expired_qf:
                del _display_states[tid]

        # ── Stage 3 + handoff: Align and send to worker (throttled) ──────────
        for track in active_tracks:
            tid = track.track_id

            # Throttle: don't send every single frame to the worker
            last = last_submitted.get(tid, -(INFERENCE_SKIP + 1))
            if (frame_idx - last) < INFERENCE_SKIP:
                continue
            last_submitted[tid] = frame_idx

            # Don't queue if worker is already saturated
            if _task_queue.full():
                continue

            # Stage 3: Face alignment (affine via landmarks)
            aligned = align_face(frame, track.bbox, track.landmarks)
            if aligned is None or aligned.size == 0:
                continue

            try:
                _task_queue.put_nowait((aligned, tid, track.landmarks))
            except queue.Full:
                pass   # Drop this frame silently - worker will catch up

        # ── Update status message ─────────────────────────────────────────────
        if status_msg and _worker_ready.is_set():
            status_msg = f"Ready - {_known_count} known faces loaded"
            # Clear the ready message after 3 seconds
            if not hasattr(main, '_ready_time'):
                main._ready_time = time.time()
            elif (time.time() - main._ready_time) > 3.0:
                status_msg = None

        # ── Render ────────────────────────────────────────────────────────────
        with _state_lock:
            current_states = dict(_display_states)

        for track in active_tracks:
            # Flicker filter: skip rendering for tracks seen < 3 frames
            if track.frames_seen < 3:
                continue
            _draw_face_overlay(
                frame,
                track.track_id,
                track.bbox,
                current_states.get(track.track_id),
                debug_on=debug_on,
            )

        # FPS counter
        now     = time.time()
        elapsed = now - fps_time
        if elapsed >= 1.0:
            fps_disp = frame_idx / elapsed   # approx; resets per second
            fps_time = now
            frame_idx = 0

        # Status bar (replaces old FPS text overlay)
        _draw_status_bar(
            frame,
            num_faces  = sum(1 for t in active_tracks if t.frames_seen >= 3),
            queue_size = _task_queue.qsize(),
            status_msg = status_msg,
            debug_on   = debug_on,
            fps        = fps_disp,
        )

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # 'q' or ESC
            break
        elif key == ord('d'):       # Toggle debug overlay
            debug_on = not debug_on

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[Main] Shutting down...")
    _stop_flag.set()
    cap.release()
    cv2.destroyAllWindows()
    worker.join(timeout=3.0)
    print("[Main] Done.")


if __name__ == "__main__":
    main()
