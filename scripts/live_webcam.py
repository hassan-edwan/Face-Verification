"""
Live Webcam Face Recognition — Gatekeeper Architecture
=======================================================
Refactored pipeline with modular stages:

  Stage 1 — Detection   : MTCNN (replaces Haar Cascade; gives landmarks)
  Stage 2 — Tracking    : IoU tracker (stable track_id per physical face)
  Stage 3 — Alignment   : Affine transform via eye landmarks → 160×160 crop
  Stage 4 — Quality Gate: FIQA (sharpness, illumination, pose proxy)
  Stage 5 — Embedding   : FaceNet (keras-facenet, unchanged from original)
  Stage 6 — Gatekeeper  : Consensus buffer + dual-threshold de-duplication
  Stage 7 — Persistence : Save best-quality frame only on confirmed enrollment

Key behavioural changes from the original:
  • New face NOT enrolled immediately — requires CONSENSUS_FRAMES quality frames.
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

from src.embedder import Embedder, get_embedder
from mtcnn import MTCNN

from src.quality    import assess_quality, quick_quality_check
from src.tracker    import FaceTracker
from src.gatekeeper import Gatekeeper, GatekeeperDecision, CONSENSUS_FRAMES, MAX_EMBEDDINGS_PER_IDENTITY
from src.database   import FaceDatabase
from src.memory     import MemoryStore, SessionManager, SessionState
from src.speech     import SpeechTranscriber, TranscriptState, SPEECH_AVAILABLE

# ── PIPELINE CONFIG ───────────────────────────────────────────────────────────
# All tunable parameters are here. Thresholds for matching/enrollment are in
# src/gatekeeper.py. Quality thresholds are in src/quality.py.

MIN_FACE_PX       = 20          # run_014: 40→20 under ArcFace (detection + embedding hold at sub-30 px)
MTCNN_CONF_MIN    = 0.90        # Minimum MTCNN detection confidence to accept
DETECT_DOWNSAMPLE = 2           # Run MTCNN on 1/N resolution frame for speed
                                # (coordinates are scaled back up automatically)
DETECT_SKIP       = 2           # Run MTCNN every N-th frame; reuse tracks otherwise.
                                # Halves main-thread cost (MTCNN is 40-80ms).
INFERENCE_SKIP    = 2           # Send to worker at most once every N frames per
                                # track. Lower = faster first-identification.
WORKER_QUEUE_SIZE = 6           # Max outstanding tasks in the worker queue

ENROLL_BASE          = os.path.join(PROJECT_ROOT, "data", "enrollments")
DEBUG_OVERLAY       = True        # Show track IDs and quality scores on screen
STATE_STALE_TIMEOUT   = 4.0       # Seconds before QUALITY_FAIL/UNCERTAIN state is auto-cleared
PENDING_STATE_TIMEOUT = 8.0       # Seconds before ANY non-terminal state (QUEUED, pipeline
                                  # stages) is auto-cleared. Prevents permanent stalls.

# Alignment lives in src/alignment.py — shared with scripts/server.py and
# src/real_data_eval.py so the live path and the offline eval agree on
# exactly how faces are warped to canonical 160×160 inputs.
from src.alignment import align_face, align_face_5point, TARGET_SIZE

# Pick alignment geometry from the embedder's declared input_size so the
# live pipeline's crops match what ArcFace / FaceNet actually trained on.
# Resolved once at import — FACE_EMBEDDER doesn't change mid-process.
_ACTIVE_ALIGN = (align_face_5point
                 if get_embedder().input_size == (112, 112)
                 else align_face)


# ═══════════════════════════════════════════════════════════════════════════════
#  STORAGE — unchanged from original for full backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def load_persistent_faces(embedder: Embedder) -> Tuple[Dict, int]:
    """
    Scans data/enrollments/ for 'Person N' folders and loads their stored face
    images as embeddings. Unchanged from the original implementation.

    Returns:
        (known_faces dict, next_person_counter int)
    """
    known_faces    = {}
    person_counter = 1

    if not os.path.exists(ENROLL_BASE):
        os.makedirs(ENROLL_BASE, exist_ok=True)
        return known_faces, person_counter

    print("[Worker] Scanning data/enrollments/ for previously saved faces...")
    dirs = [
        d for d in os.listdir(ENROLL_BASE)
        if os.path.isdir(os.path.join(ENROLL_BASE, d))
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

        img_paths = glob.glob(os.path.join(ENROLL_BASE, d, "*.jpg"))
        if not img_paths:
            continue

        known_faces[d] = []
        for path in img_paths:
            img = cv2.imread(path)
            if img is not None:
                emb = embedder.embed(img)
                known_faces[d].append(emb)

    print(f"[Worker] Loaded {len(known_faces)} known identity/ies from disk.")
    return known_faces, person_counter


def save_person_image(person_name: str, face_bgr: np.ndarray) -> None:
    """
    Saves the best-quality aligned face crop (BGR) to data/enrollments/<person_name>/.
    Only called on ENROLLED events — one save per new identity.
    """
    person_dir = os.path.join(ENROLL_BASE, person_name)
    os.makedirs(person_dir, exist_ok=True)

    existing = glob.glob(os.path.join(person_dir, "*.jpg"))
    next_idx = len(existing) + 1
    stem     = person_name.replace(" ", "_")
    filepath = os.path.join(person_dir, f"{stem}_{str(next_idx).zfill(4)}.jpg")

    cv2.imwrite(filepath, face_bgr)
    print(f"[Persistent Storage] Saved → {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE — main thread ↔ worker thread
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
    pipeline_stage: str   = ""     # Current worker sub-stage ("quality", "embedding", "matching")


_display_states: Dict[int, DisplayState] = {}   # track_id → DisplayState
_state_lock     = threading.Lock()
_task_queue     = queue.PriorityQueue(maxsize=WORKER_QUEUE_SIZE)
_task_counter   = 0              # Monotonic tiebreaker for PriorityQueue
_stop_flag      = threading.Event()
_worker_ready   = threading.Event()              # Set when worker finishes loading
_worker_alive   = threading.Event()              # Cleared if worker thread dies
_known_count    = 0                              # Number of known faces loaded from disk
_face_db: Optional[FaceDatabase] = None          # Set by worker thread before _worker_ready
_session_mgr: Optional[SessionManager] = None    # Set by worker thread before _worker_ready
_transcriber: Optional[SpeechTranscriber] = None # Initialized in main(), not worker
# Latest primary face confidence score. Written each frame by the main thread,
# read by the speech audio thread when a phrase is finalized so the stored
# transcript event carries the recognition confidence at that moment.
_latest_primary_score: float = 0.0

# Camera reader thread state — decouples capture from processing to prevent jitter
_latest_frame: Optional[np.ndarray] = None
_frame_lock   = threading.Lock()
_camera_ok    = threading.Event()                # Set when first frame is captured


def _camera_reader(cap: cv2.VideoCapture) -> None:
    """Continuously reads frames from the camera into a shared buffer."""
    global _latest_frame
    while not _stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            _camera_ok.clear()
            break
        with _frame_lock:
            _latest_frame = frame
        _camera_ok.set()


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD — Stages 4, 5, 6, 7
# ═══════════════════════════════════════════════════════════════════════════════

def recognition_worker() -> None:
    """
    Background thread: FaceNet model + Gatekeeper.
    Stages: Quality Gate -> Embedding -> Gatekeeper -> Persist -> UI update.
    """
    try:
        embedder = get_embedder()
        print(f"[Worker] Loading {embedder.name} embedder (input {embedder.input_size})...")

        known_faces, person_counter = load_persistent_faces(embedder)
        gatekeeper = Gatekeeper(known_faces, person_counter)

        # Initialize database and seed existing disk identities
        global _known_count, _face_db, _session_mgr
        db = FaceDatabase(os.path.join(PROJECT_ROOT, "data", "faces.db"))
        _face_db = db
        for name in known_faces:
            if db.get_identity(name) is None:
                person_dir = os.path.join(ENROLL_BASE, name)
                jpgs = sorted(glob.glob(os.path.join(person_dir, "*.jpg")))
                db.create_identity(name, jpgs[0] if jpgs else "")

        # Initialize memory system
        memory_store = MemoryStore(os.path.join(PROJECT_ROOT, "data", "memory.db"))
        _session_mgr = SessionManager(memory_store)

        _known_count = len(known_faces)
        _worker_ready.set()
        _worker_alive.set()
        print("[Worker] Ready. Listening for frames...")
    except Exception as exc:
        print(f"[Worker] FATAL: Failed to initialize — {exc}")
        _worker_ready.set()   # Unblock main thread so it doesn't hang on splash
        return

    while not _stop_flag.is_set():
        try:
            task = _task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        track_id = None  # set before try so except block can always reference it
        try:
            _priority, _seq, aligned_bgr, track_id, landmarks = task

            # ── Pipeline stage: signal "quality" to UI ────────────────────────
            with _state_lock:
                st = _display_states.get(track_id)
                if st is not None:
                    st.pipeline_stage = "quality"

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
                    continue  # task_done() called by finally block
            else:
                ok, quality_score = quick_quality_check(aligned_bgr)
                if not ok:
                    continue  # task_done() called by finally block

            # ── Stage 5: Embedding Extraction ─────────────────────────────────
            with _state_lock:
                st = _display_states.get(track_id)
                if st is not None:
                    st.pipeline_stage = "embedding"

            emb = embedder.embed(aligned_bgr)   # shape (dim,), unit norm

            # ── Stage 6: Gatekeeper Decision ──────────────────────────────────
            with _state_lock:
                st = _display_states.get(track_id)
                if st is not None:
                    st.pipeline_stage = "matching"
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
                threading.Thread(
                    target=save_person_image,
                    args=(result.identity, img_to_save),
                    daemon=True,
                ).start()

            # ── Database writes (fire-and-forget, non-blocking) ───────────────
            if result.identity and db:
                if result.decision == GatekeeperDecision.FAST_ENROLLED:
                    person_dir = os.path.join(ENROLL_BASE, result.identity)
                    stem = result.identity.replace(" ", "_")
                    thumb = os.path.join(person_dir, f"{stem}_0001.jpg")
                    db.create_identity(result.identity, thumb)
                elif result.decision == GatekeeperDecision.MATCHED:
                    db.update_match(result.identity)
                elif result.decision == GatekeeperDecision.REFINED:
                    person_dir = os.path.join(ENROLL_BASE, result.identity)
                    jpgs = sorted(glob.glob(os.path.join(person_dir, "*.jpg")))
                    if jpgs:
                        db.update_thumbnail(result.identity, jpgs[-1])

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

    # Worker loop exited — signal main thread so it stops submitting
    _worker_alive.clear()
    print("[Worker] Thread exiting.")


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
#  UI RENDERING — bounding box + label overlay
# ═══════════════════════════════════════════════════════════════════════════════


# Colour palette (BGR)
_COLOR_SCANNING   = (200, 200, 200)  # White   — no verdict yet (neutral)
_COLOR_IDENTIFYING= (0,  140, 255)   # Orange  — worker processing
_COLOR_RECOGNIZED = (40, 210,  40)   # Green   — MATCHED / REFINED / steady state
_COLOR_LEARNING   = (0,  210, 230)   # Cyan    — REFINING (building consensus)
_COLOR_ADJUST     = (0,  190, 230)   # Amber   — uncertain angle (actionable)
_COLOR_QUALITY    = (80,  80, 180)   # Muted red — quality failure
_COLOR_FAST       = (0,  200, 100)   # Teal-green — FAST_ENROLLED (fresh, not yet HD)
_COLOR_STATUSBAR  = (30,  30,  30)   # Dark gray — global status bar background


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
        White      — Scanning (no worker verdict yet)
        Orange     — Identifying (worker processing, animated dots)
        Green      — Recognized / Verified
        Cyan       — Learning (building consensus, progress bar shown)
        Amber      — Adjust position (uncertain angle)
        Muted red  — Quality issue (specific actionable message)
        Teal       — New face (fast enrolled, refinement pending)
    """
    x, y, w, h = bbox
    now = time.time()

    # Auto-expire stale states so the track can retry (prevents permanent stalls)
    fade_alpha = 1.0
    if state is not None and state.decision not in ("MATCHED", "REFINED"):
        elapsed = now - state.last_updated
        timeout = STATE_STALE_TIMEOUT if state.decision in ("QUALITY_FAIL", "UNCERTAIN") else PENDING_STATE_TIMEOUT
        if elapsed > timeout:
            state = None   # Revert to scanning — worker will re-assess
        elif elapsed > (timeout - 1.0):
            fade_alpha = 1.0 - (elapsed - (timeout - 1.0))

    # ── Determine colour, label, and sub-label based on state ─────────────
    sub_label = ""           # Smaller hint text below bbox
    show_progress = False    # Whether to draw refinement progress bar
    progress_frac = 0.0      # Progress bar fill fraction

    if state is None:
        # No worker verdict yet — scanning phase
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
        # Confidence chip for marginal matches
        if state.score < 0.85 and (now - state.last_updated) < 3.0:
            sub_label = "(moderate match)"
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
        # Worker submitted but hasn't returned yet — identifying phase
        elapsed_since = now - state.last_updated if state else 0
        # Map pipeline stage to user-facing text
        stage_labels = {
            "quality":   "Checking quality...",
            "embedding": "Computing identity...",
            "matching":  "Searching faces...",
        }
        stage = getattr(state, 'pipeline_stage', '') if state else ''
        if elapsed_since > 3.0:
            label = "Busy - other faces in queue" if _task_queue.qsize() > 2 else "Please wait - processing"
        elif elapsed_since > 1.5:
            label = "Please wait - processing"
        elif stage in stage_labels:
            label = stage_labels[stage]
        else:
            dots = "." * (int(now) % 3 + 1)
            label = f"Analyzing{dots}"
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

    # ── Sub-label (smaller hint text below main label) ─────────────────
    if sub_label:
        sub_scl = 0.38
        (stw, sth), _ = cv2.getTextSize(sub_label, font, sub_scl, 1)
        sub_y = label_y + sth + 4
        cv2.putText(frame, sub_label, (x + 3, sub_y), font, sub_scl, (180, 180, 180), 1)

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
        # Fraction label (e.g. "3/7")
        frac_text = f"{state.frame_count}/{CONSENSUS_FRAMES}"
        cv2.putText(frame, frac_text, (bar_x2 + 4, bar_y2), font, 0.35, (180, 180, 180), 1)

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
    num_known:    int,
    num_new:      int,
    queue_size:   int,
    status_msg:   Optional[str] = None,
    debug_on:     bool = False,
    fps:          float = 0.0,
) -> None:
    """
    Draws a 36px dark status bar across the top of the frame.
    Shows face count with breakdown (left) and queue health (right).
    Optional status message overrides the face count (e.g. "Initializing...").
    """
    h, w = frame.shape[:2]
    bar_h = 36

    # Dark background strip
    cv2.rectangle(frame, (0, 0), (w, bar_h), _COLOR_STATUSBAR, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scl  = 0.50
    thickness = 1

    # Left side: face count with known/new breakdown, or status message
    if status_msg:
        left_text = status_msg
    elif num_faces == 0:
        left_text = "No faces detected"
    elif num_faces == 1:
        left_text = "1 person"
    else:
        parts = []
        if num_known > 0:
            parts.append(f"{num_known} known")
        if num_new > 0:
            parts.append(f"{num_new} new")
        if parts:
            left_text = f"{num_faces} people ({', '.join(parts)})"
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
#  INFO PANEL — right-side identity detail panel (2/5 of screen)
# ═══════════════════════════════════════════════════════════════════════════════

_thumbnail_cache: Dict[str, Optional[np.ndarray]] = {}  # identity name → 96×96 BGR or None


def _get_thumbnail(identity: Optional[str]) -> Optional[np.ndarray]:
    """Load and cache a 96x96 thumbnail for an identity from its best image on disk."""
    if not identity or not _face_db:
        return None
    # Use identity name as cache key (stable across track changes)
    cache_key = identity
    if cache_key in _thumbnail_cache:
        return _thumbnail_cache[cache_key]
    info = _face_db.get_identity(identity)
    if not info or not info.get("best_thumbnail_path"):
        _thumbnail_cache[cache_key] = None
        return None
    path = info["best_thumbnail_path"]
    if not os.path.exists(path):
        _thumbnail_cache[cache_key] = None
        return None
    img = cv2.imread(path)
    if img is None:
        _thumbnail_cache[cache_key] = None
        return None
    thumb = cv2.resize(img, (96, 96))
    _thumbnail_cache[cache_key] = thumb
    return thumb


def _select_primary_face(
    active_tracks: List,
    current_states: Dict[int, DisplayState],
) -> Optional[int]:
    """Pick the most relevant track_id for the info panel."""
    visible = [t for t in active_tracks if t.frames_seen >= 3]
    if not visible:
        return None
    # Prefer recognized faces (largest bbox)
    recognized = [
        t for t in visible
        if current_states.get(t.track_id) is not None
        and current_states[t.track_id].decision in ("MATCHED", "REFINED")
    ]
    pool = recognized if recognized else visible
    best = max(pool, key=lambda t: t.bbox[2] * t.bbox[3])
    return best.track_id


def _format_relative_time(ts: float) -> str:
    """Format a timestamp as relative time (e.g. '2 min ago')."""
    diff = time.time() - ts
    if diff < 5:
        return "just now"
    if diff < 60:
        return f"{int(diff)}s ago"
    if diff < 3600:
        return f"{int(diff / 60)} min ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    return time.strftime("%b %d, %Y", time.localtime(ts))


def _draw_info_panel(
    panel:            np.ndarray,
    state:            Optional[DisplayState],
    db_info:          Optional[dict],
    total_faces:      int,
    all_states:       Dict[int, DisplayState],
    session_info:     Optional[dict] = None,
    recent_events:    Optional[List[dict]] = None,
    transcript_state: Optional[TranscriptState] = None,
) -> None:
    """
    Draws the identity info panel (right side of split screen).
    All rendering via OpenCV primitives on the pre-allocated panel array.
    """
    h, w = panel.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray = (160, 160, 160)
    dark_gray = (80, 80, 80)

    # ── Header ────────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (w, 36), (40, 40, 40), -1)
    cv2.putText(panel, "IDENTITY INFO", (12, 24), font, 0.50, gray, 1)
    cv2.line(panel, (0, 36), (w, 36), dark_gray, 1)

    y = 50  # current drawing y position

    # ── No face state ─────────────────────────────────────────────────────────
    if state is None:
        cv2.putText(panel, "No face detected", (12, h // 2 - 10), font, 0.55, gray, 1)
        cv2.putText(panel, "System ready", (12, h // 2 + 20), font, 0.40, dark_gray, 1)
        return

    # ── Thumbnail + Name ──────────────────────────────────────────────────────
    identity = state.identity
    thumb = _get_thumbnail(identity) if identity else None
    thumb_x, thumb_y = 12, y
    thumb_size = 96

    if thumb is not None:
        panel[thumb_y:thumb_y + thumb_size, thumb_x:thumb_x + thumb_size] = thumb
    else:
        # Gray placeholder
        cv2.rectangle(panel, (thumb_x, thumb_y),
                       (thumb_x + thumb_size, thumb_y + thumb_size), dark_gray, -1)
        cv2.putText(panel, "?", (thumb_x + 35, thumb_y + 60), font, 1.0, gray, 2)

    # Name to the right of thumbnail
    name_x = thumb_x + thumb_size + 12
    display_name = identity or "Unknown"
    if db_info and db_info.get("display_name"):
        display_name = db_info["display_name"]
    cv2.putText(panel, display_name, (name_x, y + 28), font, 0.65, white, 1)

    # Internal name (if display_name differs)
    if db_info and db_info.get("display_name") and identity:
        cv2.putText(panel, identity, (name_x, y + 50), font, 0.38, gray, 1)

    y = thumb_y + thumb_size + 20

    # ── Status badge ──────────────────────────────────────────────────────────
    decision = state.decision
    status_text, status_color = {
        "MATCHED":       ("Recognized",   _COLOR_RECOGNIZED),
        "REFINED":       ("Verified",     _COLOR_RECOGNIZED),
        "FAST_ENROLLED": ("New face",     _COLOR_FAST),
        "REFINING":      ("Learning...",  _COLOR_LEARNING),
        "UNCERTAIN":     ("Uncertain",    _COLOR_ADJUST),
        "QUALITY_FAIL":  ("Quality issue", _COLOR_QUALITY),
        "QUEUED":        ("Analyzing...", _COLOR_IDENTIFYING),
    }.get(decision, ("Processing...", _COLOR_IDENTIFYING))

    cv2.putText(panel, "Status:", (12, y), font, 0.42, gray, 1)
    (tw, _), _ = cv2.getTextSize(status_text, font, 0.45, 1)
    badge_x = 80
    cv2.rectangle(panel, (badge_x, y - 14), (badge_x + tw + 10, y + 4), status_color, -1)
    cv2.putText(panel, status_text, (badge_x + 5, y), font, 0.45, (0, 0, 0), 1)
    y += 28

    # ── Confidence bar ────────────────────────────────────────────────────────
    if state.score > 0:
        pct = int(state.score * 100)
        cv2.putText(panel, f"Confidence: {pct}%", (12, y), font, 0.42, gray, 1)
        y += 6
        bar_w = w - 24
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 8), dark_gray, -1)
        fill = int(bar_w * state.score)
        bar_color = _COLOR_RECOGNIZED if state.score >= 0.70 else _COLOR_ADJUST
        cv2.rectangle(panel, (12, y), (12 + fill, y + 8), bar_color, -1)
        y += 20

    # ── Quality score ─────────────────────────────────────────────────────────
    if state.quality > 0:
        cv2.putText(panel, f"Quality: {state.quality:.2f}", (12, y), font, 0.42, gray, 1)
        y += 22

    # ── Quality failure reason ────────────────────────────────────────────────
    if state.quality_reason:
        reason_label = _quality_reason_to_label(state.quality_reason)
        cv2.putText(panel, reason_label, (12, y), font, 0.42, _COLOR_QUALITY, 1)
        y += 22

    # ── Separator ─────────────────────────────────────────────────────────────
    cv2.line(panel, (12, y), (w - 12, y), dark_gray, 1)
    y += 16

    # ── Database fields ───────────────────────────────────────────────────────
    if db_info:
        cv2.putText(panel, f"Enrolled: {_format_relative_time(db_info['created_at'])}",
                     (12, y), font, 0.40, gray, 1)
        y += 20
        cv2.putText(panel, f"Last seen: {_format_relative_time(db_info['last_seen_at'])}",
                     (12, y), font, 0.40, gray, 1)
        y += 20
        cv2.putText(panel, f"Total matches: {db_info['total_matches']}",
                     (12, y), font, 0.40, gray, 1)
        y += 20

    # ── Progress bar for REFINING ─────────────────────────────────────────────
    if decision == "REFINING" and state.frame_count > 0:
        y += 4
        cv2.putText(panel, f"Enrollment: {state.frame_count}/{CONSENSUS_FRAMES}",
                     (12, y), font, 0.42, _COLOR_LEARNING, 1)
        y += 6
        bar_w = w - 24
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 8), dark_gray, -1)
        fill = int(bar_w * state.frame_count / CONSENSUS_FRAMES)
        cv2.rectangle(panel, (12, y), (12 + fill, y + 8), _COLOR_LEARNING, -1)
        y += 20

    # ── Session state + recent memory ────────────────────────────────────────
    if session_info:
        cv2.line(panel, (12, y), (w - 12, y), dark_gray, 1)
        y += 16
        sess_state = session_info["state"]
        sess_color = {
            "ACTIVE": _COLOR_RECOGNIZED, "GATING": _COLOR_ADJUST,
            "PAUSED": _COLOR_QUALITY,
        }.get(sess_state, dark_gray)
        cv2.putText(panel, "Session:", (12, y), font, 0.42, gray, 1)
        cv2.putText(panel, sess_state, (80, y), font, 0.42, sess_color, 1)
        y += 20

    if recent_events:
        cv2.putText(panel, "Recent memory:", (12, y), font, 0.38, gray, 1)
        y += 16
        for evt in (recent_events or [])[:3]:
            # Truncate content to fit panel
            content = evt.get("content", "")[:45]
            etype = evt.get("event_type", "?")
            ts = evt.get("created_at", 0)
            age = _format_relative_time(ts) if ts else ""
            label = f"[{etype}] {content}"
            cv2.putText(panel, label, (16, y), font, 0.33, (140, 180, 140), 1)
            cv2.putText(panel, age, (w - 80, y), font, 0.28, dark_gray, 1)
            y += 15

    # ── Live transcript ───────────────────────────────────────────────────────
    if transcript_state:
        ts = transcript_state
        if ts.is_listening or ts.partial or ts.lines:
            cv2.line(panel, (12, y), (w - 12, y), dark_gray, 1)
            y += 14

            # Listening indicator (auto-driven by session gating)
            if ts.is_listening:
                dot_pulse = int(time.time() * 3) % 2  # blink every ~0.33s
                mic_color = (0, 0, 200) if dot_pulse else (0, 0, 120)
                cv2.circle(panel, (18, y - 4), 5, mic_color, -1)
                cv2.putText(panel, "LISTENING", (28, y), font, 0.38, (0, 140, 255), 1)
            else:
                cv2.putText(panel, "Transcript", (12, y), font, 0.38, gray, 1)
            y += 16

            # Show last 2 finalized lines (dimmer, stable)
            for line in ts.lines[-2:]:
                wrapped = line[:50]
                cv2.putText(panel, wrapped, (16, y), font, 0.33, (120, 140, 120), 1)
                y += 14
                if len(line) > 50:
                    cv2.putText(panel, line[50:100], (16, y), font, 0.33, (120, 140, 120), 1)
                    y += 14

            # Show partial (brighter, in-progress, with cursor)
            if ts.partial:
                cursor = "|" if int(time.time() * 2) % 2 else ""
                partial_text = ts.partial[:55] + cursor
                cv2.putText(panel, partial_text, (16, y), font, 0.35, white, 1)
                y += 16

        elif ts.error:
            cv2.putText(panel, f"Speech: {ts.error[:40]}", (12, y), font, 0.32, _COLOR_QUALITY, 1)
            y += 16

    # ── Multi-face footer ─────────────────────────────────────────────────────
    if total_faces > 0:
        footer_y = h - 20
        cv2.putText(panel, f"Tracking {total_faces} face{'s' if total_faces != 1 else ''}",
                     (12, footer_y), font, 0.38, gray, 1)
        # State dots
        dot_x = 12 + 140
        for _, s in sorted(all_states.items()):
            dot_color = {
                "MATCHED": _COLOR_RECOGNIZED, "REFINED": _COLOR_RECOGNIZED,
                "FAST_ENROLLED": _COLOR_FAST, "REFINING": _COLOR_LEARNING,
            }.get(s.decision, dark_gray)
            cv2.circle(panel, (dot_x, footer_y - 4), 5, dot_color, -1)
            dot_x += 14


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN THREAD — stages 1 & 2 (detection + tracking), rendering loop
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

    # ── Startup splash — show loading screen until worker is ready ────────────
    splash = np.zeros((480, 640, 3), dtype=np.uint8)
    splash_font = cv2.FONT_HERSHEY_SIMPLEX
    _splash_msg = "Loading face model..."
    (tw, th), _ = cv2.getTextSize(_splash_msg, splash_font, 0.7, 2)
    cv2.putText(splash, _splash_msg, ((640 - tw) // 2, (480 + th) // 2),
                splash_font, 0.7, (200, 200, 200), 2)
    cv2.imshow("Face Recognition", splash)
    _startup_deadline = time.time() + 120.0  # 2 minute max startup
    while not _worker_ready.is_set():
        if time.time() > _startup_deadline:
            print("[Main] Worker failed to start within 120s — exiting.")
            cv2.destroyAllWindows()
            _stop_flag.set()
            return
        key = cv2.waitKey(100) & 0xFF
        if key in (ord('q'), 27):
            print("[Main] Quit during startup.")
            cv2.destroyAllWindows()
            _stop_flag.set()
            return

    # ── Open camera after model is ready ──────────────────────────────────────
    splash[:] = 0
    _splash_msg = "Starting camera..."
    (tw, th), _ = cv2.getTextSize(_splash_msg, splash_font, 0.7, 2)
    cv2.putText(splash, _splash_msg, ((640 - tw) // 2, (480 + th) // 2),
                splash_font, 0.7, (200, 200, 200), 2)
    cv2.imshow("Face Recognition", splash)
    cv2.waitKey(1)

    print("[Main] Starting webcam. Press 'q' or ESC to quit. Press 'd' to toggle debug overlay.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: Could not open webcam.")
        sys.exit(1)

    # ── Start camera reader thread (decouples capture from processing) ────────
    cam_thread = threading.Thread(
        target=_camera_reader, args=(cap,), daemon=True, name="CameraReader"
    )
    cam_thread.start()
    _camera_ok.wait(timeout=5.0)  # Wait for first frame

    # Per-track frame counter for INFERENCE_SKIP throttling
    # Maps track_id → monotonic tick of last worker submission
    last_submitted: Dict[int, int] = {}
    tick = 0                       # monotonic counter — NEVER reset (used for throttle)
    fps_frames = 0                 # separate counter for FPS display — reset each second
    prev_tracks: List = []         # cached tracks for detection-skip frames

    fps_time  = time.time()
    fps_disp  = 0.0
    debug_on  = False          # Toggle with 'd' key at runtime
    status_msg: Optional[str] = f"Ready - {_known_count} known faces loaded"
    ready_time = time.time()

    # ── Initialize speech transcriber (optional — graceful if deps missing) ──
    global _transcriber
    _transcriber = SpeechTranscriber()
    if not _transcriber.available:
        print("[Main] Speech transcription unavailable (install vosk + sounddevice).")
    else:
        print("[Main] Speech ready. Auto-records during ACTIVE identity sessions.")
        # Wire finalize callback: when vosk finalizes a phrase, store it tied to
        # the active session. SessionManager.add_event is atomic under its own
        # lock, so a concurrent identity switch cannot misattribute.
        def _on_speech_finalized(text: str) -> None:
            if _session_mgr:
                _session_mgr.add_event(
                    event_type="transcript",
                    content=text,
                    source="speech",
                    confidence=_latest_primary_score,
                )
        _transcriber.set_on_finalize(_on_speech_finalized)

    # Track session state transitions so we can auto-start/stop the transcriber
    prev_session_state: SessionState = SessionState.IDLE

    while True:
        # Read latest frame from camera thread (non-blocking)
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is None:
            if not _camera_ok.is_set():
                print("[Main] Lost webcam feed — exiting.")
                break
            continue

        tick += 1
        fps_frames += 1

        # ── Stage 1: Detection ────────────────────────────────────────────────
        # Run MTCNN every DETECT_SKIP frames; reuse previous tracks otherwise.
        # MTCNN costs 40-80ms — skipping halves main-thread load.
        run_detection = (tick % DETECT_SKIP == 1) or tick <= 1
        h_full, w_full = frame.shape[:2]

        if run_detection:
            scale     = DETECT_DOWNSAMPLE
            small_rgb = cv2.cvtColor(
                cv2.resize(frame, (w_full // scale, h_full // scale)),
                cv2.COLOR_BGR2RGB,
            )
            try:
                raw_detections = detector.detect_faces(small_rgb)
            except ValueError:
                raw_detections = []

            detections: List = []
            for r in raw_detections:
                if r['confidence'] < MTCNN_CONF_MIN:
                    continue
                x, y, w, h = [v * scale for v in r['box']]
                x = max(0, x);  y = max(0, y)
                w = min(w, w_full - x);  h = min(h, h_full - y)
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    continue
                kpts = {
                    k: (int(v[0] * scale), int(v[1] * scale))
                    for k, v in r.get('keypoints', {}).items()
                }
                detections.append(((x, y, w, h), kpts, r['confidence']))

            # ── Stage 2: Tracking ─────────────────────────────────────────────
            active_tracks = tracker.update(detections)
            prev_tracks = active_tracks  # cache for skip frames
        else:
            # Skip frame: reuse previous tracks (don't call tracker.update
            # with empty list — that would age tracks and return nothing)
            active_tracks = prev_tracks

        # Remove display state for tracks that are no longer active so stale
        # labels don't linger after a face leaves the frame.
        active_ids = {t.track_id for t in active_tracks}
        now = time.time()

        # Clean up last_submitted for dead tracks (prevents unbounded memory growth)
        dead_subs = [tid for tid in last_submitted if tid not in active_ids]
        for tid in dead_subs:
            del last_submitted[tid]

        with _state_lock:
            # Clear states for tracks the IoU tracker has dropped
            stale_ids = [tid for tid in _display_states if tid not in active_ids]
            for tid in stale_ids:
                del _display_states[tid]

            # Expire stale states to prevent permanent freezes:
            # - QUALITY_FAIL / UNCERTAIN: short timeout (re-assess on next submission)
            # - QUEUED / pipeline stages: medium timeout (worker may have crashed)
            # - Terminal states (MATCHED, REFINED): never expire (identity is locked)
            _terminal = ("MATCHED", "REFINED")
            _short_expire = ("QUALITY_FAIL", "UNCERTAIN")
            expired = [
                tid for tid, s in _display_states.items()
                if s.decision not in _terminal
                and (
                    (s.decision in _short_expire and (now - s.last_updated) > STATE_STALE_TIMEOUT)
                    or (s.decision not in _short_expire and (now - s.last_updated) > PENDING_STATE_TIMEOUT)
                )
            ]
            for tid in expired:
                del _display_states[tid]

        # ── Stage 3 + handoff: Align and send to worker (throttled) ──────────
        # Skip submission if worker is dead (avoids filling a queue nobody reads)
        worker_ok = _worker_alive.is_set()
        if not worker_ok:
            status_msg = "Recognition offline"

        for track in active_tracks:
            tid = track.track_id

            if not worker_ok:
                continue

            # Throttle: don't send every single frame to the worker
            last = last_submitted.get(tid, -(INFERENCE_SKIP + 1))
            if (tick - last) < INFERENCE_SKIP:
                continue
            last_submitted[tid] = tick

            # Don't queue if worker is already saturated
            if _task_queue.full():
                continue

            # Stage 3: Face alignment (affine via landmarks)
            aligned = _ACTIVE_ALIGN(frame, track.bbox, track.landmarks)
            if aligned is None or aligned.size == 0:
                continue

            # Determine priority: 0=new (fastest), 1=refining, 2=matched (lowest)
            with _state_lock:
                ds = _display_states.get(tid)
            if ds is not None and ds.decision in ("MATCHED", "REFINED"):
                priority = 2
            elif ds is not None and ds.decision == "REFINING":
                priority = 1
            else:
                priority = 0

            try:
                _task_counter_val = tick * 1000 + tid  # monotonic tiebreaker
                _task_queue.put_nowait((priority, _task_counter_val, aligned, tid, track.landmarks))
                # Immediately mark as QUEUED so UI transitions WHITE→ORANGE
                with _state_lock:
                    if tid not in _display_states:
                        _display_states[tid] = DisplayState(
                            identity=None, score=0.0, frame_count=0,
                            decision="QUEUED", quality=0.0,
                            last_updated=time.time(),
                        )
            except queue.Full:
                pass   # Drop this frame silently — worker will catch up

        # ── Update status message ─────────────────────────────────────────────
        # Only auto-clear the initial "Ready" message, not error messages
        if status_msg and worker_ok and (time.time() - ready_time) > 3.0:
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

        # FPS counter (uses its own resettable counter, not the throttle tick)
        now     = time.time()
        elapsed = now - fps_time
        if elapsed >= 1.0:
            fps_disp = fps_frames / elapsed
            fps_time = now
            fps_frames = 0

        # Compute face count breakdown for status bar
        visible_count = sum(1 for t in active_tracks if t.frames_seen >= 3)
        known_count = sum(
            1 for t in active_tracks if t.frames_seen >= 3
            and current_states.get(t.track_id) is not None
            and current_states[t.track_id].decision in ("MATCHED", "REFINED")
        )
        new_count = sum(
            1 for t in active_tracks if t.frames_seen >= 3
            and current_states.get(t.track_id) is not None
            and current_states[t.track_id].decision in ("FAST_ENROLLED", "REFINING")
        )

        # Status bar (replaces old FPS text overlay)
        _draw_status_bar(
            frame,
            num_faces  = visible_count,
            num_known  = known_count,
            num_new    = new_count,
            queue_size = _task_queue.qsize(),
            status_msg = status_msg,
            debug_on   = debug_on,
            fps        = fps_disp,
        )

        # ── Session tick (confidence-gated identity sessions) ──────────────
        primary_tid = _select_primary_face(active_tracks, current_states)
        primary_state = current_states.get(primary_tid) if primary_tid is not None else None

        # Publish current confidence for the speech callback (read-only cross-thread).
        global _latest_primary_score
        _latest_primary_score = primary_state.score if primary_state else 0.0

        if _session_mgr:
            _session_mgr.tick(
                identity=primary_state.identity if primary_state else None,
                decision=primary_state.decision if primary_state else None,
                score=primary_state.score if primary_state else 0.0,
            )

            # Auto-drive the transcriber on session state transitions.
            # ACTIVE entry → start listening; any exit from ACTIVE → stop.
            # Stop runs in a daemon thread so the 0.2s audio-thread join never
            # stalls the camera loop.
            curr_state = _session_mgr.state
            if _transcriber and _transcriber.available:
                if curr_state == SessionState.ACTIVE and prev_session_state != SessionState.ACTIVE:
                    _transcriber.start()
                elif curr_state != SessionState.ACTIVE and prev_session_state == SessionState.ACTIVE:
                    threading.Thread(
                        target=lambda: (_transcriber.stop(), _transcriber.clear()),
                        daemon=True,
                    ).start()
            prev_session_state = curr_state

        # ── Info panel (right 2/5 of screen) ─────────────────────────────────
        cam_h, cam_w = frame.shape[:2]
        panel_w = int(cam_w * 2 / 3)
        panel = np.zeros((cam_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        db_info = None
        if primary_state and primary_state.identity and _face_db:
            db_info = _face_db.get_identity(primary_state.identity)

        session_info = None
        recent_events = []
        if _session_mgr:
            sess = _session_mgr.current_session
            if sess:
                session_info = {
                    "state": _session_mgr.state.name,
                    "identity": sess.identity_name,
                    "session_id": sess.session_id,
                    "event_count": sess.event_count,
                }
            recent_events = _session_mgr.get_recent_events(3)

        # Get transcript state (non-blocking snapshot)
        transcript_snap = _transcriber.get_state() if _transcriber else None

        _draw_info_panel(
            panel,
            state            = primary_state,
            db_info          = db_info,
            total_faces      = visible_count,
            all_states       = current_states,
            session_info     = session_info,
            recent_events    = recent_events,
            transcript_state = transcript_snap,
        )

        canvas = np.hstack([frame, panel])
        cv2.imshow("Face Recognition", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # 'q' or ESC
            break
        elif key == ord('d'):       # Toggle debug overlay
            debug_on = not debug_on

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[Main] Shutting down...")
    if _transcriber:
        _transcriber.stop()
    _stop_flag.set()
    cam_thread.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    worker.join(timeout=3.0)
    print("[Main] Done.")


if __name__ == "__main__":
    main()
