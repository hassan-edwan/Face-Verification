"""
Web UI server for the face recognition pipeline.

Runs the same 7-stage pipeline as scripts/live_webcam.py (detection → tracking →
alignment → quality → embedding → gatekeeper → persistence) in background
threads, and exposes it over HTTP:

    GET  /                   → dashboard HTML
    GET  /api/video_feed     → MJPEG stream of the rendered camera frame
    GET  /api/state          → JSON snapshot of current recognition state
    GET  /api/thumbnail/<id> → JPEG thumbnail for a known identity

The frontend polls /api/state at ~5 Hz and renders the right-side info panel
itself. The video stream only carries per-face bounding-box overlays.

Run:
    python scripts/server.py           # default: http://127.0.0.1:5000
    python scripts/server.py --port 8000

Thread layout:
    CameraReader         — pulls frames from OpenCV VideoCapture
    RecognitionWorker    — FaceNet embeddings + Gatekeeper decisions
    PipelineProcessor    — MTCNN detection, IoU tracking, submission, rendering
    Flask (main thread)  — HTTP routes; one generator per MJPEG client
"""

import os
import sys
import cv2
import glob
import time
import json
import queue
import shutil
import argparse
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, Response, jsonify, send_from_directory, send_file, abort

from mtcnn import MTCNN

from src.quality    import assess_quality, quick_quality_check
from src.tracker    import FaceTracker
from src.gatekeeper import Gatekeeper, GatekeeperDecision, CONSENSUS_FRAMES
from src.database   import FaceDatabase
from src.memory     import MemoryStore, SessionManager, SessionState
from src.speech     import SpeechTranscriber
from src.alignment  import align_face, align_face_5point, TARGET_SIZE
from src.embedder   import Embedder, get_embedder

# ── Pipeline constants (mirrors live_webcam.py) ──────────────────────────────
# TARGET_SIZE is imported from src.alignment so the live path, the
# alignment helper, and the real-data harness agree on the canonical size.
MIN_FACE_PX       = 20          # run_014: 40→20 under ArcFace (detection + embedding hold at sub-30 px)
MTCNN_CONF_MIN    = 0.90
DETECT_DOWNSAMPLE = 2
# Target detection rate (frames per second) — decoupled from the render FPS.
# 4 FPS is plenty for tracking stability while letting the render thread hit
# 30+ FPS on its own. Override via DETECTION_TARGET_FPS env var.
DETECTION_TARGET_FPS = float(os.environ.get("DETECTION_TARGET_FPS", "4"))
_DETECTION_MIN_PERIOD = 1.0 / max(1e-3, DETECTION_TARGET_FPS)
INFERENCE_SKIP    = 2
WORKER_QUEUE_SIZE = 6

# Pick alignment geometry from the embedder's declared input_size so the
# live pipeline's crops match what ArcFace / FaceNet actually trained on.
# Resolved once at import — FACE_EMBEDDER doesn't change mid-process, and
# get_embedder() only instantiates the wrapper class here; the ONNX /
# keras weight load still happens lazily on the first embed() call.
_ACTIVE_ALIGN = (align_face_5point
                 if get_embedder().input_size == (112, 112)
                 else align_face)

ENROLL_BASE       = os.path.join(PROJECT_ROOT, "data", "enrollments")
STATE_STALE_TIMEOUT   = 4.0
PENDING_STATE_TIMEOUT = 8.0
JPEG_QUALITY      = 80

# Colors (BGR for OpenCV overlays on the streamed frame)
_COLOR_SCANNING    = (200, 200, 200)
_COLOR_IDENTIFYING = (0,  140, 255)
_COLOR_RECOGNIZED  = (40, 210,  40)
_COLOR_LEARNING    = (0,  210, 230)
_COLOR_ADJUST      = (0,  190, 230)
_COLOR_QUALITY     = (80,  80, 180)
_COLOR_FAST        = (0,  200, 100)


# ═══════════════════════════════════════════════════════════════════════════════
#  PURE HELPERS — disk I/O (alignment lives in src/alignment.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_persistent_faces(embedder: Embedder) -> Tuple[Dict, int]:
    known_faces: Dict[str, List] = {}
    person_counter = 1

    if not os.path.exists(ENROLL_BASE):
        os.makedirs(ENROLL_BASE, exist_ok=True)
        return known_faces, person_counter

    print("[Worker] Scanning data/enrollments/ for previously saved faces...")
    dirs = [d for d in os.listdir(ENROLL_BASE)
            if os.path.isdir(os.path.join(ENROLL_BASE, d))]

    for d in dirs:
        if not d.startswith("Person "):
            continue
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
                # Embedder handles resize to its own input_size; enrolled
                # thumbnails are saved pre-aligned already.
                emb = embedder.embed(img)
                known_faces[d].append(emb)

    print(f"[Worker] Loaded {len(known_faces)} known identity/ies from disk.")
    return known_faces, person_counter


def save_person_image(person_name: str, face_bgr: np.ndarray) -> None:
    person_dir = os.path.join(ENROLL_BASE, person_name)
    os.makedirs(person_dir, exist_ok=True)
    existing = glob.glob(os.path.join(person_dir, "*.jpg"))
    next_idx = len(existing) + 1
    stem     = person_name.replace(" ", "_")
    filepath = os.path.join(person_dir, f"{stem}_{str(next_idx).zfill(4)}.jpg")
    cv2.imwrite(filepath, face_bgr)
    print(f"[Persistent Storage] Saved → {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisplayState:
    identity:       Optional[str]
    score:          float
    frame_count:    int
    decision:       str
    quality:        float
    last_updated:   float = 0.0
    quality_reason: str   = ""
    is_reentry:     bool  = False
    pipeline_stage: str   = ""
    bbox:           Optional[Tuple[int, int, int, int]] = None


_display_states: Dict[int, DisplayState] = {}
_state_lock     = threading.Lock()
_task_queue     = queue.PriorityQueue(maxsize=WORKER_QUEUE_SIZE)
_stop_flag      = threading.Event()
_worker_ready   = threading.Event()
_worker_alive   = threading.Event()
_known_count    = 0
_face_db: Optional[FaceDatabase]       = None
_session_mgr: Optional[SessionManager] = None
_transcriber: Optional[SpeechTranscriber] = None
_latest_primary_score: float = 0.0

# Gatekeeper is owned by the recognition worker but published here so the
# dashboard CRUD handlers can mutate identity state safely. Every read or
# write of the gatekeeper's internal dicts MUST hold _identity_mutation_lock
# — the only mutators outside the worker are PATCH/DELETE on /api/identities.
_gatekeeper: Optional[Gatekeeper] = None
_identity_mutation_lock = threading.RLock()

# Pipeline lifecycle controls.
#   _pipeline_active = SET   → camera + ML loops run normally
#   _pipeline_active = CLEAR → camera released, workers parked, no ML inference
# Toggled by /api/pipeline/{pause,resume}. The dashboard pauses on entry so
# CPU/GPU drops to ~0; the live page resumes on entry. Idempotent on both
# sides — flipping the event multiple times is safe.
_pipeline_active = threading.Event()
_pipeline_active.set()                  # default: running
_camera_index: int = 0                  # remembered across pause/resume cycles

# Camera capture — producer owned by CameraReader, consumed by
# RenderProducer (every frame) and DetectionWorker (throttled ~4 FPS).
_latest_frame: Optional[np.ndarray] = None
_latest_frame_seq: int = 0              # monotonic, bumped on every new frame
_frame_lock   = threading.Lock()
_camera_ok    = threading.Event()

# Detection output — written by DetectionWorker, read by RenderProducer.
# A brief lock wraps both lists atomically so the render thread never sees
# a torn snapshot (e.g. tracks referring to stale bboxes).
_active_tracks: List = []
_last_detection_seq: int = -1           # frame seq MTCNN last ran on
_last_detection_wall: float = 0.0       # wall time detection last completed
_detection_fps: float = 0.0
_tracks_lock  = threading.Lock()

# Rendered JPEG ready to serve via MJPEG
_latest_jpeg:     Optional[bytes] = None
_jpeg_lock        = threading.Lock()
_jpeg_available   = threading.Condition(_jpeg_lock)

# MJPEG client counter. Render loop skips cv2.imencode when this is zero —
# there's no point encoding JPEGs nobody will consume. Updated under
# _jpeg_lock because the render loop reads it right next to the encode call.
_mjpeg_clients    = 0

# Cached JSON snapshot (updated each render tick; read by /api/state polls).
_current_snapshot: dict = {"system": {"worker_ready": False}, "primary_face": None}
_snapshot_lock = threading.Lock()

# Live FPS counters, for observability in /api/state.system.
_render_fps: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA READER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def _camera_reader() -> None:
    """Tight loop: grab a frame, publish, repeat. Must never block on
    downstream consumers — overwriting _latest_frame is the drop-oldest
    semantics consumers rely on. Bumps a monotonic sequence so the detection
    worker can notice when no new frame has arrived and skip its tick.

    Owns the cv2.VideoCapture device so the dashboard pause flow can have
    the device released entirely — not just idled — while the user is on a
    page that doesn't need the camera. On resume, the device is re-opened
    against the same _camera_index. A single thread instance survives
    arbitrarily many pause/resume cycles, so there are never duplicate
    OpenCV grab loops competing for the camera.
    """
    global _latest_frame, _latest_frame_seq
    cap: Optional[cv2.VideoCapture] = None
    try:
        while not _stop_flag.is_set():
            # ── Pause gate ─────────────────────────────────────────────────
            if not _pipeline_active.is_set():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception as exc:
                        print(f"[Camera] release on pause: {exc}")
                    cap = None
                    _camera_ok.clear()
                    print("[Camera] Released device (pipeline paused).")
                # Wake-up window short enough to react to stop_flag too.
                _pipeline_active.wait(timeout=0.5)
                continue

            # ── Open lazily ───────────────────────────────────────────────
            if cap is None:
                cap = cv2.VideoCapture(_camera_index)
                if not cap.isOpened():
                    print(f"[Camera] Could not open index {_camera_index}; retrying in 1s.")
                    cap = None
                    # Don't busy-spin if the device is unavailable.
                    if _stop_flag.wait(timeout=1.0):
                        break
                    continue
                print(f"[Camera] Opened index {_camera_index}.")

            # ── Frame grab ────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                _camera_ok.clear()
                print("[Camera] read() failed — releasing and retrying.")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                if _stop_flag.wait(timeout=0.5):
                    break
                continue

            with _frame_lock:
                _latest_frame = frame
                _latest_frame_seq += 1
            _camera_ok.set()
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        _camera_ok.clear()
        print("[Camera] Reader exiting.")


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOGNITION WORKER THREAD (stages 4–7)
# ═══════════════════════════════════════════════════════════════════════════════

def recognition_worker() -> None:
    try:
        embedder = get_embedder()
        print(f"[Worker] Loading {embedder.name} embedder (input {embedder.input_size})...")
        # Triggers lazy model-weight download/load on first embed call.

        known_faces, person_counter = load_persistent_faces(embedder)
        gatekeeper = Gatekeeper(known_faces, person_counter)

        global _known_count, _face_db, _session_mgr, _gatekeeper
        # Publish the gatekeeper for the dashboard CRUD handlers. They must
        # only mutate it under _identity_mutation_lock — same lock the worker
        # acquires for every gatekeeper.process() call.
        _gatekeeper = gatekeeper
        db = FaceDatabase(os.path.join(PROJECT_ROOT, "data", "faces.db"))
        _face_db = db
        for name in known_faces:
            if db.get_identity(name) is None:
                person_dir = os.path.join(ENROLL_BASE, name)
                jpgs = sorted(glob.glob(os.path.join(person_dir, "*.jpg")))
                db.create_identity(name, jpgs[0] if jpgs else "")

        memory_store = MemoryStore(os.path.join(PROJECT_ROOT, "data", "memory.db"))
        _session_mgr = SessionManager(memory_store)

        _known_count = len(known_faces)
        _worker_ready.set()
        _worker_alive.set()
        print("[Worker] Ready. Listening for frames...")
    except Exception as exc:
        print(f"[Worker] FATAL: Failed to initialize — {exc}")
        _worker_ready.set()
        return

    while not _stop_flag.is_set():
        # Park while paused. Drain whatever's queued so the dashboard delete
        # cascade isn't waiting on a stale task to finish processing — and
        # so we don't run inference under a paused pipeline.
        if not _pipeline_active.is_set():
            while True:
                try:
                    _task_queue.get_nowait()
                    _task_queue.task_done()
                except queue.Empty:
                    break
            _pipeline_active.wait(timeout=0.5)
            continue

        try:
            task = _task_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        track_id = None
        try:
            _priority, _seq, aligned_bgr, track_id, landmarks = task

            with _state_lock:
                st = _display_states.get(track_id)
                if st is not None:
                    st.pipeline_stage = "quality"

            with _identity_mutation_lock:
                is_enrolled = gatekeeper.is_already_enrolled(track_id)
            if not is_enrolled:
                qr = assess_quality(aligned_bgr, landmarks)
                quality_score = qr.score
                if not qr.passed:
                    with _state_lock:
                        prev = _display_states.get(track_id)
                        _display_states[track_id] = DisplayState(
                            identity       = None,
                            score          = 0.0,
                            frame_count    = 0,
                            decision       = "QUALITY_FAIL",
                            quality        = quality_score,
                            last_updated   = time.time(),
                            quality_reason = qr.reason,
                            bbox           = prev.bbox if prev else None,
                        )
                    continue
            else:
                ok, quality_score = quick_quality_check(aligned_bgr)
                if not ok:
                    continue

            with _state_lock:
                st = _display_states.get(track_id)
                if st is not None:
                    st.pipeline_stage = "embedding"

            emb = embedder.embed(aligned_bgr)

            with _state_lock:
                st = _display_states.get(track_id)
                if st is not None:
                    st.pipeline_stage = "matching"
            # Lock is held for the gatekeeper call only — embedding inference
            # already happened above so the critical section stays sub-ms.
            with _identity_mutation_lock:
                result = gatekeeper.process(emb, track_id, aligned_bgr, quality_score)

            if result.decision in (GatekeeperDecision.FAST_ENROLLED,
                                   GatekeeperDecision.REFINED):
                img_to_save = (result.best_face_image
                               if result.best_face_image is not None else aligned_bgr)
                threading.Thread(target=save_person_image,
                                 args=(result.identity, img_to_save),
                                 daemon=True).start()

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

            is_reentry = False
            if result.identity and result.decision == GatekeeperDecision.MATCHED:
                with _identity_mutation_lock:
                    meta = gatekeeper._identity_metadata.get(result.identity, {})
                    tids = meta.get("tracker_ids", set())
                    if len(tids) > 1:
                        is_reentry = True

            with _state_lock:
                prev = _display_states.get(track_id)
                _display_states[track_id] = DisplayState(
                    identity       = result.identity,
                    score          = result.score,
                    frame_count    = result.frame_count,
                    decision       = result.decision.name,
                    quality        = quality_score,
                    last_updated   = time.time(),
                    is_reentry     = is_reentry,
                    bbox           = prev.bbox if prev else None,
                )

        except Exception as exc:
            import traceback
            print(f"[Worker] Unhandled error on track {track_id}: {exc}")
            traceback.print_exc()
        finally:
            _task_queue.task_done()

    _worker_alive.clear()
    print("[Worker] Thread exiting.")


# ═══════════════════════════════════════════════════════════════════════════════
#  UI LABEL / OVERLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _quality_reason_to_label(reason: str) -> str:
    r = reason.lower()
    if "blurry" in r or "laplacian" in r:
        return "Hold still"
    if "overexposed" in r:
        return "Too bright"
    if "dark" in r or "underexposed" in r:
        return "More light needed"
    if "pose" in r or "yaw" in r or "pitch" in r or "roll" in r:
        return "Face the camera"
    return "Adjust position"


def _status_for_decision(decision: str) -> Tuple[str, str]:
    """Map a gatekeeper decision to (user-facing label, semantic status token)."""
    return {
        "MATCHED":       ("Recognized",    "recognized"),
        "REFINED":       ("Verified",      "recognized"),
        "FAST_ENROLLED": ("New face",      "new"),
        "REFINING":      ("Learning",      "learning"),
        "UNCERTAIN":     ("Uncertain",     "uncertain"),
        "QUALITY_FAIL":  ("Quality issue", "quality"),
        "QUEUED":        ("Analyzing",     "analyzing"),
    }.get(decision, ("Processing",        "analyzing"))


def _color_for_decision(decision: str) -> Tuple[int, int, int]:
    return {
        "MATCHED":       _COLOR_RECOGNIZED,
        "REFINED":       _COLOR_RECOGNIZED,
        "FAST_ENROLLED": _COLOR_FAST,
        "REFINING":      _COLOR_LEARNING,
        "UNCERTAIN":     _COLOR_ADJUST,
        "QUALITY_FAIL":  _COLOR_QUALITY,
    }.get(decision, _COLOR_IDENTIFYING)


def _draw_overlays(frame: np.ndarray, tracks: List, states: Dict[int, DisplayState]) -> None:
    """Draw bounding boxes + short labels on the camera frame only.

    Rich info (confidence, transcripts, metadata) lives in the HTML panel —
    the video stream is kept visually lean so it stays jitter-free.
    """
    for t in tracks:
        if t.frames_seen < 3:
            continue
        x, y, w, h = t.bbox
        st = states.get(t.track_id)

        if st is None:
            color, label = _COLOR_SCANNING, "Scanning..."
            thickness = 1
        else:
            color = _color_for_decision(st.decision)
            thickness = 2
            if st.decision in ("MATCHED", "REFINED"):
                label = st.identity or "Unknown"
            elif st.decision == "FAST_ENROLLED":
                label = f"New — {st.identity}"
            elif st.decision == "REFINING":
                label = f"Learning {st.frame_count}/{CONSENSUS_FRAMES}"
            elif st.decision == "QUALITY_FAIL":
                label = _quality_reason_to_label(st.quality_reason)
            elif st.decision == "UNCERTAIN":
                label = "Turn slightly"
            else:
                label = "Analyzing..."

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ly = max(y, th + 6)
        cv2.rectangle(frame, (x, ly - th - 6), (x + tw + 8, ly), color, -1)
        cv2.putText(frame, label, (x + 4, ly - 3), font, 0.5, (0, 0, 0), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIMARY-FACE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _select_primary(tracks: List, states: Dict[int, DisplayState]) -> Optional[int]:
    visible = [t for t in tracks if t.frames_seen >= 3]
    if not visible:
        return None
    recognized = [
        t for t in visible
        if states.get(t.track_id) is not None
        and states[t.track_id].decision in ("MATCHED", "REFINED")
    ]
    pool = recognized if recognized else visible
    best = max(pool, key=lambda t: t.bbox[2] * t.bbox[3])
    return best.track_id


def _format_relative(ts: float) -> str:
    if not ts:
        return ""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT BUILDER — produces the JSON payload consumed by the frontend
# ═══════════════════════════════════════════════════════════════════════════════

def _build_snapshot(tracks: List, states: Dict[int, DisplayState],
                    primary_tid: Optional[int], fps: float, status_msg: Optional[str]) -> dict:
    visible = [t for t in tracks if t.frames_seen >= 3]
    recognized = sum(1 for t in visible
                     if states.get(t.track_id) is not None
                     and states[t.track_id].decision in ("MATCHED", "REFINED"))
    learning = sum(1 for t in visible
                   if states.get(t.track_id) is not None
                   and states[t.track_id].decision in ("FAST_ENROLLED", "REFINING"))

    primary_state = states.get(primary_tid) if primary_tid is not None else None
    primary_payload = None
    db_payload = None

    if primary_state is not None:
        label, status_token = _status_for_decision(primary_state.decision)
        identity = primary_state.identity
        display_name = identity or ""
        db_info = _face_db.get_identity(identity) if (identity and _face_db) else None
        if db_info:
            display_name = db_info.get("display_name") or identity
            db_payload = {
                "enrolled_at":      db_info.get("created_at", 0),
                "enrolled_label":   _format_relative(db_info.get("created_at", 0)),
                "last_seen_at":     db_info.get("last_seen_at", 0),
                "last_seen_label":  _format_relative(db_info.get("last_seen_at", 0)),
                "total_matches":    db_info.get("total_matches", 0),
                "notes":            db_info.get("notes", ""),
            }

        quality_label = _quality_reason_to_label(primary_state.quality_reason) \
            if primary_state.quality_reason else ""

        primary_payload = {
            "track_id":          primary_tid,
            "identity":          identity,
            "display_name":      display_name or "Unknown",
            "decision":          primary_state.decision,
            "status_label":      label,
            "status_token":      status_token,
            "confidence":        float(primary_state.score),
            "quality":           float(primary_state.quality),
            "frame_count":       primary_state.frame_count,
            "consensus_target":  CONSENSUS_FRAMES,
            "quality_reason":    quality_label,
            "is_reentry":        primary_state.is_reentry,
            "pipeline_stage":    primary_state.pipeline_stage,
            "has_thumbnail":     bool(db_info and db_info.get("best_thumbnail_path")
                                      and os.path.exists(db_info.get("best_thumbnail_path") or "")),
        }

    # Session + recent events
    session_payload = None
    recent_payload: List[dict] = []
    if _session_mgr:
        sess = _session_mgr.current_session
        if sess:
            session_payload = {
                "state":       _session_mgr.state.name,
                "identity":    sess.identity_name,
                "session_id":  sess.session_id,
                "event_count": sess.event_count,
                "started_at":  sess.started_at,
                "started_label": _format_relative(sess.started_at),
            }
        for evt in _session_mgr.get_recent_events(5):
            recent_payload.append({
                "event_type": evt.get("event_type", ""),
                "content":    evt.get("content", ""),
                "source":     evt.get("source", ""),
                "created_at": evt.get("created_at", 0),
                "age_label":  _format_relative(evt.get("created_at", 0)),
            })

    # Transcript
    transcript_payload = {
        "available":    bool(_transcriber and _transcriber.available),
        "is_listening": False,
        "partial":      "",
        "lines":        [],
        "error":        "",
    }
    if _transcriber:
        ts = _transcriber.get_state()
        transcript_payload.update({
            "is_listening": bool(ts.is_listening),
            "partial":      ts.partial or "",
            "lines":        list(ts.lines[-10:]),
            "error":        ts.error or "",
        })

    faces_payload = {
        "total":      len(visible),
        "recognized": recognized,
        "new":        learning,
        "tracks":     [
            {
                "track_id": t.track_id,
                "bbox":     list(t.bbox),
                "decision": (states.get(t.track_id).decision
                             if states.get(t.track_id) else "SCANNING"),
                "identity": (states.get(t.track_id).identity
                             if states.get(t.track_id) else None),
            }
            for t in visible
        ],
    }

    return {
        "timestamp": time.time(),
        "system": {
            "worker_ready":   _worker_ready.is_set(),
            "worker_alive":   _worker_alive.is_set(),
            "camera_ok":      _camera_ok.is_set(),
            "known_count":    _known_count,
            "queue_size":     _task_queue.qsize(),
            "queue_max":      WORKER_QUEUE_SIZE,
            "fps":            round(fps, 1),
            "detection_fps":  round(_detection_fps, 1),
            "status_message": status_msg or "",
        },
        "primary_face":  primary_payload,
        "database":      db_payload,
        "session":       session_payload,
        "recent_events": recent_payload,
        "transcript":    transcript_payload,
        "faces":         faces_payload,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION WORKER — MTCNN + IoU tracker + FaceNet-worker handoff
#
#  Runs at DETECTION_TARGET_FPS (default 4 Hz). Completely decoupled from the
#  render thread: it writes `_active_tracks` under a brief lock and submits
#  aligned crops to the embedding worker. The render thread never waits on it.
# ═══════════════════════════════════════════════════════════════════════════════

def detection_worker() -> None:
    """Grabs the latest camera frame on its own schedule, runs MTCNN +
    IoU tracking, and publishes tracks to shared state. Never touches the
    JPEG/MJPEG path."""
    global _active_tracks, _last_detection_seq, _last_detection_wall, _detection_fps

    print("[Detection] Loading MTCNN detector...")
    detector = MTCNN()
    tracker  = FaceTracker()
    print(f"[Detection] Ready. Target rate = {DETECTION_TARGET_FPS:.1f} FPS.")

    last_submitted: Dict[int, int] = {}
    local_tick = 0
    last_run_wall = 0.0
    fps_frames = 0
    fps_time = time.time()
    last_seen_seq = -1

    while not _stop_flag.is_set():
        # Park while paused — no MTCNN inference, no shared-state churn.
        if not _pipeline_active.is_set():
            _pipeline_active.wait(timeout=0.5)
            # Reset cadence trackers so we don't burst on resume.
            last_run_wall = 0.0
            last_seen_seq = -1
            continue

        now = time.time()

        # Cadence gate: run at most DETECTION_TARGET_FPS.
        elapsed = now - last_run_wall
        if elapsed < _DETECTION_MIN_PERIOD:
            time.sleep(_DETECTION_MIN_PERIOD - elapsed)
            continue

        # Grab latest frame (non-copy ref is fine — camera won't mutate it in place;
        # it swaps the _latest_frame attribute on every new grab).
        with _frame_lock:
            frame = _latest_frame
            frame_seq = _latest_frame_seq
        if frame is None:
            time.sleep(0.02)
            continue
        # Skip if no new frame has arrived since last detection — wasted work.
        if frame_seq == last_seen_seq:
            time.sleep(0.02)
            continue
        last_seen_seq = frame_seq
        last_run_wall = now
        local_tick += 1
        fps_frames += 1

        h_full, w_full = frame.shape[:2]
        scale = DETECT_DOWNSAMPLE
        small_rgb = cv2.cvtColor(
            cv2.resize(frame, (w_full // scale, h_full // scale)),
            cv2.COLOR_BGR2RGB,
        )
        try:
            raw = detector.detect_faces(small_rgb)
        except ValueError:
            raw = []
        except Exception as exc:
            print(f"[Detection] MTCNN error: {exc}")
            continue

        detections: List = []
        for r in raw:
            if r['confidence'] < MTCNN_CONF_MIN:
                continue
            x, y, w, h = [v * scale for v in r['box']]
            x = max(0, x); y = max(0, y)
            w = min(w, w_full - x); h = min(h, h_full - y)
            if w < MIN_FACE_PX or h < MIN_FACE_PX:
                continue
            kpts = {k: (int(v[0] * scale), int(v[1] * scale))
                    for k, v in r.get('keypoints', {}).items()}
            detections.append(((x, y, w, h), kpts, r['confidence']))

        active_tracks = tracker.update(detections)
        active_ids = {t.track_id for t in active_tracks}

        # Publish tracks for the render thread.
        with _tracks_lock:
            _active_tracks = list(active_tracks)
            _last_detection_seq = frame_seq
            _last_detection_wall = now

        # Prune display-state entries for disappeared tracks. Expiration of
        # stale non-terminal states is kept here (not the render thread) so
        # the render thread stays cheap.
        dead = [tid for tid in last_submitted if tid not in active_ids]
        for tid in dead:
            del last_submitted[tid]

        with _state_lock:
            stale = [tid for tid in _display_states if tid not in active_ids]
            for tid in stale:
                del _display_states[tid]
            for t in active_tracks:
                st = _display_states.get(t.track_id)
                if st is not None:
                    st.bbox = tuple(t.bbox)
            _terminal     = ("MATCHED", "REFINED")
            _short_expire = ("QUALITY_FAIL", "UNCERTAIN")
            expired = [
                tid for tid, s in _display_states.items()
                if s.decision not in _terminal and (
                    (s.decision in _short_expire and (now - s.last_updated) > STATE_STALE_TIMEOUT)
                    or (s.decision not in _short_expire and (now - s.last_updated) > PENDING_STATE_TIMEOUT)
                )
            ]
            for tid in expired:
                del _display_states[tid]

        # Submit aligned crops to FaceNet worker (throttled per-track).
        worker_ok = _worker_alive.is_set()
        for track in active_tracks:
            tid = track.track_id
            if not worker_ok:
                continue
            last = last_submitted.get(tid, -(INFERENCE_SKIP + 1))
            if (local_tick - last) < INFERENCE_SKIP:
                continue
            last_submitted[tid] = local_tick
            if _task_queue.full():
                continue
            aligned = _ACTIVE_ALIGN(frame, track.bbox, track.landmarks)
            if aligned is None or aligned.size == 0:
                continue

            with _state_lock:
                ds = _display_states.get(tid)
            if ds is not None and ds.decision in ("MATCHED", "REFINED"):
                priority = 2
            elif ds is not None and ds.decision == "REFINING":
                priority = 1
            else:
                priority = 0

            try:
                _task_queue.put_nowait(
                    (priority, local_tick * 1000 + tid, aligned, tid, track.landmarks)
                )
                with _state_lock:
                    if tid not in _display_states:
                        _display_states[tid] = DisplayState(
                            identity=None, score=0.0, frame_count=0,
                            decision="QUEUED", quality=0.0,
                            last_updated=time.time(), bbox=tuple(track.bbox),
                        )
            except queue.Full:
                pass

        # FPS meter (detection-side)
        if now - fps_time >= 1.0:
            _detection_fps = fps_frames / (now - fps_time)
            fps_frames = 0
            fps_time = now

    print("[Detection] Exiting.")


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER PRODUCER — pure producer: grab frame, draw overlays, encode, publish
#
#  Zero ML in this loop. Reads published `_active_tracks` + `_display_states`
#  under brief locks and writes the JPEG + JSON snapshot. Can hit 30+ FPS
#  on modest hardware because MTCNN no longer blocks it.
# ═══════════════════════════════════════════════════════════════════════════════

def render_producer() -> None:
    """Pure producer loop — see module docstring for contract."""
    global _latest_jpeg, _current_snapshot, _latest_primary_score, _render_fps

    fps_frames = 0
    fps_time   = time.time()
    status_msg: Optional[str] = f"Ready — {_known_count} known face(s) loaded"
    ready_time = time.time()
    prev_session_state = SessionState.IDLE
    last_rendered_seq = -1

    # Snapshot rebuild cadence: the live dashboard polls /api/state at 5 Hz
    # (web/static/app.js POLL_INTERVAL_MS=200). Rebuilding at render rate
    # (~30 FPS) would waste 6× the work. Floor at 150 ms so every client
    # poll still sees fresh data, render keeps its full JPEG frame rate.
    snapshot_min_period = 0.15
    last_snapshot_wall = 0.0

    # Wire speech auto-drive once at startup (cheap closure; no-op if speech
    # is unavailable or already wired).
    if _transcriber and _transcriber.available and _session_mgr:
        def _on_finalized(text: str) -> None:
            if _session_mgr:
                _session_mgr.add_event(
                    event_type="transcript", content=text,
                    source="speech", confidence=_latest_primary_score,
                )
        _transcriber.set_on_finalize(_on_finalized)

    while not _stop_flag.is_set():
        # Park while paused — no JPEG encoding, no snapshot work.
        if not _pipeline_active.is_set():
            _pipeline_active.wait(timeout=0.5)
            last_rendered_seq = -1
            continue

        # Atomic reference grab — no copy, no lock contention with camera.
        with _frame_lock:
            src = _latest_frame
            src_seq = _latest_frame_seq
        if src is None:
            # When paused or starting up, the camera may legitimately have
            # no frame yet — wait quietly instead of treating it as a fault.
            if not _pipeline_active.is_set() or not _camera_ok.is_set():
                time.sleep(0.05)
                continue
            time.sleep(0.005)
            continue

        # Skip if nothing new has arrived (avoids re-encoding the same JPEG).
        if src_seq == last_rendered_seq:
            time.sleep(0.002)
            continue
        last_rendered_seq = src_seq

        # Copy once so overlay drawing doesn't mutate the producer's buffer.
        frame = src.copy()

        # Snapshot detection + recognition state with brief locks.
        with _tracks_lock:
            tracks_snap = list(_active_tracks)
            detection_age = time.time() - _last_detection_wall if _last_detection_wall else 999
        with _state_lock:
            states_snap = dict(_display_states)

        _draw_overlays(frame, tracks_snap, states_snap)

        # Primary-face selection + session tick (both cheap; kept here so
        # transcriber start/stop stays responsive to state changes).
        primary_tid = _select_primary(tracks_snap, states_snap)
        primary_state = states_snap.get(primary_tid) if primary_tid is not None else None
        _latest_primary_score = primary_state.score if primary_state else 0.0

        if _session_mgr:
            _session_mgr.tick(
                identity=primary_state.identity if primary_state else None,
                decision=primary_state.decision if primary_state else None,
                score=primary_state.score if primary_state else 0.0,
            )
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

        now = time.time()
        if status_msg and _worker_alive.is_set() and (now - ready_time) > 3.0:
            status_msg = None
        if not _worker_alive.is_set():
            status_msg = "Recognition offline"
        # Surface a gentle hint when detection has stalled (e.g. MTCNN hang).
        elif detection_age > 2.0:
            status_msg = f"Detection lagging ({detection_age:.0f}s)"

        # Skip JPEG encoding when nobody is consuming the MJPEG stream. On an
        # 8-core laptop, imencode + tobytes costs ~1-2 ms per frame at 640×480;
        # free CPU once the dashboard is foregrounded (it pauses the pipeline
        # entirely) and once no live viewer is connected.
        with _jpeg_lock:
            clients = _mjpeg_clients
        if clients > 0:
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with _jpeg_available:
                    _latest_jpeg = buf.tobytes()
                    _jpeg_available.notify_all()

        # Throttle snapshot rebuild to ~6.7 Hz; client polls at 5 Hz so this
        # still delivers fresh data every poll. Also force a rebuild on the
        # first render tick (last_snapshot_wall == 0) so the initial /api/state
        # returns populated data, not the cold bootstrap dict.
        if now - last_snapshot_wall >= snapshot_min_period:
            snap = _build_snapshot(tracks_snap, states_snap, primary_tid,
                                   _render_fps, status_msg)
            with _snapshot_lock:
                _current_snapshot = snap
            last_snapshot_wall = now

        # FPS meter
        fps_frames += 1
        if now - fps_time >= 1.0:
            _render_fps = fps_frames / (now - fps_time)
            fps_frames = 0
            fps_time = now

    print("[Render] Exiting.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

WEB_ROOT = os.path.join(PROJECT_ROOT, "web")
app = Flask(
    __name__,
    static_folder   = os.path.join(WEB_ROOT, "static"),
    static_url_path = "/static",
)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index():
    return send_from_directory(WEB_ROOT, "index.html")


@app.route("/api/state")
def api_state():
    with _snapshot_lock:
        snap = dict(_current_snapshot)
    return jsonify(snap)


@app.route("/api/video_feed")
def video_feed():
    """MJPEG stream. Each connected client gets its own generator; the generator
    waits on a condition variable so it emits exactly as fast as new frames
    become available (no busy-waiting, no stale frames).

    Each generator increments `_mjpeg_clients` for its lifetime so the render
    loop can skip `cv2.imencode` while no one is watching. The decrement
    lives in the generator's `finally`, which fires both on clean client
    disconnect and on server-side shutdown (via `_stop_flag`)."""
    def gen():
        global _mjpeg_clients
        boundary = b"--frame"
        last_id: Optional[int] = None
        with _jpeg_lock:
            _mjpeg_clients += 1
        try:
            while not _stop_flag.is_set():
                with _jpeg_available:
                    got = _jpeg_available.wait(timeout=1.0)
                    if not got:
                        continue
                    frame_bytes = _latest_jpeg
                    frame_id = id(frame_bytes)
                if frame_bytes is None or frame_id == last_id:
                    continue
                last_id = frame_id
                yield (boundary + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                       + frame_bytes + b"\r\n")
        finally:
            with _jpeg_lock:
                _mjpeg_clients = max(0, _mjpeg_clients - 1)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/thumbnail/<path:identity>")
def thumbnail(identity: str):
    if not _face_db:
        abort(404)
    info = _face_db.get_identity(identity)
    if not info:
        abort(404)
    path = info.get("best_thumbnail_path") or ""
    if not path or not os.path.exists(path):
        abort(404)
    resp = send_file(path, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-cache, max-age=5"
    return resp


@app.route("/api/health")
def health():
    return jsonify({
        "ok":           _worker_ready.is_set() and _worker_alive.is_set(),
        "worker_ready": _worker_ready.is_set(),
        "worker_alive": _worker_alive.is_set(),
        "camera_ok":    _camera_ok.is_set(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTITY MANAGEMENT DASHBOARD — routes
# ═══════════════════════════════════════════════════════════════════════════════

from flask import request as _flask_request   # aliased to avoid shadowing


@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(WEB_ROOT, "dashboard.html")


def _identity_to_dict(row: dict) -> dict:
    """Shape a FaceDatabase row for the dashboard API."""
    return {
        "name":          row.get("name", ""),
        "display_name":  row.get("display_name", ""),
        "notes":         row.get("notes", ""),
        "created_at":    row.get("created_at", 0),
        "last_seen_at":  row.get("last_seen_at", 0),
        "total_matches": row.get("total_matches", 0),
        "has_thumbnail": bool(row.get("best_thumbnail_path")
                              and os.path.exists(row.get("best_thumbnail_path") or "")),
    }


@app.route("/api/identities")
def api_identities_list():
    if not _face_db:
        return jsonify({"identities": []})
    rows = _face_db.list_all()
    return jsonify({"identities": [_identity_to_dict(r) for r in rows]})


@app.route("/api/identities/<path:identity>")
def api_identity_detail(identity: str):
    if not _face_db:
        abort(404)
    row = _face_db.get_identity(identity)
    if not row:
        abort(404)

    body = _identity_to_dict(row)

    # Attach session memory so the detail view can show transcripts.
    body["events"] = []
    if _session_mgr:
        events = (_session_mgr._memory  # type: ignore[attr-defined]
                  .get_recent_events(identity, limit=50))
        body["events"] = [
            {
                "event_type": e.get("event_type", ""),
                "content":    e.get("content", ""),
                "source":     e.get("source", ""),
                "created_at": e.get("created_at", 0),
                "confidence": e.get("confidence", 0),
            }
            for e in events
        ]

    return jsonify(body)


@app.route("/api/identities/<path:identity>", methods=["PATCH"])
def api_identity_update(identity: str):
    if not _face_db:
        abort(404)
    body = _flask_request.get_json(silent=True) or {}
    # Held under the same lock as DELETE + worker.process so an edit
    # cannot land between a delete cascade's existence-check and its
    # cache pop, and so the worker never reads a half-updated row.
    with _identity_mutation_lock:
        row = _face_db.get_identity(identity)
        if not row:
            abort(404)
        if "display_name" in body:
            _face_db.update_display_name(identity, str(body["display_name"])[:200])
        if "notes" in body:
            _face_db.update_notes(identity, str(body["notes"])[:4000])
        updated = _face_db.get_identity(identity)
    return jsonify(_identity_to_dict(updated)) if updated else ("", 204)


# ─────────────────────────────────────────────────────────────────────────────
#  CASCADE DELETE
# ─────────────────────────────────────────────────────────────────────────────

def _cascade_delete_identity(name: str) -> dict:
    """Atomic-ish cascading deletion of an identity from every store the
    pipeline owns. Held under _identity_mutation_lock for the entire span,
    so the recognition worker (which acquires the same lock around every
    gatekeeper call) can never observe a half-deleted person.

    Order matters: gatekeeper state first (so the worker can't re-attach a
    tracker to a soon-to-vanish identity), then DB row, then auxiliary
    stores, finally the on-disk crops. Disk failure is logged but doesn't
    roll back — the in-memory pipeline is already consistent at that point.
    """
    summary = {
        "ok": False, "name": name,
        "embeddings_removed": 0,
        "events_removed": 0,
        "files_removed": 0,
    }
    with _identity_mutation_lock:
        existed = False

        # 1) Gatekeeper in-memory state. Delegated to Gatekeeper.forget so
        #    the bank / recent-match cache invariants stay encapsulated —
        #    callers here shouldn't know the bank representation.
        if _gatekeeper is not None:
            removed = _gatekeeper.forget(name)
            if removed:
                existed = True
                summary["embeddings_removed"] = removed

        # 2) Display states — drop any track currently labelled with this
        #    identity so the live overlay doesn't briefly flash a phantom box.
        with _state_lock:
            for tid in [t for t, st in list(_display_states.items()) if st.identity == name]:
                _display_states.pop(tid, None)

        # 3) DB row.
        if _face_db is not None:
            if _face_db.delete_identity(name):
                existed = True

        if not existed:
            return summary  # ok stays False — caller returns 404

        # 4) Memory events (transcripts / system events).
        if _session_mgr is not None:
            try:
                summary["events_removed"] = _session_mgr._memory.delete_identity_events(name)  # type: ignore[attr-defined]
            except Exception as exc:
                print(f"[Delete] memory events for '{name}': {exc}")

        # 5) On-disk crops. Last so an OS-level failure (file lock, perms)
        #    can't leave the in-memory pipeline in a partially-deleted state.
        person_dir = os.path.join(ENROLL_BASE, name)
        if os.path.isdir(person_dir):
            try:
                summary["files_removed"] = sum(1 for _ in os.scandir(person_dir))
                shutil.rmtree(person_dir, ignore_errors=False)
            except OSError as exc:
                # Logged, not raised — the directory will be cleaned up the
                # next time the user manually re-runs cleanup, or by hand.
                print(f"[Delete] rmtree {person_dir}: {exc}")

        global _known_count
        _known_count = max(0, _known_count - 1)
        summary["ok"] = True
        print(f"[Delete] '{name}' purged: {summary['embeddings_removed']} embs, "
              f"{summary['events_removed']} events, {summary['files_removed']} files.")
        return summary


@app.route("/api/identities/<path:identity>", methods=["DELETE"])
def api_identity_delete(identity: str):
    if not _face_db:
        return jsonify({"ok": False, "error": "database unavailable"}), 503
    summary = _cascade_delete_identity(identity)
    if not summary.get("ok"):
        return jsonify({**summary, "error": "identity not found"}), 404
    return jsonify(summary)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE LIFECYCLE — pause/resume hooks for the dashboard
#
#  The dashboard route never needs the camera or ML — pausing on entry frees
#  the camera device entirely and parks every worker thread, so CPU/GPU drop
#  to ~zero. The live page resumes on entry. Both endpoints are idempotent.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/pipeline/pause", methods=["POST"])
def api_pipeline_pause():
    """Park camera + ML workers and free the camera device. Safe to call
    repeatedly; callers should not assume a state-change response."""
    was_active = _pipeline_active.is_set()
    _pipeline_active.clear()

    # Wake any MJPEG client blocked on the condition variable so its
    # generator returns instead of hanging until shutdown.
    with _jpeg_available:
        _jpeg_available.notify_all()

    # Stop the audio thread up front — it owns its own loop and won't
    # notice _pipeline_active by itself. Speech is meaningless on the
    # dashboard page anyway.
    if _transcriber:
        try:
            _transcriber.stop()
        except Exception as exc:
            print(f"[Pipeline] transcriber.stop on pause: {exc}")

    # Drop any in-flight track / display state so resuming starts clean
    # (no phantom bboxes carried across a pause).
    with _state_lock:
        _display_states.clear()
    with _tracks_lock:
        _active_tracks.clear()

    if was_active:
        print("[Pipeline] Paused — camera + ML idled.")
    return jsonify({
        "active":      False,
        "was_active":  was_active,
        "camera_ok":   _camera_ok.is_set(),
    })


@app.route("/api/pipeline/resume", methods=["POST"])
def api_pipeline_resume():
    """Re-enable the camera + ML loops. Camera reader will re-open the
    device on its next iteration; clients can poll /api/pipeline/status
    for camera_ok readiness."""
    was_active = _pipeline_active.is_set()
    _pipeline_active.set()
    if not was_active:
        print("[Pipeline] Resumed.")
    return jsonify({
        "active":      True,
        "was_active":  was_active,
        "camera_ok":   _camera_ok.is_set(),
    })


@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify({
        "active":       _pipeline_active.is_set(),
        "camera_ok":    _camera_ok.is_set(),
        "worker_alive": _worker_alive.is_set(),
        "worker_ready": _worker_ready.is_set(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--camera", default=0, type=int, help="OpenCV camera index.")
    args = parser.parse_args()

    global _transcriber, _camera_index
    _camera_index = args.camera
    _transcriber = SpeechTranscriber()
    if not _transcriber.available:
        print("[Main] Speech transcription unavailable (install vosk + sounddevice to enable).")

    worker = threading.Thread(target=recognition_worker, daemon=True,
                              name="RecognitionWorker")
    worker.start()

    print("[Main] Waiting for recognition worker to load FaceNet model...")
    if not _worker_ready.wait(timeout=120.0):
        print("[Main] Worker failed to start within 120s — aborting.")
        return

    # Camera reader now owns its own VideoCapture so the device can be
    # released cleanly on /api/pipeline/pause and re-opened on resume —
    # no main-thread VideoCapture handle to track. Reader survives every
    # pause/resume cycle as a single thread.
    cam_thread = threading.Thread(target=_camera_reader,
                                  daemon=True, name="CameraReader")
    cam_thread.start()
    _camera_ok.wait(timeout=5.0)

    detector_thread = threading.Thread(target=detection_worker, daemon=True,
                                       name="DetectionWorker")
    detector_thread.start()

    render_thread = threading.Thread(target=render_producer, daemon=True,
                                     name="RenderProducer")
    render_thread.start()

    print(f"[Main] Serving dashboard at http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True,
                debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        # Deterministic shutdown order: flag → audio+http-bound resources
        # → MJPEG unblockers → worker threads → DB writers.
        print("[Main] Shutting down...")
        _stop_flag.set()
        # Set pipeline_active so any worker parked on the pause event
        # wakes up and notices stop_flag instead of waiting out its timeout.
        _pipeline_active.set()

        # Unblock any MJPEG clients waiting on the condition variable.
        with _jpeg_available:
            _jpeg_available.notify_all()

        # Release transcriber FIRST so the Vosk model is freed while the
        # vosk C module is still resident (fixes interpreter-shutdown race).
        if _transcriber:
            try:
                _transcriber.close()
            except Exception as exc:
                print(f"[Main] Transcriber close error: {exc}")

        # Join pipeline threads. The camera reader releases its own
        # VideoCapture on exit (see _camera_reader's finally block), so
        # there's no main-thread cap to clean up.
        for t in (render_thread, detector_thread, cam_thread, worker):
            try:
                t.join(timeout=3.0)
            except Exception as exc:
                print(f"[Main] Thread {t.name} join error: {exc}")

        # Drain the DB writer threads (both own background loops).
        if _face_db is not None:
            try: _face_db.close()
            except Exception as exc: print(f"[Main] DB close error: {exc}")
        if _session_mgr is not None:
            mem = getattr(_session_mgr, "_memory", None)
            if mem is not None:
                try: mem.close()
                except Exception as exc: print(f"[Main] Memory close error: {exc}")

        print("[Main] Done.")


if __name__ == "__main__":
    main()
