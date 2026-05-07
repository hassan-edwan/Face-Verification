"""
Gatekeeper: Two-Stage Enrollment with Tracking Lock + Multi-Embedding Bank
===========================================================================
Decisions: MATCHED, FAST_ENROLLED, REFINING, REFINED, UNCERTAIN

Key additions over v1:
  - Tracking lock: once tracker_id → identity is set, NEVER re-match. Trust tracker.
  - Multi-embedding bank: up to MAX_EMBEDDINGS_PER_IDENTITY per identity.
    Append while space; replace worst-quality when full.
  - Identity metadata: last_seen timestamp + tracker_ids set.
"""

import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto


# ── TUNABLE THRESHOLDS ────────────────────────────────────────────────────────
MATCH_THRESHOLD             = 0.70   # cosine sim >= this  → recognised
ENROLL_THRESHOLD            = 0.40   # cosine sim <  this  → new identity
CONSENSUS_FRAMES            = 7      # frames before REFINED
MAX_EMBEDDINGS_PER_IDENTITY = 15     # embedding bank cap per person
BUFFER_MAX_AGE              = 30.0   # seconds before pending candidate expires
MIN_QUALITY_FOR_UPDATE      = 0.50   # skip embedding update below this score
# ─────────────────────────────────────────────────────────────────────────────


class GatekeeperDecision(Enum):
    MATCHED       = auto()
    FAST_ENROLLED = auto()
    REFINING      = auto()
    REFINED       = auto()
    UNCERTAIN     = auto()


@dataclass
class GatekeeperResult:
    decision:        GatekeeperDecision
    identity:        Optional[str]
    score:           float
    frame_count:     int              # embedding count (MATCHED) or consensus count
    quality_score:   float = 0.0
    best_face_image: Optional[np.ndarray] = None


@dataclass
class PendingCandidate:
    track_id:        int
    embeddings:      List[np.ndarray]     = field(default_factory=list)
    quality_scores:  List[float]          = field(default_factory=list)
    best_face_image: Optional[np.ndarray] = None
    best_quality:    float                = 0.0
    first_seen:      float                = field(default_factory=time.time)
    last_seen:       float                = field(default_factory=time.time)
    temp_name:       Optional[str]        = None


class Gatekeeper:
    def __init__(self, known_faces: Dict[str, List[np.ndarray]], person_counter: int):
        self._known_faces    = known_faces
        self._person_counter = person_counter
        self._pending:               Dict[int, PendingCandidate] = {}
        self._enrolled_tracks:       set                         = set()
        self._fast_enrolled_tracks:  Dict[int, str]              = {}

        # Tracking lock: permanent tracker_id → identity name.
        # Once set, process() returns immediately without re-matching.
        self._tracker_identity_map:  Dict[int, str]              = {}

        # Quality scores parallel to each identity's embedding list
        self._identity_quality: Dict[str, List[float]] = {
            name: [0.5] * len(embs) for name, embs in known_faces.items()
        }

        # Metadata per identity
        self._identity_metadata: Dict[str, dict] = {
            name: {"last_seen": 0.0, "tracker_ids": set()}
            for name in known_faces
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        embedding:     np.ndarray,
        track_id:      int,
        face_bgr:      np.ndarray,
        quality_score: float,
    ) -> GatekeeperResult:
        self._cleanup_expired()

        # ── Guard 1: TRACKING LOCK - fully enrolled tracks ───────────────────
        # After first assignment, ALWAYS reuse the cached identity.
        # Never re-run similarity search → prevents flickering on occlusion/angles.
        if track_id in self._enrolled_tracks:
            name = self._tracker_identity_map.get(track_id)
            if name and name in self._known_faces:
                self._update_metadata(name, track_id)
                self._try_update_embeddings(name, embedding, quality_score)
                n = len(self._known_faces[name])
                return GatekeeperResult(GatekeeperDecision.MATCHED, name, 1.0, n, quality_score)
            # Fallback: tracking map desync (shouldn't happen)
            name, score = self._find_best_match(embedding)
            return GatekeeperResult(GatekeeperDecision.MATCHED, name, score, 0, quality_score)

        # ── Guard 2: Fast-enrolled, still refining ────────────────────────────
        if track_id in self._fast_enrolled_tracks:
            return self._refine(track_id, embedding, face_bgr, quality_score)

        # ── First contact: run full decision pipeline (once per new track_id) ─
        name, score = self._find_best_match(embedding)

        if score >= MATCH_THRESHOLD:
            # Known identity → lock track immediately, no refinement needed
            self._pending.pop(track_id, None)
            self._tracker_identity_map[track_id] = name
            self._enrolled_tracks.add(track_id)
            self._update_metadata(name, track_id)
            self._try_update_embeddings(name, embedding, quality_score)
            n = len(self._known_faces.get(name, []))
            return GatekeeperResult(GatekeeperDecision.MATCHED, name, score, n, quality_score)

        if ENROLL_THRESHOLD <= score < MATCH_THRESHOLD:
            return GatekeeperResult(GatekeeperDecision.UNCERTAIN, None, score, 0, quality_score)

        # Unknown face → start accumulation buffer
        candidate = self._pending.get(track_id)
        if candidate is None:
            candidate = PendingCandidate(track_id=track_id)
            self._pending[track_id] = candidate

        candidate.embeddings.append(embedding)
        candidate.quality_scores.append(quality_score)
        candidate.last_seen = time.time()

        if quality_score > candidate.best_quality:
            candidate.best_quality    = quality_score
            candidate.best_face_image = face_bgr.copy()

        frame_count = len(candidate.embeddings)

        if self._matches_another_candidate(embedding, track_id):
            self._pending.pop(track_id, None)
            return GatekeeperResult(GatekeeperDecision.UNCERTAIN, None, score, frame_count, quality_score)

        # Fast-enroll on the very first quality frame
        if frame_count == 1:
            return self._fast_enroll(track_id, candidate, embedding)

        if frame_count >= CONSENSUS_FRAMES:
            return self._promote_to_refined(track_id, candidate)

        return GatekeeperResult(GatekeeperDecision.REFINING, candidate.temp_name, 1.0, frame_count, quality_score)

    @property
    def known_faces(self) -> Dict[str, List[np.ndarray]]:
        return self._known_faces

    @property
    def person_counter(self) -> int:
        return self._person_counter

    def is_already_enrolled(self, track_id: int) -> bool:
        return track_id in self._fast_enrolled_tracks or track_id in self._enrolled_tracks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fast_enroll(self, track_id: int, candidate: PendingCandidate, embedding: np.ndarray) -> GatekeeperResult:
        new_name = f"Person {self._person_counter}"
        self._person_counter += 1
        candidate.temp_name = new_name

        self._known_faces[new_name]        = [embedding.copy()]
        self._identity_quality[new_name]   = [candidate.best_quality]
        self._identity_metadata[new_name]  = {"last_seen": time.time(), "tracker_ids": {track_id}}

        self._fast_enrolled_tracks[track_id] = new_name
        self._tracker_identity_map[track_id] = new_name   # lock from frame 1

        print(f"[Gatekeeper] [FAST] '{new_name}' from Track {track_id}")
        return GatekeeperResult(
            decision=GatekeeperDecision.FAST_ENROLLED, identity=new_name,
            score=1.0, frame_count=1,
            quality_score=candidate.best_quality,
            best_face_image=candidate.best_face_image,
        )

    def _refine(self, track_id: int, embedding: np.ndarray, face_bgr: np.ndarray, quality_score: float) -> GatekeeperResult:
        candidate = self._pending.get(track_id)
        temp_name = self._fast_enrolled_tracks[track_id]

        if candidate is None:
            return GatekeeperResult(GatekeeperDecision.MATCHED, temp_name, 1.0, 0, quality_score)

        candidate.embeddings.append(embedding)
        candidate.quality_scores.append(quality_score)
        candidate.last_seen = time.time()

        if quality_score > candidate.best_quality:
            candidate.best_quality    = quality_score
            candidate.best_face_image = face_bgr.copy()

        frame_count = len(candidate.embeddings)

        if frame_count < CONSENSUS_FRAMES:
            return GatekeeperResult(GatekeeperDecision.REFINING, temp_name, 1.0, frame_count, quality_score)

        return self._promote_to_refined(track_id, candidate)

    def _promote_to_refined(self, track_id: int, candidate: PendingCandidate) -> GatekeeperResult:
        n_frames  = len(candidate.embeddings)
        temp_name = candidate.temp_name or f"Person {self._person_counter}"

        # Keep best-quality embeddings (up to MAX_EMBEDDINGS_PER_IDENTITY)
        if candidate.quality_scores:
            pairs = sorted(
                zip(candidate.quality_scores, candidate.embeddings),
                key=lambda x: x[0], reverse=True
            )[:MAX_EMBEDDINGS_PER_IDENTITY]
            top_qualities  = [q for q, _ in pairs]
            top_embeddings = [e for _, e in pairs]
        else:
            top_embeddings = candidate.embeddings[:MAX_EMBEDDINGS_PER_IDENTITY]
            top_qualities  = [0.5] * len(top_embeddings)

        # Final dedup using mean of top embeddings
        mean_emb = np.mean(top_embeddings, axis=0, keepdims=True)
        final_name, final_score = self._find_best_match_excluding(mean_emb, temp_name)

        if final_score >= ENROLL_THRESHOLD:
            # Resolved to a different known identity - discard temp entry
            print(f"[Gatekeeper] Refinement dedup: Track {track_id} -> '{final_name}' ({final_score:.2f})")
            self._known_faces.pop(temp_name, None)
            self._identity_quality.pop(temp_name, None)
            self._identity_metadata.pop(temp_name, None)
            self._fast_enrolled_tracks.pop(track_id, None)
            self._pending.pop(track_id, None)
            self._enrolled_tracks.add(track_id)
            self._tracker_identity_map[track_id] = final_name
            self._update_metadata(final_name, track_id)
            return GatekeeperResult(GatekeeperDecision.MATCHED, final_name, final_score, n_frames, candidate.best_quality)

        # Store top-K embeddings (replaces single-frame fast-enroll entry)
        self._known_faces[temp_name]       = top_embeddings
        self._identity_quality[temp_name]  = top_qualities
        self._update_metadata(temp_name, track_id)

        print(f"[Gatekeeper] [REFINED] '{temp_name}' ({len(top_embeddings)} embs, q={candidate.best_quality:.2f})")

        self._fast_enrolled_tracks.pop(track_id, None)
        self._pending.pop(track_id, None)
        self._enrolled_tracks.add(track_id)
        # _tracker_identity_map[track_id] already set in _fast_enroll

        return GatekeeperResult(
            decision=GatekeeperDecision.REFINED, identity=temp_name,
            score=1.0, frame_count=n_frames,
            quality_score=candidate.best_quality,
            best_face_image=candidate.best_face_image,
        )

    def _try_update_embeddings(self, identity_name: str, embedding: np.ndarray, quality_score: float) -> None:
        """Append or replace-worst embedding for a tracking-locked identity. Non-blocking."""
        if quality_score < MIN_QUALITY_FOR_UPDATE or identity_name not in self._known_faces:
            return
        embs  = self._known_faces[identity_name]
        quals = self._identity_quality.get(identity_name, [])

        if len(embs) < MAX_EMBEDDINGS_PER_IDENTITY:
            embs.append(embedding.copy())
            quals.append(quality_score)
            self._identity_quality[identity_name] = quals
        elif quals:
            worst = int(np.argmin(quals))
            if quality_score > quals[worst]:
                embs[worst]  = embedding.copy()
                quals[worst] = quality_score
                self._identity_quality[identity_name] = quals

    def _update_metadata(self, identity_name: str, track_id: int) -> None:
        if identity_name not in self._identity_metadata:
            self._identity_metadata[identity_name] = {"last_seen": 0.0, "tracker_ids": set()}
        meta = self._identity_metadata[identity_name]
        meta["last_seen"] = time.time()
        meta["tracker_ids"].add(track_id)

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._known_faces:
            return None, 0.0
        best_name, best_score = None, 0.0
        a     = embedding.flatten()
        norm_a = np.linalg.norm(a) + 1e-9
        for name, emb_list in self._known_faces.items():
            for known_emb in emb_list:
                b     = known_emb.flatten()
                score = float(np.dot(a, b) / (norm_a * (np.linalg.norm(b) + 1e-9)))
                if score > best_score:
                    best_score, best_name = score, name
        return best_name, best_score

    def _find_best_match_excluding(self, embedding: np.ndarray, exclude_name: str) -> Tuple[Optional[str], float]:
        if not self._known_faces:
            return None, 0.0
        best_name, best_score = None, 0.0
        a      = embedding.flatten()
        norm_a = np.linalg.norm(a) + 1e-9
        for name, emb_list in self._known_faces.items():
            if name == exclude_name:
                continue
            for known_emb in emb_list:
                b     = known_emb.flatten()
                score = float(np.dot(a, b) / (norm_a * (np.linalg.norm(b) + 1e-9)))
                if score > best_score:
                    best_score, best_name = score, name
        return best_name, best_score

    def _matches_another_candidate(self, embedding: np.ndarray, own_track_id: int) -> bool:
        a      = embedding.flatten()
        norm_a = np.linalg.norm(a) + 1e-9
        for tid, candidate in self._pending.items():
            if tid == own_track_id or not candidate.embeddings:
                continue
            other = np.mean(candidate.embeddings, axis=0).flatten()
            sim   = float(np.dot(a, other) / (norm_a * (np.linalg.norm(other) + 1e-9)))
            if sim >= MATCH_THRESHOLD:
                return True
        return False

    def _cleanup_expired(self) -> None:
        now     = time.time()
        expired = [tid for tid, c in self._pending.items() if (now - c.last_seen) > BUFFER_MAX_AGE]
        for tid in expired:
            if tid in self._fast_enrolled_tracks:
                name = self._fast_enrolled_tracks.pop(tid)
                print(f"[Gatekeeper] Track {tid} refinement expired - keeping '{name}' as-is.")
                self._enrolled_tracks.add(tid)
            else:
                print(f"[Gatekeeper] Candidate Track {tid} expired - discarded.")
            self._pending.pop(tid)


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE API FUNCTIONS
#  (mirror the Gatekeeper's internals for external callers / unit tests)
# ═══════════════════════════════════════════════════════════════════════════════

def match_identity(
    embedding:   np.ndarray,
    known_faces: Dict[str, List[np.ndarray]],
) -> Tuple[Optional[str], float]:
    """Compare against ALL embeddings of every identity. Return (name, best_score)."""
    best_name, best_score = None, 0.0
    a      = embedding.flatten()
    norm_a = np.linalg.norm(a) + 1e-9
    for name, emb_list in known_faces.items():
        for emb in emb_list:
            b     = emb.flatten()
            score = float(np.dot(a, b) / (norm_a * (np.linalg.norm(b) + 1e-9)))
            if score > best_score:
                best_score, best_name = score, name
    return best_name, best_score


def update_identity_embeddings(
    identity_name:    str,
    new_embedding:    np.ndarray,
    quality_score:    float,
    known_faces:      Dict[str, List[np.ndarray]],
    identity_quality: Dict[str, List[float]],
) -> bool:
    """Append or replace-worst embedding. Returns True if bank was modified."""
    if identity_name not in known_faces or quality_score < MIN_QUALITY_FOR_UPDATE:
        return False
    embs  = known_faces[identity_name]
    quals = identity_quality.get(identity_name, [])

    if len(embs) < MAX_EMBEDDINGS_PER_IDENTITY:
        embs.append(new_embedding.copy())
        quals.append(quality_score)
        identity_quality[identity_name] = quals
        return True

    if quals:
        worst = int(np.argmin(quals))
        if quality_score > quals[worst]:
            embs[worst]  = new_embedding.copy()
            quals[worst] = quality_score
            identity_quality[identity_name] = quals
            return True
    return False


def assign_identity(
    tracker_id:      int,
    embedding:       np.ndarray,
    known_faces:     Dict[str, List[np.ndarray]],
    tracker_map:     Dict[int, str],
    person_counter:  int,
    threshold:       float = MATCH_THRESHOLD,
) -> Tuple[str, bool, int]:
    """
    First-time identity assignment for a tracker_id.
    Returns (identity_name, is_new, updated_person_counter).
    Enforces tracking lock: if tracker_id already mapped, returns immediately.
    """
    if tracker_id in tracker_map:
        return tracker_map[tracker_id], False, person_counter   # tracking lock

    name, score = match_identity(embedding, known_faces)
    if score >= threshold:
        tracker_map[tracker_id] = name
        return name, False, person_counter

    # New identity
    new_name = f"Person {person_counter}"
    known_faces[new_name]   = [embedding.copy()]
    tracker_map[tracker_id] = new_name
    return new_name, True, person_counter + 1


def fast_enroll(
    face_bgr:       np.ndarray,
    tracker_id:     int,
    known_faces:    Dict[str, List[np.ndarray]],
    embedder,                                     # FaceNet instance
    person_counter: int,
) -> Tuple[str, np.ndarray, int]:
    """Create identity with 1 embedding. Returns (name, embedding, new_counter)."""
    rgb       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    embedding = embedder.embeddings(np.expand_dims(rgb, 0))
    new_name  = f"Person {person_counter}"
    known_faces[new_name] = [embedding.copy()]
    return new_name, embedding, person_counter + 1


def refine_enrollment(
    tracker_id:       int,
    face_bgr:         np.ndarray,
    identity_name:    str,
    known_faces:      Dict[str, List[np.ndarray]],
    identity_quality: Dict[str, List[float]],
    embedder,
) -> np.ndarray:
    """Add a quality-screened embedding to an existing identity (up to 15)."""
    from src.quality import quick_quality_check
    rgb       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    embedding = embedder.embeddings(np.expand_dims(rgb, 0))
    ok, q     = quick_quality_check(face_bgr)
    if ok:
        update_identity_embeddings(identity_name, embedding, q, known_faces, identity_quality)
    return embedding


def handle_detection(
    frame:          np.ndarray,
    detections:     list,
    tracker_map:    Dict[int, str],
    known_faces:    Dict[str, List[np.ndarray]],
    embedder,
    person_counter: int,
) -> Tuple[list, int]:
    """
    Main per-frame identity dispatcher.
    Enforces tracking lock: locked tracker_ids never re-run matching.

    Each detection must have: tracker_id (int), face_crop (np.ndarray BGR).
    Returns (results_list, updated_person_counter).
    """
    results = []
    for det in detections:
        tid      = det.get("tracker_id")
        face_bgr = det.get("face_crop")
        if tid is None or face_bgr is None or face_bgr.size == 0:
            continue

        # Tracking lock: identity already assigned → return immediately
        if tid in tracker_map:
            results.append({"tracker_id": tid, "identity": tracker_map[tid], "is_new": False})
            continue

        # First contact: embed then assign
        rgb       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        embedding = embedder.embeddings(np.expand_dims(rgb, 0))
        name, is_new, person_counter = assign_identity(
            tid, embedding, known_faces, tracker_map, person_counter
        )
        results.append({"tracker_id": tid, "identity": name, "is_new": is_new})

    return results, person_counter
