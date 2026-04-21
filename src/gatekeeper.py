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
# Calibrated for the ArcFace (buffalo_l / w600k_r50) embedder in
# src/embedder.py. ArcFace produces much tighter same-identity clusters
# than FaceNet (typical same-id cosine 0.30-0.60 on LFW/IJB; different-id
# 0.05-0.20). FaceNet values (0.70 / 0.60 / 0.40) would reject every
# real match. When reverting via FACE_EMBEDDER=facenet, threshold sweep
# on the fallback path before trusting these numbers.
MATCH_THRESHOLD             = 0.40   # cosine sim >= this  → strict unconditional match
REMATCH_THRESHOLD           = 0.20   # cosine sim >= this + rank-1 margin ≥ MIN_MATCH_MARGIN → re-acquisition match (tail genuines)
ENROLL_THRESHOLD            = 0.20   # cosine sim <  this  → new identity
MIN_MATCH_MARGIN            = 0.05   # run_019: required rank-1 minus rank-2 cosine gap when score sits in [REMATCH, MATCH). Decouples re-acquisition (wants loose REMATCH) from dedup (wants strict): at score=0.25 we still MATCH if rank-1 clearly wins, but refuse the match when a near-tie could merge two distinct identities.
CONSENSUS_FRAMES            = 5      # frames before REFINED (was 7; 5 gives faster enrollment)
MAX_EMBEDDINGS_PER_IDENTITY = 8      # embedding bank cap per person (was 15; 8 halves match cost)
BUFFER_MAX_AGE              = 10.0   # seconds before pending candidate expires (was 30; faster cleanup)
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

        # ── Pre-computed embedding bank for fast vectorized matching ──────────
        # _emb_bank: (N, D) matrix of L2-normalized embeddings (all identities)
        # _bank_labels: list[name] parallel to rows (kept for exclude-match mask
        # builds; numpy object arrays don't speed string compares enough to
        # justify a separate representation). A single np.dot replaces the
        # per-identity loop.
        # _bank_dirty signals *structural* invalidation (replace-worst /
        # forget). Plain appends skip the rebuild and extend in place — the
        # matmul cost is unchanged and we save ~N float32 copies per enrol.
        self._emb_bank: Optional[np.ndarray] = None
        self._bank_labels: List[str] = []
        self._bank_dirty = True
        self._rebuild_embedding_bank()

        # Recently-matched cache: stores (normalized_emb, name) for the last few
        # matches. Checked before full bank search for near-instant re-identification.
        # Kept intact on appends — a cached name is still valid if the identity
        # just gained more embeddings. Cleared only on forget / replace-worst.
        self._recent_matches: List[Tuple[np.ndarray, str]] = []
        self._recent_max = 5

    # ── Public API ─────────────────────────────────────────────────────────────

    def cleanup_dead_tracks(self, active_track_ids: set) -> None:
        """Remove internal state for tracks the IoU tracker has dropped.
        Prevents memory leaks in _pending and _fast_enrolled_tracks."""
        dead_pending = [tid for tid in self._pending if tid not in active_track_ids]
        for tid in dead_pending:
            if tid in self._fast_enrolled_tracks:
                # Keep the identity (already fast-enrolled) but stop refining
                self._enrolled_tracks.add(tid)
                self._fast_enrolled_tracks.pop(tid, None)
            self._pending.pop(tid, None)

    def process(
        self,
        embedding:     np.ndarray,
        track_id:      int,
        face_bgr:      np.ndarray,
        quality_score: float,
    ) -> GatekeeperResult:
        self._cleanup_expired()

        # ── Guard 1: TRACKING LOCK — fully enrolled tracks ───────────────────
        # After first assignment the cached identity is normally reused
        # without re-matching (prevents flickering on pose / occlusion /
        # lighting drift). Run_021 adds a drift-break: if the incoming
        # embedding is too dissimilar from the locked identity's bank
        # AND another identity is a clearer match, the tracker has
        # likely swapped face-assignments (crowded-scene S3 failure
        # mode) — break the lock and re-lock to whoever the embedder
        # now thinks this is. Drift-break must fire BEFORE
        # `_try_update_embeddings` or a wrong-identity frame would
        # silently pollute the correct identity's bank.
        if track_id in self._enrolled_tracks:
            name = self._tracker_identity_map.get(track_id)
            if name and name in self._known_faces:
                bank_rows = self._known_faces[name]
                cos_self = self._max_cosine_against_bank(embedding, bank_rows)
                if cos_self < REMATCH_THRESHOLD:
                    best_name, best_score, best_margin = \
                        self._find_best_match_with_margin(embedding)
                    is_strict  = best_score >= MATCH_THRESHOLD
                    is_reacq   = (best_score >= REMATCH_THRESHOLD
                                  and best_margin >= MIN_MATCH_MARGIN)
                    if (best_name is not None and best_name != name
                            and (is_strict or is_reacq)):
                        # Tracker swap confirmed — re-lock and update the
                        # new identity's bank instead of the old one's.
                        self._tracker_identity_map[track_id] = best_name
                        self._update_metadata(best_name, track_id)
                        self._try_update_embeddings(best_name, embedding,
                                                    quality_score)
                        n = len(self._known_faces.get(best_name, []))
                        print(f"[Gatekeeper] Track {track_id} drift-break: "
                              f"'{name}' → '{best_name}' "
                              f"(cos_self={cos_self:.2f}, "
                              f"cos_new={best_score:.2f})")
                        return GatekeeperResult(GatekeeperDecision.MATCHED,
                                                best_name, best_score,
                                                n, quality_score)
                    # Drift without a clear alternative — hold the lock
                    # but skip the bank update so we don't pollute with
                    # a low-confidence embedding.
                    self._update_metadata(name, track_id)
                    return GatekeeperResult(GatekeeperDecision.MATCHED,
                                            name, cos_self,
                                            len(bank_rows), quality_score)
                # Normal path: embedding confirms the lock.
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
        # The match decision has to serve two different situations with one
        # call site:
        #   (a) Cross-track dedup at enrollment — "is this a duplicate of
        #       someone already in the bank?" — wants strict threshold, else
        #       distinct subjects whose mugshots happen to score ~0.25
        #       against each other get merged (run_017 measured 46/130 = 35%
        #       collision rate at REMATCH=0.20).
        #   (b) Re-acquisition of a lost track — "did this new track_id
        #       return the same walker?" — wants loose threshold to admit
        #       pose/distance-drifted tail genuines (d1_far SCface queries
        #       often sit at cosine 0.20–0.35 against their own bank entry).
        #
        # Run_019 splits them with a rank-1-vs-rank-2 margin check:
        #   - score ≥ MATCH_THRESHOLD              → unconditional MATCH
        #   - score ≥ REMATCH_THRESHOLD AND margin
        #     ≥ MIN_MATCH_MARGIN                   → re-acquisition MATCH
        #     (high score or clear rank-1 winner means the embedder is
        #     confident this is the same person; ambiguous near-ties are
        #     refused to avoid collision merges).
        #   - otherwise                            → UNCERTAIN / accumulate.
        name, score, margin = self._find_best_match_with_margin(embedding)

        is_strict_match    = score >= MATCH_THRESHOLD
        is_reacquisition   = (score >= REMATCH_THRESHOLD
                              and margin >= MIN_MATCH_MARGIN)

        if is_strict_match or is_reacquisition:
            # Known identity → lock track immediately, no refinement needed
            self._pending.pop(track_id, None)
            self._tracker_identity_map[track_id] = name
            self._enrolled_tracks.add(track_id)
            self._update_metadata(name, track_id)
            self._try_update_embeddings(name, embedding, quality_score)
            n = len(self._known_faces.get(name, []))
            return GatekeeperResult(GatekeeperDecision.MATCHED, name, score, n, quality_score)

        if ENROLL_THRESHOLD <= score < MATCH_THRESHOLD:
            # Ambiguous band: either below REMATCH, or above REMATCH but
            # rank-1 / rank-2 cosines are too close to justify re-lock.
            # Refusing the match protects against collision; live pipeline
            # will continue to see the track and may eventually resolve via
            # a higher-quality later frame.
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

    def forget(self, name: str) -> int:
        """Purge a single identity from every in-memory store the gatekeeper
        owns: embedding bank, quality scores, metadata, and any tracker-id
        locks pointing at it. Returns the number of embeddings removed
        (0 if the identity wasn't known).

        Callers (cascade-delete in server.py) used to mutate these dicts
        directly — that's brittle because the bank representation now has
        invariants (see `_append_to_bank`). Routing the deletion through
        here keeps the invalidation ordering correct."""
        embs = self._known_faces.pop(name, None)
        if embs is None:
            return 0
        removed = len(embs)
        self._identity_quality.pop(name, None)
        self._identity_metadata.pop(name, None)

        # Drop every tracker_id that was locked onto this name. Also drop
        # the matching pending / fast-enrolled entries so a subsequent
        # re-enroll of the same face gets a fresh track, not a half-zombie.
        dead_tracks = [tid for tid, n in self._tracker_identity_map.items() if n == name]
        for tid in dead_tracks:
            self._tracker_identity_map.pop(tid, None)
            self._enrolled_tracks.discard(tid)
            self._fast_enrolled_tracks.pop(tid, None)
            self._pending.pop(tid, None)
        for tid in [t for t, n in list(self._fast_enrolled_tracks.items()) if n == name]:
            self._fast_enrolled_tracks.pop(tid, None)
            self._pending.pop(tid, None)

        # Structural change → full rebuild + recent-cache clear.
        self._invalidate_bank()
        return removed

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
        # Pure append → incremental bank update, recent-match cache stays valid.
        self._append_to_bank(new_name, embedding)

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
            # Resolved to a different known identity — discard temp entry
            print(f"[Gatekeeper] Refinement dedup: Track {track_id} -> '{final_name}' ({final_score:.2f})")
            self._known_faces.pop(temp_name, None)
            self._identity_quality.pop(temp_name, None)
            self._identity_metadata.pop(temp_name, None)
            self._fast_enrolled_tracks.pop(track_id, None)
            self._pending.pop(track_id, None)
            self._enrolled_tracks.add(track_id)
            self._tracker_identity_map[track_id] = final_name
            self._update_metadata(final_name, track_id)
            self._invalidate_bank()
            return GatekeeperResult(GatekeeperDecision.MATCHED, final_name, final_score, n_frames, candidate.best_quality)

        # Store top-K embeddings (replaces single-frame fast-enroll entry)
        self._known_faces[temp_name]       = top_embeddings
        self._identity_quality[temp_name]  = top_qualities
        self._update_metadata(temp_name, track_id)
        self._invalidate_bank()

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
            # Append-only update: stack one normalized row onto the bank and
            # leave the recent-match cache alone. No O(N·D) rebuild.
            self._append_to_bank(identity_name, embedding)
        elif quals:
            worst = int(np.argmin(quals))
            if quality_score > quals[worst]:
                embs[worst]  = embedding.copy()
                quals[worst] = quality_score
                self._identity_quality[identity_name] = quals
                # Replace-worst flips a cached row's direction → full rebuild.
                self._invalidate_bank()

    def _update_metadata(self, identity_name: str, track_id: int) -> None:
        if identity_name not in self._identity_metadata:
            self._identity_metadata[identity_name] = {"last_seen": 0.0, "tracker_ids": set()}
        meta = self._identity_metadata[identity_name]
        meta["last_seen"] = time.time()
        meta["tracker_ids"].add(track_id)

    def _rebuild_embedding_bank(self) -> None:
        """Rebuild the pre-normalized (N, D) embedding matrix for vectorized matching."""
        all_embs = []
        labels = []
        for name, emb_list in self._known_faces.items():
            for emb in emb_list:
                all_embs.append(emb.flatten())
                labels.append(name)
        if all_embs:
            bank = np.array(all_embs, dtype=np.float32)       # (N, D)
            norms = np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9
            self._emb_bank = bank / norms                      # L2-normalized
        else:
            self._emb_bank = None
        self._bank_labels = labels
        self._bank_dirty = False

    def _append_to_bank(self, name: str, embedding: np.ndarray) -> None:
        """Incremental version of _rebuild for the common `one new row` case.
        Avoids rebuilding the whole (N, D) matrix from Python lists — we just
        vstack a single normalized row. On a cold bank (no rows yet) we defer
        to the full rebuild path."""
        if self._bank_dirty or self._emb_bank is None:
            self._rebuild_embedding_bank()
            return
        v = embedding.flatten().astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        self._emb_bank = np.vstack([self._emb_bank, v[np.newaxis, :]])
        self._bank_labels.append(name)

    def _invalidate_bank(self) -> None:
        """Mark the embedding bank for rebuild on next match call. Used on
        structural changes (replace-worst, forget) that can flip cached
        match results — clearing the recent-match cache protects against
        stale name hits."""
        self._bank_dirty = True
        self._recent_matches.clear()

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._known_faces:
            return None, 0.0

        q = embedding.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        # Fast path: check recently-matched identities first (~0.1ms vs full search).
        # Uses REMATCH_THRESHOLD because the cached embedding is a single sample;
        # a full-bank match against the same identity is always ≥ the cache hit,
        # so accepting at REMATCH_THRESHOLD here is bounded above by the bank.
        for cached_emb, cached_name in self._recent_matches:
            score = float(np.dot(q, cached_emb))
            if score >= REMATCH_THRESHOLD:
                return cached_name, score

        # Full vectorized search
        if self._bank_dirty:
            self._rebuild_embedding_bank()
        if self._emb_bank is None:
            return None, 0.0
        scores = q @ self._emb_bank.T                         # (N,)
        best_idx = int(np.argmax(scores))
        best_name = self._bank_labels[best_idx]
        best_score = float(scores[best_idx])

        # Update recent cache on successful match
        if best_score >= MATCH_THRESHOLD:
            self._recent_matches = [
                (c, n) for c, n in self._recent_matches if n != best_name
            ]
            self._recent_matches.insert(0, (q.copy(), best_name))
            if len(self._recent_matches) > self._recent_max:
                self._recent_matches.pop()

        return best_name, best_score

    def _max_cosine_against_bank(self, embedding: np.ndarray,
                                 bank_rows: List[np.ndarray]) -> float:
        """Max cosine between `embedding` and any row in `bank_rows`.
        `_known_faces` stores raw embeddings (not L2-normalized), so
        this normalizes both sides on the fly. Returns -1.0 for an
        empty bank — callers treat that as 'no signal, skip drift
        check'."""
        if not bank_rows:
            return -1.0
        q = embedding.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        best = -1.0
        for row in bank_rows:
            r = row.flatten().astype(np.float32)
            r = r / (np.linalg.norm(r) + 1e-9)
            c = float(np.dot(q, r))
            if c > best:
                best = c
        return best

    def _find_best_match_with_margin(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float, float]:
        """Rank-1 match augmented with the rank-1 vs rank-2 margin over
        DISTINCT identities. Used by the first-contact decision to
        distinguish a clean re-acquisition (rank-1 clearly separated) from
        a collision-risk near-tie (two identities bunched close together).

        Returns (rank1_name, rank1_score, margin). If the gallery holds
        fewer than two identities the margin is float('inf') — any match
        is unambiguous because there's nothing to confuse it with."""
        if not self._known_faces:
            return None, 0.0, 0.0

        q = embedding.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        if self._bank_dirty:
            self._rebuild_embedding_bank()
        if self._emb_bank is None:
            return None, 0.0, 0.0

        scores = q @ self._emb_bank.T                         # (N,)
        # Collapse to best-per-identity so margin is measured across
        # DISTINCT people, not across multiple bank rows of the same person.
        best_per_identity: Dict[str, float] = {}
        for name, s in zip(self._bank_labels, scores):
            if s > best_per_identity.get(name, -2.0):
                best_per_identity[name] = float(s)

        ranked = sorted(best_per_identity.items(), key=lambda t: -t[1])
        if not ranked:
            return None, 0.0, 0.0
        best_name, best_score = ranked[0]
        if len(ranked) == 1:
            return best_name, best_score, float("inf")
        second_score = ranked[1][1]
        margin = best_score - second_score

        # Keep the recent-match cache warm for the strict-match path — same
        # behavior as `_find_best_match`, just with the margin tacked on.
        if best_score >= MATCH_THRESHOLD:
            self._recent_matches = [
                (c, n) for c, n in self._recent_matches if n != best_name
            ]
            self._recent_matches.insert(0, (q.copy(), best_name))
            if len(self._recent_matches) > self._recent_max:
                self._recent_matches.pop()

        return best_name, best_score, margin

    def _find_best_match_excluding(self, embedding: np.ndarray, exclude_name: str) -> Tuple[Optional[str], float]:
        if not self._known_faces:
            return None, 0.0
        if self._bank_dirty:
            self._rebuild_embedding_bank()
        if self._emb_bank is None:
            return None, 0.0
        q = embedding.flatten().astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = q @ self._emb_bank.T                         # (N,)
        # Mask out the excluded identity
        mask = np.array([name != exclude_name for name in self._bank_labels], dtype=bool)
        if not mask.any():
            return None, 0.0
        scores[~mask] = -1.0
        best_idx = int(np.argmax(scores))
        return self._bank_labels[best_idx], float(scores[best_idx])

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
                print(f"[Gatekeeper] Track {tid} refinement expired — keeping '{name}' as-is.")
                self._enrolled_tracks.add(tid)
            else:
                print(f"[Gatekeeper] Candidate Track {tid} expired — discarded.")
            self._pending.pop(tid)


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE API FUNCTIONS
#  (mirror the Gatekeeper's internals for external callers / unit tests)
# ═══════════════════════════════════════════════════════════════════════════════

def match_identity(
    embedding:   np.ndarray,
    known_faces: Dict[str, List[np.ndarray]],
) -> Tuple[Optional[str], float]:
    """Compare against ALL embeddings of every identity. Return (name, best_score).
    Uses vectorized cosine similarity for speed."""
    if not known_faces:
        return None, 0.0
    all_embs = []
    labels = []
    for name, emb_list in known_faces.items():
        for emb in emb_list:
            all_embs.append(emb.flatten())
            labels.append(name)
    if not all_embs:
        return None, 0.0
    bank = np.array(all_embs, dtype=np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9
    bank = bank / norms
    q = embedding.flatten().astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = q @ bank.T
    best_idx = int(np.argmax(scores))
    return labels[best_idx], float(scores[best_idx])


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
