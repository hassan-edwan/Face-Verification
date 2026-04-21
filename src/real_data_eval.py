"""
Real-data evaluation harness — grounds the synthetic numbers in
actual surveillance / off-angle imagery.

Runs the **full live pipeline** on every image pair:
    cv2.imread → MTCNN.detect_faces → align_face (src/alignment.py)
               → embedder.embed → Gatekeeper.process()

Parallels `src/live_eval.py` in shape — same `ScenarioResult` / aggregate
fields — but operates on paths from `data/real_eval/` instead of
synthesized embeddings. The artifact written to `outputs/runs/run_NNN.json`
carries `eval_type: "real"` (see `live_eval.build_artifact`).

Dataset loaders:
    - SCface       : scface_mugshot_to_dN  (distance 1/2/3)
    - ChokePoint   : chokepoint_within_clip_sanity (same-clip first vs
                     last frame — *not* for decision-making, retained
                     as a near-duplicate sanity signal),
                     chokepoint_cross_camera (same session, different
                     portal camera mid-frames),
                     chokepoint_cross_session (same person, different
                     session — the actual generalization test).

Beyond the gatekeeper-success rate the harness also computes
**TAR @ FAR** per scenario. Genuine scores are cosine(enroll_emb,
query_emb) for same-identity pairs; impostor scores are cosine against
*other* subjects' query images drawn from the same scenario. Scoring
thresholds derived from genuine-only success conflate a better
embedding with a looser match gate — TAR@FAR separates the two.

Designed to be SLOW (minutes per run, MTCNN + ArcFace per image) — call
from a `scripts/models/run_NNN.py` script, not inline anywhere hot.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .alignment import align_face, align_face_5point, TARGET_SIZE
from .embedder  import Embedder, get_embedder
from .gatekeeper import (
    Gatekeeper, GatekeeperDecision, REMATCH_THRESHOLD,
)
from .live_eval  import ScenarioResult, RunReport
from .quality    import quick_quality_check


# ─────────────────────────────────────────────────────────────────────────────
# Config — mirrors the live pipeline's detection knobs so real-data eval
# sees exactly what server.py / live_webcam.py see
# ─────────────────────────────────────────────────────────────────────────────

MIN_FACE_PX     = 20       # run_014: 40→20 — unlocks SCface d1 (18-30 px) + chokepoint bookends
MTCNN_CONF_MIN  = 0.90     # match server.py

# Default data roots. Overridable at callsite so tests can point at fixtures.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "real_eval"


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PairSpec:
    """One enroll → query pair."""
    subject_id:  str
    enroll_path: Path
    query_path:  Path
    scenario_id: str        # e.g. "scface_mugshot_to_d1"
    meta:        Dict[str, str] = field(default_factory=dict)
    # Optional GT left-eye (x, y) in source-image pixels. When set, the
    # detection stage picks the MTCNN face whose own left_eye keypoint is
    # closest to this coordinate (used by chokepoint, where >1 person may
    # appear in the same frame). Unset for SCface (single face per frame
    # → "largest" heuristic is correct).
    enroll_left_eye: Optional[Tuple[float, float]] = None
    query_left_eye:  Optional[Tuple[float, float]] = None


@dataclass
class PairResult:
    spec:              PairSpec
    enroll_ok:         bool
    query_decision:    Optional[str]           # GatekeeperDecision.name or None
    matched_identity:  Optional[str]
    enrolled_identity: Optional[str]
    success:           bool
    time_ms:           float
    error:             str = ""                # "no_face" / "mtcnn_reject" / ""
    # Cosine(enroll_emb, query_emb) when both embeddings succeeded. Used
    # for TAR@FAR alongside impostor scores. None if either side failed
    # MTCNN / min-size / confidence; those pairs can't contribute to
    # either TAR or FAR.
    genuine_score:     Optional[float] = None


@dataclass
class EmbedResult:
    """Cached output of _detect_and_embed + quality scoring. Shared
    between the gatekeeper pass and the TAR@FAR pass so every image is
    only embedded once per scenario run."""
    embedding:     np.ndarray
    aligned_bgr:   np.ndarray
    quality_score: float      # quick_quality_check(aligned_bgr); 0.0 if gate fails


# (path_str, left_eye_rounded_or_None) → EmbedResult | None (cached miss)
EmbedCache = Dict[Tuple[str, Optional[Tuple[int, int]]], Optional[EmbedResult]]


@dataclass
class RealScenario:
    id:           str
    description:  str
    dataset:      str                # "scface" | "chokepoint" | "tinyface"
    pairs:        List[PairSpec]
    # Oracle operates on the per-pair results and returns (successes, trials).
    # Default: count `success=True` pairs.
    oracle:       Callable[[List[PairResult]], Tuple[int, int]] = None


def _default_oracle(results: List[PairResult]) -> Tuple[int, int]:
    successes = sum(1 for r in results if r.success)
    return successes, len(results)


# ─────────────────────────────────────────────────────────────────────────────
# Model holder — lazy-loaded, shared across scenarios within one run
# ─────────────────────────────────────────────────────────────────────────────


class _Models:
    """Lazy-loaded MTCNN + embedder. Single instance per harness run;
    loading each model costs 5–10 s, so we do it once.

    The embedder is whichever backend `get_embedder()` returns — ArcFace
    (buffalo_l / w600k_r50.onnx) by default, or FaceNet via the
    `FACE_EMBEDDER=facenet` env var."""
    _mtcnn = None
    _embedder: Optional[Embedder] = None

    def mtcnn(self):
        if self._mtcnn is None:
            from mtcnn import MTCNN
            self._mtcnn = MTCNN()
        return self._mtcnn

    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder


# ─────────────────────────────────────────────────────────────────────────────
# Per-image pipeline — one image in, (embedding, quality_score, aligned) out
# ─────────────────────────────────────────────────────────────────────────────


def _detect_and_embed(img_path: Path, models: _Models,
                      expected_left_eye: Optional[Tuple[float, float]] = None,
                      ) -> Optional[EmbedResult]:
    """Run MTCNN → align → embed → quality-score. Returns an
    `EmbedResult` or None if no acceptable face was found.

    When `expected_left_eye` is provided, selects the detection whose own
    `left_eye` MTCNN keypoint is closest to that coordinate (used when a
    frame has multiple faces and we need the specific labeled person).
    Rejects the match if the nearest detection's keypoint is farther than
    `max(bbox_min_side / 3, 20 px)` from the expected coord — that's the
    "MTCNN missed the labeled person" signal.

    Quality score comes from `quick_quality_check(aligned_bgr)`, matching
    the live refinement path. A hard rejection here (score = 0.0) is
    later consumed by the gatekeeper's `MIN_QUALITY_FOR_UPDATE` gate —
    identical live-pipeline semantics."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        detections = models.mtcnn().detect_faces(img_rgb)
    except Exception:
        return None
    if not detections:
        return None

    detections = [d for d in detections if d.get("confidence", 0.0) >= MTCNN_CONF_MIN]
    if not detections:
        return None

    if expected_left_eye is None:
        best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    else:
        ex, ey = expected_left_eye
        def _dist(d):
            kp = (d.get("keypoints") or {}).get("left_eye")
            if kp is None:
                return float("inf")
            return ((kp[0] - ex) ** 2 + (kp[1] - ey) ** 2) ** 0.5
        best, best_dist = min(((d, _dist(d)) for d in detections),
                              key=lambda t: t[1])
        bx, by, bw, bh = best["box"]
        tolerance = max(min(bw, bh) / 3.0, 20.0)
        if best_dist > tolerance:
            return None
    x, y, w, h = best["box"]
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
        return None

    # Pick alignment to match the embedder's input geometry.
    # ArcFace: 5-point / 112×112; FaceNet: 2-point / 160×160.
    embedder = models.embedder()
    if embedder.name == "arcface":
        aligned_bgr = align_face_5point(img_bgr, (x, y, w, h),
                                        best.get("keypoints"))
    else:
        aligned_bgr = align_face(img_bgr, (x, y, w, h),
                                 best.get("keypoints"))
    emb = embedder.embed(aligned_bgr)

    # Real quality score on the aligned crop (used to be hard-coded
    # 0.9 / 0.7 → every frame sailed through the gatekeeper's update
    # gate regardless of blur / exposure). `quick_quality_check` returns
    # (passed, score); on failure we emit 0.0 so MIN_QUALITY_FOR_UPDATE
    # skips the bank update exactly like live.
    passed, qscore = quick_quality_check(aligned_bgr)
    return EmbedResult(embedding=emb, aligned_bgr=aligned_bgr,
                       quality_score=qscore if passed else 0.0)


def _embed_cached(path: Path, left_eye: Optional[Tuple[float, float]],
                  models: _Models, cache: EmbedCache) -> Optional[EmbedResult]:
    """Cached wrapper around `_detect_and_embed`. Keyed by (path,
    rounded-left-eye) so the same image loaded with different GT-eye
    oracles (chokepoint cross-subject frames) isn't wrongly deduped."""
    key = (str(path),
           (int(round(left_eye[0])), int(round(left_eye[1]))) if left_eye else None)
    if key in cache:
        return cache[key]
    result = _detect_and_embed(path, models, expected_left_eye=left_eye)
    cache[key] = result
    return result


def _run_pair(spec: PairSpec, models: _Models,
              cache: Optional[EmbedCache] = None) -> PairResult:
    """Enroll via track_id=1, then query on a fresh track_id=2. Success =
    query returns MATCHED to the identity assigned during enroll.

    Also records `genuine_score` — the raw cosine(enroll_emb, query_emb)
    — so the caller can aggregate TAR@FAR across the scenario without
    re-embedding."""
    t0 = time.monotonic()
    gk = Gatekeeper(known_faces={}, person_counter=1)
    cache = cache if cache is not None else {}

    # Suppress Gatekeeper's FAST/REFINED print spam.
    silenced = io.StringIO()

    enrolled_identity: Optional[str] = None
    enroll_ok = False
    query_decision: Optional[str] = None
    matched_identity: Optional[str] = None
    error = ""
    genuine_score: Optional[float] = None

    with contextlib.redirect_stdout(silenced):
        enroll_data = _embed_cached(spec.enroll_path, spec.enroll_left_eye,
                                    models, cache)
        if enroll_data is None:
            error = "no_face_on_enroll"
        else:
            res = gk.process(embedding=enroll_data.embedding, track_id=1,
                             face_bgr=enroll_data.aligned_bgr,
                             quality_score=enroll_data.quality_score)
            enrolled_identity = res.identity
            enroll_ok = res.decision in (
                GatekeeperDecision.FAST_ENROLLED,
                GatekeeperDecision.REFINED,
                GatekeeperDecision.MATCHED,
            )

            if enroll_ok:
                query_data = _embed_cached(spec.query_path, spec.query_left_eye,
                                           models, cache)
                if query_data is None:
                    error = "no_face_on_query"
                else:
                    qres = gk.process(embedding=query_data.embedding, track_id=2,
                                      face_bgr=query_data.aligned_bgr,
                                      quality_score=query_data.quality_score)
                    query_decision   = qres.decision.name
                    matched_identity = qres.identity
                    genuine_score    = _cosine(enroll_data.embedding,
                                               query_data.embedding)
            else:
                error = "enroll_rejected"

    success = (enroll_ok
               and query_decision == GatekeeperDecision.MATCHED.name
               and matched_identity is not None
               and matched_identity == enrolled_identity)

    return PairResult(
        spec=spec,
        enroll_ok=enroll_ok,
        query_decision=query_decision,
        matched_identity=matched_identity,
        enrolled_identity=enrolled_identity,
        success=success,
        time_ms=(time.monotonic() - t0) * 1000.0,
        error=error,
        genuine_score=genuine_score,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalize then dot. Matches the gatekeeper's `_find_best_match`
    scoring so TAR@FAR thresholds line up with what the live path sees."""
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    af /= (np.linalg.norm(af) + 1e-9)
    bf /= (np.linalg.norm(bf) + 1e-9)
    return float(np.dot(af, bf))


def _tar_at_far(genuines: List[float], impostors: List[float],
                far_targets: Tuple[float, ...] = (0.01, 0.001)
                ) -> Dict[str, Optional[Dict[str, float]]]:
    """Compute TAR at each target FAR. Returns
    `{str(far): {"threshold": T, "tar": R}}` (or None when sample size
    is too thin to estimate).

    Threshold T is chosen as the k-th largest impostor score where
    k = ceil(far · n_impostors) — the smallest T that admits at most
    `far` of the impostor pool. TAR is the fraction of genuines ≥ T."""
    out: Dict[str, Optional[Dict[str, float]]] = {}
    if not genuines or not impostors:
        return {f"{far:g}": None for far in far_targets}
    imp_sorted = sorted(impostors, reverse=True)
    n_imp = len(imp_sorted)
    for far in far_targets:
        k = max(1, int(np.ceil(far * n_imp)))
        # If the pool is too small to resolve this FAR (e.g. FAR=0.001
        # with only 100 impostors → k=1, threshold = max impostor →
        # noisy), emit None so the caller doesn't over-interpret.
        if k > n_imp or n_imp < int(1.0 / far):
            out[f"{far:g}"] = None
            continue
        threshold = imp_sorted[k - 1]
        tar = sum(1 for g in genuines if g >= threshold) / len(genuines)
        out[f"{far:g}"] = {"threshold": float(threshold), "tar": float(tar)}
    return out


def _compute_scenario_tar_at_far(results: List[PairResult],
                                 cache: EmbedCache,
                                 max_impostors: int = 20000,
                                 seed: int = 0,
                                 ) -> Dict:
    """Build the genuine + impostor score distributions for one scenario
    and summarize as TAR@FAR.

    Genuines: `PairResult.genuine_score` for every pair where both sides
    embedded successfully.

    Impostors: cross-subject cosine over the same pool of successfully-
    embedded (enroll, query) pairs. For (enroll_A, query_B) with A≠B we
    draw exhaustively unless the count would exceed `max_impostors`, in
    which case we sample uniformly at random with `seed`. Both members
    of an impostor pair must come from the same scenario, so the match
    conditions (distance, camera, session) are controlled — the score
    reflects identity discrimination, not apparatus drift."""
    # Collect (subject_id, enroll_emb, query_emb) for pairs that embedded.
    usable: List[Tuple[str, np.ndarray, np.ndarray]] = []
    genuines: List[float] = []
    for r in results:
        if r.genuine_score is None:
            continue
        e = cache.get((str(r.spec.enroll_path),
                       (int(round(r.spec.enroll_left_eye[0])),
                        int(round(r.spec.enroll_left_eye[1])))
                       if r.spec.enroll_left_eye else None))
        q = cache.get((str(r.spec.query_path),
                       (int(round(r.spec.query_left_eye[0])),
                        int(round(r.spec.query_left_eye[1])))
                       if r.spec.query_left_eye else None))
        if e is None or q is None:
            continue
        usable.append((r.spec.subject_id, e.embedding, q.embedding))
        genuines.append(r.genuine_score)

    # Exhaustive enroll×query grid minus diagonal (same subject).
    rng = random.Random(seed)
    n = len(usable)
    impostors: List[float] = []
    if n < 2:
        tar = _tar_at_far(genuines, impostors)
        return {"n_genuines": len(genuines), "n_impostors": 0,
                "tar_at_far": tar}

    total_off_diag = n * (n - 1)
    if total_off_diag <= max_impostors:
        pairs_iter = ((i, j) for i in range(n) for j in range(n) if i != j)
    else:
        # Uniform-random sample of off-diagonal (i, j) without replacement.
        # For large N (>20k), the reservoir is costly — fall back to
        # random.sample over a flat index range and convert back.
        flat = rng.sample(range(total_off_diag), max_impostors)
        def _to_ij(k: int) -> Tuple[int, int]:
            i, j = divmod(k, n - 1)
            if j >= i:
                j += 1  # skip diagonal
            return i, j
        pairs_iter = (_to_ij(k) for k in flat)

    for i, j in pairs_iter:
        sid_i, e_i, _ = usable[i]
        sid_j, _, q_j = usable[j]
        if sid_i == sid_j:
            continue  # belt-and-braces; grid already excludes i==j
        impostors.append(_cosine(e_i, q_j))

    return {
        "n_genuines":  len(genuines),
        "n_impostors": len(impostors),
        "tar_at_far":  _tar_at_far(genuines, impostors),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

_SCFACE_FILENAME_RE = re.compile(r"^(\d{3})_cam(\d)_(\d)\.jpe?g$", re.IGNORECASE)


def load_scface_scenarios(root: Path, max_subjects: Optional[int] = None
                          ) -> List[RealScenario]:
    """SCface: enroll with cropped frontal mugshot, query surveillance at each
    distance bucket. Camera choice within a distance is deterministic (cam1)
    so baseline numbers are reproducible run-to-run."""
    scface_root = root / "scface"
    if not scface_root.exists():
        return []

    mugshot_dir      = scface_root / "mugshot"
    surveillance_dir = scface_root / "surveillance"
    if not (mugshot_dir.exists() and surveillance_dir.exists()):
        return []

    # Discover subjects from mugshots — 001_frontal.JPG → "001".
    mugshots: Dict[str, Path] = {}
    for p in sorted(mugshot_dir.iterdir()):
        m = re.match(r"^(\d{3})_frontal\.jpe?g$", p.name, re.IGNORECASE)
        if m:
            mugshots[m.group(1)] = p

    if max_subjects:
        keep_ids = sorted(mugshots.keys())[:max_subjects]
        mugshots = {k: mugshots[k] for k in keep_ids}

    # Bucket surveillance by (subject, distance, cam).
    bucket: Dict[Tuple[str, str, str], Path] = {}
    for p in sorted(surveillance_dir.iterdir()):
        m = _SCFACE_FILENAME_RE.match(p.name)
        if m:
            subj, cam, dist = m.group(1), m.group(2), m.group(3)
            bucket[(subj, dist, cam)] = p

    # Build one scenario per distance. For each subject, pick the lowest
    # cam number available at that distance (usually cam1) so the choice is
    # deterministic and comparable across runs.
    scenarios: List[RealScenario] = []
    for dist, label in (("1", "d1_far"), ("2", "d2_mid"), ("3", "d3_close")):
        pairs: List[PairSpec] = []
        for subj, enroll in mugshots.items():
            candidates = sorted(
                (cam for (s, d, cam) in bucket if s == subj and d == dist)
            )
            if not candidates:
                continue
            cam = candidates[0]
            query = bucket[(subj, dist, cam)]
            pairs.append(PairSpec(
                subject_id=subj,
                enroll_path=enroll,
                query_path=query,
                scenario_id=f"scface_mugshot_to_{label}",
                meta={"dist": dist, "cam": cam},
            ))
        scenarios.append(RealScenario(
            id=f"scface_mugshot_to_{label}",
            description=(
                f"SCface mugshot → surveillance at distance {dist} "
                f"({'4.2m' if dist=='1' else '2.6m' if dist=='2' else '1.0m'}), "
                f"deterministic cam1 choice."
            ),
            dataset="scface",
            pairs=pairs,
            oracle=_default_oracle,
        ))
    return scenarios


def _parse_chokepoint_gt(xml_path: Path
                         ) -> Dict[str, List[Tuple[int,
                                                   Tuple[float, float],
                                                   Tuple[float, float]]]]:
    """Parse one ChokePoint groundtruth XML. Returns
    {person_id: [(frame_num, (left_eye_xy), (right_eye_xy)), ...]} sorted
    by frame. Frames without any <person> tag (nobody in frame) and
    persons missing either eye coord are dropped."""
    per_person: Dict[str, List[Tuple[int,
                                     Tuple[float, float],
                                     Tuple[float, float]]]] = {}
    tree = ET.parse(str(xml_path))
    for frame in tree.getroot().findall("frame"):
        try:
            frame_num = int(frame.get("number"))
        except (TypeError, ValueError):
            continue
        for person in frame.findall("person"):
            pid = person.get("id")
            le = person.find("leftEye")
            re = person.find("rightEye")
            if pid is None or le is None or re is None:
                continue
            try:
                le_xy = (float(le.get("x")), float(le.get("y")))
                re_xy = (float(re.get("x")), float(re.get("y")))
            except (TypeError, ValueError):
                continue
            per_person.setdefault(pid, []).append((frame_num, le_xy, re_xy))
    for pid in per_person:
        per_person[pid].sort(key=lambda t: t[0])
    return per_person


def load_chokepoint_scenarios(root: Path, max_sessions: Optional[int] = None
                              ) -> List[RealScenario]:
    """ChokePoint with per-frame GT (<person id=>, <leftEye>, <rightEye>).

    Three scenarios:
      - chokepoint_within_clip_sanity : per labeled person, enroll on
        their first visible frame, query on their last, same camera,
        same session. This is two frames of one ~1–2 s walk — near-
        duplicate matching, **not** a generalization test. Retained as
        a regression-floor sanity signal; do not use for decision-making.
        (Former name: chokepoint_temporal_stability — renamed to reflect
        what it actually measures.)
      - chokepoint_cross_camera : per person visible in ≥2 portal
        cameras of the same session, enroll mid-frame of the lowest-
        numbered camera, query mid-frame of the next. Tests pose
        change across simultaneous viewpoints.
      - chokepoint_cross_session : per person ID appearing in ≥2
        different sessions, enroll mid-frame of the earliest session,
        query mid-frame of a later one. Different walk, different time,
        often different camera set. This is the true re-identification
        test — and assumes ChokePoint's convention that `<person id>`
        is consistent across sessions for the same physical walker."""
    choke_root = root / "chokepoint"
    gt_dir     = choke_root / "groundtruth"
    if not (choke_root.exists() and gt_dir.exists()):
        return []

    # session_cam ("P1E_S1_C1") -> {pid: [(frame, le, re), ...]}
    per_sc: Dict[str, Dict[str, List[Tuple[int,
                                           Tuple[float, float],
                                           Tuple[float, float]]]]] = {}
    for xml_path in sorted(gt_dir.glob("*.xml")):
        per_sc[xml_path.stem] = _parse_chokepoint_gt(xml_path)

    if max_sessions:
        keep = sorted(per_sc.keys())[:max_sessions]
        per_sc = {k: per_sc[k] for k in keep}

    def _frame_path(session_cam: str, frame_num: int) -> Path:
        # Physical layout: chokepoint/P1E_S1/P1E_S1_C1/P1E_S1_C1/00000233.jpg
        session = session_cam.rsplit("_", 1)[0]
        return (choke_root / session / session_cam / session_cam /
                f"{frame_num:08d}.jpg")

    # ---- Scenario 1: within-clip sanity (same camera, first vs last) ----
    # Near-duplicate frames from one walk — kept as a sanity floor, not a
    # generalization signal. See docstring.
    within_clip_pairs: List[PairSpec] = []
    for sc, person_dict in per_sc.items():
        for pid, frames in person_dict.items():
            if len(frames) < 2:
                continue
            first, last = frames[0], frames[-1]
            if first[0] == last[0]:
                continue
            enroll_path = _frame_path(sc, first[0])
            query_path  = _frame_path(sc, last[0])
            if not (enroll_path.exists() and query_path.exists()):
                continue
            within_clip_pairs.append(PairSpec(
                subject_id=f"{sc}_{pid}",
                enroll_path=enroll_path,
                query_path=query_path,
                scenario_id="chokepoint_within_clip_sanity",
                enroll_left_eye=first[1],
                query_left_eye=last[1],
                meta={"session_cam": sc, "person_id": pid,
                      "enroll_frame": f"{first[0]:08d}",
                      "query_frame":  f"{last[0]:08d}",
                      "span_frames":  str(last[0] - first[0])},
            ))

    # ---- Scenario 2: cross-camera (same session, different camera) ----
    # Group session_cams by session prefix (P1E_S1 = C1, C2, C3).
    sessions: Dict[str, Dict[str, Tuple[str, Dict[str, List]]]] = {}
    for sc, person_dict in per_sc.items():
        parts = sc.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].startswith("C"):
            continue
        session, cam = parts[0], parts[1]
        sessions.setdefault(session, {})[cam] = (sc, person_dict)

    cross_pairs: List[PairSpec] = []
    for session, cams in sessions.items():
        if len(cams) < 2:
            continue
        # pid -> {cam: (session_cam, frames)}
        pid_by_cam: Dict[str, Dict[str, Tuple[str, List]]] = {}
        for cam, (sc, person_dict) in cams.items():
            for pid, frames in person_dict.items():
                pid_by_cam.setdefault(pid, {})[cam] = (sc, frames)
        for pid, cam_data in pid_by_cam.items():
            if len(cam_data) < 2:
                continue
            sorted_cams = sorted(cam_data.keys())
            enroll_cam, query_cam = sorted_cams[0], sorted_cams[1]
            enroll_sc, enroll_frames = cam_data[enroll_cam]
            query_sc,  query_frames  = cam_data[query_cam]
            # Mid-frame = person is fully in view (not entering / leaving).
            ef = enroll_frames[len(enroll_frames) // 2]
            qf = query_frames[len(query_frames) // 2]
            enroll_path = _frame_path(enroll_sc, ef[0])
            query_path  = _frame_path(query_sc,  qf[0])
            if not (enroll_path.exists() and query_path.exists()):
                continue
            cross_pairs.append(PairSpec(
                subject_id=f"{session}_{pid}",
                enroll_path=enroll_path,
                query_path=query_path,
                scenario_id="chokepoint_cross_camera",
                enroll_left_eye=ef[1],
                query_left_eye=qf[1],
                meta={"session": session, "person_id": pid,
                      "enroll_cam": enroll_cam, "query_cam": query_cam,
                      "enroll_frame": f"{ef[0]:08d}",
                      "query_frame":  f"{qf[0]:08d}"},
            ))

    # ---- Scenario 3: cross-session (same person, different walk) ----
    # Assumes ChokePoint's convention: <person id=> is consistent across
    # sessions for the same physical walker. If that assumption is wrong,
    # success rates here will collapse to near-zero and flag it loudly.
    # Group frames by raw pid, recording (session, session_cam, frames).
    pid_to_sessions: Dict[str, Dict[str, List[Tuple[str, List]]]] = {}
    for sc, person_dict in per_sc.items():
        parts = sc.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].startswith("C"):
            continue
        session = parts[0]
        for pid, frames in person_dict.items():
            if not frames:
                continue
            pid_to_sessions.setdefault(pid, {}) \
                           .setdefault(session, []) \
                           .append((sc, frames))

    cross_session_pairs: List[PairSpec] = []
    for pid, session_map in pid_to_sessions.items():
        if len(session_map) < 2:
            continue
        sorted_sessions = sorted(session_map.keys())
        enroll_session = sorted_sessions[0]
        query_session  = sorted_sessions[1]  # nearest other session; deterministic
        # Pick the camera within each session with the most frames (most
        # likely mid-walk). Ties broken by lowest cam number.
        def _pick_cam(clips: List[Tuple[str, List]]) -> Tuple[str, List]:
            return max(clips, key=lambda t: (len(t[1]), -int(t[0][-1])))
        e_sc, e_frames = _pick_cam(session_map[enroll_session])
        q_sc, q_frames = _pick_cam(session_map[query_session])
        ef = e_frames[len(e_frames) // 2]
        qf = q_frames[len(q_frames) // 2]
        enroll_path = _frame_path(e_sc, ef[0])
        query_path  = _frame_path(q_sc, qf[0])
        if not (enroll_path.exists() and query_path.exists()):
            continue
        cross_session_pairs.append(PairSpec(
            subject_id=pid,  # raw pid; deliberately NOT session-prefixed
            enroll_path=enroll_path,
            query_path=query_path,
            scenario_id="chokepoint_cross_session",
            enroll_left_eye=ef[1],
            query_left_eye=qf[1],
            meta={"person_id": pid,
                  "enroll_session": enroll_session, "enroll_session_cam": e_sc,
                  "query_session":  query_session,  "query_session_cam":  q_sc,
                  "enroll_frame": f"{ef[0]:08d}",
                  "query_frame":  f"{qf[0]:08d}"},
        ))

    scenarios: List[RealScenario] = []
    if within_clip_pairs:
        scenarios.append(RealScenario(
            id="chokepoint_within_clip_sanity",
            description=(
                "ChokePoint: per GT-labeled person, enroll their first "
                "visible frame, query their last. Same camera, same "
                "session — two frames of one walk. Near-duplicate "
                "matching; a sanity floor, NOT a generalization test."
            ),
            dataset="chokepoint",
            pairs=within_clip_pairs,
            oracle=_default_oracle,
        ))
    if cross_pairs:
        scenarios.append(RealScenario(
            id="chokepoint_cross_camera",
            description=(
                "ChokePoint: per person visible in ≥2 portal cameras of "
                "the same session, enroll mid-frame of the lowest camera, "
                "query mid-frame of the next. Tests embedding invariance "
                "across simultaneous portal viewpoints."
            ),
            dataset="chokepoint",
            pairs=cross_pairs,
            oracle=_default_oracle,
        ))
    if cross_session_pairs:
        scenarios.append(RealScenario(
            id="chokepoint_cross_session",
            description=(
                "ChokePoint: per person ID appearing in ≥2 sessions, "
                "enroll mid-frame of the earliest session, query mid-"
                "frame of the next. Different walk — the true re-"
                "identification test."
            ),
            dataset="chokepoint",
            pairs=cross_session_pairs,
            oracle=_default_oracle,
        ))
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Harness entry points
# ─────────────────────────────────────────────────────────────────────────────


def run_real_scenario(scenario: RealScenario, models: _Models,
                      progress: bool = True,
                      tar_at_far_seed: int = 0,
                      ) -> Tuple[ScenarioResult, Dict]:
    """Run one scenario through the full pipeline. `progress` prints one
    line per 10 % of pairs — helps when a run takes minutes.

    Returns `(scenario_result, diagnostics)`. `diagnostics` carries
    per-scenario TAR@FAR and sample sizes — the shape consumed by
    `run_real_all` to populate `aggregate["scenario_diagnostics"]`."""
    results: List[PairResult] = []
    cache: EmbedCache = {}
    n = len(scenario.pairs)
    milestones = {max(1, n // 10 * k) for k in range(1, 11)}
    for i, spec in enumerate(scenario.pairs, start=1):
        results.append(_run_pair(spec, models, cache=cache))
        if progress and i in milestones:
            successes = sum(1 for r in results if r.success)
            print(f"  {scenario.id}: {i}/{n}  running={successes}/{i} "
                  f"({successes/i:.0%})")
    successes, trials = (scenario.oracle or _default_oracle)(results)
    err_count = sum(1 for r in results if r.error)
    diag = _compute_scenario_tar_at_far(results, cache, seed=tar_at_far_seed)
    notes_parts = []
    if err_count:
        notes_parts.append(
            f"{err_count} pair(s) skipped with error (no face / mtcnn reject)"
        )
    tar01 = diag["tar_at_far"].get("0.01")
    if tar01:
        notes_parts.append(
            f"TAR@FAR=1%: {tar01['tar']:.2f} (n_gen={diag['n_genuines']}, "
            f"n_imp={diag['n_impostors']})"
        )
    scenario_result = ScenarioResult(
        id=scenario.id,
        trials=trials,
        successes=successes,
        success_rate=(successes / trials) if trials else 0.0,
        notes=" · ".join(notes_parts),
    )
    return scenario_result, diag


def run_real_all(datasets: Optional[List[str]] = None,
                 data_root: Optional[Path] = None,
                 max_subjects: Optional[int] = None,
                 max_sessions: Optional[int] = None,
                 models: Optional[_Models] = None,
                 ) -> Tuple[RunReport, List[str]]:
    """Run every scenario discovered in `datasets` (default: scface +
    chokepoint — tinyface is unsupported and no longer part of the
    default list). Returns (report, datasets_actually_run).

    Per-scenario TAR@FAR diagnostics are threaded into
    `report.aggregate["scenario_diagnostics"]`, a dict keyed by scenario
    id. Callers write that to the run artifact unchanged."""
    datasets  = datasets or ["scface", "chokepoint"]
    data_root = data_root or DEFAULT_DATA_ROOT
    models    = models or _Models()

    scenarios: List[RealScenario] = []
    used_datasets: List[str] = []
    if "scface" in datasets:
        s = load_scface_scenarios(data_root, max_subjects=max_subjects)
        if s:
            scenarios.extend(s)
            used_datasets.append("scface")
    if "chokepoint" in datasets:
        s = load_chokepoint_scenarios(data_root, max_sessions=max_sessions)
        if s:
            scenarios.extend(s)
            used_datasets.append("chokepoint")

    print(f"[real_data_eval] running {len(scenarios)} scenario(s): "
          f"{', '.join(s.id for s in scenarios)}")
    results: List[ScenarioResult] = []
    scenario_diagnostics: Dict[str, Dict] = {}
    for sc in scenarios:
        print(f"[real_data_eval] starting {sc.id} ({len(sc.pairs)} pairs)...")
        r, diag = run_real_scenario(sc, models)
        results.append(r)
        scenario_diagnostics[sc.id] = diag

    total_trials    = sum(r.trials for r in results)
    total_successes = sum(r.successes for r in results)
    aggregate = {
        "total_trials":    total_trials,
        "total_successes": total_successes,
        "success_rate":    (total_successes / total_trials) if total_trials else 0.0,
        "scenario_diagnostics": scenario_diagnostics,
    }
    return RunReport(scenarios=results, aggregate=aggregate), used_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Gallery-N identification harness (Phase 2 hardening)
# ─────────────────────────────────────────────────────────────────────────────
#
# The verification harness above tests each (enroll, query) pair against a
# gallery of ONE — which cannot distinguish a better embedder from a looser
# match threshold, and cannot expose near-impostor failure modes. This
# identification harness enrolls every subject in a scenario into a single
# Gatekeeper and then queries each probe against that full gallery. Reports:
#
#   - rank1_gk_accept_rate : what the live pipeline actually delivers —
#     gk._find_best_match returns the right subject AND its score clears
#     REMATCH_THRESHOLD. This is the production number.
#   - rank1_emb_accept_rate / rank5 / rank10 : embedder-only capacity. True
#     subject is the argmax (or in the top-k) of cosine against the bank,
#     ignoring REMATCH. Gap vs rank1_gk isolates how much identification
#     error is due to the match threshold vs the embedder itself.
#
# We call `gk._find_best_match` directly rather than `gk.process(...)` on
# queries — process() would mutate gallery state (FAST_ENROLLED a ghost
# Person N+1 every time the score fell below REMATCH), poisoning subsequent
# queries. `_find_best_match` is side-effect-free and matches what the
# live path's MATCHED branch consumes.


@dataclass
class IdentificationSpec:
    """One gallery-N identification scenario.

    `enrollments` is keyed by subject_id and maps to a LIST of
    (path, left_eye) pairs — one entry per enrollment template. Single-
    shot protocols pass length-1 lists. Multi-shot protocols (run_020+)
    pass multiple pose variants so the bank per identity spans more of
    face space. `_enroll_gallery` handles the list two-phase: all
    subjects' template-0 goes through the dedup gate first; extras
    enrich already-seated identities via the tracking-lock update path.
    """
    id:          str
    description: str
    dataset:     str
    # subject_id -> [(enroll_image_path, optional GT left-eye), ...]
    enrollments: Dict[str, List[Tuple[Path, Optional[Tuple[float, float]]]]]
    # (true_subject_id, query_image_path, optional GT left-eye)
    queries:     List[Tuple[str, Path, Optional[Tuple[float, float]]]]


@dataclass
class IdentificationResult:
    id:               str
    n_enrolled:       int    # subjects actually seated in the gallery
    n_queries:        int    # probe images presented
    n_embed_success:  int    # queries where MTCNN + embedding succeeded
    rank1_gk_accept:  int    # _find_best_match returns true_subject AND score >= REMATCH
    rank1_emb:        int    # true subject is nearest by cosine (threshold-free)
    rank5_emb:        int    # true subject is in the top-5 by cosine
    rank10_emb:       int    # top-10
    notes:            str = ""


def _enroll_gallery(enrollments: Dict[str, List[Tuple[Path,
                                                      Optional[Tuple[float,
                                                                     float]]]]],
                    models: _Models, cache: EmbedCache,
                    ) -> Tuple[Gatekeeper, List[str], int, int, int, int]:
    """Enroll every subject into a fresh Gatekeeper. Two-phase:
      - Phase 1: each subject's first template goes through the
        `gk.process()` first-contact decision (dedup check against
        other already-seated frontals). Same contract as single-shot.
      - Phase 2: for each successfully-seated subject, additional
        templates go through `gk.process()` on the *same* track_id,
        hitting the tracking-lock fast path at `gatekeeper.py:137`
        which calls `_try_update_embeddings`. That's the same code
        path the live pipeline uses to refine the bank across
        multiple quality frames of one walker.

    Interleaving phases would wreck dedup: subject K+1's Phase-1
    match against the bank would see all prior subjects' richer multi-
    pose banks and inflate the max-over-templates cosine, increasing
    collision risk. Two-phase keeps Phase-1 dedup identical to single-
    shot regardless of how many templates per subject are added.

    Returns (gk, enrolled_subject_ids, embed_fail, dedup_collisions,
             extra_templates_added, extra_templates_attempted).
    """
    gk = Gatekeeper(known_faces={}, person_counter=1)
    enrolled: List[str] = []
    subj_to_identity: Dict[str, str] = {}
    subj_to_track:    Dict[str, int] = {}
    identity_to_subj: Dict[str, str] = {}
    embed_fail = 0
    dedup_collision = 0

    silenced = io.StringIO()
    with contextlib.redirect_stdout(silenced):
        # ── Phase 1: seat every subject with their first template ────
        for i, (subj, templates) in enumerate(sorted(enrollments.items()),
                                              start=1):
            if not templates:
                embed_fail += 1
                continue
            first_path, first_eye = templates[0]
            emb = _embed_cached(first_path, first_eye, models, cache)
            if emb is None:
                embed_fail += 1
                continue
            res = gk.process(embedding=emb.embedding, track_id=i,
                             face_bgr=emb.aligned_bgr,
                             quality_score=emb.quality_score)
            if res.decision == GatekeeperDecision.FAST_ENROLLED:
                subj_to_identity[subj] = res.identity
                identity_to_subj[res.identity] = subj
                subj_to_track[subj]    = i
                enrolled.append(subj)
            elif res.decision == GatekeeperDecision.MATCHED:
                dedup_collision += 1
            else:
                # UNCERTAIN / REFINING on first-contact — under the
                # margin-aware first-contact rule (run_019) this means
                # the enrolling subject's embedding is ambiguously
                # close to an existing identity. Count as a non-
                # seating drop, separate from a hard embedding failure.
                embed_fail += 1

        # ── Phase 2: enrich each seated subject's bank ───────────────
        extra_added = 0
        extra_attempted = 0
        for subj, templates in sorted(enrollments.items()):
            if subj not in subj_to_identity:
                continue  # not seated in Phase 1, skip enrichment
            track_id = subj_to_track[subj]
            known_name = subj_to_identity[subj]
            for path, eye in templates[1:]:
                extra_attempted += 1
                before = len(gk._known_faces.get(known_name, []))
                emb = _embed_cached(path, eye, models, cache)
                if emb is None:
                    continue
                gk.process(embedding=emb.embedding, track_id=track_id,
                           face_bgr=emb.aligned_bgr,
                           quality_score=emb.quality_score)
                after = len(gk._known_faces.get(known_name, []))
                if after > before:
                    extra_added += 1

    gk.__dict__["_id_subject_map"] = identity_to_subj
    return gk, enrolled, embed_fail, dedup_collision, extra_added, extra_attempted


def _rank_of_truth(gk: Gatekeeper, query_emb: np.ndarray,
                   true_subject: str) -> Tuple[int, str, float]:
    """Compute (rank_of_true_subject, rank1_subject_or_None,
    rank1_cosine_score). Rank is 1-based; returns a large sentinel if
    the true subject isn't represented in the gallery."""
    if gk._bank_dirty:
        gk._rebuild_embedding_bank()
    if gk._emb_bank is None or not gk._bank_labels:
        return 10_000_000, "", 0.0

    q = query_emb.flatten().astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    scores = q @ gk._emb_bank.T  # (N_bank_rows,)

    id_to_subj: Dict[str, str] = gk.__dict__.get("_id_subject_map", {})
    # Collapse to best score per identity (there will usually be one row
    # per identity in the identification setting, but defend against
    # multiple bank rows of the same identity).
    best_per_identity: Dict[str, float] = {}
    for name, s in zip(gk._bank_labels, scores):
        if s > best_per_identity.get(name, -1.0):
            best_per_identity[name] = float(s)

    # Sort identities by best score, descending.
    ranked = sorted(best_per_identity.items(), key=lambda x: -x[1])

    rank1_identity, rank1_score = ranked[0]
    rank1_subj = id_to_subj.get(rank1_identity, rank1_identity)

    # Find the true subject's rank.
    true_rank = 10_000_000
    for i, (ident, _) in enumerate(ranked, start=1):
        if id_to_subj.get(ident) == true_subject:
            true_rank = i
            break
    return true_rank, rank1_subj, rank1_score


def run_identification(spec: IdentificationSpec, models: _Models,
                       progress: bool = True,
                       ) -> Tuple[ScenarioResult, Dict]:
    """Run one gallery-N identification scenario. Mirrors the shape of
    `run_real_scenario` so `run_identification_all` can compose both."""
    cache: EmbedCache = {}

    total_templates = sum(len(v) for v in spec.enrollments.values())
    shot_shape = (f"multi-shot (~{total_templates / max(1, len(spec.enrollments)):.1f} "
                  f"templates/subject)"
                  if total_templates > len(spec.enrollments)
                  else "single-shot")
    print(f"  [ident] enrolling gallery of {len(spec.enrollments)} subjects "
          f"[{shot_shape}]...")
    gk, enrolled, enroll_fail, collisions, extras_added, extras_attempted = \
        _enroll_gallery(spec.enrollments, models, cache)
    enrolled_set = set(enrolled)
    n_enrolled = len(enrolled)
    print(f"  [ident] gallery seated: {n_enrolled} subjects "
          f"({enroll_fail} enroll embed-fails, {collisions} dedup collisions, "
          f"{extras_added}/{extras_attempted} extra templates added)")

    n_queries = len(spec.queries)
    n_embed_success = 0
    rank1_gk_accept = 0
    rank1_emb = 0
    rank5_emb = 0
    rank10_emb = 0
    skipped_no_enroll = 0  # queries whose true subject never seated in gallery

    milestones = {max(1, n_queries // 10 * k) for k in range(1, 11)}
    silenced = io.StringIO()
    with contextlib.redirect_stdout(silenced):
        for i, (true_subj, qpath, qleye) in enumerate(spec.queries, start=1):
            if true_subj not in enrolled_set:
                skipped_no_enroll += 1
                continue
            q = _embed_cached(qpath, qleye, models, cache)
            if q is None:
                continue
            n_embed_success += 1

            true_rank, r1_subj, r1_score = _rank_of_truth(
                gk, q.embedding, true_subj
            )
            if true_rank == 1:
                rank1_emb += 1
            if true_rank <= 5:
                rank5_emb += 1
            if true_rank <= 10:
                rank10_emb += 1
            # rank1_gk_accept: live pipeline would MATCH (score >= REMATCH)
            # AND the top-1 identity is the true subject.
            if r1_subj == true_subj and r1_score >= REMATCH_THRESHOLD:
                rank1_gk_accept += 1

            if progress and i in milestones:
                print(f"  {spec.id}: {i}/{n_queries}  "
                      f"rank1_gk={rank1_gk_accept}/{i} "
                      f"({rank1_gk_accept/i:.0%})")

    # Effective trial denominator is queries with both an enrolled subject
    # and a successful query embedding. Skipped queries (true subject not
    # in gallery) are excluded — they can't be rank-1 correct by construction.
    trials = n_embed_success
    notes_bits = [
        f"gallery={n_enrolled}",
        f"rank1_gk={rank1_gk_accept}/{trials} "
        f"({(rank1_gk_accept/trials if trials else 0):.2f})",
        f"rank1_emb={rank1_emb}/{trials} "
        f"({(rank1_emb/trials if trials else 0):.2f})",
        f"rank5_emb={rank5_emb}/{trials} "
        f"({(rank5_emb/trials if trials else 0):.2f})",
    ]
    if enroll_fail or collisions:
        notes_bits.append(
            f"enroll: {enroll_fail} embed-fail, {collisions} dedup-collision"
        )
    if skipped_no_enroll:
        notes_bits.append(f"{skipped_no_enroll} queries skipped (subject not in gallery)")

    scenario_result = ScenarioResult(
        id=spec.id,
        trials=trials,
        successes=rank1_gk_accept,
        success_rate=(rank1_gk_accept / trials) if trials else 0.0,
        notes=" · ".join(notes_bits),
    )
    diag = {
        "n_enrolled":        n_enrolled,
        "n_queries":         n_queries,
        "n_embed_success":   n_embed_success,
        "rank1_gk_accept":   rank1_gk_accept,
        "rank1_emb":         rank1_emb,
        "rank5_emb":         rank5_emb,
        "rank10_emb":        rank10_emb,
        "enroll_embed_fail": enroll_fail,
        "dedup_collisions":  collisions,
        "extra_templates_added":     extras_added,
        "extra_templates_attempted": extras_attempted,
        "skipped_no_enroll": skipped_no_enroll,
        "rank1_gk_accept_rate":   (rank1_gk_accept / trials) if trials else 0.0,
        "rank1_emb_accept_rate":  (rank1_emb / trials) if trials else 0.0,
        "rank5_emb_accept_rate":  (rank5_emb / trials) if trials else 0.0,
        "rank10_emb_accept_rate": (rank10_emb / trials) if trials else 0.0,
    }
    return scenario_result, diag


def build_scface_identification_specs(root: Path,
                                      max_subjects: Optional[int] = None,
                                      ) -> List[IdentificationSpec]:
    """Collapse the verification-style SCface scenarios into gallery-N
    identification specs: enrollments = every subject's frontal mugshot
    (single-shot, length-1 template list), queries = each subject's
    cam1 image at the named distance."""
    verif_scenarios = load_scface_scenarios(root, max_subjects=max_subjects)
    specs: List[IdentificationSpec] = []
    for sc in verif_scenarios:
        enrollments: Dict[str, List[Tuple[Path,
                                          Optional[Tuple[float, float]]]]] = {}
        queries: List[Tuple[str, Path, Optional[Tuple[float, float]]]] = []
        for pair in sc.pairs:
            enrollments[pair.subject_id] = [(pair.enroll_path,
                                             pair.enroll_left_eye)]
            queries.append((pair.subject_id, pair.query_path,
                            pair.query_left_eye))
        if not enrollments:
            continue
        # id follows the existing naming convention but swaps
        # "scface_mugshot_to_" for "scface_identification_" so artifacts
        # don't collide with the verification-harness rows.
        ident_id = sc.id.replace("scface_mugshot_to_",
                                 "scface_identification_")
        specs.append(IdentificationSpec(
            id=ident_id,
            description=(sc.description
                         + " Gallery-N identification: every subject's "
                           "mugshot enrolled into one gatekeeper, each "
                           "query ranked against the full gallery."),
            dataset="scface",
            enrollments=enrollments,
            queries=queries,
        ))
    return specs


_SCFACE_ROTATION_RE = re.compile(r"^(\d{3})_([a-zA-Z]\w*)\.jpe?g$",
                                 re.IGNORECASE)

DEFAULT_MULTISHOT_POSES = ("frontal", "L1", "L4", "R1", "R4")


def load_scface_mugshot_rotation(root: Path,
                                 poses: Tuple[str, ...] = DEFAULT_MULTISHOT_POSES,
                                 ) -> Dict[str, List[Tuple[Path, str]]]:
    """Walk `scface/mugshot_rotation/` and return
    `{subject_id: [(path, pose_label), ...]}` for the requested poses.
    Files are named `NNN_{pose}.jpg` with pose in
    {frontal, L1-L4, R1-R4}. Subjects missing some requested poses
    keep whatever they have — callers decide what to do with partial
    templates."""
    mug_rot = root / "scface" / "mugshot_rotation"
    if not mug_rot.exists():
        return {}
    pose_set_lower = {p.lower() for p in poses}
    per_subject: Dict[str, List[Tuple[Path, str]]] = {}
    for p in sorted(mug_rot.iterdir()):
        m = _SCFACE_ROTATION_RE.match(p.name)
        if not m:
            continue
        subj, pose = m.group(1), m.group(2).lower()
        if pose in pose_set_lower:
            per_subject.setdefault(subj, []).append((p, pose))
    # Preserve the requested pose order for determinism (frontal first so
    # Phase 1's dedup check always runs against frontals).
    order = {p.lower(): i for i, p in enumerate(poses)}
    for subj in per_subject:
        per_subject[subj].sort(key=lambda t: order[t[1]])
    return per_subject


def build_scface_multishot_identification_specs(
    root: Path,
    poses: Tuple[str, ...] = DEFAULT_MULTISHOT_POSES,
    max_subjects: Optional[int] = None,
) -> List[IdentificationSpec]:
    """Gallery-N identification with multi-pose enrollment. Each subject
    gets N templates from `mugshot_rotation/` (default: frontal, L1, L4,
    R1, R4). Queries remain the same cam1 surveillance stills at each
    distance bucket, reused from `load_scface_scenarios`.

    Two-phase enrollment semantics are enforced inside `_enroll_gallery`
    — Phase 1 seats every subject with their FRONTAL (dedup-safe);
    Phase 2 appends the remaining poses via the tracking-lock
    update path. The frontal MUST be first in each subject's template
    list; `load_scface_mugshot_rotation` guarantees this by sorting on
    the `poses` order."""
    rotations = load_scface_mugshot_rotation(root, poses=poses)
    if not rotations:
        return []

    if max_subjects:
        keep = sorted(rotations.keys())[:max_subjects]
        rotations = {k: rotations[k] for k in keep}

    # Query side comes from the existing SCface verification scenarios,
    # same cam1 / distance selection as single-shot.
    verif = load_scface_scenarios(root, max_subjects=max_subjects)
    specs: List[IdentificationSpec] = []
    for sc in verif:
        queries: List[Tuple[str, Path, Optional[Tuple[float, float]]]] = []
        enrollments: Dict[str, List[Tuple[Path,
                                          Optional[Tuple[float, float]]]]] = {}
        for pair in sc.pairs:
            if pair.subject_id not in rotations:
                continue  # subject missing mugshot_rotation/ entries
            # enroll_left_eye is None for mugshot_rotation files — SCface
            # GT doesn't ship eye keypoints for these; MTCNN's largest-
            # face heuristic is correct (single subject per mugshot).
            enrollments[pair.subject_id] = [
                (path, None) for path, _pose in rotations[pair.subject_id]
            ]
            queries.append((pair.subject_id, pair.query_path,
                            pair.query_left_eye))
        if not enrollments:
            continue
        ident_id = sc.id.replace("scface_mugshot_to_",
                                 "scface_identification_multishot_")
        specs.append(IdentificationSpec(
            id=ident_id,
            description=(sc.description
                         + " Multi-shot identification: each subject "
                           f"enrolled with {len(poses)} pose templates "
                           f"({', '.join(poses)}); query is a single "
                           "cam1 surveillance still."),
            dataset="scface",
            enrollments=enrollments,
            queries=queries,
        ))
    return specs


def run_identification_all(datasets: Optional[List[str]] = None,
                           data_root: Optional[Path] = None,
                           max_subjects: Optional[int] = None,
                           models: Optional[_Models] = None,
                           multishot_poses: Optional[Tuple[str, ...]] = None,
                           ) -> Tuple[RunReport, List[str]]:
    """Identification counterpart to `run_real_all`. Returns a
    `RunReport` whose `aggregate["scenario_diagnostics"]` carries the
    rank-k breakdown per scenario.

    When `multishot_poses` is set, enrollment uses
    `build_scface_multishot_identification_specs` with those pose
    labels (e.g. `("frontal", "L1", "L4", "R1", "R4")`) — each
    subject seats with N templates instead of 1. Scenario IDs are
    renamed `scface_identification_multishot_*` so artifacts don't
    collide with single-shot runs."""
    datasets  = datasets or ["scface"]
    data_root = data_root or DEFAULT_DATA_ROOT
    models    = models or _Models()

    specs: List[IdentificationSpec] = []
    used_datasets: List[str] = []
    if "scface" in datasets:
        if multishot_poses is not None:
            s = build_scface_multishot_identification_specs(
                data_root, poses=multishot_poses, max_subjects=max_subjects
            )
        else:
            s = build_scface_identification_specs(data_root,
                                                  max_subjects=max_subjects)
        if s:
            specs.extend(s)
            used_datasets.append("scface")

    print(f"[real_data_eval.ident] running {len(specs)} identification "
          f"scenario(s): {', '.join(s.id for s in specs)}")
    results: List[ScenarioResult] = []
    scenario_diagnostics: Dict[str, Dict] = {}
    for sp in specs:
        print(f"[real_data_eval.ident] starting {sp.id} "
              f"(gallery={len(sp.enrollments)}, queries={len(sp.queries)})...")
        r, diag = run_identification(sp, models)
        results.append(r)
        scenario_diagnostics[sp.id] = diag

    total_trials    = sum(r.trials for r in results)
    total_successes = sum(r.successes for r in results)
    aggregate = {
        "total_trials":    total_trials,
        "total_successes": total_successes,
        "success_rate":    (total_successes / total_trials) if total_trials else 0.0,
        "scenario_diagnostics": scenario_diagnostics,
    }
    return RunReport(scenarios=results, aggregate=aggregate), used_datasets
