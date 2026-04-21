"""
Synthetic gatekeeper harness — deterministic live-pipeline evaluator.

Drives the real `src.gatekeeper.Gatekeeper.process()` with a scripted
stream of synthetic detection events (embedding + track_id + timing)
so knob tweaks in the live path are measurable without a camera, an
MTCNN run, or a FaceNet inference.

Scope: exercises the gatekeeper state machine — tracking lock, fast
enroll, refinement, re-match, bank policy. **Does not** exercise
MTCNN detection or the FaceNet encoder. Scenarios that depend on
those layers (alignment failures, false negatives from MTCNN) are
out of scope for the synthetic harness and belong in a future
video-replay harness.

Public surface:
    Event, Scenario, ScenarioResult, RunReport
    DEFAULT_SCENARIOS  — dict[str, Scenario]
    run_scenario(gatekeeper, scenario, seed, trials) -> ScenarioResult
    run_all(config_diff=None, trials=25, seed=42) -> RunReport
"""

from __future__ import annotations

import contextlib
import datetime
import io
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .gatekeeper import (
    Gatekeeper,
    GatekeeperDecision,
    MATCH_THRESHOLD,
    REMATCH_THRESHOLD,
    ENROLL_THRESHOLD,
    MIN_MATCH_MARGIN,
    CONSENSUS_FRAMES,
    MAX_EMBEDDINGS_PER_IDENTITY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Event:
    """One synthetic detection tick handed to Gatekeeper.process()."""
    t:             float                 # wall-clock seconds since scenario start
    track_id:      int
    embedding:     np.ndarray            # shape (128,), L2-normalized
    quality_score: float = 0.8           # above MIN_QUALITY_FOR_UPDATE=0.50
    bbox:          tuple  = (0, 0, 100, 100)


@dataclass
class Scenario:
    """Parameterized synthetic test case.

    `build` returns the event sequence for one trial. `oracle` inspects
    the decision timeline (list of per-event (Event, GatekeeperResult)
    tuples) and returns True iff the scenario's success criterion is met."""
    id:          str
    description: str
    build:       Callable[[np.random.Generator], List[Event]]
    oracle:      Callable[[List[tuple]], bool]


@dataclass
class ScenarioResult:
    id:           str
    trials:       int
    successes:    int
    success_rate: float
    notes:        str = ""


@dataclass
class RunReport:
    scenarios:    List[ScenarioResult]
    aggregate:    Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "scenarios": [
                {"id": s.id, "trials": s.trials, "successes": s.successes,
                 "success_rate": s.success_rate, "notes": s.notes}
                for s in self.scenarios
            ],
            "aggregate": self.aggregate,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Embedding synthesis helpers
# ─────────────────────────────────────────────────────────────────────────────


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize."""
    return v / (np.linalg.norm(v) + 1e-9)


def _sample_anchor(rng: np.random.Generator, dim: int = 128) -> np.ndarray:
    """Random unit-norm 128-d vector — stand-in for an 'identity prototype'."""
    return _unit(rng.standard_normal(dim).astype(np.float32))


def _perturb(anchor: np.ndarray, cosine_target: float,
             rng: np.random.Generator) -> np.ndarray:
    """Return a unit vector with cosine similarity ≈ `cosine_target` to
    `anchor`. Constructed as `cos·anchor + sin·orthogonal_noise`, where
    the orthogonal component is sampled from the null space of anchor.

    Cosine of the result with anchor is exactly `cosine_target` up to
    floating-point error — useful for scripting scenarios with a
    specific drift level."""
    cosine_target = float(np.clip(cosine_target, -1.0, 1.0))
    noise = rng.standard_normal(anchor.shape).astype(np.float32)
    # Remove the anchor component (Gram–Schmidt).
    noise -= np.dot(noise, anchor) * anchor
    noise = _unit(noise)
    sin = float(np.sqrt(max(0.0, 1.0 - cosine_target ** 2)))
    return _unit(cosine_target * anchor + sin * noise)


_DUMMY_BGR = np.zeros((1, 1, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Default scenarios (S1–S6)
# ─────────────────────────────────────────────────────────────────────────────
#
# Oracles return True on success, False on failure. They read the decision
# TIMELINE — a list of (Event, GatekeeperResult) — so they can assert
# per-frame invariants, not just end state.

# "Locked to an identity" — any state where the gatekeeper returns a
# non-None identity. REFINING counts: the track is already bound to a
# temp_name while consensus frames accumulate. Only UNCERTAIN (and the
# equally-unhelpful QUALITY_FAIL path, which the harness can't trigger
# since we synthesize embeddings directly) break the lock.
_LOCKED = {GatekeeperDecision.MATCHED,
           GatekeeperDecision.FAST_ENROLLED,
           GatekeeperDecision.REFINED,
           GatekeeperDecision.REFINING}


def _build_S1_single_lock(rng: np.random.Generator) -> List[Event]:
    anchor = _sample_anchor(rng)
    return [
        Event(t=0.1 * i, track_id=1,
              embedding=_perturb(anchor, 0.98, rng),
              quality_score=0.8)
        for i in range(10)
    ]


def _oracle_S1(timeline: List[tuple]) -> bool:
    """Every frame after the first must sit in a terminal lock state."""
    # Frame 0 legitimately hits FAST_ENROLLED; subsequent frames must
    # stay locked (no UNCERTAIN flicker, no re-enrollment).
    for _ev, res in timeline[1:]:
        if res.decision not in _LOCKED:
            return False
    return True


def _build_S2_two_no_cross(rng: np.random.Generator) -> List[Event]:
    a = _sample_anchor(rng)
    b = _sample_anchor(rng)
    events: List[Event] = []
    # Interleave track_id=1 (A) and track_id=2 (B) for 10 frames each.
    for i in range(10):
        t = 0.1 * i
        events.append(Event(t, 1, _perturb(a, 0.98, rng), 0.8))
        events.append(Event(t, 2, _perturb(b, 0.98, rng), 0.8))
    return events


def _oracle_S2(timeline: List[tuple]) -> bool:
    """Track 1 and track 2 must settle on distinct identities."""
    last_by_track: Dict[int, Optional[str]] = {}
    for ev, res in timeline:
        if res.decision in _LOCKED and res.identity:
            last_by_track[ev.track_id] = res.identity
    id1 = last_by_track.get(1)
    id2 = last_by_track.get(2)
    return bool(id1 and id2 and id1 != id2)


def _build_S3_crossing(rng: np.random.Generator) -> List[Event]:
    """Two tracks briefly swap their embeddings mid-sequence — simulates
    the IoU tracker flipping track→face assignment. Baseline gatekeeper
    behavior is to lock each track to whatever identity it first saw,
    so this scenario is expected to fail under the current tracking-lock.
    Future hypotheses loosening the lock should flip it."""
    a = _sample_anchor(rng)
    b = _sample_anchor(rng)
    events: List[Event] = []
    # Phase 1: both tracks stable, 6 frames.
    for i in range(6):
        t = 0.1 * i
        events.append(Event(t, 1, _perturb(a, 0.98, rng), 0.8))
        events.append(Event(t, 2, _perturb(b, 0.98, rng), 0.8))
    # Phase 2: tracks SWAP which face they see, 4 frames.
    for i in range(6, 10):
        t = 0.1 * i
        events.append(Event(t, 1, _perturb(b, 0.98, rng), 0.8))  # track 1 now sees B
        events.append(Event(t, 2, _perturb(a, 0.98, rng), 0.8))  # track 2 now sees A
    return events


def _oracle_S3(timeline: List[tuple]) -> bool:
    """Success = after the swap, at least one track is NOT still claiming
    its pre-swap identity (i.e. the system noticed something was wrong).
    Under the current hard tracking-lock, this will always fail — that's
    the baseline regression this scenario documents."""
    # Partition into pre/post swap by timestamp >= 0.6.
    post = [(ev, res) for ev, res in timeline if ev.t >= 0.6]
    pre_ids = {ev.track_id: None for ev, _ in timeline if ev.t < 0.6}
    for ev, res in timeline:
        if ev.t < 0.6 and res.decision in _LOCKED and res.identity:
            pre_ids[ev.track_id] = res.identity
    # After swap, if track 1 STILL returns the identity it had pre-swap
    # AND track 2 STILL returns its pre-swap identity, the gatekeeper
    # never detected the swap → FAIL.
    stuck = 0
    for ev, res in post:
        if (res.decision in _LOCKED and res.identity
                and res.identity == pre_ids.get(ev.track_id)):
            stuck += 1
    return stuck < len(post)  # any wobble counts as success


def _build_S4_reacquisition(rng: np.random.Generator) -> List[Event]:
    """Enroll identity A on track_id=1 (frames 0–4). Track is then
    considered lost (no events emitted for that track). Track_id=2
    appears with the same identity, perturbed ~10 % cosine — realistic
    re-entry drift — for frames 6–10."""
    a = _sample_anchor(rng)
    events: List[Event] = []
    # Phase 1: enroll via track 1
    for i in range(5):
        events.append(Event(t=0.1 * i, track_id=1,
                            embedding=_perturb(a, 0.98, rng),
                            quality_score=0.8))
    # Phase 2: gap (no events)
    # Phase 3: re-entry on track 2 with drift → cosine ≈ 0.90 against anchor
    for i in range(5):
        events.append(Event(t=2.0 + 0.1 * i, track_id=2,
                            embedding=_perturb(a, 0.90, rng),
                            quality_score=0.8))
    return events


def _oracle_S4(timeline: List[tuple]) -> bool:
    """Track 2's FIRST event must re-match to the identity track 1
    enrolled — not start a fresh 'Person N+1'."""
    track1_id: Optional[str] = None
    track2_first: Optional[str] = None
    for ev, res in timeline:
        if ev.track_id == 1 and res.decision in _LOCKED and res.identity:
            track1_id = track1_id or res.identity  # first enroll
        if ev.track_id == 2 and track2_first is None:
            if res.decision == GatekeeperDecision.MATCHED:
                track2_first = res.identity
            else:
                return False  # anything but MATCHED on first contact = fail
    return bool(track1_id and track2_first == track1_id)


def _build_S5_pose_drift(rng: np.random.Generator) -> List[Event]:
    """Enroll, then present the same identity with progressively more
    cosine drift (0.97 → 0.80). Simulates yaw change post-enrollment."""
    a = _sample_anchor(rng)
    events: List[Event] = []
    for i in range(5):
        events.append(Event(t=0.1 * i, track_id=1,
                            embedding=_perturb(a, 0.98, rng),
                            quality_score=0.8))
    # Drift: 0.95, 0.92, 0.89, 0.86, 0.83, 0.80
    for i, cos in enumerate([0.95, 0.92, 0.89, 0.86, 0.83, 0.80]):
        events.append(Event(t=0.5 + 0.1 * i, track_id=1,
                            embedding=_perturb(a, cos, rng),
                            quality_score=0.75))
    return events


def _oracle_S5(timeline: List[tuple]) -> bool:
    """Never lose the lock — all frames after the first must be terminal."""
    for _ev, res in timeline[1:]:
        if res.decision not in _LOCKED:
            return False
    return True


def _build_S6_illumination(rng: np.random.Generator) -> List[Event]:
    """Enroll, then present same identity with a fixed-direction bias
    added to every post-enroll embedding (simulates a consistent
    illumination shift that nudges the embedding in some direction)."""
    a = _sample_anchor(rng)
    # Fixed perturbation direction — same across every trial for this scenario
    # but sampled from rng so trials differ.
    bias_dir = _unit(rng.standard_normal(a.shape).astype(np.float32)
                     - np.dot(rng.standard_normal(a.shape), a) * a)
    events: List[Event] = []
    for i in range(5):
        events.append(Event(t=0.1 * i, track_id=1,
                            embedding=_perturb(a, 0.98, rng),
                            quality_score=0.8))
    # Biased: mix 88 % anchor + 12 % bias_dir → cosine ≈ 0.88
    biased = _unit(0.88 * a + 0.12 * bias_dir)
    for i in range(6):
        events.append(Event(t=0.5 + 0.1 * i, track_id=1,
                            embedding=biased + 0.01 * rng.standard_normal(a.shape),
                            quality_score=0.7))
    # Renormalize post-noise.
    for ev in events:
        ev.embedding = _unit(ev.embedding)
    return events


def _oracle_S6(timeline: List[tuple]) -> bool:
    for _ev, res in timeline[1:]:
        if res.decision not in _LOCKED:
            return False
    return True


# ─── S7 / S8 — off-angle + distance failure modes ────────────────────────────
# Added for docs/prompts/improve_pose_and_distance.md. S7 models a profile
# view (deliberate anchor-aligned drift into cosine 0.55–0.65, below the
# current REMATCH_THRESHOLD=0.60 roughly half the time). S8 is an isotropic-
# noise same-track variant that acts as a future regression floor: it passes
# trivially today under the hard tracking-lock but would catch a hypothesis
# that loosens the lock and forgets to handle noisy far-face embeddings.


def _build_S7_hard_pose(rng: np.random.Generator) -> List[Event]:
    """Enroll frontal on track_id=1, then re-acquire on track_id=2 with
    cosine 0.55–0.65 against the anchor — simulates a subject who
    enrolled frontal and returns from a side angle."""
    a = _sample_anchor(rng)
    events: List[Event] = []
    # Phase 1: enroll frontal on track 1
    for i in range(5):
        events.append(Event(t=0.1 * i, track_id=1,
                            embedding=_perturb(a, 0.98, rng),
                            quality_score=0.8))
    # Phase 2: re-entry on track 2 — hard pose drift
    for i in range(5):
        cos = float(rng.uniform(0.55, 0.65))
        events.append(Event(t=2.0 + 0.1 * i, track_id=2,
                            embedding=_perturb(a, cos, rng),
                            quality_score=0.75))
    return events


def _oracle_S7(timeline: List[tuple]) -> bool:
    """Re-match succeeds within the first 3 events of the profile phase.
    'Within 3' rather than 'first-contact' acknowledges that the first
    profile frame might reasonably land just below threshold; if 2 of 3
    succeed a diverse bank is clearly spanning the view."""
    track1_id: Optional[str] = None
    track2_events: List[tuple] = []
    for ev, res in timeline:
        if ev.track_id == 1 and res.decision in _LOCKED and res.identity:
            track1_id = track1_id or res.identity
        if ev.track_id == 2:
            track2_events.append((ev, res))
    if not track1_id or not track2_events:
        return False
    for ev, res in track2_events[:3]:
        if (res.decision == GatekeeperDecision.MATCHED
                and res.identity == track1_id):
            return True
    return False


def _build_S8_distance_noise(rng: np.random.Generator) -> List[Event]:
    """Enroll on track_id=1, then continue on the same track with
    isotropic Gaussian noise (σ ≈ 0.15) — proxy for the embedding
    degradation a small / upsampled crop produces. Same track so the
    scenario targets the post-enrollment lock + bank update path, not
    re-acquisition."""
    a = _sample_anchor(rng)
    events: List[Event] = []
    sigma = 0.15
    # Phase 1: enroll clean
    for i in range(5):
        events.append(Event(t=0.1 * i, track_id=1,
                            embedding=_perturb(a, 0.98, rng),
                            quality_score=0.8))
    # Phase 2: same track, noisy embeddings
    for i in range(6):
        noisy = a + sigma * rng.standard_normal(a.shape).astype(np.float32)
        events.append(Event(t=0.5 + 0.1 * i, track_id=1,
                            embedding=_unit(noisy),
                            quality_score=0.65))
    return events


def _oracle_S8(timeline: List[tuple]) -> bool:
    """Every post-first frame must be in a locked state. Trivially true
    under the current hard tracking-lock; serves as a regression floor
    for any future hypothesis that loosens the lock (crowded-scene fix)."""
    for _ev, res in timeline[1:]:
        if res.decision not in _LOCKED:
            return False
    return True


DEFAULT_SCENARIOS: Dict[str, Scenario] = {
    "S1_single_lock":   Scenario("S1_single_lock",
                                 "One identity, tight noise, 10 frames.",
                                 _build_S1_single_lock, _oracle_S1),
    "S2_two_no_cross":  Scenario("S2_two_no_cross",
                                 "Two identities on two tracks, no crossover.",
                                 _build_S2_two_no_cross, _oracle_S2),
    "S3_crossing":      Scenario("S3_crossing",
                                 "Two tracks swap embeddings mid-sequence.",
                                 _build_S3_crossing, _oracle_S3),
    "S4_reacquisition": Scenario("S4_reacquisition",
                                 "Same identity re-enters on a fresh track_id after a gap.",
                                 _build_S4_reacquisition, _oracle_S4),
    "S5_pose_drift":    Scenario("S5_pose_drift",
                                 "Cosine drift from 0.98 down to 0.80.",
                                 _build_S5_pose_drift, _oracle_S5),
    "S6_illumination":  Scenario("S6_illumination",
                                 "Consistent directional embedding bias post-enroll.",
                                 _build_S6_illumination, _oracle_S6),
    "S7_hard_pose":     Scenario("S7_hard_pose",
                                 "Re-acquire on a new track with cosine 0.55–0.65 drift (profile view).",
                                 _build_S7_hard_pose, _oracle_S7),
    "S8_distance_noise": Scenario("S8_distance_noise",
                                  "Same-track isotropic noise (σ=0.15) post-enrollment.",
                                  _build_S8_distance_noise, _oracle_S8),
}


# ─────────────────────────────────────────────────────────────────────────────
# Harness entry points
# ─────────────────────────────────────────────────────────────────────────────


def run_scenario(scenario: Scenario, seed: int = 0,
                 trials: int = 25) -> ScenarioResult:
    """Run `scenario` `trials` times with deterministic seeds derived
    from `seed`, each with a fresh Gatekeeper. Returns aggregated result."""
    successes = 0
    # Gatekeeper uses print() for FAST/REFINED/expired events; silencing
    # keeps the harness's stdout readable without touching gatekeeper code.
    silenced = io.StringIO()
    with contextlib.redirect_stdout(silenced):
        for trial_i in range(trials):
            rng = np.random.default_rng(seed + trial_i)
            events = scenario.build(rng)
            gk = Gatekeeper(known_faces={}, person_counter=1)
            timeline: List[tuple] = []
            for ev in events:
                res = gk.process(
                    embedding=ev.embedding,
                    track_id=ev.track_id,
                    face_bgr=_DUMMY_BGR,
                    quality_score=ev.quality_score,
                )
                timeline.append((ev, res))
            if scenario.oracle(timeline):
                successes += 1
    return ScenarioResult(
        id=scenario.id,
        trials=trials,
        successes=successes,
        success_rate=successes / trials if trials else 0.0,
    )


def run_all(config_diff: Optional[Dict] = None, *, trials: int = 25,
            seed: int = 42) -> RunReport:
    """Run every scenario in `DEFAULT_SCENARIOS` under the current
    `src.gatekeeper` module state. `config_diff` is informational only —
    threshold changes must be made in `src/gatekeeper.py` before calling."""
    _ = config_diff  # reserved for future per-call overrides; today knobs
                     # live at module scope and run_all reads whatever's
                     # currently imported.
    results: List[ScenarioResult] = []
    for name, scenario in DEFAULT_SCENARIOS.items():
        results.append(run_scenario(scenario, seed=seed, trials=trials))

    total_trials    = sum(r.trials for r in results)
    total_successes = sum(r.successes for r in results)
    aggregate = {
        "total_trials":    total_trials,
        "total_successes": total_successes,
        "success_rate":    (total_successes / total_trials) if total_trials else 0.0,
    }
    return RunReport(scenarios=results, aggregate=aggregate)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact helpers — used by scripts/models/run_NNN.py to dump JSONs.
# ─────────────────────────────────────────────────────────────────────────────


def current_knobs() -> Dict[str, float]:
    """Snapshot of the gatekeeper's tunable constants. Included in run
    artifacts so a reader can reconstruct the pipeline config without
    cloning the exact git SHA."""
    return {
        "MATCH_THRESHOLD":             MATCH_THRESHOLD,
        "REMATCH_THRESHOLD":           REMATCH_THRESHOLD,
        "ENROLL_THRESHOLD":            ENROLL_THRESHOLD,
        "MIN_MATCH_MARGIN":            MIN_MATCH_MARGIN,
        "CONSENSUS_FRAMES":            CONSENSUS_FRAMES,
        "MAX_EMBEDDINGS_PER_IDENTITY": MAX_EMBEDDINGS_PER_IDENTITY,
    }


def git_short_sha() -> str:
    """Best-effort; 'uncommitted' if git isn't present or the tree is dirty
    and the caller wants to stamp the JSON as pre-commit."""
    try:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or "uncommitted"
    except Exception:
        return "uncommitted"


def build_artifact(run_id: str, hypothesis: str,
                   config_diff: Optional[Dict], report: RunReport,
                   notes: str = "", decision: str = "inconclusive",
                   eval_type: str = "synth",
                   datasets: Optional[List[str]] = None) -> dict:
    """Shape the canonical run JSON — same schema every run_NNN.py emits.

    `eval_type` tags the run as `"synth"` (default, for `src/live_eval.py`
    scenario runs) or `"real"` (for `src/real_data_eval.py`). The synthetic
    runs predate the field; plot tooling treats missing `eval_type` as
    `"synth"` to stay backward-compatible.
    `datasets` is optional and only meaningful for real runs — lists the
    dataset roots walked (e.g. `["scface", "chokepoint"]`)."""
    artifact = {
        "run_id":      run_id,
        "eval_type":   eval_type,
        "created_at":  datetime.datetime.now(datetime.timezone.utc)
                          .replace(microsecond=0).isoformat(),
        "git_sha":     git_short_sha(),
        "hypothesis":  hypothesis,
        "config_diff": config_diff or {},
        "knobs":       current_knobs(),
        **report.to_dict(),
        "decision":    decision,
        "notes":       notes,
    }
    if datasets:
        artifact["datasets"] = datasets
    return artifact
