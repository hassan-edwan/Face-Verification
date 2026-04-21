"""
Tests for the real-data harness — structure / loaders only.

MTCNN + FaceNet are NOT loaded here (the `_Models` lazy pattern means
they aren't touched until `run_real_scenario` is called). Full pipeline
integration is validated by running `scripts/models/run_009.py`.
"""

from pathlib import Path

import numpy as np
import pytest

from src.real_data_eval import (
    DEFAULT_DATA_ROOT, PairSpec, PairResult,
    _default_oracle, _tar_at_far, _rank_of_truth,
    load_scface_scenarios, load_chokepoint_scenarios,
    build_scface_identification_specs,
    load_scface_mugshot_rotation,
    build_scface_multishot_identification_specs,
    DEFAULT_MULTISHOT_POSES,
)
from src.gatekeeper import Gatekeeper


def test_scface_loader_discovers_130_subjects_per_distance():
    """SCface is present on this machine; sanity-check the loader found
    every subject × distance combination. If this drops, the data move
    wasn't clean."""
    scenarios = load_scface_scenarios(DEFAULT_DATA_ROOT)
    assert {s.id for s in scenarios} == {
        "scface_mugshot_to_d1_far",
        "scface_mugshot_to_d2_mid",
        "scface_mugshot_to_d3_close",
    }
    # Each distance should have ~130 pairs (every subject at that distance).
    for s in scenarios:
        assert len(s.pairs) >= 120, f"{s.id} only had {len(s.pairs)} pairs"


def test_chokepoint_loader_discovers_session_cameras():
    """ChokePoint: with the per-frame GT XMLs in place, the loader builds
    three scenarios — within-clip sanity (same camera, first-vs-last
    frame of one walk), cross-camera (same person across portal cameras
    of one session), and cross-session (same pid across different
    sessions — the actual generalization test)."""
    scenarios = load_chokepoint_scenarios(DEFAULT_DATA_ROOT)
    ids = {s.id for s in scenarios}
    # cross_session requires ≥2 sessions; if the local subset only ships
    # one, that scenario will be absent — tolerate its absence but never
    # expect the old temporal_stability name to appear.
    assert "chokepoint_temporal_stability" not in ids
    assert "chokepoint_within_clip_sanity" in ids
    assert "chokepoint_cross_camera" in ids
    by_id = {s.id: s for s in scenarios}
    assert len(by_id["chokepoint_within_clip_sanity"].pairs) >= 15
    assert len(by_id["chokepoint_cross_camera"].pairs)       >= 5


def test_tar_at_far_basic_separability():
    """Genuine scores ~1.0, impostor scores ~0.0 → TAR@FAR=1 % should
    be ~1.0. Guards the threshold-picking logic against off-by-one
    errors."""
    genuines  = [0.95] * 100
    impostors = [0.05] * 1000
    out = _tar_at_far(genuines, impostors, far_targets=(0.01,))
    assert out["0.01"] is not None
    assert out["0.01"]["tar"] == 1.0


def test_tar_at_far_returns_none_for_insufficient_impostors():
    """Can't resolve FAR=0.001 with only 100 impostors — the function
    should emit None rather than a noisy 1-impostor threshold."""
    out = _tar_at_far([0.9] * 10, [0.1] * 100, far_targets=(0.001,))
    assert out["0.001"] is None


def test_rank_of_truth_finds_correct_subject_rank():
    """Hand-build a Gatekeeper with a known 3-subject bank and verify
    `_rank_of_truth` returns the right 1-based rank. Gatekeeper is used
    directly — no MTCNN / embedder touched."""
    rng = np.random.default_rng(7)
    dim = 64

    def _unit(v):
        return v / (np.linalg.norm(v) + 1e-9)

    anchors = {f"s{i}": _unit(rng.standard_normal(dim).astype(np.float32))
               for i in range(3)}
    known = {f"Person {i+1}": [emb] for i, emb in enumerate(anchors.values())}
    gk = Gatekeeper(known_faces=known, person_counter=4)
    gk.__dict__["_id_subject_map"] = {
        "Person 1": "s0", "Person 2": "s1", "Person 3": "s2",
    }

    # Query near s1 — true rank should be 1 for s1, near the bottom for s0/s2.
    query = anchors["s1"] + 0.01 * rng.standard_normal(dim).astype(np.float32)
    rank, r1_subj, r1_score = _rank_of_truth(gk, query, true_subject="s1")
    assert rank == 1
    assert r1_subj == "s1"
    assert r1_score > 0.9

    # Query the same embedding but ask about the wrong subject — rank should be > 1.
    rank, r1_subj, _ = _rank_of_truth(gk, query, true_subject="s0")
    assert rank > 1
    assert r1_subj == "s1"


def test_build_scface_identification_specs_groups_by_distance():
    """Verification pairs collapse into one IdentificationSpec per
    distance, enrollments keyed by subject_id with all 130 mugshots."""
    specs = build_scface_identification_specs(DEFAULT_DATA_ROOT)
    ids = {s.id for s in specs}
    assert ids == {"scface_identification_d1_far",
                   "scface_identification_d2_mid",
                   "scface_identification_d3_close"}
    for s in specs:
        # Gallery size should match number of queries (one per subject).
        assert len(s.enrollments) == len(s.queries)
        assert len(s.enrollments) >= 120
        # Every enrollment is length-1 for the single-shot builder.
        for subj, templates in s.enrollments.items():
            assert len(templates) == 1, (
                f"single-shot builder produced {len(templates)} templates "
                f"for {subj}"
            )


def test_load_scface_mugshot_rotation_five_poses():
    """`mugshot_rotation/` contains 130 subjects × 9 poses. Default pose
    set (frontal, L1, L4, R1, R4) should yield 5 templates per subject."""
    per_subject = load_scface_mugshot_rotation(DEFAULT_DATA_ROOT)
    assert len(per_subject) >= 120
    for subj, templates in per_subject.items():
        # Subjects with all 5 default poses available.
        poses = [p for _path, p in templates]
        assert poses[0] == "frontal", (
            f"{subj}: frontal must be first for two-phase enrollment; "
            f"got {poses}"
        )
        # Exact match isn't guaranteed if some files are missing but
        # most subjects should have the full default set.
    # Spot-check: at least one subject has the full default pose set.
    full = [s for s, ts in per_subject.items() if len(ts) == 5]
    assert len(full) >= 120, (
        f"expected ≥120 subjects with all 5 default poses; found {len(full)}"
    )


def test_build_scface_multishot_identification_specs():
    """Multi-shot builder emits one spec per distance with length-5
    enrollment lists per subject, queries inherited from surveillance."""
    specs = build_scface_multishot_identification_specs(DEFAULT_DATA_ROOT)
    ids = {s.id for s in specs}
    assert ids == {"scface_identification_multishot_d1_far",
                   "scface_identification_multishot_d2_mid",
                   "scface_identification_multishot_d3_close"}
    for s in specs:
        assert len(s.enrollments) >= 120
        # Each subject seeded with exactly len(DEFAULT_MULTISHOT_POSES) templates.
        for subj, templates in s.enrollments.items():
            assert len(templates) == len(DEFAULT_MULTISHOT_POSES)


def test_default_oracle_counts_successes():
    mk = lambda success: PairResult(
        spec=PairSpec(subject_id="x", enroll_path=Path(""), query_path=Path(""),
                      scenario_id="t"),
        enroll_ok=True, query_decision="MATCHED",
        matched_identity="x", enrolled_identity="x",
        success=success, time_ms=1.0,
    )
    res = [mk(True), mk(True), mk(False)]
    successes, trials = _default_oracle(res)
    assert successes == 2 and trials == 3


def test_missing_data_root_yields_empty_scenarios(tmp_path):
    """A fresh checkout with no data/real_eval/ should not crash; the
    loaders just return empty lists."""
    assert load_scface_scenarios(tmp_path) == []
    assert load_chokepoint_scenarios(tmp_path) == []
