"""
Tests for the synthetic gatekeeper harness.

Focus: determinism under a fixed seed, baseline behavior on current
knobs, and that the scenarios exercise the intended gatekeeper paths.
"""

import pytest

from src.live_eval import (
    DEFAULT_SCENARIOS,
    Event,
    ScenarioResult,
    build_artifact,
    current_knobs,
    run_all,
    run_scenario,
)


# ── Scenario catalogue ────────────────────────────────────────────────────────


def test_default_scenarios_all_present():
    names = set(DEFAULT_SCENARIOS.keys())
    assert names == {
        "S1_single_lock", "S2_two_no_cross", "S3_crossing",
        "S4_reacquisition", "S5_pose_drift", "S6_illumination",
        "S7_hard_pose",    "S8_distance_noise",
    }


def test_s7_s8_are_deterministic():
    """S7/S8 were added for pose+distance runs. Same seed → identical
    outcomes so run artifacts are reproducible."""
    for name in ("S7_hard_pose", "S8_distance_noise"):
        a = run_scenario(DEFAULT_SCENARIOS[name], seed=123, trials=5)
        b = run_scenario(DEFAULT_SCENARIOS[name], seed=123, trials=5)
        assert (a.successes, a.trials) == (b.successes, b.trials), name


# ── Determinism ───────────────────────────────────────────────────────────────


def test_run_scenario_is_deterministic():
    """Same seed -> identical success counts; different seed -> allowed
    to differ. We rely on this so per-run artifacts are reproducible."""
    s = DEFAULT_SCENARIOS["S4_reacquisition"]
    a = run_scenario(s, seed=42, trials=5)
    b = run_scenario(s, seed=42, trials=5)
    assert (a.successes, a.trials) == (b.successes, b.trials)


# ── Baseline behavior on current knobs ────────────────────────────────────────


@pytest.fixture(scope="module")
def baseline_report():
    # 25 trials is enough signal while staying fast (~0.5s on CPU).
    return run_all(trials=25, seed=42)


def test_s1_single_lock_is_the_regression_floor(baseline_report):
    """S1 is the baseline — if this ever drops below 100 %, the change
    broke trivial single-person recognition and the prompt's decision
    rule says revert."""
    s1 = next(s for s in baseline_report.scenarios if s.id == "S1_single_lock")
    assert s1.success_rate == 1.0, (
        "S1 is the single-person-lock baseline; must always be 1.0 unless "
        "the gatekeeper state machine itself is being rewritten."
    )


def test_s3_crossing_passes_with_drift_detection(baseline_report):
    """Run_021 added drift detection inside the tracking-lock path: if
    the incoming embedding has max-cosine below REMATCH_THRESHOLD vs
    the locked identity's bank AND another identity is a clearer
    match, the lock breaks and re-locks to the new identity.

    Before this change S3 was a documented 0 % failure (the hard
    tracking-lock made it unrecoverable by design). After: S3 should
    run at ≥ 0.9 success rate — any drop below that means drift
    detection regressed."""
    s3 = next(s for s in baseline_report.scenarios if s.id == "S3_crossing")
    assert s3.success_rate >= 0.9, (
        f"S3 crossing detection regressed: got {s3.success_rate:.2f}, "
        "expected ≥ 0.9. Drift-break logic in Gatekeeper's tracking-"
        "lock path is the expected mechanism."
    )


def test_s4_reacquisition_passes_under_rematch_threshold(baseline_report):
    """REMATCH_THRESHOLD=0.60 (landed in run_006) lets a returning face
    with ~10 % cosine drift re-lock instead of starting a new enrollment.
    If this drops, REMATCH_THRESHOLD was moved back above the drift."""
    s4 = next(s for s in baseline_report.scenarios if s.id == "S4_reacquisition")
    assert s4.success_rate >= 0.9


# ── Artifact shape ────────────────────────────────────────────────────────────


def test_build_artifact_has_required_fields():
    rep = run_all(trials=2, seed=1)
    art = build_artifact(
        run_id="run_test",
        hypothesis="test",
        config_diff={"MATCH_THRESHOLD": [0.70, 0.65]},
        report=rep,
    )
    required = {"run_id", "created_at", "git_sha", "hypothesis",
                "config_diff", "knobs", "scenarios", "aggregate",
                "decision", "notes"}
    assert required.issubset(art.keys())
    assert art["decision"] == "inconclusive"  # default until caller overrides
    # Aggregate is computed, not passed in
    assert 0.0 <= art["aggregate"]["success_rate"] <= 1.0


def test_current_knobs_exposes_match_threshold():
    knobs = current_knobs()
    assert "MATCH_THRESHOLD"    in knobs
    assert "REMATCH_THRESHOLD"  in knobs
    assert "MIN_MATCH_MARGIN"   in knobs
    assert "CONSENSUS_FRAMES"   in knobs
    # Sanity on current values — guard against accidental zeroing.
    # REMATCH can sit well below 0.30 (run_019 margin-aware design uses
    # 0.20 paired with a ≥ MIN_MATCH_MARGIN rank-1 separation check);
    # only enforce the ordering invariant and non-zero margin.
    assert 0.3 <= knobs["MATCH_THRESHOLD"] <= 1.0
    assert 0.0 < knobs["REMATCH_THRESHOLD"] <= knobs["MATCH_THRESHOLD"]
    assert 0.0 < knobs["MIN_MATCH_MARGIN"] < knobs["MATCH_THRESHOLD"]
