"""
Run 008 — Pose/distance baseline (no src/ change)
==================================================
First run under docs/prompts/improve_pose_and_distance.md. The
harness gained two new scenarios — S7_hard_pose (re-acquisition with
cosine 0.55–0.65 drift, proxy for a profile view) and S8_distance_noise
(same-track isotropic Gaussian noise, σ=0.15, proxy for a small /
upsampled crop). This run establishes their pre-change floors so
hypothesis runs under the pose/distance prompt have numbers to
compare against.

No src/ change — this is purely a scenario-addition measurement.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.live_eval import run_all, build_artifact

RUN_ID      = "run_008"
HYPOTHESIS  = (
    "Extend the synthetic harness with S7_hard_pose (cosine 0.55–0.65 "
    "re-acquisition) and S8_distance_noise (same-track σ=0.15 noise). "
    "Establishes pose/distance floors for subsequent hypothesis runs."
)
CONFIG_DIFF = {}   # no knob changes

report = run_all(trials=25, seed=42)

# Prompt class: "scenario infrastructure" — doesn't fit the design
# space of improve_pose_and_distance.md directly; it's the setup step.
notes = (
    "Scenario-addition run. No src/ change. Per the pose/distance "
    "prompt, S7/S8 baselines are required before any hypothesis can "
    "claim an improvement. S7 success ~88 % is the informative signal "
    "(uniform [0.55, 0.65] cosine against REMATCH_THRESHOLD=0.60 lets "
    "roughly half the frames pass; the S7 oracle needs only one of the "
    "first three to succeed). S8 passes trivially under the hard "
    "tracking-lock — kept as a future regression floor for any "
    "hypothesis that loosens the lock. S3 remains the documented "
    "crossing failure at 0 %. Next step: run_009 should pick a "
    "hypothesis from docs/prompts/improve_pose_and_distance.md; the "
    "recommended starting point is greedy FPS bank selection (class 1, "
    "hypothesis #1) — smallest blast radius for the biggest expected "
    "S7 lift."
)

artifact = build_artifact(
    run_id=RUN_ID, hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF, report=report,
    decision="keep",  # baseline measurement — nothing to revert
    notes=notes,
)

out_path = os.path.join(PROJECT_ROOT, "outputs", "runs", f"{RUN_ID}.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"wrote -> {out_path}")
for s in report.scenarios:
    print(f"  {s.id:22s}  {s.successes}/{s.trials}  ({s.success_rate:.0%})")
print(f"aggregate: {report.aggregate['success_rate']:.3f}")
