"""
Run 010 — Lower MIN_FACE_PX 60 → 40
====================================
Failure mode : far-face / distance (class 4 of
               docs/prompts/improve_pose_and_distance.md).
Hypothesis   : Lower MIN_FACE_PX from 60 → 40 so MTCNN detections in
               the 40–59 px band stop being hard-rejected before
               embedding. run_009 showed 283/414 pairs fail at the
               detection stage on SCface d1/d2 + ChokePoint — this
               run's targeted scenarios are exactly those.

Synthetic scenarios are unchanged by design (MIN_FACE_PX doesn't
enter the synthetic code path); skip re-running, see notes.

Expected runtime: ~10–20 minutes (dominated by MTCNN + FaceNet per
image, 414 pairs).
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.real_data_eval import run_real_all
from src.live_eval      import build_artifact

RUN_ID      = "run_010"
HYPOTHESIS  = (
    "Lower MIN_FACE_PX from 60 to 40 across scripts/server.py, "
    "scripts/live_webcam.py, and src/real_data_eval.py to recover "
    "far-face detections (run_009: 283/414 real-data pairs failed at "
    "the detection stage)."
)
CONFIG_DIFF = {"MIN_FACE_PX": [60, 40]}

report, datasets = run_real_all(datasets=["scface", "chokepoint"])

# Decision gets filled in *after* comparing to run_009 — this script
# always writes "inconclusive" and a human (or the follow-up changelog
# step) flips it to keep/revert based on the decision rule in
# docs/prompts/improve_pose_and_distance.md.
notes = (
    "MIN_FACE_PX single-knob change, measured end-to-end on SCface + "
    "ChokePoint via src/real_data_eval.py. Compare per-scenario "
    "against run_009 baseline: targeted scenarios are d1_far (0 %), "
    "d2_mid (0 %), chokepoint_temporal_stability (0 %). Regression "
    "floor: d3_close must not drop ≥ 20 % absolute from its 36.2 % "
    "baseline (16.2 % floor). Synthetic S1 regression check skipped — "
    "MIN_FACE_PX does not enter the synthetic harness (trivially "
    "100 %). DETECT_DOWNSAMPLE unchanged (still 2 in server.py / "
    "live_webcam.py); the real-data harness processes full-resolution "
    "images so it cannot measure a DETECT_DOWNSAMPLE change — saved "
    "for a future live-path run with manual verification."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=report,
    decision="inconclusive",   # resolved in the changelog step
    notes=notes,
    eval_type="real",
    datasets=datasets,
)

out_path = os.path.join(PROJECT_ROOT, "outputs", "runs", f"{RUN_ID}.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"wrote -> {out_path}")
for s in report.scenarios:
    print(f"  {s.id:36s}  {s.successes}/{s.trials}  ({s.success_rate:.0%})")
print(f"aggregate: {report.aggregate['success_rate']:.3f}")
