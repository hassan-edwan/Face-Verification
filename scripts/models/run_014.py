"""
Run 014 — MIN_FACE_PX 40 → 20 under ArcFace
============================================
Failure mode : Two scenarios are detection-bound at the 40 px floor:
               (a) SCface d1_far (4.2 m) — faces are 18-30 px per the
                   earlier diagnose_detection.py measurement, hard
                   zero across runs 009-013.
               (b) chokepoint_temporal_stability — 248/600 pairs
                   skipped with "no face" in run_013. The first-visible
                   and last-visible frames are by definition the
                   bookends where the walker is smallest / most angled
                   / entering-or-leaving frame. Match rate when
                   detection succeeds is 97 % (342/352), so the 57 %
                   aggregate is detection cost, not matching cost.

Hypothesis   : ArcFace's low-res robustness (run_011/012 showed clean
               handling at 40 px where FaceNet couldn't) extends to
               20 px. Lower the size gate and expect d1_far to move
               off 0 %, plus a meaningful lift on temporal_stability
               as bookend frames become admissible. Risk: MTCNN's own
               detection quality at 20 px may be the binding
               constraint — if so, run_015 (MTCNN_CONF_MIN drop) is
               the natural follow-up.

Decision rule: keep iff d1_far improves ≥ 20 % absolute (i.e. ≥ 20 %
               of the 130 pairs = 26 matches) OR temporal_stability
               improves ≥ 20 % absolute, AND no other scenario
               regresses ≥ 20 %. Synthetic S1 floor unaffected
               (MIN_FACE_PX doesn't enter synth path).
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("FACE_EMBEDDER", "arcface")

from src.real_data_eval import run_real_all
from src.live_eval      import build_artifact

RUN_ID      = "run_014"
HYPOTHESIS  = (
    "Under kept ArcFace, lower MIN_FACE_PX from 40 to 20. Targets "
    "(a) SCface d1_far (faces 18-30 px, 100 % rejected by current "
    "gate) and (b) chokepoint_temporal_stability bookend-frame "
    "detection failures (248/600 skipped in run_013). Expected: "
    "d1_far moves off 0 %, temporal_stability lifts as more bookends "
    "survive the size gate."
)
CONFIG_DIFF = {"MIN_FACE_PX": [40, 20]}

report, datasets = run_real_all(datasets=["scface", "chokepoint"])

notes = (
    "Compare to run_013 baseline: d1_far 0 %, d2_mid 92 %, d3_close "
    "98 %, temporal_stability 57 % (342/600, 248 skipped detection), "
    "cross_camera 99.5 %. Keep iff d1_far or temporal_stability "
    "improves ≥ 20 % absolute AND no other scenario regresses "
    "≥ 20 %. MTCNN's own detection quality at 20 px is the likely "
    "binding constraint — if this run is flat or modest, run_015 "
    "follows up with MTCNN_CONF_MIN 0.90 → 0.70."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=report,
    decision="inconclusive",
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
    print(f"  {s.id:40s}  {s.successes}/{s.trials}  ({s.success_rate:.0%})")
print(f"aggregate: {report.aggregate['success_rate']:.3f}")
