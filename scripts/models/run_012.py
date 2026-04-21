"""
Run 012 — MIN_FACE_PX 60 → 40 under ArcFace
=============================================
Failure mode : d1_far / d2_mid / chokepoint blocked before the
               embedder (run_011 kept ArcFace but MIN_FACE_PX=60 still
               rejects SCface d1 (18–30 px) and most d2 (28–46 px)).

Hypothesis   : Now that the embedder handles low-res (run_011: d3 went
               36→96 %), lowering the detection size floor lets the
               40–46 px d2 faces reach ArcFace. Prediction: d2 jumps
               to 25–50 %. d1 still blocked (faces < 40 px). d3 holds
               (minor regression risk as newly-admitted 40–59 band is
               the hardest slice of d3).

run_010 tested the same knob under FaceNet and got +4.6 % — this
run tests whether ArcFace converts size-gate-opened faces into
actual matches, rather than noise. One-hypothesis-per-run rule:
MIN_FACE_PX is the only live-pipeline knob changing; ArcFace +
thresholds are inherited kept state from run_011.
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

RUN_ID      = "run_012"
HYPOTHESIS  = (
    "Under ArcFace (run_011 kept), lower MIN_FACE_PX from 60 to 40. "
    "Expected to admit the 40–46 px d2 faces that the size gate "
    "previously rejected, now that the embedder handles low-res."
)
CONFIG_DIFF = {"MIN_FACE_PX": [60, 40]}

report, datasets = run_real_all(datasets=["scface", "chokepoint"])

notes = (
    "Compare to run_011 baseline (ArcFace, size gate=60): d1_far 0 %, "
    "d2_mid 0.8 %, d3_close 96.2 %, chokepoint 0 %. Decision keep iff "
    "one of {d1,d2,chokepoint} improves ≥20 % AND d3_close stays "
    "≥76.2 %. Synthetic S1 floor unaffected (MIN_FACE_PX doesn't enter "
    "synth path). d1_far expected to remain ~0 % — faces are 18–30 px, "
    "still below the new 40 px floor; follow-up run_013 would push to "
    "MIN_FACE_PX=20 if run_012 confirms ArcFace handles 40 px cleanly."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=report,
    decision="inconclusive",    # resolved after comparing to run_011
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
