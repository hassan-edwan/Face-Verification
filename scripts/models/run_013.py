"""
Run 013 — Chokepoint GT-grounded baseline
=========================================
Previous harness fault (runs 009-012) : `load_chokepoint_scenarios`
               treated each session-camera as one subject and paired
               frame[~25 %] with frame[~75 %] of the walk. A ChokePoint
               session contains ~25 different walkers, so enroll and
               query were almost always different people. Chokepoint's
               0 % across four runs was apparatus, not model.

Apparatus fix: per-frame GT (<person id=>, <leftEye>, <rightEye>) is
               now present at data/real_eval/chokepoint/groundtruth/.
               The rewritten loader yields two chokepoint scenarios:
                 - chokepoint_temporal_stability : per labeled person,
                   first-vs-last visible frame within one camera
                 - chokepoint_cross_camera       : same person seen in
                   2+ portal cameras of the same session

Detection also switched from "pick largest face" to "pick the MTCNN
detection whose left_eye keypoint is closest to the GT left_eye"
when a `PairSpec` carries GT eye coords, so the embedding actually
belongs to the labeled person and not some bystander.

Hypothesis  : Chokepoint's true accuracy under ArcFace + MIN_FACE_PX=40
              is > 0 %. We don't know the magnitude yet — this run is
              the GT-grounded baseline that future runs optimize from.

All live knobs kept from run_012 (ArcFace, MIN_FACE_PX=40, thresholds
MATCH=0.40 / REMATCH=0.30 / ENROLL=0.20). No src/ change beyond the
eval harness.
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

RUN_ID      = "run_013"
HYPOTHESIS  = (
    "With per-frame GT XML in place, the chokepoint loader now builds "
    "same-subject pairs (temporal_stability + cross_camera). Expect a "
    "real, nonzero baseline under kept config (ArcFace, MIN_FACE_PX=40). "
    "No model-side change — this run measures what chokepoint actually "
    "looks like once the apparatus stops comparing strangers."
)
CONFIG_DIFF = {
    "chokepoint_loader": [
        "session_cam_as_subject",
        "gt_xml_per_person_eyes",
    ],
}

report, datasets = run_real_all(datasets=["scface", "chokepoint"])

notes = (
    "Baseline for the new chokepoint apparatus. Old runs (009-012) "
    "reported chokepoint at 0 % because the loader paired different "
    "people; that number is not comparable to anything here. SCface "
    "scenarios unchanged from run_012, re-run for a complete snapshot. "
    "Decision = keep (baseline) regardless of chokepoint magnitude — "
    "the optimize rule (Δ ≥ 20 % vs prior) doesn't apply when the "
    "prior wasn't a valid measurement. Next run will target whichever "
    "chokepoint scenario is weaker (likely cross_camera — it's the "
    "pose-variation test)."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=report,
    decision="keep (baseline)",
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
