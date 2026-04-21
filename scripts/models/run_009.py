"""
Run 009 — First real-data baseline (SCface + ChokePoint)
=========================================================
First run under docs/prompts/improve_pose_and_distance.md that uses
actual surveillance imagery. No src/ change — this run establishes
the pipeline's absolute-accuracy floor on:
    - SCface (130 subjects × 3 distances) — off-angle + distance
    - ChokePoint (8 sessions × 3 cameras = 24 pairs) — temporal stability

TinyFace is deferred (see docs/real_data_eval.md — open-set identification
protocol needs a separate harness).

Expected runtime: ~10–15 minutes on CPU (MTCNN + FaceNet per image, 414
pairs total, single-threaded).
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

RUN_ID      = "run_009"
HYPOTHESIS  = (
    "First real-data baseline: run the full live pipeline "
    "(MTCNN → align → FaceNet → Gatekeeper) across SCface distance-"
    "stratified pairs + ChokePoint temporal-stability pairs. No src/ "
    "change. Establishes absolute-accuracy floors for subsequent "
    "pose/distance hypothesis runs."
)
CONFIG_DIFF = {}

report, datasets = run_real_all(datasets=["scface", "chokepoint"])

notes = (
    "Per-scenario absolute floors on real surveillance imagery. Compare "
    "against synthetic S4_reacquisition / S7_hard_pose / S8_distance_noise "
    "baselines (run_007 / run_008) — they are direction-of-travel proxies; "
    "these are the calibrated truth. SCface d1_far (4.2m) is the hardest; "
    "d3_close (1.0m) should be near-ceiling. TinyFace deferred (open-set "
    "protocol). ChokePoint scenario is temporal stability, not identity "
    "re-acquisition across subjects (no GT labels in the uploaded subset) "
    "— treat its numbers as 'does the same walking person produce stable "
    "enough embeddings to re-lock?'"
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=report,
    decision="keep",          # baseline, nothing to revert
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
