"""
Run 011 — Swap FaceNet → ArcFace (buffalo_l w600k_r50)
========================================================
Failure mode : resolution collapse at sub-40 px faces (root cause of
               run_009's 0 % at d1_far / d2_mid). FaceNet's SNR floor
               on VGGFace2-scale training made it unusable on
               surveillance-scale inputs.

Hypothesis   : Replace FaceNet with ArcFace IResNet-100 (trained on
               WebFace600K) — known-strong on low-quality / surveillance
               domains. Coordinated change: alignment also moves from
               2-point/160×160 to 5-point/112×112 (ArcFace's required
               input geometry), and cosine thresholds drop from
               0.70/0.60/0.40 to 0.40/0.30/0.20 (ArcFace same-identity
               clusters are tighter than FaceNet's).

All three changes are bundled because none is measurable alone:
swapping the backbone without threshold retuning rejects every match;
keeping 2-point alignment on a 112×112 ArcFace input gives distorted
crops. See plan file for the integration trace.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force ArcFace backend even if something else tried to set it.
os.environ.setdefault("FACE_EMBEDDER", "arcface")

from src.real_data_eval import run_real_all
from src.live_eval      import build_artifact

RUN_ID      = "run_011"
HYPOTHESIS  = (
    "Swap embedder FaceNet → ArcFace (buffalo_l / w600k_r50). Bundled "
    "with 2-point→5-point alignment (112×112 input) and cosine threshold "
    "retune (MATCH 0.70→0.40, REMATCH 0.60→0.30, ENROLL 0.40→0.20). "
    "Targets resolution collapse at d1_far / d2_mid (run_009 baseline 0 %)."
)
CONFIG_DIFF = {
    "EMBEDDER":          ["facenet", "arcface"],
    "ALIGNMENT":         ["2pt/160", "5pt/112"],
    "MATCH_THRESHOLD":   [0.70, 0.40],
    "REMATCH_THRESHOLD": [0.60, 0.30],
    "ENROLL_THRESHOLD":  [0.40, 0.20],
}

report, datasets = run_real_all(datasets=["scface", "chokepoint"])

notes = (
    "Coordinated three-part swap; one hypothesis per the run convention "
    "(swapping backbone requires alignment + threshold changes to be "
    "measurable at all). Compare per-scenario vs run_009 baseline: "
    "targeted = d1_far/d2_mid/chokepoint_temporal_stability (all at 0 %); "
    "regression floor = d3_close (36.2 %). Keep iff targeted improves "
    "≥ 20 % absolute AND d3_close stays ≥ 16.2 %. Synthetic S1 harness "
    "is unaffected (independent 128-d random vectors; thresholds looser "
    "= more permissive match, never worse)."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=report,
    decision="inconclusive",  # set after comparison to run_009
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
