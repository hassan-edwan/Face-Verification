"""
Run 007 — Baseline measurement (post-rearchitecture)
=====================================================
First run under the synthetic-harness convention. No src/ changes;
this run freezes the current baseline success rates for every
scenario in src.live_eval.DEFAULT_SCENARIOS. Future runs compare
their per-scenario deltas against this artifact.

Hypothesis:  Capture the post-rearchitecture baseline — no knob
             changes, just the first per-scenario measurement under
             the new convention. Establishes S1 as the regression
             floor and S3 as the known-failure target.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.live_eval import run_all, build_artifact

RUN_ID      = "run_007"
HYPOTHESIS  = (
    "Baseline measurement under the new synthetic-harness convention; "
    "no knob changes. Establishes per-scenario floors for run_008+."
)
CONFIG_DIFF = {}   # no-op baseline

report   = run_all(trials=25, seed=42)
# Aggregate success_rate of 0.833 with baseline knobs is expected
# (S3 at 0 % is the documented failure; other 5 scenarios all at 100 %).
# Recording decision=keep because there's no change to revert.
decision = "keep"
notes    = (
    "No src/ change; this is the baseline. S3_crossing is an intentional "
    "0 %: the gatekeeper's hard tracking-lock guarantees a crossed identity "
    "stays assigned to the wrong track until forget()ten. Targeted by "
    "future runs. Every other scenario at 100 % on 25 trials."
)

artifact = build_artifact(
    run_id=RUN_ID, hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF, report=report,
    decision=decision, notes=notes,
)

out_path = os.path.join(PROJECT_ROOT, "outputs", "runs", f"{RUN_ID}.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"wrote -> {out_path}")
for s in report.scenarios:
    print(f"  {s.id:20s}  {s.successes}/{s.trials}  ({s.success_rate:.0%})")
print(f"aggregate: {report.aggregate['success_rate']:.3f}")
