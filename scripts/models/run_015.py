"""
Run 015 — Pre-REMATCH-drop baseline under hardened harness
===========================================================
First measurement at `REMATCH_THRESHOLD = 0.30` under the new
real-data harness. This is a **baseline**, not a hypothesis test:
zero pipeline knob diff against run_014's config apart from
temporarily restoring REMATCH to its pre-015-author value so that
run_016 can measure the 0.30 → 0.20 drop against an honestly-scored
pre-state instead of run_014's old-harness numbers.

Why this re-authoring
---------------------
The originally-authored run_015 proposed REMATCH 0.30 → 0.20 and
applied the knob change to `src/gatekeeper.py`, but never executed
(`outputs/runs/run_015.json` was never produced). The pre-audit
premise — "d1_far's 53 % will lift to ~75 % at REMATCH=0.20" —
was test-set-tuned (target delta read off run_014's failure
distribution). Under the new harness it needs a real baseline
to be scored against, which is what this run provides.

Harness changes folded in since run_014
---------------------------------------
- TAR @ FAR=1 % / 0.1 % per scenario via exhaustive cross-subject
  impostor pairs (was: genuine-only `success_rate`).
- Real `quick_quality_check` on every aligned crop; the gatekeeper
  update gate now fires honestly for low-quality frames (was:
  hard-coded `quality_score = 0.9` enroll / `0.7` query).
- ChokePoint split into `within_clip_sanity` (same-clip first-vs-
  last frame, retained as a sanity floor only), `cross_camera`
  (same session, different portal cam), and `cross_session` (same
  person ID across different sessions — the true re-identification
  scenario).
- TinyFace loader stub removed; `datasets=["scface", "chokepoint"]`.

Config
------
REMATCH_THRESHOLD=0.30 (temporarily restored for this run).
All other knobs inherit from run_014's kept state:
  MATCH_THRESHOLD=0.40, ENROLL_THRESHOLD=0.20, CONSENSUS_FRAMES=5,
  MAX_EMBEDDINGS_PER_IDENTITY=8, MIN_FACE_PX=20 (run_014),
  embedder=ArcFace buffalo_l (run_011), alignment=5-point (run_011).

Decision
--------
`inconclusive`. Baselines are not keep/revert actions — they become
the denominator for the next hypothesis. Run_016 is the real
hypothesis test (REMATCH 0.30 → 0.20) and consumes these numbers.
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
from src.live_eval      import run_all, build_artifact, RunReport


RUN_ID      = "run_015"
HYPOTHESIS  = (
    "Baseline at REMATCH_THRESHOLD=0.30 under the hardened harness. "
    "Establishes an honestly-scored pre-state for run_016's REMATCH "
    "0.30 → 0.20 test. Harness adds TAR@FAR via cross-subject "
    "impostors, a real quick_quality_check on every aligned crop, "
    "new ChokePoint cross_session + renamed within_clip_sanity "
    "scenarios, and drops the TinyFace stub. Zero pipeline knob diff "
    "against run_014's kept state."
)
CONFIG_DIFF: dict = {}  # baseline — no knob change


print("[run_015] synth harness...")
synth_report = run_all()

print("[run_015] real harness...")
real_report, datasets = run_real_all(datasets=["scface", "chokepoint"])


combined_scenarios = list(synth_report.scenarios) + list(real_report.scenarios)
synth_agg = synth_report.aggregate
real_agg  = real_report.aggregate
combined_agg = {
    "total_trials":    synth_agg["total_trials"] + real_agg["total_trials"],
    "total_successes": synth_agg["total_successes"] + real_agg["total_successes"],
    "success_rate": (
        (synth_agg["total_successes"] + real_agg["total_successes"]) /
        max(1, synth_agg["total_trials"] + real_agg["total_trials"])
    ),
    "synth_aggregate": synth_agg,
    "real_aggregate":  real_agg,
}
combined_report = RunReport(scenarios=combined_scenarios, aggregate=combined_agg)

s1 = next((s for s in synth_report.scenarios if s.id.startswith("S1")), None)
s1_ok = (s1 is not None) and (s1.success_rate >= 0.95)

diag = real_agg.get("scenario_diagnostics", {})
gap_lines = []
for s in real_report.scenarios:
    d = diag.get(s.id, {})
    t01 = (d.get("tar_at_far") or {}).get("0.01")
    if t01 is None:
        gap_lines.append(
            f"{s.id}: success_rate={s.success_rate:.3f}, "
            f"TAR@FAR=1%: n/a (n_imp={d.get('n_impostors', 0)})"
        )
    else:
        gap = s.success_rate - t01["tar"]
        gap_lines.append(
            f"{s.id}: success_rate={s.success_rate:.3f}, "
            f"TAR@FAR=1%={t01['tar']:.3f} (threshold={t01['threshold']:.3f}, "
            f"gap={gap:+.3f}, n_gen={d.get('n_genuines', 0)}, "
            f"n_imp={d.get('n_impostors', 0)})"
        )

notes = (
    "Pre-REMATCH-drop baseline at REMATCH_THRESHOLD=0.30 under "
    "hardened harness. Per-scenario TAR@FAR=1 % vs genuine-only "
    "success_rate:\n  "
    + "\n  ".join(gap_lines)
    + f"\n  synth S1 floor: {'PASS' if s1_ok else 'FAIL'}"
    + (f" ({s1.success_rate:.0%})" if s1 else "")
    + ". Decision=inconclusive (baseline — no keep/revert)."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=combined_report,
    decision="inconclusive",
    notes=notes,
    eval_type="mixed",
    datasets=datasets,
)

out_path = os.path.join(PROJECT_ROOT, "outputs", "runs", f"{RUN_ID}.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"wrote -> {out_path}")
print()
print("SYNTH:")
for s in synth_report.scenarios:
    print(f"  {s.id:40s}  {s.successes}/{s.trials}  ({s.success_rate:.0%})")
print(f"  aggregate: {synth_agg['success_rate']:.3f}")
print()
print("REAL  (success_rate | TAR@FAR=1% | gap):")
for s in real_report.scenarios:
    d = diag.get(s.id, {})
    t01 = (d.get("tar_at_far") or {}).get("0.01")
    if t01 is None:
        print(f"  {s.id:40s}  sr={s.success_rate:.3f}  tar@1%=   n/a  "
              f"(n_imp={d.get('n_impostors', 0)})")
    else:
        gap = s.success_rate - t01["tar"]
        print(f"  {s.id:40s}  sr={s.success_rate:.3f}  "
              f"tar@1%={t01['tar']:.3f}  gap={gap:+.3f}  "
              f"(n_gen={d.get('n_genuines', 0)}, "
              f"n_imp={d.get('n_impostors', 0)})")
print(f"  aggregate sr: {real_agg['success_rate']:.3f}")
print()
print(f"S1 floor: {'PASS' if s1_ok else 'FAIL'}"
      + (f" ({s1.success_rate:.0%})" if s1 else ""))
