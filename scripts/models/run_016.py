"""
Run 016 — First honest measurement of REMATCH=0.20 config
==========================================================
Zero knob diff against the on-disk gatekeeper state. The REMATCH
threshold drop (0.30 → 0.20) authored as run_015 was applied to
`src/gatekeeper.py` but never executed — no `run_015.json` exists.
Run_016 is therefore the first artifact under that config, and it
lands under the hardened real-data harness:

  - TAR @ FAR=1 % / 0.1 % per scenario via exhaustive cross-subject
    impostor pairs (was: genuine-only `success_rate`)
  - Real `quick_quality_check` on every aligned crop (was: hard-coded
    `quality_score = 0.9` enroll / `0.7` query — the quality gate
    bypass flagged in the audit)
  - ChokePoint split into `within_clip_sanity` (near-duplicate first-
    vs-last frame of one walk, retained as a regression floor, no
    longer cited as generalization evidence) + `cross_camera` + a new
    `cross_session` scenario (same person-id across different walks —
    the actual re-identification signal)
  - TinyFace loader stub deleted; `datasets=["scface", "chokepoint"]`

Purpose
-------
Answer: does `REMATCH = 0.20` hold up under impostor-aware scoring?
The concern from the audit: lowering REMATCH to the same value as
ENROLL (0.20) collapses the UNCERTAIN band and pushes the match
threshold down into ArcFace's stranger-pair cosine tail (0.05–0.20).
Under the old genuine-only harness this lifts `success_rate` for free
— there are no impostors to push back. Under TAR@FAR=1 % there are.

Interpretation guide (post-run)
-------------------------------
- `success_rate` (genuine-only, historical metric): compared to
  run_014's numbers, a scenario rising here under REMATCH=0.20
  could be either a real gain or threshold inflation.
- `tar_at_far["0.01"].tar` (new, honest): the fraction of genuine
  pairs that beat the cosine threshold admitting only 1 % of
  impostors. Scenarios whose TAR@FAR sits far below their
  success_rate were threshold-inflated — the lax REMATCH admitted
  the genuines but would admit impostors at a similar rate if
  any existed. Those scenarios' "keep" decisions do not survive.
- `chokepoint_cross_session` is a brand-new signal. If its
  success_rate is near-zero, ChokePoint's `<person id>` is not
  stable across sessions under the portal subset on disk and the
  scenario needs a different pairing strategy. If meaningfully
  non-zero, it becomes the primary generalization metric for
  future knob changes (dethroning the old
  `chokepoint_temporal_stability`, now `within_clip_sanity`,
  which was same-clip near-duplicate matching).

Decision
--------
`inconclusive`. This run re-establishes the baseline under honest
measurement; keep/revert of REMATCH=0.20 belongs to run_017 after
reading these numbers.
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


RUN_ID      = "run_016"
HYPOTHESIS  = (
    "Zero pipeline knob diff — harness protocol change only. First "
    "honest measurement of the REMATCH=0.20 config (authored as "
    "run_015, applied to src/gatekeeper.py, but never executed — "
    "no run_015.json exists). New harness adds TAR@FAR=1%/0.1% via "
    "exhaustive cross-subject impostors, a real quick_quality_check "
    "on every aligned crop, a within_clip_sanity / cross_session "
    "split on ChokePoint, and drops the TinyFace stub. Expected: "
    "d1_far's 53 % success_rate from run_014 will NOT hold at "
    "TAR@FAR=1 %; if TAR is materially below success_rate, the "
    "REMATCH=0.20 move was threshold inflation and reverts in "
    "run_017."
)
CONFIG_DIFF: dict = {"REMATCH_THRESHOLD": [0.30, 0.20]}


print("[run_016] synth harness...")
synth_report = run_all()

print("[run_016] real harness...")
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

# S1 synth floor — documented in notes, not enforced here since this
# is a baseline, not a hypothesis test.
s1 = next((s for s in synth_report.scenarios if s.id.startswith("S1")), None)
s1_ok = (s1 is not None) and (s1.success_rate >= 0.95)

# Per-scenario success_rate vs TAR@FAR=1 % gap — the key table.
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
    "Protocol change only — zero pipeline knob diff. First measurement "
    "under REMATCH_THRESHOLD=0.20 (applied to src/gatekeeper.py with "
    "run_015's author, never executed). Per-scenario TAR@FAR=1 % vs "
    "genuine-only success_rate:\n  "
    + "\n  ".join(gap_lines)
    + f"\n  synth S1 floor: {'PASS' if s1_ok else 'FAIL'}"
    + (f" ({s1.success_rate:.0%})" if s1 else "")
    + ". Decision=inconclusive: measurement re-baseline, not a "
      "keep/revert. A scenario whose TAR@FAR=1 % sits materially "
      "below its success_rate was threshold-inflated by the "
      "REMATCH=0.20 drop; run_017 reverts REMATCH to 0.30 for the "
      "worst offender and re-measures."
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
