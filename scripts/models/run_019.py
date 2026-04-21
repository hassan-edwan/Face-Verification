"""
Run 019 — Margin-aware first-contact matching (architectural split)
====================================================================
Decouples the two conflicting roles of `REMATCH_THRESHOLD` exposed by
run_017 + run_018:

  1. Cross-track dedup at enrollment: "is this a duplicate of someone
     already in the bank?" — wants a strict threshold, else distinct
     subjects whose mugshots happen to sit at cosine ~0.25 against each
     other get merged (run_017: 46/130 collisions at REMATCH=0.20).

  2. Re-acquisition of a lost track: "did this new track_id return the
     same walker?" — wants a loose threshold, else tail-genuine
     matches at d1_far cosine ~0.25 fail to re-lock (run_018:
     d1_far rank1_gk collapsed 94 % → 52 % at REMATCH=0.30).

One knob can't satisfy both. This run adds a rank-1-vs-rank-2 margin
check in `gatekeeper._find_best_match_with_margin` and the first-
contact decision at `gatekeeper.py:152-184` now reads:

  score ≥ MATCH_THRESHOLD (0.40)                    → MATCH
  score ≥ REMATCH_THRESHOLD (0.20) AND margin ≥
    MIN_MATCH_MARGIN (0.05)                         → MATCH (re-acq)
  ENROLL_THRESHOLD ≤ score < MATCH_THRESHOLD        → UNCERTAIN
  score < ENROLL_THRESHOLD                          → accumulate (new)

The margin is computed across DISTINCT identities (best-per-name over
the bank). At cross-subject mugshot collisions, the "victim" identity's
mugshot should have a nearly flat score across the bank — low margin,
refused match, no collision. At re-acquisition of a genuine walker,
their own bank entry should rise clearly above everyone else's — high
margin, match accepted even in the 0.20–0.40 tail.

Hypothesis
----------
  - Identification on SCface: dedup collisions drop from 46/130
    (run_017) to ≤ 10, rank1_gk on d1_far stays ≥ 85 %.
  - Verification on SCface: d1_far success_rate retains most of
    run_016's 91 % — at most a ~5 pt regression (the genuines
    caught by REMATCH=0.20 alone that sit in ambiguous margin).
  - Synth S1 floor holds ≥ 95 %.

Config
------
  New knob: MIN_MATCH_MARGIN = 0.05
  All others inherit run_018 → run_016 lineage (REMATCH=0.20,
  MATCH=0.40, ENROLL=0.20, CONSENSUS=5, MAX=8, MIN_FACE_PX=20,
  embedder=ArcFace buffalo_l, 5-point alignment).
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("FACE_EMBEDDER", "arcface")

from src.real_data_eval import run_real_all, run_identification_all
from src.live_eval      import run_all, build_artifact, RunReport


RUN_ID      = "run_019"
HYPOTHESIS  = (
    "Add rank-1-vs-rank-2 margin check at first-contact. MIN_MATCH_"
    "MARGIN=0.05. Scores in [REMATCH=0.20, MATCH=0.40) now require "
    "clear rank-1 separation to be accepted as re-acquisition; "
    "scores ≥ MATCH=0.40 match unconditionally. Expected: "
    "identification dedup collisions drop 46 → ≤ 10 (cross-subject "
    "mugshot near-ties refused), rank1_gk on d1_far stays ≥ 85 % "
    "(genuine re-acq keeps its clear rank-1 winner in most cases). "
    "Verification d1_far retains most of run_016\u2019s 91 %. S1 floor "
    "holds. If all true, REMATCH and MATCH are now serving their "
    "natural semantics and single-knob mis-calibration is resolved."
)
CONFIG_DIFF: dict = {"MIN_MATCH_MARGIN": [None, 0.05]}


print("[run_019] synth harness...")
synth_report = run_all()

print("[run_019] verification harness (SCface + ChokePoint)...")
real_report, real_datasets = run_real_all(datasets=["scface", "chokepoint"])

print("[run_019] identification harness (SCface gallery-130)...")
ident_report, ident_datasets = run_identification_all(datasets=["scface"])


datasets = sorted(set(real_datasets + ident_datasets))
combined_scenarios = (list(synth_report.scenarios)
                      + list(real_report.scenarios)
                      + list(ident_report.scenarios))

synth_agg = synth_report.aggregate
real_agg  = real_report.aggregate
ident_agg = ident_report.aggregate

total_t = (synth_agg["total_trials"] + real_agg["total_trials"]
           + ident_agg["total_trials"])
total_s = (synth_agg["total_successes"] + real_agg["total_successes"]
           + ident_agg["total_successes"])
combined_agg = {
    "total_trials":             total_t,
    "total_successes":          total_s,
    "success_rate":             (total_s / total_t) if total_t else 0.0,
    "synth_aggregate":          synth_agg,
    "real_aggregate":           real_agg,
    "identification_aggregate": ident_agg,
}
combined_report = RunReport(scenarios=combined_scenarios, aggregate=combined_agg)

s1 = next((s for s in synth_report.scenarios if s.id.startswith("S1")), None)
s1_ok = (s1 is not None) and (s1.success_rate >= 0.95)

real_diag  = real_agg.get("scenario_diagnostics", {})
ident_diag = ident_agg.get("scenario_diagnostics", {})

real_lines = []
for s in real_report.scenarios:
    d = real_diag.get(s.id, {})
    t01 = (d.get("tar_at_far") or {}).get("0.01")
    if t01 is None:
        real_lines.append(f"{s.id}: sr={s.success_rate:.3f}, TAR@1%=n/a")
    else:
        gap = s.success_rate - t01["tar"]
        real_lines.append(
            f"{s.id}: sr={s.success_rate:.3f}, TAR@1%={t01['tar']:.3f} "
            f"(gap={gap:+.3f})"
        )

ident_lines = []
for s in ident_report.scenarios:
    d = ident_diag.get(s.id, {})
    ident_lines.append(
        f"{s.id}: n_enrol={d.get('n_enrolled', 0)}/130, "
        f"coll={d.get('dedup_collisions', 0)}, "
        f"rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f}, "
        f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f}"
    )

notes = (
    "Margin-aware first-contact matching (MIN_MATCH_MARGIN=0.05). "
    "Verification and identification measured together so the audit's "
    "single-knob problem can be judged across both axes. "
    "Verification TAR@FAR comparison:\n  "
    + "\n  ".join(real_lines)
    + "\nIdentification (gallery-130):\n  "
    + "\n  ".join(ident_lines)
    + f"\n  synth S1 floor: {'PASS' if s1_ok else 'FAIL'}"
    + (f" ({s1.success_rate:.0%})" if s1 else "")
    + ". Decision pending post-hoc review of all three tables."
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
print("VERIFICATION (sr | TAR@1% | gap):")
for s in real_report.scenarios:
    d = real_diag.get(s.id, {})
    t01 = (d.get("tar_at_far") or {}).get("0.01")
    if t01 is None:
        print(f"  {s.id:40s}  sr={s.success_rate:.3f}  tar@1%=   n/a")
    else:
        gap = s.success_rate - t01["tar"]
        print(f"  {s.id:40s}  sr={s.success_rate:.3f}  "
              f"tar@1%={t01['tar']:.3f}  gap={gap:+.3f}")
print()
print("IDENTIFICATION (n_enrol | collisions | rank1_gk | rank1_emb | rank5_emb):")
for s in ident_report.scenarios:
    d = ident_diag.get(s.id, {})
    print(f"  {s.id:40s}  "
          f"enrol={d.get('n_enrolled', 0)}/130  "
          f"coll={d.get('dedup_collisions', 0)}  "
          f"rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f}  "
          f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f}  "
          f"rank5_emb={d.get('rank5_emb_accept_rate', 0):.3f}")
print()
print(f"S1 floor: {'PASS' if s1_ok else 'FAIL'}"
      + (f" ({s1.success_rate:.0%})" if s1 else ""))
