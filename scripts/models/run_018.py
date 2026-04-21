"""
Run 018 — Revert REMATCH_THRESHOLD 0.20 → 0.30 under gallery-130 identification
===============================================================================
Hypothesis: run_017 exposed a 35 % enrollment dedup collision rate at
REMATCH=0.20 (46 of 130 SCface subjects merged into existing identities
during gallery build). Reverting REMATCH to 0.30 should admit more
subjects as unique identities. If the dedup rate drops to single digits
AND rank1_gk retrieval on the newly-seated subjects stays close to
run_017's 94 %+, REMATCH=0.20 is a mis-calibration and should be
reverted. If dedup stays high, the collision is embedder-bound (SCface
frontal mugshots have cross-subject cosines > 0.30 for many pairs) and
REMATCH can't fix it — a different mitigation (multi-template
enrollment, a better embedder, or stricter cross-track dedup using
MATCH_THRESHOLD instead of REMATCH) is needed.

Decision rule
-------------
- **revert** REMATCH to 0.30 (i.e. keep the change in run_018) iff
  dedup collisions drop by ≥ 20 absolute subjects AND
  rank1_gk on SCface d1_far drops by ≤ 5 pts absolute from run_017's
  94 % AND S1 synth floor holds ≥ 95 %.
- **keep** REMATCH at 0.20 (revert run_018) iff dedup stays high OR
  rank1_gk collapses — either tells us 0.30 isn't the right lever.
- **inconclusive** otherwise.

Note on scope: this run only measures identification + synth. The
verification-harness numbers at REMATCH=0.30 are already known from
run_015 (d1_far sr=53 %, TAR@FAR=1 %=98 %), so re-running them is
duplicative. The open question is identification, which run_017 only
measured at REMATCH=0.20.

Config change
-------------
  REMATCH_THRESHOLD: 0.20 → 0.30
All other knobs inherit run_017: MATCH=0.40, ENROLL=0.20, CONSENSUS=5,
MAX_EMBEDDINGS=8, MIN_FACE_PX=20, embedder=ArcFace buffalo_l, 5-pt.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("FACE_EMBEDDER", "arcface")

from src.real_data_eval import run_identification_all
from src.live_eval      import run_all, build_artifact, RunReport


RUN_ID      = "run_018"
HYPOTHESIS  = (
    "Revert REMATCH_THRESHOLD 0.20 → 0.30 under gallery-130 "
    "identification. Run_017 at REMATCH=0.20 produced 46/130 (35 %) "
    "enrollment dedup collisions on SCface — distinct subjects merging "
    "into existing identities at first-contact. Expected at "
    "REMATCH=0.30: dedup collisions drop to single digits; rank1_gk "
    "on surviving enrollees remains ≥ 90 % on d1_far (per run_015\u2019s "
    "verification numbers the embedder is strong at this threshold). "
    "If dedup stays high, collisions are embedder-bound (frontal "
    "mugshot cosines > 0.30 for many cross-subject pairs) and "
    "REMATCH is not the right lever."
)
CONFIG_DIFF: dict = {"REMATCH_THRESHOLD": [0.20, 0.30]}


print("[run_018] synth harness (regression floor)...")
synth_report = run_all()

print("[run_018] identification harness...")
ident_report, datasets = run_identification_all(datasets=["scface"])


combined_scenarios = list(synth_report.scenarios) + list(ident_report.scenarios)
synth_agg = synth_report.aggregate
ident_agg = ident_report.aggregate
combined_agg = {
    "total_trials":    synth_agg["total_trials"] + ident_agg["total_trials"],
    "total_successes": synth_agg["total_successes"] + ident_agg["total_successes"],
    "success_rate": (
        (synth_agg["total_successes"] + ident_agg["total_successes"]) /
        max(1, synth_agg["total_trials"] + ident_agg["total_trials"])
    ),
    "synth_aggregate":          synth_agg,
    "identification_aggregate": ident_agg,
}
combined_report = RunReport(scenarios=combined_scenarios, aggregate=combined_agg)

s1 = next((s for s in synth_report.scenarios if s.id.startswith("S1")), None)
s1_ok = (s1 is not None) and (s1.success_rate >= 0.95)

diag = ident_agg.get("scenario_diagnostics", {})
table_lines = []
for s in ident_report.scenarios:
    d = diag.get(s.id, {})
    table_lines.append(
        f"{s.id}: n_enrolled={d.get('n_enrolled', 0)}/130, "
        f"dedup_collisions={d.get('dedup_collisions', 0)}, "
        f"rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f}, "
        f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f}, "
        f"rank5_emb={d.get('rank5_emb_accept_rate', 0):.3f} "
        f"(n={d.get('n_embed_success', 0)})"
    )

# Decision is written by the script speculatively; user flips the
# JSON field post-hoc after reading the table.
notes = (
    f"REMATCH=0.30 under gallery-130 identification. Run_017 at "
    f"REMATCH=0.20 showed 46/130 dedup collisions (gallery seated "
    f"only 84 uniques). Key numbers:\n  "
    + "\n  ".join(table_lines)
    + f"\n  synth S1 floor: {'PASS' if s1_ok else 'FAIL'}"
    + (f" ({s1.success_rate:.0%})" if s1 else "")
    + ". Decision set by run author post-hoc per the rule in the "
      "run\u2019s docstring."
)

artifact = build_artifact(
    run_id=RUN_ID,
    hypothesis=HYPOTHESIS,
    config_diff=CONFIG_DIFF,
    report=combined_report,
    decision="inconclusive",
    notes=notes,
    eval_type="identification",
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
print("IDENTIFICATION (n_enrol | collisions | rank1_gk | rank1_emb | rank5_emb):")
for s in ident_report.scenarios:
    d = diag.get(s.id, {})
    print(f"  {s.id:40s}  "
          f"enrol={d.get('n_enrolled', 0)}/130  "
          f"coll={d.get('dedup_collisions', 0)}  "
          f"rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f}  "
          f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f}  "
          f"rank5_emb={d.get('rank5_emb_accept_rate', 0):.3f}  "
          f"(n={d.get('n_embed_success', 0)})")
print()
print(f"S1 floor: {'PASS' if s1_ok else 'FAIL'}"
      + (f" ({s1.success_rate:.0%})" if s1 else ""))
