"""
Run 017 — Gallery-130 identification on SCface
===============================================
First measurement under the new identification harness. Enrolls every
SCface subject's frontal mugshot into one Gatekeeper, then queries each
subject's cam1 image at each distance bucket against that gallery.

Why this matters
----------------
Runs 009–016 reported "success_rate" on a gallery of ONE (enroll one
subject, query that subject's surveillance frame, check the gatekeeper
returns the right name). With gallery=1 there are no near-impostors,
no competing identities, no Pareto trade between TAR and FAR. Those
conditions do not hold in the live pipeline, which accumulates many
identities. The audit flagged that run_014's 53 % d1_far and run_016's
91 % d1_far could collapse to any rate at gallery=130 — that gap is
what this run measures.

Metrics
-------
- `rank1_gk_accept_rate` (the `success_rate` column): probability
  `gk._find_best_match(q)` returns the true subject AND its cosine
  clears REMATCH_THRESHOLD. This is what the live pipeline delivers.
- `rank1_emb_accept_rate`: true subject is the argmax of cosine against
  the gallery, ignoring REMATCH. Embedder-only capacity.
- `rank5_emb` / `rank10_emb`: open-set charity metrics — true subject
  is in top-5 / top-10. Wide gap from rank-1 means lookalikes are
  crowding the correct answer even if the cosine score is "good enough".

Gap reading
-----------
  rank1_gk < rank1_emb : REMATCH is rejecting genuinely correct top-1
                         matches. Lowering REMATCH would help.
  rank1_emb < rank5_emb: Embedder is placing the right subject in the
                         near-neighborhood but can't separate it from
                         lookalikes. Adaptive threshold / per-scene
                         calibration won't fix this — only a better
                         embedder (or more enrollment templates per
                         subject) would.

Config (unchanged from run_016; no pipeline knobs touched)
----------------------------------------------------------
  MATCH_THRESHOLD=0.40, REMATCH_THRESHOLD=0.20, ENROLL_THRESHOLD=0.20,
  CONSENSUS_FRAMES=5, MAX_EMBEDDINGS_PER_IDENTITY=8, MIN_FACE_PX=20,
  embedder=ArcFace buffalo_l, alignment=5-point.
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


RUN_ID      = "run_017"
HYPOTHESIS  = (
    "First measurement of gallery-130 identification on SCface under "
    "the current ArcFace / REMATCH=0.20 config (unchanged from run_016). "
    "Phase 2 of the hardening: tests the audit's prediction that "
    "gallery-1 success_rate overstates live-pipeline performance. "
    "Reports rank-1 gk accept (production number), rank-1 emb (embedder "
    "ceiling ignoring REMATCH), and rank-5/10 emb (lookalike crowding). "
    "Expected: d3_close rank-1 remains near-ceiling; d2_mid near-ceiling; "
    "d1_far drops materially from run_016's gallery-1 91 % as 129 "
    "distractors compete for the top slot."
)
CONFIG_DIFF: dict = {}  # measurement extension, no pipeline knobs changed


print("[run_017] synth harness (regression floor + harness liveness)...")
synth_report = run_all()

print("[run_017] identification harness...")
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
    "synth_aggregate":         synth_agg,
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
        f"{s.id}: rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f} "
        f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f} "
        f"rank5_emb={d.get('rank5_emb_accept_rate', 0):.3f} "
        f"rank10_emb={d.get('rank10_emb_accept_rate', 0):.3f} "
        f"(gallery={d.get('n_enrolled', 0)}, "
        f"n={d.get('n_embed_success', 0)})"
    )

notes = (
    "Gallery-130 identification baseline under ArcFace / REMATCH=0.20. "
    "Zero pipeline knob diff from run_016; the comparison is "
    "verification (gallery=1) vs identification (gallery=N) under "
    "identical config.\n  "
    + "\n  ".join(table_lines)
    + f"\n  synth S1 floor: {'PASS' if s1_ok else 'FAIL'}"
    + (f" ({s1.success_rate:.0%})" if s1 else "")
    + ". Decision=inconclusive: baseline. Per-scenario rank1_gk vs "
      "rank1_emb gap tells you whether REMATCH is the bottleneck "
      "(lower REMATCH would help) or the embedder is "
      "(only a better embedder or more templates per subject would)."
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
print("IDENTIFICATION (rank1_gk | rank1_emb | rank5_emb | rank10_emb):")
for s in ident_report.scenarios:
    d = diag.get(s.id, {})
    print(f"  {s.id:40s}  "
          f"rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f}  "
          f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f}  "
          f"rank5_emb={d.get('rank5_emb_accept_rate', 0):.3f}  "
          f"rank10_emb={d.get('rank10_emb_accept_rate', 0):.3f}  "
          f"(gallery={d.get('n_enrolled', 0)}, "
          f"n={d.get('n_embed_success', 0)})")
print()
print(f"S1 floor: {'PASS' if s1_ok else 'FAIL'}"
      + (f" ({s1.success_rate:.0%})" if s1 else ""))
