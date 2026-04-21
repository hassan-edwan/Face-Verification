"""
Run 021 — Drift-detection in the tracking-lock path (S3 crossing fix)
======================================================================
S3 (crowded-scene crossing) has been at 0 % since run_007 — the hard
tracking-lock at `gatekeeper.py:137` was designed to prevent flicker
by NEVER re-running similarity search once a track was bound to an
identity. That protection turns pathological when the IoU tracker
swaps track→face assignments mid-sequence: the locked track keeps
returning its pre-swap identity while silently polluting that
identity's bank with the new (wrong) face's embeddings.

Run_021 adds a drift-break inside the locked path:
  - On every locked-track frame, compute max cosine between the
    incoming embedding and the locked identity's bank rows.
  - If max cosine < REMATCH_THRESHOLD (0.20), the embedding is no
    longer plausibly the locked identity. Run `_find_best_match_with_margin`
    against the full bank.
  - If another identity scores high enough to pass the run_019
    first-contact gate (score ≥ MATCH_THRESHOLD, OR score ≥ REMATCH_
    THRESHOLD AND margin ≥ MIN_MATCH_MARGIN), BREAK the lock:
    re-lock the track to the new identity and update that identity's
    bank (not the old one's — preventing silent corruption).
  - If no alternative is confidently better (just low-quality or
    occluded frame), HOLD the lock but skip the bank update.

Why this doesn't break benign drift
-----------------------------------
  - S5 pose drift: cosine 0.80–0.98, well above REMATCH=0.20. Drift-
    break doesn't fire. Normal bank refinement continues.
  - S6 illumination bias: cosine ≈ 0.88. Same.
  - S8 distance noise: cosine ≈ 0.85–0.95. Same.
  - Multi-shot enrollment (run_020) Phase 2: SCface same-subject
    pose variants sit at ArcFace cosine > 0.30 even for L4/R4 extreme
    profiles. Drift doesn't fire; even if it did at one outlier
    frame, the drift-break only re-locks when ANOTHER identity beats
    the gate — unlikely during enrollment where the bank contains
    the same subject's other templates.

Hypothesis
----------
  - **S3 synth: 0 % → ≥ 95 %** (primary target).
  - All other synth scenarios hold at 100 %.
  - Verification TAR@FAR unchanged across all 6 scenarios (drift
    detection only affects locked-track frames — verification pairs
    use fresh Gatekeepers per pair with only 2 unique track_ids
    that never both hit the locked path).
  - Identification (multi-shot): rank1_gk on d1/d2/d3 unchanged
    within ±1 pt vs run_020 (Phase 2 enrichment runs through the
    locked path; we need the drift-break to NOT fire on pose
    variants of the same subject).
  - S1 floor 100 %.

Config
------
  No new knob. Drift-break uses REMATCH_THRESHOLD as the "plausibly
  same person" floor and the run_019 first-contact gate as the re-
  lock admission rule. Pure behavioral change in the tracking-lock
  branch of `Gatekeeper.process()`.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("FACE_EMBEDDER", "arcface")

from src.real_data_eval import (
    run_real_all, run_identification_all, DEFAULT_MULTISHOT_POSES,
)
from src.live_eval      import run_all, build_artifact, RunReport


RUN_ID      = "run_021"
HYPOTHESIS  = (
    "Add drift-detection to the Gatekeeper tracking-lock path. When "
    "the incoming embedding's max cosine against the locked identity's "
    "bank drops below REMATCH_THRESHOLD=0.20, rerun the full match "
    "against the bank and re-lock to a different identity iff it "
    "passes the run_019 first-contact gate (MATCH_THRESHOLD=0.40 OR "
    "REMATCH_THRESHOLD=0.20 with margin ≥ MIN_MATCH_MARGIN=0.05). "
    "Primary target: synth S3 crossing 0 % → ≥ 95 %. Regression "
    "guards: all other synth scenarios ≥ 95 %, verification TAR@FAR "
    "unchanged, identification multi-shot rank1_gk within ±1 pt of "
    "run_020."
)
CONFIG_DIFF: dict = {"gatekeeper_drift_break": ["off", "on"]}


print("[run_021] synth harness (S3 primary target)...")
synth_report = run_all()

print("[run_021] verification harness (SCface + ChokePoint)...")
real_report, real_datasets = run_real_all(datasets=["scface", "chokepoint"])

print("[run_021] identification harness (multi-shot, 5 poses)...")
ident_report, ident_datasets = run_identification_all(
    datasets=["scface"],
    multishot_poses=DEFAULT_MULTISHOT_POSES,
)


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

# Primary target: S3
s3 = next((s for s in synth_report.scenarios if s.id == "S3_crossing"), None)
s3_ok = (s3 is not None) and (s3.success_rate >= 0.95)

s1 = next((s for s in synth_report.scenarios if s.id.startswith("S1")), None)
s1_ok = (s1 is not None) and (s1.success_rate >= 0.95)

# Collect regression-guard signals from verification + identification.
real_diag  = real_agg.get("scenario_diagnostics", {})
ident_diag = ident_agg.get("scenario_diagnostics", {})

synth_table = "\n  ".join(
    f"{s.id}: {s.success_rate:.3f} ({s.successes}/{s.trials})"
    for s in synth_report.scenarios
)
real_table = "\n  ".join(
    f"{s.id}: sr={s.success_rate:.3f}, "
    f"TAR@1%={((real_diag.get(s.id, {}).get('tar_at_far') or {}).get('0.01') or {}).get('tar', 0):.3f}"
    for s in real_report.scenarios
)
ident_table = "\n  ".join(
    f"{s.id}: enrol={ident_diag.get(s.id, {}).get('n_enrolled', 0)}/130, "
    f"coll={ident_diag.get(s.id, {}).get('dedup_collisions', 0)}, "
    f"rank1_gk={ident_diag.get(s.id, {}).get('rank1_gk_accept_rate', 0):.3f}, "
    f"rank1_emb={ident_diag.get(s.id, {}).get('rank1_emb_accept_rate', 0):.3f}"
    for s in ident_report.scenarios
)

notes = (
    f"S3 primary target: {'PASS' if s3_ok else 'FAIL'}"
    f" ({s3.success_rate:.0%})" if s3 else "S3 missing"
) + (
    f". S1 floor: {'PASS' if s1_ok else 'FAIL'}"
    f" ({s1.success_rate:.0%})" if s1 else "S1 missing"
) + (
    f".\nSynth:\n  {synth_table}"
    f"\nVerification:\n  {real_table}"
    f"\nIdentification (multi-shot):\n  {ident_table}"
    f"\nDecision set post-hoc."
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
    marker = " <-- S3" if s.id == "S3_crossing" else ""
    print(f"  {s.id:40s}  {s.successes}/{s.trials}  "
          f"({s.success_rate:.0%}){marker}")
print(f"  aggregate: {synth_agg['success_rate']:.3f}")
print()
print("VERIFICATION (sr | TAR@1%):")
for s in real_report.scenarios:
    d = real_diag.get(s.id, {})
    t01 = (d.get("tar_at_far") or {}).get("0.01")
    t01_str = f"{t01['tar']:.3f}" if t01 else "   n/a"
    print(f"  {s.id:40s}  sr={s.success_rate:.3f}  tar@1%={t01_str}")
print()
print("IDENTIFICATION (enrol | coll | rank1_gk | rank1_emb):")
for s in ident_report.scenarios:
    d = ident_diag.get(s.id, {})
    print(f"  {s.id:50s}  "
          f"enrol={d.get('n_enrolled', 0)}/130  "
          f"coll={d.get('dedup_collisions', 0)}  "
          f"rank1_gk={d.get('rank1_gk_accept_rate', 0):.3f}  "
          f"rank1_emb={d.get('rank1_emb_accept_rate', 0):.3f}")
print()
print(f"S3 target: {'PASS' if s3_ok else 'FAIL'}"
      + (f" ({s3.success_rate:.0%})" if s3 else ""))
print(f"S1 floor:  {'PASS' if s1_ok else 'FAIL'}"
      + (f" ({s1.success_rate:.0%})" if s1 else ""))
