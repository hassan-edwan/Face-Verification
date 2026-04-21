"""
One-shot post-run helper: compare run_011 vs run_009 baseline, apply
the decision rule, amend run_011.json's `decision` + `notes`, append
a row to outputs/runs/README.md. Throwaway — not part of the harness.
"""

import json
import pathlib
import sys


NEW = "run_011"
OLD = "run_009"


def main():
    new_p = pathlib.Path(f"outputs/runs/{NEW}.json")
    old_p = pathlib.Path(f"outputs/runs/{OLD}.json")
    new = json.loads(new_p.read_text())
    old = json.loads(old_p.read_text())

    new_by = {s["id"]: s for s in new["scenarios"]}
    old_by = {s["id"]: s for s in old["scenarios"]}

    targeted = {"scface_mugshot_to_d1_far", "scface_mugshot_to_d2_mid",
                "chokepoint_temporal_stability"}
    regression_floor = "scface_mugshot_to_d3_close"

    print(f"{'SCENARIO':38s} {'run_009':>14s}  {'run_011':>14s}  delta")
    print("-" * 80)
    deltas = {}
    for sid in sorted(new_by):
        o = old_by.get(sid)
        n = new_by[sid]
        o_rate = o["success_rate"] if o else 0
        n_rate = n["success_rate"]
        d = n_rate - o_rate
        deltas[sid] = d
        o_str = f"{o['successes']}/{o['trials']} ({o_rate*100:.1f}%)" if o else "    -    "
        n_str = f"{n['successes']}/{n['trials']} ({n_rate*100:.1f}%)"
        tag = "  ← TARGETED" if sid in targeted else (
              "  ← FLOOR"    if sid == regression_floor else "")
        print(f"{sid:38s} {o_str:>14s}  {n_str:>14s}  {d*100:+5.1f}%{tag}")

    # Decision rule per docs/prompts/improve_pose_and_distance.md
    targeted_hit    = any(deltas.get(s, 0) >= 0.20 for s in targeted)
    floor_regressed = deltas.get(regression_floor, 0) <= -0.20

    print()
    print(f"targeted scenario lifted ≥20%?  {targeted_hit}")
    print(f"d3_close regressed ≥20%?        {floor_regressed}")

    if not floor_regressed and (targeted_hit or deltas.get(regression_floor, 0) >= 0.20):
        decision = "keep"
    elif floor_regressed:
        decision = "revert"
    else:
        decision = "inconclusive"
    print(f"DECISION: {decision}")
    return decision, deltas


if __name__ == "__main__":
    main()
