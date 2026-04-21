"""
Per-run scenario plot — bar chart of scenario success rates. When the
run artifact carries per-scenario TAR@FAR diagnostics (real-data runs
under the hardened harness, run_015+), adds a second bar per scenario
for TAR@FAR=1 % alongside the historical success_rate. The visible gap
between the two is the honest read: success_rate can be threshold-
inflated; TAR@FAR can't.

Auto-discovers every outputs/runs/run_*.json and emits
outputs/plots/run_NNN_scenarios.png alongside. Skips runs whose JSON
predates the synthetic-harness schema (no "scenarios" field).

Usage:
    python scripts/plot_run.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


RUNS_DIR = "outputs/runs"
OUT_DIR  = "outputs/plots"


def plot_run(json_path: str, out_path: str) -> bool:
    with open(json_path) as f:
        data = json.load(f)
    scenarios = data.get("scenarios")
    if not scenarios:
        return False

    ids   = [s["id"]           for s in scenarios]
    rates = [s["success_rate"] for s in scenarios]
    run_id = data.get("run_id", os.path.basename(json_path))
    # eval_type is missing on pre-schema-extension runs (007/008);
    # default to "synth" so old plots keep labeling correctly.
    eval_type = data.get("eval_type", "synth")

    # Merge per-scenario diagnostics from every possible stashing site.
    # A single run can carry both verification (TAR@FAR) and
    # identification (rank-k) diagnostics if it exercises both harnesses,
    # so taking the union is required — stopping at the first dict would
    # hide one of the two.
    agg = data.get("aggregate", {}) or {}
    diag: dict = {}
    for sub in (agg.get("scenario_diagnostics"),
                (agg.get("real_aggregate") or {}).get("scenario_diagnostics"),
                (agg.get("identification_aggregate") or {}).get(
                    "scenario_diagnostics")):
        if sub:
            diag.update(sub)
    # Per-scenario secondary metrics. A scenario may have TAR@FAR
    # (verification) or rank1_emb/rank5_emb (identification) — or none
    # (synth). We unify both into two secondary-bar slots so mixed runs
    # (like run_019) can show verification and identification scenarios
    # in one plot without either type losing its companion metric.
    secondary1 = []   # TAR@FAR=1% for verif, rank1_emb for ident, None otherwise
    secondary2 = []   # rank5_emb only for ident; None for verif/synth
    for sid in ids:
        d = diag.get(sid, {})
        r1e = d.get("rank1_emb_accept_rate")
        r5e = d.get("rank5_emb_accept_rate")
        t01 = (d.get("tar_at_far") or {}).get("0.01")
        if r1e is not None:
            secondary1.append(r1e)
            secondary2.append(r5e)
        elif t01 is not None:
            secondary1.append(t01["tar"])
            secondary2.append(None)
        else:
            secondary1.append(None)
            secondary2.append(None)
    has_any_secondary = any(s is not None for s in secondary1)
    has_any_rank5     = any(s is not None for s in secondary2)

    # Scale width with scenario count so per-label text stays readable
    # even when a run_NNN bundles 4+ scenarios.
    width = max(10, 1 + 1.5 * len(ids))
    fig, ax = plt.subplots(figsize=(width, 5.5))

    if has_any_secondary:
        # Two or three grouped bars per scenario. The secondary-1 slot
        # carries "the honest metric" — TAR@FAR=1 % for verification
        # scenarios and rank1_emb for identification. They land in the
        # same slot, same color, because they answer the same question
        # from different angles: what rate does the embedder / threshold
        # deliver when calibrated against a real non-match distribution?
        x = np.arange(len(ids))
        bar_w = 0.28 if has_any_rank5 else 0.38
        sr_offset = (-bar_w if has_any_rank5 else -bar_w / 2)
        sr_bars = ax.bar(x + sr_offset, rates, bar_w,
                         color=[_color_for(r) for r in rates],
                         label="success_rate (live pipeline)",
                         edgecolor="black", linewidth=0.3)
        s1_offset = 0.0 if has_any_rank5 else bar_w / 2
        s1_heights = [s if s is not None else 0.0 for s in secondary1]
        s1_colors  = ["#4a78c4" if s is not None else "#e5e5e5"
                      for s in secondary1]
        s1_bars = ax.bar(x + s1_offset, s1_heights, bar_w,
                         color=s1_colors,
                         label="TAR@FAR=1% (verif) / rank1_emb (ident)",
                         edgecolor="black", linewidth=0.3)
        if has_any_rank5:
            s2_heights = [s if s is not None else 0.0 for s in secondary2]
            s2_colors  = ["#8fb5e0" if s is not None else "#e5e5e5"
                          for s in secondary2]
            s2_bars = ax.bar(x + bar_w, s2_heights, bar_w,
                             color=s2_colors, label="rank5_emb",
                             edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        for bar, rate in zip(sr_bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.015,
                    f"{rate:.0%}", ha="center", fontsize=8)
        for bar, s in zip(s1_bars, secondary1):
            if s is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "n/a", ha="center", fontsize=7, color="#888")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, s + 0.015,
                        f"{s:.0%}", ha="center", fontsize=8)
        if has_any_rank5:
            for bar, s in zip(s2_bars, secondary2):
                if s is None:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                            "n/a", ha="center", fontsize=7, color="#888")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, s + 0.015,
                            f"{s:.0%}", ha="center", fontsize=8)
    else:
        bars = ax.bar(ids, rates, color=[_color_for(r) for r in rates])
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                    f"{rate:.0%}", ha="center", fontsize=9)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title(
        f"{run_id} [{eval_type}]  —  {data.get('hypothesis', '')[:70]}",
        fontsize=11,
    )
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4,
               label="0.80 reference")
    ax.legend(loc="lower right", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _color_for(rate: float) -> str:
    if rate >= 0.9:
        return "#3dcc6f"
    if rate >= 0.6:
        return "#f0b040"
    return "#e06060"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "run_*.json")))
    if not paths:
        print(f"No run_*.json under {RUNS_DIR}/")
        return
    rendered = 0
    for p in paths:
        run_label = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(OUT_DIR, f"{run_label}_scenarios.png")
        if plot_run(p, out):
            print(f"  rendered -> {out}")
            rendered += 1
        else:
            print(f"  skipped  {p}  (no 'scenarios' field)")
    print(f"\nDone. {rendered} plot(s) written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
