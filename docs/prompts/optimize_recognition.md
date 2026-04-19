# Optimize Recognition — Iteration Prompt

> Paste this prompt verbatim to run the next recognition-model
> optimization cycle. Short invocation: *"Run the next recognition-model
> optimization cycle per `docs/prompts/optimize_recognition.md`."*

---

You are optimizing the live face-recognition pipeline in this repo.

## SCOPE

**In scope.** Constants and logic in `src/gatekeeper.py`, `src/quality.py`,
`src/tracker.py`, `src/similarity.py`. Pre-processing (alignment,
cropping, normalization) and post-processing (consensus, embedding-bank
policy, recent-match cache). You may rearchitect the evaluation
methodology if a better signal for LIVE behavior demands it — strict
offline LFW F1 is not sacred, but tracking discipline is.

**Out of scope.** Retraining FaceNet; swapping the embedder; adding new
ML dependencies; touching the enrichment subsystem (deleted); changing
the camera / detection / tracking thread model beyond knob tuning or
localized algorithmic fixes.

## FAILURE MODES — fix in rank order

1. **Re-acquisition.** Person leaves frame and re-enters within ~60 s;
   the IoU tracker assigns a fresh `track_id` and the new track fails
   to re-match the stored embeddings even though the face is the same.
2. **Crowded-scene identity swap.** Two tracked faces cross paths; IoU
   matching (`IOU_MATCH_THRESHOLD = 0.25`) flips `track_id → identity`;
   the gatekeeper's tracking-lock then cements the wrong identity and
   `forget(name)` is the only recovery.
3. **Fast-enroll noise.** `Gatekeeper._fast_enroll` captures a
   single-frame embedding that just scraped past the lenient
   `quick_quality_check` gate; later matches against that identity are
   fragile because the bank's first row is low-signal.
4. **Pose / lighting drift.** The refined embedding bank (up to
   `MAX_EMBEDDINGS_PER_IDENTITY = 8`) is dominated by similar frames.
   Out-of-distribution viewing conditions drop the cosine score below
   `MATCH_THRESHOLD` even though the identity hasn't changed.

## ITERATION PROTOCOL — follow exactly

Before changing any knob or logic:

1. Pick **one** failure mode + **one** hypothesis. Both go into the
   run's `hypothesis` field verbatim.
2. Choose the next `run_id`: `run_NNN` where `NNN = max(existing) + 1`.
   (First new run is `run_006`.)
3. Create `scripts/models/run_NNN.py` by copying the most recent run
   script and modifying only what the current hypothesis needs. The
   script MUST import the current `src/*` modules so the eval reflects
   the actual live-path config.

Each run ships exactly these artifacts:

| Artifact                              | Purpose                          |
| ------------------------------------- | -------------------------------- |
| `scripts/models/run_NNN.py`           | Reproducible experiment entry    |
| `outputs/runs/run_NNN.json`           | Schema below                     |
| `outputs/plots/run_NNN_<name>.png`    | Versioned — never overwrite      |
| one line in `outputs/runs/README.md`  | Rolling changelog                |
| one git commit `run_NNN: <hypothesis>`| Contains artifacts + `src/` diff |

## RUN JSON SCHEMA (required fields)

```json
{
  "run_id":     "run_006",
  "created_at": "<ISO-8601 UTC>",
  "git_sha":    "<short sha of the commit containing the src/ diff>",
  "hypothesis": "<one sentence: what you changed and why>",
  "config_diff": {
    "MATCH_THRESHOLD": [0.70, 0.65],
    "CONSENSUS_FRAMES": [5, 7]
  },
  "eval": {
    "offline": {
      "pairs_csv":      "configs/pairs_v4.csv",
      "best_threshold": 0.43,
      "val_metrics":    {"tpr": ..., "fpr": ..., "f1": ...},
      "test_metrics":   {"tpr": ..., "fpr": ..., "f1": ...}
    },
    "live": [
      {"scenario": "S1_single_lock",    "trials": 5, "successes": 5, "failures": 0, "notes": ""},
      {"scenario": "S4_reacquisition",  "trials": 5, "successes": 4, "failures": 1, "notes": "..."}
    ]
  },
  "decision": "keep | revert | inconclusive",
  "notes":    "freeform; reference related runs, anomalies, next steps"
}
```

Fields are **additive**. Never remove or silently rename existing
fields — past runs must stay readable. Older artifacts used
`best_threshold_from_val`; new runs nest it under `eval.offline`.

## EVALUATION

**Offline (regression floor).** Re-run the pair eval using
`configs/pairs_v4.csv` against current `src/`. Offline F1 is the
regression floor: a change that helps live but crashes LFW F1 is still
a revert.

**Live (structured operator protocol).** The run script does NOT drive
the camera. The operator executes each scenario 5 trials and hands back
success counts; the script writes them into `eval.live`.

| ID | Scenario                                               |
| -- | ------------------------------------------------------ |
| S1 | Single person, stable lock — baseline, must not regress |
| S2 | Two people, no crossing — stability                    |
| S3 | Two people crossing path                               |
| S4 | Person A exits for 30 s, returns — re-acquisition      |
| S5 | Person A rotates ±45° yaw after enrollment — pose drift |
| S6 | Person A walks into dim / bright area — illumination   |

**Rearchitecting eval.** If the existing schema can't capture what a
hypothesis needs (e.g. re-acquisition latency in seconds), extend
`eval.live[*]` with new named fields. Do not rename existing ones.

## DECISION RULE

Mark `decision: "keep"` **only if all three hold:**

- Offline F1 regression ≤ 2 absolute points.
- The targeted live scenario improves by ≥ 20 % success rate.
- No other live scenario regresses by ≥ 20 %.

Otherwise `"revert"` — `git revert` the `src/` change but **keep** the
run artifacts. Negative results are preserved for future reference.
`"inconclusive"` is allowed: explain why in `notes` and propose the
next hypothesis.

## CONSTRAINTS

- **One hypothesis per run.** Bundled changes make attribution guesswork.
- **Prior runs are frozen** — never edit `run_NNN.py` or
  `outputs/runs/run_NNN.json` from a completed run.
- **Never snapshot source files into `outputs/`.** Git is the archive;
  the `git_sha` field in the JSON closes the loop.
- **Do not mirror knob values into `CLAUDE.md`.** Git blame over
  `src/gatekeeper.py` etc. is authoritative. Touch `CLAUDE.md` only
  when the *convention itself* changes.
