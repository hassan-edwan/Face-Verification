# Real-Data Evaluation Harness

`src/real_data_eval.py` runs the full live pipeline (MTCNN → align →
`get_embedder()` → Gatekeeper) on images from `data/real_eval/`.
Embedder defaults to **ArcFace** (buffalo_l / w600k_r50 ONNX, 512-d,
112×112 input); `FACE_EMBEDDER=facenet` reverts to the legacy
keras_facenet path for comparison. Produces a run artifact tagged
`eval_type: "real"` that plays in the same numbering and changelog
as synthetic runs.

## Dataset layout

All under `data/real_eval/` (gitignored via the existing `data/` rule):

```
data/real_eval/
├─ scface/
│  ├─ mugshot/                  130 enrollment stills (NNN_frontal.JPG)
│  ├─ mugshot_rotation/         multi-pose mugshots (NNN_{frontal,L1-L4,R1-R4}.jpg)
│  ├─ mugshot_original/         original-resolution mugshots (unused by the harness)
│  ├─ surveillance/             2,860 surveillance stills (NNN_camX_Y.jpg)
│  ├─ surveillance_by_distance/ pre-split distance_1/2/3 + ir_cam8
│  └─ splits/                   26 .txt files (metadata + per-split listings)
└─ chokepoint/
   ├─ P1E_S1 .. P1L_S4/         8 sessions × 3 cameras, frames as 8-digit JPGs
   └─ groundtruth/*.xml         per-frame <person id>/<leftEye>/<rightEye>
```

## Scenarios

**SCface** (`scface_mugshot_to_{d1_far,d2_mid,d3_close}`) — for each of
130 subjects, enroll on the cropped frontal mugshot and query one
surveillance still at the named distance (cam1 deterministic). Distance
1 = 4.2 m (hardest), 2 = 2.6 m, 3 = 1.0 m (closest).

**ChokePoint** — three scenarios, built from the per-frame GT XMLs
(`<person id>`, `<leftEye>`, `<rightEye>`):

- `chokepoint_within_clip_sanity` — per labeled person, enroll on
  their first visible frame of one session-camera, query the last.
  Near-duplicate matching on ~1–2 s of one walk. **Sanity floor,
  not a generalization test.** Former name `chokepoint_temporal_stability`
  was removed to stop the measurement being cited as evidence of
  pose / distance invariance.
- `chokepoint_cross_camera` — same person, same session, different
  portal camera. Tests pose change across simultaneous viewpoints.
- `chokepoint_cross_session` — same person ID (ChokePoint's subject
  IDs are stable across sessions), enroll in the earliest session,
  query a later one. Different walk, different time — the true
  re-identification test.

**TinyFace** — removed. Previously shipped as a no-op loader stub
(`load_tinyface_scenarios` → `[]`). Open-set identification
(Gallery + Probe + Distractor) doesn't fit the pair harness and
was never actually evaluated despite run notes mentioning it. If
TinyFace coverage is ever needed, it must live in a dedicated
open-set harness.

## Running

```bash
PYTHONIOENCODING=utf-8 python -m scripts.models.run_009
```

Runtime: ~10–15 minutes on CPU for the full SCface + ChokePoint set
(414 pairs, one MTCNN + FaceNet pass each). Use `max_subjects=N` or
`max_sessions=N` on the loader calls for quicker smoke tests.

## Metrics

Two numbers per scenario:

- **`success_rate`** — genuine-pair gatekeeper success. Same as
  before: the fraction of (enroll, query) pairs for the same identity
  that end with `query_decision == MATCHED` and `matched_identity ==
  enrolled_identity`. Preserved for historical comparability with
  runs 009–015.
- **`tar_at_far` (in `aggregate.scenario_diagnostics`)** — the
  fraction of genuine pairs whose cosine(enroll_emb, query_emb)
  exceeds the threshold that yields a 1 % / 0.1 % false-accept rate
  on same-scenario cross-subject impostor pairs. Impostors are
  drawn exhaustively (enroll of A × query of B where A ≠ B, both
  embeddings succeeded) up to a cap of 20 000. `tar_at_far = null`
  when the impostor pool is too thin to resolve the target FAR
  (e.g. FAR=0.001 with <1 000 impostors).

A lax REMATCH_THRESHOLD lifts `success_rate` for free while
`tar_at_far` stays honest (the FAR is measured, not assumed).
When the two metrics disagree, trust TAR@FAR.

## Quality gate

The harness now runs `src.quality.quick_quality_check` on every
aligned crop and threads the returned score into `gk.process(...,
quality_score=...)`. Frames below the gate (blurry, over/under-exposed)
emit `quality_score=0.0` and therefore never update the gatekeeper's
embedding bank — matching the live refinement path. Prior runs
hard-coded `0.9` on enroll and `0.7` on query, which bypassed this
gate entirely.

## Per-run artifact

`outputs/runs/run_NNN.json` with standard schema plus:
- `eval_type: "real"`
- `datasets: ["scface", "chokepoint"]` (or subset)
- `scenarios[*]` carries the existing fields; `notes` also summarizes
  TAR@FAR=1 % inline.
- `aggregate.scenario_diagnostics: {scenario_id: {n_genuines, n_impostors,
  tar_at_far: {"0.01": {threshold, tar} | null, "0.001": ... | null}}}`

## Relationship to the synthetic harness

Synthetic harness (`src/live_eval.py`, scenarios S1–S8):
- Fast (seconds per run), deterministic, no camera / FaceNet.
- Measures gatekeeper state-machine logic with fabricated embeddings.
- **Direction-of-travel** signal: did knob change N move S7 up or down?

Real-data harness (`src/real_data_eval.py`):
- Slow (minutes per run), real images, full pipeline.
- Measures end-to-end accuracy on actual surveillance / off-angle data.
- **Absolute calibration**: what fraction of real 4.2 m SCface subjects
  does the pipeline re-match?

Per-change protocol: iterate fast with synthetic; gate keeps / reverts
with real-data runs periodically. A synthetic win that doesn't show up
in real-data after a few runs is a false positive.
