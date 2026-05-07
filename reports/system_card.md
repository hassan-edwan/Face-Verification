# System Card - Face Verification (v1.0-final)

| | |
|---|---|
| **System version** | v1.0-final |
| **Source run** | `outputs/runs/run_005.json` (`run_005_center_crop_preprocessing`) |
| **Operating threshold** | **0.43** (cosine similarity, score ≥ 0.43 ⇒ SAME) |
| **Confidence calibration** | Platt sigmoid, A = -14.184, B = 6.374, fit on val (`configs/calibration.json`) |
| **Inference path** | `src/verifier.py` → `src/cli.py` (`python -m src.cli`) |
| **Container** | `Dockerfile` at repo root; entrypoint = `python -m src.cli` |
| **Pinned environment** | `requirements.txt` |
| **Reproducibility** | `reports/reproducibility_checklist.md` |

---

## 1. System overview

The system decides whether two face images depict the **same identity**. The
production inference path is `src.verifier.FaceVerifier`, the same path the
CLI (`python -m src.cli`) and load test (`scripts/loadtest.py`) drive. The
pipeline matches `scripts/models/run_005.py` exactly:

```
image path
   └── cv2.imread (BGR)            ─┐
       cv2.cvtColor BGR → RGB      │  preprocessing
       center-crop margin = 0.15   │  (≈0.8 ms / image, CPU)
       cv2.resize → 160 × 160      ─┘
   └── FaceNet (keras-facenet)  ─────  embedding (≈160 ms / image, CPU bs=1)
       512-D vector
   └── cosine_similarity(eA, eB) ───  scoring (≈24 µs / pair)
   └── score ≥ 0.43 ?  decision = 1 (SAME)  :  decision = 0 (DIFFERENT)
   └── confidence = Platt(score)         calibrated P(same | score) ∈ [0, 1]
```

Inputs are arbitrary face crops on disk (any cv2-readable format). The
embedding model, threshold, and Platt parameters are frozen at release time;
nothing is trained at inference.

## 2. Intended use

In scope:

- Pairwise verification of a small number of cooperative, well-lit, roughly
  frontal face crops, used inside a larger trust workflow (e.g. session-level
  enrollment in the bundled webcam demo, manual review of duplicate-account
  candidates, or coursework demos).
- Threshold-sweep / ROC analyses against held-out pairs derived from the
  Labeled Faces in the Wild (LFW) split policy in `configs/manifest.json`.
- Latency / throughput baselining via `scripts/profile_pipeline.py` and load
  testing via `scripts/loadtest.py`.

Explicitly out of scope:

- Identification (1-to-N search) over a population of unknown size. The
  threshold was tuned for 1-to-1 verification on a balanced LFW pair set, not
  for hit-rate at low FPR over millions of identities.
- High-stakes authorisation decisions (border control, payments, access to
  protected resources). The system has no liveness check and no spoof
  detection.
- Surveillance, demographic inference, or any decision affecting an individual
  without an informed human in the loop.
- Verification of subjects that the FaceNet pretraining did not represent
  well (children, occluded faces, heavily off-angle faces, low-resolution or
  motion-blurred frames).

## 3. Data summary

- **Source**: Labeled Faces in the Wild (LFW), version `lfw:0.1.1`, ingested
  via `scripts/plumbing/ingest_lfw.py`.
- **Splits** (identity-disjoint, seed = 42, see `configs/manifest.json`):
  - train: 11,051 images / 4,599 identities
  - val:    1,192 images /   575 identities
  - test:     990 images /   575 identities
- **Pair sets** evolve `pairs_v1.csv` → `pairs_v4.csv`. The final system
  uses `configs/pairs_v4.csv` (2,643 pairs) which adds uniform identity
  weighting, deduplication, and a per-identity cap to the original LFW pairs.
- **Major data limitations**: LFW is a celebrity dataset of mostly Western
  public figures, biased toward adults, frontal poses, and good lighting. It
  is not representative of the broader population. Labels are identity-only;
  no demographic, consent, or capture-condition metadata is shipped, which
  means we **cannot run a credible subgroup fairness analysis** with what is
  in this repo (see §6).

## 4. Operating threshold and key metrics

Threshold was selected on **val** by argmax F1 over a sweep `[0, 1]` step
0.01, then frozen and applied to **test**. Threshold = **0.43**, recorded in
both `outputs/runs/run_005.json` and `configs/calibration.json`.

**Test set @ threshold 0.43** (n = 829 pairs; 329 same / 500 different):

| Metric | Value |
| --- | ---: |
| TP | 325 |
| FP | 3 |
| FN | 4 |
| TN | 497 |
| TPR (recall) | 0.988 |
| FPR | 0.006 |
| F1 | 0.989 |
| Accuracy | 0.992 |

**Val set @ threshold 0.43** (n = 814): TPR 1.000, FPR 0.010, F1 0.992.

These numbers describe LFW-like pairs only. They are **not** an estimate of
performance on faces collected under different conditions, on different
populations, or against adversarial inputs. On the test split the system
trades one extra missed match (FN=4 vs FN=0 on val) for a comparable
false-accept rate.

The full threshold sweep is in `outputs/runs/run_005.json`; the operating
point and its sweep neighbourhood (0.40–0.45) all sit within ±0.005 F1 of the
chosen 0.43, so the operating point is not knife-edge sensitive.

**Confidence calibration.** The CLI reports a Platt-scaled probability
alongside the binary decision: `confidence = 1 / (1 + exp(A · score + B))`
with A = -14.184, B = 6.374 fit on the val split (see
`scripts/fit_calibration.py`). Confidence is a calibrated probability that
the pair is the same identity given the cosine score; it is independent of
the 0.43 hard threshold and lets downstream consumers route uncertain pairs
to a human reviewer.

## 5. Failure modes and limitations

These are the failure modes we have observed or have strong reason to expect.
Each was visible during the iterative pair-design work in Milestones 1–2 and
is consistent with how `src/verifier.FaceVerifier` is used today:

1. **Off-angle / non-frontal faces.** Yaw or pitch beyond ~30° drops cosine
   scores below the 0.43 threshold even for true matches. This is the main
   reason the live webcam pipeline (`scripts/live_webcam.py`) gates frames
   through `src/quality.py` and refuses to enroll on poor pose.
2. **Low-quality input.** Severe blur, harsh under/over-exposure, or face
   crops smaller than ~60 px push embeddings toward the centre of the
   embedding space, raising both false matches *and* false rejections. The
   live system rejects these via FIQA; the offline CLI does **not** - it will
   embed whatever you give it.
3. **Tight crops vs full LFW frames.** The system was tuned on LFW frames
   with the `center_crop margin=0.15` preprocessing baked into
   `src/verifier.load_and_preprocess`. Feeding it a tight pre-cropped face
   (where the relevant pixels are already at the edges) effectively crops
   them out and degrades scores. Inputs should be face-region images with
   ~15 % surrounding padding, similar to raw LFW frames.
4. **Visually similar different-identity pairs.** Twins, siblings, parents
   and children, or simply unrelated people with similar bone structure and
   hairstyle remain the dominant source of FPs (3/500 on test). Any
   downstream workflow needs a human reviewer for stakes higher than a demo.
5. **No spoof / liveness defence.** A printed photo, screen replay, or 3D
   mask will pass the verifier whenever the underlying image is sharp enough
   and produces a high-similarity embedding. Do not deploy this system
   anywhere a high-similarity score alone makes a decision.
6. **Out-of-distribution lighting / colour.** Strong tinted lighting, IR
   captures, or heavy filters were not represented in LFW and degrade
   accuracy; this has not been quantified.
7. **Reproducibility caveat.** The bundled `requirements.txt` pins
   `tensorflow==2.20.0` and `keras-facenet==0.3.2`. Running with newer TF /
   keras-facenet versions can shift cosine scores by small amounts and
   therefore the operating threshold drifts; downstream consumers must use
   the pinned environment or re-calibrate via `scripts/fit_calibration.py`.

## 6. Fairness-related risks

We do **not** publish per-subgroup metrics because LFW does not ship the
metadata required to define subgroups credibly, and inferring demographic
attributes from images for the purpose of a fairness audit is itself a
practice we want to avoid here. Instead, we list the risk categories that
matter for this system and what an operator of the system should do about
them:

- **Population coverage.** LFW is heavily skewed toward Western public
  figures, adult men, and clear lighting. Empirically, embedding models
  trained on similar distributions are known in the broader literature to
  perform worse on darker-skinned subjects, women, children, and people
  wearing occlusions (e.g. glasses, head coverings). We have no reason to
  believe this system is exempt. Operators should not use LFW-derived
  numbers as a stand-in for performance on their target population.
- **Quality-correlated unfairness.** The system fails harder on low-quality
  images (§5). If the population whose images are systematically lower
  quality (older cameras, harsher lighting, smaller crops) is also a
  protected group, that quality gap converts into an accuracy gap. This is
  the most actionable fairness concern: improving image-quality gating
  (`src/quality.py`) helps it directly.
- **Misuse / dual-use.** Face verification systems can be repurposed for
  surveillance, tracking, or identification at a distance without consent.
  The threshold sweep, calibrated metrics, and Docker packaging in this repo
  are all dual-use. Anyone deploying this system should obtain informed
  consent from the people whose faces are being compared, and should not
  deploy it where individuals cannot opt out.
- **Threshold transferability.** A threshold calibrated on LFW will tend to
  be **too permissive** on harder distributions (lower-quality cameras,
  different demographics) because the FaceNet embeddings are noisier there.
  Operators who deploy this in a new domain must re-run the threshold sweep
  on representative pairs from that domain, refit the Platt sigmoid, and
  update `configs/calibration.json` before trusting the 0.43 cutoff.

If a downstream user of this repo needs a defensible subgroup fairness
analysis, the right path is to (a) collect a labelled subgroup test set with
informed consent, (b) re-run a threshold sweep stratified by subgroup using
the existing `scripts/models/run_005.py` machinery, and (c) report TPR/FPR
per subgroup plus the equal-error rate. The current artifacts do not stand
in for that.

## 7. Operational constraints

- **Hardware**: CPU is supported and is the documented baseline (see
  `reports/profiling_report.md`). End-to-end latency at batch size 1 on the
  reference CPU (Intel i7-1355U, 12 logical cores) is **~188 ms / pair
  median** (p95 ~234 ms). The embedding stage dominates (~99 % of wall time).
- **Throughput**: Single-pair throughput is ~5–6 pairs/s on CPU. Batch
  embedding at bs=32 reaches ~41 images/s (~20 pairs/s). Beyond bs=32 the
  per-image gain plateaus; memory pressure rises faster than throughput.
  Concurrent load characteristics are reproducible via
  `scripts/loadtest.py --pairs <csv> --concurrency N --duration T`.
- **GPU**: Not required, not measured here. If a GPU is present, TF will use
  it automatically; users adopting GPU should re-run the profiling script and
  report their own numbers (`environment.tf_visible_gpus` records the
  device).
- **Inputs**: any cv2-readable still image. The system has **no detector**
  in the offline path; you must supply face crops. The webcam path
  (`scripts/live_webcam.py`) bundles MTCNN detection + IoU tracking + FIQA
  for live inputs but is not what the System Card metrics describe.
- **Determinism**: cosine similarity is deterministic. The FaceNet forward
  pass is deterministic up to floating-point rounding (`oneDNN` may shuffle
  reduction order); seed pinning is at the dataset / pair-generation level
  (seed=42), not at the inference level.
- **Latency budget**: The system is suitable for interactive verification
  (sub-second per pair) on CPU. It is **not** suitable for hard real-time
  use (e.g. per-frame video at 30 FPS on CPU); the live demo handles this by
  off-loading embedding to a worker thread and throttling submissions per
  track.

## 8. Reproducibility pointer

- Environment, build and CLI commands: `README.md`, "How to run".
- Step-by-step reproduction with frozen versions:
  `reports/reproducibility_checklist.md`.
- Final tag: **`v1.0-final`**.
- Profiling artifact: `reports/profiling/latency_summary.{json,md}`.
- Frozen threshold + Platt calibration: `configs/calibration.json`.
- Source run JSON: `outputs/runs/run_005.json`.
