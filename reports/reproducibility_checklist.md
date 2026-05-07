# Reproducibility Checklist - v1.0-final

This is the **minimum, exact** path to recreate the v1.0-final release from
a fresh clone, run the verification CLI, and reproduce the profiling artifact.
A grader should not need to read anything else in the repository to run it.

| Artifact | Path |
| --- | --- |
| System Card | `reports/system_card.md` |
| Profiling report | `reports/profiling_report.md` |
| Auto-generated profiling tables | `reports/profiling/latency_summary.{md,json}` |
| Frozen threshold + Platt calibration | `configs/calibration.json` |
| Source run JSON (operating threshold + sweep) | `outputs/runs/run_005.json` |
| Final tag | **`v1.0-final`** |

---

## 0. Prerequisites

- Git
- One of:
  - **Python 3.11** + `venv` (native install), or
  - **Docker** (for the containerised CLI; no Python needed locally)
- ~2 GB free disk for LFW + cached TF model weights

## 1. Clone the tagged release

```bash
git clone <repo-url> face-verification
cd face-verification
git checkout v1.0-final
```

## 2A. Native Python setup (Linux / macOS / Windows)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## 2B. Docker setup (alternative to 2A)

```bash
docker build -t face-verifier .
```

The image is CPU-only, includes the frozen pipeline + CLI entrypoint, and
pre-downloads the FaceNet weights at build time so `docker run` works
offline. `configs/calibration.json` is baked in.

## 3. Get the dataset

The CLI itself only needs two image paths, but the offline evaluation and
the profiling script need LFW on disk. From the project root:

```bash
python scripts/plumbing/ingest_lfw.py
```

This populates `data/lfw/` (gitignored) using `tensorflow-datasets`.

## 4. Run the verification CLI on a sample pair

**Native:**

```bash
# same identity (expected: decision=1, score >> 0.43)
python -m src.cli \
    --left  data/lfw/Daniel_Radcliffe/Daniel_Radcliffe_0001.jpg \
    --right data/lfw/Daniel_Radcliffe/Daniel_Radcliffe_0002.jpg

# different identity (expected: decision=0, score < 0.43)
python -m src.cli \
    --left  data/lfw/Daniel_Radcliffe/Daniel_Radcliffe_0001.jpg \
    --right data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
```

**Docker:** mount the dataset directory into the container as `/data`:

```bash
docker run --rm -v "$(pwd)/data:/data" face-verifier \
    --left  /data/lfw/Daniel_Radcliffe/Daniel_Radcliffe_0001.jpg \
    --right /data/lfw/Daniel_Radcliffe/Daniel_Radcliffe_0002.jpg
```

Expected output (one JSON line per call):

```json
{"decision": 1, "score": 0.788, "confidence": 0.999, "threshold": 0.43,
 "latency_ms": 190.2, "left": "...", "right": "..."}
```

The threshold and Platt parameters are loaded from `configs/calibration.json`;
override the threshold with `--threshold 0.50` if needed.

For a CSV-driven batch:

```bash
python -m src.cli --pairs configs/your_pairs.csv
# CSV columns required: left_path,right_path[,label]
```

## 5. Reproduce the profiling baseline

```bash
python scripts/profile_pipeline.py \
    --reps 30 --warmup 2 --n-images 64 \
    --batch-sizes 1 2 4 8 16 32
```

Writes:

- `reports/profiling/latency_summary.json` (machine-readable)
- `reports/profiling/latency_summary.md` (human-readable table)

Expected runtime on the reference CPU (i7-1355U): ~2–4 minutes including
TF/keras-facenet startup and warmup.

## 6. Reproduce the core evaluation result (optional)

This is the run that produced the operating threshold and the test metrics
quoted in the System Card:

```bash
python scripts/plumbing/make_pairs_v1.py
python scripts/plumbing/make_pairs_v2.py
python scripts/plumbing/make_pairs_v3.py
python scripts/plumbing/make_pairs_v4.py
python -m scripts.models.run_005       # writes outputs/runs/run_005.json
python -m scripts.fit_calibration      # writes configs/calibration.json
```

Produces `outputs/runs/run_005.json` with the threshold sweep and test
metrics, then refits the Platt sigmoid on the val split. The headline
numbers should match those in `reports/system_card.md` §4 (TPR 0.988,
FPR 0.006, F1 0.989, threshold 0.43).

## 7. Run the test suite (optional but recommended)

```bash
pytest tests/ -v
```

Exercises pair-generation invariants, the cosine implementation, the
verifier, and a CLI smoke test.

## 8. (Optional) Concurrency / load characteristics

```bash
python scripts/loadtest.py --pairs configs/your_pairs.csv \
    --concurrency 4 --duration 30 --output reports/loadtest.json
```

Reports completed-pair count, throughput, and latency p50/p90/p95/p99.

## 9. Final-tag check

```bash
git tag --list | grep v1.0-final
git show v1.0-final --stat | head
```

The tag points to the commit containing every artifact listed in the table
at the top of this checklist.

---

## What "match" means here

For step 4, the score is deterministic up to the small floating-point
variation that `oneDNN` introduces on CPU; the **decision** and the score to
two decimal places should match `decision=1 / score≈0.79` and
`decision=0 / score≈-0.03` on the two example pairs. For step 5, the
absolute latency depends on hardware, but the **shape** of the numbers -
embed dominating wall time, throughput roughly 6–7× higher at bs=32 than at
bs=1, score below 1 ms - must hold.
