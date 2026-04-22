# Face Verification — LFW

Face verification pipeline using FaceNet embeddings and cosine similarity on LFW.
Milestone 3 adds an embedding-based verifier packaged behind a CLI and Docker
image, with Platt-scaled confidence and a local load test.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Make packages importable
touch src/__init__.py scripts/__init__.py scripts/plumbing/__init__.py
# Windows: type nul > src\__init__.py  (repeat for others)
```

---

## Reproduce Everything

Run these commands in order from the project root:

```bash
# 1. Download LFW and write configs/manifest.json
python scripts/plumbing/ingest_lfw.py

# 2. Generate pair CSVs (each version improves on the previous)
python scripts/plumbing/make_pairs_v1.py    # baseline
python scripts/plumbing/make_pairs_v2.py    # + uniform identity weighting
python scripts/plumbing/make_pairs_v3.py    # + deduplication
python scripts/plumbing/make_pairs_v4.py    # + identity cap

# 3. Validate pairs and run JSON
python scripts/pipeline_validator.py

# 4. Evaluation runs (2–5 min each)
python -m scripts.models.run_001                # baseline
python -m scripts.models.run_002                # uniform weighting
python -m scripts.models.run_003                # deduplication
python -m scripts.models.run_004                # identity cap
python -m scripts.models.run_005                # + center-crop preprocessing

# 5. Error analysis (requires pairs_v4.csv + run_005.json)
python -m scripts.error_analysis

# 6. Plots
python scripts/plots/plot_confusion_matrix.py
python scripts/plots/plot_roc.py

# 7. Cross-version pair comparison report (optional)
python scripts/compare_pairs.py

# 8. Fit calibration (Platt sigmoid on val split of run_005)
python -m scripts.fit_calibration

# 9. Tests (no dataset or model required)
pytest tests/ -v
```

---

## Verify a pair (CLI)

Frozen model: **run_005** (center-crop preprocessing, threshold picked on val).
Confidence is Platt-scaled on the val split — fit it once with
`python -m scripts.fit_calibration`.

```bash
# Single pair
python -m src.cli --left data/lfw/Ana_Guevara/Ana_Guevara_0001.jpg \
                  --right data/lfw/Ana_Guevara/Ana_Guevara_0002.jpg

# Batch (CSV with columns: left_path,right_path[,label])
python -m src.cli --pairs configs/my_pairs.csv
```

Each line of output is a JSON record:

```json
{"decision": 1, "score": 0.724, "confidence": 0.912, "threshold": 0.45,
 "latency_ms": 73.4, "left": "…", "right": "…"}
```

## Load test

```bash
python scripts/loadtest.py --pairs configs/my_pairs.csv \
    --concurrency 4 --duration 30 --output reports/loadtest.json
```

Reports completed-pair count, failure count, throughput (pairs/sec), mean, and
latency p50/p90/**p95**/p99. Writes a JSON summary if `--output` is given.


### Artifacts

| Path | Purpose |
| --- | --- |
| `outputs/runs/run_005.json` | Frozen Milestone 2 evaluation — source of threshold |
| `configs/calibration.json` | Platt parameters + threshold used by the CLI |
| `configs/pairs_v4.csv` | Deterministic pair file (train/val/test splits) |
| `reports/loadtest.json` | Optional load-test summary (if `--output` used) |

## Docker

```bash
docker build -t face-verifier .
docker run --rm -v "$PWD/data:/data" face-verifier \
    --left /data/lfw/Ana_Guevara/Ana_Guevara_0001.jpg \
    --right /data/lfw/Ana_Guevara/Ana_Guevara_0002.jpg
```

`configs/calibration.json` must exist in the tree being built — the image
bakes it in so the container is self-contained at runtime.

---

## Project Structure

```
├── configs/                            # Pair CSVs and data manifest
│   ├── manifest.json
│   ├── pairs.csv
│   ├── pairs_v2.csv
│   ├── pairs_v3.csv
│   └── pairs_v4.csv
│
├── data/
│   └── lfw/                            # Downloaded by ingest_lfw.py
│
├── outputs/
│   ├── runs/                           # run_001.json … run_005.json
│   └── error_analysis/                 # Montages and error CSVs
│
├── scripts/
│   ├── plumbing/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── ingest_lfw.py
│   │   ├── generate_pairs_v1.py
│   │   ├── generate_pairs_v2.py
│   │   ├── generate_pairs_v3.py
│   │   └── generate_pairs_v4.py
│   ├── models/                         # Evaluation runs
│   │   ├── run_001.py
│   │   ├── run_002.py
│   │   ├── run_003.py
│   │   ├── run_004.py
│   │   └── run_005.py
│   ├── plots/                          # Visualisations
│   │   ├── plot_confusion_matrix.py
│   │   └── plot_roc.py
│   ├── error_analysis.py
│   ├── pipeline_validator.py
│   ├── pairs_comparison.py
│   ├── bench_similarity.py
│   ├── fit_calibration.py              # Platt scaling on run_005 val split
│   └── loadtest.py                     # concurrency + latency percentiles
│
├── src/
│   ├── __init__.py
│   ├── similarity.py                   # cosine_similarity()
│   ├── verifier.py                     # FaceVerifier (embed → score → decide + confidence)
│   └── cli.py                          # `python -m src.cli`
│
├── tests/
│   ├── test_unit.py
│   ├── test_integration.py
│   └── test_verifier.py
│
├── Dockerfile
├── .dockerignore
└── requirements.txt
```