# Face Verification — LFW

Face verification pipeline using FaceNet embeddings and cosine similarity on LFW.

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

# 8. Tests (no dataset or model required)
pytest tests/ -v
```

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
│   └── bench_similarity.py
│
├── src/
│   ├── __init__.py
│   └── similarity.py                   # cosine_similarity()
│
├── tests/
│   ├── test_unit.py
│   └── test_integration.py
│
└── requirements.txt
```