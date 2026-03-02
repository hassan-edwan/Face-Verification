# Face Verification Project: Milestone 1

A deterministic pipeline for Face Verification using the LFW (Labeled Faces in the Wild) dataset. 

## Quick Start

Follow these steps to reproduce the dataset ingestion, pair generation, and similarity benchmarks.

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
.\venv\Scripts\activate   # Windows

# Install pinned dependencies
pip install -r requirements.txt

#ingest lfw
python scripts/ingest_lfw.py

#generate pairs
python scripts/generate_pairs.py

#run similarity benchmark
python -m scripts.bench_similarity
