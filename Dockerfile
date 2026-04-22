# Face verifier — embedding-based CLI.
#
# Build:
#   docker build -t face-verifier .
#
# Single-pair verify (mount local images into /data):
#   docker run --rm -v "$PWD/examples:/data" face-verifier \
#       --left /data/a.jpg --right /data/b.jpg
#
# Batch:
#   docker run --rm -v "$PWD/examples:/data" face-verifier \
#       --pairs /data/pairs.csv

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime deps (libGL for cv2, glib for image I/O).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake sources + frozen configs (including calibration.json + pair CSVs).
# Data, outputs, and reports are deliberately excluded via .dockerignore.
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Pre-download FaceNet weights so `docker run` is offline-ready.
RUN python -c "from keras_facenet import FaceNet; FaceNet()"

ENTRYPOINT ["python", "-m", "src.cli"]
