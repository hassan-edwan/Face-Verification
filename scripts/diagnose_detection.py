"""
One-shot diagnostic: measure what MTCNN actually sees on SCface by
distance. Not a run artifact — just a throwaway numbers dump so we
can ground a recovery plan in data instead of guesses.

Samples N images per (distance, camera), runs MTCNN (no confidence
floor, no size floor), and reports:
  - detection count: 0 detections vs 1+ detections
  - bbox size distribution (min side)
  - confidence distribution
  - detection size vs MIN_FACE_PX=60 cutoff
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from mtcnn import MTCNN

SCFACE = PROJECT_ROOT / "data" / "real_eval" / "scface"
SURV   = SCFACE / "surveillance"
MUGS   = SCFACE / "mugshot"

PATTERN = re.compile(r"^(\d{3})_cam(\d)_(\d)\.jpe?g$", re.IGNORECASE)
SAMPLES_PER_CELL = 5   # per (distance, camera) — 7 cams × 5 = 35 per distance


def sample_by_distance():
    buckets = defaultdict(list)   # dist -> list of paths
    per_cell = defaultdict(int)   # (dist, cam) -> count
    for p in sorted(SURV.iterdir()):
        m = PATTERN.match(p.name)
        if not m:
            continue
        subj, cam, dist = m.group(1), m.group(2), m.group(3)
        key = (dist, cam)
        if per_cell[key] >= SAMPLES_PER_CELL:
            continue
        per_cell[key] += 1
        buckets[dist].append(p)
    return buckets


def run():
    print("loading MTCNN...")
    det = MTCNN()
    print("done. sampling...")

    buckets = sample_by_distance()
    # Also sample mugshots for a high-quality reference point.
    mugs = sorted(MUGS.iterdir())[:10]

    for label, paths in [("mugshot", mugs)] + sorted(buckets.items()):
        sizes, confs = [], []
        zero = one_plus = 0
        under60 = under40 = 0
        img_w_list, img_h_list = [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img_h_list.append(img.shape[0])
            img_w_list.append(img.shape[1])
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                dets = det.detect_faces(rgb)
            except Exception:
                dets = []
            if not dets:
                zero += 1
                continue
            one_plus += 1
            # Largest detection per image
            best = max(dets, key=lambda d: d["box"][2] * d["box"][3])
            w, h = best["box"][2], best["box"][3]
            minside = min(w, h)
            sizes.append(minside)
            confs.append(best["confidence"])
            if minside < 60:
                under60 += 1
            if minside < 40:
                under40 += 1

        total = zero + one_plus
        print(f"\n--- {label} (n={total}) ---")
        print(f"  source img size: ~{median(img_w_list):.0f} x {median(img_h_list):.0f} px")
        print(f"  detection rate : {one_plus}/{total} ({one_plus/total*100:.0f}%)")
        if sizes:
            print(f"  bbox minside   : min={min(sizes)}, med={median(sizes):.0f}, max={max(sizes)}")
            print(f"  bbox <60 px    : {under60}/{one_plus} ({under60/one_plus*100:.0f}%)")
            print(f"  bbox <40 px    : {under40}/{one_plus} ({under40/one_plus*100:.0f}%)")
            print(f"  mtcnn conf     : min={min(confs):.3f}, med={median(confs):.3f}, max={max(confs):.3f}")
            under_conf90 = sum(1 for c in confs if c < 0.90)
            print(f"  conf <0.90     : {under_conf90}/{one_plus} ({under_conf90/one_plus*100:.0f}%)")


if __name__ == "__main__":
    run()
