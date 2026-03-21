"""
LFW Ingestion — Data Manifest Generator
=========================================
Downloads LFW via TensorFlow Datasets, writes images to disk in the
folder structure expected by the rest of the pipeline, splits identities
deterministically into train / val / test, and writes a manifest JSON.

Folder structure written:
    data/lfw/
        <Identity_Name>/
            <Identity_Name>_0001.jpg
            <Identity_Name>_0002.jpg
            ...

Split policy: by identity — no person appears in more than one split.
This prevents identity leakage between training and evaluation.

Outputs:
    data/lfw/               — images on disk
    configs/manifest.json   — split policy, counts, provenance

Usage:
    python scripts/plumbing/ingest_lfw.py
"""

import os
import json
from collections import defaultdict

import cv2
import numpy as np
import tensorflow_datasets as tfds

# ── CONFIG ───────────────────────────────────────────────────────────────────
LFW_OUT_DIR  = "data/lfw"
OUTPUT_PATH  = "configs/manifest.json"
TFDS_VERSION = "lfw:0.1.1"
SEED         = 42
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def load_raw_data(tfds_version: str):
    """
    Downloads (if necessary) and loads all LFW examples into memory.
    LFW is ~13k images and fits comfortably in RAM for indexing.
    Returns (data_list, ds_builder).
    """
    print(f"Loading {tfds_version} via TFDS...")
    ds_builder = tfds.builder(tfds_version)
    ds_builder.download_and_prepare()
    raw = tfds.as_numpy(ds_builder.as_dataset(split="all"))
    return list(raw), ds_builder


def sort_deterministically(data_list: list) -> list:
    """
    Sorts by (label, arrival order) for a stable, platform-independent order.
    Grouping by label first avoids touching pixel data for comparison.
    """
    grouped = defaultdict(list)
    for item in data_list:
        grouped[item["label"]].append(item)

    result = []
    for label in sorted(grouped.keys()):
        result.extend(grouped[label])
    return result


def split_by_identity(unique_identities: list, seed: int) -> tuple:
    """
    Shuffles identities with a seeded RNG and splits into
    train / val / test by the configured ratios.
    Returns (train_ids, val_ids, test_ids) as sets.
    """
    ids = list(unique_identities)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n         = len(ids)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    return set(ids[:train_end]), set(ids[train_end:val_end]), set(ids[val_end:])


def decode_label(raw_label) -> str:
    """Decodes a TFDS label to a plain string regardless of version."""
    if isinstance(raw_label, bytes):
        return raw_label.decode("utf-8")
    return str(raw_label)


def write_images_to_disk(data_list: list, lfw_out_dir: str) -> dict:
    """
    Writes every image in data_list to:
        <lfw_out_dir>/<Identity_Name>/<Identity_Name>_NNNN.jpg

    The index (NNNN) is 1-based and increments per identity, matching
    the LFW filename convention expected by generate_pairs and run scripts.

    Skips images that already exist so re-running is safe.
    Returns id_map: {identity_name: [written_indices]}
    """
    print(f"Writing images to {lfw_out_dir}/ ...")
    identity_counters: dict = defaultdict(int)
    id_map:            dict = defaultdict(list)
    written = skipped = 0

    for item in data_list:
        name = decode_label(item["label"])
        identity_counters[name] += 1
        idx      = identity_counters[name]
        filename = f"{name}_{str(idx).zfill(4)}.jpg"
        folder   = os.path.join(lfw_out_dir, name)
        path     = os.path.join(folder, filename)

        if os.path.exists(path):
            skipped += 1
        else:
            os.makedirs(folder, exist_ok=True)
            # TFDS provides RGB; cv2.imwrite expects BGR
            img_bgr = cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img_bgr)
            written += 1

        id_map[name].append(idx)

    print(f"  Written: {written}  |  Already existed (skipped): {skipped}")
    return dict(id_map)


def count_by_split(data_list: list,
                   train_ids: set,
                   val_ids: set,
                   test_ids: set) -> dict:
    """Counts images per split for the manifest."""
    counts = {"train": 0, "val": 0, "test": 0}
    for item in data_list:
        label = decode_label(item["label"])
        if label in train_ids:
            counts["train"] += 1
        elif label in val_ids:
            counts["val"] += 1
        else:
            counts["test"] += 1
    return counts


def ingest_lfw(tfds_version: str = TFDS_VERSION,
               lfw_out_dir: str = LFW_OUT_DIR,
               output_path: str = OUTPUT_PATH,
               seed: int = SEED) -> dict:
    """
    Full ingestion pipeline:
      1. Load all LFW images via TFDS
      2. Sort deterministically by (label, arrival order)
      3. Split identities into train / val / test
      4. Write images to data/lfw/<Identity>/<Identity>_NNNN.jpg
      5. Write manifest JSON

    Returns the manifest dict.
    """
    # 1. Load
    data_list, ds_builder = load_raw_data(tfds_version)
    print(f"Loaded {len(data_list)} images.")

    # 2. Sort
    data_list = sort_deterministically(data_list)

    # 3. Split identities
    unique_identities = sorted(set(decode_label(item["label"]) for item in data_list))
    print(f"Found {len(unique_identities)} unique identities.")
    train_ids, val_ids, test_ids = split_by_identity(unique_identities, seed)

    # 4. Write images to disk
    write_images_to_disk(data_list, lfw_out_dir)

    # 5. Count per split for manifest
    split_counts = count_by_split(data_list, train_ids, val_ids, test_ids)

    # 6. Build and save manifest
    manifest = {
        "seed":         seed,
        "split_policy": "By identity — no person appears in more than one split.",
        "data_source":  tfds_version,
        "lfw_dir":      lfw_out_dir,
        "cache_dir":    str(ds_builder.data_dir),
        "ratios": {
            "train": TRAIN_RATIO,
            "val":   VAL_RATIO,
            "test":  round(1 - TRAIN_RATIO - VAL_RATIO, 4),
        },
        "counts": {
            "train": {"images": split_counts["train"], "identities": len(train_ids)},
            "val":   {"images": split_counts["val"],   "identities": len(val_ids)},
            "test":  {"images": split_counts["test"],  "identities": len(test_ids)},
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Manifest saved → {output_path}")
    for split_name, info in manifest["counts"].items():
        print(f"  {split_name:5s}: {info['images']:5d} images  "
              f"across {info['identities']:4d} identities")

    return manifest


if __name__ == "__main__":
    ingest_lfw()