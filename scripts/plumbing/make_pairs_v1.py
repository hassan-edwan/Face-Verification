"""
Pair Generator v1 — Baseline
=============================
Generates 500 same-identity and 500 different-identity pairs per split
using simple random sampling. No deduplication or identity weighting.

Output: configs/pairs.csv
"""

import os
import random
import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
LFW_PATH    = "data/lfw"
OUTPUT_PATH = "configs/pairs.csv"
SEED        = 42
PAIRS_EACH  = 500   # same-identity pairs and different-identity pairs per split
# ─────────────────────────────────────────────────────────────────────────────


def build_id_map(lfw_path: str, identity_list: list) -> dict:
    """
    Returns {identity: [image_indices]} for all identities that have
    at least one image. Index is parsed from the filename suffix
    (e.g. 'Name_0003.jpg' → 3).
    """
    id_map = {}
    for name in identity_list:
        folder = os.path.join(lfw_path, name)
        files  = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
        indices = []
        for f in files:
            try:
                indices.append(int(f.split("_")[-1].split(".")[0]))
            except (ValueError, IndexError):
                continue
        if indices:
            id_map[name] = indices
    return id_map


def sample_same_pairs(id_map: dict, n: int, rng: random.Random) -> list:
    """
    Samples n same-identity pairs. Identity is chosen uniformly;
    two distinct image indices are then sampled from that identity.
    """
    eligible = [name for name, idxs in id_map.items() if len(idxs) >= 2]
    pairs = []
    for _ in range(n):
        person    = rng.choice(eligible)
        idx1, idx2 = rng.sample(id_map[person], 2)
        pairs.append([person, person, idx1, idx2, 1])
    return pairs


def sample_diff_pairs(id_map: dict, n: int, rng: random.Random) -> list:
    """
    Samples n different-identity pairs. Two distinct identities are chosen
    uniformly; one image index is sampled from each.
    """
    all_ids = list(id_map.keys())
    pairs = []
    for _ in range(n):
        p1, p2 = rng.sample(all_ids, 2)
        idx1   = rng.choice(id_map[p1])
        idx2   = rng.choice(id_map[p2])
        pairs.append([p1, p2, idx1, idx2, 0])
    return pairs


def generate_pairs(lfw_path: str = LFW_PATH,
                   output_path: str = OUTPUT_PATH,
                   seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    identities = sorted([
        d for d in os.listdir(lfw_path)
        if os.path.isdir(os.path.join(lfw_path, d))
    ])
    rng.shuffle(identities)

    n = len(identities)
    splits = {
        "train": identities[:int(n * 0.8)],
        "val":   identities[int(n * 0.8):int(n * 0.9)],
        "test":  identities[int(n * 0.9):]
    }

    all_pairs = []
    for split_name, ids in splits.items():
        print(f"Processing {split_name} split...")
        id_map = build_id_map(lfw_path, ids)
        for row in sample_same_pairs(id_map, PAIRS_EACH, rng):
            all_pairs.append(row + [split_name])
        for row in sample_diff_pairs(id_map, PAIRS_EACH, rng):
            all_pairs.append(row + [split_name])

    df = pd.DataFrame(
        all_pairs,
        columns=["left_identity", "right_identity",
                 "left_index", "right_index", "label", "split"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} pairs → {output_path}")
    return df


if __name__ == "__main__":
    generate_pairs()