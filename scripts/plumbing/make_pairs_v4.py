"""
Pair Generator v4 — Identity Cap
==================================
Improvement over v3: each identity may contribute at most
MAX_PAIRS_PER_IDENTITY pairs across both same and different sets.
This prevents the heavily-photographed LFW identities (~20 people with
50+ images) from dominating the evaluation set and inflating metrics
on easy, repeatedly-seen faces.

Output: configs/pairs_v4.csv
"""

import os
import random
import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
LFW_PATH               = "data/lfw"
OUTPUT_PATH            = "configs/pairs_v4.csv"
SEED                   = 42
PAIRS_EACH             = 500
MAX_PAIRS_PER_IDENTITY = 3     # hard cap per identity across same + diff pairs
MAX_ATTEMPTS           = PAIRS_EACH * 40
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


def sample_same_pairs(id_map: dict, n: int, rng: random.Random,
                      identity_pair_count: dict) -> list:
    """
    Samples n unique same-identity pairs, respecting MAX_PAIRS_PER_IDENTITY.
    Updates identity_pair_count in place so the cap is shared with
    sample_diff_pairs.
    """
    eligible = [name for name, idxs in id_map.items() if len(idxs) >= 2]
    pairs, seen = [], set()
    attempts = 0
    while len(pairs) < n and attempts < MAX_ATTEMPTS:
        attempts += 1
        person = rng.choice(eligible)
        if identity_pair_count[person] >= MAX_PAIRS_PER_IDENTITY:
            continue
        idx1, idx2 = rng.sample(id_map[person], 2)
        key = (person, min(idx1, idx2), max(idx1, idx2))
        if key in seen:
            continue
        seen.add(key)
        identity_pair_count[person] += 1
        pairs.append([person, person, idx1, idx2, 1])
    if len(pairs) < n:
        print(f"  Warning: only generated {len(pairs)}/{n} unique same-pairs")
    return pairs


def sample_diff_pairs(id_map: dict, n: int, rng: random.Random,
                      identity_pair_count: dict) -> list:
    """
    Samples n unique different-identity pairs, respecting MAX_PAIRS_PER_IDENTITY
    for both identities in every pair. Updates identity_pair_count in place.
    """
    all_ids = list(id_map.keys())
    pairs, seen = [], set()
    attempts = 0
    while len(pairs) < n and attempts < MAX_ATTEMPTS:
        attempts += 1
        p1, p2 = rng.sample(all_ids, 2)
        if (identity_pair_count[p1] >= MAX_PAIRS_PER_IDENTITY or
                identity_pair_count[p2] >= MAX_PAIRS_PER_IDENTITY):
            continue
        idx1 = rng.choice(id_map[p1])
        idx2 = rng.choice(id_map[p2])
        # Canonical order so (A,B) and (B,A) hash identically
        (p1, idx1), (p2, idx2) = sorted([(p1, idx1), (p2, idx2)])
        key = (p1, idx1, p2, idx2)
        if key in seen:
            continue
        seen.add(key)
        identity_pair_count[p1] += 1
        identity_pair_count[p2] += 1
        pairs.append([p1, p2, idx1, idx2, 0])
    if len(pairs) < n:
        print(f"  Warning: only generated {len(pairs)}/{n} unique diff-pairs")
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

        # Shared cap counter — same + diff pairs both draw from the same budget
        identity_pair_count = {name: 0 for name in id_map}

        same_rows = sample_same_pairs(id_map, PAIRS_EACH, rng, identity_pair_count)
        diff_rows = sample_diff_pairs(id_map, PAIRS_EACH, rng, identity_pair_count)

        contributors = sum(1 for c in identity_pair_count.values() if c > 0)
        print(f"  Unique identities represented: {contributors}/{len(id_map)}")

        for row in same_rows + diff_rows:
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