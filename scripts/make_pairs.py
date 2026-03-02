import os
import json
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds

def generate_pairs(manifest_path="outputs/manifest.json", num_pairs_per_split=1000, seed=42):
    # 1. Load Manifest and Dataset
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    ds_builder = tfds.builder("lfw:0.1.1")
    # We use 'all' because we will filter based on the manifest logic
    raw_data = list(tfds.as_numpy(ds_builder.as_dataset(split='all')))
    
    # Deterministic sort (Must match ingest_lfw logic)
    raw_data.sort(key=lambda x: (x['label'], x['image'].tobytes()))
    
    # 2. Reconstruct the Splits from the Manifest Policy
    # (Since we split by identity in Step 1, we group images by label)
    identities = sorted(list(set(x['label'] for x in raw_data)))
    rng = np.random.default_rng(seed)
    rng.shuffle(identities)
    
    train_end = int(len(identities) * 0.8) # Matches ingest_lfw ratios
    val_end = int(len(identities) * 0.9)
    
    split_map = {
        'train': set(identities[:train_end]),
        'val': set(identities[train_end:val_end]),
        'test': set(identities[val_end:])
    }

    all_pairs = []

    for split_name, allowed_ids in split_map.items():
        print(f"Generating pairs for {split_name}...")
        # Filter images belonging to this split
        split_images = [img for img in raw_data if img['label'] in allowed_ids]
        
        # Group by identity for easy sampling
        id_to_imgs = {}
        for idx, img in enumerate(split_images):
            lbl = img['label']
            if lbl not in id_to_imgs: id_to_imgs[lbl] = []
            id_to_imgs[lbl].append(idx)
            
        # 3. Generate Positive Pairs (Label 1)
        pos_pairs = []
        # Only use identities with at least 2 images
        eligible_pos_ids = [idx for idx, imgs in id_to_imgs.items() if len(imgs) >= 2]
        
        count = 0
        while count < num_pairs_per_split // 2:
            pid = rng.choice(eligible_pos_ids)
            idx1, idx2 = rng.choice(id_to_imgs[pid], size=2, replace=False)
            pos_pairs.append({
                "left_identity": split_images[idx1]['label'].decode('utf-8'),
                "right_identity": split_images[idx2]['label'].decode('utf-8'),
                "left_index": idx1,  # The index in our sorted list
                "right_index": idx2,
                "label": 1,
                "split": split_name
            })
            count += 1

        # 4. Generate Negative Pairs (Label 0)
        neg_pairs = []
        all_ids_list = sorted(list(id_to_imgs.keys()))
        
        count = 0
        while count < num_pairs_per_split // 2:
            id1, id2 = rng.choice(all_ids_list, size=2, replace=False)
            idx1 = rng.choice(id_to_imgs[id1])
            idx2 = rng.choice(id_to_imgs[id2])
            neg_pairs.append({
                "left_identity": split_images[idx1]['label'].decode('utf-8'),
                "right_identity": split_images[idx2]['label'].decode('utf-8'),
                "left_index": idx1,
                "right_index": idx2,
                "label": 0,
                "split": split_name
            })
            count += 1
            
        all_pairs.extend(pos_pairs)
        all_pairs.extend(neg_pairs)

    # 5. Save to Disk
    df = pd.DataFrame(all_pairs)
    output_path = "outputs/pairs.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} pairs to {output_path}")

if __name__ == "__main__":
    generate_pairs()