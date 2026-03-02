import os
import json
import tensorflow_datasets as tfds
import numpy as np

def ingest_lfw(seed=42, train_ratio=0.8, val_ratio=0.1):
    """
    Ingests LFW, sorts deterministically, and creates a manifest.
    Split policy: Split by Identity (Unique Persons).
    """
    # 1. Load the dataset (not shuffled)
    print("Downloading/Loading LFW from TFDS...")
    ds_builder = tfds.builder("lfw:0.1.1")
    ds_builder.download_and_prepare()
    
    # Load all data into a list to enforce our own ordering
    # Note: LFW is small enough (~13k images) to fit in memory for indexing
    raw_data = tfds.as_numpy(ds_builder.as_dataset(split='all'))
    
    # 2. Extract metadata for deterministic sorting
    # We sort by 'label' (person name) then by 'image/filename'
    data_list = list(raw_data)
    data_list.sort(key=lambda x: (
        x['label'], 
        x['image'].tobytes() # Using the actual image bytes ensures a unique, stable sort
    ))
    # 3. Apply Split Policy: By Identity
    # In face verification, it's safer to ensure a person in 'test' was never seen in 'train'
    unique_identities = sorted(list(set(x['label'] for x in data_list)))
    
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_identities)
    
    num_ids = len(unique_identities)
    train_end = int(num_ids * train_ratio)
    val_end = int(num_ids * (train_ratio + val_ratio))
    
    train_ids = set(unique_identities[:train_end])
    val_ids = set(unique_identities[train_end:val_end])
    test_ids = set(unique_identities[val_end:])
    
    # 4. Categorize images based on identity splits
    splits = {'train': [], 'val': [], 'test': []}
    for item in data_list:
        label = item['label']
        if label in train_ids:
            splits['train'].append(item)
        elif label in val_ids:
            splits['val'].append(item)
        else:
            splits['test'].append(item)

    # 5. Create Manifest
    manifest = {
        "seed": seed,
        "split_policy": "Split by Identity (Person). No person overlaps between sets.",
        "data_source": "tfds:lfw/0.1.0",
        "cache_dir": str(ds_builder.data_dir),
        "counts": {
            "train": {"images": len(splits['train']), "identities": len(train_ids)},
            "val": {"images": len(splits['val']), "identities": len(val_ids)},
            "test": {"images": len(splits['test']), "identities": len(test_ids)}
        }
    }

    # 6. Save Manifest
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    
    print("Ingestion complete. Manifest saved to outputs/manifest.json")
    return manifest

if __name__ == "__main__":
    ingest_lfw()